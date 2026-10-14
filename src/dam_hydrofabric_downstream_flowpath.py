#!/usr/bin/env python3
"""
Extract a downstream flowpath from a NextGen hydrofabric dataset starting from a dam location.

This script:
1. Loads the 'flowpaths' and 'nexus' layers from a NextGen hydrofabric GeoPackage.
2. Identifies the river reach that intersects or is closest to the dam location.
3. Builds a downstream adjacency graph by resolving the NextGen 2-hop topology:
   flowpath (wb-A) → nexus (nex-B) → flowpath (wb-C).
4. Traverses downstream reaches up to 100 km (configurable) or until the network ends.
5. Exports the accumulated flowpath as a GeoPackage.

Usage:
    python generate_hydrofabric_dam_flowpath.py [gpkg_path] [dam_shp_path] [output_file]

Defaults:
    gpkg_path   : data/conus_nextgen_hydrofabric/conus_nextgen.gpkg
    dam_shp_path: data/mountain_dell_dam_location.shp
    output_file : outputs/mdd_downstream_flowpath.gpkg
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import fiona
import geopandas as gpd
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


class HydrofabricDamFlowpath:
    """
    Extract the downstream flowpath from a NextGen hydrofabric dataset starting at a dam.

    The NextGen hydrofabric uses a 2-hop connectivity model:
        flowpath (wb-A)  --[toid]-->  nexus (nex-B)  --[toid]-->  flowpath (wb-C)

    Both the 'flowpaths' layer (geometries + length) and the 'nexus' layer (junction
    connectivity) are required to resolve this topology. Without resolving the nexus hop,
    every traversal step would terminate immediately because nexus IDs are never found
    as reach IDs.
    """

    def __init__(
        self,
        gpkg_path: str,
        dam_shp_path: str,
        output_path: str,
        max_distance_km: float = 100.0,
    ) -> None:
        """
        Initialize with input file paths and traversal parameters.

        Parameters:
            gpkg_path (str): Path to the NextGen hydrofabric GeoPackage.
            dam_shp_path (str): Path to the dam location shapefile (point geometry).
            output_path (str): Destination path for the output flowpath GeoPackage.
            max_distance_km (float): Maximum downstream traversal distance in km.
                Traversal stops when this limit is reached or the network ends.
        """
        self.gpkg_path = Path(gpkg_path).resolve()
        self.dam_shp_path = Path(dam_shp_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.max_distance_km = max_distance_km

        # Set during load_data()
        self.flowpaths_gdf: Optional[gpd.GeoDataFrame] = None
        self.nexus_gdf: Optional[gpd.GeoDataFrame] = None
        self.dam_point = None

        # Set during build_adjacency(): maps wb-ID -> downstream wb-ID (or None if terminal)
        self.adjacency: dict[str, Optional[str]] = {}

    # ------------------------------------------------------------------
    # Phase 1: Data Loading & Validation
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """
        Load the flowpaths layer, nexus layer, and dam shapefile.

        Validates that required layers and columns are present. Reprojects the
        dam point to the flowpaths CRS if they differ. If 'lengthkm' is absent
        from the flowpaths layer, it is derived from geometry length (assumes
        CRS units are metres) and a warning is logged.

        Raises:
            FileNotFoundError: If any input file does not exist.
            ValueError: If required layers or columns are missing, or inputs are empty.
        """
        for path in (self.gpkg_path, self.dam_shp_path):
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

        logger.info("Loading hydrofabric GeoPackage: %s", self.gpkg_path)
        available_layers = fiona.listlayers(str(self.gpkg_path))
        logger.info("Available layers: %s", available_layers)

        for required_layer in ("flowpaths", "nexus"):
            if required_layer not in available_layers:
                raise ValueError(
                    f"Required layer '{required_layer}' not found in {self.gpkg_path}. "
                    f"Available: {available_layers}"
                )

        # Load flowpaths (geometry + connectivity attributes)
        self.flowpaths_gdf = gpd.read_file(str(self.gpkg_path), layer="flowpaths")
        if self.flowpaths_gdf.empty:
            raise ValueError("The 'flowpaths' layer contains no features.")
        logger.info(
            "Flowpaths loaded: %d reaches, CRS: %s", len(self.flowpaths_gdf), self.flowpaths_gdf.crs
        )
        self._validate_flowpaths_schema()

        # Load nexus (junction points that resolve wb→nex→wb connectivity)
        self.nexus_gdf = gpd.read_file(str(self.gpkg_path), layer="nexus")
        if self.nexus_gdf.empty:
            raise ValueError("The 'nexus' layer contains no features.")
        for required_col in ("id", "toid"):
            if required_col not in self.nexus_gdf.columns:
                raise ValueError(
                    f"Nexus layer is missing required column '{required_col}'. "
                    f"Found: {self.nexus_gdf.columns.tolist()}"
                )
        logger.info("Nexus loaded: %d nodes", len(self.nexus_gdf))

        # Load and align dam shapefile CRS to flowpaths CRS
        dam_gdf = gpd.read_file(str(self.dam_shp_path))
        if dam_gdf.empty:
            raise ValueError(f"Dam shapefile contains no features: {self.dam_shp_path}")

        flowpaths_crs = self.flowpaths_gdf.crs
        if dam_gdf.crs != flowpaths_crs:
            logger.info("Reprojecting dam from %s to %s", dam_gdf.crs, flowpaths_crs)
            dam_gdf = dam_gdf.to_crs(flowpaths_crs)

        self.dam_point = dam_gdf.geometry.iloc[0]
        logger.info(
            "Dam point (flowpath CRS): x=%.2f, y=%.2f", self.dam_point.x, self.dam_point.y
        )

    def _validate_flowpaths_schema(self) -> None:
        """
        Validate that the flowpaths GeoDataFrame has 'id', 'toid', and 'lengthkm'.

        If 'lengthkm' is absent, it is derived from geometry length (in metres
        divided by 1000). A warning is logged to alert the user.

        Raises:
            ValueError: If 'id' or 'toid' columns are missing.
        """
        for required_col in ("id", "toid"):
            if required_col not in self.flowpaths_gdf.columns:
                raise ValueError(
                    f"Flowpaths layer is missing required column '{required_col}'. "
                    f"Found: {self.flowpaths_gdf.columns.tolist()}"
                )

        if "lengthkm" not in self.flowpaths_gdf.columns:
            logger.warning(
                "'lengthkm' column not found in flowpaths layer. "
                "Deriving reach length from geometry (assumes CRS units are metres)."
            )
            self.flowpaths_gdf = self.flowpaths_gdf.copy()
            self.flowpaths_gdf["lengthkm"] = self.flowpaths_gdf.geometry.length / 1000.0

    # ------------------------------------------------------------------
    # Phase 2: Starting Reach Identification
    # ------------------------------------------------------------------

    def find_starting_reach(self) -> str:
        """
        Identify the flowpath reach that intersects or is nearest to the dam point.

        First attempts a direct geometric intersection. If no reach intersects the
        dam point (e.g. the dam is slightly off the digitised line), falls back to
        the nearest reach using a Shapely STRtree spatial index.

        Returns:
            str: The 'id' of the starting reach (e.g. 'wb-2863369').

        Raises:
            RuntimeError: If no reaches are present to search against.
        """
        if self.flowpaths_gdf is None or self.flowpaths_gdf.empty:
            raise RuntimeError("Flowpaths data is not loaded. Call load_data() first.")

        logger.info("Identifying starting reach from dam location...")

        # Try direct intersection first (exact hit on digitised line)
        intersecting = self.flowpaths_gdf[self.flowpaths_gdf.geometry.intersects(self.dam_point)]
        if not intersecting.empty:
            starting_id: str = intersecting.iloc[0]["id"]
            logger.info("Dam intersects reach: %s", starting_id)
            return starting_id

        # Fall back to nearest reach using a spatial index
        logger.warning(
            "Dam does not intersect any reach; falling back to nearest-reach search."
        )
        tree = STRtree(self.flowpaths_gdf.geometry.values)
        nearest_pos = tree.nearest(self.dam_point)
        nearest_row = self.flowpaths_gdf.iloc[nearest_pos]
        distance_m: float = nearest_row.geometry.distance(self.dam_point)
        starting_id = nearest_row["id"]
        logger.info(
            "Nearest reach: %s (distance from dam: %.2f m / %.4f km)",
            starting_id,
            distance_m,
            distance_m / 1000.0,
        )
        return starting_id

    # ------------------------------------------------------------------
    # Phase 3: Adjacency Graph Construction (2-hop: wb → nex → wb)
    # ------------------------------------------------------------------

    def build_adjacency(self) -> None:
        """
        Build a direct waterbody-to-waterbody adjacency dict via nexus resolution.

        NextGen hydrofabric topology:
            flowpaths.toid  -> nex-XXXXXXX   (nexus ID)
            nexus.toid      -> wb-XXXXXXX    (next downstream waterbody)

        This method resolves the two-hop path into a single dict:
            { 'wb-A': 'wb-C', ... }  where wb-A → nex-B → wb-C

        Reaches whose nexus is not found in the nexus layer are mapped to None,
        indicating a network terminal (e.g. outlet to ocean or beyond dataset extent).

        Raises:
            RuntimeError: If load_data() has not been called first.
        """
        if self.flowpaths_gdf is None or self.nexus_gdf is None:
            raise RuntimeError("Data is not loaded. Call load_data() first.")

        logger.info("Building downstream adjacency graph (wb → nex → wb)...")

        # Step 1: build nexus-ID → downstream-waterbody-ID lookup
        nex_to_wb: dict[str, str] = dict(
            zip(self.nexus_gdf["id"], self.nexus_gdf["toid"])
        )

        # Step 2: resolve each flowpath's toid (nex-ID) to the downstream wb-ID
        self.adjacency = {}
        unresolved_count = 0
        for _, row in self.flowpaths_gdf.iterrows():
            wb_id: str = row["id"]
            nex_id: str = row["toid"]
            downstream_wb = nex_to_wb.get(nex_id)
            if downstream_wb is None:
                unresolved_count += 1
            self.adjacency[wb_id] = downstream_wb

        logger.info(
            "Adjacency graph built: %d reaches total, %d terminal (nexus not resolved).",
            len(self.adjacency),
            unresolved_count,
        )

    # ------------------------------------------------------------------
    # Phase 4: Downstream Traversal
    # ------------------------------------------------------------------

    def traverse_downstream(self, starting_id: str) -> tuple[list[str], float]:
        """
        Traverse downstream reaches from starting_id, accumulating reach IDs and distance.

        Traversal stops when any of the following conditions is met:
          - Cumulative reach length reaches or would exceed max_distance_km.
          - The network terminates (next downstream reach is None).
          - The next reach is outside the loaded dataset (subset boundary).
          - A cycle is detected (visited set guard).

        Parameters:
            starting_id (str): The 'id' of the starting flowpath reach.

        Returns:
            tuple[list[str], float]: Ordered list of collected reach IDs and total km traversed.

        Raises:
            RuntimeError: If build_adjacency() has not been called first.
        """
        if not self.adjacency:
            raise RuntimeError("Adjacency graph is empty. Call build_adjacency() first.")

        logger.info(
            "Traversing downstream from '%s' (limit: %.1f km)...",
            starting_id,
            self.max_distance_km,
        )

        # O(1) length lookup keyed by reach id
        length_lookup: dict[str, float] = dict(
            zip(self.flowpaths_gdf["id"], self.flowpaths_gdf["lengthkm"])
        )

        collected_ids: list[str] = []
        accumulated_km: float = 0.0
        visited: set[str] = set()
        current_id: Optional[str] = starting_id
        stop_reason: str = "unknown"

        while current_id is not None:
            # Guard: cycle detection
            if current_id in visited:
                stop_reason = f"cycle detected at reach '{current_id}'"
                logger.warning("Cycle detected at reach '%s'; stopping traversal.", current_id)
                break

            # Guard: reach outside loaded dataset (e.g. beyond subset boundary)
            if current_id not in length_lookup:
                stop_reason = (
                    f"reach '{current_id}' not in loaded dataset (out-of-subset terminal)"
                )
                logger.info(
                    "Reach '%s' is outside the loaded dataset; stopping traversal.", current_id
                )
                break

            reach_km: float = length_lookup[current_id]

            # Guard: distance limit — check before adding so we don't overshoot
            if accumulated_km + reach_km > self.max_distance_km:
                stop_reason = (
                    f"distance limit ({self.max_distance_km} km) would be exceeded "
                    f"(current: {accumulated_km:.4f} km, next reach: {reach_km:.4f} km)"
                )
                logger.info(
                    "Distance limit reached; stopping before adding '%s' (%.4f km). "
                    "Total traversed: %.4f km.",
                    current_id,
                    reach_km,
                    accumulated_km,
                )
                break

            visited.add(current_id)
            collected_ids.append(current_id)
            accumulated_km += reach_km

            logger.info(
                "  + '%s'  len=%.4f km  cumulative=%.4f km",
                current_id,
                reach_km,
                accumulated_km,
            )

            next_id = self.adjacency.get(current_id)
            if next_id is None:
                stop_reason = "network terminus (no downstream reach resolved from nexus)"
                logger.info("Network terminus reached after reach '%s'.", current_id)
                break

            current_id = next_id

        logger.info(
            "Traversal complete: %d reaches collected, %.4f km. Stop reason: %s",
            len(collected_ids),
            accumulated_km,
            stop_reason,
        )
        return collected_ids, accumulated_km

    # ------------------------------------------------------------------
    # Phase 5: Export
    # ------------------------------------------------------------------

    def export_flowpath(self, collected_ids: list[str]) -> None:
        """
        Filter the flowpaths GeoDataFrame to the collected reaches and write a GeoPackage.

        Writes two layers to the output GPKG:
          - 'flowpath'            : LineString geometries in downstream traversal order.
          - 'flowpath-attributes' : Hydraulic attributes (n, MusK, MusX) for the same
                                    reaches, sourced from the 'flowpath-attributes' layer
                                    of the input GPKG. This layer is written without
                                    geometry so the output GPKG is self-contained for
                                    downstream hydraulic analysis.

        Reaches are written in downstream traversal order. The output directory is
        created automatically if it does not already exist.

        Parameters:
            collected_ids (list[str]): Ordered list of reach IDs from traverse_downstream().
        """
        if not collected_ids:
            logger.error("No reaches were collected; output file will not be created.")
            return

        # Preserve traversal order by assigning a sort key
        order_map: dict[str, int] = {rid: i for i, rid in enumerate(collected_ids)}
        result_gdf = self.flowpaths_gdf[self.flowpaths_gdf["id"].isin(order_map)].copy()
        result_gdf["_order"] = result_gdf["id"].map(order_map)
        result_gdf = result_gdf.sort_values("_order").drop(columns=["_order"])
        result_gdf = result_gdf.reset_index(drop=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_gdf.to_file(str(self.output_path), driver="GPKG", layer="flowpath")
        logger.info(
            "Flowpath exported: %d reaches, CRS=%s -> %s",
            len(result_gdf),
            result_gdf.crs,
            self.output_path,
        )

        # Write flowpath-attributes for the collected reaches if the layer exists
        available_layers = fiona.listlayers(str(self.gpkg_path))
        if "flowpath-attributes" in available_layers:
            fa_df = gpd.read_file(str(self.gpkg_path), layer="flowpath-attributes")
            fa_subset = fa_df[fa_df["id"].isin(order_map)].copy()
            fa_subset["_order"] = fa_subset["id"].map(order_map)
            fa_subset = fa_subset.sort_values("_order").drop(columns=["_order"])
            fa_subset = fa_subset.reset_index(drop=True)
            # Ensure we have a GeoDataFrame (flowpath-attributes is a non-spatial
            # table, so gpd.read_file returns a plain DataFrame; wrap it back so
            # to_file() is available for GPKG writing)
            if "geometry" in fa_subset.columns:
                fa_subset = fa_subset.drop(columns=["geometry"])
            if not isinstance(fa_subset, gpd.GeoDataFrame):
                fa_subset = gpd.GeoDataFrame(fa_subset)
            fa_subset.to_file(
                str(self.output_path), driver="GPKG", layer="flowpath-attributes"
            )
            logger.info(
                "Flowpath-attributes exported: %d rows -> layer 'flowpath-attributes' in %s",
                len(fa_subset),
                self.output_path,
            )
        else:
            logger.warning(
                "'flowpath-attributes' layer not found in %s; "
                "hydraulic attributes (n, MusK, MusX) will not be written to output GPKG.",
                self.gpkg_path,
            )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full downstream flowpath extraction pipeline.

        Runs all phases in sequence:
          load_data → find_starting_reach → build_adjacency →
          traverse_downstream → export_flowpath

        Returns:
            dict: Summary with keys 'starting_id', 'reaches_collected',
                  'distance_km', and 'output_path'.
        """
        self.load_data()
        starting_id = self.find_starting_reach()
        self.build_adjacency()
        collected_ids, distance_km = self.traverse_downstream(starting_id)
        self.export_flowpath(collected_ids)

        summary = {
            "starting_id": starting_id,
            "reaches_collected": len(collected_ids),
            "distance_km": round(distance_km, 4),
            "output_path": str(self.output_path),
        }
        logger.info("Pipeline complete: %s", summary)
        return summary


# -----------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the command-line interface.

    Returns:
        argparse.ArgumentParser: Configured parser with 3 positional arguments
            and an optional --max-distance flag.
    """
    # Project root: two levels up from this file (src/ -> project root)
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Extract the downstream flowpath from a NextGen hydrofabric GeoPackage "
            "starting at a dam location, up to a configurable distance limit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "hydrofabric_gpkg_file_path",
        nargs="?",
        default=str(
            project_root
            / "data"
            / "conus_nextgen_hydrofabric"
            / "conus_nextgen.gpkg"
        ),
        help="Path to the NextGen hydrofabric GeoPackage.",
    )
    parser.add_argument(
        "dam_shape_file_path",
        nargs="?",
        default=str(project_root / "data" / "mountain_dell_dam_location.shp"),
        help="Path to the dam location shapefile (point geometry).",
    )
    parser.add_argument(
        "output_file_name",
        nargs="?",
        default=str(project_root / "outputs" / "mdd_downstream_flowpath.gpkg"),
        help="Destination path for the output flowpath GeoPackage.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        metavar="KM",
        help="Maximum downstream traversal distance in kilometres.",
    )
    return parser


def main() -> None:
    """Configure logging and run the flowpath extraction pipeline from CLI arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    workflow = HydrofabricDamFlowpath(
        gpkg_path=args.hydrofabric_gpkg_file_path,
        dam_shp_path=args.dam_shape_file_path,
        output_path=args.output_file_name,
        max_distance_km=args.max_distance,
    )
    workflow.run()


if __name__ == "__main__":
    main()
