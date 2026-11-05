#!/usr/bin/env python3
"""
HAND Dam Workflow Implementation

This script implements the Height Above Nearest Drainage (HAND) inundation mapping
workflow for aging dams

"""

import os
import sys
import shutil
import argparse
import logging
import numpy as np
import fiona
import geopandas as gpd
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from utils import setup_taudem_path, run_taudem_command


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HandDamWorkflow:
    """Main class for HAND dam workflow processing."""

    def __init__(self, taudem_path="$HOME/taudem", output_dir="outputs",
                 data_dir="data", clipped_dir="outputs/clipped",
                 outlets_on_reaches="Outlets2_utm.shp",
                 hydrofabric_gpkg_path=None):
        """
        Initialize the workflow.

        Args:
            taudem_path: Path to TauDEM installation
            output_dir: Output directory for results
            data_dir: Input data directory
            clipped_dir: Directory containing clipped rasters
            outlets_on_reaches: Name of user-supplied outlets shapefile in data directory
            hydrofabric_gpkg_path: Path to a downstream flowpath GeoPackage containing
                'flowpath' (geometry) and 'flowpath-attributes' (n, MusK, MusX) layers.
                If None, defaults to
                data/mountain_dell_ngen_hydrofabric/downstream_flowpath_conus.gpkg.
        """
        self.taudem_path = Path(os.path.expandvars(taudem_path)).expanduser().resolve()
        self.output_dir = Path(output_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.clipped_dir = Path(clipped_dir).resolve()
        self.outlets_on_reaches = outlets_on_reaches

        # Hydrofabric flowpath GPKG — default relative to data_dir
        if hydrofabric_gpkg_path is None:
            self.hydrofabric_gpkg_path = (
                self.data_dir
                / "mountain_dell_ngen_hydrofabric"
                / "downstream_flowpath_conus.gpkg"
            )
        else:
            self.hydrofabric_gpkg_path = Path(hydrofabric_gpkg_path).resolve()

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # Set up TauDEM PATH
        setup_taudem_path(str(self.taudem_path))

    def run_taudem_command(self, command, description="TauDEM command"):
        """
        Execute a TauDEM command with error handling.

        Args:
            command: Command string to execute
            description: Description for logging
        """
        run_taudem_command(command, description, cwd=str(self.output_dir))

    def copy_clipped_rasters(self):
        """Copy all clipped rasters from clipped directory to output directory."""
        logger.info("Copying clipped rasters to output directory...")

        if not self.clipped_dir.exists():
            logger.error(f"Clipped directory not found: {self.clipped_dir}")
            sys.exit(1)

        # Copy all .tif files
        for tif_file in self.clipped_dir.glob("*.tif"):
            dest_file = self.output_dir / tif_file.name
            shutil.copy2(tif_file, dest_file)
            logger.info(f"Copied {tif_file.name}")

    def copy_input_files(self):
        """Copy required input files from data directory."""
        logger.info("Copying input files...")

        # Copy stage.txt and forecast.csv
        for filename in ["stage.txt", "forecast.csv"]:
            src_file = self.data_dir / filename
            dest_file = self.output_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, dest_file)
                logger.info(f"Copied {filename}")
            else:
                logger.error(f"Required input file not found: {src_file}")
                sys.exit(1)

    def create_flow_direction_mask(self):
        """Create masked flow direction raster where hydrodem < 0."""
        logger.info("Creating flow direction mask...")

        hydrodem_path = self.output_dir / "hydrodem.tif"
        fdr_path = self.output_dir / "fdr.tif"
        output_path = self.output_dir / "fdr_masked.tif"

        with rasterio.open(hydrodem_path) as hydrodem:
            hydrodem_data = hydrodem.read(1)
            profile = hydrodem.profile

            with rasterio.open(fdr_path) as fdr:
                fdr_data = fdr.read(1)

                # Create mask: keep fdr values where hydrodem < 0, else 0
                mask_condition = hydrodem_data < 0
                result = np.where(mask_condition, fdr_data, 0)

                # Write result
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(result, 1)

        logger.info(f"Created flow direction mask: {output_path}")

    def reclassify_flow_directions(self):
        """Reclassify ArcGIS D8 flow directions to TauDEM D8 encoding."""
        logger.info("Reclassifying flow directions to TauDEM encoding...")

        input_path = self.output_dir / "fdr_masked.tif"
        output_path = self.output_dir / "pfdc.tif"

        # ArcGIS to TauDEM D8 mapping
        remap_dict = {
            1: 1,    # East
            128: 2,  # Northeast
            64: 3,   # North
            32: 4,   # Northwest
            16: 5,   # West
            8: 6,    # Southwest
            4: 7,    # South
            2: 8     # Southeast
        }

        with rasterio.open(input_path) as src:
            data = src.read(1)
            profile = src.profile

            # Create output array
            output_data = np.zeros_like(data, dtype=np.int16)

            # Apply reclassification
            for arcgis_val, taudem_val in remap_dict.items():
                output_data[data == arcgis_val] = taudem_val

            # Set nodata value for TauDEM
            profile.update(dtype=rasterio.int16, nodata=15)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(output_data, 1)

        logger.info(f"Created TauDEM flow directions: {output_path}")

    def convert_elevation_to_meters(self):
        """Convert elevation from centimeters to meters."""
        logger.info("Converting elevation to meters...")

        input_path = self.output_dir / "elevfdc.tif"
        output_path = self.output_dir / "elevNHDm.tif"

        with rasterio.open(input_path) as src:
            data = src.read(1)
            profile = src.profile

            # Convert from centimeters to meters
            converted_data = data / 100.0

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(converted_data, 1)

        logger.info(f"Created elevation in meters: {output_path}")

    def project_dam_location(self):
        """Project dam location shapefile to match raster coordinate system."""
        logger.info("Projecting dam location to raster coordinate system...")

        dam_shp = self.data_dir / "mountain_dell_dam_location.shp"
        fel_raster = self.output_dir / "fel.tif"
        output_shp = self.output_dir / "proj_damloc.shp"

        # Read dam location
        dam_gdf = gpd.read_file(dam_shp)

        # Get target CRS from fel.tif
        with rasterio.open(fel_raster) as src:
            target_crs = src.crs

        # Reproject to target CRS
        dam_projected = dam_gdf.to_crs(target_crs)

        # Save projected shapefile
        dam_projected.to_file(output_shp)

        logger.info(f"Created projected dam location: {output_shp}")

    def copy_outlets_on_reaches(self):
        """Copy user-supplied Outlets2_utm.shp from data directory to outputs directory."""
        logger.info(f"Copying user-supplied outlets file: {self.outlets_on_reaches}")

        # Source file in data directory
        source_shp = self.data_dir / self.outlets_on_reaches

        # Check if source file exists
        if not source_shp.exists():
            raise FileNotFoundError(f"User-supplied outlets file not found: {source_shp}")

        # Destination file in outputs directory
        output_shp = self.output_dir / "Outlets2_utm.shp"

        # Copy the shapefile and all its associated sidecar files
        # by finding all files with the same stem.
        source_stem = source_shp.stem
        for source_file in source_shp.parent.glob(f"{source_stem}.*"):
            dest_file = self.output_dir / source_file.name
            shutil.copy2(source_file, dest_file)

        logger.info(f"Copied outlets file to: {output_shp}")

    def create_dam_location_raster(self):
        """Create binary raster from outlet points."""
        logger.info("Creating dam location raster...")

        fel_raster = self.output_dir / "fel.tif"
        outlet_shp = self.output_dir / "outlet.shp"
        output_raster = self.output_dir / "damloc.tif"

        # Read outlet points
        outlet_gdf = gpd.read_file(outlet_shp)

        # Get raster properties from fel.tif
        with rasterio.open(fel_raster) as src:
            profile = src.profile
            transform = src.transform

            # Create geometries for rasterization
            geometries = [(geom, 1) for geom in outlet_gdf.geometry]

            # Rasterize points
            raster_data = rasterize(
                geometries,
                out_shape=(profile['height'], profile['width']),
                transform=transform,
                fill=0,
                dtype=rasterio.uint8
            )

            # Update profile for output
            profile.update(dtype=rasterio.uint8, nodata=3)

            with rasterio.open(output_raster, 'w', **profile) as dst:
                dst.write(raster_data, 1)

        logger.info(f"Created dam location raster: {output_raster}")

    def run_taudem_phase1(self):
        """Run TauDEM Phase 1: Flow direction conditioning."""
        logger.info("Running TauDEM Phase 1: Flow direction conditioning...")

        # Flow direction conditioning
        self.run_taudem_command(
            (
                "mpiexec -n 4 flowdircond "
                "-z elev_cm.tif "
                "-p pfdc.tif "
                "-zfdc elevfdc.tif"
            ),
            "Flow direction conditioning"
        )

    def run_taudem_phase2(self):
        """Run TauDEM Phase 2: Hydrologic processing."""
        logger.info("Running TauDEM Phase 2: Hydrologic processing...")

        # Pit removal
        self.run_taudem_command(
            (
                "mpiexec -n 8 pitremove "
                "-z elevNHDm.tif "
                "-fel fel.tif"
            ),
            "Pit removal"
        )

        # D8 flow direction
        self.run_taudem_command(
            (
                "mpiexec -n 8 d8flowdir "
                "-fel fel.tif "
                "-p p.tif "
                "-sd8 sd8.tif"
            ),
            "D8 flow direction"
        )

        # D-infinity flow direction
        self.run_taudem_command(
            (
                "mpiexec -n 8 dinfflowdir "
                "-fel fel.tif "
                "-ang ang.tif "
                "-slp slp.tif"
            ),
            "D-infinity flow direction"
        )

        # Contributing area
        self.run_taudem_command(
            (
                "mpiexec -n 8 aread8 "
                "-p p.tif "
                "-ad8 ad8.tif "
                "-nc"
            ),
            "Contributing area calculation"
        )

    def run_taudem_phase3(self):
        """Run TauDEM Phase 3: Stream network and outlets."""
        logger.info("Running TauDEM Phase 3: Stream network and outlets...")

        # Threshold for stream network
        self.run_taudem_command(
            (
                "mpiexec -n 8 threshold "
                "-ssa ad8.tif "
                "-src srctemp.tif "
                "-thresh 10000"
            ),
            "Stream network threshold"
        )

        # Move outlets to streams
        self.run_taudem_command(
            (
                "mpiexec -n 4 moveoutletstostreams "
                "-p p.tif "
                "-src srctemp.tif "
                "-o proj_damloc.shp "
                "-om outlet.shp"
            ),
            "Move outlets to streams"
        )

    def run_taudem_phase4(self):
        """Run TauDEM Phase 4: HAND calculation."""
        logger.info("Running TauDEM Phase 4: HAND calculation...")

        # Contributing area with weight grid (for HAND calculation)
        self.run_taudem_command( 
            (
                "mpiexec -n 8 aread8 "
                "-p p.tif "
                "-ad8 src.tif "
                "-wg damloc.tif "
                "-nc"
            ),
            "Contributing area with weight grid for HAND"
        )

        # HAND calculation
        self.run_taudem_command(
            (
                "mpiexec -n 8 dinfdistdown "
                "-ang ang.tif "
                "-fel fel.tif "
                "-src src.tif "
                "-dd hand.tif "
                "-m ave v "
                "-nc"
            ),
            "HAND calculation"
        )

    def process_additional_outlets(self):
        """Process additional outlets for stream network analysis."""
        logger.info("Processing additional outlets...")

        # Project Outlets2_utm to raster coordinate system
        outlets2_utm = self.output_dir / "Outlets2_utm.shp"
        fel_raster = self.output_dir / "fel.tif"
        outlets2_geo = self.output_dir / "Outlets2_geo.shp"

        # Read Outlets2_utm
        outlets_gdf = gpd.read_file(outlets2_utm)

        # Get target CRS from fel.tif
        with rasterio.open(fel_raster) as src:
            target_crs = src.crs

        # Reproject to target CRS
        outlets_projected = outlets_gdf.to_crs(target_crs)

        # Save projected shapefile
        outlets_projected.to_file(outlets2_geo)

        logger.info(f"Created Outlets2_geo.shp: {outlets2_geo}")

        # Move outlets to streams with maximum distance
        self.run_taudem_command(
            (
                "mpiexec -n 4 moveoutletstostreams "
                "-p p.tif "
                "-src src.tif "
                "-o Outlets2_geo.shp "
                "-om Outlets2_align.shp "
                "-md 3000"
            ),
            "Move additional outlets to streams"
        )

        # Stream network analysis
        self.run_taudem_command(
            (
                "mpiexec -n 4 streamnet "
                "-fel fel.tif "
                "-p p.tif "
                "-ad8 ad8.tif "
                "-src src.tif "
                "-ord ord2.tif "
                "-tree tree2.txt "
                "-coord coord2.txt "
                "-net net2.shp "
                "-o Outlets2_align.shp "
                "-w w2.tif"
            ),
            "Stream network analysis"
        )


    def load_hydrofabric_attributes(self) -> gpd.GeoDataFrame | None:
        """
        Load and merge the flowpath geometry and hydraulic attributes from the
        downstream flowpath GeoPackage.

        Expects the GPKG to contain:
          - 'flowpath'            : LineString geometries with an 'id' column.
          - 'flowpath-attributes' : Non-spatial table with columns 'id', 'n', 'MusK', 'MusX'.

        The two layers are merged on 'id' and the result is reprojected to the CRS
        of the TauDEM outputs (read from outputs/fel.tif) so that sjoin_nearest in
        create_catchment_list() operates in a consistent coordinate system.

        Returns:
            GeoDataFrame with geometry and hydraulic attribute columns, or None if the
            GPKG is missing, the required layers are absent, or fel.tif is not yet available.
        """
        if not self.hydrofabric_gpkg_path.exists():
            logger.warning(
                "Hydrofabric GPKG not found: %s. "
                "Manning_n will default to 0.05; MusK and MusX will default to 0.0.",
                self.hydrofabric_gpkg_path,
            )
            return None

        available_layers = fiona.listlayers(str(self.hydrofabric_gpkg_path))
        logger.info(
            "Hydrofabric GPKG layers available: %s", available_layers
        )

        # Determine geometry layer name ('flowpath' written by generate script,
        # 'flowpaths' used in source NextGen GPKGs)
        geom_layer = None
        for candidate in ("flowpath", "flowpaths"):
            if candidate in available_layers:
                geom_layer = candidate
                break

        if geom_layer is None:
            logger.warning(
                "No 'flowpath' or 'flowpaths' layer found in %s. "
                "Hydraulic attributes will use default values.",
                self.hydrofabric_gpkg_path,
            )
            return None

        flowpath_gdf = gpd.read_file(str(self.hydrofabric_gpkg_path), layer=geom_layer)
        logger.info("Loaded '%s' layer: %d reaches.", geom_layer, len(flowpath_gdf))

        # Merge hydraulic attributes if the layer is present
        if "flowpath-attributes" in available_layers:
            fa_df = gpd.read_file(
                str(self.hydrofabric_gpkg_path), layer="flowpath-attributes"
            )
            # Validate required hydraulic columns
            missing_cols = [c for c in ("id", "n", "MusK", "MusX") if c not in fa_df.columns]
            if missing_cols:
                logger.warning(
                    "'flowpath-attributes' layer is missing columns %s. "
                    "Hydraulic attributes will use default values.",
                    missing_cols,
                )
            else:
                flowpath_gdf = flowpath_gdf.merge(
                    fa_df[["id", "n", "MusK", "MusX"]], on="id", how="left"
                )
                logger.info(
                    "Merged hydraulic attributes: %d rows matched out of %d reaches.",
                    flowpath_gdf[["n", "MusK", "MusX"]].notna().all(axis=1).sum(),
                    len(flowpath_gdf),
                )
        else:
            logger.warning(
                "'flowpath-attributes' layer not found in %s. "
                "Manning_n will default to 0.05; MusK and MusX will default to 0.0.",
                self.hydrofabric_gpkg_path,
            )

        # Reproject to TauDEM output CRS (fel.tif) for spatial join consistency
        fel_raster = self.output_dir / "fel.tif"
        if fel_raster.exists():
            with rasterio.open(str(fel_raster)) as src:
                target_crs = src.crs
            if flowpath_gdf.crs != target_crs:
                logger.info(
                    "Reprojecting hydrofabric from %s to %s.",
                    flowpath_gdf.crs,
                    target_crs,
                )
                flowpath_gdf = flowpath_gdf.to_crs(target_crs)
        else:
            logger.warning(
                "fel.tif not found; skipping CRS reprojection of hydrofabric layer. "
                "Ensure the spatial join in create_catchment_list() uses matching CRS."
            )

        return flowpath_gdf

    def create_catchment_list(self):
        """
        Create catchment list CSV from the stream network shapefile, enriched with
        per-reach hydraulic attributes from the downstream flowpath GeoPackage.

        Hydraulic attributes (Manning_n, MusK, MusX) are sourced by spatially joining
        each TauDEM reach in net2.shp to the nearest hydrofabric flowpath reach.
        If no hydrofabric GPKG is available or a reach has no nearby match, default
        values are applied: Manning_n=0.05, MusK=0.0, MusX=0.0.
        """
        # Default hydraulic values used when no hydrofabric match is found
        DEFAULT_MANNING_N = 0.05
        DEFAULT_MUSK = 0.0
        DEFAULT_MUSX = 0.0

        logger.info("Creating catchment list...")

        net_shp = self.output_dir / "net2.shp"
        output_csv = self.output_dir / "catchlist.csv"

        # Read stream network shapefile and filter to outlet reaches (DSNODEID == 0)
        net_gdf = gpd.read_file(net_shp)
        filtered_gdf = net_gdf[net_gdf['DSNODEID'] == 0].copy()

        # Attempt to load hydrofabric hydraulic attributes for spatial join
        hydrofabric_gdf = self.load_hydrofabric_attributes()

        if hydrofabric_gdf is not None and not hydrofabric_gdf.empty:
            # Ensure CRS alignment before spatial join
            if filtered_gdf.crs != hydrofabric_gdf.crs:
                logger.info(
                    "Reprojecting net2.shp from %s to %s for spatial join.",
                    filtered_gdf.crs,
                    hydrofabric_gdf.crs,
                )
                filtered_gdf = filtered_gdf.to_crs(hydrofabric_gdf.crs)

            # Determine which hydraulic columns are available after merge
            hydro_cols = [c for c in ("n", "MusK", "MusX") if c in hydrofabric_gdf.columns]
            join_cols = ["geometry"] + hydro_cols

            joined_gdf = gpd.sjoin_nearest(
                filtered_gdf,
                hydrofabric_gdf[join_cols],
                how="left",
            )

            matched = joined_gdf[hydro_cols].notna().all(axis=1).sum() if hydro_cols else 0
            logger.info(
                "Spatial join complete: %d/%d reaches matched to a hydrofabric flowpath.",
                matched,
                len(joined_gdf),
            )
            if matched < len(joined_gdf):
                logger.warning(
                    "%d reaches had no hydrofabric match; default values will be applied "
                    "(Manning_n=%.2f, MusK=%.1f, MusX=%.1f).",
                    len(joined_gdf) - matched,
                    DEFAULT_MANNING_N,
                    DEFAULT_MUSK,
                    DEFAULT_MUSX,
                )
        else:
            # No hydrofabric data — carry on with the unjoined filtered GDF
            joined_gdf = filtered_gdf

        # Build catchment list rows
        catchlist_data = []
        for _, row in joined_gdf.iterrows():
            manning_n = (
                row["n"] if "n" in row.index and pd.notna(row["n"])
                else DEFAULT_MANNING_N
            )
            musk = (
                row["MusK"] if "MusK" in row.index and pd.notna(row["MusK"])
                else DEFAULT_MUSK
            )
            musx = (
                row["MusX"] if "MusX" in row.index and pd.notna(row["MusX"])
                else DEFAULT_MUSX
            )
            catchlist_data.append({
                'Id': row['LINKNO'],
                'Slope': row['Slope'],
                'Length': row['Length'],
                'Manning_n': manning_n,
                'MusK': musk,
                'MusX': musx,
            })

        # Save to CSV
        catchlist_df = pd.DataFrame(catchlist_data)
        catchlist_df.to_csv(output_csv, index=False)

        logger.info(f"Created catchment list: {output_csv}")

    def run_workflow(self):
        """Execute the complete HAND dam workflow."""
        logger.info("Starting HAND Dam Workflow...")

        try:
            # Phase 1: Data preparation
            self.copy_clipped_rasters()
            self.copy_input_files()
            self.copy_outlets_on_reaches()

            # Phase 2: Flow direction processing
            self.create_flow_direction_mask()
            self.reclassify_flow_directions()

            # Phase 3: TauDEM flow direction conditioning
            self.run_taudem_phase1()
            self.convert_elevation_to_meters()

            # Phase 4: TauDEM hydrologic processing
            self.run_taudem_phase2()

            # Phase 5: Dam location processing
            self.project_dam_location()
            self.run_taudem_phase3()
            self.create_dam_location_raster()

            # Phase 6: HAND calculation
            self.run_taudem_phase4()

            # Phase 7: Additional outlets and stream network
            self.process_additional_outlets()
            self.create_catchment_list()

            logger.info("HAND Dam Workflow completed successfully!")
            logger.info("To run inundation analysis, use the dam_inundation.py script.")

        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            sys.exit(1)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="HAND Dam Workflow - Height Above Nearest Drainage inundation mapping for aging dams"
    )

    parser.add_argument(
        "--taudem-path",
        default="$HOME/taudem",
        help="Path to TauDEM installation (default: $HOME/taudem)"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for results (default: outputs)"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Input data directory (default: data)"
    )

    parser.add_argument(
        "--clipped-dir",
        default="outputs/clipped",
        help="Directory containing clipped rasters (default: outputs/clipped)"
    )

    parser.add_argument(
        "--outlets-on-reaches",
        default="Outlets2_utm.shp",
        help="Name of user-supplied outlets shapefile in data directory (default: Outlets2_utm.shp)"
    )

    parser.add_argument(
        "--hydrofabric-flowpath-gpkg",
        default=None,
        help=(
            "Path to a downstream flowpath GeoPackage containing 'flowpath' geometry "
            "and 'flowpath-attributes' (n, MusK, MusX) layers used to enrich catchlist.csv. "
            "Default: data/mountain_dell_ngen_hydrofabric/downstream_flowpath_conus.gpkg"
        )
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run workflow
    workflow = HandDamWorkflow(
        taudem_path=args.taudem_path,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        clipped_dir=args.clipped_dir,
        outlets_on_reaches=args.outlets_on_reaches,
        hydrofabric_gpkg_path=args.hydrofabric_flowpath_gpkg,
    )

    workflow.run_workflow()


if __name__ == "__main__":
    main()