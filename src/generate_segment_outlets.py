#!/usr/bin/env python3
"""
Generate stream-segment outlets along the dam's downstream main stem.

Traces the TauDEM D8 flow path from the snapped dam outlet down the dam's
main-stem stream grid (src.tif) and places an outlet point every
``segment_length`` meters (default 2 km). Optionally adds breaks just upstream
of major tributary confluences. The output shapefile uses the same schema as the
hand-made ArcGIS ``Outlets2_utm.shp`` (``ID`` = 1 for the dam, 0 for the others),
so ``dam_hand.py`` can use it with ``streamnet`` with no other changes.

Where the trace stops (the terminal outlet), in order of precedence:
    1. ``--max-distance`` meters below the dam, if given.
    2. Otherwise, just upstream of where the main stem first enters a large lake or
       reservoir (NHDPlus HR ``NHDWaterbody``, FType 390/436, at least
       ``--min-waterbody-area`` km2), if ``--waterbody-path`` is given. Waterbodies
       near the dam (its own reservoir) are ignored.
    3. Otherwise, at the end of src.tif.
The stretch below the terminal outlet gets ``DSNODEID = -1`` in streamnet, so
``dam_hand.create_catchment_list()`` leaves it out of catchlist.csv.

Inputs, all in the dam_hand work directory and in the same CRS:
    p.tif       D8 flow directions (TauDEM encoding 1=E ... 8=SE)
    src.tif     dam main-stem stream grid (aread8 -wg damloc.tif)
    outlet.shp  dam point snapped to the stream by moveoutletstostreams
    ad8.tif     D8 contributing area (only with --confluence-ratio)
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# TauDEM D8 encoding -> (row offset, col offset)
D8_OFFSETS = {
    1: (0, 1),    # East
    2: (-1, 1),   # Northeast
    3: (-1, 0),   # North
    4: (-1, -1),  # Northwest
    5: (0, -1),   # West
    6: (1, -1),   # Southwest
    7: (1, 0),    # South
    8: (1, 1),    # Southeast
}

# NHD waterbody FTypes that end the study reach: 390 = LakePond, 436 = Reservoir
STOP_WATERBODY_FTYPES = (390, 436)
# Waterbodies within this distance of the dam are treated as the dam's own reservoir
DAM_RESERVOIR_BUFFER_M = 500.0
# Same NHDPlus HR GDB that dam_catchments reads by default, relative to the data directory
DEFAULT_WATERBODY_GDB = "NHDPLUS_H_1602_HU4_20220412_GDB/NHDPLUS_H_1602_HU4_20220412_GDB.gdb"
DEFAULT_WATERBODY_PATH = f"data/{DEFAULT_WATERBODY_GDB}"


class SegmentOutletGenerator:
    """Place outlets at fixed along-stream intervals on the dam's D8 main stem."""

    def __init__(self, work_dir="outputs", segment_length=2000.0,
                 min_segment_length=None, confluence_ratio=None,
                 max_distance=None, waterbody_path=None,
                 waterbody_layer="NHDWaterbody", min_waterbody_area_km2=10.0,
                 waterbody_required=False):
        """
        Initialize the generator.

        Args:
            work_dir: dam_hand output directory containing p.tif, src.tif, outlet.shp (and ad8.tif)
            segment_length: Target reach length in meters
            min_segment_length: Shortest reach allowed in meters (default: 0.5 * segment_length);
                shorter pieces are merged into the reach upstream
            confluence_ratio: If set, also break one cell upstream of any cell where contributing
                area grows by more than this fraction in one D8 step (e.g. 0.2 = 20%)
            max_distance: Stop tracing this many meters below the dam. When set, the
                waterbody-based stop is not used.
            waterbody_path: NHDPlus HR GDB (or any vector file) with a waterbody polygon layer.
                If set (and max_distance is not), stop just upstream of the first large lake
                or reservoir on the main stem; if the file is missing, trace to the end of src.tif
            waterbody_layer: Layer name in waterbody_path (default: NHDWaterbody)
            min_waterbody_area_km2: Smallest lake/reservoir that ends the trace, in km2
            waterbody_required: If True, a missing waterbody_path is an error instead of a
                warning (use when the user supplied the path explicitly)
        """
        self.work_dir = Path(work_dir).resolve()
        self.segment_length = float(segment_length)
        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive")
        self.min_segment_length = (0.5 * self.segment_length if min_segment_length is None
                                   else float(min_segment_length))
        self.confluence_ratio = confluence_ratio
        self.max_distance = max_distance
        self.waterbody_path = None if waterbody_path is None else Path(waterbody_path).resolve()
        self.waterbody_layer = waterbody_layer
        self.min_waterbody_area_km2 = float(min_waterbody_area_km2)
        self.waterbody_required = waterbody_required

    @staticmethod
    def _read(path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
        with rasterio.open(path) as src:
            return src.read(1), src.profile

    def trace_main_stem(self, p, src, src_nodata, transform, start_rc):
        """
        Follow D8 directions downstream from start_rc while staying on the stream grid.

        Returns:
            list of (row, col, distance_m) for each cell on the path, starting at the dam cell
        """
        nrows, ncols = p.shape
        dx, dy = abs(transform.a), abs(transform.e)
        diag = math.hypot(dx, dy)

        def on_stream(r, c):
            v = src[r, c]
            if src_nodata is not None and v == src_nodata:
                return False
            return np.isfinite(v) and v > 0

        r, c = start_rc
        if not (0 <= r < nrows and 0 <= c < ncols):
            raise ValueError(f"Dam outlet cell {start_rc} is outside the raster")
        if not on_stream(r, c):
            raise ValueError(f"Dam outlet cell {start_rc} is not on the stream grid (src.tif)")

        path = [(r, c, 0.0)]
        visited = {(r, c)}
        dist = 0.0
        while True:
            d = int(p[r, c]) if np.isfinite(p[r, c]) else 0
            if d not in D8_OFFSETS:
                break  # terminal cell: no flow direction
            dr, dc = D8_OFFSETS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols) or not on_stream(nr, nc):
                break
            if (nr, nc) in visited:
                logger.warning(f"Flow direction loop at cell ({nr}, {nc}); stopping trace")
                break
            step = dx if dr == 0 else dy if dc == 0 else diag
            if self.max_distance is not None and dist + step > self.max_distance:
                break
            dist += step
            r, c = nr, nc
            visited.add((r, c))
            path.append((r, c, dist))
        return path

    def find_waterbody_stop(self, path, transform, crs):
        """
        Find where the traced main stem first enters a large lake or reservoir.

        Returns:
            (index, name): index of the last path cell before the waterbody and its name,
            or (None, None) if the path never enters one
        """
        if not self.waterbody_path.exists():
            if self.waterbody_required:
                raise FileNotFoundError(f"Waterbody file not found: {self.waterbody_path}")
            logger.warning(f"Waterbody file not found: {self.waterbody_path}; "
                           f"tracing to the end of src.tif")
            return None, None

        pts = gpd.GeoDataFrame(
            {"idx": range(len(path))},
            geometry=[Point(transform * (c + 0.5, r + 0.5)) for r, c, _ in path],
            crs=crs,
        )
        layer_crs = gpd.read_file(self.waterbody_path, layer=self.waterbody_layer, rows=1).crs
        bbox = tuple(pts.to_crs(layer_crs).total_bounds)
        wb = gpd.read_file(self.waterbody_path, layer=self.waterbody_layer, bbox=bbox)
        if wb.empty:
            logger.warning(f"{self.waterbody_path.name} has no waterbodies within the main-stem "
                           f"extent (wrong region?); tracing to the end of src.tif")
            return None, None
        wb.columns = [c.lower() if c != wb.geometry.name else c for c in wb.columns]
        if "ftype" in wb.columns:
            wb = wb[wb["ftype"].isin(STOP_WATERBODY_FTYPES)]
        wb = wb.to_crs(crs)
        area_km2 = wb.geometry.area / 1e6  # raster CRS is projected (meters)
        wb = wb[area_km2 >= self.min_waterbody_area_km2]
        # Skip the dam's own reservoir
        wb = wb[wb.distance(pts.geometry.iloc[0]) > DAM_RESERVOIR_BUFFER_M]
        if wb.empty:
            logger.info(f"No lake/reservoir >= {self.min_waterbody_area_km2:g} km2 on the main stem")
            return None, None

        hits = gpd.sjoin(pts, wb, predicate="within")
        if hits.empty:
            logger.info(f"Main stem does not enter a lake/reservoir >= "
                        f"{self.min_waterbody_area_km2:g} km2")
            return None, None
        first = hits.sort_values("idx").iloc[0]
        entry = int(first["idx"])
        name = first.get("gnis_name")
        if not isinstance(name, str) or not name.strip():
            name = "unnamed waterbody"
        if entry < 1:
            return None, None
        return entry - 1, name

    def select_breaks(self, path, ad8=None):
        """
        Choose the path indices where outlets go (dam and terminal not included).

        Returns:
            sorted list of (index, type) where type is "segment" or "confluence"
        """
        total = path[-1][2]
        dists = np.array([pt[2] for pt in path])

        # Confluence breaks: one cell upstream of a large jump in contributing area
        confluences = []
        if self.confluence_ratio is not None and ad8 is not None:
            for i in range(1, len(path) - 1):
                a_cur = ad8[path[i][0], path[i][1]]
                a_next = ad8[path[i + 1][0], path[i + 1][1]]
                if a_cur > 0 and np.isfinite(a_next) and a_next / a_cur - 1.0 > self.confluence_ratio:
                    confluences.append(i)

        # Keep confluence breaks at least min_segment_length from the dam, the terminal
        # outlet and each other (first one wins in a cluster)
        kept, conf_dists = [], []
        for i in confluences:
            d = dists[i]
            if d < self.min_segment_length or total - d < self.min_segment_length:
                continue
            if conf_dists and d - conf_dists[-1] < self.min_segment_length:
                continue
            kept.append(i)
            conf_dists.append(d)
        confluences = kept

        # Fixed-interval breaks: first cell at or beyond each multiple of segment_length
        segments = []
        k = 1
        while k * self.segment_length < total:
            i = int(np.searchsorted(dists, k * self.segment_length))
            k += 1
            if i >= len(path) - 1:
                break
            # Merge a short last reach into the one above it
            if total - dists[i] < self.min_segment_length:
                continue
            # Drop interval breaks that would leave a short reach next to a confluence break
            if any(abs(dists[i] - cd) < self.min_segment_length for cd in conf_dists):
                continue
            segments.append(i)

        breaks = {i: "segment" for i in segments}
        breaks.update({i: "confluence" for i in confluences})
        return sorted(breaks.items())

    def generate(self, output_path=None, to_crs=None):
        """
        Generate the outlets shapefile.

        Args:
            output_path: Output shapefile (default: <work_dir>/Outlets2_geo.shp), in the raster CRS
            to_crs: Optional CRS (e.g. "EPSG:26912") for an extra reprojected copy named *_<epsg>.shp

        Returns:
            GeoDataFrame of outlets in the raster CRS
        """
        output_path = (self.work_dir / "Outlets2_geo.shp" if output_path is None
                       else Path(output_path).resolve())
        logger.info(f"Generating {self.segment_length:g} m segment outlets in {self.work_dir}")

        p, p_profile = self._read(self.work_dir / "p.tif")
        src, src_profile = self._read(self.work_dir / "src.tif")
        crs, transform = p_profile["crs"], p_profile["transform"]
        if crs is None or crs.is_geographic:
            raise ValueError(f"p.tif must be in a projected CRS with meter units, got: {crs}")
        if p.shape != src.shape:
            raise ValueError("p.tif and src.tif have different dimensions")

        ad8 = None
        if self.confluence_ratio is not None:
            ad8, _ = self._read(self.work_dir / "ad8.tif")

        outlet_shp = self.work_dir / "outlet.shp"
        if not outlet_shp.exists():
            raise FileNotFoundError(f"Snapped dam outlet not found: {outlet_shp}")
        dam_gdf = gpd.read_file(outlet_shp).to_crs(crs)
        if len(dam_gdf) != 1:
            logger.warning(f"{outlet_shp.name} has {len(dam_gdf)} points; using the first")
        dam_pt = dam_gdf.geometry.iloc[0]
        col, row = ~transform * (dam_pt.x, dam_pt.y)
        start_rc = (int(math.floor(row)), int(math.floor(col)))

        path = self.trace_main_stem(p, src, src_profile.get("nodata"), transform, start_rc)
        if len(path) < 2:
            raise ValueError("Main-stem trace is only one cell long; check src.tif and outlet.shp")
        total = path[-1][2]
        logger.info(f"Traced main stem: {len(path)} cells, {total / 1000:.2f} km")

        if self.max_distance is not None:
            logger.info(f"Stopping at --max-distance {self.max_distance:g} m (waterbody stop not used)")
        elif self.waterbody_path is not None:
            stop, name = self.find_waterbody_stop(path, transform, crs)
            if stop is not None and stop >= 1:
                path = path[:stop + 1]
                logger.info(f"Stopping where the main stem enters {name}: "
                            f"{path[-1][2] / 1000:.2f} km below the dam "
                            f"({(total - path[-1][2]) / 1000:.2f} km left out)")

        breaks = self.select_breaks(path, ad8)

        records = [(0, "dam", "dam", 1.0)]
        n_seg = n_conf = 0
        for i, kind in breaks:
            if kind == "segment":
                n_seg += 1
                records.append((i, f"seg_{n_seg:03d}", kind, 0.0))
            else:
                n_conf += 1
                records.append((i, f"confluence_{n_conf:03d}", kind, 0.0))
        records.append((len(path) - 1, "terminal", "terminal", 0.0))

        rows = []
        for i, name, kind, oid in records:
            r, c, dist = path[i]
            x, y = transform * (c + 0.5, r + 0.5)
            rows.append({"ID": oid, "Dam_Name": name, "type": kind,
                         "dist_m": round(dist, 1), "geometry": Point(x, y)})
        outlets = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        outlets.to_file(output_path)
        logger.info(f"Wrote {len(outlets)} outlets ({n_seg} segment, {n_conf} confluence, "
                     f"dam + terminal) to {output_path}")

        if to_crs:
            reproj = outlets.to_crs(to_crs)
            epsg = reproj.crs.to_epsg()
            suffix = str(epsg) if epsg else "reproj"
            reproj_path = output_path.with_name(f"{output_path.stem}_{suffix}.shp")
            reproj.to_file(reproj_path)
            logger.info(f"Wrote reprojected copy to {reproj_path}")

        return outlets


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description=("Generate outlets every N meters along the dam's downstream main stem "
                     "(TauDEM D8 network) for use by dam_hand.py")
    )
    parser.add_argument("--work-dir", default="outputs",
                        help="dam_hand output directory with p.tif, src.tif, outlet.shp (default: outputs)")
    parser.add_argument("--output", default=None,
                        help="Output shapefile (default: <work-dir>/Outlets2_geo.shp)")
    parser.add_argument("--segment-length", type=float, default=2000.0,
                        help="Target reach length in meters (default: 2000)")
    parser.add_argument("--min-segment-length", type=float, default=None,
                        help="Shortest reach in meters; shorter pieces are merged (default: 0.5 * segment length)")
    parser.add_argument("--confluence-ratio", type=float, default=None,
                        help="Also break above confluences where contributing area grows by more than "
                             "this fraction in one step, e.g. 0.2 (default: off; needs ad8.tif)")
    parser.add_argument("--max-distance", type=float, default=None,
                        help="Stop this many meters below the dam; overrides the waterbody stop "
                             "(default: off)")
    parser.add_argument("--waterbody-path", default=None,
                        help="NHDPlus HR GDB with an NHDWaterbody layer; the trace stops just upstream "
                             f"of the first large lake/reservoir (default: {DEFAULT_WATERBODY_PATH}, "
                             "skipped with a warning if missing; an explicit path must exist; "
                             "ignored with --max-distance)")
    parser.add_argument("--min-waterbody-area", type=float, default=10.0,
                        help="Smallest lake/reservoir (km2) that ends the trace (default: 10)")
    parser.add_argument("--to-crs", default=None,
                        help="Also write a reprojected copy, e.g. EPSG:26912")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        generator = SegmentOutletGenerator(
            work_dir=args.work_dir,
            segment_length=args.segment_length,
            min_segment_length=args.min_segment_length,
            confluence_ratio=args.confluence_ratio,
            max_distance=args.max_distance,
            waterbody_path=args.waterbody_path or DEFAULT_WATERBODY_PATH,
            min_waterbody_area_km2=args.min_waterbody_area,
            waterbody_required=args.waterbody_path is not None,
        )
        generator.generate(output_path=args.output, to_crs=args.to_crs)
    except Exception as e:
        logger.error(f"Outlet generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
