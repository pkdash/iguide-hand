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
import geopandas as gpd
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.features import rasterize
from utils import (resolve_user_path, setup_taudem_path, run_taudem_command, find_dam_shapefile,
                   find_nhdplus_gdb, default_output_dir, read_run_info)
from generate_segment_outlets import (SegmentOutletGenerator, default_waterbody_path,
                                      STOP_WATERBODY_FTYPES)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class HandDamWorkflow:
    """Main class for HAND dam workflow processing."""

    def __init__(self, dam_dir, taudem_path="$HOME/taudem", output_dir=None,
                 clipped_dir=None, outlets_on_reaches="Outlets2_utm.shp", segment_length=None,
                 min_segment_length=None, confluence_ratio=None, min_slope=1e-5,
                 max_distance=None, waterbody_path=None,
                 min_waterbody_area_km2=10.0, dam_shapefile=None,
                 stream_threshold=10000, move_outlet_distance=3000, dam_move_distance=200,
                 num_processes=4, manning_n=0.05):
        """
        Initialize the workflow.

        Args:
            dam_dir: Per-dam input directory (dam location shapefile, stage.txt, and
                optionally forecast.csv and the outlets shapefile)
            taudem_path: Path to TauDEM installation
            output_dir: Output directory for results (default: outputs/<dam_dir name>)
            clipped_dir: Directory containing clipped rasters (default: <output_dir>/clipped,
                where dam_catchments writes them)
            outlets_on_reaches: Name of user-supplied outlets shapefile in dam_dir
                (ignored when segment_length is set)
            segment_length: If set, generate outlets every this many meters along the dam's
                main stem instead of using outlets_on_reaches
            min_segment_length: Shortest generated reach in meters (default: 0.5 * segment_length)
            confluence_ratio: If set, also break generated reaches above confluences where
                contributing area grows by more than this fraction
            min_slope: Reaches flatter than this get the slope of the surrounding main stem
                (see reach_slopes); clamped to it only if the whole main stem is flatter
            max_distance: With segment_length, end the main stem this many meters below the dam
                (overrides the waterbody stop)
            waterbody_path: With segment_length, NHDPlus HR GDB whose NHDWaterbody layer ends the
                main stem just upstream of the first large lake/reservoir. Default (None):
                the huc12_gdb dam_catchments recorded in <output_dir>/run_info.json, else the
                NHDPlus HR GDB in dam_dir; with neither, there is no waterbody stop. An explicit
                path must exist.
            min_waterbody_area_km2: Smallest lake/reservoir (km2) that ends the main stem
            dam_shapefile: Dam location shapefile. Default (None): dam_location.shp, or the
                single *dam_location.shp, in dam_dir
            stream_threshold: Contributing-area threshold (cells) for the stream grid
            move_outlet_distance: Max cells moveoutletstostreams moves the reach outlets
            dam_move_distance: Max cells moveoutletstostreams moves the dam point down its
                flow path to the stream grid; if no stream is reached, a warning is logged
            num_processes: Number of MPI processes for TauDEM. streamnet's LINKNO numbering
                depends on it, so keep it fixed for forecasts keyed to reach IDs
            manning_n: Manning's n written for every reach in catchlist.csv
        """
        self.taudem_path = Path(os.path.expandvars(taudem_path)).expanduser().resolve()
        self.dam_dir = Path(dam_dir).expanduser().resolve()
        if not self.dam_dir.is_dir():
            raise FileNotFoundError(f"Dam directory not found: {self.dam_dir}")
        self.output_dir = (default_output_dir(self.dam_dir) if output_dir is None
                           else Path(output_dir).expanduser().resolve())
        self.clipped_dir = (self.output_dir / "clipped" if clipped_dir is None
                            else Path(clipped_dir).expanduser().resolve())
        self.outlets_on_reaches = outlets_on_reaches
        self.dam_shapefile = (find_dam_shapefile(self.dam_dir) if dam_shapefile is None
                              else resolve_user_path(dam_shapefile, self.dam_dir))
        self.stream_threshold = stream_threshold
        self.move_outlet_distance = move_outlet_distance
        self.dam_move_distance = dam_move_distance
        self.dam_snap = None  # set by check_dam_outlet()
        self.num_processes = num_processes
        self.manning_n = manning_n
        self.segment_length = segment_length
        self.min_segment_length = min_segment_length
        self.confluence_ratio = confluence_ratio
        self.min_slope = min_slope
        self.max_distance = max_distance
        self.waterbody_required = waterbody_path is not None
        if waterbody_path is not None:
            self.waterbody_path = Path(waterbody_path).expanduser().resolve()
        else:
            self.waterbody_path = default_waterbody_path(self.output_dir)
            if self.waterbody_path is None:
                try:
                    self.waterbody_path = find_nhdplus_gdb(self.dam_dir)
                except FileNotFoundError:
                    pass  # no waterbody stop; SegmentOutletGenerator warns
        self.min_waterbody_area_km2 = min_waterbody_area_km2
        # Fail before the TauDEM steps rather than at outlet generation
        if (self.waterbody_required and segment_length is not None and max_distance is None
                and not self.waterbody_path.exists()):
            raise FileNotFoundError(f"Waterbody file not found: {self.waterbody_path}")

        logger.info(f"Dam directory: {self.dam_dir}")
        logger.info(f"Dam shapefile: {self.dam_shapefile}")
        logger.info(f"Output directory: {self.output_dir}")
        if not read_run_info(self.output_dir):
            logger.warning(f"No run_info.json in {self.output_dir}; was dam_catchments run "
                           "with this output directory?")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        # stage.txt is required (input to dam_hydroprop)
        src_file = self.dam_dir / "stage.txt"
        if not src_file.exists():
            logger.error(f"Required input file not found: {src_file}")
            sys.exit(1)
        shutil.copy2(src_file, self.output_dir / "stage.txt")
        logger.info("Copied stage.txt")

        # forecast.csv is optional here: it is keyed to this run's reach IDs
        # (LINKNO), so it must match the current run. Use your own forecast.csv
        # (Id, flow_m3s) if you have one that matches, or generate one (e.g. a
        # uniform-flow forecast) after dam_hand. Either way it can be passed to
        # dam_inundation directly via --forecast-file.
        src_file = self.dam_dir / "forecast.csv"
        if src_file.exists():
            shutil.copy2(src_file, self.output_dir / "forecast.csv")
            logger.info("Copied forecast.csv")
        else:
            logger.info(
                f"No forecast.csv in {self.dam_dir}; pass your own (or a generated) forecast "
                "matching this run's reach IDs to dam_inundation via --forecast-file"
            )

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

        dam_shp = self.dam_shapefile
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
        """Copy user-supplied Outlets2_utm.shp from dam directory to outputs directory."""
        logger.info(f"Copying user-supplied outlets file: {self.outlets_on_reaches}")

        # Source file in dam directory
        source_shp = self.dam_dir / self.outlets_on_reaches

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
                f"mpiexec -n {self.num_processes} flowdircond "
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
                f"mpiexec -n {self.num_processes} pitremove "
                "-z elevNHDm.tif "
                "-fel fel.tif"
            ),
            "Pit removal"
        )

        # D8 flow direction
        self.run_taudem_command(
            (
                f"mpiexec -n {self.num_processes} d8flowdir "
                "-fel fel.tif "
                "-p p.tif "
                "-sd8 sd8.tif"
            ),
            "D8 flow direction"
        )

        # D-infinity flow direction
        self.run_taudem_command(
            (
                f"mpiexec -n {self.num_processes} dinfflowdir "
                "-fel fel.tif "
                "-ang ang.tif "
                "-slp slp.tif"
            ),
            "D-infinity flow direction"
        )

        # Contributing area
        self.run_taudem_command(
            (
                f"mpiexec -n {self.num_processes} aread8 "
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
                f"mpiexec -n {self.num_processes} threshold "
                "-ssa ad8.tif "
                "-src srctemp.tif "
                f"-thresh {self.stream_threshold:g}"
            ),
            "Stream network threshold"
        )

        # Move the dam point down its flow path to the first stream cell. Without -md,
        # TauDEM gives up after 50 cells and silently leaves the point off-stream.
        self.run_taudem_command(
            (
                f"mpiexec -n {self.num_processes} moveoutletstostreams "
                "-p p.tif "
                "-src srctemp.tif "
                "-o proj_damloc.shp "
                "-om outlet.shp "
                f"-md {self.dam_move_distance}"
            ),
            "Move dam location to stream"
        )
        self.check_dam_outlet()

    def check_dam_outlet(self):
        """
        Report where the dam point was snapped. Warn if it is not on the stream grid, or if
        the dam point or snapped outlet lies inside an NHD lake/reservoir.

        An off-stream dam point (e.g. an NID coordinate on the embankment) makes the main stem
        start on a hillslope path. A point in the reservoir sits on a flat DEM surface, where
        the stream TauDEM routes across the flat may leave by the wrong side (Quail Creek
        Dam: out past the South Dam instead of down Quail Creek). Either way the first reaches'
        length, slope, rating curve and the HAND around them are not those of the river.

        Returns:
            dict with moved_m, contributing_area_km2, on_stream and in_waterbody (None, or a
            description) (also kept in self.dam_snap)
        """
        dam = gpd.read_file(self.output_dir / "proj_damloc.shp").geometry.iloc[0]
        outlet = gpd.read_file(self.output_dir / "outlet.shp").geometry.iloc[0]
        with rasterio.open(self.output_dir / "srctemp.tif") as src:
            row, col = src.index(outlet.x, outlet.y)
            on_stream = bool(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0] == 1)
            cell_area_km2 = abs(src.res[0] * src.res[1]) / 1e6
        with rasterio.open(self.output_dir / "ad8.tif") as src:
            cells = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
        self.dam_snap = {
            "moved_m": round(dam.distance(outlet), 1),
            "contributing_area_km2": float(f"{cells * cell_area_km2:.3g}"),
            "on_stream": on_stream,
        }
        if on_stream:
            logger.info(
                f"Dam point moved {self.dam_snap['moved_m']:g} m to the stream grid "
                f"(contributing area {self.dam_snap['contributing_area_km2']:.3g} km2 ({cells:.0f} cells))"
            )
        else:
            logger.warning("=" * 70)
            logger.warning(
                f"Dam point is NOT on the stream grid: no stream cell within "
                f"{self.dam_move_distance} cells down its flow path (contributing area at the "
                f"point: {self.dam_snap['contributing_area_km2']:.3g} km2 = {cells:.0f} cells, stream threshold "
                f"{self.stream_threshold * cell_area_km2:.3g} km2 = {self.stream_threshold:g} cells). The main stem will start on a "
                "hillslope path, so results near the dam are unreliable. Move the dam point onto "
                "the channel below the dam, or raise --dam-move-distance."
            )
            logger.warning("=" * 70)

        # A point inside a lake/reservoir sits on the DEM's flat water surface. TauDEM's
        # flow directions across that flat are made up, so a "stream" through it, and the
        # side of the reservoir it leaves by, need not be the dam's outflow channel.
        self.dam_snap["in_waterbody"] = None
        waterbodies = self.find_waterbodies_containing({"dam point": dam, "snapped dam outlet": outlet})
        if waterbodies:
            self.dam_snap["in_waterbody"] = "; ".join(f"{k} in {v}" for k, v in waterbodies.items())
        if "snapped dam outlet" in waterbodies:
            logger.warning("=" * 70)
            logger.warning(
                f"Snapped dam outlet is inside {waterbodies['snapped dam outlet']}. The DEM is "
                "flat there, so the main stem follows a made-up route across the reservoir and may "
                "leave it by the wrong side instead of through the dam. Move the dam point onto "
                "the channel just below the dam embankment."
            )
            logger.warning("=" * 70)
        elif "dam point" in waterbodies:
            # The snap followed the flat to its lowest exit, which for an on-stream reservoir
            # is the dam; worth a look on a map when the reservoir has several exits
            logger.info(
                f"Dam point is inside {waterbodies['dam point']}; the snapped outlet, "
                f"{self.dam_snap['moved_m']:g} m away, is outside it. Check that the outlet is "
                "just below the dam, not on another side of the reservoir."
            )
        return self.dam_snap

    def find_waterbodies_containing(self, points):
        """
        Return {label: waterbody name} for the points (in the raster CRS) that fall inside an
        NHD lake/reservoir in self.waterbody_path. Returns {} if no GDB is available.
        """
        if self.waterbody_path is None or not Path(self.waterbody_path).exists():
            logger.info("No NHDPlus GDB available; skipping the dam-in-reservoir check")
            return {}
        with rasterio.open(self.output_dir / "fel.tif") as src:
            crs = src.crs
        pts = gpd.GeoSeries(list(points.values()), index=list(points.keys()), crs=crs)
        try:
            layer_crs = gpd.read_file(self.waterbody_path, layer="NHDWaterbody", rows=1).crs
            # Small search box around the points (degrees or meters, per the layer CRS)
            pad = 1e-3 if layer_crs.is_geographic else 100
            minx, miny, maxx, maxy = pts.to_crs(layer_crs).total_bounds
            bbox = (minx - pad, miny - pad, maxx + pad, maxy + pad)
            wb = gpd.read_file(self.waterbody_path, layer="NHDWaterbody", bbox=bbox)
        except Exception as e:  # missing layer, unreadable GDB: the check is advisory
            logger.info(f"Skipping the dam-in-reservoir check: {e}")
            return {}
        wb.columns = [c.lower() if c != wb.geometry.name else c for c in wb.columns]
        if "ftype" in wb.columns:
            wb = wb[wb["ftype"].isin(STOP_WATERBODY_FTYPES)]
        wb = wb.to_crs(crs)
        found = {}
        for label, pt in pts.items():
            hit = wb[wb.contains(pt)]
            if not hit.empty:
                name = hit.iloc[0].get("gnis_name")
                found[label] = name if isinstance(name, str) and name.strip() else "an unnamed waterbody"
        return found

    def run_taudem_phase4(self):
        """Run TauDEM Phase 4: HAND calculation."""
        logger.info("Running TauDEM Phase 4: HAND calculation...")

        # Contributing area with weight grid (for HAND calculation)
        self.run_taudem_command( 
            (
                f"mpiexec -n {self.num_processes} aread8 "
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
                f"mpiexec -n {self.num_processes} dinfdistdown "
                "-ang ang.tif "
                "-fel fel.tif "
                "-src src.tif "
                "-dd hand.tif "
                "-m ave v "
                "-nc"
            ),
            "HAND calculation"
        )

    def generate_segment_outlets(self):
        """Generate Outlets2_geo.shp every segment_length meters along the dam's main stem."""
        generator = SegmentOutletGenerator(
            work_dir=self.output_dir,
            segment_length=self.segment_length,
            min_segment_length=self.min_segment_length,
            confluence_ratio=self.confluence_ratio,
            max_distance=self.max_distance,
            waterbody_path=self.waterbody_path,
            min_waterbody_area_km2=self.min_waterbody_area_km2,
            waterbody_required=self.waterbody_required,
        )
        generator.generate(output_path=self.output_dir / "Outlets2_geo.shp")

    def process_additional_outlets(self):
        """Process additional outlets for stream network analysis."""
        logger.info("Processing additional outlets...")

        outlets2_geo = self.output_dir / "Outlets2_geo.shp"

        if self.segment_length is None:
            # Project user-supplied Outlets2_utm to raster coordinate system
            outlets2_utm = self.output_dir / "Outlets2_utm.shp"
            fel_raster = self.output_dir / "fel.tif"

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
        # else: generated outlets are already in the raster CRS and on src.tif cells

        # Move outlets to streams with maximum distance
        self.run_taudem_command(
            (
                f"mpiexec -n {self.num_processes} moveoutletstostreams "
                "-p p.tif "
                "-src src.tif "
                "-o Outlets2_geo.shp "
                "-om Outlets2_align.shp "
                f"-md {self.move_outlet_distance}"
            ),
            "Move additional outlets to streams"
        )

        # Stream network analysis
        self.run_taudem_command(
            (
                f"mpiexec -n {self.num_processes} streamnet "
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


    def reach_slopes(self, reaches):
        """
        Return {LINKNO: slope} for the study reaches, fixing reaches that are flat in the DEM.

        streamnet's Slope is a reach's end-to-end drop on the pit-filled DEM (fel.tif) over
        its length. Pit filling leaves flat stretches, so a reach that falls inside one can
        get a drop of ~0 even where the river is clearly sloping (Quail Creek Dam: 0 m over
        2 km between reaches sloping 2-3e-3). With Manning's equation, such a slope needs an
        unrealistically large depth to carry the flow.

        A reach whose slope is below min_slope therefore gets the slope of a longer stretch
        of the main stem: the reach plus k reaches upstream and k downstream (total drop over
        total length), starting at k = 1 and growing until the slope reaches min_slope. If
        even the whole main stem is flatter than that, the slope is clamped to min_slope.

        Args:
            reaches: GeoDataFrame of study reaches from net2.shp (LINKNO, DSLINKNO,
                strmDrop, Length, Slope)
        """
        r = reaches.set_index("LINKNO")
        slopes = r["Slope"].astype(float).to_dict()
        if self.min_slope is None:
            return slopes

        # Neighbours along the main stem (upstream: the link draining into this one; if
        # there are several, the one with the largest contributing area)
        downstream = {l: d for l, d in r["DSLINKNO"].items() if d in r.index}
        upstream = {}
        for l, d in downstream.items():
            if d not in upstream or r.loc[l, "DSContArea"] > r.loc[upstream[d], "DSContArea"]:
                upstream[d] = l

        for link, slope in list(slopes.items()):
            if slope >= self.min_slope:
                continue
            window, new_slope = [link], slope
            up, down = upstream.get(link), downstream.get(link)
            while new_slope < self.min_slope and (up is not None or down is not None):
                window += [l for l in (up, down) if l is not None]
                up = upstream.get(up) if up is not None else None
                down = downstream.get(down) if down is not None else None
                new_slope = r.loc[window, "strmDrop"].sum() / r.loc[window, "Length"].sum()
            if new_slope >= self.min_slope:
                logger.warning(
                    f"Reach {link}: slope {slope:.2e} is below --min-slope {self.min_slope:.0e} "
                    f"(flat in the pit-filled DEM); using {new_slope:.2e}, the slope over "
                    f"{r.loc[window, 'Length'].sum() / 1000:.1f} km of main stem around it "
                    f"({len(window)} reaches)"
                )
                slopes[link] = new_slope
            else:
                logger.warning(
                    f"Reach {link}: slope {slope:.2e} is below --min-slope and the whole main "
                    f"stem is flatter ({new_slope:.2e}); set to {self.min_slope:.2e}"
                )
                slopes[link] = self.min_slope
        return slopes

    def create_catchment_list(self):
        """Create catchment list CSV from stream network shapefile."""
        logger.info("Creating catchment list...")

        net_shp = self.output_dir / "net2.shp"
        output_csv = self.output_dir / "catchlist.csv"

        # Read stream network shapefile
        net_gdf = gpd.read_file(net_shp)

        # Filter rows where DSNODEID == 0 (internal nodes)
        filtered_gdf = net_gdf[net_gdf['DSNODEID'] == 0]
        slopes = self.reach_slopes(filtered_gdf)

        # Create catchment list dataframe
        catchlist_data = []
        for _, row in filtered_gdf.iterrows():
            catchlist_data.append({
                'Id': row['LINKNO'],
                'Slope': slopes[row['LINKNO']],
                'Length': row['Length'],
                'Manning_n': self.manning_n
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
            if self.segment_length is None:
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
            if self.segment_length is not None:
                self.generate_segment_outlets()
            self.process_additional_outlets()
            self.create_catchment_list()

            logger.info("HAND Dam Workflow completed successfully!")
            logger.info(
                "Next: run dam_hydroprop (hydraulic properties), then dam_inundation "
                "(inundation depth for a forecast)."
            )

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
        "--dam-dir",
        required=True,
        help="Per-dam input directory, e.g. data/<dam>, holding the dam location shapefile, "
             "stage.txt, and optionally forecast.csv and the outlets shapefile"
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: outputs/<dam-dir name>)"
    )

    parser.add_argument(
        "--dam-shapefile",
        default=None,
        help="Dam location shapefile; a bare filename is looked up in --dam-dir "
             "(default: dam_location.shp, or the single "
             "*dam_location.shp, in --dam-dir)"
    )

    parser.add_argument(
        "--clipped-dir",
        default=None,
        help="Directory containing clipped rasters (default: <output-dir>/clipped)"
    )

    parser.add_argument(
        "--outlets-on-reaches",
        default="Outlets2_utm.shp",
        help="Name of user-supplied outlets shapefile in --dam-dir (default: Outlets2_utm.shp). "
             "Ignored when --segment-length is set"
    )

    parser.add_argument(
        "--segment-length",
        type=float,
        default=None,
        help="Generate outlets every this many meters along the dam's downstream main stem "
             "(e.g. 2000) instead of using --outlets-on-reaches"
    )

    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=None,
        help="Shortest generated reach in meters; shorter pieces are merged (default: 0.5 * segment length)"
    )

    parser.add_argument(
        "--confluence-ratio",
        type=float,
        default=None,
        help="With --segment-length, also break above confluences where contributing area grows "
             "by more than this fraction in one cell step, e.g. 0.2 (default: off)"
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="With --segment-length, end the main stem this many meters below the dam; "
             "overrides the waterbody stop (default: off)"
    )

    parser.add_argument(
        "--waterbody-path",
        default=None,
        help="With --segment-length, NHDPlus HR GDB whose NHDWaterbody layer ends the main stem "
             "just upstream of the first large lake/reservoir (default: the --huc12-gdb that "
             "dam_catchments recorded in <output-dir>/run_info.json, else the GDB in --dam-dir; "
             "skipped with a warning if "
             "unavailable; an explicit path must exist; ignored with --max-distance)"
    )

    parser.add_argument(
        "--min-waterbody-area",
        type=float,
        default=10.0,
        help="Smallest lake/reservoir in km2 that ends the main stem (default: 10)"
    )

    parser.add_argument(
        "--min-slope",
        type=float,
        default=1e-5,
        help="Reaches flatter than this get the slope over the main stem around them; "
             "clamped to it only if the whole main stem is flatter (default: 1e-5)"
    )

    parser.add_argument(
        "--stream-threshold",
        type=float,
        default=10000,
        help="Contributing-area threshold in cells for the TauDEM stream grid (default: 10000)"
    )

    parser.add_argument(
        "--move-outlet-distance",
        type=int,
        default=3000,
        help="Max distance in cells that moveoutletstostreams moves reach outlets (default: 3000)"
    )

    parser.add_argument(
        "--dam-move-distance",
        type=int,
        default=200,
        help="Max distance in cells that the dam point is moved down its flow path to the stream "
             "grid; a warning is logged if no stream is reached (default: 200)"
    )

    parser.add_argument(
        "--manning-n",
        type=float,
        default=0.05,
        help="Manning's n written for every reach in catchlist.csv (default: 0.05)"
    )

    parser.add_argument("--np", type=int, default=4,
                        help="Number of MPI processes (default: 4). Reach IDs (LINKNO) from streamnet "
                             "depend on it, so keep it fixed when reusing a forecast.csv")

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
    try:
        workflow = HandDamWorkflow(
            dam_dir=args.dam_dir,
            taudem_path=args.taudem_path,
            output_dir=args.output_dir,
            clipped_dir=args.clipped_dir,
            outlets_on_reaches=args.outlets_on_reaches,
            segment_length=args.segment_length,
            min_segment_length=args.min_segment_length,
            confluence_ratio=args.confluence_ratio,
            min_slope=args.min_slope,
            max_distance=args.max_distance,
            waterbody_path=args.waterbody_path,
            min_waterbody_area_km2=args.min_waterbody_area,
            dam_shapefile=args.dam_shapefile,
            stream_threshold=args.stream_threshold,
            move_outlet_distance=args.move_outlet_distance,
            dam_move_distance=args.dam_move_distance,
            num_processes=args.np,
            manning_n=args.manning_n,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    workflow.run_workflow()


if __name__ == "__main__":
    main()