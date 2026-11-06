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
from utils import (setup_taudem_path, run_taudem_command, find_dam_shapefile,
                   find_nhdplus_gdb, default_output_dir, read_run_info)
from generate_segment_outlets import SegmentOutletGenerator, default_waterbody_path

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
                 stream_threshold=10000, move_outlet_distance=3000, num_processes=4, manning_n=0.05):
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
            min_slope: Reach slopes below this value are raised to it in catchlist.csv
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
                              else Path(dam_shapefile).expanduser().resolve())
        self.stream_threshold = stream_threshold
        self.move_outlet_distance = move_outlet_distance
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

        # Move outlets to streams
        self.run_taudem_command(
            (
                f"mpiexec -n {self.num_processes} moveoutletstostreams "
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


    def create_catchment_list(self):
        """Create catchment list CSV from stream network shapefile."""
        logger.info("Creating catchment list...")

        net_shp = self.output_dir / "net2.shp"
        output_csv = self.output_dir / "catchlist.csv"

        # Read stream network shapefile
        net_gdf = gpd.read_file(net_shp)

        # Filter rows where DSNODEID == 0 (internal nodes)
        filtered_gdf = net_gdf[net_gdf['DSNODEID'] == 0]

        # Create catchment list dataframe
        catchlist_data = []
        for _, row in filtered_gdf.iterrows():
            slope = row['Slope']
            if self.min_slope is not None and slope < self.min_slope:
                logger.warning(
                    f"Reach {row['LINKNO']}: slope {slope:.2e} below minimum; "
                    f"set to {self.min_slope:.2e}"
                )
                slope = self.min_slope
            catchlist_data.append({
                'Id': row['LINKNO'],
                'Slope': slope,
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
        help="Dam location shapefile (default: dam_location.shp, or the single "
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
        help="Reach slopes below this are raised to it in catchlist.csv (default: 1e-5)"
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
            num_processes=args.np,
            manning_n=args.manning_n,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    workflow.run_workflow()


if __name__ == "__main__":
    main()