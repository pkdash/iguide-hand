#!/usr/bin/env python3
"""
Calculate bounding box for all HUC12s downstream from a dam location.

This script:
1. Loads dam location and HUC12 geodatabase
2. Finds the starting HUC12 containing the dam
3. Traverses downstream HUC12s using connectivity data
4. Calculates cumulative distance using sqrt(area) formula
5. Stops at 100km distance or CLOSED BASIN, whichever comes first
6. Computes bounding box in reference raster projection
7. Optionally clips all NHDPlus rasters in a directory to the bounding box
"""

import os
import sys
import argparse
import subprocess
import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import CRS
import logging
from pathlib import Path
from osgeo import gdal

# Logger will be configured in main()
logger = logging.getLogger(__name__)


class HUC12BoundingBoxCalculator:
    """Calculate bounding box for downstream HUC12s from dam location, and optionally clip NHDPlus rasters to that bounding box."""

    def __init__(self, dam_shapefile, huc12_gdb_path, ref_projection_raster_path, output_dir="outputs",
                 raster_dir=None, clipped_dir="clipped", buffer_distance=None):
        """
        Initialize the calculator.

        Args:
            dam_shapefile: Path to dam location shapefile
            huc12_gdb_path: Path to HUC12 geodatabase
            ref_projection_raster_path: Path to reference raster for target projection
            output_dir: Directory for output files
            raster_dir: Directory containing NHDPlus rasters to clip
            clipped_dir: Subdirectory name for clipped NHDPlus rasters
            buffer_distance: Buffer distance in meters around bounding box
        """
        self.dam_shapefile = dam_shapefile
        self.huc12_gdb_path = huc12_gdb_path
        self.ref_projection_raster_path = ref_projection_raster_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Raster clipping parameters
        self.raster_dir = Path(raster_dir) if raster_dir else None
        self.clipped_dir = self.output_dir / clipped_dir
        self.buffer_distance = buffer_distance

        # Target projection from reference raster
        self.target_crs = None
        self.dam_location = None
        self.huc12_data = None
        self.downstream_huc12s = []
        self.total_distance = 0.0
        self.max_distance_km = 100.0

        # Raster clipping results
        self.clipped_rasters = []
        self.failed_rasters = []

    def load_target_projection(self):
        """Load target projection from reference raster."""
        try:
            import rasterio
            with rasterio.open(self.ref_projection_raster_path) as src:
                self.target_crs = src.crs
            logger.info(f"Target CRS loaded: {self.target_crs}")
        except ImportError:
            logger.error("rasterio not available, using GDAL to get projection")
            # Fallback to GDAL
            dataset = gdal.Open(self.ref_projection_raster_path)
            if dataset:
                self.target_crs = CRS.from_wkt(dataset.GetProjection())
                logger.info(f"Target CRS loaded via GDAL: {self.target_crs}")
            else:
                raise ValueError(f"Could not open {self.ref_projection_raster_path}")

    def load_dam_location(self):
        """Load and reproject dam location to target CRS."""
        logger.info("Loading dam location...")
        self.dam_location = gpd.read_file(self.dam_shapefile)

        # Reproject to target CRS
        if self.dam_location.crs != self.target_crs:
            logger.info(f"Reprojecting dam from {self.dam_location.crs} to {self.target_crs}")
            self.dam_location = self.dam_location.to_crs(self.target_crs)

        logger.info(f"Dam location loaded: {len(self.dam_location)} features")
        logger.info(f"Dam coordinates: {self.dam_location.geometry.iloc[0].x:.2f}, {self.dam_location.geometry.iloc[0].y:.2f}")

    def load_huc12_data(self):
        """Load HUC12 data from geodatabase."""
        logger.info("Loading HUC12 data...")
        self.huc12_data = gpd.read_file(self.huc12_gdb_path, layer='WBDHU12')

        # Reproject to target CRS
        if self.huc12_data.crs != self.target_crs:
            logger.info(f"Reprojecting HUC12 from {self.huc12_data.crs} to {self.target_crs}")
            self.huc12_data = self.huc12_data.to_crs(self.target_crs)

        logger.info(f"HUC12 data loaded: {len(self.huc12_data)} features")

        # Create lookup dictionary for faster access
        self.huc12_lookup = {row['huc12']: idx for idx, row in self.huc12_data.iterrows()}
        logger.info("HUC12 lookup dictionary created")

    def find_starting_huc12(self):
        """Find the HUC12 that contains the dam location."""
        logger.info("Finding starting HUC12...")

        dam_point = self.dam_location.geometry.iloc[0]

        # Perform spatial intersection
        containing_huc12s = self.huc12_data[self.huc12_data.geometry.contains(dam_point)]

        if len(containing_huc12s) == 0:
            # If no exact containment, find nearest HUC12
            logger.warning("Dam not contained in any HUC12, finding nearest...")
            distances = self.huc12_data.geometry.distance(dam_point)
            nearest_idx = distances.idxmin()
            starting_huc12 = self.huc12_data.loc[nearest_idx]
            logger.info(f"Nearest HUC12: {starting_huc12['huc12']} (distance: {distances.iloc[nearest_idx]:.2f}m)")
        elif len(containing_huc12s) == 1:
            starting_huc12 = containing_huc12s.iloc[0]
            logger.info(f"Dam contained in HUC12: {starting_huc12['huc12']}")
        else:
            # Multiple containing HUC12s (shouldn't happen but handle it)
            logger.warning(f"Dam contained in multiple HUC12s ({len(containing_huc12s)}), using first one")
            starting_huc12 = containing_huc12s.iloc[0]
            logger.info(f"Selected HUC12: {starting_huc12['huc12']}")

        return starting_huc12

    def calculate_huc12_distance(self, huc12_row):
        """Calculate distance contribution of a HUC12 using sqrt(area) formula."""
        area_sqkm = huc12_row['areasqkm']
        if pd.isna(area_sqkm) or area_sqkm <= 0:
            logger.warning(f"Invalid area for HUC12 {huc12_row['huc12']}: {area_sqkm}")
            return 0.0
        return np.sqrt(area_sqkm)

    def traverse_downstream_huc12s(self):
        """Traverse downstream HUC12s following connectivity until distance limit or closed basin."""
        logger.info("Starting downstream traversal...")

        # Find starting HUC12
        current_huc12 = self.find_starting_huc12()
        self.downstream_huc12s = []
        self.total_distance = 0.0

        visited_huc12s = set()  # Prevent infinite loops

        while current_huc12 is not None:
            current_huc12_id = current_huc12['huc12']

            # Check for infinite loop
            if current_huc12_id in visited_huc12s:
                logger.warning(f"Circular reference detected at HUC12 {current_huc12_id}, stopping")
                break

            visited_huc12s.add(current_huc12_id)

            # Calculate distance contribution
            distance_contribution = self.calculate_huc12_distance(current_huc12)

            # Check if adding this HUC12 would exceed distance limit
            if self.total_distance + distance_contribution > self.max_distance_km:
                logger.info(f"Distance limit ({self.max_distance_km} km) would be exceeded, stopping at HUC12 {current_huc12_id}")
                break

            # Add current HUC12 to downstream list
            self.downstream_huc12s.append(current_huc12)
            self.total_distance += distance_contribution

            logger.info(f"Added HUC12 {current_huc12_id}: {current_huc12['name'][:50]}... "
                       f"(distance: +{distance_contribution:.2f} km, total: {self.total_distance:.2f} km)")

            # Check if this is a terminal basin
            tohuc = current_huc12['tohuc']
            if pd.isna(tohuc) or tohuc == 'CLOSED BASIN' or tohuc == '':
                logger.info(f"Reached terminal basin at HUC12 {current_huc12_id}")
                break

            # Find next HUC12
            if tohuc in self.huc12_lookup:
                next_idx = self.huc12_lookup[tohuc]
                current_huc12 = self.huc12_data.iloc[next_idx]
            else:
                logger.warning(f"Next HUC12 {tohuc} not found in dataset, stopping")
                break

        logger.info(f"Traversal complete: {len(self.downstream_huc12s)} HUC12s, total distance: {self.total_distance:.2f} km")

    def calculate_bounding_box(self):
        """Calculate bounding box of all downstream HUC12s."""
        if not self.downstream_huc12s:
            logger.error("No downstream HUC12s found, cannot calculate bounding box")
            return None

        logger.info("Calculating bounding box...")

        # Create GeoDataFrame from downstream HUC12s
        downstream_gdf = gpd.GeoDataFrame(self.downstream_huc12s, crs=self.target_crs)

        # Calculate total bounds
        minx, miny, maxx, maxy = downstream_gdf.total_bounds

        bounding_box = {
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy,
            'width': maxx - minx,
            'height': maxy - miny,
            'crs': str(self.target_crs)
        }

        logger.info("Bounding box calculated:")
        logger.info(f"  Min X: {minx:.2f}")
        logger.info(f"  Min Y: {miny:.2f}")
        logger.info(f"  Max X: {maxx:.2f}")
        logger.info(f"  Max Y: {maxy:.2f}")
        logger.info(f"  Width: {bounding_box['width']:.2f} m")
        logger.info(f"  Height: {bounding_box['height']:.2f} m")
        logger.info(f"  CRS: {bounding_box['crs']}")

        return bounding_box

    def discover_rasters(self):
        """Discover .tif and .tiff files in the raster directory, excluding auxiliary files."""
        if not self.raster_dir or not self.raster_dir.exists():
            logger.error(f"Raster directory not found: {self.raster_dir}")
            return []

        logger.info(f"Discovering raster files in: {self.raster_dir}")

        # File extensions to include
        raster_extensions = {'.tif', '.tiff'}

        # Patterns to exclude (auxiliary files)
        exclude_patterns = {
            '.aux.xml', '.ovr', '.xml', '.tfw', '.tiff.aux.xml', '.tif.aux.xml',
            '.vat.cpg', '.vat.dbf', '.tif.ovr', '.tiff.ovr'
        }

        raster_files = []

        for file_path in self.raster_dir.iterdir():
            if file_path.is_file():
                # Check if it has a raster extension
                if file_path.suffix.lower() in raster_extensions:
                    # Check if it's not an auxiliary file
                    is_auxiliary = False
                    for exclude_pattern in exclude_patterns:
                        if str(file_path).endswith(exclude_pattern):
                            is_auxiliary = True
                            break

                    if not is_auxiliary:
                        raster_files.append(file_path)
                        logger.debug(f"Found raster: {file_path.name}")

        logger.info(f"Discovered {len(raster_files)} raster files for clipping")
        return sorted(raster_files)

    def get_raster_compression(self, raster_path):
        """Get compression settings from input raster using GDAL."""
        try:
            dataset = gdal.Open(str(raster_path))
            if dataset is None:
                logger.warning(f"Could not open raster: {raster_path}")
                return []

            # Get creation options from the raster
            creation_options = []

            # Try to get compression info
            band = dataset.GetRasterBand(1)
            if band:
                # Check for common compression types
                metadata = dataset.GetMetadata()
                if 'COMPRESSION' in metadata:
                    compression = metadata['COMPRESSION']
                    creation_options.append(f"COMPRESS={compression}")
                    logger.debug(f"Found compression: {compression} for {raster_path.name}")

                # Check if it's tiled
                block_x, block_y = band.GetBlockSize()
                if block_x < dataset.RasterXSize or block_y < dataset.RasterYSize:
                    creation_options.append("TILED=YES")
                    logger.debug(f"Raster is tiled: {raster_path.name}")

            dataset = None  # Close dataset
            return creation_options

        except Exception as e:
            logger.warning(f"Could not get compression info for {raster_path}: {e}")
            return ["COMPRESS=LZW", "TILED=YES"]  # Default fallback

    def clip_rasters(self, bounding_box):
        """Clip all discovered rasters using the calculated bounding box (exact values, no resampling)."""
        if not bounding_box:
            logger.error("No bounding box available for raster clipping")
            return

        if not self.raster_dir:
            logger.error("No raster directory specified for clipping")
            return

        # Discover raster files
        raster_files = self.discover_rasters()
        if not raster_files:
            logger.warning("No raster files found for clipping")
            return

        # Create clipped directory
        self.clipped_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created clipped raster directory: {self.clipped_dir}")

        # Calculate clipping bounds with optional buffer
        minx, miny, maxx, maxy = (
            bounding_box['minx'], bounding_box['miny'],
            bounding_box['maxx'], bounding_box['maxy']
        )

        if self.buffer_distance:
            logger.info(f"Applying buffer of {self.buffer_distance} meters to bounding box")
            minx -= self.buffer_distance
            miny -= self.buffer_distance
            maxx += self.buffer_distance
            maxy += self.buffer_distance

        logger.info(f"Clipping bounds: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

        # Reset clipping results
        self.clipped_rasters = []
        self.failed_rasters = []

        # Process each raster file
        for i, raster_path in enumerate(raster_files, 1):
            logger.info(f"Clipping raster {i}/{len(raster_files)}: {raster_path.name}")

            try:
                # Define output path
                output_path = self.clipped_dir / raster_path.name

                # Get compression settings from input raster
                creation_options = self.get_raster_compression(raster_path)

                # Build gdal_translate command
                cmd_parts = [
                    "gdal_translate",
                    "-projwin", str(minx), str(maxy), str(maxx), str(miny),  # note Y order
                    "-of", "GTiff",  # output format
                    "-co", "TILED=YES"  # ensure tiling for efficiency
                ]

                # Add creation options safely
                for option in creation_options:
                    if "=" not in option:
                        logger.warning(f"Invalid creation option (skipped): {option}")
                        continue
                    cmd_parts.extend(["-co", option])

                # Input and output paths
                cmd_parts.extend([str(raster_path), str(output_path)])

                # Execute gdal_translate command
                subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Check if output file was created
                if output_path.exists():
                    self.clipped_rasters.append({
                        'input': raster_path.name,
                        'output': output_path.name,
                        'size_mb': output_path.stat().st_size / (1024 * 1024)
                    })
                    logger.info(f"Successfully clipped: {raster_path.name} -> {output_path.name}")
                else:
                    raise FileNotFoundError(f"Output file not created: {output_path}")

            except subprocess.CalledProcessError as e:
                error_msg = f"gdal_translate failed for {raster_path.name}: {e.stderr}"
                logger.error(error_msg)
                self.failed_rasters.append({
                    'input': raster_path.name,
                    'error': error_msg
                })

            except Exception as e:
                error_msg = f"Unexpected error clipping {raster_path.name}: {str(e)}"
                logger.error(error_msg)
                self.failed_rasters.append({
                    'input': raster_path.name,
                    'error': error_msg
                })

        # Log summary
        logger.info(f"Raster clipping complete: {len(self.clipped_rasters)} successful, {len(self.failed_rasters)} failed")

        if self.clipped_rasters:
            total_size = sum(r['size_mb'] for r in self.clipped_rasters)
            logger.info(f"Total size of clipped rasters: {total_size:.1f} MB")

        if self.failed_rasters:
            logger.warning(f"Failed to clip {len(self.failed_rasters)} rasters")
            for failed in self.failed_rasters:
                logger.warning(f"  - {failed['input']}: {failed['error']}")

    def save_outputs(self, bounding_box):
        """Save downstream HUC12s shapefile and processing report."""
        logger.info("Saving outputs...")

        # Save downstream HUC12s as shapefile
        if self.downstream_huc12s:
            downstream_gdf = gpd.GeoDataFrame(self.downstream_huc12s, crs=self.target_crs)
            downstream_shapefile = self.output_dir / "downstream_huc12s_from_dam.shp"
            downstream_gdf.to_file(downstream_shapefile)
            logger.info(f"Downstream HUC12s saved to: {downstream_shapefile}")

        # Save bounding box coordinates
        if bounding_box:
            bbox_file = self.output_dir / "bounding_box_coordinates.txt"
            with open(bbox_file, 'w') as f:
                f.write("Bounding Box for Downstream HUC12s from Dam Location\n")
                f.write("=" * 55 + "\n\n")
                f.write(f"Coordinate Reference System: {bounding_box['crs']}\n\n")
                f.write(f"Min X: {bounding_box['minx']:.6f}\n")
                f.write(f"Min Y: {bounding_box['miny']:.6f}\n")
                f.write(f"Max X: {bounding_box['maxx']:.6f}\n")
                f.write(f"Max Y: {bounding_box['maxy']:.6f}\n\n")
                f.write(f"Width: {bounding_box['width']:.2f} meters\n")
                f.write(f"Height: {bounding_box['height']:.2f} meters\n\n")
                f.write(f"Number of HUC12s: {len(self.downstream_huc12s)}\n")
                f.write(f"Total Distance: {self.total_distance:.2f} km\n")
            logger.info(f"Bounding box coordinates saved to: {bbox_file}")

        # Save detailed processing report
        report_file = self.output_dir / "processing_report.txt"
        with open(report_file, 'w') as f:
            f.write("HUC12 Downstream Analysis Report\n")
            f.write("=" * 35 + "\n\n")
            f.write(f"Dam Location File: {self.dam_shapefile}\n")
            f.write(f"HUC12 Database: {self.huc12_gdb_path}\n")
            f.write(f"Target Projection: {self.target_crs}\n")
            f.write(f"Maximum Distance: {self.max_distance_km} km\n")
            if self.raster_dir:
                f.write(f"Raster Directory: {self.raster_dir}\n")
                f.write(f"Clipped Raster Directory: {self.clipped_dir}\n")
                if self.buffer_distance:
                    f.write(f"Buffer Distance: {self.buffer_distance} meters\n")
            f.write("\n")

            f.write("Analysis Results:\n")
            f.write(f"  Number of downstream HUC12s: {len(self.downstream_huc12s)}\n")
            f.write(f"  Total distance traveled: {self.total_distance:.2f} km\n")
            if self.clipped_rasters or self.failed_rasters:
                f.write(f"  Clipped rasters: {len(self.clipped_rasters)} successful, {len(self.failed_rasters)} failed\n")
            f.write("\n")

            if bounding_box:
                f.write("Bounding Box (in target projection):\n")
                f.write(f"  Min X: {bounding_box['minx']:.6f}\n")
                f.write(f"  Min Y: {bounding_box['miny']:.6f}\n")
                f.write(f"  Max X: {bounding_box['maxx']:.6f}\n")
                f.write(f"  Max Y: {bounding_box['maxy']:.6f}\n")
                f.write(f"  Width: {bounding_box['width']:.2f} m\n")
                f.write(f"  Height: {bounding_box['height']:.2f} m\n")
                if self.buffer_distance:
                    f.write(f"  Buffer applied: {self.buffer_distance} m\n")
                f.write("\n")

            # Raster clipping results
            if self.clipped_rasters:
                f.write("Successfully Clipped Rasters:\n")
                f.write("-" * 30 + "\n")
                total_size = 0
                for i, raster in enumerate(self.clipped_rasters, 1):
                    f.write(f"{i:2d}. {raster['input']} -> {raster['output']} ({raster['size_mb']:.1f} MB)\n")
                    total_size += raster['size_mb']
                f.write(f"\nTotal clipped raster size: {total_size:.1f} MB\n\n")

            if self.failed_rasters:
                f.write("Failed Raster Clipping:\n")
                f.write("-" * 25 + "\n")
                for i, failed in enumerate(self.failed_rasters, 1):
                    f.write(f"{i:2d}. {failed['input']}: {failed['error']}\n")
                f.write("\n")

            f.write("Downstream HUC12 Details:\n")
            f.write("-" * 25 + "\n")
            for i, huc12 in enumerate(self.downstream_huc12s, 1):
                distance_contrib = self.calculate_huc12_distance(huc12)
                f.write(f"{i:2d}. HUC12: {huc12['huc12']} | "
                       f"Name: {huc12['name'][:40]:<40} | "
                       f"Area: {huc12['areasqkm']:8.2f} km² | "
                       f"Distance: {distance_contrib:6.2f} km | "
                       f"ToHUC: {huc12['tohuc']}\n")

        logger.info(f"Processing report saved to: {report_file}")

    def run_analysis(self, clip_rasters=False):
        """Run the complete analysis."""
        logger.info("Starting HUC12 bounding box analysis...")

        try:
            # Load data
            self.load_target_projection()
            self.load_dam_location()
            self.load_huc12_data()

            # Perform analysis
            self.traverse_downstream_huc12s()
            bounding_box = self.calculate_bounding_box()

            # Clip rasters if requested
            if clip_rasters and bounding_box:
                logger.info("Starting raster clipping...")
                self.clip_rasters(bounding_box)

            # Save outputs
            self.save_outputs(bounding_box)

            logger.info("Analysis completed successfully!")
            return bounding_box

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate bounding box for all HUC12s downstream from a dam location.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Use default files (bounding box only)
            python %(prog)s

            # Specify custom dam location
            python %(prog)s --dam-shapefile my_dam.shp

            # Change maximum distance and output directory
            python %(prog)s --max-distance 50 --output-dir results

            # Enable raster clipping with calculated bounding box
            python %(prog)s --clip-rasters

            # Raster clipping with 1km buffer around bounding box
            python %(prog)s --clip-rasters --buffer 1000

            # Custom raster directories and clipping
            python %(prog)s --clip-rasters --raster-dir /path/to/rasters --clipped-dir my_clipped

            # Combined: custom distance, clipping with buffer, and custom output
            python %(prog)s --max-distance 75 --clip-rasters --buffer 500 --output-dir results

            # Use different HUC12 database and reference raster
            python %(prog)s --huc12-gdb /path/to/huc12.gdb --ref-projection-raster /path/to/dem.tif
        """
    )

    parser.add_argument(
        "--dam-shapefile",
        default="data/mountain_dell_dam_location.shp",
        help="Path to dam location shapefile (default: %(default)s)"
    )

    parser.add_argument(
        "--huc12-gdb",
        default="data/NHDPLUS_H_1602_HU4_20220412_GDB/NHDPLUS_H_1602_HU4_20220412_GDB.gdb",
        help="Path to HUC12 geodatabase (default: %(default)s)"
    )

    parser.add_argument(
        "--ref-projection-raster",
        default="data/NHDPLUS_H_1602_HU4_20220412_RASTER/hydrodem.tif",
        help="Path to reference raster for target projection (default: %(default)s)"
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for results (default: %(default)s)"
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        help="Maximum distance in kilometers to traverse downstream (default: %(default)s)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )

    # Raster clipping arguments
    parser.add_argument(
        "--clip-rasters",
        action="store_true",
        help="Enable raster clipping using calculated bounding box"
    )

    parser.add_argument(
        "--raster-dir",
        default="data/NHDPLUS_H_1602_HU4_20220412_RASTER", # Default raster directory
        help="Directory containing rasters to clip (default: %(default)s)"
    )

    parser.add_argument(
        "--clipped-dir",
        default="clipped",
        help="Subdirectory name within output-dir for clipped rasters (default: %(default)s)"
    )

    parser.add_argument(
        "--buffer",
        type=float,
        help="Buffer distance in meters to add around bounding box (optional)"
    )

    return parser.parse_args()


def setup_logging(verbose=False, quiet=False):
    """Setup logging configuration."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main execution function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose, args.quiet)

    # Check if files exist
    for filepath in [args.dam_shapefile, args.huc12_gdb, args.ref_projection_raster]:
        if not os.path.exists(filepath):
            logger.error(f"Required file not found: {filepath}")
            sys.exit(1)

    # Create calculator and run analysis
    calculator = HUC12BoundingBoxCalculator(
        dam_shapefile=args.dam_shapefile,
        huc12_gdb_path=args.huc12_gdb,
        ref_projection_raster_path=args.ref_projection_raster,
        output_dir=args.output_dir,
        raster_dir=args.raster_dir if args.clip_rasters else None,
        clipped_dir=args.clipped_dir,
        buffer_distance=args.buffer
    )

    # Set maximum distance
    calculator.max_distance_km = args.max_distance

    try:
        bounding_box = calculator.run_analysis(clip_rasters=args.clip_rasters)

        if bounding_box and not args.quiet:
            print("\n" + "="*60)
            print("BOUNDING BOX RESULTS")
            print("="*60)
            print(f"Coordinate System: {bounding_box['crs']}")
            print(f"Min X: {bounding_box['minx']:.6f}")
            print(f"Min Y: {bounding_box['miny']:.6f}")
            print(f"Max X: {bounding_box['maxx']:.6f}")
            print(f"Max Y: {bounding_box['maxy']:.6f}")
            print(f"Width: {bounding_box['width']:.2f} meters")
            print(f"Height: {bounding_box['height']:.2f} meters")
            if calculator.buffer_distance:
                print(f"Buffer Applied: {calculator.buffer_distance} meters")
            print(f"Number of HUC12s: {len(calculator.downstream_huc12s)}")
            print(f"Total Distance: {calculator.total_distance:.2f} km")

            # Add raster clipping results
            if args.clip_rasters:
                print("\nRaster Clipping Results:")
                print(f"Successfully clipped: {len(calculator.clipped_rasters)} rasters")
                if calculator.failed_rasters:
                    print(f"Failed to clip: {len(calculator.failed_rasters)} rasters")
                if calculator.clipped_rasters:
                    total_size = sum(r['size_mb'] for r in calculator.clipped_rasters)
                    print(f"Total clipped size: {total_size:.1f} MB")
                    print(f"Output directory: {calculator.clipped_dir}")

            print("="*60)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()