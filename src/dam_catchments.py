#!/usr/bin/env python3
"""
Calculate bounding box for all HUC12s downstream from a dam location.

This script:
1. Loads dam location and HUC12 geodatabase
2. Finds the starting HUC12 containing the dam
3. Traverses downstream HUC12s using connectivity data
4. Calculates cumulative distance using sqrt(area) formula
5. Stops at --max-distance (default 100 km), a CLOSED BASIN, or the edge of the HU4
   GDB (with a warning), whichever comes first
6. Computes bounding box in reference raster projection
7. Optionally clips all NHDPlus rasters in a directory to the bounding box
"""

import sys
import argparse
import subprocess
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import rasterio

from utils import (find_dam_shapefile, find_nhdplus_gdb, find_nhdplus_raster_dir,
                   default_output_dir, write_run_info)

# Logger will be configured in main()
logger = logging.getLogger(__name__)

# A dam outside every HUC12 but within this distance (e.g. on a boundary) starts from the
# nearest HUC12; farther away it is almost certainly in a different HU4 than --huc12-gdb
MAX_DAM_HUC12_OFFSET_M = 1000.0


class HUC12BoundingBoxCalculator:
    """
    Calculate bounding box for downstream HUC12s from dam location, and optionally clip NHDPlus
    rasters to that bounding box.
    """

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
        # Convert all paths to absolute Path objects
        self.dam_shapefile = Path(dam_shapefile).resolve()
        self.huc12_gdb_path = Path(huc12_gdb_path).resolve()
        self.ref_projection_raster_path = Path(ref_projection_raster_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Raster clipping parameters
        self.raster_dir = Path(raster_dir).resolve() if raster_dir else None
        self.clipped_dir = self.output_dir / clipped_dir
        self.buffer_distance = buffer_distance

        # Target projection from reference raster
        self.target_crs = None
        self.dam_location = None
        self.huc12_data = None
        self.downstream_huc12s = []
        self.total_distance = 0.0
        self.max_distance_km = 100.0
        self.stop_reason = None

        # Raster clipping results
        self.clipped_rasters = []
        self.failed_rasters = []

        # Key names for HUC12 data (set during load)
        self.huc12_key_name_huc12 = 'huc12'
        self.huc12_key_name_name = 'name'
        self.huc12_key_name_areasqkm = 'areasqkm'
        self.huc12_key_name_tohuc = 'tohuc'

    def load_target_projection(self):
        """Load target projection from reference raster."""
        try:
            with rasterio.open(str(self.ref_projection_raster_path)) as src:
                self.target_crs = src.crs
            logger.info(f"Target CRS loaded: {self.target_crs}")
        except Exception as e:
            raise ValueError(f"Could not open {self.ref_projection_raster_path}: {str(e)}")

    def load_dam_location(self):
        """Load and reproject dam location to target CRS."""
        logger.info("Loading dam location...")
        self.dam_location = gpd.read_file(str(self.dam_shapefile))

        # Reproject to target CRS
        if self.dam_location.crs != self.target_crs:
            logger.info(f"Reprojecting dam from {self.dam_location.crs} to {self.target_crs}")
            self.dam_location = self.dam_location.to_crs(self.target_crs)

        logger.info(f"Dam location loaded: {len(self.dam_location)} features")
        logger.info(
            f"Dam coordinates: "
            f"{self.dam_location.geometry.iloc[0].x:.2f}, "
            f"{self.dam_location.geometry.iloc[0].y:.2f}"
        )

    def load_huc12_data(self):
        """Load HUC12 data from geodatabase."""
        logger.info("Loading HUC12 data...")
        self.huc12_data = gpd.read_file(str(self.huc12_gdb_path), layer='WBDHU12')

        # Reproject to target CRS
        if self.huc12_data.crs != self.target_crs:
            logger.info(f"Reprojecting HUC12 from {self.huc12_data.crs} to {self.target_crs}")
            self.huc12_data = self.huc12_data.to_crs(self.target_crs)

        logger.info(f"HUC12 data loaded: {len(self.huc12_data)} features")

        # Set the key names from the first row of HUC4 gdb data (WBDHU12 layer)
        # Not sure why the keys are not all lowercase for different HUC4 gdb data sets
        first_row = self.huc12_data.iloc[0]
        key_reset_count = 0
        for key in first_row.keys():
            if 'huc12' == key.lower():
                self.huc12_key_name_huc12 = key
                key_reset_count += 1
            elif 'name' == key.lower():
                self.huc12_key_name_name = key
                key_reset_count += 1
            elif 'areasqkm' == key.lower():
                self.huc12_key_name_areasqkm = key
                key_reset_count += 1
            elif 'tohuc' == key.lower():
                self.huc12_key_name_tohuc = key
                key_reset_count += 1
            if key_reset_count == 4:
                break

        # Create lookup dictionary for faster access
        self.huc12_lookup = {row[self.huc12_key_name_huc12]: idx for idx, row in self.huc12_data.iterrows()}
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
            nearest_dist = distances.min()
            starting_huc12 = self.huc12_data.loc[nearest_idx]
            if nearest_dist > MAX_DAM_HUC12_OFFSET_M:
                raise ValueError(
                    f"Dam is {nearest_dist / 1000:.1f} km from the nearest HUC12 "
                    f"({starting_huc12[self.huc12_key_name_huc12]}) in {self.huc12_gdb_path.name}; "
                    "it is probably in a different HU4. Pass the --huc12-gdb and --raster-dir "
                    "for the HU4 that contains the dam."
                )
            logger.info(
                f"Nearest HUC12: {starting_huc12[self.huc12_key_name_huc12]} "
                f"(distance: {nearest_dist:.2f}m)"
            )
        elif len(containing_huc12s) == 1:
            starting_huc12 = containing_huc12s.iloc[0]
            logger.info(f"Dam contained in HUC12: {starting_huc12[self.huc12_key_name_huc12]}")
        else:
            # Multiple containing HUC12s (shouldn't happen but handle it)
            logger.warning(f"Dam contained in multiple HUC12s ({len(containing_huc12s)}), using first one")
            starting_huc12 = containing_huc12s.iloc[0]
            logger.info(f"Selected HUC12: {starting_huc12[self.huc12_key_name_huc12]}")

        return starting_huc12

    def calculate_huc12_distance(self, huc12_row):
        """Calculate distance contribution of a HUC12 using sqrt(area) formula."""
        area_sqkm = huc12_row[self.huc12_key_name_areasqkm]
        if pd.isna(area_sqkm) or area_sqkm <= 0:
            logger.warning(f"Invalid area for HUC12 {huc12_row[self.huc12_key_name_huc12]}: {area_sqkm}")
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
            current_huc12_id = current_huc12[self.huc12_key_name_huc12]

            # Check for infinite loop
            if current_huc12_id in visited_huc12s:
                logger.warning(f"Circular reference detected at HUC12 {current_huc12_id}, stopping")
                self.stop_reason = f"circular tohuc reference at HUC12 {current_huc12_id}"
                break

            visited_huc12s.add(current_huc12_id)

            # Calculate distance contribution
            distance_contribution = self.calculate_huc12_distance(current_huc12)

            # Check if adding this HUC12 would exceed distance limit
            if self.total_distance + distance_contribution > self.max_distance_km:
                logger.info(
                    f"Distance limit ({self.max_distance_km} km) would be exceeded, "
                    f"stopping at HUC12 {current_huc12_id}"
                )
                self.stop_reason = (f"distance limit ({self.max_distance_km:g} km) "
                                    f"reached before HUC12 {current_huc12_id}")
                break

            # Add current HUC12 to downstream list
            self.downstream_huc12s.append(current_huc12)
            self.total_distance += distance_contribution

            logger.info(
                f"Added HUC12 {current_huc12_id}: {current_huc12[self.huc12_key_name_name][:50]}... "
                f"(distance: +{distance_contribution:.2f} km, total: {self.total_distance:.2f} km)"
            )

            # Check if this is a terminal basin
            tohuc = current_huc12[self.huc12_key_name_tohuc]
            if pd.isna(tohuc) or tohuc == 'CLOSED BASIN' or tohuc == '':
                logger.info(f"Reached terminal basin at HUC12 {current_huc12_id}")
                self.stop_reason = f"terminal basin at HUC12 {current_huc12_id} (tohuc: {tohuc})"
                break

            # Find next HUC12
            if tohuc in self.huc12_lookup:
                next_idx = self.huc12_lookup[tohuc]
                current_huc12 = self.huc12_data.iloc[next_idx]
            else:
                next_hu4 = str(tohuc)[:4]
                self.stop_reason = (
                    f"left the HU4 GDB: next HUC12 {tohuc} (HU4 {next_hu4}) is not in "
                    f"{self.huc12_gdb_path.name}"
                )
                logger.warning("=" * 70)
                logger.warning(
                    f"Downstream walk left the HU4 dataset: HUC12 {current_huc12_id} drains to "
                    f"{tohuc} (HU4 {next_hu4}), which is not in {self.huc12_gdb_path.name}."
                )
                logger.warning(
                    f"Domain is truncated at {self.total_distance:.2f} km of the "
                    f"{self.max_distance_km:g} km limit. Only a single HU4 is supported; "
                    f"results stop at the HU4 boundary."
                )
                logger.warning("=" * 70)
                break

        logger.info(
            f"Traversal complete: {len(self.downstream_huc12s)} HUC12s, "
            f"total distance: {self.total_distance:.2f} km, stopped: {self.stop_reason}"
        )


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
        """Get compression settings from input raster using rasterio."""
        try:
            with rasterio.open(str(raster_path)) as src:
                # Get creation options from the raster
                creation_options = []

                # Check for common compression types
                if hasattr(src, 'compression') and src.compression:
                    creation_options.append(f"COMPRESS={src.compression.name}")
                    logger.debug(f"Found compression: {src.compression.name} for {raster_path.name}")

                # Check if it's tiled
                if src.block_shapes and src.block_shapes[0] != (src.height, src.width):
                    creation_options.append("TILED=YES")
                    logger.debug(f"Raster is tiled: {raster_path.name}")

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
                error_msg = f"Error clipping {raster_path.name}: {str(e)}"
                logger.error(error_msg)
                self.failed_rasters.append({
                    'input': raster_path.name,
                    'error': error_msg
                })

        # Log summary
        logger.info(
            f"Raster clipping complete: {len(self.clipped_rasters)} successful, "
            f"{len(self.failed_rasters)} failed"
        )

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
            f.write(f"  Stop reason: {self.stop_reason}\n")
            if self.clipped_rasters or self.failed_rasters:
                f.write(
                    f"  Clipped rasters: {len(self.clipped_rasters)} successful, "
                    f"{len(self.failed_rasters)} failed\n"
                )
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
                f.write(f"{i:2d}. HUC12: {huc12[self.huc12_key_name_huc12]} | "
                       f"Name: {huc12[self.huc12_key_name_name][:40]:<40} | "
                       f"Area: {huc12[self.huc12_key_name_areasqkm]:8.2f} km² | "
                       f"Distance: {distance_contrib:6.2f} km | "
                       f"ToHUC: {huc12[self.huc12_key_name_tohuc]}\n")

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
            write_run_info(
                self.output_dir,
                dam_shapefile=self.dam_shapefile,
                huc12_gdb=self.huc12_gdb_path,
                raster_dir=self.raster_dir,
                clipped_dir=self.clipped_dir if self.raster_dir else None,
            )

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
            # data/my_dam holds the dam shapefile plus its own NHDPlus HR HU4 data:
            #   NHDPLUS_H_XXXX_HU4_GDB/NHDPLUS_H_XXXX_HU4_GDB.gdb and NHDPLUS_H_XXXX_HU4_RASTER/
            # Bounding box + clipped rasters into outputs/my_dam/clipped
            python %(prog)s --dam-dir data/my_dam

            # Change maximum distance, buffer, and output directory
            python %(prog)s --dam-dir data/my_dam --max-distance 50 --buffer 1000 \\
                --output-dir results/my_dam

            # Use NHDPlus data stored elsewhere
            python %(prog)s --dam-dir data/my_dam --huc12-gdb /path/to/hu4.gdb --raster-dir /path/to/hu4_rasters

            # Bounding box only, no raster clipping
            python %(prog)s --dam-dir data/my_dam --no-clip-rasters
        """
    )

    parser.add_argument(
        "--dam-dir",
        required=True,
        help="Per-dam input directory, e.g. data/<dam>, holding dam_location.shp "
             "(or a single *dam_location.shp) and the dam's NHDPlus HR HU4 GDB and raster folder"
    )

    parser.add_argument(
        "--dam-shapefile",
        default=None,
        help="Dam location shapefile (default: found in --dam-dir)"
    )

    parser.add_argument(
        "--huc12-gdb",
        default=None,
        help="NHDPlus HR HU4 geodatabase with the WBDHU12 layer for the dam's HU4 "
             "(default: the single *.gdb in --dam-dir or one folder down)"
    )

    parser.add_argument(
        "--raster-dir",
        default=None,
        help="NHDPlus HR HU4 raster directory (hydrodem.tif, fdr.tif, ...) to clip "
             "(default: the single subfolder of --dam-dir containing hydrodem.tif)"
    )

    parser.add_argument(
        "--ref-projection-raster",
        default=None,
        help="Reference raster for the target projection (default: <raster-dir>/hydrodem.tif)"
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: outputs/<dam-dir name>)"
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
        "--no-clip-rasters",
        dest="clip_rasters",
        action="store_false",
        help="Only compute the bounding box; do not clip the rasters in --raster-dir"
    )

    parser.add_argument(
        "--clipped-dir",
        default="clipped",
        help="Subdirectory name within output-dir for clipped rasters (default: %(default)s)"
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=4000,
        help="Buffer distance in meters to add around bounding box (default: %(default)s)"
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

    # Convert paths to absolute and check if files exist
    dam_dir = Path(args.dam_dir).expanduser().resolve()
    if not dam_dir.is_dir():
        logger.error(f"Dam directory not found: {dam_dir}")
        sys.exit(1)
    try:
        dam_shapefile = (find_dam_shapefile(dam_dir) if args.dam_shapefile is None
                         else Path(args.dam_shapefile).expanduser().resolve())
        huc12_gdb = (find_nhdplus_gdb(dam_dir) if args.huc12_gdb is None
                     else Path(args.huc12_gdb).expanduser().resolve())
        raster_dir = (find_nhdplus_raster_dir(dam_dir) if args.raster_dir is None
                      else Path(args.raster_dir).expanduser().resolve())
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    ref_projection_raster = (raster_dir / "hydrodem.tif" if args.ref_projection_raster is None
                             else Path(args.ref_projection_raster).expanduser().resolve())
    output_dir = (default_output_dir(dam_dir) if args.output_dir is None
                  else Path(args.output_dir).expanduser().resolve())

    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Dam directory: {dam_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dam shapefile: {dam_shapefile}")
    logger.info(f"HUC12 GDB: {huc12_gdb}")
    logger.info(f"Reference raster: {ref_projection_raster}")

    # Validate required files exist
    for filepath in [dam_shapefile, huc12_gdb, raster_dir, ref_projection_raster]:
        if not filepath.exists():
            logger.error(f"Required file not found: {filepath}")
            logger.error(f"  Absolute path: {filepath.absolute()}")
            logger.error(f"  Current working directory: {Path.cwd()}")
            sys.exit(1)

    # Create calculator and run analysis
    calculator = HUC12BoundingBoxCalculator(
        dam_shapefile=dam_shapefile,
        huc12_gdb_path=huc12_gdb,
        ref_projection_raster_path=ref_projection_raster,
        output_dir=output_dir,
        raster_dir=raster_dir if args.clip_rasters else None,
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
            print(f"Stopped: {calculator.stop_reason}")

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
