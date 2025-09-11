#!/usr/bin/env python3
"""
Generate flowlines from a dam location using GDAL, TauDEM, and NHD data.

This script implements a complete workflow to:
1. Find HUC12 for dam location
2. Build downstream HUC12 connectivity network
3. Process DEM data with TauDEM
4. Generate flowlines and stream networks

Author: Hydrologist Assistant
Date: 2025-09-10
"""

import os
import sys
import subprocess
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlowlineGenerator:
    """Main class for generating flowlines from dam location."""

    def __init__(self, dam_shapefile, nhd_gdb, dem_file, output_dir):
        self.dam_shapefile = dam_shapefile
        self.nhd_gdb = nhd_gdb
        self.dem_file = dem_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set TauDEM path
        os.environ['PATH'] = f"{os.path.expanduser('~/taudem')}:{os.environ['PATH']}"

        # Initialize data containers
        self.dam_location = None
        self.dam_huc12 = None
        self.downstream_hucs = []
        self.target_extent = None
        self.huc12_data = None
        self.target_crs = "EPSG:26912"  # UTM Zone 12N

    def load_dam_location(self):
        """Load dam location from shapefile."""
        logger.info("Loading dam location...")
        try:
            self.dam_location = gpd.read_file(self.dam_shapefile)
            if len(self.dam_location) == 0:
                raise ValueError("No features found in dam shapefile")

            # Get the first (and should be only) dam point
            dam_point = self.dam_location.iloc[0]
            logger.info(f"Dam loaded: {dam_point.get('Dam_Name', 'Unknown')} at {dam_point.geometry}")
            return True

        except Exception as e:
            logger.error(f"Error loading dam location: {e}")
            return False

    def find_dam_huc12(self):
        """Find the HUC12 that contains the dam location."""
        logger.info("Finding HUC12 for dam location...")
        try:
            # Load HUC12 boundaries
            self.huc12_data = gpd.read_file(self.nhd_gdb, layer='WBDHU12')
            huc12_gdf = self.huc12_data

            # Ensure same CRS
            if self.dam_location.crs != huc12_gdf.crs:
                dam_reproj = self.dam_location.to_crs(huc12_gdf.crs)
            else:
                dam_reproj = self.dam_location

            # Find intersecting HUC12
            dam_point = dam_reproj.iloc[0].geometry
            intersecting = huc12_gdf[huc12_gdf.intersects(dam_point)]

            if len(intersecting) == 0:
                logger.error("Dam location does not intersect any HUC12")
                return False

            self.dam_huc12 = intersecting.iloc[0]
            logger.info(f"Dam is in HUC12: {self.dam_huc12['huc12']} - {self.dam_huc12['name']}")

            # Save dam HUC12 for reference
            dam_huc_file = self.output_dir / "dam_huc12.shp"
            intersecting.to_file(dam_huc_file)
            logger.info(f"Dam HUC12 saved to: {dam_huc_file}")

            return True

        except Exception as e:
            logger.error(f"Error finding dam HUC12: {e}")
            return False

    def check_dam_in_nhd_points(self):
        """Check if dam location exists in NHD points database."""
        logger.info("Checking if dam exists in NHD points...")
        try:
            # Load NHD points
            nhd_points = gpd.read_file(self.nhd_gdb, layer='NHDPoint')

            # Ensure same CRS
            if self.dam_location.crs != nhd_points.crs:
                dam_reproj = self.dam_location.to_crs(nhd_points.crs)
            else:
                dam_reproj = self.dam_location

            # Create buffer around dam (100m)
            dam_buffer = dam_reproj.buffer(100)  # 100m buffer

            # Find points within buffer
            nearby_points = nhd_points[nhd_points.intersects(dam_buffer.iloc[0])]

            if len(nearby_points) > 0:
                logger.info(f"Found {len(nearby_points)} NHD points near dam location")
                # Save nearby points
                nearby_file = self.output_dir / "nearby_nhd_points.shp"
                nearby_points.to_file(nearby_file)
                logger.info(f"Nearby NHD points saved to: {nearby_file}")
            else:
                logger.info("No NHD points found near dam location")

            return True

        except Exception as e:
            logger.error(f"Error checking NHD points: {e}")
            return False

    def build_downstream_huc12_network(self, max_distance_km=100):
        """Build downstream HUC12 network using NHDPlus connectivity."""
        logger.info(f"Building downstream HUC12 network (max {max_distance_km}km)...")
        try:
            if self.dam_huc12 is None:
                logger.error("Dam HUC12 not identified. Run find_dam_huc12 first.")
                return False

            all_hucs = self.huc12_data

            # Create a dictionary for quick HUC lookup
            huc_dict = {huc['huc12']: huc for _, huc in all_hucs.iterrows()}

            downstream_hucs = []
            visited_hucs = set()
            current_huc_code = self.dam_huc12['huc12']
            cumulative_distance = 0.0

            while current_huc_code and current_huc_code not in visited_hucs and cumulative_distance < max_distance_km:
                visited_hucs.add(current_huc_code)

                if current_huc_code not in huc_dict:
                    logger.warning(f"HUC {current_huc_code} not found in HUC data.")
                    break

                current_huc = huc_dict[current_huc_code]
                downstream_hucs.append(current_huc)

                # Estimate distance traveled by approximating HUC diameter in km
                cumulative_distance += np.sqrt(current_huc.get('areasqkm', 0))

                to_huc_code = current_huc.get('tohuc')
                if not to_huc_code or to_huc_code == '000000000000' or 'CLOSED BASIN' in to_huc_code:
                    logger.info(f"Reached end of chain at HUC {current_huc_code} (ToHUC: {to_huc_code}).")
                    break

                current_huc_code = to_huc_code

            if downstream_hucs:
                self.downstream_hucs = gpd.GeoDataFrame(downstream_hucs, crs=all_hucs.crs)
                downstream_file = self.output_dir / "downstream_huc12s.shp"
                self.downstream_hucs.to_file(downstream_file)
                logger.info(f"Found {len(self.downstream_hucs)} downstream HUC12s.")
                logger.info(f"Downstream HUC12s saved to: {downstream_file}")
                return True
            else:
                logger.error("No downstream HUCs found.")
                return False

        except Exception as e:
            logger.error(f"Error building downstream network: {e}")
            import traceback; traceback.print_exc()
            return False

    def create_study_area_extent(self, buffer_km=2):
        """Create buffered extent around downstream HUCs for raster processing."""
        logger.info(f"Creating study area with {buffer_km}km buffer...")
        try:
            if self.downstream_hucs is None or len(self.downstream_hucs) == 0:
                logger.error("Downstream HUCs list is empty. Cannot create study area.")
                return False

            # Reproject to a projected CRS for buffering
            downstream_utm = self.downstream_hucs.to_crs(self.target_crs)

            # Create a unified geometry of all HUCs
            combined_hucs = downstream_utm.unary_union

            # Buffer the combined geometry
            buffer_m = buffer_km * 1000
            study_area = combined_hucs.buffer(buffer_m)

            # Create a GeoDataFrame for the study area
            study_area_gdf = gpd.GeoDataFrame({'id': [1]},
                                              geometry=[study_area],
                                              crs=self.target_crs)

            # Save study area
            study_area_file = self.output_dir / "study_area.shp"
            study_area_gdf.to_file(study_area_file)
            logger.info(f"Study area saved to: {study_area_file}")

            # Get extent for raster processing
            bounds = study_area.bounds
            self.target_extent = {
                'xmin': bounds[0],
                'ymin': bounds[1],
                'xmax': bounds[2],
                'ymax': bounds[3]
            }
            logger.info(f"Study area extent: {self.target_extent}")
            return True

        except Exception as e:
            logger.error(f"Error creating study area: {e}")
            return False

    def prepare_dem_data(self, target_resolution=30):
        """Prepare DEM data for TauDEM processing."""
        logger.info("Preparing DEM data...")
        try:
            # Check if we have target extent
            if self.target_extent is None:
                logger.error("No target extent defined")
                return False

            dem_subset_file = self.output_dir / "dem_subset.tif"

            # Use gdalwarp to reproject, clip, and set resolution in one step
            logger.info("Reprojecting, clipping, and resampling DEM...")
            warp_cmd = [
                'gdalwarp',
                '-overwrite',  # Allow overwriting existing files
                '-t_srs', self.target_crs,
                '-tr', str(target_resolution), str(target_resolution),
                '-te', str(self.target_extent['xmin']), str(self.target_extent['ymin']),
                       str(self.target_extent['xmax']), str(self.target_extent['ymax']),
                '-r', 'bilinear',
                '-of', 'GTiff',
                self.dem_file,
                str(dem_subset_file)
            ]

            result = subprocess.run(warp_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"DEM preparation (gdalwarp) failed: {result.stderr}")
                return False

            logger.info(f"DEM subset created: {dem_subset_file}")

            # Store the subset DEM file for TauDEM processing
            self.dem_subset_file = dem_subset_file

            return True

        except Exception as e:
            logger.error(f"Error preparing DEM data: {e}")
            return False

    def run_taudem_preprocessing(self):
        """Run TauDEM preprocessing steps."""
        logger.info("Running TauDEM preprocessing...")
        try:
            if not hasattr(self, 'dem_subset_file'):
                logger.error("No subset DEM file available")
                return False

            # Define output files
            dem_filled_file = self.output_dir / "dem_filled.tif"
            flow_dir_file = self.output_dir / "flow_directions.tif"
            flow_acc_file = self.output_dir / "flow_accumulation.tif"

            # Step 1: Fill pits
            logger.info("Filling pits with TauDEM pitremove...")
            pitremove_cmd = [
                'mpiexec', '-n', '8', 'pitremove',
                '-z', str(self.dem_subset_file),
                '-fel', str(dem_filled_file)
            ]

            result = subprocess.run(pitremove_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"Pitremove failed: {result.stderr}")
                return False

            logger.info(f"Filled DEM created: {dem_filled_file}")

            # Step 2: Generate D8 flow directions
            logger.info("Generating D8 flow directions with TauDEM d8flowdir...")
            d8flowdir_cmd = [
                'mpiexec', '-n', '8', 'd8flowdir',
                '-fel', str(dem_filled_file),
                '-p', str(flow_dir_file)
            ]

            result = subprocess.run(d8flowdir_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"D8flowdir failed: {result.stderr}")
                return False
            logger.info(f"Flow directions created: {flow_dir_file}")

            # Step 3: Generate D8 contributing area
            logger.info("Generating D8 contributing area with TauDEM aread8...")
            aread8_cmd = [
                'mpiexec', '-n', '8', 'aread8',
                '-p', str(flow_dir_file),
                '-ad8', str(flow_acc_file)
            ]
            result = subprocess.run(aread8_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"Aread8 failed: {result.stderr}")
                return False
            logger.info(f"Flow accumulation created: {flow_acc_file}")

            # Store file paths for later use
            self.dem_filled_file = dem_filled_file
            self.flow_dir_file = flow_dir_file
            self.flow_acc_file = flow_acc_file

            return True

        except Exception as e:
            logger.error(f"Error in TauDEM preprocessing: {e}")
            return False

    def generate_main_stem_flowlines(self):
        """
        Generate a single continuous downstream flowline from the dam location
        using NHDPlus connectivity. This creates a `net2.shp`-like file.
        """
        logger.info("Generating main stem flowlines using NHD connectivity...")
        try:
            # First, check what layers are available in the NHD GDB
            try:
                # Try to load NHD flowlines - check multiple possible layer names
                flowline_layer_names = ['NHDFlowline', 'NHDFlowLine', 'Flowline', 'flowline']
                all_flowlines = None
                for layer_name in flowline_layer_names:
                    try:
                        all_flowlines = gpd.read_file(self.nhd_gdb, layer=layer_name)
                        logger.info(f"Successfully loaded flowlines from layer: {layer_name}")
                        break
                    except Exception:
                        continue

                if all_flowlines is None:
                    raise ValueError("Could not find flowline layer in NHD GDB")

                # Debug: Print available columns
                logger.info(f"Available flowline columns: {list(all_flowlines.columns)}")

                # Try to load flow connectivity data
                flow_layer_names = ['NHDPlusFlow', 'NHDFlow', 'Flow', 'flow']
                flow_data = None
                for layer_name in flow_layer_names:
                    try:
                        flow_data = gpd.read_file(self.nhd_gdb, layer=layer_name)
                        logger.info(f"Successfully loaded flow data from layer: {layer_name}")
                        break
                    except Exception:
                        continue

            except Exception as e:
                logger.error(f"Error loading NHD flowline data: {e}")
                logger.info("Attempting alternative flowline generation using geometric connectivity...")
                return self._generate_flowlines_geometric_fallback()

            # Reproject to target CRS
            dam_utm = self.dam_location.to_crs(self.target_crs)
            dam_point = dam_utm.geometry.iloc[0]
            all_flowlines_utm = all_flowlines.to_crs('EPSG:26912')

            # Debug: Print available columns and find the correct NHDPlusID column name
            logger.info(f"Available flowline columns: {list(all_flowlines_utm.columns)}")
            if flow_data is not None:
                logger.info(f"Available flow data columns: {list(flow_data.columns)}")

            # Find the correct NHDPlusID column name
            nhdplusid_col = None
            for col in all_flowlines_utm.columns:
                if 'nhdplusid' in col.lower():
                    nhdplusid_col = col
                    break

            if nhdplusid_col is None:
                raise ValueError("Could not find NHDPlusID column in flowlines data")

            logger.info(f"Using NHDPlusID column: {nhdplusid_col}")

            # Find the correct flow data column names if flow data is available
            from_col = None
            to_col = None
            if flow_data is not None:
                for col in flow_data.columns:
                    if 'from' in col.lower() and 'nhdp' in col.lower():
                        from_col = col
                    elif 'to' in col.lower() and 'nhdp' in col.lower():
                        to_col = col

                if from_col and to_col:
                    logger.info(f"Using flow data columns: {from_col} -> {to_col}")
                else:
                    logger.warning("Could not find proper flow connectivity columns")
                    flow_data = None

            # Step 1: Find flowline closest to dam
            logger.info("Finding flowline closest to dam...")
            distances = all_flowlines_utm.geometry.distance(dam_point)
            closest_flowline = all_flowlines_utm.loc[distances.idxmin()]
            logger.info(f"Closest flowline ID: {closest_flowline[nhdplusid_col]} at {distances.min():.2f}m distance.")

            # Step 2: Trace downstream using NHDPlusFlow
            main_stem_segments = []
            visited_ids = set()
            current_nhdplusid = closest_flowline[nhdplusid_col]

            for _ in range(200): # Safety break
                if current_nhdplusid in visited_ids:
                    logger.warning(f"Cycle detected at NHDPlusID {current_nhdplusid}. Stopping trace.")
                    break
                visited_ids.add(current_nhdplusid)

                # Find the flowline segment
                current_segment_rows = all_flowlines_utm[all_flowlines_utm[nhdplusid_col] == current_nhdplusid]
                if current_segment_rows.empty:
                    logger.info(f"End of path: NHDPlusID {current_nhdplusid} not found in flowline data.")
                    break

                current_segment = current_segment_rows.iloc[0]
                main_stem_segments.append(current_segment)

                # Find next downstream segment
                if flow_data is not None and from_col and to_col:
                    downstream_info = flow_data[flow_data[from_col] == current_nhdplusid]
                    if downstream_info.empty:
                        logger.info("Reached terminal path.")
                        break

                    current_nhdplusid = downstream_info.iloc[0][to_col]
                else:
                    logger.info("No flow connectivity data available, using geometric fallback.")
                    break

            if not main_stem_segments:
                logger.error("Could not trace any downstream flowlines.")
                return False

            # Step 3: Create and save main stem GeoDataFrame
            main_stem_gdf = gpd.GeoDataFrame(main_stem_segments, crs=self.target_crs)
            main_stem_file = self.output_dir / "main_stem_flowlines.shp"
            main_stem_gdf.to_file(main_stem_file)
            logger.info(f"Saved {len(main_stem_gdf)} main stem segments to {main_stem_file}")
            self.main_stem_file = main_stem_file

            # Note: Downstream endpoints will be created separately after stream network is finalized

            return True

        except Exception as e:
            logger.error(f"Error generating main stem flowlines: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_flowlines_geometric_fallback(self):
        """
        Fallback method to generate flowlines using geometric connectivity
        when NHD flowline data is not available.
        """
        logger.info("Using geometric fallback for flowline generation...")
        try:
            # For this fallback, we'll create a simplified main stem based on HUC12 connectivity
            # and the assumption that flow generally follows HUC connectivity

            if not self.downstream_hucs or len(self.downstream_hucs) == 0:
                logger.error("No downstream HUCs available for geometric fallback")
                return False

            # Create a simplified flowline by connecting HUC12 centroids
            main_stem_points = []

            # Start with dam location
            dam_utm = self.dam_location.to_crs(self.target_crs)
            start_point = dam_utm.geometry.iloc[0]
            main_stem_points.append((start_point.x, start_point.y))

            # Add centroid of each downstream HUC12
            downstream_utm = self.downstream_hucs.to_crs(self.target_crs)
            for _, huc in downstream_utm.iterrows():
                centroid = huc.geometry.centroid
                main_stem_points.append((centroid.x, centroid.y))

            # Create a single LineString from all points
            main_stem_line = LineString(main_stem_points)

            # Create GeoDataFrame similar to net2.shp structure
            main_stem_data = {
                'LINKNO': [0],
                'DSLINKNO': [-1],
                'USLINKNO1': [-1],
                'USLINKNO2': [-1],
                'DSNODEID': [0],
                'strmOrder': [1],
                'Length': [main_stem_line.length],
                'Magnitude': [1],
                'DSContArea': [0.0],
                'strmDrop': [0.0],
                'Slope': [0.0],
                'StraightL': [main_stem_line.length],
                'USContArea': [0.0],
                'WSNO': [0],
                'DOUTEND': [0.0],
                'DOUTSTART': [main_stem_line.length],
                'DOUTMID': [main_stem_line.length / 2],
                'geometry': [main_stem_line]
            }

            main_stem_gdf = gpd.GeoDataFrame(main_stem_data, crs=self.target_crs)

            # Save main stem flowlines
            main_stem_file = self.output_dir / "main_stem_flowlines.shp"
            main_stem_gdf.to_file(main_stem_file)
            logger.info(f"Saved geometric fallback main stem to {main_stem_file}")
            self.main_stem_file = main_stem_file

            # Note: Downstream endpoints will be created separately after stream network is finalized

            return True

        except Exception as e:
            logger.error(f"Error in geometric fallback: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_flow_accumulation_stats(self):
        """Analyze flow accumulation statistics to determine appropriate threshold."""
        logger.info("Analyzing flow accumulation statistics...")
        try:
            if not hasattr(self, 'flow_acc_file'):
                logger.error("Flow accumulation file not available")
                return None

            # Use GDAL to get basic statistics
            stats_cmd = ['gdalinfo', '-stats', str(self.flow_acc_file)]
            result = subprocess.run(stats_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

            if result.returncode != 0:
                logger.warning("Could not get flow accumulation statistics")
                return 0.01  # Default fallback

            # Parse statistics from gdalinfo output
            stats_lines = result.stdout.split('\n')
            min_val, max_val, mean_val = None, None, None

            for line in stats_lines:
                if 'STATISTICS_MINIMUM=' in line:
                    min_val = float(line.split('=')[1])
                elif 'STATISTICS_MAXIMUM=' in line:
                    max_val = float(line.split('=')[1])
                elif 'STATISTICS_MEAN=' in line:
                    mean_val = float(line.split('=')[1])

            if all(v is not None for v in [min_val, max_val, mean_val]):
                # Use a threshold that's approximately 10% of maximum or 2x mean, whichever is smaller
                threshold = min(max_val * 0.1, mean_val * 2)
                logger.info(f"Flow accumulation stats - Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}")
                logger.info(f"Calculated threshold: {threshold:.6f}")
                return threshold
            else:
                logger.warning("Could not parse flow accumulation statistics")
                return 0.01

        except Exception as e:
            logger.error(f"Error analyzing flow accumulation: {e}")
            return 0.01

    def run_taudem_streamnet(self):
        """Run TauDEM streamnet to generate final stream network."""
        logger.info("Running TauDEM streamnet...")
        try:
            if not all(hasattr(self, attr) for attr in ['flow_dir_file', 'endpoints_file']):
                logger.error("Missing required files for streamnet")
                return False

            # Define output files
            stream_raster_file = self.output_dir / "stream_network.tif"
            stream_vector_file = self.output_dir / "stream_network.shp"
            watershed_file = self.output_dir / "watersheds.tif"

            # First, create a stream raster using threshold on flow accumulation
            # Get adaptive threshold based on flow accumulation statistics
            threshold_value = self.analyze_flow_accumulation_stats()

            logger.info(f"Generating stream network with TauDEM threshold (threshold: {threshold_value})...")
            threshold_cmd = [
                'mpiexec', '-n', '8', 'threshold',
                '-ssa', str(self.flow_acc_file),
                '-src', str(stream_raster_file),
                '-thresh', str(threshold_value)
            ]

            result = subprocess.run(threshold_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"Threshold failed: {result.stderr}")
                return False

            logger.info(f"Stream raster created: {stream_raster_file}")

            # Create a single outlet point from the most downstream endpoint for streamnet
            logger.info("Creating outlet point for streamnet...")
            endpoints_gdf = gpd.read_file(self.endpoints_file) # This file is now from main_stem_flowlines

            # Find the most downstream point (lowest Y coordinate in UTM)
            min_y_idx = endpoints_gdf.geometry.y.idxmin()
            outlet_point = endpoints_gdf.iloc[[min_y_idx]]

            # Save outlet point
            outlet_file = self.output_dir / "outlet_points.shp"
            outlet_point.to_file(outlet_file)
            logger.info(f"Outlet point saved to: {outlet_file}")

            # Now run streamnet with outlet points
            logger.info("Generating stream network with TauDEM streamnet...")
            streamnet_cmd = [
                'mpiexec', '-n', '8', 'streamnet',
                '-p', str(self.flow_dir_file),
                '-fel', str(self.dem_filled_file),
                '-ad8', str(self.flow_acc_file),
                '-src', str(stream_raster_file),
                '-o', str(outlet_file),  # Add outlet points
                '-ord', str(self.output_dir / "stream_order.tif"),
                '-tree', str(self.output_dir / "tree.txt"),
                '-coord', str(self.output_dir / "coordinates.txt"),
                '-net', str(stream_vector_file),
                '-w', str(watershed_file)
            ]

            result = subprocess.run(streamnet_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                logger.error(f"Streamnet failed: {result.stderr}")
                logger.info("Attempting alternative streamnet approach without outlets...")

                # Try without outlet points as fallback
                streamnet_cmd_fallback = [
                    'mpiexec', '-n', '8', 'streamnet',
                    '-p', str(self.flow_dir_file),
                    '-fel', str(self.dem_filled_file),
                    '-ad8', str(self.flow_acc_file),
                    '-src', str(stream_raster_file),
                    '-ord', str(self.output_dir / "stream_order.tif"),
                    '-tree', str(self.output_dir / "tree.txt"),
                    '-coord', str(self.output_dir / "coordinates.txt"),
                    '-net', str(stream_vector_file),
                    '-w', str(watershed_file)
                ]

                result = subprocess.run(streamnet_cmd_fallback, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                if result.returncode != 0:
                    logger.error(f"Fallback streamnet also failed: {result.stderr}")
                    return False

            logger.info(f"Stream network raster created: {stream_raster_file}")
            logger.info(f"Stream network vector created: {stream_vector_file}")
            logger.info(f"Watersheds created: {watershed_file}")

            # Verify the vector file has features
            try:
                vector_check = gpd.read_file(stream_vector_file)
                logger.info(f"Stream network vector contains {len(vector_check)} features")
                if len(vector_check) == 0:
                    logger.warning("Stream network vector is empty - this may indicate issues with outlet points or stream threshold")
            except Exception as e:
                logger.warning(f"Could not verify stream network vector: {e}")

            # Store output files
            self.stream_raster_file = stream_raster_file
            self.stream_vector_file = stream_vector_file
            self.watershed_file = watershed_file

            # Replace the generic stream network with main stem flowlines
            self.create_main_stream_network()

            # Create downstream endpoints for each flowline in the stream network
            self.create_downstream_endpoints()

            return True

        except Exception as e:
            logger.error(f"Error in TauDEM streamnet: {e}")
            return False

    def create_main_stream_network(self):
        """Replace the generic stream network with the main stem flowlines for better representation."""
        try:
            logger.info("Creating main stream network from NHD flowlines...")

            # Check if main stem flowlines exist
            if not hasattr(self, 'main_stem_file') or not self.main_stem_file.exists():
                logger.warning("Main stem flowlines not found - keeping original stream network")
                return

            # Create a backup of the original stream network
            original_stream_network = self.stream_vector_file
            backup_stream_network = self.output_dir / "stream_network_all_streams.shp"

            # Backup the comprehensive stream network
            if original_stream_network.exists():
                logger.info(f"Backing up comprehensive stream network to: {backup_stream_network}")
                # Copy all related files (.shp, .shx, .dbf, .prj)
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = original_stream_network.with_suffix(ext)
                    dst = backup_stream_network.with_suffix(ext)
                    if src.exists():
                        subprocess.run(['cp', str(src), str(dst)], check=True)

            # Load and copy main stem flowlines as the new stream network
            main_stem_gdf = gpd.read_file(self.main_stem_file)
            logger.info(f"Replacing stream network with {len(main_stem_gdf)} main stem flowlines")

            # Save main stem as the primary stream network
            main_stem_gdf.to_file(self.stream_vector_file)

            logger.info(f"Main stream network created: {self.stream_vector_file}")
            logger.info(f"Comprehensive stream network backed up to: {backup_stream_network}")

        except Exception as e:
            logger.error(f"Error creating main stream network: {e}")

    def create_downstream_endpoints(self):
        """Create downstream endpoints for each flowline in the stream network."""
        try:
            # Load the final stream network
            stream_network_gdf = gpd.read_file(self.stream_vector_file)
            logger.info(f"Creating downstream endpoints for {len(stream_network_gdf)} flowlines")

            # Extract downstream endpoints for each flowline
            endpoints = []
            endpoint_attributes = []

            for idx, segment in stream_network_gdf.iterrows():
                geom = segment.geometry
                if geom.geom_type == 'LineString':
                    endpoint = Point(geom.coords[-1])
                elif geom.geom_type == 'MultiLineString':
                    # For MultiLineString, get the end of the last linestring
                    endpoint = Point(geom.geoms[-1].coords[-1])
                else:
                    continue

                endpoints.append(endpoint)
                # Copy relevant attributes from the flowline
                endpoint_attributes.append({
                    'flowline_id': idx,
                    'permanent_': segment.get('permanent_', ''),
                    'gnis_name': segment.get('gnis_name', ''),
                    'reachcode': segment.get('reachcode', ''),
                    'nhdplusid': segment.get('nhdplusid', ''),
                    'lengthkm': segment.get('lengthkm', 0)
                })

            # Create GeoDataFrame with endpoints
            endpoints_gdf = gpd.GeoDataFrame(
                endpoint_attributes,
                geometry=endpoints,
                crs=self.target_crs
            )

            # Save as downstream_flowline_endpoints.shp
            endpoints_file = self.output_dir / "downstream_flowline_endpoints.shp"
            endpoints_gdf.to_file(endpoints_file)
            logger.info(f"Saved {len(endpoints_gdf)} downstream flowline endpoints to {endpoints_file}")
            self.endpoints_file = endpoints_file

            return True

        except Exception as e:
            logger.error(f"Error creating downstream endpoints: {e}")
            import traceback
            traceback.print_exc()
            return False
            logger.info("Keeping original stream network")

    def generate_final_outputs(self):
        """Generate final flowline products and validation."""
        logger.info("Generating final outputs...")
        try:
            # Copy flow direction file as final output
            final_flow_dir = self.output_dir / "final_flow_directions.tif"
            subprocess.run(['cp', str(self.flow_dir_file), str(final_flow_dir)])

            # Generate summary report
            report_file = self.output_dir / "processing_report.txt"
            with open(report_file, 'w') as f:
                f.write("FLOWLINE GENERATION PROCESSING REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Dam Location: {self.dam_location.iloc[0].get('Dam_Name', 'Unknown')}\n")
                f.write(f"Dam HUC12: {self.dam_huc12['huc12']} - {self.dam_huc12['name']}\n")
                f.write(f"Number of downstream HUC12s: {len(self.downstream_hucs)}\n")
                f.write("Processing resolution: 30m\n")
                f.write("Output projection: UTM Zone 12N (EPSG:26912)\n\n")

                f.write("OUTPUT FILES:\n")
                f.write(f"- Final Flow directions: {final_flow_dir}\n")
                f.write(f"- Main stem flowlines (like net2.shp): {self.main_stem_file}\n")
                f.write(f"- Primary stream network (main flow path): {self.stream_vector_file}\n")
                f.write(f"- Comprehensive stream network (all streams): {self.output_dir / 'stream_network_all_streams.shp'}\n")
                f.write(f"- Outlet points for TauDEM: {self.endpoints_file}\n")
                f.write(f"- Stream network (raster): {self.stream_raster_file}\n")
                f.write(f"- Watersheds: {self.watershed_file}\n")
                f.write(f"- Study area: {self.output_dir / 'study_area.shp'}\n")

            logger.info(f"Processing report saved to: {report_file}")
            logger.info("Flowline generation completed successfully!")

            return True

        except Exception as e:
            logger.error(f"Error generating final outputs: {e}")
            return False

    def run_complete_workflow(self):
        """Run the complete flowline generation workflow."""
        logger.info("Starting complete flowline generation workflow...")

        # Phase 1: Dam Location Analysis and HUC12 Identification
        if not self.load_dam_location():
            return False
        if not self.find_dam_huc12():
            return False
        if not self.check_dam_in_nhd_points():
            return False

        # Phase 2: Downstream HUC12 Assembly
        if not self.build_downstream_huc12_network():
            return False
        if not self.create_study_area_extent():
            return False

        # Phase 3: Raster Data Preparation
        if not self.prepare_dem_data():
            return False

        # Phase 4: TauDEM Processing
        if not self.run_taudem_preprocessing():
            return False

        # Phase 5: Main Stem Flowline Generation
        if not self.generate_main_stem_flowlines():
            return False

        # Phase 7: Stream Network Generation
        if not self.run_taudem_streamnet():
            return False

        # Phase 8: Final Output Generation
        if not self.generate_final_outputs():
            return False

        logger.info("Complete workflow finished successfully!")
        return True


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Generate flowlines from dam location using GDAL, TauDEM, and NHD data"
    )
    parser.add_argument(
        "--dam-shapefile",
        default="data/mountain_dell_dam_location.shp",
        help="Path to dam location shapefile"
    )
    parser.add_argument(
        "--nhd-gdb",
        default="data/NHDPLUS_H_1602_HU4_20220412_GDB/NHDPLUS_H_1602_HU4_20220412_GDB.gdb",
        help="Path to NHD Plus geodatabase"
    )
    parser.add_argument(
        "--dem-file",
        default="data/elevNHDm.tif",
        help="Path to DEM file"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        help="Maximum downstream distance in kilometers"
    )

    args = parser.parse_args()

    # Create flowline generator
    generator = FlowlineGenerator(
        dam_shapefile=args.dam_shapefile,
        nhd_gdb=args.nhd_gdb,
        dem_file=args.dem_file,
        output_dir=args.output_dir
    )

    # Run workflow
    success = generator.run_complete_workflow()

    if success:
        logger.info("Flowline generation completed successfully!")
        sys.exit(0)
    else:
        logger.error("Flowline generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()