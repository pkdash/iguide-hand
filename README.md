# iGuide HAND Dam Workflow

This repository contains Python scripts for Height Above Nearest Drainage (HAND) inundation mapping workflow for aging dams. The workflow combines HUC12 watershed analysis with TauDEM hydrologic processing to generate flood inundation maps.

## Overview

### compute_bounding_box_for_dam_catchments.py

This script calculates bounding boxes for all HUC12 watersheds downstream from a dam location. Key features include:

- **Dam Location Analysis**: Finds the starting HUC12 watershed containing the dam location
- **Downstream Traversal**: Follows HUC12 connectivity data to identify all downstream watersheds
- **Distance Calculation**: Uses sqrt(area) formula to calculate cumulative distance along the flow path
- **Automatic Termination**: Stops at 100km distance or closed basins (terminal watersheds)
- **Bounding Box Computation**: Calculates spatial bounds in the target raster projection
- **Raster Clipping**: Optionally clips all NHDPlus rasters to the calculated bounding box with optional buffer

The script outputs downstream HUC12 shapefiles, bounding box coordinates, and optionally clipped rasters for efficient processing.

### hand_dam_workflow.py

This script implements the complete HAND dam workflow for inundation mapping. Key components include:

- **Data Preparation**: Copies and preprocesses clipped rasters and input files
- **Flow Direction Processing**: Creates flow direction masks and reclassifies ArcGIS D8 to TauDEM encoding
- **TauDEM Integration**: Runs TauDEM tools for pit removal, flow direction, and contributing area calculations
- **HAND Calculation**: Computes Height Above Nearest Drainage using D-infinity distance-down method
- **Stream Network Analysis**: Generates stream networks and processes outlet locations
- **Inundation Modeling**: Calculates hydraulic geometry and inundation depths for flood forecasting

The workflow produces HAND rasters, stream networks, and inundation depth maps for flood risk assessment.

## Installation

### Prerequisites

Before installing Python dependencies, ensure you have the following external tools installed:

- **TauDEM** (Terrain Analysis Using Digital Elevation Models) - C++ tools for hydrologic analysis
- **MPI** (Message Passing Interface) - Required for TauDEM parallel processing
- **GDAL** command line tools - For raster processing operations

### Python Environment Setup

1. **Create a Python virtual environment**:

   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   ```

   ```bash
   # On Windows:
   python -m venv .venv
   ```

2. **Activate the virtual environment**:

   ```bash
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:

   ```bash
   python -c "import geopandas, rasterio, numpy, pandas; print('All dependencies installed successfully!')"
   ```

## Computing HUC12 Bounding Box and Clipping HUC12 Rasters

The `compute_bounding_box_for_dam_catchments.py` script calculates downstream watershed boundaries and optionally clips rasters to reduce processing area.

### Basic Usage with Raster Clipping

```bash
python src/compute_bounding_box_for_dam_catchments.py --clip-rasters --buffer 4000
```

This command will:

- Calculate the bounding box for all HUC12 watersheds downstream from the dam location
- Clip all NHDPlus rasters to the calculated bounding box
- Add a 4000-meter buffer around the bounding box before clipping
- Save clipped rasters to `outputs/clipped/` directory
- Generate processing reports and downstream HUC12 shapefiles

### Output Files

- `outputs/downstream_huc12s_from_dam.shp` - Downstream HUC12 watersheds shapefile
- `outputs/bounding_box_coordinates.txt` - Bounding box coordinates in target projection
- `outputs/processing_report.txt` - Detailed processing summary
- `outputs/clipped/` - Directory containing clipped raster files

## HAND Workflow

The `hand_dam_workflow.py` script executes the complete HAND inundation mapping workflow using TauDEM tools.

### Basic Usage

```bash
python src/hand_dam_workflow.py --taudem-path /path/to/taudem
```

This command will:

- Copy clipped rasters and input files to the working directory
- Process flow directions and elevation data
- Run TauDEM hydrologic analysis tools
- Calculate HAND (Height Above Nearest Drainage) values
- Generate stream networks and process outlet locations
- Compute inundation depths for flood forecasting

### Prerequisites for HAND Workflow

Before running the HAND workflow, ensure you have:

1. **Completed HUC12 bounding box calculation** with `--clip-rasters` option
2. **TauDEM installed** and accessible at the specified path
3. **Required input files** in the `data/` directory:
   - `stage.txt` - Stage height data
   - `forecast.csv` - Forecast data
   - `Outlets2_utm.shp` - Outlet locations shapefile
   - Dam location shapefile and HUC12 geodatabase

### HAND Workflow Output Files

The workflow generates numerous output files including:

- `hand.tif` - Height Above Nearest Drainage raster
- `inundepth.tif` - Inundation depth raster
- `hydroprop.csv` - Hydraulic properties table
- `depths.csv` - Depth calculations
- Various intermediate TauDEM processing files

## Directory Structure

```text
iguide-hand/
├── src/
│   ├── compute_bounding_box_for_dam_catchments.py
│   ├── hand_dam_workflow.py
│   └── ...
├── data/
│   ├── mountain_dell_dam_location.shp
│   ├── NHDPLUS_H_1602_HU4_20220412_GDB/
│   ├── NHDPLUS_H_1602_HU4_20220412_RASTER/
│   ├── stage.txt
│   ├── forecast.csv
│   └── Outlets2_utm.shp
├── outputs/
│   ├── clipped/          # Clipped rasters
│   ├── hand.tif          # HAND results
│   ├── inundepth.tif     # Inundation depths
│   └── ...
├── requirements.txt
└── README.md
```

## Workflow Sequence

For a complete analysis, run the scripts in the following order:

1. **First**, calculate bounding box and clip rasters:

   ```bash
   python src/compute_bounding_box_for_dam_catchments.py --clip-rasters --buffer 4000
   ```

2. **Then**, run the HAND workflow:

   ```bash
   python src/hand_dam_workflow.py --taudem-path /usr/local/taudem
   ```

3. **Finally**, deactivate the virtual environment when finished:

   ```bash
   deactivate
   ```

This sequence ensures that the HAND workflow operates on efficiently clipped data, reducing processing time and memory requirements.

## Support

For questions or issues related to this workflow, please refer to the processing reports generated by each script, which contain detailed logs and diagnostic information.
