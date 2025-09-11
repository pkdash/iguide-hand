#!/usr/bin/env python3
"""
Rasterize a shapefile using GDAL in Python.

Usage:
    python Rasterize.py shapefile raster --xmin xmin --ymin ymin --xmax xmax --ymax ymax --cellsize_x cellsize_x --cellsize_y cellsize_y [--burn burn_value]
"""

from osgeo import gdal, ogr
import argparse

gdal.UseExceptions()


def main():
    parser = argparse.ArgumentParser(description="Rasterize a shapefile using GDAL in Python.")
    parser.add_argument("shapefile", help="Input shapefile path")
    parser.add_argument("raster", help="Output raster path")
    parser.add_argument("--xmin", type=float, required=True, help="Minimum X coordinate")
    parser.add_argument("--ymin", type=float, required=True, help="Minimum Y coordinate")
    parser.add_argument("--xmax", type=float, required=True, help="Maximum X coordinate")
    parser.add_argument("--ymax", type=float, required=True, help="Maximum Y coordinate")
    parser.add_argument("--cellsize_x", type=float, required=True, help="Pixel size in X direction")
    parser.add_argument("--cellsize_y", type=float, required=True, help="Pixel size in Y direction")
    parser.add_argument("--burn", type=float, default=1, help="Burn value (default=1)")

    args = parser.parse_args()

    # Compute raster size
    # x_res = int((args.xmax - args.xmin) / args.cellsize_x)
    # y_res = int((args.ymax - args.ymin) / args.cellsize_y)

    # Compute raster size (force at least 1 cell)
    x_res = max(1, round((args.xmax - args.xmin) / args.cellsize_x))
    y_res = max(1, round((args.ymax - args.ymin) / args.cellsize_y))

    # Create raster
    raster_ds = gdal.GetDriverByName("GTiff").Create(args.raster, x_res, y_res, 1, gdal.GDT_Float32)
    raster_ds.SetGeoTransform((args.xmin, args.cellsize_x, 0, args.ymax, 0, -args.cellsize_y))

    # Open shapefile and get layer
    source_ds = ogr.Open(args.shapefile)
    if source_ds is None:
        print(f"Could not open shapefile: {args.shapefile}")
        return
    layer = source_ds.GetLayer()
    raster_ds.SetProjection(layer.GetSpatialRef().ExportToWkt())

    # Rasterize
    gdal.RasterizeLayer(raster_ds, [1], layer, burn_values=[args.burn])

    # Close dataset
    raster_ds = None
    print(f"Rasterization complete: {args.raster}")

if __name__ == "__main__":
    main()
