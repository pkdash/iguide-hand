#!/usr/bin/env python3
"""
run_rasterize.py
Wrapper script to run the gdal_rasterize command from Python.
"""

import subprocess
import argparse
import sys
import shlex

def main():
    parser = argparse.ArgumentParser(description="Run gdal_rasterize from Python")
    parser.add_argument("--xmin", type=float, required=True)
    parser.add_argument("--ymin", type=float, required=True)
    parser.add_argument("--xmax", type=float, required=True)
    parser.add_argument("--ymax", type=float, required=True)
    parser.add_argument("--cellsize_resx", type=float, required=True)
    parser.add_argument("--cellsize_resy", type=float, required=True)
    parser.add_argument("--shapefile", required=True, help="Input shapefile")
    parser.add_argument("--output", required=True, help="Output raster filename")
    args = parser.parse_args()

    cmd = [
        "gdal_rasterize",
        "-ot", "Int16",
        "-of", "GTiff",
        "-burn", "1",
        "-tr", str(args.cellsize_resx), str(args.cellsize_resy),
        "-te", str(args.xmin), str(args.ymin), str(args.xmax), str(args.ymax),
        args.shapefile,
        args.output
    ]

    # Escape each argument to prevent command injection
    safe_cmd = [shlex.quote(arg) for arg in cmd]

    print("Running:", " ".join(safe_cmd))
    try:
        subprocess.run(safe_cmd, check=True, shell=False)
        print(f"Raster created: {args.output}")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error running gdal_rasterize: {e}")

if __name__ == "__main__":
    main()
