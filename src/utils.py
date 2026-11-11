#!/usr/bin/env python3
"""
Utility functions for HAND Dam Workflow

Common functions shared across multiple workflow scripts.
"""

import json
import os
import sys
import shutil
import shlex
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Written by dam_catchments into the output directory so later stages can reuse its inputs
RUN_INFO_FILE = "run_info.json"
DAM_SHAPEFILE_NAME = "dam_location.shp"


def resolve_user_path(value, dam_dir):
    """
    Resolve a user-supplied path flag. A bare name (no directory part, e.g. my_dam.shp)
    is looked up in dam_dir; absolute paths and paths with a directory part (./x, sub/x)
    resolve as usual (relative to the current directory).
    """
    path = Path(value).expanduser()
    if not path.is_absolute() and path.parent == Path("."):
        # "./x" also has parent "." but was typed explicitly relative to the CWD
        if not str(value).startswith(("./", ".\\")):
            path = Path(dam_dir) / path
    return path.resolve()


def find_dam_shapefile(dam_dir):
    """
    Locate the dam location shapefile in a per-dam data directory.

    Uses <dam_dir>/dam_location.shp if present, otherwise the single *dam_location.shp
    in dam_dir (e.g. my_dam_location.shp).

    Raises:
        FileNotFoundError: if there is no match, or more than one
    """
    dam_dir = Path(dam_dir).expanduser().resolve()
    exact = dam_dir / DAM_SHAPEFILE_NAME
    if exact.exists():
        return exact
    matches = sorted(dam_dir.glob(f"*{DAM_SHAPEFILE_NAME}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {DAM_SHAPEFILE_NAME} or *{DAM_SHAPEFILE_NAME} in {dam_dir}; "
            "add one, pass --dam-shapefile, or pass --dam-lat/--dam-lon to dam_catchments"
        )
    raise FileNotFoundError(
        f"Several dam shapefiles in {dam_dir}: {[m.name for m in matches]}; "
        "rename one to dam_location.shp or pass --dam-shapefile"
    )


def _one_match(matches, what, dam_dir, flag):
    """Return the single match, or raise FileNotFoundError naming the override flag."""
    # Dedupe on the real path (symlinked folders), report the paths as found in dam_dir
    matches = sorted({m.resolve(): m for m in matches}.values())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {what} found in {dam_dir} or its subfolders; "
                                f"add one or pass {flag}")
    raise FileNotFoundError(f"Several {what}s in {dam_dir}: "
                            f"{[str(m.relative_to(dam_dir)) for m in matches]}; pass {flag}")


def find_nhdplus_gdb(dam_dir):
    """
    Locate the dam's NHDPlus HR HU4 geodatabase: the single *.gdb with a WBDHU12 layer,
    directly in dam_dir or one folder down (e.g. NHDPLUS_H_1602_HU4_GDB/NHDPLUS_H_1602_HU4_GDB.gdb).
    Other geodatabases, such as elev_source.gdb in the raster folder, are skipped.
    """
    import geopandas as gpd

    dam_dir = Path(dam_dir).expanduser().resolve()
    matches = []
    for pattern in ("*.gdb", "*/*.gdb"):
        for p in dam_dir.glob(pattern):
            try:
                if p.is_dir() and "WBDHU12" in set(gpd.list_layers(p)["name"]):
                    matches.append(p)
            except Exception as e:  # unreadable GDB: not the one we want
                logger.debug(f"Skipping {p}: {e}")
    return _one_match(matches, "NHDPlus HR geodatabase (*.gdb with a WBDHU12 layer)",
                      dam_dir, "--huc12-gdb")


def find_nhdplus_raster_dir(dam_dir):
    """Locate the dam's NHDPlus HR HU4 raster folder: the single subfolder holding hydrodem.tif."""
    dam_dir = Path(dam_dir).expanduser().resolve()
    matches = [p.parent for p in dam_dir.glob("*/hydrodem.tif")]
    return _one_match(matches, "NHDPlus HR raster folder (with hydrodem.tif)", dam_dir,
                      "--raster-dir")


def default_output_dir(dam_dir):
    """Default per-dam output directory: outputs/<dam_dir name>."""
    return (Path("outputs") / Path(dam_dir).expanduser().resolve().name).resolve()


def write_run_info(output_dir, **paths):
    """Merge the given paths into <output_dir>/run_info.json (values stored as strings)."""
    info_path = Path(output_dir) / RUN_INFO_FILE
    info = read_run_info(output_dir)
    info.update({k: str(v) for k, v in paths.items() if v is not None})
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    logger.info(f"Wrote run info: {info_path}")


def read_run_info(output_dir):
    """Return the dict in <output_dir>/run_info.json, or {} if absent or unreadable."""
    info_path = Path(output_dir) / RUN_INFO_FILE
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read {info_path}: {e}")
        return {}


def setup_taudem_path(taudem_path):
    """
    Add TauDEM to system PATH.
    
    Args:
        taudem_path: Path to TauDEM installation
    """
    current_path = os.environ.get("PATH", "")
    if taudem_path not in current_path:
        os.environ["PATH"] = f"{taudem_path}:{current_path}"
        logger.info(f"Added TauDEM path: {taudem_path}")

    # Check if TauDEM is available
    mpiexec_path = shutil.which("mpiexec")
    if mpiexec_path is None:
        logger.warning("mpiexec not found in PATH. TauDEM commands may fail.")
    else:
        logger.info(f"Found mpiexec at: {mpiexec_path}")


def run_taudem_command(command, description="TauDEM command", cwd=None):
    """
    Execute a TauDEM command with error handling.

    Args:
        command: Command string to execute
        description: Description for logging
        cwd: Working directory for command execution
    """
    logger.info(f"Running {description}: {command}")
    cmd_list = shlex.split(command)
    try:
        result = subprocess.run(
            cmd_list,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Warnings: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"TauDEM command failed: {' '.join(cmd_list)}")
        logger.error(f"Error: {e.stderr}")
        sys.exit(1)
