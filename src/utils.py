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

# Per-dam layout: <dams>/<dam>/inputs and <dams>/<dam>/outputs, with the NHDPlus HR
# downloads shared in <dams>/../nhdplus and referenced from inputs/nhdplus.json
INPUTS_DIR_NAME = "inputs"
OUTPUTS_DIR_NAME = "outputs"
NHD_CACHE_DIR_NAME = "nhdplus"
DAMS_DIR_NAME = "dams"
NHDPLUS_REF_FILE = "nhdplus.json"
_NHDPLUS_REF_KEYS = {"gdb": "gdb", "raster": "raster_dir"}


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


def _load_nhdplus_ref_json(ref_path):
    """
    The dict in nhdplus.json, or None (with a warning naming the file) if it cannot be
    read, is not a JSON object, or has "gdb"/"raster_dir" values that are not strings.
    """
    try:
        ref = json.loads(ref_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read {ref_path}: {e}")
        return None
    if not isinstance(ref, dict):
        logger.warning(f"Ignoring {ref_path}: expected a JSON object, got {type(ref).__name__}")
        return None
    bad = [k for k in _NHDPLUS_REF_KEYS.values()
           if ref.get(k) is not None and not isinstance(ref[k], str)]
    if bad:
        logger.warning(f"Ignoring {ref_path}: {', '.join(bad)} must be a path string")
        return None
    return ref


def read_nhdplus_ref(inputs_dir):
    """
    Return the dict in <inputs_dir>/nhdplus.json with "gdb" / "raster_dir" resolved to
    absolute Paths (relative paths are relative to inputs_dir), or {} if absent or unreadable.
    """
    inputs_dir = Path(inputs_dir).expanduser().resolve()
    ref_path = inputs_dir / NHDPLUS_REF_FILE
    if not ref_path.exists():
        return {}
    ref = _load_nhdplus_ref_json(ref_path)
    if ref is None:
        return {}
    for key in _NHDPLUS_REF_KEYS.values():
        if ref.get(key):
            ref[key] = (inputs_dir / Path(ref[key]).expanduser()).resolve()
    return ref


def write_nhdplus_ref(inputs_dir, hu4=None, gdb=None, raster_dir=None):
    """
    Merge NHDPlus HR locations into <inputs_dir>/nhdplus.json, so a dam can use a shared
    download without copying it or creating symlinks (which need extra rights on Windows).
    Paths are stored relative to inputs_dir, so the whole tree can be moved; absolute when
    no relative path exists (another drive on Windows).
    """
    inputs_dir = Path(inputs_dir).expanduser().resolve()
    ref_path = inputs_dir / NHDPLUS_REF_FILE
    ref = {}
    if ref_path.exists():
        ref = _load_nhdplus_ref_json(ref_path)
        if ref is None:  # keep what the user wrote before replacing it
            backup = ref_path.with_name(ref_path.name + ".bak")
            shutil.copy2(ref_path, backup)
            logger.warning(f"Saved the unusable {ref_path} as {backup}")
            ref = {}
    if hu4 is not None:
        ref["hu4"] = str(hu4)
    for key, path in (("gdb", gdb), ("raster_dir", raster_dir)):
        if path is None:
            continue
        path = Path(path).expanduser().resolve()
        try:
            ref[key] = Path(os.path.relpath(path, inputs_dir)).as_posix()
        except ValueError:  # different drive on Windows
            ref[key] = str(path)
    ref_path.write_text(json.dumps(ref, indent=2) + "\n")
    logger.info(f"Wrote NHDPlus HR reference: {ref_path}")


def _from_nhdplus_ref(dam_dir, product, error):
    """
    The product's path from <dam_dir>/nhdplus.json; re-raises error if the file does not
    name it, FileNotFoundError naming the file if the path it names is gone.
    """
    path = read_nhdplus_ref(dam_dir).get(_NHDPLUS_REF_KEYS[product])
    if path is None:
        raise error
    if not path.exists():
        raise FileNotFoundError(
            f"{dam_dir / NHDPLUS_REF_FILE} points to {path}, which does not exist; fix or "
            "delete that file, or rerun dam_catchments with --dam-lat/--dam-lon to restore the data")
    return path


def find_nhdplus_gdb(dam_dir):
    """
    Locate the dam's NHDPlus HR HU4 geodatabase: the single *.gdb with a WBDHU12 layer,
    directly in dam_dir or one folder down (e.g. NHDPLUS_H_1602_HU4_GDB/NHDPLUS_H_1602_HU4_GDB.gdb).
    Other geodatabases, such as elev_source.gdb in the raster folder, are skipped.
    If there is none, the "gdb" entry of <dam_dir>/nhdplus.json is used.
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
    try:
        return _one_match(matches, "NHDPlus HR geodatabase (*.gdb with a WBDHU12 layer)",
                          dam_dir, "--huc12-gdb")
    except FileNotFoundError as e:
        if matches:
            raise
        return _from_nhdplus_ref(dam_dir, "gdb", e)


def find_nhdplus_raster_dir(dam_dir):
    """
    Locate the dam's NHDPlus HR HU4 raster folder: the single subfolder holding hydrodem.tif,
    else the "raster_dir" entry of <dam_dir>/nhdplus.json.
    """
    dam_dir = Path(dam_dir).expanduser().resolve()
    matches = [p.parent for p in dam_dir.glob("*/hydrodem.tif")]
    try:
        return _one_match(matches, "NHDPlus HR raster folder (with hydrodem.tif)", dam_dir,
                          "--raster-dir")
    except FileNotFoundError as e:
        if matches:
            raise
        return _from_nhdplus_ref(dam_dir, "raster", e)


def dam_inputs_dir(dam_dir, create=False):
    """
    The input folder of a dam folder: <dam_dir>/inputs.

    Args:
        dam_dir: the dam folder, e.g. dams/mountain_dell (not its inputs/ subfolder)
        create: create the folder (with its parents) if it is missing

    Raises:
        FileNotFoundError: the folder is missing (and create is False), dam_dir is itself
            an inputs/ folder
    """
    dam_dir = Path(dam_dir).expanduser().resolve()
    inputs = dam_dir / INPUTS_DIR_NAME
    if inputs.is_dir():
        return inputs
    if dam_dir.name == INPUTS_DIR_NAME:
        raise FileNotFoundError(
            f"{dam_dir} looks like a dam's {INPUTS_DIR_NAME}/ folder; pass the dam folder "
            f"above it ({dam_dir.parent}) as --dam-dir")
    if create:
        inputs.mkdir(parents=True, exist_ok=True)
        return inputs
    raise FileNotFoundError(
        f"No {INPUTS_DIR_NAME}/ folder in {dam_dir}; put the dam inputs in {inputs}, or pass "
        "--dam-lat/--dam-lon to dam_catchments to create it")


def default_output_dir(dam_dir):
    """Default per-dam output directory: <dam_dir>/outputs (independent of the CWD)."""
    return Path(dam_dir).expanduser().resolve() / OUTPUTS_DIR_NAME


def default_nhd_cache_dir(dam_dir):
    """
    Default NHDPlus HR download folder.

    For the standard layout (<root>/dams/<dam>) this is the shared <root>/nhdplus, so dams in
    one HU4 download it once. For a dam folder anywhere else we cannot know what the
    grandparent is, so the cache goes inside the dam folder (<dam_dir>/nhdplus) instead of
    possibly outside the project. Pass --nhd-cache-dir to share a cache explicitly.
    """
    dam_dir = Path(dam_dir).expanduser().resolve()
    if dam_dir.parent.name == DAMS_DIR_NAME:
        return dam_dir.parent.parent / NHD_CACHE_DIR_NAME
    return dam_dir / NHD_CACHE_DIR_NAME


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
