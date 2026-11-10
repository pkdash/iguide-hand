"""
Dam input preparation helpers.

Builds a self-contained per-dam folder (``data/<dam>/``) for the HAND pipeline from
a dam's latitude/longitude:

- ``find_hu4_for_point``: the HU4 that contains the dam (USGS WBD map service)
- ``nhdplus_hr_urls`` / ``download_nhdplus_hr``: the NHDPlus HR HU4 GDB and raster
  zips from the USGS National Map staged products bucket
- ``write_dam_point``: ``<name>_dam_location.shp``
- ``write_stage_file``: ``stage.txt``

and, after ``dam_hand``:

- ``write_uniform_forecast``: a uniform-flow ``forecast.csv`` keyed to the run's reaches

Used by ``notebooks/dam_inundation_pipeline.ipynb``; not a console script.
"""

import json
import logging
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from utils import DAM_SHAPEFILE_NAME, find_nhdplus_gdb, find_nhdplus_raster_dir

logger = logging.getLogger(__name__)

WBD_HU4_QUERY_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/2/query"
TNM_BUCKET_URL = "https://prd-tnm.s3.amazonaws.com"
NHDPLUS_HR_PREFIX = "StagedProducts/Hydrography/NHDPlusHR/VPU/Current"
# Bucket subfolder for each product
NHDPLUS_HR_PRODUCTS = {"gdb": "GDB", "raster": "Raster"}

# Stage heights (m) used for Mountain Dell Dam: 15 values from 0.1 to 40 m
DEFAULT_STAGES = [0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40]

HTTP_TIMEOUT = 120  # seconds


def _get_json(url, params):
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full_url, timeout=HTTP_TIMEOUT) as response:
        return json.load(response)


def find_hu4_for_point(lat, lon):
    """Return the 4-digit HUC (e.g. "1602") containing the point (WGS84 lat/lon)."""
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "huc4,name",
        "returnGeometry": "false",
        "f": "json",
    }
    result = _get_json(WBD_HU4_QUERY_URL, params)
    if "error" in result:
        raise RuntimeError(f"WBD query failed: {result['error']}")
    features = result.get("features", [])
    if not features:
        raise ValueError(f"No HU4 found at lat={lat}, lon={lon}; is the point inside the US?")
    attrs = features[0]["attributes"]
    logger.info(f"Dam is in HU4 {attrs['huc4']} ({attrs['name']})")
    return attrs["huc4"]


def nhdplus_hr_urls(hu4):
    """
    Return {"gdb": (url, size_bytes), "raster": (url, size_bytes)} for the HU4's
    NHDPlus HR zips. Names may carry a release date (NHDPLUS_H_1602_HU4_20220412_GDB.zip),
    so they are found by listing the bucket rather than built from a pattern.
    """
    hu4 = str(hu4).zfill(4)
    urls = {}
    for key, folder in NHDPLUS_HR_PRODUCTS.items():
        prefix = f"{NHDPLUS_HR_PREFIX}/{folder}/NHDPLUS_H_{hu4}_HU4"
        listing_url = f"{TNM_BUCKET_URL}/?{urllib.parse.urlencode({'list-type': 2, 'prefix': prefix})}"
        with urllib.request.urlopen(listing_url, timeout=HTTP_TIMEOUT) as response:
            xml = response.read().decode()
        zips = [(k, int(s)) for k, s in
                re.findall(r"<Key>([^<]+\.zip)</Key>.*?<Size>(\d+)</Size>", xml, re.S)]
        if not zips:
            raise FileNotFoundError(f"No NHDPlus HR {folder} zip for HU4 {hu4} under {prefix}")
        key_name, size = sorted(zips)[-1]  # latest release if several
        urls[key] = (f"{TNM_BUCKET_URL}/{key_name}", size)
    return urls


def _download(url, dest, size=None):
    """Stream url to dest, printing progress every ~5%."""
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info(f"Downloading {url} ({size / 1e9:.2f} GB)" if size else f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response, open(tmp, "wb") as out:
        total = size or int(response.headers.get("Content-Length", 0))
        done, next_report = 0, 0.05
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total and done / total >= next_report:
                print(f"  {dest.name}: {done / total:4.0%} of {total / 1e9:.2f} GB", flush=True)
                next_report += 0.05
    tmp.rename(dest)
    return dest


def _extract(zip_path, target_dir):
    """
    Unzip into target_dir, skipping macOS metadata. If the zip wraps everything in
    one folder named like target_dir, flatten it, so the GDB / hydrodem.tif end up
    directly in target_dir (where find_nhdplus_gdb / find_nhdplus_raster_dir look).

    Extracts into <target_dir>.partial and renames it to target_dir only when done, so
    an interrupted extraction is never mistaken for a complete one; a leftover
    .partial folder is removed and the extraction starts over.
    """
    target_dir = Path(target_dir)
    partial = target_dir.with_name(target_dir.name + ".partial")
    if partial.exists():
        logger.info(f"Removing incomplete extraction {partial}")
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    logger.info(f"Extracting {zip_path.name} to {target_dir}")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX/")]
        z.extractall(partial, members=members)
    nested = partial / target_dir.name
    if nested.is_dir():
        for item in nested.iterdir():
            shutil.move(str(item), partial / item.name)
        nested.rmdir()
    partial.rename(target_dir)


def download_nhdplus_hr(hu4, dam_dir, cache_dir=None, keep_zips=False):
    """
    Make the HU4's NHDPlus HR GDB and raster folder available in dam_dir.

    Does nothing if dam_dir already has them. Otherwise each product is downloaded
    and extracted to <cache_dir>/<zip name>/ (skipped if that folder exists already),
    then symlinked into dam_dir when cache_dir differs from dam_dir. Use a shared
    cache_dir (e.g. data/) for several dams in the same HU4.

    Args:
        hu4: 4-digit HUC code
        dam_dir: per-dam input directory
        cache_dir: where to keep extracted NHDPlus data (default: dam_dir)
        keep_zips: keep the downloaded zip files after extraction

    Returns:
        (gdb_path, raster_dir) as found by the utils.py finders
    """
    dam_dir = Path(dam_dir).expanduser().resolve()
    dam_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = dam_dir if cache_dir is None else Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    finders = {"gdb": find_nhdplus_gdb, "raster": find_nhdplus_raster_dir}
    missing = []
    for key, finder in finders.items():
        try:
            logger.info(f"NHDPlus HR {key} already in dam folder: {finder(dam_dir)}")
        except FileNotFoundError:
            missing.append(key)
    if missing:
        urls = nhdplus_hr_urls(hu4)
        for key in missing:
            url, size = urls[key]
            zip_name = url.rsplit("/", 1)[-1]
            extracted = cache_dir / Path(zip_name).stem
            if extracted.is_dir():
                logger.info(f"Using cached {extracted}")
            else:
                free = shutil.disk_usage(cache_dir).free
                if free < 3 * size:  # zip + extracted contents
                    logger.warning(f"Only {free / 1e9:.1f} GB free in {cache_dir}; "
                                   f"{zip_name} needs about {3 * size / 1e9:.1f} GB")
                zip_path = cache_dir / zip_name
                if not zip_path.exists():
                    _download(url, zip_path, size)
                _extract(zip_path, extracted)
                if not keep_zips:
                    zip_path.unlink()
            link = dam_dir / extracted.name
            if extracted != link and not link.exists():
                link.symlink_to(extracted, target_is_directory=True)
                logger.info(f"Linked {link} -> {extracted}")
    return find_nhdplus_gdb(dam_dir), find_nhdplus_raster_dir(dam_dir)


def write_dam_point(dam_dir, lat, lon, name, tolerance_m=10.0):
    """
    Write <dam_dir>/<name>_dam_location.shp (EPSG:4326) unless the folder already has a
    dam shapefile. Returns the dam shapefile path.

    An existing shapefile is never overwritten (it may be the user's own data). If its point
    is more than tolerance_m from (lat, lon), a ValueError is raised, so changed coordinates
    are not silently ignored.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    dam_dir = Path(dam_dir).expanduser().resolve()
    dam_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dam_dir.glob(f"*{DAM_SHAPEFILE_NAME}"))
    if existing:
        current = gpd.read_file(existing[0]).to_crs(4326).geometry.iloc[0]
        # Distance in meters, in the point's UTM zone
        pts = gpd.GeoSeries([current, Point(lon, lat)], crs=4326)
        offset = pts.to_crs(pts.estimate_utm_crs())
        offset_m = offset.iloc[0].distance(offset.iloc[1])
        if offset_m > tolerance_m:
            raise ValueError(
                f"{existing[0]} holds a dam point at lat {current.y:.6f}, lon {current.x:.6f}, "
                f"{offset_m:.0f} m from the requested lat {lat}, lon {lon}. Delete that shapefile "
                "to use the new coordinates, or change the coordinates to match it."
            )
        logger.info(f"Using existing dam shapefile: {existing[0]}")
        return existing[0]
    path = dam_dir / f"{name}_{DAM_SHAPEFILE_NAME}"
    gpd.GeoDataFrame({"ID": [1], "Dam_Name": [name]},
                     geometry=[Point(lon, lat)], crs="EPSG:4326").to_file(path)
    logger.info(f"Wrote dam shapefile: {path}")
    return path


def write_stage_file(dam_dir, stages=None, overwrite=False):
    """Write <dam_dir>/stage.txt (header 'Stage', one height in m per line)."""
    path = Path(dam_dir).expanduser().resolve() / "stage.txt"
    if path.exists() and not overwrite:
        logger.info(f"Using existing stage file: {path}")
        return path
    stages = DEFAULT_STAGES if stages is None else stages
    path.write_text("Stage\n" + "\n".join(f"{s:g}" for s in stages) + "\n")
    logger.info(f"Wrote {len(stages)} stage heights to {path}")
    return path


def write_uniform_forecast(output_dir, flow_m3s, name=None):
    """
    Write a forecast with the same flow on every reach in <output_dir>/catchlist.csv.
    Default file name: forecast_uniform<flow>.csv. Returns the path.
    """
    import pandas as pd

    from dam_inundation import FORECAST_FLOW_COL, FORECAST_ID_COL

    output_dir = Path(output_dir).expanduser().resolve()
    catchlist = pd.read_csv(output_dir / "catchlist.csv", skipinitialspace=True)
    path = output_dir / (name or f"forecast_uniform{flow_m3s:g}.csv")
    pd.DataFrame({FORECAST_ID_COL: catchlist["Id"], FORECAST_FLOW_COL: flow_m3s}).to_csv(path, index=False)
    logger.info(f"Wrote uniform {flow_m3s:g} m3/s forecast for {len(catchlist)} reaches: {path}")
    return path
