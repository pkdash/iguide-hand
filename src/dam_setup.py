"""
Dam input preparation helpers.

Fills a dam's input folder (``dams/<dam>/inputs/``) for the HAND pipeline from a dam's
latitude/longitude (or a point shapefile, read with ``read_dam_point``):

- ``find_hu4_for_point``: the HU4 that contains the dam (USGS WBD map service)
- ``nhdplus_hr_urls`` / ``download_nhdplus_hr``: the NHDPlus HR HU4 GDB and raster
  zips from the USGS National Map staged products bucket, extracted into a shared cache
  (default ``nhdplus/`` next to ``dams/``) and referenced from ``inputs/nhdplus.json``
- ``dam_slug``: folder/file name from a dam label (``"Mountain Dell Dam"`` -> ``mountain_dell``)
- ``read_dam_point``: lat/lon of the single point in a dam shapefile
- ``write_dam_point``: ``<name>_dam_location.shp``
- ``write_stage_file``: ``stage.txt``

Used by ``notebooks/dam_inundation_pipeline.ipynb`` and by ``dam_catchments --dam-lat/--dam-lon``
(or ``--dam-shapefile``);
not a console script itself. The uniform-flow forecast helper, ``write_uniform_forecast``, is in
``dam_inundation``.
"""

import json
import logging
import re
import shutil
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from utils import (DAM_SHAPEFILE_NAME, find_dam_shapefile, find_nhdplus_gdb, find_nhdplus_raster_dir,
                   write_nhdplus_ref)

logger = logging.getLogger(__name__)

WBD_HU4_QUERY_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/2/query"
TNM_BUCKET_URL = "https://prd-tnm.s3.amazonaws.com"
NHDPLUS_HR_PREFIX = "StagedProducts/Hydrography/NHDPlusHR/VPU/Current"
# Bucket subfolder for each product
NHDPLUS_HR_PRODUCTS = {"gdb": "GDB", "raster": "Raster"}

# Stage heights (m) used for Mountain Dell Dam: 15 values from 0.1 to 40 m
DEFAULT_STAGES = [0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40]

HTTP_TIMEOUT = 120  # seconds
RETRYABLE_HTTP_CODES = (502, 503, 504)  # transient gateway/server errors


def _urlopen_with_retry(url, attempts=3, delay=5.0):
    """
    urllib.request.urlopen, retrying transient HTTP errors: the USGS services occasionally
    answer a good request with 502/503/504.
    """
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(url, timeout=HTTP_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_HTTP_CODES or attempt == attempts:
                raise
            logger.warning(f"HTTP {e.code} from {url} (attempt {attempt}/{attempts}); retrying")
            time.sleep(delay)


def _get_json(url, params):
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    with _urlopen_with_retry(full_url) as response:
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
        with _urlopen_with_retry(listing_url) as response:
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
    if total and done != total:
        tmp.unlink()
        raise OSError(f"Download of {url} stopped at {done} of {total} bytes; rerun to retry")
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
    if target_dir.exists():  # only reached when replacing (--force-download), after a good extraction
        shutil.rmtree(target_dir)
    _rename_with_retry(partial, target_dir)


def _rename_with_retry(src, dst, attempts=5, delay=2.0):
    """
    src.rename(dst), retrying on transient Windows locks (WinError 5): antivirus and the
    search indexer briefly hold a handle on freshly extracted files.
    """
    for attempt in range(1, attempts + 1):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            logger.warning(f"Rename {src.name} -> {dst.name} blocked "
                           f"(attempt {attempt}/{attempts}); retrying")
            time.sleep(delay)


def _release_key(folder):
    """Sort key for cached product folders: a dated release (HU4_<yyyymmdd>_) beats an undated one."""
    m = re.search(r"_HU4_(\d{8})_", folder.name)
    return (m is not None, m.group(1) if m else "", folder.name)


def _cached_product(cache_dir, hu4, key):
    """
    The latest extracted <cache_dir>/NHDPLUS_H_<hu4>_HU4*_<GDB|RASTER> folder (by the release
    date in its name), or None. Logs the choice when several releases are cached.
    """
    suffix = {"gdb": "GDB", "raster": "RASTER"}[key]
    found = sorted((p for p in cache_dir.glob(f"NHDPLUS_H_{hu4}_HU4*_{suffix}") if p.is_dir()),
                   key=_release_key)
    if len(found) > 1:
        logger.info(f"Using {found[-1].name} of the cached releases "
                    f"{', '.join(p.name for p in found)}")
    return found[-1] if found else None


def _product_path(extracted, key):
    """The GDB (or raster folder) inside an extracted NHDPlus HR product folder."""
    if key == "gdb":
        return find_nhdplus_gdb(extracted)
    if not (extracted / "hydrodem.tif").exists():
        raise FileNotFoundError(f"No hydrodem.tif in {extracted}")
    return extracted


def download_nhdplus_hr(hu4, dam_dir, cache_dir=None, keep_zips=False, products=("gdb", "raster"),
                        force=False):
    """
    Make the HU4's NHDPlus HR GDB and raster folder available to a dam.

    Does nothing if dam_dir (the dam's inputs folder) already has them, either directly or
    through its nhdplus.json. Otherwise each product is extracted to <cache_dir>/<zip name>/
    (downloaded unless that folder exists already). When cache_dir differs from dam_dir,
    the location is recorded in <dam_dir>/nhdplus.json rather than symlinked, since symlinks
    need extra rights on Windows. Use a shared cache_dir for several dams in the same HU4.

    Args:
        hu4: 4-digit HUC code
        dam_dir: the dam's input folder (dams/<dam>/inputs)
        cache_dir: where to keep extracted NHDPlus data (default: dam_dir)
        keep_zips: keep the downloaded zip files after extraction
        products: which of "gdb" and "raster" to provide (e.g. only "raster" when the
            GDB is supplied from elsewhere)
        force: download the products again and replace the extracted folders in cache_dir,
            ignoring what is already there. The old folder is replaced only after the new
            extraction succeeded. Raises FileNotFoundError if a product sits directly in
            dam_dir outside cache_dir (the finders would keep preferring it); remove it first.

    Returns:
        (gdb_path, raster_dir) as found by the utils.py finders; None for a product
        not in products
    """
    dam_dir = Path(dam_dir).expanduser().resolve()
    dam_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = dam_dir if cache_dir is None else Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_finders = {"gdb": find_nhdplus_gdb, "raster": find_nhdplus_raster_dir}
    unknown = set(products) - set(all_finders)
    if unknown:
        raise ValueError(f"Unknown NHDPlus HR product(s) {sorted(unknown)}; use 'gdb' and/or 'raster'")
    finders = {k: f for k, f in all_finders.items() if k in products}
    missing = []
    for key, finder in finders.items():
        try:
            found = finder(dam_dir)
        except FileNotFoundError:
            missing.append(key)
            continue
        if not force:
            logger.info(f"NHDPlus HR {key} already in dam folder: {found}")
        elif found.is_relative_to(dam_dir) and not found.is_relative_to(cache_dir):
            raise FileNotFoundError(
                f"--force-download: the NHDPlus HR {key} {found} is in {dam_dir} itself, which "
                "is searched before the downloaded copy; move or delete it first")
        else:
            missing.append(key)
    urls = None
    for key in missing:
        extracted = None if force else _cached_product(cache_dir, hu4, key)
        if extracted is not None:
            logger.info(f"Using cached {extracted}")
            try:
                _product_path(extracted, key)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{e}; the cached folder is incomplete. Rerun with --force-download "
                    "to download it again (or delete the folder)") from e
        else:
            if urls is None:
                urls = nhdplus_hr_urls(hu4)
            url, size = urls[key]
            zip_name = url.rsplit("/", 1)[-1]
            extracted = cache_dir / Path(zip_name).stem
            free = shutil.disk_usage(cache_dir).free
            if free < 3 * size:  # zip + extracted contents
                logger.warning(f"Only {free / 1e9:.1f} GB free in {cache_dir}; "
                               f"{zip_name} needs about {3 * size / 1e9:.1f} GB")
            zip_path = cache_dir / zip_name
            if force:
                zip_path.unlink(missing_ok=True)  # a kept zip may be the stale copy
            if not zip_path.exists():
                _download(url, zip_path, size)
            try:
                _extract(zip_path, extracted)
            except zipfile.BadZipFile:
                zip_path.unlink()  # corrupt: download again next time
                raise
            if not keep_zips:
                zip_path.unlink()
        if cache_dir != dam_dir:
            path = _product_path(extracted, key)
            write_nhdplus_ref(dam_dir, hu4=hu4, **{"gdb" if key == "gdb" else "raster_dir": path})
    return tuple(finders[k](dam_dir) if k in finders else None for k in ("gdb", "raster"))


def dam_slug(label):
    """
    Folder/file name for a dam label: ASCII, lowercase, runs of other characters -> "_",
    and a trailing "dam" word dropped ("Mountain Dell Dam" -> "mountain_dell").
    Raises ValueError if nothing usable is left.
    """
    ascii_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_label.lower()).strip("_")
    slug = re.sub(r"_dam$", "", slug)
    if not slug:
        raise ValueError(f"Cannot make a folder name from dam label {label!r}; "
                         "use letters or digits, or set the name explicitly")
    return slug


def read_dam_point(path):
    """
    Return (lat, lon), WGS84, of the dam point in a shapefile.

    The file must be a .shp with a coordinate system (.prj) and exactly one feature, a
    single Point (not MultiPoint, line or polygon).

    Raises:
        FileNotFoundError: the file does not exist
        ValueError: not a shapefile, no CRS, not exactly one point, or the point does not
            convert to valid WGS84 coordinates (usually a wrong .prj)
    """
    import geopandas as gpd
    import math

    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".shp":
        raise ValueError(f"{path} is not a shapefile (.shp)")
    if not path.is_file():
        raise FileNotFoundError(f"Dam shapefile not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"{path} has no coordinate system; add its .prj file")
    if len(gdf) != 1:
        raise ValueError(f"{path} has {len(gdf)} features; it must hold exactly one dam point")
    geom = gdf.geometry.iloc[0]
    if geom is None or geom.is_empty or geom.geom_type != "Point":
        kind = "an empty geometry" if geom is None or geom.is_empty else f"a {geom.geom_type}"
        raise ValueError(f"{path} holds {kind}; the dam location must be a single Point")
    point = gdf.to_crs(4326).geometry.iloc[0]
    lat, lon = point.y, point.x
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"{path}: the point converts to lat {lat}, lon {lon}, which is not a "
                         f"valid location; check that its .prj ({gdf.crs.name}) matches the coordinates")
    return lat, lon


def write_dam_point(dam_dir, lat, lon, name, tolerance_m=10.0, label=None):
    """
    Write <dam_dir>/<name>_dam_location.shp (EPSG:4326) unless the folder already has a
    dam shapefile. Its Dam_Name attribute is label (default: name). Returns the dam
    shapefile path.

    An existing shapefile is never overwritten (it may be the user's own data). If its point
    is more than tolerance_m from (lat, lon), a ValueError is raised, so changed coordinates
    are not silently ignored.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    dam_dir = Path(dam_dir).expanduser().resolve()
    dam_dir.mkdir(parents=True, exist_ok=True)
    if any(dam_dir.glob(f"*{DAM_SHAPEFILE_NAME}")):
        # The same file the pipeline will use (raises if ambiguous)
        existing = find_dam_shapefile(dam_dir)
        points = gpd.read_file(existing).to_crs(4326).geometry
        if points.empty:
            raise ValueError(f"{existing} has no dam point; delete it to write a new one")
        current = points.iloc[0]
        # Distance in meters, in the point's UTM zone
        pts = gpd.GeoSeries([current, Point(lon, lat)], crs=4326)
        offset = pts.to_crs(pts.estimate_utm_crs())
        offset_m = offset.iloc[0].distance(offset.iloc[1])
        if offset_m > tolerance_m:
            raise ValueError(
                f"{existing} holds a dam point at lat {current.y:.6f}, lon {current.x:.6f}, "
                f"{offset_m:.0f} m from the requested lat {lat}, lon {lon}. Delete that shapefile "
                "to use the new coordinates, or change the coordinates to match it."
            )
        logger.info(f"Using existing dam shapefile: {existing}")
        return existing
    path = dam_dir / f"{name}_{DAM_SHAPEFILE_NAME}"
    gpd.GeoDataFrame({"ID": [1], "Dam_Name": [label or name]},
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
