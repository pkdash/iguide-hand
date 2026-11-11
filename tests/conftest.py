"""
Shared fixtures: tiny but real NHDPlus HR stand-ins, served offline.

The fake HU4 "9999" has a GDB zip with the .gdb at the zip root and a raster zip that, like
the real TNM raster zips, wraps everything in one folder and carries __MACOSX entries.
"""

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import dam_setup

HU4 = "9999"
GDB_STEM = f"NHDPLUS_H_{HU4}_HU4_20240101_GDB"
RASTER_STEM = f"NHDPLUS_H_{HU4}_HU4_20240101_RASTER"
# A point inside the fake HUC12 (WGS84)
LAT, LON = 40.75, -111.72


def _write_gdb(path):
    huc12 = gpd.GeoDataFrame(
        {"huc12": ["999900000001"], "tohuc": ["CLOSED BASIN"], "name": ["Fake"]},
        geometry=[box(LON - 0.1, LAT - 0.1, LON + 0.1, LAT + 0.1)], crs="EPSG:4326")
    huc12.to_file(path, layer="WBDHU12", driver="OpenFileGDB")


def _write_hydrodem(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=3, width=3, count=1, dtype="float32",
                       crs="EPSG:5070", transform=from_origin(0, 30, 10, 10)) as dst:
        dst.write(np.arange(9, dtype="float32").reshape(1, 3, 3))


def _zip_dir(src_dir, zip_path, arc_root, macosx=False):
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in sorted(src_dir.rglob("*")):
            if f.is_file():
                arcname = Path(arc_root) / f.relative_to(src_dir)
                z.write(f, arcname.as_posix())
                if macosx:
                    z.writestr(f"__MACOSX/{arcname.parent.as_posix()}/._{f.name}", b"junk")


@pytest.fixture(scope="session")
def nhdplus_zips(tmp_path_factory):
    """{"gdb": (file:// url, size), "raster": (...)} for the fake HU4, as nhdplus_hr_urls returns."""
    root = tmp_path_factory.mktemp("tnm")
    gdb_src = root / "gdb_src"
    gdb_src.mkdir()
    _write_gdb(gdb_src / f"{GDB_STEM}.gdb")
    raster_src = root / "raster_src"
    _write_hydrodem(raster_src / "hydrodem.tif")

    bucket = root / "bucket"
    bucket.mkdir()
    gdb_zip, raster_zip = bucket / f"{GDB_STEM}.zip", bucket / f"{RASTER_STEM}.zip"
    _zip_dir(gdb_src, gdb_zip, ".")
    _zip_dir(raster_src, raster_zip, RASTER_STEM, macosx=True)
    return {key: (z.as_uri(), z.stat().st_size) for key, z in (("gdb", gdb_zip), ("raster", raster_zip))}


@pytest.fixture
def offline_tnm(monkeypatch, nhdplus_zips):
    """
    Point dam_setup at the fake bucket and record network calls.
    Returns a dict with lists "hu4_lookups", "url_lookups" and "downloads".
    """
    calls = {"hu4_lookups": [], "url_lookups": [], "downloads": []}

    def fake_find_hu4(lat, lon):
        calls["hu4_lookups"].append((lat, lon))
        return HU4

    def fake_urls(hu4):
        calls["url_lookups"].append(hu4)
        assert hu4 == HU4
        return nhdplus_zips

    real_download = dam_setup._download

    def spy_download(url, dest, size=None):
        calls["downloads"].append(url.rsplit("/", 1)[-1])
        return real_download(url, dest, size)

    monkeypatch.setattr(dam_setup, "find_hu4_for_point", fake_find_hu4)
    monkeypatch.setattr(dam_setup, "nhdplus_hr_urls", fake_urls)
    monkeypatch.setattr(dam_setup, "_download", spy_download)
    return calls
