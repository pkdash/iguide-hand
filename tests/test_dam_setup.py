"""Offline tests for dam_setup.py (dam folder preparation used by the notebook and the CLI)."""

import io
import zipfile

import geopandas as gpd
import pytest

import dam_setup
from conftest import GDB_STEM, HU4, LAT, LON, RASTER_STEM
from utils import find_dam_shapefile


# --- find_hu4_for_point ------------------------------------------------------------------

def test_find_hu4_for_point_returns_huc4(monkeypatch):
    seen = {}

    def fake_get_json(url, params):
        seen.update(params)
        return {"features": [{"attributes": {"huc4": "1602", "name": "Great Salt Lake"}}]}

    monkeypatch.setattr(dam_setup, "_get_json", fake_get_json)
    assert dam_setup.find_hu4_for_point(40.75, -111.72) == "1602"
    assert seen["geometry"] == "-111.72,40.75"  # x (lon) first
    assert seen["inSR"] == 4326


def test_find_hu4_for_point_no_feature(monkeypatch):
    monkeypatch.setattr(dam_setup, "_get_json", lambda url, params: {"features": []})
    with pytest.raises(ValueError, match="No HU4"):
        dam_setup.find_hu4_for_point(0, 0)


def test_find_hu4_for_point_service_error(monkeypatch):
    monkeypatch.setattr(dam_setup, "_get_json", lambda url, params: {"error": {"code": 500}})
    with pytest.raises(RuntimeError, match="WBD query failed"):
        dam_setup.find_hu4_for_point(40.75, -111.72)


# --- nhdplus_hr_urls ---------------------------------------------------------------------

def _s3_listing(*entries):
    contents = "".join(f"<Contents><Key>{k}</Key><LastModified>x</LastModified>"
                       f"<Size>{s}</Size></Contents>" for k, s in entries)
    return f'<?xml version="1.0"?><ListBucketResult>{contents}</ListBucketResult>'.encode()


def test_nhdplus_hr_urls_picks_latest_release(monkeypatch):
    prefix = dam_setup.NHDPLUS_HR_PREFIX
    listings = {
        "GDB": _s3_listing((f"{prefix}/GDB/NHDPLUS_H_0101_HU4_20200101_GDB.zip", 10),
                           (f"{prefix}/GDB/NHDPLUS_H_0101_HU4_20220412_GDB.zip", 20)),
        "Raster": _s3_listing((f"{prefix}/Raster/NHDPLUS_H_0101_HU4_20220412_RASTER.zip", 30)),
    }
    requested = []

    def fake_urlopen(url, timeout=None):
        requested.append(url)
        folder = "Raster" if "%2FRaster%2F" in url else "GDB"
        return io.BytesIO(listings[folder])

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", fake_urlopen)
    urls = dam_setup.nhdplus_hr_urls(101)  # zero-padded to "0101"
    bucket = dam_setup.TNM_BUCKET_URL
    assert urls == {
        "gdb": (f"{bucket}/{prefix}/GDB/NHDPLUS_H_0101_HU4_20220412_GDB.zip", 20),
        "raster": (f"{bucket}/{prefix}/Raster/NHDPLUS_H_0101_HU4_20220412_RASTER.zip", 30),
    }
    assert all("NHDPLUS_H_0101_HU4" in u for u in requested)


def test_nhdplus_hr_urls_missing_product(monkeypatch):
    monkeypatch.setattr(dam_setup.urllib.request, "urlopen",
                        lambda url, timeout=None: io.BytesIO(_s3_listing()))
    with pytest.raises(FileNotFoundError, match="No NHDPlus HR GDB zip for HU4 0101"):
        dam_setup.nhdplus_hr_urls("0101")


# --- _extract ------------------------------------------------------------------------------

def test_extract_flattens_wrapper_folder_and_skips_macosx(tmp_path):
    zip_path = tmp_path / "X_RASTER.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("X_RASTER/hydrodem.tif", b"tif")
        z.writestr("X_RASTER/sub/a.txt", b"a")
        z.writestr("__MACOSX/X_RASTER/._hydrodem.tif", b"junk")
    target = tmp_path / "X_RASTER"
    stale = tmp_path / "X_RASTER.partial"
    stale.mkdir()
    (stale / "leftover").write_text("old")

    dam_setup._extract(zip_path, target)

    assert (target / "hydrodem.tif").read_bytes() == b"tif"
    assert (target / "sub" / "a.txt").exists()
    assert not (target / "X_RASTER").exists()
    assert not (target / "__MACOSX").exists()
    assert not (target / "leftover").exists()
    assert not stale.exists()


# --- download_nhdplus_hr -----------------------------------------------------------------

def test_download_into_dam_dir_and_reuse(tmp_path, offline_tnm):
    dam_dir = tmp_path / "dam"
    gdb, raster_dir = dam_setup.download_nhdplus_hr(HU4, dam_dir)

    assert gdb == dam_dir / GDB_STEM / f"{GDB_STEM}.gdb"
    assert raster_dir == dam_dir / RASTER_STEM
    assert (raster_dir / "hydrodem.tif").exists()
    assert sorted(offline_tnm["downloads"]) == [f"{GDB_STEM}.zip", f"{RASTER_STEM}.zip"]
    assert not list(dam_dir.glob("*.zip"))  # zips removed by default

    # Second call: data found in the dam dir, nothing listed or downloaded
    offline_tnm["downloads"].clear()
    assert dam_setup.download_nhdplus_hr(HU4, dam_dir) == (gdb, raster_dir)
    assert offline_tnm["downloads"] == []
    assert offline_tnm["url_lookups"] == [HU4]


def test_download_into_shared_cache_links_dam_dirs(tmp_path, offline_tnm):
    cache = tmp_path / "cache"
    dam_a, dam_b = tmp_path / "a", tmp_path / "b"
    dam_setup.download_nhdplus_hr(HU4, dam_a, cache_dir=cache, keep_zips=True)
    gdb_b, raster_b = dam_setup.download_nhdplus_hr(HU4, dam_b, cache_dir=cache)

    assert len(offline_tnm["downloads"]) == 2  # second dam reused the cache
    assert (cache / f"{RASTER_STEM}.zip").exists()  # keep_zips
    for dam in (dam_a, dam_b):
        assert (dam / RASTER_STEM).is_symlink()
        assert (dam / RASTER_STEM).resolve() == (cache / RASTER_STEM).resolve()
    assert (raster_b / "hydrodem.tif").exists()
    assert gdb_b.name == f"{GDB_STEM}.gdb"


def test_download_only_requested_products(tmp_path, offline_tnm):
    gdb, raster_dir = dam_setup.download_nhdplus_hr(HU4, tmp_path / "dam", products=("raster",))
    assert gdb is None
    assert (raster_dir / "hydrodem.tif").exists()
    assert offline_tnm["downloads"] == [f"{RASTER_STEM}.zip"]


def test_download_unknown_product(tmp_path):
    with pytest.raises(ValueError, match="Unknown NHDPlus HR product"):
        dam_setup.download_nhdplus_hr(HU4, tmp_path, products=("dem",))


# --- dam_slug ------------------------------------------------------------------------------

@pytest.mark.parametrize("label, slug", [
    ("Mountain Dell Dam", "mountain_dell"),
    ("mountain_dell", "mountain_dell"),           # already a slug: unchanged
    ("Quail Creek", "quail_creek"),
    ("  St. Mary's  Dam ", "st_mary_s"),
    ("Presa Peñitas", "presa_penitas"),           # accents dropped
    ("Lake Dam Reservoir", "lake_dam_reservoir"),  # only a trailing "dam" is dropped
    ("Hoover-Dam", "hoover"),
    ("Dam", "dam"),                               # nothing before "dam": kept
])
def test_dam_slug(label, slug):
    assert dam_setup.dam_slug(label) == slug


@pytest.mark.parametrize("label", ["", "  ", "!!!", "水坝"])
def test_dam_slug_empty_raises(label):
    with pytest.raises(ValueError, match="Cannot make a folder name"):
        dam_setup.dam_slug(label)


# --- write_dam_point / write_stage_file ----------------------------------------------------

def test_write_dam_point_new(tmp_path):
    path = dam_setup.write_dam_point(tmp_path / "dam", LAT, LON, "my")
    assert path.name == "my_dam_location.shp"
    assert find_dam_shapefile(tmp_path / "dam") == path
    gdf = gpd.read_file(path)
    assert gdf.crs.to_epsg() == 4326
    assert (gdf.geometry.iloc[0].y, gdf.geometry.iloc[0].x) == pytest.approx((LAT, LON))
    assert gdf["Dam_Name"].iloc[0] == "my"  # no label: the name


def test_write_dam_point_label(tmp_path):
    path = dam_setup.write_dam_point(tmp_path, LAT, LON, "mountain_dell", label="Mountain Dell Dam")
    assert path.name == "mountain_dell_dam_location.shp"
    assert gpd.read_file(path)["Dam_Name"].iloc[0] == "Mountain Dell Dam"


def test_write_dam_point_existing_within_tolerance_is_kept(tmp_path):
    first = dam_setup.write_dam_point(tmp_path, LAT, LON, "my")
    # ~5 m north, different name: the existing file is reused, not rewritten
    again = dam_setup.write_dam_point(tmp_path, LAT + 0.000045, LON, "other")
    assert again == first
    assert len(list(tmp_path.glob("*.shp"))) == 1


def test_write_dam_point_checks_the_file_the_pipeline_uses(tmp_path):
    dam_setup.write_dam_point(tmp_path, LAT, LON, "abc")  # near the requested point
    gpd.GeoDataFrame({"ID": [1]}, geometry=gpd.points_from_xy([LON], [LAT + 0.05]),
                     crs=4326).to_file(tmp_path / "dam_location.shp")  # ~5.5 km away
    # find_dam_shapefile prefers dam_location.shp, so that is the one checked
    with pytest.raises(ValueError, match="dam_location.shp holds a dam point"):
        dam_setup.write_dam_point(tmp_path, LAT, LON, "abc")


def test_write_dam_point_ambiguous_existing_raises(tmp_path):
    dam_setup.write_dam_point(tmp_path / "a", LAT, LON, "one")
    for f in (tmp_path / "a").iterdir():
        (tmp_path / "a" / f.name.replace("one_", "two_")).write_bytes(f.read_bytes())
    with pytest.raises(FileNotFoundError, match="Several dam shapefiles"):
        dam_setup.write_dam_point(tmp_path / "a", LAT, LON, "one")


def test_truncated_download_raises_and_leaves_nothing(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 10)
    dest = tmp_path / "out.zip"
    with pytest.raises(OSError, match="stopped at 10 of 20 bytes"):
        dam_setup._download(src.as_uri(), dest, size=20)
    assert not dest.exists() and not (tmp_path / "out.zip.part").exists()


def test_write_dam_point_existing_far_raises(tmp_path):
    dam_setup.write_dam_point(tmp_path, LAT, LON, "my")
    with pytest.raises(ValueError, match="Delete that shapefile"):
        dam_setup.write_dam_point(tmp_path, LAT + 0.01, LON, "my")  # ~1.1 km


def test_write_stage_file_default_and_keep(tmp_path):
    path = dam_setup.write_stage_file(tmp_path)
    lines = path.read_text().split()
    assert lines[0] == "Stage"
    assert [float(v) for v in lines[1:]] == dam_setup.DEFAULT_STAGES
    path.write_text("Stage\n1\n2\n")
    dam_setup.write_stage_file(tmp_path)  # existing file kept
    assert path.read_text() == "Stage\n1\n2\n"
