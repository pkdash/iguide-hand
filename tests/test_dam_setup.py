"""Offline tests for dam_setup.py (dam folder preparation used by the notebook and the CLI)."""

import io
import json
import shutil
import urllib.error
import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiPoint, Point

import dam_setup
import utils
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


def test_get_json_retries_transient_gateway_errors(monkeypatch):
    monkeypatch.setattr(dam_setup.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def flaky(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(url, 504, "Gateway Time-out", None, None)
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", flaky)
    assert dam_setup._get_json("http://x", {"a": 1}) == {"ok": True}
    assert calls["n"] == 3


def test_get_json_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr(dam_setup.time, "sleep", lambda _: None)

    def bad(url, timeout=None):
        raise urllib.error.HTTPError(url, 400, "Bad Request", None, None)

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", bad)
    with pytest.raises(urllib.error.HTTPError):
        dam_setup._get_json("http://x", {})


@pytest.mark.parametrize("error", [TimeoutError("timed out"), ConnectionResetError(54, "reset"),
                                   urllib.error.URLError("temporary failure")])
def test_get_json_retries_timeouts_and_dropped_connections(monkeypatch, error):
    monkeypatch.setattr(dam_setup.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def flaky(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise error
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", flaky)
    assert dam_setup._get_json("http://x", {}) == {"ok": True}
    assert calls["n"] == 2


def test_get_json_retries_a_timeout_while_reading(monkeypatch):
    monkeypatch.setattr(dam_setup.time, "sleep", lambda _: None)
    calls = {"n": 0}

    class SlowBody(io.BytesIO):
        def read(self, *args):
            raise TimeoutError("read timed out")

    def flaky(url, timeout=None):
        calls["n"] += 1
        return SlowBody() if calls["n"] == 1 else io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", flaky)
    assert dam_setup._get_json("http://x", {}) == {"ok": True}


def test_get_json_gives_up_after_three_timeouts(monkeypatch):
    monkeypatch.setattr(dam_setup.time, "sleep", lambda _: None)
    calls = {"n": 0}

    def down(url, timeout=None):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", down)
    with pytest.raises(TimeoutError):
        dam_setup._get_json("http://x", {})
    assert calls["n"] == 3


def test_retried_http_error_is_closed(monkeypatch):
    monkeypatch.setattr(dam_setup.time, "sleep", lambda _: None)
    errors = []

    def flaky(url, timeout=None):
        if not errors:
            errors.append(urllib.error.HTTPError(url, 503, "Unavailable", None, io.BytesIO(b"x")))
            raise errors[0]
        return io.BytesIO(b"{}")

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", flaky)
    dam_setup._get_json("http://x", {})
    assert errors[0].fp is None or errors[0].fp.closed


def test_download_retries_a_gateway_error_on_open(tmp_path, monkeypatch):
    monkeypatch.setattr(dam_setup.time, "sleep", lambda _: None)
    real_urlopen = dam_setup.urllib.request.urlopen
    src = tmp_path / "src.zip"
    src.write_bytes(b"zipdata")
    calls = {"n": 0}

    def flaky(url, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(url, 503, "Slow Down", None, None)
        return real_urlopen(url, timeout=timeout)

    monkeypatch.setattr(dam_setup.urllib.request, "urlopen", flaky)
    assert dam_setup._download(src.as_uri(), tmp_path / "out.zip").read_bytes() == b"zipdata"
    assert calls["n"] == 2


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


@pytest.fixture
def windows_locks(monkeypatch):
    """Turn on the Windows-only lock retries, with no real waiting."""
    monkeypatch.setattr(dam_setup, "RETRY_FILE_LOCKS", True)
    sleeps = []
    monkeypatch.setattr(dam_setup.time, "sleep", sleeps.append)
    return sleeps


def _locked(times):
    """An action raising a WinError-5-like PermissionError the first `times` calls."""
    state = {"calls": 0}

    def action():
        state["calls"] += 1
        if state["calls"] <= times:
            raise PermissionError(5, "Access is denied")
        return "done"
    return action, state


def test_retry_on_lock_retries_on_windows_with_growing_waits(windows_locks):
    action, state = _locked(3)
    assert dam_setup._retry_on_lock(action, "Renaming x") == "done"
    assert state["calls"] == 4
    assert windows_locks == [1, 2, 4]


def test_retry_on_lock_gives_up_after_the_last_attempt(windows_locks):
    action, state = _locked(99)
    with pytest.raises(PermissionError):
        dam_setup._retry_on_lock(action, "Renaming x", attempts=3)
    assert state["calls"] == 3


def test_retry_on_lock_does_not_retry_off_windows(monkeypatch):
    monkeypatch.setattr(dam_setup, "RETRY_FILE_LOCKS", False)
    monkeypatch.setattr(dam_setup.time, "sleep",
                        lambda _: (_ for _ in ()).throw(AssertionError("waited off Windows")))
    action, state = _locked(1)
    with pytest.raises(PermissionError):
        dam_setup._retry_on_lock(action, "Renaming x")
    assert state["calls"] == 1  # a real permission problem fails at once, as before


def test_retry_on_lock_does_not_retry_other_errors(windows_locks):
    def missing():
        raise FileNotFoundError("gone")
    with pytest.raises(FileNotFoundError):
        dam_setup._retry_on_lock(missing, "Renaming x")
    assert windows_locks == []


def test_extract_retries_rename_blocked_by_transient_lock(tmp_path, monkeypatch, windows_locks):
    zip_path = tmp_path / "X_RASTER.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("hydrodem.tif", b"tif")
    target = tmp_path / "X_RASTER"

    real_rename = dam_setup.Path.rename
    state = {"failures": 2}

    def flaky_rename(self, dst):
        if state["failures"] and self.suffix == ".partial":
            state["failures"] -= 1
            raise PermissionError(5, "Access is denied")  # WinError 5
        return real_rename(self, dst)

    monkeypatch.setattr(dam_setup.Path, "rename", flaky_rename)

    dam_setup._extract(zip_path, target)

    assert (target / "hydrodem.tif").read_bytes() == b"tif"
    assert state["failures"] == 0


def test_extract_retries_removing_a_locked_stale_partial(tmp_path, monkeypatch, windows_locks):
    zip_path = tmp_path / "X_RASTER.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("hydrodem.tif", b"tif")
    stale = tmp_path / "X_RASTER.partial"
    stale.mkdir()
    real_rmtree = dam_setup.shutil.rmtree
    state = {"failures": 1}

    def flaky_rmtree(path, *args, **kwargs):
        if state["failures"]:
            state["failures"] -= 1
            raise PermissionError(32, "The process cannot access the file")  # WinError 32
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(dam_setup.shutil, "rmtree", flaky_rmtree)
    dam_setup._extract(zip_path, tmp_path / "X_RASTER")
    assert (tmp_path / "X_RASTER" / "hydrodem.tif").exists()
    assert state["failures"] == 0


def test_download_retries_rename_of_part_file(tmp_path, monkeypatch, windows_locks):
    src = tmp_path / "src.zip"
    src.write_bytes(b"zipdata")
    real_rename = dam_setup.Path.rename
    state = {"failures": 2}

    def flaky_rename(self, dst):
        if state["failures"] and self.suffix == ".part":
            state["failures"] -= 1
            raise PermissionError(5, "Access is denied")
        return real_rename(self, dst)

    monkeypatch.setattr(dam_setup.Path, "rename", flaky_rename)
    dest = dam_setup._download(src.as_uri(), tmp_path / "out.zip")
    assert dest.read_bytes() == b"zipdata"
    assert not (tmp_path / "out.zip.part").exists()


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


def test_download_into_shared_cache_is_referenced_not_linked(tmp_path, offline_tnm):
    cache = tmp_path / "cache"
    dam_a, dam_b = tmp_path / "a", tmp_path / "b"
    dam_setup.download_nhdplus_hr(HU4, dam_a, cache_dir=cache, keep_zips=True)
    gdb_b, raster_b = dam_setup.download_nhdplus_hr(HU4, dam_b, cache_dir=cache)

    assert len(offline_tnm["downloads"]) == 2  # second dam reused the cache
    assert offline_tnm["url_lookups"] == [HU4]  # ... without even listing the bucket
    assert (cache / f"{RASTER_STEM}.zip").exists()  # keep_zips
    for dam in (dam_a, dam_b):
        assert not any(p.is_symlink() for p in dam.iterdir())  # symlinks fail on Windows
        ref = utils.read_nhdplus_ref(dam)
        assert ref["hu4"] == HU4
        assert ref["raster_dir"] == cache / RASTER_STEM
        assert ref["gdb"] == cache / GDB_STEM / f"{GDB_STEM}.gdb"
    assert raster_b == cache / RASTER_STEM and (raster_b / "hydrodem.tif").exists()
    assert gdb_b == cache / GDB_STEM / f"{GDB_STEM}.gdb"


def test_pointer_survives_moving_the_whole_tree(tmp_path, offline_tnm):
    root = tmp_path / "root"
    dam_setup.download_nhdplus_hr(HU4, root / "dams" / "x" / "inputs", cache_dir=root / "nhdplus")
    shutil.move(root, tmp_path / "elsewhere")
    inputs = tmp_path / "elsewhere" / "dams" / "x" / "inputs"
    assert utils.find_nhdplus_raster_dir(inputs) == tmp_path / "elsewhere" / "nhdplus" / RASTER_STEM
    assert utils.find_nhdplus_gdb(inputs).name == f"{GDB_STEM}.gdb"


def test_pointer_stores_absolute_path_when_no_relative_one_exists(tmp_path, monkeypatch):
    def other_drive(path, start=None):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(utils.os.path, "relpath", other_drive)
    target = tmp_path / "r"
    (tmp_path / "in").mkdir()
    utils.write_nhdplus_ref(tmp_path / "in", raster_dir=target)
    raw = json.loads((tmp_path / "in" / "nhdplus.json").read_text())
    assert raw["raster_dir"] == str(target.resolve())


def test_data_in_inputs_wins_over_pointer(tmp_path, offline_tnm):
    inputs = tmp_path / "inputs"
    dam_setup.download_nhdplus_hr(HU4, inputs, cache_dir=tmp_path / "cache")
    local = inputs / "local_raster"
    local.mkdir()
    (local / "hydrodem.tif").write_bytes(b"")
    assert utils.find_nhdplus_raster_dir(inputs) == local


def test_stale_pointer_names_the_file(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    utils.write_nhdplus_ref(inputs, raster_dir=tmp_path / "gone", gdb=tmp_path / "gone.gdb")
    for finder in (utils.find_nhdplus_raster_dir, utils.find_nhdplus_gdb):
        with pytest.raises(FileNotFoundError, match="nhdplus.json"):
            finder(inputs)


def test_newer_cached_release_is_used(tmp_path, offline_tnm):
    cache = tmp_path / "cache"
    for stem in (f"NHDPLUS_H_{HU4}_HU4_20200101_RASTER", f"NHDPLUS_H_{HU4}_HU4_20230101_RASTER"):
        (cache / stem).mkdir(parents=True)
        (cache / stem / "hydrodem.tif").write_bytes(b"")
    _, raster_dir = dam_setup.download_nhdplus_hr(HU4, tmp_path / "in", cache_dir=cache,
                                                  products=("raster",))
    assert raster_dir.name == f"NHDPLUS_H_{HU4}_HU4_20230101_RASTER"
    assert offline_tnm["url_lookups"] == [] and offline_tnm["downloads"] == []


def test_dated_cached_release_beats_undated_and_choice_is_logged(tmp_path, offline_tnm, caplog):
    caplog.set_level("INFO")
    cache = tmp_path / "cache"
    for stem in (f"NHDPLUS_H_{HU4}_HU4_RASTER", f"NHDPLUS_H_{HU4}_HU4_20230101_RASTER"):
        (cache / stem).mkdir(parents=True)
        (cache / stem / "hydrodem.tif").write_bytes(b"")
    _, raster_dir = dam_setup.download_nhdplus_hr(HU4, tmp_path / "in", cache_dir=cache,
                                                  products=("raster",))
    assert raster_dir.name == f"NHDPLUS_H_{HU4}_HU4_20230101_RASTER"
    assert "cached releases" in caplog.text and f"NHDPLUS_H_{HU4}_HU4_RASTER" in caplog.text


def test_incomplete_cached_folder_says_to_force_download(tmp_path, offline_tnm):
    cache = tmp_path / "cache"
    (cache / RASTER_STEM).mkdir(parents=True)  # no hydrodem.tif
    with pytest.raises(FileNotFoundError, match="--force-download"):
        dam_setup.download_nhdplus_hr(HU4, tmp_path / "in", cache_dir=cache, products=("raster",))
    assert offline_tnm["downloads"] == []
    assert (cache / RASTER_STEM).is_dir()  # nothing deleted


def test_force_download_replaces_cached_folder(tmp_path, offline_tnm):
    cache, inputs = tmp_path / "cache", tmp_path / "in"
    dam_setup.download_nhdplus_hr(HU4, inputs, cache_dir=cache, keep_zips=True)
    (cache / RASTER_STEM / "hydrodem.tif").write_bytes(b"damaged")
    (cache / RASTER_STEM / "stray.txt").write_text("x")
    offline_tnm["downloads"].clear()
    gdb, raster_dir = dam_setup.download_nhdplus_hr(HU4, inputs, cache_dir=cache, force=True)
    assert sorted(offline_tnm["downloads"]) == [f"{GDB_STEM}.zip", f"{RASTER_STEM}.zip"]
    assert raster_dir == cache / RASTER_STEM
    assert (raster_dir / "hydrodem.tif").stat().st_size > len(b"damaged")
    assert not (raster_dir / "stray.txt").exists()
    assert not list(cache.glob("*.partial")) and not list(cache.glob("*.zip"))
    assert utils.read_nhdplus_ref(inputs)["raster_dir"] == raster_dir


def test_force_download_failure_keeps_the_old_copy(tmp_path, offline_tnm, monkeypatch):
    cache, inputs = tmp_path / "cache", tmp_path / "in"
    dam_setup.download_nhdplus_hr(HU4, inputs, cache_dir=cache, products=("raster",))

    def broken(url, dest, size=None):
        raise OSError("network down")

    monkeypatch.setattr(dam_setup, "_download", broken)
    with pytest.raises(OSError):
        dam_setup.download_nhdplus_hr(HU4, inputs, cache_dir=cache, products=("raster",), force=True)
    assert (cache / RASTER_STEM / "hydrodem.tif").exists()


def test_force_download_refuses_data_placed_in_inputs(tmp_path, offline_tnm):
    inputs = tmp_path / "in"
    local = inputs / "local_raster"
    local.mkdir(parents=True)
    (local / "hydrodem.tif").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="move or delete it first"):
        dam_setup.download_nhdplus_hr(HU4, inputs, cache_dir=tmp_path / "cache",
                                      products=("raster",), force=True)
    assert offline_tnm["downloads"] == [] and (local / "hydrodem.tif").exists()


def test_force_download_replaces_data_extracted_into_the_dam_folder(tmp_path, offline_tnm):
    dam_dir = tmp_path / "dam"  # cache_dir defaults to dam_dir
    dam_setup.download_nhdplus_hr(HU4, dam_dir, products=("raster",))
    offline_tnm["downloads"].clear()
    _, raster_dir = dam_setup.download_nhdplus_hr(HU4, dam_dir, products=("raster",), force=True)
    assert offline_tnm["downloads"] == [f"{RASTER_STEM}.zip"]
    assert (raster_dir / "hydrodem.tif").exists()


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


# --- read_dam_point -----------------------------------------------------------------------

def _shapefile(path, geoms, crs=4326):
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"ID": list(range(len(geoms)))}, geometry=geoms, crs=crs).to_file(path)
    return path


def test_read_dam_point_reprojects_to_wgs84(tmp_path):
    utm = gpd.GeoSeries(gpd.points_from_xy([LON], [LAT]), crs=4326).to_crs(32612)
    shp = _shapefile(tmp_path / "dam.shp", list(utm), crs=32612)
    assert dam_setup.read_dam_point(shp) == pytest.approx((LAT, LON))


@pytest.mark.parametrize("geoms, message", [
    ([Point(LON, LAT), Point(LON + 0.01, LAT)], "2 features"),
    ([LineString([(LON, LAT), (LON + 0.01, LAT)])], "LineString"),
    ([MultiPoint([(LON, LAT)])], "MultiPoint"),
    ([Point(500000, 4500000)], "not a valid location"),  # UTM numbers with a WGS84 .prj
])
def test_read_dam_point_rejects_anything_but_one_point(tmp_path, geoms, message):
    shp = _shapefile(tmp_path / "dam.shp", geoms)
    with pytest.raises(ValueError, match=message):
        dam_setup.read_dam_point(shp)


def test_read_dam_point_needs_prj_and_shp(tmp_path):
    shp = _shapefile(tmp_path / "dam.shp", [Point(LON, LAT)])
    shp.with_suffix(".prj").unlink()
    with pytest.raises(ValueError, match="no coordinate system"):
        dam_setup.read_dam_point(shp)
    with pytest.raises(ValueError, match="not a shapefile"):
        dam_setup.read_dam_point(tmp_path / "dam.gpkg")
    with pytest.raises(FileNotFoundError):
        dam_setup.read_dam_point(tmp_path / "missing.shp")


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


def test_locked_zip_after_extraction_is_a_warning(tmp_path, offline_tnm, monkeypatch, windows_locks,
                                                  caplog):
    real_unlink = dam_setup.Path.unlink

    def locked_unlink(self, *args, **kwargs):
        if self.suffix == ".zip":
            raise PermissionError(32, "The process cannot access the file")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(dam_setup.Path, "unlink", locked_unlink)
    gdb, raster_dir = dam_setup.download_nhdplus_hr(HU4, tmp_path / "dam")
    assert gdb.exists() and (raster_dir / "hydrodem.tif").exists()
    assert "delete it by hand" in caplog.text

