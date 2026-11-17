"""dam_catchments CLI: preparing dams/<dam>/inputs from --dam-lat/--dam-lon or --dam-shapefile (offline)."""

import json
import sys

import geopandas as gpd
import pytest
from shapely.geometry import Point

import dam_catchments
import dam_setup
from conftest import GDB_STEM, HU4, LAT, LON, RASTER_STEM


class FakeCalculator:
    """Records constructor arguments instead of walking HUC12s; writes run_info like the real one."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_distance_km = None
        FakeCalculator.instances.append(self)

    def run_analysis(self, clip_rasters=True):
        self.kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        dam_catchments.write_run_info(self.kwargs["output_dir"],
                                      dam_shapefile=self.kwargs["dam_shapefile"])
        return None  # main() prints the summary only for a bounding box


@pytest.fixture
def run_cli(monkeypatch, offline_tnm, tmp_path):
    """Run dam_catchments.main() with the given arguments; returns (calculator, calls)."""
    FakeCalculator.instances = []
    monkeypatch.setattr(dam_catchments, "HUC12BoundingBoxCalculator", FakeCalculator)
    # dam_catchments holds its own reference to find_hu4_for_point
    monkeypatch.setattr(dam_catchments, "find_hu4_for_point", dam_setup.find_hu4_for_point)
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["dam_catchments", *map(str, argv)])
        dam_catchments.main()
        return FakeCalculator.instances[-1], offline_tnm

    return run


def test_lat_lon_builds_dam_dir_from_scratch(run_cli, tmp_path):
    dam_dir = tmp_path / "dams" / "new_dam"
    inputs, cache = dam_dir / "inputs", tmp_path / "nhdplus"
    out = tmp_path / "out"
    calc, calls = run_cli("--dam-dir", dam_dir, "--dam-lat", LAT, "--dam-lon", LON,
                          "--output-dir", out)

    shp = inputs / "new_dam_dam_location.shp"
    point = gpd.read_file(shp).to_crs(4326).geometry.iloc[0]
    assert (point.y, point.x) == pytest.approx((LAT, LON))
    assert (inputs / "stage.txt").read_text().startswith("Stage\n")
    assert calls["hu4_lookups"] == [(LAT, LON)]
    assert calc.kwargs["dam_shapefile"] == shp
    # NHDPlus data in the shared cache next to dams/, referenced from inputs/nhdplus.json
    assert calc.kwargs["huc12_gdb_path"] == cache / GDB_STEM / f"{GDB_STEM}.gdb"
    assert calc.kwargs["raster_dir"] == cache / RASTER_STEM
    assert calc.kwargs["ref_projection_raster_path"] == cache / RASTER_STEM / "hydrodem.tif"
    info = json.loads((out / "run_info.json").read_text())
    assert info["hu4"] == HU4
    assert info["dam_shapefile"] == str(shp)


def test_default_cache_is_shared_through_pointer_file_not_symlinks(run_cli, tmp_path):
    run_cli("--dam-dir", "dams/a", "--dam-lat", LAT, "--dam-lon", LON)
    calc, calls = run_cli("--dam-dir", "dams/b", "--dam-lat", LAT, "--dam-lon", LON)

    assert len(calls["downloads"]) == 2  # dam b reused dam a's download
    assert calc.kwargs["raster_dir"] == tmp_path / "nhdplus" / RASTER_STEM
    for dam in ("a", "b"):
        ref = json.loads((tmp_path / "dams" / dam / "inputs" / "nhdplus.json").read_text())
        assert ref["hu4"] == HU4
        assert ref["raster_dir"] == f"../../../nhdplus/{RASTER_STEM}"  # relative: tree can move
        assert ref["gdb"] == f"../../../nhdplus/{GDB_STEM}/{GDB_STEM}.gdb"
    assert not [p for p in (tmp_path / "dams").rglob("*") if p.is_symlink()]


def test_stale_pointer_is_repaired_from_the_cache(run_cli, tmp_path):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    # The cache moved: the pointer is stale
    (tmp_path / "nhdplus").rename(tmp_path / "moved")
    calc, calls = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                          "--nhd-cache-dir", "moved")
    assert len(calls["downloads"]) == 2  # first run only: the moved cache was found
    assert calc.kwargs["raster_dir"] == tmp_path / "moved" / RASTER_STEM


def test_stale_pointer_without_lat_lon_names_the_file(run_cli, tmp_path, caplog):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    (tmp_path / "nhdplus").rename(tmp_path / "moved")
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x")
    assert exc.value.code == 1
    assert "nhdplus.json" in caplog.text


def test_default_output_dir_and_dam_name(run_cli, tmp_path):
    calc, _ = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                      "--dam-name", "named")
    assert calc.kwargs["output_dir"] == tmp_path / "dams" / "x" / "outputs"
    assert calc.kwargs["dam_shapefile"].name == "named_dam_location.shp"
    assert gpd.read_file(calc.kwargs["dam_shapefile"])["Dam_Name"].iloc[0] == "named"


def test_dam_label_sets_dam_name_field(run_cli, tmp_path):
    calc, _ = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                      "--dam-label", " Mountain Dell Dam ")
    assert calc.kwargs["dam_shapefile"].name == "x_dam_location.shp"
    assert gpd.read_file(calc.kwargs["dam_shapefile"])["Dam_Name"].iloc[0] == "Mountain Dell Dam"


def test_dam_label_needs_lat_lon_and_text(run_cli, capsys):
    for extra, msg in ((["--dam-label", "A Dam"], "need --dam-lat/--dam-lon"),
                       (["--dam-lat", LAT, "--dam-lon", LON, "--dam-label", " "], "must not be empty")):
        with pytest.raises(SystemExit) as exc:
            run_cli("--dam-dir", "dams/x", *extra)
        assert exc.value.code == 2
        assert msg in capsys.readouterr().err


def test_hu4_flag_skips_lookup_and_cache_dir_is_referenced(run_cli, tmp_path):
    calc, calls = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                          "--hu4", HU4, "--nhd-cache-dir", "shared")
    assert calls["hu4_lookups"] == []
    assert (tmp_path / "shared" / RASTER_STEM / "hydrodem.tif").exists()
    ref = json.loads((tmp_path / "dams" / "x" / "inputs" / "nhdplus.json").read_text())
    assert ref["raster_dir"] == f"../../../shared/{RASTER_STEM}"
    assert calc.kwargs["raster_dir"] == tmp_path / "shared" / RASTER_STEM


def test_nhd_cache_dir_equal_to_inputs_extracts_there(run_cli, tmp_path):
    calc, _ = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                      "--nhd-cache-dir", "dams/x/inputs")
    inputs = tmp_path / "dams" / "x" / "inputs"
    assert calc.kwargs["raster_dir"] == inputs / RASTER_STEM
    assert not (inputs / "nhdplus.json").exists()


def test_existing_files_are_kept(run_cli, tmp_path):
    inputs = tmp_path / "dams" / "x" / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "stage.txt").write_text("Stage\n1\n")
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    calc, calls = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert (inputs / "stage.txt").read_text() == "Stage\n1\n"
    assert len(calls["downloads"]) == 2  # first run only
    assert len(list(inputs.glob("*dam_location.shp"))) == 1


def test_dam_dir_without_inputs_folder_is_an_error_without_coordinates(run_cli, tmp_path, caplog):
    old = tmp_path / "dams" / "x"
    dam_setup.write_dam_point(old, LAT, LON, "x")  # inputs directly in the dam folder
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", old)
    assert exc.value.code == 1
    assert "No inputs/ folder" in caplog.text
    assert not (old / "inputs").exists()


def test_inputs_folder_as_dam_dir_is_rejected(run_cli, tmp_path, caplog):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x/inputs")
    assert exc.value.code == 1
    assert "pass the dam folder" in caplog.text


def test_huc12_gdb_override_downloads_only_raster(run_cli, tmp_path, nhdplus_zips):
    elsewhere = tmp_path / "elsewhere.gdb"
    elsewhere.mkdir()  # main() only checks it exists; the fake calculator does not read it
    calc, calls = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                          "--huc12-gdb", "./elsewhere.gdb")
    assert calls["downloads"] == [f"{RASTER_STEM}.zip"]
    assert calc.kwargs["huc12_gdb_path"] == elsewhere


def test_both_overrides_skip_hu4_and_download(run_cli, tmp_path):
    (tmp_path / "g.gdb").mkdir()
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "hydrodem.tif").write_bytes(b"")
    calc, calls = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                          "--huc12-gdb", "./g.gdb", "--raster-dir", "./r")
    assert calls["hu4_lookups"] == [] and calls["downloads"] == []
    assert not (tmp_path / "dams" / "x" / "outputs" / "run_info.json").read_text().count('"hu4"')


@pytest.mark.parametrize("argv, message", [
    (["--dam-lat", LAT], "must be given together"),
    (["--dam-lon", LON], "must be given together"),
    (["--dam-lat", 95, "--dam-lon", LON], "outside [-90, 90]"),
    (["--dam-lat", LAT, "--dam-lon", -200], "outside [-180, 180]"),
    (["--dam-lat", LAT, "--dam-lon", LON, "--dam-shapefile", "d.shp"], "cannot be combined"),
    (["--hu4", "1602"], "need --dam-lat/--dam-lon"),
    (["--nhd-cache-dir", "data"], "need --dam-lat/--dam-lon"),
    (["--dam-lat", LAT, "--dam-lon", LON, "--hu4", "16"], "4-digit"),
])
def test_bad_flag_combinations_are_usage_errors(run_cli, capsys, argv, message):
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", *argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_moved_coordinates_fail_instead_of_reusing_old_point(run_cli, tmp_path, caplog):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-lat", LAT + 0.01, "--dam-lon", LON)
    assert exc.value.code == 1
    assert "Delete that shapefile" in caplog.text


def test_network_failure_is_a_clean_error(run_cli, monkeypatch, caplog):
    import urllib.error

    def offline(lat, lon):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(dam_catchments, "find_hu4_for_point", offline)
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert exc.value.code == 1
    assert "no network" in caplog.text


def test_without_lat_lon_behaviour_is_unchanged(run_cli, tmp_path, caplog):
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/missing")
    assert exc.value.code == 1
    assert "Dam directory not found" in caplog.text
    assert not (tmp_path / "dams" / "missing").exists()

    (tmp_path / "dams" / "empty").mkdir(parents=True)
    with pytest.raises(SystemExit):
        run_cli("--dam-dir", "dams/empty")
    assert "--dam-lat/--dam-lon" in caplog.text  # hint in the missing-inputs error
    assert not (tmp_path / "dams" / "empty" / "inputs").exists()

    (tmp_path / "dams" / "empty" / "inputs").mkdir()
    caplog.clear()
    with pytest.raises(SystemExit):
        run_cli("--dam-dir", "dams/empty")
    assert "--dam-lat/--dam-lon" in caplog.text  # hint in the missing-shapefile error


def test_rerun_with_data_present_needs_no_network(run_cli, monkeypatch):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)

    def offline(*args):
        raise AssertionError("network used although the data is in the dam dir")

    monkeypatch.setattr(dam_catchments, "find_hu4_for_point", offline)
    monkeypatch.setattr(dam_setup, "nhdplus_hr_urls", offline)
    calc, _ = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert calc.kwargs["raster_dir"].name == RASTER_STEM


def test_hu4_recorded_only_when_downloaded(run_cli, tmp_path):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    info = tmp_path / "dams" / "x" / "outputs" / "run_info.json"
    assert json.loads(info.read_text())["hu4"] == HU4
    info.unlink()
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON, "--hu4", HU4)
    assert "hu4" not in json.loads(info.read_text())  # nothing downloaded, --hu4 unverified


@pytest.mark.parametrize("name", ["../other", "a/b", "..", ""])
def test_dam_name_must_be_plain(run_cli, capsys, name):
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON, "--dam-name", name)
    assert exc.value.code == 2
    assert "plain name" in capsys.readouterr().err


def test_bad_zip_is_a_clean_error(run_cli, monkeypatch, tmp_path, caplog):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    monkeypatch.setattr(dam_setup, "nhdplus_hr_urls",
                        lambda hu4: {"gdb": (bad.as_uri(), 9), "raster": (bad.as_uri(), 9)})
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert exc.value.code == 1
    assert "Could not prepare dam directory" in caplog.text
    assert not list((tmp_path / "nhdplus").glob("*.zip"))  # removed, so a rerun re-downloads


def test_hu4_mismatch_with_pointer_is_an_error(run_cli, tmp_path, caplog):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    ref = tmp_path / "dams" / "x" / "inputs" / "nhdplus.json"
    before = ref.read_text()
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON, "--hu4", "1234")
    assert exc.value.code == 1
    assert "nhdplus.json" in caplog.text and HU4 in caplog.text and "1234" in caplog.text
    assert ref.read_text() == before  # nothing changed


def test_custom_output_dir_warns_about_later_stages(run_cli, caplog):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
            "--output-dir", "results/x")
    assert "same --output-dir" in caplog.text


def test_default_output_dir_does_not_warn(run_cli, caplog):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert "same --output-dir" not in caplog.text


def test_force_download_redownloads_existing_data(run_cli, tmp_path):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    raster = tmp_path / "nhdplus" / RASTER_STEM
    (raster / "hydrodem.tif").write_bytes(b"damaged")
    _, calls = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                       "--force-download")
    assert len(calls["downloads"]) == 4  # two on the first run, two again
    assert (raster / "hydrodem.tif").stat().st_size > len(b"damaged")


def test_force_download_leaves_products_given_by_flags_alone(run_cli, tmp_path):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    gdb = tmp_path / "nhdplus" / GDB_STEM / f"{GDB_STEM}.gdb"
    _, calls = run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON,
                       "--huc12-gdb", gdb, "--force-download")
    assert calls["downloads"][2:] == [f"{RASTER_STEM}.zip"]


def test_force_download_needs_coordinates(run_cli, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--force-download")
    assert exc.value.code == 2
    assert "--force-download" in capsys.readouterr().err


# --- --dam-shapefile -----------------------------------------------------------------------

def _point_shapefile(path, lat=LAT, lon=LON, crs=32612):
    """A one-point shapefile, in UTM by default so the copy has to reproject."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"ID": [1]}, geometry=[Point(lon, lat)], crs=4326).to_crs(crs).to_file(path)
    return path


def test_dam_shapefile_builds_dam_dir_from_scratch(run_cli, tmp_path):
    _point_shapefile(tmp_path / "survey" / "dam.shp")
    calc, calls = run_cli("--dam-dir", "dams/new", "--dam-shapefile", "survey/dam.shp",
                          "--dam-label", "New Dam")

    inputs = tmp_path / "dams" / "new" / "inputs"
    shp = inputs / "new_dam_location.shp"  # copied, so dam_hand / dam_outlets find it
    copy = gpd.read_file(shp)
    assert copy.crs.to_epsg() == 4326 and copy["Dam_Name"].iloc[0] == "New Dam"
    assert (copy.geometry.iloc[0].y, copy.geometry.iloc[0].x) == pytest.approx((LAT, LON))
    assert calls["hu4_lookups"] == [pytest.approx((LAT, LON))]
    assert len(calls["downloads"]) == 2
    assert (inputs / "stage.txt").exists()
    assert calc.kwargs["dam_shapefile"] == shp
    assert calc.kwargs["raster_dir"] == tmp_path / "nhdplus" / RASTER_STEM
    info = json.loads((tmp_path / "dams" / "new" / "outputs" / "run_info.json").read_text())
    assert info["hu4"] == HU4 and info["dam_shapefile"] == str(shp)


def test_dam_shapefile_in_inputs_is_used_as_is(run_cli, tmp_path):
    inputs = tmp_path / "dams" / "x" / "inputs"
    shp = _point_shapefile(inputs / "a_dam_location.shp")
    _point_shapefile(inputs / "b_dam_location.shp", lat=LAT + 0.05)  # picking one is allowed
    calc, calls = run_cli("--dam-dir", "dams/x", "--dam-shapefile", "a_dam_location.shp")
    assert calc.kwargs["dam_shapefile"] == shp
    assert len(calls["downloads"]) == 2
    assert sorted(p.name for p in inputs.glob("*.shp")) == ["a_dam_location.shp", "b_dam_location.shp"]


def test_dam_shapefile_found_without_flag_does_not_download(run_cli, tmp_path, caplog):
    _point_shapefile(tmp_path / "dams" / "x" / "inputs" / "dam_location.shp")
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x")
    assert exc.value.code == 1
    assert "WBDHU12" in caplog.text or ".gdb" in caplog.text  # missing NHDPlus data
    assert not list((tmp_path / "dams" / "x" / "inputs").glob("nhdplus.json"))


def test_dam_shapefile_rerun_needs_no_network(run_cli, monkeypatch, tmp_path):
    _point_shapefile(tmp_path / "dam.shp")
    run_cli("--dam-dir", "dams/x", "--dam-shapefile", "./dam.shp")

    def offline(*args):
        raise AssertionError("network used although the data is in the dam dir")

    monkeypatch.setattr(dam_catchments, "find_hu4_for_point", offline)
    monkeypatch.setattr(dam_setup, "nhdplus_hr_urls", offline)
    calc, _ = run_cli("--dam-dir", "dams/x", "--dam-shapefile", "./dam.shp")
    assert calc.kwargs["dam_shapefile"].name == "x_dam_location.shp"


def test_dam_shapefile_far_from_existing_point_fails(run_cli, tmp_path, caplog):
    run_cli("--dam-dir", "dams/x", "--dam-lat", LAT, "--dam-lon", LON)
    _point_shapefile(tmp_path / "moved.shp", lat=LAT + 0.01)
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-shapefile", "./moved.shp")
    assert exc.value.code == 1
    assert "Delete that shapefile" in caplog.text


def test_bad_dam_shapefile_fails_before_creating_anything(run_cli, tmp_path, caplog):
    gpd.GeoDataFrame({"ID": [1, 2]}, geometry=[Point(LON, LAT), Point(LON, LAT + 0.01)],
                     crs=4326).to_file(tmp_path / "two.shp")
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-shapefile", "./two.shp")
    assert exc.value.code == 1
    assert "exactly one dam point" in caplog.text
    assert not (tmp_path / "dams").exists()


def test_dam_shapefile_must_be_shp(run_cli, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "dams/x", "--dam-shapefile", "dam.gpkg")
    assert exc.value.code == 2
    assert "must be a shapefile" in capsys.readouterr().err
