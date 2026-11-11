"""dam_catchments CLI: preparing the dam folder from --dam-lat/--dam-lon (offline)."""

import json
import sys

import geopandas as gpd
import pytest

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
    dam_dir = tmp_path / "data" / "new_dam"
    out = tmp_path / "out"
    calc, calls = run_cli("--dam-dir", dam_dir, "--dam-lat", LAT, "--dam-lon", LON,
                          "--output-dir", out)

    shp = dam_dir / "new_dam_dam_location.shp"
    point = gpd.read_file(shp).to_crs(4326).geometry.iloc[0]
    assert (point.y, point.x) == pytest.approx((LAT, LON))
    assert (dam_dir / "stage.txt").read_text().startswith("Stage\n")
    assert calls["hu4_lookups"] == [(LAT, LON)]
    assert calc.kwargs["dam_shapefile"] == shp
    assert calc.kwargs["huc12_gdb_path"] == dam_dir / GDB_STEM / f"{GDB_STEM}.gdb"
    assert calc.kwargs["raster_dir"] == dam_dir / RASTER_STEM
    assert calc.kwargs["ref_projection_raster_path"] == dam_dir / RASTER_STEM / "hydrodem.tif"
    info = json.loads((out / "run_info.json").read_text())
    assert info["hu4"] == HU4
    assert info["dam_shapefile"] == str(shp)


def test_default_output_dir_and_dam_name(run_cli, tmp_path):
    calc, _ = run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON,
                      "--dam-name", "named")
    assert calc.kwargs["output_dir"] == tmp_path / "outputs" / "x"
    assert calc.kwargs["dam_shapefile"].name == "named_dam_location.shp"
    assert gpd.read_file(calc.kwargs["dam_shapefile"])["Dam_Name"].iloc[0] == "named"


def test_dam_label_sets_dam_name_field(run_cli, tmp_path):
    calc, _ = run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON,
                      "--dam-label", " Mountain Dell Dam ")
    assert calc.kwargs["dam_shapefile"].name == "x_dam_location.shp"
    assert gpd.read_file(calc.kwargs["dam_shapefile"])["Dam_Name"].iloc[0] == "Mountain Dell Dam"


def test_dam_label_needs_lat_lon_and_text(run_cli, capsys):
    for extra, msg in ((["--dam-label", "A Dam"], "need --dam-lat/--dam-lon"),
                       (["--dam-lat", LAT, "--dam-lon", LON, "--dam-label", " "], "must not be empty")):
        with pytest.raises(SystemExit) as exc:
            run_cli("--dam-dir", "data/x", *extra)
        assert exc.value.code == 2
        assert msg in capsys.readouterr().err


def test_hu4_flag_skips_lookup_and_cache_dir_links(run_cli, tmp_path):
    calc, calls = run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON,
                          "--hu4", HU4, "--nhd-cache-dir", "data")
    assert calls["hu4_lookups"] == []
    assert (tmp_path / "data" / RASTER_STEM / "hydrodem.tif").exists()
    assert (tmp_path / "data" / "x" / RASTER_STEM).is_symlink()


def test_existing_files_are_kept(run_cli, tmp_path):
    dam_dir = tmp_path / "data" / "x"
    dam_dir.mkdir(parents=True)
    (dam_dir / "stage.txt").write_text("Stage\n1\n")
    run_cli("--dam-dir", dam_dir, "--dam-lat", LAT, "--dam-lon", LON)
    calc, calls = run_cli("--dam-dir", dam_dir, "--dam-lat", LAT, "--dam-lon", LON)
    assert (dam_dir / "stage.txt").read_text() == "Stage\n1\n"
    assert len(calls["downloads"]) == 2  # first run only
    assert len(list(dam_dir.glob("*dam_location.shp"))) == 1


def test_huc12_gdb_override_downloads_only_raster(run_cli, tmp_path, nhdplus_zips):
    elsewhere = tmp_path / "elsewhere.gdb"
    elsewhere.mkdir()  # main() only checks it exists; the fake calculator does not read it
    calc, calls = run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON,
                          "--huc12-gdb", "./elsewhere.gdb")
    assert calls["downloads"] == [f"{RASTER_STEM}.zip"]
    assert calc.kwargs["huc12_gdb_path"] == elsewhere


def test_both_overrides_skip_hu4_and_download(run_cli, tmp_path):
    (tmp_path / "g.gdb").mkdir()
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "hydrodem.tif").write_bytes(b"")
    calc, calls = run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON,
                          "--huc12-gdb", "./g.gdb", "--raster-dir", "./r")
    assert calls["hu4_lookups"] == [] and calls["downloads"] == []
    assert not (tmp_path / "outputs" / "x" / "run_info.json").read_text().count('"hu4"')


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
        run_cli("--dam-dir", "data/x", *argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_moved_coordinates_fail_instead_of_reusing_old_point(run_cli, tmp_path, caplog):
    run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON)
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "data/x", "--dam-lat", LAT + 0.01, "--dam-lon", LON)
    assert exc.value.code == 1
    assert "Delete that shapefile" in caplog.text


def test_network_failure_is_a_clean_error(run_cli, monkeypatch, caplog):
    import urllib.error

    def offline(lat, lon):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(dam_catchments, "find_hu4_for_point", offline)
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert exc.value.code == 1
    assert "no network" in caplog.text


def test_without_lat_lon_behaviour_is_unchanged(run_cli, tmp_path, caplog):
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "data/missing")
    assert exc.value.code == 1
    assert "Dam directory not found" in caplog.text
    assert not (tmp_path / "data" / "missing").exists()

    (tmp_path / "data" / "empty").mkdir(parents=True)
    with pytest.raises(SystemExit):
        run_cli("--dam-dir", "data/empty")
    assert "--dam-lat/--dam-lon" in caplog.text  # hint in the missing-shapefile error


def test_rerun_with_data_present_needs_no_network(run_cli, monkeypatch):
    run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON)

    def offline(*args):
        raise AssertionError("network used although the data is in the dam dir")

    monkeypatch.setattr(dam_catchments, "find_hu4_for_point", offline)
    monkeypatch.setattr(dam_setup, "nhdplus_hr_urls", offline)
    calc, _ = run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert calc.kwargs["raster_dir"].name == RASTER_STEM


def test_hu4_recorded_only_when_downloaded(run_cli, tmp_path):
    run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON)
    info = tmp_path / "outputs" / "x" / "run_info.json"
    assert json.loads(info.read_text())["hu4"] == HU4
    info.unlink()
    run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON, "--hu4", "1601")
    assert "hu4" not in json.loads(info.read_text())  # nothing downloaded, --hu4 unverified


@pytest.mark.parametrize("name", ["../other", "a/b", "..", ""])
def test_dam_name_must_be_plain(run_cli, capsys, name):
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON, "--dam-name", name)
    assert exc.value.code == 2
    assert "plain name" in capsys.readouterr().err


def test_bad_zip_is_a_clean_error(run_cli, monkeypatch, tmp_path, caplog):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    monkeypatch.setattr(dam_setup, "nhdplus_hr_urls",
                        lambda hu4: {"gdb": (bad.as_uri(), 9), "raster": (bad.as_uri(), 9)})
    with pytest.raises(SystemExit) as exc:
        run_cli("--dam-dir", "data/x", "--dam-lat", LAT, "--dam-lon", LON)
    assert exc.value.code == 1
    assert "Could not prepare dam directory" in caplog.text
    assert not list((tmp_path / "data" / "x").glob("*.zip"))  # removed, so a rerun re-downloads
