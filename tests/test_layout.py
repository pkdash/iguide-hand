"""The dams/<dam>/{inputs,outputs} layout: utils helpers and the per-stage CLI defaults (offline)."""

import sys

import pytest

import dam_hand
import dam_hydroprop
import dam_setup
import generate_segment_outlets
from conftest import LAT, LON
from utils import (dam_inputs_dir, default_nhd_cache_dir, default_output_dir, find_nhdplus_gdb,
                   read_nhdplus_ref, write_nhdplus_ref)


# --- utils helpers ---------------------------------------------------------------------------

def test_default_output_dir_does_not_depend_on_cwd(tmp_path, monkeypatch):
    dam = tmp_path / "dams" / "x"
    monkeypatch.chdir(tmp_path)
    assert default_output_dir("dams/x") == dam / "outputs"
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert default_output_dir(dam) == dam / "outputs"


def test_default_nhd_cache_dir_is_next_to_dams(tmp_path):
    assert default_nhd_cache_dir(tmp_path / "dams" / "x") == tmp_path / "nhdplus"


def test_dam_inputs_dir(tmp_path):
    dam = tmp_path / "dams" / "x"
    with pytest.raises(FileNotFoundError, match="No inputs/ folder"):
        dam_inputs_dir(dam)
    assert dam_inputs_dir(dam, create=True) == dam / "inputs"
    assert (dam / "inputs").is_dir()
    assert dam_inputs_dir(dam) == dam / "inputs"


def test_dam_inputs_dir_does_not_detect_the_old_flat_layout(tmp_path):
    dam = tmp_path / "data" / "x"
    dam.mkdir(parents=True)
    (dam / "stage.txt").write_text("")
    with pytest.raises(FileNotFoundError, match="No inputs/ folder"):
        dam_inputs_dir(dam)


def test_dam_inputs_dir_rejects_the_inputs_folder_itself(tmp_path):
    inputs = tmp_path / "dams" / "x" / "inputs"
    inputs.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="pass the dam folder"):
        dam_inputs_dir(inputs)


# --- HandDamWorkflow ---------------------------------------------------------------------------

def test_hand_workflow_reads_inputs_and_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(dam_hand, "setup_taudem_path", lambda path: None)
    dam = tmp_path / "dams" / "x"
    shp = dam_setup.write_dam_point(dam / "inputs", LAT, LON, "x")
    dam_setup.write_stage_file(dam / "inputs")

    monkeypatch.chdir(tmp_path)  # relative --dam-dir, as typed on the command line
    wf = dam_hand.HandDamWorkflow(dam_dir="dams/x", taudem_path=tmp_path)
    assert wf.dam_dir == dam
    assert wf.inputs_dir == dam / "inputs"
    assert wf.output_dir == dam / "outputs" and wf.output_dir.is_dir()
    assert wf.clipped_dir == dam / "outputs" / "clipped"
    assert wf.dam_shapefile == shp

    wf.copy_input_files()
    assert (dam / "outputs" / "stage.txt").exists()


def test_hand_workflow_needs_an_inputs_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(dam_hand, "setup_taudem_path", lambda path: None)
    dam_setup.write_dam_point(tmp_path / "x", LAT, LON, "x")
    with pytest.raises(FileNotFoundError, match="No inputs/ folder"):
        dam_hand.HandDamWorkflow(dam_dir=tmp_path / "x", taudem_path=tmp_path)


# --- dam_hydroprop / dam_outlets CLI defaults ------------------------------------------------

class Recorder:
    """Stands in for a stage class: records its keyword arguments, does nothing."""
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        Recorder.instances.append(self)

    def run(self):
        pass

    def generate(self, **kwargs):
        pass


@pytest.fixture
def run_main(monkeypatch, tmp_path):
    Recorder.instances = []
    monkeypatch.setattr(dam_hydroprop, "DamHydroProp", Recorder)
    monkeypatch.setattr(generate_segment_outlets, "SegmentOutletGenerator", Recorder)
    monkeypatch.chdir(tmp_path)

    def run(module, *argv):
        monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, argv)])
        module.main()
        return Recorder.instances[-1].kwargs

    return run


def test_hydroprop_default_output_dir(run_main, tmp_path):
    kwargs = run_main(dam_hydroprop, "--dam-dir", "dams/x")
    assert kwargs["output_dir"] == tmp_path / "dams" / "x" / "outputs"
    assert run_main(dam_hydroprop, "--output-dir", "o")["output_dir"] == "o"


def test_outlets_default_work_dir(run_main, tmp_path):
    kwargs = run_main(generate_segment_outlets, "--dam-dir", "dams/x")
    assert kwargs["work_dir"] == tmp_path / "dams" / "x" / "outputs"
    assert run_main(generate_segment_outlets, "--work-dir", "w")["work_dir"] == "w"


@pytest.mark.parametrize("module, message", [
    (dam_hydroprop, "pass --dam-dir"),
    (generate_segment_outlets, "pass --dam-dir"),
])
def test_dam_dir_or_output_dir_is_required(run_main, capsys, module, message):
    with pytest.raises(SystemExit) as exc:
        run_main(module)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_default_nhd_cache_dir_outside_dams_stays_in_dam_folder(tmp_path):
    assert default_nhd_cache_dir(tmp_path / "my_dam") == tmp_path / "my_dam" / "nhdplus"


BAD_REFS = ["[]", '"text"', '{"gdb": 5}', '{"raster_dir": ["x"]}', "{not json"]


@pytest.mark.parametrize("text", BAD_REFS)
def test_unusable_nhdplus_json_reads_as_empty_and_names_the_file(tmp_path, caplog, text):
    (tmp_path / "nhdplus.json").write_text(text)
    assert read_nhdplus_ref(tmp_path) == {}
    assert "nhdplus.json" in caplog.text
    with pytest.raises(FileNotFoundError):  # not TypeError / AttributeError
        find_nhdplus_gdb(tmp_path)


@pytest.mark.parametrize("text", BAD_REFS)
def test_writing_over_unusable_nhdplus_json_keeps_a_backup(tmp_path, text):
    ref = tmp_path / "nhdplus.json"
    ref.write_text(text)
    write_nhdplus_ref(tmp_path, hu4="1602")
    assert (tmp_path / "nhdplus.json.bak").read_text() == text
    assert read_nhdplus_ref(tmp_path)["hu4"] == "1602"
