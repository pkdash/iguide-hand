"""dam_inundation CLI (exactly one of --forecast-file / --uniform-flow) and write_uniform_forecast."""

import sys

import pandas as pd
import pytest

import dam_inundation


class FakeAnalysis:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.forecast_file = kwargs["forecast_file"]
        FakeAnalysis.instances.append(self)

    def run_analysis(self):
        pass


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    FakeAnalysis.instances = []
    monkeypatch.setattr(dam_inundation, "DamInundationAnalysis", FakeAnalysis)
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["dam_inundation", *map(str, argv)])
        dam_inundation.main()
        return FakeAnalysis.instances[-1]

    return run


@pytest.fixture
def out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "catchlist.csv").write_text("Id, Slope, Length, Manning_n\n7, 0.01, 2000, 0.05\n"
                                       "9, 0.02, 1500, 0.05\n")
    return out


@pytest.mark.parametrize("argv, message", [
    ([], "one of the arguments --forecast-file --uniform-flow is required"),
    (["--forecast-file", "f.csv", "--uniform-flow", "10"], "not allowed with argument"),
    (["--uniform-flow", "0"], "must be a finite number > 0"),
    (["--uniform-flow", "-5"], "must be a finite number > 0"),
    (["--uniform-flow", "nan"], "must be a finite number > 0"),
    (["--uniform-flow", "inf"], "must be a finite number > 0"),
])
def test_forecast_choice_usage_errors(run_cli, capsys, out_dir, argv, message):
    with pytest.raises(SystemExit) as exc:
        run_cli("--output-dir", out_dir, *argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_uniform_flow_writes_and_uses_forecast(run_cli, out_dir):
    analysis = run_cli("--output-dir", out_dir, "--uniform-flow", "4000")
    path = out_dir / "forecast_uniform4000.csv"
    assert analysis.forecast_file == path
    forecast = pd.read_csv(path)
    assert forecast["Id"].tolist() == [7, 9]
    assert (forecast["flow_m3s"] == 4000).all()


def test_uniform_flow_uses_catchlist_flag(run_cli, tmp_path):
    out = tmp_path / "out2"
    out.mkdir()
    (tmp_path / "cl.csv").write_text("Id, Slope\n3, 0.1\n")
    analysis = run_cli("--output-dir", out, "--uniform-flow", "1.5", "--catchlist", "cl.csv")
    assert pd.read_csv(analysis.forecast_file)["Id"].tolist() == [3]


def test_uniform_flow_without_catchlist_fails_cleanly(run_cli, tmp_path, caplog):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as exc:
        run_cli("--output-dir", tmp_path / "empty", "--uniform-flow", "10")
    assert exc.value.code == 1
    assert "uniform-flow forecast" in caplog.text


def test_forecast_file_bare_name_is_looked_up_in_dam_dir(run_cli, tmp_path, out_dir):
    dam_dir = tmp_path / "data" / "x"
    dam_dir.mkdir(parents=True)
    analysis = run_cli("--dam-dir", dam_dir, "--output-dir", out_dir,
                       "--forecast-file", "forecast.csv")
    assert analysis.kwargs["forecast_file"] == "forecast.csv"
    assert analysis.kwargs["dam_dir"] == str(dam_dir)


# --- write_uniform_forecast ------------------------------------------------------------------

def test_write_uniform_forecast(tmp_path):
    pd.DataFrame({"Id": [11, 12, 13], "Slope": [0.01] * 3}).to_csv(
        tmp_path / "catchlist.csv", index=False, sep=",")
    path = dam_inundation.write_uniform_forecast(tmp_path, 4000)
    assert path == tmp_path / "forecast_uniform4000.csv"
    forecast = pd.read_csv(path)
    assert list(forecast.columns) == ["Id", "flow_m3s"]
    assert forecast["Id"].tolist() == [11, 12, 13]
    assert (forecast["flow_m3s"] == 4000).all()


def test_write_uniform_forecast_explicit_catchlist(tmp_path):
    other = tmp_path / "elsewhere.csv"
    other.write_text("Id, Slope\n5, 0.1\n")  # TauDEM style: space after comma
    out = tmp_path / "out"
    out.mkdir()
    path = dam_inundation.write_uniform_forecast(out, 2.5, catchlist=other)
    assert pd.read_csv(path)["Id"].tolist() == [5]
    assert path.name == "forecast_uniform2.5.csv"


def test_uniform_forecast_names_do_not_collide(tmp_path):
    (tmp_path / "catchlist.csv").write_text("Id\n1\n")
    a = dam_inundation.write_uniform_forecast(tmp_path, 1234567)
    b = dam_inundation.write_uniform_forecast(tmp_path, 1234568)
    assert (a.name, b.name) == ("forecast_uniform1234567.csv", "forecast_uniform1234568.csv")
