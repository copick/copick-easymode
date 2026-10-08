"""The command end to end: the report, the exit status, and one worker process per device."""

import json
import os
import shutil

import pytest

pytest.importorskip("keras")
pytest.importorskip("easymode.core.distribution")

from click.testing import CliRunner  # noqa: E402
from conftest import make_model_dir, make_project  # noqa: E402

from copick_easymode.cli.inference import easymode  # noqa: E402
from copick_easymode.core.dispatch import dispatch_easymode_inference  # noqa: E402

RUNS = {"r1": (6, 36, 40), "r2": (5, 40, 36), "r3": (6, 32, 44)}


def _invoke(config, model_dir, report, *extra):
    args = ["-c", str(config), "-t", "wbp@10.0", "--tta", "1", "--user-id", "test", "--session-id", "7"]
    args += ["--model-dir", str(model_dir), "--offline", "--report", str(report), *extra]
    return CliRunner().invoke(easymode, args, catch_exceptions=False)


def _segmented(config, run_name):
    import copick

    run = copick.from_file(str(config)).get_run(run_name)
    return bool(run.get_segmentations(name="tiny", user_id="test", session_id="7", voxel_size=10.0))


def test_a_cpu_run_reports_what_it_did_and_exits_zero(tmp_path, isolated_env):
    config, _ = make_project(tmp_path, RUNS)
    model_dir = make_model_dir(tmp_path)
    report_path = tmp_path / "out" / "easymode.json"

    result = _invoke(config, model_dir, report_path, "-m", "tiny", "--cpu", "--threads", "3")
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert report["tool"] == "copick-easymode" and report["status"] == "complete"
    assert (report["processed"], report["skipped"], report["errors"]) == (3, 0, [])
    assert report["tomogram"] == "wbp@10.0" and report["runs"] == ["r1", "r2", "r3"]
    assert (report["user_id"], report["session_id"], report["tta"], report["threshold"]) == ("test", "7", 1, 0.5)
    weights = model_dir / "models" / "tiny_t1.scnm"
    assert report["models"] == [
        {
            "feature": "tiny",
            "object": "tiny",
            "weights": str(weights),
            "kind": "scnm",
            "tag": "t1",
            "timestamp": "20260101000000",
            "bytes": weights.stat().st_size,
            "apix": None,
        },
    ]
    assert report["missing"] == {} and report["model_directory"] == str(model_dir)
    assert report["online"] is False and report["writable"] is True
    assert report["cpu"] is True and report["devices"] == [] and report["threads_per_worker"] == 3
    [worker] = report["workers"]
    assert (worker["gpu"], worker["runs"], worker["exitcode"], worker["visible_devices"]) == (
        None,
        ["r1", "r2", "r3"],
        0,
        "-1",
    )
    assert [item["status"] for item in worker["items"]] == ["processed"] * 3
    assert report["started_utc"] and report["finished_utc"]
    # Only this process wrote the config: the object was added before any worker started.
    assert [o["name"] for o in json.loads(config.read_text())["pickable_objects"]] == ["tiny"]
    assert all(_segmented(config, run) for run in RUNS)
    assert not list((model_dir).glob("*.lock")) and not (model_dir / ".cache").exists()  # offline: nothing written

    again = _invoke(config, model_dir, report_path, "-m", "tiny", "--cpu")
    assert again.exit_code == 0
    assert json.loads(report_path.read_text())["skipped"] == 3


def test_a_missing_model_fails_before_any_worker(tmp_path, isolated_env):
    config, _ = make_project(tmp_path, RUNS)
    model_dir = make_model_dir(tmp_path)
    report_path = tmp_path / "easymode.json"

    result = _invoke(config, model_dir, report_path, "-m", "tiny,nonexistent", "--cpu")
    assert result.exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["workers"] == []
    assert list(report["missing"]) == ["nonexistent"] and report["missing"]["nonexistent"]
    assert [m["feature"] for m in report["models"]] == ["tiny"]
    assert "nonexistent" in report["errors"][0]
    assert not any(_segmented(config, run) for run in RUNS)


def test_a_failed_run_or_an_unknown_run_fails_the_command(tmp_path, isolated_env):
    config, _ = make_project(tmp_path, RUNS)
    model_dir = make_model_dir(tmp_path)
    report_path = tmp_path / "easymode.json"
    # r2's tomogram exists in the project but its array is gone: reading it fails.
    shutil.rmtree(next((tmp_path / "overlay" / "ExperimentRuns" / "r2").rglob("wbp.zarr")) / "0")

    result = _invoke(config, model_dir, report_path, "-m", "tiny", "--cpu", "-r", "r1,r2,r3,r9")
    assert result.exit_code == 1
    assert "Errors encountered: 2" in result.output
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["processed"] == 2
    assert report["errors"][0] == "Run 'r9' not found in the project"
    assert report["errors"][1].startswith("Error processing tiny in r2")
    assert report["workers"][0]["exitcode"] == 1 and report["workers"][0]["errors"] == report["errors"][1:]
    assert _segmented(config, "r1") and _segmented(config, "r3") and not _segmented(config, "r2")


def test_a_refused_device_is_reported(tmp_path, isolated_env, monkeypatch):
    config, _ = make_project(tmp_path, RUNS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    report_path = tmp_path / "easymode.json"
    result = _invoke(config, make_model_dir(tmp_path), report_path, "-m", "tiny", "--gpus", "1")
    assert result.exit_code == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and "not in this job's allocation" in report["errors"][0]


def test_two_workers_each_see_their_own_device(tmp_path, isolated_env, monkeypatch):
    config, _ = make_project(tmp_path, RUNS)
    model_dir = make_model_dir(tmp_path)
    report_path = tmp_path / "easymode.json"
    # Two "devices" that are not GPUs: each spawned worker gets exactly one of them, and TensorFlow,
    # finding no such GPU, computes on the CPU.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "fake-a,fake-b")

    report = dispatch_easymode_inference(
        str(config),
        [],
        "wbp",
        10.0,
        ["tiny"],
        "test",
        "7",
        tta=1,
        threads=4,
        model_dir=str(model_dir),
        offline=True,
        add_objects=True,
        report_path=str(report_path),
    )
    assert report["status"] == "complete", report["errors"]
    assert report["devices"] == ["fake-a", "fake-b"] and report["threads_per_worker"] == 2
    workers = report["workers"]
    assert [(w["gpu"], w["visible_devices"], w["threads"], w["exitcode"]) for w in workers] == [
        ("fake-a", "fake-a", 2, 0),
        ("fake-b", "fake-b", 2, 0),
    ]
    assert [w["runs"] for w in workers] == [["r1", "r3"], ["r2"]]
    assert [w["processed"] for w in workers] == [2, 1] and report["processed"] == 3
    assert json.loads(report_path.read_text()) == report
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "fake-a,fake-b"  # the parent's own environment is untouched
    assert all(_segmented(config, run) for run in RUNS)


def test_max_workers_caps_the_devices_and_one_worker_runs_in_process(tmp_path, isolated_env, monkeypatch):
    config, _ = make_project(tmp_path, RUNS)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "fake-a,fake-b")
    report_path = tmp_path / "easymode.json"
    result = _invoke(config, make_model_dir(tmp_path), report_path, "-m", "tiny", "--max-workers", "1")
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert report["devices"] == ["fake-a"]
    [worker] = report["workers"]
    assert (worker["gpu"], worker["visible_devices"], worker["runs"]) == ("fake-a", "fake-a", ["r1", "r2", "r3"])
