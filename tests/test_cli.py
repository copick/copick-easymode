"""Offline CLI parsing and validation tests."""

from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from copick_easymode.cli.inference import easymode


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """A config file, no easymode environment variables, and a dispatcher that must not be reached."""
    for var in ("COPICK_EASYMODE_MODEL_DIR", "COPICK_EASYMODE_OFFLINE", "COPICK_CONFIG"):
        monkeypatch.delenv(var, raising=False)
    config_path = tmp_path / "project.json"
    config_path.write_text("{}", encoding="utf-8")
    dispatch = Mock(side_effect=AssertionError("inference must not start"))
    monkeypatch.setattr("copick_easymode.core.dispatch.dispatch_easymode_inference", dispatch)
    return config_path, dispatch


def _report(**updates):
    report = {"status": "complete", "processed": 1, "skipped": 0, "errors": []}
    report.update(updates)
    return report


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-m", "ribosome", "-t", "wbp"], "Invalid tomogram URI"),
        (["-m", "ribosome", "-t", "wbp@0"], "Voxel size must be positive"),
        (["-m", "ribosome", "-t", "wbp@-10"], "Voxel size must be positive"),
        (["-m", "ribosome", "-t", "wbp@nan"], "Voxel size must be positive"),
        (["-m", "ribosome", "-t", "wbp@inf"], "Voxel size must be positive"),
        (["-m", " , ", "-t", "wbp@10"], "No models specified"),
        (["-m", "ribosome", "-t", "wbp@10", "--tta", "0"], "TTA must be between 1 and 16"),
        (["-m", "ribosome", "-t", "wbp@10", "--tta", "17"], "TTA must be between 1 and 16"),
        (["-m", "ribosome", "-t", "wbp@10", "--batch-size", "0"], "Batch size must be at least 1"),
        (["-m", "ribosome", "-t", "wbp@10", "--threshold", "1.1"], "Threshold must be between 0.0 and 1.0"),
        (["-m", "ribosome", "-t", "wbp@10", "--threshold", "-0.1"], "Threshold must be between 0.0 and 1.0"),
        (["-m", "ribosome", "-t", "wbp@10", "--threshold", "nan"], "Threshold must be between 0.0 and 1.0"),
        (["-m", "ribosome", "-t", "wbp@10", "--cpu", "--gpus", "0"], "--cpu and --gpus exclude each other"),
        (["-m", "ribosome", "-t", "wbp@10", "--threads", "0"], "Invalid value for '--threads'"),
        (["-m", "ribosome", "-t", "wbp@10", "--max-workers", "0"], "Invalid value for '--max-workers'"),
    ],
)
def test_invalid_options_fail_before_inference_starts(cli_env, args, message):
    config_path, dispatch = cli_env

    result = CliRunner().invoke(easymode, ["-c", str(config_path), *args])

    assert result.exit_code != 0
    assert message in result.output
    dispatch.assert_not_called()


def test_command_parses_lists_and_delegates(cli_env, tmp_path):
    config_path, dispatch = cli_env
    dispatch.side_effect = None
    dispatch.return_value = _report()

    result = CliRunner().invoke(
        easymode,
        [
            "-c",
            str(config_path),
            "-m",
            " Ribosome,Membrane ",
            "-t",
            "wbp@10",
            "-r",
            "run-1, run-2",
            "--tta",
            "1",
            "--batch-size",
            "2",
            "--threshold",
            "0.25",
            "--xla",
            "--gpus",
            "0,1",
            "--max-workers",
            "2",
            "--threads",
            "8",
            "--model-dir",
            str(tmp_path / "models"),
            "--offline",
            "--report",
            str(tmp_path / "report.json"),
            "--no-add-objects",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    dispatch.assert_called_once_with(
        config_path=str(config_path),
        run_names=["run-1", "run-2"],
        tomo_type="wbp",
        voxel_size=10.0,
        models=["ribosome", "membrane"],
        user_id="copick",
        session_id="1",
        tta=1,
        batch_size=2,
        threshold=0.25,
        overwrite=True,
        xla=True,
        gpus="0,1",
        cpu=False,
        max_workers=2,
        threads=8,
        model_dir=str(tmp_path / "models"),
        offline=True,
        add_objects=False,
        report_path=str(tmp_path / "report.json"),
        debug=False,
        logger=dispatch.call_args.kwargs["logger"],
    )


def test_xla_is_off_unless_asked_for(cli_env):
    config_path, dispatch = cli_env
    dispatch.side_effect = None
    dispatch.return_value = _report()

    for flag, expected in (([], False), (["--no-xla"], False), (["--xla"], True)):
        result = CliRunner().invoke(easymode, ["-c", str(config_path), "-m", "ribosome", "-t", "wbp@10", *flag])
        assert result.exit_code == 0, result.output
        assert dispatch.call_args.kwargs["xla"] is expected


def test_a_failed_report_exits_nonzero_and_lists_the_errors(cli_env):
    config_path, dispatch = cli_env
    dispatch.side_effect = None
    dispatch.return_value = _report(status="failed", processed=0, errors=["Run 'r9' not found in the project"])

    result = CliRunner().invoke(easymode, ["-c", str(config_path), "-m", "ribosome", "-t", "wbp@10", "--cpu"])

    assert result.exit_code == 1
    assert "Errors encountered: 1" in result.output
    assert "Run 'r9' not found in the project" in result.output
