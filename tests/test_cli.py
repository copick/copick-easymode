"""Offline CLI parsing and validation tests."""

from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from copick_easymode.cli.inference import easymode


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-m", "ribosome", "-t", "wbp"], "Invalid tomogram URI"),
        (["-m", "ribosome", "-t", "wbp@0"], "Voxel size must be positive"),
        (["-m", "ribosome", "-t", "wbp@10", "--tta", "0"], "TTA must be between 1 and 16"),
        (["-m", "ribosome", "-t", "wbp@10", "--batch-size", "0"], "Batch size must be at least 1"),
        (["-m", "ribosome", "-t", "wbp@10", "--threshold", "1.1"], "Threshold must be between 0.0 and 1.0"),
    ],
)
def test_invalid_options_fail_before_loading_config(monkeypatch, tmp_path, args, message):
    config_path = tmp_path / "project.json"
    config_path.write_text("{}", encoding="utf-8")
    from_file = Mock(side_effect=AssertionError("config must not load"))
    monkeypatch.setattr("copick.from_file", from_file)

    result = CliRunner().invoke(easymode, ["-c", str(config_path), *args])

    assert result.exit_code != 0
    assert message in result.output
    from_file.assert_not_called()


def test_command_parses_lists_and_delegates(monkeypatch, tmp_path):
    config_path = tmp_path / "project.json"
    config_path.write_text("{}", encoding="utf-8")
    root = object()
    run = Mock(return_value={"processed": 1, "skipped": 0, "errors": []})
    monkeypatch.setattr("copick.from_file", lambda path: root)
    monkeypatch.setattr("copick_easymode.core.inference.run_easymode_inference", run)

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
            "--no-add-objects",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        root=root,
        run_names=["run-1", "run-2"],
        tomo_type="wbp",
        voxel_size=10.0,
        models=["ribosome", "membrane"],
        user_id="copick",
        session_id="1",
        tta=1,
        batch_size=2,
        threshold=0.25,
        gpus=None,
        add_objects=False,
        overwrite=True,
        config_path=None,
        logger=run.call_args.kwargs["logger"],
    )
