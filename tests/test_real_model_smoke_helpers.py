"""Deterministic tests for the opt-in real-model evidence helper."""

import argparse
import hashlib
import json

import numpy as np
import pytest
from conftest import make_project

from copick_easymode.core import easymode_env
from copick_easymode.core.easymode_env import ModelResolution, ResolvedModel
from copick_easymode.core.inference import Stages
from scripts import real_model_smoke
from scripts.real_model_smoke import discover_huggingface_revision, parse_tomogram, sha256_array, sha256_file


def test_hash_helpers_include_array_contract(tmp_path):
    path = tmp_path / "weights.h5"
    path.write_bytes(b"weights")
    array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

    assert sha256_file(path) == hashlib.sha256(b"weights").hexdigest()
    assert sha256_array(array) != sha256_array(array.astype(np.float64))
    assert sha256_array(array) != sha256_array(array.reshape(1, 2, 4))


def test_revision_discovery_reads_easymodes_local_directory_download_record(tmp_path):
    # easymode 1.x: hf_hub_download(local_dir=MODEL_DIRECTORY) records the commit beside the download.
    model = tmp_path / "models" / "ribosome_sv3-3d.h5"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    record = tmp_path / ".cache" / "huggingface" / "download" / "models" / "ribosome_sv3-3d.h5.metadata"
    record.parent.mkdir(parents=True)
    record.write_text("0123abc\netag\n1792000000.0\n", encoding="utf-8")

    assert discover_huggingface_revision(model) == "0123abc"


def test_revision_discovery_reads_huggingface_snapshot_layout(tmp_path):
    model = tmp_path / "ribosome.h5"
    model.write_bytes(b"weights")
    snapshot = tmp_path / "models--mgflast--easymode" / "snapshots" / "abc123" / model.name
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(model.read_bytes())

    assert discover_huggingface_revision(model) == "abc123"


def test_revision_discovery_reads_local_directory_cache_reference(tmp_path):
    model = tmp_path / "ribosome.h5"
    model.write_bytes(b"weights")
    reference = tmp_path / "models--mgflast--easymode" / "refs" / "main"
    reference.parent.mkdir(parents=True)
    reference.write_text("def456\n", encoding="utf-8")

    assert discover_huggingface_revision(model) == "def456"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("wbp@10", ("wbp", 10.0)), ("denoised@7.5", ("denoised", 7.5))],
)
def test_tomogram_parser(value, expected):
    assert parse_tomogram(value) == expected


@pytest.mark.parametrize("value", ["wbp", "@10", "wbp@0", "wbp@nan", "wbp@not-a-number"])
def test_tomogram_parser_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_tomogram(value)


@pytest.fixture
def smoke(tmp_path, isolated_env, monkeypatch):
    """A one-run project, a resolved model on disk and stages whose ``infer`` the test chooses."""
    config, _root = make_project(tmp_path, {"run-1": (6, 20, 24)})
    weights = tmp_path / "easymode-models" / "models" / "ribosome_t1.h5"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    record = (
        tmp_path / "easymode-models" / ".cache" / "huggingface" / "download" / "models" / f"{weights.name}.metadata"
    )
    record.parent.mkdir(parents=True)
    record.write_text("rev789\netag\n1.0\n", encoding="utf-8")
    model = ResolvedModel("ribosome", "ribosome", str(weights), "h5", tag="t1", timestamp="20260101000000", apix=10.0)
    resolution = ModelResolution(model_directory=str(weights.parent.parent), online=False, writable=True)
    resolution.models.append(model)
    monkeypatch.setattr(easymode_env, "resolve_models", lambda *args, **kwargs: resolution)
    stages = Stages(prepare=lambda volume: volume, infer=None, finish=lambda raw, prepared: raw)
    monkeypatch.setattr(real_model_smoke, "load_stages", lambda model, settings: stages)
    args = argparse.Namespace(
        config=config,
        run="run-1",
        tomogram=("wbp", 10.0),
        model="ribosome",
        user_id="smoke_user",
        session_id="1",
        threshold=0.5,
        model_dir=None,
        offline=True,
        overwrite=False,
        add_object=True,
        cpu_only=True,
    )
    return args, stages


def test_the_smoke_records_evidence_through_the_production_path(smoke):
    args, stages = smoke
    stages.infer = lambda volume: (volume > 0).astype(np.float32)

    record = real_model_smoke.run_smoke(args)

    assert record["stats"] == {"processed": 1, "skipped": 0, "errors": []}
    assert record["segmentation_entity"] == "ribosome:smoke-user/1@10.0"
    assert (record["output_zarr_format"], record["output_ome_zarr_version"]) == (3, "0.5")
    assert record["foreground_voxels"] > 0 and record["input_shape"] == (6, 20, 24)
    assert (record["model_kind"], record["model_tag"], record["huggingface_revision"]) == ("h5", "t1", "rev789")
    assert record["model_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert {"read", "prepare", "infer", "finish", "write"} <= set(record["stage_seconds"])
    assert "/" not in record["model_file"] and str(args.config.parent) not in json.dumps(record, default=str)
    assert [o["name"] for o in json.loads(args.config.read_text())["pickable_objects"]] == ["ribosome"]


def test_the_smoke_refuses_an_all_background_result_and_writes_nothing(smoke):
    import copick

    args, stages = smoke
    stages.infer = lambda volume: np.zeros(volume.shape, dtype=np.float32)

    with pytest.raises(RuntimeError, match="would produce an empty segmentation"):
        real_model_smoke.run_smoke(args)
    assert not copick.from_file(str(args.config)).get_run("run-1").get_segmentations(name="ribosome")
