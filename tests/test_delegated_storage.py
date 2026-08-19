"""End-to-end tests for copick-owned OME-Zarr reads and writes."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zarr
from copick.impl.filesystem import CopickConfigFSSpec, CopickRootFSSpec
from copick.models import PickableObject
from copick.util.ome import get_level_path
from ome_zarr_models.v05.image import Image
from zarr.codecs import GzipCodec

from copick_easymode.core.easymode_adapter import EasymodeRuntime
from copick_easymode.core.inference import run_easymode_inference


class _Backend:
    def __init__(self):
        self.clear_count = 0

    def clear_session(self):
        self.clear_count += 1


class _TensorFlow:
    def __init__(self):
        self.keras = SimpleNamespace(backend=_Backend())
        self.config = SimpleNamespace(
            list_physical_devices=lambda kind: [],
            experimental=SimpleNamespace(set_memory_growth=lambda device, enabled: None),
        )


@dataclass
class _Fixture:
    root: CopickRootFSSpec
    run: object
    tomogram: object
    values: np.ndarray


def _axes():
    return [{"name": axis, "type": "space", "unit": "angstrom"} for axis in ("z", "y", "x")]


def _multiscales(path: str, version: str):
    return [
        {
            "version": version,
            "axes": _axes(),
            "datasets": [
                {
                    "path": path,
                    "coordinateTransformations": [{"type": "scale", "scale": [10.0, 10.0, 10.0]}],
                },
            ],
        },
    ]


def _snapshot(path: str) -> dict[str, bytes]:
    root = Path(path)
    return {item.relative_to(root).as_posix(): item.read_bytes() for item in root.rglob("*") if item.is_file()}


def _project(tmp_path: Path, zarr_format: int, level_path: str) -> _Fixture:
    obj = PickableObject(name="ribosome", is_particle=False, label=1, color=[0, 255, 0, 255])
    root = CopickRootFSSpec(
        CopickConfigFSSpec(overlay_root=f"local://{tmp_path}", pickable_objects=[obj]),
    )
    run = root.new_run("run")
    voxel_spacing = run.new_voxel_spacing(10.0)
    tomogram = voxel_spacing.new_tomogram("wbp")
    values = ((np.indices((4, 5, 6)) * np.array([11, 5, 2])[:, None, None, None]).sum(0) % 17).astype(
        np.float32,
    )

    if zarr_format == 3 and level_path == "0":
        # Exercise the canonical OME-Zarr 0.5 writer contract owned by copick.
        tomogram.from_numpy(values, levels=1)
        return _Fixture(root, run, tomogram, values)

    group = zarr.group(store=tomogram.zarr(), overwrite=True, zarr_format=zarr_format)
    options = {"chunks": (2, 3, 4)}
    if zarr_format == 2:
        options["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
    else:
        options["compressors"] = (GzipCodec(level=2),)
    group.create_array(level_path, data=values, **options)
    multiscales = _multiscales(level_path, "0.4" if zarr_format == 2 else "0.5")
    if zarr_format == 2:
        group.attrs["multiscales"] = multiscales
    else:
        group.attrs["ome"] = {"version": "0.5", "multiscales": multiscales}
    return _Fixture(root, run, tomogram, values)


def _runtime(*, model_path="model.keras"):
    tensorflow = _TensorFlow()
    runtime = EasymodeRuntime(
        tensorflow=tensorflow,
        get_model=lambda name: (model_path, {"apix": 10.0}),
        load_model=lambda path: {"path": path},
    )
    return runtime, tensorflow


def _probabilities(volume: np.ndarray) -> np.ndarray:
    values = np.array([0.49, 0.5, 0.51], dtype=np.float32)
    return np.resize(values, volume.shape)


def _add_v3_run(fixture: _Fixture, name: str):
    run = fixture.root.new_run(name)
    voxel_spacing = run.new_voxel_spacing(10.0)
    tomogram = voxel_spacing.new_tomogram("wbp")
    tomogram.from_numpy(fixture.values, levels=1)
    return run, tomogram


@pytest.mark.parametrize(("zarr_format", "level_path"), [(2, "0"), (2, "s0"), (3, "0"), (3, "s0")])
def test_equivalent_v2_v3_inputs_delegate_to_copick_and_write_v3(tmp_path, zarr_format, level_path):
    fixture = _project(tmp_path, zarr_format, level_path)
    source_before = _snapshot(fixture.tomogram.overlay_path)
    seen = []
    runtime, tensorflow = _runtime()

    def segmenter(*, volume, input_apix, **kwargs):
        seen.append((volume.copy(), input_apix))
        return _probabilities(volume)

    stats = run_easymode_inference(
        fixture.root,
        run_names=[],
        tomo_type="wbp",
        voxel_size=10.0,
        models=["ribosome"],
        user_id="tester",
        session_id="1",
        tta=1,
        threshold=0.5,
        add_objects=False,
        runtime=runtime,
        segmenter=segmenter,
    )

    assert stats == {"processed": 1, "skipped": 0, "errors": []}
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], fixture.values)
    assert seen[0][1] == 10.0
    assert _snapshot(fixture.tomogram.overlay_path) == source_before
    assert tensorflow.keras.backend.clear_count == 1

    segmentation = fixture.run.get_segmentations(
        name="ribosome",
        user_id="tester",
        session_id="1",
        voxel_size=10.0,
        is_multilabel=False,
    )[0]
    expected = (_probabilities(fixture.values) >= 0.5).astype(np.uint8)
    np.testing.assert_array_equal(segmentation.numpy(), expected)
    group = zarr.open_group(segmentation.zarr(), mode="r")
    assert group.metadata.zarr_format == 3
    assert Image.from_zarr(group).ome_zarr_version == "0.5"
    assert group[get_level_path(group, 0)].dtype == np.dtype(np.uint8)


def test_default_segmenter_is_resolved_at_call_time(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 3, "s0")
    runtime, _ = _runtime()
    segmenter = Mock(side_effect=lambda **kwargs: _probabilities(kwargs["volume"]))
    monkeypatch.setattr("copick_easymode.core.inference.segment_tomogram_from_array", segmenter)

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome"],
        "tester",
        "1",
        add_objects=False,
        runtime=runtime,
    )

    assert stats == {"processed": 1, "skipped": 0, "errors": []}
    segmenter.assert_called_once()


def _legacy_segmentation(fixture: _Fixture, values: np.ndarray):
    segmentation = fixture.run.new_segmentation(
        name="ribosome",
        voxel_size=10.0,
        user_id="tester",
        session_id="1",
        is_multilabel=False,
    )
    group = zarr.group(store=segmentation.zarr(), overwrite=True, zarr_format=2)
    group.create_array(
        "legacy",
        data=values,
        chunks=(2, 3, 4),
        chunk_key_encoding={"name": "v2", "separator": "/"},
    )
    group.attrs["multiscales"] = _multiscales("legacy", "0.4")
    return segmentation


def test_no_overwrite_skips_before_read_or_inference(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 2, "s0")
    existing = _legacy_segmentation(fixture, np.ones(fixture.values.shape, dtype=np.uint8))
    before = _snapshot(existing.path)
    segmenter = Mock(side_effect=AssertionError("inference must not run"))
    monkeypatch.setattr(fixture.tomogram, "numpy", Mock(side_effect=AssertionError("tomogram must not be read")))
    runtime, _ = _runtime()

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome"],
        "tester",
        "1",
        overwrite=False,
        add_objects=False,
        runtime=runtime,
        segmenter=segmenter,
    )

    assert stats == {"processed": 0, "skipped": 1, "errors": []}
    segmenter.assert_not_called()
    assert _snapshot(existing.path) == before


def test_overwrite_reuses_entity_and_replaces_legacy_store_with_v3(tmp_path):
    fixture = _project(tmp_path, 3, "s0")
    existing = _legacy_segmentation(fixture, np.ones(fixture.values.shape, dtype=np.uint8))
    existing_path = existing.path
    runtime, _ = _runtime()

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome"],
        "tester",
        "1",
        threshold=0.5,
        overwrite=True,
        add_objects=False,
        runtime=runtime,
        segmenter=lambda **kwargs: _probabilities(kwargs["volume"]),
    )

    assert stats == {"processed": 1, "skipped": 0, "errors": []}
    segmentations = fixture.run.get_segmentations(
        name="ribosome",
        user_id="tester",
        session_id="1",
        voxel_size=10.0,
        is_multilabel=False,
    )
    assert len(segmentations) == 1
    assert segmentations[0].path == existing_path
    expected = (_probabilities(fixture.values) >= 0.5).astype(np.uint8)
    np.testing.assert_array_equal(segmentations[0].numpy(), expected)
    group = zarr.open_group(segmentations[0].zarr(), mode="r")
    assert group.metadata.zarr_format == 3
    assert Image.from_zarr(group).ome_zarr_version == "0.5"
    assert "legacy" not in group
    output_keys = _snapshot(segmentations[0].path)
    assert ".zgroup" not in output_keys
    assert ".zattrs" not in output_keys
    assert not any(key == "legacy" or key.startswith("legacy/") for key in output_keys)


def test_failed_inference_preserves_existing_segmentation(tmp_path):
    fixture = _project(tmp_path, 3, "s0")
    expected = np.ones(fixture.values.shape, dtype=np.uint8)
    existing = _legacy_segmentation(fixture, expected)
    before = _snapshot(existing.path)
    runtime, tensorflow = _runtime()

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome"],
        "tester",
        "1",
        overwrite=True,
        add_objects=False,
        runtime=runtime,
        segmenter=Mock(side_effect=RuntimeError("inference failed")),
    )

    assert stats["processed"] == 0
    assert stats["skipped"] == 0
    assert len(stats["errors"]) == 1
    assert "inference failed" in stats["errors"][0]
    assert _snapshot(existing.path) == before
    np.testing.assert_array_equal(existing.numpy(), expected)
    assert tensorflow.keras.backend.clear_count == 1


def test_failed_inference_preserves_existing_v3_segmentation(tmp_path):
    fixture = _project(tmp_path, 3, "s0")
    expected = np.ones(fixture.values.shape, dtype=np.uint8)
    existing = fixture.run.new_segmentation(
        name="ribosome",
        voxel_size=10.0,
        user_id="tester",
        session_id="1",
        is_multilabel=False,
    )
    existing.from_numpy(expected)
    before = _snapshot(existing.path)
    runtime, tensorflow = _runtime()

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome"],
        "tester",
        "1",
        overwrite=True,
        add_objects=False,
        runtime=runtime,
        segmenter=Mock(side_effect=RuntimeError("inference failed")),
    )

    assert stats["processed"] == 0
    assert stats["skipped"] == 0
    assert len(stats["errors"]) == 1
    assert "inference failed" in stats["errors"][0]
    assert _snapshot(existing.path) == before
    np.testing.assert_array_equal(existing.numpy(), expected)
    assert tensorflow.keras.backend.clear_count == 1


def test_missing_model_records_error_cleans_runtime_and_continues(tmp_path):
    fixture = _project(tmp_path, 3, "s0")
    fixture.root.new_run("missing-input")
    tensorflow = _TensorFlow()
    get_model = Mock(side_effect=[("model.keras", {"apix": 10.0}), (None, None)])
    runtime = EasymodeRuntime(tensorflow, get_model, lambda path: object())

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome", "missing"],
        "tester",
        "1",
        add_objects=False,
        runtime=runtime,
        segmenter=lambda **kwargs: np.ones(kwargs["volume"].shape, dtype=np.float32),
    )

    assert stats["processed"] == 1
    assert stats["skipped"] == 1
    assert len(stats["errors"]) == 1
    assert "missing" in stats["errors"][0]
    assert tensorflow.keras.backend.clear_count == 2


def test_missing_tomogram_type_skips_without_inference(tmp_path):
    fixture = _project(tmp_path, 3, "s0")
    runtime, _ = _runtime()
    segmenter = Mock(side_effect=AssertionError("inference must not run"))

    stats = run_easymode_inference(
        fixture.root,
        [],
        "sirt",
        10.0,
        ["ribosome"],
        "tester",
        "1",
        add_objects=False,
        runtime=runtime,
        segmenter=segmenter,
    )

    assert stats == {"processed": 0, "skipped": 1, "errors": []}
    segmenter.assert_not_called()


def test_processing_failure_does_not_prevent_later_run(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 3, "s0")
    successful_run, _ = _add_v3_run(fixture, "run-2")
    monkeypatch.setattr(fixture.tomogram, "numpy", Mock(side_effect=RuntimeError("read failed")))
    runtime, tensorflow = _runtime()

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome"],
        "tester",
        "1",
        add_objects=False,
        runtime=runtime,
        segmenter=lambda **kwargs: np.ones(kwargs["volume"].shape, dtype=np.float32),
    )

    assert stats["processed"] == 1
    assert stats["skipped"] == 0
    assert len(stats["errors"]) == 1
    assert "read failed" in stats["errors"][0]
    assert len(successful_run.get_segmentations(name="ribosome")) == 1
    assert tensorflow.keras.backend.clear_count == 1


def test_sanitized_identifiers_skip_existing_output_before_inference(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 3, "s0")
    existing = fixture.run.new_segmentation(
        name="ribosome",
        voxel_size=10.0,
        user_id="my-user",
        session_id="session-one",
        is_multilabel=False,
    )
    existing.from_numpy(np.ones(fixture.values.shape, dtype=np.uint8))
    monkeypatch.setattr(fixture.tomogram, "numpy", Mock(side_effect=AssertionError("tomogram must not be read")))
    runtime, _ = _runtime()
    segmenter = Mock(side_effect=AssertionError("inference must not run"))

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["ribosome"],
        "my_user",
        "session_one",
        add_objects=False,
        runtime=runtime,
        segmenter=segmenter,
    )

    assert stats == {"processed": 0, "skipped": 1, "errors": []}
    segmenter.assert_not_called()


def test_model_name_is_raw_for_easymode_and_sanitized_for_copick(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 3, "s0")
    get_model = Mock(return_value=("model.keras", {"apix": 10.0}))
    tensorflow = _TensorFlow()
    runtime = EasymodeRuntime(tensorflow, get_model, lambda path: object())
    save_config = Mock()
    monkeypatch.setattr(fixture.root, "save_config", save_config)

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["nuclear_envelope"],
        "tester",
        "1",
        config_path="config.json",
        runtime=runtime,
        segmenter=lambda **kwargs: np.ones(kwargs["volume"].shape, dtype=np.float32),
    )

    assert stats == {"processed": 1, "skipped": 0, "errors": []}
    get_model.assert_called_once_with("nuclear_envelope")
    assert fixture.root.get_object("nuclear-envelope") is not None
    segmentations = fixture.run.get_segmentations(name="nuclear-envelope", user_id="tester", session_id="1")
    assert len(segmentations) == 1
    save_config.assert_called_once_with("config.json")
    assert tensorflow.keras.backend.clear_count == 1


def test_object_creation_failure_records_error_and_continues(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 3, "s0")
    original_new_object = fixture.root.new_object

    def create_object(**kwargs):
        if kwargs["name"] == "actin":
            raise RuntimeError("config is read-only")
        return original_new_object(**kwargs)

    new_object = Mock(side_effect=create_object)
    save_config = Mock()
    monkeypatch.setattr(fixture.root, "new_object", new_object)
    monkeypatch.setattr(fixture.root, "save_config", save_config)
    runtime, tensorflow = _runtime()

    stats = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["membrane", "actin", "ribosome"],
        "tester",
        "1",
        config_path="config.json",
        runtime=runtime,
        segmenter=lambda **kwargs: np.ones(kwargs["volume"].shape, dtype=np.float32),
    )

    assert stats["processed"] == 2
    assert stats["skipped"] == 0
    assert len(stats["errors"]) == 1
    assert "config is read-only" in stats["errors"][0]
    assert [call.kwargs["name"] for call in new_object.call_args_list] == ["membrane", "actin"]
    assert fixture.root.get_object("membrane") is not None
    save_config.assert_called_once_with("config.json")
    assert tensorflow.keras.backend.clear_count == 3


def test_new_object_is_saved_once_and_existing_object_does_not_rewrite_config(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 3, "s0")
    save_config = Mock()
    monkeypatch.setattr(fixture.root, "save_config", save_config)
    runtime, _ = _runtime()

    def segmenter(**kwargs):
        return np.ones(kwargs["volume"].shape, dtype=np.float32)

    first = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["membrane"],
        "tester",
        "1",
        config_path="config.json",
        runtime=runtime,
        segmenter=segmenter,
    )

    assert first == {"processed": 1, "skipped": 0, "errors": []}
    assert fixture.root.get_object("membrane") is not None
    save_config.assert_called_once_with("config.json")

    save_config.reset_mock()
    second = run_easymode_inference(
        fixture.root,
        [],
        "wbp",
        10.0,
        ["membrane"],
        "tester",
        "1",
        overwrite=True,
        config_path="config.json",
        runtime=runtime,
        segmenter=segmenter,
    )

    assert second == {"processed": 1, "skipped": 0, "errors": []}
    save_config.assert_not_called()
