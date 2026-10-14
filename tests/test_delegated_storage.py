"""Copick owns every OME-Zarr read and write: legacy and current inputs, OME-Zarr 0.5 / Zarr v3 output.

The per-run loop (:func:`copick_easymode.core.inference.segment_runs`) reads tomograms with ``tomo.numpy()``
and writes with ``new_segmentation(...).from_numpy()``; these tests give it explicit stages, so no easymode or
model is involved, and check what copick reads and leaves on disk.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import zarr
from copick.impl.filesystem import CopickConfigFSSpec, CopickRootFSSpec
from copick.models import PickableObject
from copick.util.ome import get_level_path
from ome_zarr_models.v05.image import Image
from zarr.codecs import GzipCodec

from copick_easymode.core.inference import InferenceSettings, Stages, new_stats, segment_runs


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


def _probabilities(volume: np.ndarray) -> np.ndarray:
    values = np.array([0.49, 0.5, 0.51], dtype=np.float32)
    return np.resize(values, volume.shape)


def _stages(seen=None, infer=_probabilities) -> Stages:
    def prepare(volume):
        if seen is not None:
            seen.append(volume.copy())
        return volume

    return Stages(prepare=prepare, infer=infer, finish=lambda raw, prepared: raw)


def _segment(fixture: _Fixture, stages: Stages, overwrite: bool = False) -> dict:
    settings = InferenceSettings(
        tomo_type="wbp",
        voxel_size=10.0,
        user_id="tester",
        session_id="1",
        tta=1,
        threshold=0.5,
        overwrite=overwrite,
    )
    stats = segment_runs(
        fixture.root.runs,
        feature="ribosome",
        object_name="ribosome",
        stages=stages,
        settings=settings,
        stats=new_stats(),
    )
    return {key: stats[key] for key in ("processed", "skipped", "errors")}


def _segmentations(fixture: _Fixture):
    return fixture.run.get_segmentations(
        name="ribosome",
        user_id="tester",
        session_id="1",
        voxel_size=10.0,
        is_multilabel=False,
    )


def _assert_v3_uint8(segmentation, expected: np.ndarray) -> zarr.Group:
    np.testing.assert_array_equal(segmentation.numpy(), expected)
    group = zarr.open_group(segmentation.zarr(), mode="r")
    assert group.metadata.zarr_format == 3
    assert Image.from_zarr(group).ome_zarr_version == "0.5"
    assert group[get_level_path(group, 0)].dtype == np.dtype(np.uint8)
    return group


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


def _v3_segmentation(fixture: _Fixture, values: np.ndarray):
    segmentation = fixture.run.new_segmentation(
        name="ribosome",
        voxel_size=10.0,
        user_id="tester",
        session_id="1",
        is_multilabel=False,
    )
    segmentation.from_numpy(values)
    return segmentation


@pytest.mark.parametrize(("zarr_format", "level_path"), [(2, "0"), (2, "s0"), (3, "0"), (3, "s0")])
def test_equivalent_v2_v3_inputs_delegate_to_copick_and_write_v3(tmp_path, zarr_format, level_path):
    fixture = _project(tmp_path, zarr_format, level_path)
    source_before = _snapshot(fixture.tomogram.overlay_path)
    seen = []

    assert _segment(fixture, _stages(seen)) == {"processed": 1, "skipped": 0, "errors": []}

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], fixture.values)
    assert _snapshot(fixture.tomogram.overlay_path) == source_before
    [segmentation] = _segmentations(fixture)
    _assert_v3_uint8(segmentation, (_probabilities(fixture.values) >= 0.5).astype(np.uint8))


def test_without_overwrite_a_legacy_segmentation_is_skipped_before_read_or_inference(tmp_path, monkeypatch):
    fixture = _project(tmp_path, 2, "s0")
    existing = _legacy_segmentation(fixture, np.ones(fixture.values.shape, dtype=np.uint8))
    before = _snapshot(existing.path)
    monkeypatch.setattr(fixture.tomogram, "numpy", lambda *a, **k: pytest.fail("the tomogram must not be read"))

    stages = Stages(*(lambda *a: pytest.fail("no stage may run") for _ in range(3)))
    assert _segment(fixture, stages) == {"processed": 0, "skipped": 1, "errors": []}
    assert _snapshot(existing.path) == before


def test_overwrite_reuses_the_entity_and_replaces_a_legacy_store_with_v3(tmp_path):
    fixture = _project(tmp_path, 3, "s0")
    existing_path = _legacy_segmentation(fixture, np.ones(fixture.values.shape, dtype=np.uint8)).path

    assert _segment(fixture, _stages(), overwrite=True) == {"processed": 1, "skipped": 0, "errors": []}

    [segmentation] = _segmentations(fixture)
    assert segmentation.path == existing_path
    group = _assert_v3_uint8(segmentation, (_probabilities(fixture.values) >= 0.5).astype(np.uint8))
    assert "legacy" not in group
    output_keys = _snapshot(segmentation.path)
    assert ".zgroup" not in output_keys
    assert ".zattrs" not in output_keys
    assert not any(key == "legacy" or key.startswith("legacy/") for key in output_keys)


@pytest.mark.parametrize("make_existing", [_legacy_segmentation, _v3_segmentation], ids=["legacy-v2", "v3"])
def test_a_failed_inference_preserves_the_existing_store(tmp_path, make_existing):
    fixture = _project(tmp_path, 3, "s0")
    expected = np.ones(fixture.values.shape, dtype=np.uint8)
    existing = make_existing(fixture, expected)
    before = _snapshot(existing.path)

    def infer(prepared):
        raise RuntimeError("inference failed")

    outcome = _segment(fixture, _stages(infer=infer), overwrite=True)

    assert (outcome["processed"], outcome["skipped"]) == (0, 0)
    assert outcome["errors"] == ["Error processing ribosome in run: inference failed"]
    assert _snapshot(existing.path) == before
    np.testing.assert_array_equal(existing.numpy(), expected)
