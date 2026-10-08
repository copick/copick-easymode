"""The overlapped loop: the same result as the serial one, run by run, and a failing run stops nothing."""

import numpy as np
import pytest

keras = pytest.importorskip("keras")
pytest.importorskip("easymode.segmentation.inference")

from conftest import make_project, make_scnm  # noqa: E402

from copick_easymode.core.inference import (  # noqa: E402
    InferenceSettings,
    Stages,
    finish_tomogram,
    infer_tomogram,
    new_stats,
    prepare_tomogram,
    segment_runs,
    segment_tomogram_from_array,
)
from copick_easymode.core.scnm import (  # noqa: E402
    _TTA_FLIPS,
    _TTA_ROTATIONS,
    _infer_slab,
    _infer_slices,
    _preprocess,
    _unpad_xy,
    finish_scnm,
    infer_scnm,
    load_scnm,
    prepare_scnm,
    segment_volume_scnm,
)

RUNS = {"r1": (8, 40, 48), "r2": (6, 44, 36), "r3": (8, 36, 40), "r4": (7, 48, 44)}


def _reference_h5(model, volume, input_apix, model_apix=10.0, tta=1, batch_size=2):
    """copick-easymode 0.3.0's ``segment_tomogram_from_array``, verbatim."""
    from easymode.segmentation.inference import _pad_volume, _segment_tomogram_instance
    from scipy.ndimage import zoom

    volume = volume.astype(np.float32)
    oj, ok, ol = volume.shape
    scale = float(input_apix) / float(model_apix)
    if abs(scale - 1.0) > 0.05:
        volume = zoom(volume, scale, order=1)
    _j, _k, _l = volume.shape
    _k_margin = min(int(0.2 * _k), 64)
    _l_margin = min(int(0.2 * _l), 64)
    volume -= np.mean(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin])
    volume /= np.std(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin]) + 1e-7
    volume, padding = _pad_volume(volume)
    segmented_volume = np.zeros_like(volume)
    tile_size = tuple(min(256, s) for s in segmented_volume.shape)
    overlap = [0 if tile_size[i] == segmented_volume.shape[i] else 48 for i in range(3)]
    k_xy = [0, 2, 2, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3]
    k_fx = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    k_yz = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    for j in range(tta):
        tta_vol = volume.copy()
        tta_vol = np.rot90(tta_vol, k=k_xy[j], axes=(1, 2))
        tta_vol = tta_vol if not k_fx[j] else np.flip(tta_vol, axis=1)
        tta_vol = np.rot90(tta_vol, k=2 * k_yz[j], axes=(0, 1))
        segmented_tta_vol = _segment_tomogram_instance(tta_vol, model, batch_size, tile_size, overlap)
        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-2 * k_yz[j], axes=(0, 1))
        segmented_tta_vol = segmented_tta_vol if not k_fx[j] else np.flip(segmented_tta_vol, axis=1)
        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-k_xy[j], axes=(1, 2))
        segmented_volume += segmented_tta_vol
    segmented_volume /= tta
    (j0, j1), (k0, k1), (l0, l1) = padding
    segmented_volume = segmented_volume[
        j0 : segmented_volume.shape[0] - j1,
        k0 : segmented_volume.shape[1] - k1,
        l0 : segmented_volume.shape[2] - l1,
    ]
    if abs(scale - 1.0) > 0.05:
        sj, sk, sl = segmented_volume.shape
        segmented_volume = zoom(segmented_volume, (oj / sj, ok / sk, ol / sl), order=1)
    return segmented_volume.astype(np.float32)


def _reference_scnm(scnm, volume, data_apix, tta=1, batch_size=1):
    """copick-easymode 0.3.0's ``segment_volume_scnm``, verbatim."""
    from scipy.ndimage import zoom

    original_shape = volume.shape
    prepared, padding = _preprocess(volume, data_apix, scnm)
    pt, pb, pl, pr = padding
    seg = np.zeros((prepared.shape[0], prepared.shape[1] - pt - pb, prepared.shape[2] - pl - pr), dtype=np.float32)
    for k in range(tta):
        vi = np.rot90(prepared, k=_TTA_ROTATIONS[k], axes=(1, 2))
        if _TTA_FLIPS[k]:
            vi = np.flip(vi, axis=2)
        vi = np.ascontiguousarray(vi)
        si = _infer_slab(scnm, vi) if scnm.is_slab else _infer_slices(scnm, vi, batch_size)
        if _TTA_FLIPS[k]:
            si = np.flip(si, axis=2)
        si = np.rot90(si, k=-_TTA_ROTATIONS[k], axes=(1, 2))
        seg += _unpad_xy(si, padding)
    seg = np.clip(seg / tta, 0.0, 1.0)
    if seg.shape != tuple(original_shape):
        seg = zoom(
            seg,
            (1.0, original_shape[1] / seg.shape[1], original_shape[2] / seg.shape[2]),
            order=1,
            prefilter=False,
        )
    return seg.astype(np.float32)


@pytest.fixture(scope="module")
def conv3d():
    """A tiny 3D network with the input contract easymode's tiling expects."""
    inp = keras.Input(shape=(None, None, None, 1))
    x = keras.layers.Conv3D(2, 3, padding="same")(inp)
    x = keras.layers.Conv3D(1, 3, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, x)


def _settings(voxel_size, **kw):
    return InferenceSettings(tomo_type="wbp", voxel_size=voxel_size, user_id="test", session_id="1", **kw)


def _capturing(stages, captured):
    """``stages`` whose finished probability maps are also kept, per run, in the order they finish."""

    def finish(raw, prepared):
        out = stages.finish(raw, prepared)
        captured.append(out)
        return out

    return Stages(prepare=stages.prepare, infer=stages.infer, finish=finish)


def _segmentation(run, name="tiny"):
    segs = run.get_segmentations(name=name, user_id="test", session_id="1", voxel_size=run_voxel(run))
    return segs[0].numpy() if segs else None


def run_voxel(run):
    return run.voxel_spacings[0].voxel_size


@pytest.mark.parametrize("tta", [1, 3])
def test_the_h5_stages_compose_to_the_released_function(conv3d, tta):
    volume = np.random.default_rng(3).normal(size=(9, 38, 45)).astype(np.float32)
    expected = _reference_h5(conv3d, volume, input_apix=12.0, model_apix=10.0, tta=tta, batch_size=1)
    assert np.array_equal(segment_tomogram_from_array(conv3d, volume, 12.0, 10.0, tta=tta, batch_size=1), expected)
    prepared = prepare_tomogram(volume, 12.0, 10.0)
    assert np.array_equal(finish_tomogram(infer_tomogram(conv3d, prepared, tta, 1), prepared, tta), expected)


def test_the_scnm_stages_compose_to_the_released_function(tmp_path):
    scnm = load_scnm(make_scnm(tmp_path))
    volume = np.random.default_rng(4).normal(size=(6, 36, 44)).astype(np.float32)
    expected = _reference_scnm(scnm, volume, data_apix=12.0, tta=3)
    assert np.array_equal(segment_volume_scnm(scnm, volume, 12.0, tta=3), expected)
    prepared = prepare_scnm(scnm, volume, 12.0)
    assert np.array_equal(finish_scnm(infer_scnm(scnm, prepared, 3), prepared, 3), expected)


def test_the_overlapped_loop_writes_exactly_the_serial_result(tmp_path, conv3d):
    _config, root = make_project(tmp_path, RUNS, voxel_size=12.0, objects=["tiny"])
    stages = Stages(
        prepare=lambda v: prepare_tomogram(v, 12.0, 10.0),
        infer=lambda p: infer_tomogram(conv3d, p, 2, 1),
        finish=lambda raw, p: finish_tomogram(raw, p, 2),
    )
    captured = []
    stats = segment_runs(
        root.runs,
        feature="tiny",
        object_name="tiny",
        stages=_capturing(stages, captured),
        settings=_settings(12.0, tta=2, threshold=0.5),
        stats=new_stats(),
    )
    assert (stats["processed"], stats["skipped"], stats["errors"]) == (4, 0, [])
    assert [item["run"] for item in stats["items"]] == [r.name for r in root.runs]  # recorded in run order
    assert all({"read", "prepare", "infer", "finish", "write"} <= set(item["seconds"]) for item in stats["items"])
    assert len(captured) == len(root.runs)
    for run, probabilities in ((run, captured[i]) for i, run in enumerate(root.runs)):
        tomogram = run.get_voxel_spacing(12.0).get_tomogram("wbp").numpy()
        serial = segment_tomogram_from_array(conv3d, tomogram, 12.0, 10.0, tta=2, batch_size=1)
        assert np.array_equal(probabilities, serial), run.name
        assert np.array_equal(_segmentation(run), (serial >= 0.5).astype(np.uint8)), run.name

    # A second pass skips every run whose segmentation exists, and reads nothing.
    again = segment_runs(
        root.runs,
        feature="tiny",
        object_name="tiny",
        stages=stages,
        settings=_settings(12.0),
        stats=new_stats(),
    )
    assert (again["processed"], again["skipped"], again["errors"]) == (0, 4, [])
    assert all(item["seconds"] == {} for item in again["items"])


def test_a_failing_run_is_recorded_and_the_others_finish(tmp_path):
    _config, root = make_project(tmp_path, RUNS, objects=["tiny"])
    scnm = load_scnm(make_scnm(tmp_path / "model"))

    def prepare(volume):
        if volume.shape == RUNS["r2"]:
            raise RuntimeError("unreadable tomogram")
        return prepare_scnm(scnm, volume, 10.0)

    def infer(prepared):
        if prepared.original_shape == RUNS["r3"]:
            raise RuntimeError("out of GPU memory")
        return infer_scnm(scnm, prepared, 1)

    stats = segment_runs(
        root.runs,
        feature="tiny",
        object_name="tiny",
        stages=Stages(prepare=prepare, infer=infer, finish=lambda raw, p: finish_scnm(raw, p, 1)),
        settings=_settings(10.0, tta=1),
        stats=new_stats(),
    )
    assert stats["processed"] == 2 and stats["skipped"] == 0
    assert stats["errors"] == [
        "Error processing tiny in r2: unreadable tomogram",
        "Error processing tiny in r3: out of GPU memory",
    ]
    assert {item["run"]: item["status"] for item in stats["items"]} == {
        "r1": "processed",
        "r2": "error",
        "r3": "error",
        "r4": "processed",
    }
    written = {run.name: _segmentation(run) is not None for run in root.runs}
    assert written == {"r1": True, "r2": False, "r3": False, "r4": True}
