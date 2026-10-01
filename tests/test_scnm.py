"""The .scnm loader and its inference port, on tiny models shaped like Ais's two kinds."""

import json
import tarfile

import numpy as np
import pytest

keras = pytest.importorskip("keras")
h5py = pytest.importorskip("h5py")

from copick_easymode.core.models import copick_name  # noqa: E402
from copick_easymode.core.scnm import MAX_TTA, is_scnm, load_scnm, segment_volume_scnm  # noqa: E402


def _keras2_groups(h5_path):
    """Write ``groups: 1`` into every transposed convolution, as Keras 2 serialized it."""
    with h5py.File(h5_path, "r+") as fh:
        config = json.loads(fh.attrs["model_config"])
        for layer in config["config"]["layers"]:
            if layer["class_name"].endswith("Transpose"):
                layer["config"]["groups"] = 1
        fh.attrs["model_config"] = json.dumps(config)


def _scnm(tmp_path, title, model, meta):
    weights = tmp_path / f"{title}_weights.h5"
    model.save(weights)
    _keras2_groups(weights)
    metadata = tmp_path / f"{title}_metadata.json"
    metadata.write_text(json.dumps({"title": title, **meta}))
    path = tmp_path / f"{title}.scnm"
    with tarfile.open(path, "w") as archive:
        archive.add(weights, arcname=weights.name)
        archive.add(metadata, arcname=metadata.name)
    return path


@pytest.fixture
def slab_model(tmp_path):
    """ezm-3d: a (Y, X, depth, 1) slab in, the same slab of probabilities out."""
    inp = keras.Input(shape=(None, None, 4, 1), batch_size=1)
    x = keras.layers.Conv3D(2, 3, padding="same")(inp)
    x = keras.layers.Conv3DTranspose(1, 3, padding="same", activation="sigmoid")(x)
    meta = {"apix": 10.0, "model_depth": 4, "arch": "ezm-3d", "normalization": "global_mad", "z_jitter": 2}
    return _scnm(tmp_path, "slabby", keras.Model(inp, x), meta)


@pytest.fixture
def slice_model(tmp_path):
    """ezm-2d (2.5D): a (Y, X, depth) window of slices in, one slice of probabilities out."""
    inp = keras.Input(shape=(None, None, 4), batch_size=1)
    x = keras.layers.Conv2DTranspose(2, 3, padding="same")(inp)
    x = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    meta = {"apix": 10.0, "model_depth": 4, "arch": "ezm-2d", "normalization": "global_mad", "z_jitter": 2}
    return _scnm(tmp_path, "slicey", keras.Model(inp, x), meta)


def test_a_keras2_slab_model_loads_and_segments_to_the_input_shape(slab_model, tmp_path):
    with pytest.raises(Exception, match="groups"):  # what load_scnm's layer shim is for
        keras.models.load_model(tmp_path / "slabby_weights.h5", compile=False)
    scnm = load_scnm(slab_model)
    assert is_scnm(slab_model) and scnm.is_slab and scnm.dimensionality == 3
    assert (scnm.title, scnm.apix, scnm.depth, scnm.z_jitter) == ("slabby", 10.0, 4, 2)

    volume = np.random.default_rng(0).normal(size=(11, 40, 52)).astype(np.float32)
    seg = segment_volume_scnm(scnm, volume, data_apix=10.0, tta=2)
    assert seg.shape == volume.shape and seg.dtype == np.float32
    assert seg.min() >= 0.0 and seg.max() <= 1.0 and seg.max() > 0.0


def test_a_slice_model_rescales_to_its_apix_and_back(slice_model):
    scnm = load_scnm(slice_model)
    assert not scnm.is_slab and scnm.dimensionality == 2.5

    volume = np.random.default_rng(1).normal(size=(6, 36, 44)).astype(np.float32)
    seg = segment_volume_scnm(scnm, volume, data_apix=5.0, tta=MAX_TTA)
    assert seg.shape == volume.shape
    assert seg.min() >= 0.0 and seg.max() <= 1.0


def test_tta_beyond_ais_range_is_refused(slice_model):
    with pytest.raises(ValueError, match="tta"):
        segment_volume_scnm(load_scnm(slice_model), np.zeros((4, 32, 32), np.float32), 10.0, tta=MAX_TTA + 1)


def test_a_feature_keeps_its_easymode_name_but_gets_copicks():
    assert copick_name("atp_synthase") == "atp-synthase"
    assert copick_name("ribosome") == "ribosome"
    assert not is_scnm("models/ribosome_sv3-3d.h5")
