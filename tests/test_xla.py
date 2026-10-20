"""The XLA wrapper: called like the model, attributes passed through, the plain model after a failed call."""

import dataclasses
import logging

import numpy as np
import pytest

keras = pytest.importorskip("keras")
pytest.importorskip("tensorflow")

from conftest import make_scnm  # noqa: E402

from copick_easymode.core.scnm import load_scnm  # noqa: E402
from copick_easymode.core.xla import XLAModel  # noqa: E402


def _slab_model():
    inp = keras.Input(shape=(None, None, 4, 1))
    x = keras.layers.Conv3D(2, 3, padding="same")(inp)
    return keras.Model(inp, keras.layers.Conv3D(1, 1, activation="sigmoid")(x))


def test_the_wrapper_passes_attributes_through_to_the_model(tmp_path):
    model = _slab_model()
    wrapped = XLAModel(model, name="slab")
    assert wrapped.input is model.input
    assert (wrapped.input_shape, wrapped.output_shape) == (model.input_shape, model.output_shape)
    assert wrapped.layers == model.layers
    with pytest.raises(AttributeError):
        wrapped.no_such_attribute  # noqa: B018

    # An .scnm model reads its kind from the network's shapes, through the wrapper alike.
    scnm = load_scnm(make_scnm(tmp_path / "models"))
    compiled = dataclasses.replace(scnm, model=XLAModel(scnm.model))
    assert (compiled.is_slab, compiled.dimensionality) == (scnm.is_slab, scnm.dimensionality) == (False, 2.5)


def test_the_compiled_call_returns_what_the_model_does(caplog):
    model = _slab_model()
    wrapped = XLAModel(model, logger=logging.getLogger("xla-test"), name="slab")
    tile = np.random.default_rng(0).normal(size=(1, 16, 24, 4, 1)).astype(np.float32)

    with caplog.at_level(logging.INFO, logger="xla-test"):
        first = wrapped(tile, training=False).numpy()
        wrapped(tile, training=False)
    np.testing.assert_allclose(first, model(tile, training=False).numpy(), atol=1e-5)
    assert wrapped.fallback is None
    # One compile per input shape, logged once.
    assert [r.getMessage().split(":")[0] for r in caplog.records] == ["XLA compiled slab for input (1, 16, 24, 4, 1)"]


def test_a_failing_compiled_call_falls_back_to_the_model_for_good(caplog):
    model = _slab_model()
    wrapped = XLAModel(model, logger=logging.getLogger("xla-test"), name="slab")
    calls = []

    def broken(x):
        calls.append(x)
        raise RuntimeError("libdevice not found at ./libdevice.10.bc\nmore detail")

    wrapped._compiled = broken
    tile = np.random.default_rng(1).normal(size=(1, 16, 16, 4, 1)).astype(np.float32)

    with caplog.at_level(logging.INFO, logger="xla-test"):
        outputs = [wrapped(tile, training=False).numpy() for _ in range(3)]
    expected = model(tile, training=False).numpy()
    assert all(np.allclose(out, expected) for out in outputs)
    assert len(calls) == 1  # given up after the first failure
    assert wrapped.fallback == "RuntimeError: libdevice not found at ./libdevice.10.bc"
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "continuing without XLA" in warning.getMessage() and "libdevice" in warning.getMessage()
