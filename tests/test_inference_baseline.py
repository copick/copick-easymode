"""Freeze numerical behavior before changing the external runtime boundary."""

import hashlib

import numpy as np
import pytest

from copick_easymode.core import easymode_adapter
from copick_easymode.core.inference import segment_tomogram_from_array


@pytest.fixture
def identity_easymode(monkeypatch):
    functions = easymode_adapter.InferenceFunctions(
        pad_volume=lambda volume: (volume, ((0, 0), (0, 0), (0, 0))),
        segment_tomogram_instance=lambda volume, model, batch_size, tile_size, overlap: volume,
    )
    monkeypatch.setattr(easymode_adapter, "get_inference_functions", lambda: functions)


@pytest.mark.parametrize(
    ("input_apix", "expected_digest"),
    [
        (10.0, "973aa66800b0ffcda9db1c32d76642e9f202e2493b8d2156136b6171ed616d51"),
        (20.0, "a4639623698ebe9481f6ad6a18a1cedb0d2238d9c6af0489ac8ba2ac57a8bec9"),
    ],
)
def test_pre_migration_numerical_result_is_frozen(identity_easymode, input_apix, expected_digest):
    shape = (5, 8, 10)
    volume = ((np.indices(shape) * np.array([17, 5, 2])[:, None, None, None]).sum(0) % 23).astype(np.float32)

    result = segment_tomogram_from_array(
        model=object(),
        volume=volume,
        input_apix=input_apix,
        model_apix=10.0,
        tta=1,
        batch_size=1,
    )

    assert result.shape == shape
    assert result.dtype == np.float32
    digest = hashlib.sha256(np.round(result, 5).tobytes()).hexdigest()
    assert digest == expected_digest
