"""Tests for the immutable easymode compatibility boundary."""

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from copick_easymode.core import easymode_adapter
from copick_easymode.core.models import KNOWN_MODELS, list_available_models


def _install_distribution(monkeypatch, **attributes):
    easymode = ModuleType("easymode")
    core = ModuleType("easymode.core")
    distribution = ModuleType("easymode.core.distribution")
    for name, value in attributes.items():
        setattr(distribution, name, value)
    monkeypatch.setitem(sys.modules, "easymode", easymode)
    monkeypatch.setitem(sys.modules, "easymode.core", core)
    monkeypatch.setitem(sys.modules, "easymode.core.distribution", distribution)


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"title": "ribosome"}, {"title": "membrane"}], ["ribosome", "membrane"]),
        (["ribosome", "membrane"], ["ribosome", "membrane"]),
    ],
)
def test_model_listing_normalizes_current_and_legacy_records(monkeypatch, records, expected):
    monkeypatch.setattr(easymode_adapter, "list_remote_models", lambda: records)
    assert list_available_models() == expected


def test_model_listing_falls_back_when_distribution_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        easymode_adapter,
        "list_remote_models",
        lambda: (_ for _ in ()).throw(easymode_adapter.EasymodeCompatibilityError("missing")),
    )
    assert list_available_models() == KNOWN_MODELS


def test_missing_inference_api_has_exact_install_guidance(monkeypatch):
    for module_name in [name for name in sys.modules if name == "easymode" or name.startswith("easymode.")]:
        monkeypatch.delitem(sys.modules, module_name)
    original_import = builtins.__import__

    def reject_easymode(name, *args, **kwargs):
        if name == "easymode" or name.startswith("easymode."):
            raise ImportError("easymode is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_easymode)

    with pytest.raises(easymode_adapter.EasymodeCompatibilityError) as exc_info:
        easymode_adapter.get_inference_functions()

    assert easymode_adapter.EASYMODE_COMMIT in str(exc_info.value)
    assert "--no-deps" in str(exc_info.value)


def test_runtime_validates_distribution_functions(monkeypatch):
    _install_distribution(monkeypatch, get_model=None, load_model=lambda path: path)
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace())

    with pytest.raises(easymode_adapter.EasymodeCompatibilityError, match="not callable"):
        easymode_adapter.load_runtime()
