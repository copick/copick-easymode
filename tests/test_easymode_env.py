"""The easymode boundary: the only module that imports easymode, and its compatibility error."""

import ast
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import copick_easymode
from copick_easymode.core import easymode_env
from copick_easymode.core.models import KNOWN_MODELS, list_available_models


def _module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


_absent = object()


def _distribution(**overrides):
    """What easymode 1.2.6's ``easymode.core.distribution`` offers this package."""
    attributes = {
        "MODEL_CACHE_DIR": "/models",
        "REGISTRY_CACHE": "/models/registry.json",
        "_online": None,
        "get_model": lambda name, **kwargs: (None, None),
        "load_model": lambda path: {"weights": path},
        "list_remote_models": lambda: [],
        "is_online": lambda: False,
    }
    attributes.update(overrides)
    return _module("easymode.core.distribution", **{k: v for k, v in attributes.items() if v is not _absent})


@pytest.fixture
def fake_easymode(monkeypatch, isolated_env):
    """Install a fake easymode whose distribution is ``install(**overrides)``."""
    for name in [name for name in sys.modules if name == "easymode" or name.startswith("easymode.")]:
        monkeypatch.delitem(sys.modules, name)
    config = _module("easymode.core.config", settings={"MODEL_DIRECTORY": "/models"}, version="1.2.6")

    def install(**overrides):
        distribution = _distribution(**overrides)
        for module in (_module("easymode"), _module("easymode.core"), config, distribution):
            monkeypatch.setitem(sys.modules, module.__name__, module)
        return distribution

    return install


@pytest.fixture
def no_easymode(monkeypatch, isolated_env):
    """easymode is not importable, whether or not it is installed."""
    for name in [name for name in sys.modules if name == "easymode" or name.startswith("easymode.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "easymode", None)


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"title": "ribosome"}, {"title": "membrane"}], ["ribosome", "membrane"]),
        (["ribosome", "membrane"], ["ribosome", "membrane"]),
    ],
)
def test_model_listing_normalizes_current_and_legacy_records(monkeypatch, records, expected):
    monkeypatch.setattr(easymode_env, "import_easymode", lambda: SimpleNamespace(list_remote_models=lambda: records))
    assert list_available_models() == expected


def test_model_listing_falls_back_when_easymode_is_unavailable(monkeypatch):
    def unavailable():
        raise easymode_env.EasymodeCompatibilityError("missing")

    monkeypatch.setattr(easymode_env, "import_easymode", unavailable)
    assert list_available_models() == KNOWN_MODELS


@pytest.mark.parametrize("entry", ["import_easymode", "inference_functions"])
def test_a_missing_easymode_names_the_supported_release_and_its_install_command(no_easymode, entry):
    with pytest.raises(easymode_env.EasymodeCompatibilityError) as exc_info:
        getattr(easymode_env, entry)()

    message = str(exc_info.value)
    assert easymode_env.EASYMODE_VERSION in message
    assert easymode_env.EASYMODE_COMMIT in message
    assert "--no-deps" in message
    assert "No module named 'easymode" in message or "import of easymode halted" in message


def test_easymode_without_the_registry_distribution_is_refused(fake_easymode):
    # easymode 1.0.0 had neither the registry cache nor the _online flag.
    fake_easymode(REGISTRY_CACHE=_absent, _online=_absent)

    with pytest.raises(easymode_env.EasymodeCompatibilityError, match="lacks REGISTRY_CACHE, _online") as exc_info:
        easymode_env.import_easymode()
    assert easymode_env.EASYMODE_INSTALL in str(exc_info.value)


def test_distribution_functions_must_be_callable(fake_easymode):
    distribution = fake_easymode(get_model=None)

    with pytest.raises(easymode_env.EasymodeCompatibilityError, match="lacks get_model"):
        easymode_env.import_easymode()
    with pytest.raises(easymode_env.EasymodeCompatibilityError, match="lacks get_model"):
        easymode_env.go_offline(distribution)


def test_a_compatible_easymode_is_used_through_the_boundary(fake_easymode):
    distribution = fake_easymode()

    assert easymode_env.import_easymode() is distribution
    assert easymode_env.easymode_version() == "1.2.6"
    assert easymode_env.load_network("/models/ribosome.h5") == {"weights": "/models/ribosome.h5"}
    easymode_env.go_offline(distribution)
    assert distribution._online is False


def test_the_private_inference_functions_are_checked(fake_easymode, monkeypatch):
    fake_easymode()
    inference = _module("easymode.segmentation.inference", _pad_volume=lambda volume: volume)
    monkeypatch.setitem(sys.modules, "easymode.segmentation", _module("easymode.segmentation"))
    monkeypatch.setitem(sys.modules, "easymode.segmentation.inference", inference)

    with pytest.raises(easymode_env.EasymodeCompatibilityError, match="_segment_tomogram_instance"):
        easymode_env.inference_functions()

    inference._segment_tomogram_instance = lambda volume, model, batch_size, tile_size, overlap: volume
    functions = easymode_env.inference_functions()
    assert functions.pad_volume is inference._pad_volume
    assert functions.segment_tomogram_instance is inference._segment_tomogram_instance


def test_no_other_module_imports_easymode():
    package = Path(copick_easymode.__file__).parent
    boundary = Path(easymode_env.__file__)
    imported = re.compile(r"^easymode(\.\w+)*$")
    named = re.compile(r"^easymode(\.\w+)+$")  # "easymode" alone is also the command's name
    offenders = []
    for path in sorted(package.rglob("*.py")):
        if path == boundary:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names if imported.match(alias.name)]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if imported.match(node.module or "") else []
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # importlib.import_module("easymode..."), sys.modules["easymode..."]
                names = [node.value] if named.match(node.value) else []
            else:
                continue
            offenders += [f"{path.relative_to(package)}:{node.lineno} {name}" for name in names]
    assert not offenders
