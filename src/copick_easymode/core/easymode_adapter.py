"""Compatibility boundary for the immutable easymode 1.0.0 source revision."""

from dataclasses import dataclass
from typing import Any, Callable

EASYMODE_COMMIT = "a42377e0b887364050bf47c63700a3dd1c0fa0d0"
EASYMODE_INSTALL = (
    f"python -m pip install --no-deps 'easymode @ git+https://github.com/mgflast/easymode.git@{EASYMODE_COMMIT}'"
)


class EasymodeCompatibilityError(RuntimeError):
    """Raised when the supported easymode source API is unavailable."""


@dataclass(frozen=True)
class InferenceFunctions:
    """Private easymode inference functions used by this integration."""

    pad_volume: Callable
    segment_tomogram_instance: Callable


@dataclass(frozen=True)
class EasymodeRuntime:
    """Heavy runtime components loaded only when inference starts."""

    tensorflow: Any
    get_model: Callable
    load_model: Callable


def _compatibility_error(detail: str) -> EasymodeCompatibilityError:
    return EasymodeCompatibilityError(
        "The supported easymode 1.0.0 runtime is missing or incompatible "
        f"({detail}). Install its immutable source revision with: {EASYMODE_INSTALL}",
    )


def get_inference_functions() -> InferenceFunctions:
    """Load and validate the private inference functions used by the adapter."""
    try:
        from easymode.segmentation.inference import _pad_volume, _segment_tomogram_instance
    except (ImportError, AttributeError) as error:
        raise _compatibility_error(str(error)) from error

    if not callable(_pad_volume) or not callable(_segment_tomogram_instance):
        raise _compatibility_error("required inference attributes are not callable")
    return InferenceFunctions(_pad_volume, _segment_tomogram_instance)


def load_runtime() -> EasymodeRuntime:
    """Load TensorFlow and easymode's public model distribution functions."""
    try:
        import tensorflow as tf
        from easymode.core.distribution import get_model, load_model
    except (ImportError, AttributeError) as error:
        raise _compatibility_error(str(error)) from error

    if not callable(get_model) or not callable(load_model):
        raise _compatibility_error("required model distribution attributes are not callable")
    return EasymodeRuntime(tf, get_model, load_model)


def list_remote_models():
    """Return upstream model records without exposing its import path elsewhere."""
    try:
        from easymode.core.distribution import list_remote_models as upstream_list_remote_models
    except (ImportError, AttributeError) as error:
        raise _compatibility_error(str(error)) from error
    return upstream_list_remote_models()


def get_model(name: str, *, silent: bool = True):
    """Resolve one upstream model through the supported distribution API."""
    try:
        from easymode.core.distribution import get_model as upstream_get_model
    except (ImportError, AttributeError) as error:
        raise _compatibility_error(str(error)) from error
    return upstream_get_model(name, silent=silent)
