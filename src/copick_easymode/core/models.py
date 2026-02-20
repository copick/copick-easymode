"""
Available easymode models and utilities.

This module provides information about available pretrained segmentation models
from easymode that can be used with copick.
"""

from typing import Optional

# Known pretrained segmentation models from easymode (Hugging Face: mgflast/easymode)
# These are the models that can be used for inference
KNOWN_MODELS = [
    # 3D-only models
    "actin",
    "cytoplasmic_granule",
    "membrane",
    "microtubule",
    "mitochondrial_granule",
    "prohibitin",
    "ribosome",
    "tric",
    "vault",
    "void",
    # Models with both 3D and 2D options
    "cytoplasm",
    "mitochondrion",
    "npc",
    "nuclear_envelope",
    "nucleus",
]


def list_available_models(online: bool = True) -> list[str]:
    """
    List available easymode models.

    Args:
        online: If True, fetch the latest list from HuggingFace. If False, return cached list.

    Returns:
        List of available model names.
    """
    if online:
        try:
            from easymode.core.distribution import list_remote_models

            return list_remote_models()
        except Exception:
            # Fall back to known models if online fetch fails
            return KNOWN_MODELS.copy()
    return KNOWN_MODELS.copy()


def validate_model_name(name: str) -> bool:
    """
    Check if a model name is valid (available for download).

    Args:
        name: The model name to validate.

    Returns:
        True if the model is available, False otherwise.
    """
    try:
        from easymode.core.distribution import get_model

        path, meta = get_model(name, silent=True)
        return path is not None
    except Exception:
        return False


def get_model_info(name: str) -> Optional[dict]:
    """
    Get metadata information for a model.

    Args:
        name: The model name.

    Returns:
        Dictionary with model metadata (apix, timestamp) or None if not found.
    """
    try:
        from easymode.core.distribution import get_model

        path, meta = get_model(name, silent=True)
        if path is not None:
            return meta
        return None
    except Exception:
        return None
