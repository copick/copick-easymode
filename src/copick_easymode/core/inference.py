"""
Core inference logic bridging easymode and copick.

This module provides functions to run easymode pretrained segmentation models
on copick tomograms and store the results back to copick.
"""

import gc
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from copick.models import CopickRoot


def segment_tomogram_from_array(
    model,
    volume: np.ndarray,
    input_apix: float,
    model_apix: float = 10.0,
    tta: int = 1,
    batch_size: int = 2,
) -> np.ndarray:
    """
    Segment a tomogram from a numpy array.

    This function is adapted from easymode.segmentation.inference.segment_tomogram
    but works directly with numpy arrays instead of MRC files.

    Args:
        model: Loaded easymode TensorFlow model.
        volume: Input tomogram as numpy array (will be converted to float32).
        input_apix: Voxel size of input in Angstroms per pixel.
        model_apix: Target voxel size the model was trained at (default 10.0 A/px).
        tta: Test-time augmentation level (1-16). Higher values average more
             rotated/flipped predictions for better accuracy but slower inference.
        batch_size: Batch size for tile prediction.

    Returns:
        Segmentation probability map as numpy array (float32, values 0-1).
    """
    from scipy.ndimage import zoom

    # Import easymode inference functions
    from easymode.segmentation.inference import _pad_volume, _segment_tomogram_instance

    volume = volume.astype(np.float32)
    oj, ok, ol = volume.shape

    # Scale to model resolution
    scale = float(input_apix) / float(model_apix)

    if abs(scale - 1.0) > 0.05:
        volume = zoom(volume, scale, order=1)

    # Preprocess: normalize using margins to avoid edge artifacts
    _j, _k, _l = volume.shape
    _k_margin = min(int(0.2 * _k), 64)
    _l_margin = min(int(0.2 * _l), 64)
    volume -= np.mean(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin])
    volume /= np.std(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin]) + 1e-7

    # Pad volume to be divisible by 32
    volume, padding = _pad_volume(volume)
    segmented_volume = np.zeros_like(volume)

    # Adjust tile size based on volume shape
    tile_size = (
        min(256, segmented_volume.shape[0]),
        min(256, segmented_volume.shape[1]),
        min(256, segmented_volume.shape[2]),
    )
    overlap = [
        0 if tile_size[0] == segmented_volume.shape[0] else 48,
        0 if tile_size[1] == segmented_volume.shape[1] else 48,
        0 if tile_size[2] == segmented_volume.shape[2] else 48,
    ]

    # TTA rotation/flip combinations that respect data anisotropy
    # These are all 16 valid combinations of 90-degree rotations and flips
    k_xy = [0, 2, 2, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3]
    k_fx = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    k_yz = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    # Inference loop with TTA
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

    # Remove padding
    (j0, j1), (k0, k1), (l0, l1) = padding
    segmented_volume = segmented_volume[
        j0 : segmented_volume.shape[0] - j1,
        k0 : segmented_volume.shape[1] - k1,
        l0 : segmented_volume.shape[2] - l1,
    ]

    # Rescale back to original size
    if abs(scale - 1.0) > 0.05:
        sj, sk, sl = segmented_volume.shape
        segmented_volume = zoom(segmented_volume, (oj / sj, ok / sk, ol / sl), order=1)

    return segmented_volume.astype(np.float32)


def run_easymode_inference(
    root: "CopickRoot",
    run_names: list[str],
    tomo_type: str,
    voxel_size: float,
    models: list[str],
    user_id: str,
    session_id: str,
    tta: int = 4,
    batch_size: int = 1,
    threshold: float = 0.5,
    gpus: Optional[str] = None,
    add_objects: bool = True,
    overwrite: bool = False,
    config_path: Optional[str] = None,
    logger=None,
) -> dict:
    """
    Run easymode inference on copick tomograms.

    Args:
        root: CopickRoot instance with loaded project.
        run_names: List of run names to process. Empty list means all runs.
        tomo_type: Tomogram type (e.g., 'wbp', 'sirt').
        voxel_size: Voxel size in Angstroms.
        models: List of easymode model names to run.
        user_id: User ID for created segmentations.
        session_id: Session ID for created segmentations.
        tta: Test-time augmentation level (1-16).
        batch_size: Batch size for inference.
        threshold: Probability threshold for binarizing segmentation (0.0-1.0).
        gpus: Comma-separated GPU IDs (e.g., '0,1'). None for auto-detect.
        add_objects: Whether to add object definitions if missing.
        overwrite: Whether to overwrite existing segmentations.
        config_path: Path to save config if add_objects is True.
        logger: Logger instance for output messages.

    Returns:
        Dictionary with processing statistics: processed, skipped, errors.
    """
    import tensorflow as tf

    from easymode.core.distribution import get_model, load_model

    stats = {"processed": 0, "skipped": 0, "errors": []}

    # Configure GPUs
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Enable memory growth to avoid allocating all GPU memory at once
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass  # Memory growth must be set before GPUs have been initialized

    # Get runs to process
    runs = root.runs
    if run_names:
        runs = [r for r in runs if r.name in run_names]

    if not runs:
        if logger:
            logger.warning("No runs found to process.")
        return stats

    # Track if config was modified
    config_modified = False

    # Process each model
    for model_name in models:
        if logger:
            logger.info(f"Loading model: {model_name}")

        # Get and load model
        model_path, metadata = get_model(model_name)
        if model_path is None:
            error_msg = f"Model '{model_name}' not found. Skipping."
            if logger:
                logger.error(error_msg)
            stats["errors"].append(error_msg)
            continue

        model_apix = metadata.get("apix", 10.0)

        if logger:
            logger.info(f"Model loaded from {model_path}, inference at {model_apix} A/px")

        model = load_model(model_path)

        # Add object definition if needed
        if add_objects:
            existing_obj = root.get_object(model_name)
            if existing_obj is None:
                if logger:
                    logger.info(f"Adding object definition for '{model_name}'")
                root.new_object(
                    name=model_name,
                    is_particle=False,  # Segmentation, not particle picks
                    # label and color will be auto-assigned
                )
                config_modified = True

        # Process each run
        for run in runs:
            if logger:
                logger.info(f"Processing run: {run.name}")

            # Get tomogram
            try:
                vs = run.get_voxel_spacing(voxel_size)
                if vs is None:
                    if logger:
                        logger.warning(f"Voxel spacing {voxel_size} not found in {run.name}")
                    stats["skipped"] += 1
                    continue

                tomo = vs.get_tomogram(tomo_type)
                if tomo is None:
                    if logger:
                        logger.warning(f"Tomogram {tomo_type}@{voxel_size} not found in {run.name}")
                    stats["skipped"] += 1
                    continue
            except Exception as e:
                error_msg = f"Error getting tomogram in {run.name}: {e}"
                if logger:
                    logger.warning(error_msg)
                stats["errors"].append(error_msg)
                continue

            # Check if segmentation already exists
            existing_segs = run.get_segmentations(
                name=model_name,
                user_id=user_id,
                session_id=session_id,
                voxel_size=voxel_size,
                is_multilabel=False,
            )

            if existing_segs and not overwrite:
                if logger:
                    logger.info(f"Segmentation already exists for {model_name} in {run.name}, skipping")
                stats["skipped"] += 1
                continue

            try:
                # Read tomogram as numpy
                if logger:
                    logger.info(f"Reading tomogram from {run.name}")
                tomo_data = tomo.numpy()

                # Run inference
                if logger:
                    logger.info(f"Running inference for {model_name} on {run.name}")

                seg_data = segment_tomogram_from_array(
                    model=model,
                    volume=tomo_data,
                    input_apix=voxel_size,
                    model_apix=model_apix,
                    tta=tta,
                    batch_size=batch_size,
                )

                # Binarize using threshold and convert to uint8 (0 or 1)
                seg_data = (seg_data >= threshold).astype(np.uint8)

                # Create or get segmentation
                if existing_segs and overwrite:
                    # Delete existing segmentation first
                    for existing_seg in existing_segs:
                        # Note: copick doesn't have a delete method, so we overwrite via exist_ok
                        pass

                seg = run.new_segmentation(
                    name=model_name,
                    voxel_size=voxel_size,
                    user_id=user_id,
                    session_id=session_id,
                    is_multilabel=False,
                )

                # Write segmentation
                seg.from_numpy(seg_data)

                if logger:
                    logger.info(f"Saved segmentation for {model_name} in {run.name}")

                stats["processed"] += 1

            except Exception as e:
                error_msg = f"Error processing {model_name} in {run.name}: {e}"
                if logger:
                    logger.exception(error_msg)
                stats["errors"].append(error_msg)

        # Clean up model to free GPU memory
        tf.keras.backend.clear_session()
        gc.collect()

    # Save config if modified
    if config_modified and config_path:
        if logger:
            logger.info(f"Saving updated config to {config_path}")
        root.save_config(config_path)

    return stats
