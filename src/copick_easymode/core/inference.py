"""
Core inference logic bridging easymode and copick.

This module provides functions to run easymode pretrained segmentation models
on copick tomograms and store the results back to copick.

Each model runs over the runs in three overlapped stages, as Ais's ``_segmentation_thread`` does: while the
GPU segments run ``i`` in the calling thread, a reader thread reads and preprocesses run ``i + 1`` (existence
and skip checks, ``tomo.numpy()``, the model's rescale, normalization and padding) and a writer thread
finishes run ``i - 1`` (unpad, rescale back, threshold, write the segmentation). At most one run is prepared
ahead and one is being written, so memory stays at about three tomograms. The stages compose to exactly the
serial result.
"""

import contextlib
import dataclasses
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from copick_easymode.core import easymode_env
from copick_easymode.core.models import copick_name

if TYPE_CHECKING:
    from copick.models import CopickRoot, CopickRun

    from copick_easymode.core.easymode_env import ResolvedModel


# ---- the .h5 path, in three stages ---------------------------------------------------------------------------


@dataclass
class PreparedVolume:
    """A tomogram ready for an ``.h5`` model, and what maps the model's output back onto it.

    ``volume`` is the network input; the overlapped loop drops it once inference is done.
    """

    volume: Any
    padding: tuple
    original_shape: tuple
    rescaled: bool


def prepare_tomogram(volume: np.ndarray, input_apix: float, model_apix: float = 10.0) -> PreparedVolume:
    """Stage 1 (CPU) of :func:`segment_tomogram_from_array`: rescale to the model's voxel size, normalize, pad."""
    from scipy.ndimage import zoom

    easymode = easymode_env.inference_functions()

    volume = volume.astype(np.float32)
    original_shape = volume.shape

    # Scale to model resolution
    scale = float(input_apix) / float(model_apix)
    rescaled = abs(scale - 1.0) > 0.05
    if rescaled:
        volume = zoom(volume, scale, order=1)

    # Preprocess: normalize using margins to avoid edge artifacts
    _j, _k, _l = volume.shape
    _k_margin = min(int(0.2 * _k), 64)
    _l_margin = min(int(0.2 * _l), 64)
    volume -= np.mean(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin])
    volume /= np.std(volume[:, _k_margin:-_k_margin, _l_margin:-_l_margin]) + 1e-7

    # Pad volume to be divisible by 32
    volume, padding = easymode.pad_volume(volume)
    return PreparedVolume(volume=volume, padding=padding, original_shape=original_shape, rescaled=rescaled)


def infer_tomogram(model, prepared: PreparedVolume, tta: int = 1, batch_size: int = 2) -> np.ndarray:
    """Stage 2 (GPU): the sum of the ``tta`` augmented passes over the padded volume."""
    easymode = easymode_env.inference_functions()
    volume = prepared.volume
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

        segmented_tta_vol = easymode.segment_tomogram_instance(tta_vol, model, batch_size, tile_size, overlap)

        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-2 * k_yz[j], axes=(0, 1))
        segmented_tta_vol = segmented_tta_vol if not k_fx[j] else np.flip(segmented_tta_vol, axis=1)
        segmented_tta_vol = np.rot90(segmented_tta_vol, k=-k_xy[j], axes=(1, 2))
        segmented_volume += segmented_tta_vol

    return segmented_volume


def finish_tomogram(segmented_volume: np.ndarray, prepared: PreparedVolume, tta: int = 1) -> np.ndarray:
    """Stage 3 (CPU): average the passes, remove the padding and rescale back to the input's shape."""
    from scipy.ndimage import zoom

    segmented_volume /= tta

    # Remove padding
    (j0, j1), (k0, k1), (l0, l1) = prepared.padding
    segmented_volume = segmented_volume[
        j0 : segmented_volume.shape[0] - j1,
        k0 : segmented_volume.shape[1] - k1,
        l0 : segmented_volume.shape[2] - l1,
    ]

    # Rescale back to original size
    if prepared.rescaled:
        oj, ok, ol = prepared.original_shape
        sj, sk, sl = segmented_volume.shape
        segmented_volume = zoom(segmented_volume, (oj / sj, ok / sk, ol / sl), order=1)

    return segmented_volume.astype(np.float32)


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
    but works directly with numpy arrays instead of MRC files. It is
    :func:`prepare_tomogram`, :func:`infer_tomogram` and :func:`finish_tomogram` in sequence.

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
    prepared = prepare_tomogram(volume, input_apix, model_apix)
    return finish_tomogram(infer_tomogram(model, prepared, tta, batch_size), prepared, tta)


# ---- one model over many runs, overlapped --------------------------------------------------------------------


@dataclass
class Stages:
    """One loaded model's three stages: ``prepare`` and ``finish`` run beside ``infer``, which owns the GPU."""

    prepare: Callable[[np.ndarray], Any]
    infer: Callable[[Any], np.ndarray]
    finish: Callable[[np.ndarray, Any], np.ndarray]
    #: The XLA-compiled model ``infer`` calls (``settings.xla``), whose ``fallback`` says whether it was given up.
    xla: Any = None


def copick_identity(value: str, what: str = "name") -> str:
    """``value`` as copick stores a user ID, session ID or object name; a ``ValueError`` names one it cannot."""
    try:
        return copick_name(value)
    except ValueError as exc:
        raise ValueError(f"Invalid copick {what} {value!r}: {exc}") from None


@dataclass
class InferenceSettings:
    """What every run of one inference call shares (picklable, so workers receive it as is).

    ``user_id`` and ``session_id`` are held as copick stores them. copick sanitizes the IDs of what it writes
    (underscores, whitespace and path characters become dashes) but looks them up verbatim, so the check for an
    existing segmentation and the writer must both use the sanitized IDs; build the settings with
    :func:`copick_settings` to also resolve a missing user ID to the project's.
    """

    tomo_type: str
    voxel_size: float
    user_id: Optional[str]
    session_id: str
    tta: int = 4
    batch_size: int = 1
    threshold: float = 0.5
    overwrite: bool = False
    #: Run each model as one XLA-compiled function (:class:`copick_easymode.core.xla.XLAModel`).
    xla: bool = False

    def __post_init__(self):
        if self.user_id is not None:
            self.user_id = copick_identity(self.user_id, "user ID")
        self.session_id = copick_identity(self.session_id, "session ID")


def copick_settings(
    root: "CopickRoot",
    *,
    user_id: Optional[str],
    session_id: str,
    logger=None,
    **options,
) -> InferenceSettings:
    """:class:`InferenceSettings` for ``root``, with copick's identities.

    A missing ``user_id`` is the project's (``root.config.user_id``), as ``new_segmentation`` would make it; a
    sanitized ID is logged. Raises ``ValueError`` before any work when there is no user ID or an ID copick cannot
    store.
    """
    if user_id is None:
        user_id = root.config.user_id
    if user_id is None:
        raise ValueError("No user ID: pass one, or set user_id in the copick config")
    settings = InferenceSettings(user_id=user_id, session_id=session_id, **options)
    for what, requested, used in (
        ("user ID", user_id, settings.user_id),
        ("session ID", session_id, settings.session_id),
    ):
        if logger is not None and used != requested:
            logger.info(f"Using copick {what} '{used}' for the requested '{requested}'")
    return settings


@dataclass
class _Item:
    run: "CopickRun"
    status: str = "ready"  # ready -> processed | skipped | error
    message: str = ""
    prepared: Any = None
    raw: Optional[np.ndarray] = None
    seconds: Dict[str, float] = field(default_factory=dict)

    def end(self, status: str, message: str = "") -> "_Item":
        self.status, self.message = status, message
        self.prepared = self.raw = None
        return self


def new_stats() -> dict:
    """Processing statistics: counts, error messages, one record per run and model, and each model that gave up XLA."""
    return {"processed": 0, "skipped": 0, "errors": [], "items": [], "xla_fallbacks": []}


class _Quiet:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _timed(item: _Item, stage: str, func, *args):
    started = time.perf_counter()
    try:
        return func(*args)
    finally:
        item.seconds[stage] = round(time.perf_counter() - started, 3)


def segment_runs(
    runs: Sequence["CopickRun"],
    *,
    feature: str,
    object_name: str,
    stages: Stages,
    settings: InferenceSettings,
    stats: dict,
    logger=None,
) -> dict:
    """Segment ``runs`` with one loaded model, overlapping reads and writes with inference.

    Each run's outcome is recorded in ``stats`` (all bookkeeping happens in the calling thread); a run that
    fails is recorded and the others continue. Tomograms are read and segmentations found and written through
    copick alone, under copick's own identities (``object_name`` is sanitized as copick stores it).
    """
    log = logger if logger is not None else _Quiet()
    s = settings
    object_name = copick_identity(object_name, "object name")

    def read(run) -> _Item:
        item = _Item(run=run)
        log.info(f"Processing run: {run.name}")
        try:
            vs = run.get_voxel_spacing(s.voxel_size)
            if vs is None:
                log.warning(f"Voxel spacing {s.voxel_size} not found in {run.name}")
                return item.end("skipped", f"voxel spacing {s.voxel_size} not found")
            tomograms = vs.get_tomograms(tomo_type=s.tomo_type)
            if not tomograms:
                log.warning(f"Tomogram {s.tomo_type}@{s.voxel_size} not found in {run.name}")
                return item.end("skipped", f"tomogram {s.tomo_type}@{s.voxel_size} not found")
            tomo = tomograms[0]
            existing = run.get_segmentations(
                name=object_name,
                user_id=s.user_id,
                session_id=s.session_id,
                voxel_size=s.voxel_size,
                is_multilabel=False,
            )
        except Exception as e:
            message = f"Error getting tomogram in {run.name}: {e}"
            log.warning(message)
            return item.end("error", message)
        if existing and not s.overwrite:
            log.info(f"Segmentation already exists for {feature} in {run.name}, skipping")
            return item.end("skipped", "segmentation exists")
        try:
            log.info(f"Reading tomogram from {run.name}")
            volume = _timed(item, "read", tomo.numpy)
            item.prepared = _timed(item, "prepare", stages.prepare, volume)
        except Exception as e:
            message = f"Error processing {feature} in {run.name}: {e}"
            log.exception(message)
            return item.end("error", message)
        return item

    def write(item: _Item) -> _Item:
        run = item.run
        try:
            probabilities = _timed(item, "finish", stages.finish, item.raw, item.prepared)
            item.raw = item.prepared = None
            # Binarize using threshold and convert to uint8 (0 or 1)
            seg_data = (probabilities >= s.threshold).astype(np.uint8)
            del probabilities
            started = time.perf_counter()
            seg = run.new_segmentation(
                name=object_name,
                voxel_size=s.voxel_size,
                user_id=s.user_id,
                session_id=s.session_id,
                is_multilabel=False,
                exist_ok=s.overwrite,
            )
            seg.from_numpy(seg_data)
            item.seconds["write"] = round(time.perf_counter() - started, 3)
        except Exception as e:
            message = f"Error processing {feature} in {run.name}: {e}"
            log.exception(message)
            return item.end("error", message)
        timing = ", ".join(f"{stage} {sec:.1f} s" for stage, sec in item.seconds.items())
        log.info(f"Saved segmentation for {feature} in {run.name} ({timing})")
        return item.end("processed")

    def record(item: _Item) -> None:
        entry = {"run": item.run.name, "feature": feature, "status": item.status, "seconds": dict(item.seconds)}
        if item.message:
            entry["message"] = item.message
        stats["items"].append(entry)
        if item.status == "processed":
            stats["processed"] += 1
        elif item.status == "skipped":
            stats["skipped"] += 1
        else:
            stats["errors"].append(item.message)

    runs = list(runs)
    if not runs:
        return stats
    with ThreadPoolExecutor(1, thread_name_prefix="easymode-read") as reader, ThreadPoolExecutor(
        1,
        thread_name_prefix="easymode-write",
    ) as writer:
        upcoming = reader.submit(read, runs[0])
        writing = None
        for i in range(len(runs)):
            item = upcoming.result()
            # Run i + 1 is read and prepared while run i is on the GPU: one ahead, never more.
            upcoming = reader.submit(read, runs[i + 1]) if i + 1 < len(runs) else None
            if item.status == "ready":
                log.info(f"Running inference for {feature} on {item.run.name}")
                try:
                    item.raw = _timed(item, "infer", stages.infer, item.prepared)
                except Exception as e:
                    message = f"Error processing {feature} in {item.run.name}: {e}"
                    log.exception(message)
                    item.end("error", message)
                else:
                    if hasattr(item.prepared, "volume"):
                        item.prepared.volume = None  # the network input is not needed to finish
            if item.status != "ready":
                record(item)
                continue
            # One run is written at a time: run i - 1 had all of run i's inference to finish.
            if writing is not None:
                record(writing.result())
            writing = writer.submit(write, item)
        if writing is not None:
            record(writing.result())
    return stats


# ---- loading a resolved model and running it -----------------------------------------------------------------


def _enable_memory_growth(tf) -> None:
    # Enable memory growth to avoid allocating all GPU memory at once
    for device in tf.config.list_physical_devices("GPU"):
        # Memory growth must be set before GPUs have been initialized
        with contextlib.suppress(RuntimeError):
            tf.config.experimental.set_memory_growth(device, True)


def load_stages(model: "ResolvedModel", settings: InferenceSettings, logger=None) -> Stages:
    """Load a resolved model's weights and bind its three stages to ``settings``.

    With ``settings.xla`` the network is wrapped in an :class:`~copick_easymode.core.xla.XLAModel`, which the stages
    call exactly as they call the plain network, and which ``Stages.xla`` keeps for its fallback.
    """
    log = logger if logger is not None else _Quiet()

    def compiled(network):
        if not settings.xla:
            return None
        from copick_easymode.core.xla import XLAModel

        return XLAModel(network, logger=logger, name=model.feature)

    if model.kind == "scnm":
        from copick_easymode.core.scnm import MAX_TTA, finish_scnm, infer_scnm, load_scnm, prepare_scnm

        # An Ais 2D-engine model (most easymode features): its own metadata says how it runs.
        scnm = load_scnm(model.weights)
        model_tta = min(settings.tta, MAX_TTA)
        log.info(
            f"Model loaded from {model.weights}: {scnm.dimensionality}D {'slab' if scnm.is_slab else 'slice'}"
            f" model, inference at {scnm.apix} A/px",
        )
        if model_tta != settings.tta:
            log.info(f"TTA {settings.tta} reduced to {model_tta}, the most an .scnm model supports")
        xla = compiled(scnm.model)
        if xla is not None:
            scnm = dataclasses.replace(scnm, model=xla)
        return Stages(
            prepare=lambda volume: prepare_scnm(scnm, volume, settings.voxel_size),
            infer=lambda prepared: infer_scnm(scnm, prepared, model_tta),
            finish=lambda raw, prepared: finish_scnm(raw, prepared, model_tta),
            xla=xla,
        )

    # Imported here rather than first inside a stage thread (it configures TensorFlow at import).
    easymode_env.inference_functions()
    model_apix = model.apix if model.apix is not None else 10.0
    network = easymode_env.load_network(model.weights)
    log.info(f"Model loaded from {model.weights}, inference at {model_apix} A/px")
    xla = compiled(network)
    if xla is not None:
        network = xla
    return Stages(
        prepare=lambda volume: prepare_tomogram(volume, settings.voxel_size, model_apix),
        infer=lambda prepared: infer_tomogram(network, prepared, settings.tta, settings.batch_size),
        finish=lambda raw, prepared: finish_tomogram(raw, prepared, settings.tta),
        xla=xla,
    )


def select_runs(root: "CopickRoot", run_names: Sequence[str]) -> Tuple[List["CopickRun"], List[str]]:
    """The project's runs named in ``run_names`` (all runs when empty), and the requested names it lacks."""
    runs = list(root.runs)
    if not run_names:
        return runs, []
    wanted = set(run_names)
    known = {r.name for r in runs}
    return [r for r in runs if r.name in wanted], [n for n in dict.fromkeys(run_names) if n not in known]


def segment_with_models(
    runs: Sequence["CopickRun"],
    models: Sequence["ResolvedModel"],
    settings: InferenceSettings,
    stats: Optional[dict] = None,
    logger=None,
) -> dict:
    """Run every resolved model over ``runs``, one model at a time, freeing each before the next loads.

    A model that fails to load is recorded as an error and the next one still runs. A model that gave up XLA is
    recorded in ``stats["xla_fallbacks"]`` with the reason; its runs still count as processed.
    """
    import tensorflow as tf

    stats = new_stats() if stats is None else stats
    _enable_memory_growth(tf)
    for model in models:
        if logger:
            logger.info(f"Loading model: {model.feature}")
        stages = None
        try:
            try:
                stages = load_stages(model, settings, logger)
            except Exception as e:
                message = f"Error loading model '{model.feature}' from {model.weights}: {e}"
                if logger:
                    logger.exception(message)
                stats["errors"].append(message)
                continue
            segment_runs(
                runs,
                feature=model.feature,
                object_name=model.object,
                stages=stages,
                settings=settings,
                stats=stats,
                logger=logger,
            )
        finally:
            if stages is not None and stages.xla is not None and stages.xla.fallback:
                stats.setdefault("xla_fallbacks", []).append({"feature": model.feature, "reason": stages.xla.fallback})
            # Free the model's GPU memory after every attempt, a failed load included.
            stages = None
            tf.keras.backend.clear_session()
            gc.collect()
    return stats


def add_object_definitions(root: "CopickRoot", object_names: Sequence[str], logger=None) -> bool:
    """Add a segmentation object for every name the project lacks; returns whether the config changed."""
    modified = False
    for name in dict.fromkeys(object_names):
        if root.get_object(name) is None:
            if logger:
                logger.info(f"Adding object definition for '{name}'")
            root.new_object(
                name=name,
                is_particle=False,  # Segmentation, not particle picks
                # label and color will be auto-assigned
            )
            modified = True
    return modified


def run_easymode_inference(
    root: "CopickRoot",
    run_names: List[str],
    tomo_type: str,
    voxel_size: float,
    models: List[str],
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
    model_dir: Optional[str] = None,
    xla: bool = False,
) -> dict:
    """
    Run easymode inference on copick tomograms in this process.

    Reads and writes overlap with inference (see the module docstring). For one worker process per GPU, use
    :func:`copick_easymode.core.dispatch.dispatch_easymode_inference`.

    Args:
        root: CopickRoot instance with loaded project.
        run_names: List of run names to process. Empty list means all runs.
        tomo_type: Tomogram type (e.g., 'wbp', 'sirt').
        voxel_size: Voxel size in Angstroms.
        models: List of easymode model names to run.
        user_id: User ID for created segmentations (None: the project's), stored as copick sanitizes it.
        session_id: Session ID for created segmentations, stored as copick sanitizes it.
        tta: Test-time augmentation level (1-16; an .scnm model uses at most 8).
        batch_size: Batch size for inference.
        threshold: Probability threshold for binarizing segmentation (0.0-1.0).
        gpus: Comma-separated GPU IDs (e.g., '0,1'), set as this process's CUDA_VISIBLE_DEVICES before
            TensorFlow starts. None for auto-detect. This process computes on one GPU.
        add_objects: Whether to add object definitions if missing.
        overwrite: Whether to overwrite existing segmentations.
        config_path: Path to save config if add_objects is True.
        logger: Logger instance for output messages.
        model_dir: easymode model directory to use instead of easymode's setting (in memory only; default:
            ``$COPICK_EASYMODE_MODEL_DIR``, else the setting).
        xla: Run each model as one XLA-compiled function (see :mod:`copick_easymode.core.xla`).

    Returns:
        Dictionary with processing statistics: processed, skipped, errors, items (one record per run and
        model, with its stage timings) and xla_fallbacks (each model that gave up XLA, with the reason). A model
        that is not found, whose object cannot be added or that fails to load is recorded as an error and the next
        model still runs.
    """
    stats = new_stats()
    settings = copick_settings(
        root,
        tomo_type=tomo_type,
        voxel_size=voxel_size,
        user_id=user_id,
        session_id=session_id,
        tta=tta,
        batch_size=batch_size,
        threshold=threshold,
        overwrite=overwrite,
        xla=xla,
        logger=logger,
    )

    # Configure GPUs before TensorFlow initializes
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Get runs to process
    runs, unknown = select_runs(root, run_names)
    for name in unknown:
        message = f"Run '{name}' not found in the project"
        if logger:
            logger.warning(message)
        stats["errors"].append(message)

    if not runs:
        if logger:
            logger.warning("No runs found to process.")
        return stats

    distribution = easymode_env.import_easymode(model_dir, logger=logger)

    config_modified = False
    for model_name in models:
        resolved, reason = easymode_env.resolve_model(distribution, model_name, logger)
        if resolved is None:
            error_msg = f"Model '{model_name}' not found ({reason}). Skipping."
            if logger:
                logger.error(error_msg)
            stats["errors"].append(error_msg)
            continue
        # Add object definition if needed
        if add_objects:
            try:
                config_modified |= add_object_definitions(root, [resolved.object], logger)
            except Exception as e:
                error_msg = f"Error adding object '{resolved.object}' for model '{model_name}': {e}. Skipping."
                if logger:
                    logger.exception(error_msg)
                stats["errors"].append(error_msg)
                continue
        segment_with_models(runs, [resolved], settings, stats, logger)

    # Save config if modified
    if config_modified and config_path:
        if logger:
            logger.info(f"Saving updated config to {config_path}")
        root.save_config(config_path)

    return stats
