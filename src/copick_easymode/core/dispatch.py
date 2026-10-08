"""
One inference worker process per GPU for ``copick inference easymode``.

TensorFlow computes on one GPU per process, so the runs are split over one worker process per GPU, the way
``easymode segment`` and ``ais segment`` do it. The parent process

1. resolves the GPUs from the allocation (:mod:`copick_easymode.core.devices`),
2. opens the project, checks the requested runs and, with ``add_objects``, adds the object definitions and
   saves the config (the only process that ever writes it),
3. resolves every model once (:func:`copick_easymode.core.easymode_env.resolve_models`): a missing model fails
   here, before any GPU time is spent,
4. starts one worker per GPU (``multiprocessing`` spawn), each with exactly its own device in
   ``CUDA_VISIBLE_DEVICES`` and its share of the CPU threads in its environment from the start, so before
   TensorFlow initializes. The runs are dealt round-robin over their sorted names. Workers re-open the
   project from the config path, load the resolved weights with easymode kept offline and run the overlapped
   loop of :mod:`copick_easymode.core.inference`. A single worker runs in this process, by the same code.
5. collects every worker's outcome and writes the report.

The report (``report_path``) is JSON, written when the call starts (``"status": "running"``), when a worker
finishes and at the end (``"complete"`` or ``"failed"``), on every exit path after the arguments were parsed.
"""

import contextlib
import json
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

from copick_easymode.core.devices import (
    ENV_VISIBLE,
    device_label,
    nvidia_smi_devices,
    plan_workers,
    threads_per_worker,
    visible_gpus,
    worker_environment,
)
from copick_easymode.core.inference import InferenceSettings, new_stats, segment_with_models, select_runs

REPORT_TOOL = "copick-easymode"


@dataclass
class WorkerSpec:
    """Everything one worker needs (picklable: it crosses into a spawned process)."""

    index: int
    gpu: Optional[str]
    cpu: bool
    runs: List[str]
    config_path: str
    models: List[dict]
    settings: InferenceSettings
    threads: int
    model_dir: Optional[str] = None
    lock_path: Optional[str] = None
    debug: bool = False
    result_path: Optional[str] = None

    @property
    def label(self) -> str:
        return f"worker {self.index} {device_label(self.gpu, self.cpu)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _worker_entry(spec: WorkerSpec, **updates) -> dict:
    entry = {
        "index": spec.index,
        "gpu": spec.gpu,
        "runs": list(spec.runs),
        "exitcode": None,
        "seconds": None,
        "processed": 0,
        "skipped": 0,
        "errors": [],
        "visible_devices": None,
        "threads": spec.threads,
        "items": [],
    }
    entry.update(updates)
    return entry


def write_report(path, report: dict) -> None:
    """Write the report atomically, so a reader never sees half of it."""
    if not path:
        return
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = f"{path}.{os.getpid()}.partial"
    with open(partial, "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(partial, path)


# ---- the worker ----------------------------------------------------------------------------------------------


class _Prefixed(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{self.extra['label']}] {msg}", kwargs


class _WorkerFormatter(logging.Formatter):
    """Every line of a worker's record, tracebacks included, starts with ``[worker k gpu X]``."""

    def __init__(self, label: str):
        super().__init__("%(message)s")
        self.label = label

    def format(self, record):
        level = "" if record.levelno == logging.INFO else f"{record.levelname}: "
        lines = super().format(record).splitlines() or [""]
        return "\n".join([f"[{self.label}] {level}{lines[0]}"] + [f"[{self.label}] {line}" for line in lines[1:]])


def _bound_tensorflow_threads(tf, threads: int) -> None:
    for setter in (
        tf.config.threading.set_intra_op_parallelism_threads,
        tf.config.threading.set_inter_op_parallelism_threads,
    ):
        # TensorFlow may have started already; the environment variables applied then.
        with contextlib.suppress(RuntimeError):
            setter(int(threads))


def run_worker(spec: WorkerSpec, logger=None, prefixed: bool = False) -> dict:
    """Segment one worker's runs with every model; returns its report entry. Never raises for a run's failure.

    Sets the worker's device and thread bounds in this process's environment first, so call it before
    TensorFlow is imported (a spawned worker always does). Log lines get the worker's ``[worker k gpu X]``
    prefix unless ``logger`` already adds it (``prefixed``).
    """
    environment = worker_environment(spec.gpu, spec.threads, spec.cpu)
    tensorflow_loaded = "tensorflow" in sys.modules
    os.environ.update(environment)
    logger = logger if logger is not None else logging.getLogger(__name__)
    log = logger if prefixed else _Prefixed(logger, {"label": spec.label})
    if tensorflow_loaded and ENV_VISIBLE in environment:
        log.warning(
            f"TensorFlow was imported before this worker set {ENV_VISIBLE}={environment[ENV_VISIBLE]}; "
            "it may use other devices",
        )

    from copick_easymode.core.easymode_env import ResolvedModel, go_offline, import_easymode

    started = time.monotonic()
    visible = os.environ.get(ENV_VISIBLE)
    stats = new_stats()
    log.info(f"Starting: {len(spec.runs)} run(s), {spec.threads} thread(s), {ENV_VISIBLE}={visible!r}")
    try:
        distribution = import_easymode(spec.model_dir, lock_path=spec.lock_path, logger=log)
        go_offline(distribution)
        import copick
        import tensorflow as tf

        _bound_tensorflow_threads(tf, spec.threads)
        root = copick.from_file(spec.config_path)
        runs = []
        for name in spec.runs:
            run = root.get_run(name)
            if run is None:
                message = f"Run '{name}' not found in the project"
                log.error(message)
                stats["errors"].append(message)
            else:
                runs.append(run)
        models = [ResolvedModel(**m) for m in spec.models]
        segment_with_models(runs, models, spec.settings, stats, log)
    except Exception as exc:
        message = f"Worker {spec.index} ({device_label(spec.gpu, spec.cpu)}) failed: {type(exc).__name__}: {exc}"
        log.exception(message)
        stats["errors"].append(message)
    seconds = round(time.monotonic() - started, 1)
    log.info(
        f"Finished in {seconds} s: {stats['processed']} processed, {stats['skipped']} skipped, "
        f"{len(stats['errors'])} error(s)",
    )
    return _worker_entry(
        spec,
        exitcode=0 if not stats["errors"] else 1,
        seconds=seconds,
        processed=stats["processed"],
        skipped=stats["skipped"],
        errors=list(stats["errors"]),
        visible_devices=visible,
        items=stats["items"],
    )


def _worker_process(spec: WorkerSpec) -> None:
    """A spawned worker: log with its prefix, run, leave the result for the parent, exit 1 on any error."""
    level = logging.DEBUG if spec.debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_WorkerFormatter(spec.label))
    # Libraries' own records get the prefix too, until copick reconfigures the root logger on import.
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for noisy in ("absl", "gql", "h5py", "fsspec", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # The worker's own records keep it regardless.
    logger = logging.getLogger(f"{__name__}.worker{spec.index}")
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False
    result = run_worker(spec, logger, prefixed=True)
    write_report(spec.result_path, result)
    sys.exit(0 if not result["errors"] else 1)


@contextlib.contextmanager
def _environment(updates: Dict[str, str]):
    """This process's environment with ``updates`` applied, restored afterwards (a child started inside
    inherits the updated values from its first instruction)."""
    saved = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _collect(spec: WorkerSpec, process, seconds: float) -> dict:
    result = None
    if spec.result_path and os.path.isfile(spec.result_path):
        try:
            with open(spec.result_path) as fh:
                result = json.load(fh)
        except (OSError, ValueError):
            result = None
    code = process.exitcode
    if result is None:
        result = _worker_entry(spec)
        result["errors"].append(
            f"Worker {spec.index} ({device_label(spec.gpu, spec.cpu)}) exited with code {code} before reporting "
            f"(killed or crashed); its runs {', '.join(spec.runs)} were not all segmented",
        )
    elif code != 0 and not result["errors"]:
        result["errors"].append(f"Worker {spec.index} ({device_label(spec.gpu, spec.cpu)}) exited with code {code}")
    result["exitcode"] = code
    result["seconds"] = seconds
    return result


def run_workers(specs: Sequence[WorkerSpec], logger, on_result: Optional[Callable[[dict], None]] = None) -> List[dict]:
    """Run the workers to completion; one runs in this process, several each in a spawned process."""
    if len(specs) == 1:
        result = run_worker(specs[0], logger)
        if on_result:
            on_result(result)
        return [result]

    from multiprocessing.connection import wait

    context = multiprocessing.get_context("spawn")
    scratch = tempfile.mkdtemp(prefix="copick-easymode-workers-")
    launched = {}
    results = {}
    try:
        for spec in specs:
            spec.result_path = os.path.join(scratch, f"worker-{spec.index}.json")
            process = context.Process(target=_worker_process, args=(spec,), name=f"copick-easymode-{spec.index}")
            with _environment(worker_environment(spec.gpu, spec.threads, spec.cpu)):
                process.start()
            launched[process.sentinel] = (spec, process, time.monotonic())
            logger.info(
                f"[{spec.label}] started as process {process.pid}: {len(spec.runs)} run(s) {','.join(spec.runs)}",
            )
        pending = dict(launched)
        while pending:
            for sentinel in wait(list(pending)):
                spec, process, started = pending.pop(sentinel)
                process.join()
                result = _collect(spec, process, round(time.monotonic() - started, 1))
                results[spec.index] = result
                (logger.info if result["exitcode"] == 0 else logger.error)(
                    f"[{spec.label}] exit {result['exitcode']} after {result['seconds']} s: {result['processed']} "
                    f"processed, {result['skipped']} skipped, {len(result['errors'])} error(s)",
                )
                if on_result:
                    on_result(result)
    finally:
        for _spec, process, _started in launched.values():
            if process.is_alive():
                process.terminate()
                process.join(30)
        shutil.rmtree(scratch, ignore_errors=True)
    return [results[spec.index] for spec in specs]


# ---- the parent ----------------------------------------------------------------------------------------------


def dispatch_easymode_inference(
    config_path: str,
    run_names: Sequence[str],
    tomo_type: str,
    voxel_size: float,
    models: Sequence[str],
    user_id: Optional[str],
    session_id: str,
    *,
    tta: int = 4,
    batch_size: int = 1,
    threshold: float = 0.5,
    overwrite: bool = False,
    gpus: Optional[str] = None,
    cpu: bool = False,
    max_workers: Optional[int] = None,
    threads: Optional[int] = None,
    model_dir: Optional[str] = None,
    offline: bool = False,
    add_objects: bool = True,
    report_path: Optional[str] = None,
    debug: bool = False,
    logger=None,
    env: Optional[Dict[str, str]] = None,
    probe=nvidia_smi_devices,
) -> dict:
    """Run easymode inference with one worker process per GPU; returns the report.

    Args:
        config_path: The copick config; workers re-open the project from it.
        run_names: Runs to process; empty means every run of the project.
        tomo_type: Tomogram type (e.g., 'wbp').
        voxel_size: Voxel size in Angstroms.
        models: easymode features to run, one after another.
        user_id: User ID for created segmentations.
        session_id: Session ID for created segmentations.
        tta: Test-time augmentation level (1-16; an .scnm model uses at most 8).
        batch_size: Batch size for inference.
        threshold: Probability threshold for binarizing a segmentation.
        overwrite: Overwrite existing segmentations instead of skipping their runs.
        gpus: GPUs to use (see :func:`copick_easymode.core.devices.visible_gpus`); None means all of the
            allocation.
        cpu: One worker without a GPU.
        max_workers: At most this many workers (GPUs).
        threads: CPU threads to divide among the workers (default: the CPUs this process may run on, at most
            ``SLURM_CPUS_ON_NODE``).
        model_dir: easymode model directory, in memory only (default: ``$COPICK_EASYMODE_MODEL_DIR``, else
            easymode's setting).
        offline: Never contact the model registry; use only weights already in the model directory.
        add_objects: Add missing object definitions to the config (this process only, before any worker).
        report_path: Where to write the JSON report.
        debug: Debug logging in the workers.
        logger: This process's logger.
        env: The environment to resolve devices and threads from (default: this process's).
        probe: Lists GPUs when ``CUDA_VISIBLE_DEVICES`` is not set (default: ``nvidia-smi -L``).

    Returns:
        The report; ``report["status"]`` is ``"complete"`` only when every run was processed or skipped
        without an error, every model was found and every worker exited cleanly.
    """
    from copick_easymode import __version__

    log = logger if logger is not None else logging.getLogger(__name__)
    clock = time.monotonic()
    errors: List[str] = []
    report = {
        "tool": REPORT_TOOL,
        "version": __version__,
        "status": "running",
        "config": os.path.abspath(str(config_path)),
        "tomogram": f"{tomo_type}@{voxel_size}",
        "runs": sorted(dict.fromkeys(run_names)),
        "user_id": user_id,
        "session_id": session_id,
        "tta": tta,
        "threshold": threshold,
        "batch_size": batch_size,
        "overwrite": overwrite,
        "models_requested": list(models),
        "models": [],
        "missing": {},
        "model_directory": None,
        "online": None,
        "writable": None,
        "easymode_version": None,
        "cpu": cpu,
        "devices": [],
        "threads_per_worker": None,
        "workers": [],
        "processed": 0,
        "skipped": 0,
        "errors": errors,
        "started_utc": _utc_now(),
        "finished_utc": None,
        "seconds": None,
    }

    def tally() -> None:
        workers = report["workers"]
        report["processed"] = sum(w.get("processed") or 0 for w in workers)
        report["skipped"] = sum(w.get("skipped") or 0 for w in workers)
        report["errors"] = errors + [e for w in workers for e in w.get("errors") or []]

    write_report(report_path, report)
    try:
        devices = [] if cpu else visible_gpus(gpus, env, probe)
        if max_workers:
            devices = devices[: max(1, int(max_workers))]
        report["devices"] = devices

        import copick

        root = copick.from_file(str(config_path))
        runs, unknown = select_runs(root, run_names)
        for name in unknown:
            message = f"Run '{name}' not found in the project"
            log.error(message)
            errors.append(message)
        names = sorted(r.name for r in runs)
        report["runs"] = names
        if not names:
            log.warning("No runs found to process.")
            return report

        from copick_easymode.core.easymode_env import resolve_models

        resolution = resolve_models(models, model_dir=model_dir, offline=offline, logger=log)
        config_module = sys.modules.get("easymode.core.config")
        report.update(
            model_directory=resolution.model_directory,
            online=resolution.online,
            writable=resolution.writable,
            models=[m.to_dict() for m in resolution.models],
            missing=dict(resolution.missing),
            easymode_version=getattr(config_module, "version", None),
        )
        for m in resolution.models:
            size = f"{m.bytes / 1e6:.0f} MB" if m.bytes is not None else "size unknown"
            log.info(f"Model {m.feature}: {m.tag} ({m.timestamp}), {m.kind}, {m.weights}, {size}")
        if resolution.missing:
            reasons = "; ".join(f"{feature}: {why}" for feature, why in resolution.missing.items())
            message = (
                f"easymode model(s) not available in {resolution.model_directory} (online: {resolution.online}, "
                f"writable: {resolution.writable}): {reasons}"
            )
            log.error(message)
            errors.append(message)
            return report

        if add_objects:
            from copick_easymode.core.inference import add_object_definitions

            if add_object_definitions(root, [m.object for m in resolution.models], log):
                log.info(f"Saving updated config to {config_path}")
                root.save_config(str(config_path))

        if not cpu and not devices:
            log.warning(
                "No GPU found in this allocation (CUDA_VISIBLE_DEVICES unset and nvidia-smi lists none); "
                "running one worker, on whatever device TensorFlow finds",
            )
        plan = plan_workers(names, devices, cpu=cpu, max_workers=max_workers)
        per_worker = threads_per_worker(len(plan), threads, env)
        settings = InferenceSettings(
            tomo_type=tomo_type,
            voxel_size=voxel_size,
            user_id=user_id,
            session_id=session_id,
            tta=tta,
            batch_size=batch_size,
            threshold=threshold,
            overwrite=overwrite,
        )
        specs = [
            WorkerSpec(
                index=index,
                gpu=gpu,
                cpu=cpu,
                runs=shard,
                config_path=os.path.abspath(str(config_path)),
                models=[m.to_dict() for m in resolution.models],
                settings=settings,
                threads=per_worker,
                model_dir=resolution.model_directory,
                debug=debug,
            )
            for index, (gpu, shard) in enumerate(plan)
        ]
        report["threads_per_worker"] = per_worker
        report["workers"] = [_worker_entry(spec) for spec in specs]
        log.info(
            f"{len(names)} run(s) on {len(specs)} worker(s) "
            f"({', '.join(device_label(spec.gpu, spec.cpu) for spec in specs)}), {per_worker} thread(s) each",
        )
        write_report(report_path, report)

        def finished(result: dict) -> None:
            report["workers"][result["index"]] = result
            tally()
            write_report(report_path, report)

        report["workers"] = run_workers(specs, log, on_result=finished)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        log.exception(f"Inference failed: {message}")
        errors.append(message)
    except BaseException:
        errors.append("Interrupted")
        raise
    finally:
        tally()
        clean = all(w.get("exitcode") == 0 for w in report["workers"])
        report["status"] = "complete" if not report["errors"] and not report["missing"] and clean else "failed"
        report["finished_utc"] = _utc_now()
        report["seconds"] = round(time.monotonic() - clock, 1)
        write_report(report_path, report)
    return report
