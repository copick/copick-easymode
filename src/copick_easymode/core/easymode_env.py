"""
easymode's configuration, imported safely by concurrent processes, and its models resolved once.

**The import race.** ``easymode/core/config.py`` runs ``parse_settings()`` at import, and that function
*rewrites* ``~/easymode/settings.txt`` (opening for writing truncates it, then the JSON is dumped). Processes
importing it at the same time race: one that reads inside another's truncate-then-write window gets partial
JSON, the ``except`` branch calls ``parse_settings()`` again without returning its value, so
``config.settings`` is ``None`` and ``easymode.core.distribution`` fails on ``settings["MODEL_DIRECTORY"]``
before any model loads. So every process here imports ``config`` and ``distribution`` under an exclusive
``flock`` (by default beside the settings file, so concurrent jobs of the same user serialize too) and verifies
that the settings are a mapping, reloading if a foreign writer still raced.

**The model directory.** easymode reads its model directory only from ``MODEL_DIRECTORY`` in that settings
file. A ``model_dir`` (``--model-dir`` or ``$COPICK_EASYMODE_MODEL_DIR``) replaces it in memory, between
importing ``config`` and ``distribution`` (which derives ``MODEL_CACHE_DIR`` and ``REGISTRY_CACHE`` from it at
import); the settings file is never edited for it.

**Resolve once, then run offline.** The models are resolved once, before any worker starts, under a lock in the
model directory: easymode downloads what is missing or outdated when the directory is writable and it is
online, and otherwise only finds what is there. Workers then load the resolved weights with easymode kept
offline: online, easymode re-reads the remote registry and rewrites ``registry.json`` in every process, which
races between workers and fails in a directory the user cannot write.
"""

import contextlib
import importlib
import io
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX: no cross-process lock
    fcntl = None

ENV_MODEL_DIR = "COPICK_EASYMODE_MODEL_DIR"
ENV_IMPORT_LOCK = "COPICK_EASYMODE_IMPORT_LOCK"
IMPORT_LOCK_NAME = ".copick-easymode-import.lock"
#: Serializes downloads into one model directory across processes and jobs (a download takes minutes).
FETCH_LOCK_NAME = ".copick-easymode-fetch.lock"
IMPORT_LOCK_TIMEOUT = 900.0
FETCH_LOCK_TIMEOUT = 3600.0


def _log(logger, level: str, message: str) -> None:
    if logger is not None:
        getattr(logger, level)(message)


def default_lock_path() -> Path:
    """The import lock: ``$COPICK_EASYMODE_IMPORT_LOCK``, else beside ``~/easymode/settings.txt``."""
    configured = (os.environ.get(ENV_IMPORT_LOCK) or "").strip()
    if configured:
        return Path(configured)
    return Path(os.path.expanduser("~")) / "easymode" / IMPORT_LOCK_NAME


@contextlib.contextmanager
def file_lock(path, timeout: float, logger=None):
    """Hold an exclusive ``flock`` on ``path`` (created if needed); yields whether the lock is held.

    The lock is polled so a wedged peer surfaces as a ``TimeoutError`` rather than a silent hang. A lock file
    that cannot be created, or a filesystem without ``flock``, degrades to running unlocked, with a warning.
    """
    path = Path(path)
    if fcntl is None:
        yield False
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+")  # noqa: SIM115 - held open for as long as the lock is
    except OSError as exc:
        _log(logger, "warning", f"Cannot create the lock file {path} ({exc}); continuing without it")
        yield False
        return
    locked = False
    try:
        started = time.monotonic()
        while not locked:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(
                        f"Could not lock {path} within {timeout:g} s; another process holds it",
                    ) from None
                time.sleep(0.2)
            except OSError as exc:  # e.g. ENOLCK on a filesystem without locking
                _log(logger, "warning", f"Cannot lock {path} ({exc}); continuing without it")
                break
        waited = time.monotonic() - started
        if locked and waited > 0.5:
            _log(logger, "info", f"Waited {waited:.1f} s for {path}")
        yield locked
    finally:
        if locked:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _settings_ok(config) -> bool:
    settings = getattr(config, "settings", None)
    return isinstance(settings, dict) and bool(settings.get("MODEL_DIRECTORY"))


def _same_dir(a, b) -> bool:
    return os.path.abspath(str(a)) == os.path.abspath(str(b))


def _normalized(model_dir) -> Optional[str]:
    model_dir = model_dir or os.environ.get(ENV_MODEL_DIR) or None
    return os.path.abspath(os.path.expanduser(str(model_dir))) if model_dir else None


def import_easymode(model_dir=None, *, lock_path=None, logger=None, attempts: int = 5, pause: float = 0.3):
    """Import ``easymode.core.config`` and ``easymode.core.distribution`` safely; returns ``distribution``.

    Args:
        model_dir: easymode model directory to use instead of the settings file's ``MODEL_DIRECTORY`` (default:
            ``$COPICK_EASYMODE_MODEL_DIR``, else the setting). Applied in memory only.
        lock_path: The import lock (default: :func:`default_lock_path`).
        logger: Receives what happened, if given.
        attempts: How often a settings object broken by a foreign writer is reloaded before giving up.
        pause: Seconds between those reloads.
    """
    model_dir = _normalized(model_dir)
    config = sys.modules.get("easymode.core.config")
    distribution = sys.modules.get("easymode.core.distribution")
    if (
        distribution is not None
        and _settings_ok(config)
        and (model_dir is None or _same_dir(distribution.MODEL_CACHE_DIR, model_dir))
    ):
        return distribution

    with file_lock(lock_path or default_lock_path(), IMPORT_LOCK_TIMEOUT, logger):
        if model_dir:
            try:
                os.makedirs(model_dir, exist_ok=True)
            except OSError as exc:  # a read-only directory must already hold the models; resolution says if not
                _log(logger, "warning", f"Cannot create the model directory {model_dir}: {exc}")
        config = importlib.import_module("easymode.core.config")
        for attempt in range(1, attempts + 1):
            if _settings_ok(config):
                if model_dir:
                    config.settings["MODEL_DIRECTORY"] = model_dir
                distribution = sys.modules.get("easymode.core.distribution")
                if distribution is None:
                    distribution = importlib.import_module("easymode.core.distribution")
                elif model_dir and not _same_dir(distribution.MODEL_CACHE_DIR, model_dir):
                    distribution = importlib.reload(distribution)
                if model_dir and not _same_dir(distribution.MODEL_CACHE_DIR, model_dir):
                    raise RuntimeError(
                        f"easymode resolved its model directory to {distribution.MODEL_CACHE_DIR}, not {model_dir}; "
                        "this easymode reads it some other way",
                    )
                return distribution
            _log(
                logger,
                "warning",
                f"easymode settings not loaded (settings={getattr(config, 'settings', None)!r}); "
                f"reloading (attempt {attempt}/{attempts})",
            )
            time.sleep(pause)
            config = importlib.reload(config)
    raise RuntimeError(
        "easymode.core.config.settings never became a mapping with MODEL_DIRECTORY; the settings file "
        f"{getattr(config, 'settings_path', '?')} is being rewritten by another process",
    )


def go_offline(distribution) -> None:
    """Make easymode answer from the model directory alone: no registry fetch, no download, no write."""
    if not hasattr(distribution, "_online"):
        raise RuntimeError(
            "easymode.core.distribution has no _online flag, so it cannot be kept off the network; "
            "this copick-easymode supports easymode 1.x",
        )
    distribution._online = False


@dataclass
class ResolvedModel:
    """One easymode feature, resolved to a weights file before any worker starts."""

    feature: str
    object: str
    weights: str
    kind: str  # "h5" (easymode's own 3D networks) or "scnm" (Ais 2D-engine models)
    tag: Optional[str] = None
    timestamp: Optional[str] = None
    bytes: Optional[int] = None
    #: The voxel size an .h5 model runs at (its metadata, else 10 A); an .scnm model carries its own.
    apix: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelResolution:
    """What resolving the requested features found, and where."""

    model_directory: Optional[str] = None
    online: Optional[bool] = None
    writable: Optional[bool] = None
    models: List[ResolvedModel] = field(default_factory=list)
    #: Feature -> why it could not be resolved.
    missing: Dict[str, str] = field(default_factory=dict)


class _Tee(io.TextIOBase):
    """Passes what easymode prints straight through to ``target`` and keeps its lines.

    It never logs: a log handler writing to ``sys.stdout`` would write back into it.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.lines: List[str] = []
        self._partial = ""

    def writable(self):
        return True

    def write(self, text):
        if self.target is not None:
            self.target.write(text)
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        self.lines.extend(line.strip() for line in lines if line.strip())
        return len(text)

    def flush(self):
        if self.target is not None:
            self.target.flush()


def resolve_model(distribution, feature: str, logger=None):
    """Ask easymode for one feature's weights; returns ``(ResolvedModel, None)`` or ``(None, reason)``.

    easymode explains a miss only by printing it, so what it prints is kept to name the reason.
    """
    from copick_easymode.core.models import copick_name
    from copick_easymode.core.scnm import is_scnm

    tee = _Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            weights, metadata = distribution.get_model(feature)
    except Exception as exc:  # reported per model; the caller decides what a miss means
        return None, f"{type(exc).__name__}: {exc}"
    if not weights:
        return None, tee.lines[-1] if tee.lines else f"easymode has no model for '{feature}'"
    if not os.path.isfile(weights):
        return None, f"easymode returned {weights}, which does not exist"
    metadata = metadata or {}
    kind = "scnm" if is_scnm(weights) else "h5"
    return (
        ResolvedModel(
            feature=feature,
            object=copick_name(feature),
            weights=str(weights),
            kind=kind,
            tag=metadata.get("tag"),
            timestamp=None if metadata.get("timestamp") is None else str(metadata.get("timestamp")),
            bytes=os.path.getsize(weights),
            apix=float(metadata.get("apix", 10.0)) if kind == "h5" else metadata.get("apix"),
        ),
        None,
    )


def resolve_models(
    features: Sequence[str],
    *,
    model_dir=None,
    offline: bool = False,
    lock_path=None,
    logger=None,
) -> ModelResolution:
    """Resolve every feature once, before inference.

    Downloads what is missing or outdated when the model directory is writable and easymode is online (under a
    lock in that directory, so concurrent jobs fetch once); otherwise, or with ``offline``, only finds what is
    already there. Never raises for a missing model: the result names it in ``missing``.
    """
    distribution = import_easymode(model_dir, lock_path=lock_path, logger=logger)
    directory = str(distribution.MODEL_CACHE_DIR)
    registry = getattr(distribution, "REGISTRY_CACHE", os.path.join(directory, "registry.json"))
    writable = (
        os.path.isdir(directory)
        and os.access(directory, os.W_OK)
        and (not os.path.exists(registry) or os.access(registry, os.W_OK))
    )
    if offline or not writable:
        go_offline(distribution)  # online, easymode would rewrite registry.json there
    online = bool(distribution.is_online())
    result = ModelResolution(model_directory=directory, online=online, writable=writable)

    fetching = online and writable
    lock = file_lock(Path(directory) / FETCH_LOCK_NAME, FETCH_LOCK_TIMEOUT, logger) if fetching else None
    previous_umask = os.umask(0o002) if fetching else None  # a group-shared directory stays updatable
    try:
        with lock if lock is not None else contextlib.nullcontext():
            for feature in features:
                model, reason = resolve_model(distribution, feature, logger)
                if model is None:
                    result.missing[feature] = reason
                else:
                    result.models.append(model)
    finally:
        if previous_umask is not None:
            os.umask(previous_umask)
    return result
