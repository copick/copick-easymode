"""Run one opt-in, checksum-recorded easymode inference through the production path and public copick APIs.

The model is resolved by the easymode boundary, loaded by ``load_stages`` and run over the run by the same
overlapped per-run loop (``segment_runs``) the command uses, so the record describes what
``copick inference easymode`` does: copick reads the tomogram, the model segments it, copick writes the result.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import zarr
from copick.util.ome import get_level_path
from ome_zarr_models.v05.image import Image

from copick_easymode.core import easymode_env
from copick_easymode.core.devices import ENV_VISIBLE, NO_DEVICE
from copick_easymode.core.inference import (
    Stages,
    add_object_definitions,
    copick_identity,
    copick_settings,
    load_stages,
    new_stats,
    segment_runs,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def discover_huggingface_revision(model_path: Path) -> Optional[str]:
    """Return the Hugging Face revision a cached model file came from, when the cache records one.

    easymode 1.x downloads into its model directory with ``hf_hub_download(local_dir=...)``, which records the
    commit as the first line of ``<local_dir>/.cache/huggingface/download/<file>.metadata``. A ``cache_dir``
    download keeps it as ``snapshots/<revision>/`` or in ``refs/main`` instead.
    """
    for parent in model_path.parents:
        relative = model_path.relative_to(parent)
        metadata = parent / ".cache" / "huggingface" / "download" / relative.parent / f"{relative.name}.metadata"
        if metadata.is_file():
            lines = metadata.read_text(encoding="utf-8").splitlines()
            if lines and lines[0].strip():
                return lines[0].strip()
    for candidate in model_path.parent.rglob(model_path.name):
        parts = candidate.parts
        if "snapshots" in parts:
            index = parts.index("snapshots")
            if index + 1 < len(parts):
                return parts[index + 1]
    for reference in model_path.parent.glob("models--*/refs/main"):
        revision = reference.read_text(encoding="utf-8").strip()
        if revision:
            return revision
    return None


def package_versions() -> dict[str, str]:
    packages = ("copick-easymode", "copick", "easymode", "tensorflow", "keras", "numpy", "scipy", "zarr")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def installed_easymode_commit() -> Optional[str]:
    """The source commit the installed easymode was built from, when pip recorded one (a git install)."""
    try:
        direct_url = importlib.metadata.distribution("easymode").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
    except ValueError:
        return None


def parse_tomogram(value: str) -> tuple[str, float]:
    try:
        tomo_type, voxel_size = value.split("@", 1)
        parsed_voxel_size = float(voxel_size)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tomogram must use type@voxel_size") from error
    if not tomo_type or not np.isfinite(parsed_voxel_size) or parsed_voxel_size <= 0:
        raise argparse.ArgumentTypeError("tomogram type must be nonempty and voxel size must be positive")
    return tomo_type, parsed_voxel_size


def _json_value(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _ome_zarr_version(group: zarr.Group) -> Optional[str]:
    if group.metadata.zarr_format == 3:
        return group.attrs.get("ome", {}).get("version")
    multiscales = group.attrs.get("multiscales", [{}])
    return multiscales[0].get("version") if multiscales else None


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.cpu_only:
        os.environ[ENV_VISIBLE] = NO_DEVICE  # before TensorFlow initializes

    import copick

    root = copick.from_file(str(args.config))
    run = root.get_run(args.run)
    if run is None:
        raise ValueError(f"Run {args.run!r} was not found")
    tomo_type, voxel_size = args.tomogram
    voxel_spacing = run.get_voxel_spacing(voxel_size)
    if voxel_spacing is None:
        raise ValueError(f"Voxel spacing {voxel_size} was not found in run {args.run!r}")
    tomograms = voxel_spacing.get_tomograms(tomo_type=tomo_type)
    if not tomograms:
        raise ValueError(f"Tomogram {tomo_type}@{voxel_size} was not found in run {args.run!r}")
    tomogram = tomograms[0]
    settings = copick_settings(
        root,
        tomo_type=tomo_type,
        voxel_size=voxel_size,
        user_id=args.user_id,
        session_id=args.session_id,
        tta=1,
        batch_size=1,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )
    entity_name = copick_identity(args.model, "object name")
    if root.get_object(entity_name) is None and not args.add_object:
        raise ValueError(f"Object {entity_name!r} is absent; add it to the config or pass --add-object")

    resolution = easymode_env.resolve_models([args.model], model_dir=args.model_dir, offline=args.offline)
    if resolution.missing:
        raise FileNotFoundError(f"Model {args.model!r} could not be resolved: {resolution.missing[args.model]}")
    [model] = resolution.models
    if args.add_object and add_object_definitions(root, [model.object]):
        root.save_config(str(args.config))

    import tensorflow as tf

    stages = load_stages(model, settings)
    evidence: dict[str, Any] = {}

    def prepare(volume):
        evidence.update(input_shape=volume.shape, input_dtype=str(volume.dtype), input_sha256=sha256_array(volume))
        return stages.prepare(volume)

    def finish(raw, prepared):
        probabilities = stages.finish(raw, prepared)
        if probabilities.size == 0 or not np.isfinite(probabilities).all():
            raise ValueError("Real-model probability map must be nonempty and finite")
        if not np.any(probabilities >= settings.threshold):
            raise ValueError("Real-model smoke would produce an empty segmentation")
        evidence["probabilities"] = probabilities
        return probabilities

    stats = segment_runs(
        [run],
        feature=model.feature,
        object_name=model.object,
        stages=Stages(prepare=prepare, infer=stages.infer, finish=finish),
        settings=settings,
        stats=new_stats(),
    )
    if stats["processed"] != 1 or stats["errors"]:
        raise RuntimeError(f"Smoke output was not saved successfully: {stats!r}")
    [item] = stats["items"]

    segmentation = run.get_segmentations(
        name=model.object,
        user_id=settings.user_id,
        session_id=settings.session_id,
        voxel_size=voxel_size,
        is_multilabel=False,
    )[0]
    saved = segmentation.numpy()
    probabilities = evidence.pop("probabilities")
    np.testing.assert_array_equal(saved, (probabilities >= settings.threshold).astype(np.uint8))
    foreground_voxels = int(np.count_nonzero(saved))
    if foreground_voxels == 0:
        raise ValueError("Real-model smoke produced an empty segmentation")
    output_group = zarr.open_group(segmentation.zarr(), mode="r")
    output_ome_zarr_version = Image.from_zarr(output_group).ome_zarr_version
    if output_group.metadata.zarr_format != 3 or output_ome_zarr_version != "0.5":
        raise ValueError("Saved segmentation is not OME-Zarr 0.5 / Zarr v3")
    if output_group[get_level_path(output_group, 0)].dtype != np.dtype(np.uint8):
        raise ValueError("Saved segmentation is not uint8")
    input_group = zarr.open_group(tomogram.zarr(), mode="r")

    weights = Path(model.weights)
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "easymode_version": easymode_env.easymode_version(),
        "easymode_source_commit": installed_easymode_commit(),
        "easymode_supported_commit": easymode_env.EASYMODE_COMMIT,
        "packages": package_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [{"name": device.name, "type": device.device_type} for device in tf.config.list_physical_devices()],
        "model": args.model,
        "model_kind": model.kind,
        "model_tag": model.tag,
        "model_timestamp": model.timestamp,
        "model_apix": model.apix,
        "model_file": weights.name,
        "model_sha256": sha256_file(weights),
        "huggingface_revision": discover_huggingface_revision(weights),
        "model_directory_online": resolution.online,
        "tomogram": f"{tomo_type}@{voxel_size}",
        "backend": root.config.config_type,
        "input_zarr_format": input_group.metadata.zarr_format,
        "input_ome_zarr_version": _ome_zarr_version(input_group),
        **evidence,
        "probability_shape": probabilities.shape,
        "probability_dtype": str(probabilities.dtype),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_sha256": sha256_array(probabilities),
        "stage_seconds": item["seconds"],
        "threshold": settings.threshold,
        "segmentation_entity": f"{model.object}:{settings.user_id}/{settings.session_id}@{voxel_size}",
        "output_zarr_format": output_group.metadata.zarr_format,
        "output_ome_zarr_version": output_ome_zarr_version,
        "foreground_voxels": foreground_voxels,
        "segmentation_sha256": sha256_array(saved),
        "stats": {key: stats[key] for key in ("processed", "skipped", "errors")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tomogram", required=True, type=parse_tomogram)
    parser.add_argument("--model", default="ribosome")
    parser.add_argument("--user-id", default="copick-easymode-smoke")
    parser.add_argument("--session-id", default="1")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-dir", default=None, help="easymode model directory (default: easymode's own)")
    parser.add_argument("--offline", action="store_true", help="use only the models already in the model directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--add-object", action="store_true")
    parser.add_argument("--allow-gpu", action="store_false", dest="cpu_only")
    parser.set_defaults(cpu_only=True)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"configuration does not exist: {args.config}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be between 0.0 and 1.0")

    record = run_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, default=_json_value) + "\n", encoding="utf-8")
    print(f"Validated real-model inference; evidence written to {args.output}")


if __name__ == "__main__":
    main()
