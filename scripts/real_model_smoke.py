"""Run one opt-in, checksum-recorded easymode inference through public copick APIs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from copick.util.escape import sanitize_name
from copick.util.ome import get_level_path
from ome_zarr_models.v05.image import Image

from copick_easymode.core.easymode_adapter import EASYMODE_COMMIT, EasymodeRuntime, load_runtime
from copick_easymode.core.inference import run_easymode_inference, segment_tomogram_from_array


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


def discover_huggingface_revision(model_path: Path) -> str | None:
    """Return a nearby Hugging Face snapshot revision when the cache exposes one."""
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


def parse_tomogram(value: str) -> tuple[str, float]:
    try:
        tomo_type, voxel_size = value.split("@", 1)
        parsed_voxel_size = float(voxel_size)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tomogram must use type@voxel_size") from error
    if not tomo_type or parsed_voxel_size <= 0:
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


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
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
    entity_name = sanitize_name(args.model, suppress_warnings=True)
    user_id = sanitize_name(args.user_id, suppress_warnings=True)
    session_id = sanitize_name(args.session_id, suppress_warnings=True)
    if root.get_object(entity_name) is None and not args.add_object:
        raise ValueError(f"Object {entity_name!r} is absent; add it to the config or pass --add-object")

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    runtime = load_runtime()
    model_path_value, metadata = runtime.get_model(args.model)
    if model_path_value is None:
        raise FileNotFoundError(f"Model {args.model!r} could not be resolved")
    model_path = Path(model_path_value).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = runtime.load_model(str(model_path))

    volume = tomogram.numpy()
    input_shape = volume.shape
    input_dtype = str(volume.dtype)
    input_sha256 = sha256_array(volume)
    started = time.perf_counter()
    probabilities = segment_tomogram_from_array(
        model=model,
        volume=volume,
        input_apix=voxel_size,
        model_apix=(metadata or {}).get("apix", 10.0),
        tta=1,
        batch_size=1,
    )
    inference_seconds = time.perf_counter() - started
    if probabilities.size == 0 or not np.isfinite(probabilities).all():
        raise ValueError("Real-model probability map must be nonempty and finite")
    if not np.any(probabilities >= args.threshold):
        raise ValueError("Real-model smoke would produce an empty segmentation")
    del volume

    cached_runtime = EasymodeRuntime(
        tensorflow=runtime.tensorflow,
        get_model=lambda name: (str(model_path), metadata),
        load_model=lambda path: model,
    )
    stats = run_easymode_inference(
        root=root,
        run_names=[args.run],
        tomo_type=tomo_type,
        voxel_size=voxel_size,
        models=[args.model],
        user_id=args.user_id,
        session_id=args.session_id,
        tta=1,
        batch_size=1,
        threshold=args.threshold,
        gpus=None,
        add_objects=args.add_object,
        overwrite=args.overwrite,
        config_path=str(args.config) if args.add_object else None,
        runtime=cached_runtime,
        segmenter=lambda **kwargs: probabilities,
    )
    if stats["processed"] != 1 or stats["errors"]:
        raise RuntimeError(f"Smoke output was not saved successfully: {stats!r}")

    segmentation = run.get_segmentations(
        name=entity_name,
        user_id=user_id,
        session_id=session_id,
        voxel_size=voxel_size,
        is_multilabel=False,
    )[0]
    saved = segmentation.numpy()
    expected = (probabilities >= args.threshold).astype(np.uint8)
    np.testing.assert_array_equal(saved, expected)
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
    input_zarr_format = input_group.metadata.zarr_format
    if input_zarr_format == 3:
        input_ome_zarr_version = input_group.attrs.get("ome", {}).get("version")
    else:
        input_multiscales = input_group.attrs.get("multiscales", [{}])
        input_ome_zarr_version = input_multiscales[0].get("version") if input_multiscales else None

    devices = [
        {"name": device.name, "type": device.device_type}
        for device in runtime.tensorflow.config.list_physical_devices()
    ]
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "easymode_source_commit": EASYMODE_COMMIT,
        "packages": package_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": devices,
        "model": args.model,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "model_metadata": metadata,
        "huggingface_revision": discover_huggingface_revision(model_path),
        "tomogram": f"{tomo_type}@{voxel_size}",
        "backend": root.config.config_type,
        "input_zarr_format": input_zarr_format,
        "input_ome_zarr_version": input_ome_zarr_version,
        "input_shape": input_shape,
        "input_dtype": input_dtype,
        "input_sha256": input_sha256,
        "probability_shape": probabilities.shape,
        "probability_dtype": str(probabilities.dtype),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_sha256": sha256_array(probabilities),
        "inference_seconds": inference_seconds,
        "threshold": args.threshold,
        "segmentation_entity": f"{entity_name}:{user_id}/{session_id}@{voxel_size}",
        "output_zarr_format": output_group.metadata.zarr_format,
        "output_ome_zarr_version": output_ome_zarr_version,
        "foreground_voxels": foreground_voxels,
        "segmentation_sha256": sha256_array(saved),
        "stats": stats,
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
