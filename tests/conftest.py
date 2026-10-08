"""Small builders shared by the tests: a filesystem copick project, tiny Keras models, a model directory."""

import json
import tarfile

import numpy as np
import pytest


def make_project(tmp_path, runs, voxel_size=10.0, tomo_type="wbp", objects=()):
    """A filesystem copick project with one random tomogram per run; ``runs`` maps name -> (Z, Y, X)."""
    import copick

    config = tmp_path / "copick_config.json"
    config.write_text(
        json.dumps(
            {
                "config_type": "filesystem",
                "name": "test",
                "version": "1.0.0",
                "pickable_objects": [
                    {"name": name, "is_particle": False, "label": i + 1} for i, name in enumerate(objects)
                ],
                "overlay_root": f"local://{tmp_path / 'overlay'}",
                "overlay_fs_args": {"auto_mkdir": True},
            },
        ),
    )
    root = copick.from_file(str(config))
    rng = np.random.default_rng(7)
    for name, shape in runs.items():
        run = root.new_run(name)
        tomo = run.new_voxel_spacing(voxel_size).new_tomogram(tomo_type)
        tomo.from_numpy(rng.normal(size=shape).astype(np.float32), levels=1)
    return config, root


def make_scnm(directory, title="tiny", meta=None):
    """A tiny 2.5D Ais slice model (``ezm-2d``) packed as an ``.scnm``."""
    keras = pytest.importorskip("keras")
    directory.mkdir(parents=True, exist_ok=True)
    inp = keras.Input(shape=(None, None, 4), batch_size=1)
    x = keras.layers.Conv2D(2, 3, padding="same")(inp)
    x = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    weights = directory / f"{title}_weights.h5"
    keras.Model(inp, x).save(weights)
    metadata = directory / f"{title}_metadata.json"
    metadata.write_text(
        json.dumps({"title": title, "apix": 10.0, "model_depth": 4, "normalization": "global_mad", **(meta or {})}),
    )
    path = directory / f"{title}.scnm"
    with tarfile.open(path, "w") as archive:
        archive.add(weights, arcname=weights.name)
        archive.add(metadata, arcname=metadata.name)
    weights.unlink()
    metadata.unlink()
    return path


def make_model_dir(tmp_path, feature="tiny", tag="t1", timestamp="20260101000000"):
    """An easymode model directory holding one ``.scnm`` feature, as easymode's cache lays it out."""
    model_dir = tmp_path / "easymode-models"
    weights = make_scnm(model_dir / "models", title=f"{feature}_{tag}")
    (model_dir / "models" / f"{feature}_{tag}.json").write_text(
        json.dumps({"timestamp": timestamp, "feature": feature, "tag": tag}),
    )
    (model_dir / "registry.json").write_text(
        json.dumps(
            {feature: {"default": tag, "models": {tag: {"weights": f"models/{weights.name}", "timestamp": timestamp}}}},
        ),
    )
    return model_dir


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """A private HOME (easymode writes ~/easymode/settings.txt) and import lock; the device and thread
    variables a worker sets on itself are restored afterwards."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COPICK_EASYMODE_IMPORT_LOCK", str(tmp_path / "import.lock"))
    monkeypatch.delenv("COPICK_EASYMODE_MODEL_DIR", raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    for var in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    return home
