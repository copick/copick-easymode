"""What inference writes, skips and keeps under copick's identities, and how a failing model or run is contained.

None of this needs easymode: the per-run loop takes explicit stages, and the model loop and the dispatcher get
their models and stages from a fake easymode boundary.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from conftest import make_project

from copick_easymode.core import easymode_env, inference
from copick_easymode.core.dispatch import dispatch_easymode_inference
from copick_easymode.core.easymode_env import ModelResolution, ResolvedModel
from copick_easymode.core.inference import (
    InferenceSettings,
    Stages,
    copick_settings,
    new_stats,
    run_easymode_inference,
    segment_runs,
)

RUNS = {"r1": (6, 20, 24), "r2": (5, 24, 20)}


def _probabilities(volume):
    """0.49, 0.5, 0.51 repeated: the threshold keeps exactly the values at or above it."""
    return np.resize(np.array([0.49, 0.5, 0.51], dtype=np.float32), volume.shape)


def _stages(seen=None, infer=None):
    def prepare(volume):
        if seen is not None:
            seen.append(volume.copy())
        return volume

    return Stages(prepare=prepare, infer=infer or _probabilities, finish=lambda raw, prepared: raw)


def _settings(**updates):
    values = {"tomo_type": "wbp", "voxel_size": 10.0, "user_id": "tester", "session_id": "1", "tta": 1}
    values.update(updates)
    return InferenceSettings(**values)


def _segment(root, stages, object_name="ribosome", **settings):
    return segment_runs(
        root.runs,
        feature=object_name,
        object_name=object_name,
        stages=stages,
        settings=_settings(**settings),
        stats=new_stats(),
    )


def _outcome(stats):
    return stats["processed"], stats["skipped"], stats["errors"]


def _segmentations(run, name="ribosome", user_id="tester", session_id="1"):
    return run.get_segmentations(name=name, user_id=user_id, session_id=session_id, voxel_size=10.0)


def _tomogram(run):
    return run.get_voxel_spacing(10.0).get_tomograms(tomo_type="wbp")[0]


def _existing(run, values=1, name="ribosome", user_id="tester", session_id="1"):
    shape = _tomogram(run).numpy().shape
    seg = run.new_segmentation(voxel_size=10.0, name=name, session_id=session_id, is_multilabel=False, user_id=user_id)
    seg.from_numpy(np.full(shape, values, dtype=np.uint8))
    return seg


def _never(what):
    return Mock(side_effect=AssertionError(f"{what} must not happen"))


# ---- identities ----------------------------------------------------------------------------------------------


def test_settings_hold_the_ids_copick_stores():
    settings = _settings(user_id="my_user", session_id="session one")
    assert (settings.user_id, settings.session_id) == ("my-user", "session-one")
    with pytest.raises(ValueError, match="Invalid copick user ID '___'"):
        _settings(user_id="___")
    with pytest.raises(ValueError, match="Invalid copick session ID ''"):
        _settings(session_id="")


def test_a_missing_user_id_is_the_projects(tmp_path):
    _config, root = make_project(tmp_path, RUNS)
    root.config.user_id = None
    with pytest.raises(ValueError, match="No user ID"):
        copick_settings(root, user_id=None, session_id="1", tomo_type="wbp", voxel_size=10.0)

    root.config.user_id = "project_user"
    logger = Mock()
    settings = copick_settings(root, user_id=None, session_id="1", tomo_type="wbp", voxel_size=10.0, logger=logger)
    assert settings.user_id == "project-user"
    logger.info.assert_called_once_with("Using copick user ID 'project-user' for the requested 'project_user'")


def test_sanitized_ids_find_existing_output_before_anything_is_read(tmp_path, monkeypatch):
    _config, root = make_project(tmp_path, RUNS, objects=["ribosome"])
    for run in root.runs:
        _existing(run, user_id="my-user", session_id="session-one")
        monkeypatch.setattr(_tomogram(run), "numpy", _never("reading the tomogram"))

    stats = _segment(
        root,
        Stages(_never("prepare"), _never("infer"), _never("finish")),
        user_id="my_user",
        session_id="session_one",
    )

    assert _outcome(stats) == (0, 2, [])


def test_a_raw_object_name_is_written_as_copick_stores_it(tmp_path):
    _config, root = make_project(tmp_path, RUNS, objects=["nuclear-envelope"])

    stats = _segment(root, _stages(), object_name="nuclear_envelope")

    assert _outcome(stats) == (2, 0, [])
    assert all(len(_segmentations(run, name="nuclear-envelope")) == 1 for run in root.runs)


# ---- skip, overwrite, keep -----------------------------------------------------------------------------------


def test_the_model_sees_the_stored_tomogram_and_the_threshold_is_inclusive(tmp_path):
    _config, root = make_project(tmp_path, RUNS, objects=["ribosome"])
    seen = []

    stats = _segment(root, _stages(seen), threshold=0.5)

    assert _outcome(stats) == (2, 0, [])
    assert len(seen) == len(root.runs)
    for i, run in enumerate(root.runs):
        stored = _tomogram(run).numpy()
        np.testing.assert_array_equal(seen[i], stored)
        [seg] = _segmentations(run)
        written = seg.numpy()
        assert written.dtype == np.uint8
        np.testing.assert_array_equal(written, (_probabilities(stored) >= 0.5).astype(np.uint8))


def test_without_overwrite_an_existing_segmentation_is_skipped_before_reading(tmp_path, monkeypatch):
    _config, root = make_project(tmp_path, RUNS, objects=["ribosome"])
    for run in root.runs:
        _existing(run)
        monkeypatch.setattr(_tomogram(run), "numpy", _never("reading the tomogram"))

    stats = _segment(root, Stages(_never("prepare"), _never("infer"), _never("finish")))

    assert _outcome(stats) == (0, 2, [])
    assert [item["status"] for item in stats["items"]] == ["skipped", "skipped"]


def test_overwrite_reuses_the_entity_and_replaces_its_data(tmp_path):
    _config, root = make_project(tmp_path, RUNS, objects=["ribosome"])
    before = {run.name: _existing(run).path for run in root.runs}

    stats = _segment(root, _stages(), overwrite=True)

    assert _outcome(stats) == (2, 0, [])
    for run in root.runs:
        [seg] = _segmentations(run)
        assert seg.path == before[run.name]
        expected = (_probabilities(_tomogram(run).numpy()) >= 0.5).astype(np.uint8)
        np.testing.assert_array_equal(seg.numpy(), expected)


def test_a_failed_inference_keeps_the_existing_segmentation(tmp_path):
    _config, root = make_project(tmp_path, RUNS, objects=["ribosome"])
    for run in root.runs:
        _existing(run, values=7)

    stats = _segment(root, _stages(infer=Mock(side_effect=RuntimeError("out of GPU memory"))), overwrite=True)

    assert _outcome(stats) == (
        0,
        0,
        [f"Error processing ribosome in {run.name}: out of GPU memory" for run in root.runs],
    )
    for run in root.runs:
        [seg] = _segmentations(run)
        assert np.all(seg.numpy() == 7)


def test_a_missing_tomogram_type_is_skipped_without_inference(tmp_path):
    _config, root = make_project(tmp_path, RUNS, objects=["ribosome"])

    stats = _segment(root, Stages(_never("prepare"), _never("infer"), _never("finish")), tomo_type="sirt")

    assert _outcome(stats) == (0, 2, [])


def test_a_failing_read_does_not_stop_the_next_run(tmp_path, monkeypatch):
    _config, root = make_project(tmp_path, RUNS, objects=["ribosome"])
    first, second = root.runs
    monkeypatch.setattr(_tomogram(first), "numpy", Mock(side_effect=RuntimeError("read failed")))

    stats = _segment(root, _stages())

    assert _outcome(stats) == (1, 0, ["Error processing ribosome in r1: read failed"])
    assert not _segmentations(first) and len(_segmentations(second)) == 1


# ---- models, in this process ---------------------------------------------------------------------------------


def _fake_distribution(get_model):
    """What the boundary checks on easymode's distribution, with ``get_model`` in charge."""
    return SimpleNamespace(
        MODEL_CACHE_DIR="/models",
        REGISTRY_CACHE="/models/registry.json",
        _online=None,
        get_model=get_model,
        load_model=_never("loading weights outside load_stages"),
        list_remote_models=lambda: [],
        is_online=lambda: False,
    )


@pytest.fixture
def fake_boundary(monkeypatch, tmp_path):
    """easymode's side of inference, faked: the features in ``available`` resolve, a feature in ``failing``
    fails to load, and every other model writes :func:`_probabilities`. ``calls`` records what was asked."""
    import tensorflow as tf

    weights = tmp_path / "weights.h5"
    weights.write_bytes(b"not read")
    fake = SimpleNamespace(
        weights=weights,
        available={"ribosome", "membrane", "actin", "nuclear_envelope"},
        failing=set(),
        asked=[],
        loaded=[],
        released=0,
    )

    def get_model(name, **kwargs):
        fake.asked.append(name)
        return (str(weights), {"apix": 10.0}) if name in fake.available else (None, None)

    def load_stages(model, settings, logger=None):
        fake.loaded.append(model.feature)
        if model.feature in fake.failing:
            raise RuntimeError("weights are corrupt")
        return _stages()

    def clear_session():
        fake.released += 1

    monkeypatch.setattr(easymode_env, "import_easymode", lambda *args, **kwargs: _fake_distribution(get_model))
    monkeypatch.setattr(inference, "load_stages", load_stages)
    monkeypatch.setattr(tf.keras.backend, "clear_session", clear_session)
    return fake


def _run(root, models, **kwargs):
    return run_easymode_inference(root, [], "wbp", 10.0, models, "tester", "1", tta=1, **kwargs)


def test_the_model_name_is_easymodes_and_the_object_name_copicks(tmp_path, monkeypatch, fake_boundary):
    _config, root = make_project(tmp_path, RUNS)
    save_config = Mock()
    monkeypatch.setattr(root, "save_config", save_config)

    stats = _run(root, ["nuclear_envelope"], config_path="config.json")

    assert _outcome(stats) == (2, 0, [])
    assert fake_boundary.asked == ["nuclear_envelope"]
    assert root.get_object("nuclear-envelope") is not None
    assert all(len(_segmentations(run, name="nuclear-envelope")) == 1 for run in root.runs)
    save_config.assert_called_once_with("config.json")


def test_a_missing_model_is_recorded_and_the_next_model_runs(tmp_path, fake_boundary):
    _config, root = make_project(tmp_path, RUNS)

    stats = _run(root, ["missing", "ribosome"])

    assert stats["processed"] == 2
    [error] = stats["errors"]
    assert error.startswith("Model 'missing' not found") and "Skipping" in error
    assert fake_boundary.loaded == ["ribosome"]


def test_an_object_that_cannot_be_added_is_recorded_and_the_next_model_runs(tmp_path, monkeypatch, fake_boundary):
    _config, root = make_project(tmp_path, RUNS)
    original = root.new_object

    def new_object(**kwargs):
        if kwargs["name"] == "actin":
            raise RuntimeError("config is read-only")
        return original(**kwargs)

    save_config = Mock()
    monkeypatch.setattr(root, "new_object", new_object)
    monkeypatch.setattr(root, "save_config", save_config)

    stats = _run(root, ["membrane", "actin", "ribosome"], config_path="config.json")

    assert stats["processed"] == 4
    [error] = stats["errors"]
    assert "actin" in error and "config is read-only" in error
    assert fake_boundary.loaded == ["membrane", "ribosome"]
    save_config.assert_called_once_with("config.json")


def test_a_model_that_fails_to_load_is_released_and_the_next_model_runs(tmp_path, fake_boundary):
    _config, root = make_project(tmp_path, RUNS)
    fake_boundary.failing.add("membrane")

    stats = _run(root, ["membrane", "ribosome"])

    assert stats["processed"] == 2
    [error] = stats["errors"]
    assert error.startswith("Error loading model 'membrane'") and "weights are corrupt" in error
    assert fake_boundary.released == 2  # once per model, the failed load included


def test_the_config_is_saved_once_and_only_when_an_object_was_added(tmp_path, monkeypatch, fake_boundary):
    _config, root = make_project(tmp_path, RUNS)
    save_config = Mock()
    monkeypatch.setattr(root, "save_config", save_config)

    assert _outcome(_run(root, ["membrane"], config_path="config.json")) == (2, 0, [])
    save_config.assert_called_once_with("config.json")

    save_config.reset_mock()
    assert _outcome(_run(root, ["membrane"], config_path="config.json", overwrite=True)) == (2, 0, [])
    save_config.assert_not_called()


# ---- the dispatcher: the parent settles, the workers write ---------------------------------------------------


def _dispatch(config, tmp_path, user_id="my_user", session_id="session_one", **kwargs):
    return dispatch_easymode_inference(
        str(config),
        [],
        "wbp",
        10.0,
        ["nuclear_envelope"],
        user_id,
        session_id,
        tta=1,
        cpu=True,
        report_path=str(tmp_path / "report.json"),
        **kwargs,
    )


@pytest.fixture
def resolved(monkeypatch, fake_boundary, tmp_path):
    """The parent resolves ``nuclear_envelope``; ``resolved.missing`` names features it could not."""
    resolution = ModelResolution(model_directory=str(tmp_path), online=False, writable=True)
    resolution.models.append(
        ResolvedModel(
            feature="nuclear_envelope",
            object="nuclear-envelope",
            weights=str(fake_boundary.weights),
            kind="h5",
            apix=10.0,
        ),
    )
    resolve_models = Mock(return_value=resolution)
    monkeypatch.setattr(easymode_env, "resolve_models", resolve_models)
    return SimpleNamespace(resolution=resolution, resolve_models=resolve_models)


def test_the_parent_settles_copick_ids_and_the_worker_writes_under_them(tmp_path, isolated_env, resolved):
    import copick

    config, _root = make_project(tmp_path, RUNS)

    report = _dispatch(config, tmp_path)

    assert report["status"] == "complete", report["errors"]
    assert (report["user_id"], report["session_id"]) == ("my-user", "session-one")
    assert [o["name"] for o in json.loads(config.read_text())["pickable_objects"]] == ["nuclear-envelope"]
    root = copick.from_file(str(config))
    for run in root.runs:
        assert len(_segmentations(run, name="nuclear-envelope", user_id="my-user", session_id="session-one")) == 1

    # A second call finds every segmentation under the same IDs and reads nothing.
    again = _dispatch(config, tmp_path)
    assert (again["status"], again["processed"], again["skipped"]) == ("complete", 0, 2)


def test_an_id_copick_cannot_store_fails_before_any_model_is_resolved(tmp_path, isolated_env, resolved):
    config, _root = make_project(tmp_path, RUNS)

    report = _dispatch(config, tmp_path, user_id="___")

    assert report["status"] == "failed" and report["workers"] == []
    assert "Invalid copick user ID '___'" in report["errors"][0]
    resolved.resolve_models.assert_not_called()


def test_a_missing_model_fails_the_call_before_any_worker_starts(tmp_path, isolated_env, resolved, fake_boundary):
    config, _root = make_project(tmp_path, RUNS)
    resolved.resolution.missing["actin"] = "easymode has no model for 'actin'"

    report = _dispatch(config, tmp_path)

    assert report["status"] == "failed" and report["workers"] == []
    assert "actin" in report["errors"][0]
    assert fake_boundary.loaded == []
