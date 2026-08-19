"""Deterministic tests for the opt-in real-model evidence helper."""

import argparse
import hashlib

import numpy as np
import pytest

from scripts.real_model_smoke import discover_huggingface_revision, parse_tomogram, sha256_array, sha256_file


def test_hash_helpers_include_array_contract(tmp_path):
    path = tmp_path / "weights.h5"
    path.write_bytes(b"weights")
    array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

    assert sha256_file(path) == hashlib.sha256(b"weights").hexdigest()
    assert sha256_array(array) != sha256_array(array.astype(np.float64))
    assert sha256_array(array) != sha256_array(array.reshape(1, 2, 4))


def test_revision_discovery_reads_huggingface_snapshot_layout(tmp_path):
    model = tmp_path / "ribosome.h5"
    model.write_bytes(b"weights")
    snapshot = tmp_path / "models--mgflast--easymode" / "snapshots" / "abc123" / model.name
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(model.read_bytes())

    assert discover_huggingface_revision(model) == "abc123"


def test_revision_discovery_reads_local_directory_cache_reference(tmp_path):
    model = tmp_path / "ribosome.h5"
    model.write_bytes(b"weights")
    reference = tmp_path / "models--mgflast--easymode" / "refs" / "main"
    reference.parent.mkdir(parents=True)
    reference.write_text("def456\n", encoding="utf-8")

    assert discover_huggingface_revision(model) == "def456"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("wbp@10", ("wbp", 10.0)), ("denoised@7.5", ("denoised", 7.5))],
)
def test_tomogram_parser(value, expected):
    assert parse_tomogram(value) == expected


@pytest.mark.parametrize("value", ["wbp", "@10", "wbp@0", "wbp@not-a-number"])
def test_tomogram_parser_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_tomogram(value)
