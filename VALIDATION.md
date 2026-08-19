# Migration validation

The default gate is deterministic and model-free. It installs the committed lock, runs on Python 3.11-3.13,
exercises legacy Zarr v2 and supported Zarr v3 copick projects, and builds and inspects the wheel and source
distribution. It does not publish either artifact.

```bash
uv sync --locked --extra test --extra dev
uv run --no-sync pytest -q -p no:warnings tests
uv build
uv run --no-sync python scripts/inspect_distribution.py dist
```

The tagged-runtime compatibility check uses the immutable source revision and bypasses its conflicting dependency
metadata:

```bash
uv pip install --python .venv/bin/python --no-deps \
  "easymode @ git+https://github.com/mgflast/easymode.git@a42377e0b887364050bf47c63700a3dd1c0fa0d0"
uv run --no-sync python -c \
  "from copick_easymode.core.easymode_adapter import get_inference_functions; get_inference_functions()"
```

## Opt-in real-model smoke

The real-model smoke is intentionally manual because it downloads mutable external model files and can be expensive
on hosted runners. Run it against one small, valid tomogram in an existing copick configuration:

```bash
uv run --no-sync python scripts/real_model_smoke.py \
  --config /absolute/path/to/config.json \
  --run run-1 \
  --tomogram wbp@10.0 \
  --model ribosome \
  --add-object \
  --output /absolute/path/to/easymode-smoke.json
```

The command fixes `tta=1` and `batch_size=1`, requires CPU execution by default, writes the result through
`CopickSegmentation.from_numpy()`, and records the input, probability map, output, package versions, device list,
model metadata, model-file SHA-256, discoverable Hugging Face revision, elapsed time, and final statistics. Use a
dedicated user/session or an isolated overlay; pass `--overwrite` only when replacing that exact smoke output is
intended. Omit `--add-object` when the model already has a matching pickable-object definition; when supplied, the
flag updates the referenced configuration.

The same command is the application-level handoff probe for filesystem, S3-compatible, SSH, and read-only
ML Croissant/portal configurations. Remote/read-only sources must have a writable overlay. Backend credentials and
reconnect fault injection remain owned by the copick integration environment; this package must not add store or
retry logic. Record the configuration kind and fault-injection evidence alongside the generated JSON. SMB and GPU
results are optional and non-gating.
