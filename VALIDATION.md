# Migration validation

The default gate is deterministic and model-free. It installs the committed lock without easymode, runs on Python
3.11-3.13 (and 3.13 on Windows), exercises legacy Zarr v2 and supported Zarr v3 copick projects, and builds and
inspects the wheel and source distribution. It does not publish either artifact. Tests that need easymode skip here.

```bash
uv sync --locked --extra test --extra dev
uv run --no-sync pytest -q tests
uv build
uv run --no-sync python scripts/inspect_distribution.py dist
```

The tagged-runtime compatibility check installs the supported easymode release (1.2.6) from its source commit,
bypassing its conflicting dependency metadata, checks the easymode boundary, and runs the whole suite again, now
including the multi-GPU dispatch and overlapped-loop tests that run tiny models through easymode:

```bash
uv sync --locked --extra test
uv pip install --no-deps \
  "easymode @ git+https://github.com/mgflast/easymode.git@00ab9035b19ad30dca46fbe97bc77e6f11a2d981"
uv run --no-sync python -c \
  "from copick_easymode.core import easymode_env as env; env.import_easymode(); env.inference_functions()"
uv run --no-sync pytest -q tests
```

Importing easymode rewrites `~/easymode/settings.txt`; set `HOME` to a scratch directory to keep your own settings
out of it.

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

The command resolves the model through the easymode boundary (`--model-dir` selects a model directory, `--offline`
uses only what is already in it), loads it with `load_stages` and runs it over the run through the same overlapped
per-run loop the command uses, so copick reads the tomogram and `CopickSegmentation.from_numpy()` writes the result.
It fixes `tta=1` and `batch_size=1`, requires CPU execution by default (`--allow-gpu` lifts that), and records the
input, probability map, output, package versions, the installed easymode commit, device list, model kind, tag and
timestamp, model-file SHA-256, discoverable Hugging Face revision, per-stage timings, and final statistics. Use a
dedicated user/session or an isolated overlay; pass `--overwrite` only when replacing that exact smoke output is
intended. Omit `--add-object` when the model already has a matching pickable-object definition; when supplied, the
flag updates the referenced configuration.

The generated JSON is safe to commit without manual redaction: it records the model filename and logical
segmentation entity instead of absolute local paths, uses the installed package versions verbatim, and adds a UTC
recording timestamp. The command rejects an all-background result before anything is written and records the
foreground voxel count, so choose a tomogram on which the selected model produces foreground at the requested
threshold. Do not commit a manually edited or empty-output record; valid replacement evidence is pending until such
a real-model run is available.

The same command is the application-level handoff probe for filesystem, S3-compatible, SSH, and read-only
ML Croissant/portal configurations. Remote/read-only sources must have a writable overlay. Backend credentials and
reconnect fault injection remain owned by the copick integration environment; this package must not add store or
retry logic. Record the configuration kind and fault-injection evidence alongside the generated JSON. SMB and GPU
results are optional and non-gating.
