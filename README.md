# copick-easymode

Easymode pretrained segmentation integration for copick CLI.

**[Preprint](https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1) | [easymode docs](https://mgflast.github.io/easymode) | [easymode repo](https://github.com/mgflast/easymode)**

This plugin provides CLI commands to run [easymode](https://github.com/mgflast/easymode) pretrained segmentation models
on tomograms stored in copick projects. For more information about easymode, visit the [documentation](https://mgflast.github.io/easymode).
If you use this plugin, please cite the easymode preprint (see [Citation](#citation)).

## Installation

easymode 1.0.0 is not published to PyPI (only older `0.0.x` releases are), so it is installed
from GitHub. Its packaging metadata also (incorrectly) pins `numpy<2` / `tensorflow<2.12`, which
conflicts with copick's `numpy>=2` — even though easymode runs fine on `numpy>=2`. To avoid that
conflict, install copick-easymode and its dependencies **first** (this brings in copick and
`numpy>=2`), then install easymode from GitHub with `--no-deps` so its bad pins are ignored:

```bash
git clone https://github.com/copick/copick-easymode.git
cd copick-easymode

# 1. Install copick-easymode + dependencies (copick, numpy>=2, tensorflow>=2.16, easymode's runtime deps)
pip install -e .

# 2. Install easymode from GitHub WITHOUT dependency resolution.
#    --no-deps keeps your numpy>=2 stack intact, and also upgrades over any older easymode
#    (e.g. a 0.0.x already installed from PyPI).
pip install --no-deps git+https://github.com/mgflast/easymode.git
```

Verify the install:

```bash
python -c "import numpy, easymode, importlib.metadata as m; print('numpy', numpy.__version__, '| easymode', m.version('easymode'))"
# expected: numpy 2.x | easymode 1.0.0
```

## Usage

After installation, the `copick inference easymode` command becomes available:

```bash
# Basic usage - segment ribosomes in all runs
copick inference easymode -c config.json -m ribosome -t wbp@10.0

# Segment multiple features
copick inference easymode -c config.json -m ribosome,membrane,microtubule -t wbp@10.0

# Segment specific runs
copick inference easymode -c config.json -m membrane -t wbp@10.0 --run run001,run002

# Use GPUs 0 and 1: one worker process each, the runs split between them
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --gpus 0,1

# Run on the CPU
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --cpu

# High quality with test-time augmentation
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --tta 16

# Don't add object definitions to config
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --no-add-objects

# Overwrite existing segmentations
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --overwrite

# A shared model directory, no downloads, and a JSON report of what happened
copick inference easymode -c config.json -m actin -t wbp@10.0 --model-dir /shared/easymode --offline --report easymode.json
```

## Multiple GPUs and throughput

TensorFlow computes on one GPU per process, so `copick inference easymode` starts **one worker process per GPU**
(like `easymode segment` and `ais segment`) and deals the runs round-robin over their sorted names. Each worker sees
exactly its own device in `CUDA_VISIBLE_DEVICES`, set before TensorFlow starts, and gets an equal share of the CPU
threads (`OMP_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`). With a single GPU the one worker runs
in the command's own process.

- **Which GPUs.** By default every GPU the process may use: the entries of `CUDA_VISIBLE_DEVICES` when it is set (an
  empty value or `-1` means none), otherwise the devices `nvidia-smi -L` lists. `--gpus` selects from those: an integer
  is a position in `CUDA_VISIBLE_DEVICES` when it is set (under SLURM, `--gpus 0,1` means the job's first two GPUs,
  whatever their physical numbers), otherwise an `nvidia-smi` index; any other entry must be one of the allocation's
  device UUIDs. A GPU outside the allocation is refused. `--max-workers` caps the number of GPUs used, and `--cpu` runs
  one worker without a GPU.
- **Threads.** `--threads` CPU threads are divided among the workers (default: the CPUs the process may run on, at most
  the job's `SLURM_CPUS_ON_NODE`).
- **Overlap.** Within a worker, while the GPU segments one tomogram, a reader thread reads and preprocesses the next
  (`tomo.numpy()`, rescaling, normalization, padding) and a writer thread finishes the previous one (rescaling back,
  thresholding, writing the segmentation). At most one tomogram is prepared ahead and one is being written. The results
  are identical to running the steps one after another.
- **Models are resolved once**, in the parent process, before any worker starts: easymode downloads what is missing
  (into a writable model directory, under a lock there, so concurrent jobs fetch once, and with umask 002 so a
  group-shared directory stays updatable by the group), or, offline or in a directory you cannot write, only finds
  what is there. A missing model fails the command before any GPU time is spent. Workers load the resolved weights
  with easymode kept offline.
- **Model directory.** `--model-dir` (or `COPICK_EASYMODE_MODEL_DIR`) points easymode at a model directory (the one
  holding `registry.json` and `models/`) for this command only; easymode's settings file is not changed. `--offline`
  never contacts the model registry and writes nothing into the directory.
- **Concurrent imports.** easymode rewrites `~/easymode/settings.txt` whenever it is imported, which breaks processes
  importing it at the same moment. Every process here imports it under a file lock beside that file
  (`~/easymode/.copick-easymode-import.lock`, or `COPICK_EASYMODE_IMPORT_LOCK`), so workers and concurrent jobs
  cannot collide.
- **Config writes.** With `--add-objects`, only the parent process adds object definitions and saves the config,
  before the workers start; workers re-open the project from the config file and never write it.
- **Logs** of each worker start with `[worker k gpu X]`.

### Exit status and report

The command exits non-zero when any run failed, a model could not be found, a requested run is not in the project,
a requested GPU is not in the allocation, or a worker died; it still ends with the `Errors encountered: N` summary.
Runs whose tomogram is missing or whose segmentation already exists are skipped, not failed.

`--report PATH` writes a JSON report when the command starts (`"status": "running"`), as each worker finishes, and at
the end (`"complete"` or `"failed"`), whatever the outcome:

```json
{
  "tool": "copick-easymode", "version": "0.3.0", "status": "complete",
  "config": "/data/copick_config.json", "tomogram": "wbp@10.0", "runs": ["TS_01", "TS_02", "TS_03"],
  "user_id": "copick", "session_id": "1", "tta": 4, "threshold": 0.5, "batch_size": 1, "overwrite": false,
  "models_requested": ["microtubule"],
  "models": [{"feature": "microtubule", "object": "microtubule", "weights": "/shared/easymode/models/microtubule_sv3-3d.h5",
              "kind": "h5", "tag": "sv3-3d", "timestamp": "20260916223602", "bytes": 515770304, "apix": 10.0}],
  "missing": {}, "model_directory": "/shared/easymode", "online": true, "writable": true, "easymode_version": "1.2.5",
  "cpu": false, "devices": ["0", "1"], "threads_per_worker": 8,
  "workers": [
    {"index": 0, "gpu": "0", "runs": ["TS_01", "TS_03"], "exitcode": 0, "seconds": 412.3, "processed": 2, "skipped": 0,
     "errors": [], "visible_devices": "0", "threads": 8,
     "items": [{"run": "TS_01", "feature": "microtubule", "status": "processed",
                "seconds": {"read": 2.1, "prepare": 1.9, "infer": 198.0, "finish": 1.2, "write": 0.8}}]},
    {"index": 1, "gpu": "1", "runs": ["TS_02"], "exitcode": 0, "seconds": 205.7, "processed": 1, "skipped": 0,
     "errors": [], "visible_devices": "1", "threads": 8, "items": []}
  ],
  "processed": 3, "skipped": 0, "errors": [],
  "started_utc": "2026-10-08T20:00:00Z", "finished_utc": "2026-10-08T20:06:55Z", "seconds": 414.9
}
```

`missing` names each model that could not be resolved, with easymode's reason. A worker's `gpu` is the
`CUDA_VISIBLE_DEVICES` value it ran with (`null` with `--cpu`), and its `exitcode` is its process exit status (0 when
all of its runs were handled without an error).

### Python API

`copick_easymode.core.dispatch_easymode_inference(config_path, run_names, tomo_type, voxel_size, models, user_id,
session_id, ...)` is the command without the CLI: it takes the config path (workers re-open the project) and returns the
report. `copick_easymode.core.run_easymode_inference(root, ...)` keeps running everything in the calling process, with
reads and writes overlapped.

## Available Models

Every feature in easymode's model registry can be run; `easymode list` prints them with their versions.
easymode publishes them in two formats, and both run in the same process and on the same GPU:

- **`.h5` (3D)**: `ribosome`, `microtubule` and `tric`, loaded through `easymode.core.distribution.load_model`.
- **`.scnm` (Ais 2D-engine models, 2.5D slice or 3D slab)**: every other feature, e.g. `actin`, `membrane`,
  `proteasome`, `atp_synthase`, `cytoplasm`, `nucleus`. `copick_easymode.core.scnm` loads them in Keras 3 and runs
  a port of `ais segment`'s inference (Ais 1.2.38), so neither Ais nor its TensorFlow 2.11 is needed. `--tta` above 8
  is reduced to 8 for these, the most Ais supports.

copick does not allow underscores in object names, so a feature such as `atp_synthase` is written to copick as
`atp-synthase` (segmentation and object name alike); pass the easymode name to `-m`.

## Command Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to copick configuration file (or set `COPICK_CONFIG` env var) |
| `-m, --model` | Comma-separated list of models to run (required) |
| `-t, --tomogram` | Tomogram URI as `type@voxel_size` e.g., `wbp@10.0` (required) |
| `-r, --run` | Run name(s) to process, comma-separated. Empty = all runs |
| `--gpus` | Comma-separated GPUs, one worker process each: positions in `CUDA_VISIBLE_DEVICES` when set (else `nvidia-smi` indices), or device UUIDs. Default: every GPU of the allocation |
| `--cpu` | Run on the CPU with one worker |
| `--max-workers` | Use at most this many GPUs (worker processes) |
| `--threads` | CPU threads to divide among the workers. Default: the CPUs the process may run on, at most `SLURM_CPUS_ON_NODE` |
| `--model-dir` | easymode model directory for this command (or `COPICK_EASYMODE_MODEL_DIR`). Default: easymode's setting |
| `--offline` | Use only the weights already in the model directory; never contact the registry (or `COPICK_EASYMODE_OFFLINE`) |
| `--report` | Write a JSON report of the settings, models, devices, workers, errors and timings |
| `--tta` | Test-time augmentation level 1-16 (8 at most for an `.scnm` model). Higher = better but slower. Default: 4 |
| `--batch-size` | Batch size for inference. Default: 1 |
| `--threshold` | Probability threshold for binarizing a segmentation. Default: 0.5 |
| `--add-objects/--no-add-objects` | Add object definitions to config if missing. Default: enabled |
| `--overwrite/--no-overwrite` | Overwrite existing segmentations. Default: disabled |
| `--user-id` | User ID for created segmentations. Default: copick |
| `--session-id` | Session ID for created segmentations. Default: 1 |
| `--debug/--no-debug` | Enable debug logging |

## Object Definitions

When `--add-objects` is enabled (default), the plugin automatically adds object definitions to your copick config for any segmented features that don't already exist. These are added with minimal defaults:

- `is_particle`: False (segmentation target)
- `label`: Auto-assigned (next available integer)
- `color`: Auto-assigned

You can edit the config file afterward to add additional metadata like `emdb_id`, `pdb_id`, `radius`, etc.

## Output

Segmentations are stored in the copick project at:

```
{overlay_root}/ExperimentRuns/{run_name}/VoxelSpacing{voxel_size:.3f}/Segmentations/
```

Each segmentation is stored as a zarr array with OME-Zarr metadata.

## Requirements

- Python >= 3.10, < 3.13
- copick >= 1.24.1
- numpy >= 2.0.2
- TensorFlow >= 2.16
- easymode (installed separately from GitHub — see [Installation](#installation))

## Citation

This plugin runs the pretrained **easymode** models. If you use it in your research, please cite the easymode preprint:

> So-Last, M. G. F., Hale, T., Burt, A., & Allegretti, M. (2026). *Easymode: general pretrained networks for cellular cryo-ET enable flexible approaches to subtomogram averaging.* bioRxiv. https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1

## License

GPLv3 License
