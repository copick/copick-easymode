# copick-easymode

Easymode pretrained segmentation integration for copick CLI.

**[Preprint](https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1) | [easymode docs](https://mgflast.github.io/easymode) | [easymode repo](https://github.com/mgflast/easymode)**

This plugin provides CLI commands to run [easymode](https://github.com/mgflast/easymode) pretrained segmentation models
on tomograms stored in copick projects. For more information about easymode, visit the [documentation](https://mgflast.github.io/easymode).
If you use this plugin, please cite the easymode preprint (see [Citation](#citation)).

## Installation

The `v2.0` development line targets copick 2, OME-Zarr 0.5, and Zarr v3. It supports Python
3.11-3.13 and reads legacy OME-Zarr 0.4 / Zarr v2 projects through copick's compatibility layer.

easymode 1.0.0 is not published to PyPI, and its source metadata pins versions that conflict
with copick's NumPy 2 stack. Install this repository and its dependencies first, then install the
immutable easymode 1.0.0 source revision with `--no-deps`:

```bash
git clone --branch v2.0 https://github.com/copick/copick-easymode.git
cd copick-easymode

# 1. Install copick-easymode + dependencies (copick 2 alpha, NumPy 2, TensorFlow 2.20+).
pip install -e .

# 2. Install the audited easymode-1.0.0 commit WITHOUT dependency resolution.
pip install --no-deps "easymode @ git+https://github.com/mgflast/easymode.git@a42377e0b887364050bf47c63700a3dd1c0fa0d0"
```

Verify the install:

```bash
python -c "import numpy, easymode, importlib.metadata as m; print('numpy', numpy.__version__, '| easymode', m.version('easymode'))"
# expected: NumPy 2.x. The audited tag currently exposes legacy distribution
# metadata version 0.0.5 even though its Git tag is easymode-1.0.0.
```

See [VALIDATION.md](VALIDATION.md) for deterministic tests and the opt-in,
checksum-recorded real-model/backend smoke. Model weights are never downloaded by normal pull-request tests.

## Usage

After installation, the `copick inference easymode` command becomes available:

```bash
# Basic usage - segment ribosomes in all runs
copick inference easymode -c config.json -m ribosome -t wbp@10.0

# Segment multiple features
copick inference easymode -c config.json -m ribosome,membrane,microtubule -t wbp@10.0

# Segment specific runs
copick inference easymode -c config.json -m membrane -t wbp@10.0 --run run001,run002

# Use specific GPUs
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --gpus 0,1

# High quality with test-time augmentation
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --tta 16

# Don't add object definitions to config
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --no-add-objects

# Overwrite existing segmentations
copick inference easymode -c config.json -m ribosome -t wbp@10.0 --overwrite
```

## Available Models

The following pretrained segmentation models are available:

| Model | Description |
|-------|-------------|
| `ribosome` | Ribosome particles |
| `membrane` | Cellular membranes |
| `microtubule` | Microtubules |
| `actin` | Actin filaments |
| `cytoplasm` | Cytoplasm region |
| `mitochondrion` | Mitochondria |
| `nucleus` | Nuclear region |
| `nuclear_envelope` | Nuclear envelope |
| `npc` | Nuclear pore complex |
| `cytoplasmic_granule` | Cytoplasmic granules |
| `mitochondrial_granule` | Mitochondrial granules |
| `prohibitin` | Prohibitin complexes |
| `tric` | TRiC/CCT chaperonin |
| `vault` | Vault particles |
| `void` | Void/empty regions |

## Command Options

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to copick configuration file (or set `COPICK_CONFIG` env var) |
| `-m, --model` | Comma-separated list of models to run (required) |
| `-t, --tomogram` | Tomogram URI as `type@voxel_size` e.g., `wbp@10.0` (required) |
| `-r, --run` | Run name(s) to process, comma-separated. Empty = all runs |
| `--gpus` | Comma-separated GPU IDs. Default: all available |
| `--tta` | Test-time augmentation level 1-16. Higher = better but slower. Default: 4 |
| `--batch-size` | Batch size for inference. Default: 1 |
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

- Python >= 3.11, < 3.14
- copick >= 2.0.0a1, < 3
- numpy >= 2.0.2
- TensorFlow >= 2.20, < 3
- easymode 1.0.0 commit `a42377e0b887364050bf47c63700a3dd1c0fa0d0` (installed separately; see [Installation](#installation))

## Citation

This plugin runs the pretrained **easymode** models. If you use it in your research, please cite the easymode preprint:

> So-Last, M. G. F., Hale, T., Burt, A., & Allegretti, M. (2026). *Easymode: general pretrained networks for cellular cryo-ET enable flexible approaches to subtomogram averaging.* bioRxiv. https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1

## License

GPLv3 License
