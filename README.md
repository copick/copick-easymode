# copick-easymode

Easymode pretrained segmentation integration for copick CLI.

This plugin provides CLI commands to run [easymode](https://github.com/mgflast/easymode) pretrained segmentation models
on tomograms stored in copick projects. For more information about easymode, visit the [documentation](https://mgflast.github.io/easymode).
If you use this plugin, please cite the easymode authors.

## Installation

```bash
git clone https://github.com/copick/copick-easymode.git
cd copick-easymode
pip install -e .
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

- Python >= 3.10
- copick >= 0.8.0
- easymode
- TensorFlow >= 2.10.0

## License

GPLv3 License
