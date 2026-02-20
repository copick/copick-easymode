"""
CLI command for easymode inference on copick tomograms.

This module provides the `copick inference easymode` command for running
pretrained segmentation models on copick data.
"""

import click

from copick.cli.util import add_config_option, add_debug_option, add_user_session_options


@click.command(
    name="easymode",
    short_help="Segment tomograms using easymode pretrained models.",
    context_settings={"show_default": True},
    no_args_is_help=True,
)
@click.option(
    "--model",
    "-m",
    "models",
    type=str,
    required=True,
    help="Comma-separated list of models/features to run (e.g., 'ribosome,membrane').",
)
@click.option(
    "--tomogram",
    "-t",
    type=str,
    required=True,
    help="Tomogram URI in format 'type@voxel_size' (e.g., 'wbp@10.0').",
)
@click.option(
    "--run",
    "-r",
    type=str,
    default="",
    help="Run name or comma-separated list of runs. Empty = all runs.",
)
@click.option(
    "--gpus",
    type=str,
    default=None,
    help="Comma-separated GPU IDs (e.g., '0,1'). Default: all available.",
)
@click.option(
    "--tta",
    type=int,
    default=4,
    help="Test-time augmentation level (1-16). Higher = better but slower.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for inference.",
)
@click.option(
    "--add-objects/--no-add-objects",
    is_flag=True,
    default=True,
    help="Add object definitions to config if missing.",
)
@click.option(
    "--overwrite/--no-overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing segmentations.",
)
@add_config_option
@add_debug_option
@add_user_session_options
@click.pass_context
def easymode(
    ctx: click.Context,
    models: str,
    tomogram: str,
    run: str,
    gpus: str,
    tta: int,
    batch_size: int,
    add_objects: bool,
    overwrite: bool,
    config: str,
    debug: bool,
    user_id: str,
    session_id: str,
):
    """
    Segment copick tomograms using easymode pretrained models.

    This command runs inference using easymode's pretrained segmentation models
    on tomograms stored in a copick project and saves the results as copick
    segmentations.

    Available models include: ribosome, membrane, microtubule, actin, cytoplasm,
    mitochondrion, nucleus, nuclear_envelope, npc, and more.

    \b
    Examples:

    \b
    # Segment ribosomes in all runs
    copick inference easymode -c config.json -m ribosome -t wbp@10.0

    \b
    # Segment multiple features
    copick inference easymode -c config.json -m ribosome,membrane -t wbp@10.0

    \b
    # Segment specific runs with GPU selection
    copick inference easymode -c config.json -m membrane -t wbp@10.0 --run run001,run002 --gpus 0,1

    \b
    # High quality inference with TTA
    copick inference easymode -c config.json -m ribosome -t wbp@10.0 --tta 16 --batch-size 2

    \b
    # Skip adding object definitions to config
    copick inference easymode -c config.json -m ribosome -t wbp@10.0 --no-add-objects
    """
    # Deferred imports for CLI performance
    import copick

    from copick.util.log import get_logger

    from copick_easymode.core.inference import run_easymode_inference

    logger = get_logger(__name__, debug=debug)

    # Validate config
    if not config:
        logger.critical("Configuration file is required. Use -c/--config or set COPICK_CONFIG.")
        ctx.fail("Configuration file is required.")

    # Parse tomogram URI
    try:
        if "@" not in tomogram:
            raise ValueError("Missing '@' separator")
        tomo_type, voxel_str = tomogram.split("@", 1)
        voxel_size = float(voxel_str)
        if not tomo_type:
            raise ValueError("Empty tomogram type")
    except ValueError as e:
        logger.critical(f"Invalid tomogram URI: {tomogram}. Expected format: 'type@voxel_size' (e.g., 'wbp@10.0')")
        ctx.fail(f"Invalid tomogram URI: {tomogram}. Error: {e}")

    # Parse models
    model_list = [m.strip().lower() for m in models.split(",") if m.strip()]
    if not model_list:
        logger.critical("No models specified.")
        ctx.fail("No models specified.")

    # Parse runs
    run_list = [r.strip() for r in run.split(",") if r.strip()] if run else []

    # Validate TTA
    if tta < 1 or tta > 16:
        logger.critical(f"TTA must be between 1 and 16, got {tta}")
        ctx.fail(f"TTA must be between 1 and 16, got {tta}")

    # Load copick project
    try:
        logger.info(f"Loading copick project from {config}")
        root = copick.from_file(config)
    except Exception as e:
        logger.critical(f"Failed to load copick config: {e}")
        ctx.fail(f"Failed to load copick config: {e}")

    logger.info(f"Models: {model_list}")
    logger.info(f"Tomogram: {tomo_type}@{voxel_size}")
    logger.info(f"Runs: {run_list if run_list else 'all'}")
    logger.info(f"TTA: {tta}, Batch size: {batch_size}")
    logger.info(f"GPUs: {gpus if gpus else 'auto'}")
    logger.info(f"Add objects: {add_objects}, Overwrite: {overwrite}")

    # Run inference
    try:
        stats = run_easymode_inference(
            root=root,
            run_names=run_list,
            tomo_type=tomo_type,
            voxel_size=voxel_size,
            models=model_list,
            user_id=user_id,
            session_id=session_id,
            tta=tta,
            batch_size=batch_size,
            gpus=gpus,
            add_objects=add_objects,
            overwrite=overwrite,
            config_path=config if add_objects else None,
            logger=logger,
        )
    except Exception as e:
        logger.critical(f"Inference failed: {e}")
        ctx.fail(f"Inference failed: {e}")

    # Report results
    logger.info(f"Inference completed: {stats['processed']} processed, {stats['skipped']} skipped")
    if stats["errors"]:
        logger.warning(f"Errors encountered: {len(stats['errors'])}")
        for error in stats["errors"]:
            logger.warning(f"  - {error}")

    if stats["processed"] == 0 and stats["skipped"] == 0:
        logger.warning("No tomograms were processed. Check run names and tomogram URIs.")
