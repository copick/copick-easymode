"""
CLI command for easymode inference on copick tomograms.

This module provides the `copick inference easymode` command for running
pretrained segmentation models on copick data.
"""

import click
from copick.cli.util import add_config_option, add_debug_option, add_user_session_options


def add_easymode_inference_options(func: click.Command) -> click.Command:
    """
    Add easymode inference options: --tta, --batch-size, --threshold.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the inference options added.
    """
    opts = [
        click.option(
            "--tta",
            required=False,
            type=int,
            default=4,
            show_default=True,
            help="Test-time augmentation level (1-16; an .scnm model uses at most 8). Higher = better but slower.",
        ),
        click.option(
            "--batch-size",
            required=False,
            type=int,
            default=1,
            show_default=True,
            help="Batch size for inference.",
        ),
        click.option(
            "--threshold",
            required=False,
            type=float,
            default=0.5,
            show_default=True,
            help="Probability threshold for binarizing segmentation (0.0-1.0).",
        ),
    ]

    for opt in opts:
        func = opt(func)

    return func


def add_parallel_options(func: click.Command) -> click.Command:
    """
    Add device and worker options: --gpus, --cpu, --max-workers, --threads.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the device options added.
    """
    opts = [
        click.option(
            "--gpus",
            required=False,
            type=str,
            default=None,
            help="Comma-separated GPUs to use, one worker process each: positions in CUDA_VISIBLE_DEVICES when it "
            "is set (else nvidia-smi indices), or device UUIDs. A GPU outside the allocation is refused. "
            "Default: every GPU this process may use.",
        ),
        click.option(
            "--cpu",
            is_flag=True,
            default=False,
            help="Run on the CPU, with one worker and no GPU.",
        ),
        click.option(
            "--max-workers",
            required=False,
            type=click.IntRange(min=1),
            default=None,
            help="Use at most this many GPUs (worker processes). Default: one per GPU.",
        ),
        click.option(
            "--threads",
            required=False,
            type=click.IntRange(min=1),
            default=None,
            help="CPU threads to divide among the workers. Default: the CPUs this process may run on, at most "
            "SLURM_CPUS_ON_NODE.",
        ),
    ]

    for opt in reversed(opts):  # listed in --help in this order
        func = opt(func)

    return func


def add_model_report_options(func: click.Command) -> click.Command:
    """
    Add model directory and report options: --model-dir, --offline, --report.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the model and report options added.
    """
    opts = [
        click.option(
            "--model-dir",
            required=False,
            type=click.Path(file_okay=False),
            default=None,
            envvar="COPICK_EASYMODE_MODEL_DIR",
            show_envvar=True,
            help="easymode model directory (the one holding registry.json and models/). Default: easymode's "
            "MODEL_DIRECTORY setting. Applies to this command only; easymode's settings file is not changed.",
        ),
        click.option(
            "--offline",
            is_flag=True,
            default=False,
            envvar="COPICK_EASYMODE_OFFLINE",
            show_envvar=True,
            help="Never contact the model registry: use only the weights already in the model directory, and "
            "write nothing there.",
        ),
        click.option(
            "--report",
            "report_path",
            required=False,
            type=click.Path(dir_okay=False),
            default=None,
            help="Write a JSON report: settings, resolved models, devices, each worker's runs and outcome, "
            "errors and timings.",
        ),
    ]

    for opt in reversed(opts):  # listed in --help in this order
        func = opt(func)

    return func


def add_object_overwrite_options(func: click.Command) -> click.Command:
    """
    Add object and overwrite options: --add-objects, --overwrite.

    Args:
        func (click.Command): The Click command to which the options will be added.

    Returns:
        click.Command: The Click command with the object/overwrite options added.
    """
    opts = [
        click.option(
            "--add-objects/--no-add-objects",
            is_flag=True,
            default=True,
            show_default=True,
            help="Add object definitions to config if missing.",
        ),
        click.option(
            "--overwrite/--no-overwrite",
            is_flag=True,
            default=False,
            show_default=True,
            help="Overwrite existing segmentations.",
        ),
    ]

    for opt in opts:
        func = opt(func)

    return func


@click.command(
    name="easymode",
    short_help="Segment tomograms using easymode pretrained models.",
    context_settings={"show_default": True},
    no_args_is_help=True,
)
@add_config_option
@click.option(
    "--model",
    "-m",
    "models",
    required=True,
    type=str,
    help="Comma-separated list of models/features to run (e.g., 'ribosome,membrane').",
)
@click.option(
    "--tomogram",
    "-t",
    required=True,
    type=str,
    help="Tomogram URI in format 'type@voxel_size' (e.g., 'wbp@10.0').",
)
@click.option(
    "--run",
    "-r",
    required=False,
    type=str,
    default="",
    help="Run name or comma-separated list of runs. Empty = all runs.",
)
@add_easymode_inference_options
@add_parallel_options
@add_model_report_options
@add_user_session_options
@add_object_overwrite_options
@add_debug_option
@click.pass_context
def easymode(
    ctx: click.Context,
    config: str,
    models: str,
    tomogram: str,
    run: str,
    tta: int,
    batch_size: int,
    threshold: float,
    gpus: str,
    cpu: bool,
    max_workers: int,
    threads: int,
    model_dir: str,
    offline: bool,
    report_path: str,
    user_id: str,
    session_id: str,
    add_objects: bool,
    overwrite: bool,
    debug: bool,
):
    """
    Segment copick tomograms using easymode pretrained models.

    This command runs inference using easymode's pretrained segmentation models
    on tomograms stored in a copick project and saves the results as copick
    segmentations. Each requested feature is written back as its own single-label
    segmentation at the input tomogram's voxel size; with ``--add-objects`` (on by
    default) each feature is also registered as a pickable object in the config.

    One worker process runs per GPU, and the runs are split among them. Each worker
    reads and preprocesses the next tomogram and writes the previous segmentation
    while its GPU segments the current one.

    Available models include: ribosome, membrane, microtubule, actin, cytoplasm,
    mitochondrion, nucleus, nuclear_envelope, npc, and more.

    \b
    URI Format:
        Tomograms:     type@voxel_spacing
        Segmentations: name:user_id/session_id@voxel_spacing

    \b
    Examples:
        # Segment ribosomes in all runs, on every GPU of the allocation
        copick inference easymode -c config.json -m ribosome -t wbp@10.0

        # Segment multiple features
        copick inference easymode -c config.json -m ribosome,membrane -t wbp@10.0

        # Segment specific runs on GPUs 0 and 1 (one worker process each)
        copick inference easymode -c config.json -m membrane -t wbp@10.0 --run run001,run002 --gpus 0,1

        # High quality inference with TTA
        copick inference easymode -c config.json -m ribosome -t wbp@10.0 --tta 16 --batch-size 2

        # Skip adding object definitions to config
        copick inference easymode -c config.json -m ribosome -t wbp@10.0 --no-add-objects

        # A shared model directory, no downloads, and a JSON report
        copick inference easymode -c config.json -m actin -t wbp@10.0 --model-dir /shared/easymode --offline --report easymode.json

    \b
    See Also:
        copick convert seg2mesh: turn an easymode segmentation into a surface mesh

    \b
    Notes:
        A CUDA GPU is strongly recommended; the weights download automatically on first use.
        The exit status is non-zero when a model is missing, a run fails or a worker dies.

    \b
    Acknowledgements:
        This command uses pretrained models from easymode by Mart G.F. Last.
        Docs: https://mgflast.github.io/easymode
        Repo: https://github.com/mgflast/easymode
        If you use these models in your research, please cite the easymode authors.
    """
    # Deferred imports for CLI performance
    from copick.util.log import get_logger

    from copick_easymode.core.dispatch import dispatch_easymode_inference

    logger = get_logger(__name__, debug=debug)

    # Acknowledge easymode authors
    logger.info("Using easymode pretrained models by Mart G. F. So-Last et al.")
    logger.info(
        "If you use these models, please cite the easymode preprint: So-Last et al. (2026), 'Easymode: general pretrained networks for cellular cryo-ET enable flexible approaches to subtomogram averaging', bioRxiv — https://www.biorxiv.org/content/10.64898/2026.05.19.726344v1",
    )
    logger.info("easymode docs: https://mgflast.github.io/easymode | repo: https://github.com/mgflast/easymode")

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

    if cpu and gpus:
        ctx.fail("--cpu and --gpus exclude each other.")

    logger.info(f"Models: {model_list}")
    logger.info(f"Tomogram: {tomo_type}@{voxel_size}")
    logger.info(f"Runs: {run_list if run_list else 'all'}")
    logger.info(f"TTA: {tta}, Batch size: {batch_size}, Threshold: {threshold}")
    logger.info(f"GPUs: {'none (--cpu)' if cpu else gpus if gpus else 'all of the allocation'}")
    logger.info(f"Add objects: {add_objects}, Overwrite: {overwrite}")

    # Run inference
    report = dispatch_easymode_inference(
        config_path=config,
        run_names=run_list,
        tomo_type=tomo_type,
        voxel_size=voxel_size,
        models=model_list,
        user_id=user_id,
        session_id=session_id,
        tta=tta,
        batch_size=batch_size,
        threshold=threshold,
        overwrite=overwrite,
        gpus=gpus,
        cpu=cpu,
        max_workers=max_workers,
        threads=threads,
        model_dir=model_dir,
        offline=offline,
        add_objects=add_objects,
        report_path=report_path,
        debug=debug,
        logger=logger,
    )

    # Report results
    logger.info(f"Inference completed: {report['processed']} processed, {report['skipped']} skipped")
    if report["errors"]:
        logger.warning(f"Errors encountered: {len(report['errors'])}")
        for error in report["errors"]:
            logger.warning(f"  - {error}")

    if report["processed"] == 0 and report["skipped"] == 0 and not report["errors"]:
        logger.warning("No tomograms were processed. Check run names and tomogram URIs.")

    if report_path:
        logger.info(f"Report written to {report_path}")
    if report["status"] != "complete":
        ctx.exit(1)
