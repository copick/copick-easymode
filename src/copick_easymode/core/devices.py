"""
Which GPUs ``copick inference easymode`` may use, and how runs and CPU threads are split over them.

One inference worker runs per GPU, and each worker sees exactly one device. The devices come from the
allocation and never from outside it:

* ``CUDA_VISIBLE_DEVICES``, when the scheduler (or the user) set it. An empty value or ``-1`` means no
  device, and nothing is probed around that.
* Otherwise the devices ``nvidia-smi -L`` lists, by UUID, so a worker's device cannot be renumbered.

``--gpus`` selects from that allocation: an integer is a position in ``CUDA_VISIBLE_DEVICES`` when it is set
(what CUDA calls device ``i`` in this process) and an ``nvidia-smi`` index otherwise; any other entry must be one
of the allocation's own values (a UUID). Anything else is refused rather than resolved against a device the job
was not given.
"""

import os
import re
import shutil
import subprocess
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

ENV_VISIBLE = "CUDA_VISIBLE_DEVICES"
#: The CPU thread pools a worker is bounded by (OpenMP, and TensorFlow's intra-op and inter-op pools).
THREAD_VARS = ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
#: The device value that hides every GPU from a CPU worker.
NO_DEVICE = "-1"

_NVIDIA_SMI_LINE = re.compile(r"^GPU\s+(\d+):.*\(UUID:\s*(GPU-[0-9a-fA-F-]+)\)")


class DeviceError(ValueError):
    """A requested GPU is not one this process may use."""


def parse_nvidia_smi(text: str) -> List[Tuple[str, str]]:
    """The ``(index, UUID)`` pairs ``nvidia-smi -L`` lists, in its order.

    The UUID is the device's identity: a partial allocation can list ``GPU 2`` and ``GPU 7``, which must not
    become devices 0 and 1, and ``CUDA_VISIBLE_DEVICES`` accepts UUIDs verbatim.
    """
    devices = []
    for line in text.splitlines():
        match = _NVIDIA_SMI_LINE.match(line.strip())
        if match:
            devices.append((match.group(1), match.group(2)))
    return devices


def nvidia_smi_devices() -> List[Tuple[str, str]]:
    """The ``(index, UUID)`` pairs ``nvidia-smi -L`` reports; empty when the tool or a GPU is absent."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        listing = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(listing)


def visible_gpus(
    requested: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    probe: Callable[[], List[Tuple[str, str]]] = nvidia_smi_devices,
) -> List[str]:
    """The GPUs to run on, in worker order, as the values each worker's ``CUDA_VISIBLE_DEVICES`` gets.

    Args:
        requested: ``--gpus``, a comma-separated list; empty or None means every GPU of the allocation.
        env: The environment to read ``CUDA_VISIBLE_DEVICES`` from (default: this process's).
        probe: Lists ``(index, UUID)`` pairs when ``CUDA_VISIBLE_DEVICES`` is not set.

    Raises:
        DeviceError: A requested GPU is not in the allocation.
    """
    env = os.environ if env is None else env
    wanted = [item.strip() for item in (requested or "").split(",") if item.strip()]
    if ENV_VISIBLE in env:
        # The scheduler spoke: an empty value or -1 means no device, and is never probed around.
        raw = env.get(ENV_VISIBLE) or ""
        allocation = [g.strip() for g in raw.split(",") if g.strip() and g.strip() != NO_DEVICE]
        by_index = dict(enumerate(allocation))
        where = f"{ENV_VISIBLE}={raw!r}"
    else:
        listed = probe()
        allocation = [uuid for _, uuid in listed]
        by_index = {int(index): uuid for index, uuid in listed}
        where = f"{ENV_VISIBLE} is not set and nvidia-smi lists {[f'GPU {i}' for i, _ in listed] or 'none'}"
    if not wanted:
        return list(dict.fromkeys(allocation))
    chosen = []
    for item in wanted:
        if item.isdigit() and int(item) in by_index:
            chosen.append(by_index[int(item)])
        elif not item.isdigit() and item in allocation:
            chosen.append(item)
        else:
            raise DeviceError(
                f"GPU {item!r} is not in this job's allocation ({where}); refusing to address a device outside it",
            )
    return list(dict.fromkeys(chosen))


def shard_runs(runs: Iterable[str], n_workers: int) -> List[List[str]]:
    """Round-robin over the sorted run names: disjoint, complete and deterministic, with no empty shard."""
    ordered = sorted(dict.fromkeys(runs))
    n = max(1, int(n_workers))
    return [shard for shard in (ordered[i::n] for i in range(n)) if shard]


def _affinity() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def total_threads(
    threads: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    available: Callable[[], int] = _affinity,
) -> int:
    """The CPU threads to divide among the workers.

    ``threads`` when given; otherwise the CPUs this process may run on (its affinity mask), capped by
    ``SLURM_CPUS_ON_NODE``, the job's CPUs on this node, for a scheduler that does not bind CPUs.
    ``SLURM_CPUS_PER_TASK`` is not used: an allocation of 16 one-CPU tasks would bound one process to one thread.
    """
    env = os.environ if env is None else env
    if threads:
        return max(1, int(threads))
    total = max(1, int(available()))
    slurm = (env.get("SLURM_CPUS_ON_NODE") or "").strip()
    if slurm.isdigit() and int(slurm) > 0:
        total = min(total, int(slurm))
    return total


def threads_per_worker(
    n_workers: int,
    threads: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    available: Callable[[], int] = _affinity,
) -> int:
    """Each worker's share of the CPU threads, at least one."""
    return max(1, total_threads(threads, env, available) // max(1, int(n_workers)))


def worker_environment(gpu: Optional[str], threads: int, cpu: bool = False) -> dict:
    """The environment a worker needs before TensorFlow starts: its one device and its thread bounds.

    ``cpu`` hides every GPU. A worker with neither a GPU nor ``cpu`` (no device could be listed) keeps the
    inherited ``CUDA_VISIBLE_DEVICES`` and lets TensorFlow choose, as a single process always did.
    """
    env = {var: str(int(threads)) for var in THREAD_VARS}
    if cpu:
        env[ENV_VISIBLE] = NO_DEVICE
    elif gpu is not None:
        env[ENV_VISIBLE] = str(gpu)
    return env


def plan_workers(
    runs: Sequence[str],
    devices: Sequence[str],
    *,
    cpu: bool = False,
    max_workers: Optional[int] = None,
) -> List[Tuple[Optional[str], List[str]]]:
    """``(gpu, runs)`` per worker: one per device (at most ``max_workers``), or one without a GPU."""
    if not runs:
        return []
    slots: List[Optional[str]] = [None] if cpu or not devices else list(devices)
    if max_workers:
        slots = slots[: max(1, int(max_workers))]
    shards = shard_runs(runs, len(slots))
    return [(slots[i], shard) for i, shard in enumerate(shards)]


def device_label(gpu: Optional[str], cpu: bool = False) -> str:
    """How a worker's device reads in log lines: ``gpu 1``, ``cpu`` or ``gpu auto``."""
    if cpu:
        return "cpu"
    return f"gpu {gpu}" if gpu is not None else "gpu auto"
