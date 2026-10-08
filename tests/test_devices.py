"""Which GPUs a run may use, how runs are dealt to workers, and each worker's thread share."""

import pytest

from copick_easymode.core.devices import (
    THREAD_VARS,
    DeviceError,
    parse_nvidia_smi,
    plan_workers,
    shard_runs,
    threads_per_worker,
    visible_gpus,
    worker_environment,
)

FOUR = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_CPUS_ON_NODE": "32"}
LISTING = (
    "GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-1111aaaa-0000-0000-0000-000000000002)\n"
    "GPU 7: NVIDIA H100 80GB HBM3 (UUID: GPU-1111aaaa-0000-0000-0000-000000000007)\n"
)
UUID2, UUID7 = "GPU-1111aaaa-0000-0000-0000-000000000002", "GPU-1111aaaa-0000-0000-0000-000000000007"


def _no_probe():
    raise AssertionError("nvidia-smi must not be consulted when CUDA_VISIBLE_DEVICES is set")


def test_the_scheduler_list_is_the_allocation_and_integers_are_positions_in_it():
    assert visible_gpus(None, FOUR, probe=_no_probe) == ["0", "1", "2", "3"]
    assert visible_gpus("0,2", FOUR, probe=_no_probe) == ["0", "2"]
    with pytest.raises(DeviceError, match="'5' is not in this job's allocation"):
        visible_gpus("0,5", FOUR, probe=_no_probe)
    # Positions, not values: in an allocation of physical GPUs 2 and 3, "--gpus 0,1" means those two ...
    assert visible_gpus("0,1", {"CUDA_VISIBLE_DEVICES": "2,3"}, probe=_no_probe) == ["2", "3"]
    # ... and "3" is not a fourth device of this job.
    with pytest.raises(DeviceError):
        visible_gpus("3", {"CUDA_VISIBLE_DEVICES": "2,3"}, probe=_no_probe)
    uuids = {"CUDA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb"}
    assert visible_gpus("1", uuids, probe=_no_probe) == ["GPU-bbbb"]
    assert visible_gpus("GPU-aaaa,GPU-aaaa", uuids, probe=_no_probe) == ["GPU-aaaa"]
    with pytest.raises(DeviceError):
        visible_gpus("GPU-zzzz", uuids, probe=_no_probe)


def test_an_empty_or_negative_scheduler_list_means_no_gpu_and_is_never_probed_around():
    assert visible_gpus(None, {"CUDA_VISIBLE_DEVICES": ""}, probe=_no_probe) == []
    assert visible_gpus(None, {"CUDA_VISIBLE_DEVICES": "-1"}, probe=_no_probe) == []
    with pytest.raises(DeviceError):
        visible_gpus("0", {"CUDA_VISIBLE_DEVICES": "-1"}, probe=_no_probe)


def test_without_the_variable_nvidia_smi_devices_keep_their_identity():
    assert parse_nvidia_smi(LISTING) == [("2", UUID2), ("7", UUID7)]
    assert parse_nvidia_smi("No devices were found\n") == []

    def probe():
        return parse_nvidia_smi(LISTING)

    assert visible_gpus(None, {}, probe=probe) == [UUID2, UUID7]
    assert visible_gpus("7", {}, probe=probe) == [UUID7]  # an nvidia-smi index ...
    assert visible_gpus(UUID2, {}, probe=probe) == [UUID2]
    with pytest.raises(DeviceError, match="nvidia-smi lists"):  # ... never a position in its listing
        visible_gpus("0", {}, probe=probe)
    assert visible_gpus(None, {}, probe=list) == []


def test_runs_are_dealt_round_robin_over_sorted_names_with_no_empty_shard():
    runs = ["e", "a", "c", "b", "d"]
    assert shard_runs(runs, 4) == [["a", "e"], ["b"], ["c"], ["d"]]
    assert shard_runs(["b", "a"], 4) == [["a"], ["b"]]  # fewer runs than GPUs: fewer workers
    assert shard_runs(runs, 1) == [["a", "b", "c", "d", "e"]]
    assert shard_runs(["a", "a", "b"], 2) == [["a"], ["b"]]
    assert shard_runs([], 4) == []

    assert plan_workers(runs, ["0", "1", "2", "3"]) == [("0", ["a", "e"]), ("1", ["b"]), ("2", ["c"]), ("3", ["d"])]
    assert plan_workers(runs, ["1", "3"], max_workers=1) == [("1", ["a", "b", "c", "d", "e"])]
    assert plan_workers(runs, ["0", "1"], cpu=True) == [(None, ["a", "b", "c", "d", "e"])]
    assert plan_workers(runs, []) == [(None, ["a", "b", "c", "d", "e"])]
    assert plan_workers([], ["0", "1"]) == []


def test_each_worker_sees_exactly_one_device_and_its_share_of_the_threads():
    env = worker_environment("2", 8)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert all(env[var] == "8" for var in THREAD_VARS)
    assert worker_environment(None, 4, cpu=True)["CUDA_VISIBLE_DEVICES"] == "-1"
    assert "CUDA_VISIBLE_DEVICES" not in worker_environment(None, 4)  # no GPU listed: inherit, TF decides

    def cpus(n):
        return lambda: n

    assert threads_per_worker(4, env=FOUR, available=cpus(32)) == 8
    assert threads_per_worker(4, threads=6, env=FOUR, available=cpus(32)) == 1  # --threads wins
    assert threads_per_worker(3, threads=1, env={}) == 1  # never zero
    # The job's CPUs on this node cap a mask the scheduler did not narrow (256 CPUs, 16 allocated) ...
    assert threads_per_worker(2, env={"SLURM_CPUS_ON_NODE": "16"}, available=cpus(256)) == 8
    # ... and 16 one-CPU tasks still give one process all 16 (SLURM_CPUS_PER_TASK=1 is not the bound).
    assert threads_per_worker(1, env={"SLURM_CPUS_ON_NODE": "16", "SLURM_CPUS_PER_TASK": "1"}, available=cpus(16)) == 16
    assert threads_per_worker(1, env={}, available=cpus(12)) == 12
