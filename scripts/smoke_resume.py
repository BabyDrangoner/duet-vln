"""Real CUDA/rollout acceptance: pause, SIGKILL, restore from persistent storage.

Use a unique smoke run. This evaluates just eight train_dev instructions and
does not estimate research gains. No runtime is stopped or restarted here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from vln_improve.pipeline import (ROOT, MountedCheckpointStore, atomic_json,
    backup_mount_identity, digest, validate_backup_root)
from vln_improve.resumable import ResumableTrainer


def assert_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_equal(a, b)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backup-root")
    parser.add_argument("--backup-backend", choices=("drive", "filesystem"), default="drive")
    parser.add_argument("--cache", default="outputs/cache-fit8")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args()
    if args.backup_root is None:
        if args.backup_backend == "filesystem":
            parser.error("filesystem backup requires an explicit existing --backup-root")
        args.backup_root = "/content/drive/MyDrive/VLN-Research/runs"
    backup_root = Path(args.backup_root).expanduser().resolve()
    mount_identity = backup_mount_identity(backup_root, backend=args.backup_backend,
                                          allow_local=args.allow_local_backup_for_tests)
    def check_backup():
        return validate_backup_root(backup_root, backend=args.backup_backend,
            allow_local=args.allow_local_backup_for_tests, expected_identity=mount_identity)
    evidence = ROOT / "outputs" / (args.run_id + "-acceptance")
    evidence.mkdir(parents=True, exist_ok=False)
    backup = backup_root / args.run_id
    if backup.exists():
        raise ValueError("acceptance test requires a new run_id")
    config = json.loads((ROOT / "configs/pipeline.json").read_text())
    config.update(run_id=args.run_id, scope="smoke", cache=[args.cache],
                  local_root=str(evidence / "vm1"), backup_root=str(backup.parent),
                  backup_backend=args.backup_backend,
                  checkpoint_every_steps=2, checkpoint_every_seconds=60,
                  keep_local=2, keep_backup=3, max_process_seconds=1800,
                  max_vm_age_seconds=None, evaluation_reserve_seconds=120)
    config["training"].update(epochs=2, batch_size=8)
    config["validation"].update(limit=8, timeout_seconds=300)
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       TOKENIZERS_PARALLELISM="false")
    simulator = str(ROOT / "third_party/Matterport3DSimulator/build")
    environment["PYTHONPATH"] = simulator + os.pathsep + environment.get("PYTHONPATH", "")

    def launch(stage, settings, flags=()):
        check_backup()
        path = evidence / f"{stage}-config.json"
        atomic_json(path, settings)
        command = [sys.executable, "-u", "-m", "vln_improve.pipeline", "--config", str(path), *flags]
        if args.allow_local_backup_for_tests:
            command.append("--allow-local-backup-for-tests")
        log = (evidence / f"{stage}.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        return process

    def wait(process, expected):
        try:
            code = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if code != expected:
            raise RuntimeError(f"stage returned {code}, expected {expected}; logs in {evidence}")

    first = launch("01-pause", config, ("--stop-after-steps", "3"))
    wait(first, 75)
    status1 = json.loads((Path(config["local_root"]) / args.run_id / "status.json").read_text())
    assert status1["global_step"] == 3 and status1["status"] == "paused"
    print(json.dumps({"stage": "paused", "step": status1["global_step"]}), flush=True)

    # A fresh local directory and a missing cache simulate losing /content.
    cold = copy.deepcopy(config)
    cold.update(local_root=str(evidence / "vm2"), cache=[str(evidence / "missing-original-cache")])
    second = launch("02-kill", cold, ("--require-resume",))
    started = time.monotonic()
    try:
        while second.poll() is None:
            log = (evidence / "02-kill.log").read_text()
            if '"reason": "validation_pending"' in log:
                os.killpg(second.pid, signal.SIGKILL)
                break
            if time.monotonic() - started > 300:
                raise TimeoutError("did not reach pending validation")
            time.sleep(0.05)
    finally:
        if second.poll() is None:
            os.killpg(second.pid, signal.SIGKILL)
    wait(second, -signal.SIGKILL)
    killed_store = MountedCheckpointStore(evidence / "inspect-after-kill", backup,
        keep_local=2, keep_backup=3, backup_check=check_backup)
    killed, _, _ = killed_store.restore("latest")
    assert killed["control"]["evaluation_pending"] is True
    killed_step = killed["trainer"]["global_step"]
    print(json.dumps({"stage": "killed_after_backup", "step": killed_step}), flush=True)

    fresh = copy.deepcopy(cold)
    fresh["local_root"] = str(evidence / "vm3")
    third = launch("03-recover", fresh, ("--require-resume",))
    wait(third, 0)
    local = Path(fresh["local_root"]) / args.run_id
    result = json.loads((local / "status.json").read_text())
    assert result["status"] == "complete" and result["resumed"]
    store = MountedCheckpointStore(local, backup, keep_local=2, keep_backup=3, backup_check=check_backup)
    final, _, _ = store.restore("latest")
    best, _, best_manifest = store.restore("best")
    assert len(final["control"]["validation_history"]) == 2
    assert final["control"]["last_evaluated_epoch"] == 2
    assert best_manifest["is_best"]
    best_head = Path(best_manifest["local_path"]) / "head.pt"
    assert digest(best_head) == best["control"]["best"]["head_sha256"]

    reference = ResumableTrainer([ROOT / args.cache], **config["training"])
    while not reference.done:
        reference.step()
    expected = reference.state_dict()
    for key in ("head", "optimizer", "history", "global_step", "epoch", "record_cursor", "epoch_totals"):
        assert_equal(final["trainer"][key], expected[key])
    local_count = len(list((local / "snapshots").glob("step-*")))
    cloud_count = len(list((backup / "snapshots").glob("step-*")))
    assert local_count <= config["keep_local"] + 1
    assert cloud_count <= config["keep_backup"] + 1
    summary = {"status": "passed", "verification": result["verification"],
               "backup_backend": args.backup_backend, "backup_mount_identity": mount_identity,
               "gpu": torch.cuda.get_device_name(), "run_id": args.run_id,
               "paused_step": 3, "killed_step": killed_step,
               "recovered_final_step": result["global_step"],
               "exact_cuda_training_state_match": True,
               "validation_epochs": len(final["control"]["validation_history"]),
               "validation_subset_episodes": 8, "best": result["best"],
               "local_snapshots": local_count, "backup_snapshots": cloud_count,
               "backup_run": str(backup), "evidence": str(evidence)}
    check_backup()
    atomic_json(backup / "acceptance.json", summary)
    check_backup()
    atomic_json(evidence / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
