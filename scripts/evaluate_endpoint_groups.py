#!/usr/bin/env python3
"""Evaluate a fixed final head from the matched endpoint-group experiment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vln_improve.protocol import file_sha256, object_sha256, resolve_config
from vln_improve.study_ledger import StudyLedger

CODE_FILES = ("scripts/evaluate_endpoint_groups.py",
              "scripts/train_endpoint_groups.py",
              "src/vln_improve/endpoint_group_training.py", "src/vln_improve/endpoint_group_objectives.py",
              "src/vln_improve/endpoint_pair_training.py", "src/vln_improve/endpoint_controls.py",
              "src/vln_improve/endpoint_pairs.py", "src/vln_improve/checkpoint_store.py", "src/vln_improve/resumable.py", "scripts/run_duet.py", "scripts/prepare_duet.py",
              "scripts/collect_stopping_traces.py", "src/vln_improve/stopping_traces.py",
              "src/vln_improve/stopping_diagnostics.py", "src/vln_improve/diagnostics.py",
              "src/vln_improve/pipeline.py",
              "src/vln_improve/endpoint_policy.py", "src/vln_improve/endpoint_probe.py",
              "src/vln_improve/protocol.py", "src/vln_improve/study_ledger.py",
              "src/vln_improve/features.py", "src/vln_improve/capture.py", "src/vln_improve/head.py")


def validate_registration(args, code_sha):
    """Gate official validation without opening its baseline/results files."""
    if args.split != "val_unseen":
        if args.access_id or args.ledger:
            raise ValueError("training-scene evaluation must not use a validation access")
        return None
    if (not args.head or args.limit is not None or not args.access_id or not args.ledger
            or not args.execution_backup_root):
        raise ValueError("official validation requires a frozen head, full split, registration and execution backup root")
    study = json.loads(args.study.read_text())
    if study["baseline"]["selection_episodes"] != 2349:
        raise ValueError("study does not describe the complete official validation split")
    access = StudyLedger(args.ledger, study).lookup(args.access_id)
    if access["outcome"] is not None:
        raise ValueError("completed validation access cannot be rerun")
    expected = {"category": args.category, "split": "val_unseen", "label_use": "evaluation",
                "parameter_fitting_split": "train_fit", "subset": False, "subset_ids": [],
                "config_sha256": file_sha256(args.experiment), "checkpoint_sha256": file_sha256(args.head),
                "code_sha256": code_sha, "seed": args.seed, "expected_episodes": 2349}
    if any(access["registration"]["request"].get(k) != v for k, v in expected.items()):
        raise ValueError("registered validation protocol differs from this evaluation")
    return access["registration"]


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(fd)
        except OSError as error:
            # Drive FUSE can omit directory fsync. File fsync and SHA read-back
            # remain mandatory; genuine I/O errors must still abort the run.
            if error.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EBADF}:
                raise
    finally:
        os.close(fd)


def _exclusive_json(path, payload):
    """Create once and persist. Even an interrupted/partial file blocks reuse."""
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _exclusive_bytes(path, raw)


def _exclusive_bytes(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def execution_claim_path(ledger_path, access_id):
    """Sidecars share the canonical ledger location, independent of output paths."""
    ledger = Path(ledger_path).resolve(strict=True)
    root = ledger.with_name(ledger.name + ".executions")
    return root / (object_sha256(access_id) + ".claim.json")


def cloud_execution_claim_path(backup_root, registration):
    # Stable across VM replacements, ledger copies, and output renames. The
    # supplied backup root must be the same study-level Drive directory.
    key = object_sha256([registration["study_id"], registration["access_id"]])
    return Path(backup_root) / (key + ".claim.json")


def _verified_copy_once(source, target):
    raw = source.read_bytes()
    _exclusive_bytes(target, raw)
    if file_sha256(target) != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"execution backup read-back failed: {target}")


@contextmanager
def validation_execution(args, registration, code_sha):
    """Permanently claim one registered run; leave ledger closeout to its owner.

    A completed/failed sidecar records this process's outcome. A started claim
    without an outcome records an interrupted or still-running attempt. All
    three states block another run, including a different output path. Keep
    these sidecars with the ledger; never delete a claim to retry an access.
    ``register_study_access.py complete/fail`` remains the sole ledger closeout.
    """
    if args.split != "val_unseen":
        if registration is not None:
            raise ValueError("training-scene evaluation cannot claim a validation access")
        yield None
        return
    # Recheck before claiming; callers cannot supply a stale or fabricated row.
    if registration is None or validate_registration(args, code_sha) != registration:
        raise ValueError("validation registration changed before execution")
    from vln_improve.pipeline import validate_backup_root
    validate_backup_root(args.execution_backup_root)
    cloud_path = cloud_execution_claim_path(args.execution_backup_root, registration)
    cloud_path.parent.mkdir(parents=True, exist_ok=True)
    if cloud_path.exists() or cloud_path.is_symlink():
        raise ValueError("validation access already has a cloud execution claim; reruns need a new access ID")
    path = execution_claim_path(args.ledger, args.access_id)
    path.parent.mkdir(exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("execution claims require an ordinary ledger sidecar directory")
    _fsync_directory(path.parent.parent)
    ledger = Path(args.ledger).resolve()
    ledger_bytes = ledger.read_bytes()
    ledger_fingerprint = object_sha256({"study_id": registration["study_id"], "canonical_path": str(ledger)})
    payload = {"schema": "endpoint_group_validation_execution_v1", "status": "started",
               "access_id": args.access_id, "study_id": registration["study_id"], "ledger": str(ledger),
               "ledger_fingerprint": ledger_fingerprint,
               "ledger_snapshot_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
               "registration_sha256": object_sha256(registration), "code_sha256": code_sha,
               "output": str(args.output.resolve()), "backup": str(args.backup.resolve()),
               "baseline_report": str(args.baseline_report.resolve()), "cloud_claim": str(cloud_path),
               "started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    try:
        _exclusive_json(path, payload)
    except FileExistsError as error:
        raise ValueError("validation access already has an execution claim; reruns need a new access ID") from error
    claim = {"path": str(path), "sha256": file_sha256(path),
             "outcome_path": str(path.with_name(path.name.removesuffix(".claim.json") + ".outcome.json")),
             "cloud_path": str(cloud_path), "ledger_fingerprint": ledger_fingerprint,
             "ledger_snapshot_sha256": payload["ledger_snapshot_sha256"]}
    started = time.monotonic()
    cloud_claim_created = False

    def outcome(status, **extra):
        local_outcome = Path(claim["outcome_path"])
        _exclusive_json(local_outcome, {
            "schema": "endpoint_group_validation_execution_outcome_v1", "status": status,
            "access_id": args.access_id, "claim_sha256": claim["sha256"],
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started, **extra})
        if cloud_claim_created:
            validate_backup_root(args.execution_backup_root)
            _verified_copy_once(local_outcome, cloud_path.with_name(cloud_path.name.removesuffix(".claim.json") + ".outcome.json"))

    try:
        try:
            _verified_copy_once(path, cloud_path)
            cloud_claim_created = True
        except FileExistsError as error:
            raise ValueError("validation access already has a cloud execution claim; reruns need a new access ID") from error
        snapshot = path.with_name(path.name.removesuffix(".claim.json") + ".ledger.jsonl")
        _exclusive_bytes(snapshot, ledger_bytes)
        _verified_copy_once(snapshot, cloud_path.with_name(cloud_path.name.removesuffix(".claim.json") + ".ledger.jsonl"))
        # If the owner closed the ledger during setup, do not start navigation.
        if validate_registration(args, code_sha) != registration:
            raise ValueError("validation registration changed after execution claim")
        yield claim
        outcome("completed", report=str(args.output.resolve()), report_sha256=file_sha256(args.output))
    except BaseException as error:
        try:
            outcome("failed", error_type=type(error).__name__, error=str(error))
        except BaseException as recording_error:
            note = f"Execution outcome could not be recorded: {recording_error}; permanent claim remains at {path}"
            if hasattr(error, "add_note"):
                error.add_note(note)
            else:
                print(note, file=sys.stderr)
        raise


def validate_group_head_metadata(meta, experiment, seed, experiment_sha):
    """Reject intermediate, differently configured, or unbound experiment heads."""
    from train_endpoint_groups import validate_experiment
    from vln_improve.endpoint_group_training import training_code_identity
    arm = validate_experiment(experiment)
    expected = {"epochs": 20, "batch_groups": 8, "seed": seed, "arm": arm,
                "lr": experiment["optimizer"]["lr"], "weight_decay": experiment["optimizer"]["weight_decay"],
                "optimizer": experiment["optimizer"]["name"], "experiment_sha256": experiment_sha,
                "pair_weight": experiment["pair_weight"], "natural_weight": .5, "augmentation_weight": .5,
                "max_states": 15, "feature_dim": 1536, "hidden_dim": 128, "activation": "relu",
                "monitor_every": 5, "primary_checkpoint": "fixed_final_epoch",
                "engineering_best_checkpoint": experiment["engineering_best_monitor"]}
    if (any(meta.get("train_config", {}).get(k) != v for k, v in expected.items())
            or meta.get("epoch") != 20 or meta.get("global_step") != 1280
            or meta.get("pending_dev") is not False or meta.get("selection_purpose") != "fixed_final_epoch"):
        raise ValueError("head is not the registered fully trained final checkpoint")
    if meta.get("training_code_identity") != training_code_identity():
        raise ValueError("training implementation identity changed or is incomplete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true")
    parser.add_argument("--head", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--category", choices=("pilot", "confirmatory"), default="pilot")
    parser.add_argument("--split", choices=("train_dev", "val_unseen"))
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0, help="Head training seed; navigation seed stays zero")
    parser.add_argument("--study", type=Path, default=ROOT / "configs/research_study.json")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--access-id")
    parser.add_argument("--execution-backup-root", type=Path,
                        help="Required for val_unseen: the shared study-level Drive execution-claim directory")
    args = parser.parse_args(argv)
    code_files = {n: file_sha256(ROOT / n) for n in CODE_FILES}
    code_sha = object_sha256(code_files)
    if args.print_code_sha256:
        print(code_sha); return
    if any(x is None for x in (args.head, args.experiment, args.split, args.baseline_report, args.output, args.backup)):
        parser.error("head, experiment, split, baseline-report, output, backup are required")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.output.exists():
        raise ValueError("refusing to overwrite an evaluation")
    # This gate runs before baseline read/hash, head loading, or navigation.
    registration = validate_registration(args, code_sha)
    from vln_improve.pipeline import validate_backup_root
    validate_backup_root(args.backup)
    args.backup.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists():
            raise ValueError("refusing to overwrite an evaluation")
        with validation_execution(args, registration, code_sha) as claim:
            report = run_evaluation(args, code_files, code_sha, registration, claim)
    print(json.dumps({"output": str(args.output), "summary": report["summary"],
                      "endpoint_changes": sum(x["endpoint_changed"] for x in report["endpoint_decisions"]),
                      "all_online_path_and_termination_parity": True}, indent=2))
    return report


def run_evaluation(args, code_files, code_sha, registration, claim):
    """Run only after main has checked registration and claimed its access."""
    from vln_improve.pipeline import atomic_json, validate_backup_root
    cfg = resolve_config(args.config, ROOT)
    reference = json.loads(args.baseline_report.read_text())
    from collect_stopping_traces import validate_reference
    validate_reference(reference, cfg, split=args.split, seed=0,
                       checkpoint_sha256=file_sha256(cfg["base_checkpoint"]),
                       expected_episodes={"train_dev": 2890, "val_unseen": 2349}[args.split])
    baseline = {r["instr_id"]: r["trajectory"] for r in reference["trajectories"]}
    if len(baseline) != len(reference["episodes"]):
        raise ValueError("baseline instruction inventory mismatch")
    import torch
    from vln_improve.endpoint_probe import load_endpoint_head
    from vln_improve.endpoint_policy import EndpointReranker
    import run_duet
    source_lock = run_duet.verify()
    sys.path.insert(0, str(run_duet.DEFAULT_DEST / "map_nav_src"))
    import r2r.agent as upstream
    head, head_meta = (None, None) if args.head is None else load_endpoint_head(args.head, device="cuda")
    if head_meta is not None:
        experiment = json.loads(args.experiment.read_text())
        validate_group_head_metadata(head_meta, experiment, args.seed, file_sha256(args.experiment))
        dataset = Path(cfg["dataset_root"])
        expected_common = {
            "base_checkpoint_sha256": file_sha256(cfg["base_checkpoint"]),
            "feature_sha256": file_sha256(dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"),
            "annotation_sha256": file_sha256(dataset / "R2R/annotations/R2R_train_enc.json"),
            "connectivity_sha256": object_sha256({p.name: file_sha256(p) for p in sorted((dataset / "R2R/connectivity").glob("*_connectivity.json"))}),
            "model": cfg["model"], "upstream_lock": source_lock, "partition_seed": cfg["partition_seed"],
            "dev_fraction": cfg["dev_fraction"], "torch_version": str(torch.__version__)}
        if head_meta["common_identity"]["common_provenance"] != expected_common:
            raise ValueError("head and navigation backbone/data/feature provenance differ")
    base_class, old_parse = upstream.GMapNavAgent, run_duet.parse_cli
    completed = {}

    class RerankingAgent(base_class):
        def rollout(self, *positional, **kwargs):
            if positional or getattr(self, "decision_hook", None) is not None:
                raise ValueError("endpoint evaluation conflicts with another policy hook")
            ranker = EndpointReranker(self, head, baseline)
            old_move = self.make_equiv_action
            self.decision_hook, self.make_equiv_action = ranker, ranker.make_equiv_action
            try:
                result = super().rollout(**kwargs)
            finally:
                self.decision_hook, self.make_equiv_action = None, old_move
            if not ranker.completed or len(result) != 1 or result[0]["instr_id"] != ranker.instr_id:
                raise ValueError("endpoint rollout did not complete one expected instruction")
            # Upstream test can repeat one item when wrapping its final batch.
            if ranker.instr_id in completed and completed[ranker.instr_id] != ranker.result:
                raise ValueError("repeated endpoint rollout changed its result")
            completed[ranker.instr_id] = ranker.result
            return result

    raw = args.output.with_name(args.output.stem + ".raw.json")
    try:
        upstream.GMapNavAgent = RerankingAgent
        run_duet.parse_cli = lambda: SimpleNamespace(mode="baseline", config=args.config, split=args.split,
                        output=raw, cache=None, head=None, limit=args.limit, seed=0)
        run_duet.main()
    finally:
        upstream.GMapNavAgent, run_duet.parse_cli = base_class, old_parse
    report = json.loads(raw.read_text())
    expected_ids = set(sorted(baseline)[:args.limit]) if args.limit is not None else set(baseline)
    actual_ids = {r["instr_id"] for r in report["episodes"]}
    trajectory_ids = {r["instr_id"] for r in report["trajectories"]}
    if (set(completed) != actual_ids or actual_ids != expected_ids or trajectory_ids != expected_ids
            or len(report["episodes"]) != len(expected_ids) or len(report["trajectories"]) != len(expected_ids)):
        raise ValueError("endpoint evaluation lost an instruction")
    if head is None and any(r["trajectory"] != baseline[r["instr_id"]] for r in report["trajectories"]):
        raise ValueError("endpoint identity mode changed complete trajectories")
    report["metadata"].update({"mode": "endpoint_group", "group_arm": head_meta["train_config"]["arm"],
          "head_training_seed": args.seed, "navigation_seed": 0, "checkpoint_selection": "fixed_final_epoch",
          "endpoint_only": True, "head_sha256": file_sha256(args.head) if args.head else None,
          "head_metadata": head_meta, "endpoint_experiment_sha256": file_sha256(args.experiment),
          "endpoint_code_sha256": code_sha, "endpoint_code_files": code_files,
          "baseline_report_sha256": file_sha256(args.baseline_report), "access_id": args.access_id,
          "registration_sha256": object_sha256(registration) if registration else None,
          "validation_execution": claim,
          "all_online_path_and_termination_parity": True,
          "historical_score": "goal_head_logit" if head else "baseline_stop_probability",
          "scope": "complete navigation with original moves/online stopping and fully charged historical return"})
    report["endpoint_decisions"] = [completed[k] for k in sorted(completed)]
    atomic_json(args.output, report)
    validate_backup_root(args.backup)
    target = args.backup / args.output.name
    if target.exists() and file_sha256(target) != file_sha256(args.output):
        raise ValueError("cloud evaluation already exists with different bytes")
    atomic_json(target, report)
    if file_sha256(target) != file_sha256(args.output):
        raise ValueError("evaluation Drive read-back failed")
    return report


if __name__ == "__main__":
    main()
