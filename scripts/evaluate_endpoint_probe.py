#!/usr/bin/env python3
"""Complete DUET navigation with a fixed final historical-endpoint scorer."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vln_improve.protocol import file_sha256, object_sha256, resolve_config
from vln_improve.study_ledger import StudyLedger

CODE_FILES = ("scripts/evaluate_endpoint_probe.py", "scripts/run_duet.py", "scripts/prepare_duet.py",
              "scripts/collect_stopping_traces.py", "src/vln_improve/stopping_traces.py",
              "src/vln_improve/stopping_diagnostics.py", "src/vln_improve/diagnostics.py",
              "src/vln_improve/pipeline.py",
              "src/vln_improve/endpoint_policy.py", "src/vln_improve/endpoint_probe.py",
              "src/vln_improve/protocol.py", "src/vln_improve/study_ledger.py",
              "src/vln_improve/features.py", "src/vln_improve/capture.py", "src/vln_improve/head.py")


def validate_registration(args, reference, code_sha):
    if args.split != "val_unseen":
        if args.access_id or args.ledger:
            raise ValueError("training-scene evaluation must not use a validation access")
        return None
    if not args.head or args.limit is not None or not args.access_id or not args.ledger:
        raise ValueError("official validation requires a frozen head, full split and registration")
    study = json.loads(args.study.read_text())
    access = StudyLedger(args.ledger, study).lookup(args.access_id)
    if access["outcome"] is not None:
        raise ValueError("completed validation access cannot be rerun")
    expected = {"category": "pilot", "split": "val_unseen", "label_use": "evaluation",
                "parameter_fitting_split": "train_fit", "subset": False, "subset_ids": [],
                "config_sha256": file_sha256(args.experiment), "checkpoint_sha256": file_sha256(args.head),
                "code_sha256": code_sha, "seed": args.seed, "expected_episodes": len(reference["episodes"])}
    if any(access["registration"]["request"].get(k) != v for k, v in expected.items()):
        raise ValueError("registered validation protocol differs from this evaluation")
    return access["registration"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true")
    parser.add_argument("--head", type=Path, help="Omit only for training-scene identity smoke")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--experiment", type=Path, default=ROOT / "configs/endpoint_probe_pilot.json")
    parser.add_argument("--split", choices=("train_dev", "val_unseen"))
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/research_study.json")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--access-id")
    args = parser.parse_args()
    code_files = {n: file_sha256(ROOT / n) for n in CODE_FILES}
    code_sha = object_sha256(code_files)
    if args.print_code_sha256:
        print(code_sha); return
    if any(x is None for x in (args.split, args.baseline_report, args.output, args.backup)):
        parser.error("split, baseline-report, output, backup are required")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.output.exists():
        raise ValueError("refusing to overwrite an evaluation")
    from vln_improve.pipeline import atomic_json, validate_backup_root
    validate_backup_root(args.backup)
    args.backup.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = resolve_config(args.config, ROOT)
    reference = json.loads(args.baseline_report.read_text())
    registration = validate_registration(args, reference, code_sha)
    from collect_stopping_traces import validate_reference
    validate_reference(reference, cfg, split=args.split, seed=args.seed,
                       checkpoint_sha256=file_sha256(cfg["base_checkpoint"]),
                       expected_episodes=len(reference["episodes"]))
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
        expected_training = {"epochs": experiment["epochs"], "batch_episodes": experiment["batch_episodes"],
                             "seed": experiment["seed"], "lr": experiment["optimizer"]["lr"],
                             "weight_decay": experiment["optimizer"]["weight_decay"],
                             "optimizer": experiment["optimizer"]["name"]}
        if any(head_meta["train_config"].get(k) != v for k, v in expected_training.items()):
            raise ValueError("head training settings differ from the registered experiment")
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
                        output=raw, cache=None, head=None, limit=args.limit, seed=args.seed)
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
    report["metadata"].update({"mode": "endpoint_probe" if head is not None else "endpoint_identity",
          "endpoint_only": True, "head_sha256": file_sha256(args.head) if args.head else None,
          "head_metadata": head_meta, "endpoint_experiment_sha256": file_sha256(args.experiment),
          "endpoint_code_sha256": code_sha, "endpoint_code_files": code_files,
          "baseline_report_sha256": file_sha256(args.baseline_report), "access_id": args.access_id,
          "registration_sha256": object_sha256(registration) if registration else None,
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
    print(json.dumps({"output": str(args.output), "summary": report["summary"],
                      "endpoint_changes": sum(x["endpoint_changed"] for x in completed.values()),
                      "all_online_path_and_termination_parity": True}, indent=2))


if __name__ == "__main__":
    main()
