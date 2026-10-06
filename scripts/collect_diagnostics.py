#!/usr/bin/env python3
"""Collect baseline-only FP32 evidence/arrival episodes without modifying DUET."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vln_improve.diagnostics import (
    DiagnosticObserver, DiagnosticStore, atomic_json, select_diagnostic_records,
)
from vln_improve.pipeline import validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256, resolve_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--split", choices=["train_fit", "train_dev"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="Episode collection directory")
    parser.add_argument("--backup", type=Path, help="Persistent Drive collection directory")
    parser.add_argument("--per-scan", type=int, default=4, help="Distinct paths per scan, one instruction per path")
    parser.add_argument("--max-scans", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-parity", action="store_true", help="Also run plain baseline on this exact selection")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args(argv)
    config = resolve_config(args.config, ROOT)
    model = config["model"]
    if model["batch_size"] != 1 or model.get("act_visited_nodes", False) or not model.get("enc_full_graph", False):
        parser.error("M0 requires batch_size=1, enc_full_graph=true, act_visited_nodes=false")
    if not 1 <= model["max_action_len"] <= 15:
        parser.error("M0 supports max_action_len from 1 to 15")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    verification = "local-only-no-cloud-backup"
    checker = None
    if args.backup is not None:
        checker = lambda: validate_backup_root(args.backup, allow_local=args.allow_local_backup_for_tests)
        verification = checker()
    elif args.allow_local_backup_for_tests:
        parser.error("--allow-local-backup-for-tests requires --backup")

    import run_duet
    lock = run_duet.verify()  # Preserve the existing pinned-source gate.
    sys.path.insert(0, str(run_duet.DEFAULT_DEST / "map_nav_src"))
    import r2r.agent as upstream_agent
    import torch

    original_agent = upstream_agent.GMapNavAgent
    original_selector = run_duet.select_partition
    original_parser = run_duet.parse_cli
    original_hash = run_duet.file_sha256
    hash_cache = {}
    state = {}
    observers = []
    invocation = uuid.uuid4().hex
    args.output.mkdir(parents=True, exist_ok=True)

    def cached_sha(path):
        key = str(Path(path).resolve())
        if key not in hash_cache:
            hash_cache[key] = file_sha256(path)
        return hash_cache[key]

    def selector(records, split, dev_fraction, partition_seed):
        partition = original_selector(records, split, dev_fraction, partition_seed)
        selected = select_diagnostic_records(partition, per_scan=args.per_scan, seed=args.seed,
                                             max_scans=args.max_scans, max_episodes=args.max_episodes)
        if not selected:
            raise ValueError("diagnostic selection is empty")
        selection = [{"scan": row["scan"], "instr_id": row["instr_id"], "path_id": str(row["path_id"])}
                     for row in sorted(selected, key=lambda row: row["instr_id"])]
        dataset = Path(config["dataset_root"])
        identity = {
            "schema": "duet_diagnostic_collection_v1", "split": split,
            "usage": "training_diagnostics" if split == "train_fit" else "analysis_only",
            "model": model, "seed": args.seed, "max_sources": 15,
            "dev_fraction": dev_fraction, "partition_seed": partition_seed,
            "selection": selection, "selection_sha256": object_sha256(selection),
            "sampling": {"per_scan": args.per_scan, "max_scans": args.max_scans,
                         "max_episodes": args.max_episodes, "one_instruction_per_path": True},
            "base_checkpoint_sha256": cached_sha(config["base_checkpoint"]),
            "feature_sha256": cached_sha(dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"),
            "annotation_sha256": cached_sha(dataset / "R2R/annotations/R2R_train_enc.json"),
            "connectivity_sha256": object_sha256({path.name: cached_sha(path)
                                                  for path in sorted((dataset / "R2R/connectivity").glob("*_connectivity.json"))}),
            "upstream_lock": lock,
            "implementation_sha256": object_sha256({name: cached_sha(ROOT / name) for name in (
                "scripts/run_duet.py", "scripts/collect_diagnostics.py", "src/vln_improve/diagnostics.py",
                "src/vln_improve/evidence.py", "src/vln_improve/protocol.py")}),
            "torch_version": str(torch.__version__),
        }
        if "store" not in state:
            state["store"] = DiagnosticStore(args.output, identity, backup=args.backup, verify_backup=checker)
            state["selection"] = selection
        elif state["store"].identity != identity:
            raise ValueError("diagnostic identity changed within one invocation")
        return selected

    class CollectingAgent(original_agent):
        def __init__(self, *agent_args, **agent_kwargs):
            super().__init__(*agent_args, **agent_kwargs)
            observers.append(DiagnosticObserver(self, state["store"], split=args.split))

    def run(report, collect):
        upstream_agent.GMapNavAgent = CollectingAgent if collect else original_agent
        run_duet.parse_cli = lambda: SimpleNamespace(
            mode="baseline", config=args.config, split=args.split, output=report,
            cache=None, head=None, limit=None, seed=args.seed,
        )
        run_duet.main()
        value = json.loads(report.read_bytes())
        value["metadata"].update({"diagnostic_collection": True, "subset": True,
                                  "scope": "diagnostic", "usage": state["store"].identity["usage"],
                                  "selection_sha256": state["store"].identity["selection_sha256"]})
        atomic_json(report, value)
        return value

    try:
        run_duet.select_partition = selector
        run_duet.file_sha256 = cached_sha
        reference = None
        if args.check_parity:
            reference = run(args.output / f"baseline-parity-{invocation}.json", False)
            gc.collect()
            torch.cuda.empty_cache()
        report = args.output / f"collection-rollout-{invocation}.json"
        result = run(report, True)
        parity = None
        if reference is not None:
            parity = reference["trajectories"] == result["trajectories"] and reference["episodes"] == result["episodes"]
            if not parity:
                raise AssertionError("diagnostic observer changed trajectories or official per-episode metrics")
        manifests = [state["store"].find(row["scan"], row["instr_id"]) for row in state["selection"]]
        summary = {
            "schema": "duet_diagnostic_summary_v1", "status": "complete",
            "split": args.split, "usage": state["store"].identity["usage"],
            "identity_sha256": state["store"].identity_sha256,
            "num_episodes": len(manifests), "num_scans": len({item["scan_id"] for item in manifests}),
            "num_states": sum(item["num_states"] for item in manifests),
            "num_eligible_states": sum(item["coverage"]["num_eligible_states"] for item in manifests),
            "num_candidate_states": sum(item["coverage"]["num_candidate_states"] for item in manifests),
            "num_multi_source_candidate_states": sum(item["coverage"]["num_multi_source_candidate_states"] for item in manifests),
            "num_states_with_multi_source": sum(item["coverage"]["num_states_with_multi_source"] for item in manifests),
            "num_natural_arrival_pairs": sum(item["coverage"]["num_pairs"] for item in manifests),
            "new_episodes": sum(observer.new_episodes for observer in observers),
            "reused_rollouts": sum(observer.reused_episodes for observer in observers),
            "observer_baseline_trajectory_parity": parity, "verification": verification,
            "rollout_report": str(report), "scope": "collection_and_coverage_only",
            "future_label_replay_implemented": False,
        }
        atomic_json(args.output / "collection-summary.json", summary)
        if args.backup is not None:
            checker()
            atomic_json(args.backup / "collection-summary.json", summary)
            if json.loads((args.backup / "collection-summary.json").read_bytes()) != summary:
                raise ValueError("collection summary cloud read-back mismatch")
        print(json.dumps(summary, indent=2))
    finally:
        for observer in observers:
            observer.close()
        upstream_agent.GMapNavAgent = original_agent
        run_duet.select_partition = original_selector
        run_duet.parse_cli = original_parser
        run_duet.file_sha256 = original_hash


if __name__ == "__main__":
    main()
