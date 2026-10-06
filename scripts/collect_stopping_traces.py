#!/usr/bin/env python3
"""Collect small, read-only full-split STOP traces with exact baseline parity."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vln_improve.diagnostics import atomic_json
from vln_improve.pipeline import validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256, resolve_config
from vln_improve.stopping_traces import StopTraceObserver, StopTraceStore, exact_report_parity, index_report, summarize_store
from vln_improve.study_ledger import StudyLedger


CODE_FILES = (
    "scripts/collect_stopping_traces.py", "src/vln_improve/stopping_traces.py",
    "src/vln_improve/stopping_diagnostics.py", "scripts/run_duet.py", "scripts/prepare_duet.py",
    "src/vln_improve/diagnostics.py", "src/vln_improve/protocol.py", "src/vln_improve/study_ledger.py",
    "src/vln_improve/pipeline.py", "src/vln_improve/features.py", "src/vln_improve/head.py", "src/vln_improve/capture.py",
)


def trace_code_identity():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def validate_access(*, split, access_id, ledger_path, study_path, diagnostic_config, spec,
                    checkpoint_sha256, code_sha256, seed):
    """Read the existing registration; never register, finish, or fit here."""
    if split != "val_unseen":
        if access_id is not None or ledger_path is not None:
            raise ValueError("training-split analysis must not consume a validation registration")
        return None
    if not access_id or ledger_path is None or not Path(ledger_path).is_file():
        raise ValueError("val_unseen requires a pre-registered --access-id and existing --ledger")
    study = json.loads(Path(study_path).read_bytes())
    access = StudyLedger(Path(ledger_path), study).lookup(access_id)
    registration, outcome = access["registration"], access["outcome"]
    if outcome is not None:
        raise ValueError("validation access has a terminal outcome; a rerun needs separate registration")
    request = registration["request"]
    expected = {"category": "diagnostic_analysis", "split": "val_unseen", "label_use": "analysis",
                "parameter_fitting_split": "train_fit", "subset": False, "subset_ids": [],
                "config_sha256": file_sha256(diagnostic_config), "checkpoint_sha256": checkpoint_sha256,
                "code_sha256": code_sha256, "seed": seed, "expected_episodes": spec["episodes"]}
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("validation registration differs from this fixed diagnostic/config/code/checkpoint")
    if checkpoint_sha256 != study["baseline"]["base_checkpoint_sha256"]:
        raise ValueError("diagnostic checkpoint differs from the study baseline")
    return registration


def validate_reference(report, config, *, split, seed, checkpoint_sha256, expected_episodes):
    metadata = report.get("metadata", {})
    expected = {"dataset": "r2r", "split": split, "seed": seed, "mode": "baseline", "feedback": "argmax",
                "subset": False, "base_checkpoint_sha256": checkpoint_sha256,
                "model": config["model"], "partition_seed": config["partition_seed"],
                "dev_fraction": config["dev_fraction"], "num_episodes": expected_episodes}
    if any(metadata.get(key) != value for key, value in expected.items()) or "head_sha256" in metadata:
        raise ValueError("reference is not the matching frozen, full-split baseline/config")
    metrics, _ = index_report(report)
    if len(metrics) != expected_episodes:
        raise ValueError("reference episode count differs from registered full split")


@contextmanager
def output_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".collector.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another STOP collector owns this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true", help="Print the exact multi-file code identity before registration; no data/GPU access")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--diagnostic-config", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--split", choices=("train_fit", "train_dev", "val_unseen"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--study", type=Path, default=ROOT / "configs/research_study.json")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--access-id")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args(argv)
    code_files = trace_code_identity()
    code_sha = object_sha256(code_files)
    if args.print_code_sha256:
        print(code_sha)
        return
    if any(getattr(args, key) is None for key in ("diagnostic_config", "baseline_report", "split", "output", "backup")):
        parser.error("--diagnostic-config, --baseline-report, --split, --output and --backup are required")
    config = resolve_config(args.config, ROOT)
    model = config["model"]
    if (model.get("batch_size") != 1 or model.get("act_visited_nodes", False) or not model.get("enc_full_graph")
            or model.get("fusion") != "dynamic" or not 1 <= model["max_action_len"] <= 15 or args.seed < 0):
        parser.error("requires batch-one, full graph, dynamic fusion, unvisited-only actions, maxlen<=15 and nonnegative seed")
    spec = json.loads(args.diagnostic_config.read_bytes())
    checkpoint_sha = file_sha256(config["base_checkpoint"])
    if (spec.get("schema_version") != 1 or spec.get("split") != args.split or spec.get("seed") != args.seed
            or type(spec.get("episodes")) is not int or spec["episodes"] < 1
            or spec.get("base_checkpoint_sha256") != checkpoint_sha):
        raise ValueError("diagnostic specification differs from the requested run")
    # Gate validation use before reading the baseline's detailed results.
    registration = validate_access(split=args.split, access_id=args.access_id, ledger_path=args.ledger,
                                   study_path=args.study, diagnostic_config=args.diagnostic_config, spec=spec,
                                   checkpoint_sha256=checkpoint_sha, code_sha256=code_sha, seed=args.seed)
    baseline_sha = file_sha256(args.baseline_report)
    if spec.get("baseline_report_sha256") != baseline_sha:
        raise ValueError("baseline report differs from the diagnostic specification")
    reference = json.loads(args.baseline_report.read_bytes())
    validate_reference(reference, config, split=args.split, seed=args.seed,
                       checkpoint_sha256=checkpoint_sha, expected_episodes=spec["episodes"])
    checker = lambda: validate_backup_root(args.backup, allow_local=args.allow_local_backup_for_tests)
    verification = checker()
    with output_lock(args.output):
        run_collection(args, config, spec, reference, registration, code_files, checker, verification)


def run_collection(args, config, spec, reference, registration, code_files, checker, verification):
    import run_duet
    lock = run_duet.verify()
    sys.path.insert(0, str(run_duet.DEFAULT_DEST / "map_nav_src"))
    import r2r.agent as upstream_agent

    original_agent, original_selector = upstream_agent.GMapNavAgent, run_duet.select_partition
    original_parser, original_hash = run_duet.parse_cli, run_duet.file_sha256
    hashes, observers, state = {}, [], {}
    def cached_sha(path):
        key = str(Path(path).resolve())
        if key not in hashes:
            hashes[key] = file_sha256(path)
        return hashes[key]
    dataset = Path(config["dataset_root"])
    raw_split = "train" if args.split.startswith("train_") else args.split
    assets = {
        "base_checkpoint_sha256": cached_sha(config["base_checkpoint"]),
        "feature_id": cached_sha(dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"),
        "train_annotation_sha256": cached_sha(dataset / "R2R/annotations/R2R_train_enc.json"),
        "connectivity_sha256": object_sha256({path.name: cached_sha(path) for path in sorted((dataset / "R2R/connectivity").glob("*_connectivity.json"))}),
        "upstream_commit": lock["commit"],
    }
    if any(reference["metadata"].get(key) != value for key, value in assets.items()):
        raise ValueError("runtime assets/source differ from the recorded baseline")
    baseline_metrics, _ = index_report(reference)

    def selector(records, split, dev_fraction, partition_seed):
        selected = original_selector(records, split, dev_fraction, partition_seed)
        selection = [{"scan": row["scan"], "instr_id": row["instr_id"], "path_id": str(row["path_id"])}
                     for row in sorted(selected, key=lambda row: row["instr_id"])]
        if (len(selection) != spec["episodes"] or {row["instr_id"] for row in selection} != set(baseline_metrics)
                or any(baseline_metrics[row["instr_id"]]["scan_id"] != row["scan"] for row in selection)):
            raise ValueError("actual full split differs from registered baseline instructions/scans")
        identity = {"schema": "duet_stopping_trace_collection_v1", "usage": "analysis_only", "split": split,
                    "selection": selection, "selection_sha256": object_sha256(selection), "model": config["model"],
                    "seed": args.seed, "runtime_config_sha256": cached_sha(args.config),
                    "diagnostic_config_sha256": cached_sha(args.diagnostic_config),
                    "baseline_report_sha256": cached_sha(args.baseline_report),
                    "annotation_sha256": cached_sha(dataset / f"R2R/annotations/R2R_{raw_split}_enc.json"),
                    "assets": assets, "upstream_lock": lock, "code_files": code_files,
                    "code_sha256": object_sha256(code_files), "access_id": args.access_id,
                    "registration_sha256": object_sha256(registration) if registration else None}
        if "store" not in state:
            state["store"] = StopTraceStore(args.output, identity, reference, backup=args.backup, verify_backup=checker)
        elif state["store"].identity != identity:
            raise ValueError("STOP collection identity changed during one invocation")
        return selected

    class CollectingAgent(original_agent):
        def __init__(self, *agent_args, **agent_kwargs):
            super().__init__(*agent_args, **agent_kwargs)
            observers.append(StopTraceObserver(self, state["store"]))

    report_path = args.output / f"collection-rollout-{uuid.uuid4().hex}.json"
    try:
        upstream_agent.GMapNavAgent = CollectingAgent
        run_duet.select_partition, run_duet.file_sha256 = selector, cached_sha
        run_duet.parse_cli = lambda: SimpleNamespace(mode="baseline", config=args.config, split=args.split,
                                                    output=report_path, cache=None, head=None, limit=None, seed=args.seed)
        run_duet.main()
        actual = json.loads(report_path.read_bytes())
        exact_report_parity(reference, actual)
        analysis = summarize_store(state["store"])
        summary = {"schema": "duet_stopping_trace_summary_v1", "status": "complete", "usage": "analysis_only",
                   "split": args.split, "identity_sha256": state["store"].identity_sha256,
                   "num_episodes": analysis["overall"]["episodes"], "num_states": analysis["overall"]["states"],
                   "new_episodes": sum(o.new_episodes for o in observers), "reused_rollouts": sum(o.reused_episodes for o in observers),
                   "all_instruction_trajectory_and_metric_parity": True, "verification": verification,
                   "resources": actual["resources"], "access_id": args.access_id,
                   "analysis_sha256": object_sha256(analysis), "rollout_report": report_path.name}
        for name, payload in (("stopping-analysis.json", analysis), ("collection-summary.json", summary), (report_path.name, actual)):
            atomic_json(args.output / name, payload)
            checker()
            atomic_json(args.backup / name, payload)
            if json.loads((args.backup / name).read_bytes()) != payload:
                raise ValueError("STOP final artifact Drive read-back mismatch")
        print(json.dumps({"summary": summary, "overall": analysis["overall"], "opportunity_support": analysis["opportunity_support"]}, indent=2))
    finally:
        for observer in observers:
            observer.close()
        upstream_agent.GMapNavAgent = original_agent
        run_duet.select_partition, run_duet.parse_cli, run_duet.file_sha256 = original_selector, original_parser, original_hash


if __name__ == "__main__":
    main()
