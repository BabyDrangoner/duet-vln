#!/usr/bin/env python3
"""Collect the fixed D3b train-only shared-history simulator pilot; no fitting."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vln_improve.protocol import file_sha256, object_sha256, resolve_config, select_partition

CODE_FILES = ("scripts/collect_endpoint_pairs.py", "src/vln_improve/endpoint_pairs.py",
              "src/vln_improve/endpoint_probe.py", "src/vln_improve/diagnostics.py",
              "src/vln_improve/protocol.py", "src/vln_improve/pipeline.py",
              "scripts/run_duet.py", "scripts/prepare_duet.py")


def code_identity():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def validate_spec(spec, config):
    expected = {"schema_version": 1, "experiment_id": "D3b-paired-endpoint-simulator-pilot",
        "phase": "pilot", "seed": 0, "batch_size": 1,
        "selection": {"train_fit": 32, "train_dev": 16,
                      "order": "ascending_selection_hash_from_primary_path_disjoint_manifest"},
        "rollouts_per_pair": 4, "history_orders": ["A_then_B", "B_then_A"], "instruction_slots": ["A", "B"],
        "max_observed_states": 15, "feature_schema": "concat_global_local_stop_crossmodal_v1",
        "feature_dim": 1536, "feature_dtype": "float32", "training_updates": 0,
        "validation_accesses": 0, "automatic_expansion": False}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("D3b pilot configuration changed; no automatic expansion or fitting allowed")
    model = config["model"]
    if (model.get("batch_size") != 1 or model.get("max_action_len") != 15
            or model.get("fusion") != "dynamic" or not model.get("enc_full_graph")
            or model.get("act_visited_nodes", False)):
        raise ValueError("requires fixed DUET batch-one full-graph dynamic unvisited-only configuration")
    for key in ("coverage_report_sha256", "base_checkpoint_sha256"):
        value = spec.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("missing source/checkpoint SHA")


def check_records(pairs, records):
    from vln_improve.endpoint_pairs import content_hash
    for pair in pairs:
        a, b = [records[instr] for instr in pair["instr_ids"]]
        for index, row in enumerate((a, b)):
            if (row["scan"] != pair["scan"] or str(row["path_id"]) != pair["path_ids"][index]
                    or row["path"][0] != pair["start"] or row["path"][-1] != pair["goal_vpids"][index]
                    or row["heading"] != pair["heading_rad"][index]):
                raise ValueError("original annotation differs from the D3 selected pair")
        if content_hash(a["instr_encoding"]) == content_hash(b["instr_encoding"]):
            raise ValueError("paired encoded instructions are identical; no contradictory-label collection")
        for order, (first, second) in (("A_then_B", (a, b)), ("B_then_A", (b, a))):
            walk = first["path"] + list(reversed(first["path"]))[1:] + second["path"][1:]
            history = pair["histories"][order]
            if history["reference_walk"] != walk or history["observed_vpids"] != list(dict.fromkeys(walk)):
                raise ValueError("D3 shared history differs from original reference paths")


def build_runtime(config, pairs, split, seed, output):
    """Construct the same upstream environment/agent as run_duet, without its eval loop."""
    import numpy as np
    import torch
    import run_duet
    lock = run_duet.verify()
    if not torch.cuda.is_available():
        raise RuntimeError("D3b simulator feature collection requires the root-controlled CUDA runtime")
    torch.cuda.set_device(0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sys.path.insert(0, str(run_duet.DEFAULT_DEST / "map_nav_src"))
    from r2r.parser import parse_args
    from r2r.data_utils import construct_instrs
    from r2r.env import R2RNavBatch
    from r2r.agent import GMapNavAgent
    from models.graph_utils import GraphMap
    from utils.data import ImageFeaturesDB
    argv = ["duet", "--root_dir", config["dataset_root"], "--output_dir", str(output / "upstream_logs"),
            "--seed", str(seed), "--test"]
    for key, value in config["model"].items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv.extend(["--" + key, str(value)])
    previous = sys.argv
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = previous
    for key, value in config["model"].items():
        if not hasattr(args, key) or getattr(args, key) != value:
            raise ValueError(f"upstream option not applied: {key}")
    original = construct_instrs(args.anno_dir, "r2r", ["train"], "bert", args.max_instr_len, is_test=True)
    partition = select_partition(original, split, config["dev_fraction"], config["partition_seed"])
    selected_ids = {instr for pair in pairs for instr in pair["instr_ids"]}
    records = {r["instr_id"]: r for r in partition if r["instr_id"] in selected_ids}
    if set(records) != selected_ids or len(records) != 2 * len(pairs):
        raise ValueError("selected original instructions are missing or repeated in the correct partition")
    check_records(pairs, records)
    feature_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    env = R2RNavBatch(feature_db, copy.deepcopy(list(records.values())), args.connectivity_dir,
                      batch_size=1, angle_feat_size=args.angle_feat_size, seed=seed, name=split)
    agent = GMapNavAgent(args, env, rank=0)
    weights = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=True)
    agent.vln_bert.load_state_dict({k.removeprefix("module."): v for k, v in weights["vln_bert"]["state_dict"].items()}, strict=True)
    del weights
    for model in agent.models:
        model.requires_grad_(False); model.eval()
    return agent, records, GraphMap, lock


def run(args):
    import torch
    from vln_improve.endpoint_pairs import PairStore, SCHEMA, collect_pair, selected_pairs
    from vln_improve.pipeline import validate_backup_root
    from prepare_duet import verify
    config = resolve_config(args.config, ROOT)
    spec = json.loads(args.collection_config.read_text())
    validate_spec(spec, config)
    if file_sha256(args.coverage_report) != spec["coverage_report_sha256"]:
        raise ValueError("D3 coverage report SHA differs from the fixed pilot")
    coverage = json.loads(args.coverage_report.read_text())
    if coverage["identity"]["config_sha256"] != file_sha256(args.config):
        raise ValueError("D3 and collection use different runtime configurations")
    pairs = selected_pairs(coverage, args.split, spec["selection"][args.split])
    dataset = Path(config["dataset_root"])
    annotation = dataset / "R2R/annotations/R2R_train_enc.json"
    anno_sha = file_sha256(annotation)
    if anno_sha != coverage["identity"]["annotation_sha256"]:
        raise ValueError("original train annotations changed since D3")
    source_records = json.loads(annotation.read_text())
    scans = sorted({row["scan"] for row in source_records})
    graph_sha = {scan: file_sha256(dataset / "R2R/connectivity" / f"{scan}_connectivity.json") for scan in scans}
    if object_sha256(graph_sha) != coverage["identity"]["connectivity_sha256"]:
        raise ValueError("training connectivity changed since D3")
    checkpoint_sha = file_sha256(config["base_checkpoint"])
    if checkpoint_sha != spec["base_checkpoint_sha256"]:
        raise ValueError("collection checkpoint differs from frozen baseline")
    features = dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
    upstream_lock = verify()
    identity = {"schema": SCHEMA, "split": args.split,
        "usage": "training" if args.split == "train_fit" else "analysis_only",
        "selection": pairs, "selection_sha256": object_sha256(pairs), "seed": spec["seed"],
        "runtime_config_sha256": file_sha256(args.config), "collection_config_sha256": file_sha256(args.collection_config),
        "coverage_report_sha256": file_sha256(args.coverage_report), "code_files": code_identity(),
        "common_provenance": {"base_checkpoint_sha256": checkpoint_sha, "feature_sha256": file_sha256(features),
            "annotation_sha256": anno_sha,
            "connectivity_sha256": object_sha256({p.name: file_sha256(p) for p in sorted((dataset / "R2R/connectivity").glob("*_connectivity.json"))}),
            "model": config["model"], "upstream_lock": upstream_lock, "partition_seed": config["partition_seed"],
            "dev_fraction": config["dev_fraction"], "torch_version": str(torch.__version__)},
        "feature_schema": spec["feature_schema"], "feature_dim": 1536,
        "execution": spec["execution"], "candidate_cache": spec["candidate_cache"]}
    checker = lambda: validate_backup_root(args.backup, allow_local=args.allow_local_backup_for_tests)
    checker()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".collection.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = PairStore(args.output, args.backup, identity, checker)
        pending = [pair for pair in pairs if store.find(pair) is None]
        if not pending:
            summary = store.seal({"new_pairs": 0, "reused_pairs": len(pairs), "wall_seconds": 0,
                                  "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None})
        else:
            agent, records, graph_class, runtime_lock = build_runtime(config, pairs, args.split, spec["seed"], args.output)
            if runtime_lock != upstream_lock:
                raise ValueError("upstream source changed during runtime initialization")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for index, pair in enumerate(pending):
                payload = collect_pair(agent, pair, records, graph_class, store.identity_sha256)
                manifest = store.commit(payload)
                print(json.dumps({"event": "complete_pair_backed_up", "split": args.split,
                                  "new_pairs": index + 1, "pending_pairs": len(pending) - index - 1,
                                  "pair": pair["selection_hash"], "states": manifest["states"]}), flush=True)
                del payload
            torch.cuda.synchronize()
            summary = store.seal({"new_pairs": len(pending), "reused_pairs": len(pairs) - len(pending),
                                  "wall_seconds": time.monotonic() - started,
                                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        print(json.dumps({key: summary[key] for key in ("split", "pairs", "rollouts", "states",
                         "all_shared_history_exact_parity", "resources", "navigation_metrics", "interpretation")}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--collection-config", type=Path, default=ROOT / "configs/endpoint_pair_collection.json")
    parser.add_argument("--coverage-report", type=Path, default=ROOT / "outputs/study-20261003/d3-endpoint-pair-coverage.json")
    parser.add_argument("--split", choices=("train_fit", "train_dev"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.print_code_sha256:
        print(object_sha256(code_identity()))
        return
    if args.split is None or args.output is None or args.backup is None:
        parser.error("--split, --output, and --backup are required")
    run(args)


if __name__ == "__main__":
    main()
