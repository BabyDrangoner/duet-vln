#!/usr/bin/env python3
"""Collect fixed 512/128 paired features from the verified D3c common pool."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import collect_endpoint_pairs as pilot
from analyze_endpoint_pairs import load_records, read_json
from prepare_endpoint_controls import validate_spec as validate_controls_spec, verify_primary
from vln_improve.protocol import file_sha256, object_sha256, resolve_config, select_partition

CODE_FILES = ("scripts/collect_endpoint_pairs_full.py", "scripts/prepare_endpoint_controls.py",
              "scripts/analyze_endpoint_pairs.py", *pilot.CODE_FILES)


def code_identity():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_spec(spec, config):
    expected = {"schema_version": 1, "experiment_id": "D3-full-paired-endpoint-common-pool", "phase": "full",
        "seed": 0, "selection_seed": 20261003, "batch_size": 1,
        "selection": {"train_fit": 512, "train_dev": 128,
                      "source": "D3c selected common pool in ascending original D3 selection_hash order"},
        "rollouts_per_pair": 4, "history_orders": ["A_then_B", "B_then_A"], "instruction_slots": ["A", "B"],
        "max_observed_states": 15, "feature_schema": "concat_global_local_stop_crossmodal_v1",
        "feature_dim": 1536, "feature_dtype": "float32", "training_updates": 0,
        "validation_accesses": 0, "automatic_expansion": False}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("full collection requires the fixed common-pool budget and separate model/selection seeds")
    for key in ("coverage_report_sha256", "controls_report_sha256", "pilot_audit_sha256",
                "pilot_collection_config_sha256", "base_checkpoint_sha256"):
        if not _sha(spec.get(key)):
            raise ValueError(f"missing/invalid fixed source SHA: {key}")
    model = config["model"]
    if (model.get("batch_size") != 1 or model.get("max_action_len") != 15
            or model.get("fusion") != "dynamic" or model.get("enc_full_graph") is not True
            or model.get("act_visited_nodes", False) is not False):
        raise ValueError("requires frozen DUET full-graph dynamic batch-one unvisited-only protocol")
    pilot_path = ROOT / "configs/endpoint_pair_collection.json"
    if file_sha256(pilot_path) != spec["pilot_collection_config_sha256"]:
        raise ValueError("frozen pilot collection configuration changed")
    original = read_json(pilot_path)
    for key in ("execution", "candidate_cache"):
        if spec.get(key) != original[key]:
            raise ValueError("full collection changes the verified pilot execution protocol")


def validate_pilot_audit(audit):
    expected = {"endpoint-pair-pilot-train-fit": 32, "endpoint-pair-pilot-train-dev": 16}
    if not isinstance(audit, dict) or set(audit.get("splits", {})) != set(expected):
        raise ValueError("pilot audit must cover both fixed training splits")
    for key, count in expected.items():
        row = audit["splits"][key]
        if (row.get("pairs") != count or row.get("rollouts") != 4 * count
                or row.get("all_shared_history_exact_parity") is not True
                or row.get("all_drive_hashes_match") is not True
                or not _sha(row.get("manifest_sha256"))):
            raise ValueError("pilot parity or Drive audit has not passed")


def common_pairs(controls, coverage, split, *, count, selection_seed=20261003):
    """Rebuild the D3c choice, never take the original D3 first N pairs."""
    if split not in {"train_fit", "train_dev"} or type(count) is not int or count < 1 or selection_seed != 20261003:
        raise ValueError("invalid train-only common-pool selection")
    if (controls.get("schema") != "duet_endpoint_controls_geometry_v1" or controls.get("coverage_pass") is not True
            or controls.get("usage") != "train_only_geometry_diagnostic"
            or coverage.get("schema") != "duet_endpoint_pair_coverage_v1" or coverage.get("coverage_pass") is not True
            or coverage.get("usage") != "train_only_geometry_diagnostic"
            or coverage["specification"]["selection_seed"] != selection_seed):
        raise ValueError("D3/D3c schema, gates, or D3 selection seed differs")
    for key in ("config_sha256", "annotation_sha256", "connectivity_sha256"):
        if controls["identity"].get(key) != coverage["identity"].get(key):
            raise ValueError("D3 and D3c common source identities differ")
    prior = coverage["splits"][split]
    primary = prior["primary_path_disjoint_manifest"]
    if prior["primary_manifest_sha256"] != object_sha256(primary):
        raise ValueError("D3 primary checksum differs")
    result = controls["splits"][split]
    manifest = result["manifest"]
    if (result.get("coverage_pass") is not True or result.get("shortfall_pairs") != 0
            or result.get("requested_pairs") != count or manifest.get("requested_pairs") != count
            or result.get("manifest_sha256") != object_sha256(manifest)):
        raise ValueError("D3c split is incomplete or its manifest changed")
    pool, available = manifest["controls"], []
    known_paths, known_hashes, expected_controls = set(), set(), set()
    for pair in sorted(primary, key=lambda p: p["selection_hash"]):
        if (pair["selection_hash"] != object_sha256([selection_seed, pair["scan"], *pair["path_ids"]])
                or pair["selection_hash"] in known_hashes):
            raise ValueError("pair selection hash must use D3 selection seed, not GPU seed")
        known_hashes.add(pair["selection_hash"])
        ids, flags = [], []
        for index, pid in enumerate(pair["path_ids"]):
            association = (pair["scan"], pid)
            if association in known_paths:
                raise ValueError("D3 common pool repeats a source path")
            known_paths.add(association)
            instr = pair["instr_ids"][index]
            key = object_sha256([pair["scan"], pid, instr])
            expected_controls.add(key)
            item = pool.get(key)
            if (not isinstance(item, dict) or item.get("control_id") != key or item.get("scan") != pair["scan"]
                    or item.get("path_id") != pid or item.get("instr_id") != instr
                    or item.get("start") != pair["start"] or item.get("goal") != pair["goal_vpids"][index]
                    or item.get("heading") != pair["heading_rad"][index] or type(item.get("eligible")) is not bool):
                raise ValueError("C2 control association differs from D3")
            if item["eligible"]:
                original = item["original_path"]
                positive, negative = item["positive_history"], item["overshoot_history"]
                distance = item["selected_q_goal_distance_m"]
                if (positive["observed_vpids"] != list(dict.fromkeys(original))
                        or positive["observed_vpids"][-1] != item["goal"]
                        or negative["observed_vpids"][-1] != item["selected_q"]
                        or item["selected_q"] in original or not isinstance(distance, (int, float))
                        or not math.isfinite(distance) or not 3 <= distance <= 6
                        or item["selected_q_hash"] != object_sha256([0, item["scan"], pid, item["selected_q"]])
                        or any(not 2 <= h["observed_states"] <= 15 or h["observed_states"] != len(h["observed_vpids"])
                               or len(set(h["observed_vpids"])) != len(h["observed_vpids"])
                               for h in (positive, negative))):
                    raise ValueError("C2 feasible control violates its fixed geometric definition")
            ids.append(key); flags.append(item["eligible"])
        if all(flags):
            available.append({"pair": pair, "control_ids": ids, "natural_instr_ids": pair["instr_ids"]})
    if set(pool) != expected_controls:
        raise ValueError("C2 manifest has missing or extra source paths")
    expected = available[:count]
    if len(expected) != count or manifest.get("selected_pairs") != expected:
        raise ValueError("selection is not the fixed first feasible D3c common pool")
    return copy.deepcopy([item["pair"] for item in expected])


def load_sources(args, config, spec):
    for path, name in ((args.coverage_report, "coverage_report_sha256"),
                       (args.controls_report, "controls_report_sha256"), (args.pilot_audit, "pilot_audit_sha256")):
        if file_sha256(path) != spec[name]:
            raise ValueError(f"source artifact SHA differs: {name}")
    coverage, controls = read_json(args.coverage_report), read_json(args.controls_report)
    validate_pilot_audit(read_json(args.pilot_audit))
    validate_controls_spec(controls["specification"], config)
    if (controls["identity"].get("coverage_report_sha256") != spec["coverage_report_sha256"]
            or controls["specification"].get("coverage_report_sha256") != spec["coverage_report_sha256"]
            or coverage["identity"].get("config_sha256") != file_sha256(args.config)):
        raise ValueError("common pool does not bind the same D3/config source")
    pairs = common_pairs(controls, coverage, args.split, count=spec["selection"][args.split],
                         selection_seed=spec["selection_seed"])
    dataset = Path(config["dataset_root"])
    annotation = dataset / "R2R/annotations/R2R_train_enc.json"
    if file_sha256(annotation) != coverage["identity"]["annotation_sha256"]:
        raise ValueError("train annotation changed since D3/D3c")
    records = load_records(annotation)
    scans = sorted({r["scan"] for r in records})
    graphs = {scan: file_sha256(dataset / "R2R/connectivity" / f"{scan}_connectivity.json") for scan in scans}
    if (object_sha256(graphs) != coverage["identity"]["connectivity_sha256"]
            or controls["identity"].get("connectivity_files") != graphs):
        raise ValueError("training connectivity changed since D3/D3c")
    selected_records = {r["path_id"]: r for r in select_partition(records, args.split,
                         config["dev_fraction"], config["partition_seed"])}
    verify_primary(pairs, selected_records, expected_sha=object_sha256(pairs), selection_seed=spec["selection_seed"])
    source_pool = controls["splits"][args.split]["manifest"]["controls"]
    for pair in pairs:
        for pid, instr in zip(pair["path_ids"], pair["instr_ids"]):
            item = source_pool[object_sha256([pair["scan"], pid, instr])]
            if item["original_path"] != selected_records[pid]["path"]:
                raise ValueError("C2 source reference path differs from original train annotation")
    return pairs, file_sha256(annotation)


def run(args):
    import torch
    from prepare_duet import verify
    from vln_improve.endpoint_pairs import PairStore, SCHEMA, collect_pair
    from vln_improve.pipeline import validate_backup_root
    config, spec = resolve_config(args.config, ROOT), read_json(args.collection_config)
    validate_spec(spec, config)
    pairs, annotation_sha = load_sources(args, config, spec)
    dataset = Path(config["dataset_root"])
    checkpoint_sha = file_sha256(config["base_checkpoint"])
    if checkpoint_sha != spec["base_checkpoint_sha256"]:
        raise ValueError("full collection differs from the fixed official base checkpoint")
    lock = verify()
    identity = {"schema": SCHEMA, "split": args.split,
        "usage": "training" if args.split == "train_fit" else "analysis_only",
        "selection": pairs, "selection_sha256": object_sha256(pairs), "seed": spec["seed"],
        "runtime_config_sha256": file_sha256(args.config), "collection_config_sha256": file_sha256(args.collection_config),
        "coverage_report_sha256": spec["coverage_report_sha256"], "code_files": code_identity(),
        "common_provenance": {"base_checkpoint_sha256": checkpoint_sha,
            "feature_sha256": file_sha256(dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"),
            "annotation_sha256": annotation_sha,
            "connectivity_sha256": object_sha256({p.name: file_sha256(p) for p in sorted((dataset / "R2R/connectivity").glob("*_connectivity.json"))}),
            "model": config["model"], "upstream_lock": lock, "partition_seed": config["partition_seed"],
            "dev_fraction": config["dev_fraction"], "torch_version": str(torch.__version__)},
        "feature_schema": spec["feature_schema"], "feature_dim": spec["feature_dim"],
        "execution": spec["execution"], "candidate_cache": spec["candidate_cache"]}
    checker = lambda: validate_backup_root(args.backup, allow_local=args.allow_local_backup_for_tests)
    checker()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".collection.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = PairStore(args.output, args.backup, identity, checker)
        pending = [p for p in pairs if store.find(p) is None]
        if pending:
            agent, records, graph_class, runtime_lock = pilot.build_runtime(config, pairs, args.split, spec["seed"], args.output)
            if runtime_lock != lock:
                raise ValueError("upstream source changed during runtime construction")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for index, pair in enumerate(pending):
                payload = collect_pair(agent, pair, records, graph_class, store.identity_sha256)
                manifest = store.commit(payload)
                print(json.dumps({"event": "full_common_pair_backed_up", "split": args.split,
                    "new_pairs": index + 1, "remaining_pairs": len(pending) - index - 1,
                    "pair": pair["selection_hash"], "states": manifest["states"]}), flush=True)
                del payload
            torch.cuda.synchronize()
            resources = {"wall_seconds": time.monotonic() - started,
                         "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        else:
            resources = {"wall_seconds": 0, "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
        resources.update(new_pairs=len(pending), reused_pairs=len(pairs) - len(pending))
        summary = store.seal(resources)
        print(json.dumps({key: summary[key] for key in ("split", "pairs", "rollouts", "states",
                         "all_shared_history_exact_parity", "resources", "navigation_metrics", "interpretation")}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--collection-config", type=Path, default=ROOT / "configs/endpoint_pair_full.json")
    parser.add_argument("--coverage-report", type=Path, default=ROOT / "outputs/study-20261003/d3-endpoint-pair-coverage.json")
    parser.add_argument("--controls-report", type=Path, default=ROOT / "outputs/study-20261003/d3c-endpoint-controls-geometry.json")
    parser.add_argument("--pilot-audit", type=Path, default=ROOT / "outputs/study-20261003/d3b-pilot-audit.json")
    parser.add_argument("--split", choices=("train_fit", "train_dev"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--print-code-sha256", action="store_true")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.print_code_sha256:
        print(object_sha256(code_identity()))
        return
    if args.split is None or args.output is None or args.backup is None:
        parser.error("--split, --output, --backup are required")
    run(args)


if __name__ == "__main__":
    main()
