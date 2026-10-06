#!/usr/bin/env python3
"""Collect a pinned common pool's two natural and four C2 histories per pair."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vln_improve.protocol import file_sha256, object_sha256, resolve_config

CODE_FILES = ("scripts/collect_endpoint_controls.py", "src/vln_improve/endpoint_controls.py",
              "scripts/collect_endpoint_pairs.py", "src/vln_improve/endpoint_pairs.py",
              "src/vln_improve/endpoint_probe.py", "src/vln_improve/diagnostics.py",
              "src/vln_improve/protocol.py", "src/vln_improve/pipeline.py",
              "scripts/run_duet.py", "scripts/prepare_duet.py", "scripts/prepare_endpoint_controls.py")


def code_identity():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def validate_spec(spec, config):
    selections = {"smoke": {"train_fit": 2, "train_dev": 1},
                  "full": {"train_fit": 512, "train_dev": 128}}
    if spec.get("phase") not in selections:
        raise ValueError("control collection phase must be explicitly smoke or full")
    expected = {"schema_version": 1, "experiment_id": "D3-controls-common-pool", "seed": 0,
                "cache_schema": "duet_endpoint_controls_v2",
                "length_convention": "official_connectivity_actual_edges_with_preserved_execution_graph_values",
                "selection": selections[spec["phase"]], "pool_selection": selections["full"],
                "selection_order": "ascending_selection_hash_prefix_of_fixed_common_pool",
                "automatic_expansion": False, "rollouts_per_pair": 6,
                "natural_rollouts_per_pair": 2, "c2_rollouts_per_pair": 4, "max_observed_states": 15,
                "feature_schema": "concat_global_local_stop_crossmodal_v1", "feature_dim": 1536,
                "feature_dtype": "float32", "training_updates": 0, "validation_accesses": 0}
    if any(spec.get(k) != value for k, value in expected.items()):
        raise ValueError("control collection specification differs from the fixed common-pool protocol")
    for key in ("controls_report_sha256", "base_checkpoint_sha256"):
        value = spec.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
            raise ValueError("control report/checkpoint SHA is not pinned")
    model = config["model"]
    if (model.get("dataset") != "r2r" or model.get("batch_size") != 1 or model.get("max_action_len") != 15
            or model.get("fusion") != "dynamic" or model.get("enc_full_graph") is not True
            or model.get("act_visited_nodes", False)):
        raise ValueError("controls require fixed batch-one full-graph original DUET")


def resolve_selection(report, split, count):
    if (split not in {"train_fit", "train_dev"} or report.get("schema") != "duet_endpoint_controls_geometry_v1"
            or report.get("coverage_pass") is not True or report.get("validation_accesses") != 0
            or report.get("usage") != "train_only_geometry_diagnostic"):
        raise ValueError("train-only common-pool feasibility gate failed")
    source = report["splits"][split]
    manifest = source["manifest"]
    if (source.get("coverage_pass") is not True or source.get("manifest_sha256") != object_sha256(manifest)
            or manifest["requested_pairs"] != count or len(manifest["selected_pairs"]) != count):
        raise ValueError("common pool was resized or its manifest changed")
    selected, seen_paths = [], set()
    for row in manifest["selected_pairs"]:
        pair = row["pair"]
        if row["natural_instr_ids"] != pair["instr_ids"] or len(row["control_ids"]) != 2:
            raise ValueError("natural control instruction pool differs")
        controls = {}
        for i, slot in enumerate(("A", "B")):
            control_id = row["control_ids"][i]
            control = manifest["controls"][control_id]
            expected_id = object_sha256([pair["scan"], pair["path_ids"][i], pair["instr_ids"][i]])
            if (control_id != expected_id or control["control_id"] != control_id or control["eligible"] is not True
                    or control["scan"] != pair["scan"] or control["path_id"] != pair["path_ids"][i]
                    or control["instr_id"] != pair["instr_ids"][i] or control["goal"] != pair["goal_vpids"][i]
                    or control["start"] != pair["start"] or control["heading"] != pair["heading_rad"][i]
                    or pair["path_ids"][i] in seen_paths):
                raise ValueError("common source/C2 association changed or path is repeated")
            seen_paths.add(pair["path_ids"][i])
            controls[slot] = copy.deepcopy(control)
        selected.append({"pair": copy.deepcopy(pair), "controls": controls})
    hashes = [entry["pair"]["selection_hash"] for entry in selected]
    if hashes != sorted(hashes) or len(set(hashes)) != count:
        raise ValueError("common-pool deterministic order changed")
    return selected


def run(args):
    import torch
    from collect_endpoint_pairs import build_runtime
    from prepare_duet import verify
    from prepare_endpoint_controls import validate_spec as validate_geometry_spec
    from vln_improve.endpoint_controls import ControlStore, SCHEMA, collect_control_pair
    from vln_improve.pipeline import validate_backup_root
    config = resolve_config(args.config, ROOT)
    spec = json.loads(args.collection_config.read_text())
    validate_spec(spec, config)
    if file_sha256(args.controls_report) != spec["controls_report_sha256"]:
        raise ValueError("common-pool report changed")
    report = json.loads(args.controls_report.read_text())
    validate_geometry_spec(report["specification"], config)
    if report["identity"]["config_sha256"] != file_sha256(args.config):
        raise ValueError("common-pool runtime config changed")
    selection = resolve_selection(report, args.split, spec["pool_selection"][args.split])
    selection = selection[:spec["selection"][args.split]]
    dataset = Path(config["dataset_root"])
    annotation = dataset / "R2R/annotations/R2R_train_enc.json"
    annotation_sha = file_sha256(annotation)
    if annotation_sha != report["identity"]["annotation_sha256"]:
        raise ValueError("train annotation changed since common-pool preparation")
    graph_dir = dataset / "R2R/connectivity"
    graph_hashes = {scan: file_sha256(graph_dir / f"{scan}_connectivity.json")
                    for scan in report["identity"]["connectivity_files"]}
    if (graph_hashes != report["identity"]["connectivity_files"]
            or object_sha256(graph_hashes) != report["identity"]["connectivity_sha256"]):
        raise ValueError("train geometry changed since common-pool preparation")
    checkpoint_sha = file_sha256(config["base_checkpoint"])
    if checkpoint_sha != spec["base_checkpoint_sha256"]:
        raise ValueError("controls baseline checkpoint changed")
    upstream_lock = verify()
    identity = {"schema": SCHEMA, "split": args.split,
        "usage": "training" if args.split == "train_fit" else "analysis_only", "selection": selection,
        "selection_sha256": object_sha256(selection), "seed": spec["seed"],
        "runtime_config_sha256": file_sha256(args.config), "collection_config_sha256": file_sha256(args.collection_config),
        "controls_report_sha256": file_sha256(args.controls_report), "controls_source_identity": report["identity"],
        "controls_split_manifest_sha256": report["splits"][args.split]["manifest_sha256"],
        "phase": spec["phase"],
        "code_files": code_identity(), "feature_schema": spec["feature_schema"], "feature_dim": 1536,
        "common_provenance": {"base_checkpoint_sha256": checkpoint_sha,
            "feature_sha256": file_sha256(dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"),
            "annotation_sha256": annotation_sha,
            "connectivity_sha256": object_sha256({p.name: file_sha256(p) for p in sorted(graph_dir.glob("*_connectivity.json"))}),
            "model": config["model"], "upstream_lock": upstream_lock, "partition_seed": config["partition_seed"],
            "dev_fraction": config["dev_fraction"], "torch_version": str(torch.__version__)}}
    checker = lambda: validate_backup_root(args.backup, allow_local=args.allow_local_backup_for_tests)
    checker()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".collection.lock").open("a") as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = ControlStore(args.output, args.backup, identity, checker)
        pending = [entry for entry in selection if store.find(entry) is None]
        resources = {"new_pairs": len(pending), "reused_pairs": len(selection) - len(pending), "wall_seconds": 0,
                     "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}
        if pending:
            pairs = [entry["pair"] for entry in selection]
            agent, records, graph_class, runtime_lock = build_runtime(config, pairs, args.split, spec["seed"], args.output)
            if runtime_lock != upstream_lock:
                raise ValueError("upstream source changed during initialization")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for index, entry in enumerate(pending):
                payload = collect_control_pair(agent, entry, records, graph_class, store.identity_sha256)
                manifest = store.commit(payload)
                print(json.dumps({"event": "controls_pair_backed_up", "split": args.split,
                    "pair": entry["pair"]["selection_hash"], "new_pairs": index + 1,
                    "remaining_pairs": len(pending) - index - 1, "states": manifest["states"]}), flush=True)
            torch.cuda.synchronize()
            resources.update(wall_seconds=time.monotonic() - started,
                             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        summary = store.seal(resources)
        print(json.dumps({k: summary[k] for k in ("split", "pairs", "rollouts", "states", "resources")}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-sha256", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--collection-config", type=Path)
    parser.add_argument("--controls-report", type=Path)
    parser.add_argument("--split", choices=("train_fit", "train_dev"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args(argv)
    if args.print_code_sha256:
        print(object_sha256(code_identity()))
        return
    if any(getattr(args, key) is None for key in ("collection_config", "controls_report", "split", "output", "backup")):
        parser.error("--collection-config, --controls-report, --split, --output, --backup are required")
    run(args)


if __name__ == "__main__":
    main()
