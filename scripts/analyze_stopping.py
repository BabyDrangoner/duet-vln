#!/usr/bin/env python3
"""Analyze train-only STOP endpoint opportunities without inference or fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_diagnostics import validate_episode
from vln_improve.diagnostics import SCHEMA, load_episode
from vln_improve.pipeline import atomic_json
from vln_improve.protocol import file_sha256, object_sha256
from vln_improve.stopping_diagnostics import analyze_episode, summarize_episodes


def _indexed(rows, key, name):
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get(key), str) for row in rows):
        raise ValueError(f"invalid {name}")
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {name} identifier")
    return result


def analyze_collection(directory, *, split, rollout_report):
    if split not in {"train_fit", "train_dev"}:
        raise ValueError("STOP diagnostics only allow train_fit/train_dev")
    root, report_path = Path(directory), Path(rollout_report)
    marker = root / "COLLECTION.json"
    if marker.is_symlink() or report_path.is_symlink():
        raise ValueError("collection/report must not be symlinked")
    collection = json.loads(marker.read_bytes())
    identity = collection.get("identity")
    if (collection.get("schema") != SCHEMA or not isinstance(identity, dict)
            or identity.get("schema") != "duet_diagnostic_collection_v1"
            or collection.get("identity_sha256") != object_sha256(identity)):
        raise ValueError("collection identity schema/checksum mismatch")
    usage = "training_diagnostics" if split == "train_fit" else "analysis_only"
    if identity.get("split") != split or identity.get("usage") != usage:
        raise ValueError("collection split/usage mismatch")
    selected = _indexed(identity.get("selection"), "instr_id", "selection")
    if not selected or identity.get("selection_sha256") != object_sha256(identity["selection"]):
        raise ValueError("empty or invalid collection selection checksum")
    model = identity.get("model", {})
    if model.get("enc_full_graph") is not True or model.get("fusion") != "dynamic" or model.get("batch_size") != 1:
        raise ValueError("requires full-graph, dynamic-fusion, batch-one diagnostic protocol")
    report = json.loads(report_path.read_bytes())
    metadata = report.get("metadata", {})
    expected = {"split": split, "usage": usage, "feedback": "argmax", "mode": "baseline",
                "diagnostic_collection": True, "subset": True, "selection_sha256": identity["selection_sha256"],
                "base_checkpoint_sha256": identity["base_checkpoint_sha256"],
                "feature_id": identity["feature_sha256"], "train_annotation_sha256": identity["annotation_sha256"],
                "connectivity_sha256": identity["connectivity_sha256"],
                "upstream_commit": identity["upstream_lock"]["commit"], "seed": identity["seed"],
                "max_action_len": model["max_action_len"]}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("rollout report provenance differs from collection")
    metrics = _indexed(report.get("episodes"), "instr_id", "rollout metrics")
    trajectories = _indexed(report.get("trajectories"), "instr_id", "rollout trajectories")
    if set(metrics) != set(selected) or set(trajectories) != set(selected):
        raise ValueError("rollout episode selection differs from collection")
    names = {"episode-" + object_sha256([row["scan"], instr]): row for instr, row in selected.items()}
    directories = sorted(root.glob("episode-*"))
    if {p.name for p in directories} != set(names):
        raise ValueError("collection is incomplete or has unexpected episodes")
    rows, manifests = [], {}
    for episode in directories:
        inputs, labels, manifest = load_episode(episode, expected_identity_sha256=collection["identity_sha256"])
        association, _, _, _ = validate_episode(inputs, labels, split)
        selected_row = names[episode.name]
        if association["scan_id"] != selected_row["scan"] or association["instr_id"] != selected_row["instr_id"]:
            raise ValueError("saved episode is outside the selected associations")
        instr = association["instr_id"]
        row = analyze_episode(inputs, labels, manifest, max_action_len=model["max_action_len"],
                              metrics=metrics[instr], reported_trajectory=trajectories[instr]["trajectory"])
        rows.append(row)
        manifests[episode.name] = file_sha256(episode / "manifest.json")
    scans = sorted({row["association"]["scan_id"] for row in rows})
    return {
        "schema": "duet_stop_opportunity_analysis_v1", "split": split, "usage": usage,
        "scope": "fixed_complete_baseline_trajectories_final_endpoint_rescoring_only",
        "collection_identity_sha256": collection["identity_sha256"], "episode_manifest_sha256": manifests,
        "rollout_report_sha256": file_sha256(report_path),
        "implementation_sha256": object_sha256({str(path.relative_to(ROOT)): file_sha256(path) for path in (
            Path(__file__), ROOT / "src/vln_improve/stopping_diagnostics.py", ROOT / "scripts/replay_diagnostics.py")}),
        "overall": summarize_episodes(rows),
        "per_scan": {scan: summarize_episodes([r for r in rows if r["association"]["scan_id"] == scan]) for scan in scans},
        "by_termination_flag": {flag: summarize_episodes([r for r in rows if r["termination_flags"][flag]])
                                for flag in ("argmax_stop", "no_legal_move", "action_limit")},
        "episodes": rows,
        "missing": {"counterfactual_spl": "GT reference path length is not cached; start-goal shortest distance is not substituted",
                    "continue_after_baseline_stop": "not observed; no online policy benefit is estimated",
                    "unobserved_intermediate_node_goal_labels": "excluded from observed-history endpoint oracle"},
        "interpretation": {
            "cost": "complete original movement prefix plus final discovered-graph return; no prefix cropping",
            "zero_move": "max-margin/logmeanexp unavailable for entire episode if any history state has no move; coverage and paired baseline are explicit",
            "flags": "termination flags may overlap; grouped counts are not disjoint",
            "sample_size": "report independent episode counts; one or two recoverable errors cannot establish a research mechanism",
            "generalization": "train_dev isolates adaptation only; the frozen official DUET has seen these training scenes",
            "causality": "score changes and candidate-count associations do not establish causal count bias or novelty",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", "--collection", type=Path, required=True)
    parser.add_argument("--split", choices=("train_fit", "train_dev"), required=True)
    parser.add_argument("--rollout-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_collection(args.diagnostics, split=args.split, rollout_report=args.rollout_report)
    # Do not allow an output path to overwrite the immutable input collection.
    if args.output.resolve().is_relative_to(args.diagnostics.resolve()) or args.output.resolve() == args.rollout_report.resolve():
        raise ValueError("output must be outside the input collection and rollout report")
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "split": args.split, "overall": result["overall"]}, indent=2))


if __name__ == "__main__":
    main()
