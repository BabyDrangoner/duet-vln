#!/usr/bin/env python3
"""Describe evidence coverage and oracle action disagreement in every saved state.

This CPU-only analysis does not select new examples, fit a model, or equate
teacher/execution optimality with navigation success or instruction following.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from replay_diagnostics import validate_episode
from vln_improve.diagnostics import SCHEMA, load_episode
from vln_improve.pipeline import atomic_json
from vln_improve.protocol import file_sha256, object_sha256


OUTCOMES = ("optimal", "non_optimal", "unscorable_stop", "unscorable_nonstop")
COUNT_FIELDS = ("source_count_total", "observation_count", "duplicate_count", "changed_count",
                "overflow_count", "duplicate_event_count")


def _candidate_stats():
    return {"count": 0, "multi_source_count": 0, "source_count_histogram": Counter(),
            "naturally_paired_count": 0}


def _table(categories):
    return {category: {outcome: 0 for outcome in OUTCOMES} for category in categories}


def _oracle_stats():
    return {"finite_regrets": [], "optimal": 0, "non_optimal": 0,
            "unscorable_stop": 0, "unscorable_nonstop": 0,
            "finite_stop_count": 0, "finite_nonstop_count": 0,
            "by_any_multisource_candidate": _table(("none", "present")),
            "by_selected_candidate_sources": _table(("stop", "single_source", "multi_source"))}


def _stats():
    return {"episodes": 0, "recorded_states": 0, "eligible_decisions": 0, "ineligible_states": 0,
            "recorded_states_by_step": Counter(), "eligible_decisions_by_step": Counter(),
            "all_recorded_candidates": _candidate_stats(), "eligible_candidates": _candidate_stats(),
            "selected_nonstop_candidates": _candidate_stats(), "selected_stop_count": 0,
            "states_with_multisource_candidate": 0,
            "teacher": _oracle_stats(), "execution": _oracle_stats(),
            "unique_proxy_targets": 0, "naturally_arrived_targets": 0, "unknown_arrival_targets": 0,
            "observation_counts_per_unique_target": Counter({field: 0 for field in COUNT_FIELDS}),
            "targets_with_repeated_source_observations": 0,
            "targets_with_overflow": 0}


def _source_count(snapshot):
    counts = snapshot["counts"]
    if any(type(counts.get(field)) is not int or counts[field] < 0 for field in COUNT_FIELDS):
        raise ValueError("evidence counts must be nonnegative integers")
    count = counts["source_count_total"]
    if (count < len(snapshot["sources"]) or count < 1
            or counts.get("stored_count") != len(snapshot["sources"])
            or counts["observation_count"] != count + counts["duplicate_count"] + counts["changed_count"]):
        raise ValueError("evidence source/observation counts disagree")
    return count


def _add_candidate(bucket, count, paired):
    bucket["count"] += 1
    bucket["source_count_histogram"][count] += 1
    bucket["multi_source_count"] += int(count >= 2)
    bucket["naturally_paired_count"] += int(paired)


def _add_episode(stats, inputs, labels, pairs):
    stats["episodes"] += 1
    oracle = {item["step"]: item for item in labels["states"]}
    final_snapshots = {}
    for state in inputs["states"]:
        step = state["step"]
        stats["recorded_states"] += 1
        stats["recorded_states_by_step"][step] += 1
        multi_present = False
        for target, snapshot in state["candidate_evidence"].items():
            count = _source_count(snapshot)
            paired = target in pairs and pairs[target]["training_only"]["arrival_step"] > step
            _add_candidate(stats["all_recorded_candidates"], count, paired)
            final_snapshots[target] = snapshot
            multi_present |= count >= 2
            if state["eligible_decision"]:
                _add_candidate(stats["eligible_candidates"], count, paired)
        if not state["eligible_decision"]:
            stats["ineligible_states"] += 1
            continue
        stats["eligible_decisions"] += 1
        stats["eligible_decisions_by_step"][step] += 1
        stats["states_with_multisource_candidate"] += int(multi_present)
        valid = state["valid_mask"]
        selected = int(state["base_logits"].masked_fill(~valid, -torch.inf).argmax())
        if state.get("baseline_argmax", selected) != selected:
            raise ValueError("recorded baseline argmax disagrees with logits")
        if selected == 0:
            stats["selected_stop_count"] += 1
            selected_bucket = "stop"
        else:
            target = state["nav_inputs"]["gmap_vpids"][0][selected]
            count = _source_count(state["candidate_evidence"][target])
            paired = target in pairs and pairs[target]["training_only"]["arrival_step"] > step
            _add_candidate(stats["selected_nonstop_candidates"], count, paired)
            selected_bucket = "multi_source" if count >= 2 else "single_source"
        for kind in ("teacher", "execution"):
            result = stats[kind]
            costs = oracle[step][kind + "_cost"]
            finite = torch.isfinite(costs)
            if bool(finite[selected]):
                regret = float(costs[selected] - costs[finite].min())
                if regret < -1e-9:
                    raise ValueError("negative oracle regret")
                result["finite_regrets"].append(regret)
                result["finite_stop_count" if selected == 0 else "finite_nonstop_count"] += 1
                outcome = "optimal" if selected in oracle[step][kind + "_optimal_indices"] else "non_optimal"
            else:
                outcome = "unscorable_stop" if selected == 0 else "unscorable_nonstop"
            result[outcome] += 1
            result["by_any_multisource_candidate"]["present" if multi_present else "none"][outcome] += 1
            result["by_selected_candidate_sources"][selected_bucket][outcome] += 1
    stats["unique_proxy_targets"] += len(final_snapshots)
    stats["naturally_arrived_targets"] += len(pairs)
    stats["unknown_arrival_targets"] += len(set(final_snapshots) - set(pairs))
    for snapshot in final_snapshots.values():
        counts = snapshot["counts"]
        for field in COUNT_FIELDS:
            stats["observation_counts_per_unique_target"][field] += counts[field]
        stats["targets_with_repeated_source_observations"] += int(counts["observation_count"] > counts["source_count_total"])
        stats["targets_with_overflow"] += int(counts["overflow_count"] > 0)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _finish(stats):
    for name in ("recorded_states_by_step", "eligible_decisions_by_step"):
        stats[name] = {str(key): value for key, value in sorted(stats[name].items())}
    for name in ("all_recorded_candidates", "eligible_candidates", "selected_nonstop_candidates"):
        bucket = stats[name]
        bucket["source_count_histogram"] = {str(key): value for key, value in sorted(bucket["source_count_histogram"].items())}
        bucket["multi_source_fraction"] = _rate(bucket["multi_source_count"], bucket["count"])
        bucket["natural_arrival_fraction"] = _rate(bucket["naturally_paired_count"], bucket["count"])
    stats["states_with_multisource_fraction"] = _rate(stats["states_with_multisource_candidate"], stats["eligible_decisions"])
    stats["unique_target_natural_arrival_fraction"] = _rate(stats["naturally_arrived_targets"], stats["unique_proxy_targets"])
    stats["observation_counts_per_unique_target"] = dict(stats["observation_counts_per_unique_target"])
    for kind in ("teacher", "execution"):
        result = stats[kind]
        regrets = result.pop("finite_regrets")
        result["finite_regret_count"] = len(regrets)
        result["finite_regret_mean"] = statistics.mean(regrets) if regrets else None
        result["finite_regret_median"] = statistics.median(regrets) if regrets else None
        result["finite_regret_max"] = max(regrets) if regrets else None
        result["optimal_fraction_among_finite"] = _rate(result["optimal"], len(regrets))
        for table_name in ("by_any_multisource_candidate", "by_selected_candidate_sources"):
            for cell in result[table_name].values():
                cell["finite_count"] = cell["optimal"] + cell["non_optimal"]
                cell["non_optimal_fraction_among_finite"] = _rate(cell["non_optimal"], cell["finite_count"])
        result["multisource_presence_fraction_by_outcome"] = {
            outcome: _rate(result["by_any_multisource_candidate"]["present"][outcome], result[outcome])
            for outcome in OUTCOMES
        }
        selected = result["by_selected_candidate_sources"]
        result["selected_multisource_fraction_by_outcome_among_nonstop"] = {
            outcome: _rate(selected["multi_source"][outcome],
                           selected["multi_source"][outcome] + selected["single_source"][outcome])
            for outcome in OUTCOMES
        }
    return stats


def summarize_collection(collection_dir: str | Path, *, split: str):
    root = Path(collection_dir)
    if split not in {"train_fit", "train_dev"}:
        raise ValueError("only train_fit or analysis-only train_dev may be summarized")
    marker = root / "COLLECTION.json"
    if marker.is_symlink():
        raise ValueError("collection identity must not be a symlink")
    collection = json.loads(marker.read_bytes())
    identity = collection.get("identity") if isinstance(collection, dict) else None
    if (not isinstance(identity, dict) or collection.get("schema") != SCHEMA
            or collection.get("identity_sha256") != object_sha256(identity)
            or identity.get("schema") != "duet_diagnostic_collection_v1"):
        raise ValueError("collection identity schema/checksum mismatch")
    usage = "training_diagnostics" if split == "train_fit" else "analysis_only"
    if identity.get("split") != split or identity.get("usage") != usage:
        raise ValueError("collection split/usage mismatch")
    selected = identity.get("selection")
    if not isinstance(selected, list) or not selected:
        raise ValueError("collection selection is empty")
    if any(not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key]
                                          for key in ("scan", "instr_id")) for row in selected):
        raise ValueError("collection selection contains invalid associations")
    names = {"episode-" + object_sha256([row["scan"], row["instr_id"]]): row for row in selected}
    episodes = sorted(root.glob("episode-*"))
    if len(names) != len(selected) or {path.name for path in episodes} != set(names):
        raise ValueError("collection is incomplete or contains unexpected/duplicate episodes")
    total = _stats()
    by_scan, manifest_hashes = {}, {}
    for episode in episodes:
        inputs, labels, manifest = load_episode(episode, expected_identity_sha256=collection["identity_sha256"])
        association, states, oracle, pairs = validate_episode(inputs, labels, split)
        selected_row = names[episode.name]
        if association["scan_id"] != selected_row["scan"] or association["instr_id"] != selected_row["instr_id"]:
            raise ValueError("episode association differs from selected instruction")
        scene = by_scan.setdefault(association["scan_id"], _stats())
        _add_episode(total, inputs, labels, pairs)
        _add_episode(scene, inputs, labels, pairs)
        manifest_hashes[episode.name] = file_sha256(episode / "manifest.json")
    return {
        "schema": "duet_diagnostic_descriptive_summary_v1", "split": split, "usage": usage,
        "scope": "all_saved_states_and_eligible_oracle_action_disagreement",
        "collection_identity_sha256": collection["identity_sha256"],
        "episode_manifest_sha256": manifest_hashes,
        "num_scans": len(by_scan), "overall": _finish(total),
        "per_scan": {scan: _finish(stats) for scan, stats in sorted(by_scan.items())},
        "interpretation": {
            "oracle_outcomes": "teacher/execution cost optimality; not navigation success or full instruction-following correctness",
            "stop": "STOP with nonfinite oracle cost is separate and receives no fabricated zero regret",
            "candidate_denominator": "legal unvisited non-STOP candidate occurrences; the same target at different steps is counted separately",
            "duplicates_denominator": "final cumulative counters once per episode-target, not a sum of repeated state snapshots",
            "natural_arrival": "only later natural observations; missing arrival remains unknown and selection-biased",
            "causality": "cross-tabulation is descriptive association and does not control geometry or policy score",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--split", choices=("train_fit", "train_dev"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize_collection(args.collection, split=args.split)
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "split": args.split,
                      "scans": result["num_scans"], "episodes": result["overall"]["episodes"],
                      "eligible_decisions": result["overall"]["eligible_decisions"],
                      "teacher": result["overall"]["teacher"]}, indent=2))


if __name__ == "__main__":
    main()
