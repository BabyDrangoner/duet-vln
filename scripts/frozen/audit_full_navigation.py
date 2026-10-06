#!/usr/bin/env python3
"""Independently audit the frozen E1 full val_unseen evaluations.

This file only reads completed reports. It never executes a policy, selects a
checkpoint, reads/writes an access ledger, or changes frozen evaluator code.
Pass both --annotations and --connectivity on the runtime for independent path
metric recomputation; omit both for a clearly labelled report-only audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import random

ARMS = ("C1", "C2", "C3", "M")
CONTRASTS = tuple(("baseline", arm) for arm in ARMS) + (("C1", "C2"), ("C2", "C3"), ("C3", "M"))
PLAN_SHA = "69ab33efb7b69917b313d8f59b376a6b75707275fb3de8aa196d936dca6581d3"
ANNOTATION_SHA = "29110ed14c22cba6ba12bfc2e5f4d3bfdc27a253ff47f55a7a06ab97c9b71d13"
METRICS = {"sr": "success", "spl": "spl", "nDTW": "nDTW"}
SUMMARY_FIELDS = {
    "action_steps": ("action_steps", 1), "steps": ("trajectory_steps", 1),
    "lengths": ("trajectory_lengths", 1), "nav_error": ("nav_error", 1),
    "oracle_error": ("oracle_error", 1), "sr": ("success", 100),
    "oracle_sr": ("oracle_success", 100), "spl": ("spl", 100),
    "nDTW": ("nDTW", 100), "SDTW": ("SDTW", 100), "CLS": ("CLS", 100),
}
SHARED_METADATA = (
    "dataset", "feature_id", "base_checkpoint_sha256", "upstream_commit",
    "max_action_len", "feedback", "model_config_sha256", "partition_seed",
    "dev_fraction", "train_annotation_sha256", "connectivity_sha256",
    "source_and_integration_sha256", "split", "protocol_sha256", "subset",
    "model", "seed", "num_episodes",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def obj_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def indexed(rows, label):
    ids = [r["instr_id"] for r in rows]
    require(len(ids) == len(set(ids)), label + ": duplicate instruction IDs")
    return {r["instr_id"]: r for r in rows}


def flatten(path):
    require(bool(path) and all(isinstance(p, list) and p for p in path), "empty trajectory/segment")
    result = [node for segment in path for node in segment]
    require(all(isinstance(node, str) and node for node in result), "invalid trajectory node")
    return result


def close(actual, expected, label, errors=None):
    require(math.isfinite(actual) and math.isfinite(expected), label + ": nonfinite metric")
    difference = abs(actual - expected)
    require(math.isclose(actual, expected, abs_tol=2e-10, rel_tol=2e-12),
            f"{label}: recomputed={actual}, reported={expected}")
    if errors is not None:
        errors[label.split(":")[-1]] = max(errors.get(label.split(":")[-1], 0), difference)


def validate_reports(plan, baseline_path, arm_paths):
    require(sha(baseline_path) == plan["baseline_reference"]["sha256"], "baseline hash differs from frozen plan")
    require(tuple(row["arm"] for row in plan["entries"]) == ARMS, "plan arm order differs")
    reports = {"baseline": read(baseline_path), **{arm: read(arm_paths[arm]) for arm in ARMS}}
    baseline = reports["baseline"]
    baseline_episodes = indexed(baseline["episodes"], "baseline")
    ids = set(baseline_episodes)
    require(len(ids) == 2349, "baseline must cover all 2349 instructions")
    require(baseline["metadata"]["split"] == "val_unseen" and baseline["metadata"]["subset"] is False,
            "audit requires full val_unseen")
    source_files = plan["source_identity"]["execution_code_files"]
    require(obj_sha(source_files) == plan["source_identity"]["execution_code_files_sha256"], "plan aggregate code identity invalid")
    episode_maps, trajectory_maps = {}, {}
    summary_checks = {}
    for name, report in reports.items():
        for key in SHARED_METADATA:
            require(report["metadata"][key] == baseline["metadata"][key], name + ": shared metadata differs: " + key)
        episodes = indexed(report["episodes"], name + " episodes")
        trajectories = indexed(report["trajectories"], name + " trajectories")
        require(set(episodes) == set(trajectories) == ids, name + ": instruction inventory differs")
        for instr, episode in episodes.items():
            require(episode["scan_id"] == baseline_episodes[instr]["scan_id"], name + ": scan differs")
            for field in ("success", "spl", "oracle_success", "nDTW", "SDTW", "CLS"):
                require(0 <= episode[field] <= 1, f"{name}:{instr}:{field} outside [0,1]")
            require(episode["success"] in (0.0, 1.0), "fractional success")
            require(episode["success"] == float(episode["nav_error"] < 3.0), "SR threshold mismatch")
            nodes = flatten(trajectories[instr]["trajectory"])
            require(episode["trajectory_steps"] == len(nodes) - 1, "trajectory steps mismatch")
            require(episode["action_steps"] == len(trajectories[instr]["trajectory"]) - 1, "action steps mismatch")
        errors, recomputed = {}, {}
        for metric, (field, factor) in SUMMARY_FIELDS.items():
            value = factor * math.fsum(episode[field] for episode in episodes.values()) / len(ids)
            close(value, report["summary"][metric], name + ":summary:" + metric, errors)
            recomputed[metric] = value
        summary_checks[name] = {"recomputed_summary": recomputed, "maximum_absolute_errors": errors}
        episode_maps[name], trajectory_maps[name] = episodes, trajectories
    for row in plan["entries"]:
        arm, report = row["arm"], reports[row["arm"]]
        metadata = report["metadata"]
        expected = {
            "group_arm": arm, "head_training_seed": 0, "navigation_seed": 0,
            "checkpoint_selection": "fixed_final_epoch", "endpoint_only": True,
            "head_sha256": row["fixed_checkpoint"]["head_sha256"],
            "endpoint_experiment_sha256": row["registration_request"]["config_sha256"],
            "endpoint_code_sha256": plan["source_identity"]["execution_code_files_sha256"],
            "endpoint_code_files": source_files, "baseline_report_sha256": sha(baseline_path),
            "access_id": row["registration_request"]["access_id"],
            "all_online_path_and_termination_parity": True, "historical_score": "goal_head_logit",
        }
        for key, value in expected.items():
            require(metadata.get(key) == value, arm + ": frozen identity differs: " + key)
        decisions = indexed(report["endpoint_decisions"], arm + " endpoint decisions")
        require(set(decisions) == ids, arm + ": decision coverage differs")
        for instr, decision in decisions.items():
            prefix = decision["prefix_path"]
            require(decision["online_path_and_termination_parity"] is True, arm + ": reported parity failure")
            base_path = trajectory_maps["baseline"][instr]["trajectory"]
            arm_path = trajectory_maps[arm][instr]["trajectory"]
            require(base_path[:len(prefix)] == arm_path[:len(prefix)] == prefix, arm + ": online prefix differs")
            for path, endpoint_field in ((base_path, "baseline_endpoint"), (arm_path, "probe_endpoint")):
                require(len(path) in (len(prefix), len(prefix) + 1), arm + ": extra return segments")
                require(flatten(path)[-1] == decision[endpoint_field], arm + ": endpoint differs from trajectory")
                require((len(path) == len(prefix)) == (flatten(prefix)[-1] == decision[endpoint_field]),
                        arm + ": historical return presence differs")
            require(decision["endpoint_changed"] == (decision["baseline_endpoint"] != decision["probe_endpoint"]),
                    arm + ": endpoint change flag differs")
            if not decision["endpoint_changed"]:
                require(base_path == arm_path, arm + ": unchanged endpoint has changed path")
        if arm != "C1":
            prior = indexed(reports["C1"]["endpoint_decisions"], "C1")
            for instr in ids:
                require(decisions[instr]["prefix_path"] == prior[instr]["prefix_path"], "cross-arm online prefix differs")
                require(decisions[instr]["online_decisions"] == prior[instr]["online_decisions"], "cross-arm decision count differs")
    return reports, episode_maps, trajectory_maps, summary_checks


def load_graphs(directory, scans, expected_identity):
    files = sorted(Path(directory).glob("*_connectivity.json"))
    inventory = {p.name: sha(p) for p in files}
    require(obj_sha(inventory) == expected_identity, "complete connectivity inventory hash differs")
    graphs, selected_hashes = {}, {}
    for scan in sorted(scans):
        path = Path(directory) / (scan + "_connectivity.json")
        records = read(path)
        graph = defaultdict(dict)
        for i, row in enumerate(records):
            if not row["included"]:
                continue
            for j, connected in enumerate(row["unobstructed"]):
                other = records[j]
                if connected and other["included"]:
                    require(other["unobstructed"][i], "connectivity must be symmetric")
                    length = sum((row["pose"][k] - other["pose"][k]) ** 2 for k in (3, 7, 11)) ** 0.5
                    graph[row["image_id"]][other["image_id"]] = length
        require(bool(graph), "empty navigation graph")
        graphs[scan] = dict(graph)
        selected_hashes[scan] = sha(path)
    return graphs, {"all_files_sha256": inventory, "inventory_sha256": obj_sha(inventory), "used_files_sha256": selected_hashes}


class Distances:
    def __init__(self, graph):
        self.graph, self.cached = graph, {}

    def __call__(self, source, target):
        if source not in self.cached:
            # Independent Dijkstra implementation, no imports from evaluator or upstream.
            distances, queue = {source: 0.0}, [(0.0, source)]
            while queue:
                length, node = heapq.heappop(queue)
                if length != distances[node]:
                    continue
                for other, edge in self.graph[node].items():
                    candidate = length + edge
                    if candidate < distances.get(other, math.inf):
                        distances[other] = candidate
                        heapq.heappush(queue, (candidate, other))
            self.cached[source] = distances
        return self.cached[source][target]


def path_length(nodes, distance):
    return math.fsum(distance(a, b) for a, b in zip(nodes, nodes[1:]))


def recompute_metrics(segments, reference, distance):
    prediction = flatten(segments)
    require(prediction[0] == reference[0], "prediction start differs from reference")
    nav_error = distance(prediction[-1], reference[-1])
    oracle_error = min(distance(node, reference[-1]) for node in prediction)
    actual, shortest_reference = path_length(prediction, distance), path_length(reference, distance)
    success = float(nav_error < 3.0)
    previous = [0.0] + [math.inf] * len(reference)
    for node in prediction:
        current = [math.inf]
        for j, goal in enumerate(reference, 1):
            current.append(distance(node, goal) + min(previous[j], current[-1], previous[j - 1]))
        previous = current
    dtw = previous[-1]
    ndtw = math.exp(-dtw / (3.0 * len(reference)))
    coverage = math.fsum(math.exp(-min(distance(goal, node) for node in prediction) / 3.0)
                         for goal in reference) / len(reference)
    expected_length = coverage * shortest_reference
    cls = coverage * expected_length / (expected_length + abs(expected_length - actual))
    return {"nav_error": nav_error, "oracle_error": oracle_error,
            "action_steps": len(segments) - 1, "trajectory_steps": len(prediction) - 1,
            "trajectory_lengths": actual, "success": success,
            "spl": success * shortest_reference / max(actual, shortest_reference, 0.01),
            "oracle_success": float(oracle_error < 3.0), "DTW": dtw,
            "nDTW": ndtw, "SDTW": success * ndtw, "CLS": cls}


def graph_audit(reports, episodes, trajectories, annotations, connectivity):
    require(sha(annotations) == ANNOTATION_SHA, "validation annotation hash differs from frozen asset manifest")
    ground_truth = {}
    for row in read(annotations):
        for i in range(len(row["instructions"])):
            instr = f"{row['path_id']}_{i}"
            require(instr not in ground_truth, "duplicate annotation instruction")
            ground_truth[instr] = (row["scan"], row["path"])
    require(set(ground_truth) == set(episodes["baseline"]), "annotation instruction inventory differs")
    scans = {scan for scan, _ in ground_truth.values()}
    graphs, identity = load_graphs(connectivity, scans, reports["baseline"]["metadata"]["connectivity_sha256"])
    distances = {scan: Distances(graph) for scan, graph in graphs.items()}
    errors, checked = {}, 0
    for name in ("baseline",) + ARMS:
        per_metric = {}
        for instr in sorted(ground_truth):
            scan, reference = ground_truth[instr]
            require(episodes[name][instr]["scan_id"] == scan, "annotation scan differs")
            values = recompute_metrics(trajectories[name][instr]["trajectory"], reference, distances[scan])
            for key, value in values.items():
                close(value, episodes[name][instr][key], f"{name}:{instr}:{key}", per_metric)
                checked += 1
        errors[name] = per_metric
    return {"status": "passed", "episodes_checked": 5 * len(ground_truth), "metric_values_checked": checked,
            "implementation": "standalone stdlib graph construction, Dijkstra, path metrics and rolling-row DTW",
            "numeric_tolerance": {"absolute": 2e-10, "relative": 2e-12,
                                  "reason": "independent accumulation order and scalar exp may differ by floating-point roundoff"},
            "max_absolute_errors": errors,
            "annotation": {"path": str(annotations), "sha256": sha(annotations)},
            "connectivity": identity}, distances


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def bootstrap(base, method):
    scenes = defaultdict(list)
    for instr in sorted(base):
        require(base[instr]["scan_id"] == method[instr]["scan_id"], "paired scene mismatch")
        scenes[base[instr]["scan_id"]].append(instr)
    totals = []
    for scene in sorted(scenes):
        ids = scenes[scene]
        totals.append((len(ids), {metric: math.fsum(method[i][field] - base[i][field] for i in ids)
                                  for metric, field in METRICS.items()}))
    rng, samples = random.Random(0), {metric: [] for metric in METRICS}
    for _ in range(20000):
        selected = [totals[rng.randrange(len(totals))] for _ in totals]
        denominator = sum(row[0] for row in selected)
        for metric in METRICS:
            samples[metric].append(100 * math.fsum(row[1][metric] for row in selected) / denominator)
    result = {}
    for metric, field in METRICS.items():
        bm = math.fsum(base[i][field] for i in sorted(base)) / len(base)
        mm = math.fsum(method[i][field] for i in sorted(base)) / len(base)
        result[metric] = {"baseline": bm, "method": mm, "delta_pp": 100 * (mm - bm),
                          "ci95_pp": [quantile(samples[metric], .025), quantile(samples[metric], .975)]}
    return {"n_episodes": len(base), "n_scenes": len(scenes), "metrics": result}


def return_decomposition(arm, reports, episodes, trajectories, distances):
    base, method = episodes["baseline"], episodes[arm]
    categories = {key: [] for key in ("rescued", "harmed", "both_success", "both_failure")}
    changed, detailed, base_return, method_return = [], [], [], []
    decisions = indexed(reports[arm]["endpoint_decisions"], arm)
    for instr in sorted(base):
        bs, ms = bool(base[instr]["success"]), bool(method[instr]["success"])
        category = "both_success" if bs and ms else "harmed" if bs else "rescued" if ms else "both_failure"
        categories[category].append(instr)
        decision = decisions[instr]
        if decision["endpoint_changed"]:
            changed.append(instr)
        length_delta = method[instr]["trajectory_lengths"] - base[instr]["trajectory_lengths"]
        value = {"instr_id": instr, "scan_id": base[instr]["scan_id"], "category": category,
                 "endpoint_changed": decision["endpoint_changed"], "return_distance_delta_m": length_delta,
                 "spl_delta": method[instr]["spl"] - base[instr]["spl"]}
        if distances is not None:
            distance = distances[base[instr]["scan_id"]]
            prefix_length = path_length(flatten(decision["prefix_path"]), distance)
            br = base[instr]["trajectory_lengths"] - prefix_length
            mr = method[instr]["trajectory_lengths"] - prefix_length
            require(br > -2e-10 and mr > -2e-10, "negative historical return length")
            base_return.append(br)
            method_return.append(mr)
            value.update(baseline_return_m=br, method_return_m=mr)
        detailed.append(value)
    contributions = {}
    for key, ids in categories.items():
        contributions[key] = {"count": len(ids), "ids": ids,
                             "spl_contribution_pp": 100 * math.fsum(method[i]["spl"] - base[i]["spl"] for i in ids) / len(base),
                             "return_distance_delta_total_m": math.fsum(method[i]["trajectory_lengths"] - base[i]["trajectory_lengths"] for i in ids),
                             "changed_endpoints": sum(decisions[i]["endpoint_changed"] for i in ids)}
    close(math.fsum(row["spl_contribution_pp"] for row in contributions.values()),
          100 * math.fsum(method[i]["spl"] - base[i]["spl"] for i in base) / len(base), "SPL decomposition")
    result = {"changed_endpoints": len(changed), "unchanged_endpoints": len(base) - len(changed),
              "rescued_count": len(categories["rescued"]), "harmed_count": len(categories["harmed"]),
              "delta_sr_pp_from_rescues_minus_harms": 100 * (len(categories["rescued"]) - len(categories["harmed"])) / len(base),
              "return_distance_delta_total_m": math.fsum(row["return_distance_delta_m"] for row in detailed),
              "return_distance_delta_mean_m": math.fsum(row["return_distance_delta_m"] for row in detailed) / len(base),
              "categories": contributions, "per_episode": detailed,
              "cost_definition": "method full path length minus baseline full path length; equal to historical return difference because online prefixes match"}
    if distances is not None:
        result["baseline_return_total_m"] = math.fsum(base_return)
        result["method_return_total_m"] = math.fsum(method_return)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument("--" + arm, type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--connectivity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require((args.annotations is None) == (args.connectivity is None), "pass both annotation and connectivity paths")
    require(sha(args.plan) == PLAN_SHA, "frozen plan hash differs")
    require(not args.output.exists(), "audit output exists; preserve it and use a new filename")
    arm_paths = {arm: getattr(args, arm) for arm in ARMS}
    plan = read(args.plan)
    reports, episodes, trajectories, summary_checks = validate_reports(plan, args.baseline, arm_paths)
    graph_check, distances = ({"status": "not_run", "reason": "report-only mode; no ground-truth or connectivity loaded"}, None)
    if args.annotations:
        graph_check, distances = graph_audit(reports, episodes, trajectories, args.annotations, args.connectivity)
    comparisons = {method + "_minus_" + base: bootstrap(episodes[base], episodes[method]) for base, method in CONTRASTS}
    result = {"schema": "e1_full_val_unseen_independent_audit_v1", "status": "passed",
              "recorded_utc": datetime.now(timezone.utc).isoformat(), "audit_code_sha256": sha(__file__),
              "verification_mode": "independent_graph_and_report_audit" if args.annotations else "report_only_audit",
              "scope": "all four frozen seed-0 final heads, full 2349-instruction R2R val_unseen development validation; not official blind test",
              "input_sha256": {"plan": sha(args.plan), "baseline": sha(args.baseline), **{arm: sha(path) for arm, path in arm_paths.items()}},
              "checks": {"all_instructions_present": True, "all_frozen_head_and_protocol_identities_match": True,
                         "all_online_prefixes_and_reported_termination_match": True, "unchanged_endpoints_have_identical_paths": True,
                         "all_11_summary_fields_recomputed": True},
              "summary_checks": summary_checks, "graph_metric_audit": graph_check,
              "bootstrap": {"unit": "scan_id", "paired": True, "resamples": 20000, "seed": 0,
                            "confidence_level": .95, "method": "percentile", "aggregation": "episode_weighted",
                            "scope": "evaluation scenes; single training seed, no training-seed uncertainty",
                            "multiplicity": "seven predeclared contrasts, pointwise intervals; no multiple-comparison adjustment"},
              "comparisons": comparisons,
              "return_cost_decomposition": {arm: return_decomposition(arm, reports, episodes, trajectories, distances) for arm in ARMS},
              "limitations": ["val_unseen is already exposed development validation, not a blind test.",
                              "All heads share the single training seed 0; this audit cannot establish robustness across seeds.",
                              "Online policy logits are not saved in reports; prefix parity is checked directly and termination parity also relies on frozen runtime assertions."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "mode": result["verification_mode"],
                      "summary": {name: row["recomputed_summary"] for name, row in summary_checks.items()},
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
