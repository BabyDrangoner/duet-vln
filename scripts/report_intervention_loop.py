#!/usr/bin/env python3
"""Audit and summarize the prespecified two-arm E2 full navigation loop.

This CPU-only report reads completed navigation outputs. It never trains,
changes a policy, selects a checkpoint, or appends validation accesses.
Independent graph metrics come from the SHA-pinned frozen E1 audit helper.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "outputs/study-20261005/audit_full_navigation.py"
HELPER_SHA256 = "ad652843e77526b1a72a66e4a36cc9175523050b7714ed21cf76df0e9083d96d"
BASELINE_SHA256 = "32e4b21422a86c2b8a7e81eb1b423b903ac19cf25972d450c003bdd33d57cd61"
ANNOTATION_SHA256 = "29110ed14c22cba6ba12bfc2e5f4d3bfdc27a253ff47f55a7a06ab97c9b71d13"
ARMS = ("relative", "absolute")
NAMES = ("baseline",) + ARMS
CONTRASTS = (("baseline", "relative"), ("baseline", "absolute"), ("absolute", "relative"))
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 20261006, 10000
PRIMARY_METRICS = {"sr": "success", "spl": "spl", "nDTW": "nDTW"}
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


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def indexed(rows, label):
    require(isinstance(rows, list) and bool(rows), label + ": missing rows")
    ids = [row.get("instr_id") for row in rows]
    require(all(isinstance(i, str) and i for i in ids), label + ": invalid instruction ID")
    require(len(ids) == len(set(ids)), label + ": duplicate instruction IDs")
    return dict(zip(ids, rows))


def flatten(path):
    require(isinstance(path, list) and bool(path) and all(isinstance(p, list) and p for p in path),
            "empty trajectory/segment")
    nodes = [node for segment in path for node in segment]
    require(all(isinstance(node, str) and node for node in nodes), "invalid trajectory node")
    return nodes


def close(actual, expected, label):
    require(isinstance(actual, (int, float)) and isinstance(expected, (int, float))
            and not isinstance(actual, bool) and not isinstance(expected, bool)
            and math.isfinite(actual) and math.isfinite(expected), label + ": invalid numeric value")
    require(math.isclose(actual, expected, abs_tol=2e-10, rel_tol=2e-12),
            f"{label}: computed={actual}, reported={expected}")
    return abs(actual - expected)


def load_helper(path=HELPER):
    require(sha(path) == HELPER_SHA256, "frozen independent metric helper SHA differs")
    spec = importlib.util.spec_from_file_location("e2_pinned_metric_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_head_metadata(name, metadata):
    head = metadata.get("head_metadata", {})
    config = head.get("config", {})
    require(metadata.get("mode") == "e2_endpoint_intervention", name + ": wrong navigation mode")
    require(head.get("schema") == "e2_intervention_head_v1" and head.get("arm") == name,
            name + ": head schema/arm differs")
    require(config.get("arm") == name and config.get("seed") == 0,
            name + ": training arm/seed differs")
    require(valid_sha(metadata.get("head_sha256")), name + ": missing frozen head SHA")
    require(valid_sha(metadata.get("experiment_sha256"))
            and config.get("experiment_sha256") == metadata["experiment_sha256"],
            name + ": head not bound to experiment configuration")
    require(head.get("hidden_dim") == config.get("hidden_dim") == 128,
            name + ": unexpected head capacity")
    fixed = {"epochs": 20, "batch_size": 64, "lr": 1e-4, "weight_decay": .01,
             "monitor_every_epochs": 2, "risk_weight": 0.}
    require(all(config.get(k) == v for k, v in fixed.items()), name + ": fixed training budget differs")
    require(type(head.get("epoch")) is int and head["epoch"] in range(2, 21, 2)
            and head.get("global_step") == head["epoch"] * 128,
            name + ": not a trained monitored first-loop checkpoint")
    require(head.get("selection_rule") == "natural_train_dev_SR_and_SPL_at_least_baseline_then_SR_SPL_earliest",
            name + ": checkpoint selection rule differs")
    require(valid_sha(head.get("initial_head_sha256")), name + ": missing initialization identity")
    require(isinstance(head.get("data_identity"), dict) and bool(head["data_identity"]),
            name + ": missing training data identity")
    files = metadata.get("e2_code_files")
    require(isinstance(files, dict) and bool(files) and all(valid_sha(v) for v in files.values())
            and metadata.get("e2_code_sha256") == object_sha(files), name + ": invalid evaluation code identity")
    training_files = head.get("code_identity")
    require(isinstance(training_files, dict) and bool(training_files)
            and all(valid_sha(v) for v in training_files.values()), name + ": missing training code identity")
    require(all(training_files[k] == files[k] for k in set(training_files) & set(files)),
            name + ": shared training/evaluation code changed")
    require(isinstance(metadata.get("access_id"), str)
            and re.fullmatch(r"V[0-9]+", metadata["access_id"]) is not None,
            name + ": missing pilot access ID")
    claim = metadata.get("validation_execution", {})
    require(valid_sha(claim.get("sha256")) and valid_sha(claim.get("ledger_snapshot_sha256")),
            name + ": missing permanent execution claim identity")


def verify_decisions(name, report, base_paths, method_paths, episodes):
    if "endpoint_decisions" not in report:
        return {"status": "not_available", "reason": "report has no endpoint_decisions; online-prefix parity cannot be audited"}, None
    require(report["metadata"].get("all_online_path_and_termination_parity") is True,
            name + ": runtime parity assertion missing/false")
    decisions = indexed(report["endpoint_decisions"], name + " endpoint decisions")
    require(set(decisions) == set(base_paths), name + ": incomplete endpoint decision inventory")
    for instr, decision in decisions.items():
        label = name + ":" + instr
        base_path = base_paths[instr]["trajectory"]
        method_path = method_paths[instr]["trajectory"]
        prefix = decision["prefix_path"]
        terminal = flatten(prefix)[-1]
        require(decision.get("scan_id") == episodes[instr]["scan_id"], label + ": decision scene differs")
        require(decision.get("condition") == "natural"
                and decision.get("perturbation", {}).get("applied") is False,
                label + ": official navigation must be natural")
        if "online_path_and_termination_parity" in decision:
            require(decision["online_path_and_termination_parity"] is True, label + ": decision parity false")
        require(base_path[:len(prefix)] == method_path[:len(prefix)] == prefix,
                label + ": online prefix differs")
        require(decision.get("baseline_trajectory") == base_path
                and decision.get("selected_trajectory") == method_path,
                label + ": stored complete path differs")
        require(decision.get("termination_endpoint") == terminal, label + ": termination differs")
        candidates = decision.get("candidate_vpids")
        require(isinstance(candidates, list) and 1 <= len(candidates) <= 15
                and len(candidates) == len(set(candidates))
                and candidates == [segment[-1] for segment in prefix],
                label + ": candidates differ from actual decision observations")
        baseline, selected = decision["baseline_endpoint"], decision["selected_endpoint"]
        require(baseline in candidates and selected in candidates, label + ": illegal endpoint")
        for path, endpoint in ((base_path, baseline), (method_path, selected)):
            require(len(path) in (len(prefix), len(prefix) + 1), label + ": extra return segments")
            require(flatten(path)[-1] == endpoint, label + ": endpoint differs from trajectory")
            require((len(path) == len(prefix)) == (terminal == endpoint), label + ": return presence differs")
            # DUET graph.path omits the source node. The complete flattened
            # trajectory still contains the terminal-to-return first edge;
            # graph_audit independently checks that edge is executable.
        require(type(decision.get("endpoint_changed")) is bool
                and decision["endpoint_changed"] == (baseline != selected), label + ": change flag differs")
        if baseline == selected:
            require(base_path == method_path, label + ": KEEP path differs")
        gains = decision.get("predicted_gains")
        require(isinstance(gains, list) and len(gains) == len(candidates)
                and all(isinstance(row, list) and len(row) == 2
                        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                and math.isfinite(v) and -1 <= v <= 1 for v in row) for row in gains),
                label + ": invalid predicted gains")
        require(gains[candidates.index(baseline)] == [0., 0.], label + ": anchor prediction nonzero")
        eligible = [i for i, row in enumerate(gains) if row[0] >= 0 and row[1] > 0]
        best = max((gains[i][1] for i in eligible), default=None)
        winners = [i for i in eligible if gains[i][1] == best]
        predicted_selection = candidates[winners[0]] if len(winners) == 1 else baseline
        require(selected == predicted_selection, label + ": selection differs from fixed prediction rule")
    return {"status": "passed", "episodes_checked": len(decisions),
            "online_prefixes_equal": True, "original_return_equal": True,
            "selected_paths_equal": True, "keep_paths_equal": True,
            "observed_candidate_inventory_equal": True, "prediction_rule_replayed": True}, decisions


def validate_reports(reports, *, expected_episodes=2349, expected_scenes=11):
    require(set(reports) == set(NAMES), "need baseline, relative and absolute reports")
    baseline = reports["baseline"]
    metadata = baseline["metadata"]
    require(metadata.get("mode") == "baseline" and metadata.get("split") == "val_unseen"
            and metadata.get("subset") is False and metadata.get("seed") == 0,
            "baseline must be seed-0 full val_unseen")
    require(metadata.get("num_episodes") == expected_episodes, "baseline episode count differs")
    episode_maps, trajectory_maps, summary_checks, decision_checks, decision_maps = {}, {}, {}, {}, {}
    original = indexed(baseline["episodes"], "baseline")
    ids = set(original)
    require(len(ids) == expected_episodes, "wrong instruction coverage")
    require(len({e["scan_id"] for e in original.values()}) == expected_scenes, "wrong scene coverage")
    for name in NAMES:
        report = reports[name]
        for key in SHARED_METADATA:
            require(key in metadata and report["metadata"].get(key) == metadata[key],
                    name + ": shared protocol differs: " + key)
        episodes = indexed(report["episodes"], name + " episodes")
        trajectories = indexed(report["trajectories"], name + " trajectories")
        require(set(episodes) == set(trajectories) == ids, name + ": instruction inventory differs")
        for instr, episode in episodes.items():
            label = name + ":" + instr
            require(episode["scan_id"] == original[instr]["scan_id"], label + ": scene differs")
            for field in ("success", "spl", "oracle_success", "nDTW", "SDTW", "CLS"):
                value = episode[field]
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and 0 <= value <= 1, label + ": bounded metric invalid: " + field)
            require(episode["success"] in (0., 1.) and episode["oracle_success"] in (0., 1.),
                    label + ": nonbinary success")
            require(episode["success"] == float(episode["nav_error"] < 3.), label + ": SR threshold mismatch")
            require(episode["oracle_success"] == float(episode["oracle_error"] < 3.), label + ": oracle threshold mismatch")
            for field in ("nav_error", "oracle_error", "trajectory_lengths", "DTW"):
                require(isinstance(episode[field], (int, float)) and math.isfinite(episode[field])
                        and episode[field] >= 0, label + ": invalid distance: " + field)
            path = trajectories[instr]["trajectory"]
            require(episode["trajectory_steps"] == len(flatten(path)) - 1, label + ": step count mismatch")
            require(episode["action_steps"] == len(path) - 1, label + ": action count mismatch")
        recomputed, errors = {}, {}
        for metric, (field, factor) in SUMMARY_FIELDS.items():
            value = factor * math.fsum(episodes[i][field] for i in sorted(ids)) / len(ids)
            errors[metric] = close(value, report["summary"][metric], name + ":summary:" + metric)
            recomputed[metric] = value
        episode_maps[name], trajectory_maps[name] = episodes, trajectories
        summary_checks[name] = {"recomputed_summary": recomputed, "absolute_errors": errors}
    for name in ARMS:
        validate_head_metadata(name, reports[name]["metadata"])
        require(reports[name]["metadata"].get("baseline_report_sha256") == BASELINE_SHA256,
                name + ": baseline report identity differs")
        check, decisions = verify_decisions(name, reports[name], trajectory_maps["baseline"], trajectory_maps[name], episode_maps[name])
        decision_checks[name], decision_maps[name] = check, decisions
    relative, absolute = (reports[name]["metadata"] for name in ARMS)
    for key in ("e2_code_files", "e2_code_sha256"):
        require(relative[key] == absolute[key], "two arms evaluated with different code")
    for key in ("data_identity", "code_identity", "initial_head_sha256", "hidden_dim"):
        require(relative["head_metadata"][key] == absolute["head_metadata"][key], "two-arm training identity differs: " + key)
    rc, ac = (m["head_metadata"]["config"] for m in (relative, absolute))
    require({k: v for k, v in rc.items() if k not in {"arm", "experiment_sha256"}}
            == {k: v for k, v in ac.items() if k not in {"arm", "experiment_sha256"}},
            "two-arm training settings differ beyond objective")
    require(relative["experiment_sha256"] != absolute["experiment_sha256"],
            "distinct arms must have distinct method configuration identities")
    require(relative["access_id"] != absolute["access_id"], "two arms share one validation access")
    if all(decision_maps.values()):
        require(all(decision_maps["relative"][i]["prefix_path"] == decision_maps["absolute"][i]["prefix_path"]
                    for i in ids), "two-arm online prefixes differ")
    return episode_maps, trajectory_maps, summary_checks, decision_checks


def graph_audit(reports, episodes, trajectories, annotations, connectivity, helper):
    require(sha(annotations) == ANNOTATION_SHA256, "validation annotation asset SHA differs")
    truth = {}
    for row in read(annotations):
        for index in range(len(row["instructions"])):
            instr = f"{row['path_id']}_{index}"
            require(instr not in truth, "duplicate annotation instruction")
            truth[instr] = (row["scan"], row["path"])
    require(set(truth) == set(episodes["baseline"]), "annotations differ from complete report inventory")
    scans = {scan for scan, _ in truth.values()}
    graphs, identity = helper.load_graphs(connectivity, scans, reports["baseline"]["metadata"]["connectivity_sha256"])
    distances = {scan: helper.Distances(graph) for scan, graph in graphs.items()}
    errors, checked, transitions = {}, 0, 0
    for name in NAMES:
        per_metric = {}
        for instr in sorted(truth):
            scan, reference = truth[instr]
            require(episodes[name][instr]["scan_id"] == scan, "annotation/report scene differs")
            segments = trajectories[name][instr]["trajectory"]
            nodes = flatten(segments)
            for left, right in zip(nodes, nodes[1:]):
                require(left == right or right in graphs[scan].get(left, {}),
                        f"{name}:{instr}: nonexecutable trajectory edge {left}->{right}")
                transitions += 1
            values = helper.recompute_metrics(segments, reference, distances[scan])
            for key, value in values.items():
                error = close(value, episodes[name][instr][key], f"{name}:{instr}:{key}")
                per_metric[key] = max(per_metric.get(key, 0.), error)
                checked += 1
        errors[name] = per_metric
    return {"status": "passed", "episodes_checked": len(NAMES) * len(truth),
            "metric_values_checked": checked, "actual_graph_transitions_checked": transitions,
            "implementation": "SHA-pinned independent stdlib graph construction, Dijkstra, path metrics and DTW",
            "helper_sha256": HELPER_SHA256, "max_absolute_errors": errors,
            "numeric_tolerance": {"absolute": 2e-10, "relative": 2e-12},
            "annotation": {"path": str(annotations), "sha256": sha(annotations)}, "connectivity": identity}


def quantile(values, q):
    require(bool(values) and 0 <= q <= 1, "invalid quantile input")
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if lo != hi else values[lo]


def paired_comparison(base, method, base_paths, method_paths, *, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    require(set(base) == set(method) == set(base_paths) == set(method_paths) and bool(base), "paired ID mismatch")
    require(type(replicates) is int and replicates > 0, "bootstrap replicates must be positive")
    scenes = defaultdict(list)
    counts = {key: 0 for key in ("rescues", "harms", "both_success", "both_failure", "endpoint_changes")}
    for instr in sorted(base):
        require(base[instr]["scan_id"] == method[instr]["scan_id"], "paired scene mismatch")
        scenes[base[instr]["scan_id"]].append(instr)
        bs, ms = bool(base[instr]["success"]), bool(method[instr]["success"])
        key = "both_success" if bs and ms else "harms" if bs else "rescues" if ms else "both_failure"
        counts[key] += 1
        counts["endpoint_changes"] += flatten(base_paths[instr]["trajectory"])[-1] != flatten(method_paths[instr]["trajectory"])[-1]
    totals = [(len(ids), {metric: math.fsum(method[i][field] - base[i][field] for i in ids)
                           for metric, field in PRIMARY_METRICS.items()}) for _, ids in sorted(scenes.items())]
    rng, samples = random.Random(seed), {metric: [] for metric in PRIMARY_METRICS}
    for _ in range(replicates):
        selected = [totals[rng.randrange(len(totals))] for _ in totals]
        denominator = sum(row[0] for row in selected)
        for metric in PRIMARY_METRICS:
            samples[metric].append(100 * math.fsum(row[1][metric] for row in selected) / denominator)
    metrics = {}
    for metric, field in PRIMARY_METRICS.items():
        bm = math.fsum(base[i][field] for i in sorted(base)) / len(base)
        mm = math.fsum(method[i][field] for i in sorted(base)) / len(base)
        metrics[metric] = {"baseline_percent": 100 * bm, "method_percent": 100 * mm,
                           "delta_pp": 100 * (mm - bm),
                           "ci95_pp": [quantile(samples[metric], .025), quantile(samples[metric], .975)]}
    other = {field: math.fsum(method[i][field] - base[i][field] for i in sorted(base)) / len(base)
             for field in ("nav_error", "trajectory_lengths")}
    sr_delta = 100 * (counts["rescues"] - counts["harms"]) / len(base)
    close(sr_delta, metrics["sr"]["delta_pp"], "rescues-minus-harms SR identity")
    return {"episodes": len(base), "scenes": len(scenes), "metrics": metrics,
            "paired_counts": counts, "endpoint_change_percent": 100 * counts["endpoint_changes"] / len(base),
            "delta_sr_pp_from_counts": sr_delta, "mean_differences_m": other}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "relative", "absolute", "annotations", "connectivity", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "preserve previous report; output already exists")
    require(sha(args.baseline) == BASELINE_SHA256, "baseline differs from frozen original report")
    started = time.monotonic()
    helper = load_helper()
    paths = {name: getattr(args, name) for name in NAMES}
    reports = {name: read(path) for name, path in paths.items()}
    episodes, trajectories, summaries, prefix_checks = validate_reports(reports)
    graph_checks = graph_audit(reports, episodes, trajectories, args.annotations, args.connectivity, helper)
    comparisons = {method + "_minus_" + base: paired_comparison(episodes[base], episodes[method], trajectories[base], trajectories[method])
                   for base, method in CONTRASTS}
    scenes = sorted({r["scan_id"] for r in episodes["baseline"].values()})
    by_scene = {}
    for name in NAMES:
        by_scene[name] = {}
        for scene in scenes:
            rows = [r for r in episodes[name].values() if r["scan_id"] == scene]
            by_scene[name][scene] = {"episodes": len(rows), **{
                metric: factor * math.fsum(row[field] for row in rows) / len(rows)
                for metric, (field, factor) in SUMMARY_FIELDS.items()}}
    result = {"schema": "e2_complete_navigation_loop_report_v1", "status": "passed",
        "recorded_utc": datetime.now(timezone.utc).isoformat(), "report_code_sha256": sha(__file__),
        "scope": "first fixed two-arm seed-0 complete val_unseen development loop; not blind test or confirmed paper contribution",
        "input_sha256": {name: sha(path) for name, path in paths.items()},
        "head_identities": {name: {k: reports[name]["metadata"][k] for k in
            ("head_sha256", "head_metadata", "experiment_sha256", "access_id", "e2_code_sha256")}
            for name in ARMS},
        "checks": {"complete_2349_instructions_11_scenes": True, "shared_backbone_and_navigation_protocol": True,
                   "matched_training_data_capacity_initialization_and_budget": True,
                   "all_summary_metrics_recomputed": True},
        "summary_checks": summaries, "online_prefix_checks": prefix_checks,
        "graph_metric_audit": graph_checks, "by_scene": by_scene,
        "comparisons": comparisons,
        "bootstrap": {"unit": "scan_id", "paired": True, "resamples": BOOTSTRAP_REPLICATES,
                      "seed": BOOTSTRAP_SEED, "confidence_level": .95, "method": "percentile",
                      "aggregation": "episode-weighted with whole-scene resampling and multiplicity",
                      "multiplicity": "three predeclared comparisons; pointwise intervals, no multiple-comparison correction"},
        "resources": {"cpu_report_wall_seconds": time.monotonic() - started, "gpu_seconds": 0, "new_navigation_episodes": 0},
        "limitations": ["Validation is exposed development data, not a hidden test.",
                        "One training seed does not establish training robustness.",
                        "This checks predictions, complete trajectories and metrics; model-selection provenance additionally requires the frozen training and ledger records.",
                        "An interval describes uncertainty across 11 evaluation scenes, not unseen training-seed variation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "summary": {n: r["recomputed_summary"] for n, r in summaries.items()},
                      "comparisons": comparisons}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
