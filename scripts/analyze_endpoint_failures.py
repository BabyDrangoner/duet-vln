#!/usr/bin/env python3
"""Describe fixed E1 endpoint failures; never fit, rank, or simulate a policy.

Requires a matching open diagnostic ledger registration before reading any
validation report. Hash verification itself does not parse report contents.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def index(rows, association=False):
    pairs = [(r["association"]["instr_id"] if association else r["instr_id"], r) for r in rows]
    result = dict(pairs)
    require(len(result) == len(pairs), "duplicate instruction identifier")
    return result


def close(a, b, name, tolerance=2e-5):
    # The S1 graph distances were recorded in FP32; report path metrics use FP64.
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=tolerance, rel_tol=1e-6), name)


def bucket(value, upper_bounds):
    for i, bound in enumerate(upper_bounds):
        if value <= bound:
            return ("le_" if i == 0 else "gt_" + str(upper_bounds[i - 1]) + "_le_") + str(bound)
    raise ValueError(f"value {value} outside preregistered bins")


def category(base_success, new_success, changed):
    if not base_success and new_success:
        return "rescue"
    if base_success and not new_success:
        return "harm"
    if base_success and new_success:
        return "both_success_changed" if changed else "both_success_unchanged"
    return "both_failure"


def mean(values):
    return math.fsum(values) / len(values) if values else None


def support(rows):
    return {"instructions": len(rows), "independent_paths": len({(r["scan_id"], r["path_id"]) for r in rows}),
            "scenes": len({r["scan_id"] for r in rows})}


def summarize(rows, denominator):
    n = len(rows)
    result = {**support(rows), "whole_split_denominator": denominator,
              "changed_endpoints": sum(r["changed"] for r in rows),
              "rescues": sum(r["category"] == "rescue" for r in rows),
              "harms": sum(r["category"] == "harm" for r in rows)}
    for metric in ("sr", "spl", "ndtw"):
        result[metric + "_contribution_pp"] = 100 * math.fsum(r["delta_" + metric] for r in rows) / denominator
        result[metric + "_delta_within_stratum_pp"] = (100 * mean([r["delta_" + metric] for r in rows])) if n else None
    for name in ("return_delta_m", "baseline_return_m", "selected_return_m", "baseline_spl", "selected_spl",
                 "baseline_stop_probability", "selected_baseline_stop_probability",
                 "baseline_endpoint_age_steps", "selected_endpoint_age_steps",
                 "baseline_legal_move_count", "selected_legal_move_count", "history_states"):
        result["mean_" + name] = mean([r[name] for r in rows])
    result["total_return_delta_m"] = math.fsum(r["return_delta_m"] for r in rows)
    result["costlier_returns"] = sum(r["return_delta_m"] > 1e-8 for r in rows)
    result["cheaper_returns"] = sum(r["return_delta_m"] < -1e-8 for r in rows)
    result["selected_age_bins"] = dict(Counter(r["selected_age_bin"] for r in rows))
    result["selected_move_count_bins"] = dict(Counter(r["selected_move_count_bin"] for r in rows))
    return result


def validate_registration(config_path, config, ledger, study):
    from vln_improve.study_ledger import StudyLedger
    registration = StudyLedger(ledger, read(study)).lookup(config["access_id"])
    require(registration["outcome"] is None, "diagnostic access has already been closed")
    request = registration["registration"]["request"]
    expected = {"category": "diagnostic_analysis", "split": "val_unseen", "label_use": "analysis",
                "config_sha256": sha(config_path), "code_sha256": sha(__file__), "seed": 0,
                "expected_episodes": config["expected_episodes"], "subset": False, "subset_ids": []}
    for key, value in expected.items():
        require(request.get(key) == value, "diagnostic registration mismatch: " + key)
    return registration["registration"]


def analyze(config, documents):
    arms = config["arms"]
    reports = {name: documents[name] for name in ["baseline", *arms]}
    episodes = {name: index(r["episodes"]) for name, r in reports.items()}
    trajectories = {name: index(r["trajectories"]) for name, r in reports.items()}
    base = episodes["baseline"]
    ids = set(base)
    require(len(ids) == config["expected_episodes"], "full instruction inventory mismatch")
    s1 = documents["S1"]
    require(s1["all_instruction_trajectory_and_metric_parity"] is True, "S1 parity not verified")
    require(s1["split"] == "val_unseen" and s1["usage"] == "analysis_only", "S1 scope mismatch")
    trace = index(s1["episodes"], association=True)
    collection = documents["S1_collection"]
    require(s1["identity_sha256"] == collection["identity_sha256"], "S1 identity mismatch")
    require(set(trace) == ids, "S1 instruction coverage mismatch")
    for name in reports:
        require(set(episodes[name]) == set(trajectories[name]) == ids, name + " coverage mismatch")
    decisions = {arm: index(reports[arm]["endpoint_decisions"]) for arm in arms}
    for arm in arms:
        require(set(decisions[arm]) == ids, "endpoint decision coverage mismatch")

    shared, rows_by_arm = [], {arm: [] for arm in arms}
    for instr in sorted(ids):
        b, t = base[instr], trace[instr]
        states = t["states"]
        require([r["step"] for r in states] == list(range(len(states))), "nonconsecutive S1 history")
        by_node = {r["viewpoint"]: r for r in states}
        require(len(by_node) == len(states) == t["num_states"], "repeated history nodes")
        require(t["association"]["scan_id"] == b["scan_id"], "scene mismatch")
        base_method = t["methods"]["baseline_probability"]
        node = trajectories["baseline"][instr]["trajectory"][-1][-1]
        require(base_method["viewpoint"] == node and bool(b["success"]) == base_method["success"], "S1 endpoint mismatch")
        close(base_method["nav_error_m"], b["nav_error"], "S1 baseline nav_error mismatch")
        close(base_method["complete_trajectory_length_m"], b["trajectory_lengths"], "S1 baseline length mismatch")
        baseline_state = by_node[node]
        require(t["observed_history_has_success"] == any(r["success"] for r in states), "history opportunity flag mismatch")
        term = "+".join(k for k in ("argmax_stop", "no_legal_move", "action_limit") if t["termination_flags"][k])
        require(bool(term), "episode lacks termination reason")
        item = {"instr_id": instr, "scan_id": b["scan_id"], "path_id": t["path_id"],
                "baseline_success": bool(b["success"]), "terminal_success": states[-1]["success"],
                "history_has_success": t["observed_history_has_success"],
                "recoverable_baseline_failure": bool(t["observed_history_has_success"] and not b["success"]),
                "standard_oracle_success": bool(b["oracle_success"]), "history_states": len(states),
                "termination_combination": term,
                "baseline_endpoint_age_steps": len(states) - 1 - baseline_state["step"],
                "baseline_stop_probability": baseline_state["scores"]["baseline_probability"],
                "baseline_legal_move_count": baseline_state["move_count"],
                "baseline_return_m": base_method["return_distance_m"], "prefix_length_m": t["prefix_length_m"]}
        shared.append(item)
        for arm in arms:
            e, d = episodes[arm][instr], decisions[arm][instr]
            require(d["baseline_endpoint"] == node and d["online_path_and_termination_parity"], "arm baseline/parity mismatch")
            require(d["online_decisions"] == len(states), "arm history size mismatch")
            require([segment[-1] for segment in d["prefix_path"]] == [s["viewpoint"] for s in states], "arm prefix observations differ")
            selected = by_node[d["probe_endpoint"]]
            require(bool(e["success"]) == selected["success"], "selected endpoint label mismatch")
            close(e["nav_error"], selected["distance_to_goal_m"], "selected nav_error mismatch")
            changed = node != selected["viewpoint"]
            require(changed == d["endpoint_changed"], "changed flag mismatch")
            if not changed:
                require(trajectories[arm][instr]["trajectory"] == trajectories["baseline"][instr]["trajectory"], "unchanged endpoint changed path")
            age = len(states) - 1 - selected["step"]
            return_delta = e["trajectory_lengths"] - b["trajectory_lengths"]
            selected_return = item["baseline_return_m"] + return_delta
            require(selected_return > -2e-5, "negative inferred return length")
            rows_by_arm[arm].append({**item, "arm": arm, "changed": changed,
                "category": category(bool(b["success"]), bool(e["success"]), changed),
                "selected_success": bool(e["success"]), "delta_sr": e["success"] - b["success"],
                "delta_spl": e["spl"] - b["spl"], "delta_ndtw": e["nDTW"] - b["nDTW"],
                "baseline_spl": b["spl"], "selected_spl": e["spl"], "return_delta_m": return_delta,
                "selected_return_m": max(0.0, selected_return), "selected_endpoint_age_steps": age,
                "selected_baseline_stop_probability": selected["scores"]["baseline_probability"],
                "selected_legal_move_count": selected["move_count"],
                "selected_age_bin": bucket(age, config["bins"]["endpoint_age_steps"]),
                "selected_move_count_bin": bucket(selected["move_count"], config["bins"]["legal_move_count"])})
    n = len(ids)
    opportunities = [r for r in shared if r["recoverable_baseline_failure"]]
    opportunity_count = len(opportunities)
    overall = {"all": support(shared), "baseline_success": support([r for r in shared if r["baseline_success"]]),
               "baseline_failure": support([r for r in shared if not r["baseline_success"]]),
               "observed_history_has_success": support([r for r in shared if r["history_has_success"]]),
               "recoverable_baseline_failure": support(opportunities),
               "baseline_failure_without_observed_success": support([r for r in shared if not r["baseline_success"] and not r["history_has_success"]]),
               "standard_oracle_only_failure": support([r for r in shared if not r["baseline_success"] and r["standard_oracle_success"] and not r["history_has_success"]]),
               "observed_history_oracle_sr_percent": 100 * sum(r["history_has_success"] for r in shared) / n,
               "no_harm_max_possible_sr_gain_pp": 100 * opportunity_count / n}
    findings = {}
    for arm, rows in rows_by_arm.items():
        all_summary = summarize(rows, n)
        all_summary["rescue_fraction_of_observed_opportunities"] = all_summary["rescues"] / opportunity_count if opportunity_count else None
        all_summary["harm_fraction_of_baseline_successes"] = all_summary["harms"] / overall["baseline_success"]["instructions"]
        by_category = {c: summarize([r for r in rows if r["category"] == c], n) for c in
                       ("rescue", "harm", "both_success_changed", "both_success_unchanged", "both_failure")}
        for metric in ("sr", "spl", "ndtw"):
            close(sum(v[metric + "_contribution_pp"] for v in by_category.values()), all_summary[metric + "_contribution_pp"], "decomposition total mismatch", 1e-12)
        strata = {}
        for field in config["strata"]:
            groups = defaultdict(list)
            for row in rows:
                value = row[field]
                bins = {"history_states": "history_states", "baseline_endpoint_age_steps": "endpoint_age_steps",
                        "baseline_stop_probability": "baseline_stop_probability"}
                key = bucket(value, config["bins"][bins[field]]) if field in bins else str(value)
                groups[key].append(row)
            strata[field] = {k: summarize(v, n) for k, v in sorted(groups.items())}
        findings[arm] = {"overall": all_summary, "by_outcome": by_category, "strata": strata}
    transitions = Counter((a["category"], b["category"]) for a, b in zip(rows_by_arm["C3"], rows_by_arm["M"]))
    training = {}
    for name in ("train_controls", "dev_controls", "train_pairs", "dev_pairs"):
        data = documents[name]
        training[name] = {k: data[k] for k in ("split", "usage", "pairs", "rollouts", "states", "natural_rollouts", "c2_rollouts") if k in data}
        training[name]["natural_states"] = sum(f.get("natural_states", 0) for f in data["files"]) or None
        training[name]["c2_states"] = sum(f.get("c2_states", 0) for f in data["files"]) or None
    return {"scope": config["scope"], "opportunities": overall, "arms": findings,
            "M_versus_C3_outcome_transitions": [{"C3": a, "M": b, "instructions": count} for (a,b),count in sorted(transitions.items())],
            "training_manifest_counts": training, "missing_evidence": config["missing_evidence"],
            "limits": ["Descriptive development-set evidence; no causal attribution or new-policy performance estimate.",
                       "Endpoint age and confidence bins were fixed before this analysis; no gate threshold was searched.",
                       "Training pair histories are forced and do not measure natural closed-loop navigation.",
                       "No new confidence interval or significance test is added for exploratory subgroups."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/e2_failure_diagnostic.json")
    parser.add_argument("--study", type=Path, default=ROOT / "configs/research_study.json")
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    registration = validate_registration(args.config, config, args.ledger, args.study)
    output = ROOT / config["output"]
    require(not output.exists(), "refusing to overwrite existing diagnostic result")
    for item in config["inputs"].values():
        require(sha(ROOT / item["path"]) == item["sha256"], "input hash differs: " + item["path"])
    documents = {k: read(ROOT / item["path"]) for k, item in config["inputs"].items()}
    result = analyze(config, documents)
    result.update({"schema": "e2_endpoint_failure_diagnostic_result_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "config_sha256": sha(args.config), "script_sha256": sha(__file__), "registration": registration,
                   "inputs": config["inputs"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "opportunities": result["opportunities"]}))


if __name__ == "__main__":
    main()
