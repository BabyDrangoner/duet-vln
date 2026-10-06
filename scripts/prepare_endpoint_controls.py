#!/usr/bin/env python3
"""CPU-only C2 overshoot feasibility and immutable common endpoint-training pool."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_endpoint_pairs import (
    load_graph, load_records, pair_summary, read_json, shortest_distances,
    walk_length, write_new_json,
)
from vln_improve.protocol import file_sha256, object_sha256, resolve_config, select_partition

CODE_FILES = ("scripts/prepare_endpoint_controls.py", "scripts/analyze_endpoint_pairs.py", "src/vln_improve/protocol.py")
CANDIDATE_KEYS = ("reachable_nodes", "distance_annulus", "annulus_on_original_path",
                  "q_not_on_original_path", "rejected_observation_budget", "rejected_discovered_reachability", "feasible_q")


def code_identity():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def validate_spec(spec, config):
    expected = {"schema_version": 1, "diagnostic_id": "D3c-train-only-endpoint-control-feasibility",
        "source_annotation": "R2R/annotations/R2R_train_enc.json", "allowed_splits": ["train_fit", "train_dev"],
        "generation_seed": 0, "max_observed_states": 15,
        "q_distance_m": {"minimum_inclusive": 3.0, "maximum_inclusive": 6.0},
        "common_pool": {"source": "D3 primary_path_disjoint_manifest only; no rematching",
            "order": "ascending original selection_hash", "require_both_original_paths_feasible": True,
            "train_fit_pairs": 512, "train_dev_pairs": 128,
            "on_shortfall": "coverage_pass=false; preserve complete audit and available selection; no automatic downsizing or collection"},
        "access_policy": {"gpu_calls": 0, "training_updates": 0, "validation_accesses": 0,
                          "load_features_or_models": False, "load_policy_error_results": False}}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("C2 feasibility rules changed; no automatic relaxation or downsizing")
    if (spec.get("partition") != {"dev_fraction": config["dev_fraction"], "partition_seed": config["partition_seed"]}
            or config["model"].get("max_action_len") != 15):
        raise ValueError("C2 and D3 partition/observation budget differ")
    sha = spec.get("coverage_report_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid D3 report SHA")


def reconstruct_path(parents, source, target):
    path = [target]
    while path[-1] != source:
        if path[-1] not in parents or len(path) > len(parents) + 1:
            raise ValueError("missing/cyclic shortest-path parent chain")
        path.append(parents[path[-1]])
    return list(reversed(path))


def history_from_walk(graph, walk, goal):
    """Generalize the unchanged D3 first-visit/discovered-map check to one walk."""
    length = walk_length(graph, walk)
    observed = list(dict.fromkeys(walk))
    if goal not in observed:
        raise ValueError("history never observes its true goal")
    known, proxy = {observed[0]}, [observed[0]]
    for target in observed[1:]:
        distances, parents = shortest_distances(graph, proxy[-1], allowed=known | {target})
        if target not in distances:
            raise ValueError("not discovered-map reachable")
        route = reconstruct_path(parents, proxy[-1], target)
        if any(v not in known for v in route[1:-1]):
            raise ValueError("unobserved shortest-path intermediate")
        proxy.extend(route[1:])
        known.add(target)
    return {"reference_walk": list(walk), "reference_walk_states": len(walk), "reference_walk_length_m": length,
            "observed_vpids": observed, "observed_states": len(observed),
            "known_map_shortest_proxy_walk": proxy, "known_map_shortest_proxy_length_m": walk_length(graph, proxy),
            "proxy_matches_reference_walk": proxy == walk, "goal_first_observation_index": observed.index(goal)}


def prepare_control(record, graph, *, seed=0):
    """Only this path and its own goal are inputs; the paired goal is unavailable."""
    if seed != 0:
        raise ValueError("C2 generation seed is fixed to zero")
    route, goal = record["path"], record["path"][-1]
    walk_length(graph, route)  # Bad source data is a hard error, not a filtered sample.
    observed = list(dict.fromkeys(route))
    base = {"scan": record["scan"], "path_id": record["path_id"], "start": route[0], "goal": goal,
            "heading": record["heading"], "original_path": list(route), "eligible": False,
            "positive_history": None, "overshoot_history": None, "selected_q": None,
            "candidate_analysis_performed": False, "candidate_counts": {k: 0 for k in CANDIDATE_KEYS},
            "candidate_audit": []}
    if observed[-1] != goal:
        return dict(base, rejection="positive_first_visit_does_not_end_at_goal")
    if len(observed) > 15:
        return dict(base, rejection="positive_observation_budget")
    positive = history_from_walk(graph, route, goal)
    base["positive_history"] = positive
    distance, parents = shortest_distances(graph, goal)
    counts = base["candidate_counts"]
    counts["reachable_nodes"] = len(distance)
    annulus = [q for q, d in distance.items() if 3.0 <= d <= 6.0]
    counts["distance_annulus"] = len(annulus)
    counts["annulus_on_original_path"] = sum(q in route for q in annulus)
    candidates = [q for q in annulus if q not in route]
    counts["q_not_on_original_path"] = len(candidates)
    ranked = sorted(candidates, key=lambda q: object_sha256([seed, record["scan"], record["path_id"], q]))
    base["candidate_analysis_performed"] = True
    chosen = None
    for q in ranked:
        extension = reconstruct_path(parents, goal, q)
        full_walk = route + extension[1:]
        states = len(set(full_walk))
        audit = {"q": q, "goal_distance_m": distance[q], "observed_states": states,
                 "q_selection_hash": object_sha256([seed, record["scan"], record["path_id"], q])}
        if states > 15:
            counts["rejected_observation_budget"] += 1
            base["candidate_audit"].append(dict(audit, feasible=False, rejection="observation_budget"))
            continue
        try:
            history = history_from_walk(graph, full_walk, goal)
        except ValueError as error:
            # Connectivity itself was checked above and on the reconstructed
            # route, so only the explicit discovered-map gate can filter here.
            if str(error) not in {"not discovered-map reachable", "unobserved shortest-path intermediate"}:
                raise
            counts["rejected_discovered_reachability"] += 1
            base["candidate_audit"].append(dict(audit, feasible=False, rejection="discovered_reachability"))
            continue
        if history["observed_vpids"][-1] != q:
            raise ValueError("new overshoot endpoint is not the final first observation")
        counts["feasible_q"] += 1
        base["candidate_audit"].append(dict(audit, feasible=True, rejection=None))
        if chosen is None:
            chosen = (q, history, audit["q_selection_hash"], distance[q])
    if chosen is None:
        return dict(base, rejection="no_feasible_q")
    q, history, sha, goal_distance = chosen
    return dict(base, eligible=True, rejection=None, overshoot_history=history, selected_q=q,
                selected_q_hash=sha, selected_q_goal_distance_m=goal_distance,
                endpoint_labels={"positive_history": True, "overshoot_history": False})


def verify_primary(primary, records, *, expected_sha, selection_seed):
    if object_sha256(primary) != expected_sha:
        raise ValueError("D3 primary manifest checksum differs")
    seen_paths, seen_pairs = set(), set()
    for pair in primary:
        if (pair["selection_hash"] in seen_pairs or len(pair["path_ids"]) != 2
                or len(set(pair["path_ids"])) != 2 or pair["heading_difference_deg"] > 0.000001
                or pair["goal_separation_m"] <= 6):
            raise ValueError("invalid D3 primary pair")
        expected = object_sha256([selection_seed, pair["scan"], *pair["path_ids"]])
        if pair["selection_hash"] != expected:
            raise ValueError("D3 pair selection hash differs")
        seen_pairs.add(pair["selection_hash"])
        for index, pid in enumerate(pair["path_ids"]):
            if pid in seen_paths or pid not in records:
                raise ValueError("D3 primary paths overlap or are outside this train partition")
            seen_paths.add(pid)
            row = records[pid]
            if (row["scan"] != pair["scan"] or row["path"][0] != pair["start"]
                    or row["path"][-1] != pair["goal_vpids"][index]
                    or row["heading"] != pair["heading_rad"][index]):
                raise ValueError("D3 path association differs from original annotations")
            instr = pair["instr_ids"][index]
            prefix = pid + "_"
            suffix = instr.removeprefix(prefix)
            if not instr.startswith(prefix) or not suffix.isdigit() or int(suffix) >= row["num_instructions"]:
                raise ValueError("D3 instruction does not belong to the original path")


def analyze_split(primary, records, graphs, *, requested_pairs, selection_seed, primary_sha):
    by_id = {r["path_id"]: r for r in records}
    verify_primary(primary, by_id, expected_sha=primary_sha, selection_seed=selection_seed)
    controls, audit, eligible = {}, [], []
    for pair in sorted(primary, key=lambda row: row["selection_hash"]):
        ids = []
        for index, pid in enumerate(pair["path_ids"]):
            control_id = object_sha256([pair["scan"], pid, pair["instr_ids"][index]])
            result = prepare_control(by_id[pid], graphs[pair["scan"]], seed=0)
            result["instr_id"] = pair["instr_ids"][index]
            result["control_id"] = control_id
            controls[control_id] = result
            ids.append(control_id)
        feasible = [controls[key]["eligible"] for key in ids]
        audit.append({"selection_hash": pair["selection_hash"], "scan": pair["scan"], "path_ids": pair["path_ids"],
                      "control_ids": ids, "path_feasible": feasible, "both_feasible": all(feasible),
                      "path_rejections": [controls[key]["rejection"] for key in ids]})
        if all(feasible):
            eligible.append({"pair": copy.deepcopy(pair), "control_ids": ids,
                             "natural_instr_ids": list(pair["instr_ids"])})
    selected = eligible[:requested_pairs]
    statuses = Counter("eligible" if item["eligible"] else item["rejection"] for item in controls.values())
    candidates = {key: sum(item["candidate_counts"][key] for item in controls.values()) for key in CANDIDATE_KEYS}
    eligible_pairs = [item["pair"] for item in eligible]
    selected_pairs = [item["pair"] for item in selected]
    manifest = {"requested_pairs": requested_pairs, "selected_pairs": selected, "controls": controls}
    return {"source_primary": pair_summary(primary), "eligible_common_pool": pair_summary(eligible_pairs),
            "selected_common_pool": pair_summary(selected_pairs), "requested_pairs": requested_pairs,
            "coverage_pass": len(selected) == requested_pairs,
            "shortfall_pairs": max(0, requested_pairs - len(selected)),
            "path_status_counts": dict(sorted(statuses.items())), "candidate_filter_counts": candidates,
            "paths_without_candidate_analysis": sum(not item["candidate_analysis_performed"] for item in controls.values()),
            "pair_filter_counts": {"both_feasible": sum(a["both_feasible"] for a in audit),
                "only_A_feasible": sum(a["path_feasible"] == [True, False] for a in audit),
                "only_B_feasible": sum(a["path_feasible"] == [False, True] for a in audit),
                "neither_feasible": sum(a["path_feasible"] == [False, False] for a in audit)},
            "pair_audit": audit, "manifest": manifest, "manifest_sha256": object_sha256(manifest)}


def run(config_path, diagnostic_path, coverage_path):
    cfg = resolve_config(config_path, ROOT)
    spec = read_json(diagnostic_path)
    validate_spec(spec, cfg)
    if file_sha256(coverage_path) != spec["coverage_report_sha256"]:
        raise ValueError("D3 report SHA differs from the predeclared source")
    coverage = read_json(coverage_path)
    if (coverage.get("schema") != "duet_endpoint_pair_coverage_v1" or coverage.get("coverage_pass") is not True
            or coverage.get("usage") != "train_only_geometry_diagnostic"
            or coverage["identity"]["config_sha256"] != file_sha256(config_path)):
        raise ValueError("D3 source schema, gate, or runtime configuration differs")
    dataset = Path(cfg["dataset_root"])
    annotation = dataset / spec["source_annotation"]
    if file_sha256(annotation) != coverage["identity"]["annotation_sha256"]:
        raise ValueError("train annotation changed since D3")
    records = load_records(annotation)
    scans = sorted({r["scan"] for r in records})
    graph_paths = {scan: dataset / "R2R/connectivity" / f"{scan}_connectivity.json" for scan in scans}
    graph_hashes = {scan: file_sha256(path) for scan, path in graph_paths.items()}
    if object_sha256(graph_hashes) != coverage["identity"]["connectivity_sha256"]:
        raise ValueError("training connectivity changed since D3")
    graphs = {scan: load_graph(path) for scan, path in graph_paths.items()}
    splits = {}
    for split in ("train_fit", "train_dev"):
        source = coverage["splits"][split]
        rows = select_partition(records, split, cfg["dev_fraction"], cfg["partition_seed"])
        splits[split] = analyze_split(source["primary_path_disjoint_manifest"], rows, graphs,
            requested_pairs=spec["common_pool"][f"{split}_pairs"],
            selection_seed=coverage["specification"]["selection_seed"], primary_sha=source["primary_manifest_sha256"])
    return {"schema": "duet_endpoint_controls_geometry_v1", "diagnostic_id": spec["diagnostic_id"],
            "usage": "train_only_geometry_diagnostic", "training_updates": 0, "gpu_calls": 0, "validation_accesses": 0,
            "identity": {"config_sha256": file_sha256(config_path), "diagnostic_config_sha256": file_sha256(diagnostic_path),
                "coverage_report_sha256": file_sha256(coverage_path), "annotation_sha256": file_sha256(annotation),
                "connectivity_sha256": object_sha256(graph_hashes), "connectivity_files": graph_hashes,
                "implementation": code_identity(), "implementation_sha256": object_sha256(code_identity())},
            "specification": spec, "splits": splits, "coverage_pass": all(x["coverage_pass"] for x in splits.values()),
            "missing": {"simulator_parity": "Not run: actual candidate view/angle and DUET shortest-route ties require later collection checks.",
                "method_evidence": "Geometry feasibility is not a training or navigation result.",
                "full_collection_authorization": "This script does not launch collection or change its budget."}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--diagnostic-config", type=Path, default=ROOT / "configs/endpoint_controls_diagnostic.json")
    parser.add_argument("--coverage-report", type=Path, default=ROOT / "outputs/study-20261003/d3-endpoint-pair-coverage.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-code-sha256", action="store_true")
    args = parser.parse_args(argv)
    if args.print_code_sha256:
        print(object_sha256(code_identity()))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite an existing control manifest")
    result = run(args.config, args.diagnostic_config, args.coverage_report)
    write_new_json(args.output, result)
    import json
    print(json.dumps({"output": str(args.output), "coverage_pass": result["coverage_pass"],
        "splits": {key: {k: value[k] for k in ("requested_pairs", "eligible_common_pool", "selected_common_pool",
                                              "shortfall_pairs", "path_status_counts", "candidate_filter_counts")}
                   for key, value in result["splits"].items()}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
