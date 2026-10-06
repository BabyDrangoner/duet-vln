#!/usr/bin/env python3
"""D3: CPU-only coverage of real training-instruction endpoint pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import heapq
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vln_improve.protocol import file_sha256, object_sha256, resolve_config, select_partition


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"nonfinite JSON value: {value}")))


def validate_spec(spec, cfg):
    if spec.get("schema_version") != 1 or spec.get("diagnostic_id") != "D3-train-only-endpoint-pair-coverage":
        raise ValueError("unsupported D3 specification")
    if spec.get("source_annotation") != "R2R/annotations/R2R_train_enc.json":
        raise ValueError("D3 only reads original R2R_train_enc.json")
    if spec.get("allowed_splits") != ["train_fit", "train_dev"]:
        raise ValueError("D3 is training-only")
    if spec.get("partition") != {"dev_fraction": cfg["dev_fraction"], "partition_seed": cfg["partition_seed"]}:
        raise ValueError("diagnostic and baseline partitions differ")
    pair = spec.get("pairing", {})
    expected = {
        "same_scan": True, "same_start": True, "different_path_ids": True,
        "different_goals": True, "goal_separation_m_strictly_greater_than": 6.0,
        "exclude_other_goal_anywhere_on_reference_path": True,
        "heading_difference_unit": "degrees_modulo_360", "heading_exact_tolerance_deg": 0.000001,
        "heading_report_thresholds_deg": [15.0, 45.0],
        "primary_heading_rule": "exact_within_tolerance", "primary_requires_shared_prefix_edges": 0,
    }
    if pair != expected:
        raise ValueError("D3 pairing rules changed; revise the implementation and register a new specification")
    history = spec.get("history", {})
    if (history.get("construction") != "path_A_then_reverse_A_to_start_then_path_B_and_opposite_order"
            or history.get("observation_rule") != "first_visit_nodes_only; repeated visited nodes are transit, not new observations"
            or history.get("max_observed_states_each_order") != 15
            or history.get("max_physical_walk_states") is not None
            or cfg["model"].get("max_action_len") != 15):
        raise ValueError("unsupported history or observation budget")
    policy = spec.get("access_policy", {})
    if any(policy.get(key) != 0 for key in ("validation_accesses", "gpu_calls", "training_updates")):
        raise ValueError("D3 must remain CPU-only with no fitting or validation access")
    if any(policy.get(key) is not False for key in (
            "load_visual_or_text_features", "load_baseline_error_details")):
        raise ValueError("D3 may not read features or error details")
    if type(spec.get("selection_seed")) is not int:
        raise ValueError("selection_seed must be an integer")
    count = spec.get("sampling", {}).get("max_examples_per_report_stratum")
    if type(count) is not int or count < 0:
        raise ValueError("invalid example bound")
    for key in ("train_fit_min_path_disjoint_pairs", "train_fit_min_scans",
                "train_dev_min_path_disjoint_pairs", "train_dev_min_scans"):
        if type(spec.get("coverage_gate", {}).get(key)) is not int or spec["coverage_gate"][key] < 1:
            raise ValueError("invalid coverage gate")


def load_records(path):
    raw = read_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("training annotation must be a nonempty list")
    seen, result = set(), []
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("invalid annotation row")
        scan, route, heading = row.get("scan"), row.get("path"), row.get("heading")
        pid = row.get("path_id")
        instructions = row.get("instructions")
        if (not isinstance(scan, str) or not scan or Path(scan).name != scan
                or not isinstance(pid, (str, int)) or isinstance(pid, bool)
                or not isinstance(route, list) or len(route) < 2
                or any(not isinstance(v, str) or not v for v in route)
                or isinstance(heading, bool) or not isinstance(heading, (int, float))
                or not math.isfinite(heading) or not isinstance(instructions, list) or not instructions
                or any(not isinstance(x, str) or not x.strip() for x in instructions)):
            raise ValueError("annotation lacks original path, heading, or real instructions")
        key = str(pid)
        if key in seen:
            raise ValueError("duplicate original path_id")
        seen.add(key)
        result.append({"scan": scan, "path_id": key, "path": route,
                       "heading": float(heading), "num_instructions": len(instructions)})
    return result


def load_graph(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty connectivity")
    ids = [row.get("image_id") for row in rows]
    if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid or duplicate connectivity node")
    positions, graph = {}, {}
    for row in rows:
        if not isinstance(row.get("included"), bool):
            raise ValueError("invalid included flag")
        links = row.get("unobstructed")
        if not isinstance(links, list) or len(links) != len(rows) or any(type(x) is not bool for x in links):
            raise ValueError("invalid connectivity matrix")
        if row["included"]:
            pose = row.get("pose")
            if not isinstance(pose, list) or len(pose) != 16 or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pose):
                raise ValueError("invalid pose")
            positions[row["image_id"]] = tuple(float(pose[i]) for i in (3, 7, 11))
            graph[row["image_id"]] = {}
    for i, row in enumerate(rows):
        if not row["included"]:
            continue
        for j, connected in enumerate(row["unobstructed"]):
            if i == j or not connected or not rows[j]["included"]:
                continue
            if not rows[j]["unobstructed"][i]:
                raise ValueError("connectivity is not bidirectional")
            a, b = ids[i], ids[j]
            distance = math.dist(positions[a], positions[b])
            if distance <= 0:
                raise ValueError("zero-length connectivity edge")
            graph[a][b] = distance
    return graph


def shortest_distances(graph, source, allowed=None):
    dist, parent, queue = {source: 0.0}, {}, [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != dist[node]:
            continue
        for target, weight in sorted(graph[node].items()):
            if allowed is not None and target not in allowed:
                continue
            candidate = distance + weight
            if candidate < dist.get(target, math.inf):
                dist[target], parent[target] = candidate, node
                heapq.heappush(queue, (candidate, target))
    return dist, parent


def walk_length(graph, walk):
    if any(v not in graph for v in walk):
        raise ValueError("reference path uses an excluded/missing viewpoint")
    length = 0.0
    for a, b in zip(walk, walk[1:]):
        if b not in graph[a] or a not in graph[b]:
            raise ValueError("reference path contains a non-adjacent edge")
        length += graph[a][b]
    return length


def heading_difference_deg(a, b):
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def make_history(graph, first, second):
    if first[0] != second[0]:
        raise ValueError("shared history requires the same start")
    walk = first + list(reversed(first))[1:] + second[1:]
    full_length = walk_length(graph, walk)
    observed = list(dict.fromkeys(walk))
    visited, proxy_walk = {observed[0]}, [observed[0]]
    for target in observed[1:]:
        # Every intermediate is already observed; this uses no hidden future edges.
        distances, parents = shortest_distances(graph, proxy_walk[-1], allowed=visited | {target})
        if target not in distances:
            raise ValueError("new observation is not a discovered-map reachable candidate")
        route, node = [], target
        while node != proxy_walk[-1]:
            route.append(node)
            node = parents[node]
        proxy_walk.extend(reversed(route))
        visited.add(target)
    return {
        "reference_walk": walk, "reference_walk_states": len(walk),
        "reference_walk_length_m": full_length,
        "observed_vpids": observed, "observed_states": len(observed),
        "known_map_shortest_proxy_walk": proxy_walk,
        "known_map_shortest_proxy_length_m": walk_length(graph, proxy_walk),
        "proxy_matches_reference_walk": proxy_walk == walk,
        "goal_first_observation_indices": {first[-1]: observed.index(first[-1]), second[-1]: observed.index(second[-1])},
    }


def independent_pairs(rows):
    used, selected = set(), []
    for row in sorted(rows, key=lambda r: r["selection_hash"]):
        keys = {(row["scan"], pid) for pid in row["path_ids"]}
        if used.isdisjoint(keys):
            selected.append(row)
            used.update(keys)
    return selected


def pair_summary(rows):
    by_scan = Counter(row["scan"] for row in rows)
    return {"pairs": len(rows), "independent_path_ids": len({(r["scan"], p) for r in rows for p in r["path_ids"]}),
            "scans": len(by_scan), "pairs_per_scan": dict(sorted(by_scan.items()))}


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return {"count": len(values), "min": values[0], "median": median, "max": values[-1]}


def analyze_split(records, graphs, spec):
    grouped = defaultdict(list)
    for row in records:
        walk_length(graphs[row["scan"]], row["path"])
        grouped[(row["scan"], row["path"][0])].append(row)
    stages = defaultdict(list)
    eligible = []
    distance_cache = {}
    seed = spec["selection_seed"]
    for (scan, start), group in sorted(grouped.items()):
        graph = graphs[scan]
        for a, b in combinations(sorted(group, key=lambda r: r["path_id"]), 2):
            info = {"scan": scan, "start": start, "path_ids": [a["path_id"], b["path_id"]]}
            stages["same_start_different_paths"].append(info)
            if a["path"][-1] == b["path"][-1]:
                continue
            stages["different_goals"].append(info)
            cache_key = (scan, a["path"][-1])
            if cache_key not in distance_cache:
                distance_cache[cache_key] = shortest_distances(graph, cache_key[1])[0]
            separation = distance_cache[cache_key].get(b["path"][-1])
            if separation is None:
                raise ValueError("same-start reference goals are disconnected")
            if separation <= 6.0:
                continue
            stages["goal_separation_gt_6m"].append(info)
            if b["path"][-1] in a["path"] or a["path"][-1] in b["path"]:
                continue
            stages["no_cross_goal_on_reference_paths"].append(info)
            histories = [make_history(graph, a["path"], b["path"]), make_history(graph, b["path"], a["path"])]
            if any(h["observed_states"] > 15 for h in histories):
                continue
            stages["both_histories_observed_states_le_15"].append(info)
            prefix_nodes = 0
            for va, vb in zip(a["path"], b["path"]):
                if va != vb:
                    break
                prefix_nodes += 1
            indices = [min(range(row["num_instructions"]), key=lambda i: object_sha256(
                [seed, scan, row["path_id"], i])) for row in (a, b)]
            eligible.append(dict(info, goal_vpids=[a["path"][-1], b["path"][-1]],
                goal_separation_m=separation, heading_rad=[a["heading"], b["heading"]],
                heading_difference_deg=heading_difference_deg(a["heading"], b["heading"]),
                shared_prefix_edges=prefix_nodes - 1,
                instr_ids=[f"{row['path_id']}_{index}" for row, index in zip((a, b), indices)],
                selection_hash=object_sha256([seed, scan, a["path_id"], b["path_id"]]),
                histories={"A_then_B": histories[0], "B_then_A": histories[1]}))
    strata = {
        "exact_heading_primary": [r for r in eligible if r["heading_difference_deg"] <= 0.000001],
        "heading_le_15deg_descriptive": [r for r in eligible if r["heading_difference_deg"] <= 15.0],
        "heading_le_45deg_descriptive": [r for r in eligible if r["heading_difference_deg"] <= 45.0],
        "heading_gt_45deg_descriptive": [r for r in eligible if r["heading_difference_deg"] > 45.0],
    }
    summaries = {}
    for name, rows in strata.items():
        selected = independent_pairs(rows)
        summaries[name] = dict(pair_summary(rows), path_disjoint=pair_summary(selected),
            shared_prefix_edges_histogram=dict(sorted(Counter(str(r["shared_prefix_edges"]) for r in rows).items())),
            heading_difference_deg=distribution([r["heading_difference_deg"] for r in rows]),
            observed_states=distribution([h["observed_states"] for r in rows for h in r["histories"].values()]),
            reference_walk_states=distribution([h["reference_walk_states"] for r in rows for h in r["histories"].values()]),
            reference_walk_length_m=distribution([h["reference_walk_length_m"] for r in rows for h in r["histories"].values()]),
            proxy_differs_from_reference_orders=sum(not h["proxy_matches_reference_walk"] for r in rows for h in r["histories"].values()),
            examples=selected[:spec["sampling"]["max_examples_per_report_stratum"]])
    stage_names = ("same_start_different_paths", "different_goals", "goal_separation_gt_6m",
                   "no_cross_goal_on_reference_paths", "both_histories_observed_states_le_15")
    primary = independent_pairs(strata["exact_heading_primary"])
    return {"source_paths": len(records), "source_instructions": sum(r["num_instructions"] for r in records),
            "source_scans": len({r["scan"] for r in records}),
            "stages": {key: pair_summary(stages[key]) for key in stage_names},
            "heading_strata_are_cumulative": True, "heading_strata": summaries,
            "primary_path_disjoint_manifest": primary,
            "primary_manifest_sha256": object_sha256(primary)}


def run(config_path, spec_path):
    cfg = resolve_config(config_path, ROOT)
    spec = read_json(spec_path)
    validate_spec(spec, cfg)
    dataset = Path(cfg["dataset_root"])
    annotation = dataset / spec["source_annotation"]
    records = load_records(annotation)
    scans = sorted({r["scan"] for r in records})
    graph_paths = {scan: dataset / "R2R/connectivity" / f"{scan}_connectivity.json" for scan in scans}
    graphs = {scan: load_graph(path) for scan, path in graph_paths.items()}
    splits = {split: analyze_split(select_partition(records, split, cfg["dev_fraction"], cfg["partition_seed"]), graphs, spec)
              for split in ("train_fit", "train_dev")}
    gates = {}
    for split, report in splits.items():
        primary = report["heading_strata"]["exact_heading_primary"]["path_disjoint"]
        gates[split] = {"pairs": primary["pairs"], "scans": primary["scans"],
            "required_pairs": spec["coverage_gate"][f"{split}_min_path_disjoint_pairs"],
            "required_scans": spec["coverage_gate"][f"{split}_min_scans"]}
        gates[split]["pass"] = (primary["pairs"] >= gates[split]["required_pairs"]
                               and primary["scans"] >= gates[split]["required_scans"])
    return {"schema": "duet_endpoint_pair_coverage_v1", "diagnostic_id": spec["diagnostic_id"],
        "usage": "train_only_geometry_diagnostic", "gpu_calls": 0, "training_updates": 0, "validation_accesses": 0,
        "identity": {"config_sha256": file_sha256(config_path), "diagnostic_config_sha256": file_sha256(spec_path),
            "annotation_sha256": file_sha256(annotation),
            "connectivity_sha256": object_sha256({s: file_sha256(p) for s, p in graph_paths.items()}),
            "implementation_sha256": object_sha256({"scripts/analyze_endpoint_pairs.py": file_sha256(__file__),
                "src/vln_improve/protocol.py": file_sha256(ROOT / "src/vln_improve/protocol.py")})},
        "specification": spec, "splits": splits, "coverage_gate": gates,
        "coverage_pass": all(row["pass"] for row in gates.values()),
        "missing": {"simulator_replay": "Not run: actual DUET candidate-angle and shortest-route tie parity remain unverified.",
                    "instruction_semantics": "Geometric endpoint labels do not prove full instruction-following or unambiguous language.",
                    "visual_confusability": "No visual features or model scores loaded; no hard-negative claim established.",
                    "navigation_gain": "No training, policy execution, or validation analysis performed."}}


def write_new_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard link is atomic and refuses an existing destination.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--diagnostic-config", type=Path, default=ROOT / "configs/endpoint_pair_diagnostic.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite an existing report")
    result = run(args.config, args.diagnostic_config)
    write_new_json(args.output, result)
    print(json.dumps({"output": str(args.output), "coverage_pass": result["coverage_pass"],
                      "coverage_gate": result["coverage_gate"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
