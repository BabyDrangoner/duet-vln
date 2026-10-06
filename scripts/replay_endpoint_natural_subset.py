#!/usr/bin/env python3
"""CPU endpoint replay on all 256 sealed natural development instructions.

No simulator, backbone forward, fitting, sample selection, or official validation.
The primary scores use one cached state per CPU forward. This is not a claim of
bitwise equality with a future warm-candidate-cache, full-split CUDA evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time

import networkx as nx
import numpy as np
import torch

ARMS = ("C1", "C2", "C3", "M")
SCHEMA = "duet_endpoint_natural_subset_replay_v1"
EXPONENTIAL_METRICS = ("nDTW", "SDTW", "CLS")
EXACT_METRICS = ("nav_error", "oracle_error", "action_steps", "trajectory_steps", "trajectory_lengths",
                 "success", "spl", "oracle_success", "DTW")
MAX_EXPONENTIAL_ULPS = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def progress(stage, **values):
    print(json.dumps({"event": stage, **values}, sort_keys=True), flush=True)


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"symlinked input: {path}")
    require(path.is_dir() if directory else path.is_file(), f"missing ordinary input: {path}")
    return path


def read_json(path):
    def unique(items):
        out = {}
        for key, value in items:
            require(key not in out, f"duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(ordinary(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def checked_content(value):
    require(value.get("content_sha256") == object_sha({k: v for k, v in value.items()
                                                       if k != "content_sha256"}), "content SHA mismatch")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ordinary(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_code(source, mapping):
    require(isinstance(mapping, dict) and mapping, "missing source file identity")
    for name, expected in mapping.items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in name,
                "unsafe source identity path")
        require(sha(ordinary(source / rel)) == expected, f"frozen implementation changed: {name}")


def load_connectivity(directory, scans, expected_sha):
    """Verify the ten files against the hash-bound complete connectivity inventory."""
    manifest = read_json(directory / "manifest.json")
    require(manifest.get("schema") == "duet_connectivity_subset_v1"
            and manifest.get("source_connectivity_sha256") == expected_sha
            and manifest.get("selected_scans") == sorted(scans), "connectivity manifest identity differs")
    inventory = manifest.get("all_files")
    require(isinstance(inventory, dict) and object_sha(inventory) == expected_sha,
            "complete connectivity inventory SHA differs")
    graphs, distances, selected_hashes = {}, {}, {}
    for scan in sorted(scans):
        name = scan + "_connectivity.json"
        path = ordinary(directory / name)
        require(sha(path) == inventory.get(name), f"connectivity file SHA differs: {scan}")
        selected_hashes[name] = inventory[name]
        data, graph = read_json(path), nx.Graph()
        # The same insertion order and weight expression as upstream utils/data.py.
        for i, item in enumerate(data):
            if item["included"]:
                for j, connected in enumerate(item["unobstructed"]):
                    if connected and data[j]["included"]:
                        require(data[j]["unobstructed"][i], "asymmetric connectivity")
                        other = data[j]
                        weight = ((item["pose"][3] - other["pose"][3]) ** 2
                                  + (item["pose"][7] - other["pose"][7]) ** 2
                                  + (item["pose"][11] - other["pose"][11]) ** 2) ** .5
                        graph.add_edge(item["image_id"], other["image_id"], weight=weight)
                        graph.nodes[item["image_id"]]["execution_position"] = [
                            float(np.float32(item["pose"][index])) for index in (3, 7, 11)]
        require(graph.number_of_nodes() > 0, "empty connectivity scene")
        graphs[scan] = graph
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(graph))
    return graphs, distances, {"manifest_sha256": sha(directory / "manifest.json"),
                               "source_connectivity_sha256": expected_sha, "files": selected_hashes}


def reconstruct_natural(run, graph_module):
    """Rebuild only nodes observed somewhere in this fixed historical rollout.

    Never-observed frontier nodes are never Floyd intermediates in upstream.
    Removing them cannot change paths between retained nodes. Retained edges
    are still revealed in original chronological/candidate order, using saved
    execution coordinates, not the official graph or FP32 distance matrix.
    """
    states = run["states"]
    require(bool(states), "empty natural history")
    observed = [state["viewpoint"] for state in states]
    require(len(set(observed)) == len(observed), "repeated observed endpoint")
    positions = {state["viewpoint"]: state["position"] for state in states}
    require(all(len(p) == 3 and all(math.isfinite(float(x)) for x in p) for p in positions.values()),
            "invalid execution coordinates")
    graph = graph_module.FloydGraph()
    seen, revealed, retained_order, checked_pairs = [], set(), {}, 0
    for step, state in enumerate(states):
        current = state["viewpoint"]
        require(state["step"] == step, "natural state order changed")
        candidates = state["vp_cand_vpids"]
        require(candidates and candidates[0] is None and current not in candidates
                and len(set(candidates)) == len(candidates), "invalid local candidate order")
        retained_order.setdefault(current, None)
        for other in candidates[1:]:
            if other in positions:
                retained_order.setdefault(other, None)
                graph.add_edge(current, other, graph_module.calc_position_distance(positions[current], positions[other]))
                revealed.add(frozenset((current, other)))
        graph.update(current)
        seen.append(current)
        nodes = state["gmap_vpids"]
        require(nodes and nodes[0] is None and len(set(nodes)) == len(nodes), "invalid global node IDs")
        expected_visited = [False] + [node in seen for node in nodes[1:]]
        require(state["masks"]["gmap_visited_masks"] == expected_visited
                and all(state["masks"]["gmap_masks"]), "saved visited/full graph masks differ")
        expected_ids = [None] + [node for node in retained_order if node in seen] + [node for node in retained_order if node not in seen]
        require([node for node in nodes if node is None or node in positions] == expected_ids,
                "retained node insertion order differs")
        require(state["gmap_step_ids"] == [0 if node not in seen else seen.index(node) + 1 for node in nodes],
                "saved graph step IDs differ")
        retained = [node for node in nodes[1:] if node in positions]
        matrix = np.asarray(state["gmap_pair_dists"], dtype=np.float32)
        require(matrix.shape == (len(nodes), len(nodes)) and np.isfinite(matrix).all(), "invalid saved distances")
        for a in retained:
            for b in retained:
                expected = np.float32(graph.distance(a, b))
                require(expected == matrix[nodes.index(a), nodes.index(b)],
                        f"exact FP32 execution distance differs at step {step}: {a}/{b}")
                checked_pairs += 1
        action = state["executed_action"]
        flags = state["termination"]
        require(flags["step_limit"] == (step == 14)
                and flags["no_candidates"] == all(expected_visited[1:])
                and flags["argmax_stop"] == (state["baseline_argmax_index"] == 0)
                and state["baseline_argmax_vpid"] == nodes[state["baseline_argmax_index"]]
                and (action is None) == any(flags.values()), "saved online termination differs")
        prefix = state["trajectory_prefix"]
        require(prefix and prefix[-1][-1] == current and prefix[0] == [observed[0]], "invalid online prefix")
        if step + 1 < len(states):
            require(action == states[step + 1]["viewpoint"] and action not in seen,
                    "natural action does not reach its next observation")
            segment = graph.path(current, action)
            require(prefix + [segment] == states[step + 1]["trajectory_prefix"],
                    "reconstructed executed action path differs from saved online prefix")
            require(all(frozenset((a, b)) in revealed for a, b in zip([current] + segment, segment)),
                    "executed path used unrevealed edge")
        else:
            require(action is None, "natural episode lacks terminal decision")
    probabilities = run["natural_stop_probability"]
    require(probabilities.shape == (len(states),) and torch.isfinite(probabilities).all(), "invalid STOP probabilities")
    selected = int(probabilities.argmax())  # unique nodes; first insertion wins equal scores
    baseline = endpoint_trajectory(states[-1]["trajectory_prefix"], observed[selected], graph)
    require(baseline == run["trajectory"], "baseline probability fallback did not reproduce saved full trajectory")
    require(run["trajectory"] == states[-1]["trajectory_prefix"] + run["fallback_segment"],
            "saved baseline fallback segmentation differs")
    return graph, {"states": len(states), "checked_pair_distances": checked_pairs,
                   "checked_online_moves": len(states) - 1, "baseline_selected_state": selected,
                   "all_prefix_paths_exact": True, "all_retained_pair_distances_exact_fp32": True,
                   "full_baseline_fallback_path_exact": True}


def endpoint_trajectory(prefix, target, graph):
    result = copy.deepcopy(prefix)
    current = prefix[-1][-1]
    if current != target:
        result.append(graph.path(current, target))
    return result


def navigation_metrics(trajectory, gt_path, graph, shortest, eval_utils):
    """Match upstream R2RBatch._eval_item, including return segments and GT length."""
    path = sum(trajectory, [])
    require(path and gt_path and path[0] == gt_path[0], "trajectory/GT start differs")
    require(all(a != b and graph.has_edge(a, b) for a, b in zip(path[:-1], path[1:])),
            "replayed path contains a repeated/nonexistent edge")
    nav_error = shortest[path[-1]][gt_path[-1]]
    oracle_error = min(shortest[node][gt_path[-1]] for node in path)
    length = np.sum([shortest[a][b] for a, b in zip(path[:-1], path[1:])])
    gt_length = np.sum([shortest[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
    success = float(nav_error < 3.)
    scores = {"nav_error": nav_error, "oracle_error": oracle_error,
              "action_steps": len(trajectory) - 1, "trajectory_steps": len(path) - 1,
              "trajectory_lengths": length, "success": success,
              "spl": success * gt_length / max(length, gt_length, .01),
              "oracle_success": float(oracle_error < 3.)}
    scores.update(eval_utils.cal_dtw(shortest, path, gt_path, success, 3.))
    scores["CLS"] = eval_utils.cal_cls(shortest, path, gt_path, 3.)
    require(all(math.isfinite(float(v)) for v in scores.values()), "nonfinite navigation metric")
    return {k: float(v) for k, v in scores.items()}


def validate_baseline_metrics(expected, actual):
    """Bound only exponential-function rounding across CPU implementations.

    Positive IEEE-754 binary64 bit patterns have numerical order, so their
    integer distance counts representable floats exactly (including subnormals).
    No relative/absolute tolerance applies to SR, SPL, distances, DTW or steps.
    """
    require(set(actual) == set(EXACT_METRICS + EXPONENTIAL_METRICS), "baseline metric key set differs")
    for name in EXACT_METRICS + EXPONENTIAL_METRICS:
        require(type(expected.get(name)) in (int, float) and type(actual[name]) in (int, float)
                and math.isfinite(float(expected[name])) and math.isfinite(float(actual[name])),
                f"invalid baseline metric: {name}")
    for name in EXACT_METRICS:
        require(expected[name] == actual[name], f"baseline exact navigation metric differs: {name}")
    differences = {}
    for name in EXPONENTIAL_METRICS:
        before, after = float(expected[name]), float(actual[name])
        require(before >= 0. and after >= 0., f"negative exponential navigation metric: {name}")
        # Normalize signed zero; +0 and -0 denote exactly the same metric.
        before_bits = struct.unpack(">Q", struct.pack(">d", before if before else 0.))[0]
        after_bits = struct.unpack(">Q", struct.pack(">d", after if after else 0.))[0]
        ulps = abs(after_bits - before_bits)
        require(ulps <= MAX_EXPONENTIAL_ULPS, f"baseline exponential metric exceeds 8 ULP: {name} ({ulps})")
        differences[name] = {"expected": before, "actual": after, "difference": after - before,
                             "absolute_difference": abs(after - before), "ulp_distance": ulps,
                             "exact": before == after}
    return {"sr_spl_exact": True, "non_exponential_metrics_exact": True,
            "exponential_metrics_within_8ulp": True, "exponential_metrics": differences}


def summarize_baseline_roundoff(episodes):
    audits = [row["baseline_metric_verification"] for row in episodes]
    by_metric = {}
    for name in EXPONENTIAL_METRICS:
        values = [audit["exponential_metrics"][name] for audit in audits]
        by_metric[name] = {"nonexact_instructions": sum(not value["exact"] for value in values),
                           "max_absolute_difference": max(value["absolute_difference"] for value in values),
                           "max_ulp_distance": max(value["ulp_distance"] for value in values)}
    return {"status": "all_exponential_metrics_within_8ulp", "maximum_allowed_ulps": MAX_EXPONENTIAL_ULPS,
            "ulp_definition": "integer distance between nonnegative IEEE-754 binary64 bit patterns",
            "nonexact_instructions": sum(any(not value["exact"] for value in audit["exponential_metrics"].values())
                                          for audit in audits),
            "nonexact_metric_values": sum(row["nonexact_instructions"] for row in by_metric.values()),
            "max_absolute_difference": max(row["max_absolute_difference"] for row in by_metric.values()),
            "max_ulp_distance": max(row["max_ulp_distance"] for row in by_metric.values()),
            "by_metric": by_metric}


def score_head(head, features):
    require(features.device.type == "cpu" and features.dtype == torch.float32
            and features.ndim == 2 and features.shape[1] == 1536 and torch.isfinite(features).all(),
            "invalid natural endpoint features")
    with torch.inference_mode():
        single = torch.cat([head(row.reshape(1, 1536)).reshape(1) for row in features])
        batched = head(features)
    selected, batched_selected = int(single.argmax()), int(batched.argmax())
    sorted_values = torch.sort(single, descending=True).values
    margin = float(sorted_values[0] - sorted_values[1]) if len(single) > 1 else None
    return {"scores": single.tolist(), "selected_state": selected,
            "top1_margin": margin, "batched_selected_state": batched_selected,
            "batched_top1_stable": selected == batched_selected,
            "max_single_vs_batched_score_error": float((single - batched).abs().max()),
            "primary_forward_shape": [1, 1536]}


def summarize(rows):
    return {"instructions": len(rows), "sr": 100. * sum(r["success"] for r in rows) / len(rows),
            "spl": 100. * sum(r["spl"] for r in rows) / len(rows),
            **{key: sum(row[key] for row in rows) / len(rows)
               for key in ("nav_error", "trajectory_lengths", "action_steps", "trajectory_steps", "return_length_m")},
            **{key: 100. * sum(row[key] for row in rows) / len(rows) for key in ("oracle_success", "nDTW", "SDTW", "CLS")}}


def compare_all(reports, metadata, compare):
    contrasts = [("baseline", arm) for arm in ARMS] + [("C1", "C2"), ("C2", "C3"), ("C3", "M")]
    out = {}
    for before, after in contrasts:
        left, right = reports[before], reports[after]
        require([r["instr_id"] for r in left] == [r["instr_id"] for r in right], "comparison inventory changed")
        result = compare({"metadata": metadata, "episodes": left}, {"metadata": metadata, "episodes": right},
                         resamples=20000, seed=0)
        result["rescued_instructions"] = [a["instr_id"] for a, b in zip(left, right) if not a["success"] and b["success"]]
        result["harmed_instructions"] = [a["instr_id"] for a, b in zip(left, right) if a["success"] and not b["success"]]
        result["mean_return_length_delta_m"] = sum(b["return_length_m"] - a["return_length_m"]
                                                   for a, b in zip(left, right)) / len(left)
        out[f"{after}-{before}"] = result
    return out


def run(args):
    started = time.monotonic()
    source, study = ordinary(args.source, directory=True), ordinary(args.study, directory=True)
    require(not args.output.exists() and not args.output.is_symlink(), "refusing to overwrite output")
    sys.path[:0] = [str(source / "src"), str(source / "scripts")]
    from vln_improve.endpoint_group_training import load_endpoint_group_cache, training_code_identity
    from vln_improve.endpoint_probe import load_endpoint_head
    from analyze_endpoint_group_results import load_report
    from evaluate_endpoint_groups import validate_group_head_metadata
    from collect_stopping_traces import validate_reference
    from compare_metrics import compare_results
    # Fail if a pre-imported library silently resolves to a different checkout.
    import vln_improve.endpoint_group_training as training_module
    require(Path(training_module.__file__).resolve() == source / "src/vln_improve/endpoint_group_training.py",
            "training loader imported from a different source checkout")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    lock_path = study / "e1-full-data-lock.json"
    lock = read_json(lock_path)
    checked_content(lock)
    require(lock.get("schema") == "duet_endpoint_full_group_data_lock_v1"
            and lock.get("usage") == "fixed_before_E1_training", "unexpected data lock")
    check_code(source, lock["training_code_identity"])
    require(lock["training_code_identity"] == training_code_identity(), "incomplete training source identity")
    dev_lock = lock["splits"]["train_dev"]
    require(dev_lock["groups"] == 128 and lock["splits"]["train_fit"]["groups"] == 512,
            "data lock is not the full registered group source")
    pair_dir, control_dir = study / "endpoint-pair-full-train-dev", study / "endpoint-controls-full-v2-train-dev"
    require(sha(ordinary(pair_dir / "manifest.json")) == dev_lock["paired_manifest_sha256"]
            and sha(ordinary(control_dir / "manifest.json")) == dev_lock["controls_manifest_sha256"],
            "root cache manifest differs from locked bytes")
    progress("strict_load_started", groups=128, instructions=256)
    cache = load_endpoint_group_cache(pair_dir, control_dir, "train_dev", expected_data_sha256=dev_lock["data_sha256"])
    require(len(cache.groups) == 128 and cache.source_identity == dev_lock["source_identity"]
            and cache.support == dev_lock["support"], "locked development source/support differs")
    progress("strict_load_complete", groups=len(cache.groups), data_sha256=cache.data_sha256)
    common = cache.common_provenance
    check_code(source, cache.control_cache.identity["code_files"])
    check_code(source, cache.pair_cache.identity["code_files"])
    upstream = source / "third_party/VLN-DUET"
    check_code(upstream, {name: item["prepared"] for name, item in common["upstream_lock"]["files"].items()})
    graph_module = load_module(upstream / "map_nav_src/models/graph_utils.py", "endpoint_replay_floyd")
    eval_utils = load_module(upstream / "map_nav_src/r2r/eval_utils.py", "endpoint_replay_eval_utils")
    baseline_path = study / "baseline-train-dev.json"
    baseline = read_json(baseline_path)
    parity = lock["train_dev_natural_baseline_parity"]
    require(parity == {"instructions": 256, "all_full_trajectories_exact": True,
                       "baseline_sha256": sha(baseline_path)}, "baseline differs from locked natural-trajectory reference")
    validate_reference(baseline, common, split="train_dev", seed=0,
                       checkpoint_sha256=common["base_checkpoint_sha256"], expected_episodes=2890)
    metadata = baseline["metadata"]
    require(metadata["feature_id"] == common["feature_sha256"]
            and metadata["train_annotation_sha256"] == common["annotation_sha256"]
            and metadata["connectivity_sha256"] == common["connectivity_sha256"]
            and metadata["upstream_commit"] == common["upstream_lock"]["commit"], "baseline asset provenance differs")
    reference_paths = {r["instr_id"]: r["trajectory"] for r in baseline["trajectories"]}
    reference_metrics = {r["instr_id"]: r for r in baseline["episodes"]}
    heads, head_sources, head_metadata = {}, {}, {}
    for arm in ARMS:
        root = study / f"e1-seed0-{arm}"
        summary = read_json(root / "training-summary.json")
        checkpoint = summary["final_checkpoint"]
        rel = Path(checkpoint["head_relative_path"])
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe final checkpoint path")
        head_path = ordinary(root / rel)
        report, descriptor, meta = load_report(root / "dev-final.json", arm, head_path)
        require(summary["status"] == "complete" and summary["completed_epochs"] == 20
                and summary["global_step"] == 1280 and summary["seed"] == 0,
                "training summary is not the fixed seed-zero final")
        experiment_path = source / "configs" / f"endpoint_group_{arm}.json"
        require(sha(experiment_path) == lock["experiment_sha256"][arm], "registered experiment config changed")
        validate_group_head_metadata(meta, read_json(experiment_path), 0, sha(experiment_path))
        require(meta["data_identity"] == {"train": lock["splits"]["train_fit"]["data_sha256"],
                "dev": dev_lock["data_sha256"], "train_sources": lock["splits"]["train_fit"]["source_identity"],
                "dev_sources": dev_lock["source_identity"]}, "final head data identity differs from data lock")
        require(meta["common_identity"] == {"common_provenance": common,
                "feature_dim": 1536, "feature_schema": "concat_global_local_stop_crossmodal_v1"},
                "head and cached observations provenance differ")
        if head_metadata:
            require(meta["initial_head_sha256"] == head_metadata["C1"]["initial_head_sha256"], "head initialization differs")
        require([(r["pair_id"], r["scan_id"]) for r in report["groups"]]
                == [(g["pair"]["selection_hash"], g["pair"]["scan"]) for g in cache.groups],
                "final head report and development cache pair order differ")
        heads[arm], _ = load_endpoint_head(head_path, device="cpu")
        head_sources[arm], head_metadata[arm] = descriptor, meta
    progress("fixed_final_heads_verified", arms=list(ARMS), epoch=20, global_step=1280, device="cpu")
    scans = {g["pair"]["scan"] for g in cache.groups}
    require(len(scans) == 10, "unexpected natural subset scene inventory")
    graphs, distances, connectivity = load_connectivity(ordinary(args.connectivity, directory=True), scans,
                                                       common["connectivity_sha256"])
    progress("official_connectivity_verified", scenes=len(graphs))
    entries = {g["pair"]["selection_hash"]: g["control_entry"] for g in cache.control_cache.groups}
    rows, episodes = [], {name: [] for name in ("baseline", *ARMS)}
    unique_ids = set()
    for group in cache.groups:
        pair = group["pair"]
        scan = pair["scan"]
        for slot in ("A", "B"):
            natural = group["natural"][slot]
            instr = natural["instr_id"]
            require(instr not in unique_ids and instr in reference_paths, "repeated/missing natural instruction")
            unique_ids.add(instr)
            graph, audit = reconstruct_natural(natural, graph_module)
            require(natural["trajectory"] == reference_paths[instr], "natural trajectory differs from full baseline")
            require(reference_metrics[instr]["scan_id"] == scan, "baseline scene association differs")
            gt_path = entries[pair["selection_hash"]]["controls"][slot]["original_path"]
            require(gt_path[-1] == natural["labels"]["goal_vpid"], "original goal differs")
            for state, distance in zip(natural["states"], natural["labels"]["distance_to_goal"]):
                require(state["position"] == graphs[scan].nodes[state["viewpoint"]]["execution_position"],
                        "saved execution coordinate differs from float32 MatterSim connectivity position")
                require(float(distance) == distances[scan][state["viewpoint"]][gt_path[-1]], "label/connectivity distance differs")
            baseline_scores = navigation_metrics(natural["trajectory"], gt_path, graphs[scan], distances[scan], eval_utils)
            metric_audit = validate_baseline_metrics(reference_metrics[instr], baseline_scores)
            prefix = natural["states"][-1]["trajectory_prefix"]
            prefix_walk = sum(prefix, [])
            prefix_length = float(np.sum([distances[scan][a][b] for a, b in zip(prefix_walk[:-1], prefix_walk[1:])]))
            base_row = {"instr_id": instr, "scan_id": scan, **baseline_scores,
                        "return_length_m": baseline_scores["trajectory_lengths"] - prefix_length}
            episodes["baseline"].append(base_row)
            decisions = {}
            for arm in ARMS:
                scores = score_head(heads[arm], natural["features"])
                target = natural["states"][scores["selected_state"]]["viewpoint"]
                trajectory = endpoint_trajectory(prefix, target, graph)
                metrics = navigation_metrics(trajectory, gt_path, graphs[scan], distances[scan], eval_utils)
                row = {"instr_id": instr, "scan_id": scan, **metrics,
                       "return_length_m": metrics["trajectory_lengths"] - prefix_length}
                episodes[arm].append(row)
                decisions[arm] = {**scores, "endpoint": target, "trajectory": trajectory, "metrics": row,
                    "endpoint_changed": target != natural["trajectory"][-1][-1],
                    "return_segment": trajectory[len(prefix):]}
            rows.append({"pair_id": pair["selection_hash"], "scan_id": scan, "instr_id": instr,
                "state_viewpoints": [s["viewpoint"] for s in natural["states"]],
                "online_prefix": prefix, "online_prefix_length_m": prefix_length,
                "baseline_trajectory": natural["trajectory"], "baseline_metrics": base_row,
                "baseline_metric_verification": metric_audit,
                "baseline_stop_probabilities": natural["natural_stop_probability"].tolist(),
                "graph_reconstruction": audit, "arms": decisions})
    require(len(rows) == len(unique_ids) == 256, "incomplete natural subset; no sample filtering is permitted")
    roundoff = summarize_baseline_roundoff(rows)
    progress("replay_complete", instructions=len(rows), all_baseline_paths_and_sr_spl_exact=True,
             exponential_metric_roundoff=roundoff)
    for items in episodes.values():
        items.sort(key=lambda x: x["instr_id"])
    comparison_metadata = {**metadata, "subset": True, "num_episodes": 256,
                           "mode": "fixed_cached_natural_endpoint_replay", "cohort_schema": SCHEMA}
    result = {"schema": SCHEMA, "split": "train_dev", "usage": "analysis_only", "subset": True,
        "scope": "fixed cached natural trajectories and features; offline endpoint replay on 256 original instructions",
        "instructions": 256, "original_pairs": 128, "scenes": 10,
        "full_train_dev_instructions": 2890, "full_navigation_evaluation": False,
        "parameter_fitting": False, "official_validation_accesses": 0,
        "source": {"data_lock_sha256": sha(lock_path), "dev_data_sha256": cache.data_sha256,
                   "source_identity": cache.source_identity, "baseline_sha256": sha(baseline_path),
                   "heads": head_sources, "connectivity": connectivity,
                   "training_code_identity": lock["training_code_identity"],
                   "script_sha256": sha(Path(__file__))},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "networkx": nx.__version__,
                    "device": "cpu", "threads": 1, "head_training_seed": 0, "navigation_seed": 0},
        "verification": {"all_256_original_baseline_paths_exact": True,
                         "all_original_baseline_sr_spl_exact": True,
                         "all_original_baseline_non_exponential_metrics_exact": True,
                         "exponential_metric_roundoff": roundoff,
                         "all_single_vs_batched_top1_stable": all(d["arms"][a]["batched_top1_stable"] for d in rows for a in ARMS),
                         "states": sum(d["graph_reconstruction"]["states"] for d in rows)},
        "summary": {name: summarize(items) for name, items in episodes.items()},
        "comparisons": compare_all(episodes, comparison_metadata, compare_results), "episodes": rows,
        "limitations": [
            "Selected common-pool training-scene development subset: 256 instructions in 10 scenes, not the full 2890-instruction split.",
            "Natural controls reset the candidate cache per instruction; full run_duet reuses its warm cache across instructions. Equal baseline paths do not establish equal hidden features.",
            "CPU arithmetic can differ from CUDA. Single-state CPU scores define this replay; batched top1 and score margins are reported without filtering any instruction.",
            "Baseline SR/SPL, distances, DTW and steps must match exactly. Only nDTW/SDTW/CLS allow at most 8 IEEE-754 binary64 ULP across CPU exponential implementations; every actual difference and aggregate maximum is reported.",
            "The cached features are visit-time prefix STOP tokens. Historical nodes are not re-encoded at the final decision.",
            "All original moves and online termination are retained. Every replayed retrospective return is reconstructed on the discovered execution graph and charged using official connectivity distances.",
            "All four fixed-final heads are reported. No head, threshold, epoch, seed, or sample is selected from these outcomes.",
            "The paired scene bootstrap uses 20000 draws and seed 0, conditional on these fixed heads and this subset; intervals do not include training-seed uncertainty or multiplicity adjustment."],
        "wall_seconds": time.monotonic() - started}
    result["content_sha256"] = object_sha(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "connectivity", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordinary(args.output.parent, directory=True)
    raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    # Exclusive publication never replaces another run. Partial writes have no
    # valid content checksum and remain visible for diagnosis rather than retry.
    with args.output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    require(sha(args.output) == hashlib.sha256(raw).hexdigest(), "output byte read-back mismatch")
    checked_content(read_json(args.output))
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "instructions": 256, "subset": True, "summary": report["summary"],
                      "verification": report["verification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
