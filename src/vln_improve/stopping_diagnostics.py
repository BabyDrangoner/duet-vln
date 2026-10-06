"""CPU-only endpoint opportunity analysis on complete, fixed DUET trajectories.

This module never evaluates an online stopping policy or fits a score. Its
oracle is restricted to actual decision observations, not intermediate nodes.
"""

from __future__ import annotations

from collections import Counter
import math

import torch


METHODS = ("baseline_probability", "last_position", "raw_stop", "max_margin", "logmeanexp_margin")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _close(left, right, name):
    if not math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-4):
        raise ValueError(f"{name} mismatch: {left} versus {right}")


def _distance(state, source, target):
    nav = state["nav_inputs"]
    nodes = nav["gmap_vpids"][0]
    if source not in nodes or target not in nodes or source is None or target is None:
        raise ValueError("path endpoint absent from the contemporaneous discovered graph")
    value = float(nav["gmap_pair_dists"][0, nodes.index(source), nodes.index(target)])
    if not math.isfinite(value) or value < 0 or value >= 95959595:
        raise ValueError("invalid or unreachable discovered-graph path distance")
    if source == target and value != 0:
        raise ValueError("graph self distance must be zero")
    return value


def _segment_distance(state, source, segment, target):
    if not isinstance(segment, list) or not segment or segment[-1] != target:
        raise ValueError("saved path segment does not end at the expected action/return node")
    nav = state["nav_inputs"]
    nodes = nav["gmap_vpids"][0]
    for node in segment[:-1]:
        if node not in nodes or not bool(nav["gmap_visited_masks"][0, nodes.index(node)]):
            raise ValueError("saved path traverses an unobserved intermediate node")
    sequence = [source] + segment
    length = sum(_distance(state, a, b) for a, b in zip(sequence[:-1], sequence[1:]))
    _close(length, _distance(state, source, target), "saved segment versus discovered-graph distance")
    return length


def analyze_episode(inputs, labels, manifest, *, max_action_len, metrics, reported_trajectory):
    """Analyze one validated diagnostic payload and verify its recorded rollout.

    The collection wrapper also runs the full diagnostic schema validator.
    Unknown reference-path length means no counterfactual SPL is reported.
    """
    if type(max_action_len) is not int or not 1 <= max_action_len <= 15:
        raise ValueError("expected diagnostic action limit between 1 and 15")
    association = inputs["association"]
    if (association != labels["association"] or metrics.get("instr_id") != association["instr_id"]
            or metrics.get("scan_id") != association["scan_id"]):
        raise ValueError("episode/rollout association mismatch")
    states = inputs["states"]
    targets = labels["states"]
    if (not states or [s["step"] for s in states] != list(range(len(states)))
            or len(states) > max_action_len or len(targets) != len(states)
            or sorted(t["step"] for t in targets) != list(range(len(states)))):
        raise ValueError("all consecutive states, including the final state, are required")
    targets = {t["step"]: t for t in targets}
    current_nodes = [s["current_viewpoint"] for s in states]
    if len(set(current_nodes)) != len(current_nodes):
        raise ValueError("diagnostic protocol does not allow repeated decision endpoints")
    trajectory = manifest["trajectory"]
    if trajectory.get("instr_id") != association["instr_id"]:
        raise ValueError("manifest trajectory association mismatch")
    path = trajectory["path"]
    if path != reported_trajectory or not path or path[0] != [current_nodes[0]]:
        raise ValueError("manifest and rollout trajectories differ")

    state_rows = []
    for state in states:
        step = state["step"]
        logits, valid = state["base_logits"], state["valid_mask"]
        if (logits.device.type != "cpu" or valid.device.type != "cpu" or logits.dtype != torch.float32
                or valid.dtype != torch.bool or logits.ndim != 1 or valid.shape != logits.shape
                or not bool(valid[0]) or not torch.isfinite(logits[valid]).all()
                or not torch.isneginf(logits[~valid]).all()):
            raise ValueError("invalid CPU FP32 masked baseline logits")
        nav = state["nav_inputs"]
        nodes = nav["gmap_vpids"][0]
        if len(nodes) != len(logits) or nodes[0] is not None:
            raise ValueError("STOP must occupy index zero")
        expected_mask = nav["gmap_masks"][0] & ~nav["gmap_visited_masks"][0]
        if not torch.equal(valid, expected_mask):
            raise ValueError("recorded legal mask differs from graph mask")
        selected = int(logits.argmax())
        if state.get("baseline_argmax") != selected:
            raise ValueError("recorded baseline argmax disagrees with logits")
        moves = logits[valid][1:]
        stop = targets[step].get("stop", {})
        distance = _number(stop.get("distance_to_goal"), "stop distance_to_goal")
        if (distance < 0 or type(stop.get("within_success_radius")) is not bool
                or stop["within_success_radius"] != (distance < 3)
                or type(stop.get("exact_goal")) is not bool
                or (stop["exact_goal"] and distance != 0)):
            raise ValueError("STOP labels disagree with the 3m success definition")
        flags = {"argmax_stop": selected == 0, "no_legal_move": len(moves) == 0,
                 "action_limit": step == max_action_len - 1}
        if state["eligible_decision"] != (not flags["no_legal_move"] and not flags["action_limit"]):
            raise ValueError("eligible_decision disagrees with forced-ending flags")
        if any(flags.values()) != (step == len(states) - 1):
            raise ValueError("saved states do not match baseline termination conditions")
        if step < len(states) - 1 and nodes[selected] != current_nodes[step + 1]:
            raise ValueError("recorded action does not reach next decision state")
        z_stop = float(logits[0])
        state_rows.append({
            "step": step, "viewpoint": state["current_viewpoint"], "eligible_decision": state["eligible_decision"],
            "move_count": len(moves), "distance_to_goal_m": distance, "success": distance < 3,
            "termination_flags": flags, "scores": {
                "baseline_probability": float(torch.softmax(logits, dim=0)[0]), "raw_stop": z_stop,
                "max_margin": z_stop - float(moves.max()) if len(moves) else None,
                "logmeanexp_margin": z_stop - float(torch.logsumexp(moves, 0)) + math.log(len(moves)) if len(moves) else None,
            },
        })

    # Python's first maximum implements the upstream dictionary's strict > tie rule.
    baseline_index = max(range(len(states)), key=lambda i: state_rows[i]["scores"]["baseline_probability"])
    baseline_node = current_nodes[baseline_index]
    if path[-1][-1] != baseline_node:
        raise ValueError("historical STOP argmax does not match saved final endpoint")
    returned = baseline_node != current_nodes[-1]
    if len(path) != len(states) + int(returned):
        raise ValueError("saved trajectory has an unexpected number of movement/return segments")
    prefix_length = sum(_segment_distance(states[t], current_nodes[t], path[t + 1], current_nodes[t + 1])
                        for t in range(len(states) - 1))
    baseline_return = (_segment_distance(states[-1], current_nodes[-1], path[-1], baseline_node)
                       if returned else 0.0)
    _close(prefix_length + baseline_return, _number(metrics.get("trajectory_lengths"), "rollout length"),
           "complete path versus rollout length")
    _close(state_rows[baseline_index]["distance_to_goal_m"], _number(metrics.get("nav_error"), "rollout nav_error"),
           "final navigation error")
    if _number(metrics.get("success"), "rollout success") != int(state_rows[baseline_index]["success"]):
        raise ValueError("rollout success differs from endpoint label")

    def endpoint(index):
        row = state_rows[index]
        cost = _distance(states[-1], current_nodes[-1], current_nodes[index])
        return {"selected_step": index, "viewpoint": current_nodes[index], "success": row["success"],
                "nav_error_m": row["distance_to_goal_m"], "return_distance_m": cost,
                "complete_trajectory_length_m": prefix_length + cost,
                "selected_state_move_count": row["move_count"]}

    results = {"baseline_probability": endpoint(baseline_index), "last_position": endpoint(len(states) - 1)}
    for method in METHODS[2:]:
        if any(row["scores"][method] is None for row in state_rows):
            results[method] = {"available": False, "reason": "a_historical_state_has_no_legal_move"}
        else:
            chosen = max(range(len(states)), key=lambda i: state_rows[i]["scores"][method])
            results[method] = endpoint(chosen)
    successful = [i for i, row in enumerate(state_rows) if row["success"]]
    standard_oracle = _number(metrics.get("oracle_success"), "rollout oracle_success")
    if standard_oracle not in (0, 1) or (successful and not standard_oracle):
        raise ValueError("observed successful state contradicts the rollout oracle-success metric")
    oracle_index = min(successful, key=lambda i: _distance(states[-1], current_nodes[-1], current_nodes[i])) if successful else None
    baseline_ok, last_ok = results["baseline_probability"]["success"], results["last_position"]["success"]
    category = ("both_success" if baseline_ok else "fallback_harmed") if last_ok else ("fallback_rescued" if baseline_ok else "both_failure")
    return {"association": association, "num_states": len(states), "all_states_included": True,
            "states": state_rows, "termination_flags": state_rows[-1]["termination_flags"],
            "prefix_length_m": prefix_length, "start_goal_shortest_distance_m": state_rows[0]["distance_to_goal_m"],
            "methods": results, "fallback_category": category, "fallback_changed_endpoint": returned,
            "observed_history_has_success": bool(successful),
            "observed_history_success_but_baseline_failed": bool(successful) and not baseline_ok,
            "observed_history_success_min_return": endpoint(oracle_index) if oracle_index is not None else None,
            "rollout_standard_oracle_success": bool(standard_oracle),
            "parity": {"historical_stop_endpoint": True, "complete_trajectory": True, "trajectory_length": True}}


def summarize_episodes(rows):
    """Keep explicit denominators, including unavailable zero-move scores."""
    count = len(rows)
    categories = Counter(row["fallback_category"] for row in rows)
    summary = {"episodes": count, "states": sum(row["num_states"] for row in rows),
               "fallback_categories": {key: categories[key] for key in (
                   "fallback_rescued", "fallback_harmed", "both_success", "both_failure")},
               "fallback_changed_endpoint": sum(row["fallback_changed_endpoint"] for row in rows),
               "observed_history_has_success": sum(row["observed_history_has_success"] for row in rows),
               "observed_history_success_but_baseline_failed": sum(row["observed_history_success_but_baseline_failed"] for row in rows),
               "standard_oracle_success_but_baseline_failed": sum(row["rollout_standard_oracle_success"] and not row["methods"]["baseline_probability"]["success"] for row in rows),
               "termination_flags": {flag: sum(row["termination_flags"][flag] for row in rows)
                                     for flag in ("argmax_stop", "no_legal_move", "action_limit")}, "methods": {}}
    for method in METHODS:
        covered = [row for row in rows if row["methods"][method].get("available", True)]
        results = [row["methods"][method] for row in covered]
        n = len(results)
        summary["methods"][method] = {
            "covered_episodes": n, "unavailable_episodes": count - n,
            "successes": sum(r["success"] for r in results),
            "sr_percent": 100 * sum(r["success"] for r in results) / n if n else None,
            "baseline_sr_percent_same_episodes": 100 * sum(row["methods"]["baseline_probability"]["success"] for row in covered) / n if n else None,
            "rescued_vs_baseline": sum(r["methods"][method]["success"] and not r["methods"]["baseline_probability"]["success"] for r in covered),
            "harmed_vs_baseline": sum(not r["methods"][method]["success"] and r["methods"]["baseline_probability"]["success"] for r in covered),
            **{f"mean_{key}": sum(r[key] for r in results) / n if n else None
               for key in ("nav_error_m", "return_distance_m", "complete_trajectory_length_m")},
        }
    return summary
