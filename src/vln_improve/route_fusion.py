"""CPU-only route-fusion arithmetic on frozen baseline states.

These are single-state diagnostic choices, not simulated trajectories or SR/SPL.
The caller supplies the hash-verified upstream GraphMap implementation.
"""
from __future__ import annotations

from collections import Counter
import math

import torch

MODES = ("original_shared_sum", "executed_firsthop_local", "mean_backward_local", "logsumexp_backward_local",
         "hspr_style_remote_double_global")
SHIFT_INVARIANT = {"executed_firsthop_local", "mean_backward_local", "logsumexp_backward_local"}
SHIFT_ATOL = 1e-12


class RevealedRoutes:
    """Add only prefix-observed edges, in the original candidate/update order."""

    def __init__(self, positions, graph_factory):
        self.positions = positions
        self.factory = graph_factory
        self.graph = None
        self.edges = set()
        self.last_step = -1
        self.current = None

    def advance(self, state):
        current, step = state["current_viewpoint"], state["step"]
        if step != self.last_step + 1:
            raise ValueError("route reconstruction needs the complete chronological prefix")
        nav = state["nav_inputs"]
        candidates = nav["vp_cand_vpids"][0][1:]
        if current in candidates or len(set(candidates)) != len(candidates):
            raise ValueError("invalid local candidate IDs")
        observation = {"viewpoint": current, "position": self.positions[current],
                       "candidate": [{"viewpointId": vp, "position": self.positions[vp]} for vp in candidates]}
        if self.graph is None:
            self.graph = self.factory(current)
        self.graph.update_graph(observation)
        self.graph.node_step_ids[current] = step + 1
        self.edges.update(frozenset((current, vp)) for vp in candidates)
        self.current, self.last_step = current, step
        visited = [vp for vp in self.graph.node_positions if self.graph.graph.visited(vp)]
        unseen = [vp for vp in self.graph.node_positions if not self.graph.graph.visited(vp)]
        ids = nav["gmap_vpids"][0]
        if ids != [None] + visited + unseen:
            raise ValueError("reconstructed graph node order differs from saved state")
        expected_visited = torch.tensor([False] + [vp in visited for vp in ids[1:]])
        if not torch.equal(nav["gmap_visited_masks"][0], expected_visited):
            raise ValueError("reconstructed visited set differs from saved state")
        if nav["gmap_step_ids"][0].tolist() != [self.graph.node_step_ids.get(vp, 0) for vp in ids]:
            raise ValueError("reconstructed graph step IDs differ from saved state")
        distances = torch.zeros((len(ids), len(ids)), dtype=torch.float32)
        for i in range(1, len(ids)):
            for j in range(i + 1, len(ids)):
                distances[i, j] = distances[j, i] = self.graph.graph.distance(ids[i], ids[j])
        error = float((distances - nav["gmap_pair_dists"][0]).abs().max())
        if not torch.equal(distances, nav["gmap_pair_dists"][0]):
            raise ValueError(f"reconstructed full pair distances differ from saved FP32 state: max error={error}")
        return error

    def route(self, target):
        if self.graph is None or target not in self.graph.node_positions:
            raise ValueError("target is not in the revealed graph")
        path = [self.current] + self.graph.graph.path(self.current, target)
        if path[-1] != target or len(set(path)) != len(path):
            raise ValueError("invalid revealed route")
        if any(frozenset((a, b)) not in self.edges for a, b in zip(path, path[1:])):
            raise ValueError("route contains an unrevealed edge")
        if any(not self.graph.graph.visited(vp) for vp in path[1:-1]):
            raise ValueError("route uses an unobserved intermediate viewpoint")
        return path


def route_context(state, routes):
    nav = state["nav_inputs"]
    ids, local_ids = nav["gmap_vpids"][0], nav["vp_cand_vpids"][0]
    if not ids or ids[0] is not None or not local_ids or local_ids[0] is not None:
        raise ValueError("STOP must be index zero in both action lists")
    valid = nav["gmap_masks"][0] & ~nav["gmap_visited_masks"][0]
    if valid.dtype != torch.bool or not valid[0] or not torch.equal(valid, state["valid_mask"]):
        raise ValueError("saved legal mask differs from navigation masks")
    if not nav["gmap_masks"].all() or len(ids) != len(valid):
        raise ValueError("requires unpadded batch-one complete graph")
    local_mask = nav["vp_nav_masks"][0]
    if len(local_ids) > len(local_mask) or not local_mask[:len(local_ids)].all() or local_mask[len(local_ids):].any():
        raise ValueError("local masks and candidate IDs disagree")
    visited = {vp for vp, mask in zip(ids, nav["gmap_visited_masks"][0]) if mask}
    if routes.current != state["current_viewpoint"] or routes.last_step != state["step"]:
        raise ValueError("route prefix and saved state disagree")
    backward = [j for j, vp in enumerate(local_ids) if j and vp in visited]
    remote = [j for j, vp in enumerate(ids) if j and valid[j] and vp not in local_ids]
    if remote and not backward:
        raise ValueError("remote candidates with K=0 cannot be assigned a fabricated backward score")
    firsthops = {}
    for j in remote:
        route = routes.route(ids[j])
        if len(route) < 3 or route[1] not in local_ids or route[1] not in visited:
            raise ValueError("remote route first hop is not a visited local candidate")
        firsthops[j] = local_ids.index(route[1])
    for key, mask in (("base_logits", valid), ("base_global_logits", valid), ("base_local_logits", local_mask)):
        values = state[key]
        if (values.device.type != "cpu" or values.dtype != torch.float32 or values.shape != mask.shape
                or not torch.isfinite(values[mask]).all() or not torch.isneginf(values[~mask]).all()):
            raise ValueError(f"invalid saved FP32 CPU logits: {key}")
    return {"ids": ids, "local_ids": local_ids, "valid": valid, "local_mask": local_mask,
            "backward": backward, "remote": remote, "firsthops": firsthops}


def fuse_scores(state, context, mode, *, dtype=torch.float32, local_shift=0.0):
    """Use already fuse-weighted logits, preserving upstream addition order."""
    if mode not in MODES:
        raise ValueError("unknown fixed fusion control")
    global_scores = state["base_global_logits"].to(dtype).clone()
    local = state["base_local_logits"].to(dtype).clone()
    local[context["local_mask"]] += local_shift
    scores = global_scores.clone()
    scores[0] += local[0]
    backward = context["backward"]
    backward_sum = torch.tensor(0.0, dtype=dtype)
    for j in backward:
        backward_sum += local[j]
    for j, vp in enumerate(context["ids"]):
        if j == 0 or not context["valid"][j]:
            continue
        if vp in context["local_ids"]:
            scores[j] += local[context["local_ids"].index(vp)]
        elif mode == "original_shared_sum":
            scores[j] += backward_sum
        elif mode == "executed_firsthop_local":
            scores[j] += local[context["firsthops"][j]]
        elif mode == "mean_backward_local":
            scores[j] += backward_sum / len(backward)
        elif mode == "logsumexp_backward_local":
            scores[j] += torch.logsumexp(local[backward], 0)
        else:
            scores[j] += global_scores[j]
    return scores


def inspect_fusion(state, routes):
    context = route_context(state, routes)
    valid = context["valid"]
    scores = {mode: fuse_scores(state, context, mode) for mode in MODES}
    original = scores["original_shared_sum"]
    error = float((original[valid] - state["base_logits"][valid]).abs().max())
    if not torch.equal(original[valid], state["base_logits"][valid]):
        raise ValueError(f"original finite-legal fused reconstruction differs: max error={error}")
    chosen = int(original.argmax())
    if state.get("baseline_argmax") != chosen:
        raise ValueError("saved baseline argmax disagrees with saved logits")
    shifts, numeric_ties, max_shift_error = {}, {}, 0.0
    double = {mode: fuse_scores(state, context, mode, dtype=torch.float64) for mode in MODES}
    for shift in (-1.0, 1.0):
        changed, ties = {}, {}
        for mode in MODES:
            shifted = fuse_scores(state, context, mode, dtype=torch.float64, local_shift=shift)
            expected = torch.full_like(shifted, shift)
            if mode == "original_shared_sum":
                expected[context["remote"]] = shift * len(context["backward"])
            elif mode == "hspr_style_remote_double_global":
                expected[context["remote"]] = 0.0
            response_error = float(((shifted - double[mode])[valid] - expected[valid]).abs().max())
            max_shift_error = max(max_shift_error, response_error)
            if response_error > SHIFT_ATOL:
                raise ValueError(f"common-shift response identity failed for {mode}: {response_error}")
            changed[mode] = int(shifted.argmax()) != int(double[mode].argmax())
            ties[mode] = False
            if mode in SHIFT_INVARIANT and changed[mode]:
                before, after = int(double[mode].argmax()), int(shifted.argmax())
                gap = float(abs(double[mode][before] - double[mode][after]))
                if gap > 2 * SHIFT_ATOL:
                    raise ValueError(f"common-shift argmax invariance failed for {mode}")
                ties[mode] = True
        shifts[str(int(shift))] = changed
        numeric_ties[str(int(shift))] = ties
    return {"chosen": {mode: int(values.argmax()) for mode, values in scores.items()},
            "remote_indices": context["remote"], "return_neighbors": len(context["backward"]),
            "remote_firsthop_groups": len(set(context["firsthops"].values())),
            "stop_only": int(valid.sum()) == 1, "max_fused_reconstruction_error": error,
            "max_shift_response_error": max_shift_error, "shift_changes": shifts,
            "numeric_tie_shift_changes": numeric_ties,
            "fp32_fp64_argmax_differs": {mode: int(scores[mode].argmax()) != int(double[mode].argmax()) for mode in MODES}}


def _oracle_outcome(costs, choice, optimal):
    finite = torch.isfinite(costs)
    if not finite.any():
        return {"optimal": None, "regret": None, "nonfinite_choice": True}
    cost = float(costs[choice])
    return {"optimal": choice in optimal, "regret": cost - float(costs[finite].min()) if math.isfinite(cost) else None,
            "nonfinite_choice": not math.isfinite(cost)}


def analyze_episode(states, oracle, trajectory, positions, graph_factory, association):
    """Validate every recorded state, then return only compact state-level facts."""
    if list(states) != list(range(len(states))) or not states:
        raise ValueError("states must be the full zero-based chronological episode")
    path = trajectory.get("path")
    if (trajectory.get("instr_id") != association["instr_id"] or not isinstance(path, list)
            or not path or path[0] != [states[0]["current_viewpoint"]]
            or any(not isinstance(segment, list) or not segment or any(not isinstance(vp, str) for vp in segment)
                   for segment in path)):
        raise ValueError("manifest trajectory association/path is invalid")
    routes, rows, segment_index = RevealedRoutes(positions, graph_factory), [], 1
    for step, state in states.items():
        distance_error = routes.advance(state)
        result = inspect_fusion(state, routes)
        chosen = result["chosen"]["original_shared_sum"]
        next_state = states.get(step + 1)
        moves = state["eligible_decision"] and chosen != 0
        if moves:
            target = state["nav_inputs"]["gmap_vpids"][0][chosen]
            expected = routes.route(target)[1:]
            if (next_state is None or next_state["current_viewpoint"] != target
                    or segment_index >= len(path) or path[segment_index] != expected):
                raise ValueError("baseline chosen action/actual next state/manifest route disagree")
            segment_index += 1
        elif next_state is not None:
            raise ValueError("episode continues after a STOP or forced terminal decision")
        costs, candidate_costs = {}, {}
        for kind in ("teacher", "execution"):
            values = oracle[step][kind + "_cost"]
            optimal = oracle[step][kind + "_optimal_indices"]
            costs[kind] = {mode: _oracle_outcome(values, choice, optimal) for mode, choice in result["chosen"].items()}
            candidates = [i for i in range(1, len(state["valid_mask"])) if state["valid_mask"][i]]
            remote = result["remote_indices"]
            candidate_costs[kind] = {}
            for group, indices in (("all_nonstop", candidates), ("remote", remote),
                                   ("local_unvisited", [i for i in candidates if i not in remote])):
                outcomes = [_oracle_outcome(values, i, optimal) for i in indices]
                finite_regrets = [value["regret"] for value in outcomes if value["regret"] is not None]
                candidate_costs[kind][group] = {"candidate_occurrences": len(indices),
                                                "finite_regret_count": len(finite_regrets),
                                                "finite_regret_sum": sum(finite_regrets),
                                                "optimal_candidates": sum(value["optimal"] is True for value in outcomes)}
        rows.append({**association, "step": step, "eligible": state["eligible_decision"],
                     "next_path_checked": bool(moves), "max_pair_distance_error": distance_error,
                     **result, "oracle": costs, "candidate_oracle": candidate_costs})
    remaining = path[segment_index:]
    if len(remaining) > 1:
        raise ValueError("manifest has unexpected trajectory segments after termination")
    if remaining:
        stop_node = remaining[0][-1]
        if not routes.graph.graph.visited(stop_node) or remaining[0] != routes.route(stop_node)[1:]:
            raise ValueError("terminal fallback path differs from the revealed graph")
    return rows


def summarize_rows(rows):
    """State-weighted paired statistics; finite denominators are always explicit."""
    eligible = [row for row in rows if row["eligible"]]
    coverage = {"recorded_states": len(rows), "eligible_states": len(eligible),
                "ineligible_states": len(rows) - len(eligible),
                "recorded_stop_only_states": sum(row["stop_only"] for row in rows),
                "eligible_stop_only_states": sum(row["stop_only"] for row in eligible),
                "actual_next_paths_checked": sum(row["next_path_checked"] for row in rows)}
    predicates = {
        "has_remote": lambda r: bool(r["remote_indices"]),
        "return_neighbors_ge2": lambda r: r["return_neighbors"] >= 2,
        "remote_and_return_neighbors_ge2": lambda r: bool(r["remote_indices"]) and r["return_neighbors"] >= 2,
        "remote_firsthop_groups_ge2": lambda r: r["remote_firsthop_groups"] >= 2,
        "baseline_selects_remote": lambda r: r["chosen"]["original_shared_sum"] in r["remote_indices"],
    }
    for name, predicate in predicates.items():
        coverage[name] = sum(predicate(row) for row in eligible)
        coverage["recorded_" + name] = sum(predicate(row) for row in rows)
    coverage["baseline_selects_remote_fraction"] = coverage["baseline_selects_remote"] / len(eligible) if eligible else None
    coverage["return_neighbor_histogram"] = dict(sorted(Counter(str(row["return_neighbors"]) for row in eligible).items()))
    coverage["recorded_return_neighbor_histogram"] = dict(sorted(Counter(str(row["return_neighbors"]) for row in rows).items()))
    coverage["remote_firsthop_group_histogram"] = dict(sorted(Counter(str(row["remote_firsthop_groups"]) for row in eligible).items()))
    candidate_stats = {}
    for kind in ("teacher", "execution"):
        candidate_stats[kind] = {}
        for group in ("all_nonstop", "remote", "local_unvisited"):
            values = {key: sum(row["candidate_oracle"][kind][group][key] for row in eligible) for key in (
                "candidate_occurrences", "finite_regret_count", "finite_regret_sum", "optimal_candidates")}
            values["mean_finite_regret"] = (values["finite_regret_sum"] / values["finite_regret_count"]
                                             if values["finite_regret_count"] else None)
            candidate_stats[kind][group] = values
    comparisons = {}
    for mode in MODES:
        values = {"argmax_changes": sum(r["chosen"][mode] != r["chosen"]["original_shared_sum"] for r in eligible),
                  "fp32_fp64_argmax_differences": sum(r["fp32_fp64_argmax_differs"][mode] for r in eligible)}
        for kind in ("teacher", "execution"):
            paired, regrets = [], []
            stats = dict(states=len(eligible), without_finite_oracle=0, nonfinite_selected_cost=0,
                         optimal_choices=0, fixes=0, damages=0, finite_regret_count=0,
                         paired_finite_regret_count=0, nonfinite_to_finite=0, finite_to_nonfinite=0)
            for row in eligible:
                base, alt = row["oracle"][kind]["original_shared_sum"], row["oracle"][kind][mode]
                if alt["optimal"] is None:
                    stats["without_finite_oracle"] += 1
                    continue
                stats["optimal_choices"] += int(alt["optimal"])
                stats["nonfinite_selected_cost"] += int(alt["nonfinite_choice"])
                stats["fixes"] += int(base["optimal"] is False and alt["optimal"] is True)
                stats["damages"] += int(base["optimal"] is True and alt["optimal"] is False)
                stats["nonfinite_to_finite"] += int(base["regret"] is None and alt["regret"] is not None)
                stats["finite_to_nonfinite"] += int(base["regret"] is not None and alt["regret"] is None)
                if alt["regret"] is not None:
                    regrets.append(alt["regret"])
                if base["regret"] is not None and alt["regret"] is not None:
                    paired.append(alt["regret"] - base["regret"])
            stats.update(finite_regret_count=len(regrets), finite_regret_sum=sum(regrets),
                         mean_finite_regret=sum(regrets) / len(regrets) if regrets else None,
                         paired_finite_regret_count=len(paired), paired_regret_delta_sum=sum(paired),
                         paired_mean_regret_delta=sum(paired) / len(paired) if paired else None)
            values[kind] = stats
        comparisons[mode] = values
    return {"coverage": coverage, "controls": comparisons, "candidate_descriptive": candidate_stats,
            "shift_argmax_changes": {shift: {mode: sum(r["shift_changes"][shift][mode] for r in eligible)
                                               for mode in MODES} for shift in ("-1", "1")},
            "numeric_tie_shift_argmax_changes": {shift: {mode: sum(r["numeric_tie_shift_changes"][shift][mode] for r in eligible)
                                                           for mode in MODES} for shift in ("-1", "1")},
            "integrity": {key: max((r[key] for r in rows), default=0.0) for key in (
                "max_fused_reconstruction_error", "max_pair_distance_error", "max_shift_response_error")}}


def summarize_houses(rows):
    by_house = {scan: summarize_rows([row for row in rows if row["scan_id"] == scan])
                for scan in sorted({row["scan_id"] for row in rows})}
    macro = {}
    for mode in MODES:
        macro[mode] = {}
        for kind in ("teacher", "execution"):
            values = [house["controls"][mode][kind]["paired_mean_regret_delta"] for house in by_house.values()]
            finite = [value for value in values if value is not None]
            macro[mode][kind] = {"houses_with_finite_pairs": len(finite),
                                "mean_house_paired_regret_delta": sum(finite) / len(finite) if finite else None}
    return {"overall": summarize_rows(rows), "per_house": by_house, "house_macro": macro}
