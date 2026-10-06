"""E2 prototype: compare observed endpoints with the original DUET endpoint.

This module does not run navigation or change the frozen E1 implementation.
Policy inputs use observations and discovered-graph costs only. Supervised
metric targets have a separate interface and must be built on training data.
The selection rule constrains predictions; it is not a safety guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SCALAR_NAMES = (
    "candidate_stop_probability", "baseline_stop_probability",
    "candidate_visit_age_fraction", "baseline_visit_age_fraction",
    "log1p_prefix_length_m", "candidate_return_over_prefix_plus_one",
    "baseline_return_over_prefix_plus_one", "return_difference_over_prefix_plus_one",
    "candidate_is_terminal", "candidate_is_baseline",
)


def _number(value, name, *, nonnegative=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError(f"invalid {name}")
    return float(value)


def _ids(values, name):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"invalid {name}")
    return tuple(values)


@dataclass(frozen=True)
class InterventionInputs:
    candidate_vpids: tuple[str, ...]
    baseline_index: int
    node_features: Tensor                 # [N, 768], terminal global node tokens
    terminal_context: Tensor              # [1536], terminal global/local STOP tokens
    scalar_features: Tensor               # [N, len(SCALAR_NAMES)]


@dataclass(frozen=True)
class InterventionBatch:
    candidate_vpids: tuple[tuple[str, ...], ...]
    baseline_indices: Tensor              # [B]
    node_features: Tensor                 # [B, N, 768]
    terminal_context: Tensor              # [B, 1536]
    scalar_features: Tensor               # [B, N, len(SCALAR_NAMES)]
    valid_mask: Tensor                    # [B, N], excludes padding


def build_intervention_inputs(
    final_nav_inputs: dict, final_nav_outs: dict, *,
    observed_vpids: Sequence[str], visit_steps: Mapping[str, int],
    stop_probabilities: Mapping[str, float], baseline_vpid: str,
    termination_vpid: str, prefix_length_m: float,
    return_distances_m: Mapping[str, float],
) -> InterventionInputs:
    """Align final-context node tokens using the actual decision-observation IDs.

    A visited graph node alone is insufficient to become an endpoint candidate.
    ``return_distances_m`` must come from the discovered execution graph at
    termination. No environment/goal-distance lookup is made here. Scalar
    normalization is a fixed formula, with no data-fitted statistics.
    """
    ids = _ids(observed_vpids, "observed decision IDs")
    if len(ids) > 15 or baseline_vpid not in ids or termination_vpid not in ids:
        raise ValueError("invalid observed endpoint/baseline/terminal inventory")
    for mapping, name in ((visit_steps, "visit steps"),
                          (stop_probabilities, "STOP probabilities"),
                          (return_distances_m, "return distances")):
        if not isinstance(mapping, Mapping) or set(mapping) != set(ids):
            raise ValueError(f"{name} must cover exactly the observed decisions")
    if (any(type(v) is not int for v in visit_steps.values())
            or sorted(visit_steps.values()) != list(range(len(ids)))
            or visit_steps[termination_vpid] != len(ids) - 1):
        raise ValueError("visit steps must identify the actual terminal observation")
    probabilities = {v: _number(stop_probabilities[v], "STOP probability") for v in ids}
    if any(p > 1 for p in probabilities.values()):
        raise ValueError("STOP probability must be in [0, 1]")
    # Original DUET uses strict > in chronological insertion order.
    chronological = sorted(ids, key=visit_steps.__getitem__)
    if max(chronological, key=probabilities.__getitem__) != baseline_vpid:
        raise ValueError("baseline endpoint differs from original STOP selection")
    prefix = _number(prefix_length_m, "observed prefix length")
    returns = {v: _number(return_distances_m[v], "discovered return distance") for v in ids}
    if returns[termination_vpid] != 0:
        raise ValueError("terminal return distance must be zero")

    global_tokens, local_tokens = final_nav_outs["gmap_embeds"], final_nav_outs["vp_embeds"]
    for tokens, name in ((global_tokens, "global tokens"), (local_tokens, "local tokens")):
        if (not isinstance(tokens, Tensor) or tokens.ndim != 3 or tokens.shape[0] != 1
                or tokens.shape[1] < 1 or tokens.shape[2] != 768
                or not tokens.is_floating_point()):
            raise ValueError(f"invalid terminal {name}")
    if global_tokens.device != local_tokens.device:
        raise ValueError("terminal token devices differ")
    local_ids = final_nav_inputs["vp_cand_vpids"]
    if (not isinstance(local_ids, (list, tuple)) or len(local_ids) != 1
            or not isinstance(local_ids[0], (list, tuple)) or not local_ids[0]
            or local_ids[0][0] is not None or len(local_ids[0]) > local_tokens.shape[1]):
        raise ValueError("terminal local STOP ID must be None at index zero")
    global_ids = final_nav_inputs["gmap_vpids"]
    if (not isinstance(global_ids, (list, tuple)) or len(global_ids) != 1
            or not isinstance(global_ids[0], (list, tuple)) or not global_ids[0]
            or global_ids[0][0] is not None
            or len(global_ids[0]) > global_tokens.shape[1]):
        raise ValueError("invalid terminal global node IDs")
    graph_ids = tuple(global_ids[0])
    if (any(not isinstance(v, str) or not v for v in graph_ids[1:])
            or len(set(graph_ids)) != len(graph_ids)):
        raise ValueError("invalid terminal global node ID inventory")
    masks = {}
    for key, shape in (("gmap_masks", global_tokens.shape[:2]),
                       ("gmap_visited_masks", global_tokens.shape[:2]),
                       ("vp_masks", local_tokens.shape[:2])):
        mask = final_nav_inputs[key]
        if (not isinstance(mask, Tensor) or mask.dtype != torch.bool
                or mask.shape != shape or mask.device != global_tokens.device):
            raise ValueError(f"invalid terminal {key}")
        masks[key] = mask
    if (not bool(masks["gmap_masks"][0, 0]) or bool(masks["gmap_visited_masks"][0, 0])
            or not bool(masks["vp_masks"][0, 0])
            or bool(masks["gmap_masks"][0, len(graph_ids):].any())):
        raise ValueError("invalid STOP or global padding mask")
    indices = []
    for node in ids:
        if node not in graph_ids:
            raise ValueError("observed decision is absent from terminal graph")
        index = graph_ids.index(node)
        if not bool(masks["gmap_masks"][0, index] & masks["gmap_visited_masks"][0, index]):
            raise ValueError("observed decision is invalid/unvisited in terminal graph")
        indices.append(index)
    nodes = global_tokens[0, indices].detach().float().clone()
    context = torch.cat((global_tokens[0, 0], local_tokens[0, 0])).detach().float().clone()
    if not torch.isfinite(nodes).all() or not torch.isfinite(context).all():
        raise ValueError("nonfinite observed terminal representation")
    end_step, denominator = len(ids) - 1, prefix + 1.0
    scalars = torch.tensor([
        [probabilities[v], probabilities[baseline_vpid],
         (end_step - visit_steps[v]) / max(1, end_step),
         (end_step - visit_steps[baseline_vpid]) / max(1, end_step),
         math.log1p(prefix), returns[v] / denominator,
         returns[baseline_vpid] / denominator,
         (returns[v] - returns[baseline_vpid]) / denominator,
         float(v == termination_vpid), float(v == baseline_vpid)] for v in ids
    ], dtype=torch.float32, device=nodes.device)
    if not torch.isfinite(scalars).all():
        raise ValueError("nonfinite observable scalar representation")
    return InterventionInputs(ids, ids.index(baseline_vpid), nodes, context, scalars)


def collate_interventions(examples: Sequence[InterventionInputs], device="cpu") -> InterventionBatch:
    if not examples:
        raise ValueError("empty intervention batch")
    width = max(len(x.candidate_vpids) for x in examples)
    nodes = torch.zeros(len(examples), width, 768, device=device)
    scalars = torch.zeros(len(examples), width, len(SCALAR_NAMES), device=device)
    contexts = torch.zeros(len(examples), 1536, device=device)
    mask = torch.zeros(len(examples), width, dtype=torch.bool, device=device)
    anchors = []
    for row, item in enumerate(examples):
        ids = _ids(item.candidate_vpids, "candidate IDs")
        n = len(ids)
        if (not 1 <= n <= 15 or type(item.baseline_index) is not int
                or not 0 <= item.baseline_index < n
                or item.node_features.shape != (n, 768)
                or item.terminal_context.shape != (1536,)
                or item.scalar_features.shape != (n, len(SCALAR_NAMES))
                or any(not t.is_floating_point() or not torch.isfinite(t).all() for t in
                       (item.node_features, item.terminal_context, item.scalar_features))):
            raise ValueError("invalid intervention example")
        nodes[row, :n] = item.node_features.detach().to(device)
        contexts[row] = item.terminal_context.detach().to(device)
        scalars[row, :n] = item.scalar_features.detach().to(device)
        mask[row, :n] = True
        anchors.append(item.baseline_index)
    return InterventionBatch(tuple(x.candidate_vpids for x in examples),
        torch.tensor(anchors, dtype=torch.int64, device=device), nodes, contexts, scalars, mask)


class InterventionHead(nn.Module):
    """Shared node encoder, baseline comparison, and two bounded gain outputs."""
    def __init__(self, hidden_dim=128):
        super().__init__()
        if type(hidden_dim) is not int or not 1 <= hidden_dim <= 128:
            raise ValueError("hidden width must be in [1, 128]")
        self.node_encoder = nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU())
        self.context_encoder = nn.Sequential(nn.Linear(1536, hidden_dim), nn.ReLU())
        self.comparison = nn.Sequential(nn.Linear(hidden_dim * 4 + len(SCALAR_NAMES), hidden_dim),
                                        nn.ReLU(), nn.Linear(hidden_dim, 2))
        # An untrained head preserves the baseline exactly.
        nn.init.zeros_(self.comparison[-1].weight)
        nn.init.zeros_(self.comparison[-1].bias)

    def forward(self, batch: InterventionBatch) -> Tensor:
        nodes, context, scalars = batch.node_features, batch.terminal_context, batch.scalar_features
        mask, anchors = batch.valid_mask, batch.baseline_indices
        b, n, _ = nodes.shape
        if (nodes.shape != (b, n, 768) or context.shape != (b, 1536)
                or scalars.shape != (b, n, len(SCALAR_NAMES)) or b == 0 or n == 0
                or mask.shape != (b, n) or mask.dtype != torch.bool
                or anchors.shape != (b,) or anchors.dtype != torch.int64
                or (anchors < 0).any() or (anchors >= n).any()
                or any(t.device != nodes.device for t in (context, scalars, mask, anchors))
                or nodes.device != self.comparison[-1].weight.device
                or not mask[torch.arange(b, device=nodes.device), anchors].all()
                or not torch.isfinite(nodes[mask]).all() or not torch.isfinite(scalars[mask]).all()
                or not torch.isfinite(context).all()):
            raise ValueError("invalid intervention batch")
        # Padding is excluded before the MLP, including NaNs in padded storage.
        dtype = self.comparison[-1].weight.dtype
        h = self.node_encoder(nodes.detach().masked_fill(~mask[..., None], 0).to(dtype))
        anchor = h[torch.arange(b, device=h.device), anchors][:, None].expand_as(h)
        c = self.context_encoder(context.detach().to(dtype))[:, None].expand_as(h)
        x = torch.cat((h, anchor, h - anchor, c,
                       scalars.detach().masked_fill(~mask[..., None], 0).to(dtype)), -1)
        prediction = self.comparison(x).tanh()
        anchor_mask = torch.arange(n, device=h.device)[None] == anchors[:, None]
        return prediction.masked_fill((~mask | anchor_mask)[..., None], 0)


def select_intervention(predicted_gains: Tensor, candidate_vpids: Sequence[str],
                        baseline_vpid: str, *, valid_mask: Tensor | None = None) -> str:
    """Keep DUET unless one alternative predicts nonnegative SR and positive SPL.

    All ties at the best predicted SPL return the original baseline endpoint.
    Values are fractions, not percentages. This is a predicted-gain rule only.
    """
    ids = _ids(candidate_vpids, "candidate IDs")
    if (baseline_vpid not in ids or not isinstance(predicted_gains, Tensor)
            or not predicted_gains.is_floating_point() or predicted_gains.ndim != 2
            or predicted_gains.shape[1] != 2 or predicted_gains.shape[0] < len(ids)):
        raise ValueError("invalid endpoint gain predictions")
    n = predicted_gains.shape[0]
    if valid_mask is None:
        valid_mask = torch.arange(n, device=predicted_gains.device) < len(ids)
    if (valid_mask.shape != (n,) or valid_mask.dtype != torch.bool
            or valid_mask.device != predicted_gains.device
            or bool(valid_mask[len(ids):].any())
            or not bool(valid_mask[ids.index(baseline_vpid)])
            or not torch.isfinite(predicted_gains[valid_mask]).all()
            or not torch.equal(predicted_gains[ids.index(baseline_vpid)],
                               torch.zeros(2, device=predicted_gains.device, dtype=predicted_gains.dtype))):
        raise ValueError("invalid candidate mask or nonzero baseline prediction")
    eligible = valid_mask & (predicted_gains[:, 0] >= 0) & (predicted_gains[:, 1] > 0)
    if not bool(eligible.any()):
        return baseline_vpid
    gains = predicted_gains[:, 1].masked_fill(~eligible, -torch.inf)
    winners = torch.nonzero(gains == gains.max()).flatten()
    return ids[int(winners.item())] if len(winners) == 1 else baseline_vpid


def build_intervention_targets(*, candidate_goal_distances_m: Tensor,
                               candidate_total_lengths_m: Tensor,
                               reference_path_length_m: float,
                               baseline_index: int) -> Tensor:
    """Training-only exact per-candidate [delta SR, delta SPL], in fractions.

    Total lengths include the unchanged online prefix plus each actually
    executed discovered-graph return route, measured on the evaluation graph.
    ``reference_path_length_m`` is the evaluator's sum of shortest distances
    between consecutive reference-path nodes, NOT start-to-goal distance.
    The caller authenticates these separately stored supervision fields.
    """
    d, lengths = candidate_goal_distances_m, candidate_total_lengths_m
    ref = _number(reference_path_length_m, "reference path length")
    if (not isinstance(d, Tensor) or not isinstance(lengths, Tensor)
            or d.ndim != 1 or d.numel() == 0 or lengths.shape != d.shape
            or not d.is_floating_point() or not lengths.is_floating_point()
            or d.device != lengths.device or not torch.isfinite(d).all()
            or not torch.isfinite(lengths).all() or (d < 0).any() or (lengths < 0).any()
            or type(baseline_index) is not int or not 0 <= baseline_index < len(d)):
        raise ValueError("invalid training-only intervention targets")
    # FP64 preserves the graph metric convention before the training loss casts.
    success = (d.detach().double() < 3.0).double()
    spl = success * ref / lengths.detach().double().clamp_min(max(ref, 0.01))
    metrics = torch.stack((success, spl), -1)
    delta = metrics - metrics[baseline_index]
    if not torch.equal(delta[baseline_index], torch.zeros_like(delta[baseline_index])):
        raise ValueError("baseline target is not exactly the reference metric")
    return delta


def intervention_loss(predicted_gains: Tensor, target_gains: Tensor, valid_mask: Tensor,
                      baseline_indices: Tensor, *, risk_weight: float = 0.0) -> dict[str, Tensor]:
    """Episode-normalized regression; optional penalty for predicted harmful gains.

    The optional term penalizes positive predictions where that metric's true
    delta is negative, weighted by the observed harm. It is a hypothesis for
    later matched ablations, disabled by default; it supplies no risk bound.
    """
    risk_weight = _number(risk_weight, "risk weight")
    p, y, mask, anchors = predicted_gains, target_gains, valid_mask, baseline_indices
    if (p.ndim != 3 or p.shape[-1] != 2 or p.shape[0] == 0 or p.shape[1] == 0 or y.shape != p.shape
            or mask.shape != p.shape[:2] or mask.dtype != torch.bool
            or anchors.shape != p.shape[:1] or anchors.dtype != torch.int64
            or any(t.device != p.device for t in (y, mask, anchors))
            or not mask.any(-1).all() or (anchors < 0).any() or (anchors >= p.shape[1]).any()
            or not mask[torch.arange(len(p), device=p.device), anchors].all()
            or not torch.isfinite(p[mask]).all() or not torch.isfinite(y[mask]).all()
            or not (y[torch.arange(len(p), device=p.device), anchors] == 0).all()
            or not (p[torch.arange(len(p), device=p.device), anchors] == 0).all()
            or (y[mask].abs() > 1).any()):
        raise ValueError("invalid intervention loss inputs")
    # Do not let padded NaNs or constant zero anchors dilute an episode's loss.
    eligible = mask & (torch.arange(p.shape[1], device=p.device)[None] != anchors[:, None])
    p = p.masked_fill(~eligible[..., None], 0)
    y = y.detach().to(p.dtype).masked_fill(~eligible[..., None], 0)
    pointwise = F.smooth_l1_loss(p, y, reduction="none").mean(-1)
    false_gain = (p.clamp_min(0).square() * (-y).clamp_min(0)).mean(-1)
    counts = eligible.sum(-1).clamp_min(1)
    regression = ((pointwise * eligible).sum(-1) / counts).mean()
    risk = ((false_gain * eligible).sum(-1) / counts).mean()
    return {"loss": regression + risk_weight * risk, "regression": regression,
            "false_gain_penalty": risk}
