"""Matched endpoint objectives over complete source pairs and real prefixes.

The data loaders authenticate the collections. This module only assembles head
inputs from frozen features; goal membership and goal indices are supervision.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .endpoint_probe import FEATURE_DIM

ARMS = ("C1", "C2", "C3", "M")
ORDERS = ("A_then_B", "B_then_A")
SLOTS = ("A", "B")
MAX_STATES = 15
PAIR_WEIGHT = 0.1


def _run_fields(run, *, paired_slot=None):
    features = run["features"]
    labels = run["labels"]["within_success_radius"]
    if paired_slot is not None:
        if (run["instruction_slot"] != paired_slot or labels.ndim != 2
                or labels.shape[1] != 2):
            raise ValueError("paired instruction/label-column mismatch")
        labels = labels[:, paired_slot]
    if (not isinstance(features, Tensor) or not isinstance(labels, Tensor)
            or features.dtype != torch.float32 or labels.dtype != torch.bool
            or features.ndim != 2 or features.shape[1] != FEATURE_DIM
            or not 1 <= len(features) <= MAX_STATES or labels.shape != (len(features),)
            or not torch.isfinite(features).all()):
        raise ValueError("invalid group rollout features/labels")
    return features.detach(), labels


def prepare_group_batch(groups, arm, device="cpu"):
    """Build [groups,6,15,1536], retaining masks for every true decision state.

    Slots 0/1 are the common natural histories. C1 reuses these in its four
    augmentation slots; C3/M use order-major, instruction-minor paired histories.
    Padding matches head row counts and never counts as an observation or loss.
    """
    if arm not in ARMS or not groups:
        raise ValueError("expected a supported arm and a nonempty group batch")
    count = len(groups)
    features = torch.zeros(count, 6, MAX_STATES, FEATURE_DIM, dtype=torch.float32, device=device)
    labels = torch.zeros(count, 6, MAX_STATES, dtype=torch.float32, device=device)
    mask = torch.zeros(count, 6, MAX_STATES, dtype=torch.bool, device=device)
    goal_steps = torch.full((count, 2, 2), -1, dtype=torch.int64, device=device)
    for g, group in enumerate(groups):
        natural = [group["natural"][slot] for slot in SLOTS]
        runs = [(r, None) for r in natural]
        if arm == "C1":
            runs += [(r, None) for r in (natural[0], natural[0], natural[1], natural[1])]
        elif arm == "C2":
            runs += [(group["c2"][slot][context], None)
                     for slot in SLOTS for context in ("reference", "overshoot")]
        else:
            for o, order in enumerate(ORDERS):
                pair_runs = group["paired"][order]
                viewpoints = [[s["viewpoint"] for s in pair_runs[slot]["states"]] for slot in SLOTS]
                if viewpoints[0] != viewpoints[1] or len(set(viewpoints[0])) != len(viewpoints[0]):
                    raise ValueError("paired physical observation order differs")
                goals = group["pair"]["goal_vpids"]
                if len(goals) != 2 or goals[0] == goals[1] or not set(goals).issubset(viewpoints[0]):
                    raise ValueError("both original goals must be observed exactly once")
                indices = [viewpoints[0].index(v) for v in goals]
                goal_steps[g, o] = torch.tensor(indices, dtype=torch.int64, device=device)
                for i, slot in enumerate(SLOTS):
                    run = pair_runs[slot]
                    if run["instr_id"] != group["pair"]["instr_ids"][i]:
                        raise ValueError("paired original instruction association differs")
                    _, target = _run_fields(run, paired_slot=i)
                    if not bool(target[indices[i]]) or bool(target[indices[1-i]]):
                        raise ValueError("paired goals do not have the required label flip")
                    runs.append((run, i))
        for slot, (run, instruction) in enumerate(runs):
            x, y = _run_fields(run, paired_slot=instruction)
            n = len(x)
            features[g, slot, :n] = x.to(device)
            labels[g, slot, :n] = y.to(device, dtype=torch.float32)
            mask[g, slot, :n] = True
    return {"features": features, "labels": labels, "mask": mask,
            "goal_steps": goal_steps, "arm": arm}


def score_group_batch(head, batch):
    x = batch["features"]
    return head(x.reshape(-1, FEATURE_DIM)).reshape(x.shape[:3])


def per_group_metrics(logits, batch):
    """Differentiable per-group losses plus paired-order correctness diagnostics."""
    labels, mask = batch["labels"], batch["mask"]
    if (logits.shape != labels.shape or mask.shape != labels.shape or logits.ndim != 3
            or logits.shape[1:] != (6, MAX_STATES) or not torch.isfinite(logits).all()
            or mask.dtype != torch.bool or not mask.any(dim=-1).all()):
        raise ValueError("invalid padded group scores/masks")
    state_losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    episode_bce = (state_losses * mask).sum(-1) / mask.sum(-1)
    natural = episode_bce[:, :2].mean(-1)
    augmentation = episode_bce[:, 2:].mean(-1)
    bce = 0.5 * (natural + augmentation)
    ranking = torch.zeros_like(bce)
    correct_orders = torch.zeros((len(logits), 2), dtype=torch.bool, device=logits.device)
    if batch["arm"] in {"C3", "M"}:
        indices = batch["goal_steps"]
        if indices.shape != (len(logits), 2, 2) or (indices < 0).any() or (indices >= MAX_STATES).any():
            raise ValueError("invalid paired goal indices")
        group_indices = torch.arange(len(logits), device=logits.device)
        order_losses = []
        for o in range(2):
            a, b = indices[:, o, 0], indices[:, o, 1]
            slot_a, slot_b = 2 + 2 * o, 3 + 2 * o
            if not all(mask[group_indices, slot, t].all() for slot in (slot_a, slot_b) for t in (a, b)):
                raise ValueError("goal index refers to padding")
            aa, ab = logits[group_indices, slot_a, a], logits[group_indices, slot_a, b]
            ba, bb = logits[group_indices, slot_b, a], logits[group_indices, slot_b, b]
            order_losses.append(0.5 * (F.softplus(ab-aa) + F.softplus(ba-bb)))
            correct_orders[:, o] = (aa > ab) & (bb > ba)
        ranking = torch.stack(order_losses, dim=-1).mean(-1)
    return {"natural_bce": natural, "augmentation_bce": augmentation, "bce": bce,
            "ranking": ranking, "both_instructions_correct_by_order": correct_orders,
            "both_orders_correct": correct_orders.all(-1)}


def group_objective(logits, batch, arm):
    if arm not in ARMS or arm != batch["arm"]:
        raise ValueError("loss arm differs from assembled data")
    values = per_group_metrics(logits, batch)
    bce, ranking = values["bce"].mean(), values["ranking"].mean()
    loss = bce + PAIR_WEIGHT * ranking if arm == "M" else bce
    return {"loss": loss, "bce": bce, "ranking": ranking}
