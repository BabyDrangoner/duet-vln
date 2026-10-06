"""Offline interventions on frozen navigation states; never a policy adapter.

Future panoramas are experimental inputs here, not deployable observations or
reliability ground truth. Callers must keep the resulting labels out of policy
inputs. All interventions clone the state and rerun the whole navigation model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Callable

import torch


NAV_KEYS = {
    "txt_embeds", "txt_masks", "gmap_img_embeds", "gmap_step_ids", "gmap_pos_fts",
    "gmap_masks", "gmap_pair_dists", "gmap_visited_masks", "gmap_vpids",
    "vp_img_embeds", "vp_pos_fts", "vp_masks", "vp_nav_masks", "vp_obj_masks", "vp_cand_vpids",
}


def clone_navigation(inputs: dict, device=None) -> dict:
    if set(inputs) != NAV_KEYS:
        raise ValueError(f"navigation input whitelist mismatch: {set(inputs) ^ NAV_KEYS}")
    result = {}
    for key, value in inputs.items():
        result[key] = value.detach().to(device=device or value.device).clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
    if result["gmap_img_embeds"].ndim != 3 or result["gmap_img_embeds"].shape[0] != 1:
        raise ValueError("offline replay requires an individual batch=1 state")
    return result


def navigation_hash(inputs: dict) -> str:
    """Content identity including dtypes/shapes/IDs; no labels are accepted."""
    inputs = clone_navigation(inputs, "cpu")
    digest = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def legal_mask(inputs: dict) -> torch.Tensor:
    mask = inputs["gmap_masks"] & ~inputs["gmap_visited_masks"]
    if mask.dtype != torch.bool or mask.shape[0] != 1 or not mask[0, 0]:
        raise ValueError("invalid legal navigation mask")
    return mask[0]


def replace_candidate(inputs: dict, candidate_index: int, feature: torch.Tensor) -> dict:
    result = clone_navigation(inputs)
    mask = legal_mask(result)
    if type(candidate_index) is not int or not 0 < candidate_index < len(mask) or not mask[candidate_index]:
        raise ValueError("replacement requires a legal non-STOP candidate")
    reference = result["gmap_img_embeds"][0, candidate_index]
    if not isinstance(feature, torch.Tensor) or feature.shape != reference.shape or not torch.isfinite(feature).all():
        raise ValueError("replacement must be a finite feature vector of matching shape")
    result["gmap_img_embeds"][0, candidate_index] = feature.to(reference)
    return result


def candidate_margin(logits: torch.Tensor, valid: torch.Tensor, index: int) -> float:
    if logits.ndim != 1 or logits.shape != valid.shape or not valid[index]:
        raise ValueError("margin requires a legal candidate with matching logits")
    others = valid.clone()
    others[index] = False
    if not others.any() or not torch.isfinite(logits[valid]).all():
        raise ValueError("margin needs finite scores and a competing legal action")
    return float((logits[index] - torch.logsumexp(logits[others], 0)).item())


def _scores(nav_fn: Callable, inputs: dict) -> torch.Tensor:
    with torch.no_grad():
        scores = nav_fn(inputs)["fused_logits"].detach().float()
    valid = legal_mask(inputs)
    if scores.shape != (1, len(valid)) or not torch.isfinite(scores[0, valid]).all():
        raise ValueError("replayed model returned invalid scores")
    return scores[0].masked_fill(~valid, -torch.inf)


def replay_identity(nav_fn: Callable, inputs: dict, expected_logits: torch.Tensor) -> dict:
    before = navigation_hash(inputs)
    scores = _scores(nav_fn, clone_navigation(inputs))
    valid = legal_mask(inputs)
    expected = expected_logits.to(scores).reshape_as(scores)
    if not torch.isfinite(expected[valid]).all():
        raise ValueError("recorded legal logits are not finite")
    torch.testing.assert_close(scores[valid], expected[valid], atol=1e-6, rtol=1e-5)
    if int(scores.argmax()) != int(expected.masked_fill(~valid, -torch.inf).argmax()):
        raise ValueError("replay changed the action ordering")
    if navigation_hash(inputs) != before:
        raise ValueError("identity replay mutated the recorded policy state")
    return {"logits": scores, "max_absolute_error": float((scores[valid] - expected[valid]).abs().max()),
            "navigation_sha256": before}


def evaluate_replacement(nav_fn: Callable, inputs: dict, baseline_logits: torch.Tensor,
                         candidate_index: int, replacement: torch.Tensor) -> dict:
    """An offline score perturbation, not an estimate of navigation improvement."""
    before = navigation_hash(inputs)
    alternate = replace_candidate(inputs, candidate_index, replacement)
    changed = _scores(nav_fn, alternate)
    valid = legal_mask(inputs)
    baseline_logits = baseline_logits.to(changed)
    result = {"delta_margin": candidate_margin(changed, valid, candidate_index)
              - candidate_margin(baseline_logits, valid, candidate_index),
              "argmax": int(changed.argmax()),
              "argmax_changed": int(changed.argmax()) != int(baseline_logits.argmax()),
              "stop_logit_shift": float(changed[0] - baseline_logits[0]),
              "logits": [float(value) if bool(valid[i]) else None for i, value in enumerate(changed)],
              "replacement_norm": float(replacement.float().norm()),
              "original_norm": float(inputs["gmap_img_embeds"][0, candidate_index].float().norm())}
    if navigation_hash(inputs) != before:
        raise ValueError("counterfactual mutated an earlier policy state")
    return result


def observable_features(state: dict, candidate_index: int) -> tuple[dict, dict]:
    """Return score/geometry (P0) and additional current-source (P1) features.

    No arrival, target, oracle, episode outcome, or label argument is accepted.
    Association IDs are used only for joining this candidate's observed sources.
    """
    inputs = state["nav_inputs"]
    valid = legal_mask(inputs).cpu()
    logits = state["base_logits"].detach().cpu().float().reshape(-1).masked_fill(~valid, -torch.inf)
    if not valid[candidate_index] or candidate_index == 0:
        raise ValueError("features require a legal non-STOP candidate")
    target = inputs["gmap_vpids"][0][candidate_index]
    snapshot = state["candidate_evidence"][target]
    if snapshot.get("association", {}).get("target_id") != target:
        raise ValueError("candidate evidence belongs to a different target")
    sources = snapshot["sources"]
    if not sources:
        raise ValueError("candidate has no recorded source observation")
    if snapshot.get("status") != "unvisited" or snapshot["available_at_step"] > state["step"]:
        raise ValueError("candidate evidence was unavailable at the recorded decision")
    if any(not (0 <= s["first_step"] <= s["last_step"] <= state["step"])
           or not (s["first_step"] <= s["feature_step"] <= s["last_step"]) for s in sources):
        raise ValueError("future source observation cannot be used by an observable probe")
    probabilities = torch.log_softmax(logits, 0)
    positions = inputs["gmap_pos_fts"][0, candidate_index].cpu().float()
    distances = torch.tensor([math.sqrt(sum(float(v) ** 2 for v in s["relative_position"])) for s in sources])
    headings = torch.tensor([s["heading"] for s in sources])
    elevations = torch.tensor([s["elevation"] for s in sources])
    global_scores = state["base_global_logits"].detach().cpu().float().reshape(-1).masked_fill(~valid, -torch.inf)
    local_scores = state["base_local_logits"].detach().cpu().float().reshape(-1)
    local_valid = inputs["vp_nav_masks"][0].cpu()
    local_ids = inputs["vp_cand_vpids"][0]
    if not torch.isfinite(global_scores[valid]).all() or not torch.isfinite(local_scores[local_valid]).all():
        raise ValueError("recorded branch scores are not finite on legal actions")
    global_logp = torch.log_softmax(global_scores, 0)
    local_logp = torch.log_softmax(local_scores.masked_fill(~local_valid, -torch.inf), 0)
    local_index = local_ids.index(target) if target in local_ids else None
    ranked = logits[valid].sort(descending=True).values
    p0 = {
        "candidate_log_probability": float(probabilities[candidate_index]),
        "candidate_margin": candidate_margin(logits, valid, candidate_index),
        "candidate_rank": int((logits[valid] > logits[candidate_index]).sum()),
        "stop_log_probability": float(probabilities[0]),
        "global_candidate_log_probability": float(global_logp[candidate_index]),
        "local_candidate_log_probability": float(local_logp[local_index]) if local_index is not None else 0.0,
        "global_stop_log_probability": float(global_logp[0]),
        "local_stop_log_probability": float(local_logp[0]),
        "policy_entropy": float(-(probabilities[valid].exp() * probabilities[valid]).sum()),
        "top_two_gap": float(ranked[0] - ranked[1]),
        "candidate_is_argmax": float(candidate_index == int(logits.argmax())),
        "legal_actions": int(valid.sum()), "step": state["step"],
        "candidate_is_local": float(target in inputs["vp_cand_vpids"][0]),
        "discovery_age": state["step"] - min(s["first_step"] for s in sources),
        "source_distance_min": float(distances.min()), "source_distance_mean": float(distances.mean()),
        "source_heading_cos_mean": float(headings.cos().mean()),
        "source_heading_sin_mean": float(headings.sin().mean()),
        "source_elevation_mean": float(elevations.mean()),
        **{f"coverage_{key}": int(value) for key, value in snapshot["counts"].items()},
        **{f"position_{i}": float(value) for i, value in enumerate(positions)},
    }
    features = torch.stack([s["feature"].float() for s in sources])
    centered = features - features.mean(0)
    normalized = torch.nn.functional.normalize(features, dim=1)
    pairs = torch.triu_indices(len(sources), len(sources), offset=1)
    cosines = (normalized @ normalized.T)[pairs[0], pairs[1]]
    p1 = {
        "source_feature_variance": float(centered.square().sum(1).mean()),
        "source_cosine_mean": float(cosines.mean()) if len(cosines) else 1.0,
        "source_cosine_min": float(cosines.min()) if len(cosines) else 1.0,
        "source_norm_mean": float(features.norm(dim=1).mean()),
        "source_norm_std": float(features.norm(dim=1).std(unbiased=False)),
    }
    if not all(math.isfinite(float(v)) for v in (*p0.values(), *p1.values())):
        raise ValueError("observable feature contains a non-finite value")
    return p0, p1
