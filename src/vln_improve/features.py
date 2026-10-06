"""Inference-only features for residual scoring of DUET global actions."""

from collections.abc import Sequence

import torch
from torch import Tensor


FEATURE_SCHEMA = "duet_action_features_v1"


def _tensor(data: dict, key: str, ndim: int) -> Tensor:
    value = data[key]
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{key} must be a {ndim}-dimensional tensor")
    return value.detach()


def _ids(value, batch_size: int, width: int, mask: Tensor, name: str):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain one ID list per batch row")
    if len(value) != batch_size:
        raise ValueError(f"{name} batch length does not match the tensors")
    for row, ids in enumerate(value):
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise ValueError(f"{name}[{row}] must be an ID list")
        if not 1 <= len(ids) <= width:
            raise ValueError(f"{name}[{row}] length must be between 1 and {width}")
        if ids[0] is not None:
            raise ValueError(f"{name}[{row}] must have stop ID None at index 0")
        if any(not isinstance(vpid, str) or not vpid for vpid in ids[1:]):
            raise ValueError(f"{name}[{row}] must contain nonempty node IDs after stop")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}[{row}] contains duplicate candidate IDs")
        if bool(mask[row, len(ids):].any()):
            raise ValueError(f"{name}[{row}] has legal mask entries beyond its ID length")
    return value


def _log_probs(logits: Tensor, mask: Tensor, name: str) -> Tensor:
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"{name} has a row with no legal actions")
    if not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError(f"{name} must be finite for every legal action")
    normalized = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=1)
    return normalized.clamp(min=-30.0, max=0.0).masked_fill(~mask, 0.0)


def build_features(nav_inputs: dict, nav_outs: dict) -> tuple[Tensor, Tensor, Tensor]:
    """Return detached ``(features, fused_logits, valid_global_mask)``.

    Feature order is global embedding, ID-aligned local embedding, six scalars
    (global/fused/local log probability, local flag, stop flag, log1p action
    count), then global position features. Invalid feature rows are zero and
    their returned base logits are -inf. No observation or target is read.
    """
    global_embeds = _tensor(nav_outs, "gmap_embeds", 3)
    local_embeds = _tensor(nav_outs, "vp_embeds", 3)
    positions = _tensor(nav_inputs, "gmap_pos_fts", 3)
    global_logits = _tensor(nav_outs, "global_logits", 2)
    fused_logits = _tensor(nav_outs, "fused_logits", 2)
    local_logits = _tensor(nav_outs, "local_logits", 2)
    global_mask = _tensor(nav_inputs, "gmap_masks", 2)
    visited_mask = _tensor(nav_inputs, "gmap_visited_masks", 2)
    local_mask = _tensor(nav_inputs, "vp_nav_masks", 2)

    batch_size, width, hidden_size = global_embeds.shape
    if batch_size == 0 or width == 0 or hidden_size == 0:
        raise ValueError("gmap_embeds dimensions must be nonzero")
    local_width = local_embeds.shape[1]
    if local_embeds.shape != (batch_size, local_width, hidden_size):
        raise ValueError("vp_embeds must match gmap_embeds batch and hidden dimensions")
    if positions.shape[:2] != (batch_size, width):
        raise ValueError("gmap_pos_fts must match the global action dimensions")
    for name, value in (("global_logits", global_logits), ("fused_logits", fused_logits),
                        ("gmap_masks", global_mask), ("gmap_visited_masks", visited_mask)):
        if value.shape != (batch_size, width):
            raise ValueError(f"{name} must match the global action dimensions")
    for name, value in (("local_logits", local_logits), ("vp_nav_masks", local_mask)):
        if value.shape != (batch_size, local_width):
            raise ValueError(f"{name} must match the local action dimensions")
    tensors = (global_embeds, local_embeds, positions, global_logits, fused_logits,
               local_logits, global_mask, visited_mask, local_mask)
    if any(value.device != global_embeds.device for value in tensors):
        raise ValueError("All navigation tensors must be on the same device")
    if any(mask.dtype != torch.bool for mask in (global_mask, visited_mask, local_mask)):
        raise ValueError("Navigation masks must have boolean dtype")

    global_ids = _ids(nav_inputs["gmap_vpids"], batch_size, width, global_mask, "gmap_vpids")
    local_ids = _ids(nav_inputs["vp_cand_vpids"], batch_size, local_width, local_mask,
                     "vp_cand_vpids")
    valid = global_mask & ~visited_mask
    global_logits = global_logits.float()
    fused_logits = fused_logits.float()
    local_logits = local_logits.float()
    global_log_probs = _log_probs(global_logits, valid, "global_logits")
    fused_log_probs = _log_probs(fused_logits, valid, "fused_logits")
    local_log_probs = _log_probs(local_logits, local_mask, "local_logits")

    global_embeds = global_embeds.float()
    local_embeds = local_embeds.float()
    aligned_local = torch.zeros_like(global_embeds)
    aligned_log_probs = torch.zeros_like(fused_logits)
    is_local = torch.zeros_like(fused_logits)
    for row, ids in enumerate(global_ids):
        local_indices = {vpid: index for index, vpid in enumerate(local_ids[row])}
        for global_index, vpid in enumerate(ids):
            local_index = local_indices.get(vpid)
            if local_index is not None and bool(local_mask[row, local_index]):
                aligned_local[row, global_index] = local_embeds[row, local_index]
                aligned_log_probs[row, global_index] = local_log_probs[row, local_index]
                is_local[row, global_index] = 1.0
    is_stop = torch.zeros_like(fused_logits)
    is_stop[:, 0] = 1.0
    action_count = torch.log1p(valid.sum(dim=1).float()).unsqueeze(1).expand_as(fused_logits)
    scalars = torch.stack((global_log_probs, fused_log_probs, aligned_log_probs,
                           is_local, is_stop, action_count), dim=-1)
    features = torch.cat((global_embeds, aligned_local, scalars, positions.float()), dim=-1)
    features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Embeddings and positions must be finite for legal global actions")
    base_logits = fused_logits.masked_fill(~valid, -torch.inf)
    return features, base_logits, valid
