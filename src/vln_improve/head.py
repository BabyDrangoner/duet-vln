"""A bounded action correction head; the supplied navigation features stay frozen."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


class ResidualActionHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, max_delta: float = 1.0):
        super().__init__()
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not math.isfinite(max_delta) or max_delta <= 0:
            raise ValueError("max_delta must be finite and positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_delta = float(max_delta)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "max_delta": self.max_delta,
        }

    def forward(self, features: Tensor, base_logits: Tensor, valid_mask: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape [B, N, {self.feature_dim}]")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("a batch needs at least one state and candidate")
        if base_logits.shape != features.shape[:2] or valid_mask.shape != base_logits.shape:
            raise ValueError("logits and mask must match features' [B, N] dimensions")
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must have boolean dtype")
        if not features.is_floating_point() or not base_logits.is_floating_point():
            raise ValueError("features and base_logits must be floating-point tensors")
        if features.device != base_logits.device or features.device != valid_mask.device:
            raise ValueError("features, logits, and mask must be on the same device")
        if not valid_mask.any(dim=-1).all():
            raise ValueError("every state must have at least one legal action")
        if not torch.isfinite(base_logits[valid_mask]).all():
            raise ValueError("valid base_logits must be finite")
        if not torch.isfinite(features[valid_mask]).all():
            raise ValueError("valid features must be finite")

        # Mask padding before the MLP: invalid cache entries need not be finite.
        frozen_features = features.detach().masked_fill(~valid_mask.unsqueeze(-1), 0)
        frozen_features = frozen_features.to(dtype=self.network[0].weight.dtype)
        delta = self.max_delta * torch.tanh(self.network(frozen_features).squeeze(-1))
        corrected = base_logits.detach().to(dtype=delta.dtype) + delta
        if not torch.isfinite(corrected[valid_mask]).all():
            raise ValueError("corrected valid logits must be finite")
        return corrected.masked_fill(~valid_mask, -torch.inf)


def save_head_checkpoint(
    head: ResidualActionHead,
    path: str | Path,
    *,
    provenance: Mapping[str, Any],
    train_args: Mapping[str, Any],
    seed: int,
    metrics: Any,
) -> dict[str, Any]:
    """Save plain metadata and CPU weights. Metrics describe training, not navigation."""
    state_dict = {name: value.detach().cpu() for name, value in head.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in state_dict.values()):
        raise ValueError("cannot save a head with non-finite parameters")
    metadata = {
        "schema_version": 1,
        "head_config": head.config,
        "provenance": dict(provenance),
        "train_args": dict(train_args),
        "seed": seed,
        "metrics": metrics,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "state_dict": state_dict}, destination)
    return metadata


def load_head_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_provenance: Mapping[str, Any] | None = None,
) -> tuple[ResidualActionHead, dict[str, Any]]:
    """Load with weights_only=True and optionally enforce the base/data identity."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema_version", "head_config", "state_dict", "provenance", "train_args", "seed", "metrics"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("invalid head checkpoint schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported head checkpoint schema_version")
    if not isinstance(payload["provenance"], dict):
        raise ValueError("checkpoint provenance must be a dictionary")
    if expected_provenance is not None and payload["provenance"] != dict(expected_provenance):
        raise ValueError("checkpoint provenance does not match the requested base/data")
    if not isinstance(payload["head_config"], dict):
        raise ValueError("invalid head_config")
    if set(payload["head_config"]) != {"feature_dim", "hidden_dim", "max_delta"}:
        raise ValueError("invalid head_config fields")
    head = ResidualActionHead(**payload["head_config"])
    weights = payload["state_dict"]
    if not isinstance(weights, dict) or not all(
        isinstance(value, Tensor) and torch.isfinite(value).all() for value in weights.values()
    ):
        raise ValueError("checkpoint state_dict must contain finite tensors")
    head.load_state_dict(weights, strict=True)
    head.to(device).eval()
    return head, {key: value for key, value in payload.items() if key != "state_dict"}
