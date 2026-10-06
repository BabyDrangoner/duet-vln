"""Small outcome comparators with an observation-only prediction interface.

Candidate zero is the action the frozen navigator would execute. Cached inputs
must be captured before replacing that action; histories contain only the prefix
available at that decision. Full-continuation outcomes enter ``record_loss`` only.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


FEATURE_DIM = 1549
MAX_CANDIDATES = 4
MAX_HISTORY_TOKENS = 29
MODES = ("relative", "absolute", "teacher")


class ContinuationComparator(nn.Module):
    """Score candidates using a shared encoder and a causal prefix GRU.

All modes have identical parameters. Relative mode subtracts candidate zero's
two raw scores; absolute mode returns those raw scores unchanged. Teacher mode
uses column zero as action logits. No mode reads outcomes during prediction.

``history=False`` retains the current STOP context (the last history token),
using the same GRU and parameter count as the full-history model. Callers must
construct the causal prefix: the model cannot infer timestamps from embeddings.
"""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 128,
                 mode: str = "relative", history: bool = True):
        super().__init__()
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if type(history) is not bool:
            raise ValueError("history must be boolean")
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.mode, self.history = mode, history
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                     nn.LayerNorm(hidden_dim), nn.GELU())
        self.history_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.comparison = nn.Sequential(nn.Linear(4 * hidden_dim + 4, hidden_dim),
                                        nn.GELU(), nn.Linear(hidden_dim, 2))
        # Equal initial scores keep candidate zero when ties prefer the baseline.
        nn.init.zeros_(self.comparison[-1].weight)
        nn.init.zeros_(self.comparison[-1].bias)

    @property
    def config(self) -> dict:
        return {"feature_dim": self.feature_dim, "hidden_dim": self.hidden_dim,
                "mode": self.mode, "history": self.history}

    def _inputs(self, features: Tensor, history_features: Tensor,
                progress: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        for name, value in (("features", features), ("history_features", history_features),
                            ("progress", progress)):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if (features.ndim != 2 or features.shape[1] != self.feature_dim
                or not 1 <= features.shape[0] <= MAX_CANDIDATES):
            raise ValueError(f"features must have shape [K, {self.feature_dim}], 1 <= K <= 4")
        if (history_features.ndim != 2 or history_features.shape[1] != self.feature_dim
                or not 1 <= history_features.shape[0] <= MAX_HISTORY_TOKENS):
            raise ValueError(f"history_features must have shape [T, {self.feature_dim}], 1 <= T <= 29")
        if progress.shape != (4,):
            raise ValueError("progress must have shape [4]")
        weight = self.encoder[0].weight
        # Frozen DUET inputs never acquire gradients. Cached CPU features and
        # online CUDA features use the same conversion and prediction path.
        return tuple(value.detach().to(device=weight.device, dtype=weight.dtype)
                     for value in (features, history_features, progress))

    def forward(self, features: Tensor, history_features: Tensor, progress: Tensor) -> Tensor:
        features, history_features, progress = self._inputs(features, history_features, progress)
        if not self.history:
            history_features = history_features[-1:]
        candidates = self.encoder(features)
        history_tokens = self.encoder(history_features).unsqueeze(0)
        _, hidden = self.history_encoder(history_tokens)
        baseline = candidates[0:1].expand_as(candidates)
        context = hidden[-1].expand_as(candidates)
        pair = torch.cat((candidates, baseline, candidates - baseline, context,
                          progress.unsqueeze(0).expand(len(candidates), -1)), dim=-1)
        raw = self.comparison(pair)
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("comparator produced non-finite scores")
        return raw - raw[0:1] if self.mode == "relative" else raw

    def score_record(self, record: Mapping) -> Tensor:
        """Read exactly the three policy input fields; labels are never inspected."""
        return self(record["features"], record["history_features"], record["progress"])

    def relative_scores(self, features: Tensor, history_features: Tensor, progress: Tensor) -> Tensor:
        """Return predicted SR/SPL gains for either outcome regression mode.

Teacher logits have no SR/SPL units and cannot use an outcome-gain threshold.
"""
        if self.mode == "teacher":
            raise ValueError("teacher logits are not predicted SR/SPL gains")
        scores = self(features, history_features, progress)
        return scores if self.mode == "relative" else scores - scores[0:1]


def _positive_weight(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def record_loss(model: ContinuationComparator, record: Mapping, *, sr_weight: float = 1.0,
                spl_weight: float = 1.0, rescue_weight: float = 1.0,
                harm_weight: float = 1.0) -> Tensor:
    """Return a scalar loss for one complete candidate set.

Outcome modes use Smooth L1 (beta=1) on SR/SPL fractions, with equal weights by
default. Optional rescue/harm weights multiply candidates according to actual
SR minus baseline SR; normalization uses the sum of the applied weights. The
absolute control uses the same weighting rule, inputs and parameter structure.
Teacher mode uses cross entropy on column zero; target -1 contributes zero.
Training code must persist any nondefault loss weights in experiment metadata.
"""
    weights = [_positive_weight(name, value) for name, value in
               (("sr_weight", sr_weight), ("spl_weight", spl_weight),
                ("rescue_weight", rescue_weight), ("harm_weight", harm_weight))]
    prediction = model.score_record(record)
    if model.mode == "teacher":
        target = record["teacher_target"]
        if type(target) is not int or not -1 <= target < len(prediction):
            raise ValueError("teacher_target must be -1 or a legal candidate index")
        if target == -1:
            return prediction.sum() * 0.0
        return F.cross_entropy(prediction[:, 0].unsqueeze(0),
                               torch.tensor([target], device=prediction.device))

    utilities = record["utilities"]
    if (not isinstance(utilities, Tensor) or not utilities.is_floating_point()
            or utilities.shape != prediction.shape or not bool(torch.isfinite(utilities).all())):
        raise ValueError("utilities must be finite floating-point [K, 2] outcomes")
    utilities = utilities.detach().to(prediction)
    if (not bool(((utilities[:, 0] == 0) | (utilities[:, 0] == 1)).all())
            or not bool(((utilities[:, 1] >= 0) & (utilities[:, 1] <= 1)).all())
            or not bool((utilities[:, 1] <= utilities[:, 0]).all())):
        raise ValueError("utilities require binary SR and SPL fractions with zero SPL on failure")
    gains = utilities - utilities[0:1]
    targets = gains if model.mode == "relative" else utilities
    candidate_weights = torch.ones(len(prediction), device=prediction.device, dtype=prediction.dtype)
    candidate_weights = torch.where(gains[:, 0] > 0, weights[2], candidate_weights)
    candidate_weights = torch.where(gains[:, 0] < 0, weights[3], candidate_weights)
    channel_weights = prediction.new_tensor(weights[:2])
    element_weights = candidate_weights[:, None] * channel_weights[None, :]
    errors = F.smooth_l1_loss(prediction, targets, reduction="none", beta=1.0)
    return (errors * element_weights).sum() / element_weights.sum()
