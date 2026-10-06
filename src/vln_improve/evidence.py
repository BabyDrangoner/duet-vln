"""Episode-local collection of pre-instruction candidate evidence and arrivals.

DUET's candidate ``pano_embeds`` are contextualized panorama embeddings before
instruction interaction, not raw directional image features. This collector
does not change DUET's graph averages, logits, masks, or chosen actions.

Only the first ``max_sources`` distinct sources keep feature tensors. Later
sources keep lightweight fingerprints/counters, so coverage is not truncated
to the tensor capacity. Within a retained source, the latest changed observation
replaces its slot; an identical repeat only updates timestamps/counters.

Node IDs serve association and deduplication. ``model_inputs`` excludes all
IDs. Future arrival features exist only in explicitly separate training pairs;
the collector never computes a visual-distance reliability label.
"""

from __future__ import annotations

import copy
import hashlib
import math
from numbers import Real
import struct
from typing import Any, Sequence

import torch


EVIDENCE_SCHEMA = "duet_preinstruction_source_evidence_v1"
COUNT_FIELDS = (
    "source_count_total", "stored_count", "observation_count", "duplicate_count",
    "changed_count", "overflow_count", "duplicate_event_count",
)


def _copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return copy.deepcopy(value)


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_FIELDS}


class EpisodeEvidenceMemory:
    """Collect one episode; construct a fresh instance for every batch slot reset.

    ``step`` is an integer navigation decision index. ``heading``/``elevation``
    are source-relative angles in radians; ``relative_position`` is the known
    target-minus-source displacement in metres, in a consistent frame. The
    caller computes these from already observed candidates, never hidden maps.

    Call ``arrive`` only when a panorama was actually observed at that node.
    DUET may traverse intermediate graph nodes without encoding their panoramas;
    those nodes must not receive fabricated arrival observations.
    """

    def __init__(self, episode_id: str, *, scan_id: str, instr_id: str,
                 max_sources: int = 4, feature_dim: int | None = None) -> None:
        self.episode_id = _identifier(episode_id, "episode_id")
        self.scan_id = _identifier(scan_id, "scan_id")
        self.instr_id = _identifier(instr_id, "instr_id")
        if type(max_sources) is not int or max_sources < 1:
            raise ValueError("max_sources must be a positive integer")
        if feature_dim is not None and (type(feature_dim) is not int or feature_dim < 1):
            raise ValueError("feature_dim must be null or a positive integer")
        self.max_sources, self.feature_dim = max_sources, feature_dim
        self._last_step = -1
        self._targets: dict[str, dict[str, Any]] = {}
        self._visited: set[str] = set()
        self._closed_counts: dict[str, dict[str, int]] = {}
        self._pairs: list[dict[str, Any]] = []
        self._num_pairs = 0
        self._arrival_without_proxy = 0
        self._repeat_arrivals = 0
        self._ignored_visited = 0

    def _step(self, step: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if step < self._last_step:
            raise ValueError("episode observation steps must be nondecreasing")

    def _feature(self, feature: torch.Tensor) -> torch.Tensor:
        if not isinstance(feature, torch.Tensor) or not feature.is_floating_point():
            raise ValueError("feature must be a floating-point tensor")
        if feature.ndim != 1 or feature.numel() == 0 or not torch.isfinite(feature).all():
            raise ValueError("feature must be a finite nonempty vector")
        if self.feature_dim is not None and feature.numel() != self.feature_dim:
            raise ValueError("feature dimension changed within the episode")
        # Explicit float32 is a collection format, not an upstream model update.
        result = feature.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
        if not torch.isfinite(result).all():
            raise ValueError("feature is outside the collection float32 range")
        return result

    def _association(self, target_id: str) -> dict[str, str]:
        return {"episode_id": self.episode_id, "scan_id": self.scan_id,
                "instr_id": self.instr_id, "target_id": target_id}

    def observe_proxy(self, target_id: str, source_id: str, *, step: int,
                      heading: float, elevation: float,
                      relative_position: Sequence[float], feature: torch.Tensor) -> str:
        """Record available evidence; return added/repeated/updated/overflow status.

        A repeated call for the same source at the same step is an instrumentation
        duplicate, and does not increase observation_count. Conflicting values
        for that same event are rejected. A revisit on a later step is a distinct
        observation; it can repeat or change the source's last representation.
        """
        target_id = _identifier(target_id, "target_id")
        source_id = _identifier(source_id, "source_id")
        if source_id == target_id:
            raise ValueError("a proxy target must differ from its observed source")
        self._step(step)
        if target_id in self._visited:
            self._last_step = step
            self._ignored_visited += 1
            return "ignored_visited"
        heading, elevation = _number(heading, "heading"), _number(elevation, "elevation")
        if isinstance(relative_position, torch.Tensor):
            relative_position = relative_position.detach().cpu().tolist()
        if isinstance(relative_position, (str, bytes, dict)):
            raise ValueError("relative_position must contain three coordinates")
        try:
            coordinates = list(relative_position)
        except TypeError as error:
            raise ValueError("relative_position must contain three coordinates") from error
        if len(coordinates) != 3:
            raise ValueError("relative_position must contain three coordinates")
        position = [_number(value, "relative_position") for value in coordinates]
        vector = self._feature(feature)
        fingerprint = hashlib.sha256(
            vector.numpy().tobytes() + struct.pack("!5d", heading, elevation, *position)
        ).hexdigest()
        target = self._targets.get(target_id)
        if target is None:
            target = {"counts": _counts(), "sources": {}, "stored": {}}
        previous = target["sources"].get(source_id)
        if previous is not None and step == previous["last_step"]:
            if fingerprint != previous["fingerprint"]:
                raise ValueError("conflicting observations for one source at the same step")
            target["counts"]["duplicate_event_count"] += 1
            self._last_step = step
            return "duplicate_event"
        self.feature_dim = vector.numel()
        self._last_step = step
        self._targets[target_id] = target
        counts = target["counts"]
        counts["observation_count"] += 1
        new_source = previous is None
        changed = previous is not None and previous["fingerprint"] != fingerprint
        repeated = previous is not None and not changed
        if new_source:
            counts["source_count_total"] += 1
            source = {"first_step": step, "last_step": step, "feature_step": step,
                      "observation_count": 1, "duplicate_count": 0, "changed_count": 0,
                      "fingerprint": fingerprint}
            target["sources"][source_id] = source
        else:
            source = previous
            source["last_step"] = step
            source["observation_count"] += 1
            source["fingerprint"] = fingerprint
            if repeated:
                source["duplicate_count"] += 1
                counts["duplicate_count"] += 1
            if changed:
                source["feature_step"] = step
                source["changed_count"] += 1
                counts["changed_count"] += 1
        retained = source_id in target["stored"]
        if new_source and len(target["stored"]) < self.max_sources:
            retained = True
        if retained and (new_source or changed):
            target["stored"][source_id] = {
                "feature": vector, "heading": heading, "elevation": elevation,
                "relative_position": position,
            }
        counts["stored_count"] = len(target["stored"])
        if not retained:
            counts["overflow_count"] += 1
            return "overflow"
        return "added" if new_source else ("updated" if changed else "repeated")

    def policy_snapshot(self, target_id: str) -> dict[str, Any]:
        """Copy currently available proxies, never arrival labels or hidden IDs.

        The association fields are for audit/joining only. Use model_inputs for
        the numeric feature interface. Unknown and already observed targets have
        no unvisited proxy entries; old snapshots remain unchanged after arrival.
        """
        target_id = _identifier(target_id, "target_id")
        target = self._targets.get(target_id)
        sources = []
        if target is not None:
            for source_id, evidence in target["stored"].items():
                metadata = target["sources"][source_id]
                sources.append({"source_id": source_id,
                                **{key: value for key, value in metadata.items() if key != "fingerprint"},
                                **_copy(evidence)})
        return {
            "schema_version": 1, "feature_schema": EVIDENCE_SCHEMA,
            "association": self._association(target_id), "available_at_step": self._last_step,
            "status": "visited" if target_id in self._visited else ("unvisited" if target else "unknown"),
            "capacity_policy": "first_distinct_sources_latest_observation",
            "max_sources": self.max_sources, "feature_dim": self.feature_dim,
            "counts": dict(target["counts"]) if target is not None else _counts(),
            "sources": sources,
        }

    def model_inputs(self, target_id: str) -> dict[str, torch.Tensor]:
        """Return ID-free CPU tensors; future observation fields are absent."""
        snapshot = self.policy_snapshot(target_id)
        sources = snapshot["sources"]
        count = len(sources)
        return {
            "features": torch.stack([value["feature"] for value in sources]) if count
            else torch.empty((0, self.feature_dim or 0), dtype=torch.float32),
            "angles": torch.tensor([[value["heading"], value["elevation"]] for value in sources], dtype=torch.float32).reshape(count, 2),
            "relative_positions": torch.tensor([value["relative_position"] for value in sources], dtype=torch.float32).reshape(count, 3),
            "steps": torch.tensor([[value["first_step"], value["last_step"], value["feature_step"]]
                                   for value in sources], dtype=torch.int64).reshape(count, 3),
            "counts": torch.tensor([snapshot["counts"][key] for key in COUNT_FIELDS], dtype=torch.int64),
        }

    def arrive(self, target_id: str, *, step: int, feature: torch.Tensor) -> dict[str, Any] | None:
        """Seal a first observed arrival into a separate diagnostic/training pair.

        No pair is fabricated for an unobserved target. Targets without earlier
        evidence return None; their arrivals are counted separately. Revisited
        targets never replace their first-arrival target or reopen proxy memory.
        """
        target_id = _identifier(target_id, "target_id")
        self._step(step)
        if target_id in self._visited:
            self._last_step = step
            self._repeat_arrivals += 1
            return None
        vector = self._feature(feature)
        target = self._targets.get(target_id)
        if target is not None and any(value["last_step"] >= step for value in target["sources"].values()):
            raise ValueError("arrival must be strictly later than every paired proxy observation")
        evidence = self.policy_snapshot(target_id) if target is not None else None
        self.feature_dim = vector.numel()
        self._last_step = step
        self._visited.add(target_id)
        if target is None:
            self._arrival_without_proxy += 1
            return None
        self._closed_counts[target_id] = dict(target["counts"])
        del self._targets[target_id]
        pair = {
            "schema_version": 1, "kind": "arrival_observation_pair",
            "feature_schema": EVIDENCE_SCHEMA, "association": self._association(target_id),
            "inference_snapshot": evidence,
            "training_only": {"arrival_step": step, "arrival_feature": vector,
                              "arrival_observed": True, "reliability_label": None},
        }
        self._pairs.append(pair)
        self._num_pairs += 1
        return _copy(pair)

    def drain_pairs(self) -> list[dict[str, Any]]:
        """Consume detached copies for a training/diagnostic writer, not a policy."""
        result = _copy(self._pairs)
        self._pairs.clear()
        return result

    def pending_targets(self) -> list[str]:
        return list(self._targets)

    def summary(self) -> dict[str, Any]:
        """Observation coverage, including overflow and unresolved targets."""
        targets = {key: {**value, "arrival_observed": True} for key, value in self._closed_counts.items()}
        targets.update({key: {**value["counts"], "arrival_observed": False}
                        for key, value in self._targets.items()})
        return {
            "episode_id": self.episode_id, "scan_id": self.scan_id, "instr_id": self.instr_id,
            "last_step": self._last_step, "num_pairs": self._num_pairs,
            "num_pending_targets": len(self._targets), "arrival_without_proxy": self._arrival_without_proxy,
            "repeat_arrivals": self._repeat_arrivals, "ignored_visited_proxies": self._ignored_visited,
            "targets": targets,
        }
