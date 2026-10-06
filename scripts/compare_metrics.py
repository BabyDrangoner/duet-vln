#!/usr/bin/env python3
"""Compare two paired VLN evaluations using only the Python standard library.

This accepts the project's normalized JSON schema, not arbitrary upstream output:
    {"metadata": { ... REQUIRED_PROTOCOL_FIELDS ... },
     "episodes": [{"instr_id": "1_0", "scan_id": "scan-A",
                   "success": 1, "spl": 0.75}, ...]}

Episode success and SPL must be fractions in [0, 1], not percentages. Metadata
identifies the shared evaluation protocol and frozen base checkpoint. Extra
episode/metadata fields are allowed but are not used in this comparison.

The point estimates are episode-weighted means. Bootstrap replicates sample
scenes with replacement and include every paired episode in each sampled scene.
Both methods use the same sampled scenes. Percentile intervals describe the
observed paired evaluations, not variation across training random seeds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any


REQUIRED_PROTOCOL_FIELDS = (
    "dataset",
    "split",
    "feature_id",
    "base_checkpoint_sha256",
    "protocol_sha256",
    "upstream_commit",
    "max_action_len",
    "feedback",
)


def _unit_interval(value: Any, location: str, *, allow_bool: bool = False) -> float:
    if isinstance(value, bool) and not allow_bool:
        raise ValueError(f"{location}: expected a number in [0, 1], got a boolean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{location}: expected a number in [0, 1]")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{location}: expected a finite number in [0, 1]")
    return float(value)


def validate_result(payload: Any, source: str) -> tuple[dict, dict]:
    """Validate a normalized result and index its episodes by instruction ID."""
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: top-level JSON must be an object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: metadata must be an object")
    for key in REQUIRED_PROTOCOL_FIELDS:
        if key not in metadata:
            raise ValueError(f"{source}: missing metadata.{key}")
        value = metadata[key]
        if key == "max_action_len":
            if type(value) is not int or value <= 0:
                raise ValueError(f"{source}: metadata.{key} must be a positive integer")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}: metadata.{key} must be a nonempty string")
    for field in ("base_checkpoint_sha256", "protocol_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", metadata[field]):
            raise ValueError(f"{source}: metadata.{field} must be a 64-digit hex digest")

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{source}: episodes must be a nonempty list")
    indexed = {}
    for index, episode in enumerate(episodes):
        location = f"{source}: episodes[{index}]"
        if not isinstance(episode, dict):
            raise ValueError(f"{location} must be an object")
        for field in ("instr_id", "scan_id", "success", "spl"):
            if field not in episode:
                raise ValueError(f"{location}: missing {field}")
        for field in ("instr_id", "scan_id"):
            if not isinstance(episode[field], str) or not episode[field].strip():
                raise ValueError(f"{location}.{field} must be a nonempty string")
        instr_id = episode["instr_id"]
        if instr_id in indexed:
            raise ValueError(f"{source}: duplicate instr_id {instr_id!r}")
        indexed[instr_id] = {
            "scan_id": episode["scan_id"],
            "success": _unit_interval(episode["success"], f"{location}.success", allow_bool=True),
            "spl": _unit_interval(episode["spl"], f"{location}.spl"),
        }
    return metadata, indexed


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linearly interpolate the requested percentile in an already sorted list."""
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def compare_results(
    baseline: Any,
    method: Any,
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict:
    """Return paired mean differences and scene-cluster bootstrap 95% CIs.

    Deltas and their confidence intervals are in percentage points, while the
    baseline and method means remain fractions. At least two scenes are required
    to estimate variation across scenes. The function does not select a winner.
    """
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    baseline_meta, baseline_episodes = validate_result(baseline, "baseline")
    method_meta, method_episodes = validate_result(method, "method")
    for field in REQUIRED_PROTOCOL_FIELDS:
        if baseline_meta[field] != method_meta[field]:
            raise ValueError(
                f"protocol mismatch for metadata.{field}: "
                f"baseline={baseline_meta[field]!r}, method={method_meta[field]!r}"
            )

    baseline_ids = set(baseline_episodes)
    method_ids = set(method_episodes)
    if baseline_ids != method_ids:
        missing = sorted(baseline_ids - method_ids)
        extra = sorted(method_ids - baseline_ids)
        raise ValueError(
            "episode set mismatch: "
            f"missing from method ({len(missing)}): {missing[:5]}; "
            f"extra in method ({len(extra)}): {extra[:5]}"
        )

    # Each cluster stores paired metric differences, keeping the episode-weighted
    # estimand even when different scenes have different numbers of episodes.
    clusters = defaultdict(list)
    baseline_values = {"sr": [], "spl": []}
    method_values = {"sr": [], "spl": []}
    for instr_id in sorted(baseline_ids):
        base = baseline_episodes[instr_id]
        candidate = method_episodes[instr_id]
        if base["scan_id"] != candidate["scan_id"]:
            raise ValueError(
                f"scan_id mismatch for instr_id {instr_id!r}: "
                f"baseline={base['scan_id']!r}, method={candidate['scan_id']!r}"
            )
        clusters[base["scan_id"]].append(
            (candidate["success"] - base["success"], candidate["spl"] - base["spl"])
        )
        for metric, field in (("sr", "success"), ("spl", "spl")):
            baseline_values[metric].append(base[field])
            method_values[metric].append(candidate[field])

    if len(clusters) < 2:
        raise ValueError("at least two distinct scan_id values are required for a scene-cluster confidence interval")
    cluster_totals = [
        (len(clusters[scan_id]),
         math.fsum(values[0] for values in clusters[scan_id]),
         math.fsum(values[1] for values in clusters[scan_id]))
        for scan_id in sorted(clusters)
    ]
    n_scenes = len(cluster_totals)
    rng = random.Random(seed)
    bootstrap_deltas = {"sr": [], "spl": []}
    for _ in range(resamples):
        selected = [cluster_totals[rng.randrange(n_scenes)] for _ in range(n_scenes)]
        n_selected = sum(cluster[0] for cluster in selected)
        bootstrap_deltas["sr"].append(100.0 * math.fsum(cluster[1] for cluster in selected) / n_selected)
        bootstrap_deltas["spl"].append(100.0 * math.fsum(cluster[2] for cluster in selected) / n_selected)

    metrics = {}
    n_episodes = len(baseline_ids)
    for metric in ("sr", "spl"):
        baseline_mean = math.fsum(baseline_values[metric]) / n_episodes
        method_mean = math.fsum(method_values[metric]) / n_episodes
        differences = sorted(bootstrap_deltas[metric])
        metrics[metric] = {
            "baseline": baseline_mean,
            "method": method_mean,
            "delta_pp": 100.0 * (method_mean - baseline_mean),
            "ci95_pp": [_percentile(differences, 0.025), _percentile(differences, 0.975)],
        }
    warnings = []
    if n_scenes < 10:
        warnings.append("Fewer than 10 scenes: the scene-cluster interval may be unstable.")
    if resamples < 1000:
        warnings.append("Fewer than 1000 bootstrap resamples: interval endpoints have limited Monte Carlo precision.")
    return {
        "schema_version": 1,
        "metadata": {field: baseline_meta[field] for field in REQUIRED_PROTOCOL_FIELDS},
        "n_episodes": n_episodes,
        "n_scenes": n_scenes,
        "bootstrap": {
            "unit": "scan_id",
            "paired": True,
            "resamples": resamples,
            "seed": seed,
            "confidence_level": 0.95,
            "method": "percentile",
            "aggregation": "episode_weighted",
            "scope": "evaluation scenes; does not include variation across training seeds",
        },
        "metrics": metrics,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline result in the project's normalized JSON schema")
    parser.add_argument("--method", type=Path, required=True, help="Method result in the same schema")
    parser.add_argument("--resamples", type=int, default=10000, help="Number of paired scene bootstrap draws (default: 10000)")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap random seed (default: 0)")
    parser.add_argument("--output", type=Path, help="Also save the comparison JSON to this path")
    args = parser.parse_args(argv)
    try:
        with args.baseline.open(encoding="utf-8") as stream:
            baseline = json.load(stream)
        with args.method.open(encoding="utf-8") as stream:
            method = json.load(stream)
        result = compare_results(baseline, method, resamples=args.resamples, seed=args.seed)
        serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            if args.output.resolve() in {args.baseline.resolve(), args.method.resolve()}:
                raise ValueError("--output must not overwrite either input result")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
    except (OSError, ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
