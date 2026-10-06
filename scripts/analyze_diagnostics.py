#!/usr/bin/env python3
"""Fit fixed diagnostic probes on train_fit and evaluate scene-disjoint train_dev.

Natural and simulated-branch arrivals must be analyzed in separate reports.
Arrival deltas are training targets or explicitly labelled offline references.
The report is neither a deployed policy nor evidence of navigation improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vln_improve.probes import (
    binary_metrics, fit_logistic, fit_ridge, group_kfold,
    paired_cluster_bootstrap, regression_metrics,
)


ALPHA = 1.0
TARGETS = {"teacher_regret": "regression", "execution_regret": "regression",
           "teacher_optimal": "classification", "execution_optimal": "classification"}
INTERVENTIONS = ("arrival", "arrival_normmatched", "noise_normmatched")
ARRIVAL_KINDS = ("natural", "simulated_branch")
GROUPS = ("P0", "P0_P1", "P0_true_arrival_delta", "P0_normmatched_arrival_delta",
          "P0_noise_normmatched_delta", "P3_predicted_arrival_delta")
P3_TARGET = "interventions.arrival.delta_margin"


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {number} of {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"row {number} of {path} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"no diagnostic rows in {path}")
    return rows


def _number(value, name: str) -> float:
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


def _validate(rows: list[dict], split: str, columns: dict | None = None) -> dict:
    if not rows:
        raise ValueError(f"{split} has no diagnostic rows")
    names = columns
    seen = set()
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"diagnostic rows must have split={split!r}; official validation is forbidden")
        if row.get("arrival_kind") not in ARRIVAL_KINDS:
            raise ValueError("arrival_kind must explicitly be natural or simulated_branch")
        for key in ("scan_id", "instr_id", "target_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{key} must be a nonempty string")
        if type(row.get("step")) is not int or row["step"] < 0:
            raise ValueError("step must be a nonnegative integer")
        key = tuple(row[field] for field in ("scan_id", "instr_id", "step", "target_id"))
        if key in seen:
            raise ValueError("duplicate diagnostic candidate row")
        seen.add(key)
        current_names = {}
        for block in ("p0", "p1"):
            values = row.get(block)
            if not isinstance(values, dict) or not values or not all(isinstance(k, str) and k for k in values):
                raise ValueError(f"{block} must contain named numeric features")
            current_names[block] = sorted(values)
            for name, value in values.items():
                _number(value, f"{block}.{name}")
        if names is None:
            names = current_names
        if names != current_names:
            raise ValueError("feature columns changed across rows or splits")
        labels = row.get("labels")
        if not isinstance(labels, dict):
            raise ValueError("labels must be a separate dictionary")
        for name, kind in TARGETS.items():
            value = _number(labels.get(name), f"labels.{name}")
            if (kind == "classification" and value not in (0, 1)) or (kind == "regression" and value < 0):
                raise ValueError(f"invalid diagnostic label: {name}")
        if _number(labels.get("base_chosen"), "labels.base_chosen") not in (0, 1):
            raise ValueError("base_chosen must be binary")
        interventions = row.get("interventions")
        if not isinstance(interventions, dict):
            raise ValueError("interventions must be a separate dictionary")
        optional = [name for name in ("shuffled_arrival", "arrival_headingmatched", "arrival_headingmatched_normmatched",
                                      "shuffled_normmatched", "shuffled_headingmatched") if name in interventions]
        for name in (*INTERVENTIONS, *optional):
            value = interventions.get(name)
            if not isinstance(value, dict):
                raise ValueError(f"missing required intervention: {name}")
            _number(value.get("delta_margin"), f"interventions.{name}.delta_margin")
    return names


def _features(rows, columns, block):
    return np.array([[row[block][column] for column in columns[block]] for row in rows], dtype=np.float64)


def _delta(rows, intervention):
    return np.array([row["interventions"][intervention]["delta_margin"] for row in rows], dtype=np.float64)


def _per_row_loss(targets, predictions, kind):
    if kind == "regression":
        return np.abs(targets - predictions)
    clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
    return -(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))


def _comparison(first, reference, scenes, n_bootstrap, seed):
    if len(set(scenes)) < 2:
        return {"difference_a_minus_b": float(np.mean(first - reference)), "ci95": None,
                "num_samples": len(first), "num_clusters": len(set(scenes)),
                "n_bootstrap": 0, "reason": "at_least_two_development_scenes_required"}
    return paired_cluster_bootstrap(first, reference, scenes, n_bootstrap=n_bootstrap, seed=seed)


def _class_counts(train_y, dev_y):
    return {"train_positive": int(train_y.sum()), "train_negative": int(len(train_y) - train_y.sum()),
            "dev_positive": int(dev_y.sum()), "dev_negative": int(len(dev_y) - dev_y.sum())}


def _coverage(rows):
    """Describe observable source/choice coverage, without changing fitting weights."""
    source_key = "coverage_source_count_total"
    cells = {}
    for chosen in (False, True):
        for multi in (False, True):
            selected = [row for row in rows if source_key in row["p0"]
                        and bool(row["labels"]["base_chosen"]) == chosen
                        and (row["p0"][source_key] > 1) == multi]
            cells[f"{'chosen' if chosen else 'unchosen'}_{'multi' if multi else 'single'}_source"] = {
                "rows": len(selected), "scans": sorted({row["scan_id"] for row in selected}),
                "class_counts": {target: {"positive": sum(row["labels"][target] == 1 for row in selected),
                                           "negative": sum(row["labels"][target] == 0 for row in selected)}
                                 for target, kind in TARGETS.items() if kind == "classification"},
            }
    return {
        "source_count_feature": source_key,
        "rows_without_source_count": sum(source_key not in row["p0"] for row in rows),
        "strata": cells,
        "per_scan_class_counts": {
            scan: {target: {"positive": sum(row["scan_id"] == scan and row["labels"][target] == 1 for row in rows),
                            "negative": sum(row["scan_id"] == scan and row["labels"][target] == 0 for row in rows)}
                   for target, kind in TARGETS.items() if kind == "classification"}
            for scan in sorted({row["scan_id"] for row in rows})},
    }


def _heading_control(train_rows, dev_rows, columns, n_bootstrap, seed):
    """Compare historical-heading and actual-arrival references on identical rows."""
    intervention = "arrival_headingmatched"
    train = [row for row in train_rows if intervention in row["interventions"]]
    dev = [row for row in dev_rows if intervention in row["interventions"]]
    train_scenes, dev_scenes = [row["scan_id"] for row in train], [row["scan_id"] for row in dev]
    result = {
        "status": "not_provided" if not train and not dev else "insufficient_data",
        "subset_rule": "rows_with_an_explicit_arrival_headingmatched_control_in_each_split",
        "comparison": "P0, P0+true arrival and P0+headingmatched arrival refitted and evaluated on exactly the same subset",
        "data": {"train_rows": len(train), "dev_rows": len(dev),
                 "excluded_train_rows": len(train_rows) - len(train), "excluded_dev_rows": len(dev_rows) - len(dev),
                 "train_scans": sorted(set(train_scenes)), "dev_scans": sorted(set(dev_scenes)),
                 "train_row_coverage": len(train) / len(train_rows), "dev_row_coverage": len(dev) / len(dev_rows),
                 "dev_candidate_keys": [{key: row[key] for key in ("scan_id", "instr_id", "step", "target_id")} for row in dev]},
        "limitations": ["Historical-heading encoding is an offline future-information reference; it never enters P3 inputs or changes the P3 target.",
                        "The main arrival reference retains the acquisition protocol's observed arrival orientation.",
                        "Heading control availability may select a subset; compare only references refitted on these same rows."],
    }
    if len(set(train_scenes)) < 2 or len(set(dev_scenes)) < 2:
        result["reason"] = "at_least_two_training_and_two_development_scenes_with_heading_controls_required"
        return result
    train_p0, dev_p0 = _features(train, columns, "p0"), _features(dev, columns, "p0")
    groups = {"P0": (train_p0, dev_p0)}
    for name, delta_name in (("P0_true_arrival_delta", "arrival"),
                             ("P0_headingmatched_arrival_delta", intervention)):
        groups[name] = (np.column_stack((train_p0, _delta(train, delta_name))),
                        np.column_stack((dev_p0, _delta(dev, delta_name))))
    results, predictions, comparisons = {}, {}, {}
    completed = 0
    for target, kind in TARGETS.items():
        train_y = np.array([row["labels"][target] for row in train], dtype=np.float64)
        dev_y = np.array([row["labels"][target] for row in dev], dtype=np.float64)
        class_counts = _class_counts(train_y, dev_y) if kind == "classification" else None
        if class_counts is not None and any(count == 0 for count in class_counts.values()):
            results[target] = {"status": "insufficient_data", "reason": "both_classes_required_in_each_heading_subset",
                               "class_counts": class_counts}
            continue
        results[target], predictions[target], losses = {"status": "complete"}, {}, {}
        if class_counts is not None:
            results[target]["class_counts"] = class_counts
        for name, (X_train, X_dev) in groups.items():
            prediction = (fit_ridge if kind == "regression" else fit_logistic)(X_train, train_y, X_dev, alpha=ALPHA)
            results[target][name] = {"metrics": (regression_metrics if kind == "regression" else binary_metrics)(dev_y, prediction),
                                     "num_features": X_train.shape[1]}
            predictions[target][name] = prediction.tolist()
            losses[name] = _per_row_loss(dev_y, prediction, kind)
        comparisons[target] = {}
        for name, candidate, reference in (
            ("true_arrival_vs_P0", "P0_true_arrival_delta", "P0"),
            ("headingmatched_arrival_vs_P0", "P0_headingmatched_arrival_delta", "P0"),
            ("headingmatched_vs_true_arrival", "P0_headingmatched_arrival_delta", "P0_true_arrival_delta"),
        ):
            comparisons[target][name] = {
                "candidate": candidate, "reference": reference, "cohort": "same_heading_subset",
                "loss": "absolute_error" if kind == "regression" else "log_loss",
                **_comparison(losses[candidate], losses[reference], dev_scenes, n_bootstrap, seed),
            }
        completed += 1
    result.update(status="complete" if completed == len(TARGETS) else "partial",
                  development_results=results, development_predictions=predictions,
                  paired_loss_comparisons=comparisons)
    return result


def _symmetric_shuffled_control(train_rows, dev_rows, columns, n_bootstrap, seed, *, variant):
    """Match object identity controls while holding the encoding convention fixed."""
    true_intervention = f"arrival_{variant}"
    shuffled_intervention = f"shuffled_{variant}"
    true_group = f"P0_{variant}_arrival_delta"
    shuffled_group = f"P0_shuffled_{variant}_delta"
    train = [row for row in train_rows if all(name in row["interventions"]
                                             for name in (true_intervention, shuffled_intervention))]
    dev = [row for row in dev_rows if all(name in row["interventions"]
                                        for name in (true_intervention, shuffled_intervention))]
    train_scenes, dev_scenes = [row["scan_id"] for row in train], [row["scan_id"] for row in dev]
    provided = any(shuffled_intervention in row["interventions"] for row in train_rows + dev_rows)
    result = {
        "status": "insufficient_data" if provided else "not_provided",
        "intervention_pair": {"true": true_intervention, "shuffled": shuffled_intervention},
        "subset_rule": f"rows_with_both_{true_intervention}_and_{shuffled_intervention}_in_each_split",
        "comparison": "P0 and both future-information references refitted and evaluated on exactly the same subset",
        "encoding_control": ("Both actual legal-path arrival representations are scaled to the historical target proxy norm."
                             if variant == "normmatched" else
                             "Target and donor are both encoded at the same historical-state heading and elevation."),
        "data": {"train_rows": len(train), "dev_rows": len(dev),
                 "excluded_train_rows": len(train_rows) - len(train), "excluded_dev_rows": len(dev_rows) - len(dev),
                 "train_scans": sorted(set(train_scenes)), "dev_scans": sorted(set(dev_scenes)),
                 "train_row_coverage": len(train) / len(train_rows), "dev_row_coverage": len(dev) / len(dev_rows),
                 "dev_candidate_keys": [{key: row[key] for key in ("scan_id", "instr_id", "step", "target_id")} for row in dev]},
        "limitations": ["Only rows containing both members of this intervention pair enter this comparison.",
                        "Donor availability selects a subset; compare these results only with references refitted on this same subset.",
                        "The encoding convention and donor provenance must be enforced upstream; numeric JSON cannot certify them.",
                        "These privileged references never change P3 inputs, targets or the main analysis cohort."],
    }
    if len(set(train_scenes)) < 2 or len(set(dev_scenes)) < 2:
        result["reason"] = "at_least_two_training_and_two_development_scenes_with_paired_controls_required"
        return result
    train_p0, dev_p0 = _features(train, columns, "p0"), _features(dev, columns, "p0")
    groups = {"P0": (train_p0, dev_p0)}
    for group, intervention in ((true_group, true_intervention), (shuffled_group, shuffled_intervention)):
        groups[group] = (np.column_stack((train_p0, _delta(train, intervention))),
                         np.column_stack((dev_p0, _delta(dev, intervention))))
    results, predictions, comparisons = {}, {}, {}
    completed = 0
    for target, kind in TARGETS.items():
        train_y = np.array([row["labels"][target] for row in train], dtype=np.float64)
        dev_y = np.array([row["labels"][target] for row in dev], dtype=np.float64)
        class_counts = _class_counts(train_y, dev_y) if kind == "classification" else None
        if class_counts is not None and any(count == 0 for count in class_counts.values()):
            results[target] = {"status": "insufficient_data", "reason": "both_classes_required_in_each_paired_subset",
                               "class_counts": class_counts}
            continue
        results[target], predictions[target], losses = {"status": "complete"}, {}, {}
        if class_counts is not None:
            results[target]["class_counts"] = class_counts
        for group, (X_train, X_dev) in groups.items():
            prediction = (fit_ridge if kind == "regression" else fit_logistic)(X_train, train_y, X_dev, alpha=ALPHA)
            results[target][group] = {"metrics": (regression_metrics if kind == "regression" else binary_metrics)(dev_y, prediction),
                                      "num_features": X_train.shape[1]}
            predictions[target][group] = prediction.tolist()
            losses[group] = _per_row_loss(dev_y, prediction, kind)
        comparisons[target] = {}
        for name, candidate, reference in (("true_vs_P0", true_group, "P0"),
                                           ("shuffled_vs_P0", shuffled_group, "P0"),
                                           ("true_vs_shuffled", true_group, shuffled_group)):
            comparisons[target][name] = {
                "candidate": candidate, "reference": reference, "cohort": f"same_{variant}_shuffled_subset",
                "loss": "absolute_error" if kind == "regression" else "log_loss",
                **_comparison(losses[candidate], losses[reference], dev_scenes, n_bootstrap, seed),
            }
        completed += 1
    result.update(status="complete" if completed == len(TARGETS) else "partial",
                  development_results=results, development_predictions=predictions,
                  paired_loss_comparisons=comparisons)
    return result


def _shuffled_control(train_rows, dev_rows, columns, n_bootstrap, seed):
    """Refit all four controls on one donor subset, preserving the main cohort."""
    train = [row for row in train_rows if "shuffled_arrival" in row["interventions"]]
    dev = [row for row in dev_rows if "shuffled_arrival" in row["interventions"]]
    train_scenes, dev_scenes = [row["scan_id"] for row in train], [row["scan_id"] for row in dev]
    result = {
        "status": "insufficient_data",
        "subset_rule": "rows_with_an_explicit_shuffled_arrival_control_in_each_split",
        "comparison": "P0, P0+true arrival, P0+normmatched arrival and P0+shuffled all refitted on exactly this training subset and evaluated on exactly this development subset",
        "data": {"train_rows": len(train), "dev_rows": len(dev),
                 "excluded_train_rows": len(train_rows) - len(train), "excluded_dev_rows": len(dev_rows) - len(dev),
                 "train_scans": sorted(set(train_scenes)), "dev_scans": sorted(set(dev_scenes)),
                 "train_row_coverage": len(train) / len(train_rows), "dev_row_coverage": len(dev) / len(dev_rows),
                 "dev_candidate_keys": [{key: row[key] for key in ("scan_id", "instr_id", "step", "target_id")} for row in dev]},
        "limitations": ["Donor availability creates an additional selected subset.",
                        "Subset shuffled results cannot be compared with full-cohort P0 or true-arrival metrics.",
                        "Normmatched arrival versus raw shuffled arrival does not hold feature scale fixed; use normmatched_shuffled_control for a symmetric norm comparison when provided.",
                        "Norm-matching probes a scale confound but does not establish a causal reliability label."],
    }
    if len(set(train_scenes)) < 2 or len(set(dev_scenes)) < 2:
        result["reason"] = "at_least_two_training_and_two_development_scenes_with_shuffled_controls_required"
        return result
    train_p0, dev_p0 = _features(train, columns, "p0"), _features(dev, columns, "p0")
    groups = {"P0": (train_p0, dev_p0)}
    for name, intervention in (("P0_true_arrival_delta", "arrival"),
                               ("P0_normmatched_arrival_delta", "arrival_normmatched"),
                               ("P0_shuffled_arrival_delta", "shuffled_arrival")):
        groups[name] = (np.column_stack((train_p0, _delta(train, intervention))),
                        np.column_stack((dev_p0, _delta(dev, intervention))))
    results, predictions, comparisons = {}, {}, {}
    completed = 0
    for target, kind in TARGETS.items():
        train_y = np.array([row["labels"][target] for row in train], dtype=np.float64)
        dev_y = np.array([row["labels"][target] for row in dev], dtype=np.float64)
        class_counts = None
        if kind == "classification":
            class_counts = _class_counts(train_y, dev_y)
            if any(count == 0 for count in class_counts.values()):
                results[target] = {"status": "insufficient_data", "reason": "both_classes_required_in_each_shuffled_subset",
                                   "class_counts": class_counts}
                continue
        fit = fit_ridge if kind == "regression" else fit_logistic
        metric = regression_metrics if kind == "regression" else binary_metrics
        results[target], predictions[target], losses = {"status": "complete"}, {}, {}
        for name, (X_train, X_dev) in groups.items():
            prediction = fit(X_train, train_y, X_dev, alpha=ALPHA)
            results[target][name] = {"metrics": metric(dev_y, prediction), "num_features": X_train.shape[1]}
            predictions[target][name] = prediction.tolist()
            losses[name] = _per_row_loss(dev_y, prediction, kind)
        if class_counts is not None:
            results[target]["class_counts"] = class_counts
        comparisons[target] = {}
        for name, candidate, reference in (
            ("shuffled_vs_P0", "P0_shuffled_arrival_delta", "P0"),
            ("true_arrival_vs_P0", "P0_true_arrival_delta", "P0"),
            ("normmatched_arrival_vs_P0", "P0_normmatched_arrival_delta", "P0"),
            ("true_arrival_vs_shuffled", "P0_true_arrival_delta", "P0_shuffled_arrival_delta"),
            ("normmatched_arrival_vs_shuffled", "P0_normmatched_arrival_delta", "P0_shuffled_arrival_delta"),
            ("normmatched_vs_true_arrival", "P0_normmatched_arrival_delta", "P0_true_arrival_delta"),
        ):
            comparisons[target][name] = {
                "candidate": candidate, "reference": reference, "cohort": "same_shuffled_subset",
                "loss": "absolute_error" if kind == "regression" else "log_loss",
                **_comparison(losses[candidate], losses[reference], dev_scenes, n_bootstrap, seed),
            }
        completed += 1
    result.update(status="complete" if completed == len(TARGETS) else "partial",
                  development_results=results, development_predictions=predictions,
                  paired_loss_comparisons=comparisons)
    return result


def analyze(train_rows: list[dict], dev_rows: list[dict], *, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    columns = _validate(train_rows, "train_fit")
    _validate(dev_rows, "train_dev", columns)
    arrival_kinds = {row["arrival_kind"] for row in train_rows + dev_rows}
    if len(arrival_kinds) != 1:
        raise ValueError("arrival_kind cohorts cannot be mixed within or across splits; run separate reports")
    arrival_kind = next(iter(arrival_kinds))
    train_scenes = [row["scan_id"] for row in train_rows]
    dev_scenes = [row["scan_id"] for row in dev_rows]
    if set(train_scenes) & set(dev_scenes):
        raise ValueError("training and development scenes overlap")
    if len(set(train_scenes)) < 2:
        raise ValueError("P3 cross-fitting requires at least two training scenes")
    train_p0, dev_p0 = _features(train_rows, columns, "p0"), _features(dev_rows, columns, "p0")
    train_p1, dev_p1 = _features(train_rows, columns, "p1"), _features(dev_rows, columns, "p1")
    train_combined, dev_combined = np.column_stack((train_p0, train_p1)), np.column_stack((dev_p0, dev_p1))
    train_delta, dev_delta = _delta(train_rows, "arrival"), _delta(dev_rows, "arrival")
    oof_delta = np.full(len(train_rows), np.nan)
    fold_manifest = []
    for fitting, held_out in group_kfold(train_scenes, folds=min(5, len(set(train_scenes))), seed=seed):
        oof_delta[held_out] = fit_ridge(train_combined[fitting], train_delta[fitting], train_combined[held_out], alpha=ALPHA)
        fold_manifest.append({"fit_scans": sorted({train_scenes[i] for i in fitting}),
                              "held_out_scans": sorted({train_scenes[i] for i in held_out}),
                              "fit_rows": len(fitting), "held_out_rows": len(held_out)})
    predicted_dev_delta = fit_ridge(train_combined, train_delta, dev_combined, alpha=ALPHA)
    groups = {
        "P0": (train_p0, dev_p0), "P0_P1": (train_combined, dev_combined),
        "P0_true_arrival_delta": (np.column_stack((train_p0, train_delta)), np.column_stack((dev_p0, dev_delta))),
        "P0_normmatched_arrival_delta": (np.column_stack((train_p0, _delta(train_rows, "arrival_normmatched"))),
                                         np.column_stack((dev_p0, _delta(dev_rows, "arrival_normmatched")))),
        "P0_noise_normmatched_delta": (np.column_stack((train_p0, _delta(train_rows, "noise_normmatched"))),
                                       np.column_stack((dev_p0, _delta(dev_rows, "noise_normmatched")))),
        "P3_predicted_arrival_delta": (np.column_stack((train_p0, oof_delta)),
                                      np.column_stack((dev_p0, predicted_dev_delta))),
    }
    results, predictions, comparisons, target_status = {}, {}, {}, {}
    for target, kind in TARGETS.items():
        train_y = np.array([row["labels"][target] for row in train_rows], dtype=np.float64)
        dev_y = np.array([row["labels"][target] for row in dev_rows], dtype=np.float64)
        class_counts = _class_counts(train_y, dev_y) if kind == "classification" else None
        if class_counts is not None and any(count == 0 for count in class_counts.values()):
            unavailable = {"status": "insufficient_data", "reason": "both_classes_required_in_training_and_development",
                           "class_counts": class_counts}
            target_status[target] = unavailable
            results[target] = {name: {**unavailable, "metrics": None, "num_features": matrices[0].shape[1]}
                               for name, matrices in groups.items()}
            predictions[target], comparisons[target] = {}, {}
            continue
        target_status[target] = {"status": "complete"}
        if class_counts is not None:
            target_status[target]["class_counts"] = class_counts
        target_results, target_predictions, losses = {}, {}, {}
        for name, (X_train, X_dev) in groups.items():
            prediction = (fit_ridge if kind == "regression" else fit_logistic)(X_train, train_y, X_dev, alpha=ALPHA)
            metrics = (regression_metrics if kind == "regression" else binary_metrics)(dev_y, prediction)
            target_results[name] = {"status": "complete", "metrics": metrics, "num_features": X_train.shape[1]}
            if class_counts is not None:
                target_results[name]["class_counts"] = class_counts
            target_predictions[name] = prediction.tolist()
            losses[name] = _per_row_loss(dev_y, prediction, kind)
        results[target], predictions[target] = target_results, target_predictions
        comparisons[target] = {
            name: {"reference": "P0", "loss": "absolute_error" if kind == "regression" else "log_loss",
                   **_comparison(losses[name], losses["P0"], dev_scenes, n_bootstrap, seed)}
            for name in groups if name != "P0"
        }
        # M versus an equally observable feature set is at least as important as
        # M versus geometry/scores alone; P3 does not gain a new inference input.
        comparisons[target]["P3_vs_P0_P1"] = {
            "reference": "P0_P1", "loss": "absolute_error" if kind == "regression" else "log_loss",
            **_comparison(losses["P3_predicted_arrival_delta"], losses["P0_P1"], dev_scenes, n_bootstrap, seed),
        }
        for name, reference in (("normmatched_vs_true_arrival", "P0_true_arrival_delta"),
                                ("normmatched_vs_noise", "P0_noise_normmatched_delta")):
            comparisons[target][name] = {
                "candidate": "P0_normmatched_arrival_delta", "reference": reference,
                "loss": "absolute_error" if kind == "regression" else "log_loss",
                **_comparison(losses["P0_normmatched_arrival_delta"], losses[reference], dev_scenes, n_bootstrap, seed),
            }
    if arrival_kind == "natural":
        cohort_limitations = ["Only naturally reached candidates have arrival observations; this is a selected, policy-dependent sample."]
    else:
        cohort_limitations = [
            "Simulated branches acquire observations for preselected candidates at frozen baseline states; this remains a selected sample of the baseline state and candidate distribution.",
            "Branch acquisition removes the requirement of natural arrival for the selected candidates, but does not demonstrate generalization to states visited by a changed policy.",
            "Legal branch execution and isolation from the original policy trajectory must be audited upstream; numeric JSON cannot certify them.",
        ]
    return {
        "schema_version": 3, "kind": f"offline_{arrival_kind}_arrival_diagnostics",
        "status": "complete" if all(item["status"] == "complete" for item in target_status.values()) else "partial",
        "claims_navigation_improvement": False,
        "protocol": {"training_split": "train_fit", "development_split": "train_dev",
                     "official_validation_used": False, "alpha": ALPHA,
                     "alpha_selection": "fixed_before_development_evaluation",
                     "standardization": "fit_on_each_training_fold_only",
                     "bootstrap_unit": "scan", "bootstrap_sign": "model_minus_reference; negative_loss_difference_is_better",
                     "auprc_definition": "average_precision_step_integral_with_ties_grouped",
                     "p3_prediction_target": P3_TARGET,
                     "arrival_kind": arrival_kind,
                     "cohort_policy": "one_explicit_arrival_kind_per_report; no_pooled_fit_across_acquisition_protocols",
                     "arrival_orientation": "legal_execution_path_arrival_heading_and_elevation" if arrival_kind == "simulated_branch" else "naturally_observed_arrival_heading_and_elevation",
                     "shuffled_arrival_orientation": "donor_legal_execution_path_arrival_heading_and_elevation" if arrival_kind == "simulated_branch" else "naturally_observed_donor_arrival_heading_and_elevation",
                     "seed": seed, "n_bootstrap": n_bootstrap},
        "data": {"train_rows": len(train_rows), "dev_rows": len(dev_rows),
                 "train_scans": sorted(set(train_scenes)), "dev_scans": sorted(set(dev_scenes)),
                 "dev_candidate_keys": [{key: row[key] for key in ("scan_id", "instr_id", "step", "target_id")}
                                        for row in dev_rows],
                 "train_base_chosen_fraction": float(np.mean([row["labels"]["base_chosen"] for row in train_rows])),
                 "dev_base_chosen_fraction": float(np.mean([row["labels"]["base_chosen"] for row in dev_rows])),
                 "selection_coverage": {"train": _coverage(train_rows), "dev": _coverage(dev_rows)}},
        "features": columns,
        "group_interpretation": {
            "P0": "observable baseline scores, geometry, age and coverage controls",
            "P0_P1": "same-state observable baseline plus evidence statistics",
            "P0_true_arrival_delta": "offline future-information reference; not a deployable feature or guaranteed upper bound",
            "P0_normmatched_arrival_delta": "offline future-information reference with replacement norm matched to the original; probes the feature-scale confound",
            "P0_shuffled_arrival_delta": "offline mismatched-arrival control; source/donor selection must be audited upstream",
            "P0_headingmatched_arrival_delta": "offline privileged reference encoded at the historical state's heading/elevation; matched-subset analysis only",
            "P0_shuffled_normmatched_delta": "offline donor actual-arrival representation scaled to the historical target proxy norm; symmetric matched-subset control",
            "P0_shuffled_headingmatched_delta": "offline donor encoded at the historical state's heading/elevation; symmetric matched-subset control",
            "P0_noise_normmatched_delta": "offline norm-matched perturbation control",
            "P3_predicted_arrival_delta": "train-scene cross-fitted predicted delta; dev prediction uses train-fitted P0+P1 only",
        },
        "p3_delta_prediction": {"target": P3_TARGET, "folds": fold_manifest,
                                "train_oof_metrics": regression_metrics(train_delta, oof_delta),
                                "dev_metrics": regression_metrics(dev_delta, predicted_dev_delta),
                                "train_oof_predictions": oof_delta.tolist(),
                                "dev_predictions": predicted_dev_delta.tolist()},
        "development_results": results, "paired_loss_comparisons": comparisons,
        "target_status": target_status,
        "development_predictions": predictions,
        "shuffled_control": _shuffled_control(train_rows, dev_rows, columns, n_bootstrap, seed),
        "heading_control": _heading_control(train_rows, dev_rows, columns, n_bootstrap, seed),
        "normmatched_shuffled_control": _symmetric_shuffled_control(train_rows, dev_rows, columns, n_bootstrap, seed,
                                                                   variant="normmatched"),
        "headingmatched_shuffled_control": _symmetric_shuffled_control(train_rows, dev_rows, columns, n_bootstrap, seed,
                                                                      variant="headingmatched"),
        "limitations": [
            *cohort_limitations,
            "Episode candidates are also correlated within instructions; confidence intervals resample whole scenes.",
            "Teacher and execution regret are diagnostic oracle costs, not complete instruction-following correctness.",
            "A future-information reference is an offline probe, not a causal reliability label or a guaranteed upper bound.",
            "Norm-matching checks a feature-scale confound; remaining effects may still reflect viewpoint change or out-of-distribution inputs.",
            "P3 is a linear learned projection of P0+P1 and may differ through regularization; compare it directly with P0+P1.",
            "Feature and shuffled-donor provenance must be enforced during replay; numeric JSON cannot certify their origin.",
            "These predictive associations do not demonstrate SR/SPL improvement, a successful policy intervention, or publishable novelty.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = analyze(read_jsonl(args.train), read_jsonl(args.dev), n_bootstrap=args.n_bootstrap, seed=args.seed)
    report["input_files"] = {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for name, path in (("train", args.train), ("dev", args.dev))}
    report["analysis_provenance"] = {
        "source_files": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in (("analyzer", Path(__file__)), ("probes", ROOT / "src/vln_improve/probes.py"))},
        "python_version": sys.version.split()[0], "numpy_version": np.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "train_rows": report["data"]["train_rows"],
                      "dev_rows": report["data"]["dev_rows"], "claims_navigation_improvement": False}))


if __name__ == "__main__":
    main()
