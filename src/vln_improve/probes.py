"""Small numeric diagnostic probes, fitted only on training scenes.

No policy objects or future-observation containers enter this module. Callers
must choose eligible numeric columns and keep diagnostic targets out of policy
inputs. Standardization statistics are fitted exclusively on X_train.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric matrix")
    return result


def _vector(value, name: str, *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (not allow_empty and len(result) == 0) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty numeric vector")
    return result


def _alpha(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("alpha must be finite and nonnegative")
    return float(value)


def _design(X_train, y_train, X_test):
    train, test = _matrix(X_train, "X_train"), _matrix(X_test, "X_test")
    targets = _vector(y_train, "y_train")
    if len(train) != len(targets) or train.shape[1] != test.shape[1]:
        raise ValueError("training rows and train/test feature dimensions must match")
    # Zero-column matrices are permitted for an intercept-only reference probe.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            center = train.mean(axis=0)
            scale = train.std(axis=0)
            scale = np.where(scale == 0, 1.0, scale)
            train_z = (train - center) / scale
            test_z = (test - center) / scale
        except FloatingPointError as error:
            raise ValueError("feature range overflows float64 standardization") from error
    return (np.column_stack((train_z, np.ones(len(train)))), targets,
            np.column_stack((test_z, np.ones(len(test)))))


def _linear(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = design @ coefficients
        except FloatingPointError as error:
            raise ValueError("linear prediction exceeds the finite numeric range") from error
    if not np.isfinite(result).all():
        raise ValueError("linear prediction is non-finite")
    return result


def fit_ridge(X_train, y_train, X_test, alpha: float = 1.0) -> np.ndarray:
    """Return test predictions from squared-error ridge with an unpenalized intercept.

    Objective: sum((prediction - y)**2) + alpha * sum(feature_weights**2).
    A least-squares solve on an augmented design also supports alpha=0 and
    constant/rank-deficient columns without taking an unstable matrix inverse.
    """
    penalty = _alpha(alpha)
    train, targets, test = _design(X_train, y_train, X_test)
    regularizer = np.eye(train.shape[1], dtype=np.float64) * math.sqrt(penalty)
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.lstsq(np.vstack((train, regularizer)),
                                   np.concatenate((targets, np.zeros(train.shape[1]))), rcond=None)[0]
    return _linear(test, coefficients)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _binary_labels(targets: np.ndarray) -> None:
    if not np.isin(targets, [0.0, 1.0]).all():
        raise ValueError("binary targets must be exactly 0 or 1")


def fit_logistic(X_train, y_train, X_test, alpha: float = 1.0,
                 max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    """Return probabilities from L2 logistic regression without penalizing bias.

    Objective: sum(binary_log_loss) + alpha/2 * sum(feature_weights**2).
    Damped Newton steps use stable logaddexp/sigmoid calculations and an Armijo
    line search. A single-class training set returns its constant class prior;
    that case has no finite unpenalized-intercept maximum-likelihood estimate.
    """
    penalty = _alpha(alpha)
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(tol, (int, float)) or isinstance(tol, bool) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    train, targets, test = _design(X_train, y_train, X_test)
    _binary_labels(targets)
    prior = float(targets.mean())
    if prior in (0.0, 1.0):
        return np.full(len(test), prior, dtype=np.float64)
    weights = np.zeros(train.shape[1], dtype=np.float64)
    weights[-1] = math.log(prior) - math.log1p(-prior)
    regularizer = np.ones(train.shape[1], dtype=np.float64) * (penalty / len(train))
    regularizer[-1] = 0.0

    def objective(coefficients):
        logits = _linear(train, coefficients)
        return float(np.mean(np.logaddexp(0.0, logits) - targets * logits)
                     + 0.5 * np.dot(regularizer * coefficients, coefficients))

    converged = False
    for _ in range(max_iter):
        probabilities = _sigmoid(_linear(train, weights))
        gradient = train.T @ (probabilities - targets) / len(train) + regularizer * weights
        if np.max(np.abs(gradient)) <= tol:
            converged = True
            break
        variance = probabilities * (1.0 - probabilities)
        hessian = train.T @ (variance[:, None] * train) / len(train) + np.diag(regularizer)
        direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        decrease = float(np.dot(gradient, direction))
        if not math.isfinite(decrease) or decrease <= 0:
            direction, decrease = gradient, float(np.dot(gradient, gradient))
        old_objective = objective(weights)
        step = 1.0
        for _ in range(60):
            candidate = weights - step * direction
            candidate_objective = objective(candidate)
            if candidate_objective <= old_objective - 1e-4 * step * decrease:
                break
            step *= 0.5
        else:
            raise RuntimeError("logistic probe line search failed to improve its objective")
        if np.max(np.abs(candidate - weights)) <= tol * max(1.0, float(np.max(np.abs(weights)))):
            weights = candidate
            converged = True
            break
        weights = candidate
    if not converged:
        raise RuntimeError("logistic probe did not converge; increase max_iter or regularization")
    return _sigmoid(_linear(test, weights))


def binary_metrics(y_true, probabilities) -> dict:
    """Binary metrics; AUROC/AP-style AUPRC are undefined for a single class.

    AUPRC is the step integral (average precision), grouping tied scores at one
    threshold. Log loss clips probabilities to [1e-15, 1-1e-15] so predictions
    at exactly 0 or 1 remain finite. Other metrics use the original probabilities.
    """
    labels, scores = _vector(y_true, "y_true"), _vector(probabilities, "probabilities")
    if len(labels) != len(scores) or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("probabilities must match targets and lie in [0, 1]")
    _binary_labels(labels)
    clipped = np.clip(scores, 1e-15, 1.0 - 1e-15)
    result = {
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))),
        "brier": float(np.mean((scores - labels)**2)), "auroc": None, "auprc": None,
        "num_samples": len(labels),
    }
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return result
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0) + 1, len(labels)]
    ranks = np.empty(len(labels), dtype=np.float64)
    start = 0
    for end in ends:
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    result["auroc"] = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                             / (positives * negatives))
    descending = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[descending], labels[descending]
    thresholds = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(labels) - 1]
    true_positive = np.cumsum(sorted_labels)[thresholds]
    precision = true_positive / (thresholds + 1)
    recall = true_positive / positives
    result["auprc"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return result


def regression_metrics(y_true, predictions) -> dict:
    labels, values = _vector(y_true, "y_true"), _vector(predictions, "predictions")
    if len(labels) != len(values):
        raise ValueError("predictions must match targets")
    with np.errstate(over="raise", invalid="raise"):
        try:
            mae = float(np.mean(np.abs(values - labels)))
        except FloatingPointError as error:
            raise ValueError("errors exceed the finite numeric range") from error
    return {"mae": mae, "num_samples": len(labels)}


def _groups(scan_ids: Sequence[str], size: int) -> tuple[np.ndarray, np.ndarray]:
    scans = np.asarray(scan_ids, dtype=object)
    if scans.ndim != 1 or len(scans) != size or not all(isinstance(value, str) and value for value in scans):
        raise ValueError("scan_ids must contain one nonempty scene string per sample")
    unique = np.array(sorted(set(scans)), dtype=object)
    lookup = {scene: index for index, scene in enumerate(unique)}
    inverse = np.array([lookup[scene] for scene in scans], dtype=np.int64)
    return unique, inverse


def paired_cluster_bootstrap(values_a, values_b, scan_ids: Sequence[str],
                             n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Bootstrap mean(a-b) by resampling entire scenes, retaining paired rows.

    Each draw samples num_clusters scenes with replacement. All rows from each
    sampled scene are included together; larger scenes retain their row weight.
    Inputs should be additive per-example losses/errors, not a scalar AUROC.
    The returned interval is a 95% percentile interval, not a significance test.
    """
    first, second = _vector(values_a, "values_a"), _vector(values_b, "values_b")
    if len(first) != len(second):
        raise ValueError("paired values must have equal lengths")
    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scenes, inverse = _groups(scan_ids, len(first))
    if len(scenes) < 2:
        raise ValueError("a scene bootstrap requires at least two clusters")
    with np.errstate(over="raise", invalid="raise"):
        try:
            difference = first - second
            sums = np.bincount(inverse, weights=difference, minlength=len(scenes))
            counts = np.bincount(inverse, minlength=len(scenes))
            rng = np.random.default_rng(seed)
            draws = np.empty(n_bootstrap, dtype=np.float64)
            for index in range(n_bootstrap):
                sampled = rng.integers(0, len(scenes), size=len(scenes))
                draws[index] = sums[sampled].sum() / counts[sampled].sum()
            point = float(difference.mean())
        except FloatingPointError as error:
            raise ValueError("bootstrap values exceed the finite numeric range") from error
    if not np.isfinite(draws).all() or not math.isfinite(point):
        raise ValueError("bootstrap statistics must be finite")
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"difference_a_minus_b": point, "ci95": [float(low), float(high)],
            "num_samples": len(first), "num_clusters": len(scenes), "n_bootstrap": n_bootstrap}


def group_kfold(scan_ids: Sequence[str], folds: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic scene-disjoint folds, greedily balancing sample counts.

    Seeded tie-breaking affects equal-size scenes only. Returned row indices
    preserve input order. Feature preprocessing must still be fitted within
    each training fold; fit_ridge/fit_logistic enforce this themselves.
    """
    if type(folds) is not int or folds < 2:
        raise ValueError("folds must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scenes, inverse = _groups(scan_ids, len(scan_ids))
    if folds > len(scenes):
        raise ValueError("folds cannot exceed the number of scenes")
    counts = np.bincount(inverse, minlength=len(scenes))
    shuffled = np.random.default_rng(seed).permutation(len(scenes))
    order = sorted(shuffled.tolist(), key=lambda index: -counts[index])
    assignment = np.zeros(len(scenes), dtype=np.int64)
    loads = np.zeros(folds, dtype=np.int64)
    for scene in order:
        fold = int(np.argmin(loads))
        assignment[scene] = fold
        loads[fold] += counts[scene]
    return [(np.flatnonzero(assignment[inverse] != fold), np.flatnonzero(assignment[inverse] == fold))
            for fold in range(folds)]
