import numpy as np
import pytest

from vln_improve.probes import (
    binary_metrics, fit_logistic, fit_ridge, group_kfold,
    paired_cluster_bootstrap, regression_metrics,
)


def test_ridge_generalizes_to_separate_samples_and_shifted_feature_range():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(100, 3))
    test = rng.normal(loc=5, size=(40, 3))
    weights = np.array([2.0, -3.0, 0.5])
    targets = train @ weights + 13.0
    expected = test @ weights + 13.0
    predicted = fit_ridge(train, targets, test, alpha=0)
    assert regression_metrics(expected, predicted)["mae"] < 1e-12


def test_logistic_generalizes_to_held_out_samples():
    rng = np.random.default_rng(15)
    train = rng.normal(size=(500, 3))
    test = rng.normal(size=(300, 3))
    weights = np.array([3.0, -2.0, 0.5])
    # Independent labels for independent examples, not accuracy on fitted data.
    train_probability = 1 / (1 + np.exp(-(train @ weights + 0.3)))
    test_probability = 1 / (1 + np.exp(-(test @ weights + 0.3)))
    y_train = (rng.uniform(size=len(train)) < train_probability).astype(int)
    y_test = (rng.uniform(size=len(test)) < test_probability).astype(int)
    predicted = fit_logistic(train, y_train, test)
    measured = binary_metrics(y_test, predicted)
    prior = binary_metrics(y_test, np.full(len(test), y_train.mean()))
    assert measured["auroc"] > 0.85
    assert measured["log_loss"] < prior["log_loss"] - 0.2
    assert measured["brier"] < prior["brier"]


@pytest.mark.parametrize("fit", [fit_ridge, fit_logistic])
def test_test_set_does_not_influence_fitted_standardization(fit):
    train = np.array([[-2.0, 1.0], [-1.0, 2.0], [0.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
    labels = np.array([0, 0, 1, 1, 1])
    test = np.array([[-0.5, 1.3], [0.3, 1.8]])
    reference = fit(train, labels, test, alpha=3)
    expanded = fit(train, labels, np.vstack((test, [[1e10, -1e10]])), alpha=3)
    np.testing.assert_array_equal(reference, expanded[:2])


def test_constant_columns_and_unregularized_ridge_intercept():
    train = np.ones((20, 3)) * np.array([1, 5, -9])
    test = np.array([[0, 99, -1], [20, 5, -9]], dtype=float)
    predicted = fit_ridge(train, np.full(20, 17.5), test, alpha=1e12)
    np.testing.assert_allclose(predicted, 17.5, rtol=0, atol=1e-10)
    rank_deficient = fit_ridge(train, np.arange(20), test, alpha=0)
    np.testing.assert_allclose(rank_deficient, np.mean(np.arange(20)), atol=1e-12)


def test_constant_columns_and_unregularized_logistic_intercept():
    train = np.ones((20, 2))
    labels = np.array([1] * 14 + [0] * 6)
    predicted = fit_logistic(train, labels, np.array([[99.0, -200.0], [1.0, 1.0]]), alpha=1e12)
    np.testing.assert_allclose(predicted, 0.7, atol=1e-12)


@pytest.mark.parametrize("target", [0, 1])
def test_single_class_training_and_metrics(target):
    predicted = fit_logistic(np.arange(10).reshape(-1, 1), np.full(10, target), [[-99], [99]])
    np.testing.assert_array_equal(predicted, np.full(2, target))
    measured = binary_metrics(np.full(2, target), predicted)
    assert measured["auroc"] is None and measured["auprc"] is None
    assert measured["brier"] == 0
    assert np.isfinite(measured["log_loss"])


def test_extreme_test_logits_and_endpoint_probabilities_are_finite():
    train = np.array([[-3], [-2], [-1], [1], [2], [3]], dtype=float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        predicted = fit_logistic(train, [0, 0, 0, 1, 1, 1], [[-1e200], [1e200]])
        np.testing.assert_array_equal(predicted, [0.0, 1.0])
        measured = binary_metrics([1, 0], predicted)
    assert np.isfinite(measured["log_loss"])
    assert measured["log_loss"] > 30
    assert measured["brier"] == 1
    assert measured["auroc"] == 0


def test_auroc_average_precision_and_tied_scores_have_known_values():
    measured = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert measured["auroc"] == pytest.approx(0.75)
    assert measured["auprc"] == pytest.approx(5 / 6)
    tied = binary_metrics([0, 1, 0, 1], [0.5] * 4)
    assert tied["auroc"] == 0.5 and tied["auprc"] == 0.5
    permuted = binary_metrics([1, 0, 1, 0], [0.5] * 4)
    assert tied == permuted


def test_binary_metrics_and_mae_match_direct_calculations():
    labels, scores = np.array([0, 1, 1, 0]), np.array([0.2, 0.6, 0.8, 0.1])
    result = binary_metrics(labels, scores)
    assert result["log_loss"] == pytest.approx(-np.log([0.8, 0.6, 0.8, 0.9]).mean())
    assert result["brier"] == pytest.approx(np.mean((scores - labels)**2))
    assert result["num_samples"] == 4
    assert regression_metrics([1, 2, 3], [0, 2, 5]) == {"mae": 1.0, "num_samples": 3}


def test_intercept_only_probe_and_empty_test_matrix():
    empty_features = np.empty((4, 0))
    np.testing.assert_allclose(fit_ridge(empty_features, [1, 2, 3, 4], np.empty((2, 0))), [2.5, 2.5])
    np.testing.assert_allclose(fit_logistic(empty_features, [0, 1, 1, 1], np.empty((2, 0))), [0.75, 0.75])
    assert fit_ridge([[0], [1]], [0, 1], np.empty((0, 1))).shape == (0,)
    assert fit_logistic([[0], [1]], [0, 1], np.empty((0, 1))).shape == (0,)


def test_bootstrap_resamples_entire_scenes_with_original_episode_weights():
    first = np.array([0.0, 0.0, 0.0, 10.0, 30.0, 30.0])
    second = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    scans = ["a", "a", "a", "b", "c", "c"]
    result = paired_cluster_bootstrap(first, second, scans, n_bootstrap=500, seed=9)
    # Independently reconstruct sampled rows, including duplicates of ALL rows
    # within each chosen house, rather than resampling individual episodes.
    members = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5])]
    rng = np.random.default_rng(9)
    draws = []
    for _ in range(500):
        scene_indices = rng.integers(0, 3, size=3)
        rows = np.concatenate([members[scene] for scene in scene_indices])
        draws.append(np.mean(first[rows] - second[rows]))
    assert result["difference_a_minus_b"] == pytest.approx(np.mean(first - second))
    np.testing.assert_array_equal(result["ci95"], np.quantile(draws, [0.025, 0.975]))
    assert result["num_clusters"] == 3 and result["num_samples"] == 6
    assert result == paired_cluster_bootstrap(first, second, scans, n_bootstrap=500, seed=9)


def test_paired_bootstrap_retains_pairing_and_difference_sign():
    values = np.array([50.0, -5.0, 99.0, 0.0])
    result = paired_cluster_bootstrap(values, values + 3, ["a", "a", "b", "b"], 100, seed=2)
    assert result["difference_a_minus_b"] == -3
    assert result["ci95"] == [-3.0, -3.0]


def test_scene_folds_are_disjoint_deterministic_and_cover_every_test_row():
    scans = np.array(["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2 + ["e", "f"])
    splits = group_kfold(scans, folds=3, seed=12)
    seen = []
    repeated = group_kfold(scans, folds=3, seed=12)
    for (train, test), (train_again, test_again) in zip(splits, repeated):
        assert len(train) > 0 and len(test) > 0
        assert not set(scans[train]) & set(scans[test])
        assert set(train) | set(test) == set(range(len(scans)))
        assert not set(train) & set(test)
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(test, test_again)
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(len(scans)))


def test_scene_held_out_ridge_generalization():
    rng = np.random.default_rng(33)
    scans = np.repeat([f"scene-{index}" for index in range(9)], 10)
    inputs = rng.normal(size=(len(scans), 3))
    targets = 2 * inputs[:, 0] - 3 * inputs[:, 1] + 0.7 * inputs[:, 2] + 8
    predictions = np.full(len(scans), np.nan)
    for train, test in group_kfold(scans, folds=3):
        predictions[test] = fit_ridge(inputs[train], targets[train], inputs[test], alpha=0)
    assert regression_metrics(targets, predictions)["mae"] < 1e-12


@pytest.mark.parametrize("case", ["nan", "dimensions", "negative_alpha", "wrong_targets", "no_convergence"])
def test_probe_invalid_inputs_are_rejected(case):
    train = [[-1.0], [0.0], [1.0]]
    with pytest.raises((ValueError, RuntimeError)):
        if case == "nan":
            fit_ridge([[float("nan")], [0], [1]], [0, 0, 1], [[0]])
        elif case == "dimensions":
            fit_ridge(train, [0, 0, 1], [[0, 1]])
        elif case == "negative_alpha":
            fit_logistic(train, [0, 0, 1], [[0]], alpha=-1)
        elif case == "wrong_targets":
            fit_logistic(train, [0, 0.2, 1], [[0]])
        elif case == "no_convergence":
            fit_logistic(train, [0, 0, 1], [[0]], max_iter=1)


@pytest.mark.parametrize("labels,scores", [([0, 1], [0.0, float("nan")]),
                                           ([0, 1], [0.0, 1.1]), ([0, 2], [0.1, 0.8]),
                                           ([0, 1], [0.5]), ([], [])])
def test_invalid_binary_metrics_rejected(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)


def test_insufficient_scene_clusters_and_invalid_fold_count_are_rejected():
    with pytest.raises(ValueError, match="two clusters"):
        paired_cluster_bootstrap([0, 1], [0, 1], ["same", "same"])
    with pytest.raises(ValueError, match="number of scenes"):
        group_kfold(["a", "a", "b"], folds=3)
    with pytest.raises(ValueError, match="scan_ids"):
        group_kfold(["a", None], folds=2)
