import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.analyze_diagnostics import GROUPS, TARGETS, analyze, main, read_jsonl


def rows(split, scenes, *, seed, rows_per_scene=16):
    rng = np.random.default_rng(seed)
    result = []
    for scene in scenes:
        for index in range(rows_per_scene):
            score, evidence = rng.normal(size=2)
            teacher = max(0.0, score + 1.5 * evidence)
            execution = max(0.0, score + 1.8 * evidence + 0.2)
            result.append({
                "split": split, "arrival_kind": "natural", "scan_id": scene, "instr_id": f"{scene}-instruction-{index}",
                "step": index % 4, "target_id": f"node-{index}",
                "p0": {"base_score": float(score), "constant_control": 1.0},
                "p1": {"evidence_statistic": float(evidence)},
                "labels": {"teacher_regret": teacher, "execution_regret": execution,
                           "teacher_optimal": int(teacher == 0), "execution_optimal": int(execution == 0),
                           "base_chosen": int(score > 0)},
                "interventions": {
                    "arrival": {"delta_margin": float(-2 * evidence + 0.1 * score)},
                    "arrival_normmatched": {"delta_margin": float(-1.8 * evidence)},
                    "shuffled_arrival": {"delta_margin": float(rng.normal())},
                    "noise_normmatched": {"delta_margin": float(rng.normal())},
                    "last_source": {"delta_margin": 0.0},
                },
            })
    return result


@pytest.fixture
def data():
    return (rows("train_fit", ["train-a", "train-b", "train-c", "train-d"], seed=1),
            rows("train_dev", ["dev-a", "dev-b", "dev-c"], seed=2))


def test_report_contains_all_fixed_probes_and_scene_bootstrap(data):
    train, dev = data
    report = analyze(train, dev, n_bootstrap=30, seed=9)
    assert report["claims_navigation_improvement"] is False
    assert report["protocol"]["alpha"] == 1
    assert report["protocol"]["official_validation_used"] is False
    assert report["protocol"]["bootstrap_unit"] == "scan"
    assert report["data"]["train_rows"] == 64 and report["data"]["dev_rows"] == 48
    assert set(report["development_results"]) == set(TARGETS)
    for target, result in report["development_results"].items():
        assert set(result) == set(GROUPS)
        compared = report["paired_loss_comparisons"][target]["P0_P1"]
        assert compared["reference"] == "P0"
        assert compared["num_clusters"] == 3 and compared["n_bootstrap"] == 30
        assert len(compared["ci95"]) == 2
        assert "P3_vs_P0_P1" in report["paired_loss_comparisons"][target]
    # Synthetic signal genuinely generalizes to independent development scenes.
    classification = report["development_results"]["teacher_optimal"]
    assert classification["P0_P1"]["metrics"]["log_loss"] < classification["P0"]["metrics"]["log_loss"]
    assert report["paired_loss_comparisons"]["teacher_optimal"]["P0_P1"]["difference_a_minus_b"] < 0
    assert any("selected" in limitation for limitation in report["limitations"])
    assert any("do not demonstrate" in limitation for limitation in report["limitations"])
    assert report["shuffled_control"]["status"] == "complete"


def test_p3_train_cross_fitting_never_fits_on_its_held_out_scene(data):
    train, dev = data
    report = analyze(train, dev, n_bootstrap=10)
    held_scenes = []
    for fold in report["p3_delta_prediction"]["folds"]:
        assert not set(fold["fit_scans"]) & set(fold["held_out_scans"])
        assert fold["fit_rows"] + fold["held_out_rows"] == len(train)
        held_scenes.extend(fold["held_out_scans"])
    assert sorted(held_scenes) == sorted({row["scan_id"] for row in train})
    changed = copy.deepcopy(train)
    changed_scene = changed[0]["scan_id"]
    for row in changed:
        if row["scan_id"] == changed_scene:
            row["interventions"]["arrival"]["delta_margin"] += 100
    alternate = analyze(changed, dev, n_bootstrap=10)
    own_scene = [index for index, row in enumerate(train) if row["scan_id"] == changed_scene]
    original_predictions = np.array(report["p3_delta_prediction"]["train_oof_predictions"])
    changed_predictions = np.array(alternate["p3_delta_prediction"]["train_oof_predictions"])
    np.testing.assert_array_equal(original_predictions[own_scene], changed_predictions[own_scene])


def test_development_arrival_labels_never_enter_p3_predictions(data):
    train, dev = data
    first = analyze(train, dev, n_bootstrap=10)
    altered = copy.deepcopy(dev)
    for row in altered:
        row["interventions"]["arrival"]["delta_margin"] += 1000
        row["interventions"]["shuffled_arrival"]["delta_margin"] -= 1000
    second = analyze(train, altered, n_bootstrap=10)
    assert first["p3_delta_prediction"]["dev_predictions"] == second["p3_delta_prediction"]["dev_predictions"]
    assert first["p3_delta_prediction"]["dev_metrics"] != second["p3_delta_prediction"]["dev_metrics"]
    for target in TARGETS:
        for group in ("P0", "P0_P1", "P3_predicted_arrival_delta"):
            assert first["development_predictions"][target][group] == second["development_predictions"][target][group]


def test_development_targets_are_evaluation_only(data):
    train, dev = data
    original = analyze(train, dev, n_bootstrap=10)
    altered = copy.deepcopy(dev)
    for row in altered:
        row["labels"]["teacher_optimal"] = 1 - row["labels"]["teacher_optimal"]
        row["labels"]["teacher_regret"] += 50
    changed = analyze(train, altered, n_bootstrap=10)
    assert original["development_predictions"] == changed["development_predictions"]
    assert original["development_results"] != changed["development_results"]


@pytest.mark.parametrize("mutation,match", [
    ("overlap", "overlap"), ("official_validation", "official validation"),
    ("missing_split", "split"), ("schema", "columns"),
    ("duplicate", "duplicate"), ("nan", "finite"), ("missing_intervention", "intervention"),
])
def test_invalid_or_leaking_analysis_inputs_rejected(data, mutation, match):
    train, dev = copy.deepcopy(data)
    if mutation == "overlap":
        dev[0]["scan_id"] = train[0]["scan_id"]
    elif mutation == "official_validation":
        dev[0]["split"] = "val_unseen"
    elif mutation == "missing_split":
        del dev[0]["split"]
    elif mutation == "schema":
        dev[0]["p1"]["extra_column"] = 1
    elif mutation == "duplicate":
        dev.append(copy.deepcopy(dev[0]))
    elif mutation == "nan":
        dev[0]["interventions"]["arrival"]["delta_margin"] = float("nan")
    elif mutation == "missing_intervention":
        del dev[0]["interventions"]["noise_normmatched"]
    with pytest.raises(ValueError, match=match):
        analyze(train, dev, n_bootstrap=10)


def test_one_development_scene_reports_unavailable_interval(data):
    train, dev = data
    subset = [row for row in dev if row["scan_id"] == "dev-a"]
    report = analyze(train, subset, n_bootstrap=20)
    interval = report["paired_loss_comparisons"]["teacher_regret"]["P0_P1"]
    assert interval["ci95"] is None
    assert interval["num_clusters"] == 1 and interval["n_bootstrap"] == 0
    assert "two" in interval["reason"]


def test_one_training_scene_cannot_silently_skip_cross_fitting(data):
    train, dev = data
    with pytest.raises(ValueError, match="at least two training scenes"):
        analyze([row for row in train if row["scan_id"] == "train-a"], dev, n_bootstrap=10)


def test_jsonl_cli_saves_reproducible_input_hashes_and_finite_json(data, tmp_path):
    train, dev = data
    train_path, dev_path, output = tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "report.json"
    train_path.write_text("\n".join(json.dumps(row) for row in train) + "\n")
    dev_path.write_text("\n".join(json.dumps(row) for row in dev) + "\n")
    assert len(read_jsonl(train_path)) == len(train)
    main(["--train", str(train_path), "--dev", str(dev_path), "--output", str(output), "--n-bootstrap", "20"])
    report = json.loads(output.read_text())
    assert report["input_files"]["train"]["path"] == str(train_path.resolve())
    assert len(report["input_files"]["train"]["sha256"]) == 64
    provenance = report["analysis_provenance"]
    assert provenance["python_version"] == sys.version.split()[0]
    assert provenance["numpy_version"] == np.__version__
    for source in provenance["source_files"].values():
        assert source["sha256"] == hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
    assert set(provenance["source_files"]) == {"analyzer", "probes"}
    assert report["claims_navigation_improvement"] is False
    assert "NaN" not in output.read_text()


def test_missing_shuffled_controls_do_not_remove_main_analysis_rows(data):
    train, dev = data
    reference = analyze(train, dev, n_bootstrap=10)
    train, dev = copy.deepcopy(data)
    for index, row in enumerate(train + dev):
        if index % 2:
            del row["interventions"]["shuffled_arrival"]
    report = analyze(train, dev, n_bootstrap=10)
    assert report["data"]["train_rows"] == len(train) and report["data"]["dev_rows"] == len(dev)
    assert report["development_results"] == reference["development_results"]
    assert report["development_predictions"] == reference["development_predictions"]
    subset = report["shuffled_control"]
    assert subset["status"] == "complete"
    assert subset["data"]["train_row_coverage"] == 0.5
    assert subset["data"]["dev_row_coverage"] == 0.5
    # Independent main analysis on just the donor subset reconstructs the
    # correct matched P0. It differs from the full-cohort P0.
    subtrain = [row for row in train if "shuffled_arrival" in row["interventions"]]
    subdev = [row for row in dev if "shuffled_arrival" in row["interventions"]]
    independently_fitted = analyze(subtrain, subdev, n_bootstrap=10)
    for target in TARGETS:
        assert subset["development_predictions"][target]["P0"] == independently_fitted["development_predictions"][target]["P0"]
        assert subset["paired_loss_comparisons"][target]["shuffled_vs_P0"]["reference"] == "P0"
        assert subset["paired_loss_comparisons"][target]["shuffled_vs_P0"]["cohort"] == "same_shuffled_subset"
        for group in ("P0_true_arrival_delta", "P0_normmatched_arrival_delta"):
            assert subset["development_predictions"][target][group] == independently_fitted["development_predictions"][target][group]


def test_absent_shuffled_controls_report_insufficient_data_without_imputation(data):
    train, dev = copy.deepcopy(data)
    for row in train + dev:
        del row["interventions"]["shuffled_arrival"]
    report = analyze(train, dev, n_bootstrap=10)
    subset = report["shuffled_control"]
    assert subset["status"] == "insufficient_data"
    assert subset["data"]["train_rows"] == subset["data"]["dev_rows"] == 0
    assert "development_predictions" not in subset
    assert len(report["development_predictions"]["teacher_optimal"]["P0"]) == len(dev)


def test_single_class_shuffled_subset_skips_only_its_classification_target(data):
    train, dev = copy.deepcopy(data)
    for row in train + dev:
        if row["labels"]["teacher_optimal"] != 1:
            del row["interventions"]["shuffled_arrival"]
    report = analyze(train, dev, n_bootstrap=10)
    subset = report["shuffled_control"]
    assert subset["status"] == "partial"
    assert subset["development_results"]["teacher_regret"]["status"] == "complete"
    skipped = subset["development_results"]["teacher_optimal"]
    assert skipped["status"] == "insufficient_data"
    assert skipped["class_counts"]["train_negative"] == 0
    assert "teacher_optimal" not in subset["development_predictions"]
    assert report["development_results"]["teacher_optimal"]["P0"]["metrics"]["auroc"] is not None


def test_normmatched_reference_is_added_without_changing_p3_target(data):
    train, dev = data
    original = analyze(train, dev, n_bootstrap=10)
    assert original["schema_version"] == 3
    assert original["protocol"]["p3_prediction_target"] == "interventions.arrival.delta_margin"
    assert original["p3_delta_prediction"]["target"] == "interventions.arrival.delta_margin"
    altered_train, altered_dev = copy.deepcopy(data)
    for row in altered_train + altered_dev:
        row["interventions"]["arrival_normmatched"]["delta_margin"] = 0.0
    altered = analyze(altered_train, altered_dev, n_bootstrap=10)
    assert original["p3_delta_prediction"] == altered["p3_delta_prediction"]
    for target in TARGETS:
        assert original["development_predictions"][target]["P3_predicted_arrival_delta"] == altered["development_predictions"][target]["P3_predicted_arrival_delta"]
        assert original["development_predictions"][target]["P0_true_arrival_delta"] == altered["development_predictions"][target]["P0_true_arrival_delta"]
        assert original["development_predictions"][target]["P0_normmatched_arrival_delta"] != altered["development_predictions"][target]["P0_normmatched_arrival_delta"]
        assert "normmatched_vs_true_arrival" in original["paired_loss_comparisons"][target]
        assert "normmatched_vs_noise" in original["paired_loss_comparisons"][target]


def test_matched_subset_true_and_normmatched_controls_have_paired_shuffled_comparisons(data):
    from vln_improve.probes import paired_cluster_bootstrap

    train, dev = copy.deepcopy(data)
    for index, row in enumerate(train + dev):
        if index % 3 == 0:
            del row["interventions"]["shuffled_arrival"]
    report = analyze(train, dev, n_bootstrap=20, seed=9)
    subset = report["shuffled_control"]
    subset_dev = [row for row in dev if "shuffled_arrival" in row["interventions"]]
    labels = np.array([row["labels"]["teacher_regret"] for row in subset_dev])
    predictions = subset["development_predictions"]["teacher_regret"]
    shuffled_loss = np.abs(labels - np.array(predictions["P0_shuffled_arrival_delta"]))
    for comparison_name, group in (("true_arrival_vs_shuffled", "P0_true_arrival_delta"),
                                    ("normmatched_arrival_vs_shuffled", "P0_normmatched_arrival_delta")):
        actual = subset["paired_loss_comparisons"]["teacher_regret"][comparison_name]
        expected = paired_cluster_bootstrap(np.abs(labels - np.array(predictions[group])), shuffled_loss,
                                            [row["scan_id"] for row in subset_dev], n_bootstrap=20, seed=9)
        assert actual["cohort"] == "same_shuffled_subset"
        assert actual["reference"] == "P0_shuffled_arrival_delta"
        assert actual["candidate"] == group
        for key in expected:
            assert actual[key] == expected[key]
        assert actual["num_samples"] == len(subset_dev)
        assert actual["num_samples"] < len(dev)


@pytest.mark.parametrize("split", ["train", "dev"])
@pytest.mark.parametrize("target", ["teacher_optimal", "execution_optimal"])
@pytest.mark.parametrize("constant", [0, 1])
def test_single_class_main_target_is_explicitly_unavailable(data, split, target, constant):
    train, dev = copy.deepcopy(data)
    changed = train if split == "train" else dev
    for row in changed:
        row["labels"][target] = constant
    report = analyze(train, dev, n_bootstrap=10)
    assert report["status"] == "partial"
    status = report["target_status"][target]
    assert status["status"] == "insufficient_data"
    assert status["class_counts"][f"{split}_positive"] == constant * len(changed)
    assert status["class_counts"][f"{split}_negative"] == (1 - constant) * len(changed)
    for group in GROUPS:
        value = report["development_results"][target][group]
        assert value["status"] == "insufficient_data"
        assert value["metrics"] is None
        assert value["class_counts"] == status["class_counts"]
    assert report["development_predictions"][target] == {}
    assert report["paired_loss_comparisons"][target] == {}
    # Other targets and cross-fitted delta learning continue normally.
    assert report["target_status"]["teacher_regret"]["status"] == "complete"
    assert len(report["development_predictions"]["teacher_regret"]["P0"]) == len(dev)
    assert len(report["p3_delta_prediction"]["dev_predictions"]) == len(dev)


def test_missing_normmatched_intervention_cannot_silently_drop_new_reference(data):
    train, dev = copy.deepcopy(data)
    del dev[0]["interventions"]["arrival_normmatched"]
    with pytest.raises(ValueError, match="arrival_normmatched"):
        analyze(train, dev, n_bootstrap=10)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "mixed_train", "mixed_dev", "different_splits"])
def test_ambiguous_or_mixed_arrival_cohorts_are_rejected(data, mutation):
    train, dev = copy.deepcopy(data)
    if mutation == "missing":
        del train[0]["arrival_kind"]
    elif mutation == "unknown":
        dev[0]["arrival_kind"] = "shadow_adjacent"
    elif mutation == "mixed_train":
        train[0]["arrival_kind"] = "simulated_branch"
    elif mutation == "mixed_dev":
        dev[0]["arrival_kind"] = "simulated_branch"
    else:
        for row in dev:
            row["arrival_kind"] = "simulated_branch"
    with pytest.raises(ValueError, match="arrival_kind"):
        analyze(train, dev, n_bootstrap=10)


def test_branch_reports_its_actual_selection_bias_and_preserves_all_probe_definitions(data):
    train, dev = data
    natural = analyze(train, dev, n_bootstrap=10)
    train, dev = copy.deepcopy(data)
    for row in train + dev:
        row["arrival_kind"] = "simulated_branch"
    branch = analyze(train, dev, n_bootstrap=10)
    assert natural["kind"] == "offline_natural_arrival_diagnostics"
    assert branch["kind"] == "offline_simulated_branch_arrival_diagnostics"
    assert branch["protocol"]["arrival_kind"] == "simulated_branch"
    assert branch["protocol"]["arrival_orientation"] == "legal_execution_path_arrival_heading_and_elevation"
    assert any("baseline state and candidate distribution" in item for item in branch["limitations"])
    assert not any("Only naturally reached" in item for item in branch["limitations"])
    for key in ("development_predictions", "development_results", "paired_loss_comparisons", "p3_delta_prediction", "shuffled_control"):
        assert branch[key] == natural[key]
    assert branch["protocol"]["alpha"] == 1
    assert branch["protocol"]["p3_prediction_target"] == "interventions.arrival.delta_margin"


def test_optional_heading_reference_uses_one_matched_subset_and_never_changes_main_fit(data):
    from vln_improve.probes import paired_cluster_bootstrap

    train, dev = copy.deepcopy(data)
    baseline = analyze(train, dev, n_bootstrap=10)
    assert baseline["heading_control"]["status"] == "not_provided"
    for index, row in enumerate(train + dev):
        if index % 2 == 0:
            row["interventions"]["arrival_headingmatched"] = {
                "delta_margin": row["interventions"]["arrival"]["delta_margin"] + row["p0"]["base_score"]}
    report = analyze(train, dev, n_bootstrap=10)
    heading = report["heading_control"]
    assert heading["status"] == "complete"
    assert heading["data"]["train_row_coverage"] == heading["data"]["dev_row_coverage"] == 0.5
    for key in ("development_predictions", "development_results", "paired_loss_comparisons", "p3_delta_prediction", "shuffled_control"):
        assert report[key] == baseline[key]
    subtrain = [row for row in train if "arrival_headingmatched" in row["interventions"]]
    subdev = [row for row in dev if "arrival_headingmatched" in row["interventions"]]
    independently_fitted = analyze(subtrain, subdev, n_bootstrap=10)
    for target in TARGETS:
        for group in ("P0", "P0_true_arrival_delta"):
            assert heading["development_predictions"][target][group] == independently_fitted["development_predictions"][target][group]
    prediction = heading["development_predictions"]["teacher_regret"]
    y = np.array([row["labels"]["teacher_regret"] for row in subdev])
    expected = paired_cluster_bootstrap(np.abs(y - np.array(prediction["P0_headingmatched_arrival_delta"])),
                                        np.abs(y - np.array(prediction["P0_true_arrival_delta"])),
                                        [row["scan_id"] for row in subdev], n_bootstrap=10, seed=0)
    comparison = heading["paired_loss_comparisons"]["teacher_regret"]["headingmatched_vs_true_arrival"]
    assert comparison["cohort"] == "same_heading_subset"
    assert comparison["num_samples"] == len(subdev)
    for key, value in expected.items():
        assert comparison[key] == value


def test_development_heading_values_are_never_p3_inputs(data):
    train, dev = copy.deepcopy(data)
    for row in train + dev:
        row["interventions"]["arrival_headingmatched"] = copy.deepcopy(row["interventions"]["arrival"])
    original = analyze(train, dev, n_bootstrap=10)
    for row in dev:
        row["interventions"]["arrival_headingmatched"]["delta_margin"] += 1000
    changed = analyze(train, dev, n_bootstrap=10)
    assert changed["p3_delta_prediction"] == original["p3_delta_prediction"]
    assert changed["development_predictions"] == original["development_predictions"]
    assert changed["heading_control"]["development_predictions"] != original["heading_control"]["development_predictions"]


def test_heading_subset_class_insufficiency_is_explicit_and_does_not_stop_other_targets(data):
    train, dev = copy.deepcopy(data)
    for row in train + dev:
        if row["labels"]["teacher_optimal"] == 1:
            row["interventions"]["arrival_headingmatched"] = copy.deepcopy(row["interventions"]["arrival"])
    heading = analyze(train, dev, n_bootstrap=10)["heading_control"]
    assert heading["status"] == "partial"
    target = heading["development_results"]["teacher_optimal"]
    assert target["status"] == "insufficient_data"
    assert target["class_counts"]["train_negative"] == target["class_counts"]["dev_negative"] == 0
    assert "teacher_optimal" not in heading["development_predictions"]
    assert heading["development_results"]["teacher_regret"]["status"] == "complete"


def test_invalid_heading_reference_is_rejected(data):
    train, dev = copy.deepcopy(data)
    dev[0]["interventions"]["arrival_headingmatched"] = {"delta_margin": float("nan")}
    with pytest.raises(ValueError, match="arrival_headingmatched.*finite"):
        analyze(train, dev, n_bootstrap=10)


def test_selection_coverage_distinguishes_choice_source_count_and_scene_class_counts(data):
    train, dev = copy.deepcopy(data)
    for index, row in enumerate(train + dev):
        row["p0"]["coverage_source_count_total"] = 1 + index % 3
    report = analyze(train, dev, n_bootstrap=10)
    for split, samples in (("train", train), ("dev", dev)):
        coverage = report["data"]["selection_coverage"][split]
        assert coverage["rows_without_source_count"] == 0
        assert sum(cell["rows"] for cell in coverage["strata"].values()) == len(samples)
        chosen_multi = [row for row in samples if row["labels"]["base_chosen"] and row["p0"]["coverage_source_count_total"] > 1]
        assert coverage["strata"]["chosen_multi_source"]["rows"] == len(chosen_multi)
        negative = sum(item["teacher_optimal"]["negative"] for item in coverage["per_scan_class_counts"].values())
        assert negative == sum(row["labels"]["teacher_optimal"] == 0 for row in samples)


def test_symmetric_shuffled_controls_are_optional_for_existing_natural_reports(data):
    report = analyze(*data, n_bootstrap=10)
    for variant in ("normmatched", "headingmatched"):
        control = report[f"{variant}_shuffled_control"]
        assert control["status"] == "not_provided"
        assert control["data"]["train_rows"] == control["data"]["dev_rows"] == 0
        assert "development_predictions" not in control
    assert report["shuffled_control"]["status"] == "complete"


@pytest.mark.parametrize("variant", ["normmatched", "headingmatched"])
def test_symmetric_shuffled_references_use_identical_rows_for_fit_and_comparison(data, variant):
    from vln_improve.probes import fit_logistic, fit_ridge, paired_cluster_bootstrap

    train, dev = copy.deepcopy(data)
    original = analyze(train, dev, n_bootstrap=10)
    true_key, shuffle_key = f"arrival_{variant}", f"shuffled_{variant}"
    for index, row in enumerate(train + dev):
        # Unequal availability must lead to an intersection, never imputation.
        if variant == "headingmatched" and index % 3:
            row["interventions"][true_key] = {"delta_margin": row["interventions"]["arrival"]["delta_margin"] + 0.2}
        if index % 2 == 0:
            row["interventions"][shuffle_key] = {"delta_margin": row["interventions"]["shuffled_arrival"]["delta_margin"] - 0.3}
    report = analyze(train, dev, n_bootstrap=10)
    control = report[f"{variant}_shuffled_control"]
    assert control["status"] == "complete"
    for key in ("development_predictions", "development_results", "paired_loss_comparisons", "p3_delta_prediction", "shuffled_control"):
        assert report[key] == original[key]
    selected_train = [row for row in train if true_key in row["interventions"] and shuffle_key in row["interventions"]]
    selected_dev = [row for row in dev if true_key in row["interventions"] and shuffle_key in row["interventions"]]
    assert control["data"]["train_rows"] == len(selected_train) < len(train)
    assert control["data"]["dev_rows"] == len(selected_dev) < len(dev)
    assert control["data"]["dev_candidate_keys"] == [
        {key: row[key] for key in ("scan_id", "instr_id", "step", "target_id")} for row in selected_dev]
    columns = sorted(train[0]["p0"])
    train_p0 = np.array([[row["p0"][name] for name in columns] for row in selected_train])
    dev_p0 = np.array([[row["p0"][name] for name in columns] for row in selected_dev])
    true_group, shuffle_group = f"P0_{variant}_arrival_delta", f"P0_shuffled_{variant}_delta"
    for target, kind in TARGETS.items():
        train_y = np.array([row["labels"][target] for row in selected_train])
        dev_y = np.array([row["labels"][target] for row in selected_dev])
        fit = fit_ridge if kind == "regression" else fit_logistic
        losses = {}
        for group, intervention in (("P0", None), (true_group, true_key), (shuffle_group, shuffle_key)):
            X_train, X_dev = train_p0, dev_p0
            if intervention:
                X_train = np.column_stack((train_p0, [row["interventions"][intervention]["delta_margin"] for row in selected_train]))
                X_dev = np.column_stack((dev_p0, [row["interventions"][intervention]["delta_margin"] for row in selected_dev]))
            prediction = fit(X_train, train_y, X_dev, alpha=1.0)
            np.testing.assert_array_equal(control["development_predictions"][target][group], prediction)
            if kind == "regression":
                losses[group] = np.abs(dev_y - prediction)
            else:
                p = np.clip(prediction, 1e-15, 1 - 1e-15)
                losses[group] = -(dev_y * np.log(p) + (1 - dev_y) * np.log1p(-p))
        expected = paired_cluster_bootstrap(losses[true_group], losses[shuffle_group],
                                            [row["scan_id"] for row in selected_dev], n_bootstrap=10, seed=0)
        comparison = control["paired_loss_comparisons"][target]["true_vs_shuffled"]
        assert comparison["candidate"] == true_group and comparison["reference"] == shuffle_group
        assert comparison["cohort"] == f"same_{variant}_shuffled_subset"
        for key, value in expected.items():
            assert comparison[key] == value


@pytest.mark.parametrize("variant", ["normmatched", "headingmatched"])
def test_symmetric_shuffled_single_class_subset_skips_only_affected_target(data, variant):
    train, dev = copy.deepcopy(data)
    for row in train + dev:
        if row["labels"]["teacher_optimal"] == 1:
            row["interventions"][f"arrival_{variant}"] = copy.deepcopy(row["interventions"]["arrival"])
            row["interventions"][f"shuffled_{variant}"] = copy.deepcopy(row["interventions"]["shuffled_arrival"])
    report = analyze(train, dev, n_bootstrap=10)
    control = report[f"{variant}_shuffled_control"]
    assert control["status"] == "partial"
    target = control["development_results"]["teacher_optimal"]
    assert target["status"] == "insufficient_data"
    assert target["class_counts"]["train_negative"] == target["class_counts"]["dev_negative"] == 0
    assert "teacher_optimal" not in control["development_predictions"]
    assert control["development_results"]["teacher_regret"]["status"] == "complete"
    assert report["target_status"]["teacher_optimal"]["status"] == "complete"


@pytest.mark.parametrize("variant", ["normmatched", "headingmatched"])
def test_symmetric_shuffled_missing_training_controls_is_not_fabricated(data, variant):
    train, dev = copy.deepcopy(data)
    for row in dev:
        row["interventions"][f"arrival_{variant}"] = copy.deepcopy(row["interventions"]["arrival"])
        row["interventions"][f"shuffled_{variant}"] = copy.deepcopy(row["interventions"]["shuffled_arrival"])
    control = analyze(train, dev, n_bootstrap=10)[f"{variant}_shuffled_control"]
    assert control["status"] == "insufficient_data"
    assert control["data"]["train_rows"] == 0
    assert control["data"]["dev_rows"] == len(dev)
    assert "development_predictions" not in control


@pytest.mark.parametrize("intervention", ["shuffled_normmatched", "shuffled_headingmatched", "arrival_headingmatched_normmatched"])
def test_optional_symmetric_control_values_must_be_finite(data, intervention):
    train, dev = copy.deepcopy(data)
    dev[0]["interventions"][intervention] = {"delta_margin": float("nan")}
    with pytest.raises(ValueError, match=f"{intervention}.*finite"):
        analyze(train, dev, n_bootstrap=10)
