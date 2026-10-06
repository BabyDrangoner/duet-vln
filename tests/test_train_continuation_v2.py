"""CPU acceptance of sampling, causal selection, and durable E3-v2 training.

All data are synthetic. No navigation benchmark or GPU access is performed.
"""
from __future__ import annotations

import copy
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from scripts import train_continuation_v2 as training
from vln_improve.checkpoint_store import CheckpointStore
from vln_improve.continuation_learning import record_loss
from vln_improve.continuation_v2 import SCHEMA as DATA_SCHEMA, SCHEDULES, save_bundle
from vln_improve.protocol import file_sha256


@pytest.fixture(autouse=True)
def cpu_single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_record(instr_id="fit-0", scan_id="fit-house", condition="natural", step=1,
                baseline=(1.0, .7), outcomes=None, scores=None, teacher_target=1):
    generator = torch.Generator().manual_seed(100 + step)
    features = torch.randn(3, 7, generator=generator)
    features[:, :2] = torch.tensor(scores if scores is not None else
                                  [[0., 0.], [.3, .2], [.1, -.2]])
    if outcomes is None:
        outcomes = [baseline, (1., .9), (0., 0.)]
    return {"features": features,
            "history_features": torch.randn(2 * step + 1, 7, generator=generator),
            "progress": torch.tensor([step / 14, (15 - step) / 15, .01, .75]),
            "utilities": torch.tensor(outcomes, dtype=torch.float32),
            "teacher_target": teacher_target, "candidate_actions": ["keep", "rescue", None],
            "instr_id": instr_id, "scan_id": scan_id, "condition": condition, "step": step}


def make_dataset(root: Path, split="train_fit", *, instructions=2, states=2):
    prefix = "fit" if split == "train_fit" else "dev"
    bundles = []
    for index in range(instructions):
        for condition in SCHEDULES:
            baseline = (1., .7) if condition == "natural" else (0., 0.)
            identity = dict(instr_id=f"{prefix}-{index}", scan_id=f"{prefix}-house-{index}",
                            condition=condition)
            records = [make_record(**identity, step=step, baseline=baseline)
                       for step in range(1, states + 1)]
            branches = [{**identity, "target_step": row["step"], "target_action": action,
                         "metrics": {"success": outcome[0], "spl": outcome[1]},
                         "is_anchor": candidate == 0}
                        for row in records
                        for candidate, (action, outcome) in enumerate(zip(
                            row["candidate_actions"], [baseline, (1., .9), (0., 0.)]))]
            bundles.append({"reference": {**identity, "metrics": {
                "success": baseline[0], "spl": baseline[1]}}, "records": records,
                "branches": branches})
    selection = {"split": split, "instr_ids": [f"{prefix}-{i}" for i in range(instructions)],
                 "scan_ids": [f"{prefix}-house-{i}" for i in range(instructions)],
                 "conditions": list(SCHEDULES), "count": instructions, "seed": 0, "smoke": True}
    manifest = {"selection": selection, "provenance": {"schema": "synthetic-v2",
                "base_checkpoint_sha256": "a" * 64, "feature_sha256": "b" * 64}}
    return training.Dataset(root=root, manifest=manifest, bundles=bundles,
                            identity={"split": split, "manifest_sha256": prefix + "-fixture-v1"})


def set_branch_outcome(bundle, row, candidate, outcome):
    """Keep the training tensor and original-precision replay metric paired."""
    row["utilities"][candidate] = torch.tensor(outcome, dtype=row["utilities"].dtype)
    matching = [branch for branch in bundle["branches"]
                if (branch["target_step"], branch["target_action"])
                == (row["step"], row["candidate_actions"][candidate])]
    assert len(matching) == 1
    matching[0]["metrics"] = {"success": float(outcome[0]), "spl": float(outcome[1])}


class FeatureScores(torch.nn.Module):
    """An observable deterministic policy: outcomes cannot affect its scores."""

    def __init__(self, mode="relative"):
        super().__init__()
        self.mode = mode
        self.calls = []

    def forward(self, features, history_features, progress):
        self.calls.append((features.detach().clone(), history_features.detach().clone(),
                           progress.detach().clone()))
        return features[:, :2]

    def score_record(self, record):
        return self(record["features"], record["history_features"], record["progress"])


def assert_tree_equal(actual, expected, *, rtol=0, atol=0):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_equal(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_tree_equal(left, right, rtol=rtol, atol=atol)
    else:
        assert actual == expected


def small_config(**overrides):
    config = training.default_config()
    config.update(epochs=2, candidate_epochs=[1, 2], batch_size=2, hidden_dim=8,
                  feature_dim=7, checkpoint_every_steps=1, device="cpu")
    config.update(overrides)
    return config


def restore_latest(local, backup):
    return CheckpointStore(local, backup).restore("latest")[0]


def test_epoch_plan_weights_original_instructions_equally_and_covers_every_record():
    records = [make_record(instr_id="a", step=1),
               *[make_record(instr_id="b", step=i) for i in (1, 2, 3)],
               *[make_record(instr_id="c", step=i) for i in (1, 2)]]
    plan = training.build_epoch_plan(records, records, seed=9, epoch=0)
    assert len(plan) == 9
    by_instruction, by_record = defaultdict(float), defaultdict(float)
    for item in plan:
        assert type(item["record_index"]) is int
        assert item["weight"] > 0
        index = item["record_index"]
        by_instruction[records[index]["instr_id"]] += item["weight"]
        by_record[index] += item["weight"]
    assert set(by_record) == set(range(len(records)))
    assert dict(by_instruction) == pytest.approx({"a": 3., "b": 3., "c": 3.})
    assert dict(by_record) == pytest.approx({0: 3., 1: 1., 2: 1., 3: 1., 4: 1.5, 5: 1.5})


def test_olddata_plan_preserves_full_data_update_budget_and_instruction_weight(tmp_path):
    full = make_dataset(tmp_path).records
    # Unequal state counts expose accidental record-weighted sampling.
    full = [r for i, r in enumerate(full) if not (r["instr_id"] == "fit-0" and i % 3 == 0)]
    old = [r for r in full if r["condition"] in {"natural", "perturb_step2"}]
    plans = [training.build_epoch_plan(rows, full, seed=7, epoch=1) for rows in (full, old)]
    assert len(plans[0]) == len(plans[1])
    weights = []
    for rows, plan in zip((full, old), plans):
        value = defaultdict(float)
        for item in plan:
            value[rows[item["record_index"]]["instr_id"]] += item["weight"]
        weights.append(dict(value))
        assert {item["record_index"] for item in plan} == set(range(len(rows)))
    assert weights[0] == pytest.approx(weights[1])


def test_epoch_plan_deterministic_despite_global_rng_and_changes_between_epochs(tmp_path):
    rows = make_dataset(tmp_path, instructions=3).records
    first = training.build_epoch_plan(rows, rows, seed=5, epoch=0)
    random.seed(923); np.random.seed(17); torch.manual_seed(84)
    assert first == training.build_epoch_plan(rows, rows, seed=5, epoch=0)
    assert first != training.build_epoch_plan(rows, rows, seed=5, epoch=1)


def test_olddata_plan_rejects_missing_original_instruction(tmp_path):
    rows = make_dataset(tmp_path).records
    subset = [r for r in rows if r["instr_id"] == "fit-0"]
    with pytest.raises(ValueError):
        training.build_epoch_plan(subset, rows, seed=0, epoch=0)


def test_olddata_and_full_data_training_execute_the_same_update_budget(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    # The partial last batch also counts once in both experimental arms.
    config = small_config(batch_size=3)
    counts = []
    histories = []
    for arm in ("relative-history", "relative-olddata"):
        trainer = training.ContinuationTrainer(fit, dev, arm, config)
        while not trainer.done:
            if trainer.pending_dev:
                trainer.monitor_dev()
            else:
                trainer.step()
        counts.append((trainer.global_step, trainer.optimizer_updates))
        histories.append(trainer.training_history)
    assert counts == [(12, 12), (12, 12)]
    assert [row["sample_slots"] for row in histories[0]] == [16, 16]
    assert [row["sample_slots"] for row in histories[1]] == [16, 16]
    assert [row["repeated_slots"] for row in histories[0]] == [0, 0]
    assert [row["repeated_slots"] for row in histories[1]] == [8, 8]


@pytest.mark.parametrize("key,first,second", (
    ("model_config_sha256", "a" * 64, "b" * 64),
    ("seed", 0, 1), ("feature_dtype", "float16", "float32"),
    ("schema", DATA_SCHEMA, "different-schema")))
def test_fit_dev_must_share_the_actual_collector_protocol(tmp_path, key, first, second):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    fit.manifest["provenance"][key] = first
    dev.manifest["provenance"][key] = second
    with pytest.raises(ValueError, match="provenance"):
        training.ContinuationTrainer(fit, dev, "relative-history", small_config())


def test_cached_evaluation_chooses_earliest_eligible_state_not_best_future_label(tmp_path):
    data = make_dataset(tmp_path, split="train_dev", instructions=1)
    for bundle in data.bundles:
        first, later = bundle["records"]
        set_branch_outcome(bundle, first, 1, (0., 0.))
        set_branch_outcome(bundle, later, 1, (1., 1.))
        # Input order is not trajectory order; inference must sort by step.
        bundle["records"] = [later, first]
    report = training.evaluate_cached(FeatureScores(), data, sr_threshold=.1, spl_threshold=0.)
    assert report["interventions"] == len(SCHEDULES)
    for condition in SCHEDULES:
        assert report["conditions"][condition]["successes"] == 0
        assert report["conditions"][condition]["spl"] == 0.
    assert report["eligible"] is False


def test_cached_evaluation_skips_rejected_early_gate_and_keeps_empty_references(tmp_path):
    data = make_dataset(tmp_path, split="train_dev", instructions=1)
    for bundle in data.bundles:
        bundle["records"][0]["features"][:, :2] = 0.
        set_branch_outcome(bundle, bundle["records"][1], 1, (1., .8))
    # A reference lacking eligible states still belongs in the denominator.
    natural = next(b for b in data.bundles if b["reference"]["condition"] == "natural")
    natural["records"] = []
    report = training.evaluate_cached(FeatureScores(), data, sr_threshold=.1, spl_threshold=0.)
    assert report["interventions"] == len(SCHEDULES) - 1
    assert report["conditions"]["natural"]["episodes"] == 1
    assert report["conditions"]["natural"]["interventions"] == 0
    assert report["conditions"]["natural"]["sr"] == pytest.approx(1.)
    assert report["conditions"]["natural"]["spl"] == pytest.approx(.7)
    for condition in set(SCHEDULES) - {"natural"}:
        assert report["conditions"][condition]["sr"] == pytest.approx(1.)
        assert report["conditions"][condition]["spl"] == pytest.approx(.8)


@pytest.mark.parametrize("natural_outcome", ((0., 0.), (1., .2)))
def test_natural_sr_or_spl_regression_blocks_hard_condition_improvement(tmp_path, natural_outcome):
    data = make_dataset(tmp_path, split="train_dev", instructions=1)
    for bundle in data.bundles:
        for row in bundle["records"]:
            outcome = natural_outcome if row["condition"] == "natural" else (1., .9)
            set_branch_outcome(bundle, row, 1, outcome)
    report = training.evaluate_cached(FeatureScores(), data, sr_threshold=.1, spl_threshold=0.)
    assert report["hard_net_successes"] > 0
    assert report["eligible"] is False


def test_cached_policy_prediction_does_not_read_outcomes_or_teacher_targets(tmp_path):
    data = make_dataset(tmp_path, split="train_dev", instructions=1)
    changed = copy.deepcopy(data)
    for bundle in changed.bundles:
        for row in bundle["records"]:
            row["teacher_target"] = 0
            for candidate in range(1, len(row["candidate_actions"])):
                set_branch_outcome(bundle, row, candidate, (0., 0.))
    policies = [FeatureScores(), FeatureScores()]
    reports = [training.evaluate_cached(model, rows, sr_threshold=.1, spl_threshold=0.)
               for model, rows in zip(policies, (data, changed))]
    assert_tree_equal(policies[0].calls, policies[1].calls)
    assert reports[0]["interventions"] == reports[1]["interventions"]
    assert reports[0]["conditions"]["natural"]["successes"] == 1
    assert reports[1]["conditions"]["natural"]["successes"] == 0


@pytest.mark.parametrize("spl", (.612345678901, .612345699999))
def test_cached_evaluation_preserves_replay_precision_and_does_not_reject_equal_spl(tmp_path, spl):
    data = make_dataset(tmp_path, split="train_dev", instructions=1)
    natural = next(bundle for bundle in data.bundles if bundle["reference"]["condition"] == "natural")
    natural["reference"]["metrics"]["spl"] = spl
    for row in natural["records"]:
        set_branch_outcome(natural, row, 0, (1., spl))
        set_branch_outcome(natural, row, 1, (1., spl))
        # Rounded training targets are valid supervision, but not metric evidence.
        assert float(row["utilities"][1, 1]) != spl
    report = training.evaluate_cached(FeatureScores(), data, sr_threshold=.1, spl_threshold=0.)
    metrics = report["conditions"]["natural"]
    assert metrics["interventions"] == 1
    assert metrics["spl"] == metrics["baseline_spl"] == spl
    assert report["eligible"] is True


def test_cached_evaluation_reads_replay_metrics_even_when_training_targets_are_unavailable(tmp_path):
    data = make_dataset(tmp_path, split="train_dev", instructions=1)
    for row in data.records:
        del row["utilities"]
    report = training.evaluate_cached(FeatureScores(), data, sr_threshold=.1, spl_threshold=0.)
    assert report["conditions"]["natural"]["spl"] == .9
    assert report["eligible"] is True


def test_threshold_score_cache_detaches_without_device_transfer_or_repeated_forward(tmp_path, monkeypatch):
    data = make_dataset(tmp_path, split="train_dev", instructions=1)
    for row in data.records:
        row["features"].requires_grad_(True)

    class CountingTeacher(FeatureScores):
        def __init__(self):
            super().__init__(mode="teacher")
            self.forward_counts = defaultdict(int)

        def forward(self, features, history_features, progress):
            self.forward_counts[features.data_ptr()] += 1
            return super().forward(features, history_features, progress)

    def forbidden_cpu(self, *args, **kwargs):
        raise AssertionError("cached scores must stay on the online policy device")

    model, cache, previous = CountingTeacher(), {}, {}
    # This detects the old detach().cpu() path even on a CPU-only test host.
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden_cpu)
        for threshold in (0., .025, .05, .1, .4):
            training.evaluate_cached(model, data, threshold, 0., score_cache=cache)
            for key, scores in cache.items():
                bundle_index, record_index = key
                source = data.bundles[bundle_index]["records"][record_index]["features"]
                assert scores.requires_grad is False
                assert scores.device == source.device and scores.dtype == source.dtype
                assert torch.equal(scores, source[:, :2])
                if key in previous:
                    cached_object, exact_values = previous[key]
                    assert scores is cached_object
                    assert torch.equal(scores, exact_values)
                else:
                    previous[key] = (scores, scores.clone())
    assert len(cache) == len(data.records)
    assert len(model.calls) == len(data.records)
    assert dict(model.forward_counts) == {row["features"].data_ptr(): 1 for row in data.records}


@pytest.mark.parametrize("cold", (False, True))
@pytest.mark.parametrize("pause_step", (3, 8))
def test_full_training_state_is_identical_after_hot_or_cold_strict_resume(tmp_path, cold, pause_step):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    config = small_config()
    kwargs = dict(fit=fit, dev=dev, arm="relative-history", config=config, backup_check=lambda: None)
    expected_local, expected_backup = tmp_path / "expected-local", tmp_path / "expected-backup"
    expected = training.train_run(**kwargs, local_dir=expected_local, backup_dir=expected_backup)
    assert expected["status"] == "complete"
    expected_state = restore_latest(expected_local, expected_backup)
    local, backup = tmp_path / "interrupted-local", tmp_path / "interrupted-backup"
    paused = training.train_run(**kwargs, local_dir=local, backup_dir=backup, stop_after_steps=pause_step)
    assert paused["status"] == "paused" and paused["global_step"] == pause_step
    assert paused["pending_dev"] is (pause_step == 8)
    if cold:
        shutil.rmtree(local)
        local = tmp_path / "fresh-local"
    # Recovery must overwrite ambient RNG, including Python and NumPy.
    random.seed(984); np.random.seed(348); torch.manual_seed(193)
    actual = training.train_run(**kwargs, local_dir=local, backup_dir=backup, require_resume=True)
    assert actual["status"] == "complete"
    assert actual["global_step"] == expected["global_step"]
    actual_state = restore_latest(local, backup)
    assert_tree_equal(actual_state, expected_state)


def test_strict_resume_refuses_a_missing_checkpoint(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    with pytest.raises(ValueError, match="require-resume.*no valid checkpoint"):
        training.train_run(fit, dev, "relative-history", small_config(),
                           tmp_path / "local", tmp_path / "backup", lambda: None,
                           require_resume=True)


@pytest.mark.parametrize("change", ("configuration", "fit-identity", "dev-identity", "arm"))
def test_resume_rejects_changed_experiment_identity(tmp_path, change):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    config, arm = small_config(), "relative-history"
    local, backup = tmp_path / "local", tmp_path / "backup"
    training.train_run(fit, dev, arm, config, local, backup, lambda: None, stop_after_steps=1)
    if change == "configuration":
        config["lr"] *= 2
    elif change == "fit-identity":
        fit.identity["manifest_sha256"] = "different-fit"
    elif change == "dev-identity":
        dev.identity["manifest_sha256"] = "different-dev"
    else:
        arm = "absolute-history"
    with pytest.raises(ValueError):
        training.train_run(fit, dev, arm, config, local, backup, lambda: None, require_resume=True)


def test_teacher_valid_labels_preserve_equal_total_instruction_weight(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    for row in fit.records:
        row["teacher_target"] = -1
    labeled = [fit.records[0], *fit.records[8:11]]
    for index, row in enumerate(labeled):
        row["teacher_target"] = 1 if index % 2 == 0 else 2
    config = small_config(epochs=1, candidate_epochs=[1], batch_size=len(fit.records))
    trainer = training.ContinuationTrainer(fit, dev, "teacher-history", config)
    expected = copy.deepcopy(trainer.head)
    optimizer = torch.optim.AdamW(expected.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    optimizer.zero_grad(set_to_none=True)
    # Each of two original instructions has eight slots. Their 1 vs 3
    # supervised states receive total weight eight each, with a fixed batch
    # denominator of sixteen; missing labels never become expert targets.
    terms = []
    # Preserve addition order. Batched versus separate matrix kernels may still
    # differ by rounding; checkpoint-resume tests continue to require exactness.
    # These fixture weights are calculated independently of trainer counters.
    fixture_weights = {"fit-0": 8., "fit-1": 8 / 3}
    for item in trainer.plan:
        row = fit.records[item["record_index"]]
        assert item["weight"] == 1.
        if row["teacher_target"] != -1:
            terms.append(record_loss(expected, row) * fixture_weights[row["instr_id"]])
    weighted_loss = torch.stack(terms).sum() / config["batch_size"]
    weighted_loss.backward()
    optimizer.step()
    trainer.step()
    for name, actual in trainer.head.state_dict().items():
        # A common logit bias has theoretically zero CE gradient. Adam may
        # amplify roundoff there; this shift cancels out of every softmax.
        tolerance = 1e-5 if name == "comparison.2.bias" else 5e-7
        torch.testing.assert_close(actual, expected.state_dict()[name], rtol=1e-4, atol=tolerance)
    assert_tree_equal(trainer.optimizer.state_dict(), optimizer.state_dict(), rtol=1e-4, atol=5e-7)
    for row in labeled:
        actual_probability = trainer.head.score_record(row)[:, 0].softmax(0)
        expected_probability = expected.score_record(row)[:, 0].softmax(0)
        torch.testing.assert_close(actual_probability, expected_probability, rtol=1e-5, atol=2e-7)
    row = trainer.training_history[-1]
    assert row["teacher_labeled_slots"] == len(labeled)
    assert row["teacher_missing_slots"] == len(fit.records) - len(labeled)
    assert row["weighted_loss"] == pytest.approx(float(weighted_loss.detach()))
    assert row["teacher_instructions_with_labels"] == 2
    assert row["teacher_instructions_without_labels"] == 0
    assert trainer.optimizer_updates == 1


def test_teacher_all_missing_batch_has_no_optimizer_or_weight_decay_update(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    for row in fit.records:
        row["teacher_target"] = -1
    config = small_config(epochs=1, candidate_epochs=[1], batch_size=2, weight_decay=.5)
    plan = training.build_epoch_plan(fit.records, fit.records, seed=config["seed"], epoch=0)
    first_batch_indices = {item["record_index"] for item in plan[:config["batch_size"]]}
    labeled_index = next(index for index in range(len(fit.records)) if index not in first_batch_indices)
    fit.records[labeled_index]["teacher_target"] = 1
    trainer = training.ContinuationTrainer(fit, dev, "teacher-history", config)
    before = copy.deepcopy(trainer.head.state_dict())
    trainer.step()
    assert_tree_equal(trainer.head.state_dict(), before)
    assert trainer.optimizer.state_dict()["state"] == {}
    assert trainer.optimizer_updates == 0
    assert trainer.epoch_teacher_labeled == 0
    assert trainer.epoch_teacher_missing == config["batch_size"]
    assert trainer.epoch_weight_sum == 0.
    while not trainer.pending_dev:
        trainer.step()
    assert trainer.training_history[-1]["teacher_labeled_slots"] == 1
    assert trainer.training_history[-1]["teacher_missing_slots"] == len(fit.records) - 1
    assert trainer.training_history[-1]["weighted_loss"] is not None
    assert trainer.training_history[-1]["teacher_instructions_with_labels"] == 1
    assert trainer.training_history[-1]["teacher_instructions_without_labels"] == 1
    assert trainer.optimizer_updates == 1


def test_teacher_rejects_dataset_without_any_expert_labels(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    for row in fit.records:
        row["teacher_target"] = -1
    with pytest.raises(ValueError, match="teacher|label|expert"):
        training.ContinuationTrainer(fit, dev, "teacher-history", small_config())


@pytest.mark.parametrize("arm", tuple(training.ARMS))
def test_training_step_uses_one_vectorized_loss_call(tmp_path, monkeypatch, arm):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    config = small_config(batch_size=4)
    if arm == "teacher-history":
        plan = training.build_epoch_plan(fit.records, fit.records, config["seed"], 0)
        fit.records[plan[0]["record_index"]]["teacher_target"] = -1
    trainer = training.ContinuationTrainer(fit, dev, arm, config)
    original = training.batch_record_losses
    calls = []

    def spy(model, records, **weights):
        assert model is trainer.head
        rows = list(records)
        calls.append(len(rows))
        return original(model, rows, **weights)

    def forbidden_single_record(*args, **kwargs):
        raise AssertionError("training step must use the batched prediction path")

    monkeypatch.setattr(training, "batch_record_losses", spy)
    monkeypatch.setattr(trainer.head, "score_record", forbidden_single_record)
    trainer.step()
    assert len(calls) == 1
    assert calls[0] == (3 if arm == "teacher-history" else 4)
    assert trainer.global_step == trainer.optimizer_updates == 1


@pytest.mark.parametrize("arm", ("relative-history", "absolute-history", "teacher-history"))
def test_final_partial_batch_keeps_fixed_denominator_and_record_weights(tmp_path, monkeypatch, arm):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    config = small_config(epochs=1, candidate_epochs=[1], batch_size=3)
    trainer = training.ContinuationTrainer(fit, dev, arm, config)
    while trainer.cursor + config["batch_size"] < len(trainer.plan):
        trainer.step()
    remaining = trainer.plan[trainer.cursor:]
    assert len(remaining) == 1
    reference = copy.deepcopy(trainer.head)
    reference.zero_grad(set_to_none=True)
    terms = []
    for item in remaining:
        row = trainer.records[item["record_index"]]
        # Every instruction has eight fully labeled states in this fixture,
        # so teacher's additional per-instruction weight correction is one.
        loss = record_loss(reference, row, **{name: config[name] for name in
            ("sr_weight", "spl_weight", "rescue_weight", "harm_weight")})
        terms.append(loss * item["weight"])
    (torch.stack(terms).sum() / config["batch_size"]).backward()
    original = training.batch_record_losses
    calls = []

    def spy(model, records, **weights):
        rows = list(records)
        calls.append(len(rows))
        return original(model, rows, **weights)

    monkeypatch.setattr(training, "batch_record_losses", spy)
    trainer.step()
    assert calls == [1]
    assert trainer.global_step == trainer.optimizer_updates == 6
    assert trainer.pending_dev is True
    for name, parameter in trainer.head.named_parameters():
        expected = dict(reference.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-4, atol=5e-6,
                                   msg=lambda message: name + ": " + message)


@pytest.mark.parametrize("change", ("plan", "plan_digest", "cursor", "step"))
def test_resume_rejects_inconsistent_sampler_or_progress(tmp_path, change):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    trainer = training.ContinuationTrainer(fit, dev, "relative-history", small_config())
    trainer.step()
    state = trainer.state_dict()
    if change == "plan":
        state["plan"][0]["weight"] *= 2
    elif change == "plan_digest":
        state["plan_sha256"] = "0" * 64
    elif change == "cursor":
        state["cursor"] += 1
    else:
        state["global_step"] += 1
    with pytest.raises(ValueError, match="resume"):
        trainer.load_state_dict(state)


def persist_dataset_fixture(tmp_path, mutate=None):
    """Produce real tensor files, SHA pointers and identical backup manifests."""
    local, backup = tmp_path / "cache", tmp_path / "persistent-cache"
    local.mkdir(); backup.mkdir()
    pointers = []
    for condition, schedule in SCHEDULES.items():
        step = max(schedule) + 1 if schedule else 0
        row = make_record(instr_id="fit-0", scan_id="fit-house", condition=condition, step=step)
        row["features"] = torch.zeros((3, 1549), dtype=torch.float16)
        row["history_features"] = torch.zeros((2 * step + 1, 1549), dtype=torch.float16)
        row["progress"] = row["progress"].half()
        states = [{"step": index, "executed_action": "keep", "raw_argmax_action": "keep",
                   "forced_terminal": False, "candidate_vpids": [None, "keep", "rescue"],
                   "logits": [0., 1., .5], "valid_mask": [True, True, True],
                   "visited_mask": [False, False, False]}
                  for index in range(step + 1)]
        ref = {"instr_id": "fit-0", "scan_id": "fit-house", "condition": condition,
               "seed": 0, "metrics": {"success": 1., "spl": .7}, "states": states}
        branches = [{"target_step": step, "target_action": action, "is_anchor": index == 0,
                     "instr_id": "fit-0", "scan_id": "fit-house", "condition": condition,
                     "metrics": {"success": float(utility[0]), "spl": float(utility[1])}}
                    for index, (action, utility) in enumerate(zip(row["candidate_actions"], row["utilities"]))]
        bundle = {"schema": DATA_SCHEMA, "split": "train_fit", "reference": ref,
                  "records": [row], "branches": branches}
        if mutate is not None:
            mutate(bundle)
        task = {"instr_id": "fit-0", "condition": condition}
        pointer = save_bundle(local / "bundles", backup / "bundles", task, bundle, lambda: None)
        pointers.append(dict(pointer, task=task))
    manifest = {"schema": DATA_SCHEMA, "complete": True, "backup_root": str(backup),
                "selection": {"split": "train_fit", "instr_ids": ["fit-0"],
                    "scan_ids": ["fit-house"], "conditions": list(SCHEDULES)},
                "provenance": {"schema": DATA_SCHEMA}, "bundles": pointers}
    raw = json.dumps(manifest, sort_keys=True, indent=2)
    for root in (local, backup):
        (root / "dataset-manifest.json").write_text(raw)
    return local, backup, manifest


def test_dataset_loader_verifies_real_bundles_and_restores_a_missing_local_tensor(tmp_path):
    local, backup, manifest = persist_dataset_fixture(tmp_path)
    first = manifest["bundles"][0]
    (local / "bundles" / first["file"]).unlink()
    checks = []
    dataset = training.load_dataset(local, "train_fit", check_backup=lambda: checks.append(True))
    assert len(dataset.bundles) == len(SCHEDULES)
    assert len(dataset.records) == len(SCHEDULES)
    assert dataset.records[0]["features"].shape == (3, 1549)
    assert file_sha256(local / "bundles" / first["file"]) == first["sha256"]
    assert dataset.identity["manifest_sha256"] == file_sha256(local / "dataset-manifest.json")
    assert len(checks) >= 1 + 2 * len(SCHEDULES)


def test_dataset_loader_refuses_missing_persistent_tensor_even_if_local_exists(tmp_path):
    local, backup, manifest = persist_dataset_fixture(tmp_path)
    (backup / "bundles" / manifest["bundles"][0]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        training.load_dataset(local, "train_fit")


@pytest.mark.parametrize("which", ("local", "backup"))
def test_dataset_loader_rejects_tensor_bytes_that_do_not_match_pointer_sha(tmp_path, which):
    local, backup, manifest = persist_dataset_fixture(tmp_path)
    root = local if which == "local" else backup
    path = root / "bundles" / manifest["bundles"][0]["file"]
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="corrupt"):
        training.load_dataset(local, "train_fit")


def test_dataset_loader_rejects_outcomes_that_disagree_with_full_continuation(tmp_path):
    def mutate(bundle):
        bundle["records"][0]["utilities"][1, 1] = .1
    local, _, _ = persist_dataset_fixture(tmp_path, mutate)
    with pytest.raises(ValueError, match="utilities.*outcomes"):
        training.load_dataset(local, "train_fit")


def test_dataset_loader_rejects_candidate_zero_different_from_executed_action(tmp_path):
    def mutate(bundle):
        actions = bundle["records"][0]["candidate_actions"]
        actions[0], actions[1] = actions[1], actions[0]
    local, _, _ = persist_dataset_fixture(tmp_path, mutate)
    with pytest.raises(ValueError, match="candidate"):
        training.load_dataset(local, "train_fit")


def test_dataset_loader_rejects_missing_full_continuation_label(tmp_path):
    local, _, _ = persist_dataset_fixture(tmp_path, lambda bundle: bundle["branches"].pop())
    with pytest.raises(ValueError, match="no full continuation label"):
        training.load_dataset(local, "train_fit")


def test_dataset_loader_rejects_local_manifest_that_disagrees_with_backup(tmp_path):
    local, _, manifest = persist_dataset_fixture(tmp_path)
    manifest["provenance"]["tampered"] = True
    (local / "dataset-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifests differ"):
        training.load_dataset(local, "train_fit")


def test_all_arms_can_be_prepared_at_step_zero_then_cold_strict_resumed(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    config = small_config(epochs=1, candidate_epochs=[1], batch_size=16)
    original_local, backup = tmp_path / "original-local", tmp_path / "suite-backup"
    for arm in training.ARMS:
        report = training.train_run(fit, dev, arm, config, original_local / arm, backup / arm,
                                    lambda: None, stop_requested=lambda: True)
        assert report["status"] == "paused" and report["global_step"] == 0
        state = restore_latest(original_local / arm, backup / arm)
        assert state["optimizer_updates"] == 0
        assert state["optimizer"]["state"] == {}
    shutil.rmtree(original_local)
    for arm in training.ARMS:
        report = training.train_run(fit, dev, arm, config, tmp_path / "fresh-local" / arm,
                                    backup / arm, lambda: None, require_resume=True)
        assert report["status"] == "complete" and report["resumed"] is True
        assert report["global_step"] == 1
        assert report["selected_checkpoint"] is not None


def scope_dataset(root, split, count, *, rescues=0, rescue_scans=12, smoke=False):
    """Only the already-loaded identity and branch outcomes matter to this gate."""
    prefix = "fit" if split == "train_fit" else "dev"
    ids = [f"{prefix}-{index}" for index in range(count)]
    bundles = []
    for index, instr_id in enumerate(ids):
        scan_id = f"{prefix}-house-{index % rescue_scans}"
        for condition in SCHEDULES:
            # The same rescue repeated across conditions remains one instruction.
            bundles.append({"reference": {"instr_id": instr_id, "scan_id": scan_id,
                "condition": condition, "metrics": {"success": 0., "spl": 0.}},
                "branches": [{"metrics": {"success": float(index < rescues), "spl": 0.}}],
                "records": [{"instr_id": instr_id, "scan_id": scan_id, "condition": condition}]})
    manifest = {"selection": {"split": split, "instr_ids": ids,
                "scan_ids": sorted({b["reference"]["scan_id"] for b in bundles}),
                "conditions": list(SCHEDULES), "smoke": smoke},
                "provenance": {"schema": DATA_SCHEMA}}
    return training.Dataset(root, manifest, bundles, {"manifest_sha256": prefix + "-scope"})


def test_training_scope_smoke_requires_an_explicit_flag(tmp_path):
    fit = scope_dataset(tmp_path / "fit", "train_fit", 2, rescues=2, smoke=True)
    dev = scope_dataset(tmp_path / "dev", "train_dev", 2, smoke=True)
    with pytest.raises(ValueError, match="explicit.*engineering-smoke"):
        training.validate_training_scope(fit, dev)
    support = training.validate_training_scope(fit, dev, engineering_smoke=True)
    assert support["passed"] is False
    assert support["unique_rescuable_fit_instructions"] == 2
    assert support["rescuable_fit_scans"] == 2
    assert "not_statistical_power_or_navigation_performance" in support["scope"]


@pytest.mark.parametrize("fit_count,dev_count,rescues,scans,accepted", (
    (512, 128, 32, 12, True), (2048, 128, 32, 12, True),
    (256, 128, 32, 12, False), (512, 127, 32, 12, False),
    (512, 128, 31, 12, False), (512, 128, 32, 11, False)))
def test_formal_training_scope_requires_fixed_panels_and_diverse_rescue_support(
        tmp_path, fit_count, dev_count, rescues, scans, accepted):
    fit = scope_dataset(tmp_path / "fit", "train_fit", fit_count, rescues=rescues, rescue_scans=scans)
    dev = scope_dataset(tmp_path / "dev", "train_dev", dev_count)
    if accepted:
        support = training.validate_training_scope(fit, dev)
        assert support["passed"] is True
        assert support["unique_rescuable_fit_instructions"] == 32
        assert support["rescuable_fit_scans"] == 12
    else:
        with pytest.raises(ValueError, match="formal training"):
            training.validate_training_scope(fit, dev)


def test_smoke_head_keeps_scope_and_exact_training_panel_identity(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    trainer = training.ContinuationTrainer(fit, dev, "relative-history", small_config(engineering_smoke=True))
    payload = trainer.head_payload()
    assert payload["scope"] == payload["provenance"]["scope"] == "engineering_smoke"
    assert payload["selection_status"] == "engineering_smoke_unselected_snapshot"
    assert payload["provenance"]["fit_selection"] == fit.manifest["selection"]
    assert payload["provenance"]["dev_selection"] == dev.manifest["selection"]


def test_selected_head_export_is_verified_portable_and_preserves_smoke_scope(tmp_path):
    fit, dev = make_dataset(tmp_path / "fit"), make_dataset(tmp_path / "dev", "train_dev")
    config = small_config(epochs=1, candidate_epochs=[1], batch_size=16, engineering_smoke=True)
    local, backup = tmp_path / "run", tmp_path / "persistent-run"
    report = training.train_run(fit, dev, "relative-history", config, local, backup, lambda: None)
    selected = report["selected_checkpoint"]
    path = local / "selected-head.pt"
    assert selected["export_file"] == path.name
    assert file_sha256(path) == file_sha256(backup / path.name) == selected["head_sha256"]
    head = torch.load(path, map_location="cpu", weights_only=True)
    assert head["schema"] == "e3_continuation_head_v2"
    assert head["model_config"] == {"feature_dim": 7, "hidden_dim": 8, "mode": "relative", "history": True}
    assert head["thresholds"] == {"sr": selected["sr_threshold"], "spl": selected["spl_threshold"]}
    assert head["scope"] == head["provenance"]["scope"] == "engineering_smoke"
    assert head["selection_status"].startswith("engineering_smoke_")
    assert head["provenance"]["fit_selection"] == fit.manifest["selection"]
    assert head["provenance"]["dev_selection"] == dev.manifest["selection"]
