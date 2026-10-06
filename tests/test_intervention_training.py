import copy
from pathlib import Path
import shutil

import pytest
import torch

from vln_improve.checkpoint_store import CheckpointStore
from vln_improve.endpoint_intervention import collate_interventions, select_intervention
from vln_improve.endpoint_pairs import content_hash
from vln_improve.intervention_training import (
    AbsoluteUtilityHead, InterventionTrainer, absolute_loss, evaluate_records, load_head,
    make_head, predict_gains, prepare_batch, record_inputs, train_intervention, validate_splits,
)
from scripts.train_endpoint_intervention import validate_cache_protocol


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def records(split, count=6):
    result = []
    generator = torch.Generator().manual_seed(41 if split == "fit" else 42)
    for i in range(count):
        n = 2 + i % 3
        ids = [f"{split}-{i}-{j}" for j in range(n)]
        success = torch.tensor([float((i + j) % 3 != 0) for j in range(n)], dtype=torch.float64)
        utilities = torch.stack((success, success * torch.linspace(.3, .9, n, dtype=torch.float64)), -1)
        result.append({"schema": "e2_intervention_record_v1", "instr_id": f"{split}-{i}",
            "scan_id": f"{split}-scene-{i % 2}", "condition": "natural", "baseline_endpoint": ids[0],
            "inputs": {"candidate_vpids": ids, "baseline_index": 0,
                "node_features": torch.randn(n, 768, generator=generator),
                "terminal_context": torch.randn(1536, generator=generator),
                "scalar_features": torch.randn(n, 10, generator=generator)},
            "targets": utilities - utilities[0], "utilities": utilities,
            "candidate_metrics": [{"success": float(u[0]), "spl": float(u[1]), "nDTW": .4 + .1 * j}
                                  for j, u in enumerate(utilities)]})
    return result


def config(arm="relative"):
    return {"arm": arm, "seed": 0, "epochs": 3, "batch_size": 2, "hidden_dim": 8,
            "lr": 1e-3, "weight_decay": .01, "monitor_every_epochs": 2, "risk_weight": 0.}


def assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right


def test_matched_capacity_initialization_and_keep():
    torch.manual_seed(0); relative = make_head("relative", 8)
    torch.manual_seed(0); absolute = make_head("absolute", 8)
    assert content_hash(relative.state_dict()) == content_hash(absolute.state_dict())
    assert sum(p.numel() for p in relative.parameters()) == sum(p.numel() for p in absolute.parameters())
    batch, _, _ = prepare_batch(records("fit"), "cpu")
    for model in (relative, absolute):
        gains = predict_gains(model, batch)
        assert torch.equal(gains, torch.zeros_like(gains))
        for i, ids in enumerate(batch.candidate_vpids):
            assert select_intervention(gains[i], ids, ids[0], valid_mask=batch.valid_mask[i]) == ids[0]
    raw = absolute(batch)
    assert torch.all(raw[batch.valid_mask] == .5)
    loss = absolute_loss(raw, torch.zeros_like(raw), batch.valid_mask)
    loss.backward()
    assert absolute.comparison[-1].bias.grad.abs().sum() > 0


def test_absolute_subtracts_learned_anchor_then_identical_gate():
    model = make_head("absolute", 8)
    with torch.no_grad():
        model.comparison[-1].weight.fill_(.1)
    batch, _, _ = prepare_batch(records("fit"), "cpu")
    raw = model(batch)
    gains = predict_gains(model, batch)
    for i in range(len(raw)):
        n = len(batch.candidate_vpids[i])
        assert torch.equal(gains[i, :n], raw[i, :n] - raw[i, 0])
        assert torch.equal(gains[i, 0], torch.zeros(2))
    assert torch.equal(gains[~batch.valid_mask], torch.zeros_like(gains[~batch.valid_mask]))


def test_episode_normalized_absolute_loss_and_padding():
    prediction = torch.tensor([[[1., 1.], [1., 1.], [float("nan"), float("nan")]],
                               [[0., 0.], [0., 0.], [0., 0.]]], requires_grad=True)
    utilities = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    assert float(absolute_loss(prediction, utilities, mask)) == .25


@pytest.mark.parametrize("arm", ("relative", "absolute"))
@pytest.mark.parametrize("stop_after", (4, 6))
def test_recovery_from_empty_local_store_matches_continuous(tmp_path, arm, stop_after):
    fit, dev = records("fit"), records("dev", 4)
    cfg, identity = config(arm), {"fit": "fixed-fixture", "dev": "disjoint-fixture"}
    full = train_intervention(fit, dev, cfg, data_identity=identity,
        local_dir=tmp_path / "full", backup_dir=tmp_path / "full-backup", checkpoint_every_steps=1)
    first = train_intervention(fit, dev, cfg, data_identity=identity,
        local_dir=tmp_path / "resumed", backup_dir=tmp_path / "backup", checkpoint_every_steps=1,
        stop_after_updates=stop_after)
    assert first["status"] == "interrupted"
    assert first["global_step"] == stop_after
    if stop_after == 6:
        assert first["pending_dev"] is True
    shutil.rmtree(tmp_path / "resumed")
    restarted = train_intervention(fit, dev, cfg, data_identity=identity,
        local_dir=tmp_path / "resumed", backup_dir=tmp_path / "backup", checkpoint_every_steps=1)
    assert restarted["status"] == "complete" and restarted["resumed"] is True
    assert restarted["global_step"] == 9
    assert restarted["training_history"] == full["training_history"]
    assert restarted["dev_history"] == full["dev_history"]
    assert [r["epoch"] for r in restarted["dev_history"]] == [2, 3]
    full_state, full_head, _ = CheckpointStore(tmp_path / "full", tmp_path / "full-backup").restore("latest")
    state, head, _ = CheckpointStore(tmp_path / "resumed", tmp_path / "backup").restore("latest")
    assert_exact(state, full_state)
    assert content_hash(head) == content_hash(full_head)
    selected = restarted["selected_checkpoint"]
    model = load_head(tmp_path / "resumed" / selected["head_relative_path"])
    metrics = evaluate_records(model, dev)
    expected = next(r for r in restarted["dev_history"] if r["epoch"] == selected["epoch"])
    assert metrics["per_episode"] == expected["per_episode"]
    assert restarted["initial_head_sha256"] == full["initial_head_sha256"]


def test_validation_rejects_leakage_wrong_targets_and_perturbed_dev():
    fit, dev = records("fit"), records("dev", 4)
    broken = copy.deepcopy(dev); broken[0]["scan_id"] = fit[0]["scan_id"]
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(fit, broken)
    broken = copy.deepcopy(dev); broken[0]["condition"] = "perturb_step2"
    with pytest.raises(ValueError, match="nonnatural"):
        validate_splits(fit, broken)
    broken = copy.deepcopy(fit); broken[0]["targets"][1, 1] += .01
    with pytest.raises(ValueError, match="disagree"):
        validate_splits(broken, dev)


def test_restoration_rejects_sampler_optimizer_and_config_changes():
    fit, dev = records("fit"), records("dev", 4)
    trainer = InterventionTrainer(fit, dev, config(), data_identity={"fixture": True})
    trainer.step()
    saved = trainer.state_dict()
    for key, value, message in (("cursor", 4, "cursor"), ("current_order_sha256", "different", "sampler")):
        broken = copy.deepcopy(saved); broken[key] = value
        with pytest.raises(ValueError, match=message):
            trainer.load_state_dict(broken)
    broken = copy.deepcopy(saved); broken["optimizer"]["state"][0]["step"] += 1
    with pytest.raises(ValueError, match="AdamW step"):
        trainer.load_state_dict(broken)
    changed = InterventionTrainer(fit, dev, {**config(), "epochs": 4}, data_identity={"fixture": True})
    with pytest.raises(ValueError, match="config"):
        changed.load_state_dict(saved)


def test_selection_uses_actual_metrics_and_keeps_earliest_tie(monkeypatch):
    trainer = InterventionTrainer(records("fit"), records("dev", 4), config(), data_identity={})
    panel = {"sr": .8, "spl": .7, "eligible": True, "research_success": True, "changes": 1}
    monkeypatch.setattr("vln_improve.intervention_training.evaluate_records", lambda *a, **k: dict(panel))
    trainer.epoch = 2; trainer.pending_dev = True
    _, best = trainer.monitor_dev()
    assert best and trainer.best_epoch == 2
    trainer.epoch = 3; trainer.pending_dev = True
    _, best = trainer.monitor_dev()
    assert not best and trainer.best_epoch == 2
    panel.update(sr=.9, spl=.6, eligible=False)
    trainer.pending_dev = True
    _, best = trainer.monitor_dev()
    assert not best and trainer.best_epoch == 2


def test_no_eligible_checkpoint_uses_fixed_trained_final(tmp_path, monkeypatch):
    panel = {"sr": .1, "spl": .1, "eligible": False, "research_success": False, "changes": 2}
    monkeypatch.setattr("vln_improve.intervention_training.evaluate_records", lambda *a, **k: dict(panel))
    result = train_intervention(records("fit"), records("dev", 4), config(), data_identity={},
        local_dir=tmp_path / "local", backup_dir=tmp_path / "backup")
    assert result["best_epoch"] is None
    assert result["selected_checkpoint"]["selection_reason"] == "dev_gate_failed_final_exploratory"
    assert result["selected_checkpoint"]["epoch"] == 3
    assert result["selected_checkpoint"]["head_sha256"] == result["final_checkpoint"]["head_sha256"]


def test_keep_is_eligible_but_not_research_success():
    metrics = evaluate_records(make_head("relative", 8), records("dev", 4))
    assert metrics["eligible"] is True
    assert metrics["changes"] == 0 and metrics["research_success"] is False
    assert metrics["sr"] == metrics["baseline_sr"] and metrics["spl"] == metrics["baseline_spl"]


def test_protocol_requires_matching_conditions_membership_and_master_digest():
    natural = records("fit")
    perturbed = copy.deepcopy(natural)
    for record in perturbed:
        record["condition"] = "perturb_step2"
    dev = records("dev", 4)
    provenance = {"experiment_sha256": "master", "base_checkpoint_sha256": "backbone"}
    meta = {"provenance": provenance, "collection": {"scope": "research"}}
    identity = {"fit": [copy.deepcopy(meta), copy.deepcopy(meta)], "dev": copy.deepcopy(meta)}
    spec = {"collection_config_sha256": "master", "baseline": {"checkpoint_sha256": "backbone"},
        "collection": {"fit_conditions": ["natural", "perturb_step2"], "fit_instruction_count": 6,
            "fit_expected_scenes": 2, "dev_instruction_count": 4, "dev_expected_scenes": 2},
        "fit_episodes": 12, "epochs": 3, "batch_size": 2, "max_updates_per_arm": 18}
    validate_cache_protocol(natural + perturbed, dev, identity, spec, "per-arm-digest")
    assert identity["scope"] == "research"
    changed = copy.deepcopy(perturbed); changed[0]["instr_id"] = "different-id"
    with pytest.raises(ValueError, match="membership"):
        validate_cache_protocol(natural + changed, dev, identity, spec, "per-arm-digest")
    broken = copy.deepcopy(identity); broken["dev"]["provenance"]["base_checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        validate_cache_protocol(natural + perturbed, dev, broken, spec, "per-arm-digest")
    with pytest.raises(ValueError, match="scoped"):
        validate_cache_protocol(natural, dev, identity, spec, "per-arm-digest", engineering_smoke=True)


def test_numerically_equal_dev_metrics_keep_earliest_checkpoint(monkeypatch):
    trainer = InterventionTrainer(records("fit"), records("dev", 4), config(), data_identity={})
    panel = {"sr": .8, "spl": .7, "eligible": True, "research_success": False, "changes": 0}
    monkeypatch.setattr("vln_improve.intervention_training.evaluate_records", lambda *a, **k: dict(panel))
    trainer.epoch = 2; trainer.pending_dev = True
    assert trainer.monitor_dev()[1]
    panel["spl"] += 1e-13
    trainer.epoch = 3; trainer.pending_dev = True
    assert not trainer.monitor_dev()[1]
    assert trainer.best_epoch == 2
