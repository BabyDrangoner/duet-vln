import copy
import json
import random
import shutil

import numpy as np
import pytest
import torch

from vln_improve.features import FEATURE_SCHEMA
from vln_improve.head import load_head_checkpoint
from vln_improve.resumable import ResumableTrainer


def cache_at(path, count=11):
    path.mkdir(parents=True)
    records = []
    for index in range(count):
        length = 2 + index % 3
        target = index % length
        features = torch.zeros(length, 2, dtype=torch.float16)
        features[target, 0] = 1
        features[:, 1] = index / count
        records.append({
            "features": features, "base_logits": torch.arange(length, dtype=torch.float32) * 0.2,
            "valid_mask": torch.ones(length, dtype=torch.bool), "target": target,
            "hard": index % 2 == 0, "instr_id": f"instruction-{index}", "scan_id": "train-house",
        })
    names = []
    for start in range(0, count, 128):
        name = f"shard-{start:04d}.pt"
        names.append(name)
        torch.save(records[start:start + 128], path / name)
    (path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "feature_schema": FEATURE_SCHEMA, "feature_dim": 2,
        "split": "train_fit", "num_records": count, "shards": names,
        "provenance": {"dataset": "synthetic", "feature_id": "test-two-features",
                       "base_checkpoint_sha256": "a" * 64, "upstream_commit": "b" * 40,
                       "max_action_len": 15, "feedback": "argmax"},
    }))
    return path


def trainer(cache, **options):
    return ResumableTrainer([cache], **{
        "epochs": 3, "batch_size": 4, "hidden_dim": 8, "seed": 42, **options,
    })


def finish(value):
    results = []
    while not value.done:
        results.append(value.step())
    return results


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for left, right in zip(first, second):
            assert_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize("stop_after", [0, 1, 2, 3, 4, 8, 9])
def test_exact_resume_across_batch_and_epoch_boundaries(tmp_path, stop_after):
    cache = cache_at(tmp_path / "cache")
    continuous = trainer(cache)
    expected_results = finish(continuous)
    expected = continuous.state_dict()
    interrupted = trainer(cache)
    actual_results = [interrupted.step() for _ in range(stop_after)]
    path = tmp_path / "resume.pt"
    torch.save(interrupted.state_dict(), path)
    restored = trainer(cache)
    restored.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    actual_results.extend(finish(restored))
    assert actual_results == expected_results
    assert_equal(restored.state_dict(), expected)
    assert restored.global_step == 9 and restored.epoch == 3 and restored.done
    assert [item["num_records"] for item in restored.history] == [11, 11, 11]
    with pytest.raises(RuntimeError, match="already complete"):
        restored.step()


def test_multi_shard_resume_and_final_short_batch(tmp_path):
    cache = cache_at(tmp_path / "cache", count=131)
    continuous = trainer(cache, batch_size=64, epochs=2)
    finish(continuous)
    interrupted = trainer(cache, batch_size=64, epochs=2)
    interrupted.step()
    interrupted.step()
    state = interrupted.state_dict()
    assert state["record_cursor"] == 128
    resumed = trainer(cache, batch_size=64, epochs=2)
    resumed.load_state_dict(state)
    boundary = resumed.step()
    assert boundary["epoch_completed"] and boundary["epoch_metrics"]["num_records"] == 131
    finish(resumed)
    assert_equal(resumed.state_dict(), continuous.state_dict())


def test_resume_after_cache_moves_to_new_vm_path(tmp_path):
    original = cache_at(tmp_path / "old-vm")
    first = trainer(original)
    first.step()
    state = first.state_dict()
    relocated = tmp_path / "new-vm"
    shutil.copytree(original, relocated)
    resumed = trainer(relocated)
    resumed.load_state_dict(state)
    finish(first)
    finish(resumed)
    assert_equal(resumed.state_dict(), first.state_dict())


@pytest.mark.parametrize("option", [
    {"epochs": 4}, {"batch_size": 3}, {"lr": 0.01}, {"hidden_dim": 9},
    {"max_delta": 2}, {"hard_weight": 2}, {"kl_weight": 0.2}, {"seed": 43},
])
def test_resume_rejects_changed_hyperparameters(tmp_path, option):
    cache = cache_at(tmp_path / "cache")
    state = trainer(cache).state_dict()
    with pytest.raises(ValueError, match="configuration changed"):
        trainer(cache, **option).load_state_dict(state)


@pytest.mark.parametrize("change", ["shard", "manifest"])
def test_resume_hashes_content_not_only_filenames(tmp_path, change):
    cache = cache_at(tmp_path / "cache")
    state = trainer(cache).state_dict()
    if change == "shard":
        path = cache / "shard-0000.pt"
        records = torch.load(path, weights_only=True)
        records[0]["features"][0, 0] += 1
        torch.save(records, path)
    else:
        path = cache / "manifest.json"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="content or provenance changed"):
        trainer(cache).load_state_dict(state)


def test_snapshot_is_independent_of_later_updates(tmp_path):
    value = trainer(cache_at(tmp_path / "cache"))
    value.step()
    saved = value.state_dict()
    frozen = copy.deepcopy(saved)
    value.step()
    assert_equal(saved, frozen)


def test_python_numpy_torch_rng_is_restored(tmp_path):
    cache = cache_at(tmp_path / "cache")
    value = trainer(cache)
    value.step()
    saved = value.state_dict()
    expected = (random.random(), np.random.rand(), torch.rand(4))
    resumed = trainer(cache)
    resumed.load_state_dict(saved)
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert_equal(actual, expected)


def test_export_compatible_with_navigation_head_loader(tmp_path):
    value = trainer(cache_at(tmp_path / "cache"))
    value.step()
    path = tmp_path / "head.pt"
    torch.save(value.head_payload(), path)
    loaded, metadata = load_head_checkpoint(path, expected_provenance=value.provenance)
    assert_equal(loaded.state_dict(), value.head.state_dict())
    assert metadata["metrics"]["kind"] == "offline_training_only"
    assert metadata["train_args"]["global_step"] == 1


@pytest.mark.parametrize("changes", [
    {"epoch": -1}, {"record_cursor": 1}, {"global_step": 300},
    {"epoch_totals": {"num_records": 0}}, {"history": [{}]},
    {"scheduler": {}}, {"scaler": {}}, {"head": {"bad": torch.tensor(float("nan"))}},
])
def test_rejects_invalid_resume_state(tmp_path, changes):
    cache = cache_at(tmp_path / "cache")
    saved = trainer(cache).state_dict()
    saved.update(changes)
    with pytest.raises(ValueError):
        trainer(cache).load_state_dict(saved)


def test_validation_cache_is_rejected(tmp_path):
    cache = cache_at(tmp_path / "cache")
    path = cache / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["split"] = "val_unseen"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="validation data is forbidden"):
        trainer(cache)


@pytest.mark.parametrize("change", ["lr", "missing_moments", "wrong_step", "wrong_shape"])
def test_incomplete_or_inconsistent_optimizer_is_rejected(tmp_path, change):
    cache = cache_at(tmp_path / "cache")
    value = trainer(cache)
    value.step()
    saved = value.state_dict()
    optimizer = saved["optimizer"]
    if change == "lr":
        optimizer["param_groups"][0]["lr"] = 123
    elif change == "missing_moments":
        optimizer["state"] = {}
    elif change == "wrong_step":
        optimizer["state"][0]["step"] = torch.tensor(20.0)
    elif change == "wrong_shape":
        optimizer["state"][0]["exp_avg"] = torch.zeros(1)
    with pytest.raises(ValueError, match="optimizer"):
        trainer(cache).load_state_dict(saved)


def test_iterator_is_created_once_per_epoch_not_per_step(tmp_path, monkeypatch):
    import vln_improve.resumable as module
    calls = []
    original = module.iter_records

    def observed(manifests, seed):
        calls.append(seed)
        return original(manifests, seed)

    monkeypatch.setattr(module, "iter_records", observed)
    value = trainer(cache_at(tmp_path / "cache"))
    finish(value)
    assert calls == [42, 43, 44]
