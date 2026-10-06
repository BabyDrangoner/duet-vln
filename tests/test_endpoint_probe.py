import copy
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vln_improve.checkpoint_store import CheckpointError, CheckpointStore
from vln_improve.endpoint_probe import (
    CACHE_SCHEMA, FEATURE_SCHEMA, EndpointProbe, EndpointTrainer,
    build_endpoint_features, endpoint_probabilities, episode_normalized_bce,
    load_endpoint_cache, load_endpoint_head, train_endpoint_probe,
)
from vln_improve.protocol import file_sha256, object_sha256


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _cache(root, split, *, episodes=3):
    root.mkdir(parents=True)
    common = {"base_checkpoint_sha256": "1" * 64, "feature_sha256": "2" * 64,
              "annotation_sha256": "3" * 64, "connectivity_sha256": "4" * 64,
              "model": {"batch_size": 1, "max_action_len": 15},
              "upstream_lock": {"commit": "test-fixture"}, "partition_seed": 20261003,
              "dev_fraction": .2, "torch_version": str(torch.__version__)}
    runtime = {k: common[k] for k in ("model", "upstream_lock", "base_checkpoint_sha256", "torch_version")}
    usage = "training" if split == "train_fit" else "analysis_only"
    identity = {"schema": "duet_endpoint_features_identity_v1", "split": split, "usage": usage,
                "collection_identity_sha256": "5" * 64, "runtime": runtime,
                "common_provenance": common, "feature_schema": FEATURE_SCHEMA, "feature_dim": 1536,
                "implementation": {"endpoint.py": "6" * 64}}
    identity_sha = object_sha256(identity)
    files, counts = [], {}
    generator = torch.Generator().manual_seed(123)
    for index in range(episodes):
        n = 2 + index % 2
        association = {"episode_id": f"episode-{split}-{index}", "scan_id": f"scene-{split}",
                       "instr_id": f"instruction-{split}-{index}"}
        distance = torch.tensor([3.0] + [1.0] * (n - 1), dtype=torch.float64)
        features = torch.randn(n, 1536, generator=generator) * .1
        features[:, 0] = (distance < 3).float() * 2 - 1
        payload = {"schema": CACHE_SCHEMA, "identity_sha256": identity_sha,
                   "association": association, "input_manifest_sha256": "7" * 64,
                   "features": features, "labels": (distance < 3).float(), "steps": torch.arange(n),
                   "viewpoints": [f"view-{j}" for j in range(n)], "distance_to_goal": distance,
                   "base_stop_probability": torch.linspace(.2, .9, n)}
        name = association["episode_id"] + ".pt"
        torch.save(payload, root / name)
        files.append({"name": name, "sha256": file_sha256(root / name), "association": association,
                      "num_states": n, "positives": n - 1})
        scan = counts.setdefault(association["scan_id"], {"episodes": 0, "states": 0, "positives": 0})
        scan["episodes"] += 1; scan["states"] += n; scan["positives"] += n - 1
    manifest = {"schema": CACHE_SCHEMA, "feature_dim": 1536, "split": split, "usage": usage,
                "identity": identity, "identity_sha256": identity_sha, "files": files,
                "summary": {"episodes": episodes, "states": sum(x["num_states"] for x in files),
                            "positives": sum(x["positives"] for x in files), "per_scan": counts,
                            "all_state_three_branch_exact_logit_parity": True, "resources": {"wall_seconds": 1}}}
    _write_json(root / "IDENTITY.json", identity)
    _commit_manifest(root, manifest)
    return root


def _commit_manifest(root, manifest):
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "COMMITTED.json", {"manifest_sha256": file_sha256(root / "manifest.json")})


def _pair(tmp_path):
    return _cache(tmp_path / "fit", "train_fit"), _cache(tmp_path / "dev", "train_dev", episodes=2)


def _trainer(fit, dev, **kwargs):
    return EndpointTrainer(load_endpoint_cache(fit, expected_split="train_fit"),
                           load_endpoint_cache(dev, expected_split="train_dev"), **kwargs)


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def test_two_crossmodal_stop_tokens_are_copied_without_policy_mutation_or_labels():
    inputs = {"gmap_vpids": [[None, "a"]], "vp_cand_vpids": [[None, "b"]],
              "gmap_masks": torch.ones(1, 2, dtype=torch.bool), "vp_masks": torch.ones(1, 3, dtype=torch.bool)}
    outputs = {"gmap_embeds": torch.randn(1, 2, 768, requires_grad=True),
               "vp_embeds": torch.randn(1, 3, 768, requires_grad=True), "fused_logits": torch.randn(1, 2)}
    original = {k: v.detach().clone() for k, v in outputs.items()}
    features = build_endpoint_features(inputs, outputs)
    assert features.shape == (1, 1536) and features.dtype == torch.float32 and not features.requires_grad
    assert torch.equal(features[0, :768], outputs["gmap_embeds"][0, 0])
    assert torch.equal(features[0, 768:], outputs["vp_embeds"][0, 0])
    features.zero_()
    for key in outputs:
        assert torch.equal(outputs[key], original[key])
    head = EndpointProbe()
    x = torch.randn(4, 1536, requires_grad=True)
    head(x).sum().backward()
    assert x.grad is None and all(p.grad is not None for p in head.parameters())
    assert endpoint_probabilities(head, x).shape == (4,)
    assert sum(p.numel() for p in head.parameters()) == 196865
    inputs["gmap_vpids"][0][0] = "not-stop"
    with pytest.raises(ValueError, match="STOP"):
        build_endpoint_features(inputs, outputs)


def test_episode_weighting_does_not_reward_longer_paths_or_reweight_classes():
    logits = torch.tensor([2.0, -1.0, -1.0, -1.0], requires_grad=True)
    labels = torch.tensor([1.0, 1.0, 1.0, 1.0])
    actual = episode_normalized_bce(logits, labels, [1, 3])
    expected = (F.softplus(torch.tensor(-2.0)) + F.softplus(torch.tensor(1.0))) / 2
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.allclose(logits.grad[1:].sum(), torch.sigmoid(torch.tensor(-1.0)) / 2 - .5)
    assert not torch.allclose(actual, F.binary_cross_entropy_with_logits(logits, labels))


def test_cache_preserves_all_states_and_exact_three_metre_label_boundary(tmp_path):
    root = _cache(tmp_path / "fit", "train_fit")
    cache = load_endpoint_cache(root, expected_split="train_fit")
    assert len(cache.episodes) == 3 and len(cache.episodes[1]["steps"]) == 3
    assert cache.episodes[0]["labels"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="split"):
        load_endpoint_cache(root, expected_split="train_dev")
    with pytest.raises(ValueError, match="only"):
        load_endpoint_cache(root, expected_split="val_unseen")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["summary"]["resources"]["wall_seconds"] = 999
    _commit_manifest(root, manifest)
    changed = load_endpoint_cache(root, expected_split="train_fit")
    assert changed.manifest_sha256 != cache.manifest_sha256
    assert changed.data_sha256 == cache.data_sha256


@pytest.mark.parametrize("damage", ["commit", "bytes", "label", "steps", "nan", "usage", "extra", "association"])
def test_corrupted_or_incomplete_cache_is_rejected(tmp_path, damage):
    root = _cache(tmp_path / "fit", "train_fit")
    manifest = json.loads((root / "manifest.json").read_text())
    item = manifest["files"][0]
    path = root / item["name"]
    if damage == "commit":
        (root / "COMMITTED.json").unlink()
    elif damage == "bytes":
        path.write_bytes(b"broken")
    elif damage == "usage":
        manifest["usage"] = "analysis_only"
        _commit_manifest(root, manifest)
    elif damage == "extra":
        shutil.copyfile(path, root / "episode-extra.pt")
    else:
        payload = torch.load(path, weights_only=True)
        if damage == "label": payload["labels"][0] = 1
        if damage == "steps": payload["steps"][0] = 1
        if damage == "nan": payload["features"][0, 0] = torch.nan
        if damage == "association": payload["association"]["instr_id"] = "wrong"
        torch.save(payload, path)
        item["sha256"] = file_sha256(path)
        _commit_manifest(root, manifest)
    with pytest.raises(ValueError):
        load_endpoint_cache(root, expected_split="train_fit")


def test_training_reads_only_features_labels_and_rejects_scene_or_provenance_overlap(tmp_path):
    fit, dev = _pair(tmp_path)
    trainer = _trainer(fit, dev, epochs=2, batch_episodes=2)
    batch = [{k: v for k, v in trainer.train_cache.episodes[0].items() if k in ("features", "labels")}]
    features, labels, lengths = trainer._batch(batch)
    assert features.shape == (2, 1536) and labels.shape == (2,) and lengths == [2]
    train = load_endpoint_cache(fit, expected_split="train_fit")
    validation = load_endpoint_cache(dev, expected_split="train_dev")
    validation.episodes[0]["association"]["scan_id"] = train.episodes[0]["association"]["scan_id"]
    with pytest.raises(ValueError, match="scene"):
        EndpointTrainer(train, validation)
    validation = load_endpoint_cache(dev, expected_split="train_dev")
    validation.manifest["identity"]["common_provenance"]["feature_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="provenance"):
        EndpointTrainer(train, validation)


@pytest.mark.parametrize("pause_after", [1, 2])
def test_cpu_full_training_matches_mid_epoch_and_pending_dev_resume_exactly(tmp_path, pause_after):
    fit, dev = _pair(tmp_path)
    options = dict(epochs=3, batch_episodes=2, checkpoint_every_steps=1, keep_local=1, keep_backup=1)
    uninterrupted = train_endpoint_probe(fit, dev, tmp_path / "full-local", tmp_path / "full-drive", **options)
    paused = train_endpoint_probe(fit, dev, tmp_path / "res-local", tmp_path / "res-drive", max_steps=pause_after, **options)
    assert paused["status"] == "interrupted" and paused["global_step"] == pause_after
    assert paused["pending_dev"] == (pause_after == 2)
    # A replacement VM has no local checkpoint directory.
    shutil.rmtree(tmp_path / "res-local")
    resumed = train_endpoint_probe(fit, dev, tmp_path / "res-local", tmp_path / "res-drive", **options)
    assert resumed["status"] == "complete" and resumed["resumed"]
    assert resumed["history"] == uninterrupted["history"] and resumed["global_step"] == 6
    complete_store = CheckpointStore(tmp_path / "full-local", tmp_path / "full-drive")
    resumed_store = CheckpointStore(tmp_path / "res-local", tmp_path / "res-drive")
    full_state, full_head, _ = complete_store.restore()
    state, head, _ = resumed_store.restore()
    assert state["config"]["optimizer"] == "AdamW"
    _equal(full_state, state); _equal(full_head, head)
    _, best_head, best_manifest = resumed_store.restore("best")
    assert best_manifest["is_best"] and best_head["epoch"] == resumed["best_epoch"]
    assert "not_navigation_best" in best_head["selection_purpose"]
    head_file = Path(best_manifest["local_path"]) / "head.pt"
    model, meta = load_endpoint_head(head_file, expected_common_identity=best_head["common_identity"])
    assert model(torch.zeros(1, 1536)).shape == (1,)
    assert meta["epoch"] == resumed["best_epoch"]
    assert len(list((tmp_path / "res-drive" / "snapshots").glob("step-*"))) <= 2


@pytest.mark.parametrize("damage", ["cursor", "adam", "data", "pending", "weights", "best"])
def test_resume_rejects_inconsistent_training_state(tmp_path, damage):
    fit, dev = _pair(tmp_path)
    trainer = _trainer(fit, dev, epochs=2, batch_episodes=2)
    trainer.step()
    saved = trainer.state_dict()
    if damage == "cursor": saved["episode_cursor"] = 1
    if damage == "adam": next(iter(saved["optimizer"]["state"].values()))["step"] += 1
    if damage == "data": saved["data_identity"]["train"] = "f" * 64
    if damage == "pending": saved["pending_dev"] = True
    if damage == "weights": next(iter(saved["head"].values())).fill_(torch.nan)
    if damage == "best": saved["best_dev_bce"] = .1
    with pytest.raises(ValueError):
        _trainer(fit, dev, epochs=2, batch_episodes=2).load_state_dict(saved)


def test_changed_configuration_and_all_corrupt_backup_do_not_restart_training(tmp_path):
    fit, dev = _pair(tmp_path)
    local, drive = tmp_path / "local", tmp_path / "drive"
    train_endpoint_probe(fit, dev, local, drive, epochs=2, batch_episodes=2, max_steps=1)
    with pytest.raises(ValueError, match="identity"):
        train_endpoint_probe(fit, dev, local, drive, epochs=2, batch_episodes=2, lr=.01)
    shutil.rmtree(local)
    for path in (drive / "snapshots").glob("step-*/state.pt"):
        path.write_bytes(b"corrupt")
    with pytest.raises(CheckpointError, match="No loadable"):
        train_endpoint_probe(fit, dev, local, drive, epochs=2, batch_episodes=2)


def test_training_cli_fixed_pilot_and_cloud_summary(tmp_path):
    from train_endpoint_probe import main
    fit, dev = _pair(tmp_path)
    local, backup = tmp_path / "run", tmp_path / "backup"
    result = main(["--train-cache", str(fit), "--dev-cache", str(dev), "--local-run", str(local),
                   "--backup-run", str(backup), "--allow-local-backup-for-tests"])
    assert result["completed_epochs"] == 20 and result["status"] == "complete"
    assert (local / "training-summary.json").read_bytes() == (backup / "training-summary.json").read_bytes()
    state, _, _ = CheckpointStore(local, backup).restore()
    assert state["config"]["lr"] == .001 and state["config"]["weight_decay"] == .0001
    assert state["config"]["optimizer"] == "AdamW"
    assert state["config"]["seed"] == 0 and state["config"]["batch_episodes"] == 32
