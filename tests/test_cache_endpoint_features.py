import copy
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cache_endpoint_features import (
    verify_logits, copy_verified, atomic_tensor, prepare_cache_pair,
    reuse_episode, commit_episode,
)
from vln_improve.endpoint_probe import CACHE_SCHEMA, FEATURE_SCHEMA, load_endpoint_cache
from vln_improve.pipeline import atomic_json
from vln_improve.protocol import object_sha256, file_sha256


def test_branch_parity_rejects_change_even_when_argmax_is_unchanged():
    state = {k: torch.tensor([2., -torch.inf, 1.]) for k in
             ("base_logits", "base_global_logits", "base_local_logits")}
    out = {k: torch.tensor([[2., -torch.inf, 1.]]) for k in
           ("fused_logits", "global_logits", "local_logits")}
    verify_logits(state, out)
    out["local_logits"][0, 2] += 1e-6
    with pytest.raises(ValueError, match="local_logits"):
        verify_logits(state, out)


def test_backup_preserves_serialized_bytes_and_rejects_disagreement(tmp_path):
    src, dst = tmp_path / "episode.pt", tmp_path / "cloud.pt"
    atomic_tensor(src, {"tensor": torch.ones(2, 3)})
    copy_verified(src, dst)
    assert src.read_bytes() == dst.read_bytes()
    copy_verified(src, dst)
    dst.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="disagree"):
        copy_verified(src, dst)
    assert dst.read_bytes() == b"corrupt"


def _identity():
    common = {"base_checkpoint_sha256": "1" * 64, "feature_sha256": "2" * 64,
              "annotation_sha256": "3" * 64, "connectivity_sha256": "4" * 64,
              "model": {"batch_size": 1}, "upstream_lock": {"commit": "fixture"},
              "partition_seed": 0, "dev_fraction": .2, "torch_version": str(torch.__version__)}
    runtime = {k: common[k] for k in ("model", "upstream_lock", "base_checkpoint_sha256", "torch_version")}
    return {"schema": "duet_endpoint_features_identity_v1", "split": "train_fit", "usage": "training",
            "collection_identity_sha256": "5" * 64, "runtime": runtime, "common_provenance": common,
            "feature_schema": FEATURE_SCHEMA, "feature_dim": 1536, "implementation": {"test.py": "6" * 64}}


def _payload(identity, index):
    return {"schema": CACHE_SCHEMA, "identity_sha256": object_sha256(identity),
            "association": {"episode_id": f"episode-{index}", "scan_id": "scan", "instr_id": str(index)},
            "input_manifest_sha256": "7" * 64, "features": torch.full((2, 1536), float(index)),
            "labels": torch.tensor([0., 1.]), "steps": torch.arange(2), "viewpoints": ["a", "b"],
            "distance_to_goal": torch.tensor([4., 2.], dtype=torch.float64),
            "base_stop_probability": torch.tensor([.2, .8])}


def _seal_file(root, name):
    return root / (name + ".seal.json")


def _finalize(root, identity):
    files = [json.loads(_seal_file(root, f"episode-{i}.pt").read_text())["file"] for i in range(2)]
    manifest = {"schema": CACHE_SCHEMA, "feature_dim": 1536, "split": "train_fit", "usage": "training",
                "identity": identity, "identity_sha256": object_sha256(identity), "files": files,
                "summary": {"episodes": 2, "states": 4, "positives": 2,
                            "per_scan": {"scan": {"episodes": 2, "states": 4, "positives": 2}},
                            "all_state_three_branch_exact_logit_parity": True,
                            "resources": {"wall_seconds": 1.23}}}
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "COMMITTED.json", {"manifest_sha256": file_sha256(root / "manifest.json")})
    return manifest


def _pair(tmp_path, complete):
    local, cloud, identity = tmp_path / "local", tmp_path / "cloud", _identity()
    names = {f"episode-{i}.pt" for i in range(2)}
    assert prepare_cache_pair(local, cloud, identity, names) is None
    for i in range(2):
        commit_episode(local, cloud, f"episode-{i}.pt", _payload(identity, i))
    if complete:
        for root in (local, cloud):
            _finalize(root, identity)
    return local, cloud, identity, names


def _reuse(local, cloud, identity, index=0):
    payload = _payload(identity, index)
    return reuse_episode(local, cloud, f"episode-{index}.pt", payload["identity_sha256"],
                         payload["association"], payload["input_manifest_sha256"])


@pytest.mark.parametrize("complete", [False, True])
def test_replacement_vm_restores_exact_committed_episode_bytes(tmp_path, complete):
    local, cloud, identity, names = _pair(tmp_path, complete)
    original = {p.name: p.read_bytes() for p in cloud.iterdir()}
    shutil.rmtree(local)
    manifest = prepare_cache_pair(local, cloud, identity, names)
    assert (manifest is not None) == complete
    for index in range(2):
        restored = _reuse(local, cloud, identity, index)
        assert restored["identity_sha256"] == object_sha256(identity)
    assert {p.name: p.read_bytes() for p in local.iterdir()} == original
    assert {p.name: p.read_bytes() for p in cloud.iterdir()} == original
    if complete:
        load_endpoint_cache(local, expected_split="train_fit")


def test_complete_missing_files_can_be_restored_from_each_other_without_resigning(tmp_path):
    local, cloud, identity, names = _pair(tmp_path, True)
    original_manifest = (local / "manifest.json").read_bytes()
    (local / "episode-0.pt").unlink()
    (cloud / "episode-1.pt").unlink()
    (cloud / "COMMITTED.json").unlink()  # interrupted collection-marker replication
    _seal_file(local, "episode-1.pt").unlink()  # full manifest remains authoritative
    assert prepare_cache_pair(local, cloud, identity, names) is not None
    for root in (local, cloud):
        assert (root / "manifest.json").read_bytes() == original_manifest
        load_endpoint_cache(root, expected_split="train_fit")
        assert len(list(root.glob("*.seal.json"))) == 2


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("altered_copy", ["local", "cloud", "both"])
def test_valid_torch_save_rewrite_with_unchanged_identity_is_rejected(tmp_path, complete, altered_copy):
    local, cloud, identity, names = _pair(tmp_path, complete)
    roots = [local, cloud] if altered_copy == "both" else [local if altered_copy == "local" else cloud]
    for root in roots:
        path = root / "episode-0.pt"
        old = torch.load(path, weights_only=True)
        changed = copy.deepcopy(old)
        changed["features"][0, 0] += .25
        assert all(changed[k] == old[k] for k in ("identity_sha256", "association", "input_manifest_sha256"))
        atomic_tensor(path, changed)  # valid, finite torch file; only the old digest reveals tampering
    before = {str(p): p.read_bytes() for root in (local, cloud) for p in root.iterdir()}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        if complete:
            prepare_cache_pair(local, cloud, identity, names)
        else:
            _reuse(local, cloud, identity)
    assert {str(p): p.read_bytes() for root in (local, cloud) for p in root.iterdir()} == before


@pytest.mark.parametrize("complete", [False, True])
def test_two_individually_valid_but_conflicting_commits_are_rejected(tmp_path, complete):
    local, cloud, identity, names = _pair(tmp_path, complete)
    path = cloud / "episode-0.pt"
    changed = torch.load(path, weights_only=True)
    changed["features"][0, 0] += 1
    atomic_tensor(path, changed)
    seal_path = _seal_file(cloud, path.name)
    seal = json.loads(seal_path.read_text())
    seal["file"]["sha256"] = file_sha256(path)
    atomic_json(seal_path, seal)
    if complete:
        _finalize(cloud, identity)
        load_endpoint_cache(cloud, expected_split="train_fit")
    with pytest.raises(ValueError, match="disagree"):
        if complete:
            prepare_cache_pair(local, cloud, identity, names)
        else:
            _reuse(local, cloud, identity)


def test_unsealed_partial_pt_is_not_adopted_or_resigned(tmp_path):
    local, cloud, identity, names = _pair(tmp_path, False)
    _seal_file(local, "episode-0.pt").unlink()
    with pytest.raises(ValueError, match="uncommitted.*no seal"):
        _reuse(local, cloud, identity)
    assert not _seal_file(local, "episode-0.pt").exists()
    with pytest.raises(ValueError, match="overwrite"):
        commit_episode(local, cloud, "episode-0.pt", _payload(identity, 0))


@pytest.mark.parametrize("complete", [False, True])
def test_missing_from_both_copies_does_not_recompute_a_committed_episode(tmp_path, complete):
    local, cloud, identity, names = _pair(tmp_path, complete)
    for root in (local, cloud):
        (root / "episode-0.pt").unlink()
    with pytest.raises(ValueError, match="missing from both"):
        if complete:
            prepare_cache_pair(local, cloud, identity, names)
        else:
            _reuse(local, cloud, identity)


@pytest.mark.parametrize("damage", ["commit", "manifest", "foreign_identity", "extra_episode"])
def test_invalid_completed_cache_is_not_reopened_as_partial(tmp_path, damage):
    local, cloud, identity, names = _pair(tmp_path, True)
    if damage == "commit":
        atomic_json(local / "COMMITTED.json", {"manifest_sha256": "0" * 64})
    elif damage == "manifest":
        value = json.loads((local / "manifest.json").read_text())
        value["summary"]["states"] += 1
        atomic_json(local / "manifest.json", value)
    elif damage == "foreign_identity":
        altered = copy.deepcopy(identity)
        altered["implementation"]["test.py"] = "f" * 64
        atomic_json(local / "IDENTITY.json", altered)
    else:
        shutil.copyfile(local / "episode-0.pt", local / "episode-extra.pt")
    with pytest.raises(ValueError):
        prepare_cache_pair(local, cloud, identity, names)


@pytest.mark.parametrize("field", ["input_manifest_sha256", "association", "labels"])
def test_sealed_payload_must_match_source_and_label_contract(tmp_path, field):
    local, cloud, identity, names = _pair(tmp_path, False)
    expected = _payload(identity, 0)
    if field == "input_manifest_sha256":
        expected[field] = "f" * 64
    elif field == "association":
        expected[field]["instr_id"] = "different"
    else:
        for root in (local, cloud):
            path = root / "episode-0.pt"
            payload = torch.load(path, weights_only=True)
            payload["labels"][0] = 1
            atomic_tensor(path, payload)
            seal_path = _seal_file(root, path.name)
            seal = json.loads(seal_path.read_text())
            seal["file"]["sha256"] = file_sha256(path)
            atomic_json(seal_path, seal)
    with pytest.raises(ValueError):
        reuse_episode(local, cloud, "episode-0.pt", expected["identity_sha256"],
                      expected["association"], expected["input_manifest_sha256"])
