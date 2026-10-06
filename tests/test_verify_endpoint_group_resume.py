import copy
import json
from pathlib import Path
import sys

import pytest
import torch

from vln_improve.checkpoint_store import CheckpointError
from vln_improve.endpoint_group_training import load_endpoint_group_cache
from vln_improve.protocol import file_sha256, object_sha256
from test_endpoint_group_training import _sources

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from verify_endpoint_group_resume import (
    _delete_owned_interrupted, _store_inventory, acceptance_subset, assert_exact, run_acceptance,
)
import verify_endpoint_group_resume as acceptance_module


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def caches(tmp_path):
    fit = _sources(tmp_path, "train_fit", 9, 0)
    dev = _sources(tmp_path, "train_dev", 5, 20)
    return load_endpoint_group_cache(*fit, "train_fit"), load_endpoint_group_cache(*dev, "train_dev")


def test_acceptance_subset_has_explicit_source_and_recomputed_identity(caches):
    train, _ = caches
    subset = acceptance_subset(train, 8)
    assert len(train.groups) == 9 and len(subset.groups) == 8
    assert subset.source_identity["subset_of"]["data_sha256"] == train.data_sha256
    assert subset.source_identity["pair_ids"] == [g["pair"]["selection_hash"] for g in train.groups[:8]]
    assert subset.support["groups"] == 8 and subset.support["original_paths"] == 16
    assert subset.data_sha256 == object_sha256(subset.source_identity) != train.data_sha256
    assert subset.source_identity["ordered_pair_sha256"] == train.source_identity["ordered_pair_sha256"][:8]
    with pytest.raises(ValueError): acceptance_subset(train, 10)


def test_real_whole_acceptance_restores_from_deleted_vm_and_seals_both_reports(tmp_path, caches):
    train, dev = caches
    before = {p: file_sha256(p) for cache in (train, dev) for root in (cache.pair_cache.root, cache.control_cache.root)
              for p in root.rglob("*") if p.is_file()}
    calls = []
    result = run_acceptance(train, dev, tmp_path/"accept-local", tmp_path/"accept-drive", device="cpu",
                            verify_backup=lambda: calls.append(True))
    assert result["status"] == "passed" and result["scope"] == "acceptance_only"
    assert result["not_navigation_result"] is True and all(result["checks"].values())
    assert result["source_counts"] == {"full_train": 9, "full_dev": 5, "subset_train": 8, "subset_dev": 4}
    assert result["runs"]["paused"]["global_step"] == 2
    assert result["runs"]["resumed"]["resumed"] is True
    assert result["runs"]["resumed"]["global_step"] == 12
    assert result["content_sha256"] == object_sha256({k:v for k,v in result.items() if k != "content_sha256"})
    assert len(calls) > 10
    for name in ("ACCEPTANCE.json", "resume-acceptance.json", "COMMITTED.json"):
        assert (tmp_path/"accept-local"/name).read_bytes() == (tmp_path/"accept-drive"/name).read_bytes()
    seal = json.loads((tmp_path/"accept-local"/"COMMITTED.json").read_text())
    assert seal["report_sha256"] == file_sha256(tmp_path/"accept-local"/"resume-acceptance.json")
    assert all(file_sha256(p) == sha for p,sha in before.items())
    with pytest.raises(ValueError, match="refusing overwrite"):
        run_acceptance(train, dev, tmp_path/"accept-local", tmp_path/"accept-drive", device="cpu", verify_backup=lambda: None)
    child = tmp_path/"accept-local"/"interrupted"
    inventory = _store_inventory(child)
    (child/"foreign-note").write_text("preserve")
    marker = json.loads((tmp_path/"accept-local"/"ACCEPTANCE.json").read_text())
    with pytest.raises(ValueError, match="unexpected"):
        _delete_owned_interrupted(tmp_path/"accept-local", marker, inventory)
    assert (child/"foreign-note").read_text() == "preserve"


def test_unknown_roots_nested_sources_and_symlinks_are_never_removed(tmp_path, caches):
    train, dev = caches
    local, backup = tmp_path/"local", tmp_path/"backup"
    local.mkdir(); (local/"user-file").write_text("keep")
    with pytest.raises(ValueError, match="refusing overwrite"):
        run_acceptance(train, dev, local, backup, device="cpu", verify_backup=lambda: None)
    assert (local/"user-file").read_text() == "keep" and not backup.exists()
    with pytest.raises(ValueError, match="non-nested"):
        run_acceptance(train, dev, train.pair_cache.root/"unsafe", backup, device="cpu", verify_backup=lambda: None)
    link = tmp_path/"link"; link.symlink_to(local, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        run_acceptance(train, dev, link, backup, device="cpu", verify_backup=lambda: None)
    child = local/"interrupted"; child.mkdir(); (child/"foreign").write_text("keep")
    with pytest.raises(ValueError, match="unexpected"):
        _store_inventory(child)
    assert (child/"foreign").read_text() == "keep"


def test_full_state_comparison_catches_optimizer_rng_and_history_changes():
    value = {"head": torch.tensor([1.]), "optimizer": {"exp_avg": torch.zeros(2)},
             "rng": (1, [2, 3]), "history": [{"metric": .5}]}
    for mutation in ("optimizer", "rng", "history", "dtype"):
        other = copy.deepcopy(value)
        if mutation == "optimizer": other["optimizer"]["exp_avg"][0] = 1
        elif mutation == "rng": other["rng"] = [1, [2, 3]]
        elif mutation == "history": other["history"][0]["metric"] = .6
        else: other["head"] = other["head"].double()
        with pytest.raises(ValueError): assert_exact(value, other)
    with pytest.raises(ValueError): assert_exact(torch.tensor([0.]), torch.tensor([-0.]))


def test_cloud_tamper_is_rejected_before_local_copy_is_deleted(tmp_path, caches, monkeypatch):
    original = acceptance_module._backup_payload
    def corrupt(store, checkpoint_id):
        data = store.backup_dir/"snapshots"/checkpoint_id/"state.pt"
        payload = torch.load(data, map_location="cpu", weights_only=True)
        next(iter(payload["head"].values())).add_(1)
        torch.save(payload, data)  # Still a valid torch archive with unchanged identity fields.
        return original(store, checkpoint_id)
    monkeypatch.setattr(acceptance_module, "_backup_payload", corrupt)
    with pytest.raises(CheckpointError):
        run_acceptance(*caches, tmp_path/"local", tmp_path/"drive", device="cpu", verify_backup=lambda: None)
    assert list((tmp_path/"local"/"interrupted"/"snapshots").glob("*/state.pt"))
    assert not (tmp_path/"local"/"COMMITTED.json").exists()
