import json
import multiprocessing
import os
from pathlib import Path
import shutil

import pytest
import torch

from vln_improve import checkpoint_store as module
from vln_improve.checkpoint_store import BackupError, CheckpointError, CheckpointStore


def make_store(tmp_path, **kwargs):
    return CheckpointStore(tmp_path / "local", tmp_path / "drive", **kwargs)


def save(store, step, *, best=False):
    return store.save({"step": step, "optimizer": {"tensor": torch.tensor([step])}},
                      {"model": torch.tensor([step + 1.0])},
                      step=step, is_best=best, metrics={"spl": float(step)})


def ids(root):
    return {path.name for path in (root / "snapshots").iterdir() if module.ID_RE.fullmatch(path.name)}


def test_roundtrip_step_zero_unique_ids_and_distinct_best_latest(tmp_path):
    store = make_store(tmp_path)
    first = save(store, 0, best=True)
    second = save(store, 0)
    assert first != second
    state, head, manifest = store.restore()
    assert manifest["checkpoint_id"] == second
    assert manifest["step"] == 0
    assert manifest["metrics"] == {"spl": 0.0}
    assert Path(manifest["local_path"]).is_dir()
    torch.testing.assert_close(state["optimizer"]["tensor"], torch.tensor([0]))
    torch.testing.assert_close(head["model"], torch.tensor([1.0]))
    assert store.restore("best")[2]["checkpoint_id"] == first


def test_save_copy_failure_preserves_all_local_points_and_backup_refs(tmp_path, monkeypatch):
    store = make_store(tmp_path, keep_local=1, keep_backup=1)
    first = save(store, 1, best=True)
    refs = {name: (store.backup_dir / name).read_bytes() for name in ("latest.json", "best.json")}
    original = shutil.copyfile

    def fail_copy(source, target):
        raise OSError("drive disconnected")

    monkeypatch.setattr(shutil, "copyfile", fail_copy)
    with pytest.raises(BackupError, match="local checkpoint retained"):
        save(store, 2, best=True)
    failed_id = (ids(store.local_dir) - {first}).pop()
    assert len(ids(store.local_dir)) == 2
    assert ids(store.backup_dir) == {first}
    assert all((store.backup_dir / name).read_bytes() == data for name, data in refs.items())
    assert not list((store.backup_dir / "snapshots").glob(".pending-*"))
    monkeypatch.setattr(shutil, "copyfile", original)
    third = save(store, 3)
    assert failed_id in ids(store.local_dir)  # Never prune the unbacked state.
    assert third in ids(store.local_dir)


def test_copy_checksum_failure_does_not_publish_pointer(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    first = save(store, 1)
    original = shutil.copyfile

    def corrupt_copy(source, target):
        result = original(source, target)
        if Path(target).name == "state.pt":
            with open(target, "ab") as stream:
                stream.write(b"broken")
        return result

    monkeypatch.setattr(shutil, "copyfile", corrupt_copy)
    with pytest.raises(BackupError, match="checksum"):
        save(store, 2)
    assert json.loads((store.backup_dir / "latest.json").read_text())["checkpoint_id"] == first
    assert len(ids(store.local_dir)) == 2
    assert ids(store.backup_dir) == {first}


def test_pointer_write_failure_restores_both_old_backup_refs(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    save(store, 1, best=True)
    before = {name: (store.backup_dir / name).read_bytes() for name in ("latest.json", "best.json")}
    original = module._atomic_bytes
    failed = False

    def failing_atomic(path, data):
        nonlocal failed
        if path == store.backup_dir / "best.json" and not failed:
            failed = True
            raise OSError("pointer write interrupted")
        return original(path, data)

    monkeypatch.setattr(module, "_atomic_bytes", failing_atomic)
    with pytest.raises(BackupError, match="pointer write interrupted"):
        save(store, 2, best=True)
    assert all((store.backup_dir / name).read_bytes() == data for name, data in before.items())
    assert len(ids(store.local_dir)) == 2


def test_retention_keeps_recent_plus_best_and_leaves_unknown_content(tmp_path):
    store = make_store(tmp_path, keep_local=2, keep_backup=3)
    best = save(store, 0, best=True)
    unknown = store.local_dir / "snapshots" / "my-experiment"
    unknown.mkdir()
    (unknown / "notes.txt").write_text("keep")
    last = [save(store, step) for step in range(1, 8)]
    assert ids(store.local_dir) == {best, *last[-2:]}
    assert ids(store.backup_dir) == {best, *last[-3:]}
    assert (unknown / "notes.txt").read_text() == "keep"


def test_latest_corrupt_backup_falls_back_to_previous_verified_backup(tmp_path):
    store = make_store(tmp_path)
    first = save(store, 1)
    second = save(store, 2)
    (store.backup_dir / "snapshots" / second / "state.pt").write_bytes(b"truncated")
    # Even though a newer local copy is valid, the durable backup is preferred.
    assert store.restore()[2]["checkpoint_id"] == first


def test_best_corruption_fallback_never_selects_unvalidated_latest(tmp_path):
    store = make_store(tmp_path)
    first = save(store, 1, best=True)
    second = save(store, 2, best=True)
    save(store, 3)
    (store.backup_dir / "snapshots" / second / "head.pt").write_bytes(b"bad")
    assert store.restore("best")[2]["checkpoint_id"] == first


def test_best_not_created_by_periodic_checkpoints(tmp_path):
    store = make_store(tmp_path)
    save(store, 1)
    with pytest.raises(FileNotFoundError, match="best"):
        store.restore("best")


@pytest.mark.parametrize("broken_ref", [b"{bad", b'{"checkpoint_id":"../../evil","run_id":"x"}'])
def test_corrupt_pointer_scans_only_committed_snapshots(tmp_path, broken_ref):
    store = make_store(tmp_path)
    first = save(store, 1)
    second = save(store, 2)
    (store.backup_dir / "latest.json").write_bytes(broken_ref)
    (store.backup_dir / "snapshots" / second / "COMMITTED").unlink()
    assert store.restore()[2]["checkpoint_id"] == first


def test_fresh_colab_local_directory_adopts_persistent_run_and_restores(tmp_path):
    store = make_store(tmp_path)
    identity = save(store, 3, best=True)
    shutil.rmtree(store.local_dir)
    new = make_store(tmp_path)
    assert new.run_id == store.run_id
    assert new.restore()[2]["checkpoint_id"] == identity
    assert new.restore("best")[0]["step"] == 3


def test_corrupt_local_copy_is_quarantined_and_repaired_from_drive(tmp_path):
    store = make_store(tmp_path)
    identity = save(store, 2)
    path = store.local_dir / "snapshots" / identity
    (path / "state.pt").write_bytes(b"bad")
    assert store.restore()[0]["step"] == 2
    assert (path / "state.pt").read_bytes() != b"bad"
    quarantine = list((store.local_dir / "snapshots").glob(".corrupt-*"))
    assert len(quarantine) == 1
    assert (quarantine[0] / "state.pt").read_bytes() == b"bad"


def test_different_run_identity_rejected_before_overwriting(tmp_path):
    first = make_store(tmp_path / "a")
    second = make_store(tmp_path / "b")
    marker = (second.backup_dir / "STORE.json").read_bytes()
    with pytest.raises(CheckpointError, match="different runs"):
        CheckpointStore(first.local_dir, second.backup_dir)
    assert (second.backup_dir / "STORE.json").read_bytes() == marker


def test_lock_reentrant_and_excludes_second_store_writer(tmp_path):
    first = make_store(tmp_path)
    second = make_store(tmp_path)
    with first.lock():
        with first.lock():
            save(first, 0)
        with pytest.raises(CheckpointError, match="writer lock"):
            save(second, 1)
    save(second, 2)


def test_local_serialization_failure_does_not_publish_incomplete_snapshot(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    first = save(store, 1)

    def fail_save(payload, stream):
        stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        save(store, 2)
    assert ids(store.local_dir) == ids(store.backup_dir) == {first}
    assert not list((store.local_dir / "snapshots").glob(".pending-*"))


def test_manifest_corruption_cannot_silently_change_best_status(tmp_path):
    store = make_store(tmp_path)
    identity = save(store, 0)
    for root in (store.local_dir, store.backup_dir):
        path = root / "snapshots" / identity / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["is_best"] = True
        path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        store.restore("best")


def test_unbacked_checkpoint_survives_and_is_local_fallback(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    monkeypatch.setattr(store, "_copy_snapshot", lambda *_: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(BackupError):
        save(store, 1)
    assert store.restore()[0]["step"] == 1


@pytest.mark.parametrize("kwargs", [{"keep_local": 0}, {"keep_backup": -1}, {"keep_local": True}])
def test_invalid_retention_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_store(tmp_path, **kwargs)


def test_nested_roots_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-nested"):
        CheckpointStore(tmp_path / "run", tmp_path / "run" / "drive")


def test_payload_uses_safe_torch_loading(tmp_path):
    store = make_store(tmp_path)
    store.save({"unsafe": Path("/tmp/unsafe-object")}, {}, step=0)
    with pytest.raises(CheckpointError, match="No loadable checkpoint"):
        store.restore()


def test_corrupt_existing_run_cannot_be_mistaken_for_a_fresh_run(tmp_path):
    store = make_store(tmp_path)
    identity = save(store, 1)
    for root in (store.local_dir, store.backup_dir):
        (root / "snapshots" / identity / "head.pt").write_bytes(b"bad")
        (root / "latest.json").unlink()
    with pytest.raises(CheckpointError, match="failed validation"):
        store.restore()


def test_fresh_run_is_distinct_from_corrupt_run(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.restore()


def test_best_survives_retention_when_best_pointer_is_corrupt(tmp_path):
    store = make_store(tmp_path, keep_local=1, keep_backup=1)
    identity = save(store, 0, best=True)
    for root in (store.local_dir, store.backup_dir):
        (root / "best.json").write_bytes(b"truncated")
    save(store, 1)
    save(store, 2)
    assert identity in ids(store.local_dir)
    assert identity in ids(store.backup_dir)
    assert store.restore("best")[2]["checkpoint_id"] == identity


def _kill_between_reference_writes(local, backup, exit_at):
    store = CheckpointStore(local, backup)
    original = module._atomic_bytes

    def abrupt_exit(path, data):
        original(path, data)
        if path == store.backup_dir / exit_at:
            os._exit(9)  # No Python finally blocks: equivalent to a killed VM.

    module._atomic_bytes = abrupt_exit
    save(store, 2, best=True)


@pytest.mark.parametrize("exit_at", ["latest.json", "best.json", "refs.json"])
def test_abrupt_exit_cannot_split_canonical_latest_and_best(tmp_path, exit_at):
    store = make_store(tmp_path)
    first = save(store, 1, best=True)
    process = multiprocessing.get_context("fork").Process(
        target=_kill_between_reference_writes,
        args=(str(store.local_dir), str(store.backup_dir), exit_at),
    )
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("abrupt-exit child did not finish")
    assert process.exitcode == 9
    reopened = make_store(tmp_path)
    latest = reopened.restore("latest")[2]["checkpoint_id"]
    best = reopened.restore("best")[2]["checkpoint_id"]
    assert latest == best
    if exit_at != "refs.json":
        assert latest == first
    else:
        assert latest != first


def test_corrupt_canonical_refs_falls_back_to_verified_snapshots(tmp_path):
    store = make_store(tmp_path)
    best = save(store, 1, best=True)
    latest = save(store, 2)
    (store.backup_dir / "refs.json").write_bytes(b"partial")
    assert store.restore()[2]["checkpoint_id"] == latest
    assert store.restore("best")[2]["checkpoint_id"] == best


def test_mirrors_are_not_authoritative(tmp_path):
    store = make_store(tmp_path)
    first = save(store, 1, best=True)
    second = save(store, 2)
    (store.backup_dir / "best.json").write_bytes((store.backup_dir / "latest.json").read_bytes())
    (store.backup_dir / "latest.json").write_bytes(b"corrupt")
    assert store.restore()[2]["checkpoint_id"] == second
    assert store.restore("best")[2]["checkpoint_id"] == first
