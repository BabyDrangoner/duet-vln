import hashlib
import json
from pathlib import Path
import shutil
import tarfile

import pytest
import torch

from vln_improve import run_assets
from vln_improve.features import FEATURE_SCHEMA
from vln_improve.run_assets import prepare_inputs, snapshot_code


def cache(root, value=1, split="train_fit"):
    root.mkdir(parents=True)
    torch.save([{"fixture": value}], root / "shard.pt")
    manifest = {
        "schema_version": 1, "feature_schema": FEATURE_SCHEMA, "split": split,
        "feature_dim": 2, "num_records": 1, "shards": ["shard.pt"],
        "provenance": {
            "dataset": "fixture", "feature_id": "fixture", "base_checkpoint_sha256": "a" * 64,
            "upstream_commit": "b" * 40, "max_action_len": 15, "feedback": "argmax",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_inputs_backup_and_restore_after_original_vm_loss(tmp_path):
    source = cache(tmp_path / "original")
    backup = tmp_path / "drive"
    local = tmp_path / "new-vm"
    original = (source / "shard.pt").read_bytes()
    assert prepare_inputs([source], local, backup) == [source]
    assert (backup / "inputs/COMMITTED").is_file()
    shutil.rmtree(source)
    restored = prepare_inputs([source], local, backup)
    assert restored == [local / "inputs/cache-0000"]
    assert (restored[0] / "shard.pt").read_bytes() == original
    assert prepare_inputs([source], local, backup) == restored


def test_identical_content_at_new_absolute_source_path_is_accepted(tmp_path):
    source = cache(tmp_path / "original")
    backup = tmp_path / "drive"
    local = tmp_path / "run"
    prepare_inputs([source], local, backup)
    moved = tmp_path / "elsewhere"
    shutil.copytree(source, moved)
    assert prepare_inputs([moved], local, backup) == [moved]


def test_changed_existing_source_rejected_even_when_good_cloud_copy_exists(tmp_path):
    source = cache(tmp_path / "original")
    backup = tmp_path / "drive"
    local = tmp_path / "run"
    prepare_inputs([source], local, backup)
    (source / "shard.pt").write_bytes(b"different content")
    with pytest.raises(ValueError, match="source cache changed"):
        prepare_inputs([source], local, backup)
    assert not (local / "inputs").exists()


def test_order_is_identity_and_missing_source_does_not_hide_other_source_change(tmp_path):
    first = cache(tmp_path / "a", 1)
    second = cache(tmp_path / "b", 2)
    backup = tmp_path / "drive"
    local = tmp_path / "run"
    prepare_inputs([first, second], local, backup)
    with pytest.raises(ValueError, match="position"):
        prepare_inputs([second, first], local, backup)
    with pytest.raises(ValueError, match="length"):
        prepare_inputs([first], local, backup)
    shutil.rmtree(first)
    (second / "shard.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="position 1"):
        prepare_inputs([first, second], local, backup)


@pytest.mark.parametrize("damage", ["shard", "index", "marker", "symlink"])
def test_corrupt_cloud_backup_never_replaced_with_original(tmp_path, damage):
    source = cache(tmp_path / "source")
    backup = tmp_path / "drive"
    local = tmp_path / "run"
    prepare_inputs([source], local, backup)
    inputs = backup / "inputs"
    if damage == "shard":
        (inputs / "cache-0000/shard.pt").write_bytes(b"corrupt")
    elif damage == "index":
        (inputs / "index.json").write_bytes(b"corrupt")
    elif damage == "marker":
        (inputs / "COMMITTED").unlink()
    else:
        (inputs / "cache-0000/shard.pt").unlink()
        (inputs / "cache-0000/shard.pt").symlink_to(source / "shard.pt")
    with pytest.raises(ValueError):
        prepare_inputs([source], local, backup)


def test_interrupted_initial_backup_is_not_silently_overwritten(tmp_path):
    source = cache(tmp_path / "source")
    backup = tmp_path / "drive"
    partial = backup / ".inputs-interrupted.tmp"
    partial.mkdir(parents=True)
    (partial / "partial").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="uncommitted"):
        prepare_inputs([source], tmp_path / "local", backup)
    assert (partial / "partial").read_bytes() == b"preserve"


def test_upload_failure_leaves_original_and_never_commits_partial_backup(tmp_path, monkeypatch):
    source = cache(tmp_path / "source")
    original = (source / "shard.pt").read_bytes()
    backup = tmp_path / "drive"

    def fail_copy(reader, writer, length):
        writer.write(reader.read(3))
        raise OSError("simulated Drive quota/network failure")

    monkeypatch.setattr(run_assets.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="simulated"):
        prepare_inputs([source], tmp_path / "local", backup)
    assert not (backup / "inputs").exists()
    assert (source / "shard.pt").read_bytes() == original
    assert len(list(backup.glob(".inputs-*.tmp"))) == 1


def test_changed_source_during_copy_is_detected(tmp_path, monkeypatch):
    source = cache(tmp_path / "source")
    backup = tmp_path / "drive"
    original_copy = run_assets.shutil.copyfileobj

    def corrupt_copy(reader, writer, length):
        original_copy(reader, writer, length)
        writer.write(b"unexpected extra bytes")

    monkeypatch.setattr(run_assets.shutil, "copyfileobj", corrupt_copy)
    with pytest.raises(ValueError, match="copy verification"):
        prepare_inputs([source], tmp_path / "local", backup)
    assert not (backup / "inputs").exists()


def test_invalid_existing_local_restore_is_not_silently_used(tmp_path):
    source = cache(tmp_path / "source")
    local = tmp_path / "local"
    backup = tmp_path / "drive"
    prepare_inputs([source], local, backup)
    shutil.rmtree(source)
    prepare_inputs([source], local, backup)
    (local / "inputs/cache-0000/shard.pt").write_bytes(b"corrupt local copy")
    with pytest.raises(ValueError, match="checksum"):
        prepare_inputs([source], local, backup)


@pytest.mark.parametrize("invalid", ["validation", "duplicate", "missing"])
def test_initial_input_validation(tmp_path, invalid):
    source = cache(tmp_path / "source", split="val_unseen" if invalid == "validation" else "train_fit")
    sources = [source, source] if invalid == "duplicate" else [source]
    if invalid == "missing":
        (source / "shard.pt").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_inputs(sources, tmp_path / "local", tmp_path / "drive")
    assert not (tmp_path / "drive/inputs").exists()


def test_committed_index_cannot_escape_backup_root(tmp_path):
    source = cache(tmp_path / "source")
    backup = tmp_path / "drive"
    prepare_inputs([source], tmp_path / "local", backup)
    index_path = backup / "inputs/index.json"
    value = json.loads(index_path.read_bytes())
    value["caches"][0]["files"][0]["path"] = "../../outside"
    index_path.write_text(json.dumps(value))
    (backup / "inputs/COMMITTED").write_text(hashlib.sha256(index_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="escapes"):
        prepare_inputs([source], tmp_path / "local", backup)


def create_file(root, relative, contents="fixture\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def test_source_snapshot_is_deterministic_and_excludes_data_credentials(tmp_path):
    root = tmp_path / "project"
    included = [
        "src/vln_improve/train.py", "scripts/run.py", "configs/r2r.json",
        "docs/guide.md", "tests/test_training.py", "pyproject.toml", "README.md",
        "setup_gpu.sh", "build_mattersim.sh", "download_assets.py", "uv.lock",
        "artifacts/assets-manifest.json", "artifacts/requirements-freeze.txt",
        "artifacts/mattersim-compat.patch",
    ]
    excluded = [
        "datasets/annotations.json", "outputs/eval.json", ".venv/runtime.py",
        ".git/config", "auth/token.json", "keys/private.pem", "configs/auth.json",
        "configs/credentials.json", "configs/client_secret_123.json",
        "artifacts/private.json", "src/vln_improve/__pycache__/train.pyc", "unlisted.py",
    ]
    for path in included + excluded:
        create_file(root, path)
    backup = tmp_path / "drive"
    first = snapshot_code(root, backup)
    second = snapshot_code(root, backup)
    assert first == second
    with tarfile.open(first["archive"], "r:gz") as archive:
        assert set(archive.getnames()) == set(included)
    assert first["archive_sha256"] == hashlib.sha256(Path(first["archive"]).read_bytes()).hexdigest()
    assert first["verification"] == "filesystem-readback-sha256"


def test_core_identity_changes_only_for_core_python_code(tmp_path):
    root = tmp_path / "project"
    create_file(root, "src/vln_improve/train.py")
    doc = create_file(root, "docs/guide.md")
    script = create_file(root, "scripts/run.sh")
    backup = tmp_path / "drive"
    first = snapshot_code(root, backup)
    doc.write_text("new docs")
    script.write_text("new shell script")
    second = snapshot_code(root, backup)
    assert first["code_sha256"] == second["code_sha256"]
    assert first["archive_sha256"] != second["archive_sha256"]
    create_file(root, "src/vln_improve/train.py", "new core code")
    assert snapshot_code(root, backup)["code_sha256"] != first["code_sha256"]


def test_source_snapshot_rejects_symlinks_and_corrupt_existing_archive(tmp_path):
    root = tmp_path / "project"
    source = create_file(root, "src/train.py")
    backup = tmp_path / "drive"
    result = snapshot_code(root, backup)
    Path(result["archive"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        snapshot_code(root, backup)
    source.unlink()
    outside = create_file(tmp_path, "outside.py")
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        snapshot_code(root, tmp_path / "other-drive")


def test_source_snapshot_rejects_artifacts_directory_symlink(tmp_path):
    root = tmp_path / "project"
    create_file(root, "src/train.py")
    outside = tmp_path / "outside"
    create_file(outside, "assets-manifest.json")
    (root / "artifacts").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        snapshot_code(root, tmp_path / "drive")
