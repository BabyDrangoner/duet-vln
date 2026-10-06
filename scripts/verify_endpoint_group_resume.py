#!/usr/bin/env python3
"""Acceptance-only group-training restart check on eight/four verified groups."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vln_improve.checkpoint_store import CheckpointStore, ID_RE
from vln_improve.endpoint_group_training import (
    _support, load_endpoint_group_cache, train_endpoint_groups, validate_endpoint_group_splits,
)
from vln_improve.endpoint_pairs import content_hash
from vln_improve.pipeline import atomic_json, validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256

SCHEMA = "duet_endpoint_group_resume_acceptance_v1"
PROFILE = {"scope": "acceptance_only", "arm": "M", "seed": 0,
           "fit_groups": 8, "dev_groups": 4, "epochs": 3, "batch_groups": 2,
           "monitor_every": 1, "interrupt_after_updates": 2,
           "checkpoint_every_steps": 1, "keep_local": 2, "keep_backup": 3}


def acceptance_subset(cache, count):
    """Keep immutable full-source validation and explicitly identify a prefix subset."""
    if type(count) is not int or count < 1 or len(cache.groups) < count:
        raise ValueError("acceptance source has too few groups")
    groups = cache.groups[:count]
    source = copy.deepcopy(cache.source_identity)
    source.update(schema="duet_endpoint_group_acceptance_subset_v1", scope="acceptance_only",
        subset_of={"data_sha256": cache.data_sha256, "groups": len(cache.groups),
                   "source_identity_sha256": object_sha256(cache.source_identity)},
        selection_rule="first N groups in the complete sealed cache order",
        source_indices=list(range(count)),
        pair_ids=[g["pair"]["selection_hash"] for g in groups],
        ordered_pair_sha256=[object_sha256(g["pair"]) for g in groups])
    return replace(cache, groups=groups, data_sha256=object_sha256(source),
                   source_identity=source, support=_support(groups))


def assert_exact(actual, expected, path="state"):
    """Compare tensor bytes, types and all nested optimizer/RNG/history values."""
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor)
                or actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.equal(actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                   expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8))):
            raise ValueError(f"acceptance tensor mismatch at {path}")
    elif isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"acceptance dictionary keys differ at {path}")
        for key in actual: assert_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"acceptance sequence mismatch at {path}")
        for index, (a, e) in enumerate(zip(actual, expected)):
            assert_exact(a, e, f"{path}[{index}]")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"acceptance value mismatch at {path}")


def state_content_hash(value):
    # AdamW state uses integer parameter keys; preserve their type rather than
    # relying on JSON's implicit conversion of dictionary keys to strings.
    def canonical(item):
        if isinstance(item, dict):
            return ["dict", [[type(k).__name__, k, canonical(item[k])]
                              for k in sorted(item, key=lambda k: (type(k).__name__, repr(k)))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [canonical(v) for v in item]]
        return item
    return content_hash(canonical(value))


def _ordinary_path(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("acceptance paths must not contain symbolic links")
    return path.resolve()


def _distinct_paths(paths):
    paths = [_ordinary_path(p) for p in paths]
    for i, a in enumerate(paths):
        if any(a.is_relative_to(b) or b.is_relative_to(a) for b in paths[i+1:]):
            raise ValueError("acceptance caches and roots must be distinct and non-nested")
    return paths


def _fresh_roots(local, backup, marker, checker):
    # Check both roots before creating either: an existing result is evidence,
    # even if it came from an earlier invocation of this acceptance script.
    for root in (local, backup):
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError("acceptance requires new or empty dedicated roots; refusing overwrite")
    checker()
    for root in (local, backup):
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "ACCEPTANCE.json", marker)
    if file_sha256(local / "ACCEPTANCE.json") != file_sha256(backup / "ACCEPTANCE.json"):
        raise ValueError("acceptance identity backup read-back differs")


def _store_inventory(root):
    """Only a clean CheckpointStore tree may be removed during this test."""
    allowed = {"STORE.json", "snapshots", ".writer.lock", "refs.json", "latest.json", "best.json"}
    if not root.is_dir() or root.is_symlink() or set(p.name for p in root.iterdir()) - allowed:
        raise ValueError("unexpected files in acceptance interrupted store")
    files = {}
    for entry in root.rglob("*"):
        if entry.is_symlink(): raise ValueError("symbolic link in acceptance interrupted store")
        relative = entry.relative_to(root)
        if len(relative.parts) == 1:
            if entry.name == "snapshots":
                if not entry.is_dir(): raise ValueError("invalid snapshots directory")
            elif not entry.is_file(): raise ValueError("unexpected acceptance store entry")
        elif len(relative.parts) == 2:
            if relative.parts[0] != "snapshots" or not entry.is_dir() or ID_RE.fullmatch(entry.name) is None:
                raise ValueError("unexpected acceptance snapshot directory")
        elif (len(relative.parts) != 3 or relative.parts[0] != "snapshots"
              or entry.name not in {"manifest.json", "COMMITTED", "state.pt", "head.pt"} or not entry.is_file()):
            raise ValueError("unexpected file below acceptance snapshot")
        if entry.is_file(): files[str(relative)] = file_sha256(entry)
    return files


def _delete_owned_interrupted(local_root, marker, inventory):
    directory = local_root / "interrupted"
    if json.loads((local_root / "ACCEPTANCE.json").read_text()) != marker:
        raise ValueError("acceptance ownership marker changed before local deletion")
    if _ordinary_path(directory).parent != local_root or _store_inventory(directory) != inventory:
        raise ValueError("acceptance store changed before local deletion")
    with (directory / ".writer.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # This exact directory was created by this invocation; siblings and
        # source caches are never removed, even if another check later fails.
        shutil.rmtree(directory)
    directory.mkdir()


def _backup_payload(store, checkpoint_id):
    directory = store.backup_dir / "snapshots" / checkpoint_id
    manifest = store._verify(directory, checkpoint_id)
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    head = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    return state, head, manifest


def run_acceptance(full_train, full_dev, local_root, backup_root, *, device="cuda", verify_backup=None):
    """Execute fresh acceptance runs; source caches are read-only throughout."""
    validate_endpoint_group_splits(full_train, full_dev)
    roots = _distinct_paths([full_train.pair_cache.root, full_train.control_cache.root,
                             full_dev.pair_cache.root, full_dev.control_cache.root,
                             local_root, backup_root])
    local_root, backup_root = roots[-2:]
    checker = verify_backup or (lambda: validate_backup_root(backup_root))
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("acceptance requested device is unavailable")
    train, dev = acceptance_subset(full_train, 8), acceptance_subset(full_dev, 4)
    marker = {"schema": SCHEMA, "profile": PROFILE, "not_navigation_result": True,
              "source_full": {"train": full_train.data_sha256, "dev": full_dev.data_sha256},
              "subset_identity": {"train": train.source_identity, "dev": dev.source_identity},
              "subset_data_sha256": {"train": train.data_sha256, "dev": dev.data_sha256},
              "script_sha256": file_sha256(Path(__file__)), "device": device}
    _fresh_roots(local_root, backup_root, marker, checker)
    options = {k: PROFILE[k] for k in ("arm", "seed", "epochs", "batch_groups", "monitor_every",
                                      "checkpoint_every_steps", "keep_local", "keep_backup")}
    options.update(device=device, experiment_sha256=object_sha256(marker), verify_backup=checker)
    started = time.monotonic()
    continuous = train_endpoint_groups(train, dev, local_root/"continuous", backup_root/"continuous", **options)
    interrupted = train_endpoint_groups(train, dev, local_root/"interrupted", backup_root/"interrupted",
                                        max_steps=2, **options)
    if (continuous["status"] != "complete" or continuous["global_step"] != 12
            or interrupted["status"] != "interrupted" or interrupted["global_step"] != 2):
        raise ValueError("acceptance runs did not reach their registered boundaries")
    paused_store = CheckpointStore(local_root/"interrupted", backup_root/"interrupted")
    paused_state, paused_head, paused_manifest = paused_store.restore("latest")
    if paused_state.get("global_step") != 2 or paused_head.get("global_step") != 2:
        raise ValueError("acceptance paused store fell back from the required two-update boundary")
    checker()
    cloud_state, cloud_head, cloud_manifest = _backup_payload(paused_store, paused_manifest["checkpoint_id"])
    assert_exact(paused_state, cloud_state); assert_exact(paused_head, cloud_head)
    assert_exact({k:v for k,v in paused_manifest.items() if k != "local_path"}, cloud_manifest)
    inventory = _store_inventory(local_root/"interrupted")
    _delete_owned_interrupted(local_root, marker, inventory)
    if any((local_root/"interrupted").iterdir()):
        raise ValueError("acceptance restore destination is not empty")
    resumed = train_endpoint_groups(train, dev, local_root/"interrupted", backup_root/"interrupted", **options)
    if resumed["status"] != "complete" or resumed["resumed"] is not True or resumed["global_step"] != 12:
        raise ValueError("acceptance failed to resume from cloud to the fixed final step")
    stores = [CheckpointStore(local_root/name, backup_root/name) for name in ("continuous", "interrupted")]
    states, heads, final_manifests = [], [], []
    for store in stores:
        state, head, manifest = store.restore("latest")
        if (state.get("epoch") != 3 or state.get("global_step") != 12 or state.get("pending_dev") is not False
                or head.get("epoch") != 3 or head.get("global_step") != 12 or head.get("pending_dev") is not False):
            raise ValueError("acceptance final store fell back from the required completed boundary")
        checker()
        cloud_state, cloud_head, cloud_manifest = _backup_payload(store, manifest["checkpoint_id"])
        assert_exact(state, cloud_state); assert_exact(head, cloud_head)
        assert_exact({k:v for k,v in manifest.items() if k != "local_path"}, cloud_manifest)
        states.append(state); heads.append(head); final_manifests.append(manifest)
    assert_exact(states[0], states[1]); assert_exact(heads[0], heads[1])
    report = {"schema": SCHEMA, "status": "passed", "scope": "acceptance_only", "not_navigation_result": True,
        "identity_sha256": object_sha256(marker), "identity": marker,
        "checks": {"paused_drive_payload_exact": True, "local_interrupted_tree_deleted": True,
                   "restored_from_empty_local_directory": True, "final_full_state_exact": True,
                   "final_head_exact": True, "final_drive_payload_exact": True},
        "final_state_content_sha256": state_content_hash(states[0]), "final_head_content_sha256": content_hash(heads[0]),
        "final_checkpoint_manifests": final_manifests, "paused_checkpoint_id": paused_manifest["checkpoint_id"],
        "source_counts": {"full_train": len(full_train.groups), "full_dev": len(full_dev.groups),
                          "subset_train": 8, "subset_dev": 4},
        "subset_support": {"train": train.support, "dev": dev.support},
        "runs": {"continuous": continuous, "paused": interrupted, "resumed": resumed},
        "device": device, "torch_version": str(torch.__version__),
        "gpu_name": torch.cuda.get_device_name() if device == "cuda" else None,
        "wall_seconds": time.monotonic()-started,
        "storage_evidence": "mounted path SHA read-back; not an independent server receipt"}
    report["content_sha256"] = object_sha256(report)
    atomic_json(local_root/"resume-acceptance.json", report)
    checker(); atomic_json(backup_root/"resume-acceptance.json", report)
    digest = file_sha256(local_root/"resume-acceptance.json")
    if digest != file_sha256(backup_root/"resume-acceptance.json"):
        raise ValueError("acceptance report backup read-back differs")
    for root in (local_root, backup_root):
        atomic_json(root/"COMMITTED.json", {"report_sha256": digest, "identity_sha256": object_sha256(marker)})
    if file_sha256(local_root/"COMMITTED.json") != file_sha256(backup_root/"COMMITTED.json"):
        raise ValueError("acceptance commit marker backup read-back differs")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-pairs", "train-controls", "dev-pairs", "dev-controls", "local-root", "backup-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-train-data-sha256", required=True)
    parser.add_argument("--expected-dev-data-sha256", required=True)
    parser.add_argument("--experiment", type=Path, default=ROOT/"configs/endpoint_group_M.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.allow_local_backup_for_tests and args.device != "cpu":
        raise ValueError("test-only mount bypass requires CPU")
    for value in (args.expected_train_data_sha256, args.expected_dev_data_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None: raise ValueError("invalid full data SHA")
    train = load_endpoint_group_cache(args.train_pairs, args.train_controls, "train_fit",
                                     expected_data_sha256=args.expected_train_data_sha256)
    dev = load_endpoint_group_cache(args.dev_pairs, args.dev_controls, "train_dev",
                                   expected_data_sha256=args.expected_dev_data_sha256)
    from train_endpoint_groups import validate_experiment
    if validate_experiment(json.loads(args.experiment.read_text()), train=train, dev=dev) != "M":
        raise ValueError("acceptance full-pool provenance must use the registered M source configuration")
    result = run_acceptance(train, dev, args.local_root, args.backup_root, device=args.device,
        verify_backup=lambda: validate_backup_root(args.backup_root, allow_local=args.allow_local_backup_for_tests))
    print(json.dumps({k: result[k] for k in ("status", "scope", "not_navigation_result", "checks",
                                           "source_counts", "final_state_content_sha256", "wall_seconds")}, indent=2))
    return result


if __name__ == "__main__": main()
