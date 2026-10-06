#!/usr/bin/env python3
"""Extract frozen STOP representations from sealed train-only D1 states."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch

from replay_diagnostics import load_model, validate_episode, validate_runtime
from vln_improve.counterfactual import clone_navigation, navigation_hash
from vln_improve.diagnostics import SCHEMA, load_episode
from vln_improve.endpoint_probe import build_endpoint_features, load_endpoint_cache, validate_endpoint_episode
from vln_improve.pipeline import atomic_json, validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256


COMMON_KEYS = ("base_checkpoint_sha256", "feature_sha256", "annotation_sha256",
               "connectivity_sha256", "model", "upstream_lock", "partition_seed",
               "dev_fraction", "torch_version")
CODE_FILES = ("scripts/cache_endpoint_features.py", "scripts/replay_diagnostics.py",
              "src/vln_improve/endpoint_probe.py", "src/vln_improve/counterfactual.py",
              "src/vln_improve/diagnostics.py", "src/vln_improve/protocol.py")
SEAL_SCHEMA = "duet_endpoint_episode_commit_v2"


def _ordinary_file(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected ordinary cache file: {path}")


def _read_json(path):
    _ordinary_file(path)
    return json.loads(path.read_bytes(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"invalid JSON constant: {value}")))


def _seal_path(root, name):
    return root / (name + ".seal.json")


def _seal(payload, item):
    return {"schema": SEAL_SCHEMA, "identity_sha256": payload["identity_sha256"],
            "input_manifest_sha256": payload["input_manifest_sha256"], "file": item}


def _read_seal(root, name, identity_sha):
    path = _seal_path(root, name)
    if not path.exists() and not path.is_symlink():
        return None
    value = _read_json(path)
    if (not isinstance(value, dict) or set(value) != {"schema", "identity_sha256", "input_manifest_sha256", "file"}
            or value["schema"] != SEAL_SCHEMA or value["identity_sha256"] != identity_sha
            or not isinstance(value["input_manifest_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["input_manifest_sha256"]) is None):
        raise ValueError("invalid endpoint episode seal")
    item = value["file"]
    if (not isinstance(item, dict) or set(item) != {"name", "sha256", "association", "num_states", "positives"}
            or item["name"] != name or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
        raise ValueError("invalid endpoint sealed file identity")
    return value


def _checked_payload(path, item, identity_sha):
    _ordinary_file(path)
    if file_sha256(path) != item["sha256"]:
        raise ValueError(f"committed endpoint episode SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_endpoint_episode(payload, item, identity_sha)
    return payload


def _write_seal_if_missing(root, name, seal):
    path = _seal_path(root, name)
    previous = _read_seal(root, name, seal["identity_sha256"])
    if previous is not None and previous != seal:
        raise ValueError("endpoint episode seals disagree")
    if previous is None:
        atomic_json(path, seal)
    if _read_json(path) != seal:
        raise ValueError("endpoint seal read-back mismatch")


def prepare_cache_pair(local, backup, identity, expected_names):
    """Validate every old commit before restoring only missing cache files.

    A complete manifest is immutable. Partial caches have no collection commit;
    each reusable episode must instead carry its own content-bound seal.
    """
    roots = (Path(local), Path(backup))
    resolved = [root.resolve() for root in roots]
    if resolved[0].is_relative_to(resolved[1]) or resolved[1].is_relative_to(resolved[0]):
        raise ValueError("cache copies must use separate non-nested directories")
    identity_sha = object_sha256(identity)
    expected_names = set(expected_names)
    for name in expected_names:
        if Path(name).name != name or not name.startswith("episode-") or not name.endswith(".pt"):
            raise ValueError("invalid expected cache episode name")
    committed = []
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "IDENTITY.json"
        if marker.exists() or marker.is_symlink():
            if _read_json(marker) != identity:
                raise ValueError("cache identity changed; choose a new directory")
        elif (list(root.glob("episode-*")) or (root / "manifest.json").exists()
              or (root / "COMMITTED.json").exists()):
            raise ValueError("orphan cache files lack identity")
        for path in root.glob("episode-*"):
            name = path.name.removesuffix(".seal.json")
            if name not in expected_names or (path.name != name and path.name != name + ".seal.json"):
                raise ValueError("unexpected endpoint cache episode inventory")
        commit_path, manifest_path = root / "COMMITTED.json", root / "manifest.json"
        if commit_path.exists() or commit_path.is_symlink():
            commit = _read_json(commit_path)
            manifest = _read_json(manifest_path)
            if commit != {"manifest_sha256": file_sha256(manifest_path)}:
                raise ValueError("endpoint committed manifest SHA-256 mismatch")
            if (manifest.get("identity") != identity or manifest.get("identity_sha256") != identity_sha
                    or not isinstance(manifest.get("files"), list)
                    or len(manifest["files"]) != len(expected_names)
                    or {item.get("name") for item in manifest["files"]} != expected_names):
                raise ValueError("endpoint committed manifest identity/inventory mismatch")
            committed.append((root, manifest, commit))
    if not committed:
        for root in roots:
            if not (root / "IDENTITY.json").exists():
                atomic_json(root / "IDENTITY.json", identity)
        return None
    source, manifest, commit = committed[0]
    if any(other != manifest or marker != commit for _, other, marker in committed[1:]):
        raise ValueError("local/cloud committed manifests disagree")
    repairs, seals = [], []
    # Complete preflight before copying anything: no existing file is re-signed.
    for item in manifest["files"]:
        name = item["name"]
        valid = []
        present_seals = [_read_seal(root, name, identity_sha) for root in roots]
        for root in roots:
            path = root / name
            if path.exists() or path.is_symlink():
                payload = _checked_payload(path, item, identity_sha)
                valid.append((root, payload))
        if not valid:
            raise ValueError(f"committed endpoint file is missing from both copies: {name}")
        seal = _seal(valid[0][1], item)
        if any(previous is not None and previous != seal for previous in present_seals):
            raise ValueError("endpoint seal disagrees with committed manifest")
        for root in roots:
            if not (root / name).exists():
                repairs.append((valid[0][0] / name, root / name))
            seals.append((root, name, seal))
    for root in roots:
        path = root / "manifest.json"
        if (path.exists() or path.is_symlink()) and _read_json(path) != manifest:
            raise ValueError("uncommitted manifest disagrees with valid committed copy")
    for root in roots:
        if not (root / "IDENTITY.json").exists():
            atomic_json(root / "IDENTITY.json", identity)
    for src, dst in repairs:
        copy_verified(src, dst)
    for root, name, seal in seals:
        _write_seal_if_missing(root, name, seal)
    for root in roots:
        if root != source:
            copy_verified(source / "manifest.json", root / "manifest.json")
            copy_verified(source / "COMMITTED.json", root / "COMMITTED.json")
        load_endpoint_cache(root, expected_split=identity["split"])
    return manifest


def reuse_episode(local, backup, name, identity_sha, association, input_sha):
    """Return a verified sealed episode, repairing missing copies only."""
    roots = (Path(local), Path(backup))
    found, existing_seals = [], []
    for root in roots:
        path = root / name
        seal = _read_seal(root, name, identity_sha)
        if seal is not None:
            if (seal["input_manifest_sha256"] != input_sha or seal["file"]["association"] != association):
                raise ValueError("sealed endpoint episode/source association changed")
            existing_seals.append(seal)
        if path.exists() or path.is_symlink():
            if seal is None:
                raise ValueError(f"uncommitted endpoint file has no seal; use a new cache directory: {path}")
            payload = _checked_payload(path, seal["file"], identity_sha)
            if payload["input_manifest_sha256"] != input_sha:
                raise ValueError("sealed endpoint source manifest changed")
            found.append((root, payload, seal))
    if not existing_seals:
        return None
    if any(seal != existing_seals[0] for seal in existing_seals[1:]):
        raise ValueError("local/cloud endpoint episode seals disagree")
    if not found:
        raise ValueError("sealed endpoint data missing from both copies")
    source, payload, seal = found[0]
    for root in roots:
        if not (root / name).exists():
            copy_verified(source / name, root / name)
        _write_seal_if_missing(root, name, seal)
    return payload


def commit_episode(local, backup, name, payload):
    """Commit a freshly computed episode; never adopt a pre-existing pt file."""
    roots = (Path(local), Path(backup))
    if any((root / name).exists() or (root / name).is_symlink()
           or _seal_path(root, name).exists() or _seal_path(root, name).is_symlink() for root in roots):
        raise ValueError("refusing to overwrite an endpoint episode or seal")
    item = {"name": name, "sha256": "0" * 64, "association": payload["association"],
            "num_states": len(payload["steps"]), "positives": int(payload["labels"].sum())}
    if name != payload["association"]["episode_id"] + ".pt":
        raise ValueError("endpoint filename/association mismatch")
    validate_endpoint_episode(payload, item, payload["identity_sha256"])
    atomic_tensor(roots[0] / name, payload)
    item["sha256"] = file_sha256(roots[0] / name)
    seal = _seal(payload, item)
    _write_seal_if_missing(roots[0], name, seal)
    copy_verified(roots[0] / name, roots[1] / name)
    _write_seal_if_missing(roots[1], name, seal)
    return reuse_episode(*roots, name, payload["identity_sha256"], payload["association"],
                         payload["input_manifest_sha256"])


def verify_logits(state, output):
    """Require exact recorded FP32 values, masks, and argmax on all branches."""
    for actual_name, recorded_name in (("fused_logits", "base_logits"),
                                       ("global_logits", "base_global_logits"),
                                       ("local_logits", "base_local_logits")):
        actual = output[actual_name].detach().cpu().float()
        expected = state[recorded_name].cpu()
        if actual.shape != (1, len(expected)) or not torch.equal(actual[0], expected):
            raise ValueError(f"endpoint replay changed {actual_name}")


def atomic_tensor(path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def copy_verified(source, destination):
    source, destination = Path(source), Path(destination)
    _ordinary_file(source)
    digest = file_sha256(source)
    if destination.exists() or destination.is_symlink():
        _ordinary_file(destination)
        if file_sha256(destination) != digest:
            raise ValueError(f"existing cache bytes disagree: {destination}")
        return
    temporary = destination.with_name("." + destination.name + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    if file_sha256(temporary) != digest:
        raise ValueError("cache backup read-back mismatch")
    temporary.replace(destination)
    if file_sha256(destination) != digest:
        raise ValueError("cache backup final read-back mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--split", choices=("train_fit", "train_dev"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    validate_backup_root(args.backup)
    args.output.mkdir(parents=True, exist_ok=True)
    args.backup.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".collection.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    collected = json.loads((args.collection / "COLLECTION.json").read_text())
    source = collected["identity"]
    if (collected["schema"] != SCHEMA or object_sha256(source) != collected["identity_sha256"]
            or source["split"] != args.split):
        raise ValueError("collection identity/split mismatch")
    expected_usage = "training_diagnostics" if args.split == "train_fit" else "analysis_only"
    if source["usage"] != expected_usage:
        raise ValueError("source usage mismatch")
    expected = {"episode-" + object_sha256([r["scan"], r["instr_id"]]) for r in source["selection"]}
    episodes = sorted(args.collection.glob("episode-*"))
    if not episodes or len(expected) != len(source["selection"]) or {e.name for e in episodes} != expected:
        raise ValueError("source collection episode inventory mismatch")
    model, runtime = load_model(args.config)
    validate_runtime(source, runtime)
    identity = {"schema": "duet_endpoint_features_identity_v1", "split": args.split,
                "usage": "training" if args.split == "train_fit" else "analysis_only",
                "collection_identity_sha256": collected["identity_sha256"], "runtime": runtime,
                "common_provenance": {k: source[k] for k in COMMON_KEYS},
                "feature_schema": "concat_global_local_stop_crossmodal_v1", "feature_dim": 1536,
                "implementation": {n: file_sha256(ROOT / n) for n in CODE_FILES}}
    identity_sha = object_sha256(identity)
    validate_backup_root(args.backup)
    completed_manifest = prepare_cache_pair(args.output, args.backup, identity,
                                            {name + ".pt" for name in expected})
    files, counts = [], {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for episode in episodes:
        inputs, labels, _ = load_episode(episode, expected_identity_sha256=collected["identity_sha256"])
        association, states, oracle, _ = validate_episode(inputs, labels, args.split)
        name = episode.name + ".pt"
        target = args.output / name
        input_sha = file_sha256(episode / "manifest.json")
        validate_backup_root(args.backup)
        payload = reuse_episode(args.output, args.backup, name, identity_sha, association, input_sha)
        if payload is None:
            features, distances, probabilities = [], [], []
            for step, state in states.items():
                frozen = clone_navigation(state["nav_inputs"], "cuda")
                before = navigation_hash(frozen)
                with torch.inference_mode():
                    output = model("navigation", frozen)
                    verify_logits(state, output)
                    feature = build_endpoint_features(frozen, output).cpu()
                if navigation_hash(frozen) != before:
                    raise ValueError("endpoint replay mutated inputs")
                stop = oracle[step]["stop"]
                distance = stop["distance_to_goal"]
                if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                        or not math.isfinite(distance) or distance < 0
                        or type(stop["within_success_radius"]) is not bool
                        or stop["within_success_radius"] != (distance < 3)):
                    raise ValueError("invalid training-only STOP labels")
                features.append(feature[0])
                distances.append(distance)
                probabilities.append(float(state["base_logits"].softmax(0)[0]))
            distance_tensor = torch.tensor(distances, dtype=torch.float64)
            payload = {"schema": "duet_endpoint_features_v1", "identity_sha256": identity_sha,
                       "association": association, "input_manifest_sha256": input_sha,
                       "features": torch.stack(features).float(), "steps": torch.tensor(list(states), dtype=torch.int64),
                       "viewpoints": [s["current_viewpoint"] for s in states.values()],
                       "distance_to_goal": distance_tensor, "labels": (distance_tensor < 3).float(),
                       "base_stop_probability": torch.tensor(probabilities, dtype=torch.float32)}
            validate_backup_root(args.backup)
            payload = commit_episode(args.output, args.backup, name, payload)
        n, positives = len(payload["steps"]), int(payload["labels"].sum())
        files.append({"name": name, "sha256": file_sha256(target), "association": association,
                      "num_states": n, "positives": positives})
        scan = counts.setdefault(association["scan_id"], {"episodes": 0, "states": 0, "positives": 0})
        scan["episodes"] += 1; scan["states"] += n; scan["positives"] += positives
        print(json.dumps({"event": "endpoint_episode_backed_up", "episodes": len(files),
                          "states": sum(f["num_states"] for f in files)}), flush=True)
    if completed_manifest is not None:
        if files != completed_manifest["files"]:
            raise ValueError("resumed complete endpoint cache file inventory changed")
        print(json.dumps(completed_manifest["summary"], indent=2))
        return
    manifest = {"schema": "duet_endpoint_features_v1", "feature_dim": 1536, "split": args.split,
                "usage": identity["usage"], "identity": identity, "identity_sha256": identity_sha,
                "files": files, "summary": {"episodes": len(files), "states": sum(f["num_states"] for f in files),
                "positives": sum(f["positives"] for f in files), "per_scan": counts,
                "all_state_three_branch_exact_logit_parity": True,
                "resources": {"wall_seconds": time.monotonic() - started,
                              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()}}}
    # All episodes are sealed before the complete manifest is committed. A
    # resumed complete cache keeps these exact bytes, including resource timing.
    for root in (args.output, args.backup):
        validate_backup_root(args.backup)
        atomic_json(root / "manifest.json", manifest)
        atomic_json(root / "COMMITTED.json", {"manifest_sha256": file_sha256(root / "manifest.json")})
        load_endpoint_cache(root, expected_split=args.split)
    assert file_sha256(args.output / "manifest.json") == file_sha256(args.backup / "manifest.json")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
