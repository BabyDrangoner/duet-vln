#!/usr/bin/env python3
"""Train one frozen E2 comparison arm and verify every durable recovery point."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Deterministic CUDA GEMM configuration must exist before CUDA initialization.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from vln_improve.intervention_training import ARMS, train_intervention
from vln_improve.pipeline import atomic_json, validate_backup_root
from vln_improve.protocol import file_sha256


def load_training_caches(fit_paths, dev_path):
    from vln_improve.intervention_runtime import load_records
    fit, dev, identities = [], None, {"fit": [], "dev": None}
    for split, paths in (("train_fit", fit_paths), ("train_dev", [dev_path])):
        for path in paths:
            path = Path(path)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != "e2_intervention_cache_v1" or manifest["collection"]["split"] != split:
                raise ValueError("cache schema/split differs from training request")
            records = list(load_records(path))
            identity = {"manifest_sha256": file_sha256(manifest_path), "records": len(records),
                "provenance": manifest["provenance"], "collection": manifest["collection"]}
            if split == "train_fit":
                fit.extend(records); identities["fit"].append(identity)
            else:
                dev = records; identities["dev"] = identity
    return fit, dev, identities


def validate_cache_protocol(train, dev, identity, specification, digest, *, engineering_smoke=False):
    manifests = [*identity["fit"], identity["dev"]]
    common = manifests[0]["provenance"]
    if any(m["provenance"] != common for m in manifests):
        raise ValueError("cache provenance differs across fit conditions/development")
    if (common.get("experiment_sha256") != specification.get("collection_config_sha256", digest)
            or common.get("base_checkpoint_sha256") != specification["baseline"]["checkpoint_sha256"]):
        raise ValueError("cache experiment/base checkpoint differs from fixed protocol")
    if engineering_smoke:
        if any(m["collection"].get("scope") != "engineering_smoke" for m in manifests):
            raise ValueError("engineering smoke requires explicitly scoped caches")
        identity["scope"] = "engineering_smoke"
        return
    if any(m["collection"].get("scope") != "research" for m in manifests):
        raise ValueError("formal training refuses engineering caches")
    collection = specification["collection"]
    if len(identity["fit"]) != len(collection["fit_conditions"]):
        raise ValueError("formal training fit condition inventory differs")
    panels = {condition: [r for r in train if r["condition"] == condition] for condition in collection["fit_conditions"]}
    if (len(train) != specification["fit_episodes"]
            or any(len(rows) != collection["fit_instruction_count"] for rows in panels.values())
            or any(len({r["scan_id"] for r in rows}) != collection["fit_expected_scenes"] for rows in panels.values())
            or len({frozenset(r["instr_id"] for r in rows) for rows in panels.values()}) != 1
            or len(dev) != collection["dev_instruction_count"]
            or len({r["scan_id"] for r in dev}) != collection["dev_expected_scenes"]
            or any(r["condition"] != "natural" for r in dev)
            or specification["epochs"] * ((len(train) + specification["batch_size"] - 1) // specification["batch_size"])
                != specification["max_updates_per_arm"]):
        raise ValueError("formal cache membership/count/scenes/update budget differs")
    identity["scope"] = "research"


def write_reports(result, local_run, backup_run, verify_backup):
    for path in (Path(local_run), Path(backup_run)):
        path.mkdir(parents=True, exist_ok=True)
    verify_backup()
    for name, value in (("training-summary.json", result),
                        ("dev-history.json", {"arm": result["arm"], "history": result.get("dev_history", [])})):
        atomic_json(Path(local_run) / name, value)
        atomic_json(Path(backup_run) / name, value)
        if file_sha256(Path(local_run) / name) != file_sha256(Path(backup_run) / name):
            raise ValueError("training report backup read-back mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fit-cache", type=Path, action="append", required=True)
    for name in ("dev-cache", "local-run", "backup-run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--checkpoint-every-steps", type=int, default=50)
    parser.add_argument("--checkpoint-every-seconds", type=float)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    specification = json.loads(args.config.read_text())
    keys = ("seed", "epochs", "batch_size", "hidden_dim", "lr", "weight_decay", "monitor_every_epochs", "risk_weight", "inference_batch_size")
    if any(key not in specification for key in keys):
        raise ValueError("fixed experiment config misses required training fields")
    config = {key: specification[key] for key in keys}
    config.update(arm=args.arm, experiment_sha256=file_sha256(args.config))
    if "arms" in specification and args.arm not in specification["arms"]:
        raise ValueError("training arm is absent from fixed protocol")
    if "arm" in specification and args.arm != specification["arm"]:
        raise ValueError("training arm differs from frozen per-arm specification")
    paths = [p.resolve() for p in (*args.fit_cache, args.dev_cache, args.local_run, args.backup_run)]
    for index, path in enumerate(paths):
        if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths[index + 1:]):
            raise ValueError("caches and run stores must be separate and nonnested")
    train, dev, identity = load_training_caches(args.fit_cache, args.dev_cache)
    validate_cache_protocol(train, dev, identity, specification, file_sha256(args.config), engineering_smoke=args.engineering_smoke)
    persistence = specification.get("persistence", {})
    checkpoint_seconds = args.checkpoint_every_seconds if args.checkpoint_every_seconds is not None else persistence.get("checkpoint_every_seconds", 300.)
    deadline_seconds = args.deadline_seconds if args.deadline_seconds is not None else persistence.get("max_process_seconds", 36000.)
    checker = lambda: validate_backup_root(args.backup_run, allow_local=args.allow_local_backup_for_tests)
    checker()
    stop_requested = False
    started = time.monotonic()
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = train_intervention(train, dev, config, data_identity=identity, local_dir=args.local_run,
            backup_dir=args.backup_run, device=args.device,
            checkpoint_every_steps=args.checkpoint_every_steps, checkpoint_every_seconds=checkpoint_seconds,
            keep_local=persistence.get("keep_local", 2), keep_backup=persistence.get("keep_backup", 5),
            verify_backup=checker, should_stop=lambda: stop_requested or time.monotonic() - started >= deadline_seconds,
            stop_after_updates=args.stop_after_updates)
        write_reports(result, args.local_run, args.backup_run, checker)
        print(json.dumps({k: v for k, v in result.items() if k not in {"training_history", "dev_history"}}, indent=2))
        return result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
