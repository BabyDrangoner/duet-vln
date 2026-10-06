#!/usr/bin/env python3
"""Train one registered endpoint arm/seed on the complete fixed source pool."""

import argparse
import json
from pathlib import Path
import re
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vln_improve.endpoint_group_training import (
    BEST_PURPOSE, FINAL_PURPOSE, final_dev_report, load_endpoint_group_cache, train_endpoint_groups,
    validate_endpoint_group_splits,
)
from vln_improve.endpoint_probe import FEATURE_SCHEMA
from vln_improve.pipeline import atomic_json, validate_backup_root
from vln_improve.protocol import file_sha256


def write_training_reports(result, local_run, backup_run, verify_backup):
    """Publish small verified reports after the durable checkpoint has committed."""
    result = dict(result)
    local_run, backup_run = Path(local_run), Path(backup_run)
    if result.get("status") == "complete":
        report = final_dev_report(result)
        atomic_json(local_run / "dev-final.json", report)
        verify_backup()
        atomic_json(backup_run / "dev-final.json", report)
        digest = file_sha256(local_run / "dev-final.json")
        if digest != file_sha256(backup_run / "dev-final.json"):
            raise ValueError("final dev group report backup SHA read-back mismatch")
        result["final_dev_report"] = {"relative_path": "dev-final.json", "sha256": digest,
                                      "content_sha256": report["content_sha256"], "groups": len(report["groups"])}
    atomic_json(local_run / "training-summary.json", result)
    verify_backup()
    atomic_json(backup_run / "training-summary.json", result)
    if file_sha256(local_run / "training-summary.json") != file_sha256(backup_run / "training-summary.json"):
        raise ValueError("group training summary backup SHA read-back mismatch")
    return result


def validate_experiment(spec, *, train=None, dev=None):
    arm = spec.get("arm")
    if arm not in {"C1", "C2", "C3", "M"}:
        raise ValueError("unknown registered endpoint training arm")
    expected = {"schema_version": 1, "experiment_id": f"E1-common-pool-endpoint-{arm}",
        "specified_before_training": True, "source_selection_seed": 20261003, "model_collection_seed": 0,
        "train_pairs": 512, "dev_pairs": 128, "feature_schema": FEATURE_SCHEMA,
        "architecture": {"input_dim": 1536, "hidden_dim": 128, "activation": "ReLU", "output_dim": 1},
        "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001},
        "epochs": 20, "batch_groups": 8, "monitor_every_epochs": 5, "allowed_training_seeds": [0, 1, 2],
        "natural_weight": .5, "augmentation_weight": .5, "pair_weight": .1 if arm == "M" else 0.,
        "primary_checkpoint": FINAL_PURPOSE, "engineering_best_monitor": BEST_PURPOSE,
        "fit_split": "train_fit", "monitor_split": "train_dev", "head_input": "frozen1536_STOP_tokens_only"}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("registered full-pool endpoint training protocol differs")
    sha = spec.get("controls_report_sha256")
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("missing registered common-pool controls report digest")
    if (train is None) != (dev is None):
        raise ValueError("both complete group splits are required")
    if train is not None:
        validate_endpoint_group_splits(train, dev)
        for cache, expected_count in ((train, 512), (dev, 128)):
            if (len(cache.groups) != expected_count
                    or cache.source_identity["controls"]["controls_report_sha256"] != sha
                    or cache.pair_cache.source_selection_seed != spec["source_selection_seed"]
                    or cache.pair_cache.identity["seed"] != spec["model_collection_seed"]):
                raise ValueError("group cache does not match the registered full source pool/seed")
    return arm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    for name in ("train-pairs", "train-controls", "dev-pairs", "dev-controls", "local-run", "backup-run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-train-data-sha256", required=True)
    parser.add_argument("--expected-dev-data-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    spec = json.loads(args.experiment.read_text())
    arm = validate_experiment(spec)
    paths = [p.resolve() for p in (args.train_pairs, args.train_controls, args.dev_pairs, args.dev_controls,
                                  args.local_run, args.backup_run)]
    for index, left in enumerate(paths):
        if any(left.is_relative_to(right) or right.is_relative_to(left) for right in paths[index + 1:]):
            raise ValueError("group caches and run directories must be separate and non-nested")
    for value in (args.expected_train_data_sha256, args.expected_dev_data_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("expected group digests must be SHA-256 values")
    train = load_endpoint_group_cache(args.train_pairs, args.train_controls, "train_fit",
                                      expected_data_sha256=args.expected_train_data_sha256)
    dev = load_endpoint_group_cache(args.dev_pairs, args.dev_controls, "train_dev",
                                    expected_data_sha256=args.expected_dev_data_sha256)
    validate_experiment(spec, train=train, dev=dev)
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = train_endpoint_groups(train, dev, args.local_run, args.backup_run, arm=arm,
            seed=args.seed, device=args.device, epochs=20, batch_groups=8, monitor_every=5,
            experiment_sha256=file_sha256(args.experiment),
            verify_backup=lambda: validate_backup_root(args.backup_run, allow_local=args.allow_local_backup_for_tests),
            should_stop=lambda: interrupted)
        result = write_training_reports(result, args.local_run, args.backup_run,
            lambda: validate_backup_root(args.backup_run, allow_local=args.allow_local_backup_for_tests))
        print(json.dumps(result, indent=2))
        return result
    finally:
        for sig, previous_handler in previous.items(): signal.signal(sig, previous_handler)


if __name__ == "__main__":
    main()
