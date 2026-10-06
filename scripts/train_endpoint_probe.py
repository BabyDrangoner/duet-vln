#!/usr/bin/env python3
"""Train a small frozen-feature endpoint control with durable step-boundary resume."""

import argparse
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vln_improve.endpoint_probe import train_endpoint_probe
from vln_improve.pipeline import atomic_json, validate_backup_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--backup-run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args(argv)
    roots = [x.resolve() for x in (args.train_cache, args.dev_cache, args.local_run, args.backup_run)]
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("cache and run directories must be separate and non-nested")
    interrupted = False

    def on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True

    previous = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = train_endpoint_probe(
            args.train_cache, args.dev_cache, args.local_run, args.backup_run, device=args.device,
            epochs=20, batch_episodes=32, lr=1e-3, weight_decay=1e-4, seed=0,
            verify_backup=lambda: validate_backup_root(args.backup_run, allow_local=args.allow_local_backup_for_tests),
            should_stop=lambda: interrupted)
        atomic_json(args.local_run / "training-summary.json", result)
        validate_backup_root(args.backup_run, allow_local=args.allow_local_backup_for_tests)
        atomic_json(args.backup_run / "training-summary.json", result)
        if json.loads((args.backup_run / "training-summary.json").read_bytes()) != result:
            raise ValueError("endpoint training summary backup read-back mismatch")
        print(json.dumps(result, indent=2))
        return result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
