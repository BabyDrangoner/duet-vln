"""Experiment identity and scene-level train/development separation."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select_partition(records, split, dev_fraction=0.2, partition_seed=20261003):
    if split not in ("train_fit", "train_dev"):
        return list(records)
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie between 0 and 1")
    scans = sorted({r["scan"] for r in records},
                   key=lambda s: object_sha256([partition_seed, s]))
    if len(scans) < 2:
        raise ValueError("Scene separation needs at least two training scenes")
    n_dev = min(len(scans) - 1, max(1, round(len(scans) * dev_fraction)))
    dev = set(scans[:n_dev])
    return [r for r in records if (r["scan"] in dev) == (split == "train_dev")]


def resolve_config(config_path, root):
    config = json.loads(Path(config_path).read_text())
    for key in ("dataset_root", "base_checkpoint"):
        path = Path(config[key]).expanduser()
        config[key] = str(path.resolve() if path.is_absolute() else (root / path).resolve())
    if config["model"]["dataset"] != "r2r" or config["model"]["fusion"] != "dynamic":
        raise ValueError("First implementation supports R2R with DUET dynamic fusion only")
    return config
