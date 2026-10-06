"""Offline training on bounded cache shards from training environments only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .features import FEATURE_SCHEMA
from .head import ResidualActionHead, save_head_checkpoint


PROVENANCE_FIELDS = {
    "dataset", "feature_id", "base_checkpoint_sha256", "upstream_commit", "max_action_len", "feedback"
}


@dataclass(frozen=True)
class CacheManifest:
    root: Path
    feature_dim: int
    provenance: dict[str, Any]
    shards: tuple[Path, ...]
    num_records: int
    manifest_sha256: str


def read_manifests(cache_dirs: Sequence[str | Path]) -> list[CacheManifest]:
    if not cache_dirs:
        raise ValueError("at least one --cache is required")
    manifests = []
    seen_roots = set()
    for directory in cache_dirs:
        root = Path(directory).resolve(strict=True)
        if root in seen_roots:
            raise ValueError(f"duplicate cache directory: {root}")
        seen_roots.add(root)
        manifest_path = (root / "manifest.json").resolve(strict=True)
        if not manifest_path.is_relative_to(root):
            raise ValueError("manifest path escapes the cache directory")
        raw_manifest = manifest_path.read_bytes()
        data = json.loads(raw_manifest)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported cache schema_version")
        if data.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError(f"cache feature_schema must be {FEATURE_SCHEMA!r}")
        if data.get("split") != "train_fit":
            raise ValueError("training accepts only split='train_fit'; validation data is forbidden")
        feature_dim = data.get("feature_dim")
        count = data.get("num_records")
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("cache feature_dim must be a positive integer")
        if type(count) is not int or count <= 0:
            raise ValueError("cache num_records must be a positive integer")
        provenance = data.get("provenance")
        if not isinstance(provenance, dict) or not PROVENANCE_FIELDS.issubset(provenance):
            raise ValueError("cache provenance is missing required fields")
        for field in PROVENANCE_FIELDS - {"max_action_len"}:
            if not isinstance(provenance[field], str) or not provenance[field]:
                raise ValueError(f"provenance {field} must be a nonempty string")
        if type(provenance["max_action_len"]) is not int or provenance["max_action_len"] <= 0:
            raise ValueError("provenance max_action_len must be a positive integer")
        names = data.get("shards")
        if not isinstance(names, list) or not names:
            raise ValueError("cache shards must be a nonempty list")
        paths = []
        for name in names:
            if not isinstance(name, str) or not name or Path(name).is_absolute():
                raise ValueError("shard path must be relative to its cache directory")
            candidate = (root / name).resolve()
            if not candidate.is_relative_to(root):
                raise ValueError("shard path escapes the cache directory")
            if not candidate.is_file():
                raise ValueError(f"missing shard: {name}")
            paths.append(candidate)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate shard paths in cache manifest")
        current = CacheManifest(
            root, feature_dim, provenance, tuple(paths), count,
            hashlib.sha256(raw_manifest).hexdigest(),
        )
        if manifests and (
            current.feature_dim != manifests[0].feature_dim
            or current.provenance != manifests[0].provenance
        ):
            raise ValueError("all caches must have identical feature_dim and provenance")
        manifests.append(current)
    return manifests


def _validate_record(record: Any, feature_dim: int) -> None:
    required = {"features", "base_logits", "valid_mask", "target", "hard", "instr_id", "scan_id"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("cache record is missing required fields")
    features, logits, mask = (record[key] for key in ("features", "base_logits", "valid_mask"))
    if not all(isinstance(value, Tensor) for value in (features, logits, mask)):
        raise ValueError("record features, logits and mask must be tensors")
    if features.dtype != torch.float16 or logits.dtype != torch.float32 or mask.dtype != torch.bool:
        raise ValueError("record dtypes must be float16 features, float32 logits, and bool mask")
    if features.ndim != 2 or features.shape[1] != feature_dim or features.shape[0] == 0:
        raise ValueError("invalid record feature shape")
    if logits.shape != (features.shape[0],) or mask.shape != logits.shape:
        raise ValueError("record logits and mask do not match candidate count")
    if not mask.any():
        raise ValueError("record must contain at least one legal action")
    if not torch.isfinite(features[mask]).all() or not torch.isfinite(logits[mask]).all():
        raise ValueError("valid record features and logits must be finite")
    target = record["target"]
    if type(target) is not int or not 0 <= target < len(mask) or not mask[target]:
        raise ValueError("record target must index a legal action")
    if type(record["hard"]) is not bool:
        raise ValueError("record hard must be boolean")
    for field in ("instr_id", "scan_id"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"record {field} must be a nonempty string")


def iter_records(manifests: Sequence[CacheManifest], seed: int) -> Iterator[dict[str, Any]]:
    """Load one <=128-record shard at a time; never concatenate the full cache."""
    rng = random.Random(seed)
    jobs = [(index, path) for index, manifest in enumerate(manifests) for path in manifest.shards]
    rng.shuffle(jobs)
    counts = [0] * len(manifests)
    for index, path in jobs:
        records = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(records, list) or not 1 <= len(records) <= 128:
            raise ValueError("each shard must contain a list of 1 to 128 records")
        rng.shuffle(records)
        for record in records:
            _validate_record(record, manifests[index].feature_dim)
            counts[index] += 1
            yield record
    for manifest, count in zip(manifests, counts):
        if count != manifest.num_records:
            raise ValueError(f"num_records mismatch in {manifest.root}: manifest={manifest.num_records}, actual={count}")


def collate_records(records: Sequence[dict[str, Any]], device: str | torch.device = "cpu") -> dict[str, Tensor]:
    if not records:
        raise ValueError("cannot collate an empty batch")
    feature_dim = records[0]["features"].shape[-1]
    for record in records:
        _validate_record(record, feature_dim)
    count = max(len(record["base_logits"]) for record in records)
    features = torch.zeros(len(records), count, feature_dim, dtype=torch.float32)
    logits = torch.full((len(records), count), -torch.inf, dtype=torch.float32)
    mask = torch.zeros(len(records), count, dtype=torch.bool)
    for index, record in enumerate(records):
        length = len(record["base_logits"])
        features[index, :length] = record["features"]
        logits[index, :length] = record["base_logits"]
        mask[index, :length] = record["valid_mask"]
    return {
        "features": features.to(device),
        "base_logits": logits.to(device),
        "valid_mask": mask.to(device),
        "target": torch.tensor([record["target"] for record in records], dtype=torch.long, device=device),
        "hard": torch.tensor([record["hard"] for record in records], dtype=torch.bool, device=device),
    }


def training_loss(
    corrected_logits: Tensor,
    base_logits: Tensor,
    valid_mask: Tensor,
    targets: Tensor,
    hard: Tensor,
    *,
    hard_weight: float,
    kl_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_record_ce = F.cross_entropy(corrected_logits, targets, reduction="none")
    weights = torch.where(hard, hard_weight, 1.0).to(corrected_logits.dtype)
    ce_numerator = (per_record_ce * weights).sum()
    ce = ce_numerator / weights.sum()
    base = base_logits.detach().masked_fill(~valid_mask, -torch.inf)
    base_log_p = F.log_softmax(base, dim=-1).masked_fill(~valid_mask, 0.0)
    corrected_log_p = F.log_softmax(corrected_logits, dim=-1).masked_fill(~valid_mask, 0.0)
    base_p = F.softmax(base, dim=-1)
    per_record_kl = (base_p * (base_log_p - corrected_log_p)).sum(dim=-1)
    kl = per_record_kl.mean()
    loss = ce + kl_weight * kl
    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    return loss, {"ce_numerator": ce_numerator.detach(), "weight_sum": weights.sum().detach(), "kl_sum": per_record_kl.sum().detach()}


def train_from_caches(
    cache_dirs: Sequence[str | Path],
    output: str | Path,
    *,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 0.001,
    hidden_dim: int = 128,
    max_delta: float = 1.0,
    hard_weight: float = 3.0,
    kl_weight: float = 0.1,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    if type(epochs) is not int or epochs <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive integers")
    for name, value in (("lr", lr), ("hard_weight", hard_weight)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and nonnegative")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("training device must be cpu or cuda")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    manifests = read_manifests(cache_dirs)
    random.seed(seed)
    torch.manual_seed(seed)
    head = ResidualActionHead(manifests[0].feature_dim, hidden_dim, max_delta).to(selected_device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        head.train()
        totals = {"ce_numerator": 0.0, "weight_sum": 0.0, "kl_sum": 0.0, "num_records": 0}
        buffer: list[dict[str, Any]] = []

        def update(records: list[dict[str, Any]]) -> None:
            batch = collate_records(records, selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch["features"], batch["base_logits"], batch["valid_mask"])
            loss, statistics = training_loss(
                logits, batch["base_logits"], batch["valid_mask"], batch["target"], batch["hard"],
                hard_weight=hard_weight, kl_weight=kl_weight,
            )
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
                raise ValueError("non-finite gradients")
            optimizer.step()
            for name, value in statistics.items():
                totals[name] += value.item()
            totals["num_records"] += len(records)

        for record in iter_records(manifests, seed + epoch):
            buffer.append(record)
            if len(buffer) == batch_size:
                update(buffer)
                buffer = []
        if buffer:
            update(buffer)
        ce = totals["ce_numerator"] / totals["weight_sum"]
        kl = totals["kl_sum"] / totals["num_records"]
        measured = {"epoch": epoch + 1, "train_loss": ce + kl_weight * kl, "train_ce": ce, "train_kl": kl, "num_records": totals["num_records"]}
        history.append(measured)
        print(json.dumps(measured, ensure_ascii=False), flush=True)
    args = {
        "cache": [str(manifest.root) for manifest in manifests], "output": str(Path(output)),
        "cache_manifests": [
            {"path": str(manifest.root / "manifest.json"), "sha256": manifest.manifest_sha256}
            for manifest in manifests
        ],
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "hidden_dim": hidden_dim,
        "max_delta": max_delta, "hard_weight": hard_weight, "kl_weight": kl_weight,
        "seed": seed, "device": device,
    }
    return save_head_checkpoint(
        head, output, provenance=manifests[0].provenance, train_args=args, seed=seed,
        metrics={"kind": "offline_training_only", "epochs": history},
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", required=True, help="train_fit cache; repeat for aggregated rounds")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-delta", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=3.0)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = vars(parser.parse_args(argv))
    cache_dirs = args.pop("cache")
    output = args.pop("output")
    train_from_caches(cache_dirs, output, **args)


if __name__ == "__main__":
    main()
