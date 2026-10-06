import copy
import hashlib
import json

import pytest
import torch

from vln_improve.features import FEATURE_SCHEMA
from vln_improve.head import ResidualActionHead, load_head_checkpoint
from vln_improve.train import collate_records, iter_records, read_manifests, train_from_caches, training_loss


PROVENANCE = {
    "dataset": "synthetic-test-only", "feature_id": "fixture-two-features",
    "base_checkpoint_sha256": "a" * 64, "upstream_commit": "b" * 40,
    "max_action_len": 15, "feedback": "argmax",
}


def make_record(index=0, length=2):
    target = index % 2
    features = torch.zeros(length, 2, dtype=torch.float16)
    features[:, 1] = 1
    features[target] = torch.tensor([1, 0], dtype=torch.float16)
    logits = torch.full((length,), 0.5, dtype=torch.float32)
    logits[target] = 0
    return {
        "features": features, "base_logits": logits,
        "valid_mask": torch.ones(length, dtype=torch.bool), "target": target,
        "hard": index % 3 != 0, "instr_id": f"instruction-{index}", "scan_id": "train-house",
    }


def make_cache(path, records=None, **manifest_overrides):
    path.mkdir(parents=True)
    records = records if records is not None else [make_record()]
    names = []
    for start in range(0, len(records), 128):
        name = f"shard-{start:04d}.pt"
        torch.save(records[start:start + 128], path / name)
        names.append(name)
    manifest = {
        "schema_version": 1, "feature_schema": FEATURE_SCHEMA, "feature_dim": 2, "split": "train_fit",
        "provenance": PROVENANCE.copy(), "shards": names, "num_records": len(records),
        **manifest_overrides,
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_training_learns_ranking_and_records_only_training_metrics(tmp_path):
    records = [make_record(index, 2 + index % 2) for index in range(12)]
    first = make_cache(tmp_path / "round0", records[:6])
    second = make_cache(tmp_path / "round1", records[6:])
    output = tmp_path / "trained.pt"
    metadata = train_from_caches(
        [first, second], output, epochs=15, batch_size=12, lr=0.05,
        hidden_dim=16, max_delta=2, hard_weight=3, kl_weight=0.01, seed=1,
    )
    head, stored = load_head_checkpoint(output, expected_provenance=PROVENANCE)
    batch = collate_records(records)
    predictions = head(batch["features"], batch["base_logits"], batch["valid_mask"]).argmax(-1)
    assert torch.equal(predictions, batch["target"])
    assert not torch.any(batch["base_logits"].argmax(-1) == batch["target"])
    history = metadata["metrics"]["epochs"]
    assert history[-1]["train_ce"] < history[0]["train_ce"] * 0.5
    assert stored == metadata
    assert stored["train_args"]["cache_manifests"] == [
        {"path": str(cache.resolve() / "manifest.json"),
         "sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()}
        for cache in (first, second)
    ]
    assert metadata["metrics"]["kind"] == "offline_training_only"
    assert all(epoch["num_records"] == 12 for epoch in history)
    assert not ({"sr", "spl", "success_rate"} & metadata["metrics"].keys())


def test_variable_lengths_masked_kl_and_exact_hard_weights():
    records = [make_record(0, 2), make_record(1, 4), make_record(2, 3)]
    records[1]["valid_mask"][-1] = False
    records[1]["features"][-1] = float("nan")
    records[1]["base_logits"][-1] = float("nan")
    batch = collate_records(records)
    head = ResidualActionHead(2, hidden_dim=8)
    corrected = head(batch["features"], batch["base_logits"], batch["valid_mask"])
    loss, stats = training_loss(
        corrected, batch["base_logits"], batch["valid_mask"], batch["target"], batch["hard"],
        hard_weight=3, kl_weight=0.1,
    )
    ce = torch.nn.functional.cross_entropy(corrected, batch["target"], reduction="none")
    expected = (ce * torch.tensor([1.0, 3.0, 3.0])).sum() / 7
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(stats["kl_sum"], torch.tensor(0.0), atol=1e-7, rtol=0)
    assert stats["weight_sum"] == 7
    assert torch.isneginf(corrected[~batch["valid_mask"]]).all()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())


@pytest.mark.parametrize("change", [
    {"split": "val_unseen"}, {"feature_dim": 3},
    {"provenance": {**PROVENANCE, "feature_id": "different"}},
    {"provenance": {**PROVENANCE, "base_checkpoint_sha256": "c" * 64}},
])
def test_incompatible_caches_are_rejected(tmp_path, change):
    first = make_cache(tmp_path / "a")
    second = make_cache(tmp_path / "b", **change)
    with pytest.raises(ValueError):
        train_from_caches([first, second], tmp_path / "should-not-exist.pt", epochs=1)
    assert not (tmp_path / "should-not-exist.pt").exists()


@pytest.mark.parametrize("schema", [None, "incompatible_reordered_features_v2", "missing"])
def test_missing_or_incompatible_feature_schema_is_rejected(tmp_path, schema):
    first = make_cache(tmp_path / "round0")
    second = make_cache(tmp_path / "round1", feature_schema=schema)
    if schema == "missing":
        path = second / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["feature_schema"]
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="feature_schema"):
        read_manifests([first, second])


@pytest.mark.parametrize("alias_kind", ["same_path", "symlink"])
def test_duplicate_cache_directory_is_rejected_after_resolving(tmp_path, alias_kind):
    cache = make_cache(tmp_path / "cache")
    alias = cache
    if alias_kind == "symlink":
        alias = tmp_path / "cache-alias"
        alias.symlink_to(cache, target_is_directory=True)
    with pytest.raises(ValueError, match="duplicate cache directory"):
        read_manifests([cache, alias])


def test_manifest_hash_tracks_original_bytes_not_reformatted_json(tmp_path):
    cache = make_cache(tmp_path / "cache")
    path = cache / "manifest.json"
    original = path.read_bytes()
    before = read_manifests([cache])[0]
    assert before.manifest_sha256 == hashlib.sha256(original).hexdigest()
    path.write_text(json.dumps(json.loads(original), indent=2) + "\n")
    after = read_manifests([cache])[0]
    assert after.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert after.manifest_sha256 != before.manifest_sha256
    assert after.provenance == before.provenance


@pytest.mark.parametrize("name", ["../outside.pt", "absolute", "symlink"])
def test_cache_path_escape_is_rejected(tmp_path, name):
    outside = tmp_path / "outside.pt"
    torch.save([make_record()], outside)
    shard_name = str(outside) if name == "absolute" else name
    cache = make_cache(tmp_path / "cache", shards=[shard_name])
    if name == "symlink":
        (cache / name).symlink_to(outside)
    with pytest.raises(ValueError, match="relative|escapes"):
        read_manifests([cache])


@pytest.mark.parametrize("bad_kind", ["nan", "illegal_target", "all_masked", "count", "oversized_shard"])
def test_corrupt_records_or_shards_are_rejected(tmp_path, bad_kind):
    record = make_record()
    if bad_kind == "nan":
        record["base_logits"][0] = float("nan")
    elif bad_kind == "illegal_target":
        record["valid_mask"][record["target"]] = False
    elif bad_kind == "all_masked":
        record["valid_mask"][:] = False
    cache = make_cache(tmp_path / "cache", [record], num_records=2 if bad_kind == "count" else 1)
    if bad_kind == "oversized_shard":
        torch.save([copy.deepcopy(record) for _ in range(129)], cache / "shard-0000.pt")
    with pytest.raises(ValueError):
        list(iter_records(read_manifests([cache]), seed=0))


def test_streams_more_than_one_shard_without_losing_records(tmp_path):
    cache = make_cache(tmp_path / "cache", [make_record(index) for index in range(131)])
    seen = [record["instr_id"] for record in iter_records(read_manifests([cache]), seed=42)]
    assert len(seen) == len(set(seen)) == 131
