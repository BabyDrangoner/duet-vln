import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from vln_improve.endpoint_pairs import ORDERS, SCHEMA, SLOTS, PairStore
from vln_improve.endpoint_probe import FEATURE_SCHEMA
from vln_improve.endpoint_pair_training import (
    DATA_SCHEMA, SOURCE_PAIR_SELECTION_SEED, load_endpoint_pair_cache, pair_training_examples,
    validate_endpoint_pair_splits,
)
from vln_improve.protocol import file_sha256, object_sha256
from vln_improve.diagnostics import atomic_json


def _pair(scan, number):
    paths = [str(number * 2), str(number * 2 + 1)]
    pair = {"scan": scan, "start": "s", "path_ids": paths, "instr_ids": [x + "_0" for x in paths],
            "goal_vpids": ["ga", "gb"], "heading_rad": [0., 0.], "heading_difference_deg": 0.,
            "goal_separation_m": 8., "shared_prefix_edges": 0,
            "selection_hash": object_sha256([SOURCE_PAIR_SELECTION_SEED, scan, *paths]), "histories": {}}
    for order, goals in zip(ORDERS, (["ga", "gb"], ["gb", "ga"])):
        walk = ["s", goals[0], "s", goals[1]]
        vps = ["s", *goals]
        pair["histories"][order] = {"observed_vpids": vps, "observed_states": 3,
            "reference_walk": walk, "reference_walk_states": 4, "reference_walk_length_m": 12.,
            "known_map_shortest_proxy_walk": walk, "known_map_shortest_proxy_length_m": 12.,
            "proxy_matches_reference_walk": True,
            "goal_first_observation_indices": {v: vps.index(v) for v in pair["goal_vpids"]}}
    return pair


def _identity(split, pairs):
    return {"schema": SCHEMA, "split": split, "usage": "training" if split == "train_fit" else "analysis_only",
        "selection": pairs, "selection_sha256": object_sha256(pairs), "seed": 0,
        "runtime_config_sha256": "1" * 64, "collection_config_sha256": "2" * 64,
        "coverage_report_sha256": "3" * 64, "code_files": {"collector.py": "4" * 64},
        "common_provenance": {"base_checkpoint_sha256": "5" * 64, "feature_sha256": "6" * 64,
            "annotation_sha256": "7" * 64, "connectivity_sha256": "8" * 64,
            "model": {"dataset": "r2r", "fusion": "dynamic", "batch_size": 1,
                      "max_action_len": 15, "enc_full_graph": True},
            "upstream_lock": {"revision": "test"}, "partition_seed": 20261003,
            "dev_fraction": .2, "torch_version": str(torch.__version__)},
        "feature_schema": FEATURE_SCHEMA, "feature_dim": 1536,
        "execution": "fixture of fixed forced histories", "candidate_cache": "cold cache"}


def _payload(pair, identity):
    rollouts = {}
    for order in ORDERS:
        vps = pair["histories"][order]["observed_vpids"]
        states = []
        prefix = [["s"]]
        for t, vp in enumerate(vps):
            if t:
                prefix.append([vp] if t == 1 else ["s", vp])
            states.append({"step": t, "viewpoint": vp, "heading": .5 if t else 0., "elevation": 0.,
                "view_index": 12, "position": [0., 0., 0.], "trajectory_prefix": copy.deepcopy(prefix),
                "prefix_length_m": [0., 4., 12.][t], "gmap_vpids": [None, "s", "ga", "gb"],
                "vp_cand_vpids": [None, "s"], "masks": {"test": [True]},
                "gmap_step_ids": [0, 1, 0, 0], "gmap_pair_dists": [[0.]],
                "panorama_sha256": object_sha256([order, t, "panorama"]),
                "navigation_without_text_sha256": object_sha256([order, t, "navigation"])})
        distances = torch.tensor([{"s": [4., 4.], "ga": [0., 8.], "gb": [8., 0.]}[v] for v in vps], dtype=torch.float64)
        rollouts[order] = {}
        for index, slot in enumerate(SLOTS):
            rollouts[order][slot] = {"instr_id": pair["instr_ids"][index], "instruction_slot": index,
                "feature_schema": FEATURE_SCHEMA, "mode": "forced_training_history_no_retrospective_fallback",
                "language_input_sha256": object_sha256([pair["instr_ids"][index], "tokens"]),
                "instruction_text_sha256": object_sha256([pair["instr_ids"][index], "text"]),
                "features": torch.full((3, 1536), float(index + 10)), "states": copy.deepcopy(states),
                "trajectory": copy.deepcopy(prefix), "forced_actions": vps[1:] + [None],
                "actual_length_m": 12., "natural_stop_probability": torch.tensor([.1, .3, .7]),
                "wall_seconds": .25,
                "labels": {"goal_vpids": pair["goal_vpids"], "distance_to_goals": distances.clone(),
                           "within_success_radius": distances < 3}}
    return {"schema": SCHEMA, "identity_sha256": object_sha256(identity), "pair": pair,
            "feature_schema": FEATURE_SCHEMA, "rollouts": rollouts}


def _cache(tmp_path, split="train_fit", *, count=2, scan=None, start=0):
    scan = scan or split + "-scene"
    pairs = sorted([_pair(scan, i + start) for i in range(count)], key=lambda x: x["selection_hash"])
    identity = _identity(split, pairs)
    root, backup = tmp_path / split, tmp_path / (split + "-backup")
    store = PairStore(root, backup, identity, lambda: None)
    for pair in pairs:
        store.commit(_payload(pair, identity))
    store.seal({"wall_seconds": 2., "new_pairs": count})
    return root, store


def _json(path):
    return json.loads(path.read_text())


def _commit(root, manifest):
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "COMMITTED.json", {"manifest_sha256": file_sha256(root / "manifest.json")})


def _rewrite_payload(root, change):
    manifest = _json(root / "manifest.json")
    entry = manifest["files"][0]
    directory = root / entry["name"]
    payload = torch.load(directory / "data.pt", weights_only=True)
    change(payload)
    torch.save(payload, directory / "data.pt")
    pair_manifest = _json(directory / "manifest.json")
    pair_manifest["data_sha256"] = file_sha256(directory / "data.pt")
    _commit(directory, pair_manifest)
    manifest["files"][0] = {"name": entry["name"], "manifest_sha256": file_sha256(directory / "manifest.json"), **pair_manifest}
    _commit(root, manifest)


def test_actual_pairstore_format_and_instruction_specific_goal_columns(tmp_path):
    root, _ = _cache(tmp_path)
    assert not (root / "IDENTITY.json").exists()  # actual collector embeds identity in COLLECTION
    cache = load_endpoint_pair_cache(root, "train_fit")
    assert len(cache.pairs) == 2 and cache.manifest["rollouts"] == 8
    examples = pair_training_examples(cache.pairs[0])
    assert [(x.order, x.instruction_slot) for x in examples] == [(o, s) for o in ORDERS for s in (0, 1)]
    for example in examples:
        assert example.features.shape == (3, 1536) and example.features.dtype == torch.float32
        assert example.targets.dtype == torch.float32 and example.targets.sum() == 1
        assert example.targets[example.goal_steps[example.instruction_slot]] == 1
        assert example.targets[example.goal_steps[1 - example.instruction_slot]] == 0
    assert examples[0].targets.tolist() == [0., 1., 0.]
    assert examples[1].targets.tolist() == [0., 0., 1.]
    assert examples[2].targets.tolist() == [0., 0., 1.]
    examples[0].features.zero_()
    assert cache.pairs[0]["rollouts"][ORDERS[0]]["A"]["features"].sum() > 0
    loaded = load_endpoint_pair_cache(root, "train_fit", expected_identity_sha256=cache.identity_sha256,
                                     expected_data_sha256=cache.data_sha256,
                                     expected_common_provenance=cache.common_provenance)
    assert loaded.data_sha256 == cache.data_sha256


def test_training_digest_binds_ordered_payload_bytes_but_not_resume_resources(tmp_path):
    root, store = _cache(tmp_path)
    before = load_endpoint_pair_cache(root, "train_fit")
    store.seal({"wall_seconds": 0., "new_pairs": 0, "reused_pairs": 2})
    after = load_endpoint_pair_cache(root, "train_fit")
    assert before.manifest_sha256 != after.manifest_sha256 and before.data_sha256 == after.data_sha256
    digests = [{k: f[k] for k in ("name", "manifest_sha256", "data_sha256")} for f in after.manifest["files"]]
    assert after.data_sha256 == object_sha256({"schema": DATA_SCHEMA, "collection_identity_sha256": after.identity_sha256,
                                             "pairs": digests})
    _rewrite_payload(root, lambda p: p["rollouts"][ORDERS[0]]["A"]["features"].add_(.125))
    with pytest.raises(ValueError, match="registered data digest"):
        load_endpoint_pair_cache(root, "train_fit", expected_data_sha256=before.data_sha256)


def test_source_selection_seed_is_independent_of_model_seed(tmp_path):
    root, _ = _cache(tmp_path)
    original = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    cache = load_endpoint_pair_cache(root, "train_fit", expected_selection_seed=20261003)
    assert cache.identity["seed"] == 0
    assert cache.source_selection_seed == 20261003
    assert cache.common_identity["source_pair_selection_seed"] == 20261003
    assert load_endpoint_pair_cache(root, "train_fit").data_sha256 == cache.data_sha256
    with pytest.raises(ValueError, match="selection hash"):
        load_endpoint_pair_cache(root, "train_fit", expected_selection_seed=0)
    assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize("target", ["root_commit", "pair_commit", "bytes", "missing_pair", "extra_pair", "extra_file", "reorder", "escape", "duplicate"])
def test_corrupt_partial_or_misassociated_bytes_are_rejected(tmp_path, target):
    root, _ = _cache(tmp_path)
    manifest = _json(root / "manifest.json")
    folder = root / manifest["files"][0]["name"]
    if target == "root_commit": (root / "COMMITTED.json").unlink()
    elif target == "pair_commit": (folder / "COMMITTED.json").unlink()
    elif target == "bytes":
        payload = torch.load(folder / "data.pt", weights_only=True)
        payload["rollouts"][ORDERS[0]]["A"]["features"].add_(1)
        torch.save(payload, folder / "data.pt")
    elif target == "missing_pair": shutil.rmtree(folder)
    elif target == "extra_pair": shutil.copytree(folder, root / ("pair-" + "f" * 64))
    elif target == "extra_file": shutil.copyfile(folder / "data.pt", folder / "unregistered.pt")
    else:
        if target == "reorder": manifest["files"].reverse()
        elif target == "escape": manifest["files"][0]["name"] = "../escape"
        elif target == "duplicate": manifest["files"][1] = manifest["files"][0]
        _commit(root, manifest)
    with pytest.raises(ValueError):
        load_endpoint_pair_cache(root, "train_fit")


@pytest.mark.parametrize("where", ["root", "pair", "data", "commit", "collection"])
def test_symbolic_links_are_rejected_even_if_contents_match(tmp_path, where):
    root, _ = _cache(tmp_path)
    folder = root / _json(root / "manifest.json")["files"][0]["name"]
    target = {"root": root, "pair": folder, "data": folder / "data.pt",
              "commit": folder / "COMMITTED.json", "collection": root / "COLLECTION.json"}[where]
    real = tmp_path / ("real-" + where)
    target.rename(real)
    target.symlink_to(real, target_is_directory=real.is_dir())
    with pytest.raises(ValueError, match="symbolic|ordinary"):
        load_endpoint_pair_cache(root, "train_fit")


@pytest.mark.parametrize("damage", ["slot", "instr", "goal_columns", "success", "cross_order_distance", "parity",
                                    "future_transit", "nan_feature", "label_feature_column", "language"])
def test_even_rehashed_semantic_or_label_corruption_is_rejected(tmp_path, damage):
    root, _ = _cache(tmp_path)
    def change(payload):
        run = payload["rollouts"][ORDERS[0]]["A"]
        if damage == "slot": run["instruction_slot"] = 1
        elif damage == "instr": run["instr_id"] = payload["pair"]["instr_ids"][1]
        elif damage == "goal_columns": run["labels"]["goal_vpids"].reverse()
        elif damage == "success": run["labels"]["within_success_radius"][0, 0] = True
        elif damage == "parity": run["states"][1]["heading"] += .001
        elif damage == "nan_feature": run["features"][0, 0] = torch.nan
        elif damage == "label_feature_column": run["features"] = torch.cat([run["features"], torch.ones(3, 1)], dim=1)
        elif damage == "language": run["language_input_sha256"] = "f" * 64
        elif damage == "cross_order_distance":
            for slot in SLOTS:
                payload["rollouts"][ORDERS[1]][slot]["labels"]["distance_to_goals"][0, 0] = 5.
        elif damage == "future_transit":
            for slot in SLOTS:
                r = payload["rollouts"][ORDERS[0]][slot]
                r["states"][1]["trajectory_prefix"] = [["s"], ["gb", "ga"]]
                r["states"][2]["trajectory_prefix"] = [["s"], ["gb", "ga"], ["s", "gb"]]
                r["trajectory"] = r["states"][-1]["trajectory_prefix"]
    _rewrite_payload(root, change)
    with pytest.raises(ValueError):
        load_endpoint_pair_cache(root, "train_fit")


def test_split_usage_pins_and_redundant_identity_are_checked(tmp_path):
    root, _ = _cache(tmp_path)
    for split in ("val_unseen", "train_dev"):
        with pytest.raises(ValueError): load_endpoint_pair_cache(root, split)
    with pytest.raises(ValueError, match="registered identity"):
        load_endpoint_pair_cache(root, "train_fit", expected_identity_sha256="0" * 64)
    with pytest.raises(ValueError, match="registered source"):
        load_endpoint_pair_cache(root, "train_fit", expected_common_provenance={})
    identity = _json(root / "COLLECTION.json")["identity"]
    atomic_json(root / "IDENTITY.json", identity)
    load_endpoint_pair_cache(root, "train_fit")
    identity["split"] = "val_unseen"
    atomic_json(root / "IDENTITY.json", identity)
    with pytest.raises(ValueError, match="redundant"):
        load_endpoint_pair_cache(root, "train_fit")


def test_fake_training_label_and_repeated_path_are_rejected_before_data_loading(tmp_path):
    root, _ = _cache(tmp_path)
    collection = _json(root / "COLLECTION.json")
    original = copy.deepcopy(collection)
    collection["identity"]["usage"] = "training"  # train_fit is unchanged
    collection["identity"]["split"] = "val_unseen"
    collection["identity_sha256"] = object_sha256(collection["identity"])
    atomic_json(root / "COLLECTION.json", collection)
    with pytest.raises(ValueError, match="split/usage"):
        load_endpoint_pair_cache(root, "train_fit")
    collection = original
    collection["identity"]["selection"][1] = copy.deepcopy(collection["identity"]["selection"][0])
    collection["identity"]["selection_sha256"] = object_sha256(collection["identity"]["selection"])
    collection["identity_sha256"] = object_sha256(collection["identity"])
    atomic_json(root / "COLLECTION.json", collection)
    with pytest.raises(ValueError, match="repeats"):
        load_endpoint_pair_cache(root, "train_fit")


def test_fit_dev_common_provenance_and_scene_independence(tmp_path):
    fit, _ = _cache(tmp_path / "a")
    dev, _ = _cache(tmp_path / "b", "train_dev", start=10)
    train = load_endpoint_pair_cache(fit, "train_fit")
    validation = load_endpoint_pair_cache(dev, "train_dev")
    assert validate_endpoint_pair_splits(train, validation) == train.common_identity
    validation.identity["common_provenance"]["feature_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="provenance"):
        validate_endpoint_pair_splits(train, validation)
    overlap, _ = _cache(tmp_path / "c", "train_dev", scan="train_fit-scene", start=10)
    with pytest.raises(ValueError, match="share scan"):
        validate_endpoint_pair_splits(train, load_endpoint_pair_cache(overlap, "train_dev"))
