#!/usr/bin/env python3
"""Replay frozen diagnostic states with future observations as offline controls.

Writes analysis rows, not a deployable model or navigation performance claim.
Completed episodes have atomic result files, allowing interrupted replay to
resume without silently reusing a different model, collection, or program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from prepare_duet import verify, DEFAULT_DEST
from vln_improve.counterfactual import clone_navigation, evaluate_replacement, legal_mask, observable_features, replay_identity
from vln_improve.diagnostics import SCHEMA, load_episode
from vln_improve.pipeline import atomic_json, validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256, resolve_config


def load_model(config_file):
    lock = verify()
    config = resolve_config(config_file, ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("actual DUET replay requires the CUDA runtime")
    sys.path.insert(0, str(DEFAULT_DEST / "map_nav_src"))
    from r2r.parser import parse_args
    from models.model import VLNBert
    argv = ["duet", "--root_dir", config["dataset_root"], "--output_dir", str(ROOT / "outputs/replay-upstream"), "--test"]
    for key, value in config["model"].items():
        if isinstance(value, bool):
            if value: argv.append("--" + key)
        else: argv.extend(["--" + key, str(value)])
    previous = sys.argv
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = previous
    for key, value in config["model"].items():
        if not hasattr(args, key) or getattr(args, key) != value:
            raise ValueError(f"model option was not applied: {key}")
    model = VLNBert(args).cuda().eval().requires_grad_(False)
    payload = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in payload["vln_bert"]["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    del state, payload
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return model, {"model": config["model"], "upstream_lock": lock,
                   "base_checkpoint_sha256": file_sha256(config["base_checkpoint"]),
                   "torch_version": str(torch.__version__)}


def validate_runtime(collected, runtime):
    for key in ("model", "upstream_lock", "base_checkpoint_sha256", "torch_version"):
        if key not in runtime or key not in collected or runtime[key] != collected[key]:
            raise ValueError(f"replay runtime differs from collection: {key}")


def chosen_candidates(state, seed):
    legal = state["valid_mask"].tolist()
    options = [index for index in range(1, len(legal)) if legal[index]]
    chosen = int(state["base_logits"].argmax())
    first = [chosen] if chosen in options else []
    rest = [index for index in options if index != chosen]
    rest.sort(key=lambda index: object_sha256([seed, state["step"], state["nav_inputs"]["gmap_vpids"][0][index]]))
    # Selection uses current state only, never whether a future label exists.
    return first + rest[:1]


def _step_map(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    result = {}
    for value in values:
        step = value.get("step") if isinstance(value, dict) else None
        if type(step) is not int or step < 0 or step in result:
            raise ValueError(f"{name} contains an invalid or duplicate step")
        result[step] = value
    if list(result) != sorted(result):
        raise ValueError(f"{name} steps are not chronological")
    return result


def _navigation_state(state, association):
    data = clone_navigation(state["nav_inputs"], "cpu")
    graph = data["gmap_img_embeds"]
    _, nodes, dimension = graph.shape
    text = data["txt_embeds"]
    local = data["vp_img_embeds"]
    if (text.ndim != 3 or text.shape[0] != 1 or text.shape[2] != dimension
            or local.ndim != 3 or local.shape[0] != 1 or local.shape[2] != dimension):
        raise ValueError("navigation text/local feature shapes disagree")
    shapes = {
        "txt_masks": (1, text.shape[1]), "gmap_step_ids": (1, nodes),
        "gmap_pos_fts": (1, nodes, 7), "gmap_masks": (1, nodes),
        "gmap_pair_dists": (1, nodes, nodes), "gmap_visited_masks": (1, nodes),
        "vp_pos_fts": (1, local.shape[1], 14), "vp_masks": (1, local.shape[1]),
        "vp_nav_masks": (1, local.shape[1]),
    }
    for key, shape in shapes.items():
        if not isinstance(data[key], torch.Tensor) or tuple(data[key].shape) != shape:
            raise ValueError(f"invalid navigation tensor shape: {key}")
    mask_names = {"txt_masks", "gmap_masks", "gmap_visited_masks", "vp_masks", "vp_nav_masks"}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            continue
        expected_dtype = torch.bool if key in mask_names else (torch.int64 if key == "gmap_step_ids" else torch.float32)
        if value.dtype != expected_dtype or (value.is_floating_point() and not torch.isfinite(value).all()):
            raise ValueError(f"navigation tensor dtype/value mismatch: {key}")
    if data["vp_obj_masks"] is not None:
        raise ValueError("R2R M0 replay requires vp_obj_masks=None")
    vpids = data["gmap_vpids"]
    local_ids = data["vp_cand_vpids"]
    for name, values, maximum in (("gmap_vpids", vpids, nodes), ("vp_cand_vpids", local_ids, local.shape[1])):
        if (not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], list)
                or not values[0] or values[0][0] is not None or len(values[0]) > maximum
                or not all(isinstance(value, str) and value for value in values[0][1:])
                or len(set(values[0])) != len(values[0])):
            raise ValueError(f"invalid navigation node identifiers: {name}")
    if len(vpids[0]) != nodes or not data["gmap_masks"].all():
        raise ValueError("M0 replay requires a complete unpadded batch=1 graph")
    if (data["vp_nav_masks"] & ~data["vp_masks"]).any() or not data["vp_nav_masks"][0, 0]:
        raise ValueError("invalid local navigation masks")
    if not data["vp_nav_masks"][0, :len(local_ids[0])].all() or data["vp_nav_masks"][0, len(local_ids[0]):].any():
        raise ValueError("local candidate IDs and navigation masks disagree")
    current = state["current_viewpoint"]
    if current not in vpids[0] or not data["gmap_visited_masks"][0, vpids[0].index(current)]:
        raise ValueError("current viewpoint is missing or not visited in the frozen graph")
    legal = legal_mask(data)
    if (not isinstance(state["valid_mask"], torch.Tensor) or state["valid_mask"].dtype != torch.bool
            or not torch.equal(state["valid_mask"].cpu(), legal)):
        raise ValueError("recorded legal mask differs from frozen navigation inputs")
    for key, length, valid in (("base_logits", nodes, legal), ("base_global_logits", nodes, legal),
                               ("base_local_logits", local.shape[1], data["vp_nav_masks"][0])):
        scores = state[key]
        if (not isinstance(scores, torch.Tensor) or scores.shape != (length,) or scores.dtype != torch.float32
                or not torch.isfinite(scores.cpu()[valid]).all()
                or not torch.isneginf(scores.cpu()[~valid]).all()):
            raise ValueError(f"invalid recorded branch logits: {key}")
    if type(state.get("eligible_decision")) is not bool:
        raise ValueError("eligible_decision must be boolean")
    targets = {vpids[0][index] for index in range(1, nodes) if legal[index]}
    if set(state["candidate_evidence"]) != targets:
        raise ValueError("candidate evidence does not match legal unvisited nodes")
    for target, snapshot in state["candidate_evidence"].items():
        expected = {**association, "target_id": target}
        if snapshot.get("association") != expected:
            raise ValueError("source evidence belongs to another episode or target")
        observable_features(state, vpids[0].index(target))
    return data


def validate_episode(inputs, labels, split):
    """Validate all joins before sampling; malformed unused labels also fail."""
    if split not in {"train_fit", "train_dev"}:
        raise ValueError("replay only accepts train_fit or analysis-only train_dev")
    if inputs.get("schema") != SCHEMA or labels.get("schema") != SCHEMA:
        raise ValueError("unsupported diagnostic episode schema")
    association = inputs.get("association")
    if (not isinstance(association, dict) or set(association) != {"episode_id", "scan_id", "instr_id"}
            or not all(isinstance(value, str) and value for value in association.values())
            or labels.get("association") != association):
        raise ValueError("input/label episode association mismatch")
    expected_usage = "training_diagnostics" if split == "train_fit" else "analysis_only"
    if labels.get("usage") != expected_usage:
        raise ValueError("episode label usage differs from requested split")
    states = _step_map(inputs["states"], "input states")
    oracle = _step_map(labels["states"], "oracle states")
    if set(states) != set(oracle):
        raise ValueError("input and oracle step sets differ")
    navigation = {step: _navigation_state(state, association) for step, state in states.items()}
    for step, state in states.items():
        legal = legal_mask(navigation[step])
        for kind in ("teacher", "execution"):
            costs = oracle[step][kind + "_cost"]
            if (not isinstance(costs, torch.Tensor) or costs.shape != legal.shape
                    or not costs.is_floating_point() or torch.isnan(costs).any() or (costs < 0).any()
                    or torch.isfinite(costs.cpu()[~legal]).any()):
                raise ValueError("invalid oracle cost vector")
            finite = torch.isfinite(costs)
            expected = torch.nonzero(finite & ((costs - costs[finite].min()).abs() <= 1e-6)).flatten().tolist() if finite.any() else []
            if oracle[step].get(kind + "_optimal_indices") != expected:
                raise ValueError("oracle optimal indices disagree with costs")
    pairs = {}
    for pair in labels["arrival_pairs"]:
        joined = pair["association"]
        target = joined.get("target_id")
        if not isinstance(target, str) or not target or joined != {**association, "target_id": target}:
            raise ValueError("arrival pair belongs to another episode")
        if target in pairs:
            raise ValueError("multiple first-arrival labels for one target")
        future = pair["training_only"]
        arrival_step = future["arrival_step"]
        if (type(arrival_step) is not int or arrival_step not in states
                or states[arrival_step]["current_viewpoint"] != target
                or future.get("arrival_observed") is not True or future.get("arrival_kind") != "natural"):
            raise ValueError("arrival does not match a naturally observed episode state")
        prior = pair["inference_snapshot"]
        if (prior.get("association") != joined or prior.get("status") != "unvisited"
                or not prior["sources"] or prior["available_at_step"] >= arrival_step
                or any(source["last_step"] >= arrival_step for source in prior["sources"])):
            raise ValueError("arrival must follow its source evidence strictly")
        if not any(step < arrival_step and target in state["candidate_evidence"] for step, state in states.items()):
            raise ValueError("arrival target was never an earlier candidate")
        observed = navigation[arrival_step]
        index = observed["gmap_vpids"][0].index(target)
        feature = future["arrival_feature"]
        if (not isinstance(feature, torch.Tensor) or feature.shape != observed["gmap_img_embeds"][0, index].shape
                or feature.dtype != torch.float32 or not torch.isfinite(feature).all()
                or not torch.allclose(feature.cpu(), observed["gmap_img_embeds"][0, index], atol=1e-6, rtol=1e-5)):
            raise ValueError("arrival feature differs from the actual observed state")
        pairs[target] = pair
    return association, states, oracle, pairs


def seal_result(result, *, input_manifest_sha256, replay_identity_sha256):
    payload = {key: value for key, value in result.items() if key != "content_sha256"}
    payload.update({"schema": "duet_episode_replay_v1", "input_manifest_sha256": input_manifest_sha256,
                    "replay_identity_sha256": replay_identity_sha256})
    return {**payload, "content_sha256": object_sha256(payload)}


def read_result(path, *, input_manifest_sha256, replay_identity_sha256, association, split):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("cached replay result must not be a symlink")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError("cached replay result must be an object")
    content = {key: value for key, value in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != object_sha256(content):
        raise ValueError("cached replay content checksum mismatch")
    if (result.get("schema") != "duet_episode_replay_v1"
            or result.get("input_manifest_sha256") != input_manifest_sha256
            or result.get("replay_identity_sha256") != replay_identity_sha256
            or result.get("association") != association):
        raise ValueError("cached replay references another input, program, or episode")
    if (not isinstance(result.get("rows"), list) or not isinstance(result.get("counters"), dict)
            or result["counters"].get("paired_rows") != len(result["rows"])):
        raise ValueError("cached replay row count/schema mismatch")
    seen = set()
    for row in result["rows"]:
        key = (row.get("step"), row.get("target_id"))
        if (key in seen or row.get("split") != split
                or any(row.get(field) != value for field, value in association.items())
                or row.get("future_information_is_offline_only") is not True):
            raise ValueError("cached replay contains duplicate or mismatched rows")
        seen.add(key)
    return result


def process_episode(model, inputs, labels, *, split, seed, device="cuda"):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if getattr(model, "training", False):
        raise ValueError("offline replay requires model.eval()")
    association, states, label_by_step, pairs = validate_episode(inputs, labels, split)
    nav_fn = lambda batch: model("navigation", batch)
    rows, selections = [], []
    counters = {"eligible_states": 0, "selected_candidate_states": 0,
                "selected_without_natural_arrival": 0, "paired_rows": 0,
                "missing_shuffled_control": 0, "max_replay_error": 0.0}
    for state in states.values():
        if not state["eligible_decision"]:
            continue
        counters["eligible_states"] += 1
        selected = chosen_candidates(state, seed)
        counters["selected_candidate_states"] += len(selected)
        candidate_ids = state["nav_inputs"]["gmap_vpids"][0]
        selections.append({"step": state["step"], "target_ids": [candidate_ids[index] for index in selected]})
        selected = [index for index in selected if candidate_ids[index] in pairs
                    and pairs[candidate_ids[index]]["training_only"]["arrival_step"] > state["step"]]
        counters["selected_without_natural_arrival"] += len(chosen_candidates(state, seed)) - len(selected)
        # Verify every eligible baseline state, including those without an
        # arrival label, so missing labels cannot hide a model/state mismatch.
        frozen = clone_navigation(state["nav_inputs"], device)
        replay = replay_identity(nav_fn, frozen, state["base_logits"])
        counters["max_replay_error"] = max(counters["max_replay_error"], replay["max_absolute_error"])
        if not selected:
            continue
        for index in selected:
            target = candidate_ids[index]
            pair = pairs[target]
            if any(pair["association"][key] != association[key] for key in ("episode_id", "scan_id", "instr_id")):
                raise ValueError("arrival pair belongs to another episode")
            future = pair["training_only"]["arrival_feature"].float()
            old = state["nav_inputs"]["gmap_img_embeds"][0, index].float()
            if future.norm() < 1e-8 or old.norm() < 1e-8:
                raise ValueError("zero-norm representation invalidates norm control")
            p0, p1 = observable_features(state, index)
            source = state["candidate_evidence"][target]["sources"]
            latest = max(source, key=lambda value: value["last_step"])["feature"]
            noise_seed = int(object_sha256([seed, association["episode_id"], state["step"], target])[:15], 16)
            noise = torch.randn(old.shape, generator=torch.Generator().manual_seed(noise_seed))
            controls = {"arrival": future, "arrival_normmatched": future * (old.norm() / future.norm()),
                        "last_source": latest, "noise_normmatched": noise * (old.norm() / noise.norm())}
            distractors = [item for key, item in pairs.items() if key != target
                           and item["training_only"]["arrival_step"] > state["step"]]
            if distractors:
                # Same episode/scene, closest arrival lag; still an imperfect negative control.
                distractor = min(distractors, key=lambda item: (
                    abs(item["training_only"]["arrival_step"] - pair["training_only"]["arrival_step"]),
                    object_sha256([seed, item["association"]["target_id"]])))
                controls["shuffled_arrival"] = distractor["training_only"]["arrival_feature"].float()
            else:
                counters["missing_shuffled_control"] += 1
            interventions = {name: evaluate_replacement(nav_fn, frozen, replay["logits"], index, feature)
                             for name, feature in controls.items()}
            oracle = label_by_step[state["step"]]
            label = {"base_chosen": index == int(state["base_logits"].argmax())}
            for kind in ("teacher", "execution"):
                costs = oracle[kind + "_cost"]
                finite = torch.isfinite(costs)
                if not finite[index] or not finite.any():
                    raise ValueError("candidate lacks a finite oracle diagnostic cost")
                label[kind + "_regret"] = float(costs[index] - costs[finite].min())
                label[kind + "_optimal"] = index in oracle[kind + "_optimal_indices"]
                for outcome in interventions.values():
                    outcome[kind + "_argmax_optimal"] = outcome["argmax"] in oracle[kind + "_optimal_indices"]
            rows.append({"split": split, "scan_id": association["scan_id"], "instr_id": association["instr_id"],
                         "episode_id": association["episode_id"], "step": state["step"], "target_id": target,
                         "arrival_step": pair["training_only"]["arrival_step"], "arrival_kind": "natural",
                         "p0": p0, "p1": p1, "labels": label, "interventions": interventions,
                         "navigation_sha256": replay["navigation_sha256"],
                         "future_information_is_offline_only": True})
            counters["paired_rows"] += 1
    return {"association": association, "rows": rows, "counters": counters, "selections": selections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New replay directory, resumable by identity")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--split", choices=("train_fit", "train_dev"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    collection = json.loads((args.collection / "COLLECTION.json").read_text())
    collected = collection["identity"]
    if collection.get("schema") != SCHEMA or collection.get("identity_sha256") != object_sha256(collected):
        raise ValueError("collection identity checksum/schema mismatch")
    if collected["split"] != args.split:
        raise ValueError("collection split differs from requested split")
    if collected["usage"] != ("training_diagnostics" if args.split == "train_fit" else "analysis_only"):
        raise ValueError("collection usage differs from split")
    selected_names = {"episode-" + object_sha256([row["scan"], row["instr_id"]]) for row in collected["selection"]}
    episodes = sorted(args.collection.glob("episode-*"))
    if not episodes or len(selected_names) != len(collected["selection"]) or {path.name for path in episodes} != selected_names:
        raise ValueError("collection is incomplete, duplicated, or contains unexpected episodes")
    if args.backup:
        validate_backup_root(args.backup)
        args.backup.mkdir(parents=True, exist_ok=True)
    model, runtime = load_model(args.config)
    validate_runtime(collected, runtime)
    identity = {"schema": "duet_counterfactual_replay_v1", "collection_sha256": collection["identity_sha256"],
                "runtime": runtime, "seed": args.seed,
                "implementation": {name: file_sha256(ROOT / name) for name in (
                    "scripts/replay_diagnostics.py", "src/vln_improve/counterfactual.py", "src/vln_improve/diagnostics.py")},
                "selection": "baseline_chosen_nonstop_plus_one_hash_selected_legal_candidate_before_future_availability",
                "controls": ["arrival", "arrival_normmatched", "last_source", "noise_normmatched", "shuffled_arrival_same_episode_future_nearest_lag"]}
    identity_sha256 = object_sha256(identity)
    args.output.mkdir(parents=True, exist_ok=True)
    for root in (args.output, args.backup):
        if root is None: continue
        marker = root / "REPLAY.json"
        if marker.is_symlink() or (marker.exists() and json.loads(marker.read_text()) != identity):
            raise ValueError("replay identity changed; use a fresh output directory")
        if not marker.exists():
            if any(root.glob("episode-*.json")):
                raise ValueError("existing replay results lack a REPLAY identity")
            atomic_json(marker, identity)
    started = time.monotonic()
    all_rows, episode_counters = [], []
    torch.cuda.reset_peak_memory_stats()
    for episode in episodes:
        inputs, labels, manifest = load_episode(episode, expected_identity_sha256=collection["identity_sha256"])
        validate_episode(inputs, labels, args.split)
        manifest_sha256 = file_sha256(episode / "manifest.json")
        target = args.output / (episode.name + ".json")
        result = None
        for root in (args.backup, args.output):
            if root is None: continue
            path = root / target.name
            if path.exists() or path.is_symlink():
                loaded = read_result(path, input_manifest_sha256=manifest_sha256,
                                     replay_identity_sha256=identity_sha256,
                                     association=inputs["association"], split=args.split)
                if result is not None and result != loaded:
                    raise ValueError("local and cloud replay results disagree")
                result = loaded
        if result is None:
            result = process_episode(model, inputs, labels, split=args.split, seed=args.seed)
            result = seal_result(result, input_manifest_sha256=manifest_sha256,
                                 replay_identity_sha256=identity_sha256)
        atomic_json(target, result)
        if args.backup:
            validate_backup_root(args.backup)
            atomic_json(args.backup / target.name, result)
            if file_sha256(target) != file_sha256(args.backup / target.name):
                raise ValueError("replay cloud read-back mismatch")
        all_rows.extend(result["rows"])
        episode_counters.append(result["counters"])
        print(json.dumps({"event": "episode_replay_committed", "episodes": len(episode_counters),
                          "rows": len(all_rows), "scan": manifest["scan_id"]}), flush=True)
    summary = {"status": "complete", "split": args.split, "episodes": len(episode_counters),
               "rows": len(all_rows), "scans": len({row["scan_id"] for row in all_rows}),
               "analysis_cohort": "all_selected_natural_pairs_retained_even_when_shuffled_control_is_missing",
               "replay_identity_sha256": identity_sha256,
               "max_identity_replay_error": max((c["max_replay_error"] for c in episode_counters), default=0),
               "selected_candidate_states": sum(c["selected_candidate_states"] for c in episode_counters),
               "selected_without_natural_arrival": sum(c["selected_without_natural_arrival"] for c in episode_counters),
               "missing_shuffled_control": sum(c["missing_shuffled_control"] for c in episode_counters),
               "elapsed_seconds": time.monotonic() - started,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "scope": "offline_representation_perturbation_not_navigation_improvement",
               "limitations": ["natural-arrival selection bias", "same-episode shuffled control is not fully geometry matched",
                               "orientation-matched re-encoding is not implemented", "global-node replacement can be out of distribution"]}
    outputs = {"rows.jsonl": all_rows}
    for root in (args.output, args.backup):
        if root is None: continue
        if root == args.backup: validate_backup_root(root)
        for filename, rows in outputs.items():
            text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
            temporary = root / ("." + filename + ".tmp")
            with temporary.open("w") as stream:
                stream.write(text); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(root / filename)
        atomic_json(root / "summary.json", summary)
    if args.backup:
        for filename in (*outputs, "summary.json"):
            if file_sha256(args.output / filename) != file_sha256(args.backup / filename):
                raise ValueError("combined replay cloud read-back mismatch")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
