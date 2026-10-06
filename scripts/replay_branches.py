#!/usr/bin/env python3
"""Complete the fixed D1 candidate sample with independent branch observations.

This is privileged offline analysis. Historical states, original trajectories,
and the candidate-selection rule remain immutable. It is not a policy rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from replay_diagnostics import chosen_candidates, load_model, read_result, seal_result, validate_episode, validate_runtime
from vln_improve.branch_observations import MatterportBranchEnvironment, PrefixGraph, angle_close, matched_donor
from vln_improve.counterfactual import clone_navigation, evaluate_replacement, navigation_hash, observable_features, replay_identity
from vln_improve.diagnostics import SCHEMA, load_episode
from vln_improve.pipeline import atomic_json, validate_backup_root
from vln_improve.protocol import file_sha256, object_sha256, resolve_config


COHORT = "simulated_branch_all_preselected_candidates"
CONTROLS = ["arrival", "arrival_normmatched", "arrival_headingmatched", "arrival_headingmatched_normmatched",
            "last_source", "noise_normmatched", "shuffled_arrival", "shuffled_normmatched", "shuffled_headingmatched"]


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _check_feature(value, reference):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != reference.shape
            or not torch.isfinite(value).all() or float(value.norm()) < 1e-8):
        raise ValueError("invalid/zero-norm branch representation")
    return value.detach().cpu().clone()


@torch.inference_mode()
def process_episode(model, provider, inputs, labels, *, split, seed=0, device="cuda", graph_factory=None):
    if type(seed) is not int or seed != 0:
        raise ValueError("the registered branch expansion fixes the original seed at zero")
    if model.training:
        raise ValueError("branch replay requires model.eval()")
    association, states, oracle_by_step, natural_pairs = validate_episode(inputs, labels, split)
    # Commit the whole selection before constructing any privileged observation.
    selections = [{"step": state["step"], "target_ids": [state["nav_inputs"]["gmap_vpids"][0][index]
                    for index in chosen_candidates(state, seed)]}
                  for state in states.values() if state["eligible_decision"]]
    rows, observations = [], []
    counters = {"eligible_states": 0, "selected_candidate_states": sum(len(row["target_ids"]) for row in selections),
                "paired_rows": 0, "selected_without_natural_arrival": 0, "missing_shuffled_control": 0,
                "natural_chosen_parity_checks": 0, "max_natural_feature_error": 0.0, "max_replay_error": 0.0,
                "branch_observations": 0, "historical_panorama_parity_checks": 0}
    graph = PrefixGraph(graph_factory)
    nav_fn = lambda batch: model("navigation", batch)
    for step, state in states.items():
        before = navigation_hash(state["nav_inputs"])
        historical = provider.historical(association["scan_id"], state)
        graph.advance(state, historical)
        counters["historical_panorama_parity_checks"] += 1
        if not state["eligible_decision"]:
            continue
        counters["eligible_states"] += 1
        selected = chosen_candidates(state, seed)
        ids = state["nav_inputs"]["gmap_vpids"][0]
        # Donor matching is also frozen before looking up any target panorama.
        donors = {index: matched_donor(state, index, seed) for index in selected}
        required = sorted(set(selected) | {match["index"] for match in donors.values() if match is not None})
        routes = {index: graph.route(ids[index]) for index in required}
        frozen = clone_navigation(state["nav_inputs"], device)
        replay = replay_identity(nav_fn, frozen, state["base_logits"])
        counters["max_replay_error"] = max(counters["max_replay_error"], replay["max_absolute_error"])
        branch = {}
        for index in required:
            result = provider.branch(association["scan_id"], state, routes[index])
            reference = state["nav_inputs"]["gmap_img_embeds"][0, index]
            result["feature"] = _check_feature(result["feature"], reference)
            result["headingmatched_feature"] = _check_feature(result["headingmatched_feature"], reference)
            if (not angle_close(result["heading"], routes[index]["arrival_heading"])
                    or abs(result["elevation"] - routes[index]["arrival_elevation"]) > 1e-6
                    or result["view_index"] != routes[index]["arrival_view_index"]
                    or not angle_close(result["headingmatched_heading"], state["heading"])
                    or abs(result["headingmatched_elevation"] - state["elevation"]) > 1e-6):
                raise ValueError("branch/orientation control violates the declared frame")
            branch[index] = result
            observations.append({"step": step, "target_id": ids[index], "role": "selected" if index in selected else "shuffled_donor",
                                 "route": routes[index], "feature": result["feature"].tolist(),
                                 "headingmatched_feature": result["headingmatched_feature"].tolist(),
                                 "feature_sha256": _tensor_hash(result["feature"]),
                                 "headingmatched_feature_sha256": _tensor_hash(result["headingmatched_feature"]),
                                 "target_panorama_tokens": result["target_panorama_tokens"], "target_degree": result["target_degree"],
                                 "privileged_offline_only": True})
            counters["branch_observations"] += 1
        baseline = int(state["base_logits"].argmax())
        if baseline in selected:
            target = ids[baseline]
            if step + 1 not in states or states[step + 1]["current_viewpoint"] != target:
                raise ValueError("eligible baseline action lacks the expected next historical arrival")
            arrival_state = states[step + 1]
            result = branch[baseline]
            if (not angle_close(result["heading"], arrival_state["heading"])
                    or abs(result["elevation"] - arrival_state["elevation"]) > 1e-6):
                raise ValueError("branch arrival heading differs from actual baseline arrival")
            destination_index = arrival_state["nav_inputs"]["gmap_vpids"][0].index(target)
            natural = arrival_state["nav_inputs"]["gmap_img_embeds"][0, destination_index]
            torch.testing.assert_close(result["feature"], natural, atol=1e-5, rtol=1e-5)
            counters["natural_chosen_parity_checks"] += 1
            counters["max_natural_feature_error"] = max(counters["max_natural_feature_error"], float((result["feature"] - natural).abs().max()))
        for index in selected:
            target, result = ids[index], branch[index]
            old = state["nav_inputs"]["gmap_img_embeds"][0, index].float()
            if float(old.norm()) < 1e-8:
                raise ValueError("zero-norm original representation invalidates the norm controls")
            natural_pair = natural_pairs.get(target)
            has_natural = natural_pair is not None and natural_pair["training_only"]["arrival_step"] > step
            counters["selected_without_natural_arrival"] += int(not has_natural)
            p0, p1 = observable_features(state, index)
            sources = state["candidate_evidence"][target]["sources"]
            noise_seed = int(object_sha256([seed, association["episode_id"], step, target])[:15], 16)
            noise = torch.randn(old.shape, generator=torch.Generator().manual_seed(noise_seed))
            norm = lambda feature: feature * (old.norm() / feature.norm())
            controls = {"arrival": result["feature"], "arrival_normmatched": norm(result["feature"]),
                        "arrival_headingmatched": result["headingmatched_feature"],
                        "arrival_headingmatched_normmatched": norm(result["headingmatched_feature"]),
                        "last_source": max(sources, key=lambda value: value["last_step"])["feature"],
                        "noise_normmatched": norm(noise)}
            donor = donors[index]
            if donor is None:
                counters["missing_shuffled_control"] += 1
            else:
                donor_feature = branch[donor["index"]]["feature"]
                controls["shuffled_arrival"] = donor_feature
                controls["shuffled_normmatched"] = norm(donor_feature)
                controls["shuffled_headingmatched"] = branch[donor["index"]]["headingmatched_feature"]
            interventions = {name: evaluate_replacement(nav_fn, frozen, replay["logits"], index, feature)
                             for name, feature in controls.items()}
            oracle = oracle_by_step[step]
            label = {"base_chosen": index == baseline}
            for kind in ("teacher", "execution"):
                costs = oracle[kind + "_cost"]
                finite = torch.isfinite(costs)
                if not finite[index] or not finite.any():
                    raise ValueError("selected branch candidate lacks a finite oracle cost")
                label[kind + "_regret"] = float(costs[index] - costs[finite].min())
                label[kind + "_optimal"] = index in oracle[kind + "_optimal_indices"]
                for outcome in interventions.values():
                    outcome[kind + "_argmax_optimal"] = outcome["argmax"] in oracle[kind + "_optimal_indices"]
            rows.append({**association, "step": step, "target_id": target, "split": split,
                         "analysis_cohort": COHORT, "arrival_kind": "simulated_branch",
                         "has_natural_arrival": has_natural, "branch_route": routes[index],
                         "shuffled_donor": donor, "p0": p0, "p1": p1, "labels": label,
                         "interventions": interventions, "navigation_sha256": replay["navigation_sha256"],
                         "branch_feature_sha256": _tensor_hash(result["feature"]),
                         "future_information_is_offline_only": True})
            counters["paired_rows"] += 1
        if navigation_hash(state["nav_inputs"]) != before:
            raise ValueError("branch collection changed historical navigation inputs")
    if counters["paired_rows"] != counters["selected_candidate_states"]:
        raise ValueError("branch expansion failed to retain every preselected candidate")
    return {"association": association, "analysis_cohort": COHORT, "rows": rows, "counters": counters,
            "selections": selections, "branch_observations": observations}


def read_branch_result(path, **expected):
    result = read_result(path, **expected)
    if (result.get("analysis_cohort") != COHORT
            or result["counters"].get("selected_candidate_states") != len(result["rows"])
            or any(row.get("arrival_kind") != "simulated_branch" or row.get("analysis_cohort") != COHORT
                   or "arrival_step" in row for row in result["rows"])):
        raise ValueError("cached branch result has an incompatible cohort")
    return result


def _atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--split", choices=("train_fit", "train_dev"), required=True)
    parser.add_argument("--seed", type=int, choices=(0,), default=0)
    args = parser.parse_args(argv)
    marker = args.collection / "COLLECTION.json"
    if marker.is_symlink():
        raise ValueError("collection identity must not be symlinked")
    collection = json.loads(marker.read_text())
    collected = collection["identity"]
    if (collection.get("schema") != SCHEMA or collection.get("identity_sha256") != object_sha256(collected)
            or collected.get("schema") != "duet_diagnostic_collection_v1" or collected.get("split") != args.split
            or collected.get("usage") != ("training_diagnostics" if args.split == "train_fit" else "analysis_only")):
        raise ValueError("collection identity/split/usage mismatch")
    expected_names = {"episode-" + object_sha256([row["scan"], row["instr_id"]]) for row in collected["selection"]}
    episodes = sorted(args.collection.glob("episode-*"))
    if not episodes or len(expected_names) != len(collected["selection"]) or {path.name for path in episodes} != expected_names:
        raise ValueError("branch expansion requires the complete original selected episode set")
    if args.backup:
        if args.backup.resolve().is_relative_to(args.output.resolve()) or args.output.resolve().is_relative_to(args.backup.resolve()):
            raise ValueError("branch output and cloud backup must be separate")
        validate_backup_root(args.backup)
    config = resolve_config(args.config, ROOT)
    dataset = Path(config["dataset_root"])
    feature_file = dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
    connectivity = dataset / "R2R/connectivity"
    if file_sha256(feature_file) != collected["feature_sha256"]:
        raise ValueError("branch feature file differs from the collected baseline")
    if object_sha256({path.name: file_sha256(path) for path in sorted(connectivity.glob("*_connectivity.json"))}) != collected["connectivity_sha256"]:
        raise ValueError("branch connectivity differs from the collected baseline")
    model, runtime = load_model(args.config)
    validate_runtime(collected, runtime)
    import MatterSim
    identity = {"schema": "duet_simulated_branch_replay_v1", "collection_sha256": collection["identity_sha256"],
                "runtime": runtime, "simulator_sha256": file_sha256(Path(MatterSim.__file__)), "seed": args.seed,
                "analysis_cohort": COHORT, "selection": "unchanged_chosen_candidates_seed0_before_privileged_observation",
                "route": "exact_prefix_graph_floyd_path_with_private_legal_simulator_walk",
                "orientation": "primary_last_edge_pointId_and_control_historical_heading_elevation",
                "shuffled": "same_state_distance_then_source_count_then_hash_donor_with_own_arrival_and_historical_frames",
                "controls": CONTROLS, "features_include_new_local_geometry": True,
                "implementation": {name: file_sha256(ROOT / name) for name in (
                    "scripts/replay_branches.py", "src/vln_improve/branch_observations.py", "scripts/replay_diagnostics.py",
                    "src/vln_improve/counterfactual.py", "src/vln_improve/diagnostics.py")}}
    identity_sha = object_sha256(identity)
    for root in (args.output, args.backup):
        if root is None:
            continue
        root.mkdir(parents=True, exist_ok=True)
        path = root / "REPLAY.json"
        if path.is_symlink() or (path.exists() and json.loads(path.read_text()) != identity):
            raise ValueError("branch replay identity changed; use a new output directory")
        if not path.exists():
            if any(root.glob("episode-*.json")):
                raise ValueError("branch results lack a REPLAY identity")
            atomic_json(path, identity)
    provider = MatterportBranchEnvironment(connectivity, feature_file, model,
                 image_feat_size=config["model"]["image_feat_size"], angle_feat_size=config["model"]["angle_feat_size"])
    all_rows, counters = [], []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for episode in episodes:
        inputs, labels, manifest = load_episode(episode, expected_identity_sha256=collection["identity_sha256"])
        if inputs["association"]["episode_id"] != episode.name or manifest["episode_name"] != episode.name:
            raise ValueError("episode directory name does not match its verified association")
        validate_episode(inputs, labels, args.split)
        input_sha = file_sha256(episode / "manifest.json")
        expected = {"input_manifest_sha256": input_sha, "replay_identity_sha256": identity_sha,
                    "association": inputs["association"], "split": args.split}
        filename = episode.name + ".json"
        result = None
        for root in (args.backup, args.output):
            if root is None:
                continue
            path = root / filename
            if path.exists() or path.is_symlink():
                loaded = read_branch_result(path, **expected)
                if result is not None and loaded != result:
                    raise ValueError("local/cloud branch results disagree")
                result = loaded
        if result is None:
            result = process_episode(model, provider, inputs, labels, split=args.split, seed=args.seed)
            result = seal_result(result, input_manifest_sha256=input_sha, replay_identity_sha256=identity_sha)
        for root in (args.output, args.backup):
            if root is None:
                continue
            if root == args.backup:
                validate_backup_root(root)
            path = root / filename
            if not path.exists():
                atomic_json(path, result)
            if read_branch_result(path, **expected) != result:
                raise ValueError("branch episode backup read-back differs")
        all_rows.extend(result["rows"])
        counters.append(result["counters"])
        print(json.dumps({"episode": episode.name, "rows": len(result["rows"]), "completed_episodes": len(counters)}), flush=True)
    summary = {"schema": "duet_branch_summary_v1", "status": "complete", "split": args.split,
               "usage": collected["usage"], "analysis_cohort": COHORT, "replay_identity_sha256": identity_sha,
               "rows": len(all_rows), "episodes": len(episodes), "scans": len({row["scan_id"] for row in all_rows}),
               "elapsed_seconds": time.monotonic() - started, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "counters": {key: (max(c[key] for c in counters) if key.startswith("max_") else sum(c[key] for c in counters))
                            for key in counters[0]},
               "scope": "privileged_offline_branch_observations_not_navigation_improvement",
               "limitations": ["fixed seen-distribution baseline histories", "target panorama includes newly revealed local geometry",
                               "global-node replacement can be out of distribution", "donors are approximately geometry matched",
                               "Drive mount read-back is not an independent service persistence receipt"]}
    text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in all_rows)
    for root in (args.output, args.backup):
        if root is None:
            continue
        if root == args.backup:
            validate_backup_root(root)
        _atomic_text(root / "rows.jsonl", text)
        atomic_json(root / "summary.json", summary)
    if args.backup:
        for filename in ("rows.jsonl", "summary.json"):
            if file_sha256(args.output / filename) != file_sha256(args.backup / filename):
                raise ValueError("combined branch cloud read-back mismatch")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
