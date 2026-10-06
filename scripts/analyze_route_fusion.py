#!/usr/bin/env python3
"""Fixed CPU-only route-fusion diagnostic on complete saved D1 training splits.

Never collects observations, runs a simulator, fits parameters, or reads val_unseen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from replay_diagnostics import validate_episode
from vln_improve.diagnostics import SCHEMA, atomic_json, load_episode
from vln_improve.protocol import file_sha256, object_sha256
from vln_improve.route_fusion import MODES, SHIFT_ATOL, analyze_episode, summarize_houses


class EndpointPositions:
    """Reproduce MatterSim's float32 coordinates, then its Python double exposure.

    NavGraph.cpp reads JSON pose through asFloat() into float posearr/glm::vec3.
    MatterSim.cpp promotes these values into Viewpoint's double x/y/z fields.
    DUET graph_utils subsequently subtracts the Python floats and computes the
    edge length. Rounding only the final edge distance does not reproduce this.
    Coordinates are decoded only for endpoints revealed by saved candidate lists.
    """
    def __init__(self, filename):
        data = json.loads(Path(filename).read_text())
        if not isinstance(data, list):
            raise ValueError("connectivity file must be a list")
        self.records = {row["image_id"]: row for row in data}
        if len(self.records) != len(data):
            raise ValueError("duplicate viewpoint in connectivity")
        self.used = set()

    def __getitem__(self, key):
        row = self.records[key]
        if row.get("included") is not True:
            raise ValueError("revealed endpoint is excluded in connectivity")
        pose = row["pose"]
        if not isinstance(pose, list) or len(pose) != 16:
            raise ValueError("invalid endpoint pose")
        position = [pose[i] for i in (3, 7, 11)]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in position):
            raise ValueError("nonfinite endpoint position")
        with np.errstate(over="ignore"):
            promoted = [float(np.float32(value)) for value in position]
        if any(not math.isfinite(value) for value in promoted):
            raise ValueError("endpoint position exceeds MatterSim float32 range")
        self.used.add(key)
        return promoted


def read_collection(root, split, expected_identity):
    root = Path(root)
    if split not in {"train_fit", "train_dev"}:
        raise ValueError("R0 only accepts train_fit and train_dev")
    marker = root / "COLLECTION.json"
    if marker.is_symlink():
        raise ValueError("collection identity cannot be a symlink")
    collection = json.loads(marker.read_text())
    identity = collection.get("identity")
    if (collection.get("schema") != SCHEMA or not isinstance(identity, dict)
            or identity.get("schema") != "duet_diagnostic_collection_v1"
            or collection.get("identity_sha256") != object_sha256(identity)
            or collection["identity_sha256"] != expected_identity):
        raise ValueError("collection identity differs from its content or registered R0 protocol")
    if identity.get("split") != split or identity.get("usage") != (
            "training_diagnostics" if split == "train_fit" else "analysis_only"):
        raise ValueError("collection split/usage differs from requested training partition")
    model = identity["model"]
    if (model.get("batch_size") != 1 or not model.get("enc_full_graph")
            or model.get("act_visited_nodes", False) or model.get("fusion") != "dynamic"):
        raise ValueError("R0 requires the original batch-one dynamic full-graph DUET policy")
    selected = identity.get("selection")
    if not isinstance(selected, list) or not selected or identity.get("selection_sha256") != object_sha256(selected):
        raise ValueError("selection identity is invalid")
    names = {"episode-" + object_sha256([r["scan"], r["instr_id"]]): r for r in selected}
    episodes = sorted(root.glob("episode-*"))
    if len(names) != len(selected) or {path.name for path in episodes} != set(names):
        raise ValueError("collection is incomplete or has duplicate/unexpected episodes")
    return collection, names, episodes


def load_graph_factory(upstream, collection):
    """Load only pure NumPy graph code, after matching the saved source lock."""
    upstream = Path(upstream)
    hashes = {}
    for name in ("map_nav_src/models/graph_utils.py", "map_nav_src/models/vilmodel.py"):
        expected = collection["identity"]["upstream_lock"]["files"][name]["prepared"]
        path = upstream / name
        if path.is_symlink() or file_sha256(path) != expected:
            raise ValueError(f"upstream source differs from collection: {name}")
        hashes[name] = expected
    spec = importlib.util.spec_from_file_location("r0_pinned_graph_utils", upstream / "map_nav_src/models/graph_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphMap, hashes


def analyze_collection(root, *, split, connectivity, upstream, expected_identity, connectivity_hashes):
    collection, names, episodes = read_collection(root, split, expected_identity)
    if object_sha256(connectivity_hashes) != collection["identity"]["connectivity_sha256"]:
        raise ValueError("connectivity content differs from the collection")
    graph_factory, source_hashes = load_graph_factory(upstream, collection)
    rows, manifest_hashes, positions = [], {}, {}
    for episode in episodes:
        inputs, labels, manifest = load_episode(episode, expected_identity_sha256=expected_identity)
        association, states, oracle, _ = validate_episode(inputs, labels, split)
        selected = names[episode.name]
        if association["scan_id"] != selected["scan"] or association["instr_id"] != selected["instr_id"]:
            raise ValueError("episode association differs from the fixed collection selection")
        scan = association["scan_id"]
        if scan not in positions:
            positions[scan] = EndpointPositions(Path(connectivity) / f"{scan}_connectivity.json")
        rows.extend(analyze_episode(states, oracle, manifest["trajectory"], positions[scan], graph_factory, association))
        manifest_hashes[episode.name] = file_sha256(episode / "manifest.json")
    return {"split": split, "collection_identity_sha256": expected_identity,
            "collection_file_sha256": file_sha256(Path(root) / "COLLECTION.json"),
            "episode_manifest_sha256": manifest_hashes, "upstream_source_sha256": source_hashes,
            "episodes": len(episodes), "houses": len(positions),
            "revealed_coordinate_endpoints_per_house": {scan: len(value.used) for scan, value in sorted(positions.items())},
            **summarize_houses(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-collection", type=Path, required=True)
    parser.add_argument("--dev-collection", type=Path, required=True)
    parser.add_argument("--connectivity", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/VLN-DUET")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.monotonic()
    if args.output.exists():
        raise FileExistsError("R0 report already exists; preserve it and choose a new explicit output path")
    protocol = json.loads(args.protocol.read_text())
    if (protocol.get("schema_version") != 1 or protocol.get("variants") != list(MODES)
            or protocol.get("fixed_shift_checks") != [-1.0, 1.0]
            or protocol.get("fitting") is not False
            or protocol.get("new_gpu_collection") is not False
            or protocol.get("new_official_validation_access") is not False):
        raise ValueError("registered R0 variants/scope differ from this fixed implementation")
    identities = protocol.get("collections", {})
    if set(identities) != {"train_fit", "train_dev"}:
        raise ValueError("R0 protocol requires the exact train_fit/train_dev collection identities")
    # Hash the same complete set used by collection, but never infer edges from it.
    connectivity_files = sorted(args.connectivity.glob("*_connectivity.json"))
    if not connectivity_files or any(path.is_symlink() for path in connectivity_files):
        raise ValueError("connectivity files are missing or symlinked")
    connectivity_hashes = {path.name: file_sha256(path) for path in connectivity_files}
    collections = {split: read_collection(directory, split, identities[split])[0]
                   for split, directory in (("train_fit", args.train_collection), ("train_dev", args.dev_collection))}
    scans = {split: {row["scan"] for row in value["identity"]["selection"]} for split, value in collections.items()}
    if scans["train_fit"] & scans["train_dev"]:
        raise ValueError("training and development collection houses overlap")
    torch.set_num_threads(1)
    results = {split: analyze_collection(directory, split=split, connectivity=args.connectivity,
                                         upstream=args.upstream, expected_identity=identities[split],
                                         connectivity_hashes=connectivity_hashes)
               for split, directory in (("train_fit", args.train_collection), ("train_dev", args.dev_collection))}
    report = {"schema": "duet_route_fusion_diagnostic_v1", "protocol_sha256": file_sha256(args.protocol),
              "controls": list(MODES), "collections": results,
              "resources": {"cpu_wall_seconds": time.monotonic() - start, "torch_version": str(torch.__version__)},
              "implementation_sha256": {name: file_sha256(ROOT / name) for name in (
                  "scripts/analyze_route_fusion.py", "src/vln_improve/route_fusion.py", "scripts/replay_diagnostics.py",
                  "src/vln_improve/diagnostics.py", "src/vln_improve/counterfactual.py", "src/vln_improve/protocol.py")},
              "interpretation": {
                  "scope": "complete fixed D1 states; scored controls on saved baseline states, not navigation SR/SPL",
                  "graphs": "only revealed prefix edges; connectivity supplies their endpoint positions, not unseen shortcuts",
                  "coordinates": "MatterSim JSON pose -> float32 glm::vec3 -> double Viewpoint/Python float; edge arithmetic then follows upstream graph_utils, with exact saved FP32 pair-distance parity",
                  "logits": "saved already fuse-weighted global/local logits; no additional fuse weight or new observations",
                  "hspr_control": "HSPR-style remote double-global score-rule adaptation only; not a reproduction of the full HSPR model",
                  "precision": "original reconstruction and diagnostic control choices FP32; common-shift algebra checked separately in FP64",
                  "shift": "all valid local logits plus -1/+1; original remote K*c, three local controls c, HSPR-style remote 0; STOP/local c",
                  "shift_atol": SHIFT_ATOL,
                  "numeric_ties": "the three invariant controls may only flip machine argmax within a 2e-12 FP64 tie; reported separately, otherwise rejected",
                  "causality": "common-shift checks show parameterization sensitivity, not proof of a natural causal policy error",
                  "costs": "existing teacher/execution oracle action costs; nonfinite STOP costs separate; no invented zero regret",
                  "aggregation": "one complete decision state per observation; all eligible states, with state-weighted and whole-house summaries; candidate occurrences are descriptive only",
                  "fitting": "none; no sampling, hyperparameter/threshold search, simulator, model inference, or validation access"}}
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output),
                      "eligible_states": {split: value["overall"]["coverage"]["eligible_states"] for split, value in results.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
