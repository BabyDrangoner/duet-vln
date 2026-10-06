#!/usr/bin/env python3
"""CPU-only audit of a real cached forced prefix's coordinate conventions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-report", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True,
                        help="Existing train_fit paired cache root; its selected pair must be committed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a diagnostic")
    import numpy as np
    import torch
    from collect_endpoint_controls import resolve_selection
    from vln_improve.endpoint_pairs import validate_pair
    from vln_improve.endpoint_controls import walk_length
    from vln_improve.endpoint_pair_training import _json
    from vln_improve.protocol import file_sha256, resolve_config, object_sha256
    report = json.loads(args.controls_report.read_text())
    entry = resolve_selection(report, "train_fit", 512)[0]
    cfg = resolve_config(args.config, ROOT)
    collection = _json(args.pair_cache / "COLLECTION.json")
    identity = collection["identity"]
    if identity["split"] != "train_fit" or collection["identity_sha256"] != object_sha256(identity):
        raise ValueError("cached paired source is not valid train_fit")
    pair = entry["pair"]
    if pair not in identity["selection"]:
        raise ValueError("first common-pool pair is absent from the chosen paired cache")
    folder = args.pair_cache / ("pair-" + pair["selection_hash"])
    manifest = _json(folder / "manifest.json")
    if (_json(folder / "COMMITTED.json") != {"manifest_sha256": file_sha256(folder / "manifest.json")}
            or manifest["data_sha256"] != file_sha256(folder / "data.pt")
            or manifest["identity_sha256"] != collection["identity_sha256"]):
        raise ValueError("cached real pair is not completely committed")
    payload = torch.load(folder / "data.pt", map_location="cpu", weights_only=True)
    validate_pair(payload, collection["identity_sha256"], pair)
    original = payload["rollouts"]["A_then_B"]["A"]
    goal_index = next(i for i, state in enumerate(original["states"]) if state["viewpoint"] == pair["goal_vpids"][0])
    prefix_states = original["states"][:goal_index + 1]
    expected = entry["controls"]["A"]["positive_history"]["observed_vpids"]
    if [s["viewpoint"] for s in prefix_states] != expected:
        raise ValueError("cached prefix is not the real C2 reference observation sequence")
    run = {"states": prefix_states, "trajectory": prefix_states[-1]["trajectory_prefix"],
           "actual_length_m": prefix_states[-1]["prefix_length_m"]}
    graph_path = Path(cfg["dataset_root"]) / "R2R/connectivity" / (pair["scan"] + "_connectivity.json")
    if file_sha256(graph_path) != report["identity"]["connectivity_files"][pair["scan"]]:
        raise ValueError("official graph changed since pool preparation")
    sys.path.insert(0, str(ROOT / "third_party/VLN-DUET/map_nav_src"))
    from models.graph_utils import calc_position_distance
    from utils.data import load_nav_graphs
    graph = load_nav_graphs(str(graph_path.parent), [pair["scan"]])[pair["scan"]]
    positions = {state["viewpoint"]: state["position"] for state in run["states"]}
    flattened = sum(run["trajectory"], [])
    edges = [{"source": a, "target": b, "simulator_position_length_m": float(calc_position_distance(positions[a], positions[b])),
              "official_connectivity_length_m": graph[a][b]["weight"]} for a, b in zip(flattened, flattened[1:])]
    coordinates = [{"viewpoint": v, "observed": p, "connectivity": graph.nodes[v]["position"].tolist(),
                    "connectivity_rounded_float32": graph.nodes[v]["position"].astype(np.float32).astype(np.float64).tolist()}
                   for v, p in positions.items()]
    official = walk_length(run["trajectory"], graph)
    execution_edges = sum(e["simulator_position_length_m"] for e in edges)
    result = {"schema": "endpoint_control_length_diagnostic_v1", "split": "train_fit",
        "pair": pair["selection_hash"], "instr_id": pair["instr_ids"][0], "context": "A:reference",
        "real_source": "committed A_then_B/A forced rollout, truncated at first actual observation of goal A",
        "gpu_calls": 0,
        "actual_trajectory": run["trajectory"], "states": len(run["states"]),
        "frozen_forced_rollout_length_m": run["actual_length_m"], "simulator_position_edge_sum_m": execution_edges,
        "official_connectivity_edge_sum_m": official,
        "frozen_minus_official_m": run["actual_length_m"] - official,
        "frozen_minus_simulator_edge_sum_m": run["actual_length_m"] - execution_edges,
        "all_observed_coordinates_equal_float32_connectivity": all(x["observed"] == x["connectivity_rounded_float32"] for x in coordinates),
        "coordinate_max_abs_difference_from_connectivity": max(abs(x-y) for c in coordinates for x,y in zip(c["observed"], c["connectivity"])),
        "edges": edges, "coordinates": coordinates,
        "prefixes": [{"viewpoint": s["viewpoint"], "frozen_execution_prefix_m": s["prefix_length_m"],
                      "official_prefix_m": walk_length(s["trajectory_prefix"], graph)} for s in run["states"]],
        "source": {"controls_report_sha256": file_sha256(args.controls_report),
                   "pair_identity_sha256": collection["identity_sha256"],
                   "pair_manifest_sha256": file_sha256(folder / "manifest.json"),
                   "connectivity_file_sha256": file_sha256(graph_path),
                   "frozen_forced_rollout_sha256": file_sha256(ROOT / "src/vln_improve/endpoint_pairs.py")},
        "changes_to_policy_or_frozen_collector": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("frozen_forced_rollout_length_m", "simulator_position_edge_sum_m",
        "official_connectivity_edge_sum_m", "frozen_minus_official_m", "frozen_minus_simulator_edge_sum_m",
        "all_observed_coordinates_equal_float32_connectivity", "coordinate_max_abs_difference_from_connectivity")}, indent=2))


if __name__ == "__main__":
    main()
