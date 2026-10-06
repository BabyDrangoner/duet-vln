"""Structural replay checks; no feature datasets, model backbone, or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("natural_subset_replay", ROOT / "scripts/replay_endpoint_natural_subset.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
graph_utils = replay.load_module(ROOT / "third_party/VLN-DUET/map_nav_src/models/graph_utils.py", "natural_test_graph")
eval_utils = replay.load_module(ROOT / "third_party/VLN-DUET/map_nav_src/r2r/eval_utils.py", "natural_test_metrics")


def make_run():
    positions = {"a": [0., 0., 0.], "b": [1., 0., 0.], "c": [0., 1., 0.],
                 "d": [1., 1., 0.], "frontier": [2., 0., 0.]}
    candidates = {"a": ["b", "c"], "b": ["a", "frontier", "d"],
                  "c": ["a", "d"], "d": ["b", "c", "frontier"]}
    order = ["a", "b", "c", "d"]
    graph = graph_utils.GraphMap("a")
    states, prefix = [], [["a"]]
    for step, current in enumerate(order):
        graph.update_graph({"viewpoint": current, "position": positions[current],
                            "candidate": [{"viewpointId": v, "position": positions[v]} for v in candidates[current]]})
        graph.node_step_ids[current] = step + 1
        visited = [v for v in graph.node_positions if graph.graph.visited(v)]
        unseen = [v for v in graph.node_positions if not graph.graph.visited(v)]
        nodes = [None] + visited + unseen
        pair = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
        for i in range(1, len(nodes)):
            for j in range(i + 1, len(nodes)):
                pair[i, j] = pair[j, i] = graph.graph.distance(nodes[i], nodes[j])
        action = order[step + 1] if step + 1 < len(order) else None
        states.append({"step": step, "viewpoint": current, "position": positions[current],
                       "vp_cand_vpids": [None] + candidates[current], "gmap_vpids": nodes,
                       "masks": {"gmap_masks": [True] * len(nodes),
                                 "gmap_visited_masks": [False] + [v in visited for v in nodes[1:]]},
                       "gmap_step_ids": [graph.node_step_ids.get(v, 0) for v in nodes],
                       "gmap_pair_dists": pair.tolist(), "executed_action": action,
                       "baseline_argmax_index": nodes.index(action), "baseline_argmax_vpid": action,
                       "termination": {"step_limit": False, "argmax_stop": action is None, "no_candidates": False},
                       "trajectory_prefix": copy.deepcopy(prefix)})
        if action is not None:
            prefix.append(graph.graph.path(current, action))
    probabilities = torch.tensor([.1, .4, .2, .35], dtype=torch.float32)
    trajectory = prefix + [graph.graph.path("d", "b")]
    return {"states": states, "natural_stop_probability": probabilities,
            "trajectory": trajectory, "fallback_segment": trajectory[len(prefix):]}, graph.graph


def test_full_graph_and_reduced_graph_preserve_remote_move_and_equal_length_return():
    run, original = make_run()
    reduced, audit = replay.reconstruct_natural(run, graph_utils)
    assert run["states"][2]["trajectory_prefix"][-1] == ["a", "c"]
    assert "frontier" not in reduced._dis
    assert audit["baseline_selected_state"] == 1
    assert audit["checked_online_moves"] == 3
    for a in ("a", "b", "c", "d"):
        for b in ("a", "b", "c", "d"):
            assert reduced.distance(a, b) == original.distance(a, b)
            assert reduced.path(a, b) == original.path(a, b)
    # b and c are equally short d->a routes: original strict-< tie is retained.
    assert reduced.path("d", "a") == original.path("d", "a")


@pytest.mark.parametrize("damage", ["distance", "prefix", "mask", "step", "fallback", "termination"])
def test_changed_replay_state_or_baseline_is_rejected(damage):
    run, _ = make_run()
    if damage == "distance":
        ids = run["states"][1]["gmap_vpids"]
        run["states"][1]["gmap_pair_dists"][ids.index("a")][ids.index("b")] += .01
    elif damage == "prefix":
        run["states"][2]["trajectory_prefix"][-1] = ["c"]
    elif damage == "mask":
        run["states"][1]["masks"]["gmap_visited_masks"][1] = False
    elif damage == "step":
        run["states"][1]["gmap_step_ids"][1] += 1
    elif damage == "fallback":
        run["trajectory"][-1] = ["c"]
    elif damage == "termination":
        run["states"][-1]["termination"]["no_candidates"] = True
    with pytest.raises(ValueError):
        replay.reconstruct_natural(run, graph_utils)


def test_official_metrics_charge_return_and_use_original_gt_length():
    graph = nx.Graph()
    for a, b, weight in [("a", "b", 2.), ("b", "c", 2.), ("c", "a", 3.)]:
        graph.add_edge(a, b, weight=weight)
    distances = dict(nx.all_pairs_dijkstra_path_length(graph))
    # GT length 4 is intentionally longer than shortest start->goal 3.
    gt = ["a", "b", "c"]
    result = replay.navigation_metrics([["a"], ["b"], ["c"], ["b"], ["c"]], gt, graph, distances, eval_utils)
    assert result["success"] == 1.
    assert result["trajectory_lengths"] == 8.
    assert result["spl"] == .5
    assert result["action_steps"] == 4.
    assert result["nDTW"] < 1.
    with pytest.raises(ValueError, match="repeated/nonexistent"):
        replay.navigation_metrics([["a"], ["a"]], gt, graph, distances, eval_utils)


def test_single_state_head_primary_preserves_first_tie_and_reports_batch_difference():
    class Head(torch.nn.Module):
        def forward(self, x):
            result = x[:, 0].clone()
            if len(x) > 1:
                result[1] += .01
            return result
    features = torch.zeros(3, 1536)
    features[:, 0] = torch.tensor([2., 2., 1.])
    result = replay.score_head(Head(), features)
    assert result["selected_state"] == 0  # stable strict > / dict insertion order
    assert result["top1_margin"] == 0.
    assert result["batched_selected_state"] == 1
    assert result["batched_top1_stable"] is False  # report; do not exclude this episode
    assert result["primary_forward_shape"] == [1, 1536]


def test_connectivity_requires_complete_inventory_binding_and_actual_bytes(tmp_path):
    pose_a, pose_b = [0.] * 16, [0.] * 16
    pose_b[3] = 4.
    data = [{"image_id": "a", "pose": pose_a, "included": True, "unobstructed": [False, True]},
            {"image_id": "b", "pose": pose_b, "included": True, "unobstructed": [True, False]}]
    graph_file = tmp_path / "scan_connectivity.json"
    graph_file.write_text(json.dumps(data))
    inventory = {graph_file.name: replay.sha(graph_file), "unused_connectivity.json": "1" * 64}
    digest = replay.object_sha(inventory)
    manifest = {"schema": "duet_connectivity_subset_v1", "source_connectivity_sha256": digest,
                "all_files": inventory, "selected_scans": ["scan"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    graphs, distances, _ = replay.load_connectivity(tmp_path, {"scan"}, digest)
    assert distances["scan"]["a"]["b"] == 4.
    assert graphs["scan"].nodes["b"]["execution_position"] == [4., 0., 0.]
    graph_file.write_text(json.dumps(data) + " ")  # semantic JSON same; byte seal changed
    with pytest.raises(ValueError, match="file SHA"):
        replay.load_connectivity(tmp_path, {"scan"}, digest)


def test_scene_bootstrap_uses_full_fixed_inventory_in_every_arm():
    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_metrics import compare_results
    metadata = {"dataset": "r2r", "split": "train_dev", "feature_id": "frozen", "base_checkpoint_sha256": "1" * 64,
                "protocol_sha256": "2" * 64, "upstream_commit": "commit", "max_action_len": 15, "feedback": "argmax"}
    baseline = [{"instr_id": "a", "scan_id": "scene1", "success": 0., "spl": 0., "return_length_m": 0.},
                {"instr_id": "b", "scan_id": "scene2", "success": 1., "spl": .5, "return_length_m": 1.}]
    reports = {name: copy.deepcopy(baseline) for name in ("baseline", *replay.ARMS)}
    reports["C3"][0].update(success=1., spl=.5, return_length_m=2.)
    out = replay.compare_all(reports, metadata, compare_results)
    assert len(out) == 7
    assert out["C3-baseline"]["rescued_instructions"] == ["a"]
    assert out["C3-baseline"]["harmed_instructions"] == []
    assert out["C3-baseline"]["mean_return_length_delta_m"] == 1.


def _baseline_metrics():
    return {name: .5 for name in replay.EXACT_METRICS + replay.EXPONENTIAL_METRICS}


def test_one_ulp_exponential_roundoff_is_accepted_and_fully_recorded():
    expected = _baseline_metrics()
    expected["nDTW"] = expected["SDTW"] = .9679744878638769
    actual = copy.deepcopy(expected)
    actual["nDTW"] = actual["SDTW"] = float(np.nextafter(expected["nDTW"], -np.inf))
    audit = replay.validate_baseline_metrics(expected, actual)
    for name in ("nDTW", "SDTW"):
        diff = audit["exponential_metrics"][name]
        assert diff["expected"] == expected[name]
        assert diff["actual"] == actual[name]
        assert diff["difference"] == actual[name] - expected[name]
        assert diff["ulp_distance"] == 1
        assert diff["exact"] is False
    summary = replay.summarize_baseline_roundoff([{"baseline_metric_verification": audit}])
    assert summary["nonexact_instructions"] == 1
    assert summary["nonexact_metric_values"] == 2
    assert summary["max_ulp_distance"] == 1
    assert summary["by_metric"]["CLS"]["nonexact_instructions"] == 0


@pytest.mark.parametrize("metric", replay.EXPONENTIAL_METRICS)
def test_eight_ulp_boundary_is_accepted_but_nine_is_rejected(metric):
    expected, actual = _baseline_metrics(), _baseline_metrics()
    for _ in range(8):
        actual[metric] = float(np.nextafter(actual[metric], np.inf))
    assert replay.validate_baseline_metrics(expected, actual)["exponential_metrics"][metric]["ulp_distance"] == 8
    actual[metric] = float(np.nextafter(actual[metric], np.inf))
    with pytest.raises(ValueError, match="exceeds 8 ULP"):
        replay.validate_baseline_metrics(expected, actual)


@pytest.mark.parametrize("metric", replay.EXACT_METRICS)
def test_non_exponential_metrics_reject_even_one_ulp_difference(metric):
    expected, actual = _baseline_metrics(), _baseline_metrics()
    actual[metric] = float(np.nextafter(actual[metric], np.inf))
    with pytest.raises(ValueError, match="exact navigation metric differs"):
        replay.validate_baseline_metrics(expected, actual)
