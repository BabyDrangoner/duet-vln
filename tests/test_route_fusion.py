import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import textwrap

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_route_fusion import EndpointPositions, load_graph_factory, read_collection
from vln_improve.diagnostics import SCHEMA
from vln_improve.protocol import object_sha256
from vln_improve.route_fusion import (
    MODES, SHIFT_INVARIANT, RevealedRoutes, analyze_episode, fuse_scores,
    inspect_fusion, route_context, summarize_houses,
)

GRAPH_FILE = ROOT / "third_party/VLN-DUET/map_nav_src/models/graph_utils.py"
spec = importlib.util.spec_from_file_location("r0_test_graph", GRAPH_FILE)
upstream_graph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream_graph)
GraphMap = upstream_graph.GraphMap

POSITIONS = {"A": [0., 0., 0.], "B": [2., 0., 0.], "C": [2., 2., 0.],
             "D": [0., 2., 0.], "X": [-1., 0., 0.], "Y": [3., 2., 0.], "Z": [0., 3., 0.]}


def upstream_fuse(state):
    # Execute the pinned upstream fusion block on CPU, independent of the new implementation.
    source = (ROOT / "third_party/VLN-DUET/map_nav_src/models/vilmodel.py").read_text()
    block = source.split("        # fusion\n", 1)[1].split("        # object grounding logits", 1)[0]
    env = {"torch": torch, "batch_size": 1,
           "global_logits": state["base_global_logits"][None].clone(),
           "local_logits": state["base_local_logits"][None].clone(),
           **{key: state["nav_inputs"][key] for key in ("gmap_vpids", "gmap_visited_masks", "vp_cand_vpids")}}
    exec(textwrap.dedent(block), env)
    return env["fused_logits"][0]


def make_state(graph, step, current, candidates, chosen, *, local_values=None, global_values=None, eligible=True,
               positions=None):
    positions = POSITIONS if positions is None else positions
    graph.update_graph({"viewpoint": current, "position": positions[current],
                        "candidate": [{"viewpointId": v, "position": positions[v]} for v in candidates]})
    graph.node_step_ids[current] = step + 1
    visited = [v for v in graph.node_positions if graph.graph.visited(v)]
    ids = [None] + visited + [v for v in graph.node_positions if not graph.graph.visited(v)]
    legal = torch.tensor([v is None or v not in visited for v in ids])
    pairs = torch.zeros(1, len(ids), len(ids))
    for i in range(1, len(ids)):
        for j in range(i + 1, len(ids)):
            pairs[0, i, j] = pairs[0, j, i] = graph.graph.distance(ids[i], ids[j])
    local_ids = [None] + candidates
    global_scores = torch.tensor([0.0 if legal[i] else -torch.inf for i in range(len(ids))])
    global_scores[ids.index(chosen)] = 10.0
    for key, value in (global_values or {}).items():
        global_scores[ids.index(key)] = value
    local_scores = torch.tensor([(local_values or {}).get(v, 0.) for v in local_ids])
    state = {"step": step, "current_viewpoint": current, "eligible_decision": eligible,
             "valid_mask": legal, "base_global_logits": global_scores, "base_local_logits": local_scores,
             "nav_inputs": {"gmap_vpids": [ids], "vp_cand_vpids": [local_ids],
                            "gmap_masks": torch.ones(1, len(ids), dtype=torch.bool),
                            "gmap_visited_masks": ~legal[None], "gmap_pair_dists": pairs,
                            "gmap_step_ids": torch.tensor([[graph.node_step_ids.get(v, 0) for v in ids]]),
                            "vp_nav_masks": torch.ones(1, len(local_ids), dtype=torch.bool)}}
    state["base_logits"] = upstream_fuse(state)
    state["baseline_argmax"] = int(state["base_logits"].argmax())
    return state


@pytest.fixture
def episode():
    graph = GraphMap("A")
    observations = [("A", ["B", "D", "X"], "B"), ("B", ["A", "C"], "C"),
                    ("C", ["B", "D", "Y"], "D"), ("D", ["A", "C", "Z"], "Z"),
                    ("Z", ["D"], None)]
    states = {}
    for step, (current, candidates, chosen) in enumerate(observations):
        options = {"local_values": {"A": -2., "C": -4.},
                   "global_values": {"X": 6.8, "Y": 2., "Z": 1.}} if current == "D" else {}
        states[step] = make_state(graph, step, current, candidates, chosen, **options)
    oracle = {}
    for step, state in states.items():
        costs = torch.full_like(state["base_logits"], torch.inf, dtype=torch.float64)
        costs[state["valid_mask"]] = 5.
        costs[0] = torch.inf
        optimum = state["nav_inputs"]["gmap_vpids"][0].index("X") if step == 3 else state["baseline_argmax"]
        costs[optimum] = 0.
        oracle[step] = {f"{kind}_{field}": value for kind in ("teacher", "execution")
                        for field, value in (("cost", costs.clone()), ("optimal_indices", [optimum]))}
    trajectory = {"instr_id": "instruction", "path": [["A"], ["B"], ["C"], ["D"], ["Z"]]}
    return states, oracle, trajectory, {"scan_id": "house", "instr_id": "instruction", "episode_id": "episode"}


def prefix_at(states, last):
    routes = RevealedRoutes(POSITIONS, GraphMap)
    for step in range(last + 1):
        routes.advance(states[step])
    return routes


def test_exact_upstream_fusion_firsthop_controls_and_common_shift(episode):
    states, _, _, _ = episode
    routes = prefix_at(states, 3)
    state = states[3]
    before = {key: state[key].clone() for key in ("base_logits", "base_global_logits", "base_local_logits")}
    result = inspect_fusion(state, routes)
    context = route_context(state, routes)
    ids = context["ids"]
    assert routes.route("X") == ["D", "A", "X"]
    assert routes.route("Y") == ["D", "C", "Y"]
    assert result["return_neighbors"] == result["remote_firsthop_groups"] == 2
    assert result["max_fused_reconstruction_error"] == 0
    assert ids[result["chosen"]["original_shared_sum"]] == "Z"
    assert all(ids[result["chosen"][mode]] == "X" for mode in MODES[1:])
    assert result["shift_changes"]["1"]["original_shared_sum"] is True
    assert all(not result["shift_changes"][shift][mode] for shift in ("-1", "1") for mode in SHIFT_INVARIANT)
    hspr = fuse_scores(state, context, "hspr_style_remote_double_global")
    assert hspr[ids.index("X")] == 2 * state["base_global_logits"][ids.index("X")]
    assert hspr[0] == state["base_logits"][0]
    assert hspr[ids.index("Z")] == state["base_logits"][ids.index("Z")]
    for key, value in before.items():
        assert torch.equal(value, state[key])


def test_single_return_neighbor_route_mean_and_logsumexp_equal_original(episode):
    states, _, _, _ = episode
    routes = prefix_at(states, 1)
    context = route_context(states[1], routes)
    assert len(context["backward"]) == 1
    assert context["remote"]
    expected = fuse_scores(states[1], context, "original_shared_sum")
    for mode in SHIFT_INVARIANT:
        assert torch.equal(expected, fuse_scores(states[1], context, mode))
    with pytest.raises(ValueError, match="not in the revealed"):
        routes.route("Z")  # Coordinates existing in the file do not reveal a node/edge.


@pytest.mark.parametrize("damage,match", [("distance", "pair distances"), ("visited", "visited set"),
                                           ("steps", "step IDs"), ("order", "node order")])
def test_graph_parity_rejects_changes_without_tolerance_relaxation(episode, damage, match):
    states, _, _, _ = copy.deepcopy(episode)
    state = states[0]
    if damage == "distance":
        state["nav_inputs"]["gmap_pair_dists"][0, 1, 2] += 1e-5
    elif damage == "visited":
        state["nav_inputs"]["gmap_visited_masks"][0, 1] = False
    elif damage == "steps":
        state["nav_inputs"]["gmap_step_ids"][0, 1] += 1
    else:
        state["nav_inputs"]["gmap_vpids"][0][1:] = reversed(state["nav_inputs"]["gmap_vpids"][0][1:])
    with pytest.raises(ValueError, match=match):
        prefix_at(states, 0)


def test_remote_with_no_return_neighbor_and_changed_baseline_fail(episode):
    states, _, _, _ = copy.deepcopy(episode)
    routes = prefix_at(states, 3)
    bad = copy.deepcopy(states[3])
    bad["nav_inputs"]["vp_cand_vpids"] = [[None, "Z"]]
    bad["nav_inputs"]["vp_nav_masks"] = torch.ones(1, 2, dtype=torch.bool)
    bad["base_local_logits"] = torch.zeros(2)
    with pytest.raises(ValueError, match="K=0"):
        inspect_fusion(bad, routes)
    states[3]["base_logits"][0] += 1e-7
    with pytest.raises(ValueError, match="fused reconstruction differs"):
        inspect_fusion(states[3], routes)


def test_complete_episode_checks_execution_and_aggregates_whole_states(episode):
    states, oracle, trajectory, association = episode
    rows = analyze_episode(states, oracle, trajectory, POSITIONS, GraphMap, association)
    result = summarize_houses(rows)
    coverage = result["overall"]["coverage"]
    assert coverage["eligible_states"] == 5
    assert coverage["actual_next_paths_checked"] == 4
    assert coverage["remote_firsthop_groups_ge2"] == 1
    firsthop = result["overall"]["controls"]["executed_firsthop_local"]
    assert firsthop["argmax_changes"] == 1
    assert firsthop["teacher"]["fixes"] == 1
    assert firsthop["teacher"]["damages"] == 0
    assert firsthop["teacher"]["paired_mean_regret_delta"] == -1.0
    assert result["house_macro"]["executed_firsthop_local"]["teacher"]["houses_with_finite_pairs"] == 1
    bad = copy.deepcopy(trajectory)
    bad["path"][3] = ["B", "A", "D"]
    with pytest.raises(ValueError, match="manifest route disagree"):
        analyze_episode(states, oracle, bad, POSITIONS, GraphMap, association)


def test_nonfinite_stop_cost_has_no_fabricated_zero_regret(episode):
    states, oracle, trajectory, association = copy.deepcopy(episode)
    for kind in ("teacher", "execution"):
        oracle[4][kind + "_cost"][0] = torch.inf
        oracle[4][kind + "_optimal_indices"] = [states[4]["nav_inputs"]["gmap_vpids"][0].index("X")]
    rows = analyze_episode(states, oracle, trajectory, POSITIONS, GraphMap, association)
    result = summarize_houses(rows)["overall"]["controls"]["original_shared_sum"]["teacher"]
    assert result["nonfinite_selected_cost"] == 1
    assert result["finite_regret_count"] == 4
    assert rows[-1]["oracle"]["teacher"]["original_shared_sum"]["regret"] is None


def test_stop_only_forced_ending_is_reported_separately():
    state = make_state(GraphMap("A"), 0, "A", [], None, eligible=False)
    oracle = {0: {f"{kind}_{field}": value for kind in ("teacher", "execution")
                  for field, value in (("cost", torch.tensor([0., torch.inf], dtype=torch.float64)),
                                       ("optimal_indices", [0]))}}
    association = {"scan_id": "house", "instr_id": "i", "episode_id": "e"}
    rows = analyze_episode({0: state}, oracle, {"instr_id": "i", "path": [["A"]]}, POSITIONS, GraphMap, association)
    coverage = summarize_houses(rows)["overall"]["coverage"]
    assert coverage["recorded_stop_only_states"] == 1
    assert coverage["eligible_states"] == 0


def test_collection_selection_identity_is_checked_and_val_is_refused(tmp_path):
    selected = [{"scan": "house", "instr_id": "i"}]
    identity = {"schema": "duet_diagnostic_collection_v1", "split": "train_fit", "usage": "training_diagnostics",
                "model": {"batch_size": 1, "enc_full_graph": True, "fusion": "dynamic"},
                "selection": selected, "selection_sha256": object_sha256(selected)}
    sha = object_sha256(identity)
    (tmp_path / "COLLECTION.json").write_text(json.dumps({"schema": SCHEMA, "identity": identity, "identity_sha256": sha}))
    with pytest.raises(ValueError, match="incomplete"):
        read_collection(tmp_path, "train_fit", sha)
    (tmp_path / ("episode-" + object_sha256(["house", "i"]))).mkdir()
    assert len(read_collection(tmp_path, "train_fit", sha)[2]) == 1
    with pytest.raises(ValueError, match="only accepts"):
        read_collection(tmp_path, "val_unseen", sha)
    with pytest.raises(ValueError, match="registered R0 protocol"):
        read_collection(tmp_path, "train_fit", "a" * 64)


def test_endpoint_positions_never_decode_unrevealed_positions(tmp_path):
    pose = [0.] * 16
    pose[3], pose[7], pose[11] = 1., 2., 3.
    path = tmp_path / "connectivity.json"
    path.write_text(json.dumps([{"image_id": "A", "included": True, "pose": pose},
                                {"image_id": "future", "included": True, "pose": "invalid-unused"}]))
    positions = EndpointPositions(path)
    assert positions["A"] == [1., 2., 3.]
    assert positions.used == {"A"}
    with pytest.raises(ValueError, match="endpoint pose"):
        positions["future"]


def test_endpoint_float32_rounding_then_double_arithmetic_reproduces_exact_graph(tmp_path):
    raw_positions = {"A": [100.1234567, 200.7654321, 1.00001],
                     "B": [100.4567891, 201.1234567, 1.77777777],
                     "C": [99.9876543, 202.11111111, .3333333]}
    # Independently reproduce the C++ float -> double conversion with IEEE bytes.
    simulator_positions = {key: [float(struct.unpack("f", struct.pack("f", value))[0]) for value in position]
                           for key, position in raw_positions.items()}
    rows = []
    for key, position in raw_positions.items():
        pose = [0.] * 16
        for index, value in zip((3, 7, 11), position):
            pose[index] = value
        rows.append({"image_id": key, "included": True, "pose": pose})
    filename = tmp_path / "connectivity.json"
    filename.write_text(json.dumps(rows))
    endpoints = EndpointPositions(filename)
    for key, values in simulator_positions.items():
        assert endpoints[key] == values
        assert all(type(value) is float for value in endpoints[key])
    saved = make_state(GraphMap("A"), 0, "A", ["B", "C"], "B", positions=simulator_positions)
    # Raw JSON doubles are a different coordinate source, despite identical IDs.
    with pytest.raises(ValueError, match="pair distances differ"):
        RevealedRoutes(raw_positions, GraphMap).advance(saved)
    assert RevealedRoutes(endpoints, GraphMap).advance(saved) == 0.0


def test_upstream_code_hash_is_checked_before_loading(tmp_path):
    collection = {"identity": {"upstream_lock": {"files": {
        "map_nav_src/models/graph_utils.py": {"prepared": "0" * 64}}}}}
    with pytest.raises(ValueError, match="source differs"):
        load_graph_factory(ROOT / "third_party/VLN-DUET", collection)
