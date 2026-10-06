import ast
import copy
import json
from pathlib import Path
import shutil
import sys
from types import MethodType

import networkx as nx
import numpy as np
import pytest
import torch

from vln_improve.diagnostics import atomic_json
from vln_improve.endpoint_controls import (
    ControlStore, SCHEMA, collect_control_pair, load_control_cache, natural_rollout,
    validate_control_pair, walk_length, annotate_route_lengths, validate_route_lengths,
)
from vln_improve.protocol import file_sha256, object_sha256
from test_endpoint_pairs import FakeAgent, FakeModel, GraphMap, pair_fixture, records_fixture
import test_endpoint_pairs as fixture_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU controls tests must not inspect CUDA"))


def original_rollout():
    """Execute the actual frozen upstream policy against fake features/simulator."""
    path = ROOT / "third_party/VLN-DUET/map_nav_src/r2r/agent.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GMapNavAgent")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "rollout")
    scope = {"np": np, "torch": torch, "GraphMap": GraphMap}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["rollout"]


class PolicyModel(FakeModel):
    def __call__(self, mode, inputs):
        outputs = super().__call__(mode, inputs)
        if mode == "navigation":
            ids = inputs["gmap_vpids"][0]
            current = ids[int(inputs["gmap_step_ids"][0].argmax())]
            right = float(inputs["txt_embeds"].mean()) < 20
            branch, goal = ("a", "ga") if right else ("b", "gb")
            logits = torch.full_like(outputs["fused_logits"], -10)
            if current == "s":
                logits[0, 0] = 2
                logits[0, ids.index(branch)] = 2.1
            elif current == branch:
                logits[0, 0] = 0
                logits[0, ids.index(goal)] = 3
            else:
                logits[:] = .9
                logits[0, 0] = 1
            legal = inputs["gmap_masks"] & ~inputs["gmap_visited_masks"]
            outputs["fused_logits"] = logits.masked_fill(~legal, -torch.inf)
        return outputs


def agent_fixture(monkeypatch):
    # Extra frontier branches let STOP win now while an earlier STOP probability
    # remains larger: the real upstream fallback must return to the start.
    positions = dict(fixture_module.POSITIONS, c=(0, 2, 0), d=(0, -2, 0))
    adjacency = dict(fixture_module.ADJ, s=["a", "b", "c", "d"], c=["s"], d=["s"])
    monkeypatch.setattr(fixture_module, "POSITIONS", positions)
    monkeypatch.setattr(fixture_module, "ADJ", adjacency)
    agent = FakeAgent()
    graph = nx.Graph()
    for a, targets in adjacency.items():
        for b in targets:
            graph.add_edge(a, b, weight=float(np.linalg.norm(np.array(positions[a]) - positions[b])))
    agent.env.graphs = {"scene": graph}
    agent.env.shortest_distances = {"scene": dict(nx.all_pairs_dijkstra_path_length(graph))}
    agent.args.fusion = "dynamic"
    agent.args.enc_full_graph = True
    agent.args.act_visited_nodes = False
    agent.args.detailed_output = False
    agent.vln_bert = PolicyModel()
    agent.rollout = MethodType(original_rollout(), agent)
    agent.feedback = "argmax"
    return agent


def entry_fixture():
    pair = pair_fixture()
    controls = {}
    for slot, instr, goal, positive, overshoot in (
            ("A", "1_0", "ga", ["s", "a", "ga"], ["s", "a", "ga", "b"]),
            ("B", "2_0", "gb", ["s", "b", "gb"], ["s", "b", "gb", "a"])):
        controls[slot] = {"scan": "scene", "path_id": instr.split("_")[0], "instr_id": instr,
            "start": "s", "goal": goal, "heading": 0., "original_path": positive,
            "eligible": True, "positive_history": {"observed_vpids": positive},
            "overshoot_history": {"observed_vpids": overshoot}}
    return {"pair": pair, "controls": controls}


def identity_fixture(entry):
    model = {"dataset": "r2r", "batch_size": 1, "max_action_len": 15, "fusion": "dynamic", "enc_full_graph": True}
    common = {"base_checkpoint_sha256": "1" * 64, "feature_sha256": "2" * 64,
              "annotation_sha256": "3" * 64, "connectivity_sha256": "4" * 64,
              "model": model, "upstream_lock": {"commit": "fake"}, "partition_seed": 0,
              "dev_fraction": .2, "torch_version": str(torch.__version__)}
    return {"schema": SCHEMA, "split": "train_fit", "usage": "training", "selection": [entry],
            "selection_sha256": object_sha256([entry]), "controls_report_sha256": "5" * 64,
            "collection_config_sha256": "6" * 64, "runtime_config_sha256": "7" * 64,
            "code_files": {"test": "8" * 64}, "common_provenance": common,
            "feature_schema": "concat_global_local_stop_crossmodal_v1", "feature_dim": 1536}


def payload_fixture(monkeypatch):
    agent, entry = agent_fixture(monkeypatch), entry_fixture()
    identity = identity_fixture(entry)
    payload = collect_control_pair(agent, entry, records_fixture(), GraphMap, object_sha256(identity))
    return identity, payload


def test_natural_delegates_actual_upstream_argmax_and_fallback_without_extra_observations(monkeypatch):
    agent = agent_fixture(monkeypatch)
    record = records_fixture()["1_0"]
    run = natural_rollout(agent, record)
    assert [s["viewpoint"] for s in run["states"]] == ["s", "a", "ga"]
    assert run["trajectory"] == [["s"], ["a"], ["ga"], ["a", "s"]]
    assert run["fallback_segment"] == [["a", "s"]]
    assert run["actual_length_m"] == 8 and run["states"][-1]["prefix_length_m"] == 4
    assert run["states"][-1]["termination"]["argmax_stop"]
    assert run["features"].shape == (3, 1536)
    assert "labels" not in run
    assert agent.decision_hook is None and agent.feedback == "argmax"
    # Same original policy without the collector must return exactly this route.
    agent.env.buffered_state_dict = {}
    agent.env.batch = [copy.deepcopy(record)]
    agent.env.newEpisodes(["scene"], ["s"], [0.])
    plain = agent.rollout(train_ml=None, train_rl=False, reset=False)
    assert plain[0]["path"] == run["trajectory"]


def test_goal_path_cannot_change_natural_policy_features_or_route(monkeypatch):
    agent = agent_fixture(monkeypatch)
    record = records_fixture()["1_0"]
    original = natural_rollout(agent, record)
    record["path"] = ["s", "b", "gb"]  # Same instruction/start, different offline goal.
    changed = natural_rollout(agent, record)
    assert changed["trajectory"] == original["trajectory"]
    assert torch.equal(changed["features"], original["features"])


def test_six_runs_own_goal_labels_real_routes_and_independent_language_passes(monkeypatch):
    identity, payload = payload_fixture(monkeypatch)
    validate_control_pair(payload, object_sha256(identity), entry_fixture())
    for slot in ("A", "B"):
        ref, over = payload["c2"][slot].values()
        assert bool(ref["labels"]["within_success_radius"][-1])
        assert not bool(over["labels"]["within_success_radius"][-1])
        assert over["actual_length_m"] == 10
        assert len(over["states"]) == 4  # Revisited transit nodes were not observed.
        assert len(sum(over["trajectory"], [])) == 6
        assert payload["natural"][slot]["trajectory"][-1][-1] == "s"
    assert not torch.equal(payload["natural"]["A"]["features"], payload["natural"]["B"]["features"])


@pytest.mark.parametrize("mutation", ["missing_run", "label", "slot", "fallback", "context", "future_observation"])
def test_reject_misassociated_or_partial_control_group(monkeypatch, mutation):
    identity, payload = payload_fixture(monkeypatch)
    run = payload["natural"]["A"]
    if mutation == "missing_run": del payload["c2"]["B"]["overshoot"]
    elif mutation == "label": run["labels"]["within_success_radius"][0] = True
    elif mutation == "slot": run["instruction_slot"] = 1
    elif mutation == "fallback": run["fallback_segment"] = []
    elif mutation == "context": run["context_id"] = "A:reference"
    elif mutation == "future_observation": run["states"].append(copy.deepcopy(run["states"][0]))
    with pytest.raises(ValueError): validate_control_pair(payload, object_sha256(identity), entry_fixture())


def test_store_atomic_restore_idempotent_seal_and_readonly_loader(monkeypatch, tmp_path):
    identity, payload = payload_fixture(monkeypatch)
    local, backup = tmp_path / "local", tmp_path / "drive"
    store = ControlStore(local, backup, identity, lambda: None)
    item = store.commit(payload)
    assert item["rollouts"] == 6
    result = store.seal({"wall_seconds": 12})
    digest = file_sha256(local / "manifest.json")
    loaded = load_control_cache(local, expected_split="train_fit")
    assert len(loaded.groups) == 1 and loaded.groups[0]["pair"] == payload["pair"]
    shutil.rmtree(local / ("pair-" + payload["pair"]["selection_hash"]))
    restored = ControlStore(local, backup, identity, lambda: None)
    assert restored.find(payload["control_entry"]) == item
    assert restored.seal({"wall_seconds": 999}) == result
    assert file_sha256(local / "manifest.json") == digest
    assert load_control_cache(local, expected_split="train_fit").data_sha256 == loaded.data_sha256
    with pytest.raises(ValueError): load_control_cache(local, expected_split="val_unseen")
    with pytest.raises(ValueError): restored.commit(payload)


@pytest.mark.parametrize("target", ["data", "group_manifest", "root_manifest", "root_commit", "identity"])
def test_resume_never_resigns_corrupt_bytes(monkeypatch, tmp_path, target):
    identity, payload = payload_fixture(monkeypatch)
    local, backup = tmp_path / "local", tmp_path / "drive"
    store = ControlStore(local, backup, identity, lambda: None)
    store.commit(payload); store.seal({})
    folder = local / ("pair-" + payload["pair"]["selection_hash"])
    path = {"data": folder / "data.pt", "group_manifest": folder / "manifest.json",
            "root_manifest": local / "manifest.json", "root_commit": local / "COMMITTED.json",
            "identity": local / "COLLECTION.json"}[target]
    with path.open("ab") as stream: stream.write(b"corrupt")
    before = path.read_bytes()
    with pytest.raises((ValueError, json.JSONDecodeError)):
        ControlStore(local, backup, identity, lambda: None).seal({})
    assert path.read_bytes() == before


def test_even_resigned_group_cannot_replace_sealed_group(monkeypatch, tmp_path):
    identity, payload = payload_fixture(monkeypatch)
    local, backup = tmp_path / "local", tmp_path / "drive"
    store = ControlStore(local, backup, identity, lambda: None)
    store.commit(payload); store.seal({})
    folder = local / ("pair-" + payload["pair"]["selection_hash"])
    payload["natural"]["A"]["features"][0, 0] += 1
    torch.save(payload, folder / "data.pt")
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["data_sha256"] = file_sha256(folder / "data.pt")
    atomic_json(folder / "manifest.json", manifest)
    atomic_json(folder / "COMMITTED.json", {"manifest_sha256": file_sha256(folder / "manifest.json")})
    with pytest.raises(ValueError, match="immutable root seal"):
        ControlStore(local, backup, identity, lambda: None).find(payload["control_entry"])


def test_failed_drive_copy_resumes_complete_group_without_rerun(monkeypatch, tmp_path):
    identity, payload = payload_fixture(monkeypatch)
    store = ControlStore(tmp_path / "local", tmp_path / "drive", identity, lambda: None)
    original = store._copy
    monkeypatch.setattr(store, "_copy", lambda *args: (_ for _ in ()).throw(OSError("disconnect")))
    with pytest.raises(OSError): store.commit(payload)
    monkeypatch.setattr(store, "_copy", original)
    assert store.find(payload["control_entry"])["rollouts"] == 6
    store.seal({})


def test_full_walk_length_rejects_teleport_and_counts_transit():
    graph = nx.Graph()
    graph.add_edge("s", "a", weight=2)
    graph.add_edge("a", "g", weight=2)
    assert walk_length([["s"], ["a", "g"], ["a", "s"]], graph) == 8
    with pytest.raises(ValueError): walk_length([["s"], ["g"]], graph)


def test_two_graph_lengths_remain_distinct_without_relaxing_a_tolerance(monkeypatch):
    agent, entry = agent_fixture(monkeypatch), entry_fixture()
    graph = agent.env.graphs["scene"]
    # The real-cache diagnostic found a 1.36e-7 m difference because MatterSim
    # rounds JSON coordinates to float32. Exercise that separate-metric case.
    graph["a"]["ga"]["weight"] += 2e-7
    agent.env.shortest_distances["scene"] = dict(nx.all_pairs_dijkstra_path_length(graph))
    identity = identity_fixture(entry)
    payload = collect_control_pair(agent, entry, records_fixture(), GraphMap, object_sha256(identity))
    run = payload["c2"]["A"]["reference"]
    assert run["execution_graph_length_m"] == 4.0
    assert run["execution_position_edge_sum_m"] == 4.0
    assert run["actual_length_m"] == 4.000000200000001
    assert [s["execution_graph_prefix_length_m"] for s in run["states"]] == [0., 2., 4.]
    assert run["states"][-1]["prefix_length_m"] == run["actual_length_m"]
    assert run["trajectory"] == [["s"], ["a"], ["ga"]]
    natural = payload["natural"]["A"]
    assert natural["execution_graph_length_m"] is None
    assert natural["actual_length_m"] == 8.000000400000002
    assert natural["execution_position_edge_sum_m"] == 8.0
    assert natural["trajectory"] == [["s"], ["a"], ["ga"], ["a", "s"]]
    validate_control_pair(payload, object_sha256(identity), entry)


@pytest.mark.parametrize("mutation", ["total", "prefix", "edge", "execution_source"])
def test_v2_length_audit_rejects_missing_route_costs_and_changed_conventions(monkeypatch, mutation):
    identity, payload = payload_fixture(monkeypatch)
    run = payload["c2"]["A"]["overshoot"]
    if mutation == "total": run["actual_length_m"] -= 2
    elif mutation == "prefix": run["states"][-1]["prefix_length_m"] -= 2
    elif mutation == "edge": run["length_audit"]["edges"].pop()
    elif mutation == "execution_source": run["length_audit"]["execution_graph_source"] = "official_connectivity"
    with pytest.raises(ValueError):
        validate_control_pair(payload, object_sha256(identity), entry_fixture())


def test_real_diagnostic_totals_stay_distinct_while_same_source_is_checked():
    # Values measured from the real first fit pair's 7-state prefix. This small
    # audit fixture isolates the validated two-metric relationship, not imagery.
    execution, official = 8.206755407959456, 8.206755271671916
    graph = nx.Graph()
    graph.add_edge("s", "g", weight=official)
    run = {"trajectory": [["s"], ["g"]], "actual_length_m": execution,
           "states": [{"viewpoint": "s", "position": [0., 0., 0.],
                       "trajectory_prefix": [["s"]], "prefix_length_m": 0.},
                      {"viewpoint": "g", "position": [execution, 0., 0.],
                       "trajectory_prefix": [["s"], ["g"]], "prefix_length_m": execution}]}
    annotate_route_lengths(run, graph, forced=True)
    validate_route_lengths(run, forced=True)
    assert run["actual_length_m"] == official
    assert run["execution_graph_length_m"] == execution
    assert execution - official == 1.3628753947614314e-7
    # Changing both redundant stored counters does not bypass the edge audit.
    run["execution_graph_length_m"] += 1
    run["states"][-1]["execution_graph_prefix_length_m"] += 1
    with pytest.raises(ValueError, match="same-coordinate edge sum"):
        validate_route_lengths(run, forced=True)


def test_every_original_execution_prefix_is_checked(monkeypatch):
    identity, payload = payload_fixture(monkeypatch)
    run = payload["c2"]["A"]["overshoot"]
    run["states"][1]["execution_graph_prefix_length_m"] += 1
    with pytest.raises(ValueError, match="execution prefix differs"):
        validate_control_pair(payload, object_sha256(identity), entry_fixture())
