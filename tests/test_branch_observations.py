"""CPU checks for independent branch observations and frozen-state replay."""
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/VLN-DUET/map_nav_src"))

from models.graph_utils import GraphMap
import replay_branches
from replay_diagnostics import chosen_candidates, seal_result
from vln_improve.branch_observations import (
    MatterportBranchEnvironment, PrefixGraph, encode_panorama, matched_donor,
    panorama_inputs, view_angles,
)
from vln_improve.counterfactual import navigation_hash
from test_diagnostics import collect


POSITIONS = {"A": [0.0, 0.0, 0.0], "B": [10.0, 0.0, 0.0],
             "C": [9.0, 1.0, 0.0], "X": [0.0, 1.0, 0.0]}


def observation(current, candidates):
    return {"viewpoint": current, "position": POSITIONS[current],
            "heading": 0.0, "elevation": 0.0,
            "candidate": [{"viewpointId": target, "position": POSITIONS[target],
                           "pointId": point} for target, point in candidates]}


def geometry_state(graph, ob, step):
    """Save the upstream geometry after each observed panorama in the prefix."""
    graph.update_graph(ob)
    graph.node_step_ids[ob["viewpoint"]] = step + 1
    visited = [vp for vp in graph.node_positions if graph.graph.visited(vp)]
    unseen = [vp for vp in graph.node_positions if not graph.graph.visited(vp)]
    ids = [None] + visited + unseen
    distances = torch.zeros(1, len(ids), len(ids))
    for i in range(1, len(ids)):
        for j in range(i + 1, len(ids)):
            distances[0, i, j] = distances[0, j, i] = graph.graph.distance(ids[i], ids[j])
    return {"step": step, "current_viewpoint": ob["viewpoint"],
            "heading": ob["heading"], "elevation": ob["elevation"],
            "nav_inputs": {"gmap_vpids": [ids],
                "vp_cand_vpids": [[None] + [c["viewpointId"] for c in ob["candidate"]]],
                "gmap_step_ids": torch.tensor([[graph.node_step_ids.get(vp, 0) for vp in ids]]),
                "gmap_visited_masks": torch.tensor([[False] + [graph.graph.visited(vp) for vp in ids[1:]]]),
                "gmap_pos_fts": torch.from_numpy(graph.get_pos_fts(
                    ob["viewpoint"], ids, ob["heading"], ob["elevation"]))[None],
                "gmap_pair_dists": distances}}


@pytest.fixture
def geometry_prefix():
    observations = [observation("A", [("B", 15), ("X", 31)]),
                    observation("B", [("A", 17), ("C", 26)]),
                    observation("C", [("B", 8), ("X", 29)])]
    graph = GraphMap("A")
    states = [geometry_state(graph, ob, step) for step, ob in enumerate(observations)]
    return states, observations


def test_prefix_graph_uses_only_observed_edges_and_restores_last_hop_orientation(geometry_prefix):
    states, observations = geometry_prefix
    prefix = PrefixGraph(GraphMap)
    prefix.advance(states[0], observations[0])
    with pytest.raises(ValueError, match="discovered unvisited"):
        prefix.route("C")
    prefix.advance(states[1], observations[1])
    before = prefix.route("X")
    assert before["path"] == ["B", "A", "X"]
    assert before["edge_point_ids"] == [17, 31]
    assert before["distance"] == pytest.approx(11.0)
    assert before["hops"] == 2
    assert before["arrival_view_index"] == 31
    assert before["arrival_heading"] == pytest.approx(7 * math.pi / 6)
    assert before["arrival_elevation"] == pytest.approx(math.pi / 6)
    assert frozenset(("C", "X")) not in prefix.edges
    # The future C-X shortcut becomes available only after C's own observation.
    prefix.advance(states[2], observations[2])
    assert prefix.route("X")["path"] == ["C", "X"]
    assert before["path"] == ["B", "A", "X"]


@pytest.mark.parametrize("damage,match", [
    ("step", "ordered prefix"), ("viewpoint", "ordered prefix"),
    ("heading", "orientation"), ("elevation", "orientation"),
    ("candidate_order", "candidate order"), ("node_order", "node order"),
    ("step_ids", "step IDs"), ("geometry", "geometry"),
    ("distance", "distances"), ("visited", "visited state"),
])
def test_prefix_graph_rejects_history_that_disagrees_with_frozen_state(geometry_prefix, damage, match):
    states, observations = copy.deepcopy(geometry_prefix)
    state, ob = states[0], observations[0]
    if damage == "step":
        state["step"] = 1
    elif damage == "viewpoint":
        ob["viewpoint"] = "B"
    elif damage in {"heading", "elevation"}:
        ob[damage] = 0.3
    elif damage == "candidate_order":
        ob["candidate"].reverse()
    elif damage == "node_order":
        state["nav_inputs"]["gmap_vpids"][0][-2:] = ["X", "B"]
    elif damage == "step_ids":
        state["nav_inputs"]["gmap_step_ids"][0, 1] = 0
    elif damage == "geometry":
        state["nav_inputs"]["gmap_pos_fts"][0, 2, 4] += 1
    elif damage == "visited":
        state["nav_inputs"]["gmap_visited_masks"][0, 2] = True
    else:
        state["nav_inputs"]["gmap_pair_dists"][0, 1, 2] += 1
    with pytest.raises(ValueError, match=match):
        PrefixGraph(GraphMap).advance(state, ob)


def test_prefix_route_rejects_visited_target_and_unobserved_reverse_edge(geometry_prefix):
    states, observations = geometry_prefix
    prefix = PrefixGraph(GraphMap)
    prefix.advance(states[0], observations[0])
    with pytest.raises(ValueError, match="discovered unvisited"):
        prefix.route("A")
    # The graph knows an undirected A-B edge, but an executable B-A turn needs
    # B's own candidate view index; it cannot be inferred from A's observation.
    second_ob = observation("B", [("C", 26)])
    graph = GraphMap("A")
    geometry_state(graph, observations[0], 0)
    second_state = geometry_state(graph, second_ob, 1)
    prefix.advance(second_state, second_ob)
    with pytest.raises(ValueError, match="undiscovered directed edge"):
        prefix.route("X")


@pytest.mark.parametrize("point", [-1, 36, 1.5, True])
def test_discrete_view_angles_reject_invalid_indices(point):
    with pytest.raises(ValueError, match="pointId"):
        view_angles(point)


def test_panorama_packing_retains_two_candidates_at_the_same_view_index():
    features = np.arange(36 * 6, dtype=np.float32).reshape(36, 6)
    first, second = features[7].copy(), features[7].copy() + 1000
    ob = {"feature": features, "candidate": [
        {"viewpointId": "B", "pointId": 7, "feature": first},
        {"viewpointId": "X", "pointId": 7, "feature": second}]}
    batch = panorama_inputs(ob, image_feat_size=2, device="cpu")
    expected = np.stack([first, second] + [row for i, row in enumerate(features) if i != 7])
    torch.testing.assert_close(batch["view_img_fts"], torch.from_numpy(expected[:, :2])[None])
    torch.testing.assert_close(batch["loc_fts"][0, :, :4], torch.from_numpy(expected[:, 2:]))
    assert torch.equal(batch["loc_fts"][0, :, 4:], torch.ones(37, 3))
    assert batch["view_lens"].tolist() == [37]
    assert batch["nav_types"].tolist() == [[1, 1] + [0] * 35]
    assert batch["cand_vpids"] == [["B", "X"]]

    class MaskedModel(torch.nn.Module):
        def forward(self, mode, inputs):
            assert mode == "panorama"
            mask = torch.ones(1, 37, dtype=torch.bool)
            mask[0, -1] = False
            return inputs["view_img_fts"], mask

    model = MaskedModel().eval()
    mean, embeds = encode_panorama(model, ob, image_feat_size=2, device="cpu")
    torch.testing.assert_close(mean, torch.from_numpy(expected[:-1, :2]).mean(0))
    assert mean.device.type == embeds.device.type == "cpu"
    model.train()
    with pytest.raises(ValueError, match="eval"):
        encode_panorama(model, ob, image_feat_size=2, device="cpu")


def test_donor_matching_uses_current_geometry_then_source_count_not_oracles():
    ids = [None, "A", "target", "near", "same-distance", "visited"]
    distances = torch.zeros(1, len(ids), len(ids))
    distances[0, 1] = torch.tensor([0, 0, 3, 2, 3, 3])
    state = {"step": 2, "current_viewpoint": "A",
             "nav_inputs": {"gmap_vpids": [ids], "gmap_pair_dists": distances},
             "valid_mask": torch.tensor([True, False, True, True, True, False]),
             "candidate_evidence": {target: {"counts": {"source_count_total": count}}
                 for target, count in (("target", 2), ("near", 2), ("same-distance", 1))}}
    donor = matched_donor(state, 2, seed=4)
    assert donor["target_id"] == "same-distance"
    assert donor["distance_difference"] == 0
    assert donor["source_count_difference"] == 1
    state["teacher_cost"] = [0, 0, 0, 0, 999, 0]
    state["arrival_pairs"] = [{"target_id": "near", "arrival_step": 3}]
    state["future_outcome"] = {"same-distance": "bad"}
    assert matched_donor(state, 2, seed=4) == donor
    state["valid_mask"][3:5] = False
    assert matched_donor(state, 2, seed=4) is None


class FixturePrefixGraph:
    """Route stub: the separate geometry tests exercise the real graph above."""
    def __init__(self, factory=None):
        self.step = -1

    def advance(self, state, ob):
        assert state["step"] == self.step + 1
        assert ob["viewpoint"] == state["current_viewpoint"]
        self.state, self.step = state, state["step"]

    def route(self, target):
        return {"path": [self.state["current_viewpoint"], target],
                "edge_point_ids": [12], "hops": 1, "distance": 1.0,
                "arrival_heading": 0.0, "arrival_elevation": 0.0,
                "arrival_view_index": 12}


class FixtureBranchProvider:
    def __init__(self, *, unknown_scale=1.0, headingmatched_scale=1.0, damage=None):
        self.unknown_scale = unknown_scale
        self.headingmatched_scale = headingmatched_scale
        self.damage = damage
        self.visits = []
        self.histories = []

    def historical(self, scan, state):
        self.histories.append((scan, state["step"], state["current_viewpoint"]))
        return {"viewpoint": state["current_viewpoint"]}

    def branch(self, scan, state, route):
        target = route["path"][-1]
        self.visits.append((state["step"], target))
        feature = {"B": torch.tensor([1.0, 2.5]), "C": torch.tensor([0.0, 5.0]),
                   "X": torch.tensor([3.0, 4.0]) * self.unknown_scale}[target]
        result = {"feature": feature,
                  "headingmatched_feature": (feature + torch.tensor([2.0, 1.0])) * self.headingmatched_scale,
                  "heading": 0.0, "elevation": 0.0, "view_index": 12,
                  "headingmatched_heading": 0.0, "headingmatched_elevation": 0.0,
                  "target_panorama_tokens": 36, "target_degree": 2}
        if self.damage == "natural_feature" and target == "B":
            result["feature"] = feature + 1
        elif self.damage == "zero_feature":
            result["feature"] = torch.zeros_like(feature)
        elif self.damage == "primary_heading":
            result["heading"] = 1.0
        elif self.damage == "matched_heading":
            result["headingmatched_heading"] = 1.0
        return result


@pytest.fixture
def branch_episode(tmp_path, monkeypatch):
    agent, observer, store, trajectory, payload = collect(tmp_path)
    observer.close()
    monkeypatch.setattr(replay_branches, "PrefixGraph", FixturePrefixGraph)
    return agent.vln_bert.eval(), payload[0], payload[1]


def run_branches(episode, provider=None):
    model, inputs, labels = episode
    return replay_branches.process_episode(model, provider or FixtureBranchProvider(), inputs, labels,
                                          split="train_fit", seed=0, device="cpu")


def test_branch_replay_retains_every_preselected_candidate_including_unarrived(branch_episode):
    model, inputs, labels = branch_episode
    before = [navigation_hash(state["nav_inputs"]) for state in inputs["states"]]
    provider = FixtureBranchProvider()
    result = run_branches(branch_episode, provider)
    expected = [(state["step"], state["nav_inputs"]["gmap_vpids"][0][index])
                for state in inputs["states"] if state["eligible_decision"]
                for index in chosen_candidates(state, 0)]
    assert [(row["step"], row["target_id"]) for row in result["rows"]] == expected
    assert result["counters"]["paired_rows"] == result["counters"]["selected_candidate_states"] == 5
    assert result["counters"]["selected_without_natural_arrival"] == 3
    assert {row["target_id"] for row in result["rows"] if not row["has_natural_arrival"]} == {"X"}
    assert result["counters"]["natural_chosen_parity_checks"] == 2
    assert result["counters"]["max_natural_feature_error"] == 0.0
    assert result["counters"]["max_replay_error"] == 0.0
    assert result["counters"]["historical_panorama_parity_checks"] == 3
    assert len(provider.histories) == 3
    assert len(provider.visits) == len(set(provider.visits)) == 5
    assert all(row["arrival_kind"] == "simulated_branch" and "arrival_step" not in row
               and row["future_information_is_offline_only"] for row in result["rows"])
    assert before == [navigation_hash(state["nav_inputs"]) for state in inputs["states"]]
    # Donors come from this decision's legal candidate set, including X which
    # has no natural future observation at any step of the original trajectory.
    for row in result["rows"]:
        donor = row["shuffled_donor"]
        if donor is None:
            assert (row["step"], row["target_id"]) == (2, "X")
            assert "shuffled_arrival" not in row["interventions"]
            continue
        assert donor == matched_donor(inputs["states"][row["step"]],
                                     inputs["states"][row["step"]]["nav_inputs"]["gmap_vpids"][0].index(row["target_id"]))
        assert donor["target_id"] != row["target_id"]
        donor_ob = next(ob for ob in result["branch_observations"]
                        if ob["step"] == row["step"] and ob["target_id"] == donor["target_id"])
        assert row["interventions"]["shuffled_arrival"]["replacement_norm"] == pytest.approx(
            torch.tensor(donor_ob["feature"]).norm().item())
        assert row["interventions"]["shuffled_headingmatched"]["replacement_norm"] == pytest.approx(
            torch.tensor(donor_ob["headingmatched_feature"]).norm().item())
        assert row["interventions"]["shuffled_normmatched"]["replacement_norm"] == pytest.approx(
            row["interventions"]["shuffled_normmatched"]["original_norm"], rel=1e-6)
    assert result["counters"]["missing_shuffled_control"] == 1


def test_natural_pair_availability_does_not_change_branch_sample_or_features(branch_episode):
    original = run_branches(branch_episode)
    model, inputs, labels = branch_episode
    no_natural = copy.deepcopy(labels)
    no_natural["arrival_pairs"] = []
    independent = run_branches((model, inputs, no_natural))
    assert independent["selections"] == original["selections"]
    assert len(independent["rows"]) == len(original["rows"]) == 5
    assert independent["counters"]["selected_without_natural_arrival"] == 5
    for first, second in zip(original["rows"], independent["rows"]):
        assert first["p0"] == second["p0"]
        assert first["p1"] == second["p1"]
        assert first["shuffled_donor"] == second["shuffled_donor"]
        assert first["interventions"] == second["interventions"]


def test_privileged_target_and_orientation_features_do_not_leak_into_p0_p1(branch_episode):
    original = run_branches(branch_episode)
    changed = run_branches(branch_episode, FixtureBranchProvider(unknown_scale=3, headingmatched_scale=2))
    assert original["selections"] == changed["selections"]
    for first, second in zip(original["rows"], changed["rows"]):
        assert first["p0"] == second["p0"]
        assert first["p1"] == second["p1"]
        assert first["navigation_sha256"] == second["navigation_sha256"]
        assert first["shuffled_donor"] == second["shuffled_donor"]
        assert (first["interventions"]["arrival_headingmatched"]["replacement_norm"]
                != second["interventions"]["arrival_headingmatched"]["replacement_norm"])
        if first["target_id"] == "X":
            assert first["branch_feature_sha256"] != second["branch_feature_sha256"]
            assert (first["interventions"]["arrival"]["replacement_norm"]
                    != second["interventions"]["arrival"]["replacement_norm"])
        for name in ("arrival_normmatched", "arrival_headingmatched_normmatched", "noise_normmatched"):
            assert second["interventions"][name]["replacement_norm"] == pytest.approx(
                second["interventions"][name]["original_norm"], rel=1e-6)


@pytest.mark.parametrize("damage,error", [("natural_feature", AssertionError),
    ("primary_heading", ValueError), ("matched_heading", ValueError), ("zero_feature", ValueError)])
def test_branch_replay_rejects_natural_parity_or_orientation_violations(branch_episode, damage, error):
    with pytest.raises(error):
        run_branches(branch_episode, FixtureBranchProvider(damage=damage))


def test_branch_replay_keeps_ineligible_history_but_does_not_select_it(branch_episode):
    model, inputs, labels = branch_episode
    inputs["states"][-1]["eligible_decision"] = False
    provider = FixtureBranchProvider()
    result = run_branches(branch_episode, provider)
    assert len(provider.histories) == 3
    assert result["counters"]["eligible_states"] == 2
    assert result["counters"]["paired_rows"] == 4
    assert all(row["step"] < 2 for row in result["rows"])


def test_branch_cached_result_restores_only_matching_input_program_split_and_association(branch_episode, tmp_path):
    result = run_branches(branch_episode)
    sealed = seal_result(result, input_manifest_sha256="input-hash", replay_identity_sha256="branch-code")
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(sealed))
    expected = {"input_manifest_sha256": "input-hash", "replay_identity_sha256": "branch-code",
                "association": branch_episode[1]["association"], "split": "train_fit"}
    assert replay_branches.read_branch_result(path, **expected) == sealed
    for change in ({"input_manifest_sha256": "other-input"}, {"replay_identity_sha256": "other-code"},
                   {"split": "train_dev"}, {"association": {**expected["association"], "scan_id": "other-scan"}}):
        with pytest.raises(ValueError):
            replay_branches.read_branch_result(path, **{**expected, **change})
    sealed["branch_observations"][0]["feature"][0] += 0.5
    path.write_text(json.dumps(sealed))
    with pytest.raises(ValueError, match="checksum"):
        replay_branches.read_branch_result(path, **expected)


@pytest.mark.parametrize("damage", ["cohort", "row_count", "natural_cohort", "arrival_step"])
def test_branch_cache_rejects_resealed_incompatible_cohort(branch_episode, tmp_path, damage):
    result = run_branches(branch_episode)
    if damage == "cohort":
        result["analysis_cohort"] = "naturally_arrived_only"
    elif damage == "row_count":
        result["counters"]["selected_candidate_states"] += 1
    elif damage == "natural_cohort":
        result["rows"][0]["arrival_kind"] = "natural"
    else:
        result["rows"][0]["arrival_step"] = 10
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(seal_result(result, input_manifest_sha256="input", replay_identity_sha256="code")))
    with pytest.raises(ValueError, match="cohort"):
        replay_branches.read_branch_result(path, input_manifest_sha256="input", replay_identity_sha256="code",
            association=branch_episode[1]["association"], split="train_fit")


class FakeWalker:
    """Discrete turns with navigation indices that change after rotation."""
    def __init__(self, *, missing_target=False):
        self.missing_target = missing_target
        self.starts, self.moves, self.turns = [], [], []

    def newEpisode(self, scans, viewpoints, headings, elevations):
        assert len(scans) == len(viewpoints) == len(headings) == len(elevations) == 1
        self.starts.append((scans[0], viewpoints[0], headings[0], elevations[0]))
        self.scan, self.viewpoint = scans[0], viewpoints[0]
        self.view_index = int(round(headings[0] / (math.pi / 6))) % 12
        self.view_index += (int(round(elevations[0] / (math.pi / 6))) + 1) * 12

    def getState(self):
        heading, elevation = view_angles(self.view_index)
        if self.viewpoint == "A":
            neighbors = ["decoy-a", "B"] if self.view_index == 17 else ["B", "decoy-a"]
            if self.missing_target and self.view_index == 17:
                neighbors = ["decoy-a"]
        elif self.viewpoint == "B":
            neighbors = ["decoy-b", "decoy-c", "X"] if self.view_index == 31 else ["X", "decoy-b", "decoy-c"]
        else:
            neighbors = []
        return [SimpleNamespace(scanId=self.scan, location=SimpleNamespace(viewpointId=self.viewpoint),
            viewIndex=self.view_index, heading=heading, elevation=elevation,
            navigableLocations=[SimpleNamespace(viewpointId=vp) for vp in [self.viewpoint] + neighbors])]

    def makeAction(self, indices, headings, elevations):
        assert len(indices) == len(headings) == len(elevations) == 1
        index, heading, elevation = indices[0], headings[0], elevations[0]
        assert heading in {-1, 0, 1} and elevation in {-1, 0, 1}
        if index:
            assert heading == elevation == 0
            destination = self.getState()[0].navigableLocations[index].viewpointId
            self.moves.append((self.viewpoint, destination, self.view_index, index))
            self.viewpoint = destination
        else:
            self.turns.append((self.viewpoint, self.view_index, heading, elevation))
            band = max(0, min(2, self.view_index // 12 + elevation))
            self.view_index = band * 12 + (self.view_index % 12 + heading) % 12


def isolated_physical_branch(monkeypatch, *, missing_target=False):
    provider = MatterportBranchEnvironment.__new__(MatterportBranchEnvironment)
    provider.walker = FakeWalker(missing_target=missing_target)

    def current_observation():
        state = provider.walker.getState()[0]
        return {"scan": state.scanId, "viewpoint": state.location.viewpointId,
                "viewIndex": state.viewIndex, "heading": state.heading, "elevation": state.elevation,
                "candidate": [{"pointId": 5}, {"pointId": 5}]}

    matched_calls = []

    def matched_observation(scan, viewpoint, heading, elevation):
        # This control may inspect another orientation only after the physical
        # branch reached its target; it cannot stand in for either route edge.
        assert provider.walker.viewpoint == viewpoint == "X"
        assert len(provider.walker.moves) == 2
        matched_calls.append((scan, viewpoint, heading, elevation))
        return {**current_observation(), "heading": heading, "elevation": elevation, "viewIndex": 12}

    monkeypatch.setattr(provider, "_current_observation", current_observation)
    monkeypatch.setattr(provider, "_average", lambda ob: torch.tensor([float(ob["viewIndex"]), 1.0]))
    monkeypatch.setattr(provider, "observation", matched_observation)
    heading, elevation = view_angles(31)
    route = {"path": ["A", "B", "X"], "edge_point_ids": [17, 31], "hops": 2,
             "distance": 2.0, "arrival_heading": heading, "arrival_elevation": elevation,
             "arrival_view_index": 31}
    state = {"current_viewpoint": "A", "heading": 0.0, "elevation": 0.0}
    return provider, state, route, matched_calls


def test_physical_branch_rotates_then_resolves_each_live_navigation_index(monkeypatch):
    provider, state, route, matched_calls = isolated_physical_branch(monkeypatch)
    result = provider.branch("scan", state, route)
    # Before each rotation the desired neighbor is index 1. At the executable
    # view it becomes 2 and then 3, catching stale candidate navigation indices.
    assert provider.walker.moves == [("A", "B", 17, 2), ("B", "X", 31, 3)]
    assert provider.walker.starts == [("scan", "A", 0.0, 0.0)]
    assert provider.walker.viewpoint == "X"
    assert provider.walker.view_index == result["view_index"] == 31
    assert provider.walker.turns
    assert any(turn[3] == 1 for turn in provider.walker.turns)
    assert result["heading"] == pytest.approx(route["arrival_heading"])
    assert result["elevation"] == pytest.approx(route["arrival_elevation"])
    assert matched_calls == [("scan", "X", 0.0, 0.0)]
    assert torch.equal(result["feature"], torch.tensor([31.0, 1.0]))
    assert torch.equal(result["headingmatched_feature"], torch.tensor([12.0, 1.0]))
    assert result["target_panorama_tokens"] == 37


def test_physical_branch_rejects_target_missing_after_rotation(monkeypatch):
    provider, state, route, matched_calls = isolated_physical_branch(monkeypatch, missing_target=True)
    with pytest.raises(ValueError, match="not legally navigable"):
        provider.branch("scan", state, route)
    assert provider.walker.viewpoint == "A"
    assert provider.walker.view_index == 17
    assert provider.walker.moves == []
    assert provider.walker.starts == [("scan", "A", 0.0, 0.0)]
    assert matched_calls == []
