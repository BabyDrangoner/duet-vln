"""Independent, privileged branch observations for frozen DUET diagnostics.

No policy graph is updated with a branch observation. Route construction sees
only the saved history prefix; RGB rendering and target/goal labels are unused.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from .protocol import object_sha256


def view_angles(point_id: int) -> tuple[float, float]:
    if type(point_id) is not int or not 0 <= point_id < 36:
        raise ValueError("invalid discrete panorama pointId")
    return (point_id % 12) * math.pi / 6, (point_id // 12 - 1) * math.pi / 6


def angle_close(a: float, b: float, *, atol=1e-6) -> bool:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi) <= atol


def matched_donor(state: dict, target_index: int, seed: int = 0):
    """Current-geometry matching only; never inspect oracle/arrival labels."""
    ids = state["nav_inputs"]["gmap_vpids"][0]
    current = ids.index(state["current_viewpoint"])
    distances = state["nav_inputs"]["gmap_pair_dists"][0, current]
    counts = lambda i: state["candidate_evidence"][ids[i]]["counts"]["source_count_total"]
    choices = [i for i in range(1, len(ids)) if i != target_index and bool(state["valid_mask"][i])]
    if not choices:
        return None
    def key(i):
        return (abs(float(distances[i] - distances[target_index])), abs(counts(i) - counts(target_index)),
                object_sha256([seed, state["step"], ids[target_index], ids[i]]))
    selected = min(choices, key=key)
    return {"index": selected, "target_id": ids[selected], "distance_difference": key(selected)[0],
            "source_count_difference": key(selected)[1], "rule": "current_graph_distance_then_source_count_then_hash"}


class PrefixGraph:
    """Recreate the upstream graph with the same observed edge/update order."""
    def __init__(self, graph_factory=None):
        if graph_factory is None:
            from models.graph_utils import GraphMap
            graph_factory = GraphMap
        self.factory = graph_factory
        self.graph = None
        self.current = None
        self.last_step = -1
        self.edge_points = {}
        self.edges = set()

    def advance(self, state: dict, observation: dict):
        step, current = state["step"], state["current_viewpoint"]
        if step != self.last_step + 1 or observation["viewpoint"] != current:
            raise ValueError("historical graph requires the complete ordered prefix")
        if not angle_close(observation["heading"], state["heading"]) or abs(observation["elevation"] - state["elevation"]) > 1e-6:
            raise ValueError("historical simulator orientation differs from frozen state")
        expected_candidates = state["nav_inputs"]["vp_cand_vpids"][0][1:]
        if [c["viewpointId"] for c in observation["candidate"]] != expected_candidates:
            raise ValueError("historical simulator candidate order differs from frozen state")
        if self.graph is None:
            self.graph = self.factory(current)
        self.graph.update_graph(observation)
        self.graph.node_step_ids[current] = step + 1
        for candidate in observation["candidate"]:
            other = candidate["viewpointId"]
            self.edges.add(frozenset((current, other)))
            self.edge_points[current, other] = int(candidate["pointId"])
        self.current, self.last_step = current, step
        nav = state["nav_inputs"]
        ids = nav["gmap_vpids"][0]
        visited = [vp for vp in self.graph.node_positions if self.graph.graph.visited(vp)]
        unseen = [vp for vp in self.graph.node_positions if not self.graph.graph.visited(vp)]
        if [None] + visited + unseen != ids:
            raise ValueError("reconstructed graph node order differs from frozen state")
        if nav["gmap_visited_masks"][0].tolist() != [False] + [self.graph.graph.visited(vp) for vp in ids[1:]]:
            raise ValueError("reconstructed visited state differs from frozen graph")
        expected_steps = [self.graph.node_step_ids.get(vp, 0) for vp in ids]
        if nav["gmap_step_ids"][0].tolist() != expected_steps:
            raise ValueError("reconstructed graph step IDs differ from frozen state")
        positions = self.graph.get_pos_fts(current, ids, state["heading"], state["elevation"])
        if not torch.allclose(torch.as_tensor(positions), nav["gmap_pos_fts"][0], atol=1e-5, rtol=1e-5):
            raise ValueError("reconstructed graph geometry/path hops differ from frozen state")
        distances = torch.zeros_like(nav["gmap_pair_dists"][0])
        for i in range(1, len(ids)):
            for j in range(i + 1, len(ids)):
                distances[i, j] = distances[j, i] = self.graph.graph.distance(ids[i], ids[j])
        if not torch.allclose(distances, nav["gmap_pair_dists"][0], atol=1e-5, rtol=1e-5):
            raise ValueError("reconstructed graph distances differ from frozen state")

    def route(self, target: str) -> dict:
        graph = self.graph
        if graph is None or target not in graph.node_positions or graph.graph.visited(target):
            raise ValueError("branch target must be a discovered unvisited candidate")
        path = [self.current] + graph.graph.path(self.current, target)
        if len(path) < 2 or path[-1] != target or len(set(path)) != len(path):
            raise ValueError("invalid discovered-graph path")
        if any(not graph.graph.visited(vp) for vp in path[1:-1]):
            raise ValueError("branch path uses an unobserved intermediate node")
        points = []
        for source, destination in zip(path, path[1:]):
            if frozenset((source, destination)) not in self.edges or (source, destination) not in self.edge_points:
                raise ValueError("branch route contains an undiscovered directed edge")
            points.append(self.edge_points[source, destination])
        heading, elevation = view_angles(points[-1])
        return {"path": path, "edge_point_ids": points, "hops": len(points),
                "distance": float(graph.graph.distance(self.current, target)),
                "arrival_heading": heading, "arrival_elevation": elevation,
                "arrival_view_index": points[-1]}


def panorama_inputs(observation: dict, *, image_feat_size: int, device="cuda") -> dict:
    """Exact batch=1 DUET packing, including repeated candidate view tokens."""
    candidates = observation["candidate"]
    used = {candidate["pointId"] for candidate in candidates}
    rows = [candidate["feature"] for candidate in candidates]
    rows.extend(row for index, row in enumerate(observation["feature"]) if index not in used)
    features = np.stack(rows).astype(np.float32)
    loc = np.concatenate((features[:, image_feat_size:], np.ones((len(rows), 3), dtype=np.float32)), axis=1)
    return {"view_img_fts": torch.from_numpy(features[:, :image_feat_size])[None].to(device),
            "loc_fts": torch.from_numpy(loc)[None].to(device),
            "nav_types": torch.tensor([[1] * len(candidates) + [0] * (36 - len(used))], device=device),
            "view_lens": torch.tensor([len(rows)], device=device),
            "cand_vpids": [[candidate["viewpointId"] for candidate in candidates]]}


@torch.inference_mode()
def encode_panorama(model, observation: dict, *, image_feat_size: int, device="cuda"):
    if model.training:
        raise ValueError("branch encoding requires eval mode")
    batch = panorama_inputs(observation, image_feat_size=image_feat_size, device=device)
    embeds, masks = model("panorama", batch)
    average = (embeds * masks.unsqueeze(2)).sum(1) / masks.sum(1, keepdim=True)
    if not torch.isfinite(embeds).all():
        raise ValueError("nonfinite branch panorama")
    return average[0].detach().float().cpu().clone(), embeds[0].detach().float().cpu().clone()


class MatterportBranchEnvironment:
    """Two private simulators; one walks, one enumerates panorama candidates."""
    def __init__(self, connectivity_dir, feature_file, model, *, image_feat_size=768, angle_feat_size=4, device="cuda"):
        from utils.data import ImageFeaturesDB, new_simulator, get_all_point_angle_feature
        from r2r.env import R2RNavBatch
        self.walker = new_simulator(str(connectivity_dir))
        scanner = new_simulator(str(connectivity_dir))
        self.adapter = SimpleNamespace(sim=scanner, buffered_state_dict={}, angle_feat_size=angle_feat_size)
        self.make_candidate = R2RNavBatch.make_candidate
        self.angle_features = get_all_point_angle_feature(scanner, angle_feat_size)
        self.features = ImageFeaturesDB(str(feature_file), image_feat_size)
        self.model, self.image_feat_size, self.device = model, image_feat_size, device
        self.average_cache = {}

    def _current_observation(self):
        state = self.walker.getState()[0]
        scan, viewpoint, view_id = state.scanId, state.location.viewpointId, int(state.viewIndex)
        heading, elevation = float(state.heading), float(state.elevation)
        position = (state.location.x, state.location.y, state.location.z)
        raw = self.features.get_image_feature(scan, viewpoint)
        candidates = self.make_candidate(self.adapter, raw, scan, viewpoint, view_id)
        return {"scan": scan, "viewpoint": viewpoint, "viewIndex": view_id,
                "heading": heading, "elevation": elevation, "position": position,
                "feature": np.concatenate((raw, self.angle_features[view_id]), axis=-1), "candidate": candidates}

    def observation(self, scan, viewpoint, heading, elevation):
        self.walker.newEpisode([scan], [viewpoint], [heading], [elevation])
        return self._current_observation()

    def historical(self, scan, state):
        ob = self.observation(scan, state["current_viewpoint"], state["heading"], state["elevation"])
        average, embeds = encode_panorama(self.model, ob, image_feat_size=self.image_feat_size, device=self.device)
        frozen = state["nav_inputs"]["vp_img_embeds"][0, 1:]
        if embeds.shape != frozen.shape or not torch.allclose(embeds, frozen, atol=1e-5, rtol=1e-5):
            raise ValueError("historical panorama re-encoding differs from frozen state")
        index = state["nav_inputs"]["gmap_vpids"][0].index(state["current_viewpoint"])
        if not torch.allclose(average, state["nav_inputs"]["gmap_img_embeds"][0, index], atol=1e-5, rtol=1e-5):
            raise ValueError("historical panorama mean differs from frozen graph")
        return ob

    def _rotate(self, desired):
        # MatterSim discrete actions use the sign, not the magnitude, of turns.
        for _ in range(16):
            current = int(self.walker.getState()[0].viewIndex)
            if current == desired:
                return
            delta = (desired % 12 - current % 12) % 12
            heading = (1 if delta <= 6 else -1) if delta else 0
            elevation = (desired // 12 > current // 12) - (desired // 12 < current // 12)
            self.walker.makeAction([0], [heading], [elevation])
        raise ValueError("simulator failed to rotate to the route edge orientation")

    def branch(self, scan, state, route):
        self.walker.newEpisode([scan], [state["current_viewpoint"]], [state["heading"]], [state["elevation"]])
        for source, target, point in zip(route["path"], route["path"][1:], route["edge_point_ids"]):
            if self.walker.getState()[0].location.viewpointId != source:
                raise ValueError("simulator left the declared branch route")
            self._rotate(point)
            actual = self.walker.getState()[0]
            indices = [i for i, loc in enumerate(actual.navigableLocations) if loc.viewpointId == target]
            if len(indices) != 1 or indices[0] == 0:
                raise ValueError("branch edge is not legally navigable at its candidate view")
            self.walker.makeAction([indices[0]], [0], [0])
            actual = self.walker.getState()[0]
            if actual.location.viewpointId != target or actual.viewIndex != point:
                raise ValueError("physical branch arrival differs from DUET edge semantics")
        ob = self._current_observation()
        if (ob["viewpoint"] != route["path"][-1] or not angle_close(ob["heading"], route["arrival_heading"])
                or abs(ob["elevation"] - route["arrival_elevation"]) > 1e-6):
            raise ValueError("branch endpoint orientation mismatch")
        primary = self._average(ob)
        matched = self.observation(scan, ob["viewpoint"], state["heading"], state["elevation"])
        return {"feature": primary, "headingmatched_feature": self._average(matched),
                "heading": ob["heading"], "elevation": ob["elevation"], "view_index": ob["viewIndex"],
                "headingmatched_heading": matched["heading"], "headingmatched_elevation": matched["elevation"],
                "target_panorama_tokens": len(ob["candidate"]) + 36 - len({c["pointId"] for c in ob["candidate"]}),
                "target_degree": len(ob["candidate"])}

    def _average(self, observation):
        key = (observation["scan"], observation["viewpoint"], observation["viewIndex"])
        if key not in self.average_cache:
            average, _ = encode_panorama(self.model, observation, image_feat_size=self.image_feat_size, device=self.device)
            self.average_cache[key] = average
        return self.average_cache[key].clone()
