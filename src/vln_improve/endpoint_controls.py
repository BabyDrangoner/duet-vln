"""Train-only natural/C2 endpoint controls, with immutable six-rollout groups.

Natural collection observes the upstream rollout without replacing its policy.
Goal labels and graph lengths are computed only after that rollout returns.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch

from .diagnostics import atomic_json
from .endpoint_pairs import SLOTS, content_hash, forced_rollout, nontext_hashes
from .endpoint_probe import FEATURE_DIM, FEATURE_SCHEMA
from .endpoint_probe import COMMON_KEYS
from .endpoint_probe import build_endpoint_features
from .protocol import file_sha256, object_sha256

SCHEMA = "duet_endpoint_controls_v2"
NATURAL_MODE = "original_duet_argmax_with_online_stop_and_retrospective_fallback"
C2_CONTEXTS = ("reference", "overshoot")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked controls file: {path}")
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def walk_length(trajectory, graph):
    """Measure every traversed edge, including unobserved fallback transit."""
    walk = [vp for segment in trajectory for vp in segment]
    if not walk:
        raise ValueError("empty actual trajectory")
    length = 0.0
    for a, b in zip(walk, walk[1:]):
        if a == b or not graph.has_edge(a, b):
            raise ValueError("actual trajectory contains a repeated or nonexistent edge")
        weight = float(graph[a][b]["weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("invalid actual edge length")
        length += weight
    return length


def _position_distance(a, b):
    # Exact arithmetic convention from frozen GraphMap.calc_position_distance.
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    return float(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))


def annotate_route_lengths(run, graph, *, forced):
    """Keep execution-graph numbers; measure the unchanged route on the official graph.

    MatterSim exposes float32-rounded JSON coordinates as Python floats. The
    discovered execution graph therefore has different edge weights from the
    original-connectivity evaluation graph. No tolerance equates those metrics.
    """
    original_total = float(run["actual_length_m"]) if forced else None
    original_prefixes = [float(s["prefix_length_m"]) for s in run["states"]] if forced else None
    positions = {s["viewpoint"]: s["position"] for s in run["states"]}
    walk = sum(run["trajectory"], [])
    official_total = walk_length(run["trajectory"], graph)  # Checks every actual edge.
    if any(v not in positions for v in walk):
        raise ValueError("actual route contains a node with no previous real observation")
    edges = [{"source": a, "target": b, "official_length_m": float(graph[a][b]["weight"]),
              "execution_position_length_m": _position_distance(positions[a], positions[b])}
             for a, b in zip(walk, walk[1:])]
    run["length_audit"] = {"schema": "actual_route_two_graph_lengths_v1",
        "primary_source": "official_connectivity_graph_edge_weights",
        "execution_position_source": "saved_real_observation_positions",
        "execution_graph_source": "frozen_forced_rollout" if forced else "not_captured_by_original_natural_rollout",
        "edges": edges}
    run["execution_graph_length_m"] = original_total
    run["execution_position_edge_sum_m"] = sum(e["execution_position_length_m"] for e in edges)
    run["actual_length_m"] = official_total
    for i, state in enumerate(run["states"]):
        state["execution_graph_prefix_length_m"] = original_prefixes[i] if forced else None
        state["prefix_length_m"] = walk_length(state["trajectory_prefix"], graph)


def validate_route_lengths(run, *, forced):
    audit = run.get("length_audit", {})
    expected_source = "frozen_forced_rollout" if forced else "not_captured_by_original_natural_rollout"
    if (audit.get("schema") != "actual_route_two_graph_lengths_v1"
            or audit.get("primary_source") != "official_connectivity_graph_edge_weights"
            or audit.get("execution_position_source") != "saved_real_observation_positions"
            or audit.get("execution_graph_source") != expected_source):
        raise ValueError("missing or changed route-length coordinate convention")
    walk = sum(run["trajectory"], [])
    edges = audit.get("edges", [])
    positions = {s["viewpoint"]: s["position"] for s in run["states"]}
    if len(edges) != len(walk) - 1 or any(v not in positions for v in walk):
        raise ValueError("route-length audit omits an actual edge/observation")
    for edge, a, b in zip(edges, walk, walk[1:]):
        if (edge.get("source") != a or edge.get("target") != b or a == b
                or any(type(edge.get(k)) not in (int, float) or not math.isfinite(edge[k]) or edge[k] <= 0
                       for k in ("official_length_m", "execution_position_length_m"))
                or edge["execution_position_length_m"] != _position_distance(positions[a], positions[b])):
            raise ValueError("route-length audit edge/source mismatch")
    if (run["actual_length_m"] != sum(e["official_length_m"] for e in edges)
            or run["execution_position_edge_sum_m"] != sum(e["execution_position_length_m"] for e in edges)):
        raise ValueError("route-length total differs from its complete edge sum")
    recorded = run.get("execution_graph_length_m")
    if forced:
        if type(recorded) not in (int, float) or not math.isfinite(recorded) or recorded < 0:
            raise ValueError("missing original forced execution-graph total")
        # This compares two sums using the same saved MatterSim coordinates.
        # Floyd graph accumulation may group additions differently. It never
        # compares that coordinate system with the official JSON-pose graph.
        if not math.isclose(recorded, run["execution_position_edge_sum_m"], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("original execution total differs from its same-coordinate edge sum")
    elif recorded is not None:
        raise ValueError("natural upstream rollout did not record an execution-graph total")
    for state in run["states"]:
        prefix = sum(state["trajectory_prefix"], [])
        n = len(prefix) - 1
        if (prefix != walk[:len(prefix)]
                or state["prefix_length_m"] != sum(e["official_length_m"] for e in edges[:n])):
            raise ValueError("official prefix length omits or changes actual traversed edges")
        original = state.get("execution_graph_prefix_length_m")
        if forced:
            if type(original) not in (int, float) or not math.isfinite(original) or original < 0:
                raise ValueError("missing original forced execution-graph prefix")
            execution_prefix = sum(e["execution_position_length_m"] for e in edges[:n])
            if not math.isclose(original, execution_prefix, rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError("original execution prefix differs from its same-coordinate edge sum")
        elif original is not None:
            raise ValueError("natural upstream rollout did not record an execution-graph prefix")
    if forced and run["states"][-1]["execution_graph_prefix_length_m"] != recorded:
        raise ValueError("original explicit-STOP total differs from its final execution prefix")


def natural_rollout(agent, record):
    """Capture only real decision observations; execute the original rollout."""
    if (agent.args.batch_size != 1 or agent.args.max_action_len != 15
            or agent.args.fusion != "dynamic" or not agent.args.enc_full_graph
            or getattr(agent.args, "act_visited_nodes", False)
            or getattr(agent, "decision_hook", None) is not None):
        raise ValueError("natural controls require the unmodified batch-one DUET policy")
    env = agent.env
    env.buffered_state_dict = {}
    agent.scanvp_cands = {}
    env.batch = [copy.deepcopy(record)]
    env.env.newEpisodes([record["scan"]], [record["path"][0]], [record["heading"]])
    states, features, probabilities = [], [], []
    captured = {}
    previous = {name: getattr(agent, name, None) for name in
                ("_panorama_feature_variable", "_language_variable", "decision_hook", "feedback")}
    started = time.monotonic()

    def panorama(obs):
        inputs = previous["_panorama_feature_variable"](obs)
        captured["panorama"] = inputs
        return inputs

    def language(obs):
        inputs = previous["_language_variable"](obs)
        captured["language_input_sha256"] = content_hash(inputs)
        return inputs

    def observe(nav, outputs, obs, ended, step, trajectory):
        if len(obs) != 1 or bool(ended[0]) or step != len(states):
            raise ValueError("unexpected natural control decision")
        ob = obs[0]
        if ob["instr_id"] != record["instr_id"] or ob["scan"] != record["scan"]:
            raise ValueError("natural rollout changed source instruction")
        legal = nav["gmap_masks"] & ~nav["gmap_visited_masks"]
        logits = outputs["fused_logits"]
        if (logits.shape != legal.shape or not bool(legal[0, 0])
                or not torch.isfinite(logits[legal]).all()
                or not torch.isneginf(logits[~legal]).all()):
            raise ValueError("baseline action masking differs")
        index = int(logits.argmax(-1)[0])
        stop = index == 0
        no_candidates = bool(nav["no_vp_left"][0])
        terminal = stop or no_candidates or step == agent.args.max_action_len - 1
        features.append(build_endpoint_features(nav, outputs)[0].cpu().clone())
        probabilities.append(float(logits.softmax(-1)[0, 0]))
        states.append({"step": step, "viewpoint": ob["viewpoint"],
            "heading": float(ob["heading"]), "elevation": float(ob["elevation"]),
            "view_index": int(ob["viewIndex"]), "position": [float(v) for v in ob["position"]],
            "trajectory_prefix": copy.deepcopy(trajectory[0]["path"]),
            "gmap_vpids": copy.deepcopy(nav["gmap_vpids"][0]),
            "vp_cand_vpids": copy.deepcopy(nav["vp_cand_vpids"][0]),
            "masks": {k: nav[k][0].detach().cpu().tolist() for k in
                      ("gmap_masks", "gmap_visited_masks", "vp_masks", "vp_nav_masks")},
            "gmap_step_ids": nav["gmap_step_ids"][0].detach().cpu().tolist(),
            "gmap_pair_dists": nav["gmap_pair_dists"][0].detach().cpu().tolist(),
            "baseline_argmax_index": index, "baseline_argmax_vpid": nav["gmap_vpids"][0][index],
            "executed_action": None if terminal else nav["gmap_vpids"][0][index],
            "termination": {"argmax_stop": stop, "no_candidates": no_candidates,
                            "step_limit": step == agent.args.max_action_len - 1},
            **nontext_hashes(captured.pop("panorama"), nav)})
        return outputs  # preserve the exact object used by upstream argmax/fallback

    agent._panorama_feature_variable = panorama
    agent._language_variable = language
    agent.decision_hook, agent.feedback = observe, "argmax"
    try:
        with torch.inference_mode():
            result = agent.rollout(train_ml=None, train_rl=False, reset=False)
    finally:
        for name, value in previous.items():
            setattr(agent, name, value)
    if len(result) != 1 or result[0]["instr_id"] != record["instr_id"] or not states:
        raise ValueError("natural rollout did not finish its source instruction")
    trajectory = copy.deepcopy(result[0]["path"])
    graph = env.graphs[record["scan"]]
    for state in states:
        state["prefix_length_m"] = walk_length(state["trajectory_prefix"], graph)
    prefix = states[-1]["trajectory_prefix"]
    if trajectory[:len(prefix)] != prefix or len(trajectory) not in (len(prefix), len(prefix) + 1):
        raise ValueError("unexpected upstream terminal/fallback path")
    return {"instr_id": record["instr_id"], "feature_schema": FEATURE_SCHEMA,
        "mode": NATURAL_MODE, "language_input_sha256": captured["language_input_sha256"],
        "instruction_text_sha256": hashlib.sha256(record["instruction"].encode()).hexdigest(),
        "features": torch.stack(features).float(), "states": states, "trajectory": trajectory,
        "actual_length_m": walk_length(trajectory, graph),
        "fallback_segment": trajectory[len(prefix):],
        "natural_stop_probability": torch.tensor(probabilities, dtype=torch.float32),
        "wall_seconds": time.monotonic() - started}


def collect_control_pair(agent, entry, records, graph_map_class, identity_sha256):
    pair = entry["pair"]
    result = {"schema": SCHEMA, "identity_sha256": identity_sha256, "feature_schema": FEATURE_SCHEMA,
              "pair": copy.deepcopy(pair), "control_entry": copy.deepcopy(entry), "natural": {}, "c2": {}}
    for index, slot in enumerate(SLOTS):
        record = records[pair["instr_ids"][index]]
        control = entry["controls"][slot]
        if (record["scan"] != pair["scan"] or str(record["path_id"]) != pair["path_ids"][index]
                or record["path"][-1] != pair["goal_vpids"][index]
                or control["original_path"] != record["path"] or control["instr_id"] != record["instr_id"]
                or control["goal"] != record["path"][-1] or control["heading"] != record["heading"]):
            raise ValueError("control source differs from the original train annotation")
        runs = {"natural": natural_rollout(agent, record)}
        for context, key in (("reference", "positive_history"), ("overshoot", "overshoot_history")):
            runs[context] = forced_rollout(agent, record, control[key]["observed_vpids"], graph_map_class)
        for context, run in runs.items():
            # No goal is read by natural policy capture. Supervision starts here.
            distances = torch.tensor([agent.env.shortest_distances[record["scan"]][s["viewpoint"]][record["path"][-1]]
                                      for s in run["states"]], dtype=torch.float64)
            run.update(instruction_slot=index, context_id=f"{slot}:{context}")
            run["labels"] = {"goal_vpid": record["path"][-1], "distance_to_goal": distances,
                             "within_success_radius": distances < 3}
            annotate_route_lengths(run, agent.env.graphs[record["scan"]], forced=context != "natural")
        result["natural"][slot] = runs.pop("natural")
        result["c2"][slot] = runs
    validate_control_pair(result, identity_sha256, entry)
    return result


def iter_runs(payload):
    for slot in SLOTS:
        yield slot, "natural", payload["natural"][slot]
        for context in C2_CONTEXTS:
            yield slot, context, payload["c2"][slot][context]


def validate_control_pair(payload, identity_sha256, entry):
    pair = entry["pair"]
    if (payload.get("schema") != SCHEMA or payload.get("identity_sha256") != identity_sha256
            or payload.get("pair") != pair or payload.get("control_entry") != entry
            or payload.get("feature_schema") != FEATURE_SCHEMA
            or set(payload.get("natural", {})) != set(SLOTS) or set(payload.get("c2", {})) != set(SLOTS)
            or any(set(payload["c2"][slot]) != set(C2_CONTEXTS) for slot in SLOTS)):
        raise ValueError("incomplete or mismatched six-rollout control group")
    for slot, context, run in iter_runs(payload):
        validate_route_lengths(run, forced=context != "natural")
        index = SLOTS.index(slot)
        n = len(run["states"])
        if (not 1 <= n <= 15 or run["instr_id"] != pair["instr_ids"][index]
                or run["instruction_slot"] != index or run["context_id"] != f"{slot}:{context}"
                or run["feature_schema"] != FEATURE_SCHEMA
                or not all(_sha(run[k]) for k in ("language_input_sha256", "instruction_text_sha256"))):
            raise ValueError("control run association/shape mismatch")
        for key, tensor, dtype, shape in (
                ("features", run["features"], torch.float32, (n, FEATURE_DIM)),
                ("probability", run["natural_stop_probability"], torch.float32, (n,)),
                ("distance", run["labels"]["distance_to_goal"], torch.float64, (n,)),
                ("success", run["labels"]["within_success_radius"], torch.bool, (n,))):
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != dtype
                    or tensor.shape != shape or tensor.requires_grad or not torch.isfinite(tensor).all()):
                raise ValueError(f"invalid control {key}")
        distance, success = run["labels"]["distance_to_goal"], run["labels"]["within_success_radius"]
        probability = run["natural_stop_probability"]
        if (run["labels"]["goal_vpid"] != pair["goal_vpids"][index] or (distance < 0).any()
                or not torch.equal(success, distance < 3) or ((probability < 0) | (probability > 1)).any()):
            raise ValueError("control labels/probabilities disagree")
        observed = [state["viewpoint"] for state in run["states"]]
        if observed[0] != pair["start"] or len(set(observed)) != n:
            raise ValueError("control observations are not unique actual nodes")
        for t, state in enumerate(run["states"]):
            prefix = state["trajectory_prefix"]
            if (state["step"] != t or not prefix or prefix[0] != [pair["start"]]
                    or prefix[-1][-1] != state["viewpoint"]
                    or any(not _sha(state[k]) for k in ("panorama_sha256", "navigation_without_text_sha256"))
                    or not all(isinstance(state[k], (float, int)) and math.isfinite(state[k]) for k in
                               ("heading", "elevation", "prefix_length_m"))):
                raise ValueError("invalid control observation/prefix")
            if t and prefix[:len(run["states"][t - 1]["trajectory_prefix"])] != run["states"][t - 1]["trajectory_prefix"]:
                raise ValueError("control prefixes changed history")
            if state["viewpoint"] == pair["goal_vpids"][index] and float(distance[t]) != 0:
                raise ValueError("original goal must have zero geometric distance")
        if not math.isfinite(run["actual_length_m"]) or run["actual_length_m"] < run["states"][-1]["prefix_length_m"]:
            raise ValueError("invalid complete route length")
        if context == "natural":
            if run["mode"] != NATURAL_MODE or run["trajectory"] != prefix + run["fallback_segment"]:
                raise ValueError("natural fallback was changed or omitted")
            if len(run["fallback_segment"]) > 1 or any(s["executed_action"] is None for s in run["states"][:-1]):
                raise ValueError("natural observations occurred after termination")
            for t, state in enumerate(run["states"]):
                index_argmax = state["baseline_argmax_index"]
                flags = state["termination"]
                expected_action = None if any(flags.values()) else state["baseline_argmax_vpid"]
                if (state["gmap_vpids"][index_argmax] != state["baseline_argmax_vpid"]
                        or flags["argmax_stop"] != (index_argmax == 0)
                        or flags["step_limit"] != (t == 14) or state["executed_action"] != expected_action
                        or (t < n - 1 and run["states"][t + 1]["viewpoint"] != expected_action)):
                    raise ValueError("natural baseline action/termination association changed")
            if run["states"][-1]["executed_action"] is not None:
                raise ValueError("natural rollout has no final stop")
        else:
            control_key = "positive_history" if context == "reference" else "overshoot_history"
            expected = entry["controls"][slot][control_key]["observed_vpids"]
            if (run["mode"] != "forced_training_history_no_retrospective_fallback" or observed != expected
                    or run["forced_actions"] != expected[1:] + [None] or run["trajectory"] != prefix
                    or bool(success[-1]) != (context == "reference")):
                raise ValueError("C2 fixed history/endpoint association changed")


class ControlStore:
    """Commit complete six-run groups locally and read back their Drive copy.

    Existing root seals are immutable. Restart repairs missing copies from a
    verified peer; it never recomputes a checksum to bless changed bytes.
    """
    def __init__(self, local, backup, identity, verify_backup):
        if (identity.get("schema") != SCHEMA or identity.get("split") not in {"train_fit", "train_dev"}
                or identity.get("usage") != ("training" if identity.get("split") == "train_fit" else "analysis_only")
                or not identity.get("selection") or identity.get("selection_sha256") != object_sha256(identity["selection"])
                or any(not _sha(identity.get(k)) for k in ("controls_report_sha256", "collection_config_sha256", "runtime_config_sha256"))
                or not identity.get("code_files") or any(not _sha(v) for v in identity["code_files"].values())
                or not identity.get("common_provenance") or not _sha(identity["common_provenance"].get("base_checkpoint_sha256"))):
            raise ValueError("controls require a complete immutable train-only identity")
        self.local, self.backup = Path(local).resolve(), Path(backup).resolve()
        if self.local.is_relative_to(self.backup) or self.backup.is_relative_to(self.local):
            raise ValueError("local and Drive control roots must be separate")
        self.identity, self.identity_sha256 = copy.deepcopy(identity), object_sha256(identity)
        self.verify_backup = verify_backup
        self.expected = {"pair-" + e["pair"]["selection_hash"]: e for e in identity["selection"]}
        if len(self.expected) != len(identity["selection"]):
            raise ValueError("duplicate control pair")
        self.seals = {}
        for root in (self.local, self.backup):
            if root == self.backup:
                verify_backup()
            root.mkdir(parents=True, exist_ok=True)
            wanted = {"schema": SCHEMA, "identity": identity, "identity_sha256": self.identity_sha256}
            marker = root / "COLLECTION.json"
            if marker.exists() or marker.is_symlink():
                if _json(marker) != wanted:
                    raise ValueError("controls collection identity changed")
            else:
                if list(root.glob("pair-*")) or (root / "manifest.json").exists() or (root / "COMMITTED.json").exists():
                    raise ValueError("orphan control files without collection identity")
                atomic_json(marker, wanted)
            if any(p.name not in self.expected for p in root.glob("pair-*")):
                raise ValueError("unexpected control pair")
            self.seals[root] = self._read_root(root)
        if all(self.seals.values()) and self.seals[self.local] != self.seals[self.backup]:
            raise ValueError("local/Drive root seals disagree")

    def _read_root(self, root):
        files = (root / "manifest.json", root / "COMMITTED.json")
        if not any(p.exists() or p.is_symlink() for p in files):
            return None
        manifest, commit = (_json(p) for p in files)
        if (commit != {"manifest_sha256": file_sha256(files[0])}
                or manifest.get("schema") != SCHEMA or manifest.get("identity_sha256") != self.identity_sha256
                or manifest.get("split") != self.identity["split"]
                or [f["name"] for f in manifest.get("files", [])] != list(self.expected)):
            raise ValueError("controls root commit/identity differs")
        return manifest

    def verify(self, directory, entry):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("missing or symlinked control pair")
        manifest = _json(directory / "manifest.json")
        if _json(directory / "COMMITTED.json") != {"manifest_sha256": file_sha256(directory / "manifest.json")}:
            raise ValueError("control pair commit mismatch")
        data = directory / "data.pt"
        if data.is_symlink() or not data.is_file():
            raise ValueError("missing or symlinked control payload")
        if (manifest.get("schema") != SCHEMA or manifest.get("identity_sha256") != self.identity_sha256
                or manifest.get("entry_sha256") != object_sha256(entry) or manifest.get("data_sha256") != file_sha256(data)):
            raise ValueError("control pair data/identity mismatch")
        payload = torch.load(data, map_location="cpu", weights_only=True)
        validate_control_pair(payload, self.identity_sha256, entry)
        counts = self._counts(payload)
        if any(manifest.get(k) != value for k, value in counts.items()):
            raise ValueError("control manifest counts differ")
        return manifest

    @staticmethod
    def _counts(payload):
        return {"rollouts": 6, "natural_rollouts": 2, "c2_rollouts": 4,
                "states": sum(len(r["states"]) for _, _, r in iter_runs(payload)),
                "natural_states": sum(len(r["states"]) for r in payload["natural"].values()),
                "c2_states": sum(len(r["states"]) for runs in payload["c2"].values() for r in runs.values())}

    def _copy(self, source, target, entry):
        original = self.verify(source, entry)
        stage = target.parent / (".control-stage-" + uuid.uuid4().hex)
        try:
            shutil.copytree(source, stage)
            if self.verify(stage, entry) != original:
                raise ValueError("Drive control readback mismatch")
            if target.exists() or target.is_symlink():
                raise ValueError("concurrent control writer")
            stage.rename(target)
            return self.verify(target, entry)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def find(self, entry):
        name = "pair-" + entry["pair"]["selection_hash"]
        if self.expected.get(name) != entry:
            raise ValueError("control is outside immutable selection")
        self.verify_backup()
        found = []
        # A surviving peer's root seal also constrains an unsealed/recreated
        # local copy. Never use an unpinned local group to repair a sealed Drive.
        pinned_root = self.seals[self.local] or self.seals[self.backup]
        for root in (self.local, self.backup):
            path = root / name
            item = self.verify(path, entry) if path.exists() or path.is_symlink() else None
            seal = pinned_root
            if item is not None and seal is not None:
                pinned = next(f for f in seal["files"] if f["name"] == name)
                if pinned != {"name": name, "manifest_sha256": file_sha256(path / "manifest.json"), **item}:
                    raise ValueError("pair differs from the existing immutable root seal")
            found.append(item)
        if found[0] is not None and found[1] is not None and found[0] != found[1]:
            raise ValueError("local/Drive control pair bytes disagree")
        if not any(found) and any(self.seals.values()):
            raise ValueError("sealed control group missing from both copies")
        if found[0] is None and found[1] is not None:
            return self._copy(self.backup / name, self.local / name, entry)
        if found[0] is not None and found[1] is None:
            self._copy(self.local / name, self.backup / name, entry)
        return found[0]

    def commit(self, payload):
        entry = payload["control_entry"]
        validate_control_pair(payload, self.identity_sha256, entry)
        if self.find(entry) is not None:
            raise ValueError("refusing to replace an existing control group")
        stage = self.local / (".control-stage-" + uuid.uuid4().hex)
        stage.mkdir()
        try:
            with (stage / "data.pt").open("xb") as stream:
                torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
            manifest = {"schema": SCHEMA, "identity_sha256": self.identity_sha256,
                        "entry_sha256": object_sha256(entry), "data_sha256": file_sha256(stage / "data.pt"),
                        **self._counts(payload)}
            atomic_json(stage / "manifest.json", manifest)
            atomic_json(stage / "COMMITTED.json", {"manifest_sha256": file_sha256(stage / "manifest.json")})
            self.verify(stage, entry)
            target = self.local / ("pair-" + entry["pair"]["selection_hash"])
            if target.exists() or target.is_symlink():
                raise ValueError("concurrent control writer")
            stage.rename(target)
            self.verify_backup()
            self._copy(target, self.backup / target.name, entry)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def seal(self, resources):
        # Check current root bytes too, not only the in-memory seal from startup.
        for root in (self.local, self.backup):
            current = self._read_root(root)
            if current != self.seals[root]:
                raise ValueError("existing root seal changed during collection")
        files = []
        for name, entry in self.expected.items():
            manifest = self.find(entry)
            if manifest is None:
                raise ValueError("cannot seal incomplete controls")
            files.append({"name": name, "manifest_sha256": file_sha256(self.local / name / "manifest.json"), **manifest})
        previous = self.seals[self.local] or self.seals[self.backup]
        if previous is not None:
            if previous["files"] != files:
                raise ValueError("refusing to re-sign changed group manifests")
            result = previous  # preserve original resources and every original byte
        else:
            result = {"schema": SCHEMA, "identity_sha256": self.identity_sha256,
                "split": self.identity["split"], "usage": self.identity["usage"], "files": files,
                "pairs": len(files), "rollouts": 6 * len(files), "natural_rollouts": 2 * len(files),
                "c2_rollouts": 4 * len(files), "states": sum(f["states"] for f in files),
                "resources": resources, "navigation_metrics": None,
                "interpretation": "train-only control features; C2 forced histories are not navigation evaluations"}
        for root in (self.local, self.backup):
            if root == self.backup:
                self.verify_backup()
            if self.seals[root] is None:
                atomic_json(root / "manifest.json", result)
                atomic_json(root / "COMMITTED.json", {"manifest_sha256": file_sha256(root / "manifest.json")})
                self.seals[root] = self._read_root(root)
            if self.seals[root] != result:
                raise ValueError("Drive root seal readback differs")
        if file_sha256(self.local / "manifest.json") != file_sha256(self.backup / "manifest.json"):
            raise ValueError("Drive root manifest byte readback differs")
        return result


@dataclass(frozen=True)
class ControlCache:
    root: Path
    manifest: dict
    manifest_sha256: str
    identity: dict
    identity_sha256: str
    data_sha256: str
    groups: tuple

    @property
    def common_identity(self):
        return {key: copy.deepcopy(self.identity[key]) for key in
                ("common_provenance", "feature_schema", "feature_dim")}


def load_control_cache(root, *, expected_split, expected_identity_sha256=None):
    """Read a complete sealed controls cache without repairing or rewriting it."""
    root = Path(root)
    if expected_split not in {"train_fit", "train_dev"} or root.is_symlink() or not root.is_dir():
        raise ValueError("controls loader accepts only ordinary train-only cache directories")
    collection = _json(root / "COLLECTION.json")
    identity = collection["identity"]
    digest = object_sha256(identity)
    if (collection != {"schema": SCHEMA, "identity": identity, "identity_sha256": digest}
            or identity.get("schema") != SCHEMA or identity.get("split") != expected_split
            or identity.get("usage") != ("training" if expected_split == "train_fit" else "analysis_only")
            or (expected_identity_sha256 is not None and expected_identity_sha256 != digest)
            or identity.get("feature_schema") != FEATURE_SCHEMA or identity.get("feature_dim") != FEATURE_DIM
            or identity.get("selection_sha256") != object_sha256(identity.get("selection"))
            or not identity.get("selection") or not identity.get("code_files")
            or any(not _sha(x) for x in identity["code_files"].values())
            or any(not _sha(identity.get(k)) for k in ("controls_report_sha256", "collection_config_sha256", "runtime_config_sha256"))):
        raise ValueError("controls cache identity/split differs")
    common = identity.get("common_provenance")
    if (not isinstance(common, dict) or set(common) != COMMON_KEYS
            or any(not _sha(common[k]) for k in ("base_checkpoint_sha256", "feature_sha256", "annotation_sha256", "connectivity_sha256"))):
        raise ValueError("invalid controls common provenance")
    expected = {"pair-" + e["pair"]["selection_hash"]: e for e in identity["selection"]}
    if len(expected) != len(identity["selection"]) or set(p.name for p in root.glob("pair-*")) != set(expected):
        raise ValueError("incomplete or extra controls groups")
    # Reuse exactly the read-only verification path used during collection.
    verifier = object.__new__(ControlStore)
    verifier.identity, verifier.identity_sha256, verifier.expected = identity, digest, expected
    manifest = verifier._read_root(root)
    if manifest is None:
        raise ValueError("controls cache is not sealed")
    groups, files = [], []
    for name, entry in expected.items():
        item = verifier.verify(root / name, entry)
        files.append({"name": name, "manifest_sha256": file_sha256(root / name / "manifest.json"), **item})
        groups.append(torch.load(root / name / "data.pt", map_location="cpu", weights_only=True))
    if (manifest["files"] != files or manifest.get("pairs") != len(files)
            or manifest.get("rollouts") != 6 * len(files) or manifest.get("natural_rollouts") != 2 * len(files)
            or manifest.get("c2_rollouts") != 4 * len(files) or manifest.get("states") != sum(f["states"] for f in files)):
        raise ValueError("sealed controls manifest does not match actual group bytes/counts")
    data_sha = object_sha256({"identity_sha256": digest, "ordered_files": [
        {key: item[key] for key in ("name", "manifest_sha256", "data_sha256")} for item in files]})
    return ControlCache(root, manifest, file_sha256(root / "manifest.json"), identity, digest, data_sha, tuple(groups))
