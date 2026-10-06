"""Forced train-only endpoint-pair collection and atomic four-rollout storage."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import torch

from .diagnostics import atomic_json
from .endpoint_probe import FEATURE_DIM, FEATURE_SCHEMA, build_endpoint_features
from .protocol import file_sha256, object_sha256

SCHEMA = "duet_endpoint_pairs_v1"
ORDERS = ("A_then_B", "B_then_A")
SLOTS = ("A", "B")
TEXT_KEYS = {"txt_embeds", "txt_masks"}
NAV_KEYS = {"gmap_vpids", "gmap_img_embeds", "gmap_step_ids", "gmap_pos_fts", "gmap_visited_masks",
            "gmap_pair_dists", "gmap_masks", "no_vp_left", "vp_img_embeds", "vp_pos_fts",
            "vp_masks", "vp_nav_masks", "vp_cand_vpids", "txt_embeds", "txt_masks"}
PANO_KEYS = {"view_img_fts", "loc_fts", "nav_types", "view_lens", "cand_vpids"}


def content_hash(value):
    """Exact dtype/shape/byte identity; no tensor serialization timestamps."""
    digest = hashlib.sha256()
    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError("nonfinite input/feature tensor")
            digest.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item):
                if not isinstance(key, str):
                    raise ValueError("hash dictionary keys must be strings")
                add(key); add(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"list[")
            for child in item:
                add(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b";")
    add(value)
    return digest.hexdigest()


def nontext_hashes(panorama, navigation):
    if set(panorama) != PANO_KEYS or set(navigation) != NAV_KEYS:
        raise ValueError("unexpected DUET input keys; audit before excluding any field")
    return {"panorama_sha256": content_hash(panorama),
            "navigation_without_text_sha256": content_hash({k: v for k, v in navigation.items() if k not in TEXT_KEYS})}


def selected_pairs(report, split, count):
    if split not in {"train_fit", "train_dev"} or type(count) is not int or count < 1:
        raise ValueError("invalid train-only pair selection")
    if report.get("schema") != "duet_endpoint_pair_coverage_v1" or report.get("coverage_pass") is not True:
        raise ValueError("D3 coverage gate has not passed")
    source = report["splits"][split]
    rows = source["primary_path_disjoint_manifest"]
    if source["primary_manifest_sha256"] != object_sha256(rows):
        raise ValueError("D3 primary manifest checksum differs")
    if len(rows) < count:
        raise ValueError("fixed pilot pair budget exceeds D3 coverage")
    seen, hashes = set(), set()
    for row in rows:
        key = row["selection_hash"]
        paths = {(row["scan"], pid) for pid in row["path_ids"]}
        if len(paths) != 2 or not seen.isdisjoint(paths) or key in hashes:
            raise ValueError("D3 primary pairs are not unique and path-disjoint")
        if row["heading_difference_deg"] > 0.000001 or row["goal_separation_m"] <= 6:
            raise ValueError("D3 primary geometry/heading rule differs")
        for order in ORDERS:
            vps = row["histories"][order]["observed_vpids"]
            if not 2 <= len(vps) <= 15 or len(set(vps)) != len(vps) or vps[0] != row["start"]:
                raise ValueError("D3 observation sequence invalid")
            if not set(row["goal_vpids"]).issubset(vps):
                raise ValueError("both goals must actually be observed")
        seen.update(paths); hashes.add(key)
    return copy.deepcopy(sorted(rows, key=lambda r: r["selection_hash"])[:count])


def _finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_pair(payload, identity_sha256, expected_pair):
    if (payload.get("schema") != SCHEMA or payload.get("identity_sha256") != identity_sha256
            or payload.get("pair") != expected_pair or payload.get("feature_schema") != FEATURE_SCHEMA
            or set(payload.get("rollouts", {})) != set(ORDERS)):
        raise ValueError("pair identity/schema differs")
    for order in ORDERS:
        runs = payload["rollouts"][order]
        if set(runs) != set(SLOTS):
            raise ValueError("incomplete four-rollout pair")
        vps = expected_pair["histories"][order]["observed_vpids"]
        for index, slot in enumerate(SLOTS):
            run = runs[slot]
            if (run["instr_id"] != expected_pair["instr_ids"][index] or run["instruction_slot"] != index
                    or run["feature_schema"] != FEATURE_SCHEMA or run["mode"] != "forced_training_history_no_retrospective_fallback"
                    or run["forced_actions"] != vps[1:] + [None]
                    or len(run["states"]) != len(vps)
                    or run["trajectory"] != run["states"][-1]["trajectory_prefix"]):
                raise ValueError("rollout association, endpoint, or forced action differs")
            n, features = len(vps), run["features"]
            if (not isinstance(features, torch.Tensor) or features.device.type != "cpu"
                    or features.dtype != torch.float32 or features.shape != (n, FEATURE_DIM)
                    or features.requires_grad or not torch.isfinite(features).all()):
                raise ValueError("invalid paired endpoint features")
            labels = run["labels"]
            distances, success = labels["distance_to_goals"], labels["within_success_radius"]
            if (labels["goal_vpids"] != expected_pair["goal_vpids"]
                    or not all(isinstance(x, torch.Tensor) and x.device.type == "cpu" for x in (distances, success))
                    or distances.dtype != torch.float64 or distances.shape != (n, 2)
                    or success.dtype != torch.bool or success.shape != (n, 2)
                    or not torch.isfinite(distances).all() or (distances < 0).any()
                    or not torch.equal(success, distances < 3) or success.all(dim=1).any()):
                raise ValueError("paired labels are invalid/overlapping")
            probability = run["natural_stop_probability"]
            if (not isinstance(probability, torch.Tensor) or probability.device.type != "cpu"
                    or probability.dtype != torch.float32 or probability.shape != (n,)
                    or not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any()):
                raise ValueError("invalid unmodified baseline STOP probability")
            for t, state in enumerate(run["states"]):
                if (state["step"] != t or state["viewpoint"] != vps[t]
                        or not _finite_number(state["heading"]) or not _finite_number(state["elevation"])
                        or not state["trajectory_prefix"] or state["trajectory_prefix"][-1][-1] != vps[t]
                        or state["trajectory_prefix"][0] != [vps[0]]):
                    raise ValueError("invalid state/prefix/angle sequence")
                for key in ("panorama_sha256", "navigation_without_text_sha256"):
                    sha = state[key]
                    if not isinstance(sha, str) or len(sha) != 64 or any(x not in "0123456789abcdef" for x in sha):
                        raise ValueError("invalid nontext SHA")
            for j, goal in enumerate(expected_pair["goal_vpids"]):
                t = vps.index(goal)
                if float(distances[t, j]) != 0 or not bool(success[t, j]):
                    raise ValueError("the true goal node must have zero distance and be positive")
        # Every saved state field is language independent; language identity is
        # outside this block. Feature values are expected to be text conditioned.
        if content_hash(runs["A"]["states"]) != content_hash(runs["B"]["states"]):
            raise ValueError("shared-history nontext/angle/trajectory parity failed")
        if content_hash(runs["A"]["labels"]) != content_hash(runs["B"]["labels"]):
            raise ValueError("same physical history has different geometric labels")
        if runs["A"]["language_input_sha256"] == runs["B"]["language_input_sha256"]:
            raise ValueError("paired encoded instructions are identical; do not train contradictory labels")


def forced_rollout(agent, record, observed_vpids, graph_map_class):
    """Use upstream feature helpers and action executor, with a fixed candidate sequence.

    This intentionally bypasses the natural stop/fallback decision. It does not
    produce a baseline or method navigation score. No goal labels enter inputs.
    """
    env = agent.env
    if agent.args.batch_size != 1 or len(observed_vpids) > agent.args.max_action_len:
        raise ValueError("forced rollout requires batch one and bounded history")
    if record["path"][0] != observed_vpids[0] or len(set(observed_vpids)) != len(observed_vpids):
        raise ValueError("forced history start/uniqueness mismatch")
    env.buffered_state_dict = {}
    agent.scanvp_cands = {}
    env.batch = [copy.deepcopy(record)]
    env.env.newEpisodes([record["scan"]], [record["path"][0]], [record["heading"]])
    obs = env._get_obs()
    agent._update_scanvp_cands(obs)
    gmap = graph_map_class(obs[0]["viewpoint"])
    gmap.update_graph(obs[0])
    trajectory = [{"instr_id": record["instr_id"], "path": [[obs[0]["viewpoint"]]], "details": {}}]
    states, features, probabilities = [], [], []
    prefix_length = 0.0
    started = time.monotonic()
    with torch.inference_mode():
        language = agent._language_variable(obs)
        language_sha = content_hash(language)
        txt_embeds = agent.vln_bert("language", language)
        for t, current in enumerate(observed_vpids):
            ob = obs[0]
            if ob["viewpoint"] != current or ob["instr_id"] != record["instr_id"] or ob["scan"] != record["scan"]:
                raise ValueError("simulator did not follow the fixed observation sequence")
            gmap.node_step_ids[current] = t + 1
            pano_inputs = agent._panorama_feature_variable(obs)
            pano_embeds, pano_masks = agent.vln_bert("panorama", pano_inputs)
            avg = (pano_embeds * pano_masks.unsqueeze(2)).sum(1) / pano_masks.sum(1, keepdim=True)
            gmap.update_node_embed(current, avg[0], rewrite=True)
            for j, candidate in enumerate(pano_inputs["cand_vpids"][0]):
                if not gmap.graph.visited(candidate):
                    gmap.update_node_embed(candidate, pano_embeds[0, j])
            nav = agent._nav_gmap_variable(obs, [gmap])
            nav.update(agent._nav_vp_variable(obs, [gmap], pano_embeds, pano_inputs["cand_vpids"],
                                             pano_inputs["view_lens"], pano_inputs["nav_types"]))
            nav.update(txt_embeds=txt_embeds, txt_masks=language["txt_masks"])
            hashes = nontext_hashes(pano_inputs, nav)
            outputs = agent.vln_bert("navigation", nav)
            if hashes != nontext_hashes(pano_inputs, nav):
                raise ValueError("model mutated the observation inputs")
            features.append(build_endpoint_features(nav, outputs)[0].cpu().clone())
            legal = nav["gmap_masks"] & ~nav["gmap_visited_masks"]
            logits = outputs["fused_logits"]
            if logits.shape != legal.shape or not bool(legal[0, 0]) or not torch.isfinite(logits[legal]).all():
                raise ValueError("invalid baseline legal scores")
            probabilities.append(float(logits.masked_fill(~legal, -torch.inf).softmax(-1)[0, 0]))
            states.append({"step": t, "viewpoint": current, "heading": float(ob["heading"]),
                "elevation": float(ob["elevation"]), "view_index": int(ob["viewIndex"]),
                "position": [float(x) for x in ob["position"]],
                "trajectory_prefix": copy.deepcopy(trajectory[0]["path"]),
                "prefix_length_m": prefix_length,
                "gmap_vpids": copy.deepcopy(nav["gmap_vpids"][0]),
                "vp_cand_vpids": copy.deepcopy(nav["vp_cand_vpids"][0]),
                "masks": {key: nav[key][0].detach().cpu().tolist() for key in
                          ("gmap_masks", "gmap_visited_masks", "vp_masks", "vp_nav_masks")},
                "gmap_step_ids": nav["gmap_step_ids"][0].detach().cpu().tolist(),
                "gmap_pair_dists": nav["gmap_pair_dists"][0].detach().cpu().tolist(), **hashes})
            target = observed_vpids[t + 1] if t + 1 < len(observed_vpids) else None
            if target is not None:
                ids = nav["gmap_vpids"][0]
                if target not in ids or not bool(legal[0, ids.index(target)]) or gmap.graph.visited(target):
                    raise ValueError("forced target is not a legal unvisited candidate")
                segment = gmap.graph.path(current, target)
                if not segment or segment[-1] != target or any(not gmap.graph.visited(v) for v in segment[:-1]):
                    raise ValueError("forced path has an unobserved intermediate node")
            before = copy.deepcopy(trajectory[0]["path"])
            agent.make_equiv_action([target], [gmap], obs, trajectory)
            if target is None:
                if trajectory[0]["path"] != before:
                    raise ValueError("explicit final STOP must not append a fallback route")
            else:
                if trajectory[0]["path"] != before + [segment]:
                    raise ValueError("upstream executor differs from its discovered-map route")
                prefix_length += float(gmap.graph.distance(current, target))
                obs = env._get_obs()
                agent._update_scanvp_cands(obs)
                gmap.update_graph(obs[0])
    return {"instr_id": record["instr_id"], "feature_schema": FEATURE_SCHEMA,
            "mode": "forced_training_history_no_retrospective_fallback", "language_input_sha256": language_sha,
            "instruction_text_sha256": hashlib.sha256(record["instruction"].encode()).hexdigest(),
            "features": torch.stack(features).float(), "states": states,
            "trajectory": copy.deepcopy(trajectory[0]["path"]), "forced_actions": observed_vpids[1:] + [None],
            "actual_length_m": prefix_length,
            "natural_stop_probability": torch.tensor(probabilities, dtype=torch.float32),
            "wall_seconds": time.monotonic() - started}


def collect_pair(agent, pair, records, graph_map_class, identity_sha256):
    runs = {}
    for order in ORDERS:
        runs[order] = {}
        vps = pair["histories"][order]["observed_vpids"]
        for index, slot in enumerate(SLOTS):
            record = records[pair["instr_ids"][index]]
            run = forced_rollout(agent, record, vps, graph_map_class)
            # Goal information is accessed only after the feature rollout.
            distance = torch.tensor([[agent.env.shortest_distances[pair["scan"]][v][g]
                                     for g in pair["goal_vpids"]] for v in vps], dtype=torch.float64)
            run["instruction_slot"] = index
            run["labels"] = {"goal_vpids": pair["goal_vpids"], "distance_to_goals": distance,
                             "within_success_radius": distance < 3}
            runs[order][slot] = run
    payload = {"schema": SCHEMA, "identity_sha256": identity_sha256, "pair": pair,
               "feature_schema": FEATURE_SCHEMA, "rollouts": runs}
    validate_pair(payload, identity_sha256, pair)
    return payload


class PairStore:
    """Only complete, validated groups are durable; a partial group is retried."""
    def __init__(self, local, backup, identity, verify_backup):
        split = identity.get("split")
        if (identity.get("schema") != SCHEMA or split not in {"train_fit", "train_dev"}
                or identity.get("usage") != ("training" if split == "train_fit" else "analysis_only")
                or identity.get("selection_sha256") != object_sha256(identity.get("selection"))):
            raise ValueError("pair store requires a valid train-only collection identity")
        self.local, self.backup = Path(local).resolve(), Path(backup).resolve()
        if self.local.is_relative_to(self.backup) or self.backup.is_relative_to(self.local):
            raise ValueError("pair local and backup roots must be separate")
        self.identity, self.identity_sha256 = copy.deepcopy(identity), object_sha256(identity)
        self.verify_backup = verify_backup
        self.expected = {"pair-" + p["selection_hash"]: p for p in identity["selection"]}
        if len(self.expected) != len(identity["selection"]):
            raise ValueError("duplicate selected pairs")
        for root in (self.local, self.backup):
            if root == self.backup:
                verify_backup()
            root.mkdir(parents=True, exist_ok=True)
            marker = root / "COLLECTION.json"
            wanted = {"schema": SCHEMA, "identity": identity, "identity_sha256": self.identity_sha256}
            if marker.exists():
                if marker.is_symlink() or json.loads(marker.read_text()) != wanted:
                    raise ValueError("paired collection identity changed")
            else:
                if list(root.glob("pair-*")) or (root / "manifest.json").exists():
                    raise ValueError("orphan paired data without identity")
                atomic_json(marker, wanted)
            if any(p.name not in self.expected for p in root.glob("pair-*")):
                raise ValueError("unexpected pair outside the fixed selection")

    def verify(self, directory, pair):
        directory = Path(directory)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("pair directory missing/symlinked")
        for name in ("manifest.json", "COMMITTED.json", "data.pt"):
            if (directory / name).is_symlink() or not (directory / name).is_file():
                raise ValueError("pair is partial or symlinked")
        manifest = json.loads((directory / "manifest.json").read_text())
        if json.loads((directory / "COMMITTED.json").read_text()) != {"manifest_sha256": file_sha256(directory / "manifest.json")}:
            raise ValueError("pair commit checksum differs")
        if (manifest.get("schema") != SCHEMA or manifest.get("identity_sha256") != self.identity_sha256
                or manifest.get("pair_sha256") != object_sha256(pair)
                or manifest.get("data_sha256") != file_sha256(directory / "data.pt")):
            raise ValueError("pair manifest/data identity differs")
        payload = torch.load(directory / "data.pt", map_location="cpu", weights_only=True)
        validate_pair(payload, self.identity_sha256, pair)
        if (manifest.get("rollouts") != 4 or manifest.get("shared_history_exact_parity") is not True
                or manifest.get("states") != sum(len(r["states"]) for h in payload["rollouts"].values() for r in h.values())):
            raise ValueError("pair manifest counts/parity differ from verified contents")
        return manifest

    def _copy(self, source, target, pair):
        original = self.verify(source, pair)
        stage = target.parent / (".pair-stage-" + uuid.uuid4().hex)
        try:
            shutil.copytree(source, stage)
            if self.verify(stage, pair) != original:
                raise ValueError("pair Drive readback SHA mismatch")
            if target.exists():
                raise ValueError("concurrent pair writer detected")
            stage.rename(target)
            return self.verify(target, pair)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def find(self, pair):
        name = "pair-" + pair["selection_hash"]
        if self.expected.get(name) != pair:
            raise ValueError("pair is outside immutable selection")
        local, backup = self.local / name, self.backup / name
        self.verify_backup()
        lm = self.verify(local, pair) if local.exists() or local.is_symlink() else None
        bm = self.verify(backup, pair) if backup.exists() or backup.is_symlink() else None
        if lm is not None and bm is not None and lm != bm:
            raise ValueError("local/Drive pair bytes disagree")
        if lm is not None and bm is None:
            self._copy(local, backup, pair)
        elif lm is None and bm is not None:
            lm = self._copy(backup, local, pair)
        return lm

    def commit(self, payload):
        pair = payload["pair"]
        validate_pair(payload, self.identity_sha256, pair)
        if self.find(pair) is not None:
            raise ValueError("refusing to replace a complete pair")
        stage = self.local / (".pair-stage-" + uuid.uuid4().hex)
        stage.mkdir()
        try:
            with (stage / "data.pt").open("xb") as stream:
                torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
            manifest = {"schema": SCHEMA, "identity_sha256": self.identity_sha256,
                        "pair_sha256": object_sha256(pair), "data_sha256": file_sha256(stage / "data.pt"),
                        "rollouts": 4, "states": sum(len(r["states"]) for h in payload["rollouts"].values() for r in h.values()),
                        "shared_history_exact_parity": True}
            atomic_json(stage / "manifest.json", manifest)
            atomic_json(stage / "COMMITTED.json", {"manifest_sha256": file_sha256(stage / "manifest.json")})
            self.verify(stage, pair)
            target = self.local / ("pair-" + pair["selection_hash"])
            if target.exists():
                raise ValueError("concurrent pair writer detected")
            stage.rename(target)
            self.verify_backup()
            self._copy(target, self.backup / target.name, pair)
            return manifest
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def seal(self, resources):
        files = []
        for pair in self.identity["selection"]:
            manifest = self.find(pair)
            if manifest is None:
                raise ValueError("cannot seal an incomplete pair selection")
            name = "pair-" + pair["selection_hash"]
            files.append({"name": name, "manifest_sha256": file_sha256(self.local / name / "manifest.json"), **manifest})
        result = {"schema": SCHEMA, "identity_sha256": self.identity_sha256, "split": self.identity["split"],
                  "usage": self.identity["usage"], "files": files, "pairs": len(files), "rollouts": 4 * len(files),
                  "states": sum(f["states"] for f in files), "all_shared_history_exact_parity": True,
                  "resources": resources, "navigation_metrics": None,
                  "interpretation": "forced training trajectories; no navigation performance or novelty claim"}
        for root in (self.local, self.backup):
            if root == self.backup:
                self.verify_backup()
            atomic_json(root / "manifest.json", result)
            atomic_json(root / "COMMITTED.json", {"manifest_sha256": file_sha256(root / "manifest.json")})
        if file_sha256(self.local / "manifest.json") != file_sha256(self.backup / "manifest.json"):
            raise ValueError("collection manifest Drive readback differs")
        return result
