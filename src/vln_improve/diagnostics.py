"""Read-only, episode-atomic DUET evidence diagnostics (M0 collection only)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Sequence
import uuid

import torch

from .evidence import EpisodeEvidenceMemory
from .protocol import file_sha256, object_sha256


SCHEMA = "duet_diagnostic_episode_v1"
NAVIGATION_KEYS = (
    "txt_embeds", "txt_masks", "gmap_img_embeds", "gmap_step_ids", "gmap_pos_fts",
    "gmap_masks", "gmap_pair_dists", "gmap_visited_masks", "gmap_vpids",
    "vp_img_embeds", "vp_pos_fts", "vp_masks", "vp_nav_masks", "vp_obj_masks",
    "vp_cand_vpids",
)


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_copy(item) for item in value]
    return copy.deepcopy(value)


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    _write(temporary, _json(value))
    os.replace(temporary, path)


def select_diagnostic_records(records: Sequence[dict], *, per_scan: int = 4,
                              seed: int = 0, max_scans: int | None = None,
                              max_episodes: int | None = None) -> list[dict]:
    """Choose one instruction per path, with deterministic scan-balanced order."""
    for name, value in (("per_scan", per_scan), ("max_scans", max_scans), ("max_episodes", max_episodes)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be positive")
    grouped: dict[str, dict[str, list[dict]]] = {}
    for row in records:
        grouped.setdefault(row["scan"], {}).setdefault(str(row["path_id"]), []).append(row)
    scans = sorted(grouped, key=lambda scan: object_sha256([seed, "scan", scan]))
    if max_scans is not None:
        scans = scans[:max_scans]
    by_scan = {}
    for scan in scans:
        paths = sorted(grouped[scan], key=lambda path: object_sha256([seed, scan, path]))[:per_scan]
        by_scan[scan] = [min(grouped[scan][path], key=lambda row: object_sha256([seed, row["instr_id"]]))
                         for path in paths]
    selected = [by_scan[scan][index] for index in range(per_scan) for scan in scans
                if index < len(by_scan[scan])]
    return selected if max_episodes is None else selected[:max_episodes]


class DiagnosticStore:
    """Immutable complete episodes; a partial episode is recollected after restart.

    The CLI validates actual Drive mounting. Filesystem verification here does
    not claim an independent service-side persistence acknowledgement.
    """

    def __init__(self, local: str | Path, identity: dict, *, backup: str | Path | None = None,
                 verify_backup: Callable[[], Any] | None = None):
        self.local = Path(local).resolve()
        self.backup = Path(backup).resolve() if backup is not None else None
        if self.backup is not None and (self.local.is_relative_to(self.backup) or self.backup.is_relative_to(self.local)):
            raise ValueError("local and backup diagnostic directories must be separate")
        self.identity = cpu_copy(identity)
        self.identity_sha256 = object_sha256(identity)
        self.verify_backup = verify_backup
        for root in (self.local, self.backup):
            if root is None:
                continue
            if root == self.backup and self.verify_backup:
                self.verify_backup()
            root.mkdir(parents=True, exist_ok=True)
            marker = root / "COLLECTION.json"
            expected = {"schema": SCHEMA, "identity_sha256": self.identity_sha256, "identity": self.identity}
            if marker.exists():
                if marker.is_symlink() or json.loads(marker.read_bytes()) != expected:
                    raise ValueError("diagnostic collection identity changed")
            else:
                if any(root.glob("episode-*")):
                    raise ValueError("existing episodes lack a collection identity")
                atomic_json(marker, expected)

    @staticmethod
    def episode_name(scan: str, instruction: str) -> str:
        return "episode-" + object_sha256([scan, instruction])

    def verify_episode(self, directory: Path) -> dict:
        try:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("episode is not a real directory")
            for name in ("manifest.json", "COMMITTED", "inputs.pt", "labels.pt"):
                if (directory / name).is_symlink() or not (directory / name).is_file():
                    raise ValueError(f"missing or symlinked episode file: {name}")
            raw = (directory / "manifest.json").read_bytes()
            if (directory / "COMMITTED").read_text().strip() != hashlib.sha256(raw).hexdigest():
                raise ValueError("episode commit checksum mismatch")
            manifest = json.loads(raw)
            if manifest["schema"] != SCHEMA or manifest["identity_sha256"] != self.identity_sha256:
                raise ValueError("episode provenance mismatch")
            if manifest["episode_name"] != self.episode_name(manifest["scan_id"], manifest["instr_id"]):
                raise ValueError("episode identifier mismatch")
            if set(manifest["files"]) != {"inputs.pt", "labels.pt"}:
                raise ValueError("episode file list mismatch")
            for name, expected in manifest["files"].items():
                file = directory / name
                if expected != {"size": file.stat().st_size, "sha256": file_sha256(file)}:
                    raise ValueError("episode data checksum mismatch")
            return manifest
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"corrupt diagnostic episode: {directory}") from error

    def _copy_episode(self, source: Path, destination: Path) -> dict:
        original = self.verify_episode(source)
        stage = destination.parent / f".pending-{uuid.uuid4().hex}"
        shutil.copytree(source, stage)
        if self.verify_episode(stage) != original:
            raise ValueError("diagnostic episode backup read-back mismatch")
        if destination.exists():
            raise ValueError("concurrent diagnostic episode writer detected")
        stage.rename(destination)
        return self.verify_episode(destination)

    def find(self, scan: str, instruction: str) -> dict | None:
        name = self.episode_name(scan, instruction)
        local = self.local / name
        remote = self.backup / name if self.backup is not None else None
        local_manifest = self.verify_episode(local) if local.exists() or local.is_symlink() else None
        remote_manifest = None
        if remote is not None:
            if self.verify_backup:
                self.verify_backup()
            if remote.exists() or remote.is_symlink():
                remote_manifest = self.verify_episode(remote)
        if local_manifest is not None and remote_manifest is not None and local_manifest != remote_manifest:
            raise ValueError("local and cloud episode versions disagree")
        if local_manifest is not None and remote is not None and remote_manifest is None:
            self._copy_episode(local, remote)
        if local_manifest is None and remote_manifest is not None:
            local_manifest = self._copy_episode(remote, local)
        if local_manifest is not None and (local_manifest["scan_id"] != scan or local_manifest["instr_id"] != instruction):
            raise ValueError("requested episode association does not match stored episode")
        return local_manifest

    def commit(self, inputs: dict, labels: dict, trajectory: dict, summary: dict) -> dict:
        association = inputs["association"]
        scan, instruction = association["scan_id"], association["instr_id"]
        if self.find(scan, instruction) is not None:
            raise ValueError("refusing to overwrite a committed diagnostic episode")
        name = self.episode_name(scan, instruction)
        stage = self.local / f".pending-{uuid.uuid4().hex}"
        stage.mkdir()
        for filename, value in (("inputs.pt", inputs), ("labels.pt", labels)):
            with (stage / filename).open("xb") as stream:
                torch.save(cpu_copy(value), stream)
                stream.flush()
                os.fsync(stream.fileno())
        manifest = {
            "schema": SCHEMA, "identity_sha256": self.identity_sha256,
            "episode_name": name, "scan_id": scan, "instr_id": instruction,
            "num_states": len(inputs["states"]), "usage": labels["usage"],
            "trajectory": cpu_copy(trajectory), "coverage": summary,
            "files": {filename: {"size": (stage / filename).stat().st_size,
                                 "sha256": file_sha256(stage / filename)}
                      for filename in ("inputs.pt", "labels.pt")},
        }
        raw = _json(manifest)
        _write(stage / "manifest.json", raw)
        _write(stage / "COMMITTED", (hashlib.sha256(raw).hexdigest() + "\n").encode())
        self.verify_episode(stage)
        stage.rename(self.local / name)
        # The committed local episode remains recoverable if cloud upload fails.
        return self.find(scan, instruction)


class DiagnosticObserver:
    """Attach to an unmodified DUET agent; never change model outputs or actions."""

    def __init__(self, agent: Any, store: DiagnosticStore, *, split: str):
        if split not in {"train_fit", "train_dev"}:
            raise ValueError("diagnostics only support train_fit or analysis-only train_dev")
        if agent.args.batch_size != 1 or agent.args.act_visited_nodes or not agent.args.enc_full_graph:
            raise ValueError("M0 diagnostics require batch=1, full graph, and unvisited-only actions")
        if agent.args.max_action_len > 15:
            raise ValueError("M0 max_sources=15 requires max_action_len <=15")
        self.agent, self.store, self.split = agent, store, split
        self.usage = "training_diagnostics" if split == "train_fit" else "analysis_only"
        self.panorama = None
        self.memory = None
        self.states: list[dict] = []
        self.label_states: list[dict] = []
        self.arrival_pairs: list[dict] = []
        self.observed: set[str] = set()
        self.new_episodes = self.reused_episodes = 0
        self.panorama_handle = agent.vln_bert.register_forward_hook(self._panorama_hook)
        self.original_rollout = agent.rollout
        agent.rollout = self.rollout

    def _panorama_hook(self, module, args, output):
        if args[0] == "panorama":
            embeds, masks = output
            if embeds.shape[0] != 1:
                raise ValueError("M0 panorama hook requires batch=1")
            self.panorama = {
                "embeds": cpu_copy(embeds).float(), "masks": cpu_copy(masks),
                "cand_vpids": cpu_copy(args[1]["cand_vpids"]),
                "average": cpu_copy(torch.sum(embeds * masks.unsqueeze(2), 1) / torch.sum(masks, 1, keepdim=True)),
            }
        return None

    def rollout(self, **kwargs):
        if kwargs.get("train_ml") is not None or kwargs.get("train_rl", False) or kwargs.get("reset", True) is False:
            raise ValueError("diagnostic rollout only supports reset=True inference")
        if self.agent.feedback != "argmax":
            raise ValueError("diagnostics require baseline argmax feedback")
        obs = self.agent.env.reset()
        if len(obs) != 1:
            raise ValueError("diagnostics require one episode per rollout")
        scan, instruction = str(obs[0]["scan"]), str(obs[0]["instr_id"])
        saved = self.store.find(scan, instruction)
        if saved is not None:
            self.reused_episodes += 1
            return [cpu_copy(saved["trajectory"])]
        episode_id = self.store.episode_name(scan, instruction)
        self.memory = EpisodeEvidenceMemory(episode_id, scan_id=scan, instr_id=instruction, max_sources=15)
        self.states, self.label_states, self.arrival_pairs = [], [], []
        self.observed = set()
        self.panorama = None
        previous_hook = getattr(self.agent, "decision_hook", None)
        if previous_hook is not None and previous_hook is not self:
            raise ValueError("diagnostics must use the original baseline without a trained head")
        self.agent.decision_hook = self
        try:
            arguments = dict(kwargs, reset=False)
            trajectories = self.original_rollout(**arguments)
        finally:
            self.agent.decision_hook = previous_hook
        if len(trajectories) != 1 or trajectories[0]["instr_id"] != instruction or not self.states:
            raise ValueError("diagnostic rollout did not complete its expected episode")
        association = {"episode_id": episode_id, "scan_id": scan, "instr_id": instruction}
        inputs = {"schema": SCHEMA, "association": association, "states": self.states}
        labels = {"schema": SCHEMA, "association": association, "usage": self.usage,
                  "states": self.label_states, "arrival_pairs": self.arrival_pairs,
                  "unarrived_targets": [{"target_id": target, "arrival_observed": False,
                                         "outcome": "unknown"} for target in self.memory.pending_targets()]}
        summary = self.memory.summary()
        summary["num_eligible_states"] = sum(state["eligible_decision"] for state in self.states)
        summary["num_candidate_states"] = sum(len(state["candidate_evidence"]) for state in self.states if state["eligible_decision"])
        summary["num_multi_source_candidate_states"] = sum(
            item["counts"]["source_count_total"] >= 2 for state in self.states if state["eligible_decision"]
            for item in state["candidate_evidence"].values())
        summary["num_states_with_multi_source"] = sum(
            any(item["counts"]["source_count_total"] >= 2 for item in state["candidate_evidence"].values())
            for state in self.states if state["eligible_decision"])
        self.store.commit(inputs, labels, trajectories[0], summary)
        self.new_episodes += 1
        return trajectories

    def __call__(self, nav_inputs, nav_outs, obs, ended, step, trajectory):
        if len(obs) != 1 or bool(ended[0]) or self.memory is None or self.panorama is None:
            raise ValueError("unexpected diagnostic episode/step state")
        ob = obs[0]
        current = str(ob["viewpoint"])
        pano = self.panorama
        self.panorama = None
        embeds, masks = pano["embeds"][0], pano["masks"][0]
        average = pano["average"][0]
        if current not in self.observed:
            pair = self.memory.arrive(current, step=step, feature=average)
            if pair is not None:
                pair["training_only"].update({"arrival_heading": float(ob["heading"]),
                                              "arrival_elevation": float(ob["elevation"]),
                                              "arrival_kind": "natural"})
                self.arrival_pairs.append(pair)
            self.memory.drain_pairs()
            self.observed.add(current)
        vpids = nav_inputs["gmap_vpids"][0]
        visited = nav_inputs["gmap_visited_masks"][0].detach().cpu()
        legal = (nav_inputs["gmap_masks"][0] & ~nav_inputs["gmap_visited_masks"][0]).detach().cpu().clone()
        by_vpid = {vp: index for index, vp in enumerate(vpids)}
        candidates = ob["candidate"]
        if pano["cand_vpids"][0] != [candidate["viewpointId"] for candidate in candidates]:
            raise ValueError("panorama candidate ordering changed")
        for index, candidate in enumerate(candidates):
            target = candidate["viewpointId"]
            if not bool(visited[by_vpid[target]]):
                relative = [float(target_coordinate - source_coordinate) for target_coordinate, source_coordinate
                            in zip(candidate["position"], ob["position"])]
                self.memory.observe_proxy(target, current, step=step, heading=float(candidate["heading"]),
                                          elevation=float(candidate["elevation"]), relative_position=relative,
                                          feature=embeds[index])
        evidence = {target: self.memory.policy_snapshot(target) for index, target in enumerate(vpids)
                    if target is not None and bool(legal[index])}
        for index, target in enumerate(vpids):
            if target not in evidence:
                continue
            snapshot = evidence[target]
            if snapshot["counts"]["overflow_count"] or snapshot["counts"]["observation_count"] != snapshot["counts"]["source_count_total"]:
                raise ValueError("M0 encountered source overflow/revisits; preserve full events before proceeding")
            vectors = [source["feature"] for source in snapshot["sources"]]
            if not vectors:
                raise ValueError("legal unvisited candidate has no captured evidence")
            rebuilt = vectors[0].clone()
            for vector in vectors[1:]:
                rebuilt += vector
            rebuilt /= len(vectors)
            actual = nav_inputs["gmap_img_embeds"][0, index].detach().cpu()
            if not torch.allclose(rebuilt, actual, atol=1e-6, rtol=1e-5):
                raise ValueError("captured evidence does not reconstruct DUET's graph average")
        frozen = {key: cpu_copy(nav_inputs.get(key)) for key in NAVIGATION_KEYS}
        if any(frozen[key] is None for key in NAVIGATION_KEYS if key != "vp_obj_masks"):
            raise ValueError("navigation replay input is missing")
        logits = nav_outs["fused_logits"][0].detach().cpu().float().clone()
        if not torch.isfinite(logits[legal]).all():
            raise ValueError("nonfinite legal baseline logits")
        eligible = not nav_inputs["no_vp_left"][0] and step < self.agent.args.max_action_len - 1
        self.states.append({
            "step": int(step), "nav_inputs": frozen, "base_logits": logits,
            "base_global_logits": cpu_copy(nav_outs["global_logits"][0]).float(),
            "base_local_logits": cpu_copy(nav_outs["local_logits"][0]).float(), "valid_mask": legal,
            "current_viewpoint": current, "heading": float(ob["heading"]), "elevation": float(ob["elevation"]),
            "candidate_evidence": evidence, "eligible_decision": bool(eligible),
            "baseline_argmax": int(logits.argmax()),
        })
        self.label_states.append(self._oracle_labels(ob, nav_inputs, legal, step))
        return nav_outs

    def _oracle_labels(self, observation, nav_inputs, legal, step):
        current, goal, scan = observation["viewpoint"], observation["gt_path"][-1], observation["scan"]
        distances = self.agent.env.shortest_distances[scan]
        vpids = nav_inputs["gmap_vpids"][0]
        current_index = vpids.index(current)
        teacher = torch.full((len(vpids),), float("inf"), dtype=torch.float64)
        execution = teacher.clone()
        for index, target in enumerate(vpids):
            if target is not None and bool(legal[index]):
                teacher[index] = distances[current][target] + distances[target][goal]
                execution[index] = float(nav_inputs["gmap_pair_dists"][0, current_index, index]) + distances[target][goal]
        exact = current == goal
        if exact:
            teacher[0] = execution[0] = 0.0
        def optimal(values):
            if not torch.isfinite(values).any():
                return []
            return torch.nonzero(torch.isfinite(values) & ((values - values.min()).abs() <= 1e-6)).flatten().tolist()
        return {"step": int(step), "teacher_cost": teacher, "execution_cost": execution,
                "teacher_optimal_indices": optimal(teacher), "execution_optimal_indices": optimal(execution),
                "stop": {"exact_goal": exact, "within_success_radius": distances[current][goal] < 3.0,
                         "distance_to_goal": float(distances[current][goal])}}

    def close(self):
        self.panorama_handle.remove()
        self.agent.rollout = self.original_rollout


def load_episode(directory: str | Path, *, expected_identity_sha256: str | None = None):
    """Verify COMMITTED/content hashes and return (inputs, labels, manifest).

    ``expected_identity_sha256`` should come from the collection identity when
    comparing episodes. No policy/inference path calls this labels loader.
    """
    root = Path(directory)
    # Reuse the exact store verifier without creating or mutating a collection.
    try:
        raw = (root / "manifest.json").read_bytes()
        candidate = json.loads(raw)
        identity = candidate["identity_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"invalid diagnostic episode manifest: {root}") from error
    if expected_identity_sha256 is not None and identity != expected_identity_sha256:
        raise ValueError("episode collection identity mismatch")
    class ReadOnlyVerifier:
        identity_sha256 = identity
        episode_name = staticmethod(DiagnosticStore.episode_name)
    manifest = DiagnosticStore.verify_episode(ReadOnlyVerifier(), root)
    inputs = torch.load(root / "inputs.pt", map_location="cpu", weights_only=True)
    labels = torch.load(root / "labels.pt", map_location="cpu", weights_only=True)
    if (not isinstance(inputs, dict) or not isinstance(labels, dict)
            or inputs.get("schema") != SCHEMA or labels.get("schema") != SCHEMA
            or inputs.get("association") != labels.get("association")
            or not isinstance(inputs.get("states"), list)
            or len(inputs["states"]) != manifest["num_states"]):
        raise ValueError("diagnostic episode payload schema mismatch")
    association = inputs["association"]
    if (association["scan_id"] != manifest["scan_id"] or association["instr_id"] != manifest["instr_id"]
            or labels.get("usage") != manifest["usage"]):
        raise ValueError("diagnostic episode association/usage mismatch")
    return inputs, labels, manifest
