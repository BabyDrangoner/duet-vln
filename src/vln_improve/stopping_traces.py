"""Small, analysis-only STOP traces; no features, training loader, or policy edits."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .diagnostics import atomic_json
from .protocol import object_sha256
from .stopping_diagnostics import analyze_episode, summarize_episodes


SCHEMA = "duet_stopping_trace_v1"
STATE_KEYS = {"step", "current_viewpoint", "base_logits", "valid_mask", "baseline_argmax", "eligible_decision", "nav_inputs"}
NAV_KEYS = {"gmap_vpids", "gmap_masks", "gmap_visited_masks", "gmap_pair_dists"}


def index_report(report):
    indexed = []
    for key in ("episodes", "trajectories"):
        values = report.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"baseline report is missing {key}")
        mapping = {row["instr_id"]: row for row in values}
        if len(mapping) != len(values):
            raise ValueError("baseline report repeats an instruction")
        indexed.append(mapping)
    if set(indexed[0]) != set(indexed[1]):
        raise ValueError("baseline metrics and trajectory IDs differ")
    return tuple(indexed)


def exact_report_parity(reference, actual):
    """Require every metric key/value and complete trajectory, not mean scores."""
    ref_metrics, ref_paths = index_report(reference)
    metrics, paths = index_report(actual)
    if metrics != ref_metrics or paths != ref_paths:
        raise ValueError("exact per-instruction baseline trajectory/metric parity failed")


def restore_payload(trace):
    """Convert the deliberately small JSON schema to the existing STOP core."""
    if trace.get("schema") != SCHEMA or trace.get("usage") != "analysis_only":
        raise ValueError("STOP trace must be analysis_only")
    association = trace["association"]
    label_block = trace["labels"]
    if label_block.get("usage") != "analysis_only" or label_block.get("association") != association:
        raise ValueError("trace labels have invalid usage/association")
    states = []
    for raw in trace["states"]:
        if set(raw) != STATE_KEYS or set(raw["nav_inputs"]) != NAV_KEYS:
            raise ValueError("trace contains unexpected fields (large feature arrays are forbidden)")
        state = copy.deepcopy(raw)
        mask = raw["valid_mask"]
        logits = raw["base_logits"]
        if (not isinstance(mask, list) or not mask or any(type(x) is not bool for x in mask)
                or not isinstance(logits, list) or len(logits) != len(mask)
                or any((x is None) != (not valid) for x, valid in zip(logits, mask))):
            raise ValueError("invalid JSON masked logits; only illegal scores may be null")
        state["valid_mask"] = torch.tensor(mask, dtype=torch.bool)
        state["base_logits"] = torch.tensor([x if valid else -torch.inf for x, valid in zip(logits, mask)], dtype=torch.float32)
        nav = state["nav_inputs"]
        nodes = nav["gmap_vpids"]
        if (not isinstance(nodes, list) or len(nodes) != 1 or len(nodes[0]) != len(mask)
                or nodes[0][0] is not None or any(not isinstance(n, str) or not n for n in nodes[0][1:])
                or len(set(nodes[0])) != len(nodes[0])):
            raise ValueError("invalid trace graph node IDs")
        for key in ("gmap_masks", "gmap_visited_masks"):
            value = nav[key]
            if (not isinstance(value, list) or len(value) != 1 or len(value[0]) != len(mask)
                    or any(type(x) is not bool for x in value[0])):
                raise ValueError("invalid trace graph boolean mask")
            nav[key] = torch.tensor(value, dtype=torch.bool)
        nav["gmap_pair_dists"] = torch.tensor(nav["gmap_pair_dists"], dtype=torch.float32)
        distances = nav["gmap_pair_dists"]
        if (distances.shape != (1, len(mask), len(mask)) or not torch.isfinite(distances).all()
                or (distances < 0).any() or not torch.equal(distances, distances.transpose(1, 2))
                or not torch.equal(distances.diagonal(dim1=1, dim2=2), torch.zeros(1, len(mask)))
                or not nav["gmap_masks"].all()):
            raise ValueError("invalid full-graph trace distances")
        current = state["current_viewpoint"]
        if current not in nodes[0] or not nav["gmap_visited_masks"][0, nodes[0].index(current)]:
            raise ValueError("current observation is not a visited graph node")
        states.append(state)
    return ({"association": association, "states": states}, label_block,
            {"trajectory": trace["trajectory"]})


class StopTraceStore:
    """Atomic per-episode JSON, verified Drive read-back, resumable by identity."""

    def __init__(self, local, identity, baseline, *, backup, verify_backup):
        self.local, self.backup = Path(local).resolve(), Path(backup).resolve()
        if self.local.is_relative_to(self.backup) or self.backup.is_relative_to(self.local):
            raise ValueError("trace local and backup roots must be disjoint")
        self.identity = copy.deepcopy(identity)
        self.identity_sha256 = object_sha256(identity)
        self.verify_backup = verify_backup
        self.metrics, self.trajectories = index_report(baseline)
        self.selection = {r["instr_id"]: r for r in identity["selection"]}
        if len(self.selection) != len(identity["selection"]) or set(self.selection) != set(self.metrics):
            raise ValueError("trace selection differs from the full baseline")
        marker = {"schema": SCHEMA, "identity": self.identity, "identity_sha256": self.identity_sha256}
        for root in (self.local, self.backup):
            if root == self.backup:
                self.verify_backup()
            root.mkdir(parents=True, exist_ok=True)
            path = root / "COLLECTION.json"
            if path.exists():
                if path.is_symlink() or json.loads(path.read_bytes()) != marker:
                    raise ValueError("STOP collection identity changed")
            else:
                if any(root.glob("episode-*.json")):
                    raise ValueError("orphan STOP episodes lack their collection identity")
                atomic_json(path, marker)
            if json.loads(path.read_bytes()) != marker:
                raise ValueError("STOP collection identity read-back mismatch")

    @staticmethod
    def filename(scan, instr):
        return "episode-" + object_sha256([scan, instr]) + ".json"

    def _validate(self, envelope, scan, instr):
        if (not isinstance(envelope, dict) or set(envelope) != {"trace", "content_sha256"}
                or envelope.get("content_sha256") != object_sha256(envelope.get("trace"))):
            raise ValueError("STOP episode checksum mismatch")
        trace = envelope["trace"]
        expected = {"episode_id": self.filename(scan, instr).removesuffix(".json"), "scan_id": scan, "instr_id": instr}
        if (trace.get("identity_sha256") != self.identity_sha256 or trace.get("association") != expected
                or instr not in self.selection or self.selection[instr]["scan"] != scan
                or trace.get("path_id") != self.selection[instr]["path_id"]):
            raise ValueError("STOP episode identity/selection mismatch")
        if (trace["trajectory"]["instr_id"] != instr or trace["trajectory"]["path"] != self.trajectories[instr]["trajectory"]
                or trace.get("metrics") != self.metrics[instr]):
            raise ValueError("stored trace does not match every baseline trajectory/metric")
        inputs, labels, manifest = restore_payload(trace)
        analyze_episode(inputs, labels, manifest, max_action_len=self.identity["model"]["max_action_len"],
                        metrics=trace["metrics"], reported_trajectory=self.trajectories[instr]["trajectory"])
        return trace

    def _read(self, path, scan, instr):
        if path.is_symlink() or not path.is_file():
            raise ValueError("STOP trace is missing or symlinked")
        value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON {x}")))
        return self._validate(value, scan, instr), value

    def find(self, scan, instr):
        if instr not in self.selection or self.selection[instr]["scan"] != scan:
            raise ValueError("observer received an unselected instruction")
        name = self.filename(scan, instr)
        self.verify_backup()
        local, remote = self.local / name, self.backup / name
        left = self._read(local, scan, instr) if local.exists() or local.is_symlink() else None
        right = self._read(remote, scan, instr) if remote.exists() or remote.is_symlink() else None
        if left and right and left[1] != right[1]:
            raise ValueError("local and Drive STOP traces disagree")
        if left and right is None:
            atomic_json(remote, left[1])
            right = self._read(remote, scan, instr)
        if right and left is None:
            atomic_json(local, right[1])
            left = self._read(local, scan, instr)
        if left and (right is None or left[1] != right[1]):
            raise ValueError("STOP trace cloud read-back mismatch")
        return None if left is None else left[0]

    def commit(self, trace):
        scan, instr = trace["association"]["scan_id"], trace["association"]["instr_id"]
        envelope = {"trace": trace, "content_sha256": object_sha256(trace)}
        self._validate(envelope, scan, instr)
        if self.find(scan, instr) is not None:
            raise ValueError("refusing to overwrite a committed STOP episode")
        atomic_json(self.local / self.filename(scan, instr), envelope)
        return self.find(scan, instr)


class StopTraceObserver:
    """Return the original nav_outs object; capture only small copied statistics."""

    def __init__(self, agent, store):
        if (agent.args.batch_size != 1 or agent.args.act_visited_nodes or not agent.args.enc_full_graph
                or not 1 <= agent.args.max_action_len <= 15):
            raise ValueError("STOP traces require batch-one, full graph, unvisited-only actions, maxlen<=15")
        self.agent, self.store = agent, store
        self.original_rollout = agent.rollout
        self.new_episodes = self.reused_episodes = 0
        self.states, self.labels = [], []
        agent.rollout = self.rollout

    def rollout(self, **kwargs):
        if (kwargs.get("train_ml") is not None or kwargs.get("train_rl", False)
                or kwargs.get("reset", True) is False or self.agent.feedback != "argmax"):
            raise ValueError("STOP trace supports only reset=True baseline inference")
        previous = getattr(self.agent, "decision_hook", None)
        if previous is not None:
            raise ValueError("STOP trace must not coexist with another policy hook")
        observations = self.agent.env.reset()
        if len(observations) != 1:
            raise ValueError("STOP trace requires one episode per rollout")
        observation = observations[0]
        scan, instr = str(observation["scan"]), str(observation["instr_id"])
        saved = self.store.find(scan, instr)
        if saved is not None:
            self.reused_episodes += 1
            return [copy.deepcopy(saved["trajectory"])]
        self.states, self.labels = [], []
        self.agent.decision_hook = self
        try:
            trajectories = self.original_rollout(**dict(kwargs, reset=False))
        finally:
            self.agent.decision_hook = previous
        if len(trajectories) != 1 or trajectories[0]["instr_id"] != instr or not self.states:
            raise ValueError("STOP trace rollout did not complete its expected episode")
        trajectory = trajectories[0]
        scores = self.agent.env._eval_item(scan, trajectory["path"], observation["gt_path"])
        metrics = dict(instr_id=instr, scan_id=scan, **{key: float(value) for key, value in scores.items()})
        association = {"episode_id": self.store.filename(scan, instr).removesuffix(".json"), "scan_id": scan, "instr_id": instr}
        self.store.commit({"schema": SCHEMA, "identity_sha256": self.store.identity_sha256,
                           "usage": "analysis_only", "association": association,
                           "path_id": self.store.selection[instr]["path_id"], "states": self.states,
                           "labels": {"association": association, "usage": "analysis_only", "states": self.labels},
                           "trajectory": copy.deepcopy(trajectory), "metrics": metrics})
        self.new_episodes += 1
        return trajectories

    def __call__(self, nav_inputs, nav_outs, obs, ended, step, trajectory):
        if len(obs) != 1 or bool(ended[0]):
            raise ValueError("unexpected STOP observer state")
        legal = (nav_inputs["gmap_masks"][0] & ~nav_inputs["gmap_visited_masks"][0]).detach().cpu()
        logits = nav_outs["fused_logits"][0].detach().cpu().float()
        if not torch.isfinite(logits[legal]).all() or not torch.isneginf(logits[~legal]).all():
            raise ValueError("invalid masked STOP trace logits")
        nav = {key: (nav_inputs[key].detach().cpu().tolist() if isinstance(nav_inputs[key], torch.Tensor)
                     else copy.deepcopy(nav_inputs[key])) for key in NAV_KEYS}
        current = str(obs[0]["viewpoint"])
        self.states.append({"step": int(step), "current_viewpoint": current,
                            "base_logits": [float(x) if bool(valid) else None for x, valid in zip(logits, legal)],
                            "valid_mask": legal.tolist(), "baseline_argmax": int(logits.argmax()),
                            "eligible_decision": not bool(nav_inputs["no_vp_left"][0]) and step < self.agent.args.max_action_len - 1,
                            "nav_inputs": nav})
        distance = float(self.agent.env.shortest_distances[obs[0]["scan"]][current][obs[0]["gt_path"][-1]])
        self.labels.append({"step": int(step), "stop": {"exact_goal": current == obs[0]["gt_path"][-1],
                             "within_success_radius": distance < 3, "distance_to_goal": distance}})
        return nav_outs

    def close(self):
        self.agent.rollout = self.original_rollout


def summarize_store(store):
    rows, checksums = [], {}
    expected = {store.filename(row["scan"], row["instr_id"]) for row in store.selection.values()}
    if {path.name for path in store.local.glob("episode-*.json")} != expected:
        raise ValueError("local STOP collection is incomplete or contains extra episodes")
    for instr, selected in sorted(store.selection.items()):
        trace = store.find(selected["scan"], instr)
        if trace is None:
            raise ValueError("missing STOP trace")
        inputs, labels, manifest = restore_payload(trace)
        row = analyze_episode(inputs, labels, manifest, max_action_len=store.identity["model"]["max_action_len"],
                              metrics=trace["metrics"], reported_trajectory=store.trajectories[instr]["trajectory"])
        row["path_id"] = trace["path_id"]
        rows.append(row)
        checksums[store.filename(selected["scan"], instr)] = object_sha256(trace)
    def support(chosen):
        return {"instructions": len(chosen), "independent_paths": len({(r["association"]["scan_id"], r["path_id"]) for r in chosen}),
                "scenes": len({r["association"]["scan_id"] for r in chosen})}
    return {"schema": "duet_stop_trace_analysis_v1", "usage": "analysis_only", "split": store.identity["split"],
            "identity_sha256": store.identity_sha256, "episode_content_sha256": checksums,
            "all_instruction_trajectory_and_metric_parity": True, "overall": summarize_episodes(rows),
            "opportunity_support": {
                "observed_history_success_but_baseline_failed": support([r for r in rows if r["observed_history_success_but_baseline_failed"]]),
                **{category: support([r for r in rows if r["fallback_category"] == category])
                   for category in ("fallback_rescued", "fallback_harmed")}},
            "by_termination_flag": {flag: summarize_episodes([r for r in rows if r["termination_flags"][flag]])
                                    for flag in ("argmax_stop", "no_legal_move", "action_limit")},
            "episodes": rows,
            "missing": {"counterfactual_spl": "GT reference path length is not cached; start-goal shortest distance is not substituted"},
            "interpretation": "Read-only fixed full-baseline trajectories, final endpoint rescoring only; no online continuation benefit, training, causal count-bias claim, or established novel method."}
