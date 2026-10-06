"""Rerank historical endpoints while preserving DUET's online decisions."""
from __future__ import annotations

import copy
import math

import torch

from .endpoint_probe import build_endpoint_features


class EndpointReranker:
    """One batch=1 rollout; no target labels or environment distances as input.

    The upstream caller writes its current STOP probability after the decision
    hook. We therefore replace endpoint scores only after make_equiv_action,
    immediately before the original historical endpoint selection executes.
    """

    def __init__(self, agent, head, baseline_trajectories):
        self.agent, self.head = agent, head
        self.baseline = baseline_trajectories
        self.original_move = agent.make_equiv_action
        self.scores = {}
        self.instr_id = None
        self.decisions = []
        self.completed = False
        self.result = None

    def __call__(self, nav_inputs, nav_outs, observations, ended, step, trajectory):
        if len(observations) != 1 or bool(ended[0]) or self.completed:
            raise ValueError("endpoint reranker requires one active episode")
        ob = observations[0]
        instr = str(ob["instr_id"])
        if instr not in self.baseline or self.instr_id not in (None, instr):
            raise ValueError("endpoint episode differs from reference")
        self.instr_id = instr
        current = str(ob["viewpoint"])
        if self.head is not None:
            with torch.inference_mode():
                value = self.head(build_endpoint_features(nav_inputs, nav_outs)).reshape(-1)
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise ValueError("endpoint head returned invalid score")
            self.scores[current] = float(value.item())
        self.decisions.append({"step": int(step), "viewpoint": current,
                               "argmax": int(nav_outs["fused_logits"][0].argmax())})
        # Preserve the original objects, scores, masks, and online STOP.
        return nav_outs

    def make_equiv_action(self, actions, gmaps, observations, trajectories):
        if len(actions) != 1 or len(gmaps) != 1 or self.instr_id is None:
            raise ValueError("unexpected endpoint movement batch")
        result = self.original_move(actions, gmaps, observations, trajectories)
        if actions[0] is not None:
            return result
        graph = gmaps[0]
        original = graph.node_stop_scores
        if not original or self.completed:
            raise ValueError("invalid endpoint termination state")
        # Python dict insertion order and stable max implement the upstream's
        # strict > tie rule. Reconstruct its complete trajectory before edits.
        base_node = max(original, key=lambda node: original[node]["stop"])
        baseline_path = copy.deepcopy(trajectories[0]["path"])
        current = observations[0]["viewpoint"]
        if current != base_node:
            baseline_path.append(graph.graph.path(current, base_node))
        if baseline_path != self.baseline[self.instr_id]:
            raise ValueError("online trajectory/termination changed before endpoint reranking")
        if self.head is not None:
            if set(self.scores) != set(original):
                raise ValueError("probe and baseline historical node sets differ")
            for node, score in self.scores.items():
                if not math.isfinite(score):
                    raise ValueError("nonfinite endpoint score")
                original[node] = {**original[node], "stop": score}
        selected = max(original, key=lambda node: original[node]["stop"])
        self.result = {"instr_id": self.instr_id, "online_decisions": len(self.decisions),
                       "online_path_and_termination_parity": True,
                       "baseline_endpoint": base_node, "probe_endpoint": selected,
                       "endpoint_changed": base_node != selected,
                       "prefix_path": copy.deepcopy(trajectories[0]["path"])}
        self.completed = True
        return result
