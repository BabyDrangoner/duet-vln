import copy
from types import SimpleNamespace

import pytest
import torch

import vln_improve.endpoint_policy as policy


def run_fake(monkeypatch, *, head=True, reference=None):
    monkeypatch.setattr(policy, "build_endpoint_features", lambda inputs, outputs: inputs["feature"])
    calls = []
    def move(actions, graphs, obs, paths):
        calls.append(list(actions))
        if actions[0] is not None:
            paths[0]["path"].append([obs[0]["viewpoint"], actions[0]])
    agent = SimpleNamespace(make_equiv_action=move)
    baseline = {"i": [["A"], ["A", "B"]]} if reference is None else reference
    ranker = policy.EndpointReranker(agent, torch.nn.Identity() if head else None, baseline)
    graph = SimpleNamespace(node_stop_scores={}, graph=SimpleNamespace(path=lambda a, b: [a, b]))
    paths = [{"instr_id": "i", "path": [["A"]]}]
    for step, (node, q, action) in enumerate((("A", .9, "B"), ("B", .2, None))):
        obs = [{"instr_id": "i", "viewpoint": node, "gt_path": ["FORBIDDEN"]}]
        logits = torch.tensor([[1., 2.]]) if action is not None else torch.tensor([[2., 1.]])
        outs = {"fused_logits": logits}
        before = logits.clone()
        assert ranker({"feature": torch.tensor([[q]])}, outs, obs, [False], step, paths) is outs
        assert torch.equal(logits, before)
        graph.node_stop_scores[node] = {"stop": .1 if node == "A" else .8}
        ranker.make_equiv_action([action], [graph], obs, paths)
    chosen = max(graph.node_stop_scores, key=lambda n: graph.node_stop_scores[n]["stop"])
    if chosen != "B":
        paths[0]["path"].append(graph.graph.path("B", chosen))
    return ranker, paths, calls


def test_rerank_changes_only_final_return_and_keeps_full_prefix(monkeypatch):
    ranker, paths, calls = run_fake(monkeypatch)
    assert calls == [["B"], [None]]
    assert paths[0]["path"] == [["A"], ["A", "B"], ["B", "A"]]
    assert ranker.result["online_path_and_termination_parity"]
    assert ranker.result["baseline_endpoint"] == "B"
    assert ranker.result["probe_endpoint"] == "A"


def test_no_head_is_exact_baseline_identity(monkeypatch):
    ranker, paths, _ = run_fake(monkeypatch, head=False)
    assert paths[0]["path"] == [["A"], ["A", "B"]]
    assert not ranker.result["endpoint_changed"]


def test_refuses_changed_online_path_before_endpoint_write(monkeypatch):
    with pytest.raises(ValueError, match="online trajectory/termination"):
        run_fake(monkeypatch, reference={"i": [["A"], ["A", "C"], ["C", "B"]]})
