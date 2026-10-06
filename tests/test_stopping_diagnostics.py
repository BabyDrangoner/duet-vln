import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_stopping import analyze_collection, main
from test_diagnostics import collect
from vln_improve.diagnostics import DiagnosticStore
from vln_improve.protocol import object_sha256
from vln_improve.stopping_diagnostics import analyze_episode, summarize_episodes


def _episode(*, goal="B", forced=False, no_move=False):
    nodes = [None, "A", "B", "C"] + ([] if no_move else ["X", "Y", "Z"])
    pos = {node: 4 * i for i, node in enumerate(nodes[1:])}
    association = {"episode_id": "episode", "instr_id": "instruction", "scan_id": "scan"}
    states, labels = [], []
    for step, current in enumerate(("A", "B", "C")):
        visited = torch.tensor([[False] + [node in ("A", "B", "C")[:step + 1] for node in nodes[1:]]])
        valid = ~visited[0]
        logits = torch.full((len(nodes),), -10.0)
        logits[0] = (-2, 1, 0)[step]
        if step < 2:
            logits[nodes.index(("B", "C")[step])] = (3, 1.1)[step]
        else:
            logits[valid] = -.1
            logits[0] = 0
        logits[~valid] = -torch.inf
        pairs = torch.zeros(1, len(nodes), len(nodes))
        for i, a in enumerate(nodes[1:], 1):
            for j, b in enumerate(nodes[1:], 1):
                pairs[0, i, j] = abs(pos[a] - pos[b])
        states.append({"step": step, "current_viewpoint": current, "base_logits": logits,
                       "valid_mask": valid, "baseline_argmax": int(logits.argmax()),
                       "eligible_decision": not (step == 2 and (forced or no_move)),
                       "nav_inputs": {"gmap_vpids": [nodes], "gmap_masks": torch.ones_like(visited),
                                      "gmap_visited_masks": visited, "gmap_pair_dists": pairs}})
        distance = abs(pos[current] - pos[goal])
        labels.append({"step": step, "stop": {"distance_to_goal": distance, "within_success_radius": distance < 3,
                                              "exact_goal": distance == 0}})
    path = [["A"], ["B"], ["C"]] + ([] if no_move else [["B"]])
    final = "C" if no_move else "B"
    metrics = {"instr_id": "instruction", "scan_id": "scan", "trajectory_lengths": 8 if no_move else 12,
               "nav_error": abs(pos[final] - pos[goal]), "success": float(final == goal), "oracle_success": 1.0}
    return ({"association": association, "states": states}, {"association": association, "states": labels},
            {"trajectory": {"instr_id": "instruction", "path": path}},
            {"max_action_len": 3 if forced else 15, "metrics": metrics, "reported_trajectory": copy.deepcopy(path)})


@pytest.mark.parametrize("goal,category", [("B", "fallback_rescued"), ("C", "fallback_harmed")])
def test_rescue_and_harm_include_whole_prefix_and_return(goal, category):
    inputs, labels, manifest, kwargs = _episode(goal=goal)
    result = analyze_episode(inputs, labels, manifest, **kwargs)
    assert result["fallback_category"] == category
    assert result["methods"]["baseline_probability"]["complete_trajectory_length_m"] == 12
    assert result["methods"]["last_position"]["complete_trajectory_length_m"] == 8
    assert result["methods"]["max_margin"]["viewpoint"] == "C"
    assert result["methods"]["logmeanexp_margin"]["viewpoint"] == "B"
    assert result["observed_history_has_success"]
    assert len(result["states"]) == 3


def test_forced_end_state_is_included_and_flags_can_overlap():
    inputs, labels, manifest, kwargs = _episode(forced=True)
    result = analyze_episode(inputs, labels, manifest, **kwargs)
    assert result["num_states"] == 3
    assert result["states"][-1]["eligible_decision"] is False
    assert result["termination_flags"] == {"argmax_stop": True, "no_legal_move": False, "action_limit": True}


def test_no_move_does_not_fabricate_margin_or_silently_drop_a_state():
    inputs, labels, manifest, kwargs = _episode(goal="C", no_move=True)
    result = analyze_episode(inputs, labels, manifest, **kwargs)
    assert result["states"][-1]["scores"]["baseline_probability"] == 1
    assert result["termination_flags"]["no_legal_move"]
    assert result["methods"]["max_margin"]["available"] is False
    summary = summarize_episodes([result])
    assert summary["methods"]["raw_stop"]["covered_episodes"] == 1
    assert summary["methods"]["max_margin"]["covered_episodes"] == 0
    assert summary["methods"]["max_margin"]["unavailable_episodes"] == 1
    assert summary["methods"]["max_margin"]["sr_percent"] is None


@pytest.mark.parametrize("damage", ["length", "trajectory", "selected_endpoint", "stop_label", "missing_final", "action"])
def test_corrupt_or_inconsistent_recorded_execution_is_rejected(damage):
    inputs, labels, manifest, kwargs = _episode()
    if damage == "length":
        kwargs["metrics"]["trajectory_lengths"] = 4  # Illegal prefix cropping.
    elif damage == "trajectory":
        kwargs["reported_trajectory"][-1] = ["A"]
    elif damage == "selected_endpoint":
        manifest["trajectory"]["path"][-1] = ["A"]
        kwargs["reported_trajectory"][-1] = ["A"]
    elif damage == "stop_label":
        labels["states"][1]["stop"]["within_success_radius"] = False
    elif damage == "missing_final":
        inputs["states"].pop()
    else:
        inputs["states"][0]["baseline_argmax"] = 0
    with pytest.raises(ValueError):
        analyze_episode(inputs, labels, manifest, **kwargs)


def test_raw_score_tie_uses_first_historical_node():
    inputs, labels, manifest, kwargs = _episode()
    # Common logit shift preserves each action distribution and baseline path.
    state = inputs["states"][0]
    state["base_logits"][state["valid_mask"]] += 3
    result = analyze_episode(inputs, labels, manifest, **kwargs)
    assert result["methods"]["raw_stop"]["selected_step"] == 0
    assert result["methods"]["baseline_probability"]["selected_step"] == 1


def test_baseline_probability_ties_keep_first_node_even_at_forced_end():
    inputs, labels, manifest, kwargs = _episode(forced=True)
    for state in inputs["states"]:
        state["base_logits"][0] = -1000.0  # FP32 softmax ties at exactly zero.
        state["baseline_argmax"] = int(state["base_logits"].argmax())
    manifest["trajectory"]["path"][-1] = ["B", "A"]
    kwargs["reported_trajectory"] = copy.deepcopy(manifest["trajectory"]["path"])
    kwargs["metrics"].update(trajectory_lengths=16, nav_error=4, success=0)
    result = analyze_episode(inputs, labels, manifest, **kwargs)
    assert result["methods"]["baseline_probability"]["selected_step"] == 0
    assert result["termination_flags"] == {"argmax_stop": False, "no_legal_move": False, "action_limit": True}


def test_success_radius_does_not_require_exact_goal():
    inputs, labels, manifest, kwargs = _episode()
    labels["states"][1]["stop"] = {"distance_to_goal": 2.0, "within_success_radius": True, "exact_goal": False}
    kwargs["metrics"]["nav_error"] = 2.0
    assert analyze_episode(inputs, labels, manifest, **kwargs)["fallback_category"] == "fallback_rescued"


def _collection(tmp_path):
    agent, observer, _, _, (inputs, labels, manifest) = collect(tmp_path / "fixture")
    observer.close()
    selection = [{"scan": "scan", "instr_id": "instruction", "path_id": "path"}]
    model = {"enc_full_graph": True, "fusion": "dynamic", "batch_size": 1, "max_action_len": 15}
    identity = {"schema": "duet_diagnostic_collection_v1", "split": "train_fit", "usage": "training_diagnostics",
                "selection": selection, "selection_sha256": object_sha256(selection), "model": model,
                "base_checkpoint_sha256": "base", "feature_sha256": "feature", "annotation_sha256": "annotation",
                "connectivity_sha256": "connectivity", "upstream_lock": {"commit": "commit"}, "seed": 0}
    store = DiagnosticStore(tmp_path / "collection", identity)
    store.commit(inputs, labels, manifest["trajectory"], manifest["coverage"])
    path = manifest["trajectory"]["path"]
    flat = sum(path, [])
    distances = agent.env.shortest_distances["scan"]
    length = sum(distances[a][b] for a, b in zip(flat[:-1], flat[1:]))
    report = {"metadata": {"split": "train_fit", "usage": "training_diagnostics", "feedback": "argmax", "mode": "baseline",
                           "diagnostic_collection": True, "subset": True, "selection_sha256": identity["selection_sha256"],
                           "base_checkpoint_sha256": "base", "feature_id": "feature", "train_annotation_sha256": "annotation",
                           "connectivity_sha256": "connectivity", "upstream_commit": "commit", "seed": 0, "max_action_len": 15},
              "episodes": [{"instr_id": "instruction", "scan_id": "scan", "trajectory_lengths": length,
                            "nav_error": 0, "success": 1, "oracle_success": 1}],
              "trajectories": [{"instr_id": "instruction", "trajectory": path}]}
    report_path = tmp_path / "rollout.json"
    report_path.write_text(json.dumps(report))
    return store.local, report_path


def test_collection_cli_is_cpu_only_and_keeps_verified_denominators(tmp_path, monkeypatch):
    collection, report = _collection(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("diagnostic must not inspect CUDA"))
    output = tmp_path / "analysis.json"
    main(["--diagnostics", str(collection), "--rollout-report", str(report), "--split", "train_fit", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["overall"]["episodes"] == 1
    assert result["overall"]["states"] == 3
    assert result["overall"]["methods"]["baseline_probability"]["sr_percent"] == 100
    assert "counterfactual_spl" in result["missing"]
    assert len(result["episode_manifest_sha256"]) == 1


@pytest.mark.parametrize("damage", ["checkpoint", "selection", "split", "missing_episode"])
def test_collection_identity_and_coverage_are_required(tmp_path, damage):
    collection, report = _collection(tmp_path)
    value = json.loads(report.read_text())
    if damage == "checkpoint":
        value["metadata"]["base_checkpoint_sha256"] = "other"
    elif damage == "selection":
        value["episodes"] = []
    elif damage == "split":
        value["metadata"]["split"] = "val_unseen"
    else:
        next(collection.glob("episode-*/COMMITTED")).unlink()
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        analyze_collection(collection, split="train_fit", rollout_report=report)
