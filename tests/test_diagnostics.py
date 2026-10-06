import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from vln_improve.diagnostics import (
    DiagnosticObserver, DiagnosticStore, NAVIGATION_KEYS, SCHEMA, load_episode,
    select_diagnostic_records,
)


def equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, list):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal(a, b)
    else:
        assert first == second


POSITIONS = {"A": [0.0, 0.0, 0.0], "B": [1.0, 0.0, 0.0],
             "C": [2.0, 0.0, 0.0], "X": [3.0, 0.0, 0.0]}


class FakeModel(torch.nn.Module):
    def forward(self, mode, batch):
        if mode == "panorama":
            return batch["embeds"], torch.ones(batch["embeds"].shape[:2], dtype=torch.bool)
        count = int(batch["gmap_visited_masks"].sum())
        chosen = "B" if count == 1 else ("C" if count == 2 else None)
        logits = torch.zeros_like(batch["gmap_masks"], dtype=torch.float32)
        logits[0, batch["gmap_vpids"][0].index(chosen)] = 3.0
        logits.masked_fill_(batch["gmap_visited_masks"], -torch.inf)
        return {"fused_logits": logits, "global_logits": logits.clone(),
                "local_logits": torch.zeros_like(batch["vp_masks"], dtype=torch.float32)}


class FakeEnv:
    def __init__(self, goal="C"):
        self.goal = goal
        self.resets = 0
        self.shortest_distances = {"scan": {a: {b: abs(POSITIONS[a][0] - POSITIONS[b][0])
                                                for b in POSITIONS} for a in POSITIONS}}

    def reset(self):
        self.resets += 1
        return [self.observation(0)]

    def observation(self, step):
        current = ["A", "B", "C"][step]
        targets = [["B", "X"], ["C", "X"], ["X"]][step]
        return {"scan": "scan", "instr_id": "instruction", "viewpoint": current,
                "position": POSITIONS[current], "heading": 0.0, "elevation": 0.0,
                "gt_path": ["A", "B", self.goal],
                "candidate": [{"viewpointId": target, "position": POSITIONS[target],
                               "heading": 0.1, "elevation": 0.0, "pointId": index}
                              for index, target in enumerate(targets)]}


class FakeAgent:
    def __init__(self, goal="C"):
        self.args = SimpleNamespace(batch_size=1, act_visited_nodes=False, enc_full_graph=True, max_action_len=15)
        self.env = FakeEnv(goal)
        self.vln_bert = FakeModel()
        self.feedback = "argmax"
        self.decision_hook = None
        self.actual_rollouts = 0

    def rollout(self, reset=True, **kwargs):
        self.actual_rollouts += 1
        if reset:
            self.env.reset()
        visited, sums = [], {}
        trajectory = [{"instr_id": "instruction", "path": [["A"]], "details": {}}]
        for step, values in enumerate(([[1.0, 0.0], [0.0, 1.0]], [[2.0, 2.0], [0.0, 3.0]], [[0.0, 5.0]])):
            ob = self.env.observation(step)
            current = ob["viewpoint"]
            visited.append(current)
            targets = [item["viewpointId"] for item in ob["candidate"]]
            pano, masks = self.vln_bert("panorama", {"embeds": torch.tensor([values]), "cand_vpids": [targets]})
            sums[current] = [pano[0].mean(0), 1]
            for j, target in enumerate(targets):
                if target in visited:
                    continue
                if target in sums:
                    sums[target][0] += pano[0, j]
                    sums[target][1] += 1
                else:
                    sums[target] = [pano[0, j], 1]
            vpids = [None] + visited + [target for target in sums if target not in visited]
            count = len(vpids)
            pairs = torch.zeros(1, count, count)
            for i, a in enumerate(vpids):
                for j, b in enumerate(vpids):
                    if a is not None and b is not None:
                        pairs[0, i, j] = self.env.shortest_distances["scan"][a][b]
            inputs = {
                "txt_embeds": torch.zeros(1, 3, 2), "txt_masks": torch.ones(1, 3, dtype=torch.bool),
                "gmap_img_embeds": torch.stack([torch.zeros(2)] + [sums[node][0] / sums[node][1] for node in vpids[1:]])[None],
                "gmap_step_ids": torch.zeros(1, count, dtype=torch.long), "gmap_pos_fts": torch.zeros(1, count, 7),
                "gmap_masks": torch.ones(1, count, dtype=torch.bool), "gmap_pair_dists": pairs,
                "gmap_visited_masks": torch.tensor([[node in visited for node in vpids]]), "gmap_vpids": [vpids],
                "vp_img_embeds": torch.cat([torch.zeros(1, 1, 2), pano], dim=1),
                "vp_pos_fts": torch.zeros(1, len(targets) + 1, 14),
                "vp_masks": torch.ones(1, len(targets) + 1, dtype=torch.bool),
                "vp_nav_masks": torch.ones(1, len(targets) + 1, dtype=torch.bool),
                "vp_cand_vpids": [[None] + targets], "no_vp_left": [False],
            }
            outputs = self.vln_bert("navigation", inputs)
            if self.decision_hook is not None:
                returned = self.decision_hook(inputs, outputs, [ob], [False], step, trajectory)
                assert returned is outputs
            action = vpids[int(outputs["fused_logits"].argmax())]
            if action is not None:
                trajectory[0]["path"].append([action])
        return trajectory


def collect(tmp_path, *, goal="C", split="train_fit", backup=False):
    store = DiagnosticStore(tmp_path / "local", {"fixture": 1}, backup=tmp_path / "backup" if backup else None)
    agent = FakeAgent(goal)
    observer = DiagnosticObserver(agent, store, split=split)
    trajectory = agent.rollout()
    directory = store.local / store.episode_name("scan", "instruction")
    return agent, observer, store, trajectory, load_episode(directory)


def test_observer_preserves_baseline_and_captures_pre_inplace_sources(tmp_path):
    baseline = FakeAgent().rollout()
    agent, observer, store, trajectory, payload = collect(tmp_path)
    inputs, labels, manifest = payload
    assert trajectory == baseline
    assert agent.env.resets == 1
    assert manifest["num_states"] == 3
    assert manifest["coverage"]["num_pairs"] == 2
    assert manifest["coverage"]["num_multi_source_candidate_states"] == 2
    first = inputs["states"][0]
    assert set(first["nav_inputs"]) == set(NAVIGATION_KEYS)
    assert first["nav_inputs"]["vp_obj_masks"] is None
    assert torch.equal(first["candidate_evidence"]["X"]["sources"][0]["feature"], torch.tensor([0.0, 1.0]))
    assert len(inputs["states"][2]["candidate_evidence"]["X"]["sources"]) == 3
    assert not ({"gt_path", "distance", "arrival_pairs"} & first.keys())
    assert "base_global_logits" in first and "base_local_logits" in first
    assert labels["unarrived_targets"] == [{"target_id": "X", "arrival_observed": False, "outcome": "unknown"}]
    pair = labels["arrival_pairs"][0]
    assert pair["association"]["target_id"] == "B"
    assert pair["training_only"]["arrival_step"] == 1
    assert torch.equal(pair["training_only"]["arrival_feature"], torch.tensor([1.0, 2.5]))
    assert labels["states"][2]["stop"]["exact_goal"] is True
    assert labels["states"][2]["teacher_optimal_indices"] == [0]
    observer.close()


def test_changing_ground_truth_does_not_change_policy_inputs(tmp_path):
    a = collect(tmp_path / "a", goal="C")
    b = collect(tmp_path / "b", goal="X")
    equal(a[-1][0], b[-1][0])
    assert a[3] == b[3]
    assert a[-1][1]["states"][-1]["stop"] != b[-1][1]["states"][-1]["stop"]
    a[1].close()
    b[1].close()


def test_completed_episode_is_skipped_and_cloud_restores_after_vm_loss(tmp_path):
    agent, observer, store, trajectory, payload = collect(tmp_path, backup=True)
    equal(agent.rollout(), trajectory)
    assert agent.actual_rollouts == 1
    assert observer.reused_episodes == 1
    observer.close()
    shutil.rmtree(store.local)
    restored = DiagnosticStore(tmp_path / "new-local", {"fixture": 1}, backup=tmp_path / "backup")
    other = FakeAgent()
    attached = DiagnosticObserver(other, restored, split="train_fit")
    assert other.rollout() == trajectory
    assert other.actual_rollouts == 0
    assert (restored.local / restored.episode_name("scan", "instruction") / "COMMITTED").is_file()
    attached.close()


def test_train_dev_labels_are_explicitly_analysis_only(tmp_path):
    agent, observer, store, trajectory, payload = collect(tmp_path, split="train_dev")
    assert payload[1]["usage"] == payload[2]["usage"] == "analysis_only"
    observer.close()


@pytest.mark.parametrize("filename", ["inputs.pt", "labels.pt", "manifest.json", "COMMITTED"])
def test_corruption_is_rejected_before_reuse(tmp_path, filename):
    agent, observer, store, trajectory, payload = collect(tmp_path)
    directory = store.local / store.episode_name("scan", "instruction")
    (directory / filename).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_episode(directory)
    with pytest.raises(ValueError):
        agent.rollout()
    observer.close()


def test_incomplete_episode_is_recollected_without_using_partial_files(tmp_path):
    store = DiagnosticStore(tmp_path / "local", {"fixture": 1})
    partial = store.local / ".pending-interrupted"
    partial.mkdir()
    (partial / "inputs.pt").write_bytes(b"partial")
    agent = FakeAgent()
    observer = DiagnosticObserver(agent, store, split="train_fit")
    agent.rollout()
    assert observer.new_episodes == 1
    assert (partial / "inputs.pt").read_bytes() == b"partial"
    assert store.find("scan", "instruction") is not None
    observer.close()


def test_backup_failure_keeps_committed_local_episode(tmp_path, monkeypatch):
    store = DiagnosticStore(tmp_path / "local", {"fixture": 1}, backup=tmp_path / "backup")
    def failure(source, destination):
        raise OSError("simulated upload failure")
    monkeypatch.setattr(store, "_copy_episode", failure)
    agent = FakeAgent()
    observer = DiagnosticObserver(agent, store, split="train_fit")
    with pytest.raises(OSError, match="simulated"):
        agent.rollout()
    directory = store.local / store.episode_name("scan", "instruction")
    assert load_episode(directory)[2]["num_states"] == 3
    observer.close()


def test_collection_identity_and_replay_identity_are_strict(tmp_path):
    agent, observer, store, trajectory, payload = collect(tmp_path)
    with pytest.raises(ValueError, match="identity changed"):
        DiagnosticStore(store.local, {"fixture": 2})
    directory = store.local / store.episode_name("scan", "instruction")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_episode(directory, expected_identity_sha256="different")
    observer.close()


@pytest.mark.parametrize("setting", ["batch_size", "act_visited_nodes", "enc_full_graph", "max_action_len"])
def test_unsupported_rollout_protocol_is_rejected(tmp_path, setting):
    agent = FakeAgent()
    setattr(agent.args, setting, {"batch_size": 4, "act_visited_nodes": True,
                                "enc_full_graph": False, "max_action_len": 30}[setting])
    store = DiagnosticStore(tmp_path / "local", {"fixture": 1})
    with pytest.raises(ValueError):
        DiagnosticObserver(agent, store, split="train_fit")


def test_sampling_is_path_unique_scene_balanced_and_deterministic():
    records = [{"scan": f"scan-{scan}", "path_id": path, "instr_id": f"{scan}-{path}-{instruction}"}
               for scan in range(5) for path in range(7) for instruction in range(3)]
    first = select_diagnostic_records(records, per_scan=3, seed=5, max_episodes=12)
    assert first == select_diagnostic_records(list(reversed(records)), per_scan=3, seed=5, max_episodes=12)
    assert len({(row["scan"], row["path_id"]) for row in first}) == len(first) == 12
    assert len({row["scan"] for row in first[:5]}) == 5
    assert len(select_diagnostic_records(records, per_scan=3, max_scans=2)) == 6


def test_replay_navigation_whitelist_matches_counterfactual_module():
    from vln_improve.counterfactual import NAV_KEYS
    assert set(NAVIGATION_KEYS) == NAV_KEYS
