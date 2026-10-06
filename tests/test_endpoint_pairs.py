import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from vln_improve.endpoint_pairs import (
    SCHEMA, PairStore, collect_pair, content_hash, forced_rollout, nontext_hashes,
    selected_pairs, validate_pair,
)
from vln_improve.protocol import object_sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_test_upstream_graph", ROOT / "third_party/VLN-DUET/map_nav_src/models/graph_utils.py")
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)
GraphMap = graph_module.GraphMap

POSITIONS = {"s": (0, 0, 0), "a": (2, 0, 0), "ga": (4, 0, 0), "b": (-2, 0, 0), "gb": (-4, 0, 0)}
ADJ = {"s": ["a", "b"], "a": ["s", "ga"], "ga": ["a"], "b": ["s", "gb"], "gb": ["b"]}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def pair_fixture():
    row = {"scan": "scene", "start": "s", "path_ids": ["1", "2"], "instr_ids": ["1_0", "2_0"],
           "goal_vpids": ["ga", "gb"], "heading_rad": [0., 0.], "heading_difference_deg": 0.,
           "goal_separation_m": 8., "selection_hash": "a" * 64, "histories": {}}
    for order, vps in (("A_then_B", ["s", "a", "ga", "b", "gb"]),
                       ("B_then_A", ["s", "b", "gb", "a", "ga"])):
        row["histories"][order] = {"observed_vpids": vps}
    return row


class FakeEnv:
    def __init__(self):
        self.env = self
        self.batch = None
        self.buffered_state_dict = {"warm": True}
        self.resets = 0
        self.shortest_distances = {"scene": {a: {b: float(abs(POSITIONS[a][0] - POSITIONS[b][0]))
                                                 for b in POSITIONS} for a in POSITIONS}}

    def newEpisodes(self, scans, nodes, headings):
        assert self.buffered_state_dict == {}
        self.current, self.heading = nodes[0], headings[0]
        self.resets += 1

    def _get_obs(self):
        row = self.batch[0]
        return [{"instr_id": row["instr_id"], "scan": row["scan"], "viewpoint": self.current,
                 "heading": self.heading, "elevation": 0., "viewIndex": 12,
                 "position": POSITIONS[self.current], "instr_encoding": row["instr_encoding"],
                 "candidate": [{"viewpointId": v, "position": POSITIONS[v]} for v in ADJ[self.current]]}]


class FakeModel:
    def __init__(self):
        self.languages = []

    def __call__(self, mode, inputs):
        if mode == "language":
            self.languages.append(inputs["txt_ids"].clone())
            return inputs["txt_ids"].float().unsqueeze(-1).expand(-1, -1, 768)
        if mode == "panorama":
            feature = inputs["view_img_fts"]
            return feature, torch.ones(feature.shape[:2], dtype=torch.bool)
        if mode == "navigation":
            text = inputs["txt_embeds"].mean(1, keepdim=True)
            global_embeds = inputs["gmap_img_embeds"] + text
            local_embeds = inputs["vp_img_embeds"] + text
            legal = inputs["gmap_masks"] & ~inputs["gmap_visited_masks"]
            logits = global_embeds[..., 0].masked_fill(~legal, -torch.inf)
            return {"gmap_embeds": global_embeds, "vp_embeds": local_embeds, "fused_logits": logits}
        raise AssertionError(mode)


class FakeAgent:
    def __init__(self):
        self.env, self.vln_bert = FakeEnv(), FakeModel()
        self.args = SimpleNamespace(batch_size=1, max_action_len=15)
        self.scanvp_cands = {}

    def _update_scanvp_cands(self, obs):
        pass

    def _language_variable(self, obs):
        ids = torch.tensor([obs[0]["instr_encoding"]])
        return {"txt_ids": ids, "txt_masks": torch.ones_like(ids, dtype=torch.bool)}

    def _panorama_feature_variable(self, obs):
        candidates = [c["viewpointId"] for c in obs[0]["candidate"]]
        n = len(candidates)
        return {"view_img_fts": torch.ones(1, n, 768) * (list(POSITIONS).index(obs[0]["viewpoint"]) + 1),
                "loc_fts": torch.zeros(1, n, 7), "nav_types": torch.ones(1, n, dtype=torch.long),
                "view_lens": torch.tensor([n]), "cand_vpids": [candidates]}

    def _nav_gmap_variable(self, obs, maps):
        gmap = maps[0]
        ids = [None] + list(gmap.node_positions)
        n = len(ids)
        dists = torch.zeros(1, n, n)
        for i, a in enumerate(ids[1:], 1):
            for j, b in enumerate(ids[1:], 1):
                dists[0, i, j] = float(gmap.graph.distance(a, b))
        return {"gmap_vpids": [ids], "gmap_img_embeds": torch.stack([torch.zeros(768)] + [gmap.get_node_embed(v) for v in ids[1:]])[None],
                "gmap_step_ids": torch.tensor([[0] + [gmap.node_step_ids.get(v, 0) for v in ids[1:]]]),
                "gmap_pos_fts": torch.zeros(1, n, 7), "gmap_pair_dists": dists,
                "gmap_masks": torch.ones(1, n, dtype=torch.bool),
                "gmap_visited_masks": torch.tensor([[False] + [gmap.graph.visited(v) for v in ids[1:]]]),
                "no_vp_left": [False]}

    def _nav_vp_variable(self, obs, maps, embeds, ids, lengths, nav_types):
        n = embeds.shape[1] + 1
        return {"vp_img_embeds": torch.cat([torch.zeros(1, 1, 768), embeds], 1),
                "vp_pos_fts": torch.zeros(1, n, 14), "vp_masks": torch.ones(1, n, dtype=torch.bool),
                "vp_nav_masks": torch.ones(1, n, dtype=torch.bool), "vp_cand_vpids": [[None] + ids[0]]}

    def make_equiv_action(self, actions, maps, obs, traj):
        target = actions[0]
        if target is not None:
            segment = maps[0].graph.path(obs[0]["viewpoint"], target)
            traj[0]["path"].append(segment)
            self.env.current = target
            self.env.heading = 0.5


def records_fixture():
    return {"1_0": {"instr_id": "1_0", "scan": "scene", "path_id": 1, "heading": 0.,
                     "path": ["s", "a", "ga"], "instruction": "go right", "instr_encoding": [11]},
            "2_0": {"instr_id": "2_0", "scan": "scene", "path_id": 2, "heading": 0.,
                     "path": ["s", "b", "gb"], "instruction": "go left", "instr_encoding": [22]}}


def identity_fixture(pair):
    return {"schema": SCHEMA, "split": "train_fit", "usage": "training", "selection": [pair],
            "selection_sha256": object_sha256([pair]), "code_files": {"test": "1" * 64}}


def payload_fixture():
    pair = pair_fixture()
    identity = identity_fixture(pair)
    agent = FakeAgent()
    return agent, identity, collect_pair(agent, pair, records_fixture(), GraphMap, object_sha256(identity))


def test_four_fresh_language_forwards_identical_history_and_separate_labels():
    agent, identity, payload = payload_fixture()
    assert agent.env.resets == 4
    assert [int(x[0, 0]) for x in agent.vln_bert.languages] == [11, 22, 11, 22]
    for order in payload["rollouts"].values():
        a, b = order["A"], order["B"]
        assert content_hash(a["states"]) == content_hash(b["states"])
        assert not torch.equal(a["features"], b["features"])
        assert a["features"].shape == (5, 1536)
        assert a["actual_length_m"] == 12
        assert len(sum(a["trajectory"], [])) == 7
        assert len(a["states"]) == 5  # revisited transit nodes are not observations
        assert torch.equal(a["labels"]["within_success_radius"], b["labels"]["within_success_radius"])
        assert a["forced_actions"][-1] is None
    validate_pair(payload, object_sha256(identity), pair_fixture())


@pytest.mark.parametrize("mutation", ["heading", "mask", "hash", "prefix", "label", "identical_language", "missing_rollout"])
def test_complete_pair_rejects_any_shared_history_or_label_mismatch(mutation):
    _, identity, payload = payload_fixture()
    run = payload["rollouts"]["A_then_B"]["B"]
    if mutation == "heading": run["states"][0]["heading"] += 1e-12
    if mutation == "mask": run["states"][0]["masks"]["gmap_masks"][0] = False
    if mutation == "hash": run["states"][0]["panorama_sha256"] = "f" * 64
    if mutation == "prefix": run["states"][1]["trajectory_prefix"] = [["s"], ["b", "a"]]
    if mutation == "label": run["labels"]["within_success_radius"][0, 0] = True
    if mutation == "identical_language": run["language_input_sha256"] = payload["rollouts"]["A_then_B"]["A"]["language_input_sha256"]
    if mutation == "missing_rollout": del payload["rollouts"]["B_then_A"]["B"]
    with pytest.raises(ValueError):
        validate_pair(payload, object_sha256(identity), pair_fixture())


def test_repeat_target_rejected_before_execution():
    agent = FakeAgent()
    with pytest.raises(ValueError, match="uniqueness"):
        forced_rollout(agent, records_fixture()["1_0"], ["s", "a", "s"], GraphMap)
    assert agent.env.resets == 0


def test_nontext_hash_excludes_only_text_and_rejects_unknown_input():
    from vln_improve.endpoint_pairs import NAV_KEYS, PANO_KEYS
    pano = {k: [] for k in PANO_KEYS}
    nav = {k: [] for k in NAV_KEYS}
    nav["gmap_img_embeds"] = torch.zeros(1, 2, 768)
    nav["txt_embeds"] = torch.zeros(1, 2, 768)
    original = nontext_hashes(pano, nav)
    nav["txt_embeds"].add_(1)
    assert nontext_hashes(pano, nav) == original
    nav["gmap_img_embeds"].add_(1e-12)
    assert nontext_hashes(pano, nav) != original
    nav["distance_to_goal"] = 0
    with pytest.raises(ValueError, match="unexpected"):
        nontext_hashes(pano, nav)


def test_pair_commit_backup_restore_and_no_partial_group(tmp_path):
    _, identity, payload = payload_fixture()
    local, backup = tmp_path / "local", tmp_path / "backup"
    store = PairStore(local, backup, identity, lambda: None)
    broken = copy.deepcopy(payload)
    del broken["rollouts"]["B_then_A"]
    with pytest.raises(ValueError): store.commit(broken)
    assert not list(local.glob("pair-*"))
    manifest = store.commit(payload)
    assert manifest["states"] == 20 and manifest["shared_history_exact_parity"]
    name = "pair-" + payload["pair"]["selection_hash"]
    shutil.rmtree(local / name)
    assert store.find(payload["pair"]) == manifest
    assert (local / name / "data.pt").is_file()
    with pytest.raises(ValueError, match="replace"): store.commit(payload)
    summary = store.seal({"wall_seconds": 1.})
    assert summary["pairs"] == 1 and summary["rollouts"] == 4 and summary["navigation_metrics"] is None
    with (backup / name / "data.pt").open("ab") as stream: stream.write(b"corrupt")
    with pytest.raises(ValueError, match="identity"): store.find(payload["pair"])


def test_resume_finishes_missing_drive_copy_and_rejects_changed_code(tmp_path, monkeypatch):
    _, identity, payload = payload_fixture()
    store = PairStore(tmp_path / "local", tmp_path / "backup", identity, lambda: None)
    original_copy = store._copy
    monkeypatch.setattr(store, "_copy", lambda *args: (_ for _ in ()).throw(OSError("disconnect")))
    with pytest.raises(OSError): store.commit(payload)
    assert len(list(store.local.glob("pair-*"))) == 1
    assert not list(store.backup.glob("pair-*"))
    monkeypatch.setattr(store, "_copy", original_copy)
    assert store.find(payload["pair"])["states"] == 20
    changed = copy.deepcopy(identity); changed["code_files"]["test"] = "2" * 64
    with pytest.raises(ValueError, match="identity changed"):
        PairStore(store.local, store.backup, changed, lambda: None)


def test_pair_selection_fixed_hash_order_and_train_only():
    pair = pair_fixture()
    other = copy.deepcopy(pair); other["selection_hash"] = "0" * 64
    other["path_ids"] = ["3", "4"]; other["instr_ids"] = ["3_0", "4_0"]
    rows = [pair, other]
    report = {"schema": "duet_endpoint_pair_coverage_v1", "coverage_pass": True,
              "splits": {"train_fit": {"primary_path_disjoint_manifest": rows, "primary_manifest_sha256": object_sha256(rows)}}}
    assert selected_pairs(report, "train_fit", 1) == [other]
    with pytest.raises(ValueError): selected_pairs(report, "val_unseen", 1)
    with pytest.raises(ValueError): selected_pairs(report, "train_fit", 3)


def test_cli_configuration_cannot_expand_pilot():
    sys.path.insert(0, str(ROOT / "scripts"))
    from collect_endpoint_pairs import validate_spec
    spec = json.loads((ROOT / "configs/endpoint_pair_collection.json").read_text())
    config = json.loads((ROOT / "configs/r2r.json").read_text())
    validate_spec(spec, config)
    spec["selection"]["train_fit"] = 512
    with pytest.raises(ValueError, match="expansion"): validate_spec(spec, config)
