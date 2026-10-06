import copy

import pytest
import torch

from vln_improve.counterfactual import (
    NAV_KEYS, candidate_margin, evaluate_replacement, navigation_hash,
    replace_candidate, replay_identity, observable_features,
)


def inputs():
    return {
        "txt_embeds": torch.ones(1, 2, 4), "txt_masks": torch.ones(1, 2, dtype=torch.bool),
        "gmap_img_embeds": torch.tensor([[[0., 0., 0., 0.], [1., 0., 0., 0.], [2., 0., 0., 0.]]]),
        "gmap_step_ids": torch.zeros(1, 3, dtype=torch.long), "gmap_pos_fts": torch.zeros(1, 3, 7),
        "gmap_masks": torch.ones(1, 3, dtype=torch.bool), "gmap_visited_masks": torch.tensor([[False, True, False]]),
        "gmap_pair_dists": torch.zeros(1, 3, 3), "gmap_vpids": [[None, "visited", "candidate"]],
        "vp_img_embeds": torch.zeros(1, 2, 4), "vp_pos_fts": torch.zeros(1, 2, 14),
        "vp_masks": torch.ones(1, 2, dtype=torch.bool), "vp_nav_masks": torch.ones(1, 2, dtype=torch.bool),
        "vp_obj_masks": None, "vp_cand_vpids": [[None, "candidate"]],
    }


def model(data):
    features = data["gmap_img_embeds"]
    # Mixing deliberately changes both the substituted action and STOP.
    z = features[:, :, 0] + features[:, :, 0].sum(1, keepdim=True) * torch.tensor([1., 0., 2.])
    return {"fused_logits": z.masked_fill(data["gmap_visited_masks"], -torch.inf)}


def test_replay_identity_and_full_model_intervention_preserve_input():
    data = inputs(); before = navigation_hash(data)
    baseline = replay_identity(model, data, model(data)["fused_logits"][0])
    assert baseline["max_absolute_error"] == 0
    result = evaluate_replacement(model, data, baseline["logits"], 2, torch.tensor([-3., 0., 0., 0.]))
    assert result["stop_logit_shift"] != 0
    assert result["argmax_changed"]
    assert result["logits"][1] is None
    assert navigation_hash(data) == before


@pytest.mark.parametrize("index", [0, 1, 3, -1])
def test_replacement_requires_legal_unvisited_nonstop(index):
    with pytest.raises(ValueError): replace_candidate(inputs(), index, torch.ones(4))


def test_replacement_copy_does_not_alias_feature_or_original():
    data = inputs(); future = torch.ones(4)
    alternate = replace_candidate(data, 2, future)
    future.zero_(); alternate["gmap_vpids"][0][2] = "changed"
    assert torch.equal(alternate["gmap_img_embeds"][0, 2], torch.ones(4))
    assert data["gmap_vpids"][0][2] == "candidate"
    assert data["gmap_img_embeds"][0, 2, 0] == 2


def test_replay_mismatch_is_not_silently_accepted():
    with pytest.raises(AssertionError): replay_identity(model, inputs(), torch.tensor([3., -torch.inf, 100.]))


def test_reject_hidden_fields_and_nonfinite_future():
    data = inputs(); data["gt_path"] = ["hidden"]
    with pytest.raises(ValueError): navigation_hash(data)
    with pytest.raises(ValueError): replace_candidate(inputs(), 2, torch.full((4,), torch.nan))


def test_margin_ignores_masked_scores_and_common_logit_offset():
    z = torch.tensor([2., 1000., 3.]); mask = torch.tensor([True, False, True])
    assert candidate_margin(z, mask, 2) == 1
    assert candidate_margin(z + 2000, mask, 2) == 1


def source_state():
    from vln_improve.evidence import EpisodeEvidenceMemory
    memory = EpisodeEvidenceMemory("episode", scan_id="scan", instr_id="instruction", feature_dim=4)
    memory.observe_proxy("candidate", "visited", step=0, heading=0., elevation=0.,
                         relative_position=[1., 0., 0.], feature=torch.tensor([2., 0., 0., 0.]))
    data = inputs()
    return {"nav_inputs": data, "base_logits": model(data)["fused_logits"][0],
            "base_global_logits": torch.tensor([1., -torch.inf, 3.]),
            "base_local_logits": torch.tensor([2., 4.]), "step": 1,
            "candidate_evidence": {"candidate": memory.policy_snapshot("candidate")}}


def test_observable_probe_uses_current_inputs_and_coverage_in_strong_baseline():
    state = source_state()
    p0, p1 = observable_features(state, 2)
    assert p0["coverage_source_count_total"] == 1
    assert "global_candidate_log_probability" in p0
    assert p1["source_cosine_mean"] == 1
    changed = copy.deepcopy(state)
    changed["training_only"] = {"future_arrival": torch.randn(4), "goal": "a hidden target"}
    assert observable_features(changed, 2) == (p0, p1)


def test_candidate_evidence_join_rejects_another_target():
    state = source_state()
    state["candidate_evidence"]["candidate"]["association"]["target_id"] = "another"
    with pytest.raises(ValueError, match="different target"):
        observable_features(state, 2)


@pytest.mark.parametrize("field", ["available_at_step", "feature_step", "last_step"])
def test_future_source_join_is_rejected(field):
    state = source_state()
    snap = state["candidate_evidence"]["candidate"]
    if field == "available_at_step": snap[field] = 3
    else: snap["sources"][0][field] = 3
    with pytest.raises(ValueError, match="unavailable|future"):
        observable_features(state, 2)
