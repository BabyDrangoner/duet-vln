import copy
import math

import pytest
import torch
import torch.nn.functional as F

from vln_improve.endpoint_group_objectives import (
    group_objective, per_group_metrics, prepare_group_batch, score_group_batch,
)
from vln_improve.endpoint_probe import EndpointProbe


def run(n, positive, *, offset=0):
    x = torch.arange(n, dtype=torch.float32)[:, None].expand(n, 1536).clone() + offset
    y = torch.tensor([i in positive for i in range(n)], dtype=torch.bool)
    return {"features": x, "labels": {"within_success_radius": y,
            "distance_to_goal": torch.where(y, 0., 4.).double()}}


def make_group():
    group = {"pair": {"instr_ids": ["ia", "ib"], "goal_vpids": ["a", "b"]},
             "natural": {"A": run(2, {1}), "B": run(3, {2})},
             "c2": {}, "paired": {}}
    for slot in ("A", "B"):
        group["c2"][slot] = {"reference": run(2, {1}), "overshoot": run(4, {1})}
    for order, vps in (("A_then_B", ["s", "a", "b"]), ("B_then_A", ["s", "b", "a"])):
        group["paired"][order] = {}
        y = torch.tensor([[v == "a", v == "b"] for v in vps], dtype=torch.bool)
        for i, slot in enumerate(("A", "B")):
            r = run(3, {vps.index(("a", "b")[i])}, offset=5*i)
            r.update({"instr_id": ("ia", "ib")[i], "instruction_slot": i,
                      "states": [{"viewpoint": v} for v in vps]})
            r["labels"]["within_success_radius"] = y.clone()
            group["paired"][order][slot] = r
    return group


def test_c1_is_plain_episode_mean_bce_despite_duplicate_slots():
    batch = prepare_group_batch([make_group()], "C1")
    # Deterministic same-feature predictions, including the duplicate slots.
    logits = batch["features"][..., 0]
    actual = group_objective(logits, batch, "C1")["loss"]
    expected = (F.binary_cross_entropy_with_logits(torch.tensor([0., 1.]), torch.tensor([0., 1.]))
                + F.binary_cross_entropy_with_logits(torch.tensor([0., 1., 2.]), torch.tensor([0., 0., 1.]))) / 2
    assert actual.item() == pytest.approx(expected.item())


def test_c3_and_m_share_identical_all_data_and_differ_only_by_fixed_pair_term():
    group = make_group()
    c3, method = [prepare_group_batch([group], arm) for arm in ("C3", "M")]
    for name in ("features", "labels", "mask", "goal_steps"):
        assert torch.equal(c3[name], method[name])
    torch.manual_seed(4)
    logits = torch.randn(1, 6, 15, requires_grad=True)
    c = group_objective(logits, c3, "C3")
    m = group_objective(logits, method, "M")
    assert torch.equal(m["loss"], c["bce"] + .1*c["ranking"])
    m["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[~method["mask"]].any()


def test_label_column_is_conditioned_on_instruction_not_any_goal():
    batch = prepare_group_batch([make_group()], "C3")
    assert batch["labels"][0, 2, :3].tolist() == [0, 1, 0]
    assert batch["labels"][0, 3, :3].tolist() == [0, 0, 1]
    assert batch["labels"][0, 4, :3].tolist() == [0, 0, 1]
    assert batch["labels"][0, 5, :3].tolist() == [0, 1, 0]


def test_diagnostics_require_both_instructions_and_both_orders_and_strict_ties():
    batch = prepare_group_batch([make_group()], "C3")
    logits = 2 * batch["labels"] - 1
    good = per_group_metrics(logits, batch)
    assert good["both_instructions_correct_by_order"].tolist() == [[True, True]]
    assert good["both_orders_correct"].tolist() == [True]
    assert good["ranking"].item() == pytest.approx(F.softplus(torch.tensor(-2.)).item())
    logits[0, 2, :3] = 0  # A row is tied, despite the B row being correct.
    bad = per_group_metrics(logits, batch)
    assert bad["both_instructions_correct_by_order"].tolist() == [[False, True]]
    assert not bad["both_orders_correct"].item()


def test_additive_instruction_plus_place_shortcut_cannot_satisfy_pair():
    batch = prepare_group_batch([make_group()], "C3")
    logits = torch.zeros(1, 6, 15)
    # Different instruction offsets cannot reverse the same place preference.
    for o in range(2):
        a, b = batch["goal_steps"][0, o]
        for i in range(2):
            logits[0, 2+2*o+i, a] = 1 + i*4
            logits[0, 2+2*o+i, b] = -1 + i*4
    assert not per_group_metrics(logits, batch)["both_instructions_correct_by_order"].any()


def test_features_are_detached_and_distance_metadata_cannot_change_head_inputs():
    a = make_group(); b = copy.deepcopy(a)
    a["natural"]["A"]["features"].requires_grad_(True)
    b["natural"]["A"]["labels"]["distance_to_goal"].fill_(900)
    first, second = [prepare_group_batch([g], "C3") for g in (a, b)]
    assert torch.equal(first["features"], second["features"])
    assert not first["features"].requires_grad
    head = EndpointProbe()
    group_objective(score_group_batch(head, first), first, "C3")["loss"].backward()
    assert a["natural"]["A"]["features"].grad is None
    assert all(p.grad is not None for p in head.parameters())


def test_groups_have_equal_weight_despite_different_natural_lengths():
    a, b = make_group(), make_group()
    b["natural"]["A"] = run(15, {10, 11, 12, 13, 14})
    batch = prepare_group_batch([a, b], "C2")
    logits = batch["features"][..., 0]
    per = per_group_metrics(logits, batch)
    assert group_objective(logits, batch, "C2")["loss"].item() == pytest.approx(per["bce"].mean().item())
    for i, g in enumerate((a, b)):
        one = prepare_group_batch([g], "C2")
        assert per["bce"][i].item() == pytest.approx(group_objective(one["features"][..., 0], one, "C2")["loss"].item())


@pytest.mark.parametrize("kind", ["instruction", "slot", "goal", "label", "path"])
def test_rejects_inconsistent_pair_supervision(kind):
    g = make_group(); r = g["paired"]["A_then_B"]["A"]
    if kind == "instruction": r["instr_id"] = "wrong"
    if kind == "slot": r["instruction_slot"] = 1
    if kind == "goal": g["pair"]["goal_vpids"][0] = "missing"
    if kind == "label": r["labels"]["within_success_radius"][2, 0] = True
    if kind == "path": r["states"][1]["viewpoint"] = "different"
    with pytest.raises(ValueError): prepare_group_batch([g], "M")


def test_arm_mismatch_and_invalid_padding_goal_rejected():
    batch = prepare_group_batch([make_group()], "M")
    logits = torch.zeros(1, 6, 15)
    with pytest.raises(ValueError, match="arm"): group_objective(logits, batch, "C3")
    batch["goal_steps"][0, 0, 0] = 10
    with pytest.raises(ValueError, match="padding"): per_group_metrics(logits, batch)
