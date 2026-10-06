"""E2 policy isolation, cost targets, alignment, and trainable small-head checks."""
import dataclasses
import inspect

import pytest
import torch

from vln_improve.endpoint_intervention import (
    SCALAR_NAMES, InterventionHead, build_intervention_inputs,
    build_intervention_targets, collate_interventions, intervention_loss,
    select_intervention,
)


def fixture(*, order=(None, "A", "TRANSIT", "B", "FRONTIER"), extra_padding=1):
    values = {None: 10., "A": 1., "TRANSIT": 99., "B": 2., "FRONTIER": 88.}
    width = len(order) + extra_padding
    global_tokens = torch.full((1, width, 768), float("nan"))
    for index, node in enumerate(order):
        global_tokens[0, index] = values[node]
    global_tokens.requires_grad_()
    mask = torch.tensor([[True] * len(order) + [False] * extra_padding])
    visited = torch.tensor([[v in ("A", "TRANSIT", "B") for v in order] + [False] * extra_padding])
    nav = {"gmap_vpids": [list(order)], "gmap_masks": mask,
           "gmap_visited_masks": visited, "vp_masks": torch.ones(1, 3, dtype=torch.bool),
           "vp_cand_vpids": [[None, "A", "FRONTIER"]]}
    out = {"gmap_embeds": global_tokens, "vp_embeds": torch.full((1, 3, 768), 3.)}
    kwargs = {"observed_vpids": ["A", "B"], "visit_steps": {"A": 0, "B": 1},
              "stop_probabilities": {"A": .9, "B": .2}, "baseline_vpid": "A",
              "termination_vpid": "B", "prefix_length_m": 8.,
              "return_distances_m": {"A": 4., "B": 0.}}
    return nav, out, kwargs


def example(**kwargs):
    nav, out, args = fixture(**kwargs)
    return build_intervention_inputs(nav, out, **args)


def test_final_tokens_align_by_id_and_exclude_visited_transit_and_frontier():
    a = example()
    b = example(order=(None, "B", "FRONTIER", "TRANSIT", "A"))
    assert a.candidate_vpids == ("A", "B")
    assert torch.equal(a.node_features, b.node_features)
    assert torch.equal(a.scalar_features, b.scalar_features)
    assert torch.equal(a.node_features[:, 0], torch.tensor([1., 2.]))
    assert torch.equal(a.terminal_context[:768], torch.full((768,), 10.))
    assert not a.node_features.requires_grad
    assert not a.terminal_context.requires_grad


def test_candidate_order_changes_only_alignment_not_node_identity():
    nav, out, args = fixture()
    a = build_intervention_inputs(nav, out, **args)
    args["observed_vpids"] = ["B", "A"]
    b = build_intervention_inputs(nav, out, **args)
    assert b.baseline_index == 1
    assert torch.equal(a.node_features.flip(0), b.node_features)
    assert torch.equal(a.scalar_features.flip(0), b.scalar_features)


def test_exact_observable_cost_and_age_normalization():
    x = example()
    cols = {k: i for i, k in enumerate(SCALAR_NAMES)}
    assert x.scalar_features[0, cols["candidate_return_over_prefix_plus_one"]] == pytest.approx(4 / 9)
    assert x.scalar_features[1, cols["return_difference_over_prefix_plus_one"]] == pytest.approx(-4 / 9)
    assert x.scalar_features[:, cols["candidate_visit_age_fraction"]].tolist() == [1., 0.]


class GuardedDict(dict):
    def __getitem__(self, key):
        assert key not in {"goal", "gt_path", "distance_to_goal", "reference_path_length_m"}
        return super().__getitem__(key)


def test_policy_interface_never_reads_goal_supervision_or_mutates_navigation():
    nav, out, args = fixture()
    before = out["gmap_embeds"].detach().clone()
    nav = GuardedDict(nav, goal="FORBIDDEN", gt_path="FORBIDDEN", distance_to_goal="FORBIDDEN")
    build_intervention_inputs(nav, GuardedDict(out, goal="FORBIDDEN"), **args)
    torch.testing.assert_close(out["gmap_embeds"], before, equal_nan=True)
    parameters = inspect.signature(build_intervention_inputs).parameters
    assert not set(parameters) & {"goal", "candidate_goal_distances_m", "reference_path_length_m"}
    assert not set(InterventionHead.forward.__annotations__) & {"target_gains"}


@pytest.mark.parametrize("defect", ["masked", "unvisited", "missing", "bad_baseline", "bad_steps",
                                    "nonzero_terminal_return", "bad_probability", "extra_cost",
                                    "stop_mask", "local_stop_id", "nonfinite_token", "padding_mask"])
def test_rejects_corrupted_observation_identity_or_inputs(defect):
    nav, out, args = fixture()
    if defect == "masked": nav["gmap_masks"][0, 1] = False
    elif defect == "unvisited": nav["gmap_visited_masks"][0, 1] = False
    elif defect == "missing": nav["gmap_vpids"][0][1] = "OTHER"
    elif defect == "bad_baseline": args["baseline_vpid"] = "B"
    elif defect == "bad_steps": args["visit_steps"]["B"] = 0
    elif defect == "nonzero_terminal_return": args["return_distances_m"]["B"] = .1
    elif defect == "bad_probability": args["stop_probabilities"]["A"] = 1.01
    elif defect == "extra_cost": args["return_distances_m"]["TRANSIT"] = 1.
    elif defect == "stop_mask": nav["gmap_masks"][0, 0] = False
    elif defect == "local_stop_id": nav["vp_cand_vpids"][0][0] = "A"
    elif defect == "nonfinite_token":
        out["gmap_embeds"] = out["gmap_embeds"].detach()
        out["gmap_embeds"][0, 1, 0] = float("nan")
    elif defect == "padding_mask": nav["gmap_masks"][0, -1] = True
    with pytest.raises(ValueError):
        build_intervention_inputs(nav, out, **args)


def test_original_probability_tie_uses_first_chronological_node():
    nav, out, args = fixture()
    args["stop_probabilities"] = {"A": .5, "B": .5}
    args["observed_vpids"] = ["B", "A"]
    assert build_intervention_inputs(nav, out, **args).baseline_index == 1
    args["baseline_vpid"] = "B"
    with pytest.raises(ValueError, match="original STOP"):
        build_intervention_inputs(nav, out, **args)


def singleton():
    x = example()
    return dataclasses.replace(x, candidate_vpids=("A",), node_features=x.node_features[:1],
                               scalar_features=x.scalar_features[:1])


def test_small_zero_initialized_head_preserves_baseline_for_padded_batches():
    batch = collate_interventions([example(), singleton()])
    head = InterventionHead()
    assert sum(p.numel() for p in head.parameters()) == 362370
    prediction = head(batch)
    assert prediction.shape == (2, 2, 2)
    assert torch.equal(prediction, torch.zeros_like(prediction))
    assert select_intervention(prediction[0], batch.candidate_vpids[0], "A", valid_mask=batch.valid_mask[0]) == "A"
    assert select_intervention(prediction[1], batch.candidate_vpids[1], "A", valid_mask=batch.valid_mask[1]) == "A"


def test_masked_padding_cannot_change_outputs_or_loss_even_if_nan():
    batch = collate_interventions([example(), singleton()])
    head = InterventionHead()
    torch.nn.init.normal_(head.comparison[-1].weight)
    before = head(batch)
    batch.node_features[1, 1] = float("nan")
    batch.scalar_features[1, 1] = float("nan")
    after = head(batch)
    assert torch.equal(before, after)
    target = torch.zeros_like(after)
    loss = intervention_loss(after, target, batch.valid_mask, batch.baseline_indices)["loss"]
    target[1, 1] = float("nan")
    after = after.clone()
    after[1, 1] = float("nan")
    other = intervention_loss(after, target, batch.valid_mask, batch.baseline_indices)["loss"]
    assert torch.equal(loss, other)


@pytest.mark.parametrize("scores,expected", [
    ([[0., 0.], [0., 0.], [0., 0.]], "A"),
    ([[0., 0.], [-.1, .5], [.2, -.1]], "A"),
    ([[0., 0.], [0., .1], [.2, .3]], "C"),
    ([[0., 0.], [0., .3], [.2, .3]], "A"),
    ([[0., 0.], [0., .1], [.2, 0.]], "B"),
])
def test_selection_requires_both_gains_and_retains_baseline_on_ties(scores, expected):
    assert select_intervention(torch.tensor(scores), ("A", "B", "C"), "A") == expected


def test_selection_ignores_invalid_candidates_and_rejects_nonzero_anchor():
    scores = torch.tensor([[0., 0.], [0., .1], [1., .9], [float("nan"), float("nan")]])
    mask = torch.tensor([True, True, False, False])
    assert select_intervention(scores, ("A", "B", "C"), "A", valid_mask=mask) == "B"
    scores[0, 1] = .01
    with pytest.raises(ValueError, match="baseline prediction"):
        select_intervention(scores, ("A", "B", "C"), "A", valid_mask=mask)


def test_targets_match_reference_path_spl_and_full_return_cost():
    # Reference path is 12 m although start-goal geodesic could be 8 m.
    # A is baseline, B is successful but wastes 12 m returning, C fails at 3 m.
    target = build_intervention_targets(candidate_goal_distances_m=torch.tensor([2., 0., 3., 1.]),
        candidate_total_lengths_m=torch.tensor([12., 24., 12., 9.]),
        reference_path_length_m=12., baseline_index=0)
    assert target.dtype == torch.float64
    assert torch.equal(target, torch.tensor([[0., 0.], [0., -.5], [-1., -1.], [0., 0.]], dtype=torch.float64))
    # Success gain is positive but return cost reduces achievable SPL gain.
    rescued = build_intervention_targets(candidate_goal_distances_m=torch.tensor([4., 2.]),
        candidate_total_lengths_m=torch.tensor([12., 30.]), reference_path_length_m=12., baseline_index=0)
    torch.testing.assert_close(rescued[1], torch.tensor([1., .4], dtype=torch.float64))


def test_all_failed_targets_and_degenerate_reference_length_are_finite():
    targets = build_intervention_targets(candidate_goal_distances_m=torch.tensor([3., 8.]),
        candidate_total_lengths_m=torch.tensor([0., 2.]), reference_path_length_m=0., baseline_index=0)
    assert torch.equal(targets, torch.zeros_like(targets))
    assert select_intervention(targets, ("A", "B"), "A") == "A"


def test_cost_supervision_is_not_constrained_to_execution_coordinate_rounding():
    x = example()
    # An official-connectivity metric may differ from observable cost scalars.
    target = build_intervention_targets(candidate_goal_distances_m=torch.tensor([4., 1.]),
        candidate_total_lengths_m=torch.tensor([12.0000007, 8.0000001], dtype=torch.float64),
        reference_path_length_m=9.0000003, baseline_index=x.baseline_index)
    assert target[1, 1] == 1.


def test_loss_is_per_episode_excludes_anchor_and_exposes_optional_harm_penalty():
    p = torch.tensor([[[0., 0.], [.4, .4], [.4, .4]], [[0., 0.], [.4, .4], [999., 999.]]], requires_grad=True)
    y = torch.tensor([[[0., 0.], [-1., -.5], [-1., -.5]], [[0., 0.], [-1., -.5], [999., 999.]]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    anchors = torch.tensor([0, 0])
    base = intervention_loss(p, y, mask, anchors)
    risk = intervention_loss(p, y, mask, anchors, risk_weight=2.)
    assert risk["loss"] > base["loss"]
    assert risk["false_gain_penalty"].item() == pytest.approx(.12)
    risk["loss"].backward()
    assert p.grad[0, 1].min() > 0  # gradient descent reduces harmful positive predictions
    assert torch.equal(p.grad[:, 0], torch.zeros_like(p.grad[:, 0]))
    assert torch.equal(p.grad[1, 2], torch.zeros_like(p.grad[1, 2]))
    single = intervention_loss(p[:1, :2], y[:1, :2], mask[:1, :2], anchors[:1])
    assert torch.equal(single["loss"], base["loss"])


def test_synthetic_training_and_state_reload_preserve_predictions_and_anchor():
    torch.manual_seed(3)
    batch = collate_interventions([example(), singleton()])
    head = InterventionHead()
    target = torch.tensor([[[0., 0.], [1., .4]], [[0., 0.], [0., 0.]]])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
    initial = intervention_loss(head(batch), target, batch.valid_mask, batch.baseline_indices)["loss"].item()
    for _ in range(8):
        optimizer.zero_grad()
        loss = intervention_loss(head(batch), target, batch.valid_mask, batch.baseline_indices)["loss"]
        loss.backward()
        optimizer.step()
    final = head(batch)
    assert intervention_loss(final, target, batch.valid_mask, batch.baseline_indices)["loss"] < initial
    other = InterventionHead()
    other.load_state_dict(head.state_dict())
    assert torch.equal(final, other(batch))
    assert torch.equal(final[:, 0], torch.zeros_like(final[:, 0]))
    assert torch.equal(final[1, 1], torch.zeros_like(final[1, 1]))
