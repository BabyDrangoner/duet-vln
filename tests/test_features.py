import copy
import math

import pytest
import torch

from vln_improve.features import FEATURE_SCHEMA, build_features


def sample():
    # Local actions have a different order, plus an already visited action.
    inputs = {
        "gmap_vpids": [[None, "a", "b", "remote", "visited"]],
        "vp_cand_vpids": [[None, "b", "visited", "a"]],
        "gmap_masks": torch.tensor([[True, True, True, True, True, False]]),
        "gmap_visited_masks": torch.tensor([[False, False, False, False, True, False]]),
        "vp_nav_masks": torch.tensor([[True, True, True, True, False]]),
        "gmap_pos_fts": torch.arange(18, dtype=torch.float32).reshape(1, 6, 3),
    }
    outputs = {
        "gmap_embeds": torch.arange(12, dtype=torch.float32).reshape(1, 6, 2),
        "vp_embeds": torch.tensor([[[0., 1.], [20., 21.], [90., 91.], [10., 11.], [99., 99.]]]),
        "global_logits": torch.tensor([[1., 2., 3., 4., -torch.inf, -torch.inf]]),
        "fused_logits": torch.tensor([[2., 4., 6., 8., -torch.inf, -torch.inf]]),
        "local_logits": torch.tensor([[0., 2., 9., 1., -torch.inf]]),
    }
    return inputs, outputs


def test_id_alignment_schema_and_dimensions():
    inputs, outputs = sample()
    features, base, valid = build_features(inputs, outputs)
    assert FEATURE_SCHEMA == "duet_action_features_v1"
    assert features.shape == (1, 6, 2 * 2 + 6 + 3)
    torch.testing.assert_close(features[0, 1, 2:4], torch.tensor([10., 11.]))
    torch.testing.assert_close(features[0, 2, 2:4], torch.tensor([20., 21.]))
    assert not features[0, 3, 2:4].any()
    torch.testing.assert_close(features[0, :4, 7], torch.tensor([1., 1., 1., 0.]))
    torch.testing.assert_close(features[0, :4, 8], torch.tensor([1., 0., 0., 0.]))
    torch.testing.assert_close(features[0, :4, 9], torch.full((4,), math.log1p(4)))
    torch.testing.assert_close(base[valid], outputs["fused_logits"][valid])
    assert torch.isneginf(base[~valid]).all()
    assert not features[~valid].any()


def test_local_normalization_includes_visited_directions():
    inputs, outputs = sample()
    features, _, _ = build_features(inputs, outputs)
    expected = torch.log_softmax(outputs["local_logits"][0, :4], dim=0)
    torch.testing.assert_close(features[0, 1, 6], expected[3])
    torch.testing.assert_close(features[0, 2, 6], expected[1])
    assert features[0, 3, 6] == 0  # Nonlocal candidate has no local log probability.


def test_invalid_padding_and_visited_values_do_not_reach_head():
    inputs, outputs = sample()
    outputs["gmap_embeds"][0, 4:] = torch.nan
    outputs["vp_embeds"][0, 2] = torch.nan  # Visited globally, valid locally.
    outputs["vp_embeds"][0, 4] = torch.inf
    inputs["gmap_pos_fts"][0, 4:] = torch.inf
    outputs["global_logits"][0, 4:] = torch.nan
    outputs["fused_logits"][0, 4:] = torch.inf
    outputs["local_logits"][0, 4] = torch.nan
    features, base, valid = build_features(inputs, outputs)
    assert torch.isfinite(features).all()
    assert not features[~valid].any()
    assert torch.isneginf(base[~valid]).all()


def test_features_are_detached_float32_and_ignore_ground_truth():
    inputs, outputs = sample()
    for mapping in (inputs, outputs):
        for key, value in mapping.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                mapping[key] = value.double().requires_grad_()
    reference = build_features(inputs, outputs)
    inputs["gt_path"] = object()
    inputs["obs"] = {"goal": object(), "distance": object()}
    outputs["gt_target"] = object()
    actual = build_features(inputs, outputs)
    for first, second in zip(reference, actual):
        torch.testing.assert_close(first, second)
        assert not second.requires_grad
        assert second.grad_fn is None
    assert actual[0].dtype == actual[1].dtype == torch.float32


@pytest.mark.parametrize("key", ["global_logits", "fused_logits", "local_logits"])
def test_nonfinite_legal_logits_rejected(key):
    inputs, outputs = sample()
    outputs[key][0, 1] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        build_features(inputs, outputs)


def test_no_legal_global_actions_rejected():
    inputs, outputs = sample()
    inputs["gmap_masks"][:] = False
    with pytest.raises(ValueError, match="no legal actions"):
        build_features(inputs, outputs)


def test_no_legal_local_actions_rejected():
    inputs, outputs = sample()
    inputs["vp_nav_masks"][:] = False
    with pytest.raises(ValueError, match="no legal actions"):
        build_features(inputs, outputs)


@pytest.mark.parametrize("name", ["gmap_vpids", "vp_cand_vpids"])
def test_duplicate_ids_and_wrong_stop_rejected(name):
    inputs, outputs = sample()
    inputs[name][0][2] = inputs[name][0][1]
    with pytest.raises(ValueError, match="duplicate"):
        build_features(inputs, outputs)
    inputs, outputs = sample()
    inputs[name][0][0] = "wrong-stop"
    with pytest.raises(ValueError, match="stop ID None"):
        build_features(inputs, outputs)


@pytest.mark.parametrize("name,mask", [("gmap_vpids", "gmap_masks"),
                                      ("vp_cand_vpids", "vp_nav_masks")])
def test_candidate_lengths_and_legal_padding_rejected(name, mask):
    inputs, outputs = sample()
    inputs[name][0].extend(["extra-1", "extra-2"])
    with pytest.raises(ValueError, match="length"):
        build_features(inputs, outputs)
    inputs, outputs = sample()
    inputs[mask][0, -1] = True
    with pytest.raises(ValueError, match="ID length"):
        build_features(inputs, outputs)


def test_log_probabilities_are_clamped():
    inputs, outputs = sample()
    for name in ("global_logits", "fused_logits", "local_logits"):
        outputs[name][0, 0] = -1000
    features, _, _ = build_features(inputs, outputs)
    torch.testing.assert_close(features[0, 0, 4:7], torch.full((3,), -30.0))


def test_inputs_are_not_mutated():
    inputs, outputs = sample()
    saved_inputs, saved_outputs = copy.deepcopy(inputs), copy.deepcopy(outputs)
    build_features(inputs, outputs)
    for actual, expected in ((inputs, saved_inputs), (outputs, saved_outputs)):
        for key in actual:
            if isinstance(actual[key], torch.Tensor):
                torch.testing.assert_close(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]
