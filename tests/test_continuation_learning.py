import copy

import pytest
import torch

from vln_improve.continuation_learning import ContinuationComparator, record_loss
from vln_improve import continuation_learning as learning


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def record(feature_dim=7):
    generator = torch.Generator().manual_seed(19)
    return {"features": torch.randn(3, feature_dim, generator=generator),
            "history_features": torch.randn(5, feature_dim, generator=generator),
            "progress": torch.tensor([3 / 14, 12 / 15, .12, 3 / 4]),
            "utilities": torch.tensor([[0., 0.], [1., .8], [0., 0.]]),
            "teacher_target": 1, "instr_id": "fit-1", "scan_id": "fit-house",
            "condition": "natural", "step": 3}


def nonzero_output(model):
    with torch.no_grad():
        generator = torch.Generator().manual_seed(13)
        model.comparison[-1].weight.copy_(
            torch.randn(model.comparison[-1].weight.shape, generator=generator) * .2)


def test_same_capacity_initialization_and_keep_across_modes():
    models = []
    for mode in ("relative", "absolute", "teacher"):
        torch.manual_seed(4)
        models.append(ContinuationComparator(mode=mode))
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert counts == [364162] * 3
    for model in models[1:]:
        for name, value in models[0].state_dict().items():
            assert torch.equal(value, model.state_dict()[name])
    data = record(1549)
    for model in models:
        assert torch.equal(model.score_record(data), torch.zeros(3, 2))


@pytest.mark.parametrize("mode", ("relative", "absolute", "teacher"))
def test_prediction_reads_only_three_inputs_and_never_labels(mode):
    model = ContinuationComparator(7, 8, mode=mode)
    nonzero_output(model)
    data = record()
    original = model.score_record(data)
    altered = dict(data, utilities=torch.full((3, 2), float("nan")), teacher_target=999,
                   instr_id="unknown", scan_id="other", condition="unseen", step=100)
    assert torch.equal(original, model.score_record(altered))

    class InputsOnly(dict):
        def __getitem__(self, key):
            assert key in {"features", "history_features", "progress"}
            return super().__getitem__(key)

    assert torch.equal(original, model.score_record(InputsOnly(data)))


def test_relative_anchor_zero_and_absolute_deployment_subtraction():
    relative = ContinuationComparator(7, 8)
    nonzero_output(relative)
    absolute = ContinuationComparator(7, 8, mode="absolute")
    absolute.load_state_dict(relative.state_dict())
    data = record()
    gains = relative.score_record(data)
    raw = absolute.score_record(data)
    assert torch.equal(gains[0], torch.zeros(2))
    assert torch.equal(gains, raw - raw[0:1])
    assert torch.equal(absolute.relative_scores(data["features"], data["history_features"],
                                              data["progress"]), gains)
    teacher = ContinuationComparator(7, 8, mode="teacher")
    with pytest.raises(ValueError, match="not predicted"):
        teacher.relative_scores(data["features"], data["history_features"], data["progress"])


def test_history_ablation_reads_only_current_stop_context_with_matched_capacity():
    full = ContinuationComparator(7, 8)
    nonzero_output(full)
    ablated = ContinuationComparator(7, 8, history=False)
    ablated.load_state_dict(full.state_dict())
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in ablated.parameters())
    data = record()
    changed = copy.deepcopy(data)
    changed["history_features"][:-1] = torch.randn_like(changed["history_features"][:-1]) * 10
    assert torch.equal(ablated.score_record(data), ablated.score_record(changed))
    assert not torch.equal(full.score_record(data), full.score_record(changed))
    last_only = dict(data, history_features=data["history_features"][-1:])
    assert torch.equal(ablated.score_record(data), full.score_record(last_only))


@pytest.mark.parametrize("mode", ("relative", "absolute", "teacher"))
def test_actual_optimization_reduces_loss_and_inputs_remain_frozen(mode):
    torch.manual_seed(7)
    model = ContinuationComparator(7, 12, mode=mode)
    data = record()
    for key in ("features", "history_features", "progress", "utilities"):
        data[key].requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    before = float(record_loss(model, data).detach())
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True)
        loss = record_loss(model, data)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    after = float(record_loss(model, data).detach())
    assert after < before * .1
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.history_encoder.weight_ih_l0.grad.abs().sum() > 0
    assert all(data[key].grad is None for key in ("features", "history_features", "progress", "utilities"))


def test_teacher_missing_expert_label_is_differentiable_zero():
    model = ContinuationComparator(7, 8, mode="teacher")
    data = record()
    data["teacher_target"] = -1
    del data["utilities"]
    loss = record_loss(model, data)
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_fraction_targets_and_explicit_candidate_weight_normalization():
    model = ContinuationComparator(7, 8)
    data = record()
    # At zero prediction, losses for the sole rescue are .5 (SR), .32 (SPL).
    assert record_loss(model, data).item() == pytest.approx(.82 / 6)
    weighted = record_loss(model, data, rescue_weight=3., sr_weight=2.)
    assert weighted.item() == pytest.approx((.5 * 2 + .32) * 3 / (5 * 3))
    data["utilities"] = torch.tensor([[1., .8], [0., 0.], [1., .8]])
    assert record_loss(model, data, harm_weight=3.).item() == pytest.approx(.82 * 3 / 10)


@pytest.mark.parametrize("kwargs", ({"mode": "oracle"}, {"history": 1},
                                    {"feature_dim": 0}, {"hidden_dim": True}))
def test_invalid_model_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ContinuationComparator(**kwargs)


@pytest.mark.parametrize("key,value", (
    ("features", torch.zeros(5, 7)), ("features", torch.zeros(0, 7)),
    ("features", torch.zeros(3, 8)), ("features", torch.zeros(3, 7, dtype=torch.long)),
    ("features", torch.full((3, 7), float("nan"))),
    ("history_features", torch.zeros(0, 7)), ("history_features", torch.zeros(30, 7)),
    ("history_features", torch.full((2, 7), float("inf"))),
    ("progress", torch.zeros(1, 4)), ("progress", torch.tensor([0., 1., 2., float("nan")]))))
def test_invalid_policy_inputs_rejected(key, value):
    data = record()
    data[key] = value
    with pytest.raises(ValueError):
        ContinuationComparator(7, 8).score_record(data)


@pytest.mark.parametrize("utilities", (torch.zeros(2, 2), torch.full((3, 2), float("nan")),
    torch.tensor([[0., 0.], [.5, .3], [0., 0.]]),
    torch.tensor([[0., 0.], [1., 80.], [0., 0.]]),
    torch.tensor([[0., .1], [1., .8], [0., 0.]])))
def test_invalid_outcome_units_rejected(utilities):
    with pytest.raises(ValueError, match="utilities"):
        record_loss(ContinuationComparator(7, 8), dict(record(), utilities=utilities))


@pytest.mark.parametrize("target", (True, -2, 3, .5))
def test_invalid_teacher_index_rejected(target):
    with pytest.raises(ValueError, match="teacher_target"):
        record_loss(ContinuationComparator(7, 8, mode="teacher"), dict(record(), teacher_target=target))


@pytest.mark.parametrize("weight", (0, -1, float("nan"), float("inf"), True))
def test_invalid_loss_weights_rejected(weight):
    with pytest.raises(ValueError, match="rescue_weight"):
        record_loss(ContinuationComparator(7, 8), record(), rescue_weight=weight)


def heterogeneous_records(dtype=torch.float32):
    """Exercise all candidate counts and unsorted, extreme prefix lengths."""
    examples = []
    outcomes = [((1., .7),), ((0., 0.), (1., .8)),
                ((1., .6), (0., 0.), (1., .9)),
                ((0., 0.), (0., 0.), (1., .4), (1., .9))]
    for index, (candidates, tokens, target) in enumerate(((1, 1, 0), (2, 7, -1), (3, 3, 2), (4, 29, 3))):
        generator = torch.Generator().manual_seed(105 + index)
        examples.append({"features": torch.randn(candidates, 7, generator=generator).to(dtype),
                         "history_features": torch.randn(tokens, 7, generator=generator).to(dtype),
                         "progress": torch.tensor([index / 14, .5, .12, candidates / 4], dtype=dtype),
                         "utilities": torch.tensor(outcomes[index], dtype=dtype),
                         "teacher_target": target})
    return examples


def padded_inputs(records, *, candidates=None, tokens=None, fill=float("nan")):
    """Build the public batch interface independently of score_records."""
    candidates = candidates or max(len(row["features"]) for row in records)
    tokens = tokens or max(len(row["history_features"]) for row in records)
    count, width = len(records), records[0]["features"].shape[1]
    features = torch.full((count, candidates, width), fill)
    history = torch.full((count, tokens, width), fill)
    candidate_mask = torch.zeros(count, candidates, dtype=torch.bool)
    history_mask = torch.zeros(count, tokens, dtype=torch.bool)
    for index, row in enumerate(records):
        k, t = len(row["features"]), len(row["history_features"])
        features[index, :k] = row["features"]
        history[index, :t] = row["history_features"]
        candidate_mask[index, :k] = True
        history_mask[index, :t] = True
    return dict(features=features, history_features=history,
                progress=torch.stack([row["progress"] for row in records]),
                candidate_mask=candidate_mask, history_mask=history_mask)


@pytest.mark.parametrize("mode", ("relative", "absolute", "teacher"))
@pytest.mark.parametrize("history", (True, False))
@pytest.mark.parametrize("device,cudnn_tf32", (("cpu", False), *[
    pytest.param("cuda", enabled, marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA acceptance runs on the WSL GPU"))
    for enabled in (False, True)]))
def test_batch_scores_losses_and_gradients_match_separate_records(mode, history, device, cudnn_tf32, monkeypatch):
    # Check the exact FP32 equations tightly, and separately exercise the
    # production cuDNN TF32 default. Packed and dense GRUs choose different
    # kernels; their reduced-precision accumulation has a measured wider bound.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", cudnn_tf32)
    score_tolerance = dict(rtol=2e-4, atol=2e-5) if cudnn_tf32 else dict(rtol=2e-5, atol=2e-6)
    gradient_tolerance = dict(rtol=2e-4, atol=1e-4) if cudnn_tf32 else dict(rtol=1e-4, atol=5e-6)
    torch.manual_seed(61)
    batch_model = ContinuationComparator(7, 12, mode=mode, history=history).to(device)
    nonzero_output(batch_model)
    single_model = copy.deepcopy(batch_model)
    records = heterogeneous_records(dtype=torch.float16)
    for row in records:
        for name in ("features", "history_features", "progress", "utilities"):
            row[name].requires_grad_(True)
    scores, mask = batch_model.score_records(records)
    assert scores.shape == (4, 4, 2) and mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    assert scores.device == mask.device == next(batch_model.parameters()).device
    assert scores.dtype == next(batch_model.parameters()).dtype
    assert torch.count_nonzero(scores[~mask]) == 0
    for index, row in enumerate(records):
        candidates = len(row["features"])
        assert mask[index].tolist() == [True] * candidates + [False] * (4 - candidates)
        torch.testing.assert_close(scores[index, :candidates], single_model.score_record(row),
                                   **score_tolerance)
    weights = dict(sr_weight=2., spl_weight=.75, rescue_weight=3., harm_weight=4.)
    actual = learning.batch_record_losses(batch_model, records, **weights)
    expected = torch.stack([record_loss(single_model, row, **weights) for row in records])
    assert actual.shape == (len(records),)
    torch.testing.assert_close(actual, expected, **score_tolerance)
    record_weights = actual.new_tensor([.5, 2., 1.25, 3.])
    (actual * record_weights).sum().backward()
    (expected * record_weights).sum().backward()
    for name, actual_parameter in batch_model.named_parameters():
        expected_parameter = dict(single_model.named_parameters())[name]
        assert actual_parameter.grad is not None and expected_parameter.grad is not None, name
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad,
                                   **gradient_tolerance, msg=lambda message: name + ": " + message)
    assert all(row[name].grad is None for row in records
               for name in ("features", "history_features", "progress", "utilities"))


@pytest.mark.parametrize("mode", ("relative", "absolute", "teacher"))
def test_batch_prediction_collation_reads_only_policy_inputs(mode):
    model = ContinuationComparator(7, 8, mode=mode)
    nonzero_output(model)
    records = heterogeneous_records()

    class InputsOnly(dict):
        def __getitem__(self, key):
            assert key in {"features", "history_features", "progress"}, key
            return super().__getitem__(key)

        def get(self, key, default=None):
            assert key in {"features", "history_features", "progress"}, key
            return super().get(key, default)

    expected, expected_mask = model.score_records(records)
    poisoned = [InputsOnly(dict(row, utilities=torch.full_like(row["utilities"], float("nan")),
                               teacher_target=999, goal="unavailable", future="forbidden"))
                for row in records]
    actual, mask = model.score_records(poisoned)
    assert torch.equal(actual, expected)
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize("history", (True, False))
def test_nan_padding_and_extra_future_padding_do_not_change_valid_scores(history):
    model = ContinuationComparator(7, 8, history=history)
    nonzero_output(model)
    rows = heterogeneous_records()[:3]
    compact = padded_inputs(rows)
    expanded = padded_inputs(rows, candidates=4, tokens=29)
    original = model.forward_batch(**compact)
    padded = model.forward_batch(**expanded)
    assert torch.isfinite(padded).all()
    assert torch.count_nonzero(padded[~expanded["candidate_mask"]]) == 0
    for index, row in enumerate(rows):
        count = len(row["features"])
        torch.testing.assert_close(padded[index, :count], original[index, :count], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(padded[index, :count], model.score_record(row), rtol=1e-5, atol=1e-6)
    padded.sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    # Alter another item's valid context: samples must never attend to each other.
    expanded["features"][1, :2] *= -100
    expanded["history_features"][1, :7] += 100
    changed = model.forward_batch(**expanded)
    torch.testing.assert_close(changed[0], padded[0], rtol=0, atol=0)
    torch.testing.assert_close(changed[2], padded[2], rtol=0, atol=0)


def test_batch_history_ablation_uses_last_valid_token_and_ignores_earlier_history():
    model = ContinuationComparator(7, 8, history=False)
    nonzero_output(model)
    rows = heterogeneous_records()
    before, mask = model.score_records(rows)
    for row in rows:
        row["history_features"][:-1] = 1000.
    after, after_mask = model.score_records(rows)
    assert torch.equal(before, after)
    assert torch.equal(mask, after_mask)
    for index, row in enumerate(rows):
        only_current = dict(row, history_features=row["history_features"][-1:])
        torch.testing.assert_close(after[index, :len(row["features"])], model.score_record(only_current),
                                   rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("which,error", (
    ("candidate_mask", "nonprefix"), ("candidate_mask", "empty"),
    ("candidate_mask", "dtype"), ("candidate_mask", "shape"),
    ("history_mask", "nonprefix"), ("history_mask", "empty"),
    ("history_mask", "dtype"), ("history_mask", "shape")))
def test_batch_rejects_invalid_or_nonprefix_masks(which, error):
    inputs = padded_inputs(heterogeneous_records())
    if error == "nonprefix":
        inputs[which][0] = False
        inputs[which][0, 0] = inputs[which][0, 2] = True
    elif error == "empty":
        inputs[which][0] = False
    elif error == "dtype":
        inputs[which] = inputs[which].float()
    else:
        inputs[which] = inputs[which][:, :-1]
    with pytest.raises(ValueError):
        ContinuationComparator(7, 8).forward_batch(**inputs)


@pytest.mark.parametrize("field", ("features", "history_features", "progress"))
def test_batch_rejects_nonfinite_values_inside_valid_inputs(field):
    inputs = padded_inputs(heterogeneous_records())
    inputs[field].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        ContinuationComparator(7, 8).forward_batch(**inputs)


@pytest.mark.parametrize("boundary", ("empty_batch", "too_many_candidates", "too_much_history"))
def test_batch_rejects_dimensions_outside_fixed_interface_limits(boundary):
    if boundary == "too_many_candidates":
        inputs = padded_inputs(heterogeneous_records(), candidates=5)
    elif boundary == "too_much_history":
        inputs = padded_inputs(heterogeneous_records(), tokens=30)
    else:
        inputs = {name: value[:0] for name, value in padded_inputs(heterogeneous_records()).items()}
    with pytest.raises(ValueError):
        ContinuationComparator(7, 8).forward_batch(**inputs)


def test_teacher_batch_with_no_expert_labels_has_differentiable_zero_loss():
    model = ContinuationComparator(7, 8, mode="teacher")
    nonzero_output(model)
    rows = heterogeneous_records()
    for row in rows:
        row["teacher_target"] = -1
        del row["utilities"]
    losses = learning.batch_record_losses(model, rows)
    assert torch.equal(losses, torch.zeros(len(rows)))
    assert losses.requires_grad
    losses.sum().backward()
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for parameter in model.parameters())
