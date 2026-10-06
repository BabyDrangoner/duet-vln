import copy

import pytest
import torch

from vln_improve.continuation_learning import ContinuationComparator, record_loss


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
