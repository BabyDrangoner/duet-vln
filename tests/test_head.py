import pytest
import torch

from vln_improve.head import ResidualActionHead, load_head_checkpoint, save_head_checkpoint


def test_zero_initialization_mask_bounds_and_frozen_inputs():
    head = ResidualActionHead(3, hidden_dim=8, max_delta=0.6)
    features = torch.randn(2, 3, 3, requires_grad=True)
    base = torch.tensor([[0.1, 1.0, float("nan")], [0.4, -0.2, 0.8]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    result = head(features, base, mask)
    torch.testing.assert_close(result[mask], base.detach()[mask])
    assert torch.isneginf(result[~mask]).all()
    torch.nn.functional.cross_entropy(result, torch.tensor([0, 1])).backward()
    assert features.grad is None and base.grad is None
    assert head.network[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        head.network[-1].weight.fill_(100)
        head.network[-1].bias.fill_(100)
    corrected = head(features, base, mask)
    assert torch.all((corrected[mask] - base.detach()[mask]).abs() <= 0.600001)
    assert torch.isneginf(corrected[~mask]).all()


@pytest.mark.parametrize("bad_kind", ["no_actions", "nan_logits", "infinite_logits", "nan_features"])
def test_bad_valid_inputs_are_rejected(bad_kind):
    head = ResidualActionHead(2)
    features = torch.zeros(1, 2, 2)
    logits = torch.zeros(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    if bad_kind == "no_actions":
        mask[:] = False
    elif bad_kind == "nan_logits":
        logits[0, 0] = float("nan")
    elif bad_kind == "infinite_logits":
        logits[0, 0] = float("inf")
    else:
        features[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        head(features, logits, mask)


def test_padding_with_nonfinite_values_does_not_poison_gradients():
    head = ResidualActionHead(2, hidden_dim=8)
    features = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [float("nan"), float("inf")]]])
    logits = torch.tensor([[0.2, 0.0, float("nan")]])
    mask = torch.tensor([[True, True, False]])
    result = head(features, logits, mask)
    torch.nn.functional.cross_entropy(result, torch.tensor([1])).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_checkpoint_roundtrip_and_provenance_guard(tmp_path):
    head = ResidualActionHead(2, hidden_dim=7, max_delta=0.8)
    with torch.no_grad():
        head.network[-1].bias.fill_(0.5)
    path = tmp_path / "head.pt"
    provenance = {"base_checkpoint_sha256": "a" * 64, "feature_id": "fixture"}
    save_head_checkpoint(head, path, provenance=provenance, train_args={"epochs": 1}, seed=8, metrics={"train_loss": 0.4})
    loaded, metadata = load_head_checkpoint(path, expected_provenance=provenance)
    assert metadata["seed"] == 8 and metadata["head_config"] == head.config
    assert metadata["metrics"] == {"train_loss": 0.4}
    assert not loaded.training
    for name, weight in head.state_dict().items():
        torch.testing.assert_close(weight, loaded.state_dict()[name])
    with pytest.raises(ValueError, match="provenance"):
        load_head_checkpoint(path, expected_provenance={**provenance, "feature_id": "other"})
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["network.0.weight"][0, 0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_head_checkpoint(path)
