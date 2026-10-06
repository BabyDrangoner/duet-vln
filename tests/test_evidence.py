"""Collection-only tests; no visual distance is used as a reliability label."""

import io

import pytest
import torch

from vln_improve.evidence import COUNT_FIELDS, EVIDENCE_SCHEMA, EpisodeEvidenceMemory


def memory(episode="episode-a", **kwargs):
    return EpisodeEvidenceMemory(episode, scan_id="scan-a", instr_id="instruction-a", **kwargs)


def observe(value, source="source-a", *, target="target", step=0, feature=None, **kwargs):
    return value.observe_proxy(target, source, step=step,
                               heading=kwargs.pop("heading", 0.5),
                               elevation=kwargs.pop("elevation", -0.1),
                               relative_position=kwargs.pop("relative_position", [1.0, 2.0, 0.0]),
                               feature=torch.tensor([1.0, 2.0]) if feature is None else feature,
                               **kwargs)


def equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            equal(left, right)
    else:
        assert first == second


def test_episode_instances_isolate_identical_node_ids():
    first, second = memory(), memory("episode-b")
    observe(first)
    assert second.pending_targets() == []
    assert second.drain_pairs() == []
    assert second.model_inputs("target")["features"].shape == (0, 0)
    observe(second, feature=torch.tensor([10.0, 20.0]))
    pair = first.arrive("target", step=1, feature=torch.tensor([3.0, 4.0]))
    assert pair["association"]["episode_id"] == "episode-a"
    assert second.policy_snapshot("target")["status"] == "unvisited"
    assert second.summary()["num_pairs"] == 0


def test_exact_repeated_source_counts_as_one_source_and_preserves_feature_age():
    value = memory()
    assert observe(value) == "added"
    assert observe(value) == "duplicate_event"
    assert observe(value, step=2) == "repeated"
    snapshot = value.policy_snapshot("target")
    assert snapshot["counts"] == {
        "source_count_total": 1, "stored_count": 1, "observation_count": 2,
        "duplicate_count": 1, "changed_count": 0, "overflow_count": 0, "duplicate_event_count": 1,
    }
    assert snapshot["sources"][0]["first_step"] == 0
    assert snapshot["sources"][0]["feature_step"] == 0
    assert snapshot["sources"][0]["last_step"] == 2
    assert snapshot["sources"][0]["observation_count"] == 2


def test_changed_source_updates_existing_slot_without_increasing_source_count():
    value = memory()
    observe(value)
    assert observe(value, step=1, feature=torch.tensor([5.0, 6.0])) == "updated"
    source = value.policy_snapshot("target")["sources"][0]
    assert source["first_step"] == 0 and source["feature_step"] == 1 and source["last_step"] == 1
    assert source["changed_count"] == 1 and source["duplicate_count"] == 0
    assert torch.equal(source["feature"], torch.tensor([5.0, 6.0]))
    assert value.summary()["targets"]["target"]["source_count_total"] == 1


def test_geometry_change_is_recorded_even_when_embedding_stays_identical():
    value = memory()
    observe(value)
    assert observe(value, step=1, heading=0.7) == "updated"
    assert value.policy_snapshot("target")["sources"][0]["heading"] == 0.7


def test_conflicting_same_source_same_step_is_rejected_without_mutation():
    value = memory()
    observe(value)
    before = value.policy_snapshot("target")
    with pytest.raises(ValueError, match="conflicting"):
        observe(value, feature=torch.tensor([9.0, 2.0]))
    equal(value.policy_snapshot("target"), before)


def test_capacity_limits_tensors_without_truncating_unique_source_coverage():
    value = memory(max_sources=2)
    for index in range(5):
        status = observe(value, source=f"s{index}")
        assert status == ("added" if index < 2 else "overflow")
    assert observe(value, source="s4", step=1) == "overflow"
    assert observe(value, source="s4", step=2, heading=0.8) == "overflow"
    snapshot = value.policy_snapshot("target")
    assert [source["source_id"] for source in snapshot["sources"]] == ["s0", "s1"]
    assert snapshot["counts"] == {
        "source_count_total": 5, "stored_count": 2, "observation_count": 7,
        "duplicate_count": 1, "changed_count": 1, "overflow_count": 5, "duplicate_event_count": 0,
    }
    assert value.model_inputs("target")["counts"].tolist() == [snapshot["counts"][key] for key in COUNT_FIELDS]
    assert value.model_inputs("target")["features"].shape == (2, 2)
    pair = value.arrive("target", step=3, feature=torch.tensor([9.0, 8.0]))
    assert pair["inference_snapshot"]["counts"]["source_count_total"] == 5
    assert len(pair["inference_snapshot"]["sources"]) == 2


def test_arrival_future_is_separate_and_not_accessible_before_arrival():
    value = memory()
    observe(value)
    inputs = value.model_inputs("target")
    before = value.policy_snapshot("target")
    assert value.drain_pairs() == []
    assert all(isinstance(item, torch.Tensor) for item in inputs.values())
    assert set(inputs) == {"features", "angles", "relative_positions", "steps", "counts"}
    assert "training_only" not in before and "arrival_feature" not in before
    arrival = torch.tensor([1000.0, -3000.0], requires_grad=True)
    pair = value.arrive("target", step=1, feature=arrival)
    assert pair["kind"] == "arrival_observation_pair"
    assert pair["feature_schema"] == EVIDENCE_SCHEMA
    assert pair["training_only"]["reliability_label"] is None
    assert pair["training_only"]["arrival_observed"] is True
    equal(pair["inference_snapshot"], before)
    assert value.policy_snapshot("target")["status"] == "visited"
    assert value.model_inputs("target")["features"].shape == (0, 2)
    assert torch.equal(inputs["features"], torch.tensor([[1.0, 2.0]]))


def test_replacing_future_observation_never_changes_prearrival_inputs():
    first, second = memory(), memory()
    for value in (first, second):
        observe(value)
        observe(value, source="source-b", step=1, feature=torch.tensor([-5.0, 8.0]))
    first_before, second_before = first.model_inputs("target"), second.model_inputs("target")
    equal(first_before, second_before)
    first_pair = first.arrive("target", step=2, feature=torch.tensor([1.0, 1.0]))
    second_pair = second.arrive("target", step=2, feature=torch.tensor([-100.0, 100.0]))
    equal(first_pair["inference_snapshot"], second_pair["inference_snapshot"])
    equal(first_before, second_before)
    assert not torch.equal(first_pair["training_only"]["arrival_feature"], second_pair["training_only"]["arrival_feature"])


def test_unarrived_target_has_unknown_outcome_and_no_fabricated_negative():
    value = memory()
    observe(value, target="never-reached")
    assert value.drain_pairs() == []
    assert value.summary()["targets"]["never-reached"]["arrival_observed"] is False
    assert value.summary()["num_pending_targets"] == 1
    assert value.arrive("other-node", step=1, feature=torch.tensor([2.0, 3.0])) is None
    assert value.summary()["arrival_without_proxy"] == 1
    assert value.drain_pairs() == []


def test_repeat_arrival_does_not_overwrite_first_pair_or_reopen_proxy_memory():
    value = memory()
    observe(value)
    first = value.arrive("target", step=1, feature=torch.tensor([3.0, 4.0]))
    assert value.arrive("target", step=2, feature=torch.tensor([-99.0, -99.0])) is None
    assert observe(value, step=2) == "ignored_visited"
    pairs = value.drain_pairs()
    assert len(pairs) == 1
    equal(pairs[0], first)
    assert value.drain_pairs() == []
    assert value.summary()["num_pairs"] == 1
    assert value.summary()["repeat_arrivals"] == 1
    assert value.summary()["ignored_visited_proxies"] == 1


def test_input_and_snapshot_mutations_do_not_affect_memory_or_pair_queue():
    value = memory()
    supplied_feature = torch.tensor([1.0, 2.0], requires_grad=True)
    supplied_position = [1.0, 2.0, 0.0]
    observe(value, feature=supplied_feature, relative_position=supplied_position)
    with torch.no_grad():
        supplied_feature.fill_(123)
    supplied_position[0] = 123
    first = value.policy_snapshot("target")
    assert first["sources"][0]["feature"].tolist() == [1.0, 2.0]
    assert first["sources"][0]["relative_position"][0] == 1
    first["sources"][0]["feature"].fill_(456)
    first["sources"][0]["source_id"] = "mutated-id"
    first["sources"][0]["relative_position"][0] = 456
    first["counts"]["source_count_total"] = 456
    model_inputs = value.model_inputs("target")
    model_inputs["features"].fill_(789)
    assert value.policy_snapshot("target")["sources"][0]["source_id"] == "source-a"
    assert value.policy_snapshot("target")["sources"][0]["feature"].tolist() == [1.0, 2.0]
    supplied_arrival = torch.tensor([3.0, 4.0], requires_grad=True)
    returned = value.arrive("target", step=1, feature=supplied_arrival)
    with torch.no_grad():
        supplied_arrival.fill_(999)
    returned["training_only"]["arrival_feature"].fill_(999)
    returned["inference_snapshot"]["sources"][0]["feature"].fill_(999)
    queued = value.drain_pairs()[0]
    assert queued["training_only"]["arrival_feature"].tolist() == [3.0, 4.0]
    assert queued["inference_snapshot"]["sources"][0]["feature"].tolist() == [1.0, 2.0]
    assert not queued["training_only"]["arrival_feature"].requires_grad
    assert queued["training_only"]["arrival_feature"].grad_fn is None
    assert queued["training_only"]["arrival_feature"].device.type == "cpu"
    summary = value.summary()
    summary["targets"]["target"]["source_count_total"] = 99
    assert value.summary()["targets"]["target"]["source_count_total"] == 1


def test_plain_pairs_roundtrip_with_restricted_cpu_torch_loader():
    value = memory()
    observe(value, feature=torch.tensor([1.0, 2.0], dtype=torch.float16, requires_grad=True))
    value.arrive("target", step=1, feature=torch.tensor([3.0, 4.0]))
    pair = value.drain_pairs()
    payload = io.BytesIO()
    torch.save(pair, payload)
    payload.seek(0)
    restored = torch.load(payload, map_location="cpu", weights_only=True)
    equal(pair, restored)


@pytest.mark.parametrize("mutation", ["reverse", "same_step_arrival", "wrong_dimension", "nan_feature", "nan_geometry", "self_proxy"])
def test_invalid_inputs_are_rejected(mutation):
    value = memory()
    observe(value, step=1)
    with pytest.raises(ValueError):
        if mutation == "reverse":
            observe(value, step=0)
        elif mutation == "same_step_arrival":
            value.arrive("target", step=1, feature=torch.tensor([3.0, 4.0]))
        elif mutation == "wrong_dimension":
            observe(value, step=2, feature=torch.tensor([1.0, 2.0, 3.0]))
        elif mutation == "nan_feature":
            observe(value, step=2, feature=torch.tensor([float("nan"), 2.0]))
        elif mutation == "nan_geometry":
            observe(value, step=2, relative_position=[0.0, float("inf"), 0.0])
        elif mutation == "self_proxy":
            observe(value, source="target", step=2)
    assert value.summary()["num_pairs"] == 0
    assert value.summary()["targets"]["target"]["observation_count"] == 1


@pytest.mark.parametrize("kwargs", [{"max_sources": 0}, {"max_sources": True}, {"feature_dim": 0}])
def test_invalid_capacity_and_dimensions_rejected(kwargs):
    with pytest.raises(ValueError):
        memory(**kwargs)


def test_source_ids_only_associate_and_do_not_change_numeric_inputs():
    first, second = memory(), memory("different-episode")
    observe(first, source="s1", target="t1")
    observe(second, source="renamed-source", target="renamed-target")
    equal(first.model_inputs("t1"), second.model_inputs("renamed-target"))


def test_geometry_accepts_simulator_numpy_scalars_and_detached_tensor_coordinates():
    import numpy as np
    value = memory()
    position = torch.tensor([1.0, 2.0, 0.0], requires_grad=True)
    observe(value, heading=np.float32(0.5), relative_position=position)
    with torch.no_grad():
        position.fill_(999)
    assert value.policy_snapshot("target")["sources"][0]["relative_position"] == [1.0, 2.0, 0.0]
