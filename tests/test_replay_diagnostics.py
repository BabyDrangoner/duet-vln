import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from replay_diagnostics import (
    chosen_candidates, process_episode, read_result, seal_result, validate_episode,
    validate_runtime,
)
from vln_improve.counterfactual import navigation_hash
from test_diagnostics import collect


@pytest.fixture
def episode(tmp_path):
    agent, observer, store, trajectory, payload = collect(tmp_path)
    observer.close()
    inputs, labels, manifest = payload
    return agent.vln_bert.eval(), inputs, labels


def test_cpu_replay_covers_full_navigation_and_preserves_state(episode):
    model, inputs, labels = episode
    before = [navigation_hash(state["nav_inputs"]) for state in inputs["states"]]
    result = process_episode(model, inputs, labels, split="train_fit", seed=0, device="cpu")
    assert result["counters"]["paired_rows"] == 2
    assert result["counters"]["max_replay_error"] == 0
    assert result["counters"]["selected_without_natural_arrival"] > 0
    assert {row["target_id"] for row in result["rows"]} == {"B", "C"}
    first, second = result["rows"]
    assert "shuffled_arrival" in first["interventions"]
    # B was already observed by step 1, so it is not a future donor for C.
    assert "shuffled_arrival" not in second["interventions"]
    assert result["counters"]["missing_shuffled_control"] == 1
    assert all(row["split"] == "train_fit" and row["future_information_is_offline_only"] for row in result["rows"])
    assert before == [navigation_hash(state["nav_inputs"]) for state in inputs["states"]]


def test_candidate_sampling_precedes_future_label_availability(episode):
    model, inputs, labels = episode
    full = process_episode(model, inputs, labels, split="train_fit", seed=3, device="cpu")
    missing = copy.deepcopy(labels)
    missing["arrival_pairs"] = []
    absent = process_episode(model, inputs, missing, split="train_fit", seed=3, device="cpu")
    assert full["selections"] == absent["selections"]
    assert full["counters"]["selected_candidate_states"] == absent["counters"]["selected_candidate_states"]
    assert absent["rows"] == []
    assert absent["counters"]["selected_without_natural_arrival"] == absent["counters"]["selected_candidate_states"]


def test_future_arrival_changes_interventions_but_not_earlier_observable_features(episode):
    model, inputs, labels = episode
    original = process_episode(model, inputs, labels, split="train_fit", seed=0, device="cpu")
    changed_inputs, changed_labels = copy.deepcopy(inputs), copy.deepcopy(labels)
    # Change a future actual observation consistently in its arrival state and
    # offline label. No preceding navigation input or source evidence changes.
    future = changed_labels["arrival_pairs"][1]["training_only"]["arrival_feature"]
    future.mul_(3.0)
    arrival_state = changed_inputs["states"][2]
    index = arrival_state["nav_inputs"]["gmap_vpids"][0].index("C")
    arrival_state["nav_inputs"]["gmap_img_embeds"][0, index] = future
    changed = process_episode(model, changed_inputs, changed_labels, split="train_fit", seed=0, device="cpu")
    for first, second in zip(original["rows"], changed["rows"]):
        assert first["p0"] == second["p0"]
        assert first["p1"] == second["p1"]
        assert first["navigation_sha256"] == second["navigation_sha256"]
    assert original["rows"][1]["interventions"]["arrival"]["replacement_norm"] != changed["rows"][1]["interventions"]["arrival"]["replacement_norm"]


@pytest.mark.parametrize("damage", ["duplicate_input", "duplicate_label", "missing_label", "wrong_episode",
                                   "wrong_arrival_step", "duplicate_pair", "wrong_arrival_feature",
                                   "future_source", "invalid_optimal_indices", "missing_nav_key", "different_mask"])
def test_invalid_episode_join_or_replay_state_is_rejected(episode, damage):
    model, inputs, labels = episode
    if damage == "duplicate_input":
        inputs["states"].append(copy.deepcopy(inputs["states"][-1]))
    elif damage == "duplicate_label":
        labels["states"].append(copy.deepcopy(labels["states"][-1]))
    elif damage == "missing_label":
        labels["states"].pop()
    elif damage == "wrong_episode":
        labels["arrival_pairs"][-1]["association"]["instr_id"] = "other-instruction"
    elif damage == "wrong_arrival_step":
        labels["arrival_pairs"][-1]["training_only"]["arrival_step"] = 0
    elif damage == "duplicate_pair":
        labels["arrival_pairs"].append(copy.deepcopy(labels["arrival_pairs"][-1]))
    elif damage == "wrong_arrival_feature":
        labels["arrival_pairs"][-1]["training_only"]["arrival_feature"] += 50
    elif damage == "future_source":
        inputs["states"][0]["candidate_evidence"]["B"]["sources"][0]["last_step"] = 5
    elif damage == "invalid_optimal_indices":
        labels["states"][0]["teacher_optimal_indices"] = [1]
    elif damage == "missing_nav_key":
        del inputs["states"][0]["nav_inputs"]["vp_obj_masks"]
    else:
        inputs["states"][0]["valid_mask"][1] = True
    with pytest.raises(ValueError):
        process_episode(model, inputs, labels, split="train_fit", seed=0, device="cpu")


def test_train_dev_analysis_marker_and_training_mode_guard(episode):
    model, inputs, labels = episode
    with pytest.raises(ValueError, match="usage"):
        process_episode(model, inputs, labels, split="train_dev", seed=0, device="cpu")
    labels["usage"] = "analysis_only"
    result = process_episode(model, inputs, labels, split="train_dev", seed=0, device="cpu")
    assert all(row["split"] == "train_dev" for row in result["rows"])
    model.train()
    with pytest.raises(ValueError, match="eval"):
        process_episode(model, inputs, labels, split="train_dev", seed=0, device="cpu")


def test_recorded_logits_must_match_loaded_model(episode):
    model, inputs, labels = episode
    inputs["states"][0]["base_logits"][0] += 100
    with pytest.raises((AssertionError, ValueError)):
        process_episode(model, inputs, labels, split="train_fit", seed=0, device="cpu")


def test_cached_result_checks_content_and_all_identity_dimensions(episode, tmp_path):
    model, inputs, labels = episode
    result = process_episode(model, inputs, labels, split="train_fit", seed=0, device="cpu")
    sealed = seal_result(result, input_manifest_sha256="input-hash", replay_identity_sha256="program-hash")
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(sealed))
    kwargs = {"input_manifest_sha256": "input-hash", "replay_identity_sha256": "program-hash",
              "association": inputs["association"], "split": "train_fit"}
    assert read_result(path, **kwargs) == sealed
    for change in ({"input_manifest_sha256": "other-input"}, {"replay_identity_sha256": "other-program"},
                   {"association": {**inputs["association"], "scan_id": "other-scan"}}, {"split": "train_dev"}):
        with pytest.raises(ValueError):
            read_result(path, **{**kwargs, **change})
    sealed["rows"][0]["interventions"]["arrival"]["delta_margin"] += 0.01
    path.write_text(json.dumps(sealed))
    with pytest.raises(ValueError, match="checksum"):
        read_result(path, **kwargs)


def test_cached_result_without_checksum_is_not_accepted(episode, tmp_path):
    model, inputs, labels = episode
    result = process_episode(model, inputs, labels, split="train_fit", seed=0, device="cpu")
    path = tmp_path / "old-result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="checksum"):
        read_result(path, input_manifest_sha256="input", replay_identity_sha256="code",
                    association=inputs["association"], split="train_fit")


@pytest.mark.parametrize("field", ["model", "upstream_lock", "base_checkpoint_sha256", "torch_version"])
def test_runtime_model_provenance_is_required(field):
    expected = {"model": {"hidden": 24}, "upstream_lock": {"commit": "abc"},
                "base_checkpoint_sha256": "weights", "torch_version": "2.5.1"}
    validate_runtime(expected, copy.deepcopy(expected))
    changed = copy.deepcopy(expected)
    changed[field] = "mismatch"
    with pytest.raises(ValueError, match=field):
        validate_runtime(expected, changed)


def test_actual_tiny_upstream_full_forward_runs_cpu_interventions(episode):
    transformers = pytest.importorskip("transformers")
    upstream = ROOT / "third_party/VLN-DUET/map_nav_src"
    if not upstream.exists():
        pytest.skip("pinned DUET source is unavailable")
    sys.path.insert(0, str(upstream))
    from models.vilmodel import GlocalTextPathNavCMT
    config = transformers.BertConfig(
        vocab_size=100, hidden_size=24, num_attention_heads=4, intermediate_size=48,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        max_position_embeddings=32, type_vocab_size=2, layer_norm_eps=1e-12,
        output_attentions=True, image_feat_size=8, angle_feat_size=4, obj_feat_size=0,
        num_l_layers=1, num_x_layers=1, num_pano_layers=1, update_lang_bert=False,
        use_lang2visn_attn=False, graph_sprels=True, glocal_fuse=True,
        fix_lang_embedding=False, fix_pano_embedding=False, fix_local_branch=False,
        max_action_steps=100,
    )
    torch.manual_seed(5)
    model = GlocalTextPathNavCMT(config).eval().requires_grad_(False)
    _, inputs, labels = episode
    def pad(feature):
        return torch.nn.functional.pad(feature, (0, 22))
    for state in inputs["states"]:
        data = state["nav_inputs"]
        for name in ("txt_embeds", "gmap_img_embeds", "vp_img_embeds"):
            data[name] = pad(data[name])
        for snapshot in state["candidate_evidence"].values():
            for source in snapshot["sources"]:
                source["feature"] = pad(source["feature"])
        with torch.no_grad():
            output = model("navigation", data)
        for target, source in (("base_logits", "fused_logits"), ("base_global_logits", "global_logits"),
                               ("base_local_logits", "local_logits")):
            state[target] = output[source][0].clone()
    for pair in labels["arrival_pairs"]:
        pair["training_only"]["arrival_feature"] = pad(pair["training_only"]["arrival_feature"])
        for source in pair["inference_snapshot"]["sources"]:
            source["feature"] = pad(source["feature"])
    seed = next(value for value in range(100) if 2 in chosen_candidates(inputs["states"][0], value))
    result = process_episode(model, inputs, labels, split="train_fit", seed=seed, device="cpu")
    assert result["rows"]
    assert result["counters"]["max_replay_error"] == 0.0
    assert any(abs(row["interventions"]["arrival"]["stop_logit_shift"]) > 1e-8 for row in result["rows"])
