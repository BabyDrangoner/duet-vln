import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_diagnostics import main, summarize_collection
from test_diagnostics import collect
from vln_improve.diagnostics import DiagnosticStore


def _collection(tmp_path, *, split="train_fit", edits=None, second_scene=False):
    _, observer, _, _, payload = collect(tmp_path / "fixture", split=split)
    observer.close()
    inputs, labels, manifest = payload
    if edits:
        edits(inputs, labels)
    selection = [{"scan": "scan", "instr_id": "instruction", "path_id": "path"}]
    if second_scene:
        selection.append({"scan": "other-scene", "instr_id": "other-instruction", "path_id": "other-path"})
    identity = {"schema": "duet_diagnostic_collection_v1", "split": split,
                "usage": "training_diagnostics" if split == "train_fit" else "analysis_only",
                "selection": selection, "model": {"fixture": True}}
    store = DiagnosticStore(tmp_path / "collection", identity)
    store.commit(inputs, labels, manifest["trajectory"], manifest["coverage"])
    if second_scene:
        other_inputs, other_labels = copy.deepcopy(inputs), copy.deepcopy(labels)
        association = {"scan_id": "other-scene", "instr_id": "other-instruction",
                       "episode_id": store.episode_name("other-scene", "other-instruction")}
        def reassociate(value):
            if isinstance(value, dict):
                if "association" in value:
                    value["association"].update(association)
                for child in value.values():
                    reassociate(child)
            elif isinstance(value, list):
                for child in value:
                    reassociate(child)
        reassociate(other_inputs)
        reassociate(other_labels)
        trajectory = copy.deepcopy(manifest["trajectory"])
        trajectory["instr_id"] = "other-instruction"
        store.commit(other_inputs, other_labels, trajectory, manifest["coverage"])
    return store.local


def test_all_states_candidate_occurrences_and_unique_targets_have_explicit_denominators(tmp_path):
    root = _collection(tmp_path, second_scene=True)
    result = summarize_collection(root, split="train_fit")
    stats = result["overall"]
    assert result["num_scans"] == 2
    assert stats["episodes"] == 2
    assert stats["recorded_states"] == stats["eligible_decisions"] == 6
    assert stats["recorded_states_by_step"] == {"0": 2, "1": 2, "2": 2}
    assert stats["all_recorded_candidates"]["source_count_histogram"] == {"1": 6, "2": 2, "3": 2}
    assert stats["all_recorded_candidates"]["multi_source_fraction"] == pytest.approx(0.4)
    assert stats["all_recorded_candidates"]["natural_arrival_fraction"] == pytest.approx(0.4)
    assert stats["selected_nonstop_candidates"]["count"] == 4
    assert stats["selected_nonstop_candidates"]["multi_source_count"] == 0
    assert stats["selected_stop_count"] == 2
    assert stats["unique_proxy_targets"] == 6
    assert stats["naturally_arrived_targets"] == 4
    assert stats["unknown_arrival_targets"] == 2
    assert stats["observation_counts_per_unique_target"]["observation_count"] == 10
    assert stats["teacher"]["optimal"] == 6
    assert stats["teacher"]["finite_regret_mean"] == 0.0
    assert stats["teacher"]["finite_stop_count"] == 2
    assert all(scan["eligible_decisions"] == 3 for scan in result["per_scan"].values())
    assert len(result["episode_manifest_sha256"]) == 2


def test_wrong_multisource_action_is_not_hidden_by_coverage_or_natural_arrival_filter(tmp_path):
    def edit(inputs, labels):
        state = inputs["states"][1]
        index = state["nav_inputs"]["gmap_vpids"][0].index("X")
        state["base_logits"][index] = 8
        state["baseline_argmax"] = index
        # The discovered graph can make execution cost worse than teacher cost.
        labels["states"][1]["execution_cost"][index] += 2
    result = summarize_collection(_collection(tmp_path, edits=edit), split="train_fit")
    stats = result["overall"]
    assert stats["selected_nonstop_candidates"]["source_count_histogram"] == {"1": 1, "2": 1}
    assert stats["selected_nonstop_candidates"]["naturally_paired_count"] == 1
    assert stats["teacher"]["optimal"] == 2
    assert stats["teacher"]["non_optimal"] == 1
    assert stats["teacher"]["finite_regret_mean"] == pytest.approx(2 / 3)
    assert stats["execution"]["finite_regret_mean"] == pytest.approx(4 / 3)
    table = stats["teacher"]["by_selected_candidate_sources"]
    assert table["multi_source"]["non_optimal"] == 1
    assert table["single_source"]["optimal"] == 1
    assert table["stop"]["optimal"] == 1
    assert stats["teacher"]["selected_multisource_fraction_by_outcome_among_nonstop"]["non_optimal"] == 1.0
    assert stats["teacher"]["multisource_presence_fraction_by_outcome"]["optimal"] == 0.5


def test_nonfinite_stop_is_separate_and_does_not_receive_zero_regret(tmp_path):
    def edit(inputs, labels):
        inputs["states"][0]["base_logits"][0] = 9
        inputs["states"][0]["baseline_argmax"] = 0
    stats = summarize_collection(_collection(tmp_path, edits=edit), split="train_fit")["overall"]
    for kind in ("teacher", "execution"):
        assert stats[kind]["unscorable_stop"] == 1
        assert stats[kind]["finite_regret_count"] == 2
        assert stats[kind]["finite_stop_count"] == 1
        assert stats[kind]["by_selected_candidate_sources"]["stop"]["unscorable_stop"] == 1
    assert stats["selected_stop_count"] == 2


def test_ineligible_states_are_counted_but_not_given_a_policy_decision_outcome(tmp_path):
    def edit(inputs, labels):
        for state in inputs["states"]:
            state["eligible_decision"] = False
    stats = summarize_collection(_collection(tmp_path, edits=edit), split="train_fit")["overall"]
    assert stats["recorded_states"] == stats["ineligible_states"] == 3
    assert stats["all_recorded_candidates"]["count"] == 5
    assert stats["eligible_candidates"]["count"] == stats["eligible_decisions"] == 0
    assert stats["teacher"]["finite_regret_mean"] is None
    assert stats["selected_nonstop_candidates"]["multi_source_fraction"] is None
    assert stats["unique_proxy_targets"] == 3


def test_duplicate_and_overflow_counts_are_summed_once_per_target(tmp_path):
    def edit(inputs, labels):
        # Deliberately nonzero synthetic cumulative counters verify aggregation;
        # the production M0 collector itself rejects revisits/overflow.
        for state in inputs["states"][1:]:
            counts = state["candidate_evidence"]["X"]["counts"]
            counts["duplicate_count"] = 1
            counts["changed_count"] = 1
            counts["source_count_total"] += 1
            counts["observation_count"] = counts["source_count_total"] + 2
            counts["overflow_count"] = 1
            counts["duplicate_event_count"] = 2
    stats = summarize_collection(_collection(tmp_path, edits=edit), split="train_fit")["overall"]
    counts = stats["observation_counts_per_unique_target"]
    assert counts == {"source_count_total": 6, "observation_count": 8, "duplicate_count": 1,
                      "changed_count": 1, "overflow_count": 1, "duplicate_event_count": 2}
    assert stats["targets_with_repeated_source_observations"] == 1
    assert stats["targets_with_overflow"] == 1


@pytest.mark.parametrize("damage", ["checksum", "identity", "split", "missing_episode", "extra_episode", "bad_argmax", "counts", "duplicate_step"])
def test_integrity_or_identity_mismatch_is_rejected(tmp_path, damage):
    def edit(inputs, labels):
        if damage == "bad_argmax":
            inputs["states"][0]["baseline_argmax"] = 0
        elif damage == "counts":
            inputs["states"][0]["candidate_evidence"]["B"]["counts"]["observation_count"] = 5
        elif damage == "duplicate_step":
            labels["states"][1]["step"] = 0
    root = _collection(tmp_path, edits=edit)
    episode = next(root.glob("episode-*"))
    if damage == "checksum":
        (episode / "labels.pt").write_bytes(b"corrupted")
    elif damage == "identity":
        marker = root / "COLLECTION.json"
        content = json.loads(marker.read_bytes())
        content["identity"]["model"] = {"changed": True}
        marker.write_text(json.dumps(content))
    elif damage == "missing_episode":
        shutil.rmtree(episode)
    elif damage == "extra_episode":
        (root / "episode-unexpected").mkdir()
    with pytest.raises(ValueError):
        summarize_collection(root, split="train_dev" if damage == "split" else "train_fit")


def test_cli_preserves_analysis_only_split_and_writes_strict_json(tmp_path):
    root = _collection(tmp_path, split="train_dev")
    output = tmp_path / "reports" / "summary.json"
    main(["--collection", str(root), "--split", "train_dev", "--output", str(output)])
    data = json.loads(output.read_bytes())
    assert data["usage"] == "analysis_only"
    assert data["split"] == "train_dev"
    assert data["overall"]["eligible_decisions"] == 3
    assert "not navigation success" in data["interpretation"]["oracle_outcomes"]
    assert not list(output.parent.glob("*.tmp"))
