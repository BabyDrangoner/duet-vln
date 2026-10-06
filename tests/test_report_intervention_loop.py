"""Focused independent-report checks; no real validation output is opened."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("report_intervention_loop", ROOT / "scripts/report_intervention_loop.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def synthetic_reports():
    helper = report.load_helper()
    graph = {"a": {"b": 4.}, "b": {"a": 4., "c": 4.}, "c": {"b": 4.}}
    distance = helper.Distances(graph)
    base_paths = {"1_0": [["a"], ["b"]], "2_0": [["a"], ["b"], ["c"]]}
    method_paths = {"1_0": [["a"], ["b"], ["a"]], "2_0": [["a"], ["b"], ["c"], ["b"]]}
    meta = {key: "same" for key in report.SHARED_METADATA}
    meta.update(mode="baseline", split="val_unseen", subset=False, seed=0, num_episodes=2, model={"fake": True})
    result = {}
    files = {"src/vln_improve/endpoint_intervention.py": "a" * 64}
    for arm in report.NAMES:
        paths = method_paths if arm == "relative" else base_paths
        episodes = [dict(instr_id=instr, scan_id="s" + instr[0],
                         **helper.recompute_metrics(path, ["a", "b"], distance))
                    for instr, path in paths.items()]
        summary = {key: factor * sum(row[field] for row in episodes) / 2
                   for key, (field, factor) in report.SUMMARY_FIELDS.items()}
        result[arm] = {"metadata": copy.deepcopy(meta), "episodes": episodes,
                       "trajectories": [dict(instr_id=i, trajectory=p) for i, p in paths.items()], "summary": summary}
        if arm == "baseline":
            continue
        config = {"arm": arm, "seed": 0, "epochs": 20, "batch_size": 64, "lr": 1e-4,
                  "weight_decay": .01, "monitor_every_epochs": 2, "risk_weight": 0., "hidden_dim": 128,
                  "experiment_sha256": ("b" if arm == "relative" else "c") * 64}
        head = {"schema": "e2_intervention_head_v1", "arm": arm, "config": config,
                "hidden_dim": 128, "epoch": 2, "global_step": 256,
                "selection_rule": "natural_train_dev_SR_and_SPL_at_least_baseline_then_SR_SPL_earliest",
                "initial_head_sha256": "d" * 64, "data_identity": {"fit": "fixed"}, "code_identity": files}
        result[arm]["metadata"].update(
            mode="e2_endpoint_intervention", head_metadata=head, head_sha256="e" * 64,
            experiment_sha256=config["experiment_sha256"], e2_code_files=files, e2_code_sha256=report.object_sha(files),
            access_id="V0006" if arm == "relative" else "V0007", baseline_report_sha256=report.BASELINE_SHA256,
            validation_execution={"sha256": "f" * 64, "ledger_snapshot_sha256": "0" * 64},
            all_online_path_and_termination_parity=True)
        decisions = []
        for instr, prefix in base_paths.items():
            candidates = [segment[-1] for segment in prefix]
            baseline, selected = prefix[-1][-1], paths[instr][-1][-1]
            gains = [[.1, .2] if c == selected and c != baseline else [0., 0.] for c in candidates]
            decisions.append(dict(instr_id=instr, scan_id="s" + instr[0], condition="natural",
                perturbation={"applied": False}, prefix_path=prefix, baseline_trajectory=prefix,
                selected_trajectory=paths[instr], termination_endpoint=baseline,
                candidate_vpids=candidates, baseline_endpoint=baseline, selected_endpoint=selected,
                endpoint_changed=baseline != selected, predicted_gains=gains))
        result[arm]["endpoint_decisions"] = decisions
    return result


def validate(value):
    return report.validate_reports(value, expected_episodes=2, expected_scenes=2)


def test_synthetic_reports_recompute_and_verify_real_return_source_omission():
    _, _, summaries, checks = validate(synthetic_reports())
    assert summaries["relative"]["recomputed_summary"]["sr"] == 50.
    assert all(c["status"] == "passed" for c in checks.values())
    assert all(c["prediction_rule_replayed"] for c in checks.values())


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["relative"]["episodes"].append(copy.deepcopy(r["relative"]["episodes"][0])), "duplicate"),
    (lambda r: r["relative"]["episodes"][0].update(scan_id="other"), "scene differs"),
    (lambda r: r["relative"]["summary"].update(spl=99.), "summary:spl"),
    (lambda r: r["relative"]["metadata"].update(max_action_len=100), "protocol differs"),
    (lambda r: r["absolute"]["metadata"]["head_metadata"]["config"].update(seed=3), "training arm/seed differs"),
    (lambda r: r["relative"]["metadata"]["head_metadata"]["config"].update(experiment_sha256="1" * 64), "not bound"),
    (lambda r: r["relative"]["metadata"]["head_metadata"].update(epoch=0, global_step=0), "trained monitored"),
    (lambda r: r["relative"]["metadata"]["head_metadata"].update(data_identity={"fit": "different"}), "training identity differs"),
    (lambda r: r["relative"]["endpoint_decisions"][0].update(endpoint_changed=False), "change flag differs"),
    (lambda r: r["relative"]["endpoint_decisions"][0].update(candidate_vpids=["a", "b", "frontier"]), "actual decision observations"),
    (lambda r: r["relative"]["endpoint_decisions"][0].update(predicted_gains=[[0., 0.], [0., 0.]]), "selection differs"),
    (lambda r: r["relative"]["endpoint_decisions"][0].update(prefix_path=[["x"]]), "online prefix differs"),
    (lambda r: r["relative"]["endpoint_decisions"][0]["perturbation"].update(applied=True), "must be natural"),
    (lambda r: r["relative"]["episodes"][0].update(nav_error=float("nan")), "invalid distance"),
])
def test_rejects_protocol_metric_and_policy_mismatches(mutation, match):
    value = synthetic_reports()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        validate(value)


def test_missing_endpoint_decisions_explicitly_marks_parity_unavailable():
    value = synthetic_reports()
    del value["relative"]["endpoint_decisions"]
    *_, checks = validate(value)
    assert checks["relative"]["status"] == "not_available"


def test_bootstrap_counts_and_repeatability():
    value = synthetic_reports()
    episodes, paths, *_ = validate(value)
    first = report.paired_comparison(episodes["baseline"], episodes["relative"], paths["baseline"], paths["relative"], replicates=100)
    second = report.paired_comparison(episodes["baseline"], episodes["relative"], paths["baseline"], paths["relative"], replicates=100)
    assert first == second
    assert first["paired_counts"] == {"rescues": 1, "harms": 1, "both_success": 0, "both_failure": 0, "endpoint_changes": 2}
    assert first["metrics"]["sr"]["delta_pp"] == 0.
    assert first["metrics"]["sr"]["ci95_pp"] == [-100., 100.]


def test_bootstrap_uses_whole_scenes_with_episode_weights():
    base, method, paths = {}, {}, {}
    for i, scene in enumerate(("large", "large", "small")):
        instr = str(i)
        gain = 1. if scene == "large" else -1.
        base[instr] = {"scan_id": scene, "success": float(gain < 0), "spl": float(gain < 0),
                       "nDTW": float(gain < 0), "nav_error": 1., "trajectory_lengths": 2.}
        method[instr] = {**base[instr], "success": float(gain > 0), "spl": float(gain > 0), "nDTW": float(gain > 0)}
        paths[instr] = {"trajectory": [["a"]]}
    result = report.paired_comparison(base, method, paths, paths, replicates=1000)
    assert result["scenes"] == 2 and result["episodes"] == 3
    assert result["metrics"]["sr"]["delta_pp"] == pytest.approx(100 / 3)
    assert result["metrics"]["sr"]["ci95_pp"] == [-100., 100.]


def test_tied_predictions_keep_original_endpoint():
    value = synthetic_reports()
    decision = value["absolute"]["endpoint_decisions"][1]
    decision["predicted_gains"] = [[.5, .25], [.5, .25], [0., 0.]]
    validate(value)
    decision["selected_endpoint"] = "a"
    with pytest.raises(ValueError):
        validate(value)


def test_helper_is_hash_pinned(tmp_path):
    path = tmp_path / "different.py"
    path.write_text("raise RuntimeError('should never execute')\n")
    with pytest.raises(ValueError, match="helper SHA differs"):
        report.load_helper(path)


def test_nonfinite_json_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="nonfinite JSON"):
        report.read(path)


def test_full_inventory_is_enforced_by_default():
    with pytest.raises(ValueError, match="episode count differs"):
        report.validate_reports(synthetic_reports())


def toy_graph_assets(tmp_path, monkeypatch, value):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    for scene in ("s1", "s2"):
        rows = []
        for i, node in enumerate(("a", "b", "c")):
            pose = [0.] * 16
            pose[3] = i * 4.
            rows.append({"image_id": node, "included": True, "pose": pose,
                         "unobstructed": [abs(i - j) == 1 for j in range(3)]})
        (graph_dir / (scene + "_connectivity.json")).write_text(json.dumps(rows))
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps([
        {"path_id": i, "scan": "s" + str(i), "path": ["a", "b"], "instructions": ["go"]}
        for i in (1, 2)]))
    monkeypatch.setattr(report, "ANNOTATION_SHA256", report.sha(annotation))
    inventory = {p.name: report.sha(p) for p in sorted(graph_dir.glob("*_connectivity.json"))}
    for row in value.values():
        row["metadata"]["connectivity_sha256"] = report.object_sha(inventory)
    return annotation, graph_dir


def test_independent_graph_recomputes_every_metric(tmp_path, monkeypatch):
    value = synthetic_reports()
    annotation, graph_dir = toy_graph_assets(tmp_path, monkeypatch, value)
    episodes, paths, *_ = validate(value)
    result = report.graph_audit(value, episodes, paths, annotation, graph_dir, report.load_helper())
    assert result["episodes_checked"] == 6
    assert result["metric_values_checked"] == 72
    assert result["actual_graph_transitions_checked"] > 0
    assert all(v == 0 for errors in result["max_absolute_errors"].values() for v in errors.values())


def test_independent_graph_rejects_shortest_distance_teleport(tmp_path, monkeypatch):
    value = synthetic_reports()
    annotation, graph_dir = toy_graph_assets(tmp_path, monkeypatch, value)
    episodes, paths, *_ = validate(value)
    # Shortest-distance metrics alone would accept a->c through b. A physical
    # trajectory must actually list b, so the independent edge check rejects it.
    paths["relative"]["1_0"]["trajectory"] = [["a"], ["c"]]
    with pytest.raises(ValueError, match="nonexecutable trajectory edge"):
        report.graph_audit(value, episodes, paths, annotation, graph_dir, report.load_helper())
