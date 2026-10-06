import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_endpoint_controls import (
    analyze_split, history_from_walk, prepare_control, reconstruct_path, run, validate_spec,
)
from analyze_endpoint_pairs import shortest_distances, write_new_json
from vln_improve.protocol import file_sha256, object_sha256, select_partition


def graph(edges):
    result = {}
    for a, b, length in edges:
        result.setdefault(a, {})[b] = float(length)
        result.setdefault(b, {})[a] = float(length)
    return result


def record(path, *, scan="scene", pid="1"):
    return {"scan": scan, "path_id": pid, "path": path, "heading": 0., "num_instructions": 1}


def test_inclusive_annulus_excludes_original_path_and_uses_fixed_hash():
    g = graph([("s", "g", 4), ("g", "q3", 3), ("g", "q6", 6),
               ("g", "too_close", 2.999), ("g", "too_far", 6.001)])
    row = record(["s", "g"])
    result = prepare_control(row, g)
    assert result["eligible"]
    assert result["candidate_counts"]["distance_annulus"] == 3
    assert result["candidate_counts"]["annulus_on_original_path"] == 1
    assert result["candidate_counts"]["feasible_q"] == 2
    expected = min(["q3", "q6"], key=lambda q: object_sha256([0, "scene", "1", q]))
    assert result["selected_q"] == expected
    assert result["selected_q_goal_distance_m"] in (3, 6)
    assert result["endpoint_labels"] == {"positive_history": True, "overshoot_history": False}
    assert all(a["q"] not in row["path"] for a in result["candidate_audit"])


def test_positive_first_visit_must_end_at_goal_and_budget_is_not_relaxed():
    g = graph([("s", "g", 1), ("g", "x", 1)])
    result = prepare_control(record(["s", "g", "x", "g"]), g)
    assert result["rejection"] == "positive_first_visit_does_not_end_at_goal"
    assert not result["candidate_analysis_performed"]
    nodes = [f"p{i}" for i in range(15)]
    g = graph([(a, b, 1) for a, b in zip(nodes, nodes[1:])] + [(nodes[-1], "q", 3)])
    result = prepare_control(record(nodes), g)
    assert result["positive_history"]["observed_states"] == 15
    assert result["rejection"] == "no_feasible_q"
    assert result["candidate_counts"]["rejected_observation_budget"] == 1
    assert result["candidate_audit"][0]["observed_states"] == 16


def test_no_q_is_a_reported_failure_but_invalid_edges_are_errors():
    g = graph([("s", "g", 1), ("g", "near", 2)])
    result = prepare_control(record(["s", "g"]), g)
    assert not result["eligible"] and result["rejection"] == "no_feasible_q"
    with pytest.raises(ValueError, match="non-adjacent"):
        prepare_control(record(["s", "near"]), g)


def test_dijkstra_ties_are_fixed_by_sorted_neighbors_and_retained_parent():
    edges = [("g", "y", 1), ("g", "x", 1), ("y", "q", 2), ("x", "q", 2), ("s", "g", 1)]
    one, two = graph(edges), graph(list(reversed(edges)))
    for item in (one, two):
        distances, parents = shortest_distances(item, "g")
        assert distances["q"] == 3
        assert reconstruct_path(parents, "g", "q") == ["g", "x", "q"]
    assert prepare_control(record(["s", "g"]), one) == prepare_control(record(["s", "g"]), two)


def test_revisited_transit_nodes_do_not_create_observations():
    g = graph([("s", "a", 2), ("a", "g", 2), ("s", "q", 2)])
    history = history_from_walk(g, ["s", "a", "g", "a", "s", "q"], "g")
    assert history["observed_vpids"] == ["s", "a", "g", "q"]
    assert history["known_map_shortest_proxy_walk"] == history["reference_walk"]
    assert history["reference_walk_length_m"] == 10


def pair_for(scan, pid_a, pid_b):
    pair = {"scan": scan, "start": "s", "path_ids": [pid_a, pid_b],
            "instr_ids": [pid_a + "_0", pid_b + "_0"], "goal_vpids": ["ga", "gb"],
            "heading_rad": [0., 0.], "heading_difference_deg": 0., "goal_separation_m": 8.}
    pair["selection_hash"] = object_sha256([20261003, scan, pid_a, pid_b])
    return pair


def test_pair_common_pool_requires_both_paths_and_does_not_shrink_target():
    g = graph([("s", "a", 2), ("a", "ga", 2), ("s", "b", 2), ("b", "gb", 2)])
    records = [record(["s", "a", "ga"], pid="1"), record(["s", "b", "gb"], pid="2")]
    pair = pair_for("scene", "1", "2")
    result = analyze_split([pair], records, {"scene": g}, requested_pairs=512,
                           selection_seed=20261003, primary_sha=object_sha256([pair]))
    assert result["eligible_common_pool"]["pairs"] == 1
    assert result["selected_common_pool"]["pairs"] == 1
    assert not result["coverage_pass"] and result["shortfall_pairs"] == 511
    manifest = result["manifest"]
    assert manifest["requested_pairs"] == 512 and len(manifest["controls"]) == 2
    for entry in manifest["selected_pairs"]:
        assert entry["pair"] == pair
        assert all(manifest["controls"][key]["eligible"] for key in entry["control_ids"])
    assert result["manifest_sha256"] == object_sha256(manifest)


def test_control_generation_cannot_consult_other_pair_goal():
    g = graph([("s", "g", 1), ("g", "q", 3), ("g", "r", 4)])
    one = record(["s", "g"])
    two = dict(one, other_goal="q", base_failure=True)
    three = dict(one, other_goal="r", base_failure=False)
    assert prepare_control(one, g) == prepare_control(two, g) == prepare_control(three, g)
    with pytest.raises(ValueError, match="fixed"):
        prepare_control(one, g, seed=1)


def test_fixed_spec_refuses_relaxed_distance_or_smaller_pool():
    spec = json.loads((ROOT / "configs/endpoint_controls_diagnostic.json").read_text())
    cfg = json.loads((ROOT / "configs/r2r.json").read_text())
    validate_spec(spec, cfg)
    changed = copy.deepcopy(spec); changed["common_pool"]["train_fit_pairs"] = 32
    with pytest.raises(ValueError): validate_spec(changed, cfg)
    changed = copy.deepcopy(spec); changed["q_distance_m"]["minimum_inclusive"] = 2.99
    with pytest.raises(ValueError): validate_spec(changed, cfg)


def test_end_to_end_train_only_manifest_source_binding_and_immutable_output(tmp_path):
    dataset = tmp_path / "datasets"
    anno = dataset / "R2R/annotations/R2R_train_enc.json"
    conn = dataset / "R2R/connectivity"
    anno.parent.mkdir(parents=True); conn.mkdir(parents=True)
    cfg = json.loads((ROOT / "configs/r2r.json").read_text())
    cfg["dataset_root"] = str(dataset)
    cfg_path = tmp_path / "runtime.json"; cfg_path.write_text(json.dumps(cfg))
    xyz = {"s": 0., "a": 2., "ga": 4., "b": -2., "gb": -4.}
    adjacent = {"s": ["a", "b"], "a": ["s", "ga"], "ga": ["a"], "b": ["s", "gb"], "gb": ["b"]}
    raw, normalized, all_pairs = [], [], []
    for scan in ("one", "two"):
        connectivity = []
        for node, x in xyz.items():
            pose = [0.] * 16; pose[3] = x
            connectivity.append({"image_id": node, "included": True, "pose": pose,
                                 "unobstructed": [v in adjacent[node] for v in xyz]})
        (conn / f"{scan}_connectivity.json").write_text(json.dumps(connectivity))
        for label, path in (("1", ["s", "a", "ga"]), ("2", ["s", "b", "gb"])):
            r = record(path, scan=scan, pid=scan + label)
            normalized.append(r)
            raw.append({k: v for k, v in dict(r, instructions=["the real source instruction"]).items() if k != "num_instructions"})
        all_pairs.append(pair_for(scan, scan + "1", scan + "2"))
    anno.write_text(json.dumps(raw))
    coverage = {"schema": "duet_endpoint_pair_coverage_v1", "usage": "train_only_geometry_diagnostic", "coverage_pass": True,
        "identity": {"config_sha256": file_sha256(cfg_path), "annotation_sha256": file_sha256(anno),
                     "connectivity_sha256": object_sha256({scan: file_sha256(conn / f"{scan}_connectivity.json") for scan in ("one", "two")})},
        "specification": {"selection_seed": 20261003}, "splits": {}}
    for split in ("train_fit", "train_dev"):
        scan_set = {r["scan"] for r in select_partition(normalized, split)}
        primary = [p for p in all_pairs if p["scan"] in scan_set]
        coverage["splits"][split] = {"primary_path_disjoint_manifest": primary, "primary_manifest_sha256": object_sha256(primary)}
    source = tmp_path / "d3.json"; source.write_text(json.dumps(coverage))
    spec = json.loads((ROOT / "configs/endpoint_controls_diagnostic.json").read_text())
    spec["coverage_report_sha256"] = file_sha256(source)
    diagnostic = tmp_path / "diagnostic.json"; diagnostic.write_text(json.dumps(spec))
    report = run(cfg_path, diagnostic, source)
    assert not report["coverage_pass"]
    assert report["splits"]["train_fit"]["shortfall_pairs"] == 511
    assert report["splits"]["train_dev"]["shortfall_pairs"] == 127
    assert report["gpu_calls"] == report["validation_accesses"] == report["training_updates"] == 0
    destination = tmp_path / "report.json"; write_new_json(destination, report)
    with pytest.raises(FileExistsError): write_new_json(destination, report)
    anno.write_text(anno.read_text() + " ")
    with pytest.raises(ValueError, match="annotation changed"):
        run(cfg_path, diagnostic, source)
