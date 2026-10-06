import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_endpoint_pairs_full import common_pairs, validate_pilot_audit, validate_spec
from vln_improve.protocol import object_sha256


def sources():
    pairs = []
    pool = {}
    for index in range(3):
        scan, start = "scan", f"s{index}"
        ids = [f"{index}A", f"{index}B"]
        pair = {"scan": scan, "start": start, "path_ids": ids,
                "instr_ids": [p + "_0" for p in ids], "goal_vpids": [f"g{p}" for p in ids],
                "heading_rad": [0., 0.], "heading_difference_deg": 0., "goal_separation_m": 10.,
                "histories": {}}
        pair["selection_hash"] = object_sha256([20261003, scan, *ids])
        pairs.append(pair)
        for slot, pid in enumerate(ids):
            instr, goal = pair["instr_ids"][slot], pair["goal_vpids"][slot]
            cid, q = object_sha256([scan, pid, instr]), "q" + pid
            positive = {"observed_vpids": [start, goal], "observed_states": 2}
            negative = {"observed_vpids": [start, goal, q], "observed_states": 3}
            pool[cid] = {"control_id": cid, "scan": scan, "path_id": pid, "instr_id": instr,
                         "start": start, "goal": goal, "heading": 0., "original_path": [start, goal],
                         "eligible": True, "positive_history": positive, "overshoot_history": negative,
                         "selected_q": q, "selected_q_goal_distance_m": 3.,
                         "selected_q_hash": object_sha256([0, scan, pid, q])}
    pairs.sort(key=lambda x: x["selection_hash"])
    first = pairs[0]
    first_control = object_sha256([first["scan"], first["path_ids"][0], first["instr_ids"][0]])
    pool[first_control]["eligible"] = False
    entries = [{"pair": p, "control_ids": [object_sha256([p["scan"], pid, instr])
                 for pid, instr in zip(p["path_ids"], p["instr_ids"])], "natural_instr_ids": p["instr_ids"]} for p in pairs[1:]]
    common = {k: str(i) * 64 for i, k in enumerate(("config_sha256", "annotation_sha256", "connectivity_sha256"), 1)}
    coverage = {"schema": "duet_endpoint_pair_coverage_v1", "coverage_pass": True,
                "usage": "train_only_geometry_diagnostic", "identity": copy.deepcopy(common),
                "specification": {"selection_seed": 20261003},
                "splits": {"train_fit": {"primary_path_disjoint_manifest": pairs, "primary_manifest_sha256": object_sha256(pairs)}}}
    manifest = {"requested_pairs": 2, "selected_pairs": entries, "controls": pool}
    controls = {"schema": "duet_endpoint_controls_geometry_v1", "coverage_pass": True,
                "usage": "train_only_geometry_diagnostic", "identity": copy.deepcopy(common),
                "splits": {"train_fit": {"coverage_pass": True, "shortfall_pairs": 0, "requested_pairs": 2,
                            "manifest": manifest, "manifest_sha256": object_sha256(manifest)}}}
    return coverage, controls


def refresh(controls):
    result = controls["splits"]["train_fit"]
    result["manifest_sha256"] = object_sha256(result["manifest"])


def test_uses_c2_feasible_pool_not_original_d3_first_n():
    coverage, controls = sources()
    selected = common_pairs(controls, coverage, "train_fit", count=2)
    primary = coverage["splits"]["train_fit"]["primary_path_disjoint_manifest"]
    assert selected == primary[1:] and selected != primary[:2]
    selected[0]["scan"] = "changed"
    assert controls["splits"]["train_fit"]["manifest"]["selected_pairs"][0]["pair"]["scan"] == "scan"


@pytest.mark.parametrize("mutation", ["source", "shortfall", "order", "original_first_n", "extra_control", "bad_q", "bad_control_goal"])
def test_invalid_or_changed_common_pool_cannot_be_collected(mutation):
    coverage, controls = sources()
    section = controls["splits"]["train_fit"]
    manifest = section["manifest"]
    if mutation == "source": controls["identity"]["annotation_sha256"] = "f" * 64
    if mutation == "shortfall": section["coverage_pass"] = False; section["shortfall_pairs"] = 1
    if mutation == "order": manifest["selected_pairs"].reverse()
    if mutation == "original_first_n":
        first = coverage["splits"]["train_fit"]["primary_path_disjoint_manifest"][0]
        manifest["selected_pairs"][0] = {"pair": first, "natural_instr_ids": first["instr_ids"],
            "control_ids": [object_sha256([first["scan"], p, i]) for p, i in zip(first["path_ids"], first["instr_ids"])]}
    if mutation == "extra_control": manifest["controls"]["unexpected"] = {}
    if mutation in {"bad_q", "bad_control_goal"}:
        cid = manifest["selected_pairs"][0]["control_ids"][0]
        if mutation == "bad_q": manifest["controls"][cid]["selected_q_goal_distance_m"] = 2.99
        else: manifest["controls"][cid]["goal"] = "wrong-goal"
    refresh(controls)
    with pytest.raises(ValueError): common_pairs(controls, coverage, "train_fit", count=2)


def test_gpu_seed_zero_cannot_be_used_to_validate_selection_hashes():
    coverage, controls = sources()
    assert len(common_pairs(controls, coverage, "train_fit", count=2, selection_seed=20261003)) == 2
    with pytest.raises(ValueError): common_pairs(controls, coverage, "train_fit", count=2, selection_seed=0)
    primary = coverage["splits"]["train_fit"]
    row = primary["primary_path_disjoint_manifest"][0]
    row["selection_hash"] = object_sha256([0, row["scan"], *row["path_ids"]])
    primary["primary_manifest_sha256"] = object_sha256(primary["primary_path_disjoint_manifest"])
    with pytest.raises(ValueError, match="selection hash"):
        common_pairs(controls, coverage, "train_fit", count=2)


def test_only_training_splits_and_complete_common_report_are_allowed():
    coverage, controls = sources()
    with pytest.raises(ValueError): common_pairs(controls, coverage, "val_unseen", count=2)
    controls["coverage_pass"] = False
    with pytest.raises(ValueError, match="gates"):
        common_pairs(controls, coverage, "train_fit", count=2)


def audit_fixture():
    return {"splits": {name: {"pairs": count, "rollouts": count * 4,
                "all_shared_history_exact_parity": True, "all_drive_hashes_match": True, "manifest_sha256": "f" * 64}
             for name, count in (("endpoint-pair-pilot-train-fit", 32), ("endpoint-pair-pilot-train-dev", 16))}}


def test_pilot_gate_requires_both_strict_parity_and_drive_audit():
    audit = audit_fixture(); validate_pilot_audit(audit)
    audit["splits"]["endpoint-pair-pilot-train-dev"]["all_drive_hashes_match"] = False
    with pytest.raises(ValueError): validate_pilot_audit(audit)
    audit = audit_fixture(); audit["splits"]["endpoint-pair-pilot-train-fit"]["all_shared_history_exact_parity"] = False
    with pytest.raises(ValueError): validate_pilot_audit(audit)


def test_full_spec_pins_512_128_and_distinct_model_selection_seeds():
    spec = json.loads((ROOT / "configs/endpoint_pair_full.json").read_text())
    config = json.loads((ROOT / "configs/r2r.json").read_text())
    validate_spec(spec, config)
    assert spec["seed"] == 0 and spec["selection_seed"] == 20261003
    for key, value in (("selection_seed", 0), ("seed", 1), ("controls_report_sha256", None)):
        changed = copy.deepcopy(spec); changed[key] = value
        with pytest.raises(ValueError): validate_spec(changed, config)
    changed = copy.deepcopy(spec); changed["selection"]["train_fit"] = 32
    with pytest.raises(ValueError): validate_spec(changed, config)
