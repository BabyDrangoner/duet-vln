import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("endpoint_failure_diagnostic", Path(__file__).resolve().parents[1] / "scripts/analyze_endpoint_failures.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def fixture():
    arms = ["C1", "C2", "C3", "M"]
    config = {"arms": arms, "expected_episodes": 2, "scope": "synthetic", "missing_evidence": [],
              "bins": {"endpoint_age_steps": [0, 3, 7, 14], "legal_move_count": [0, 4, 9, 1000],
                       "history_states": [5, 10, 15], "baseline_stop_probability": [.25, .5, .75, 1]},
              "strata": ["history_states", "baseline_success", "termination_combination"]}
    reports = {name: {"episodes": [], "trajectories": [], "endpoint_decisions": []} for name in ["baseline", *arms]}
    traces = []
    for i in range(2):
        # One rescue and one harm; rescue costs 2m, harm costs 2m.
        instr, base_success = str(i), bool(i)
        nodes = [f"{i}_a", f"{i}_b"]
        prefix = [[nodes[0]], [nodes[1]]]
        base = {"instr_id": instr, "scan_id": "scene", "success": int(base_success),
                "spl": .8 if base_success else 0., "nDTW": .5, "nav_error": 0. if base_success else 5.,
                "trajectory_lengths": 10., "oracle_success": 1.}
        reports["baseline"]["episodes"].append(base)
        reports["baseline"]["trajectories"].append({"instr_id": instr, "trajectory": prefix})
        states = [{"step": j, "viewpoint": node, "success": (not base_success if j == 0 else base_success),
                   "distance_to_goal_m": 0. if (not base_success if j == 0 else base_success) else 5.,
                   "scores": {"baseline_probability": .1 if j == 0 else .9}, "move_count": 3}
                  for j, node in enumerate(nodes)]
        traces.append({"association": {"instr_id": instr, "scan_id": "scene"}, "path_id": i,
                       "states": states, "num_states": 2, "observed_history_has_success": True,
                       "termination_flags": {"argmax_stop": True, "no_legal_move": False, "action_limit": False},
                       "prefix_length_m": 10., "methods": {"baseline_probability": {
                           "viewpoint": nodes[1], "success": base_success, "nav_error_m": base["nav_error"],
                           "complete_trajectory_length_m": 10., "return_distance_m": 0.}}})
        for arm in arms:
            reports[arm]["episodes"].append({**base, "success": int(not base_success), "spl": 0. if base_success else .4,
                                              "trajectory_lengths": 12., "nav_error": 5. if base_success else 0.})
            reports[arm]["trajectories"].append({"instr_id": instr, "trajectory": prefix + [[nodes[0]]]})
            reports[arm]["endpoint_decisions"].append({"instr_id": instr, "baseline_endpoint": nodes[1],
                "probe_endpoint": nodes[0], "online_path_and_termination_parity": True, "online_decisions": 2,
                "endpoint_changed": True, "prefix_path": prefix})
    reports["S1"] = {"all_instruction_trajectory_and_metric_parity": True, "split": "val_unseen", "usage": "analysis_only",
                     "identity_sha256": "a", "episodes": traces}
    reports["S1_collection"] = {"identity_sha256": "a"}
    for name in ("train_controls", "dev_controls", "train_pairs", "dev_pairs"):
        reports[name] = {"split": "train_fit", "files": [], "states": 0, "pairs": 0}
    return config, reports


def test_exact_decomposition_and_opportunity_denominator():
    result = diagnostic.analyze(*fixture())
    assert result["opportunities"]["recoverable_baseline_failure"]["instructions"] == 1
    arm = result["arms"]["M"]
    assert arm["overall"]["rescues"] == arm["overall"]["harms"] == 1
    assert arm["overall"]["spl_contribution_pp"] == pytest.approx(-20)
    assert arm["by_outcome"]["rescue"]["spl_contribution_pp"] == pytest.approx(20)
    assert arm["by_outcome"]["harm"]["spl_contribution_pp"] == pytest.approx(-40)
    assert arm["overall"]["rescue_fraction_of_observed_opportunities"] == 1
    assert arm["overall"]["mean_return_delta_m"] == 2


def test_rejects_trace_prefix_mismatch():
    config, reports = fixture()
    reports["M"]["endpoint_decisions"][0]["prefix_path"] = [["unobserved"]]
    with pytest.raises(ValueError, match="prefix observations"):
        diagnostic.analyze(config, reports)


def test_rejects_missing_instruction():
    config, reports = fixture()
    reports["C1"]["episodes"].pop()
    with pytest.raises(ValueError, match="coverage"):
        diagnostic.analyze(config, reports)


def test_bins_include_exact_boundaries_without_fitting():
    assert diagnostic.bucket(0, [0, 3, 7, 14]) == "le_0"
    assert diagnostic.bucket(3, [0, 3, 7, 14]) == "gt_0_le_3"
    with pytest.raises(ValueError, match="outside"):
        diagnostic.bucket(15, [0, 3, 7, 14])
