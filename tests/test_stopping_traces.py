import copy
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_stopping_traces import main, output_lock, validate_access, validate_reference
from test_diagnostics import FakeAgent
from test_study_ledger import budget, request, study
from vln_improve.protocol import file_sha256
from vln_improve.stopping_traces import StopTraceObserver, StopTraceStore, exact_report_parity, restore_payload, summarize_store
from vln_improve.study_ledger import StudyLedger


def _agent():
    agent = FakeAgent()
    def evaluate(scan, path, goal_path):
        sequence = sum(path, [])
        distances = agent.env.shortest_distances[scan]
        return {"trajectory_lengths": sum(distances[a][b] for a, b in zip(sequence[:-1], sequence[1:])),
                "nav_error": distances[sequence[-1]][goal_path[-1]],
                "success": float(distances[sequence[-1]][goal_path[-1]] < 3), "oracle_success": 1.,
                "spl": 1., "nDTW": 1.}
    agent.env._eval_item = evaluate
    return agent


def _store(tmp_path, *, reference_change=None):
    baseline = _agent()
    trajectory = baseline.rollout()[0]
    scores = baseline.env._eval_item("scan", trajectory["path"], ["A", "B", "C"])
    metrics = dict(instr_id="instruction", scan_id="scan", **{key: float(value) for key, value in scores.items()})
    reference = {"episodes": [metrics], "trajectories": [{"instr_id": "instruction", "trajectory": trajectory["path"]}]}
    if reference_change:
        reference_change(reference)
    identity = {"split": "val_unseen", "selection": [{"instr_id": "instruction", "scan": "scan", "path_id": "actual-path-id"}],
                "model": {"max_action_len": 15}, "usage": "analysis_only", "fixture": True}
    store = StopTraceStore(tmp_path / "local", identity, reference, backup=tmp_path / "drive", verify_backup=lambda: None)
    return store, reference


def test_read_only_trace_has_no_features_and_all_official_metrics_match(tmp_path):
    store, reference = _store(tmp_path)
    agent = _agent()
    observer = StopTraceObserver(agent, store)
    expected = copy.deepcopy(reference)
    result = agent.rollout()
    assert result[0]["path"] == reference["trajectories"][0]["trajectory"]
    assert reference == expected
    trace = store.find("scan", "instruction")
    assert trace["metrics"] == reference["episodes"][0]
    assert trace["labels"]["usage"] == trace["usage"] == "analysis_only"
    raw = next(store.local.glob("episode-*.json")).read_text()
    assert "gmap_img_embeds" not in raw and "txt_embeds" not in raw and "Infinity" not in raw
    assert len(raw) < 25000
    assert observer.new_episodes == 1
    summary = summarize_store(store)
    assert summary["all_instruction_trajectory_and_metric_parity"]
    assert summary["overall"]["episodes"] == 1
    assert summary["episodes"][0]["path_id"] == "actual-path-id"
    assert summary["opportunity_support"]["observed_history_success_but_baseline_failed"] == {
        "instructions": 0, "independent_paths": 0, "scenes": 0}
    observer.close()


def test_completed_episode_is_restored_from_drive_without_running_policy(tmp_path):
    store, reference = _store(tmp_path)
    first = StopTraceObserver(_agent(), store)
    original = first.agent.rollout()
    first.close()
    shutil.rmtree(store.local)
    restored = StopTraceStore(store.local, store.identity, reference, backup=store.backup, verify_backup=lambda: None)
    agent = _agent()
    observer = StopTraceObserver(agent, restored)
    assert agent.rollout() == original
    assert agent.actual_rollouts == 0
    assert observer.reused_episodes == 1
    observer.close()


def test_interrupted_backup_preserves_local_episode_and_retries_copy(tmp_path):
    store, reference = _store(tmp_path)
    def failing_backup():
        if list(store.local.glob("episode-*.json")):
            raise OSError("lost Drive")
    store.verify_backup = failing_backup
    agent = _agent()
    observer = StopTraceObserver(agent, store)
    with pytest.raises(OSError, match="lost Drive"):
        agent.rollout()
    observer.close()
    assert len(list(store.local.glob("episode-*.json"))) == 1
    store.verify_backup = lambda: None
    second = _agent()
    resumed = StopTraceObserver(second, store)
    second.rollout()
    assert second.actual_rollouts == 0
    assert len(list(store.backup.glob("episode-*.json"))) == 1
    resumed.close()


@pytest.mark.parametrize("damage", ["nDTW", "trajectory"])
def test_parity_rejects_any_metric_or_path_difference_before_commit(tmp_path, damage):
    def edit(report):
        if damage == "nDTW":
            report["episodes"][0]["nDTW"] = .99
        else:
            report["trajectories"][0]["trajectory"] = [["A"], ["C"]]
    store, _ = _store(tmp_path, reference_change=edit)
    observer = StopTraceObserver(_agent(), store)
    with pytest.raises(ValueError, match="baseline trajectory/metric"):
        observer.agent.rollout()
    assert not list(store.local.glob("episode-*.json"))
    observer.close()


def test_corrupted_backup_or_changed_identity_is_not_silently_reused(tmp_path):
    store, reference = _store(tmp_path)
    observer = StopTraceObserver(_agent(), store)
    observer.agent.rollout()
    observer.close()
    path = next(store.backup.glob("episode-*.json"))
    value = json.loads(path.read_text())
    value["trace"]["metrics"]["nDTW"] = .2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        store.find("scan", "instruction")
    with pytest.raises(ValueError, match="identity changed"):
        StopTraceStore(store.local, dict(store.identity, fixture=False), reference, backup=store.backup, verify_backup=lambda: None)


def test_trace_schema_rejects_feature_arrays_and_malformed_masks(tmp_path):
    store, _ = _store(tmp_path)
    observer = StopTraceObserver(_agent(), store)
    observer.agent.rollout()
    observer.close()
    trace = store.find("scan", "instruction")
    invalid = copy.deepcopy(trace)
    invalid["states"][0]["nav_inputs"]["txt_embeds"] = [[0]]
    with pytest.raises(ValueError, match="large feature"):
        restore_payload(invalid)
    invalid = copy.deepcopy(trace)
    invalid["states"][0]["base_logits"][0] = None
    with pytest.raises(ValueError, match="masked logits"):
        restore_payload(invalid)


def _registered(tmp_path):
    spec_path = tmp_path / "diagnostic.json"
    spec_path.write_text('{"episodes":2349}')
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study()))
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    ledger.register(**request(category="diagnostic_analysis", config_sha256=file_sha256(spec_path), label_use="analysis",
                             budget_path=budget(tmp_path, category="diagnostic_analysis")))
    arguments = dict(split="val_unseen", access_id="P1", ledger_path=ledger.path, study_path=study_path,
                     diagnostic_config=spec_path, spec={"episodes": 2349}, checkpoint_sha256="b" * 64,
                     code_sha256="c" * 64, seed=0)
    return ledger, arguments


def test_registered_val_access_is_read_only_and_requires_exact_identities(tmp_path):
    ledger, arguments = _registered(tmp_path)
    before = ledger.path.read_bytes()
    assert validate_access(**arguments)["request"]["category"] == "diagnostic_analysis"
    assert ledger.path.read_bytes() == before
    for key, value in (("code_sha256", "d" * 64), ("checkpoint_sha256", "d" * 64), ("seed", 1), ("access_id", None)):
        with pytest.raises(ValueError):
            validate_access(**dict(arguments, **{key: value}))
    ledger.finish("P1", status="failed", metrics={}, resources={"seconds": 0}, decision="do not reuse", error="test")
    with pytest.raises(ValueError, match="terminal outcome"):
        validate_access(**arguments)


def test_unregistered_access_or_changed_diagnostic_spec_is_rejected(tmp_path):
    _, arguments = _registered(tmp_path)
    with pytest.raises(KeyError):
        validate_access(**dict(arguments, access_id="unknown"))
    Path(arguments["diagnostic_config"]).write_text('{"episodes":2349,"changed":true}')
    with pytest.raises(ValueError, match="registration differs"):
        validate_access(**arguments)


def test_full_report_parity_rejects_same_means_but_swapped_instruction_metrics():
    reference = {"episodes": [{"instr_id": "a", "success": 1}, {"instr_id": "b", "success": 0}],
                 "trajectories": [{"instr_id": "a", "trajectory": [["A"]]}, {"instr_id": "b", "trajectory": [["B"]]}]}
    changed = copy.deepcopy(reference)
    changed["episodes"][0]["success"], changed["episodes"][1]["success"] = 0, 1
    with pytest.raises(ValueError, match="parity"):
        exact_report_parity(reference, changed)
    exact_report_parity(reference, copy.deepcopy(reference))


def test_code_identity_cli_never_checks_cuda(capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("must not inspect CUDA"))
    main(["--print-code-sha256"])
    result = capsys.readouterr().out.strip()
    assert len(result) == 64 and all(c in "0123456789abcdef" for c in result)


def test_output_lock_rejects_concurrent_collectors(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another STOP collector"):
            with output_lock(tmp_path):
                pass
