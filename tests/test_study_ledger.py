import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vln_improve.study_ledger import StudyLedger


def study():
    return {"study_id": "test-study", "baseline": {"base_checkpoint_sha256": "b" * 64,
                                                 "selection_episodes": 2349},
            "evaluation_protocol": {"parameter_fitting_split": "train_fit",
                                    "unseen_development_split": "val_unseen",
                                    "pilot_max_variants": 2, "pilot_seed": 0,
                                    "pilot_checkpoint_evaluations_per_variant": 3}}


def request(**updates):
    result = dict(access_id="P1", category="pilot", variant_id="method-one", config_sha256="a" * 64,
                  checkpoint_sha256="b" * 64, code_sha256="c" * 64, purpose="full navigation selection",
                  split="val_unseen", seed=0, expected_episodes=2349)
    return dict(result, **updates)


def budget(tmp_path, category="confirmatory", **updates):
    value = dict(schema_version=1, study_id="test-study", category=category, budget_id="frozen-v1",
                 reason="method frozen before three-seed report", max_variants=1, max_accesses=3,
                 max_accesses_per_variant=3, seeds=[0, 1, 2], splits=["val_unseen"],
                 variants={"method-one": "a" * 64})
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(dict(value, **updates)))
    return path


def test_register_and_finish_are_idempotent_but_conflicting_reuse_fails(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    initial = ledger.register(**request())
    content = ledger.path.read_bytes()
    assert ledger.register(**request()) == initial
    assert ledger.path.read_bytes() == content
    with pytest.raises(ValueError, match="different registration"):
        ledger.register(**request(checkpoint_sha256="d" * 64))
    report = tmp_path / "report.json"
    report.write_text('{"sr": 70}')
    result = dict(status="completed", metrics={"sr": 70}, resources={"seconds": 10},
                  decision="retain", report_path=report)
    completed = ledger.finish("P1", **result)
    assert ledger.finish("P1", **result) == completed
    assert len(ledger.path.read_text().splitlines()) == 2
    assert ledger.status()["pending"] == []
    report.write_text('{"sr": 71}')
    with pytest.raises(ValueError, match="different terminal outcome"):
        ledger.finish("P1", **result)


def test_failed_and_pending_accesses_consume_budget_and_cannot_be_erased(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    ledger.register(**request())
    ledger.finish("P1", status="failed", metrics={}, resources={"seconds": None},
                  decision="rerun with a new access ID", error="runtime lost")
    for i in (2, 3):
        ledger.register(**request(access_id=f"P{i}"))
    with pytest.raises(ValueError, match="per-variant access budget"):
        ledger.register(**request(access_id="P4"))
    assert ledger.status()["usage"]["pilot"]["variants"] == {"method-one": 3}
    assert ledger.status()["outcomes"] == {"P1": "failed"}
    with pytest.raises(ValueError, match="new ID to reset"):
        ledger.register(**request(access_id="P5", variant_id="renamed-method"))
    with pytest.raises(ValueError, match="requires a new variant ID"):
        ledger.register(**request(access_id="P5", config_sha256="d" * 64))


def test_baseline_shares_pilot_limits(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    ledger.register(**request(category="baseline"))
    ledger.register(**request(access_id="P2", variant_id="second", config_sha256="d" * 64))
    with pytest.raises(ValueError, match="variant budget exhausted"):
        ledger.register(**request(access_id="P3", variant_id="third", config_sha256="e" * 64))
    assert len(ledger.status()["usage"]["pilot"]["variants"]) == 2


def test_preserves_and_accounts_for_existing_manual_baseline(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    first = dict(access_id="V0001", status="registered_before_execution", method_id="DUET-official-frozen",
                 config_sha256="a" * 64, checkpoint_sha256="b" * 64, purpose="baseline", split="val_unseen",
                 subset=False, seed=0, expected_episodes=2349, variant_budget_charge=0)
    second = dict(access_id="V0001", event="completed", status="complete", metrics={"sr": 70})
    original = "".join(json.dumps(x) + "\n" for x in (first, second)).encode()
    ledger.path.write_bytes(original)
    state = ledger.status()
    assert state["usage"]["pilot"] == {"accesses": 1, "variants": {"DUET-official-frozen": 1}}
    assert "variant_budget_charge was 0" in state["legacy_notes"][0]
    access = ledger.lookup("V0001")
    assert access["registration"]["legacy"] is True
    assert access["registration"]["request"]["category"] == "baseline"
    assert access["registration"]["budget_group"] == "pilot"
    assert access["outcome"] == second
    assert ledger.path.read_bytes() == original
    ledger.register(**request(config_sha256="d" * 64))
    assert ledger.path.read_bytes().startswith(original)


def test_lookup_unknown_access_never_registers_or_changes_the_ledger(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    with pytest.raises(KeyError, match="not registered"):
        ledger.lookup("missing")
    assert not ledger.path.exists()
    ledger.register(**request())
    original = ledger.path.read_bytes()
    with pytest.raises(KeyError, match="not registered"):
        ledger.lookup("missing")
    assert ledger.path.read_bytes() == original


def test_lookup_pending_and_completed_are_read_only_independent_copies(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    registered = ledger.register(**request(subset=True, subset_ids=["1_0"], expected_episodes=1))
    original = ledger.path.read_bytes()
    access = ledger.lookup("P1")
    assert access == {"registration": registered, "outcome": None}
    access["registration"]["request"]["subset_ids"].append("unexpected")
    assert ledger.lookup("P1")["registration"]["request"]["subset_ids"] == ["1_0"]
    assert ledger.path.read_bytes() == original

    report = tmp_path / "report.json"
    report.write_text('{"sr":70}')
    completed = ledger.finish("P1", status="completed", metrics={"sr": 70}, resources={"seconds": 1},
                              decision="retain", report_path=report)
    original = ledger.path.read_bytes()
    access = ledger.lookup("P1")
    assert access["outcome"] == completed
    access["outcome"]["metrics"]["sr"] = 100
    assert ledger.lookup("P1")["outcome"]["metrics"]["sr"] == 70
    assert ledger.path.read_bytes() == original


@pytest.mark.parametrize("category", ["diagnostic_analysis", "confirmatory"])
def test_separate_categories_require_explicit_budget(tmp_path, category):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    fields = request(category=category, label_use="analysis" if category == "diagnostic_analysis" else "evaluation")
    with pytest.raises(ValueError, match="explicit budget"):
        ledger.register(**fields)
    path = budget(tmp_path, category)
    ledger.register(**dict(fields, budget_path=path))
    assert ledger.status()["usage"][category]["accesses"] == 1
    assert "pilot" not in ledger.status()["usage"]


def test_confirmatory_freeze_and_budget_revision_keep_old_usage(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    path = budget(tmp_path, max_accesses=1)
    ledger.register(**request(category="confirmatory", budget_path=path))
    with pytest.raises(ValueError, match="absent from the frozen"):
        ledger.register(**request(access_id="F2", category="confirmatory", config_sha256="d" * 64, budget_path=path))
    # A new file ID does not reset access counts; an explicit higher limit can extend them.
    budget(tmp_path, budget_id="new-name", max_accesses=1)
    with pytest.raises(ValueError, match="total access budget"):
        ledger.register(**request(access_id="F2", category="confirmatory", seed=1, budget_path=path))
    budget(tmp_path, budget_id="extension", max_accesses=2, reason="explicit documented extension")
    ledger.register(**request(access_id="F2", category="confirmatory", seed=1, budget_path=path))
    assert ledger.status()["usage"]["confirmatory"]["accesses"] == 2


@pytest.mark.parametrize("updates,match", [
    ({"split": "train_dev"}, "only val_unseen"),
    ({"split": "train_fit"}, "only val_unseen"),
    ({"parameter_fitting_split": "val_unseen"}, "fitting must remain train_fit"),
    ({"label_use": "fit"}, "validation fitting is forbidden"),
    ({"subset": True}, "all fixed IDs"),
    ({"expected_episodes": 10}, "complete official split"),
    ({"seed": 1}, "seed is outside"),
    ({"config_sha256": "not-sha"}, "SHA-256"),
])
def test_invalid_or_leaking_requests_do_not_append(tmp_path, updates, match):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    with pytest.raises(ValueError, match=match):
        ledger.register(**request(**updates))
    assert not ledger.path.exists()


def test_subset_identity_cannot_change_between_attempts(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    fields = request(subset=True, subset_ids=["2_0", "1_0"], expected_episodes=2)
    ledger.register(**fields)
    ledger.register(**dict(fields, subset_ids=["1_0", "2_0"]))
    with pytest.raises(ValueError, match="different registration"):
        ledger.register(**dict(fields, subset_ids=["1_0", "3_0"]))


def test_torn_record_and_nonfinite_results_fail_closed(tmp_path):
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    ledger.path.write_text('{"access_id":"torn"')
    with pytest.raises(ValueError, match="unterminated"):
        ledger.register(**request())
    assert ledger.path.read_text() == '{"access_id":"torn"'
    ledger.path.unlink()
    ledger.register(**request())
    with pytest.raises(ValueError, match="finite numeric"):
        ledger.finish("P1", status="failed", metrics={"sr": float("nan")},
                      resources={"seconds": 1}, decision="invalid", error="nan")
    assert len(ledger.path.read_text().splitlines()) == 1


def _contending_register(path, output):
    limited = study()
    limited["evaluation_protocol"]["pilot_checkpoint_evaluations_per_variant"] = 1
    ledger = StudyLedger(Path(path), limited)
    try:
        ledger.register(**request(access_id=str(multiprocessing.current_process().pid)))
        output.put("registered")
    except ValueError as exc:
        output.put(str(exc))


def test_local_flock_makes_budget_check_and_append_one_transaction(tmp_path):
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    children = [context.Process(target=_contending_register, args=(str(tmp_path / "ledger.jsonl"), output))
                for _ in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    results = [output.get(timeout=5) for _ in children]
    assert results.count("registered") == 1
    assert any("per-variant access budget exhausted" in value for value in results)
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1


def test_cli_registers_actual_config_and_report_hashes(tmp_path):
    from vln_improve.protocol import file_sha256

    root = Path(__file__).resolve().parents[1]
    study_path, config_path = tmp_path / "study.json", tmp_path / "method.json"
    study_path.write_text(json.dumps(study()))
    config_path.write_text('{"method":"fixed"}')
    command = [sys.executable, str(root / "scripts/register_study_access.py"),
               "--study", str(study_path), "--ledger", str(tmp_path / "accesses.jsonl")]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    registered = subprocess.run(command + ["register", "--access-id", "CLI1", "--category", "pilot",
                                "--variant-id", "method-one", "--config", str(config_path),
                                "--checkpoint-sha256", "b" * 64, "--code-sha256", "c" * 64,
                                "--purpose", "CLI integration", "--seed", "0", "--expected-episodes", "2349"],
                                env=env, check=True, capture_output=True, text=True)
    assert json.loads(registered.stdout)["record"]["request"]["config_sha256"] == file_sha256(config_path)
    metrics, resources = tmp_path / "metrics.json", tmp_path / "resources.json"
    metrics.write_text('{"sr":70}')
    resources.write_text('{"seconds":1}')
    completed = subprocess.run(command + ["complete", "--access-id", "CLI1", "--metrics", str(metrics),
                               "--resources", str(resources), "--report", str(metrics), "--decision", "retain"],
                               env=env, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["record"]["report_sha256"] == file_sha256(metrics)
    assert result["status"]["outcomes"] == {"CLI1": "completed"}
