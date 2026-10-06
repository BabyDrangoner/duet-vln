"""Finite pipeline persistence and identity checks without Drive or a GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_intervention_loop", ROOT / "scripts/run_intervention_loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def ledger_prefix():
    return b"".join((ROOT / loop.LEDGER_RELATIVE).read_bytes().splitlines(keepends=True)[:14])


def test_arm_configs_are_complete_distinct_and_do_not_mutate_master():
    master = loop.read(ROOT / "configs/e2_loop_v1.json")
    before = copy.deepcopy(master)
    values = loop.derive_arm_configs(master, "a" * 64)
    assert master == before
    assert values["relative"]["collection"] == values["absolute"]["collection"] == master["collection"]
    assert values["relative"]["collection_config_sha256"] == "a" * 64
    assert loop.object_sha(values["relative"]) != loop.object_sha(values["absolute"])
    loop.validate_master(master)
    master["epochs"] = 21
    with pytest.raises(ValueError, match="budget"):
        loop.validate_master(master)


def test_ledger_recovery_extends_only_verified_prefixes(tmp_path):
    local, cloud = tmp_path / "local.jsonl", tmp_path / "cloud.jsonl"
    cloud.write_bytes(ledger_prefix())
    loop.reconcile_ledgers(local, cloud)
    assert local.read_bytes() == cloud.read_bytes()
    local.write_bytes(local.read_bytes() + b'{"event":"test_append"}\n')
    loop.reconcile_ledgers(local, cloud)
    assert local.read_bytes() == cloud.read_bytes()
    cloud.write_bytes(cloud.read_bytes() + b'{"event":"second_append"}\n')
    loop.reconcile_ledgers(local, cloud)
    assert local.read_bytes() == cloud.read_bytes()


def test_divergent_ledgers_are_not_overwritten(tmp_path):
    local, cloud = tmp_path / "local.jsonl", tmp_path / "cloud.jsonl"
    left, right = ledger_prefix() + b'{"writer":"a"}\n', ledger_prefix() + b'{"writer":"b"}\n'
    local.write_bytes(left)
    cloud.write_bytes(right)
    with pytest.raises(ValueError, match="diverge"):
        loop.reconcile_ledgers(local, cloud)
    assert local.read_bytes() == left and cloud.read_bytes() == right


@pytest.mark.parametrize("raw", [b"{}\n", ledger_prefix()[:-1], b"x" + ledger_prefix()[1:]])
def test_bad_original_ledger_is_rejected(raw):
    with pytest.raises(ValueError):
        loop.verify_initial_ledger(raw)


def test_immutable_artifact_rejects_overwrite(tmp_path):
    source, target = tmp_path / "source.json", tmp_path / "target.json"
    loop.immutable_json(source, {"fixed": 1})
    loop.verified_copy(source, target)
    with pytest.raises(ValueError, match="frozen JSON changed"):
        loop.immutable_json(source, {"fixed": 2})
    target.write_text("different")
    with pytest.raises(ValueError, match="immutable backup differs"):
        loop.verified_copy(source, target)


def minimal_runner(tmp_path):
    runner = loop.Loop.__new__(loop.Loop)
    runner.output, runner.backup = tmp_path / "local", tmp_path / "drive"
    runner.frozen = runner.output / "frozen"
    runner.output.mkdir()
    runner.backup.mkdir()
    runner.frozen.mkdir()
    runner.master_sha = "a" * 64
    runner.source = {"execution_files_sha256": "b" * 64}
    return runner


def training_summary(runner, arm, *, complete=True):
    run = runner.output / "training" / arm
    run.mkdir(parents=True)
    head = run / "snapshots" / "chosen" / "head.pt"
    head.parent.mkdir(parents=True)
    head.write_bytes((arm + "-weights").encode())
    (runner.frozen / (arm + ".json")).write_text(json.dumps({"arm": arm}))
    result = {"status": "complete" if complete else "interrupted", "completed_epochs": 20 if complete else 8,
              "selected_checkpoint": {"head_relative_path": "snapshots/chosen/head.pt", "head_sha256": loop.digest(head),
                                      "epoch": 2, "global_step": 256, "selection_reason": "eligible_natural_train_dev_best"}}
    loop.immutable_json(run / "training-summary.json", result)


def test_both_heads_frozen_before_joint_manifest_exists(tmp_path):
    runner = minimal_runner(tmp_path)
    training_summary(runner, "relative")
    training_summary(runner, "absolute", complete=False)
    with pytest.raises(ValueError, match="both arms must finish"):
        runner.freeze_both_heads()
    assert not (runner.frozen / "selected-heads.json").exists()


def test_selected_heads_recovery_is_idempotent(tmp_path):
    runner = minimal_runner(tmp_path)
    for arm in loop.ARMS:
        training_summary(runner, arm)
    runner.freeze_both_heads()
    first = (runner.frozen / "selected-heads.json").read_bytes()
    runner.freeze_both_heads()
    assert (runner.frozen / "selected-heads.json").read_bytes() == first
    assert (runner.backup / "frozen/selected-heads.json").read_bytes() == first
    (runner.output / "training/relative/snapshots/chosen/head.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint identity differs"):
        runner.freeze_both_heads()


def test_resource_budget_reserves_time_and_honors_stop(monkeypatch):
    runner = loop.Loop.__new__(loop.Loop)
    runner.started = time.monotonic()
    runner.persistence = {"max_process_seconds": 36000, "max_vm_age_seconds": 37800, "evaluation_reserve_seconds": 1800}
    runner.stop_requested = False
    monkeypatch.setattr(loop, "vm_age_seconds", lambda: 36050.)
    assert runner.remaining(reserve=False) == pytest.approx(1750.)
    with pytest.raises(loop.ResumeRequired):
        runner.ready(reserve=True)
    runner.ready(reserve=False)
    runner.stop_requested = True
    with pytest.raises(loop.ResumeRequired):
        runner.ready(reserve=False)


def test_execution_source_change_is_detected(tmp_path):
    runner = loop.Loop.__new__(loop.Loop)
    runner.root = tmp_path
    path = tmp_path / "code.py"
    path.write_text("original\n")
    runner.master_path = tmp_path / "master.json"
    runner.master_path.write_text("{}\n")
    runner.master_sha = loop.digest(runner.master_path)
    files = {"code.py": loop.digest(path)}
    runner.source = {"execution_files": files, "execution_files_sha256": loop.object_sha(files), "master_config_sha256": runner.master_sha}
    runner.verify_source()
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="changed after freeze"):
        runner.verify_source()


def test_closed_navigation_lookup_uses_report_sha_field(tmp_path):
    path = tmp_path / "ledger"
    rows = [{"event": "registered", "access_id": "V0006", "request": {}},
            {"event": "completed", "access_id": "V0006", "report": "/saved/report.json", "report_sha256": "a" * 64}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    registration, outcome = loop.ledger_entry(path, "V0006")
    assert registration["request"] == {}
    assert outcome["report_sha256"] == "a" * 64
