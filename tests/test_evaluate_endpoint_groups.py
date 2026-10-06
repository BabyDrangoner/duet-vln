import copy
import json
import multiprocessing
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_endpoint_groups as evaluator
from evaluate_endpoint_groups import validate_group_head_metadata, validate_registration
from test_study_ledger import study, request
from vln_improve.protocol import file_sha256, object_sha256
from vln_improve.study_ledger import StudyLedger
from vln_improve.endpoint_group_training import training_code_identity

ROOT = Path(__file__).resolve().parents[1]


def final_meta(arm="C3", seed=0):
    path = ROOT / f"configs/endpoint_group_{arm}.json"
    experiment = json.loads(path.read_text())
    sha = file_sha256(path)
    meta = {"train_config": {"epochs": 20, "batch_groups": 8, "arm": arm, "seed": seed,
                            "optimizer": "AdamW", "lr": .001, "weight_decay": .0001,
                            "experiment_sha256": sha, "pair_weight": .1 if arm == "M" else 0.,
                            "natural_weight": .5, "augmentation_weight": .5, "max_states": 15,
                            "feature_dim": 1536, "hidden_dim": 128, "activation": "relu", "monitor_every": 5,
                            "primary_checkpoint": "fixed_final_epoch",
                            "engineering_best_checkpoint": experiment["engineering_best_monitor"]},
            "epoch": 20, "global_step": 1280, "pending_dev": False,
            "selection_purpose": "fixed_final_epoch",
            "training_code_identity": training_code_identity()}
    return meta, experiment, sha


@pytest.mark.parametrize("arm", ["C1", "C2", "C3", "M"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_accepts_registered_final_head_for_all_arms_seeds(arm, seed):
    meta, experiment, sha = final_meta(arm, seed)
    validate_group_head_metadata(meta, experiment, seed, sha)


@pytest.mark.parametrize("field,value", [("epoch", 15), ("global_step", 1279),
                                         ("pending_dev", True), ("selection_purpose", "best_dev")])
def test_rejects_nonfinal_or_pending_head(field, value):
    meta, experiment, sha = final_meta()
    meta[field] = value
    with pytest.raises(ValueError, match="final checkpoint"):
        validate_group_head_metadata(meta, experiment, 0, sha)


def test_rejects_training_seed_config_and_source_changes():
    meta, experiment, sha = final_meta()
    with pytest.raises(ValueError): validate_group_head_metadata(meta, experiment, 1, sha)
    with pytest.raises(ValueError): validate_group_head_metadata(meta, experiment, 0, "a"*64)
    altered = copy.deepcopy(meta)
    altered["training_code_identity"]["src/vln_improve/endpoint_probe.py"] = "b"*64
    with pytest.raises(ValueError, match="implementation identity changed"):
        validate_group_head_metadata(altered, experiment, 0, sha)
    altered = copy.deepcopy(meta); altered["train_config"]["arm"] = "M"
    with pytest.raises(ValueError): validate_group_head_metadata(altered, experiment, 0, sha)


@pytest.mark.parametrize("field,value", [("pair_weight", 0.), ("natural_weight", 1.),
                                         ("augmentation_weight", 0.), ("max_states", 30),
                                         ("feature_dim", 1537), ("hidden_dim", 256),
                                         ("activation", "gelu"), ("monitor_every", 1)])
def test_rejects_mismatched_objective_or_architecture_in_head(field, value):
    meta, experiment, sha = final_meta("M")
    meta["train_config"][field] = value
    with pytest.raises(ValueError): validate_group_head_metadata(meta, experiment, 0, sha)


def test_rejects_changed_experiment_weights_and_partial_code_identity():
    meta, experiment, sha = final_meta("M")
    changed = copy.deepcopy(experiment); changed["natural_weight"] = 1.
    with pytest.raises(ValueError): validate_group_head_metadata(meta, changed, 0, sha)
    meta["training_code_identity"].pop("scripts/train_endpoint_groups.py")
    with pytest.raises(ValueError, match="incomplete"): validate_group_head_metadata(meta, experiment, 0, sha)


def test_requires_exact_pending_full_validation_access(tmp_path):
    config, head, study_path = (tmp_path / n for n in ("experiment.json", "head.pt", "study.json"))
    config.write_text("{}"); head.write_bytes(b"frozen final head"); study_path.write_text(json.dumps(study()))
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    args = SimpleNamespace(split="val_unseen", head=head, limit=None, access_id="P1",
                           ledger=ledger.path, study=study_path, experiment=config, seed=0, category="pilot",
                           execution_backup_root=tmp_path / "cloud-executions")
    ledger.register(**request(config_sha256=file_sha256(config), checkpoint_sha256=file_sha256(head)))
    validate_registration(args, "c"*64)
    for field, value in (("category", "confirmatory"), ("seed", 1), ("limit", 10)):
        changed = copy.copy(args); setattr(changed, field, value)
        with pytest.raises(ValueError): validate_registration(changed, "c"*64)
    ledger.finish("P1", status="failed", metrics={}, resources={"seconds": None}, decision="new access needed", error="test")
    with pytest.raises(ValueError, match="cannot be rerun"): validate_registration(args, "c"*64)


def test_training_development_does_not_consume_validation_registration():
    args = SimpleNamespace(split="train_dev", access_id=None, ledger=None)
    assert validate_registration(args, "c"*64) is None
    args.access_id = "P1"
    with pytest.raises(ValueError): validate_registration(args, "c"*64)


def registered_args(tmp_path):
    code_sha = object_sha256({name: file_sha256(ROOT / name) for name in evaluator.CODE_FILES})
    config, head, study_path = (tmp_path / name for name in ("experiment.json", "head.pt", "study.json"))
    config.write_text("{}"); head.write_bytes(b"frozen final head"); study_path.write_text(json.dumps(study()))
    ledger = StudyLedger(tmp_path / "ledger.jsonl", study())
    registration = ledger.register(**request(config_sha256=file_sha256(config), checkpoint_sha256=file_sha256(head),
                                              code_sha256=code_sha))
    args = SimpleNamespace(split="val_unseen", head=head, limit=None, access_id="P1",
        ledger=ledger.path, study=study_path, experiment=config, seed=0, category="pilot",
        execution_backup_root=tmp_path / "cloud-executions", config=ROOT / "configs/r2r.json",
        output=tmp_path / "result.json", backup=tmp_path / "cloud-results",
        baseline_report=tmp_path / "official-baseline.json")
    args.baseline_report.write_text("{invalid baseline JSON}")
    return args, ledger, registration, code_sha


@pytest.fixture
def local_drive(monkeypatch):
    import vln_improve.pipeline as pipeline
    monkeypatch.setattr(pipeline, "validate_backup_root", lambda path: "local-filesystem-test")


def cli_args(args):
    names = ("head", "config", "experiment", "split", "baseline_report", "output", "backup", "seed",
             "study", "ledger", "access_id", "execution_backup_root")
    return [item for name in names if getattr(args, name) is not None
            for item in ("--" + name.replace("_", "-"), str(getattr(args, name)))]


def claim_files(args, registration):
    local = evaluator.execution_claim_path(args.ledger, args.access_id)
    cloud = evaluator.cloud_execution_claim_path(args.execution_backup_root, registration)
    outcome = lambda path: path.with_name(path.name.removesuffix(".claim.json") + ".outcome.json")
    return local, cloud, outcome(local), outcome(cloud)


@pytest.mark.parametrize("invalid", ["missing_access", "missing_cloud_root", "wrong_seed"])
def test_main_gates_before_any_baseline_read_or_hash(tmp_path, monkeypatch, invalid):
    args, _, _, _ = registered_args(tmp_path)
    if invalid == "missing_access": args.access_id = "not-registered"
    if invalid == "missing_cloud_root": args.execution_backup_root = None
    if invalid == "wrong_seed": args.seed = 1
    original_open = Path.open

    def guarded_open(path, *positional, **kwargs):
        if path == args.baseline_report:
            pytest.fail("official baseline was opened before registration was accepted")
        return original_open(path, *positional, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises((ValueError, KeyError)):
        evaluator.main(cli_args(args))


def test_main_persists_cloud_claim_and_ledger_before_first_baseline_read(tmp_path, monkeypatch, local_drive):
    args, ledger, registration, _ = registered_args(tmp_path)
    ledger_before = ledger.path.read_bytes()
    local, cloud, local_outcome, cloud_outcome = claim_files(args, registration)
    original_read = Path.read_text
    reads = []

    def guarded_read(path, *positional, **kwargs):
        if path == args.baseline_report:
            reads.append(path)
            assert local.read_bytes() == cloud.read_bytes()
            snapshot = cloud.with_name(cloud.name.removesuffix(".claim.json") + ".ledger.jsonl")
            assert snapshot.read_bytes() == ledger_before
        return original_read(path, *positional, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    with pytest.raises(json.JSONDecodeError):
        evaluator.main(cli_args(args))
    assert reads == [args.baseline_report]
    assert json.loads(local_outcome.read_text())["status"] == "failed"
    assert local_outcome.read_bytes() == cloud_outcome.read_bytes()
    assert ledger.path.read_bytes() == ledger_before  # External finish still owns the ledger.


def test_completed_claim_blocks_other_output_and_external_finish_remains_idempotent(tmp_path, local_drive):
    args, ledger, registration, code_sha = registered_args(tmp_path)
    ledger_before = ledger.path.read_bytes()
    with evaluator.validation_execution(args, registration, code_sha) as claim:
        args.output.write_text('{"sr": 70}')
    local, cloud, local_outcome, cloud_outcome = claim_files(args, registration)
    assert claim["sha256"] == file_sha256(local) == file_sha256(cloud)
    result = json.loads(local_outcome.read_text())
    assert result["status"] == "completed" and result["report_sha256"] == file_sha256(args.output)
    assert local_outcome.read_bytes() == cloud_outcome.read_bytes()
    assert ledger.path.read_bytes() == ledger_before
    changed = copy.copy(args); changed.output = tmp_path / "another-output.json"
    with pytest.raises(ValueError, match="execution claim"):
        with evaluator.validation_execution(changed, registration, code_sha):
            pytest.fail("same access reran under a different output")
    closeout = dict(status="completed", metrics={"sr": 70}, resources={"seconds": 1},
                    decision="recorded fixed evaluation", report_path=args.output)
    assert ledger.finish(args.access_id, **closeout) == ledger.finish(args.access_id, **closeout)
    assert len(ledger.path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="cannot be rerun"):
        validate_registration(changed, code_sha)


@pytest.mark.parametrize("error", [RuntimeError("failed rollout"), KeyboardInterrupt("interrupted rollout")])
def test_failed_or_interrupted_execution_keeps_both_claims(tmp_path, local_drive, error):
    args, ledger, registration, code_sha = registered_args(tmp_path)
    with pytest.raises(type(error), match=str(error)):
        with evaluator.validation_execution(args, registration, code_sha):
            raise error
    local, cloud, local_outcome, cloud_outcome = claim_files(args, registration)
    assert local.read_bytes() == cloud.read_bytes()
    assert local_outcome.read_bytes() == cloud_outcome.read_bytes()
    outcome = json.loads(local_outcome.read_text())
    assert outcome["status"] == "failed" and outcome["error_type"] == type(error).__name__
    assert ledger.lookup(args.access_id)["outcome"] is None
    with pytest.raises(ValueError, match="execution claim"):
        with evaluator.validation_execution(args, registration, code_sha):
            pytest.fail("failed access reran")
    ledger.finish(args.access_id, status="failed", metrics={}, resources={"seconds": None},
                  decision="new access required", error=str(error))
    assert len(ledger.path.read_text().splitlines()) == 2


def _claim_worker(args, registration, code_sha, barrier=None, queue=None, crash=False):
    import vln_improve.pipeline as pipeline
    pipeline.validate_backup_root = lambda path: "local-filesystem-test"
    if barrier is not None:
        barrier.wait(timeout=10)
    try:
        with evaluator.validation_execution(args, registration, code_sha):
            if crash:
                os._exit(73)
            args.output.write_text('{"sr": 70}')
        queue.put("completed")
    except ValueError as error:
        queue.put(str(error))


@pytest.mark.parametrize("copied_ledger", [False, True])
def test_concurrent_different_outputs_get_one_execution_even_with_ledger_copy(tmp_path, copied_ledger):
    args, _, registration, code_sha = registered_args(tmp_path)
    other = copy.copy(args); other.output = tmp_path / "other-output.json"
    if copied_ledger:
        other.ledger = tmp_path / "ledger-copy.jsonl"
        shutil.copyfile(args.ledger, other.ledger)
    context = multiprocessing.get_context("spawn")
    queue, barrier = context.Queue(), context.Barrier(2)
    workers = [context.Process(target=_claim_worker, args=(item, registration, code_sha, barrier, queue))
               for item in (args, other)]
    for worker in workers: worker.start()
    for worker in workers:
        worker.join(timeout=20)
        if worker.is_alive(): worker.terminate(); worker.join()
        assert worker.exitcode == 0
    results = [queue.get(timeout=3), queue.get(timeout=3)]
    assert results.count("completed") == 1
    assert sum("execution claim" in result for result in results) == 1


def test_hard_crash_and_lost_vm_files_cannot_reuse_cloud_access(tmp_path, local_drive):
    args, _, registration, code_sha = registered_args(tmp_path)
    context = multiprocessing.get_context("spawn")
    worker = context.Process(target=_claim_worker, args=(args, registration, code_sha), kwargs={"crash": True})
    worker.start(); worker.join(timeout=20)
    if worker.is_alive(): worker.terminate(); worker.join()
    assert worker.exitcode == 73
    local, cloud, local_outcome, cloud_outcome = claim_files(args, registration)
    assert local.read_bytes() == cloud.read_bytes()
    assert not local_outcome.exists() and not cloud_outcome.exists()
    shutil.rmtree(local.parent)  # Simulate VM loss; only the cloud claim survives.
    other = copy.copy(args); other.ledger = tmp_path / "restored-ledger.jsonl"
    other.output = tmp_path / "restored-output.json"
    shutil.copyfile(args.ledger, other.ledger)
    with pytest.raises(ValueError, match="cloud execution claim"):
        with evaluator.validation_execution(other, registration, code_sha):
            pytest.fail("a crashed cloud access was reused after VM replacement")


@pytest.mark.parametrize("failure", ["claim_write", "claim_readback", "ledger_snapshot", "ledger_readback"])
def test_cloud_persistence_failure_prevents_baseline_access(tmp_path, monkeypatch, local_drive, failure):
    args, _, registration, _ = registered_args(tmp_path)
    local, cloud, local_outcome, _ = claim_files(args, registration)
    original_write = evaluator._exclusive_bytes

    def broken_write(path, raw):
        if (failure == "claim_write" and path == cloud
                or failure == "ledger_snapshot" and path.parent == cloud.parent and path.name.endswith(".ledger.jsonl")):
            raise OSError("cloud write failed")
        original_write(path, raw)
        if (failure == "claim_readback" and path == cloud
                or failure == "ledger_readback" and path.parent == cloud.parent and path.name.endswith(".ledger.jsonl")):
            path.write_bytes(b"corrupted cloud bytes")

    monkeypatch.setattr(evaluator, "_exclusive_bytes", broken_write)
    original_open = Path.open

    def guarded_open(path, *positional, **kwargs):
        if path == args.baseline_report:
            pytest.fail("baseline opened without a verified cloud execution claim and ledger snapshot")
        return original_open(path, *positional, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises((OSError, ValueError), match="cloud write failed|read-back failed"):
        evaluator.main(cli_args(args))
    assert local.exists()
    assert json.loads(local_outcome.read_text())["status"] == "failed"


def test_partial_claim_from_interrupted_write_cannot_be_replaced(tmp_path, local_drive):
    args, _, registration, code_sha = registered_args(tmp_path)
    local, cloud, _, _ = claim_files(args, registration)
    local.parent.mkdir()
    local.write_bytes(b'{"status":')
    with pytest.raises(ValueError, match="execution claim"):
        with evaluator.validation_execution(args, registration, code_sha):
            pytest.fail("partial execution claim was replaced")
    assert local.read_bytes() == b'{"status":'
    assert not cloud.exists()


def test_training_development_does_not_create_execution_claims(tmp_path):
    args = SimpleNamespace(split="train_dev")
    with evaluator.validation_execution(args, None, "c" * 64) as claim:
        assert claim is None
    assert list(tmp_path.iterdir()) == []
