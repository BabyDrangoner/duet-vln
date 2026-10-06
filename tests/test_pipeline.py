"""Real upstream model forward on CPU, followed by our capture/fit/eval interface.

Small random weights exercise the API only. These are not R2R experiments.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

from vln_improve.capture import CacheWriter, DecisionHook
from vln_improve.features import build_features
from vln_improve.head import ResidualActionHead, load_head_checkpoint
from vln_improve.train import train_from_caches


@pytest.fixture
def upstream_navigation():
    transformers = pytest.importorskip("transformers")
    root = Path(__file__).resolve().parents[1] / "third_party/VLN-DUET/map_nav_src"
    if not root.exists():
        pytest.skip("Run scripts/prepare_duet.py first")
    sys.path.insert(0, str(root))
    from models.vilmodel import GlocalTextPathNavCMT
    config = transformers.BertConfig(
        vocab_size=100, hidden_size=24, num_attention_heads=4, intermediate_size=48,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        max_position_embeddings=32, type_vocab_size=2, layer_norm_eps=1e-12,
        output_attentions=True, image_feat_size=8, angle_feat_size=4, obj_feat_size=0,
        num_l_layers=1, num_x_layers=1, num_pano_layers=1, update_lang_bert=False,
        use_lang2visn_attn=False, graph_sprels=True, glocal_fuse=True,
        fix_lang_embedding=False, fix_pano_embedding=False, fix_local_branch=False,
        max_action_steps=100)
    model = GlocalTextPathNavCMT(config).eval().requires_grad_(False)
    inputs = dict(
        txt_embeds=torch.randn(2, 3, 24), txt_masks=torch.ones(2, 3, dtype=torch.bool),
        gmap_img_embeds=torch.randn(2, 4, 24), gmap_step_ids=torch.zeros(2, 4, dtype=torch.long),
        gmap_pos_fts=torch.randn(2, 4, 7), gmap_pair_dists=torch.rand(2, 4, 4),
        gmap_masks=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
        gmap_visited_masks=torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.bool),
        gmap_vpids=[[None, "current", "remote", "near"], [None, "current2", "near2"]],
        vp_img_embeds=torch.randn(2, 3, 24), vp_pos_fts=torch.randn(2, 3, 14),
        vp_masks=torch.ones(2, 3, dtype=torch.bool),
        vp_nav_masks=torch.ones(2, 3, dtype=torch.bool), vp_obj_masks=None,
        vp_cand_vpids=[[None, "near", "current"], [None, "near2", "current2"]])
    with torch.no_grad():
        outputs = model.forward_navigation_per_step(**inputs)
    inputs["no_vp_left"] = [False, False]
    return inputs, outputs


def test_real_upstream_logits_zero_head_and_no_target_access(upstream_navigation):
    inputs, outputs = upstream_navigation
    features, logits, mask = build_features(inputs, outputs)
    head = ResidualActionHead(features.shape[-1])

    class Agent:
        def _teacher_action_r4r(self, *args, **kwargs):
            raise AssertionError("Evaluation must not request labels")

    with torch.no_grad():
        result = DecisionHook(Agent(), head=head)(inputs, outputs, None, None, 0, None)
    torch.testing.assert_close(result["fused_logits"], outputs["fused_logits"], rtol=0, atol=0)
    assert torch.equal(result["fused_logits"].argmax(-1), logits.argmax(-1))
    assert torch.isneginf(result["fused_logits"][~mask]).all()


def test_capture_training_and_reloaded_inference(upstream_navigation, tmp_path):
    inputs, outputs = upstream_navigation
    provenance = dict(dataset="r2r", feature_id="fixture", base_checkpoint_sha256="a" * 64,
                      upstream_commit="fixture", max_action_len=15, feedback="argmax")
    cache = tmp_path / "cache"
    writer = CacheWriter(cache, provenance, shard_size=1)

    class Agent:
        args = SimpleNamespace(max_action_len=15)

        def _teacher_action_r4r(self, *args, **kwargs):
            assert kwargs["imitation_learning"] is False
            return torch.tensor([2, 2])

    obs = [{"instr_id": "a_0", "scan": "a"}, {"instr_id": "b_0", "scan": "b"}]
    hook = DecisionHook(Agent(), writer=writer)
    hook(inputs, outputs, obs, [False, False], 0, [])
    hook(inputs, outputs, obs, [False, False], 0, [])  # repeated tail batch must not duplicate data
    hook(inputs, outputs, obs, [False, False], 14, [])  # forced last action not used as supervision
    manifest = writer.close()
    assert manifest["num_records"] == 2
    assert len(manifest["shards"]) == 2
    output = tmp_path / "head.pt"
    train_from_caches([cache], output, epochs=2, batch_size=2, hidden_dim=8)
    head, metadata = load_head_checkpoint(output, expected_provenance=provenance)
    assert metadata["metrics"]["kind"] == "offline_training_only"
    with torch.no_grad():
        result = DecisionHook(Agent(), head=head)(inputs, outputs, None, None, 0, None)
    _, _, mask = build_features(inputs, outputs)
    assert torch.isfinite(result["fused_logits"][mask]).all()


# Persistent pipeline tests use real cache loading, optimizer steps, source
# snapshots, and checkpoint backup/restore. Only navigation is replaced with
# deterministic episode reports; these fixtures never claim benchmark results.
import copy
import json
import shutil

from vln_improve.checkpoint_store import CheckpointStore
from vln_improve.features import FEATURE_SCHEMA
from vln_improve.pipeline import digest, navigation_metrics, run_pipeline, validate_config


@pytest.fixture
def durable_run(tmp_path):
    project = tmp_path / "project"
    (project / "src/vln_improve").mkdir(parents=True)
    (project / "scripts").mkdir()
    (project / "configs").mkdir()
    (project / "src/vln_improve/fixture.py").write_text("VALUE = 1\n")
    (project / "scripts/run_duet.py").write_text("# fixture; evaluation is injected\n")
    (project / "configs/upstream.json").write_text('{"commit":"fixture"}\n')
    (project / "configs/r2r.json").write_text('{"dataset":"fixture"}\n')
    cache = project / "cache"
    cache.mkdir()
    records = []
    for index in range(5):
        records.append({
            "features": torch.tensor([[1, index], [0, 1]], dtype=torch.float16),
            "base_logits": torch.tensor([0, 1], dtype=torch.float32),
            "valid_mask": torch.tensor([True, True]), "target": index % 2,
            "hard": bool(index % 2), "instr_id": f"train-{index}", "scan_id": "train-scene",
        })
    torch.save(records, cache / "shard.pt")
    (cache / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "feature_schema": FEATURE_SCHEMA, "feature_dim": 2,
        "split": "train_fit", "num_records": len(records), "shards": ["shard.pt"],
        "provenance": {"dataset": "fixture", "feature_id": "fixture-two-features",
                       "base_checkpoint_sha256": "a" * 64, "upstream_commit": "b" * 40,
                       "max_action_len": 15, "feedback": "argmax"},
    }))
    config = {
        "schema_version": 1, "run_id": "test-run", "scope": "research", "cache": ["cache"],
        "local_root": str(tmp_path / "colab-local"), "backup_root": str(tmp_path / "persistent"),
        "checkpoint_every_steps": 2, "checkpoint_every_seconds": 3600,
        "keep_local": 2, "keep_backup": 3, "max_process_seconds": 3600,
        "max_vm_age_seconds": None, "evaluation_reserve_seconds": 1,
        "validation": {"config": "configs/r2r.json", "split": "train_dev",
                       "limit": None, "timeout_seconds": 30},
        "training": {"epochs": 3, "batch_size": 2, "hidden_dim": 4,
                     "lr": 0.01, "seed": 3, "device": "cpu"},
    }
    return project, config


def episode_report(head, *, successes=5, successful_spl=0.4, subset=False, protocol="a" * 64):
    episodes = [{"instr_id": f"dev-{index}", "success": int(index < successes),
                 "spl": successful_spl if index < successes else 0.0} for index in range(10)]
    return {
        "metadata": {"split": "train_dev", "mode": "eval", "subset": subset,
                     "protocol_sha256": protocol, "head_sha256": digest(head), "num_episodes": 10},
        "summary": {"sr": successes * 10.0, "spl": successes * 10.0 * successful_spl},
        "episodes": episodes,
    }


def fixed_evaluator(head, output, options, should_stop):
    assert not should_stop()
    report = episode_report(head, subset=options.get("limit") is not None)
    output.write_text(json.dumps(report))
    return report


def pipeline(config, project, **kwargs):
    return run_pipeline(config, project_root=project, allow_local_backup=True,
                        evaluator=kwargs.pop("evaluator", fixed_evaluator), **kwargs)


def restored_state(config, which="latest"):
    store = CheckpointStore(Path(config["local_root"]) / config["run_id"],
                            Path(config["backup_root"]) / config["run_id"],
                            keep_local=config["keep_local"], keep_backup=config["keep_backup"])
    return store.restore(which)


def tensors_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            tensors_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for left, right in zip(first, second):
            tensors_equal(left, right)
    else:
        assert first == second


def test_pipeline_fresh_vm_restores_cache_and_exact_training(durable_run):
    project, config = durable_run
    reference = copy.deepcopy(config)
    reference["run_id"] = "uninterrupted-reference"
    expected_result = pipeline(reference, project)
    expected, _, _ = restored_state(reference)
    assert expected_result["status"] == "complete"
    paused = pipeline(config, project, stop_after_steps=1)
    assert paused["status"] == "paused" and paused["global_step"] == 1
    assert paused["best"] is None
    shutil.rmtree(config["local_root"])
    shutil.rmtree(project / "cache")
    config["local_root"] += "-new-vm"
    complete = pipeline(config, project, require_resume=True)
    actual, _, _ = restored_state(config)
    assert complete["status"] == "complete" and complete["resumed"]
    assert complete["global_step"] == 9 and complete["epoch"] == 3
    for key in ("head", "optimizer", "history", "epoch_totals", "record_cursor", "global_step"):
        tensors_equal(actual["trainer"][key], expected["trainer"][key])
    local_run = Path(complete["local_run"])
    assert (local_run / "inputs/cache-0000/manifest.json").is_file()
    assert list((Path(complete["backup_run"]) / "reproducibility").glob("source-*.tar.gz"))


def test_pipeline_interrupted_evaluation_retries_same_epoch_before_training(durable_run):
    project, config = durable_run
    evaluations = []

    def interrupted(head, output, options, should_stop):
        saved = torch.load(head, weights_only=True)
        evaluations.append(saved["train_args"]["global_step"])
        raise InterruptedError("simulated session interruption during navigation")

    paused = pipeline(config, project, evaluator=interrupted)
    assert paused["status"] == "paused" and paused["reason"] == "validation_interrupted"
    state, _, _ = restored_state(config)
    assert state["control"]["evaluation_pending"]
    assert state["control"]["last_evaluated_epoch"] == 0
    assert state["trainer"]["global_step"] == 3

    def recording(head, output, options, should_stop):
        evaluations.append(torch.load(head, weights_only=True)["train_args"]["global_step"])
        return fixed_evaluator(head, output, options, should_stop)

    complete = pipeline(config, project, require_resume=True, evaluator=recording)
    assert complete["status"] == "complete"
    assert evaluations == [3, 3, 6, 9]
    state, _, _ = restored_state(config)
    assert not state["control"]["evaluation_pending"]
    assert len(state["control"]["validation_history"]) == 3


def test_pipeline_best_uses_sr_then_spl_and_preserves_best(durable_run):
    project, config = durable_run
    config["keep_local"] = config["keep_backup"] = 1

    def scored(head, output, options, should_stop):
        epoch = torch.load(head, weights_only=True)["train_args"]["completed_epochs"]
        # Epoch 2 wins the SR tie. Epoch 3 has greater SPL but lower SR.
        successes, spl = {1: (5, 0.2), 2: (5, 0.4), 3: (4, 0.9)}[epoch]
        report = episode_report(head, successes=successes, successful_spl=spl)
        output.write_text(json.dumps(report))
        return report

    complete = pipeline(config, project, evaluator=scored)
    assert complete["best"]["epoch"] == 2
    assert complete["best"]["sr"] == 50 and complete["best"]["spl"] == 20
    best_state, best_head, best_manifest = restored_state(config, "best")
    latest_state, _, latest_manifest = restored_state(config, "latest")
    assert best_state["trainer"]["epoch"] == 2 and best_manifest["is_best"]
    assert best_head["train_args"]["global_step"] == 6
    restored_best_file = Path(complete["local_run"]) / "snapshots" / best_manifest["checkpoint_id"] / "head.pt"
    assert digest(restored_best_file) == best_state["control"]["best"]["head_sha256"]
    assert latest_state["trainer"]["epoch"] == 3 and not latest_manifest["is_best"]
    for root in (Path(complete["local_run"]), Path(complete["backup_run"])):
        assert (root / "snapshots" / best_manifest["checkpoint_id"]).exists()
        assert (root / "snapshots" / latest_manifest["checkpoint_id"]).exists()
        assert len(list((root / "snapshots").glob("step-*"))) <= 2


@pytest.mark.parametrize("change", ["source", "validation_file", "validation_option", "training"])
def test_pipeline_rejects_changed_source_or_protocol_on_resume(durable_run, change):
    project, config = durable_run
    pipeline(config, project, stop_after_steps=1)
    if change == "source":
        (project / "src/vln_improve/fixture.py").write_text("VALUE = 2\n")
    elif change == "validation_file":
        (project / "configs/r2r.json").write_text('{"dataset":"different"}\n')
    elif change == "validation_option":
        config["validation"]["timeout_seconds"] += 1
    elif change == "training":
        config["training"]["lr"] = 0.02
    with pytest.raises(ValueError, match="changed"):
        pipeline(config, project, require_resume=True)


def test_research_subset_config_rejected_before_any_backup(durable_run):
    project, config = durable_run
    config["validation"]["limit"] = 8
    with pytest.raises(ValueError, match="subset best selection"):
        pipeline(config, project)
    assert not Path(config["backup_root"]).exists()


def test_smoke_subset_is_explicit_in_best_metadata(durable_run):
    project, config = durable_run
    config["scope"] = "smoke"
    config["validation"]["limit"] = 10
    complete = pipeline(config, project)
    assert complete["best"]["scope"] == "smoke"
    assert complete["best"]["subset"] is True


@pytest.mark.parametrize("mutation,match", [
    ("subset", "subset"), ("nan_episode", "per-episode"), ("nan_summary", "summary"),
    ("protocol_changed", "protocol changed"), ("protocol_missing", "fingerprint"),
    ("wrong_head", "different checkpoint"), ("wrong_split", "split"),
    ("incomplete", "incomplete"), ("duplicate", "duplicate"),
    ("summary_mismatch", "summary"),
])
def test_navigation_metrics_rejects_invalid_evidence(tmp_path, mutation, match):
    head = tmp_path / "head.pt"
    head.write_bytes(b"test checkpoint bytes")
    report = episode_report(head)
    if mutation == "subset":
        report["metadata"]["subset"] = True
    elif mutation == "nan_episode":
        report["episodes"][0]["spl"] = float("nan")
    elif mutation == "nan_summary":
        report["summary"]["sr"] = float("nan")
    elif mutation == "protocol_changed":
        report["metadata"]["protocol_sha256"] = "b" * 64
    elif mutation == "protocol_missing":
        del report["metadata"]["protocol_sha256"]
    elif mutation == "wrong_head":
        report["metadata"]["head_sha256"] = "b" * 64
    elif mutation == "wrong_split":
        report["metadata"]["split"] = "val_unseen"
    elif mutation == "incomplete":
        report["episodes"].pop()
    elif mutation == "duplicate":
        report["episodes"][1]["instr_id"] = report["episodes"][0]["instr_id"]
    elif mutation == "summary_mismatch":
        report["summary"]["spl"] += 1
    with pytest.raises(ValueError, match=match):
        navigation_metrics(report, head, scope="research", expected_protocol="a" * 64)


def test_invalid_evaluation_never_becomes_best_and_remains_pending(durable_run):
    project, config = durable_run

    def wrong_head(head, output, options, should_stop):
        report = episode_report(head)
        report["metadata"]["head_sha256"] = "c" * 64
        return report

    with pytest.raises(ValueError, match="different checkpoint"):
        pipeline(config, project, evaluator=wrong_head)
    state, _, _ = restored_state(config)
    assert state["control"]["evaluation_pending"]
    assert state["control"]["best"] is None
    assert not (Path(config["backup_root"]) / config["run_id"] / "best.json").exists()
    complete = pipeline(config, project, require_resume=True)
    assert complete["status"] == "complete"


def test_protocol_change_during_run_cannot_replace_existing_best(durable_run):
    project, config = durable_run

    def changing(head, output, options, should_stop):
        epoch = torch.load(head, weights_only=True)["train_args"]["completed_epochs"]
        return episode_report(head, successes=5 if epoch == 1 else 10,
                              protocol=("a" if epoch == 1 else "b") * 64)

    with pytest.raises(ValueError, match="protocol changed"):
        pipeline(config, project, evaluator=changing)
    state, _, _ = restored_state(config)
    assert state["control"]["best"]["epoch"] == 1
    assert state["control"]["last_evaluated_epoch"] == 1
    assert state["control"]["evaluation_pending"]


def test_require_resume_refuses_silent_restart(durable_run):
    project, config = durable_run
    with pytest.raises(FileNotFoundError):
        pipeline(config, project, require_resume=True)
