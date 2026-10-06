import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_endpoint_group_results import (
    ARMS, METRICS, analyze_reports, load_report, main, validate_report,
)
from vln_improve.endpoint_group_training import _dev_record, final_dev_report
from vln_improve.endpoint_probe import EndpointProbe, FEATURE_SCHEMA, HEAD_SCHEMA
from vln_improve.protocol import file_sha256, object_sha256


def rehash(report):
    report["content_sha256"] = object_sha256({k: v for k, v in report.items() if k != "content_sha256"})


def rebuild_monitor(report):
    rows = [{"pair_id": r["pair_id"], "scan": r["scan_id"],
             **{k: r[k] for k in ("natural_bce", "c2_bce", "paired_bce", "ranking")},
             "correct_orders": r["both_instructions_correct_by_order"], "correct_both": r["both_orders_correct"]}
            for r in report["groups"]]
    report["monitor"] = {k: v for k, v in _dev_record(rows, 20, 1280).items() if k != "per_group"}
    rehash(report)


def make_reports():
    reports = {}
    for index, arm in enumerate(ARMS):
        rows = [{"pair_id": object_sha256(["pair", i]), "scan": "scene-A" if i < 96 else "scene-B",
                 "natural_bce": 5., "c2_bce": 5., "paired_bce": 5., "ranking": 5.,
                 "correct_orders": [False, False], "correct_both": False} for i in range(128)]
        checkpoint = {"checkpoint_id": f"step-000000001280-{index:032x}", "epoch": 20, "global_step": 1280,
                      "head_sha256": str(index + 1) * 64, "selection_purpose": "fixed_final_epoch"}
        checkpoint["head_relative_path"] = f"snapshots/{checkpoint['checkpoint_id']}/head.pt"
        result = {"status": "complete", "pending_dev": False, "final_checkpoint": checkpoint,
                  "completed_epochs": 20, "global_step": 1280, "arm": arm, "seed": 0,
                  "train_data_sha256": "a" * 64, "dev_data_sha256": "b" * 64,
                  "dev_history": [_dev_record(rows, 20, 1280)]}
        reports[arm] = final_dev_report(result)
    return reports


def sources():
    return {a: {"report_sha256": str(i + 1) * 64} for i, a in enumerate(ARMS)}


def write_inputs(tmp_path):
    reports = make_reports()
    paths = {}
    torch.manual_seed(12)
    head = EndpointProbe()
    for arm, report in reports.items():
        folder = tmp_path / arm
        head_path = folder / report["checkpoint"]["head_relative_path"]
        head_path.parent.mkdir(parents=True)
        payload = {"schema": HEAD_SCHEMA, "head_config": head.config, "feature_schema": FEATURE_SCHEMA,
            "state_dict": copy.deepcopy(head.state_dict()), "epoch": 20, "global_step": 1280,
            "pending_dev": False, "selection_purpose": "fixed_final_epoch",
            "initial_head_sha256": "7" * 64, "training_code_identity": {"frozen_training.py": "8" * 64},
            "data_identity": {"train": "a" * 64, "dev": "b" * 64},
            "common_identity": {"common_provenance": {"base_checkpoint_sha256": "9" * 64}},
            "train_config": {"arm": arm, "seed": 0, "epochs": 20, "batch_groups": 8, "monitor_every": 5,
                "optimizer": "AdamW", "lr": .001, "weight_decay": .0001,
                "pair_weight": .1 if arm == "M" else 0., "natural_weight": .5, "augmentation_weight": .5,
                "max_states": 15, "feature_dim": 1536, "hidden_dim": 128, "activation": "relu",
                "primary_checkpoint": "fixed_final_epoch", "experiment_sha256": "6" * 64,
                "engineering_best_checkpoint": "equal_mean_of_natural_c2_paired_dev_episode_bce"}}
        torch.save(payload, head_path)
        report["checkpoint"]["head_sha256"] = file_sha256(head_path)
        rehash(report)
        path = folder / "dev-final.json"
        path.write_text(json.dumps(report))
        paths[arm] = path
    return paths


def test_scene_bootstrap_keeps_pairs_and_uses_resampled_pair_denominator():
    reports = make_reports()
    for row in reports["C2"]["groups"]:
        for key in ("natural_bce", "c2_bce", "paired_bce", "ranking"):
            row[key] += 1. if row["scan_id"] == "scene-A" else -1.
    rebuild_monitor(reports["C2"])
    result = analyze_reports(reports, sources())
    contrast = result["comparisons"]["C2-C1"]
    # 96*1 + 32*(-1) / 128 = .5; the unweighted scene mean would be zero.
    for metric in METRICS[:4]:
        assert contrast["metrics"][metric]["effect_after_minus_before"] == .5
        assert contrast["metrics"][metric]["ci95"] == [-1., 1.]
    assert contrast["by_scene"]["scene-A"]["pairs"] == 96
    assert contrast["by_scene"]["scene-B"]["pairs"] == 32
    assert len(contrast["per_pair"]) == 128
    assert list(result["comparisons"]) == ["C2-C1", "C3-C2", "M-C3"]
    assert result["bootstrap"]["resamples"] == 10000 and result["bootstrap"]["seed"] == 0
    draws = np.random.default_rng(0).integers(2, size=(10000, 2))
    assert result["bootstrap"]["draws_sha256"] == object_sha256(draws.tolist())
    assert result == analyze_reports(reports, sources())


def test_order_events_and_whole_pair_correctness_have_distinct_rescues_and_harms():
    reports = make_reports()
    # Pair 0 rescues order A but harms B: average order effect=0, both-orders remain false.
    # Pair 1 rescues both orders: average order effect=1, whole-pair effect=1.
    before, after = reports["C3"]["groups"], reports["M"]["groups"]
    before[0]["both_instructions_correct_by_order"] = [False, True]
    after[0]["both_instructions_correct_by_order"] = [True, False]
    after[1]["both_instructions_correct_by_order"] = [True, True]
    after[1]["both_orders_correct"] = True
    rebuild_monitor(reports["C3"]); rebuild_monitor(reports["M"])
    contrast = analyze_reports(reports, sources())["comparisons"]["M-C3"]
    events = contrast["transitions"]
    assert (events["order_event_rescues"], events["order_event_harms"]) == (3, 1)
    assert events["order_events_total"] == 256
    assert events["both_orders"]["rescues"] == 1 and events["both_orders"]["harms"] == 0
    assert contrast["metrics"][METRICS[4]]["effect_after_minus_before"] == 1 / 128
    assert contrast["metrics"][METRICS[5]]["effect_after_minus_before"] == 1 / 128
    assert contrast["by_scene"]["scene-A"]["transitions"] == events | {"order_events_total": 192,
        "by_order": [{"rescues": 2, "harms": 0, "unchanged_correct": 0, "unchanged_incorrect": 94},
                     {"rescues": 1, "harms": 1, "unchanged_correct": 0, "unchanged_incorrect": 94}],
        "both_orders": {"rescues": 1, "harms": 0, "unchanged_correct": 0, "unchanged_incorrect": 95}}


@pytest.mark.parametrize("damage", ["hash", "epoch", "step", "purpose", "split", "usage", "count", "duplicate",
                                    "bool", "both", "negative", "aggregate", "scene_aggregate", "head_sha", "data_sha", "traversal"])
def test_rejects_invalid_report_even_when_internal_hash_is_resealed(damage):
    report = make_reports()["C1"]
    if damage == "hash": report["groups"][0]["natural_bce"] += 1
    elif damage == "epoch": report["checkpoint"]["epoch"] = 15
    elif damage == "step": report["checkpoint"]["global_step"] = 1279
    elif damage == "purpose": report["selection_purpose"] = "best_dev"
    elif damage == "split": report["split"] = "val_unseen"
    elif damage == "usage": report["usage"] = "training"
    elif damage == "count": report["groups"].pop()
    elif damage == "duplicate": report["groups"][-1] = copy.deepcopy(report["groups"][0])
    elif damage == "bool": report["groups"][0]["both_instructions_correct_by_order"][0] = 1
    elif damage == "both": report["groups"][0]["both_orders_correct"] = True
    elif damage == "negative": report["groups"][0]["ranking"] = -.1
    elif damage == "aggregate": report["monitor"]["natural_bce"] += 1
    elif damage == "scene_aggregate": report["monitor"]["by_scene"]["scene-A"]["groups"] += 1
    elif damage == "head_sha": report["checkpoint"]["head_sha256"] = "invalid"
    elif damage == "data_sha": report["dev_data_sha256"] = report["train_data_sha256"]
    elif damage == "traversal": report["checkpoint"]["head_relative_path"] = "../head.pt"
    if damage != "hash": rehash(report)
    with pytest.raises(ValueError): validate_report(report, "C1")


@pytest.mark.parametrize("damage", ["seed", "train", "dev", "pair_order", "scan", "arm", "missing_arm"])
def test_rejects_unpaired_or_selected_arm_inputs(damage):
    reports = make_reports()
    altered = reports["C2"]
    if damage == "seed": altered["training_seed"] = 1
    elif damage == "train": altered["train_data_sha256"] = "c" * 64
    elif damage == "dev": altered["dev_data_sha256"] = "c" * 64
    elif damage == "pair_order": altered["groups"][0], altered["groups"][1] = altered["groups"][1], altered["groups"][0]
    elif damage == "scan": altered["groups"][0]["scan_id"] = "scene-B"
    elif damage == "arm": altered["arm"] = "C3"
    elif damage == "missing_arm": del reports["C1"]
    rebuild_monitor(altered)
    with pytest.raises(ValueError): analyze_reports(reports, sources())


def test_cli_validates_actual_heads_and_emits_all_fixed_comparisons(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CPU only")))
    paths = write_inputs(tmp_path)
    output = tmp_path / "analysis.json"
    args = ["--output", str(output)]
    for arm, flag in zip(ARMS, ("c1", "c2", "c3", "method")):
        args += ["--" + flag, str(paths[arm])]
    result = main(args)
    assert json.loads(output.read_text()) == result
    assert all(s["head_bytes_verified"] for s in result["source_reports"].values())
    assert result["content_sha256"] == object_sha256({k: v for k, v in result.items() if k != "content_sha256"})
    for comparison in result["comparisons"].values():
        for metric in comparison["metrics"].values():
            assert metric["effect_after_minus_before"] == 0 and metric["ci95"] == [0., 0.]
    with pytest.raises(ValueError, match="overwrite"): main(args)


@pytest.mark.parametrize("damage", ["missing", "bytes", "metadata_seed", "metadata_arm", "metadata_data", "metadata_pair_weight", "summary"])
def test_head_file_and_metadata_bindings_are_required(tmp_path, damage):
    paths = write_inputs(tmp_path)
    path = paths["M"]
    report = json.loads(path.read_text())
    head_path = path.parent / report["checkpoint"]["head_relative_path"]
    if damage == "missing": head_path.unlink()
    elif damage == "bytes": head_path.write_bytes(b"corrupt")
    elif damage == "summary": (path.parent / "training-summary.json").write_text("{}")
    else:
        payload = torch.load(head_path, map_location="cpu", weights_only=True)
        if damage == "metadata_seed": payload["train_config"]["seed"] = 1
        elif damage == "metadata_arm": payload["train_config"]["arm"] = "C3"
        elif damage == "metadata_data": payload["data_identity"]["dev"] = "c" * 64
        elif damage == "metadata_pair_weight": payload["train_config"]["pair_weight"] = 0.
        torch.save(payload, head_path)
        report["checkpoint"]["head_sha256"] = file_sha256(head_path)
        rehash(report); path.write_text(json.dumps(report))
    with pytest.raises(ValueError): load_report(path, "M")


def test_nonfinite_json_and_duplicate_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"schema": 1, "schema": 2}', '{"value": NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError): load_report(path, "C1")


@pytest.mark.parametrize("identity", ["data_identity", "common_identity", "training_code_identity", "initial_head_sha256"])
def test_cli_rejects_different_shared_sources_or_initialization(tmp_path, identity):
    paths = write_inputs(tmp_path)
    report = json.loads(paths["C2"].read_text())
    head_path = paths["C2"].parent / report["checkpoint"]["head_relative_path"]
    payload = torch.load(head_path, map_location="cpu", weights_only=True)
    if identity == "initial_head_sha256": payload[identity] = "5" * 64
    else: payload[identity]["changed_source"] = "5" * 64
    torch.save(payload, head_path)
    report["checkpoint"]["head_sha256"] = file_sha256(head_path)
    rehash(report); paths["C2"].write_text(json.dumps(report))
    output = tmp_path / "analysis.json"
    args = ["--output", str(output)]
    for arm, flag in zip(ARMS, ("c1", "c2", "c3", "method")):
        args += ["--" + flag, str(paths[arm])]
    with pytest.raises(ValueError, match="shared source/head initialization"):
        main(args)
    assert not output.exists()
