import copy
import json
from pathlib import Path
import shutil
import sys

import networkx as nx
import pytest
import torch

from vln_improve.checkpoint_store import CheckpointError, CheckpointStore
from vln_improve.endpoint_controls import ControlStore, NATURAL_MODE, SCHEMA as CONTROL_SCHEMA, annotate_route_lengths
from vln_improve.endpoint_group_training import (
    BEST_PURPOSE, EndpointGroupTrainer, assemble_endpoint_groups, final_dev_report, load_endpoint_group_cache,
    train_endpoint_groups, validate_endpoint_group_splits,
)
from vln_improve.endpoint_group_objectives import prepare_group_batch, score_group_batch, per_group_metrics
from vln_improve.endpoint_pairs import ORDERS, SLOTS, PairStore
from vln_improve.endpoint_probe import load_endpoint_head
from vln_improve.protocol import file_sha256, object_sha256
from test_endpoint_pair_training import _identity, _pair, _payload

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _control_run(paired, slot, context):
    index = SLOTS.index(slot)
    order = ORDERS[index]
    run = copy.deepcopy(paired["rollouts"][order][slot])
    pair = paired["pair"]
    goal, end = pair["goal_vpids"][index], "q" + slot
    vps = [pair["start"], goal] + ([end] if context == "overshoot" else [])
    run["features"] = run["features"][:len(vps)].clone()
    run["features"][:, 0] = torch.tensor([-1., 1.] + ([-1.] if len(vps) == 3 else []))
    run["states"] = run["states"][:len(vps)]
    prefix = [[vps[0]]]
    for t, state in enumerate(run["states"]):
        if t: prefix.append([vps[t]])
        state.update(viewpoint=vps[t], trajectory_prefix=copy.deepcopy(prefix), prefix_length_m=float(4*t),
                     position=[float(4*t), 0., 0.])
        if context == "natural":
            action = vps[t+1] if t+1 < len(vps) else None
            state.update(baseline_argmax_index=state["gmap_vpids"].index(action), baseline_argmax_vpid=action,
                         termination={"argmax_stop": action is None, "step_limit": False, "no_vp_left": False},
                         executed_action=action)
    run.update(trajectory=copy.deepcopy(prefix), actual_length_m=float(4*(len(vps)-1)),
               context_id=f"{slot}:{context}", natural_stop_probability=torch.full((len(vps),), .5),
               labels={"goal_vpid": goal, "distance_to_goal": torch.tensor([4., 0.] + ([4.] if len(vps)==3 else []), dtype=torch.float64),
                       "within_success_radius": torch.tensor([False, True] + ([False] if len(vps)==3 else []))})
    if context == "natural":
        run["mode"] = NATURAL_MODE; run["fallback_segment"] = []; run.pop("forced_actions")
    else:
        run["forced_actions"] = vps[1:] + [None]
    graph = nx.Graph()
    for a, b in zip(vps, vps[1:]): graph.add_edge(a, b, weight=4.)
    annotate_route_lengths(run, graph, forced=context != "natural")
    return run


def _sources(tmp_path, split, n, start):
    pairs = sorted([_pair(split+"-scene", start+i) for i in range(n)], key=lambda p: p["selection_hash"])
    identity = _identity(split, pairs)
    pp, cp = tmp_path / (split+"-pairs"), tmp_path / (split+"-controls")
    pair_store = PairStore(pp, tmp_path / (split+"-pairs-backup"), identity, lambda: None)
    entries, payloads = [], []
    for pair in pairs:
        payload = _payload(pair, identity)
        for order in ORDERS:
            for index, slot in enumerate(SLOTS):
                run = payload["rollouts"][order][slot]
                run["features"][:, 0] = torch.tensor([1. if s["viewpoint"] == pair["goal_vpids"][index] else -1. for s in run["states"]])
                run["features"][:, 1:] *= .01
        pair_store.commit(payload)
        entries.append({"pair": pair, "controls": {slot: {
            "instr_id": pair["instr_ids"][i], "goal": pair["goal_vpids"][i],
            "positive_history": {"observed_vpids": [pair["start"], pair["goal_vpids"][i]]},
            "overshoot_history": {"observed_vpids": [pair["start"], pair["goal_vpids"][i], "q"+slot]}}
            for i, slot in enumerate(SLOTS)}})
        payloads.append(payload)
    pair_store.seal({"wall_seconds": 1.})
    control_identity = {"schema": CONTROL_SCHEMA, "split": split, "usage": identity["usage"],
        "selection": entries, "selection_sha256": object_sha256(entries),
        "controls_report_sha256": "a"*64, "collection_config_sha256": "b"*64, "runtime_config_sha256": "1"*64,
        "code_files": {"control.py": "c"*64}, "common_provenance": identity["common_provenance"],
        "feature_schema": identity["feature_schema"], "feature_dim": 1536}
    controls = ControlStore(cp, tmp_path / (split+"-controls-backup"), control_identity, lambda: None)
    for entry, paired in zip(entries, payloads):
        payload = {"schema": CONTROL_SCHEMA, "identity_sha256": object_sha256(control_identity),
            "feature_schema": identity["feature_schema"], "pair": entry["pair"], "control_entry": entry,
            "natural": {s: _control_run(paired, s, "natural") for s in SLOTS},
            "c2": {s: {c: _control_run(paired, s, c) for c in ("reference", "overshoot")} for s in SLOTS}}
        controls.commit(payload)
    controls.seal({"wall_seconds": 2.})
    return pp, cp


def _caches(tmp_path, n=3):
    fit = _sources(tmp_path, "train_fit", n, 0)
    dev = _sources(tmp_path, "train_dev", 2, 10)
    return load_endpoint_group_cache(*fit, "train_fit"), load_endpoint_group_cache(*dev, "train_dev"), fit, dev


def _equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: _equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): _equal(x, y)
    else: assert a == b


def test_strict_actual_cache_assembly_preserves_pool_and_independent_collectors(tmp_path):
    train, dev, fit_paths, _ = _caches(tmp_path)
    validate_endpoint_group_splits(train, dev)
    assert len(train.groups) == 3 and train.support["original_paths"] == 6
    assert train.source_identity["paired"]["common_identity"]["code_files"] != train.source_identity["controls"]["implementation"]
    assert train.support["panels"]["natural"]["rollouts"] == 6
    assert train.support["panels"]["c2"]["states"] == 30
    assert train.support["panels"]["paired"]["states"] == 36
    # Synthetic natural/reference features and full prefixes are identical.
    # They remain two actual executions, but do not become new histories.
    assert train.support["combined_distinct_histories"]["c2"] == 12
    assert load_endpoint_group_cache(*fit_paths, "train_fit", expected_data_sha256=train.data_sha256).data_sha256 == train.data_sha256
    with pytest.raises(ValueError, match="registered digest"):
        load_endpoint_group_cache(*fit_paths, "train_fit", expected_data_sha256="0"*64)


@pytest.mark.parametrize("damage", ["model", "membership", "pair", "language", "distance", "instruction"])
def test_assembler_rejects_cross_source_mismatch(tmp_path, damage):
    train, _, _, _ = _caches(tmp_path)
    pair, controls = copy.deepcopy(train.pair_cache), copy.deepcopy(train.control_cache)
    if damage == "model": controls.identity["common_provenance"]["feature_sha256"] = "f"*64
    elif damage == "membership": controls.groups[0]["pair"]["selection_hash"] = "f"*64
    elif damage == "pair": controls.groups[0]["pair"]["heading_rad"][0] = .01
    elif damage == "language": controls.groups[0]["natural"]["A"]["language_input_sha256"] = "f"*64
    elif damage == "distance": controls.groups[0]["c2"]["A"]["reference"]["labels"]["distance_to_goal"][0] += 1
    elif damage == "instruction": controls.groups[0]["natural"]["A"]["instr_id"] = "wrong"
    with pytest.raises(ValueError): assemble_endpoint_groups(pair, controls)


def test_same_seed_all_arms_share_initialization_order_and_common_dev_monitor(tmp_path):
    train, dev, _, _ = _caches(tmp_path)
    trainers = [EndpointGroupTrainer(train, dev, arm=a, epochs=5, batch_groups=2) for a in ("C1", "C2", "C3", "M")]
    assert len({t.initial_head_sha256 for t in trainers}) == 1
    assert len({tuple(t.group_order(2)) for t in trainers}) == 1
    first = prepare_group_batch(train.groups[:2], "C3")
    second = prepare_group_batch(train.groups[:2], "M")
    for k in ("features", "mask", "labels", "goal_steps"): assert torch.equal(first[k], second[k])
    for t in trainers:
        t.epoch = 5; t.global_step = 10; t.pending_dev = True
    records = [t.monitor_dev()[0] for t in trainers]
    assert all(r == records[0] for r in records)
    r = records[0]
    assert r["common_monitor_bce"] == (r["natural_bce"] + r["c2_bce"] + r["paired_bce"])/3
    assert r["selection_purpose"] == BEST_PURPOSE
    assert {x["pair_id"] for x in r["per_group"]} == {g["pair"]["selection_hash"] for g in dev.groups}
    other = EndpointGroupTrainer(train, dev, arm="M", seed=1)
    assert other.initial_head_sha256 != trainers[0].initial_head_sha256


@pytest.mark.parametrize("pause_after", [1, 10, 11])
def test_resume_missing_vm_matches_complete_adam_rng_and_monitoring(tmp_path, pause_after):
    train, dev, _, _ = _caches(tmp_path)
    options = dict(arm="M", epochs=10, batch_groups=2, monitor_every=5,
                   checkpoint_every_steps=2, keep_local=1, keep_backup=1)
    full = train_endpoint_groups(train, dev, tmp_path/"full", tmp_path/"full-drive", **options)
    paused = train_endpoint_groups(train, dev, tmp_path/"part", tmp_path/"part-drive", max_steps=pause_after, **options)
    assert paused["status"] == "interrupted" and paused["final_checkpoint"] is None
    assert paused["pending_dev"] == (pause_after == 10)
    shutil.rmtree(tmp_path/"part")
    resumed = train_endpoint_groups(train, dev, tmp_path/"part", tmp_path/"part-drive", **options)
    assert resumed["status"] == "complete" and resumed["resumed"]
    assert [r["epoch"] for r in resumed["dev_history"]] == [5, 10]
    full_store, resumed_store = CheckpointStore(tmp_path/"full", tmp_path/"full-drive"), CheckpointStore(tmp_path/"part", tmp_path/"part-drive")
    fs, fh, _ = full_store.restore(); rs, rh, _ = resumed_store.restore()
    _equal(fs, rs); _equal(fh, rh)
    assert resumed["training_history"] == full["training_history"] and resumed["dev_history"] == full["dev_history"]
    final = resumed["final_checkpoint"]
    assert final["epoch"] == 10 and final["global_step"] == 20 and final["selection_purpose"] == "fixed_final_epoch"
    model, metadata = load_endpoint_head(tmp_path/"part"/final["head_relative_path"])
    assert metadata["pending_dev"] is False and metadata["train_config"]["arm"] == "M"
    assert all(k.startswith(("src/", "scripts/")) for k in metadata["training_code_identity"])
    assert model(torch.zeros(1,1536)).shape == (1,)
    _, _, best = resumed_store.restore("best")
    assert best["metrics"]["engineering_best_rule"] == BEST_PURPOSE
    assert len(list((tmp_path/"part"/"snapshots").glob("step-*"))) <= 2


@pytest.mark.parametrize("damage", ["cursor", "pending", "optimizer", "order", "monitor", "initial", "data", "code", "weights"])
def test_resume_rejects_inconsistent_training_state(tmp_path, damage):
    train, dev, _, _ = _caches(tmp_path)
    trainer = EndpointGroupTrainer(train, dev, arm="C3", epochs=10, batch_groups=2)
    for _ in range(10): trainer.step()
    trainer.monitor_dev()
    state = trainer.state_dict()
    if damage == "cursor": state["group_cursor"] = 1
    elif damage == "pending": state["pending_dev"] = True
    elif damage == "optimizer": next(iter(state["optimizer"]["state"].values()))["step"] += 1
    elif damage == "order": state["training_history"][0]["group_order_sha256"] = "0"*64
    elif damage == "monitor": state["dev_history"][0]["common_monitor_bce"] += .01
    elif damage == "initial": state["initial_head_sha256"] = "0"*64
    elif damage == "data": state["data_identity"]["train"] = "0"*64
    elif damage == "code": state["code_identity"]["new.py"] = "0"*64
    elif damage == "weights": next(iter(state["head"].values())).fill_(torch.nan)
    with pytest.raises(ValueError):
        EndpointGroupTrainer(train, dev, arm="C3", epochs=10, batch_groups=2).load_state_dict(state)


def test_default_twenty_epochs_final_is_independent_of_best_and_budgets_are_honest(tmp_path):
    train, dev, _, _ = _caches(tmp_path, n=1)
    result = train_endpoint_groups(train, dev, tmp_path/"run", tmp_path/"drive", arm="C1", checkpoint_every_steps=10)
    assert result["completed_epochs"] == 20 and result["global_step"] == 20
    assert [r["epoch"] for r in result["dev_history"]] == [5,10,15,20]
    assert result["final_checkpoint"]["epoch"] == 20
    budget = result["budget"]
    assert budget["actual_selected_rollouts"] == 2 and budget["logical_slots_per_epoch"] == 6
    assert budget["real_selected_states"] == 4 and budget["state_references_per_epoch"] == 12
    assert budget["padded_head_rows_per_epoch"] == 90
    assert budget["state_references_all_epochs"] == 240 and budget["padded_head_rows_all_epochs"] == 1800
    assert budget["distinct_selected_feature_and_prefix_histories"] == 2
    with pytest.raises(ValueError, match="config/data/code"):
        train_endpoint_groups(train, dev, tmp_path/"run", tmp_path/"drive", arm="M")
    for directory in (tmp_path/"run"/"snapshots", tmp_path/"drive"/"snapshots"):
        for checkpoint in directory.glob("step-*"): (checkpoint/"state.pt").write_bytes(b"corrupt")
    with pytest.raises(CheckpointError):
        train_endpoint_groups(train, dev, tmp_path/"run", tmp_path/"drive", arm="C1")


def test_production_cli_cannot_downgrade_pool_or_change_registered_training(tmp_path):
    from train_endpoint_groups import validate_experiment
    for arm in ("C1","C2","C3","M"):
        spec = json.loads((ROOT/f"configs/endpoint_group_{arm}.json").read_text())
        assert validate_experiment(spec) == arm
    spec = json.loads((ROOT/"configs/endpoint_group_M.json").read_text())
    for key, value in (("epochs",5),("batch_groups",2),("pair_weight",.2),("train_pairs",32),
                       ("allowed_training_seeds",[0]),("monitor_every_epochs",1)):
        changed = copy.deepcopy(spec); changed[key] = value
        with pytest.raises(ValueError): validate_experiment(changed)
    train, dev, _, _ = _caches(tmp_path)
    with pytest.raises(ValueError, match="full source pool"):
        validate_experiment(spec, train=train, dev=dev)


def test_final_dev_export_preserves_original_group_and_checkpoint_binding(tmp_path):
    from train_endpoint_groups import write_training_reports
    train, dev, _, _ = _caches(tmp_path, n=1)
    result = train_endpoint_groups(train, dev, tmp_path/"run", tmp_path/"drive", arm="M", epochs=5)
    report = final_dev_report(result)
    assert [(r["pair_id"], r["scan_id"]) for r in report["groups"]] == [
        (g["pair"]["selection_hash"], g["pair"]["scan"]) for g in dev.groups]
    assert report["checkpoint"] == result["final_checkpoint"]
    assert report["dev_data_sha256"] == dev.data_sha256
    assert report["content_sha256"] == object_sha256({k:v for k,v in report.items() if k != "content_sha256"})
    for row, monitored in zip(report["groups"], result["dev_history"][-1]["per_group"]):
        for key in ("natural_bce", "c2_bce", "paired_bce", "ranking"):
            assert row[key] == monitored[key]
        assert row["both_instructions_correct_by_order"] == monitored["correct_orders"]
        assert row["both_orders_correct"] == monitored["correct_both"]
    written = write_training_reports(result, tmp_path/"run", tmp_path/"drive", lambda: None)
    assert written["final_dev_report"]["sha256"] == file_sha256(tmp_path/"run"/"dev-final.json")
    for name in ("dev-final.json", "training-summary.json"):
        assert (tmp_path/"run"/name).read_bytes() == (tmp_path/"drive"/name).read_bytes()
    for damage in ("pending", "epoch", "aggregate", "duplicate"):
        bad = copy.deepcopy(result)
        if damage == "pending": bad["pending_dev"] = True
        elif damage == "epoch": bad["final_checkpoint"]["epoch"] -= 1
        elif damage == "aggregate": bad["dev_history"][-1]["natural_bce"] += 1
        else: bad["dev_history"][-1]["per_group"][-1] = copy.deepcopy(bad["dev_history"][-1]["per_group"][0])
        with pytest.raises(ValueError): final_dev_report(bad)
