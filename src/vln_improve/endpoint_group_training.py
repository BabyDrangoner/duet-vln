"""Matched source groups and restartable frozen-feature endpoint training.

The primary head is always the fixed final epoch. The independently protected
best head monitors the same three dev BCE panels for every training arm.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint_store import CheckpointStore
from .endpoint_controls import load_control_cache
from .endpoint_group_objectives import (
    ARMS, MAX_STATES, PAIR_WEIGHT, group_objective, per_group_metrics,
    prepare_group_batch, score_group_batch,
)
from .endpoint_pair_training import load_endpoint_pair_cache, validate_endpoint_pair_splits
from .endpoint_pairs import ORDERS, SLOTS, content_hash
from .endpoint_probe import EndpointProbe, FEATURE_DIM, FEATURE_SCHEMA, HEAD_SCHEMA
from .protocol import file_sha256, object_sha256
from .resumable import _cpu_copy, _restore_rng, _rng_state

GROUP_SCHEMA = "duet_endpoint_training_groups_v1"
TRAIN_SCHEMA = "duet_endpoint_group_training_v1"
FINAL_PURPOSE = "fixed_final_epoch"
BEST_PURPOSE = "equal_mean_of_natural_c2_paired_dev_episode_bce"
PANELS = ("natural", "c2", "paired")


@dataclass(frozen=True)
class EndpointGroupCache:
    split: str
    groups: tuple[dict, ...]
    data_sha256: str
    source_identity: dict
    common_provenance: dict
    support: dict
    pair_cache: object
    control_cache: object


def _panel_runs(group, panel):
    if panel == "natural":
        return [(group["natural"][s], None) for s in SLOTS]
    if panel == "c2":
        return [(group["c2"][s][c], None) for s in SLOTS for c in ("reference", "overshoot")]
    return [(group["paired"][o][s], i) for o in ORDERS for i, s in enumerate(SLOTS)]


def _support(groups):
    panels, history_sets = {}, {}
    for panel in PANELS:
        counts = {"rollouts": 0, "states": 0, "positives": 0, "negatives": 0}
        hashes = set()
        scenes = {}
        for group in groups:
            scan = group["pair"]["scan"]
            scene = scenes.setdefault(scan, {"rollouts": 0, "states": 0, "positives": 0, "negatives": 0})
            for run, slot in _panel_runs(group, panel):
                y = run["labels"]["within_success_radius"]
                y = y if slot is None else y[:, slot]
                values = {"rollouts": 1, "states": len(y), "positives": int(y.sum()), "negatives": int((~y).sum())}
                for key, value in values.items():
                    counts[key] += value; scene[key] += value
                # Exact feature/prefix duplicates only; no claim of statistical independence.
                prefix = [{k: state[k] for k in ("viewpoint", "heading", "elevation", "trajectory_prefix")}
                          for state in run["states"]]
                hashes.add(content_hash({"instruction": run["instr_id"], "features": run["features"], "prefix": prefix}))
        panels[panel] = {**counts, "distinct_feature_and_prefix_histories": len(hashes), "per_scene": scenes}
        history_sets[panel] = hashes
    return {"groups": len(groups), "original_paths": 2 * len(groups), "original_instructions": 2 * len(groups),
            "scenes": len({g["pair"]["scan"] for g in groups}), "panels": panels,
            "combined_distinct_histories": {p: len(history_sets["natural"] | history_sets[p]) for p in PANELS},
            "dedup_definition": "exact instruction ID, FP32 feature bytes, and observed viewpoint/angle/full-prefix equality"}


def assemble_endpoint_groups(pair_cache, control_cache, *, expected_data_sha256=None):
    """Join two already validated whole caches; source implementations stay separate."""
    if pair_cache.split not in {"train_fit", "train_dev"} or control_cache.identity["split"] != pair_cache.split:
        raise ValueError("paired/control training splits differ")
    common = pair_cache.common_provenance
    if (control_cache.common_identity != {"common_provenance": common, "feature_schema": FEATURE_SCHEMA,
                                          "feature_dim": FEATURE_DIM}):
        raise ValueError("paired/control backbone or feature/data provenance differs")
    paired = {p["pair"]["selection_hash"]: p for p in pair_cache.pairs}
    controls = {p["pair"]["selection_hash"]: p for p in control_cache.groups}
    if (len(paired) != len(pair_cache.pairs) or len(controls) != len(control_cache.groups)
            or list(paired) != list(controls)):
        raise ValueError("paired/control groups differ in membership or fixed order")
    groups = []
    for key, payload in paired.items():
        control = controls[key]
        pair = payload["pair"]
        if control["pair"] != pair or control["control_entry"]["pair"] != pair:
            raise ValueError("paired/control original source pair mapping differs")
        group = {"pair": pair, "natural": control["natural"], "c2": control["c2"], "paired": payload["rollouts"]}
        for index, slot in enumerate(SLOTS):
            language, distances = set(), {}
            runs = [(group["natural"][slot], None)] + [(group["c2"][slot][c], None) for c in ("reference", "overshoot")]
            runs += [(group["paired"][o][slot], index) for o in ORDERS]
            for run, column in runs:
                if run["instr_id"] != pair["instr_ids"][index] or run["instruction_slot"] != index:
                    raise ValueError("group original instruction slot mapping differs")
                language.add((run["language_input_sha256"], run["instruction_text_sha256"]))
                label = run["labels"]
                if column is None:
                    if label["goal_vpid"] != pair["goal_vpids"][index]:
                        raise ValueError("control goal differs from paired instruction goal")
                    values = label["distance_to_goal"]
                else:
                    values = label["distance_to_goals"][:, column]
                for state, distance in zip(run["states"], values):
                    vp, value = state["viewpoint"], float(distance)
                    if vp in distances and distances[vp] != value:
                        raise ValueError("same instruction/viewpoint has inconsistent cross-source distance labels")
                    distances[vp] = value
            if len(language) != 1:
                raise ValueError("paired/control language inputs or instruction text differ")
        groups.append(group)
    source = {"schema": GROUP_SCHEMA, "split": pair_cache.split,
        "paired": {"identity_sha256": pair_cache.identity_sha256, "data_sha256": pair_cache.data_sha256,
                   "common_identity": pair_cache.common_identity},
        "controls": {"identity_sha256": control_cache.identity_sha256, "data_sha256": control_cache.data_sha256,
                     "implementation": copy.deepcopy(control_cache.identity["code_files"]),
                     "collection_config_sha256": control_cache.identity["collection_config_sha256"],
                     "controls_report_sha256": control_cache.identity["controls_report_sha256"],
                     "runtime_config_sha256": control_cache.identity["runtime_config_sha256"]},
        "ordered_pair_sha256": [object_sha256(g["pair"]) for g in groups]}
    digest = object_sha256(source)
    if expected_data_sha256 is not None and digest != expected_data_sha256:
        raise ValueError("assembled group data differs from registered digest")
    return EndpointGroupCache(pair_cache.split, tuple(groups), digest, source, common,
                              _support(groups), pair_cache, control_cache)


def load_endpoint_group_cache(pair_directory, control_directory, expected_split, *, expected_data_sha256=None,
                              expected_pair_identity_sha256=None, expected_control_identity_sha256=None):
    for directory in (pair_directory, control_directory):
        path = Path(directory).absolute()
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("group source cache path contains a symbolic link")
    paired = load_endpoint_pair_cache(pair_directory, expected_split,
                                      expected_identity_sha256=expected_pair_identity_sha256)
    controls = load_control_cache(control_directory, expected_split=expected_split,
                                  expected_identity_sha256=expected_control_identity_sha256)
    return assemble_endpoint_groups(paired, controls, expected_data_sha256=expected_data_sha256)


def validate_endpoint_group_splits(train, dev):
    validate_endpoint_pair_splits(train.pair_cache, dev.pair_cache)
    if train.common_provenance != dev.common_provenance:
        raise ValueError("group fit/dev common provenance differs")
    for key in ("implementation", "collection_config_sha256", "controls_report_sha256", "runtime_config_sha256"):
        if train.source_identity["controls"][key] != dev.source_identity["controls"][key]:
            raise ValueError("group fit/dev controls protocol or source implementation differs")


def _code_identity():
    folder = Path(__file__).resolve().parent
    names = ("endpoint_group_training.py", "endpoint_group_objectives.py", "endpoint_pair_training.py",
             "endpoint_controls.py", "endpoint_pairs.py", "endpoint_probe.py", "resumable.py", "checkpoint_store.py", "protocol.py")
    mapping = {"src/vln_improve/" + name: file_sha256(folder / name) for name in names}
    mapping["scripts/train_endpoint_groups.py"] = file_sha256(folder.parents[1] / "scripts/train_endpoint_groups.py")
    return mapping


def training_code_identity():
    """Root-relative training-source digests for an independent final-head audit."""
    return _code_identity()


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value) and value >= 0


def _dev_record(rows, epoch, step):
    scenes = {}
    for row in rows:
        if (set(row) != {"pair_id", "scan", "natural_bce", "c2_bce", "paired_bce", "ranking", "correct_orders", "correct_both"}
                or any(not _finite(row[k]) for k in ("natural_bce", "c2_bce", "paired_bce", "ranking"))
                or not isinstance(row["correct_orders"], list) or len(row["correct_orders"]) != 2
                or any(type(x) is not bool for x in row["correct_orders"])
                or type(row["correct_both"]) is not bool or row["correct_both"] != all(row["correct_orders"])):
            raise ValueError("invalid common dev panel group row")
        scenes.setdefault(row["scan"], []).append(row)
    def means(items):
        return {"groups": len(items), **{p + "_bce": sum(r[p + "_bce"] for r in items) / len(items) for p in PANELS},
                "ranking": sum(r["ranking"] for r in items) / len(items),
                "correct_order_rates": [sum(r["correct_orders"][i] for r in items) / len(items) for i in range(2)],
                "both_orders_rate": sum(r["correct_both"] for r in items) / len(items)}
    overall = means(rows)
    return {"epoch": epoch, "global_step": step, **overall,
            "common_monitor_bce": sum(overall[p + "_bce"] for p in PANELS) / 3,
            "selection_purpose": BEST_PURPOSE, "by_scene": {s: means(r) for s, r in sorted(scenes.items())},
            "per_group": copy.deepcopy(rows)}


def final_dev_report(result):
    """Export fixed-final group metrics; rows remain paired at the original group."""
    final = result.get("final_checkpoint")
    history = result.get("dev_history")
    if (result.get("status") != "complete" or result.get("pending_dev") is not False
            or not isinstance(final, dict) or not isinstance(history, list) or not history
            or final.get("selection_purpose") != FINAL_PURPOSE
            or final.get("epoch") != result.get("completed_epochs")
            or final.get("global_step") != result.get("global_step")):
        raise ValueError("final group metrics require a completed fixed-final checkpoint")
    record = history[-1]
    rows = record.get("per_group")
    if (not isinstance(rows, list) or not rows
            or len({r.get("pair_id") for r in rows}) != len(rows)
            or record != _dev_record(rows, final["epoch"], final["global_step"])):
        raise ValueError("final group monitoring rows/association differ from the final checkpoint")
    report = {"schema": "duet_endpoint_group_final_dev_v1", "split": "train_dev", "usage": "analysis_only",
        "arm": result["arm"], "training_seed": result["seed"],
        "train_data_sha256": result["train_data_sha256"], "dev_data_sha256": result["dev_data_sha256"],
        "checkpoint": copy.deepcopy(final), "selection_purpose": FINAL_PURPOSE,
        "statistical_unit": "original paired group; four paired rollouts are not independent samples",
        "monitor": {k: copy.deepcopy(v) for k, v in record.items() if k != "per_group"},
        "groups": [{"pair_id": row["pair_id"], "scan_id": row["scan"],
            "natural_bce": row["natural_bce"], "c2_bce": row["c2_bce"], "paired_bce": row["paired_bce"],
            "ranking": row["ranking"],
            "both_instructions_correct_by_order": row["correct_orders"],
            "both_orders_correct": row["correct_both"]} for row in rows]}
    report["content_sha256"] = object_sha256(report)
    return report


class EndpointGroupTrainer:
    def __init__(self, train, dev, *, arm, seed=0, device="cpu", epochs=20, batch_groups=8, monitor_every=5,
                 experiment_sha256=None):
        validate_endpoint_group_splits(train, dev)
        if arm not in ARMS or type(seed) is not int or seed not in (0, 1, 2):
            raise ValueError("group training requires C1/C2/C3/M and registered seed 0/1/2")
        if (any(type(x) is not int or x < 1 for x in (epochs, batch_groups, monitor_every))
                or epochs % monitor_every or not train.groups or not dev.groups):
            raise ValueError("invalid group epoch/batch/monitoring counts")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"} or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("requested group training device is unavailable")
        self.train, self.dev = train, dev
        self.config = {"arm": arm, "seed": seed, "device": str(self.device), "epochs": epochs,
            "batch_groups": batch_groups, "monitor_every": monitor_every, "lr": .001, "weight_decay": .0001,
            "optimizer": "AdamW", "pair_weight": PAIR_WEIGHT if arm == "M" else 0., "max_states": MAX_STATES,
            "natural_weight": .5, "augmentation_weight": .5,
            "feature_dim": FEATURE_DIM, "hidden_dim": 128, "activation": "relu",
            "torch_version": str(torch.__version__), "primary_checkpoint": FINAL_PURPOSE,
            "engineering_best_checkpoint": BEST_PURPOSE}
        self.config["experiment_sha256"] = experiment_sha256
        self.data_identity = {"train": train.data_sha256, "dev": dev.data_sha256,
                              "train_sources": train.source_identity, "dev_sources": dev.source_identity}
        self.code_identity = _code_identity()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if self.device.type == "cuda": torch.cuda.manual_seed_all(seed)
        self.head = EndpointProbe().to(self.device)
        self.initial_head_sha256 = content_hash(self.head.state_dict())
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=.001, weight_decay=.0001)
        self.epoch = self.cursor = self.global_step = 0
        self.pending_dev = False
        self.epoch_sums = {k: 0. for k in ("loss", "bce", "ranking")}
        self.training_history, self.dev_history = [], []
        self.best_common_bce = self.best_epoch = None
        self.batches_per_epoch = math.ceil(len(train.groups) / batch_groups)

    @property
    def done(self):
        return self.epoch == self.config["epochs"] and not self.pending_dev

    def group_order(self, epoch):
        order = list(range(len(self.train.groups)))
        random.Random(self.config["seed"] + epoch).shuffle(order)
        return order

    def _order_sha(self, epoch):
        return object_sha256([self.train.groups[i]["pair"]["selection_hash"] for i in self.group_order(epoch)])

    def step(self):
        if self.done or self.pending_dev:
            raise ValueError("group training is done or awaits common dev monitoring")
        indices = self.group_order(self.epoch)[self.cursor:self.cursor + self.config["batch_groups"]]
        batch = prepare_group_batch([self.train.groups[i] for i in indices], self.config["arm"], self.device)
        self.head.train(); self.optimizer.zero_grad(set_to_none=True)
        values = group_objective(score_group_batch(self.head, batch), batch, self.config["arm"])
        if any(not bool(torch.isfinite(values[k])) for k in self.epoch_sums):
            raise ValueError("nonfinite group objective")
        values["loss"].backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.head.parameters()):
            raise ValueError("nonfinite group gradient")
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.head.parameters()):
            raise ValueError("nonfinite group parameters after AdamW")
        numbers = {k: float(values[k].detach()) for k in self.epoch_sums}
        for key, value in numbers.items(): self.epoch_sums[key] += value * len(indices)
        self.cursor += len(indices); self.global_step += 1
        if self.cursor == len(self.train.groups):
            self.training_history.append({"epoch": self.epoch + 1, "global_step": self.global_step,
                "groups": self.cursor, "group_order_sha256": self._order_sha(self.epoch),
                **{k: v / self.cursor for k, v in self.epoch_sums.items()}})
            self.epoch += 1; self.cursor = 0
            self.epoch_sums = {k: 0. for k in self.epoch_sums}
            self.pending_dev = self.epoch % self.config["monitor_every"] == 0
        return {"global_step": self.global_step, "groups": len(indices), **numbers}

    def monitor_dev(self):
        if not self.pending_dev:
            raise ValueError("no registered epoch awaiting common dev monitoring")
        self.head.eval()
        rows = []
        with torch.no_grad():
            for start in range(0, len(self.dev.groups), self.config["batch_groups"]):
                groups = self.dev.groups[start:start + self.config["batch_groups"]]
                panels = {}
                for arm in ("C2", "C3"):
                    batch = prepare_group_batch(groups, arm, self.device)
                    panels[arm] = {k: v.detach().cpu() for k, v in per_group_metrics(score_group_batch(self.head, batch), batch).items()}
                if not torch.equal(panels["C2"]["natural_bce"], panels["C3"]["natural_bce"]):
                    raise ValueError("common dev natural panel changed between control assemblies")
                for i, group in enumerate(groups):
                    rows.append({"pair_id": group["pair"]["selection_hash"], "scan": group["pair"]["scan"],
                        "natural_bce": float(panels["C2"]["natural_bce"][i]),
                        "c2_bce": float(panels["C2"]["augmentation_bce"][i]),
                        "paired_bce": float(panels["C3"]["augmentation_bce"][i]),
                        "ranking": float(panels["C3"]["ranking"][i]),
                        "correct_orders": panels["C3"]["both_instructions_correct_by_order"][i].tolist(),
                        "correct_both": bool(panels["C3"]["both_orders_correct"][i])})
        record = _dev_record(rows, self.epoch, self.global_step)
        self.dev_history.append(record)
        best = self.best_common_bce is None or record["common_monitor_bce"] < self.best_common_bce
        if best: self.best_common_bce, self.best_epoch = record["common_monitor_bce"], self.epoch
        self.pending_dev = False
        return record, best

    def state_dict(self):
        return {"schema": TRAIN_SCHEMA, "config": copy.deepcopy(self.config),
            "data_identity": copy.deepcopy(self.data_identity), "code_identity": copy.deepcopy(self.code_identity),
            "initial_head_sha256": self.initial_head_sha256,
            "head": _cpu_copy(self.head.state_dict()), "optimizer": _cpu_copy(self.optimizer.state_dict()),
            "rng": _rng_state(), "scheduler": None, "scaler": None,
            "epoch": self.epoch, "group_cursor": self.cursor, "global_step": self.global_step,
            "pending_dev": self.pending_dev, "epoch_sums": copy.deepcopy(self.epoch_sums),
            "training_history": copy.deepcopy(self.training_history), "dev_history": copy.deepcopy(self.dev_history),
            "best_common_bce": self.best_common_bce, "best_epoch": self.best_epoch}

    def load_state_dict(self, saved):
        if (not isinstance(saved, dict) or saved.get("schema") != TRAIN_SCHEMA
                or saved.get("config") != self.config or saved.get("data_identity") != self.data_identity
                or saved.get("code_identity") != self.code_identity
                or saved.get("initial_head_sha256") != self.initial_head_sha256
                or saved.get("scheduler") is not None or saved.get("scaler") is not None):
            raise ValueError("group resume config/data/code/initialization differs")
        epoch, cursor, step = [saved.get(k) for k in ("epoch", "group_cursor", "global_step")]
        n, batch = len(self.train.groups), self.config["batch_groups"]
        if (any(type(x) is not int for x in (epoch, cursor, step)) or not 0 <= epoch <= self.config["epochs"]
                or not 0 <= cursor < n or cursor % batch or (epoch == self.config["epochs"] and cursor)
                or step != epoch * self.batches_per_epoch + cursor // batch):
            raise ValueError("invalid group resume optimizer cursor")
        pending, training, dev = [saved.get(k) for k in ("pending_dev", "training_history", "dev_history")]
        if (type(pending) is not bool or (pending and (not epoch or epoch % self.config["monitor_every"] or cursor))
                or not isinstance(training, list) or len(training) != epoch
                or not isinstance(dev, list) or len(dev) != epoch // self.config["monitor_every"] - int(pending)):
            raise ValueError("group pending monitor/history differs from epoch")
        for index, record in enumerate(training, 1):
            if (not isinstance(record, dict) or set(record) != {"epoch", "global_step", "groups", "group_order_sha256", "loss", "bce", "ranking"}
                    or record["epoch"] != index or record["global_step"] != index * self.batches_per_epoch
                    or record["groups"] != n or record["group_order_sha256"] != self._order_sha(index - 1)
                    or any(not _finite(record[k]) for k in ("loss", "bce", "ranking"))):
                raise ValueError("group online training history/order is invalid")
        association = [(g["pair"]["selection_hash"], g["pair"]["scan"]) for g in self.dev.groups]
        for index, record in enumerate(dev, 1):
            monitor_epoch = index * self.config["monitor_every"]
            if not isinstance(record, dict) or not isinstance(record.get("per_group"), list):
                raise ValueError("invalid common dev monitoring record")
            rows = record["per_group"]
            if ([(r.get("pair_id"), r.get("scan")) for r in rows] != association
                    or record != _dev_record(rows, monitor_epoch, monitor_epoch * self.batches_per_epoch)):
                raise ValueError("common dev panels/associations/checkpoint metric changed")
        best = min(dev, key=lambda r: r["common_monitor_bce"]) if dev else None
        if (saved.get("best_common_bce") != (best["common_monitor_bce"] if best else None)
                or saved.get("best_epoch") != (best["epoch"] if best else None)):
            raise ValueError("group best-dev metadata differs from common monitoring history")
        sums = saved.get("epoch_sums")
        if (not isinstance(sums, dict) or set(sums) != {"loss", "bce", "ranking"}
                or any(not _finite(x) for x in sums.values()) or (cursor == 0 and any(sums.values()))):
            raise ValueError("invalid partial-epoch group objective totals")
        head = EndpointProbe().to(self.device)
        expected, weights = head.state_dict(), saved.get("head")
        if (not isinstance(weights, dict) or set(weights) != set(expected)
                or any(not isinstance(t, torch.Tensor) or t.shape != expected[k].shape
                       or t.dtype != expected[k].dtype or not torch.isfinite(t).all() for k, t in weights.items())):
            raise ValueError("invalid group head weights")
        head.load_state_dict(weights, strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        opt = saved.get("optimizer")
        if not isinstance(opt, dict) or opt.get("param_groups") != optimizer.state_dict()["param_groups"]:
            raise ValueError("group AdamW parameter groups changed")
        moments, parameters = opt.get("state"), list(head.parameters())
        if not isinstance(moments, dict) or set(moments) != (set(range(len(parameters))) if step else set()):
            raise ValueError("group AdamW state is missing or unexpected")
        for index, value in moments.items():
            if (not isinstance(value, dict) or set(value) != {"step", "exp_avg", "exp_avg_sq"}
                    or not isinstance(value["step"], torch.Tensor) or value["step"].numel() != 1
                    or not torch.isfinite(value["step"]).all() or float(value["step"]) != step):
                raise ValueError("group AdamW step differs from optimizer cursor")
            for name in ("exp_avg", "exp_avg_sq"):
                tensor = value[name]
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != parameters[index].shape
                        or tensor.dtype != parameters[index].dtype or not torch.isfinite(tensor).all()
                        or (name == "exp_avg_sq" and (tensor < 0).any())):
                    raise ValueError("invalid group AdamW moment")
        optimizer.load_state_dict(opt)
        _restore_rng(saved["rng"], device=self.device, seed=self.config["seed"])
        self.head, self.optimizer = head, optimizer
        self.epoch, self.cursor, self.global_step = epoch, cursor, step
        self.pending_dev, self.epoch_sums = pending, copy.deepcopy(sums)
        self.training_history, self.dev_history = copy.deepcopy(training), copy.deepcopy(dev)
        self.best_common_bce, self.best_epoch = saved["best_common_bce"], saved["best_epoch"]

    def head_payload(self):
        return {"schema": HEAD_SCHEMA, "head_config": self.head.config, "feature_schema": FEATURE_SCHEMA,
            "state_dict": _cpu_copy(self.head.state_dict()),
            "common_identity": {"common_provenance": copy.deepcopy(self.train.common_provenance),
                                "feature_schema": FEATURE_SCHEMA, "feature_dim": FEATURE_DIM},
            "data_identity": copy.deepcopy(self.data_identity), "train_config": copy.deepcopy(self.config),
            "training_code_identity": copy.deepcopy(self.code_identity), "pending_dev": self.pending_dev,
            "initial_head_sha256": self.initial_head_sha256,
            "epoch": self.epoch, "global_step": self.global_step,
            "selection_purpose": FINAL_PURPOSE, "engineering_best_rule": BEST_PURPOSE}

    def budget(self):
        panel = "natural" if self.config["arm"] == "C1" else ("c2" if self.config["arm"] == "C2" else "paired")
        support = self.train.support
        natural, augmentation = support["panels"]["natural"], support["panels"][panel]
        references = 3 * natural["states"] if panel == "natural" else natural["states"] + augmentation["states"]
        return {"original_paths": support["original_paths"], "original_instructions": support["original_instructions"],
            "groups_per_epoch": len(self.train.groups), "logical_slots_per_epoch": 6 * len(self.train.groups),
            "actual_selected_rollouts": natural["rollouts"] + (augmentation["rollouts"] if panel != "natural" else 0),
            "distinct_selected_feature_and_prefix_histories": support["combined_distinct_histories"][panel],
            "real_selected_states": natural["states"] + (augmentation["states"] if panel != "natural" else 0),
            "state_references_per_epoch": references, "padded_head_rows_per_epoch": len(self.train.groups) * 6 * MAX_STATES,
            "state_references_all_epochs": references * self.config["epochs"],
            "padded_head_rows_all_epochs": len(self.train.groups) * 6 * MAX_STATES * self.config["epochs"],
            "expected_optimizer_updates": self.batches_per_epoch * self.config["epochs"],
            "train_support": support, "dev_support": self.dev.support}


def train_endpoint_groups(train, dev, local_dir, backup_dir, *, arm, seed=0, device="cpu",
                          epochs=20, batch_groups=8, monitor_every=5, experiment_sha256=None,
                          checkpoint_every_steps=10, checkpoint_every_seconds=60., keep_local=2, keep_backup=5,
                          verify_backup=None, should_stop=None, max_steps=None):
    """Train one arm/seed through complete optimizer boundaries and verified copies.

    ``max_steps`` is an interruption-test boundary, not an early-stopping rule.
    The production CLI validates the fixed full-pool protocol and data digests.
    """
    if (type(checkpoint_every_steps) is not int or checkpoint_every_steps < 1
            or not _finite(checkpoint_every_seconds) or checkpoint_every_seconds == 0
            or (max_steps is not None and (type(max_steps) is not int or max_steps < 0))):
        raise ValueError("invalid group checkpoint or interruption-test interval")
    checker, stopping = verify_backup or (lambda: None), should_stop or (lambda: False)
    trainer = EndpointGroupTrainer(train, dev, arm=arm, seed=seed, device=device,
        epochs=epochs, batch_groups=batch_groups, monitor_every=monitor_every, experiment_sha256=experiment_sha256)
    checker()
    store = CheckpointStore(local_dir, backup_dir, keep_local=keep_local, keep_backup=keep_backup)
    started, resumed = time.monotonic(), False
    if trainer.device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    with store.lock():
        try:
            state, head, _ = store.restore("latest")
        except FileNotFoundError:
            pass
        else:
            trainer.load_state_dict(state)
            if content_hash(head) != content_hash(trainer.head_payload()):
                raise ValueError("group snapshot head differs from optimizer-state head")
            resumed = True
        last_step, last_saved = trainer.global_step, time.monotonic()
        latest_id = None

        def persist(reason, *, best=False, metrics=None):
            nonlocal last_step, last_saved, latest_id
            checker()
            detail = {"reason": reason, "primary_rule": FINAL_PURPOSE, "engineering_best_rule": BEST_PURPOSE,
                      "epoch": trainer.epoch, "arm": arm, "seed": seed}
            if metrics is not None: detail["common_monitor_bce"] = metrics["common_monitor_bce"]
            latest_id = store.save(trainer.state_dict(), trainer.head_payload(), step=trainer.global_step,
                                   is_best=best, metrics=detail)
            last_step, last_saved = trainer.global_step, time.monotonic()

        if not resumed: persist("initialized")
        invocation_step, interrupted = trainer.global_step, False
        try:
            while not trainer.done:
                if stopping() or (max_steps is not None and trainer.global_step - invocation_step >= max_steps):
                    persist("interrupted_at_complete_optimizer_boundary")
                    interrupted = True
                    break
                if trainer.pending_dev:
                    metrics, best = trainer.monitor_dev()
                    persist("common_three_panel_dev_monitored", best=best, metrics=metrics)
                    continue
                trainer.step()
                if (trainer.pending_dev or trainer.global_step - last_step >= checkpoint_every_steps
                        or time.monotonic() - last_saved >= checkpoint_every_seconds):
                    persist("common_dev_pending" if trainer.pending_dev else "periodic")
        except KeyboardInterrupt:
            # Interrupting AdamW.step can leave partially updated state. Keep the
            # last durable boundary instead of blessing that uncertain state.
            return {"status": "interrupted", "resumed": resumed, "last_durable_step": last_step,
                    "arm": arm, "seed": seed, "primary_rule": FINAL_PURPOSE}
        if trainer.done: persist("fixed_final_epoch_complete")
        final = None
        if trainer.done:
            _, final_head, final_manifest = store.restore("latest")
            if content_hash(final_head) != content_hash(trainer.head_payload()):
                raise ValueError("final checkpoint read-back differs from fixed final head")
            final = {"checkpoint_id": final_manifest["checkpoint_id"], "epoch": trainer.epoch,
                "global_step": trainer.global_step, "head_sha256": final_manifest["files"]["head.pt"]["sha256"],
                "head_relative_path": str(Path("snapshots") / final_manifest["checkpoint_id"] / "head.pt"),
                "selection_purpose": FINAL_PURPOSE}
        return {"status": "interrupted" if interrupted else "complete", "resumed": resumed,
            "arm": arm, "seed": seed, "global_step": trainer.global_step, "completed_epochs": trainer.epoch,
            "pending_dev": trainer.pending_dev, "initial_head_sha256": trainer.initial_head_sha256,
            "primary_rule": FINAL_PURPOSE, "final_checkpoint": final,
            "best_common_bce": trainer.best_common_bce, "best_epoch": trainer.best_epoch,
            "engineering_best_rule": BEST_PURPOSE, "latest_checkpoint_id": latest_id,
            "train_data_sha256": train.data_sha256, "dev_data_sha256": dev.data_sha256,
            "training_history": trainer.training_history, "dev_history": trainer.dev_history,
            "budget": trainer.budget(), "resources": {"wall_seconds_this_invocation": time.monotonic() - started,
                "optimizer_updates_this_invocation": trainer.global_step - invocation_step,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if trainer.device.type == "cuda" else None}}
