#!/usr/bin/env python3
"""Train small E3 comparators on fixed train_fit continuations; select on train_dev."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from vln_improve.continuation_learning import ContinuationComparator, batch_record_losses
from vln_improve.continuation_v2 import SCHEMA as DATA_SCHEMA, SCHEDULES, choose_action, load_bundle
from vln_improve.endpoint_pairs import content_hash
from vln_improve.pipeline import (MountedCheckpointStore, atomic_json, backup_mount_identity,
                                  validate_backup_root, validate_separate_roots)
from vln_improve.protocol import file_sha256, object_sha256
from vln_improve.resumable import _cpu_copy, _restore_rng, _rng_state
from vln_improve.intervention_runtime import verified_copy


SCHEMA = "e3_continuation_training_v2"
ARMS = {"relative-history": ("relative", True, False),
        "absolute-history": ("absolute", True, False),
        "teacher-history": ("teacher", True, False),
        "relative-nohistory": ("relative", False, False),
        "relative-olddata": ("relative", True, True)}
CODE_FILES = ("scripts/train_continuation_v2.py", "src/vln_improve/continuation_learning.py",
              "src/vln_improve/continuation_v2.py", "src/vln_improve/checkpoint_store.py",
              "src/vln_improve/pipeline.py", "src/vln_improve/resumable.py",
              "src/vln_improve/endpoint_pairs.py", "src/vln_improve/protocol.py",
              "src/vln_improve/intervention_runtime.py")


def code_identity():
    return {name: file_sha256(ROOT / name) for name in CODE_FILES}


def trees_equal(left, right):
    """Compare portable full states, including integer optimizer-state keys."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(trees_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            trees_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def default_config():
    return {"seed": 0, "epochs": 20, "candidate_epochs": [5, 10, 20], "batch_size": 32,
            "lr": 3e-4, "weight_decay": .01, "hidden_dim": 128, "feature_dim": 1549,
            "sr_thresholds": [0., .05, .1, .2, .4], "spl_thresholds": [0.],
            "sr_weight": 1., "spl_weight": 1., "rescue_weight": 4., "harm_weight": 4.,
            "checkpoint_every_steps": 100, "device": "cuda"}


@dataclass
class Dataset:
    root: Path
    manifest: dict
    bundles: list
    identity: dict

    @property
    def records(self):
        return [record for bundle in self.bundles for record in bundle["records"]]


def load_dataset(root, split, *, check_backup=lambda: None):
    """Require complete manifests, verified backup bytes and paired branch labels."""
    root = Path(root).expanduser().resolve()
    path = root / "dataset-manifest.json"
    manifest = json.loads(path.read_text())
    if (split not in {"train_fit", "train_dev"} or manifest.get("schema") != DATA_SCHEMA
            or manifest.get("complete") is not True or manifest["selection"]["split"] != split):
        raise ValueError("complete training-partition dataset required")
    backup = Path(manifest["backup_root"]).expanduser().resolve()
    validate_separate_roots(root, backup)
    check_backup()
    if file_sha256(backup / path.name) != file_sha256(path):
        raise ValueError("local and persistent dataset manifests differ")
    selection = manifest["selection"]
    ids, scans, conditions = selection["instr_ids"], selection["scan_ids"], selection["conditions"]
    if (not ids or len(ids) != len(set(ids)) or not scans or len(scans) != len(set(scans))
            or set(conditions) != set(SCHEDULES) or len(conditions) != len(SCHEDULES)):
        raise ValueError("dataset needs unique instructions/scenes and all four fixed conditions")
    bundles, seen, actual_scans = [], set(), set()
    for pointer in manifest["bundles"]:
        bundle = load_bundle(root / "bundles", backup / "bundles", pointer, check_backup=check_backup)
        ref = bundle["reference"]
        key = (ref["instr_id"], ref["condition"])
        if (bundle.get("split") != split or key in seen or key[0] not in ids or key[1] not in conditions
                or pointer["task"]["instr_id"] != key[0] or pointer["task"]["condition"] != key[1]
                or ref["scan_id"] not in scans):
            raise ValueError("bundle instruction/condition/split identity differs")
        seen.add(key)
        actual_scans.add(ref["scan_id"])
        branch_index = {}
        for branch in bundle["branches"]:
            branch_key = (branch["target_step"], branch["target_action"])
            if branch_key in branch_index:
                raise ValueError("duplicate continuation branch")
            branch_index[branch_key] = branch
        steps = set()
        for record in bundle["records"]:
            step = record["step"]
            if ((record["instr_id"], record["condition"]) != key or record["scan_id"] != ref["scan_id"]
                    or type(step) is not int or step in steps or not 0 <= step < len(ref["states"])):
                raise ValueError("record differs from its causal reference")
            steps.add(step)
            actions = record["candidate_actions"]
            if (not 1 <= len(actions) <= 4 or len(set(actions)) != len(actions)
                    or actions[0] != ref["states"][step]["executed_action"]
                    or record["features"].shape != (len(actions), 1549)
                    or record["history_features"].shape != (2 * step + 1, 1549)
                    or record["progress"].shape != (4,) or record["utilities"].shape != (len(actions), 2)):
                raise ValueError("invalid candidate inventory, causal history or feature schema")
            for index, action in enumerate(actions):
                branch = branch_index.get((step, action))
                if branch is None:
                    raise ValueError("candidate has no full continuation label")
                expected = torch.tensor([branch["metrics"]["success"], branch["metrics"]["spl"]], dtype=torch.float64)
                if not torch.allclose(record["utilities"][index].double(), expected, rtol=1e-6, atol=1e-6):
                    raise ValueError("candidate utilities differ from full branch outcomes")
            expected = torch.tensor([ref["metrics"]["success"], ref["metrics"]["spl"]], dtype=torch.float64)
            if not torch.allclose(record["utilities"][0].double(), expected, rtol=1e-6, atol=1e-6):
                raise ValueError("baseline candidate differs from reference outcomes")
        bundles.append(bundle)
    if seen != {(instr, condition) for instr in ids for condition in conditions} or actual_scans != set(scans):
        raise ValueError("dataset does not cover its fixed selection exactly")
    identity = {"manifest_sha256": file_sha256(path), "selection": selection,
                "provenance": manifest["provenance"],
                "bundle_inventory_sha256": object_sha256(manifest["bundles"])}
    return Dataset(root, manifest, bundles, identity)


def validate_splits(fit, dev):
    a, b = fit.manifest["selection"], dev.manifest["selection"]
    if a["split"] != "train_fit" or b["split"] != "train_dev":
        raise ValueError("fit/dev must be the two training partitions")
    if set(a["instr_ids"]) & set(b["instr_ids"]) or set(a["scan_ids"]) & set(b["scan_ids"]):
        raise ValueError("instruction or scene leakage between fit and dev")
    # Location/runtime fields can differ; model, assets and collector must match.
    pa, pb = fit.manifest["provenance"], dev.manifest["provenance"]
    common = ("asset_pins", "base_checkpoint_sha256", "feature_sha256", "train_annotation_sha256",
              "connectivity_sha256", "source", "source_files", "source_sha256", "experiment_sha256",
              "model_config_file_sha256", "model_config_sha256", "model", "partition_seed", "dev_fraction",
              "upstream_lock", "seed", "feature_dtype", "schema")
    for name in common:
        if pa.get(name) != pb.get(name):
            raise ValueError(f"fit/dev provenance differs: {name}")
    if not fit.records or not dev.bundles:
        raise ValueError("empty training records or development references")


def validate_training_scope(fit, dev, *, engineering_smoke=False):
    """Enforce the frozen support gate before any formal model optimization."""
    validate_splits(fit, dev)
    rescuable, rescue_scans = set(), set()
    for bundle in fit.bundles:
        ref = bundle["reference"]
        if any(branch["metrics"]["success"] > ref["metrics"]["success"] for branch in bundle["branches"]):
            rescuable.add(ref["instr_id"])
            rescue_scans.add(ref["scan_id"])
    support = {"unique_rescuable_fit_instructions": len(rescuable), "rescuable_fit_scans": len(rescue_scans),
               "required_instructions": 32, "required_scans": 12,
               "passed": len(rescuable) >= 32 and len(rescue_scans) >= 12,
               "scope": "engineering_support_gate_not_statistical_power_or_navigation_performance"}
    if not engineering_smoke:
        if fit.manifest["selection"].get("smoke") or dev.manifest["selection"].get("smoke"):
            raise ValueError("smoke caches require explicit --engineering-smoke")
        if (len(fit.manifest["selection"]["instr_ids"]) not in {512, 2048}
                or len(dev.manifest["selection"]["instr_ids"]) != 128):
            raise ValueError("formal training requires fixed 512/2048 fit and 128 dev instructions")
        if not support["passed"]:
            raise ValueError("formal training lacks 32 unique rescue instructions across 12 fit scenes")
    return support


def build_epoch_plan(records, full_records, seed, epoch):
    """Equal total weight per instruction; old-data repeats preserve update count.

Each instruction gets r slots, where r is the largest full-data state count.
All retained states occur at least once. When a state occurs m times among n
retained states, each occurrence weighs r/(n*m). Thus every instruction's total
weight is exactly r in both data arms; repeated states are never called new data.
"""
    groups, full = defaultdict(list), Counter()
    for index, record in enumerate(records):
        groups[record["instr_id"]].append(index)
    for record in full_records:
        full[record["instr_id"]] += 1
    if not groups or set(groups) != set(full):
        raise ValueError("training arms must cover identical eligible instructions")
    repeats = max(full.values())
    rng = random.Random(int(seed) + 104729 * int(epoch))
    plan = []
    for instr in sorted(groups):
        indices = groups[instr]
        if len(indices) > repeats:
            raise ValueError("retained records exceed the full-data instruction budget")
        drawn = []
        while len(drawn) < repeats:
            cycle = list(indices)
            rng.shuffle(cycle)
            drawn.extend(cycle[:repeats - len(drawn)])
        counts = Counter(drawn)
        plan.extend({"record_index": i, "weight": repeats / (len(indices) * counts[i])} for i in drawn)
    rng.shuffle(plan)
    return plan


def _aggregate(rows):
    n = len(rows)
    successes = sum(row[0] for row in rows)
    baseline_successes = sum(row[2] for row in rows)
    return {"episodes": n, "successes": int(successes), "baseline_successes": int(baseline_successes),
            "sr": successes / n, "spl": sum(row[1] for row in rows) / n,
            "baseline_sr": baseline_successes / n, "baseline_spl": sum(row[3] for row in rows) / n,
            "interventions": sum(row[4] for row in rows)}


def evaluate_cached(model, dataset, sr_threshold, spl_threshold, *, score_cache=None):
    """First causal trigger chooses once; complete branch labels only score it.

This is an offline lookup of already executed continuations, not a new online
navigation run. References with no eligible states remain in the denominator.
"""
    grouped = defaultdict(list)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for bundle_index, bundle in enumerate(dataset.bundles):
                ref = bundle["reference"]
                outcome = (float(ref["metrics"]["success"]), float(ref["metrics"]["spl"]))
                selected = outcome
                intervened = False
                for record_index, record in sorted(enumerate(bundle["records"]), key=lambda item: item[1]["step"]):
                    key = (bundle_index, record_index)
                    if score_cache is not None and key in score_cache:
                        scores = score_cache[key]
                    else:
                        scores = model.score_record(record)
                        if score_cache is not None:
                            # All thresholds must use the same device as online
                            # gating; CPU/GPU softmax can differ in the last bit.
                            score_cache[key] = scores.detach()
                    index = choose_action(scores, model.mode, sr_threshold, spl_threshold)
                    if index != 0:
                        # Access to labels happens only after the action is fixed.
                        action = record["candidate_actions"][index]
                        branches = [branch for branch in bundle["branches"]
                                    if branch["target_step"] == record["step"]
                                    and branch["target_action"] == action]
                        if len(branches) != 1:
                            raise ValueError("selected action needs one complete continuation branch")
                        metrics = branches[0]["metrics"]
                        # Training utilities are float32 tensors. The natural
                        # no-regression gate must use the original evaluator's
                        # precision for both baseline and selected branch.
                        selected = (float(metrics["success"]), float(metrics["spl"]))
                        if (selected[0] not in (0., 1.) or not math.isfinite(selected[1])
                                or not 0. <= selected[1] <= selected[0]):
                            raise ValueError("invalid selected full-continuation metrics")
                        intervened = True
                        break
                grouped[ref["condition"]].append((*selected, *outcome, int(intervened)))
    finally:
        model.train(was_training)
    conditions = {name: _aggregate(rows) for name, rows in sorted(grouped.items())}
    natural = conditions.get("natural")
    if natural is None:
        raise ValueError("development cache must include natural references")
    hard = [conditions[name] for name in ("early_two", "late_three") if name in conditions]
    hard_n = sum(item["episodes"] for item in hard)
    return {"conditions": conditions,
            "eligible": natural["sr"] >= natural["baseline_sr"] - 1e-12
                        and natural["spl"] >= natural["baseline_spl"] - 1e-12,
            "hard_net_successes": sum(item["successes"] - item["baseline_successes"] for item in hard),
            "hard_spl": sum(item["spl"] * item["episodes"] for item in hard) / hard_n if hard_n else 0.,
            "interventions": sum(item["interventions"] for item in conditions.values()),
            "hard_unit": "condition cases in early_two and late_three; not independent original instructions",
            "scope": "offline_cached_complete_continuations_on_train_dev"}


def selection_key(report):
    return (report["hard_net_successes"], report["hard_spl"], -report["interventions"],
            -report["epoch"], report["sr_threshold"], report["spl_threshold"])


class ContinuationTrainer:
    def __init__(self, fit, dev, arm, config):
        if arm not in ARMS:
            raise ValueError("unknown training arm")
        validate_splits(fit, dev)
        self.config = copy.deepcopy(config)
        self.arm = arm
        for key in ("epochs", "batch_size", "hidden_dim", "feature_dim", "checkpoint_every_steps"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (type(config["seed"]) is not int or config["seed"] < 0
                or not config["candidate_epochs"] or config["candidate_epochs"] != sorted(set(config["candidate_epochs"]))
                or config["candidate_epochs"][-1] != config["epochs"]
                or any(type(x) is not int or not 1 <= x <= config["epochs"] for x in config["candidate_epochs"])):
            raise ValueError("fixed candidate epochs must be ordered and include the final epoch")
        for key in ("lr", "sr_weight", "spl_weight", "rescue_weight", "harm_weight"):
            if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"invalid {key}")
        if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
            raise ValueError("invalid weight_decay")
        for key in ("sr_thresholds", "spl_thresholds"):
            values = config[key]
            if not values or values != sorted(set(values)) or any(not math.isfinite(x) or x < 0 for x in values):
                raise ValueError("gate thresholds must be fixed, finite, nonnegative and ordered")
        self.device = torch.device(config["device"])
        if self.device.type not in {"cpu", "cuda"} or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("training needs an available CPU or CUDA device")
        mode, history, old_data = ARMS[arm]
        self.fit, self.dev = fit, dev
        self.full_records = fit.records
        self.records = [r for r in self.full_records if not old_data or r["condition"] in {"natural", "perturb_step2"}]
        self.instruction_record_counts = Counter(r["instr_id"] for r in self.records)
        self.teacher_labeled_counts = Counter(r["instr_id"] for r in self.records if r["teacher_target"] != -1)
        if mode == "teacher" and not self.teacher_labeled_counts:
            raise ValueError("teacher arm has no labeled records")
        self.data_identity = {"fit": fit.identity, "dev": dev.identity,
                              "retained_records": len(self.records), "full_records": len(self.full_records),
                              "records_content_sha256": content_hash(self.records)}
        self.code_identity = code_identity()
        self.runtime_identity = {"torch": str(torch.__version__), "device_type": self.device.type}
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        import numpy as np
        np.random.seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        self.head = ContinuationComparator(config["feature_dim"], config["hidden_dim"], mode, history).to(self.device)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        self.initial_head_sha256 = content_hash(self.head.state_dict())
        self.epoch = self.cursor = self.global_step = 0
        self.optimizer_updates = 0
        self.pending_dev = False
        self.training_history, self.dev_history = [], []
        self.best_selection = None
        self.epoch_loss_sum = self.epoch_weight_sum = 0.
        self.epoch_teacher_labeled = self.epoch_teacher_missing = 0
        self.plan = build_epoch_plan(self.records, self.full_records, config["seed"], self.epoch)

    @property
    def done(self):
        return self.epoch == self.config["epochs"] and not self.pending_dev

    def step(self):
        if self.done or self.pending_dev:
            raise ValueError("no optimizer step allowed while complete or development is pending")
        batch = self.plan[self.cursor:self.cursor + self.config["batch_size"]]
        if not batch:
            raise ValueError("empty optimizer batch")
        self.head.train()
        self.optimizer.zero_grad(set_to_none=True)
        selected_records = []
        effective_weights = []
        for item in batch:
            record = self.records[item["record_index"]]
            if self.head.mode == "teacher" and record["teacher_target"] == -1:
                self.epoch_teacher_missing += 1
                continue
            if self.head.mode == "teacher":
                self.epoch_teacher_labeled += 1
            weight = item["weight"]
            if self.head.mode == "teacher":
                # Excluding missing experts must not underweight instructions
                # which have fewer labeled states. Zero-label instructions are
                # reported separately and cannot provide teacher supervision.
                weight *= self.instruction_record_counts[record["instr_id"]] / self.teacher_labeled_counts[record["instr_id"]]
            selected_records.append(record)
            effective_weights.append(weight)
        if selected_records:
            losses = batch_record_losses(self.head, selected_records, **{
                name: self.config[name] for name in
                ("sr_weight", "spl_weight", "rescue_weight", "harm_weight")})
            numerator = (losses * losses.new_tensor(effective_weights)).sum()
            # A fixed denominator preserves equal instruction weights across
            # the epoch, including the final partial batch.
            denominator = self.config["batch_size"]
            loss = numerator / denominator
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.head.parameters()):
                raise ValueError("nonfinite training gradients")
            self.optimizer.step()
            self.optimizer_updates += 1
            self.epoch_loss_sum += float(numerator.detach())
            self.epoch_weight_sum += sum(effective_weights)
        # A teacher batch with no expert labels has no optimizer update. Its
        # position is still consumed and reported, rather than inventing labels.
        self.cursor += len(batch)
        self.global_step += 1
        if self.cursor == len(self.plan):
            self.epoch += 1
            self.training_history.append({"epoch": self.epoch, "global_step": self.global_step,
                "weighted_loss": self.epoch_loss_sum / self.epoch_weight_sum if self.epoch_weight_sum else None,
                "sample_slots": len(self.plan), "unique_records": len(self.records),
                "repeated_slots": len(self.plan) - len(self.records),
                "instruction_weight": max(Counter(r["instr_id"] for r in self.full_records).values()),
                "teacher_labeled_slots": self.epoch_teacher_labeled,
                "teacher_missing_slots": self.epoch_teacher_missing,
                "teacher_instructions_with_labels": len(self.teacher_labeled_counts),
                "teacher_instructions_without_labels": len(self.instruction_record_counts) - len(self.teacher_labeled_counts),
                "optimizer_updates": self.optimizer_updates})
            self.cursor = 0
            self.epoch_loss_sum = self.epoch_weight_sum = 0.
            self.epoch_teacher_labeled = self.epoch_teacher_missing = 0
            self.pending_dev = self.epoch in self.config["candidate_epochs"]
            self.plan = build_epoch_plan(self.records, self.full_records, self.config["seed"], self.epoch)

    def monitor_dev(self):
        if not self.pending_dev:
            raise ValueError("development evaluation is not pending")
        score_cache = {}
        candidates = []
        for sr in self.config["sr_thresholds"]:
            for spl in self.config["spl_thresholds"]:
                report = evaluate_cached(self.head, self.dev, sr, spl, score_cache=score_cache)
                candidates.append(dict(report, epoch=self.epoch, sr_threshold=sr, spl_threshold=spl))
        self.dev_history.extend(candidates)
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        winner = max(eligible, key=selection_key) if eligible else None
        is_best = winner is not None and (self.best_selection is None or selection_key(winner) > selection_key(self.best_selection))
        if is_best:
            self.best_selection = copy.deepcopy(winner)
        self.pending_dev = False
        return is_best

    def state_dict(self):
        return {"schema": SCHEMA, "arm": self.arm, "config": self.config,
                "data_identity": self.data_identity, "code_identity": self.code_identity,
                "runtime_identity": self.runtime_identity, "initial_head_sha256": self.initial_head_sha256,
                "head": _cpu_copy(self.head.state_dict()), "optimizer": _cpu_copy(self.optimizer.state_dict()),
                "rng": _rng_state(), "epoch": self.epoch, "cursor": self.cursor,
                "global_step": self.global_step, "pending_dev": self.pending_dev,
                "optimizer_updates": self.optimizer_updates,
                "plan": copy.deepcopy(self.plan), "plan_sha256": object_sha256(self.plan),
                "training_history": copy.deepcopy(self.training_history), "dev_history": copy.deepcopy(self.dev_history),
                "best_selection": copy.deepcopy(self.best_selection), "epoch_loss_sum": self.epoch_loss_sum,
                "epoch_weight_sum": self.epoch_weight_sum, "epoch_teacher_labeled": self.epoch_teacher_labeled,
                "epoch_teacher_missing": self.epoch_teacher_missing}

    def load_state_dict(self, state):
        for key, expected in (("schema", SCHEMA), ("arm", self.arm), ("config", self.config),
                              ("data_identity", self.data_identity), ("code_identity", self.code_identity),
                              ("runtime_identity", self.runtime_identity), ("initial_head_sha256", self.initial_head_sha256)):
            if state.get(key) != expected:
                raise ValueError(f"resume identity differs: {key}")
        epoch, cursor, step = state["epoch"], state["cursor"], state["global_step"]
        plan = build_epoch_plan(self.records, self.full_records, self.config["seed"], epoch)
        batches_per_epoch = math.ceil(len(plan) / self.config["batch_size"])
        if (type(epoch) is not int or not 0 <= epoch <= self.config["epochs"]
                or type(cursor) is not int or not 0 <= cursor < len(plan)
                or cursor % self.config["batch_size"] != 0
                or type(step) is not int or step != epoch * batches_per_epoch + cursor // self.config["batch_size"]
                or state["plan"] != plan or state["plan_sha256"] != object_sha256(plan)
                or type(state["pending_dev"]) is not bool
                or (state["pending_dev"] and (epoch not in self.config["candidate_epochs"] or cursor != 0))
                or (epoch == self.config["epochs"] and cursor != 0)
                or len(state["training_history"]) != epoch):
            raise ValueError("invalid resume epoch, permutation or cursor")
        if (type(state["optimizer_updates"]) is not int or not 0 <= state["optimizer_updates"] <= step
                or (self.head.mode != "teacher" and state["optimizer_updates"] != step)):
            raise ValueError("invalid resumed optimizer update count")
        if state["optimizer"]["param_groups"] != self.optimizer.state_dict()["param_groups"]:
            raise ValueError("resume optimizer configuration differs")
        expected = self.head.state_dict()
        if set(state["head"]) != set(expected) or any(not isinstance(state["head"][k], torch.Tensor)
            or state["head"][k].shape != v.shape or not torch.isfinite(state["head"][k]).all()
            for k, v in expected.items()):
            raise ValueError("invalid resumed head")
        eligible = [item for item in state["dev_history"] if item["eligible"]]
        best = max(eligible, key=selection_key) if eligible else None
        if state["best_selection"] != best:
            raise ValueError("resume best selection differs from completed development records")
        self.head.load_state_dict(state["head"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for key in ("epoch", "cursor", "global_step", "pending_dev", "plan", "training_history", "dev_history",
                    "optimizer_updates", "best_selection", "epoch_loss_sum", "epoch_weight_sum",
                    "epoch_teacher_labeled", "epoch_teacher_missing"):
            setattr(self, key, copy.deepcopy(state[key]))
        _restore_rng(state["rng"], device=self.device, seed=self.config["seed"])

    def head_payload(self):
        selection = self.best_selection if self.best_selection is not None and self.best_selection["epoch"] == self.epoch else None
        provenance = {"fit_manifest_sha256": self.fit.identity.get("manifest_sha256", object_sha256(self.fit.identity)),
                      "dev_manifest_sha256": self.dev.identity.get("manifest_sha256", object_sha256(self.dev.identity)),
                      "fit_provenance": self.fit.manifest["provenance"],
                      "dev_provenance": self.dev.manifest["provenance"],
                      "fit_selection": self.fit.manifest["selection"],
                      "dev_selection": self.dev.manifest["selection"],
                      "source_files": self.code_identity, "source_sha256": object_sha256(self.code_identity),
                      "training_config_sha256": object_sha256(self.config)}
        for name in ("base_checkpoint_sha256", "feature_sha256", "train_annotation_sha256",
                     "connectivity_sha256", "asset_pins", "model", "partition_seed", "dev_fraction", "upstream_lock"):
            if name in self.fit.manifest["provenance"]:
                provenance[name] = self.fit.manifest["provenance"][name]
        thresholds = {"sr": selection["sr_threshold"] if selection else max(self.config["sr_thresholds"]),
                      "spl": selection["spl_threshold"] if selection else max(self.config["spl_thresholds"])}
        smoke = (self.config.get("engineering_smoke", False) or self.fit.manifest["selection"].get("smoke", False)
                 or self.dev.manifest["selection"].get("smoke", False))
        provenance["scope"] = "engineering_smoke" if smoke else "training_experiment"
        status = "passed_cached_dev_gate" if selection else (
            "failed_cached_dev_gate_fixed_final_strictest" if self.done and self.best_selection is None else "unselected_snapshot")
        return {"schema": "e3_continuation_head_v2", "arm": self.arm, "model_config": self.head.config,
                "state_dict": _cpu_copy(self.head.state_dict()), "epoch": self.epoch, "global_step": self.global_step,
                "data_identity": self.data_identity, "code_identity": self.code_identity, "config": self.config,
                "provenance": provenance, "thresholds": thresholds, "selection": copy.deepcopy(selection),
                "scope": "engineering_smoke" if smoke else "training_experiment",
                "selection_status": ("engineering_smoke_" if smoke else "") + status}


def train_run(fit, dev, arm, config, local_dir, backup_dir, backup_check, *, require_resume=False,
              stop_after_steps=None, stop_requested=lambda: False):
    """Persist full states only at committed optimizer/evaluation boundaries."""
    trainer = ContinuationTrainer(fit, dev, arm, config)
    started = time.monotonic()
    store = MountedCheckpointStore(local_dir, backup_dir, keep_local=2, keep_backup=5, backup_check=backup_check)
    resumed, last_id = False, None
    with store.lock():
        try:
            state, head, _ = store.restore("latest")
        except FileNotFoundError:
            if require_resume:
                raise ValueError("--require-resume requested but no valid checkpoint exists")
        else:
            if content_hash(head["state_dict"]) != content_hash(state["head"]):
                raise ValueError("training state and portable head differ")
            trainer.load_state_dict(state)
            resumed = True
        invocation_step = trainer.global_step

        def persist(reason, *, best=False):
            nonlocal last_id
            if code_identity() != trainer.code_identity:
                raise ValueError("training source changed while running")
            backup_check()
            last_id = store.save(trainer.state_dict(), trainer.head_payload(), step=trainer.global_step,
                                 is_best=best, metrics={"reason": reason, "epoch": trainer.epoch,
                                                        "pending_dev": trainer.pending_dev})
            return last_id

        if not resumed:
            persist("initial_state")
        while not trainer.done:
            if stop_requested() or (stop_after_steps is not None and trainer.global_step >= stop_after_steps):
                persist("paused_at_complete_update_boundary")
                break
            if trainer.pending_dev:
                best = trainer.monitor_dev()
                persist("candidate_epoch_and_gates_evaluated", best=best)
                continue
            previous_epoch = trainer.epoch
            trainer.step()
            if trainer.epoch != previous_epoch:
                persist("epoch_updates_complete_before_development")
                print(json.dumps({"arm": arm, **trainer.training_history[-1]}), flush=True)
            elif trainer.global_step % config["checkpoint_every_steps"] == 0:
                persist("periodic_update_boundary")
        final, selected = None, None

        def reference(manifest, head):
            return {"checkpoint_id": manifest["checkpoint_id"], "epoch": head["epoch"],
                    "global_step": head["global_step"], "head_sha256": manifest["files"]["head.pt"]["sha256"],
                    "head_relative_path": str(Path("snapshots") / manifest["checkpoint_id"] / "head.pt")}

        if trainer.done:
            persist("fixed_final_complete")
            final_state, final_head, final_manifest = store.restore("latest")
            if not trees_equal(final_state, trainer.state_dict()):
                raise ValueError("final full-state readback differs")
            final = reference(final_manifest, final_head)
            if trainer.best_selection is not None:
                _, best_head, best_manifest = store.restore("best")
                if best_head["selection"] != trainer.best_selection:
                    raise ValueError("durable best differs from selected development gate")
                selected = dict(reference(best_manifest, best_head),
                                sr_threshold=trainer.best_selection["sr_threshold"],
                                spl_threshold=trainer.best_selection["spl_threshold"],
                                status="passed_cached_dev_gate", eligible=True)
            else:
                selected = dict(final, sr_threshold=max(config["sr_thresholds"]),
                                spl_threshold=max(config["spl_thresholds"]),
                                status="failed_cached_dev_gate_fixed_final_strictest", eligible=False)
            selected_source = Path(local_dir) / selected["head_relative_path"]
            verified_copy(selected_source, Path(local_dir) / "selected-head.pt")
            backup_check()
            verified_copy(selected_source, Path(backup_dir) / "selected-head.pt")
            selected["export_file"] = "selected-head.pt"
        report = {"schema": SCHEMA, "status": "complete" if trainer.done else "paused", "resumed": resumed,
                  "arm": arm, "config": config, "global_step": trainer.global_step,
                  "completed_epochs": trainer.epoch, "pending_dev": trainer.pending_dev,
                  "optimizer_updates": trainer.optimizer_updates,
                  "parameter_count": sum(p.numel() for p in trainer.head.parameters()),
                  "initial_head_sha256": trainer.initial_head_sha256, "data_identity": trainer.data_identity,
                  "code_identity": trainer.code_identity, "training_history": trainer.training_history,
                  "dev_history": trainer.dev_history, "best_selection": trainer.best_selection,
                  "final_checkpoint": final, "selected_checkpoint": selected, "latest_checkpoint_id": last_id,
                  "claim": "offline training/development selection only; requires full online navigation validation",
                  "resources": {"this_attempt_seconds": time.monotonic() - started,
                      "batch_boundaries_this_attempt": trainer.global_step - invocation_step,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None,
                      "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if trainer.device.type == "cuda" else None}}
        destination = Path(local_dir) / "training-report.json"
        atomic_json(destination, report)
        backup_check()
        # Reports contain attempt timing and may change on strict resume. Publish
        # a new immutable attempt report instead of overwriting prior evidence.
        verified_copy(destination, Path(backup_dir) / f"training-report-{time.time_ns()}.json")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--arm", choices=[*ARMS, "all"], required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--engineering-smoke", action="store_true",
                        help="allow small engineering caches; exported heads remain marked as smoke")
    parser.add_argument("--stop-after-steps", type=int,
                        help="pause after this absolute committed batch boundary; omit when continuing")
    args = parser.parse_args(argv)
    if args.stop_after_steps is not None and args.stop_after_steps < 0:
        raise ValueError("--stop-after-steps must be nonnegative")
    output, backup = args.output_dir.expanduser().resolve(), args.backup_root.expanduser().resolve()
    validate_separate_roots(output, backup)
    mount = backup_mount_identity(backup, backend="filesystem")
    def check_backup():
        return validate_backup_root(backup, backend="filesystem", expected_identity=mount)
    datasets = []
    for path, split in ((args.fit_cache, "train_fit"), (args.dev_cache, "train_dev")):
        manifest = json.loads((path.expanduser().resolve() / "dataset-manifest.json").read_text())
        data_backup = Path(manifest["backup_root"]).expanduser().resolve()
        data_mount = backup_mount_identity(data_backup, backend="filesystem")
        def check_data(path=data_backup, identity=data_mount):
            return validate_backup_root(path, backend="filesystem", expected_identity=identity)
        datasets.append(load_dataset(path, split, check_backup=check_data))
    config = default_config()
    config["device"] = args.device
    config["engineering_smoke"] = args.engineering_smoke
    config["support_gate"] = validate_training_scope(*datasets, engineering_smoke=args.engineering_smoke)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        if args.arm == "all" and not args.require_resume:
            # Create every arm's durable step-zero state before the first arm
            # trains. A later --arm all --require-resume can then remain strict
            # even when interruption happened before the second arm started.
            for arm in ARMS:
                if stopping:
                    return 75
                train_run(*datasets, arm, config, output / arm, backup / arm, check_backup,
                          stop_after_steps=0, stop_requested=lambda: stopping)
        for arm in ARMS if args.arm == "all" else [args.arm]:
            report = train_run(*datasets, arm, config, output / arm, backup / arm, check_backup,
                               require_resume=args.require_resume, stop_after_steps=args.stop_after_steps,
                               stop_requested=lambda: stopping)
            print(json.dumps({"arm": arm, "status": report["status"],
                              "selected_checkpoint": report["selected_checkpoint"]}), flush=True)
            if report["status"] != "complete":
                return 75
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
