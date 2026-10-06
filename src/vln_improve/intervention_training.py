"""Matched E2 head training with natural-development selection and durable resume.

Cached candidate metrics describe executable final returns. They supervise the
head and development selection, and are never passed to the policy network.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .checkpoint_store import CheckpointStore
from .endpoint_intervention import (
    InterventionHead, InterventionInputs, SCALAR_NAMES, collate_interventions,
    intervention_loss, select_intervention,
)
from .endpoint_pairs import content_hash
from .protocol import file_sha256, object_sha256
from .resumable import _cpu_copy, _restore_rng, _rng_state

TRAIN_SCHEMA = "e2_intervention_training_v1"
HEAD_SCHEMA = "e2_intervention_head_v1"
BEST_RULE = "natural_train_dev_SR_and_SPL_at_least_baseline_then_SR_SPL_earliest"
ARMS = ("relative", "absolute")
METRIC_TOLERANCE = 1e-12


def _metric_better(candidate, previous):
    if candidate[0] > previous[0] + METRIC_TOLERANCE:
        return True
    return abs(candidate[0] - previous[0]) <= METRIC_TOLERANCE and candidate[1] > previous[1] + METRIC_TOLERANCE


class AbsoluteUtilityHead(InterventionHead):
    """Identical trainable capacity; estimate absolute SR/SPL before subtraction."""

    def forward(self, batch):
        nodes, context, scalars = batch.node_features, batch.terminal_context, batch.scalar_features
        mask, anchors = batch.valid_mask, batch.baseline_indices
        b, n, _ = nodes.shape
        # Same validation and architecture as InterventionHead; absolute
        # outputs retain the original endpoint because it needs supervision.
        if (nodes.shape != (b, n, 768) or context.shape != (b, 1536)
                or scalars.shape != (b, n, len(SCALAR_NAMES)) or b == 0 or n == 0
                or mask.shape != (b, n) or mask.dtype != torch.bool
                or anchors.shape != (b,) or anchors.dtype != torch.int64
                or (anchors < 0).any() or (anchors >= n).any()
                or any(t.device != nodes.device for t in (context, scalars, mask, anchors))
                or nodes.device != self.comparison[-1].weight.device
                or not mask[torch.arange(b, device=nodes.device), anchors].all()
                or not torch.isfinite(nodes[mask]).all() or not torch.isfinite(scalars[mask]).all()
                or not torch.isfinite(context).all()):
            raise ValueError("invalid absolute utility batch")
        dtype = self.comparison[-1].weight.dtype
        h = self.node_encoder(nodes.detach().masked_fill(~mask[..., None], 0).to(dtype))
        anchor = h[torch.arange(len(h), device=h.device), anchors][:, None].expand_as(h)
        context = self.context_encoder(context.detach().to(dtype))[:, None].expand_as(h)
        scalars = scalars.detach().masked_fill(~mask[..., None], 0).to(dtype)
        result = self.comparison(torch.cat((h, anchor, h - anchor, context, scalars), -1)).sigmoid()
        return result.masked_fill(~mask[..., None], 0)


def make_head(arm, hidden_dim=128):
    if arm not in ARMS:
        raise ValueError("unknown intervention training arm")
    model = (InterventionHead if arm == "relative" else AbsoluteUtilityHead)(hidden_dim)
    model.arm = arm
    model.hidden_dim = hidden_dim
    return model


def predict_gains(model, batch, arm=None):
    arm = arm or model.arm
    if arm not in ARMS or getattr(model, "arm", arm) != arm:
        raise ValueError("prediction arm differs from trained head")
    prediction = model(batch)
    if arm == "absolute":
        baseline = prediction[torch.arange(len(prediction), device=prediction.device), batch.baseline_indices]
        prediction = prediction - baseline[:, None]
    anchor_mask = torch.arange(prediction.shape[1], device=prediction.device)[None] == batch.baseline_indices[:, None]
    return prediction.masked_fill((~batch.valid_mask | anchor_mask)[..., None], 0)


def load_head(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != HEAD_SCHEMA:
        raise ValueError("not an E2 intervention head")
    model = make_head(payload["arm"], payload["hidden_dim"])
    expected, weights = model.state_dict(), payload.get("state_dict")
    if (not isinstance(weights, dict) or set(weights) != set(expected)
            or any(not isinstance(v, torch.Tensor) or v.shape != expected[k].shape
                   or v.dtype != expected[k].dtype or not torch.isfinite(v).all() for k, v in weights.items())):
        raise ValueError("invalid E2 head weights")
    model.load_state_dict(weights, strict=True)
    model.training_metadata = {k: copy.deepcopy(v) for k, v in payload.items() if k != "state_dict"}
    return model.to(device).eval()


def record_inputs(record):
    value = record["inputs"]
    return InterventionInputs(tuple(value["candidate_vpids"]), value["baseline_index"],
        value["node_features"], value["terminal_context"], value["scalar_features"])


def validate_records(records, *, natural_only=False):
    if not records:
        raise ValueError("empty intervention cache")
    seen = set()
    for record in records:
        identity = (record.get("instr_id"), record.get("condition"))
        if (record.get("schema") != "e2_intervention_record_v1" or identity in seen
                or not all(isinstance(record.get(k), str) and record[k] for k in ("instr_id", "scan_id"))
                or record.get("condition") not in {"natural", "perturb_step2"}
                or (natural_only and record["condition"] != "natural")):
            raise ValueError("invalid, duplicate, or nonnatural intervention record")
        seen.add(identity)
        item = record_inputs(record)
        collate_interventions([item])
        n, anchor = len(item.candidate_vpids), item.baseline_index
        targets, utilities = record.get("targets"), record.get("utilities")
        if (not isinstance(utilities, torch.Tensor) or not isinstance(targets, torch.Tensor)
                or utilities.shape != (n, 2) or targets.shape != (n, 2)
                or not utilities.is_floating_point() or not targets.is_floating_point()
                or utilities.device.type != "cpu" or targets.device.type != "cpu"
                or not torch.isfinite(utilities).all() or not torch.isfinite(targets).all()
                or (utilities < 0).any() or (utilities > 1).any()
                or not ((utilities[:, 0] == 0) | (utilities[:, 0] == 1)).all()
                or not torch.equal(targets, utilities - utilities[anchor])
                or record.get("baseline_endpoint") != item.candidate_vpids[anchor]):
            raise ValueError("candidate utility/relative targets/original endpoint disagree")
        metrics = record.get("candidate_metrics")
        if (not isinstance(metrics, list) or len(metrics) != n
                or any(float(m["success"]) != float(u[0]) or float(m["spl"]) != float(u[1])
                       for m, u in zip(metrics, utilities))):
            raise ValueError("candidate replay metrics disagree with supervision")


def validate_splits(train, dev):
    validate_records(train)
    validate_records(dev, natural_only=True)
    for key in ("instr_id", "scan_id"):
        if {r[key] for r in train} & {r[key] for r in dev}:
            raise ValueError(f"fit/dev {key} leakage")


def prepare_batch(records, device):
    batch = collate_interventions([record_inputs(r) for r in records], device=device)
    targets = torch.zeros((*batch.valid_mask.shape, 2), dtype=torch.float64, device=device)
    utilities = torch.zeros_like(targets)
    for index, record in enumerate(records):
        n = len(record["inputs"]["candidate_vpids"])
        targets[index, :n] = record["targets"].to(device)
        utilities[index, :n] = record["utilities"].to(device)
    return batch, targets, utilities


def absolute_loss(prediction, utilities, valid_mask):
    if (prediction.shape != utilities.shape or valid_mask.shape != prediction.shape[:2]
            or not torch.isfinite(prediction[valid_mask]).all()
            or not torch.isfinite(utilities[valid_mask]).all() or not valid_mask.any(-1).all()):
        raise ValueError("invalid absolute utility loss")
    prediction = prediction.masked_fill(~valid_mask[..., None], 0)
    utilities = utilities.detach().to(prediction.dtype).masked_fill(~valid_mask[..., None], 0)
    element = F.smooth_l1_loss(prediction, utilities, reduction="none").mean(-1)
    return ((element * valid_mask).sum(-1) / valid_mask.sum(-1)).mean()


def _summarize_rows(rows):
    n = len(rows)
    return {"episodes": n, "sr": sum(r["success"] for r in rows) / n,
        "spl": sum(r["spl"] for r in rows) / n,
        "ndtw": sum(r["ndtw"] for r in rows) / n if all(r["ndtw"] is not None for r in rows) else None,
        "baseline_sr": sum(r["baseline_success"] for r in rows) / n,
        "baseline_spl": sum(r["baseline_spl"] for r in rows) / n,
        "changes": sum(r["changed"] for r in rows),
        "rescues": sum(r["success"] > r["baseline_success"] for r in rows),
        "harms": sum(r["success"] < r["baseline_success"] for r in rows)}


@torch.no_grad()
def evaluate_records(model, records, *, batch_size=1, device="cpu"):
    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        subset = records[start:start + batch_size]
        batch, _, _ = prepare_batch(subset, device)
        gains = predict_gains(model, batch).cpu()
        for i, record in enumerate(subset):
            ids = record["inputs"]["candidate_vpids"]
            anchor = record["inputs"]["baseline_index"]
            endpoint = select_intervention(gains[i], ids, ids[anchor], valid_mask=batch.valid_mask[i].cpu())
            index = ids.index(endpoint)
            chosen, baseline = record["candidate_metrics"][index], record["candidate_metrics"][anchor]
            rows.append({"instr_id": record["instr_id"], "scan_id": record["scan_id"],
                "endpoint": endpoint, "baseline_endpoint": ids[anchor], "changed": index != anchor,
                "success": float(chosen["success"]), "spl": float(chosen["spl"]),
                "ndtw": float(chosen["nDTW"]) if "nDTW" in chosen else None,
                "baseline_success": float(baseline["success"]), "baseline_spl": float(baseline["spl"]),
                "predicted_gain": gains[i, index].tolist()})
    summary = _summarize_rows(rows)
    summary["eligible"] = (summary["sr"] >= summary["baseline_sr"] - METRIC_TOLERANCE
                           and summary["spl"] >= summary["baseline_spl"] - METRIC_TOLERANCE)
    summary["research_success"] = (summary["eligible"] and summary["changes"] > 0
        and (summary["sr"] > summary["baseline_sr"] + METRIC_TOLERANCE
             or summary["spl"] > summary["baseline_spl"] + METRIC_TOLERANCE))
    scenes = sorted({r["scan_id"] for r in rows})
    return {**summary, "by_scene": {s: _summarize_rows([r for r in rows if r["scan_id"] == s]) for s in scenes},
        "per_episode": rows}


def code_identity():
    here = Path(__file__).resolve().parent
    names = ("intervention_training.py", "endpoint_intervention.py", "checkpoint_store.py",
             "resumable.py", "protocol.py", "endpoint_pairs.py", "intervention_runtime.py")
    result = {"src/vln_improve/" + name: file_sha256(here / name) for name in names}
    result["scripts/train_endpoint_intervention.py"] = file_sha256(here.parents[1] / "scripts/train_endpoint_intervention.py")
    return result


def normalized_config(config):
    result = {"arm": "relative", "seed": 0, "epochs": 10, "batch_size": 64,
        "hidden_dim": 128, "lr": 1e-4, "weight_decay": .01,
        "monitor_every_epochs": 1, "risk_weight": 0., "experiment_sha256": None,
        "inference_batch_size": 1}
    if set(config) - set(result):
        raise ValueError("unknown fixed training config keys")
    result.update(config)
    if result["arm"] not in ARMS:
        raise ValueError("invalid intervention arm")
    for key in ("epochs", "batch_size", "hidden_dim", "monitor_every_epochs"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"invalid {key}")
    if (type(result["seed"]) is not int or result["seed"] < 0 or result["hidden_dim"] > 128
            or result["risk_weight"] != 0 or result["inference_batch_size"] != 1):
        raise ValueError("invalid seed/width or unmatched risk objective")
    for key in ("lr", "weight_decay"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"invalid {key}")
    if result["lr"] == 0:
        raise ValueError("zero learning rate")
    return result


class InterventionTrainer:
    def __init__(self, train, dev, config, *, data_identity, device="cpu"):
        validate_splits(train, dev)
        self.train, self.dev = train, dev
        self.config = normalized_config(config)
        self.data_identity = copy.deepcopy(data_identity)
        self.code_identity = code_identity()
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("training device must be CPU or CUDA")
        seed = self.config["seed"]
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        self.head = make_head(self.config["arm"], self.config["hidden_dim"]).to(self.device)
        self.initial_head_sha256 = content_hash(self.head.state_dict())
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.epoch = self.cursor = self.global_step = 0
        self.pending_dev = False
        self.epoch_loss_sum = 0.
        self.condition_sums = {}
        self.training_history, self.dev_history = [], []
        self.best_epoch = self.best_metrics = None

    @property
    def batches_per_epoch(self):
        return math.ceil(len(self.train) / self.config["batch_size"])

    @property
    def done(self):
        return self.epoch == self.config["epochs"] and not self.pending_dev

    def order(self, epoch):
        generator = torch.Generator().manual_seed(self.config["seed"] + 1000003 * epoch)
        return torch.randperm(len(self.train), generator=generator).tolist()

    def step(self):
        if self.done or self.pending_dev:
            raise ValueError("training awaits dev or is complete")
        indices = self.order(self.epoch)[self.cursor:self.cursor + self.config["batch_size"]]
        batch, targets, utilities = prepare_batch([self.train[i] for i in indices], self.device)
        self.head.train(); self.optimizer.zero_grad(set_to_none=True)
        prediction = self.head(batch)
        loss = (intervention_loss(prediction, targets, batch.valid_mask, batch.baseline_indices)["loss"]
                if self.config["arm"] == "relative" else absolute_loss(prediction, utilities, batch.valid_mask))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.head.parameters()):
            raise ValueError("missing/nonfinite gradient")
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.head.parameters()):
            raise ValueError("nonfinite optimizer output")
        self.epoch_loss_sum += float(loss.detach()) * len(indices)
        with torch.no_grad():
            for row, index in enumerate(indices):
                mask = batch.valid_mask[row].clone()
                target = utilities[row] if self.config["arm"] == "absolute" else targets[row]
                if self.config["arm"] == "relative":
                    mask[batch.baseline_indices[row]] = False
                value = (float(F.smooth_l1_loss(prediction[row][mask], target[mask].to(prediction.dtype)))
                         if bool(mask.any()) else 0.)
                total = self.condition_sums.setdefault(self.train[index]["condition"], {"loss_sum": 0., "episodes": 0})
                total["loss_sum"] += value; total["episodes"] += 1
        self.cursor += len(indices); self.global_step += 1
        if self.cursor == len(self.train):
            self.training_history.append({"epoch": self.epoch + 1, "global_step": self.global_step,
                "episodes": self.cursor, "loss": self.epoch_loss_sum / self.cursor,
                "order_sha256": object_sha256(self.order(self.epoch)),
                "by_condition": {k: {"episodes": v["episodes"], "loss": v["loss_sum"] / v["episodes"]}
                                 for k, v in self.condition_sums.items()}})
            self.epoch += 1; self.cursor = 0; self.epoch_loss_sum = 0.
            self.condition_sums = {}
            self.pending_dev = self.epoch % self.config["monitor_every_epochs"] == 0 or self.epoch == self.config["epochs"]
        return float(loss.detach())

    def monitor_dev(self):
        if not self.pending_dev:
            raise ValueError("no fixed development monitor pending")
        metrics = evaluate_records(self.head, self.dev, batch_size=self.config["inference_batch_size"], device=self.device)
        record = {"epoch": self.epoch, "global_step": self.global_step, **metrics}
        self.dev_history.append(record)
        key = (metrics["sr"], metrics["spl"])
        best = metrics["eligible"] and (self.best_metrics is None or _metric_better(key, self.best_metrics))
        if best:
            self.best_epoch, self.best_metrics = self.epoch, list(key)
        self.pending_dev = False
        return record, best

    def state_dict(self):
        return {"schema": TRAIN_SCHEMA, "config": copy.deepcopy(self.config),
            "data_identity": copy.deepcopy(self.data_identity), "code_identity": copy.deepcopy(self.code_identity),
            "initial_head_sha256": self.initial_head_sha256,
            "head": _cpu_copy(self.head.state_dict()), "optimizer": _cpu_copy(self.optimizer.state_dict()),
            "rng": _rng_state(), "scheduler": None, "scaler": None,
            "epoch": self.epoch, "cursor": self.cursor, "global_step": self.global_step,
            "pending_dev": self.pending_dev, "epoch_loss_sum": self.epoch_loss_sum,
            "condition_sums": copy.deepcopy(self.condition_sums),
            "training_history": copy.deepcopy(self.training_history), "dev_history": copy.deepcopy(self.dev_history),
            "best_epoch": self.best_epoch, "best_metrics": copy.deepcopy(self.best_metrics),
            "current_order_sha256": object_sha256(self.order(self.epoch))}

    def load_state_dict(self, saved):
        if (saved.get("schema") != TRAIN_SCHEMA or saved.get("config") != self.config
                or saved.get("data_identity") != self.data_identity or saved.get("code_identity") != self.code_identity
                or saved.get("initial_head_sha256") != self.initial_head_sha256
                or saved.get("scheduler") is not None or saved.get("scaler") is not None):
            raise ValueError("resume identity/config differs")
        epoch, cursor, step = (saved[k] for k in ("epoch", "cursor", "global_step"))
        if (any(type(v) is not int for v in (epoch, cursor, step)) or not 0 <= epoch <= self.config["epochs"]
                or not 0 <= cursor < len(self.train) or cursor % self.config["batch_size"]
                or (epoch == self.config["epochs"] and cursor)
                or step != epoch * self.batches_per_epoch + cursor // self.config["batch_size"]
                or saved.get("current_order_sha256") != object_sha256(self.order(epoch))):
            raise ValueError("invalid resume cursor/sampler")
        pending = saved.get("pending_dev")
        monitored = [i for i in range(1, epoch + 1) if i % self.config["monitor_every_epochs"] == 0 or i == self.config["epochs"]]
        if (type(pending) is not bool or (pending and (not monitored or monitored[-1] != epoch or cursor))
                or len(saved["training_history"]) != epoch
                or [r["epoch"] for r in saved["dev_history"]] != (monitored[:-1] if pending else monitored)):
            raise ValueError("invalid pending monitor/history")
        for index, record in enumerate(saved["training_history"], 1):
            if (record["epoch"] != index or record["global_step"] != index * self.batches_per_epoch
                    or record["episodes"] != len(self.train) or not math.isfinite(record["loss"])
                    or record["order_sha256"] != object_sha256(self.order(index - 1))):
                raise ValueError("invalid training history")
        eligible = [r for r in saved["dev_history"] if r["eligible"]]
        best = None
        for record in eligible:
            if best is None or _metric_better((record["sr"], record["spl"]), (best["sr"], best["spl"])):
                best = record
        if (saved["best_epoch"] != (best["epoch"] if best else None)
                or saved["best_metrics"] != ([best["sr"], best["spl"]] if best else None)
                or not math.isfinite(saved["epoch_loss_sum"]) or saved["epoch_loss_sum"] < 0
                or (cursor == 0 and saved["epoch_loss_sum"] != 0)):
            raise ValueError("invalid best/partial epoch state")
        conditions = saved.get("condition_sums")
        if (not isinstance(conditions, dict) or set(conditions) - {"natural", "perturb_step2"}
                or any(set(v) != {"loss_sum", "episodes"} or type(v["episodes"]) is not int
                       or v["episodes"] < 1 or not math.isfinite(v["loss_sum"]) or v["loss_sum"] < 0
                       for v in conditions.values())
                or sum(v["episodes"] for v in conditions.values()) != cursor):
            raise ValueError("invalid partial condition losses")
        weights, expected = saved["head"], self.head.state_dict()
        if (set(weights) != set(expected) or any(v.shape != expected[k].shape or v.dtype != expected[k].dtype
                or not torch.isfinite(v).all() for k, v in weights.items())):
            raise ValueError("invalid restored head")
        optimizer = saved["optimizer"]
        if optimizer.get("param_groups") != self.optimizer.state_dict()["param_groups"]:
            raise ValueError("restored AdamW configuration differs")
        moments = optimizer.get("state")
        parameters = list(self.head.parameters())
        if not isinstance(moments, dict) or set(moments) != (set(range(len(parameters))) if step else set()):
            raise ValueError("missing AdamW state")
        for index, item in moments.items():
            if set(item) != {"step", "exp_avg", "exp_avg_sq"} or float(item["step"]) != step:
                raise ValueError("AdamW step differs from cursor")
            for key in ("exp_avg", "exp_avg_sq"):
                if (item[key].shape != parameters[index].shape or item[key].dtype != parameters[index].dtype
                        or not torch.isfinite(item[key]).all() or (key == "exp_avg_sq" and (item[key] < 0).any())):
                    raise ValueError("invalid AdamW moments")
        self.head.load_state_dict(weights); self.optimizer.load_state_dict(optimizer)
        _restore_rng(saved["rng"], device=self.device, seed=self.config["seed"])
        for name in ("epoch", "cursor", "global_step", "pending_dev", "epoch_loss_sum", "training_history",
                     "dev_history", "best_epoch", "best_metrics", "condition_sums"):
            setattr(self, name, copy.deepcopy(saved[name]))

    def head_payload(self):
        return {"schema": HEAD_SCHEMA, "arm": self.config["arm"], "hidden_dim": self.config["hidden_dim"],
            "state_dict": _cpu_copy(self.head.state_dict()), "epoch": self.epoch, "global_step": self.global_step,
            "config": copy.deepcopy(self.config), "data_identity": copy.deepcopy(self.data_identity),
            "code_identity": copy.deepcopy(self.code_identity), "selection_rule": BEST_RULE,
            "config_sha256": self.config["experiment_sha256"],
            "initial_head_sha256": self.initial_head_sha256}


def train_intervention(train, dev, config, *, data_identity, local_dir, backup_dir, device="cpu",
                       checkpoint_every_steps=50, checkpoint_every_seconds=60., keep_local=2, keep_backup=5,
                       verify_backup=None, should_stop=None, stop_after_updates=None):
    """Resume only committed optimizer boundaries; dev best is immediately durable."""
    if (type(checkpoint_every_steps) is not int or checkpoint_every_steps < 1
            or not math.isfinite(checkpoint_every_seconds) or checkpoint_every_seconds <= 0
            or (stop_after_updates is not None and (type(stop_after_updates) is not int or stop_after_updates < 0))):
        raise ValueError("invalid checkpoint or interruption interval")
    checker, stopping = verify_backup or (lambda: None), should_stop or (lambda: False)
    trainer = InterventionTrainer(train, dev, config, data_identity=data_identity, device=device)
    checker()
    store = CheckpointStore(local_dir, backup_dir, keep_local=keep_local, keep_backup=keep_backup)
    started, resumed = time.monotonic(), False
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with store.lock():
        try:
            state, head, _ = store.restore("latest")
        except FileNotFoundError:
            pass
        else:
            trainer.load_state_dict(state)
            if content_hash(head) != content_hash(trainer.head_payload()):
                raise ValueError("restored head/state mismatch")
            resumed = True
        invocation_step = trainer.global_step
        last_step, last_saved, latest_id = trainer.global_step, time.monotonic(), None

        def persist(reason, *, best=False, metrics=None):
            nonlocal last_step, last_saved, latest_id
            checker()
            detail = {"reason": reason, "epoch": trainer.epoch, "arm": trainer.config["arm"], "selection_rule": BEST_RULE}
            if metrics is not None:
                detail.update({k: metrics[k] for k in ("sr", "spl", "eligible", "changes", "research_success")})
            latest_id = store.save(trainer.state_dict(), trainer.head_payload(), step=trainer.global_step,
                                   is_best=best, metrics=detail)
            last_step, last_saved = trainer.global_step, time.monotonic()

        if not resumed:
            persist("initialized_not_selectable")
        interrupted = False
        try:
            while not trainer.done:
                if stopping() or (stop_after_updates is not None and trainer.global_step - invocation_step >= stop_after_updates):
                    persist("interrupted_at_complete_optimizer_boundary")
                    interrupted = True
                    break
                if trainer.pending_dev:
                    metrics, best = trainer.monitor_dev()
                    persist("natural_train_dev_monitored", best=best, metrics=metrics)
                    continue
                trainer.step()
                if (trainer.pending_dev or trainer.global_step - last_step >= checkpoint_every_steps
                        or time.monotonic() - last_saved >= checkpoint_every_seconds):
                    persist("development_monitor_pending" if trainer.pending_dev else "periodic")
        except KeyboardInterrupt:
            return {"status": "interrupted", "resumed": resumed, "last_durable_step": last_step,
                    "arm": trainer.config["arm"], "reason": "interrupt_inside_unknown_optimizer_boundary"}
        final = selected = None
        if trainer.done:
            persist("fixed_final_epoch_complete")
            _, final_head, final_manifest = store.restore("latest")
            if content_hash(final_head) != content_hash(trainer.head_payload()):
                raise ValueError("final checkpoint read-back mismatch")

            def reference(manifest, payload):
                return {"checkpoint_id": manifest["checkpoint_id"], "epoch": payload["epoch"],
                    "global_step": payload["global_step"], "head_sha256": manifest["files"]["head.pt"]["sha256"],
                    "head_relative_path": str(Path("snapshots") / manifest["checkpoint_id"] / "head.pt")}

            final = reference(final_manifest, final_head)
            if trainer.best_epoch is not None:
                _, selected_head, selected_manifest = store.restore("best")
                if selected_head["epoch"] != trainer.best_epoch:
                    raise ValueError("durable best differs from development selection")
                selected = {**reference(selected_manifest, selected_head), "selection_reason": "eligible_natural_train_dev_best"}
            else:
                selected = {**final, "selection_reason": "dev_gate_failed_final_exploratory"}
        return {"schema": TRAIN_SCHEMA, "status": "interrupted" if interrupted else "complete", "resumed": resumed,
            "arm": trainer.config["arm"], "config": trainer.config, "global_step": trainer.global_step,
            "completed_epochs": trainer.epoch, "pending_dev": trainer.pending_dev,
            "parameter_count": sum(p.numel() for p in trainer.head.parameters()),
            "initial_head_sha256": trainer.initial_head_sha256, "data_identity": trainer.data_identity,
            "code_identity": trainer.code_identity, "selection_rule": BEST_RULE,
            "final_checkpoint": final, "selected_checkpoint": selected, "best_epoch": trainer.best_epoch,
            "best_metrics": trainer.best_metrics, "latest_checkpoint_id": latest_id,
            "training_history": trainer.training_history, "dev_history": trainer.dev_history,
            "resources": {"wall_seconds_this_invocation": time.monotonic() - started,
                "optimizer_updates_this_invocation": trainer.global_step - invocation_step,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if trainer.device.type == "cuda" else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if trainer.device.type == "cuda" else None}}
