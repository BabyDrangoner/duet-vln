"""Ordinary endpoint BCE control on frozen, language-conditioned STOP tokens.

The policy functions accept features only. Goal distances are used to validate
training labels, never to construct features or choose navigation actions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .checkpoint_store import CheckpointStore
from .protocol import file_sha256, object_sha256
from .resumable import _cpu_copy, _restore_rng, _rng_state


FEATURE_DIM = 1536
FEATURE_SCHEMA = "concat_global_local_stop_crossmodal_v1"
CACHE_SCHEMA = "duet_endpoint_features_v1"
HEAD_SCHEMA = "duet_endpoint_probe_head_v1"
TRAIN_SCHEMA = "duet_endpoint_probe_training_v1"
COMMON_KEYS = {"base_checkpoint_sha256", "feature_sha256", "annotation_sha256",
               "connectivity_sha256", "model", "upstream_lock", "partition_seed",
               "dev_fraction", "torch_version"}


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid {name} SHA-256")
    return value


def _json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected ordinary file: {path}")
    return json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError(f"nonfinite JSON constant: {x}")))


def build_endpoint_features(nav_inputs: dict, nav_outs: dict) -> Tensor:
    """Copy two cross-modal STOP tokens to detached FP32 ``[B,1536]``.

    No simulator observation, label, distance-to-goal, or future input is read.
    The original device is retained so the caller controls CPU transfer.
    """
    global_embeds, local_embeds = nav_outs["gmap_embeds"], nav_outs["vp_embeds"]
    if (not isinstance(global_embeds, Tensor) or not isinstance(local_embeds, Tensor)
            or global_embeds.ndim != 3 or local_embeds.ndim != 3
            or global_embeds.shape[0] < 1 or global_embeds.shape[1] < 1 or local_embeds.shape[1] < 1
            or global_embeds.shape[0] != local_embeds.shape[0]
            or global_embeds.shape[2] != 768 or local_embeds.shape[2] != 768
            or not global_embeds.is_floating_point() or not local_embeds.is_floating_point()
            or global_embeds.device != local_embeds.device):
        raise ValueError("cross-modal embeddings must be floating [B,N,768] and [B,P,768]")
    for key, embeds in (("gmap_vpids", global_embeds), ("vp_cand_vpids", local_embeds)):
        ids = nav_inputs[key]
        if (not isinstance(ids, (list, tuple)) or len(ids) != embeds.shape[0]
                or any(not isinstance(row, (list, tuple)) or not row or row[0] is not None
                       or len(row) > embeds.shape[1] for row in ids)):
            raise ValueError(f"{key} must identify STOP at index zero")
    for key, embeds in (("gmap_masks", global_embeds), ("vp_masks", local_embeds)):
        mask = nav_inputs[key]
        if (not isinstance(mask, Tensor) or mask.dtype != torch.bool
                or mask.shape != embeds.shape[:2] or not bool(mask[:, 0].all())):
            raise ValueError(f"{key} must include the STOP token")
    features = torch.cat((global_embeds[:, 0].detach(), local_embeds[:, 0].detach()), dim=-1).float()
    if not torch.isfinite(features).all():
        raise ValueError("endpoint STOP embeddings must be finite")
    return features


class EndpointProbe(nn.Module):
    """Fixed 1536 -> 128 -> ReLU -> 1 head; output is an endpoint logit."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, 128), nn.ReLU(), nn.Linear(128, 1))

    @property
    def config(self):
        return {"feature_dim": FEATURE_DIM, "hidden_dim": 128, "activation": "relu"}

    def forward(self, features: Tensor) -> Tensor:
        if (not isinstance(features, Tensor) or features.ndim != 2 or features.shape[0] < 1
                or features.shape[1] != FEATURE_DIM or not features.is_floating_point()
                or not torch.isfinite(features).all()):
            raise ValueError("endpoint features must be finite floating [states,1536]")
        if features.device != self.network[0].weight.device:
            raise ValueError("endpoint features and head must use the same device")
        # A frozen feature interface remains safe if a caller accidentally passes
        # a tensor with a backbone autograd graph attached.
        logits = self.network(features.detach().to(self.network[0].weight.dtype)).squeeze(-1)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite endpoint prediction")
        return logits


def endpoint_probabilities(head: EndpointProbe, features: Tensor) -> Tensor:
    """Label-free inference; history ranking may use logits to avoid saturation."""
    with torch.no_grad():
        return head(features).sigmoid()


def _association(value):
    if (not isinstance(value, dict) or set(value) != {"episode_id", "scan_id", "instr_id"}
            or any(not isinstance(x, str) or not x for x in value.values())):
        raise ValueError("invalid endpoint episode association")


def validate_endpoint_episode(payload: dict, item: dict, identity_sha256: str) -> None:
    required = {"schema", "identity_sha256", "association", "input_manifest_sha256", "features",
                "steps", "viewpoints", "distance_to_goal", "labels", "base_stop_probability"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("endpoint episode schema fields differ")
    if payload["schema"] != CACHE_SCHEMA or payload["identity_sha256"] != identity_sha256:
        raise ValueError("endpoint episode belongs to a different cache")
    _association(payload["association"])
    if payload["association"] != item["association"]:
        raise ValueError("endpoint episode association differs from its manifest")
    _sha(payload["input_manifest_sha256"], "source episode manifest")
    features, labels, steps = (payload[k] for k in ("features", "labels", "steps"))
    distance, probability = (payload[k] for k in ("distance_to_goal", "base_stop_probability"))
    if not all(isinstance(x, Tensor) and x.device.type == "cpu"
               for x in (features, labels, steps, distance, probability)):
        raise ValueError("endpoint cache tensors must be CPU tensors")
    n = item["num_states"]
    if type(n) is not int or not 1 <= n <= 15:
        raise ValueError("endpoint episode must contain 1 to 15 complete decision states")
    if (features.dtype != torch.float32 or features.shape != (n, FEATURE_DIM)
            or labels.dtype != torch.float32 or labels.shape != (n,)
            or steps.dtype != torch.int64 or not torch.equal(steps, torch.arange(n))
            or distance.dtype != torch.float64 or distance.shape != (n,)
            or probability.dtype != torch.float32 or probability.shape != (n,)):
        raise ValueError("endpoint cache tensor shape/dtype or consecutive steps mismatch")
    if not all(torch.isfinite(x).all() for x in (features, labels, distance, probability)):
        raise ValueError("nonfinite endpoint cache value")
    if (not ((labels == 0) | (labels == 1)).all() or (distance < 0).any()
            or not torch.equal(labels, (distance < 3).float())
            or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("endpoint labels/probabilities disagree with their definitions")
    viewpoints = payload["viewpoints"]
    if (not isinstance(viewpoints, list) or len(viewpoints) != n
            or any(not isinstance(x, str) or not x for x in viewpoints)
            or len(set(viewpoints)) != n):
        raise ValueError("expected unique, real decision viewpoints")
    if type(item["positives"]) is not int or item["positives"] != int(labels.sum()):
        raise ValueError("endpoint positive count differs from manifest")


@dataclass(frozen=True)
class EndpointCache:
    root: Path
    manifest: dict
    manifest_sha256: str
    data_sha256: str
    episodes: tuple[dict, ...]

    @property
    def split(self):
        return self.manifest["split"]

    @property
    def common_identity(self):
        identity = self.manifest["identity"]
        return {key: copy.deepcopy(identity[key]) for key in
                ("runtime", "common_provenance", "feature_schema", "feature_dim", "implementation")}


def load_endpoint_cache(directory: str | Path, *, expected_split: str) -> EndpointCache:
    """Read a committed whole cache, validating bytes, labels and associations."""
    if expected_split not in {"train_fit", "train_dev"}:
        raise ValueError("endpoint training loader permits only train_fit/train_dev")
    root = Path(directory).resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    digest = file_sha256(manifest_path)
    committed = _json(root / "COMMITTED.json")
    if committed != {"manifest_sha256": digest}:
        raise ValueError("endpoint manifest is uncommitted or its checksum changed")
    usage = "training" if expected_split == "train_fit" else "analysis_only"
    if (manifest.get("schema") != CACHE_SCHEMA or manifest.get("feature_dim") != FEATURE_DIM
            or manifest.get("split") != expected_split or manifest.get("usage") != usage):
        raise ValueError("endpoint cache schema/split/usage mismatch")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or manifest.get("identity_sha256") != object_sha256(identity):
        raise ValueError("endpoint cache identity checksum mismatch")
    if _json(root / "IDENTITY.json") != identity:
        raise ValueError("endpoint IDENTITY.json mismatch")
    if (identity.get("schema") != "duet_endpoint_features_identity_v1"
            or identity.get("split") != expected_split or identity.get("usage") != usage
            or identity.get("feature_schema") != FEATURE_SCHEMA or identity.get("feature_dim") != FEATURE_DIM):
        raise ValueError("endpoint feature identity differs from the manifest")
    _sha(identity.get("collection_identity_sha256"), "source collection")
    common, runtime = identity.get("common_provenance"), identity.get("runtime")
    if (not isinstance(common, dict) or set(common) != COMMON_KEYS or not isinstance(runtime, dict)
            or set(runtime) != {"model", "upstream_lock", "base_checkpoint_sha256", "torch_version"}
            or any(runtime[k] != common[k] for k in runtime)):
        raise ValueError("endpoint runtime/common provenance mismatch")
    for key in ("base_checkpoint_sha256", "feature_sha256", "annotation_sha256", "connectivity_sha256"):
        _sha(common[key], key)
    implementation = identity.get("implementation")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("endpoint source implementation identity is missing")
    for name, sha in implementation.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid implementation filename")
        _sha(sha, name)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("empty endpoint cache")
    names, instructions, episode_ids, payloads, per_scan = set(), set(), set(), [], {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"name", "sha256", "association", "num_states", "positives"}:
            raise ValueError("endpoint file manifest fields differ")
        name = item["name"]
        if (not isinstance(name, str) or Path(name).name != name or not name.startswith("episode-")
                or not name.endswith(".pt") or name in names):
            raise ValueError("invalid or repeated endpoint episode filename")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("endpoint episode file is missing or escapes the cache")
        if file_sha256(path) != _sha(item["sha256"], "episode"):
            raise ValueError("endpoint episode SHA-256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_endpoint_episode(payload, item, manifest["identity_sha256"])
        association = payload["association"]
        if name != association["episode_id"] + ".pt":
            raise ValueError("episode filename does not match its association")
        if association["instr_id"] in instructions or association["episode_id"] in episode_ids:
            raise ValueError("endpoint cache repeats an instruction/episode")
        names.add(name); instructions.add(association["instr_id"]); episode_ids.add(association["episode_id"])
        payloads.append(payload)
        scan = per_scan.setdefault(association["scan_id"], {"episodes": 0, "states": 0, "positives": 0})
        scan["episodes"] += 1; scan["states"] += item["num_states"]; scan["positives"] += item["positives"]
    if {p.name for p in root.glob("episode-*.pt")} != names:
        raise ValueError("unexpected endpoint episode files outside the manifest")
    summary = manifest.get("summary", {})
    expected = {"episodes": len(files), "states": sum(x["num_states"] for x in files),
                "positives": sum(x["positives"] for x in files), "per_scan": per_scan,
                "all_state_three_branch_exact_logit_parity": True}
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("endpoint cache summary/three-branch parity mismatch")
    # Exporter elapsed-time summaries may change when resuming an already complete
    # cache. Training identity binds every authoritative file byte and its order.
    data_sha = object_sha256({"identity": identity, "files": files})
    return EndpointCache(root, manifest, digest, data_sha, tuple(payloads))


def episode_normalized_bce(logits: Tensor, labels: Tensor, lengths: list[int]) -> Tensor:
    """Mean within each episode, then mean over episodes; no class reweighting."""
    if (not lengths or any(type(n) is not int or n < 1 for n in lengths)
            or logits.ndim != 1 or labels.shape != logits.shape or sum(lengths) != len(logits)
            or not torch.isfinite(logits).all() or not torch.isfinite(labels).all()
            or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("invalid endpoint BCE batch")
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return torch.stack([part.mean() for part in losses.split(lengths)]).mean()


def _training_code_identity():
    folder = Path(__file__).resolve().parent
    return {name: file_sha256(folder / name) for name in
            ("endpoint_probe.py", "checkpoint_store.py", "resumable.py", "protocol.py")}


class EndpointTrainer:
    """Small, restartable BCE trainer; dev BCE is monitoring, never nav-best."""

    def __init__(self, train: EndpointCache, dev: EndpointCache, *, epochs=20,
                 batch_episodes=32, lr=1e-3, weight_decay=1e-4, seed=0, device="cpu"):
        if train.split != "train_fit" or dev.split != "train_dev":
            raise ValueError("fit accepts train_fit and monitoring accepts train_dev only")
        if train.common_identity != dev.common_identity:
            raise ValueError("train/dev endpoint caches use different model/data/source provenance")
        train_scans = {x["association"]["scan_id"] for x in train.episodes}
        dev_scans = {x["association"]["scan_id"] for x in dev.episodes}
        if train_scans & dev_scans:
            raise ValueError("train/dev endpoint caches share a scene")
        if any(type(n) is not int or n < 1 for n in (epochs, batch_episodes)) or type(seed) is not int or seed < 0:
            raise ValueError("invalid endpoint training counts/seed")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in (lr, weight_decay)) or lr <= 0 or weight_decay < 0:
            raise ValueError("invalid endpoint AdamW hyperparameters")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("endpoint trainer supports CPU or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA endpoint training requested without CUDA")
        self.train_cache, self.dev_cache = train, dev
        self.config = {"epochs": epochs, "batch_episodes": batch_episodes, "lr": float(lr),
                       "weight_decay": float(weight_decay), "seed": seed, "device": str(self.device),
                       "optimizer": "AdamW", "loss": "mean_episode_mean_state_bce_no_class_weights",
                       "torch_version": str(torch.__version__), "feature_dim": FEATURE_DIM,
                       "hidden_dim": 128, "activation": "relu"}
        self.data_identity = {"train": train.data_sha256, "dev": dev.data_sha256,
                              "common": train.common_identity}
        self.code_identity = _training_code_identity()
        random.seed(seed); torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.head = EndpointProbe().to(self.device)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        self.epoch = self.cursor = self.global_step = 0
        self.pending_dev = False
        self.history = []
        self.best_dev_bce = self.best_epoch = None
        self.epoch_loss_sum = 0.0
        self.pending_train_bce = None
        self.batches_per_epoch = math.ceil(len(train.episodes) / batch_episodes)

    @property
    def done(self):
        return self.epoch == self.config["epochs"] and not self.pending_dev

    def _batch(self, episodes):
        # Deliberately restrict training input access to these two tensor fields.
        features = torch.cat([x["features"] for x in episodes]).to(self.device)
        labels = torch.cat([x["labels"] for x in episodes]).to(self.device)
        return features, labels, [len(x["labels"]) for x in episodes]

    def step(self):
        if self.done or self.pending_dev:
            raise ValueError("endpoint step requires unfinished training and no pending dev monitoring")
        order = list(range(len(self.train_cache.episodes)))
        random.Random(self.config["seed"] + self.epoch).shuffle(order)
        indices = order[self.cursor:self.cursor + self.config["batch_episodes"]]
        batch = [self.train_cache.episodes[i] for i in indices]
        features, labels, lengths = self._batch(batch)
        self.head.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = episode_normalized_bce(self.head(features), labels, lengths)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite endpoint training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.head.parameters()):
            raise ValueError("nonfinite endpoint gradient")
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.head.parameters()):
            raise ValueError("nonfinite endpoint parameters after AdamW")
        value = float(loss.detach())
        self.epoch_loss_sum += value * len(batch)
        self.cursor += len(batch)
        self.global_step += 1
        if self.cursor == len(self.train_cache.episodes):
            self.pending_train_bce = self.epoch_loss_sum / self.cursor
            self.epoch += 1; self.cursor = 0; self.epoch_loss_sum = 0.0
            self.pending_dev = True
        return {"step": self.global_step, "batch_bce": value, "episodes": len(batch),
                "epoch_completed": self.pending_dev}

    def monitor_dev(self):
        if not self.pending_dev:
            raise ValueError("no completed epoch awaiting endpoint dev monitoring")
        self.head.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(self.dev_cache.episodes), self.config["batch_episodes"]):
                episodes = self.dev_cache.episodes[start:start + self.config["batch_episodes"]]
                features, labels, lengths = self._batch(episodes)
                total += float(episode_normalized_bce(self.head(features), labels, lengths)) * len(episodes)
        bce = total / len(self.dev_cache.episodes)
        if not math.isfinite(bce):
            raise ValueError("nonfinite endpoint dev BCE")
        metrics = {"epoch": self.epoch, "global_step": self.global_step,
                   "online_train_bce": self.pending_train_bce, "train_dev_bce": bce,
                   "dev_episodes": len(self.dev_cache.episodes),
                   "selection_purpose": "engineering_best_dev_bce_not_navigation_best"}
        self.history.append(metrics)
        is_best = self.best_dev_bce is None or bce < self.best_dev_bce
        if is_best:
            self.best_dev_bce, self.best_epoch = bce, self.epoch
        self.pending_dev = False; self.pending_train_bce = None
        return metrics, is_best

    def state_dict(self):
        return {"schema": TRAIN_SCHEMA, "config": copy.deepcopy(self.config),
                "data_identity": copy.deepcopy(self.data_identity), "code_identity": copy.deepcopy(self.code_identity),
                "head": _cpu_copy(self.head.state_dict()), "optimizer": _cpu_copy(self.optimizer.state_dict()),
                "scheduler": None, "scaler": None, "rng": _rng_state(),
                "epoch": self.epoch, "episode_cursor": self.cursor, "global_step": self.global_step,
                "pending_dev": self.pending_dev, "pending_train_bce": self.pending_train_bce,
                "epoch_loss_sum": self.epoch_loss_sum, "history": copy.deepcopy(self.history),
                "best_dev_bce": self.best_dev_bce, "best_epoch": self.best_epoch}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or state.get("schema") != TRAIN_SCHEMA
                or state.get("config") != self.config or state.get("data_identity") != self.data_identity
                or state.get("code_identity") != self.code_identity):
            raise ValueError("endpoint resume config/data/code identity changed")
        if state.get("scheduler") is not None or state.get("scaler") is not None:
            raise ValueError("endpoint pilot does not use a scheduler or mixed precision")
        epoch, cursor, step = (state.get(k) for k in ("epoch", "episode_cursor", "global_step"))
        n = len(self.train_cache.episodes)
        if (any(type(x) is not int for x in (epoch, cursor, step))
                or not 0 <= epoch <= self.config["epochs"] or not 0 <= cursor < n
                or cursor % self.config["batch_episodes"] or (epoch == self.config["epochs"] and cursor)
                or step != epoch * self.batches_per_epoch + cursor // self.config["batch_episodes"]):
            raise ValueError("invalid endpoint resume optimizer-step cursor")
        pending, history = state.get("pending_dev"), state.get("history")
        if (type(pending) is not bool or not isinstance(history, list)
                or len(history) != epoch - int(pending) or (pending and cursor)
                or (pending and epoch < 1)):
            raise ValueError("endpoint pending dev/history mismatch")
        for index, entry in enumerate(history, 1):
            if (entry.get("epoch") != index or entry.get("global_step") != index * self.batches_per_epoch
                    or entry.get("dev_episodes") != len(self.dev_cache.episodes)
                    or any(not isinstance(entry.get(k), (int, float)) or isinstance(entry[k], bool)
                           or not math.isfinite(entry[k]) or entry[k] < 0
                           for k in ("train_dev_bce", "online_train_bce"))):
                raise ValueError("invalid endpoint monitoring history")
        best = min(history, key=lambda x: x["train_dev_bce"]) if history else None
        if (state.get("best_dev_bce") != (best["train_dev_bce"] if best else None)
                or state.get("best_epoch") != (best["epoch"] if best else None)):
            raise ValueError("endpoint best-dev metadata differs from history")
        total, pending_loss = state.get("epoch_loss_sum"), state.get("pending_train_bce")
        if (not isinstance(total, (int, float)) or isinstance(total, bool) or not math.isfinite(total) or total < 0
                or (cursor == 0 and total != 0)
                or (pending and (not isinstance(pending_loss, (int, float)) or isinstance(pending_loss, bool)
                                 or not math.isfinite(pending_loss) or pending_loss < 0))
                or (not pending and pending_loss is not None)):
            raise ValueError("endpoint partial-epoch loss state is invalid")
        weights = state.get("head")
        if not isinstance(weights, dict) or not all(isinstance(x, Tensor) and torch.isfinite(x).all() for x in weights.values()):
            raise ValueError("invalid endpoint head weights")
        head = EndpointProbe().to(self.device)
        head.load_state_dict(weights, strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        saved_opt = state.get("optimizer")
        if not isinstance(saved_opt, dict) or saved_opt.get("param_groups") != optimizer.state_dict()["param_groups"]:
            raise ValueError("endpoint AdamW parameter groups changed")
        states = saved_opt.get("state")
        parameters = list(head.parameters())
        if not isinstance(states, dict) or set(states) != (set(range(len(parameters))) if step else set()):
            raise ValueError("endpoint AdamW state is missing or unexpected")
        for index, value in states.items():
            if not isinstance(value, dict) or set(value) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid endpoint AdamW slots")
            if (not isinstance(value["step"], Tensor) or value["step"].numel() != 1
                    or not torch.isfinite(value["step"]).all() or float(value["step"]) != step):
                raise ValueError("endpoint AdamW step differs from the training cursor")
            for slot in ("exp_avg", "exp_avg_sq"):
                tensor = value[slot]
                if (not isinstance(tensor, Tensor) or tensor.shape != parameters[index].shape
                        or tensor.dtype != parameters[index].dtype or not torch.isfinite(tensor).all()
                        or (slot == "exp_avg_sq" and (tensor < 0).any())):
                    raise ValueError("invalid endpoint AdamW moment")
        optimizer.load_state_dict(saved_opt)
        _restore_rng(state["rng"], device=self.device, seed=self.config["seed"])
        self.head, self.optimizer = head, optimizer
        self.epoch, self.cursor, self.global_step = epoch, cursor, step
        self.pending_dev, self.pending_train_bce = pending, pending_loss
        self.epoch_loss_sum = float(total)
        self.history = copy.deepcopy(history)
        self.best_dev_bce, self.best_epoch = state["best_dev_bce"], state["best_epoch"]

    def head_payload(self):
        return {"schema": HEAD_SCHEMA, "head_config": self.head.config,
                "feature_schema": FEATURE_SCHEMA, "state_dict": _cpu_copy(self.head.state_dict()),
                "common_identity": copy.deepcopy(self.train_cache.common_identity),
                "data_identity": copy.deepcopy(self.data_identity), "train_config": copy.deepcopy(self.config),
                "epoch": self.epoch, "global_step": self.global_step,
                "selection_purpose": "engineering_best_dev_bce_not_navigation_best"}


def load_endpoint_head(path: str | Path, *, device="cpu", expected_common_identity=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    head = EndpointProbe()
    if (not isinstance(payload, dict) or payload.get("schema") != HEAD_SCHEMA
            or payload.get("head_config") != head.config or payload.get("feature_schema") != FEATURE_SCHEMA
            or (expected_common_identity is not None and payload.get("common_identity") != expected_common_identity)):
        raise ValueError("endpoint head schema/provenance mismatch")
    weights = payload.get("state_dict")
    if not isinstance(weights, dict) or not all(isinstance(x, Tensor) and torch.isfinite(x).all() for x in weights.values()):
        raise ValueError("endpoint head weights are invalid")
    head.load_state_dict(weights, strict=True)
    return head.to(device).eval(), {k: v for k, v in payload.items() if k != "state_dict"}


def train_endpoint_probe(train_cache, dev_cache, local_dir, backup_dir, *, device="cpu",
                         epochs=20, batch_episodes=32, lr=1e-3, weight_decay=1e-4, seed=0,
                         checkpoint_every_steps=10, checkpoint_every_seconds=60,
                         keep_local=2, keep_backup=5, verify_backup: Callable[[], object] | None = None,
                         should_stop: Callable[[], bool] | None = None, max_steps: int | None = None):
    """Resume/save through CheckpointStore; ``best`` means dev BCE monitoring.

    ``max_steps`` is an interruption-test boundary in the current invocation,
    not a new training horizon. Cache or identity errors never start a new run.
    """
    if type(checkpoint_every_steps) is not int or checkpoint_every_steps < 1:
        raise ValueError("checkpoint_every_steps must be positive")
    if (isinstance(checkpoint_every_seconds, bool) or not isinstance(checkpoint_every_seconds, (int, float))
            or not math.isfinite(checkpoint_every_seconds) or checkpoint_every_seconds <= 0):
        raise ValueError("checkpoint_every_seconds must be positive finite")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
        raise ValueError("max_steps must be a nonnegative interruption-test limit")
    train = load_endpoint_cache(train_cache, expected_split="train_fit")
    dev = load_endpoint_cache(dev_cache, expected_split="train_dev")
    trainer = EndpointTrainer(train, dev, epochs=epochs, batch_episodes=batch_episodes,
                              lr=lr, weight_decay=weight_decay, seed=seed, device=device)
    checker, stopping = verify_backup or (lambda: None), should_stop or (lambda: False)
    checker()
    store = CheckpointStore(local_dir, backup_dir, keep_local=keep_local, keep_backup=keep_backup)
    resumed = False
    with store.lock():
        try:
            state, head, _ = store.restore("latest")
        except FileNotFoundError:
            pass
        else:
            trainer.load_state_dict(state)
            current = trainer.head_payload()
            if set(head) != set(current) or any(head[k] != current[k] for k in head if k != "state_dict"):
                raise ValueError("endpoint snapshot state/head metadata differ")
            if (set(head["state_dict"]) != set(current["state_dict"])
                    or any(not torch.equal(head["state_dict"][k], current["state_dict"][k]) for k in current["state_dict"])):
                raise ValueError("endpoint snapshot state/head weights differ")
            resumed = True
        last_saved_step = trainer.global_step
        last_saved_at = time.monotonic()
        latest_id = None

        def persist(reason, *, is_best=False, metrics=None):
            nonlocal last_saved_step, last_saved_at, latest_id
            checker()
            detail = {"reason": reason, "selection_purpose": "engineering_best_dev_bce_not_navigation_best"}
            if metrics:
                detail.update(metrics)
            latest_id = store.save(trainer.state_dict(), trainer.head_payload(), step=trainer.global_step,
                                   is_best=is_best, metrics=detail)
            last_saved_step, last_saved_at = trainer.global_step, time.monotonic()

        if not resumed:
            persist("initialized")
        started_at_step = trainer.global_step
        interrupted = False
        try:
            while not trainer.done:
                if stopping() or (max_steps is not None and trainer.global_step - started_at_step >= max_steps):
                    persist("interrupted_at_optimizer_boundary")
                    interrupted = True
                    break
                if trainer.pending_dev:
                    metrics, best = trainer.monitor_dev()
                    persist("train_dev_bce_monitored", is_best=best, metrics=metrics)
                    continue
                trainer.step()
                if (trainer.pending_dev or trainer.global_step - last_saved_step >= checkpoint_every_steps
                        or time.monotonic() - last_saved_at >= checkpoint_every_seconds):
                    persist("dev_monitor_pending" if trainer.pending_dev else "periodic")
        except KeyboardInterrupt:
            # A Python interrupt can occur inside optimizer.step; only the last
            # durable boundary can safely be resumed. Do not save partial AdamW.
            return {"status": "interrupted", "resumed": resumed, "last_durable_step": last_saved_step,
                    "selection_purpose": "engineering_best_dev_bce_not_navigation_best"}
        if trainer.done:
            persist("completed")
        return {"status": "interrupted" if interrupted else "complete", "resumed": resumed,
                "global_step": trainer.global_step, "completed_epochs": trainer.epoch,
                "pending_dev": trainer.pending_dev, "best_dev_bce": trainer.best_dev_bce,
                "best_epoch": trainer.best_epoch, "history": trainer.history,
                "latest_checkpoint_id": latest_id, "train_data_sha256": train.data_sha256,
                "dev_data_sha256": dev.data_sha256,
                "selection_purpose": "engineering_best_dev_bce_not_navigation_best"}
