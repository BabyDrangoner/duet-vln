"""Interruptible offline training with complete, portable optimizer checkpoints.

Only complete optimizer steps are checkpoint boundaries. The cache order is
reconstructed from its seed once on restore, then streamed without replaying
updates. Callers own durable storage, navigation evaluation, and retention.
"""

from __future__ import annotations

import copy
import hashlib
import math
import platform
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from .features import FEATURE_SCHEMA
from .head import ResidualActionHead
from .train import collate_records, iter_records, read_manifests, training_loss


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [item.cpu().clone() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
        "numpy": None,
    }
    try:
        import numpy as np
    except ImportError:
        return state
    engine, keys, position, has_gauss, cached_gauss = np.random.get_state()
    # numpy arrays cannot be loaded by torch's restricted weights_only loader.
    state["numpy"] = {
        "engine": engine, "keys": keys.tolist(), "position": position,
        "has_gauss": has_gauss, "cached_gauss": cached_gauss,
    }
    return state


def _restore_rng(state: dict[str, Any], *, device: torch.device, seed: int) -> None:
    # Validate independent generators before modifying process-global state.
    probe = random.Random()
    probe.setstate(state["python"])
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    numpy_state = None
    if state["numpy"] is not None:
        import numpy as np
        value = state["numpy"]
        numpy_state = (
            value["engine"], np.array(value["keys"], dtype=np.uint32),
            value["position"], value["has_gauss"], value["cached_gauss"],
        )
        np.random.RandomState().set_state(numpy_state)
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list) or not all(
        isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1
        for value in cuda_states
    ):
        raise ValueError("invalid CUDA RNG state")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if device.type == "cuda":
        # Moving from CPU or a different GPU topology is supported, but exact
        # bitwise reproducibility is only expected on an unchanged runtime.
        torch.cuda.manual_seed_all(seed)
        for index, value in enumerate(cuda_states[:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value, index)


def _empty_totals() -> dict[str, float | int]:
    return {"ce_numerator": 0.0, "weight_sum": 0.0, "kl_sum": 0.0, "num_records": 0}


class ResumableTrainer:
    """Stream train_fit caches and expose an atomic optimizer-step boundary.

    Loading checks every manifest and shard hash, plus all training options.
    Absolute cache locations and the CPU/CUDA device may change on a new VM.
    Epoch limits are intentionally fixed; extending a run is a new experiment.
    """

    def __init__(
        self, cache_dirs: Sequence[str | Path], *, epochs: int = 3,
        batch_size: int = 32, lr: float = 0.001, hidden_dim: int = 128,
        max_delta: float = 1.0, hard_weight: float = 3.0,
        kl_weight: float = 0.1, seed: int = 0, device: str = "cpu",
    ) -> None:
        if type(epochs) is not int or epochs <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive integers")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name, value in (("lr", lr), ("hard_weight", hard_weight)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("training device must be cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.manifests = read_manifests(cache_dirs)
        self.provenance = copy.deepcopy(self.manifests[0].provenance)
        self.config = {
            "epochs": epochs, "batch_size": batch_size, "lr": float(lr),
            "hidden_dim": hidden_dim, "max_delta": float(max_delta),
            "hard_weight": float(hard_weight), "kl_weight": float(kl_weight),
            "seed": seed, "feature_dim": self.manifests[0].feature_dim,
            "feature_schema": FEATURE_SCHEMA, "optimizer": "Adam",
            "optimizer_options": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "scheduler": None, "scaler": None,
        }
        self.data_identity = [
            {
                "manifest_sha256": item.manifest_sha256, "num_records": item.num_records,
                "shards": [
                    {"name": str(path.relative_to(item.root)), "sha256": _file_hash(path)}
                    for path in item.shards
                ],
            }
            for item in self.manifests
        ]
        self.num_records = sum(item.num_records for item in self.manifests)
        self.batches_per_epoch = math.ceil(self.num_records / batch_size)
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import numpy as np
            np.random.seed(seed % (2**32))
        except ImportError:
            pass
        self.head = ResidualActionHead(self.config["feature_dim"], hidden_dim, max_delta).to(self.device)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        self.global_step = 0
        self.epoch = 0
        self.history: list[dict[str, Any]] = []
        self._cursor = 0
        self._totals = _empty_totals()
        self._iterator: Iterator[dict[str, Any]] | None = None

    @property
    def done(self) -> bool:
        return self.epoch == self.config["epochs"]

    def _records(self) -> Iterator[dict[str, Any]]:
        if self._iterator is None:
            self._iterator = iter_records(self.manifests, self.config["seed"] + self.epoch)
            for _ in range(self._cursor):
                try:
                    next(self._iterator)
                except StopIteration as error:
                    raise ValueError("cache ended before the restored cursor") from error
        return self._iterator

    def step(self) -> dict[str, Any]:
        if self.done:
            raise RuntimeError("training is already complete")
        remaining = self.num_records - self._cursor
        iterator = self._records()
        records = []
        for _ in range(min(self.config["batch_size"], remaining)):
            try:
                records.append(next(iterator))
            except StopIteration as error:
                raise ValueError("cache ended before its declared record count") from error
        epoch_completed = len(records) == remaining
        if epoch_completed:
            # Exhaustion also executes iter_records' per-cache count validation.
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("cache contains more than its declared record count")
        self.head.train()
        batch = collate_records(records, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.head(batch["features"], batch["base_logits"], batch["valid_mask"])
        loss, statistics = training_loss(
            logits, batch["base_logits"], batch["valid_mask"], batch["target"], batch["hard"],
            hard_weight=self.config["hard_weight"], kl_weight=self.config["kl_weight"],
        )
        loss.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in self.head.parameters()):
            raise ValueError("non-finite gradients")
        self.optimizer.step()
        self.global_step += 1
        self._cursor += len(records)
        self._totals["num_records"] += len(records)
        for name, value in statistics.items():
            self._totals[name] += value.item()
        result = {"global_step": self.global_step, "epoch_completed": epoch_completed, "batch_loss": loss.item()}
        if epoch_completed:
            ce = self._totals["ce_numerator"] / self._totals["weight_sum"]
            kl = self._totals["kl_sum"] / self._totals["num_records"]
            self.epoch += 1
            measured = {
                "epoch": self.epoch, "train_loss": ce + self.config["kl_weight"] * kl,
                "train_ce": ce, "train_kl": kl, "num_records": self._totals["num_records"],
            }
            self.history.append(measured)
            self._cursor = 0
            self._totals = _empty_totals()
            self._iterator = None
            result["epoch_metrics"] = copy.deepcopy(measured)
        return {**result, "epoch": self.epoch, "done": self.done}

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "kind": "resumable_offline_trainer",
            "config": copy.deepcopy(self.config), "data_identity": copy.deepcopy(self.data_identity),
            "provenance": copy.deepcopy(self.provenance), "head": _cpu_copy(self.head.state_dict()),
            "optimizer": _cpu_copy(self.optimizer.state_dict()), "scheduler": None, "scaler": None,
            "rng": _rng_state(), "global_step": self.global_step, "epoch": self.epoch,
            "record_cursor": self._cursor, "epoch_totals": copy.deepcopy(self._totals),
            "history": copy.deepcopy(self.history), "device": str(self.device),
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__)},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("kind") != "resumable_offline_trainer":
            raise ValueError("unsupported resumable trainer state")
        if state.get("config") != self.config:
            raise ValueError("training configuration changed; refusing to resume")
        if state.get("data_identity") != self.data_identity or state.get("provenance") != self.provenance:
            raise ValueError("cache content or provenance changed; refusing to resume")
        if state.get("scheduler", "missing") is not None or state.get("scaler", "missing") is not None:
            raise ValueError("unsupported scheduler/scaler state")
        epoch, cursor, step = (state.get(key) for key in ("epoch", "record_cursor", "global_step"))
        if not all(type(value) is int for value in (epoch, cursor, step)):
            raise ValueError("invalid training cursor")
        if not (0 <= epoch <= self.config["epochs"] and 0 <= cursor < self.num_records):
            raise ValueError("invalid training cursor")
        if cursor % self.config["batch_size"] or (epoch == self.config["epochs"] and cursor):
            raise ValueError("cursor must be at a completed optimizer-step boundary")
        if step != epoch * self.batches_per_epoch + cursor // self.config["batch_size"]:
            raise ValueError("global_step is inconsistent with the training cursor")
        history, totals = state.get("history"), state.get("epoch_totals")
        if not isinstance(history, list) or len(history) != epoch:
            raise ValueError("epoch history does not match the training cursor")
        for index, item in enumerate(history):
            if not isinstance(item, dict) or item.get("epoch") != index + 1 or item.get("num_records") != self.num_records:
                raise ValueError("invalid epoch history")
            if not all(isinstance(item.get(key), (int, float)) and math.isfinite(item[key])
                       for key in ("train_loss", "train_ce", "train_kl")):
                raise ValueError("invalid epoch metrics")
        if not isinstance(totals, dict) or set(totals) != set(_empty_totals()) or totals["num_records"] != cursor:
            raise ValueError("epoch statistics do not match the training cursor")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in totals.values()):
            raise ValueError("non-finite epoch statistics")
        if (cursor == 0 and totals != _empty_totals()) or (cursor and totals["weight_sum"] <= 0):
            raise ValueError("invalid accumulated epoch statistics")
        weights = state.get("head")
        if not isinstance(weights, dict) or not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in weights.values()):
            raise ValueError("head state must contain finite tensors")
        head = ResidualActionHead(self.config["feature_dim"], self.config["hidden_dim"], self.config["max_delta"]).to(self.device)
        head.load_state_dict(weights, strict=True)
        optimizer = torch.optim.Adam(head.parameters(), lr=self.config["lr"])
        optimizer_state = state.get("optimizer")
        expected_groups = optimizer.state_dict()["param_groups"]
        if not isinstance(optimizer_state, dict) or not isinstance(optimizer_state.get("state"), dict):
            raise ValueError("invalid optimizer state")
        groups = optimizer_state.get("param_groups")
        if not isinstance(groups, list) or groups != expected_groups:
            raise ValueError("optimizer hyperparameters or parameter ordering changed")
        parameter_ids = [identifier for group in groups for identifier in group["params"]]
        expected_ids = set(parameter_ids) if step else set()
        if set(optimizer_state["state"]) != expected_ids:
            raise ValueError("optimizer moments are missing or inconsistent with global_step")
        for identifier, parameter in zip(parameter_ids, head.parameters()):
            if not step:
                continue
            moments = optimizer_state["state"][identifier]
            if not isinstance(moments, dict) or set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam moment state")
            saved_step = moments["step"]
            if not isinstance(saved_step, torch.Tensor) or saved_step.numel() != 1 or saved_step.item() != step:
                raise ValueError("optimizer step differs from global_step")
            for name in ("exp_avg", "exp_avg_sq"):
                value = moments[name]
                if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError("optimizer moment shape or dtype changed")
        optimizer.load_state_dict(optimizer_state)
        for value in optimizer.state.values():
            if not all(not isinstance(item, torch.Tensor) or torch.isfinite(item).all() for item in value.values()):
                raise ValueError("optimizer state contains non-finite tensors")
        _restore_rng(state["rng"], device=self.device, seed=self.config["seed"])
        self.head, self.optimizer = head, optimizer
        self.epoch, self._cursor, self.global_step = epoch, cursor, step
        self._totals, self.history = copy.deepcopy(totals), copy.deepcopy(history)
        self._iterator = None

    def head_payload(self) -> dict[str, Any]:
        """Return the existing head.py evaluation format, without saving a file."""
        return {
            "schema_version": 1, "head_config": self.head.config,
            "state_dict": _cpu_copy(self.head.state_dict()), "provenance": copy.deepcopy(self.provenance),
            "train_args": {
                **copy.deepcopy(self.config), "device": str(self.device),
                "cache_identity": copy.deepcopy(self.data_identity), "global_step": self.global_step,
                "completed_epochs": self.epoch, "record_cursor": self._cursor,
            },
            "seed": self.config["seed"],
            "metrics": {"kind": "offline_training_only", "epochs": copy.deepcopy(self.history)},
        }
