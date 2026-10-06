"""Append-only accounting for official validation accesses; no evaluator or fitter."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .protocol import file_sha256, object_sha256

CATEGORIES = ("baseline", "pilot", "diagnostic_analysis", "confirmatory")
LEGACY_NOTE = (
    "Legacy baseline records are preserved byte-for-byte. The protocol includes the "
    "baseline in the pilot budget, so V0001 consumes one variant and one access even "
    "though its original variant_budget_charge was 0. Missing legacy code identity "
    "is not reconstructed."
)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


class StudyLedger:
    """Use one local ledger writer domain; flock is not a distributed Drive lock.

    All registrations, including failed/incomplete attempts, consume their budget.
    A retry that actually reruns an evaluation needs a new access ID.
    """

    def __init__(self, path: Path, study: dict):
        self.path = Path(path)
        self.study = study
        self.study_id = _text(study.get("study_id"), "study_id")
        self.protocol = study["evaluation_protocol"]
        if self.protocol.get("parameter_fitting_split") != "train_fit":
            raise ValueError("this study ledger requires parameter fitting only on train_fit")
        if self.protocol.get("unseen_development_split") != "val_unseen":
            raise ValueError("this ledger is scoped to official val_unseen development")

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_name(self.path.name + ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _read(self):
        registrations, outcomes = {}, {}
        if not self.path.exists():
            return registrations, outcomes
        # Fail closed on a torn append. Never silently truncate an audit record.
        with self.path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    raise ValueError(f"unterminated ledger record at line {number}; preserve and repair explicitly")
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                    _json(row)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"invalid ledger JSON at line {number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"ledger line {number} must be an object")
                access = _text(row.get("access_id"), "access_id")
                if row.get("event") == "registered":
                    if row.get("schema_version") != 1 or row.get("study_id") != self.study_id:
                        raise ValueError(f"incompatible registration: {access}")
                    if access in registrations:
                        raise ValueError(f"duplicate registration: {access}")
                    registrations[access] = row
                elif row.get("status") == "registered_before_execution":
                    # Explicit compatibility for the one existing, inspected manual record.
                    if (access != "V0001" or row.get("method_id") != "DUET-official-frozen"
                            or row.get("checkpoint_sha256") != self.study["baseline"]["base_checkpoint_sha256"]
                            or row.get("split") != "val_unseen" or row.get("subset") is not False
                            or access in registrations):
                        raise ValueError("unknown legacy registration; an explicit import is required")
                    request = dict(category="baseline", variant_id=row["method_id"],
                                   config_sha256=_sha(row["config_sha256"], "config_sha256"),
                                   checkpoint_sha256=row["checkpoint_sha256"], code_sha256=None,
                                   purpose=row["purpose"], split=row["split"], subset=False,
                                   subset_ids=[], seed=row["seed"], expected_episodes=row["expected_episodes"],
                                   label_use="evaluation", parameter_fitting_split="train_fit")
                    registrations[access] = dict(row, request=request, budget_group="pilot", legacy=True)
                elif row.get("event") in {"completed", "failed"}:
                    if access not in registrations or access in outcomes:
                        raise ValueError(f"missing registration or duplicate outcome: {access}")
                    if row.get("schema_version") is not None and (
                            row.get("schema_version") != 1 or row.get("study_id") != self.study_id):
                        raise ValueError(f"incompatible outcome: {access}")
                    outcomes[access] = row
                else:
                    raise ValueError(f"unsupported ledger record at line {number}")
        return registrations, outcomes

    def _append(self, row):
        payload = (_json(row) + "\n").encode()
        with self.path.open("ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _budget(self, category, budget_path):
        if category in {"pilot", "baseline"}:
            if budget_path is not None:
                raise ValueError("baseline/pilot limits come from the study protocol, not a separate budget")
            variants = _positive(self.protocol["pilot_max_variants"], "pilot_max_variants")
            per_variant = _positive(self.protocol["pilot_checkpoint_evaluations_per_variant"], "pilot checkpoint limit")
            budget = dict(budget_id="pilot", category="pilot", max_variants=variants,
                          max_accesses=variants * per_variant, max_accesses_per_variant=per_variant,
                          seeds=[self.protocol["pilot_seed"]], splits=["val_unseen"])
            return "pilot", dict(limits=budget, study_config_sha256=object_sha256(self.study))
        if budget_path is None:
            raise ValueError(f"{category} requires an explicit budget file")
        budget_path = Path(budget_path)
        raw_budget = budget_path.read_bytes()
        budget = json.loads(raw_budget)
        if (budget.get("schema_version") != 1 or budget.get("study_id") != self.study_id
                or budget.get("category") != category):
            raise ValueError("budget schema/study/category mismatch")
        for key in ("budget_id", "reason"):
            _text(budget.get(key), key)
        for key in ("max_variants", "max_accesses", "max_accesses_per_variant"):
            _positive(budget.get(key), key)
        seeds = budget.get("seeds")
        if (not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds))
                or any(type(x) is not int or x < 0 for x in seeds)):
            raise ValueError("budget seeds must be a nonempty list of distinct nonnegative integers")
        if budget.get("splits") != ["val_unseen"]:
            raise ValueError("budget splits must be [val_unseen]")
        if category == "confirmatory":
            variants = budget.get("variants")
            if not isinstance(variants, dict) or not variants:
                raise ValueError("confirmatory budget requires the frozen variant_id -> config_sha256 mapping")
            for key, value in variants.items():
                _text(key, "variant_id")
                _sha(value, "variant config_sha256")
        _json(budget)
        # Group by category, never budget ID: revising limits must not reset usage.
        return category, dict(limits=budget, file=str(budget_path),
                              sha256=hashlib.sha256(raw_budget).hexdigest())

    @staticmethod
    def _usage(registrations):
        groups = {}
        for row in registrations.values():
            group = groups.setdefault(row["budget_group"], {"accesses": 0, "variants": {}})
            group["accesses"] += 1
            variant = row["request"]["variant_id"]
            group["variants"][variant] = group["variants"].get(variant, 0) + 1
        return groups

    def register(self, *, access_id: str, category: str, variant_id: str, config_sha256: str,
                 checkpoint_sha256: str, code_sha256: str, purpose: str, split: str,
                 seed: int, expected_episodes: int, subset: bool = False,
                 subset_ids: list[str] | None = None, label_use: str = "evaluation",
                 parameter_fitting_split: str = "train_fit", budget_path: Path | None = None):
        _text(access_id, "access_id")
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        _text(variant_id, "variant_id")
        for key, value in (("config_sha256", config_sha256), ("checkpoint_sha256", checkpoint_sha256),
                           ("code_sha256", code_sha256)):
            _sha(value, key)
        _text(purpose, "purpose")
        if split != "val_unseen" or parameter_fitting_split != "train_fit":
            raise ValueError("only val_unseen development accesses; parameter fitting must remain train_fit")
        expected_use = "analysis" if category == "diagnostic_analysis" else "evaluation"
        if label_use != expected_use:
            raise ValueError(f"{category} requires label_use={expected_use}; validation fitting is forbidden")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _positive(expected_episodes, "expected_episodes")
        if type(subset) is not bool:
            raise ValueError("subset must be a boolean")
        ids = [] if subset_ids is None else subset_ids
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("subset_ids must be a list of nonempty instruction IDs")
        if len(ids) != len(set(ids)):
            raise ValueError("subset_ids must be unique")
        if subset:
            if not ids or expected_episodes != len(ids):
                raise ValueError("subset registration requires all fixed IDs and their exact count")
        elif ids or expected_episodes != self.study["baseline"]["selection_episodes"]:
            raise ValueError("full access must declare the complete official split without subset IDs")
        if category == "confirmatory" and subset:
            raise ValueError("confirmatory evaluation requires the full split")
        request = dict(category=category, variant_id=variant_id, config_sha256=config_sha256,
                       checkpoint_sha256=checkpoint_sha256, code_sha256=code_sha256, purpose=purpose,
                       split=split, seed=seed, expected_episodes=expected_episodes, subset=subset,
                       subset_ids=sorted(ids), label_use=label_use,
                       parameter_fitting_split=parameter_fitting_split)
        group, snapshot = self._budget(category, budget_path)
        limits = snapshot["limits"]
        with self._lock():
            registrations, _ = self._read()
            if access_id in registrations:
                previous = registrations[access_id]
                if previous["request"] != request:
                    raise ValueError(f"access ID {access_id} already has a different registration")
                return previous
            if seed not in limits["seeds"]:
                raise ValueError("seed is outside the declared budget")
            if category == "confirmatory" and limits["variants"].get(variant_id) != config_sha256:
                raise ValueError("variant/config is absent from the frozen confirmatory budget")
            for previous in registrations.values():
                identity = previous["request"]
                if identity["variant_id"] == variant_id and identity["config_sha256"] != config_sha256:
                    raise ValueError("changed method configuration requires a new variant ID")
                if (previous["budget_group"] == group and identity["config_sha256"] == config_sha256
                        and identity["variant_id"] != variant_id):
                    raise ValueError("the same method configuration cannot get a new ID to reset its budget")
            used = self._usage(registrations).get(group, {"accesses": 0, "variants": {}})
            if used["accesses"] >= limits["max_accesses"]:
                raise ValueError(f"{group} total access budget exhausted")
            if variant_id not in used["variants"] and len(used["variants"]) >= limits["max_variants"]:
                raise ValueError(f"{group} variant budget exhausted")
            if used["variants"].get(variant_id, 0) >= limits["max_accesses_per_variant"]:
                raise ValueError(f"{group} per-variant access budget exhausted")
            row = dict(schema_version=1, event="registered", study_id=self.study_id,
                       access_id=access_id, registered_utc=_now(), request=request,
                       budget_group=group, budget=snapshot)
            self._append(row)
            return row

    def finish(self, access_id: str, *, status: str, metrics: dict, resources: dict,
               decision: str, report_path: Path | None = None, error: str | None = None):
        if status not in {"completed", "failed"}:
            raise ValueError("status must be completed or failed")
        _text(decision, "decision")
        if not isinstance(metrics, dict) or not isinstance(resources, dict) or not resources:
            raise ValueError("metrics/resources must be objects; record costs or explicit unknowns")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
            raise ValueError("metrics must contain finite numeric values")
        _json(resources)
        if status == "completed" and (not metrics or report_path is None or error is not None):
            raise ValueError("completion requires metrics and a report, without error")
        if status == "failed":
            _text(error, "failure error")
        payload = dict(event=status, metrics=metrics, resources=resources, decision=decision,
                       report=None if report_path is None else str(report_path),
                       report_sha256=None if report_path is None else file_sha256(report_path), error=error)
        with self._lock():
            registrations, outcomes = self._read()
            if access_id not in registrations:
                raise ValueError("register the access before execution/analysis")
            if access_id in outcomes:
                previous = outcomes[access_id]
                if any(previous.get(k) != v for k, v in payload.items()):
                    raise ValueError("this access already has a different terminal outcome; reruns need a new ID")
                return previous
            row = dict(schema_version=1, study_id=self.study_id, access_id=access_id,
                       recorded_utc=_now(), **payload)
            self._append(row)
            return row

    def lookup(self, access_id: str) -> dict:
        """Return independent copies of a normalized registration and its outcome.

        ``outcome`` is None while pending; completed and failed records are terminal.
        This reads under the local lock without appending or rewriting ledger bytes.
        It does not reserve execution: callers must also prevent concurrent runs of
        the same pending access. Unknown IDs raise KeyError rather than registering.
        """
        _text(access_id, "access_id")
        with self._lock():
            registrations, outcomes = self._read()
            if access_id not in registrations:
                raise KeyError(f"access ID {access_id} is not registered")
            return deepcopy({"registration": registrations[access_id],
                             "outcome": outcomes.get(access_id)})

    def status(self):
        with self._lock():
            registrations, outcomes = self._read()
            return dict(study_id=self.study_id, usage=self._usage(registrations),
                        pilot_limits=self._budget("pilot", None)[1]["limits"],
                        pending=[key for key in registrations if key not in outcomes],
                        outcomes={key: value["event"] for key, value in outcomes.items()},
                        legacy_notes=[LEGACY_NOTE] if any(r.get("legacy") for r in registrations.values()) else [])
