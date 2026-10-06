#!/usr/bin/env python3
"""Run the finite E2 collection, two-arm training and navigation experiment.

Requires mounted Google Drive. Restart with the same arguments to resume
collection/training or finish an unstarted evaluation. A claimed evaluation
is never executed twice; interrupted claimed runs need a new registered ID.
Only children launched by this process receive shutdown signals.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
INITIAL_LEDGER_SHA = "26b26492a904edd4cdebbc3e7a48ccbd12ce8a8adafe09f6b8211279d0463a89"
INITIAL_LEDGER_LINES = 14
BASELINE_SHA = "32e4b21422a86c2b8a7e81eb1b423b903ac19cf25972d450c003bdd33d57cd61"
BASELINE_RELATIVE = "outputs/study-20261003/baseline-val-unseen.json"
LEDGER_RELATIVE = "outputs/study-20261003/val-unseen-access-ledger.jsonl"
ARMS = ("relative", "absolute")


class ResumeRequired(RuntimeError):
    """The fixed work is durable and can continue in another invocation."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def immutable_json(path, value):
    path, raw = Path(path), json_bytes(value)
    if path.exists():
        require(path.read_bytes() == raw, "frozen JSON changed: " + str(path))
    else:
        atomic_bytes(path, raw)


def verified_copy(source, destination, *, mutable=False):
    source, destination = Path(source), Path(destination)
    raw = source.read_bytes()
    expected = hashlib.sha256(raw).hexdigest()
    if destination.exists() and not mutable:
        require(digest(destination) == expected, "immutable backup differs: " + str(destination))
    else:
        atomic_bytes(destination, raw)
    require(digest(destination) == expected, "backup read-back SHA differs: " + str(destination))
    return expected


def verify_initial_ledger(raw):
    rows = raw.splitlines(keepends=True)
    require(len(rows) >= INITIAL_LEDGER_LINES and all(line.endswith(b"\n") for line in rows),
            "canonical ledger is truncated")
    require(hashlib.sha256(b"".join(rows[:INITIAL_LEDGER_LINES])).hexdigest() == INITIAL_LEDGER_SHA,
            "canonical ledger initial 14-row identity differs")
    for line in rows:
        require(isinstance(json.loads(line), dict), "ledger line is not an object")


def reconcile_ledgers(local, cloud):
    """Synchronize append-only prefixes; reject divergent writer histories."""
    local, cloud = Path(local), Path(cloud)
    require(cloud.exists(), "canonical Drive ledger is missing")
    cloud_raw = cloud.read_bytes()
    verify_initial_ledger(cloud_raw)
    if not local.exists():
        atomic_bytes(local, cloud_raw)
        return digest(local)
    local_raw = local.read_bytes()
    verify_initial_ledger(local_raw)
    if local_raw.startswith(cloud_raw):
        verified_copy(local, cloud, mutable=True)
    elif cloud_raw.startswith(local_raw):
        verified_copy(cloud, local, mutable=True)
    else:
        raise ValueError("local/Drive ledger histories diverge; refuse to merge or overwrite")
    require(local.read_bytes() == cloud.read_bytes(), "ledger synchronization failed")
    return digest(local)


def ledger_entry(path, access_id):
    registration = outcome = None
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("access_id") != access_id:
            continue
        if row.get("event") == "registered":
            require(registration is None, "duplicate registration")
            registration = row
        elif row.get("event") in {"completed", "failed"}:
            require(outcome is None, "duplicate outcome")
            outcome = row
    return registration, outcome


def derive_arm_configs(master, master_sha):
    require(master.get("arms") == list(ARMS), "first loop requires exactly relative and absolute")
    return {arm: dict(copy.deepcopy(master), arm=arm, collection_config_sha256=master_sha) for arm in ARMS}


def validate_master(master):
    fixed = {"experiment_id": "e2-loop-v1", "seed": 0, "epochs": 20, "batch_size": 64,
             "inference_batch_size": 1, "hidden_dim": 128, "lr": 1e-4, "weight_decay": .01,
             "monitor_every_epochs": 2, "risk_weight": 0., "fit_episodes": 8192, "max_updates_per_arm": 2560}
    require(all(master.get(k) == v for k, v in fixed.items()), "not the frozen finite E2 loop budget")
    collection = master["collection"]
    require(collection["fit_instruction_count"] == 4096 and collection["dev_instruction_count"] == 2890
            and collection["fit_expected_scenes"] == 49 and collection["dev_expected_scenes"] == 12
            and collection["fit_conditions"] == ["natural", "perturb_step2"]
            and collection["adaptive_collection"] is False, "collection plan differs")
    require(master["validation"]["max_accesses_per_arm_this_loop"] == 1, "unbounded validation request")
    require(master["baseline"]["checkpoint_sha256"] == "c1ed3ed27e1acbffec1fa81fbdd06e16498398809a7f80f349043ba74c21ba94",
            "backbone identity differs")


def vm_age_seconds():
    return float(Path("/proc/uptime").read_text().split()[0])


class Loop:
    def __init__(self, args):
        self.args, self.root = args, args.root.resolve()
        self.child_env = dict(os.environ, PYTHONPATH=str(self.root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        self.output, self.backup = args.output_dir.resolve(), args.backup_dir.resolve()
        self.master_path = (self.root / args.master_config).resolve()
        self.cloud_ledger = args.cloud_ledger.resolve()
        self.ledger = self.root / LEDGER_RELATIVE
        self.claim_root = (args.execution_backup_root or self.cloud_ledger.parent / "navigation-execution-claims").resolve()
        sys.path.insert(0, str(self.root / "src"))
        from vln_improve.pipeline import validate_backup_root
        self.verify_drive = lambda: [validate_backup_root(p) for p in (self.backup, self.cloud_ledger.parent, self.claim_root)]
        self.verify_drive()  # No local-filesystem exception exists in this entry point.
        require(not self.output.is_relative_to(self.backup) and not self.backup.is_relative_to(self.output),
                "local experiment and Drive backup must be separate")
        self.master, self.master_sha = read(self.master_path), digest(self.master_path)
        validate_master(self.master)
        require(digest(self.root / BASELINE_RELATIVE) == BASELINE_SHA, "original baseline report changed")
        self.output.mkdir(parents=True, exist_ok=True)
        self.backup.mkdir(parents=True, exist_ok=True)
        self.started, self.stop_requested = time.monotonic(), False
        self.persistence = self.master["persistence"]
        self.state_path, self.cloud_state = self.output / "loop-state.json", self.backup / "loop-state.json"
        identity = {"schema": "e2_finite_loop_state_v1", "experiment_id": self.master["experiment_id"],
                    "master_sha256": self.master_sha, "backup_dir": str(self.backup),
                    "cloud_ledger": str(self.cloud_ledger), "execution_backup_root": str(self.claim_root)}
        resumed = self.cloud_state.exists()
        if resumed:
            self.state = read(self.cloud_state)
            require(self.state.get("identity") == identity, "resumed loop identity differs")
            atomic_bytes(self.state_path, self.cloud_state.read_bytes())
        else:
            require(not self.state_path.exists(), "local state lacks its authoritative Drive journal")
            self.state = {"identity": identity, "created_utc": now(), "status": "initialized", "stages": {}, "invocations": []}
        reconcile_ledgers(self.ledger, self.cloud_ledger)
        if not resumed:
            require(digest(self.ledger) == INITIAL_LEDGER_SHA, "new loop requires the original 14-row ledger")
        self.state["invocations"].append({"started_utc": now(), "pid": os.getpid(), "resumed": resumed,
                                          "vm_age_seconds_at_start": vm_age_seconds()})
        self.save_state()
        self.frozen = self.output / "frozen"
        self.frozen.mkdir(exist_ok=True)
        for name, value in (("master.json", self.master), *[(arm + ".json", value) for arm, value in derive_arm_configs(self.master, self.master_sha).items()]):
            # Preserve the exact master bytes: its SHA binds collection identity.
            path = self.frozen / name
            if name == "master.json":
                if path.exists():
                    require(digest(path) == self.master_sha, "frozen master changed")
                else:
                    atomic_bytes(path, self.master_path.read_bytes())
            else:
                immutable_json(path, value)
            verified_copy(path, self.backup / "frozen" / name)
        self.snapshot_source()

    def save_state(self):
        self.verify_drive()
        self.state["updated_utc"] = now()
        atomic_bytes(self.state_path, json_bytes(self.state))
        verified_copy(self.state_path, self.cloud_state, mutable=True)

    def remaining(self, *, reserve=False):
        process = self.persistence["max_process_seconds"] - (time.monotonic() - self.started)
        runtime = self.persistence["max_vm_age_seconds"] - vm_age_seconds()
        return min(process, runtime) - (self.persistence["evaluation_reserve_seconds"] if reserve else 0)

    def ready(self, *, reserve=False):
        if self.stop_requested or self.remaining(reserve=reserve) <= 60:
            raise ResumeRequired("stop requested or process/VM budget reached; durable state permits continuation")

    def snapshot_source(self):
        local, cloud = self.frozen / "source-manifest.json", self.backup / "frozen/source-manifest.json"
        if cloud.exists() and not local.exists():
            verified_copy(cloud, local)
        if local.exists():
            self.source = read(local)
            self.verify_source()
            verified_copy(local, cloud)
            archive = self.frozen / "source.tar.gz"
            if not archive.exists():
                verified_copy(self.backup / "frozen/source.tar.gz", archive)
            require(digest(archive) == self.source["archive_sha256"], "frozen source archive differs")
            return
        execution = []
        for folder in ("src", "scripts"):
            execution.extend(p for p in (self.root / folder).rglob("*.py") if "__pycache__" not in p.parts)
        execution.extend(self.root / name for name in ("configs/r2r.json", "configs/research_study.json",
            "outputs/study-20261005/audit_full_navigation.py"))
        execution.append(self.master_path)
        third_party = self.root / "third_party/VLN-DUET"
        execution.extend(p for p in third_party.rglob("*.py") if not set(p.parts) & {"__pycache__", ".git", "build"})
        execution.extend(p for p in (self.root / "third_party").glob("*.json"))
        files = {str(p.relative_to(self.root)): digest(p) for p in sorted(set(execution))}
        archive_files = set(execution) | {self.root / BASELINE_RELATIVE}
        for folder in ("docs", "tests", "configs"):
            archive_files.update(p for p in (self.root / folder).rglob("*") if p.is_file()
                                 and p.suffix in {".md", ".json", ".py"} and "__pycache__" not in p.parts)
        for name in ("README.md", "pyproject.toml", ".gitignore"):
            if (self.root / name).is_file():
                archive_files.add(self.root / name)
        archive = self.frozen / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(archive_files):
                bundle.add(path, arcname=str(path.relative_to(self.root)), recursive=False)
        archive_inventory = {str(p.relative_to(self.root)): digest(p) for p in sorted(archive_files)}
        with tarfile.open(archive) as bundle:
            require({m.name for m in bundle.getmembers()} == set(archive_inventory), "source archive inventory differs")
            for member in bundle.getmembers():
                require(member.isfile() and hashlib.sha256(bundle.extractfile(member).read()).hexdigest() == archive_inventory[member.name],
                        "source archive member mismatch")
        self.source = {"schema": "e2_loop_source_freeze_v1", "execution_files": files,
            "execution_files_sha256": object_sha(files), "archive_files": archive_inventory,
            "archive_sha256": digest(archive), "master_config_sha256": self.master_sha,
            "scope": "source, configs, tests, documents and original baseline report; excludes datasets and feature caches"}
        immutable_json(local, self.source)
        verified_copy(archive, self.backup / "frozen/source.tar.gz")
        verified_copy(local, cloud)

    def verify_source(self):
        require(self.source["master_config_sha256"] == self.master_sha, "source/master identity differs")
        files = self.source["execution_files"]
        require(object_sha(files) == self.source["execution_files_sha256"], "source manifest fingerprint invalid")
        for name, expected in files.items():
            require(digest(self.root / name) == expected, "execution source changed after freeze: " + name)
        require(digest(self.master_path) == self.master_sha, "master configuration changed")

    def command(self, stage, command, *, reserve=False):
        self.ready(reserve=reserve)
        self.verify_drive()
        self.verify_source()
        attempts = self.state["stages"].setdefault(stage, [])
        log = self.output / "logs" / f"{stage}.attempt-{len(attempts) + 1:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        record = {"started_utc": now(), "command": list(map(str, command)), "log": str(log.relative_to(self.output)),
                  "status": "starting", "remaining_seconds_at_start": self.remaining(reserve=reserve)}
        attempts.append(record)
        self.state["status"], self.state["current_stage"] = "running", stage
        self.save_state()
        guarded = [sys.executable, str(self.root / "src/vln_improve/child_guard.py"),
                   "--parent-pid", str(os.getpid()), "--", *map(str, command)]
        started = time.monotonic()
        with log.open("xb") as stream:
            process = subprocess.Popen(guarded, cwd=self.root, env=self.child_env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status="running", child_pid=process.pid)
            self.save_state()
            sent = None
            heartbeat = time.monotonic()
            while process.poll() is None:
                if sent is None and (self.stop_requested or self.remaining(reserve=reserve) <= 60):
                    # This PID is our Popen child; never signal a recovered/unknown PID.
                    process.send_signal(signal.SIGTERM)
                    sent = time.monotonic()
                    record.update(stop_signal="SIGTERM", stop_reason="requested_or_budget", status="stopping")
                    self.save_state()
                if sent is not None and time.monotonic() - sent > 300:
                    record.update(status="stopping_child_still_running", elapsed_seconds=time.monotonic() - started)
                    self.save_state()
                    raise ResumeRequired("owned child did not exit after SIGTERM; inspect recorded PID before continuing")
                if time.monotonic() - heartbeat >= 60:
                    record["elapsed_seconds"] = time.monotonic() - started
                    self.save_state()
                    verified_copy(log, self.backup / record["log"], mutable=True)
                    print(json.dumps({"stage": stage, "child_pid": process.pid,
                        "elapsed_seconds": record["elapsed_seconds"], "remaining_seconds": self.remaining(reserve=reserve)}), flush=True)
                    heartbeat = time.monotonic()
                time.sleep(1)
            record.update(status="process_completed" if process.returncode == 0 else "process_failed",
                          returncode=process.returncode, elapsed_seconds=time.monotonic() - started, finished_utc=now())
            stream.flush()
            os.fsync(stream.fileno())
        verified_copy(log, self.backup / record["log"], mutable=True)
        self.save_state()
        return record

    def checked_command(self, stage, command, *, reserve=False):
        result = self.command(stage, command, reserve=reserve)
        if result["returncode"]:
            if result.get("stop_signal"):
                raise ResumeRequired(stage + " interrupted; acknowledged data/checkpoints remain resumable")
            raise RuntimeError(stage + " failed; see " + result["log"])
        return result

    def restore_report(self, relative, *, mutable=False):
        local, cloud = self.output / relative, self.backup / relative
        if cloud.exists() and (not local.exists() or mutable):
            verified_copy(cloud, local, mutable=mutable)
        if local.exists():
            verified_copy(local, cloud, mutable=mutable)
            return read(local)
        return None

    def collect(self):
        self.cache_paths = {}
        for name, split, condition in (("fit-natural", "train_fit", "natural"),
                                       ("fit-perturb", "train_fit", "perturb_step2"),
                                       ("dev-natural", "train_dev", "natural")):
            relative = Path("collections") / name
            cache, backup = self.output / relative / "cache", self.backup / relative
            self.cache_paths[name] = cache
            if (backup / "cache/manifest.json").exists():
                verified_copy(backup / "cache/manifest.json", cache / "manifest.json", mutable=True)
            report_path = relative / "collection.json"
            completed = self.restore_report(report_path)
            if completed is not None:
                require(read(cache / "manifest.json").get("complete") is True, "collection report has incomplete cache")
                continue
            command = [sys.executable, self.root / "scripts/run_endpoint_intervention.py", "--mode", "collect",
                "--split", split, "--config", self.root / "configs/r2r.json", "--experiment", self.frozen / "master.json",
                "--condition", condition, "--seed", "0", "--cache", cache,
                "--backup", backup, "--output", self.output / report_path]
            if split == "train_fit":
                command += ["--selection", "scene_stratified", "--limit", "4096"]
            self.checked_command("collect-" + name, command, reserve=True)
            self.restore_report(report_path)
        self.support()

    def support(self):
        relative = Path("support.json")
        if self.restore_report(relative) is not None:
            return
        from vln_improve.intervention_runtime import load_records
        panels, unique, scenes, manifests = {}, set(), set(), {}
        for name, cache in self.cache_paths.items():
            self.ready(reserve=True)
            manifests[name] = digest(cache / "manifest.json")
            row = {"episodes": 0, "candidates": 0, "baseline_successes": 0,
                   "positive_instructions": [], "positive_scenes": [], "positive_candidates": 0,
                   "perturbations_applied": 0, "by_scene": {}}
            positive_scenes = set()
            for record in load_records(cache):
                is_positive = bool((record["targets"][:, 1] > 0).any())
                scene, instr = record["scan_id"], record["instr_id"]
                row["episodes"] += 1
                row["candidates"] += len(record["utilities"])
                row["baseline_successes"] += int(record["utilities"][record["inputs"]["baseline_index"], 0])
                row["positive_candidates"] += int((record["targets"][:, 1] > 0).sum())
                row["perturbations_applied"] += int(record["perturbation"]["applied"])
                scene_row = row["by_scene"].setdefault(scene, {"episodes": 0, "positive_instructions": 0})
                scene_row["episodes"] += 1
                scene_row["positive_instructions"] += int(is_positive)
                if is_positive:
                    row["positive_instructions"].append(instr)
                    positive_scenes.add(scene)
                    if name.startswith("fit-"):
                        unique.add(instr)
                        scenes.add(scene)
            row["positive_instructions"].sort()
            row["positive_scenes"] = sorted(positive_scenes)
            panels[name] = row
        passed = len(unique) >= 40 and len(scenes) >= 8
        value = {"schema": "e2_loop_support_v1", "panels": panels, "cache_manifest_sha256": manifests,
                 "unique_positive_fit_instruction_count": len(unique), "positive_fit_scene_count": len(scenes),
                 "unique_positive_fit_instructions": sorted(unique), "positive_fit_scenes": sorted(scenes),
                 "readiness_passed": passed, "required_unique_instructions": 40, "required_scenes": 8,
                 "decision": "continue same finite fixed experiment; " + ("support gate passed" if passed else "support-limited exploratory/engineering result"),
                 "adaptive_collection": False}
        immutable_json(self.output / relative, value)
        verified_copy(self.output / relative, self.backup / relative)

    def train(self):
        for arm in ARMS:
            relative = Path("training") / arm
            local, backup = self.output / relative, self.backup / relative
            summary = self.restore_report(relative / "training-summary.json", mutable=True)
            if summary is None or summary.get("status") != "complete":
                command = [sys.executable, self.root / "scripts/train_endpoint_intervention.py",
                    "--config", self.frozen / (arm + ".json"), "--arm", arm,
                    "--fit-cache", self.cache_paths["fit-natural"], "--fit-cache", self.cache_paths["fit-perturb"],
                    "--dev-cache", self.cache_paths["dev-natural"], "--local-run", local, "--backup-run", backup,
                    "--device", "cuda", "--checkpoint-every-steps", "50", "--checkpoint-every-seconds", "300",
                    "--deadline-seconds", str(max(1, self.remaining(reserve=True) - 90))]
                self.checked_command("train-" + arm, command, reserve=True)
                summary = self.restore_report(relative / "training-summary.json", mutable=True)
            require(summary is not None, "trainer returned without a durable summary")
            if summary.get("status") == "interrupted":
                raise ResumeRequired("training interrupted at a durable optimizer boundary")
            require(summary.get("status") == "complete" and summary.get("completed_epochs") == 20
                    and summary.get("global_step") == 2560 and summary.get("arm") == arm,
                    "formal training did not finish its fixed budget")
            require(summary["config"]["experiment_sha256"] == digest(self.frozen / (arm + ".json")),
                    "training summary configuration differs")
        self.freeze_both_heads()

    def freeze_both_heads(self):
        entries = {}
        for arm in ARMS:
            relative = Path("training") / arm
            summary_path = self.output / relative / "training-summary.json"
            summary = read(summary_path)
            require(summary["status"] == "complete" and summary["completed_epochs"] == 20, "both arms must finish before validation")
            chosen = summary["selected_checkpoint"]
            original = self.output / relative / chosen["head_relative_path"]
            if not original.exists():
                verified_copy(self.backup / relative / chosen["head_relative_path"], original)
            require(digest(original) == chosen["head_sha256"], "selected checkpoint identity differs")
            frozen_head = self.frozen / "heads" / (arm + ".pt")
            verified_copy(original, frozen_head)
            verified_copy(frozen_head, self.backup / "frozen/heads" / (arm + ".pt"))
            entries[arm] = {**chosen, "frozen_head": str(Path("frozen/heads") / (arm + ".pt")),
                "training_summary_sha256": digest(summary_path), "method_config_sha256": digest(self.frozen / (arm + ".json"))}
        value = {"schema": "e2_both_selected_heads_frozen_v1", "entries": entries,
                 "master_config_sha256": self.master_sha, "source_files_sha256": self.source["execution_files_sha256"],
                 "rule": "both complete trained selections are immutable before any new official navigation"}
        immutable_json(self.frozen / "selected-heads.json", value)
        verified_copy(self.frozen / "selected-heads.json", self.backup / "frozen/selected-heads.json")

    def ledger_command(self, stage, arguments):
        reconcile_ledgers(self.ledger, self.cloud_ledger)
        command = [sys.executable, self.root / "scripts/register_study_access.py", "--study",
                   self.root / "configs/research_study.json", "--ledger", self.ledger, *arguments]
        try:
            return self.checked_command(stage, command)
        finally:
            reconcile_ledgers(self.ledger, self.cloud_ledger)

    def close_navigation(self, arm, access, report_path, *, success, attempt=None, error=None):
        registration, outcome = ledger_entry(self.ledger, access)
        if outcome is not None:
            require(outcome["event"] == ("completed" if success else "failed"), "closed navigation outcome differs")
            return
        report = read(report_path) if report_path.exists() else None
        metrics = report["summary"] if report is not None else {}
        resources = report.get("resources", {}) if report is not None else {
            "rollout_seconds": None, "cuda_peak_allocated_bytes": None,
            "reason": "no complete navigation report; interrupted costs unavailable"}
        resources = dict(resources, orchestration_seconds=(attempt or {}).get("elapsed_seconds"))
        metrics_path, resources_path = report_path.with_name(arm + "-metrics.json"), report_path.with_name(arm + "-resources.json")
        immutable_json(metrics_path, metrics)
        if resources_path.exists():
            # A prior closeout may have stopped after writing measured costs.
            # Preserve those exact bytes for idempotent ledger completion.
            prior = read(resources_path)
            require(all(prior.get(k) == v for k, v in resources.items() if k != "orchestration_seconds"),
                    "existing closeout resources differ from navigation report")
        else:
            immutable_json(resources_path, resources)
        arguments = ["complete" if success else "fail", "--access-id", access, "--resources", resources_path,
                     "--decision", "Record the preselected finite-loop outcome; no validation-based checkpoint or threshold changes"]
        if report is not None:
            arguments += ["--metrics", metrics_path, "--report", report_path]
        if not success:
            arguments += ["--error", error or "interrupted navigation attempt"]
        # Closeout and ledger backup must still run after a stop request. These
        # are short CPU writes, not new navigation; preserve the user's result.
        previous_stop = self.stop_requested
        self.stop_requested = False
        try:
            reconcile_ledgers(self.ledger, self.cloud_ledger)
            command = [sys.executable, self.root / "scripts/register_study_access.py", "--study",
                       self.root / "configs/research_study.json", "--ledger", self.ledger, *arguments]
            log = self.output / "logs" / (access + "-closeout.json")
            completed = subprocess.run(list(map(str, command)), cwd=self.root, env=self.child_env, capture_output=True, timeout=60)
            atomic_bytes(log, completed.stdout + completed.stderr)
            require(completed.returncode == 0, "navigation ledger closeout failed; preserve report and retry closeout only")
            reconcile_ledgers(self.ledger, self.cloud_ledger)
            verified_copy(log, self.backup / "logs" / log.name, mutable=True)
        finally:
            self.stop_requested = previous_stop
        for path in (metrics_path, resources_path):
            verified_copy(path, self.backup / path.relative_to(self.output))

    def evaluate(self):
        self.verify_source()
        frozen = read(self.frozen / "selected-heads.json")
        require(set(frozen["entries"]) == set(ARMS), "both selected heads must be frozen")
        sys.path.insert(0, str(self.root / "scripts"))
        from run_endpoint_intervention import code_identity
        from evaluate_endpoint_groups import cloud_execution_claim_path, execution_claim_path
        code_sha = code_identity(evaluation=True)[1]
        for arm in ARMS:
            self.ready()
            entry = frozen["entries"][arm]
            head, config = self.output / entry["frozen_head"], self.frozen / (arm + ".json")
            require(digest(head) == entry["head_sha256"] and digest(config) == entry["method_config_sha256"],
                    "selected head/config mutated after joint freeze")
            access, variant = (self.master["validation"][arm + "_" + key] for key in ("access_id", "variant_id"))
            report_path = self.output / "navigation" / (arm + ".json")
            self.restore_report(report_path.relative_to(self.output))
            reconcile_ledgers(self.ledger, self.cloud_ledger)
            registration, outcome = ledger_entry(self.ledger, access)
            if outcome is not None:
                require(outcome["event"] == "completed", "previous navigation attempt failed; same access ID will not be rerun")
                require(report_path.exists() and digest(report_path) == outcome["report_sha256"],
                        "completed navigation report absent/different")
                continue
            if registration is None:
                self.ledger_command("register-" + arm, ["register", "--access-id", access, "--category", "pilot",
                    "--variant-id", variant, "--config", config, "--checkpoint-sha256", entry["head_sha256"],
                    "--code-sha256", code_sha, "--purpose", "Prespecified full E2 loop; report selected trained head even if development gate fails",
                    "--seed", "0", "--expected-episodes", "2349"])
                registration, _ = ledger_entry(self.ledger, access)
            expected = {"variant_id": variant, "config_sha256": digest(config), "checkpoint_sha256": digest(head),
                        "code_sha256": code_sha, "expected_episodes": 2349, "seed": 0, "subset": False}
            require(all(registration["request"].get(k) == v for k, v in expected.items()), "existing registration differs")
            cloud_claim = cloud_execution_claim_path(self.claim_root, registration)
            local_claim = execution_claim_path(self.ledger, access)
            if cloud_claim.exists() or local_claim.exists():
                # Recover a finished evaluation without executing its policy again.
                claim = cloud_claim if cloud_claim.exists() else local_claim
                outcome_path = claim.with_name(claim.name.removesuffix(".claim.json") + ".outcome.json")
                execution_outcome = read(outcome_path) if outcome_path.exists() else {}
                if (execution_outcome.get("status") == "completed" and report_path.exists()
                        and digest(report_path) == execution_outcome.get("report_sha256")):
                    self.close_navigation(arm, access, report_path, success=True)
                    continue
                self.close_navigation(arm, access, report_path, success=False,
                    error="prior execution claim lacks a verified completed report; no automatic navigation rerun")
                raise RuntimeError("claimed navigation interrupted; a new explicit access ID is required, never reuse " + access)
            command = [sys.executable, self.root / "scripts/run_endpoint_intervention.py", "--mode", "eval", "--split", "val_unseen",
                "--config", self.root / "configs/r2r.json", "--experiment", config, "--head", head,
                "--baseline-report", self.root / BASELINE_RELATIVE, "--condition", "natural", "--seed", "0",
                "--output", report_path, "--backup", self.backup / "navigation", "--ledger", self.ledger,
                "--study", self.root / "configs/research_study.json", "--access-id", access,
                "--category", "pilot", "--execution-backup-root", self.claim_root]
            attempt = self.command("navigation-" + arm, command)
            if attempt["returncode"] == 0:
                require(report_path.exists(), "evaluation returned without report")
                self.close_navigation(arm, access, report_path, success=True, attempt=attempt)
            else:
                self.close_navigation(arm, access, report_path, success=False, attempt=attempt,
                    error="navigation child exit " + str(attempt["returncode"]))
                if attempt.get("stop_signal"):
                    raise ResumeRequired("navigation interrupted and recorded as failed; new access ID required before any retry")
                raise RuntimeError("navigation failed; fixed access will not be retried")

    def report(self):
        output = self.output / "loop-report.json"
        if self.restore_report(output.relative_to(self.output)) is None:
            r2r = read(self.root / "configs/r2r.json")
            dataset = self.root / r2r["dataset_root"]
            command = [sys.executable, self.root / "scripts/report_intervention_loop.py",
                "--baseline", self.root / BASELINE_RELATIVE, "--relative", self.output / "navigation/relative.json",
                "--absolute", self.output / "navigation/absolute.json", "--annotations", dataset / "R2R/annotations/R2R_val_unseen_enc.json",
                "--connectivity", dataset / "R2R/connectivity", "--output", output]
            self.checked_command("independent-report", command)
            verified_copy(output, self.backup / output.name)
        self.state["status"], self.state["current_stage"] = "complete", None
        self.state["report_sha256"] = digest(output)
        self.state["completed_utc"] = now()
        self.save_state()
        self.package_records()

    def package_records(self):
        """Small download bundle: reports and code metadata, never cache tensors."""
        chosen = sorted(p for p in self.output.rglob("*") if p.is_file()
                        and p.suffix in {".json", ".jsonl", ".log"}
                        and "cache" not in p.relative_to(self.output).parts
                        and "snapshots" not in p.relative_to(self.output).parts
                        and p.name != "small-records.manifest.json")
        inventory = {str(p.relative_to(self.output)): {"sha256": digest(p), "bytes": p.stat().st_size} for p in chosen}
        archive = self.output / "small-records.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in chosen:
                bundle.add(path, arcname=str(path.relative_to(self.output)), recursive=False)
        with tarfile.open(archive) as bundle:
            for item in bundle.getmembers():
                require(item.isfile() and hashlib.sha256(bundle.extractfile(item).read()).hexdigest() == inventory[item.name]["sha256"],
                        "small-records archive failed verification")
        manifest = {"schema": "e2_small_records_v1", "files": inventory, "archive_sha256": digest(archive),
                    "source_archive": "frozen/source.tar.gz", "source_archive_sha256": self.source["archive_sha256"]}
        atomic_bytes(self.output / "small-records.manifest.json", json_bytes(manifest))
        verified_copy(archive, self.backup / archive.name, mutable=True)
        verified_copy(self.output / "small-records.manifest.json", self.backup / "small-records.manifest.json", mutable=True)

    def run(self):
        self.collect()
        self.train()
        self.evaluate()
        self.report()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    for key in ("output-dir", "backup-dir", "cloud-ledger"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--master-config", type=Path, default=Path("configs/e2_loop_v1.json"))
    parser.add_argument("--execution-backup-root", type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Single local writer. Drive is an append-only backup, not a distributed lock.
    with (args.output_dir / ".orchestrator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another orchestrator already owns this local experiment") from error
        loop = Loop(args)
        def stop(signum, frame):
            loop.stop_requested = True
        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            loop.run()
            print(json.dumps({"status": "complete", "output": str(loop.output / "loop-report.json"),
                              "backup": str(loop.backup), "report_sha256": loop.state["report_sha256"]}), flush=True)
        except BaseException as error:
            loop.state.update(status="resume_required" if isinstance(error, ResumeRequired) else "failed",
                              error_type=type(error).__name__, error=str(error))
            try:
                loop.save_state()
                loop.package_records()
            except BaseException as backup_error:
                print(json.dumps({"backup_error": str(backup_error), "original_error": str(error)}), flush=True)
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
