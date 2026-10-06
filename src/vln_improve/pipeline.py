"""Single-writer, interruptible training with verified persistent checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Callable
import uuid

from .checkpoint_store import CheckpointStore
from .resumable import ResumableTrainer
from .run_assets import prepare_inputs, snapshot_code

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def backup_mount_identity(path: Path, *, backend: str = "drive", allow_local: bool = False) -> dict:
    """Validate an explicit backup backend and identify its deepest Linux mount.

    Filesystem backup protects process/restart recovery. A directory on the same
    disk is not protection against disk failure. No directory is created here.
    """
    if backend not in {"drive", "filesystem"}:
        raise ValueError("backup backend must be drive or filesystem")
    if allow_local:
        if backend != "drive":
            raise ValueError("filesystem backup cannot use the test-only local bypass")
        return {"backend": "test-only", "verification": "local-filesystem-test"}
    path = Path(path).expanduser()
    if backend == "filesystem" and not path.is_dir():
        raise RuntimeError(f"filesystem backup root must be an existing directory: {path}")
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise RuntimeError("backup mount verification requires Linux /proc/self/mountinfo")
    target = path.resolve()
    mounts = []
    for line in mountinfo.read_text().splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields, details = left.split(), right.split()
        if len(fields) < 6 or len(details) < 2:
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
        if target == mount or target.is_relative_to(mount):
            mounts.append((len(mount.parts), {"backend": backend, "mount_point": str(mount),
                "mount_root": fields[3], "filesystem": details[0], "source": details[1],
                "device": fields[2], "mount_id": fields[0], "mount_options": fields[5].split(",")}))
    deepest = max((depth for depth, _ in mounts), default=-1)
    matches = [item for depth, item in mounts if depth == deepest]
    if len(matches) > 1:
        raise RuntimeError("ambiguous stacked backup mounts at the same path; refusing to select a covered mount")
    identity = matches[0] if matches else None
    if backend == "drive" and (identity is None or identity["filesystem"] not in {"fuse.drivefs", "fuse.drive"}):
        raise RuntimeError(f"Google Drive is not mounted at backup destination: {path}")
    if backend == "filesystem":
        unsuitable = {"tmpfs", "ramfs", "rootfs", "overlay", "fuse.overlayfs", "fuse-overlayfs",
                      "unionfs", "fuse.unionfs", "aufs", "devtmpfs", "proc", "sysfs",
                      "cgroup", "cgroup2", "squashfs", "debugfs", "tracefs", "securityfs", "fusectl", "mqueue"}
        if identity is None or identity["filesystem"] in unsuitable or identity["source"].startswith(("/dev/ram", "/dev/zram")):
            raise RuntimeError(f"filesystem backup needs a persistent filesystem; mounted type is {identity['filesystem'] if identity else 'unknown'}: {path}")
        if "ro" in identity["mount_options"]:
            raise RuntimeError(f"filesystem backup mount is read-only: {path}")
    identity["verification"] = "drive-mount-readback-sha256" if backend == "drive" else "filesystem-mount-readback-sha256"
    return identity


def validate_backup_root(path: Path, *, backend: str = "drive", allow_local: bool = False,
                         expected_identity: dict | None = None) -> str:
    """Default to Drive; detect a lost/remounted filesystem before any new write."""
    identity = backup_mount_identity(path, backend=backend, allow_local=allow_local)
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError("backup mount identity changed; refusing fallback to a parent or replacement mount")
    return identity["verification"]


def validate_separate_roots(local: Path, backup: Path) -> None:
    local, backup = Path(local).resolve(), Path(backup).resolve()
    if local.is_relative_to(backup) or backup.is_relative_to(local):
        raise ValueError("local and backup roots must be separate, non-nested directories")


class MountedCheckpointStore(CheckpointStore):
    """Check the selected mount again before publishing pointers or pruning."""
    def __init__(self, *args, backup_check: Callable[[], object], **kwargs):
        self.backup_check = backup_check
        self.backup_check()
        super().__init__(*args, **kwargs)

    def _copy_snapshot(self, source, root):
        self.backup_check()
        result = super()._copy_snapshot(source, root)
        self.backup_check()
        return result

    def _update_pointers(self, root, directory, *, is_best):
        self.backup_check()
        return super()._update_pointers(root, directory, is_best=is_best)

    def _prune(self):
        self.backup_check()
        return super()._prune()


def validate_config(config: dict) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("pipeline config requires schema_version=1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", config.get("run_id", "")):
        raise ValueError("run_id must be a simple, unique directory name")
    if config.get("scope") not in {"research", "smoke"}:
        raise ValueError("scope must be research or smoke")
    if config.get("backup_backend", "drive") not in {"drive", "filesystem"}:
        raise ValueError("backup_backend must be drive or filesystem")
    if not isinstance(config.get("cache"), list) or not config["cache"] or not all(
        isinstance(p, str) and p for p in config["cache"]
    ):
        raise ValueError("cache must contain train_fit cache paths")
    for name in ("local_root", "backup_root"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"{name} is required")
    for name in ("checkpoint_every_steps", "keep_local", "keep_backup"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("checkpoint_every_seconds", "max_process_seconds", "evaluation_reserve_seconds"):
        if not isinstance(config.get(name), (int, float)) or not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    age = config.get("max_vm_age_seconds")
    if age is not None and (not isinstance(age, (int, float)) or not math.isfinite(age) or age <= 0):
        raise ValueError("max_vm_age_seconds must be null or positive")
    validation = config.get("validation", {})
    if validation.get("split") != "train_dev":
        raise ValueError("best selection is restricted to train_dev; reserve official validation for reporting")
    if not isinstance(validation.get("config"), str):
        raise ValueError("validation.config is required")
    limit = validation.get("limit")
    if limit is not None and (config["scope"] != "smoke" or type(limit) is not int or limit < 1):
        raise ValueError("subset best selection is allowed only for explicitly labelled smoke runs")
    if type(validation.get("timeout_seconds")) is not int or validation["timeout_seconds"] < 1:
        raise ValueError("validation.timeout_seconds must be positive")
    if not isinstance(config.get("training"), dict):
        raise ValueError("training settings are required")


def evaluate_head(head: Path, output: Path, options: dict, should_stop: Callable[[], bool]) -> dict:
    command = [sys.executable, str(Path(options["project_root"]) / "scripts/run_duet.py"),
               "--config", options["config"], "--mode", "eval", "--split", "train_dev",
               "--head", str(head), "--output", str(output)]
    if options.get("limit") is not None:
        command += ["--limit", str(options["limit"])]
    command = [sys.executable, str(Path(__file__).with_name("child_guard.py")),
               "--parent-pid", str(os.getpid()), "--", *command]
    with output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, cwd=options["project_root"], stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        start = time.monotonic()
        try:
            while process.poll() is None:
                if should_stop():
                    raise InterruptedError("validation interrupted; pending evaluation will be resumed")
                if time.monotonic() - start > options["timeout_seconds"]:
                    raise TimeoutError("navigation validation exceeded timeout")
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Child can exit between poll() and killpg().
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        if process.returncode:
            raise RuntimeError(f"navigation validation failed ({process.returncode}); see {output.with_suffix('.log')}")
    return json.loads(output.read_text())


def navigation_metrics(report: dict, head: Path, *, scope: str, expected_protocol: str | None) -> dict:
    metadata, episodes = report["metadata"], report["episodes"]
    if metadata.get("split") != "train_dev" or metadata.get("mode") != "eval":
        raise ValueError("invalid validation split or mode")
    subset = metadata.get("subset")
    if type(subset) is not bool or (scope == "research" and subset):
        raise ValueError("subset evaluation cannot select a research best checkpoint")
    protocol = metadata.get("protocol_sha256", "")
    if not isinstance(protocol, str) or not re.fullmatch(r"[0-9a-f]{64}", protocol):
        raise ValueError("missing evaluation protocol fingerprint")
    if expected_protocol is not None and protocol != expected_protocol:
        raise ValueError("evaluation protocol changed within this run")
    if metadata.get("head_sha256") != digest(head):
        raise ValueError("evaluation report belongs to a different checkpoint")
    if not episodes or len(episodes) != metadata.get("num_episodes"):
        raise ValueError("incomplete evaluation report")
    if len({e["instr_id"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate evaluation instruction IDs")
    for e in episodes:
        for key in ("success", "spl"):
            if not isinstance(e.get(key), (int, float)) or not math.isfinite(e[key]) or not 0 <= e[key] <= 1:
                raise ValueError(f"invalid per-episode {key}")
    values = {"sr": 100 * sum(e["success"] for e in episodes) / len(episodes),
              "spl": 100 * sum(e["spl"] for e in episodes) / len(episodes)}
    for key, value in values.items():
        summary = report["summary"].get(key)
        if not isinstance(summary, (int, float)) or not math.isfinite(summary) or abs(summary - value) > 1e-6:
            raise ValueError(f"summary {key} does not match episode results")
    return dict(values, protocol_sha256=protocol, split="train_dev", scope=scope,
                subset=subset, num_episodes=len(episodes), head_sha256=metadata["head_sha256"])


def run_pipeline(config: dict, *, project_root: Path = ROOT, allow_local_backup: bool = False,
                 stop_after_steps: int | None = None, require_resume: bool = False,
                 evaluator: Callable = evaluate_head) -> dict:
    validate_config(config)
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 1):
        raise ValueError("stop_after_steps must be positive")
    project_root = Path(project_root).resolve()
    resolve = lambda p: (project_root / Path(p).expanduser()).resolve() if not Path(p).expanduser().is_absolute() else Path(p).expanduser().resolve()
    local = resolve(config["local_root"]) / config["run_id"]
    backup_root = resolve(config["backup_root"])
    backup = backup_root / config["run_id"]
    backend = config.get("backup_backend", "drive")
    validate_separate_roots(local, backup)
    mount_identity = backup_mount_identity(backup_root if backend == "filesystem" else backup,
                                          backend=backend, allow_local=allow_local_backup)
    verification = mount_identity["verification"]
    def backup_check():
        return validate_backup_root(backup_root, backend=backend, allow_local=allow_local_backup,
                                    expected_identity=mount_identity)
    store = MountedCheckpointStore(local, backup, keep_local=config["keep_local"], keep_backup=config["keep_backup"],
                                   backup_check=backup_check)
    start = time.monotonic()
    requested = []
    old_handlers = {}

    def remaining() -> float:
        value = config["max_process_seconds"] - (time.monotonic() - start)
        max_age = config.get("max_vm_age_seconds")
        if max_age is not None:
            uptime = Path("/proc/uptime")
            if not uptime.is_file():
                raise RuntimeError("VM-age budget requires Linux /proc/uptime")
            value = min(value, max_age - float(uptime.read_text().split()[0]))
        return value

    def should_stop() -> bool:
        return bool(requested) or remaining() <= 0

    def handler(signum, frame):
        requested.append(signal.Signals(signum).name)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, handler)
        with store.lock():
            backup_check()
            caches = prepare_inputs([resolve(p) for p in config["cache"]], local, backup)
            trainer = ResumableTrainer(caches, **config["training"])
            source_files = sorted((project_root / "src/vln_improve").glob("*.py"))
            source_files += [project_root / "scripts/run_duet.py", project_root / "configs/upstream.json"]
            source_id = {str(p.relative_to(project_root)): digest(p) for p in source_files if p.is_file()}
            options = dict(config["validation"], config=str(resolve(config["validation"]["config"])),
                           project_root=str(project_root))
            identity = {"scope": config["scope"], "validation": config["validation"],
                        "validation_config_sha256": digest(Path(options["config"])), "source": source_id,
                        "backup_backend": backend, "backup_root": str(backup_root)}
            control = {"best": None, "last_evaluated_epoch": 0, "evaluation_pending": False,
                       "validation_protocol_sha256": None, "validation_history": []}
            restored = False
            try:
                state, _, manifest = store.restore("latest")
            except FileNotFoundError:
                if require_resume:
                    raise
            else:
                if state.get("pipeline_schema") != 1 or state.get("identity") != identity:
                    raise ValueError("pipeline source or validation settings changed; use a new run_id")
                trainer.load_state_dict(state["trainer"])
                control = state["control"]
                restored = True
            if not restored:
                backup_check()
                snapshot_code(project_root, backup)
            atomic_json(local / "config.json", config)
            backup_check()
            atomic_json(backup / "config.json", config)
            last_saved_step, last_saved_time = trainer.global_step, time.monotonic()
            current_id = manifest["checkpoint_id"] if restored else None
            updates = 0

            def persist(reason: str, *, is_best=False, metrics=None):
                nonlocal last_saved_step, last_saved_time, current_id
                validate_backup_root(backup, backend=backend, allow_local=allow_local_backup,
                                     expected_identity=mount_identity)
                payload = {"pipeline_schema": 1, "identity": identity, "trainer": trainer.state_dict(),
                           "control": control, "reason": reason, "backup_verification": verification,
                           "backup_mount_identity": mount_identity}
                current_id = store.save(payload, trainer.head_payload(), step=trainer.global_step,
                                        is_best=is_best, metrics=metrics)
                last_saved_step, last_saved_time = trainer.global_step, time.monotonic()
                print(json.dumps({"event": "checkpoint_backed_up", "step": trainer.global_step,
                                  "reason": reason, "is_best": is_best, "checkpoint_id": current_id}), flush=True)

            def finish(kind, reason):
                result = {"status": kind, "reason": reason, "global_step": trainer.global_step,
                          "epoch": trainer.epoch, "best": control["best"], "resumed": restored,
                          "verification": verification, "backup_backend": backend,
                          "backup_mount_identity": mount_identity,
                          "local_run": str(local), "backup_run": str(backup)}
                atomic_json(local / "status.json", result)
                backup_check()
                atomic_json(backup / "status.json", result)
                return result

            if not restored:
                persist("initial")
            try:
                while True:
                    if should_stop() or (stop_after_steps is not None and updates >= stop_after_steps):
                        persist("graceful_stop")
                        return finish("paused", requested[-1] if requested else "budget_or_step_limit")
                    if trainer.epoch > control["last_evaluated_epoch"] or control["evaluation_pending"]:
                        control["evaluation_pending"] = True
                        persist("validation_pending")
                        if remaining() < config["evaluation_reserve_seconds"]:
                            return finish("paused", "insufficient_time_for_validation")
                        head = local / "snapshots" / current_id / "head.pt"
                        report_path = local / f"eval-epoch{trainer.epoch}-step{trainer.global_step}-{uuid.uuid4().hex[:8]}.json"
                        report = evaluator(head, report_path, options, should_stop)
                        metrics = navigation_metrics(report, head, scope=config["scope"],
                                                     expected_protocol=control["validation_protocol_sha256"])
                        metrics.update(epoch=trainer.epoch, global_step=trainer.global_step)
                        best = control["best"]
                        is_best = best is None or (metrics["sr"], metrics["spl"]) > (best["sr"], best["spl"])
                        if is_best:
                            control["best"] = metrics
                        control["validation_protocol_sha256"] = metrics["protocol_sha256"]
                        control["validation_history"].append(metrics)
                        control["last_evaluated_epoch"] = trainer.epoch
                        control["evaluation_pending"] = False
                        persist("validation_complete", is_best=is_best, metrics=metrics)
                        backup_check()
                        atomic_json(backup / report_path.name, report)
                    if trainer.done:
                        return finish("complete", "target_epochs_completed")
                    trainer.step()
                    updates += 1
                    if (trainer.global_step - last_saved_step >= config["checkpoint_every_steps"]
                            or time.monotonic() - last_saved_time >= config["checkpoint_every_seconds"]):
                        persist("periodic")
            except InterruptedError:
                persist("interrupted_validation")
                return finish("paused", "validation_interrupted")
            except Exception:
                # A failed backup already retains its local snapshot. Never pretend
                # it is durable or prune it by retrying with different run settings.
                raise
    finally:
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(json.loads(args.config.read_text()), require_resume=args.require_resume,
                          stop_after_steps=args.stop_after_steps,
                          allow_local_backup=args.allow_local_backup_for_tests)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(75 if result["status"] == "paused" else 0)


if __name__ == "__main__":
    main()
