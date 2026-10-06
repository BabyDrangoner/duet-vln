#!/usr/bin/env python3
"""Real-feature E2 trainer/recovery/navigation engineering smoke on train splits.

This script uses exactly eight natural fit, eight perturbed fit and eight
natural development records. It never opens validation/test annotations. The
optional local backup exception proves recovery mechanics only, not Drive
durability or research improvement.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def assert_tree_equal(left, right, path="state"):
    import torch
    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu())):
            raise AssertionError(f"exact tensor recovery failed at {path}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise AssertionError(f"recovery dictionary keys differ at {path}")
        for key in left:
            assert_tree_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"recovery sequence differs at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_tree_equal(a, b, f"{path}[{index}]")
    elif type(left) is not type(right) or left != right:
        raise AssertionError(f"recovery scalar differs at {path}")


def validate_smoke_caches(train, dev, identity):
    from vln_improve.intervention_training import validate_splits
    validate_splits(train, dev)
    manifests = [*identity["fit"], identity["dev"]]
    if (len(identity["fit"]) != 2 or any(m["collection"].get("scope") != "engineering_smoke" for m in manifests)
            or any(m["provenance"] != manifests[0]["provenance"] for m in manifests)):
        raise ValueError("smoke requires two fit caches and one dev cache with shared engineering provenance")
    natural = [r for r in train if r["condition"] == "natural"]
    perturbed = [r for r in train if r["condition"] == "perturb_step2"]
    if (len(natural) != 8 or len(perturbed) != 8 or len(dev) != 8
            or {r["instr_id"] for r in natural} != {r["instr_id"] for r in perturbed}):
        raise ValueError("smoke requires the same eight fit instructions under both conditions and eight natural dev instructions")
    identity["scope"] = "engineering_smoke"


def verify_current_assets(runtime_config, provenance):
    from vln_improve.protocol import file_sha256, object_sha256, resolve_config
    import run_duet
    cfg = resolve_config(runtime_config, ROOT)
    dataset = Path(cfg["dataset_root"])
    current = {
        "base_checkpoint_sha256": file_sha256(cfg["base_checkpoint"]),
        "feature_sha256": file_sha256(dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"),
        "train_annotation_sha256": file_sha256(dataset / "R2R/annotations/R2R_train_enc.json"),
        "connectivity_sha256": object_sha256({p.name: file_sha256(p) for p in
            sorted((dataset / "R2R/connectivity").glob("*_connectivity.json"))}),
        "model": cfg["model"], "partition_seed": cfg["partition_seed"], "dev_fraction": cfg["dev_fraction"],
        "upstream_lock": run_duet.verify(),
    }
    if cfg["model"]["batch_size"] != 1 or any(provenance.get(k) != v for k, v in current.items()):
        raise ValueError("smoke navigation assets/partition differ from authenticated training caches")
    return current


def run_real_navigation(model, dev, expected, runtime_config, output):
    """Execute trained head on exactly the cached train_dev instruction IDs."""
    import torch
    import run_duet
    from vln_improve.intervention_runtime import TerminalIntervention
    from vln_improve.intervention_training import predict_gains
    sys.path.insert(0, str(run_duet.DEFAULT_DEST / "map_nav_src"))
    import r2r.agent as upstream

    records = {r["instr_id"]: r for r in dev}
    predicted = {r["instr_id"]: r for r in expected["per_episode"]}
    baseline_paths = {key: r["candidate_paths"][r["inputs"]["baseline_index"]] for key, r in records.items()}
    original_class, original_parse, original_partition = upstream.GMapNavAgent, run_duet.parse_cli, run_duet.select_partition
    completed = {}

    def partition(data, split, dev_fraction, partition_seed):
        if split != "train_dev":
            raise ValueError("engineering trained-head navigation is restricted to train_dev")
        rows = original_partition(data, split, dev_fraction, partition_seed)
        chosen = [r for r in rows if r["instr_id"] in records]
        if {r["instr_id"] for r in chosen} != set(records) or len(chosen) != 8:
            raise ValueError("cached development IDs are not the exact live train_dev inventory")
        return chosen

    class Agent(original_class):
        def rollout(self, *positional, **kwargs):
            if positional or getattr(self, "decision_hook", None) is not None:
                raise ValueError("unexpected existing rollout hook")
            hook = TerminalIntervention(self, condition="natural", seed=0, model=model,
                predict_gains=predict_gains, collect=False, baseline_paths=baseline_paths)
            old_move = self.make_equiv_action
            self.decision_hook, self.make_equiv_action = hook, hook.make_equiv_action
            try:
                result = super().rollout(**kwargs)
            finally:
                self.decision_hook, self.make_equiv_action = None, old_move
            decision = hook.finish(result)
            key = decision["instr_id"]
            if key in completed and decision != completed[key]:
                raise ValueError("repeated engineering navigation decision differs")
            completed[key] = decision
            return result

    try:
        upstream.GMapNavAgent, run_duet.select_partition = Agent, partition
        run_duet.parse_cli = lambda: SimpleNamespace(mode="baseline", config=runtime_config, split="train_dev",
            output=output, cache=None, head=None, limit=None, seed=0)
        run_duet.main()
    finally:
        upstream.GMapNavAgent, run_duet.parse_cli, run_duet.select_partition = original_class, original_parse, original_partition
    report = json.loads(output.read_text())
    if set(completed) != set(records):
        raise ValueError("real navigation instruction coverage differs from dev cache")
    episodes = {r["instr_id"]: r for r in report["episodes"]}
    paths = {r["instr_id"]: r["trajectory"] for r in report["trajectories"]}
    if set(episodes) != set(records) or set(paths) != set(records):
        raise ValueError("live metric/trajectory inventory differs from smoke inventory")
    maximum_metric_error = 0.
    for key, record in records.items():
        endpoint = predicted[key]["endpoint"]
        index = record["inputs"]["candidate_vpids"].index(endpoint)
        if completed[key]["selected_endpoint"] != endpoint or paths[key] != record["candidate_paths"][index]:
            raise ValueError("trained live endpoint/actual full return differs from cached candidate decision")
        for metric, value in record["candidate_metrics"][index].items():
            error = abs(float(episodes[key][metric]) - float(value))
            maximum_metric_error = max(maximum_metric_error, error)
            if not math.isfinite(error) or error > 1e-12:
                raise ValueError(f"actual full-route metric differs for {key}: {metric}")
    report["metadata"].update(mode="e2_trained_head_engineering_smoke", scope="engineering_smoke",
        new_val_unseen_accesses=0, baseline_online_path_and_termination_parity=True,
        cached_dev_endpoints_and_full_trajectories_match=True, cached_candidate_metrics_match=True,
        policy_supervision_access=False, head_metadata=model.training_metadata)
    report["endpoint_decisions"] = [completed[k] for k in sorted(completed)]
    report["verification"] = {"episodes": 8, "maximum_metric_absolute_error": maximum_metric_error,
        "candidate_metric_comparisons": sum(len(r["candidate_metrics"][0]) for r in dev),
        "selected_endpoint_and_executed_trajectory_exact": True}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-cache", type=Path, action="append", required=True)
    parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--allow-local-backup-for-tests", action="store_true")
    args = parser.parse_args(argv)
    from vln_improve.pipeline import atomic_json, validate_backup_root
    from vln_improve.protocol import file_sha256
    from vln_improve.checkpoint_store import CheckpointStore
    from vln_improve.intervention_runtime import verified_copy
    from vln_improve.intervention_training import code_identity, evaluate_records, load_head, train_intervention
    from train_endpoint_intervention import load_training_caches
    import torch

    if args.output_dir.exists() or args.backup_dir.exists():
        raise ValueError("use new output and backup directories; previous attempts must remain unchanged")
    paths = [p.resolve() for p in [*args.fit_cache, args.dev_cache, args.output_dir, args.backup_dir]]
    if any(left.is_relative_to(right) or right.is_relative_to(left)
           for index, left in enumerate(paths) for right in paths[index + 1:]):
        raise ValueError("smoke caches/output/backup must be separate nonnested directories")
    backup_kind = validate_backup_root(args.backup_dir, allow_local=args.allow_local_backup_for_tests)
    if not torch.cuda.is_available():
        raise RuntimeError("this integration smoke requires a real NVIDIA GPU")
    train, dev, identity = load_training_caches(args.fit_cache, args.dev_cache)
    validate_smoke_caches(train, dev, identity)
    assets = verify_current_assets(args.config, identity["dev"]["provenance"])
    args.output_dir.mkdir(parents=True)
    args.backup_dir.mkdir(parents=True)
    checker = lambda: validate_backup_root(args.backup_dir, allow_local=args.allow_local_backup_for_tests)
    started = time.monotonic()
    source = {**code_identity(), "scripts/verify_intervention_loop_smoke.py": file_sha256(Path(__file__))}
    results = {}
    for arm in ("relative", "absolute"):
        print(json.dumps({"stage": "engineering_training_and_recovery", "arm": arm}), flush=True)
        arm_dir, cloud = args.output_dir / arm, args.backup_dir / arm
        arm_dir.mkdir(); cloud.mkdir()
        spec = {"scope": "engineering_smoke", "arm": arm, "seed": 0, "epochs": 2, "batch_size": 4,
            "hidden_dim": 128, "lr": 1e-4, "weight_decay": .01, "monitor_every_epochs": 1,
            "risk_weight": 0., "inference_batch_size": 1, "train_episodes": len(train), "dev_episodes": len(dev),
            "collection_config_sha256": identity["dev"]["provenance"]["experiment_sha256"],
            "interruption_after_updates": 3, "backup_verification": backup_kind,
            "cache_identities": identity, "source_files": source}
        config_path = arm_dir / "smoke-config.json"
        atomic_json(config_path, spec); verified_copy(config_path, cloud / config_path.name)
        training_keys = ("arm", "seed", "epochs", "batch_size", "hidden_dim", "lr", "weight_decay",
                         "monitor_every_epochs", "risk_weight", "inference_batch_size")
        config = {k: spec[k] for k in training_keys}
        config["experiment_sha256"] = file_sha256(config_path)
        shared = dict(train=train, dev=dev, config=config, data_identity=identity,
            device="cuda", checkpoint_every_steps=1, checkpoint_every_seconds=30.,
            keep_local=2, keep_backup=5, verify_backup=checker)
        full = train_intervention(**shared, local_dir=arm_dir / "continuous", backup_dir=cloud / "continuous")
        interrupted = train_intervention(**shared, local_dir=arm_dir / "interrupted", backup_dir=cloud / "resume",
            stop_after_updates=3)
        if interrupted["status"] != "interrupted" or interrupted["global_step"] != 3:
            raise AssertionError("trainer did not stop at the requested complete update boundary")
        # A fresh path forces restoration solely from the separate backup store.
        restored_dir = arm_dir / "restored-from-empty"
        if restored_dir.exists():
            raise AssertionError("recovery target is not empty")
        recovered = train_intervention(**shared, local_dir=restored_dir, backup_dir=cloud / "resume")
        if full["status"] != "complete" or recovered["status"] != "complete" or not recovered["resumed"]:
            raise AssertionError("continuous or recovered training did not complete")
        full_state, full_head, _ = CheckpointStore(arm_dir / "continuous", cloud / "continuous").restore("latest")
        recovered_state, recovered_head, _ = CheckpointStore(restored_dir, cloud / "resume").restore("latest")
        assert_tree_equal(full_state, recovered_state)
        assert_tree_equal(full_head, recovered_head, "exported_head")
        for field in ("training_history", "dev_history", "best_epoch", "best_metrics", "initial_head_sha256"):
            assert_tree_equal(full[field], recovered[field], field)
        selected = recovered["selected_checkpoint"]
        selected_path = restored_dir / selected["head_relative_path"]
        standalone_head = arm_dir / "selected-head.pt"
        verified_copy(selected_path, standalone_head)
        model = load_head(standalone_head, device="cuda")
        cached = evaluate_records(model, dev, batch_size=1, device="cuda")
        selected_monitor = next(r for r in recovered["dev_history"] if r["epoch"] == selected["epoch"])
        assert_tree_equal(cached["per_episode"], selected_monitor["per_episode"], "selected_cached_inference")
        navigation_path = arm_dir / "trained-head-train-dev-8.json"
        print(json.dumps({"stage": "engineering_real_navigation", "arm": arm, "split": "train_dev", "episodes": 8}), flush=True)
        navigation = run_real_navigation(model, dev, cached, args.config, navigation_path)
        atomic_json(navigation_path, navigation)
        recovery_report = {"scope": "engineering_smoke", "status": "passed", "arm": arm,
            "backup_verification": backup_kind, "drive_durability_verified": not args.allow_local_backup_for_tests,
            "interruption_update": 3, "restored_from_fresh_local_path": True,
            "exact_model_optimizer_rng_sampler_history_recovery": True,
            "selected_head_cached_dev_inference_exact": True,
            "continuous": full, "interrupted": interrupted, "restored": recovered,
            "cached_selected_dev": cached, "navigation_verification": navigation["verification"],
            "selected_head_sha256": file_sha256(standalone_head),
            "navigation_report_sha256": file_sha256(navigation_path)}
        atomic_json(arm_dir / "recovery.json", recovery_report)
        for path in (standalone_head, navigation_path, arm_dir / "recovery.json"):
            checker(); verified_copy(path, cloud / path.name)
        results[arm] = {k: recovery_report[k] for k in ("status", "exact_model_optimizer_rng_sampler_history_recovery",
            "selected_head_cached_dev_inference_exact", "selected_head_sha256", "navigation_report_sha256", "navigation_verification")}
        results[arm].update(global_step=recovered["global_step"], selected_epoch=selected["epoch"],
            parameter_count=recovered["parameter_count"], changes=cached["changes"],
            cached_sr=cached["sr"], cached_spl=cached["spl"], selection_reason=selected["selection_reason"])
        del model, full_state, recovered_state, full_head, recovered_head
        gc.collect(); torch.cuda.empty_cache()
    if source != {**code_identity(), "scripts/verify_intervention_loop_smoke.py": file_sha256(Path(__file__))}:
        raise ValueError("training/smoke source changed during the integration run")
    report = {"schema": "e2_formal_trainer_real_navigation_smoke_v1", "status": "passed",
        "scope": "engineering_only_16_fit_8_dev_real_records_not_method_gain_evidence",
        "fit_episodes": len(train), "dev_episodes": len(dev), "new_val_unseen_accesses": 0,
        "new_test_accesses": 0, "gpu": torch.cuda.get_device_name(0),
        "backup_verification": backup_kind, "drive_durability_verified": not args.allow_local_backup_for_tests,
        "backup_limitation": "local runtime files may disappear on VM loss" if args.allow_local_backup_for_tests else None,
        "arms": results, "current_assets": assets, "source_sha256": source,
        "elapsed_seconds": time.monotonic() - started}
    atomic_json(args.output_dir / "smoke.json", report)
    checker(); verified_copy(args.output_dir / "smoke.json", args.backup_dir / "smoke.json")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
