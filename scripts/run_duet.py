#!/usr/bin/env python3
"""Baseline, exact zero-residual parity, training capture, and closed-loop evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prepare_duet import verify, DEFAULT_DEST
from vln_improve.protocol import file_sha256, object_sha256, resolve_config, select_partition


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "identity", "collect", "eval"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    parser.add_argument("--split", choices=["train_fit", "train_dev", "val_seen", "val_unseen"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--limit", type=int, help="Smoke check only; results marked as a subset")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    cli = parse_cli()
    if cli.mode == "collect" and (cli.split != "train_fit" or cli.cache is None):
        raise ValueError("Collection is permitted only on train_fit, with --cache")
    if cli.mode == "eval" and cli.head is None:
        raise ValueError("eval requires --head")
    if cli.mode in ("baseline", "identity") and cli.head is not None:
        raise ValueError("baseline/identity must not load a trained head")
    if cli.limit is not None and cli.limit < 1:
        raise ValueError("--limit must be positive")
    if cli.output.exists():
        raise ValueError(f"Refusing to overwrite a run: {cli.output}")
    lock = verify()
    cfg = resolve_config(cli.config, ROOT)
    dataset = Path(cfg["dataset_root"])
    checkpoint = Path(cfg["base_checkpoint"])
    raw_split = "train" if cli.split.startswith("train_") else cli.split
    annotation = dataset / f"R2R/annotations/R2R_{raw_split}_enc.json"
    train_annotation = dataset / "R2R/annotations/R2R_train_enc.json"
    feature_file = dataset / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
    for path in (checkpoint, annotation, train_annotation, feature_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required asset missing: {path}; see README.md")

    import numpy as np
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError("DUET rollout needs an NVIDIA CUDA runtime; local CPU tests are separate")
    torch.cuda.set_device(0)
    sys.path.insert(0, str(DEFAULT_DEST / "map_nav_src"))
    from r2r.parser import parse_args
    from r2r.data_utils import construct_instrs
    from r2r.env import R2RNavBatch
    from r2r.agent import GMapNavAgent
    from utils.data import ImageFeaturesDB
    from vln_improve.capture import CacheWriter, DecisionHook
    from vln_improve.features import FEATURE_SCHEMA
    from vln_improve.head import ResidualActionHead, load_head_checkpoint

    # Pass only explicitly recorded model settings to the upstream parser.
    argv = ["duet", "--root_dir", str(dataset), "--output_dir", str(cli.output.parent / "upstream_logs"),
            "--seed", str(cli.seed), "--test"]
    for key, value in cfg["model"].items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv.extend(["--" + key, str(value)])
    old_argv = sys.argv
    try:
        sys.argv = argv
        args = parse_args()
    finally:
        sys.argv = old_argv
    for key, value in cfg["model"].items():
        if not hasattr(args, key) or getattr(args, key) != value:
            raise ValueError(f"Unknown or unapplied upstream model option: {key}")
    data = construct_instrs(args.anno_dir, "r2r", [raw_split], "bert", args.max_instr_len, is_test=True)
    data = select_partition(data, cli.split, cfg["dev_fraction"], cfg["partition_seed"])
    data = sorted(data, key=lambda row: row["instr_id"])
    if cli.limit is not None:
        data = data[:cli.limit]
    if not data:
        raise ValueError("Selected split is empty")
    args.batch_size = min(args.batch_size, len(data))
    scan_ids = sorted({row["scan"] for row in data})
    graph_hashes = {scan: file_sha256(Path(args.connectivity_dir) / f"{scan}_connectivity.json") for scan in scan_ids}
    # The simulator uses this graph to construct angle features even for small subsets.
    angle_graph = Path(args.connectivity_dir) / "ZMojNkEp431_connectivity.json"
    graph_hashes["ZMojNkEp431"] = file_sha256(angle_graph)
    all_graph_hashes = {path.name: file_sha256(path)
                        for path in sorted(Path(args.connectivity_dir).glob("*_connectivity.json"))}
    implementation_hashes = {name: file_sha256(ROOT / name) for name in (
        "scripts/run_duet.py", "src/vln_improve/features.py", "src/vln_improve/head.py",
        "src/vln_improve/capture.py", "src/vln_improve/protocol.py")}
    provenance = {
        "dataset": "r2r", "feature_id": file_sha256(feature_file),
        "base_checkpoint_sha256": file_sha256(checkpoint),
        "upstream_commit": lock["commit"], "max_action_len": args.max_action_len,
        "feedback": "argmax", "feature_schema": FEATURE_SCHEMA,
        "model_config_sha256": object_sha256({k: v for k, v in cfg["model"].items() if k != "batch_size"}),
        "partition_seed": cfg["partition_seed"], "dev_fraction": cfg["dev_fraction"],
        "train_annotation_sha256": file_sha256(train_annotation),
        "connectivity_sha256": object_sha256(all_graph_hashes),
        "source_and_integration_sha256": object_sha256([lock, implementation_hashes]),
    }
    protocol = dict(provenance, source_lock=lock, annotation_sha256=file_sha256(annotation),
                    connectivity=graph_hashes, split=cli.split, seed=cli.seed,
                    instr_ids=sorted(row["instr_id"] for row in data),
                    batch_size=args.batch_size, torch_version=str(torch.__version__),
                    numpy_version=np.__version__, transformers_version=transformers.__version__)
    metadata = dict(provenance, split=cli.split, protocol_sha256=object_sha256(protocol),
                    subset=cli.limit is not None, model=cfg["model"], seed=cli.seed,
                    mode=cli.mode, num_episodes=len(data), gpu=torch.cuda.get_device_name(0))

    def seed_all():
        random.seed(cli.seed)
        np.random.seed(cli.seed)
        torch.manual_seed(cli.seed)
        torch.cuda.manual_seed_all(cli.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    feature_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    def make_env():
        seed_all()
        return R2RNavBatch(feature_db, copy.deepcopy(data), args.connectivity_dir,
                           batch_size=args.batch_size, angle_feat_size=args.angle_feat_size,
                           seed=cli.seed, name=cli.split)

    env = make_env()
    agent = GMapNavAgent(args, env, rank=0)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["vln_bert"]["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    # Unlike upstream's permissive loader, refuse missing randomly initialized weights.
    agent.vln_bert.load_state_dict(state, strict=True)
    del payload, state
    for model in agent.models:
        model.requires_grad_(False)
        model.eval()

    head = None
    if cli.head:
        head, head_meta = load_head_checkpoint(cli.head, device="cuda", expected_provenance=provenance)
        head.eval()
        metadata["head_sha256"] = file_sha256(cli.head)
        metadata["head_config"] = head_meta["head_config"]
    collection = {
        "collector": "head" if cli.head else "baseline",
        "collector_head_sha256": file_sha256(cli.head) if cli.head else None,
        "protocol_sha256": metadata["protocol_sha256"],
        "seed": cli.seed, "limit": cli.limit, "scan_ids": scan_ids,
        "instruction_set_sha256": object_sha256(sorted(row["instr_id"] for row in data)),
        "num_episodes": len(data),
    }
    writer = CacheWriter(cli.cache, provenance, collection=collection) if cli.mode == "collect" else None
    hook = DecisionHook(agent, head=head, writer=writer) if (head is not None or writer is not None) else None

    def evaluate(decision_hook):
        agent.env = make_env()
        agent.decision_hook = decision_hook
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            agent.test(use_dropout=False, feedback="argmax")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        predictions = sorted(agent.get_results(), key=lambda r: r["instr_id"])
        expected = {r["instr_id"] for r in data}
        if {r["instr_id"] for r in predictions} != expected:
            raise ValueError("Evaluation did not cover the selected instructions exactly")
        return predictions, {"rollout_seconds": elapsed,
                             "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    predictions, resources = evaluate(hook)
    if cli.mode == "identity":
        zero_head = ResidualActionHead(feature_dim=2 * 768 + 6 + args.angle_feat_size + 3).cuda().eval()
        zero_predictions, zero_resources = evaluate(DecisionHook(agent, head=zero_head))
        if predictions != zero_predictions:
            raise AssertionError("Zero residual changes trajectories; stop experiments and fix integration")
        metadata["zero_residual_trajectory_identity"] = True
        metadata["zero_head_resources"] = zero_resources
    if writer is not None:
        metadata["cache_manifest"] = writer.close()
    averages, _ = agent.env.eval_metrics(predictions)
    episodes = []
    for prediction in predictions:
        instr_id = prediction["instr_id"]
        scan, target_path = agent.env.gt_trajs[instr_id]
        scores = agent.env._eval_item(scan, prediction["trajectory"], target_path)
        episodes.append(dict(instr_id=instr_id, scan_id=scan,
                             **{key: float(value) for key, value in scores.items()}))
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(metadata=metadata, resources=resources, summary=averages,
                  episodes=episodes, trajectories=predictions)
    cli.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(cli.output), "summary": averages, "resources": resources}, indent=2))


if __name__ == "__main__":
    main()
