#!/usr/bin/env python3
"""T4 engineering smoke for E2 on eight train_fit instructions only.

Run an unchanged baseline twice, collect terminal representations without
changing any action, verify every alternative's metric target, and exercise
small-head training plus restoration from an empty local checkpoint directory.
This is not a method evaluation or evidence of navigation improvement.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def equal_tree(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("use a new smoke directory; do not overwrite previous attempts")
    from vln_improve.pipeline import validate_backup_root, atomic_json
    from vln_improve.protocol import file_sha256
    validate_backup_root(args.backup)
    args.output_dir.mkdir(parents=True)
    args.backup.mkdir(parents=True, exist_ok=True)
    import torch
    from vln_improve.endpoint_intervention import (
        InterventionHead, InterventionInputs, build_intervention_inputs,
        build_intervention_targets, collate_interventions, intervention_loss, select_intervention,
    )
    from vln_improve.checkpoint_store import CheckpointStore
    import run_duet
    run_duet.verify()
    sys.path.insert(0, str(run_duet.DEFAULT_DEST / "map_nav_src"))
    import r2r.agent as upstream

    count = 8
    started = time.monotonic()
    original_class, original_parse = upstream.GMapNavAgent, run_duet.parse_cli
    baseline_path = args.output_dir / "baseline-train-fit-8.json"
    observer_path = args.output_dir / "observer-train-fit-8.json"
    captured = {}
    examples = {}

    def run(path):
        run_duet.parse_cli = lambda: SimpleNamespace(
            mode="baseline", config=args.config, split="train_fit", output=path,
            cache=None, head=None, limit=count, seed=0)
        run_duet.main()

    class ObserverAgent(original_class):
        def rollout(self, *positional, **kwargs):
            if positional or self.decision_hook is not None:
                raise ValueError("unexpected smoke rollout hook")
            old_move = self.make_equiv_action
            visits, terminal, built = {}, {}, {}

            def observe(nav_inputs, nav_outs, obs, ended, step, trajectories):
                if len(obs) != 1 or bool(ended[0]):
                    raise ValueError("smoke requires one active instruction")
                vp = str(obs[0]["viewpoint"])
                if vp in visits:
                    raise ValueError("baseline revisited a decision node")
                visits[vp] = int(step)
                terminal.update(inputs=nav_inputs, outputs=nav_outs)
                return nav_outs  # same objects and original logits

            def move(actions, gmaps, observations, trajectories):
                result = old_move(actions, gmaps, observations, trajectories)
                if actions[0] is not None:
                    return result
                graph, ob = gmaps[0], observations[0]
                instr, scan, current = str(ob["instr_id"]), str(ob["scan"]), str(ob["viewpoint"])
                nodes = list(graph.node_stop_scores)
                if set(nodes) != set(visits):
                    raise ValueError("observed candidate inventory differs from original STOP inventory")
                stop = {v: float(graph.node_stop_scores[v]["stop"]) for v in nodes}
                anchor = max(nodes, key=stop.__getitem__)
                prefix = copy.deepcopy(trajectories[0]["path"])
                flat = [v for segment in prefix for v in segment]
                # Policy geometry uses only observed positions and discovered routes.
                prefix_observed = math.fsum(math.dist(graph.node_positions[a], graph.node_positions[b])
                                            for a, b in zip(flat, flat[1:]))
                return_known = {v: float(graph.graph.distance(current, v)) if v != current else 0.0
                                for v in nodes}
                inputs = build_intervention_inputs(
                    terminal["inputs"], terminal["outputs"], observed_vpids=nodes,
                    visit_steps=visits, stop_probabilities=stop, baseline_vpid=anchor,
                    termination_vpid=current, prefix_length_m=prefix_observed,
                    return_distances_m=return_known)
                # Ground truth is introduced only after policy inputs are complete.
                metrics, paths = [], []
                for node in nodes:
                    path = copy.deepcopy(prefix)
                    if node != current:
                        path.append(graph.graph.path(current, node))
                    paths.append(path)
                    metrics.append(self.env._eval_item(scan, path, ob["gt_path"]))
                distances = self.env.shortest_distances[scan]
                reference_length = sum(distances[a][b] for a, b in zip(ob["gt_path"], ob["gt_path"][1:]))
                target = build_intervention_targets(
                    candidate_goal_distances_m=torch.tensor([m["nav_error"] for m in metrics], dtype=torch.float64),
                    candidate_total_lengths_m=torch.tensor([m["trajectory_lengths"] for m in metrics], dtype=torch.float64),
                    reference_path_length_m=reference_length, baseline_index=inputs.baseline_index)
                expected = torch.tensor([[m["success"], m["spl"]] for m in metrics], dtype=torch.float64)
                expected -= expected[inputs.baseline_index].clone()
                if not torch.allclose(target, expected, rtol=0, atol=2e-12):
                    raise ValueError("alternative targets differ from upstream full-path evaluation")
                item = {
                    "instr_id": instr, "scan_id": scan, "candidate_vpids": nodes,
                    "baseline_endpoint": anchor, "termination_endpoint": current,
                    "baseline_trajectory": paths[inputs.baseline_index],
                    "candidates": len(nodes), "target_max_abs_error": float((target - expected).abs().max()),
                    "positive_sr_gain_candidates": int((target[:, 0] > 0).sum()),
                    "positive_spl_gain_candidates": int((target[:, 1] > 0).sum()),
                    "negative_spl_gain_candidates": int((target[:, 1] < 0).sum()),
                }
                if instr in captured and captured[instr] != item:
                    raise ValueError("wraparound smoke instruction changed")
                captured[instr] = item
                examples[instr] = (InterventionInputs(inputs.candidate_vpids, inputs.baseline_index,
                    inputs.node_features.cpu(), inputs.terminal_context.cpu(), inputs.scalar_features.cpu()), target)
                built["instr"] = instr
                return result

            self.decision_hook, self.make_equiv_action = observe, move
            try:
                result = super().rollout(**kwargs)
            finally:
                self.decision_hook, self.make_equiv_action = None, old_move
            if len(result) != 1 or result[0]["instr_id"] != built.get("instr"):
                raise ValueError("smoke instruction did not complete")
            return result

    try:
        run(baseline_path)
        upstream.GMapNavAgent = ObserverAgent
        run(observer_path)
    finally:
        upstream.GMapNavAgent, run_duet.parse_cli = original_class, original_parse
    baseline, observed = [json.loads(p.read_text()) for p in (baseline_path, observer_path)]
    if (baseline["episodes"] != observed["episodes"] or baseline["trajectories"] != observed["trajectories"]
            or len(captured) != count):
        raise ValueError("observer changed baseline trajectories/metrics or instruction count")
    for row in baseline["trajectories"]:
        if row["trajectory"] != captured[row["instr_id"]]["baseline_trajectory"]:
            raise ValueError("KEEP differs from DUET's actual historical return")

    ordered = [examples[k] for k in sorted(examples)]
    batch = collate_interventions([p[0] for p in ordered], device="cuda")
    targets = torch.zeros(*batch.valid_mask.shape, 2, dtype=torch.float64, device="cuda")
    for i, (inputs, target) in enumerate(ordered):
        targets[i, :len(inputs.candidate_vpids)] = target.to("cuda")
    torch.manual_seed(0)
    head = InterventionHead().cuda()
    with torch.no_grad():
        predictions = head(batch)
        for i, (inputs, _) in enumerate(ordered):
            anchor = inputs.candidate_vpids[inputs.baseline_index]
            assert select_intervention(predictions[i], inputs.candidate_vpids, anchor,
                                       valid_mask=batch.valid_mask[i]) == anchor
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    store = CheckpointStore(args.output_dir / "checkpoint-continuous", args.backup / "checkpoint", keep_local=2)

    def step(model, optim):
        model.train(); optim.zero_grad(set_to_none=True)
        loss = intervention_loss(model(batch), targets, batch.valid_mask, batch.baseline_indices)["loss"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite engineering training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite/missing engineering gradient")
        optim.step()
        return float(loss.detach())

    losses = []
    for step_id in range(1, 21):
        losses.append(step(head, optimizer))
        if step_id == 10:
            state = {"model": head.state_dict(), "optimizer": optimizer.state_dict(), "step": 10,
                     "losses": losses[:], "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
            store.save(state, {"schema": "e2_engineering_smoke_only", "state_dict": head.state_dict()}, step=10)
    restored_store = CheckpointStore(args.output_dir / "checkpoint-restored-empty", args.backup / "checkpoint", keep_local=2)
    state, _, _ = restored_store.restore("latest")
    restored_head = InterventionHead().cuda(); restored_head.load_state_dict(state["model"], strict=True)
    restored_optimizer = torch.optim.AdamW(restored_head.parameters(), lr=1e-4, weight_decay=0.01)
    restored_optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    restored_losses = state["losses"][:]
    for _ in range(10):
        restored_losses.append(step(restored_head, restored_optimizer))
    if not (losses == restored_losses and equal_tree(head.state_dict(), restored_head.state_dict())
            and equal_tree(optimizer.state_dict(), restored_optimizer.state_dict())):
        raise ValueError("empty-local-directory restore differs from continuous training")
    head.eval(); restored_head.eval()
    with torch.no_grad():
        if not torch.equal(head(batch), restored_head(batch)):
            raise ValueError("reloaded head inference differs")
    store.save({"model": head.state_dict(), "optimizer": optimizer.state_dict(), "step": 20,
                "losses": losses, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
               {"schema": "e2_engineering_smoke_only", "state_dict": head.state_dict()}, step=20)
    report = {
        "schema": "e2_endpoint_intervention_real_smoke_v1", "status": "passed",
        "scope": "8 train_fit engineering instructions, 20 fixed-batch gradient steps; not method training or navigation improvement evidence",
        "split": "train_fit", "new_val_unseen_accesses": 0, "gpu": torch.cuda.get_device_name(0),
        "episodes": count, "candidates": sum(x["candidates"] for x in captured.values()),
        "original_trajectory_and_all_metric_parity": True, "all_candidate_targets_match_upstream": True,
        "zero_initialized_head_keeps_original_endpoint": True, "parameters": sum(p.numel() for p in head.parameters()),
        "continuous_and_empty_local_restore_identical": True, "losses": losses,
        "elapsed_seconds": time.monotonic() - started, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "instructions": [captured[k] for k in sorted(captured)],
        "source_sha256": {p: file_sha256(ROOT / p) for p in ["scripts/smoke_endpoint_intervention.py", "src/vln_improve/endpoint_intervention.py"]},
        "baseline_report_sha256": file_sha256(baseline_path), "observer_report_sha256": file_sha256(observer_path),
        "checkpoint_backup": str(args.backup / "checkpoint"),
    }
    atomic_json(args.output_dir / "smoke.json", report)
    for path in [baseline_path, observer_path, args.output_dir / "smoke.json"]:
        destination = args.backup / path.name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(path.read_bytes()); stream.flush(); os.fsync(stream.fileno())
        if file_sha256(temporary) != file_sha256(path):
            raise ValueError("smoke report Drive staging verification failed")
        os.replace(temporary, destination)
        if file_sha256(destination) != file_sha256(path):
            raise ValueError("smoke report Drive verification failed")
    print(json.dumps({k: report[k] for k in ["status", "scope", "episodes", "candidates", "parameters",
                     "continuous_and_empty_local_restore_identical", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
