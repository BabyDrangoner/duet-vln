"""Capture labels only in training; leave the policy feature interface free of ground truth."""
import json
from pathlib import Path
import torch
from .features import build_features, FEATURE_SCHEMA


class CacheWriter:
    def __init__(self, directory, provenance, shard_size=128, collection=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError(f"Cache directory must be empty: {self.directory}")
        if not 1 <= shard_size <= 128:
            raise ValueError("shard_size must be in [1, 128]")
        self.provenance = provenance
        self.collection = dict(collection or {})
        self.shard_size = shard_size
        self.records, self.shards, self.seen = [], [], set()
        self.count = self.hard_count = 0
        self.feature_dim = None

    def add(self, record, step):
        key = (record["instr_id"], step)
        if key in self.seen:
            return
        self.seen.add(key)
        dim = record["features"].shape[-1]
        if self.feature_dim not in (None, dim):
            raise ValueError("Feature schema changed within collection")
        self.feature_dim = dim
        self.records.append(record)
        self.count += 1
        self.hard_count += int(record["hard"])
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self):
        if self.records:
            name = f"shard-{len(self.shards):05d}.pt"
            torch.save(self.records, self.directory / name)
            self.shards.append(name)
            self.records = []

    def close(self):
        self.flush()
        if self.count == 0:
            raise ValueError("No usable training decisions collected")
        manifest = dict(schema_version=1, feature_dim=self.feature_dim,
                        feature_schema=FEATURE_SCHEMA, split="train_fit",
                        provenance=self.provenance, shards=self.shards,
                        num_records=self.count, num_hard=self.hard_count,
                        collection=self.collection)
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


class DecisionHook:
    def __init__(self, agent, head=None, writer=None):
        self.agent, self.head, self.writer = agent, head, writer

    def __call__(self, nav_inputs, nav_outs, obs, ended, step, traj):
        features, base_logits, valid = build_features(nav_inputs, nav_outs)
        if self.writer is not None:
            # This call uses training targets. It is never reached in evaluation.
            targets = self.agent._teacher_action_r4r(
                obs, nav_inputs["gmap_vpids"], ended,
                visited_masks=nav_inputs["gmap_visited_masks"],
                imitation_learning=False, t=step, traj=traj)
            for i, ob in enumerate(obs):
                target = int(targets[i])
                if (ended[i] or nav_inputs["no_vp_left"][i]
                        or step == self.agent.args.max_action_len - 1
                        or target < 0 or target >= valid.shape[1] or not valid[i, target]):
                    continue
                length = len(nav_inputs["gmap_vpids"][i])
                self.writer.add({
                    "features": features[i, :length].detach().cpu().half(),
                    "base_logits": base_logits[i, :length].detach().cpu().float(),
                    "valid_mask": valid[i, :length].detach().cpu().bool(),
                    "target": target,
                    "hard": int(base_logits[i].argmax()) != target,
                    "instr_id": str(ob["instr_id"]), "scan_id": str(ob["scan"]),
                }, step)
        if self.head is None:
            return nav_outs
        corrected = self.head(features, base_logits, valid)
        return dict(nav_outs, fused_logits=corrected)
