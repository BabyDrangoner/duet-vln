#!/usr/bin/env python3
"""Summarize immutable E1 training metadata, without reading val_unseen."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "outputs/study-20261003/recovered-e1-seed0"
    result = {"scope": "Training-only immutable metadata; no validation parsing or checkpoint selection.", "arms": {}}
    for arm in ("C1", "C2", "C3", "M"):
        path = directory / f"e1-seed0-{arm}/training-summary.json"
        record = json.loads(path.read_text())
        summary = {k: record[k] for k in ("best_epoch", "best_common_bce", "completed_epochs", "primary_rule", "train_data_sha256", "dev_data_sha256")}
        summary["source"] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        summary["learning_curve"] = [{k: v for k, v in row.items() if k in (
            "epoch", "natural_bce", "c2_bce", "paired_bce", "ranking", "common_monitor_bce", "both_orders_rate")}
            for row in record["dev_history"]]
        summary["train_epoch_first_last"] = [{k: v for k, v in row.items() if k in ("epoch", "loss", "bce", "ranking")}
                                             for row in (record["training_history"][0], record["training_history"][-1])]
        summary["support"] = {}
        for partition in ("train_support", "dev_support"):
            support = record["budget"][partition]
            item = {k: support[k] for k in ("groups", "original_instructions", "original_paths", "scenes")}
            item["panels"] = {k: {**{field: value for field, value in panel.items() if field != "per_scene"},
                                  "positive_fraction": panel["positives"] / panel["states"],
                                  "mean_states_per_rollout": panel["states"] / panel["rollouts"]}
                              for k, panel in support["panels"].items()}
            summary["support"][partition] = item
        result["arms"][arm] = summary
    path = ROOT / "outputs/study-20261006-e2/training-metadata-diagnosis.json"
    text = json.dumps(result, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("existing training summary differs; do not overwrite")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
