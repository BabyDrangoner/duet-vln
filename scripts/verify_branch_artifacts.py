#!/usr/bin/env python3
"""Verify fixed D1 branch artifacts before interpreting numeric probe results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from replay_branches import CONTROLS, read_branch_result
from replay_diagnostics import chosen_candidates, validate_runtime
from vln_improve.diagnostics import load_episode
from vln_improve.protocol import file_sha256, object_sha256
from vln_improve.pipeline import atomic_json, validate_backup_root


def verify(collection_dir, replay_dir, cloud_dir, split, protocol):
    collection = json.loads((collection_dir / "COLLECTION.json").read_text())
    identity = collection["identity"]
    expected_collection = protocol["collections"]["train" if split == "train_fit" else "dev"]
    assert object_sha256(identity) == collection["identity_sha256"] == expected_collection
    replay = json.loads((replay_dir / "REPLAY.json").read_text())
    replay_sha = object_sha256(replay)
    assert replay["collection_sha256"] == expected_collection
    assert replay["seed"] == 0 and replay["controls"] == CONTROLS == protocol["controls"]
    assert replay["schema"] == "duet_simulated_branch_replay_v1"
    assert replay["analysis_cohort"] == "simulated_branch_all_preselected_candidates"
    validate_runtime(identity, replay["runtime"])
    for name, digest in replay["implementation"].items():
        assert file_sha256(ROOT / name) == digest, name
    expected_names = sorted("episode-" + object_sha256([row["scan"], row["instr_id"]])
                            for row in identity["selection"])
    assert len(set(expected_names)) == len(expected_names)
    assert sorted(p.stem for p in replay_dir.glob("episode-*.json")) == expected_names
    rows, counters, episode_hashes = [], [], {}
    for name in expected_names:
        inputs, labels, manifest = load_episode(collection_dir / name,
                                               expected_identity_sha256=expected_collection)
        result = read_branch_result(replay_dir / (name + ".json"),
            input_manifest_sha256=file_sha256(collection_dir / name / "manifest.json"),
            replay_identity_sha256=replay_sha, association=inputs["association"], split=split)
        selections = [{"step": state["step"], "target_ids": [state["nav_inputs"]["gmap_vpids"][0][i]
                        for i in chosen_candidates(state, 0)]}
                      for state in inputs["states"] if state["eligible_decision"]]
        assert result["selections"] == selections
        expected_keys = [(entry["step"], target) for entry in selections for target in entry["target_ids"]]
        assert [(row["step"], row["target_id"]) for row in result["rows"]] == expected_keys
        for row in result["rows"]:
            assert row["arrival_kind"] == "simulated_branch" and "arrival_step" not in row
            expected_controls = CONTROLS if row["shuffled_donor"] is not None else CONTROLS[:6]
            assert set(row["interventions"]) == set(expected_controls)
        rows.extend(result["rows"])
        counters.append(result["counters"])
        episode_hashes[name] = file_sha256(replay_dir / (name + ".json"))
    written_rows = [json.loads(line) for line in (replay_dir / "rows.jsonl").read_text().splitlines()]
    assert rows == written_rows
    keys = [(r["scan_id"], r["instr_id"], r["step"], r["target_id"]) for r in rows]
    assert len(set(keys)) == len(keys) == protocol["expected_candidates"][split]
    summary = json.loads((replay_dir / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["split"] == split
    assert summary["replay_identity_sha256"] == replay_sha
    assert summary["rows"] == len(rows) and summary["episodes"] == len(expected_names)
    assert summary["scans"] == len({r["scan_id"] for r in rows})
    assert summary["counters"] == {
        key: (max(c[key] for c in counters) if key.startswith("max_") else sum(c[key] for c in counters))
        for key in counters[0]}
    expected_counts = ({"eligible_states": 1139, "historical_panorama_parity_checks": 1139,
                        "natural_chosen_parity_checks": 946, "selected_without_natural_arrival": 1113}
                       if split == "train_fit" else
                       {"eligible_states": 598, "historical_panorama_parity_checks": 599,
                        "natural_chosen_parity_checks": 503, "selected_without_natural_arrival": 576})
    for key, value in expected_counts.items():
        assert summary["counters"][key] == value, key
    hashes = {name: file_sha256(replay_dir / name) for name in ("REPLAY.json", "rows.jsonl", "summary.json")}
    validate_backup_root(cloud_dir)
    for name, digest in hashes.items():
        assert file_sha256(cloud_dir / name) == digest, name
    donors = [r["shuffled_donor"] for r in rows if r["shuffled_donor"] is not None]
    def describe(key):
        values = np.asarray([d[key] for d in donors], dtype=float)
        return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
                "p95": float(np.quantile(values, .95)), "max": float(values.max()),
                "exact_match_fraction": float((values == 0).mean())} if len(values) else {"count": 0}
    return {"schema": "duet_branch_artifact_audit_v1", "status": "verified", "split": split,
            "protocol_sha256": object_sha256(protocol), "replay_identity_sha256": replay_sha,
            "collection_identity_sha256": expected_collection, "rows": len(rows),
            "episodes": len(expected_names), "artifacts_sha256": hashes,
            "episode_artifact_sha256": episode_hashes,
            "selection_and_complete_row_concatenation_verified": True,
            "source_implementation_matches_replay": True,
            "cloud_combined_artifacts_readback_verified": True,
            "cloud_verification_scope": "Drive mount combined-artifact SHA readback; no independent service receipt",
            "counters": summary["counters"], "donor_missing": len(rows) - len(donors),
            "donor_matching": {key: describe(key) for key in ("distance_difference", "source_count_difference")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--split", choices=("train_fit", "train_dev"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/branch_extension.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.collection, args.replay, args.cloud, args.split, json.loads(args.protocol.read_text()))
    result["verification_script_sha256"] = file_sha256(Path(__file__))
    atomic_json(args.output, result)
    cloud_path = args.cloud / args.output.name
    atomic_json(cloud_path, result)
    assert file_sha256(args.output) == file_sha256(cloud_path)
    print(json.dumps({k: v for k, v in result.items() if k != "episode_artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
