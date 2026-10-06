#!/usr/bin/env python3
"""Compare four fixed-final, same-seed endpoint heads on adaptation development.

This reads only CPU reports and small final heads. It never loads feature caches,
fits parameters, chooses checkpoints/seeds, or evaluates navigation episodes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vln_improve.protocol import file_sha256, object_sha256

ARMS = ("C1", "C2", "C3", "M")
COMPARISONS = (("C1", "C2"), ("C2", "C3"), ("C3", "M"))
LOSSES = ("natural_bce", "c2_bce", "paired_bce", "ranking")
METRICS = LOSSES + ("mean_order_both_instructions_correct_rate", "both_orders_correct_rate")
FINAL = "fixed_final_epoch"
MONITOR = "equal_mean_of_natural_c2_paired_dev_episode_bce"
RESAMPLES = 10000
BOOTSTRAP_SEED = 0
GROUPS = 128


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def _means(rows):
    return {"groups": len(rows), **{k: sum(r[k] for r in rows) / len(rows) for k in LOSSES},
            "correct_order_rates": [sum(r["both_instructions_correct_by_order"][i] for r in rows) / len(rows)
                                    for i in range(2)],
            "both_orders_rate": sum(r["both_orders_correct"] for r in rows) / len(rows)}


def validate_report(report, arm):
    """Authenticate internal structure and recompute every saved aggregation."""
    keys = {"schema", "split", "usage", "arm", "training_seed", "train_data_sha256", "dev_data_sha256",
            "checkpoint", "selection_purpose", "statistical_unit", "monitor", "groups", "content_sha256"}
    if not isinstance(report, dict) or set(report) != keys or arm not in ARMS:
        raise ValueError("unexpected final development report schema")
    if (not _sha(report["content_sha256"])
            or object_sha256({k: v for k, v in report.items() if k != "content_sha256"}) != report["content_sha256"]):
        raise ValueError("final development report content SHA mismatch")
    if (report["schema"] != "duet_endpoint_group_final_dev_v1" or report["split"] != "train_dev"
            or report["usage"] != "analysis_only" or report["arm"] != arm
            or type(report["training_seed"]) is not int or report["training_seed"] not in (0, 1, 2)
            or report["selection_purpose"] != FINAL
            or report["statistical_unit"] != "original paired group; four paired rollouts are not independent samples"
            or not all(_sha(report[k]) for k in ("train_data_sha256", "dev_data_sha256"))
            or report["train_data_sha256"] == report["dev_data_sha256"]):
        raise ValueError("final development protocol/arm/seed/data identity mismatch")
    checkpoint = report["checkpoint"]
    if (not isinstance(checkpoint, dict)
            or set(checkpoint) != {"checkpoint_id", "epoch", "global_step", "head_sha256", "head_relative_path", "selection_purpose"}
            or type(checkpoint["epoch"]) is not int or checkpoint["epoch"] != 20
            or type(checkpoint["global_step"]) is not int or checkpoint["global_step"] != 1280
            or checkpoint["selection_purpose"] != FINAL or not _sha(checkpoint["head_sha256"])
            or not isinstance(checkpoint["checkpoint_id"], str)
            or re.fullmatch(r"step-000000001280-[0-9a-f]{32}", checkpoint["checkpoint_id"]) is None
            or checkpoint["head_relative_path"] != f"snapshots/{checkpoint['checkpoint_id']}/head.pt"):
        raise ValueError("report is not bound to a fixed-final 20/1280 checkpoint")
    rows = report["groups"]
    if not isinstance(rows, list) or len(rows) != GROUPS:
        raise ValueError("analysis requires the complete 128 original development pairs")
    row_keys = {"pair_id", "scan_id", *LOSSES, "both_instructions_correct_by_order", "both_orders_correct"}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != row_keys or not _sha(row["pair_id"])
                or not isinstance(row["scan_id"], str) or not row["scan_id"]
                or any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) or row[k] < 0 for k in LOSSES)
                or not isinstance(row["both_instructions_correct_by_order"], list)
                or len(row["both_instructions_correct_by_order"]) != 2
                or any(type(v) is not bool for v in row["both_instructions_correct_by_order"])
                or type(row["both_orders_correct"]) is not bool
                or row["both_orders_correct"] != all(row["both_instructions_correct_by_order"])):
            raise ValueError("invalid original-pair metrics or correctness labels")
    if len({r["pair_id"] for r in rows}) != GROUPS:
        raise ValueError("duplicate original development pair")
    scenes = sorted({r["scan_id"] for r in rows})
    overall = _means(rows)
    expected_monitor = {"epoch": 20, "global_step": 1280, **overall,
        "common_monitor_bce": sum(overall[k] for k in LOSSES[:3]) / 3,
        "selection_purpose": MONITOR,
        "by_scene": {s: _means([r for r in rows if r["scan_id"] == s]) for s in scenes}}
    if report["monitor"] != expected_monitor:
        raise ValueError("saved monitor aggregation differs from original-pair rows")
    return report


def load_report(path, arm, head_path=None):
    """Verify actual head bytes and bind their CPU-loaded metadata to the report."""
    path = Path(path)
    report = validate_report(_json(path), arm)
    head_path = Path(head_path) if head_path is not None else path.parent / report["checkpoint"]["head_relative_path"]
    if not head_path.is_file() or file_sha256(head_path) != report["checkpoint"]["head_sha256"]:
        raise ValueError(f"{arm}: missing final head or actual head file SHA mismatch")
    from vln_improve.endpoint_probe import load_endpoint_head
    _, metadata = load_endpoint_head(head_path, device="cpu")
    expected = {"arm": arm, "seed": report["training_seed"], "epochs": 20, "batch_groups": 8,
        "monitor_every": 5, "optimizer": "AdamW", "lr": .001, "weight_decay": .0001,
        "pair_weight": .1 if arm == "M" else 0., "natural_weight": .5, "augmentation_weight": .5,
        "max_states": 15, "feature_dim": 1536, "hidden_dim": 128, "activation": "relu",
        "primary_checkpoint": FINAL, "engineering_best_checkpoint": MONITOR}
    if (metadata.get("epoch") != 20 or metadata.get("global_step") != 1280
            or metadata.get("pending_dev") is not False or metadata.get("selection_purpose") != FINAL
            or not isinstance(metadata.get("train_config"), dict)
            or any(metadata["train_config"].get(k) != v for k, v in expected.items())
            or not _sha(metadata["train_config"].get("experiment_sha256"))
            or not isinstance(metadata.get("data_identity"), dict)
            or metadata["data_identity"].get("train") != report["train_data_sha256"]
            or metadata["data_identity"].get("dev") != report["dev_data_sha256"]
            or not _sha(metadata.get("initial_head_sha256"))):
        raise ValueError(f"{arm}: final head metadata and report/protocol disagree")
    code = metadata.get("training_code_identity")
    if not isinstance(code, dict) or not code or not all(isinstance(k, str) and _sha(v) for k, v in code.items()):
        raise ValueError(f"{arm}: missing training code identity")
    source = {"report_path": str(path.resolve()), "report_sha256": file_sha256(path),
              "report_content_sha256": report["content_sha256"], "head_path": str(head_path.resolve()),
              "head_sha256": report["checkpoint"]["head_sha256"], "head_bytes_verified": True,
              "experiment_sha256": metadata["train_config"]["experiment_sha256"]}
    # A co-located summary is an additional independent file association, when available.
    summary_path = path.parent / "training-summary.json"
    if summary_path.is_file():
        summary = _json(summary_path)
        descriptor = summary.get("final_dev_report", {})
        if (summary.get("status") != "complete" or summary.get("pending_dev") is not False
                or summary.get("arm") != arm or summary.get("seed") != report["training_seed"]
                or summary.get("final_checkpoint") != report["checkpoint"]
                or summary.get("train_data_sha256") != report["train_data_sha256"]
                or summary.get("dev_data_sha256") != report["dev_data_sha256"]
                or descriptor.get("sha256") != source["report_sha256"]
                or descriptor.get("content_sha256") != report["content_sha256"] or descriptor.get("groups") != GROUPS):
            raise ValueError(f"{arm}: training summary binding mismatch")
        source["training_summary_sha256"] = file_sha256(summary_path)
    else:
        source["training_summary_sha256"] = None
    return report, source, metadata


def _values(rows):
    return np.asarray([[*[r[k] for k in LOSSES], sum(r["both_instructions_correct_by_order"]) / 2,
                        float(r["both_orders_correct"])] for r in rows], dtype=np.float64)


def _transitions(before, after):
    def counts(left, right):
        return {"rescues": sum(not a and b for a, b in zip(left, right)),
                "harms": sum(a and not b for a, b in zip(left, right)),
                "unchanged_correct": sum(a and b for a, b in zip(left, right)),
                "unchanged_incorrect": sum(not a and not b for a, b in zip(left, right))}
    order = [counts([r["both_instructions_correct_by_order"][i] for r in before],
                    [r["both_instructions_correct_by_order"][i] for r in after]) for i in range(2)]
    return {"order_names": ["A_then_B", "B_then_A"], "by_order": order,
            "both_orders": counts([r["both_orders_correct"] for r in before], [r["both_orders_correct"] for r in after]),
            "order_events_total": 2 * len(before),
            "order_event_rescues": sum(x["rescues"] for x in order),
            "order_event_harms": sum(x["harms"] for x in order)}


def analyze_reports(reports, sources):
    """Paired, scene-cluster bootstrap; scene draws are shared by all contrasts."""
    if set(reports) != set(ARMS) or set(sources) != set(ARMS):
        raise ValueError("exactly C1/C2/C3/M are required")
    for arm in ARMS:
        validate_report(reports[arm], arm)
    reference = reports["C1"]
    order = [(r["pair_id"], r["scan_id"]) for r in reference["groups"]]
    for arm in ARMS[1:]:
        other = reports[arm]
        if (any(other[k] != reference[k] for k in ("training_seed", "train_data_sha256", "dev_data_sha256"))
                or [(r["pair_id"], r["scan_id"]) for r in other["groups"]] != order):
            raise ValueError("four arms must share seed, data SHA and complete ordered pair/scene inventory")
    scenes = sorted({scan for _, scan in order})
    if len(scenes) < 2:
        raise ValueError("scene-cluster inference requires at least two scenes")
    memberships = [np.asarray([i for i, (_, scan) in enumerate(order) if scan == s]) for s in scenes]
    sizes = np.asarray([len(ix) for ix in memberships])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(scenes), size=(RESAMPLES, len(scenes)))
    denominators = sizes[draws].sum(axis=1)
    values = {a: _values(reports[a]["groups"]) for a in ARMS}
    comparisons = {}
    for before, after in COMPARISONS:
        delta = values[after] - values[before]
        sums = np.stack([delta[ix].sum(axis=0) for ix in memberships])
        replicated = sums[draws].sum(axis=1) / denominators[:, None]
        intervals = np.quantile(replicated, [.025, .975], axis=0, method="linear")
        metrics = {}
        for i, name in enumerate(METRICS):
            metrics[name] = {"before_mean": float(values[before][:, i].mean()),
                "after_mean": float(values[after][:, i].mean()), "effect_after_minus_before": float(delta[:, i].mean()),
                "ci95": intervals[:, i].tolist(), "preferred_direction": "lower" if name in LOSSES else "higher",
                "unit": "loss" if name in LOSSES else "proportion"}
        per_scene = {}
        for scene, ix in zip(scenes, memberships):
            left = [reports[before]["groups"][int(i)] for i in ix]
            right = [reports[after]["groups"][int(i)] for i in ix]
            per_scene[scene] = {"pairs": len(ix), "pair_ids": [r["pair_id"] for r in left],
                "metrics": {m: {"before_mean": float(values[before][ix, j].mean()),
                    "after_mean": float(values[after][ix, j].mean()), "effect_after_minus_before": float(delta[ix, j].mean())}
                            for j, m in enumerate(METRICS)}, "transitions": _transitions(left, right)}
        comparisons[f"{after}-{before}"] = {"before": before, "after": after, "metrics": metrics,
            "transitions": _transitions(reports[before]["groups"], reports[after]["groups"]), "by_scene": per_scene,
            "per_pair": [{"pair_id": pair, "scan_id": scan,
                          "effects": dict(zip(METRICS, delta[i].tolist()))} for i, (pair, scan) in enumerate(order)]}
    result = {"schema": "duet_endpoint_group_comparison_v1", "split": "train_dev", "usage": "analysis_only",
        "training_seed": reference["training_seed"], "epoch": 20, "global_step": 1280,
        "selection_purpose": FINAL, "pairs": GROUPS, "scenes": len(scenes),
        "train_data_sha256": reference["train_data_sha256"], "dev_data_sha256": reference["dev_data_sha256"],
        "source_reports": sources, "ordered_groups": [{"pair_id": p, "scan_id": s} for p, s in order],
        "bootstrap": {"resamples": RESAMPLES, "seed": BOOTSTRAP_SEED, "rng": "numpy.default_rng.PCG64",
            "unit": "scene", "sampled_scenes_per_resample": len(scenes), "with_replacement": True,
            "shared_draws_across_arms_and_contrasts": True, "within_scene": "retain all original paired groups and four arms",
            "estimand": "pair-weighted after-minus-before mean; resampled pair count is the denominator",
            "interval": "unadjusted percentile 95%, numpy linear quantiles", "draws_sha256": object_sha256(draws.tolist())},
        "comparisons": comparisons,
        "interpretation": ["Training-house adaptation development mechanism diagnostics; these are not navigation SR/SPL results.",
            "The frozen backbone has seen training scenes. This split isolates adapter fitting, not base-model unseen scenes.",
            "Rescues/harms denote pair/order ranking correctness changes, not rescued or harmed navigation episodes.",
            "Intervals describe scene-cluster resampling conditional on this training seed and these frozen heads; they omit retraining-seed uncertainty.",
            "The three contrasts and six metrics are all reported; intervals are not multiplicity adjusted, and no winner or success threshold is selected.",
            "Report hashes and CPU head metadata bind the declared data identities; this analysis does not reload feature caches."]}
    result["content_sha256"] = object_sha256(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("c1", "c2", "c3", "method"):
        parser.add_argument("--" + flag, type=Path, required=True, help="Fixed-final dev-final.json")
        parser.add_argument("--" + flag + "-head", type=Path, help="Final head.pt; defaults to report-relative checkpoint path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refusing to overwrite a comparison report")
    reports, sources, metas = {}, {}, {}
    for arm, flag in zip(ARMS, ("c1", "c2", "c3", "method")):
        reports[arm], sources[arm], metas[arm] = load_report(getattr(args, flag), arm, getattr(args, flag + "_head"))
    for arm in ARMS[1:]:
        for key in ("data_identity", "common_identity", "training_code_identity", "initial_head_sha256"):
            if metas[arm].get(key) != metas["C1"].get(key):
                raise ValueError(f"{arm}: shared source/head initialization identity differs from C1")
    result = analyze_reports(reports, sources)
    result["analysis_code_sha256"] = file_sha256(Path(__file__))
    result["numpy_version"] = np.__version__
    result["content_sha256"] = object_sha256({k: v for k, v in result.items() if k != "content_sha256"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=".endpoint-analysis-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, args.output)  # Atomic publication that refuses a concurrent overwrite.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "sha256": file_sha256(args.output), "training_seed": result["training_seed"],
                      "pairs": GROUPS, "scenes": result["scenes"], "scope": "adaptation development mechanism diagnostics"}))
    return result


if __name__ == "__main__":
    main()
