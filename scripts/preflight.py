#!/usr/bin/env python3
"""Report readiness without downloading data or starting a paid GPU job."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vln_improve.protocol import resolve_config
from prepare_duet import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2r.json")
    args = parser.parse_args()
    cfg = resolve_config(args.config, ROOT)
    root = Path(cfg["dataset_root"])
    assets = {
        "base_checkpoint": Path(cfg["base_checkpoint"]),
        "train_annotations": root / "R2R/annotations/R2R_train_enc.json",
        "val_unseen_annotations": root / "R2R/annotations/R2R_val_unseen_enc.json",
        "features": root / "R2R/features/pth_vit_base_patch16_224_imagenet.hdf5",
        "angle_feature_connectivity": root / "R2R/connectivity/ZMojNkEp431_connectivity.json",
    }
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "duet_python_recommended": "3.11", "assets": {
                  key: {"path": str(value), "exists": value.is_file()} for key, value in assets.items()},
              "modules": {key: importlib.util.find_spec(key) is not None for key in
                          ("torch", "numpy", "transformers", "h5py", "networkx", "jsonlines", "line_profiler", "MatterSim")}}
    try:
        report["upstream_commit"] = verify()["commit"]
        report["upstream_verified"] = True
    except (ValueError, FileNotFoundError) as error:
        report["upstream_verified"] = False
        report["source_error"] = str(error)
    report["cuda_available"] = False
    if report["modules"]["torch"]:
        import torch
        report["torch_version"] = str(torch.__version__)
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory}
    report["ready_for_rollout"] = bool(report["upstream_verified"] and report["cuda_available"]
                                      and all(report["modules"].values())
                                      and all(x["exists"] for x in report["assets"].values()))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_rollout"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
