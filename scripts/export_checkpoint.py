"""Export a verified inference head from a protected training run."""
import argparse
import json
from pathlib import Path
import shutil

from vln_improve.checkpoint_store import CheckpointStore
from vln_improve.pipeline import ROOT, digest, validate_backup_root, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("best", "latest"), default="best")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    local = ROOT / config["local_root"] / config["run_id"]
    backup = ROOT / config["backup_root"] / config["run_id"]
    validate_backup_root(backup)
    store = CheckpointStore(local, backup, keep_local=config["keep_local"], keep_backup=config["keep_backup"])
    _, _, manifest = store.restore(args.kind)
    source = Path(manifest["local_path"]) / "head.pt"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if digest(source) != digest(output):
            raise FileExistsError(f"refusing to overwrite a different checkpoint: {output}")
    else:
        with source.open("rb") as reader, output.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    if digest(source) != digest(output):
        raise RuntimeError("exported checkpoint checksum mismatch")
    print(json.dumps({"output": str(output), "kind": args.kind,
                      "checkpoint_id": manifest["checkpoint_id"], "metrics": manifest["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
