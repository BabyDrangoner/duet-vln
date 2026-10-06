#!/usr/bin/env python3
"""Fetch pinned upstream source and apply two explicit compatibility/instrumentation edits."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/upstream.json"
DEFAULT_DEST = ROOT / "third_party/VLN-DUET"
ANCHOR = "            nav_outs = self.vln_bert('navigation', nav_inputs)\n"
HOOK = (ANCHOR + "            if getattr(self, 'decision_hook', None) is not None:\n"
        "                nav_outs = self.decision_hook(nav_inputs, nav_outs, obs, ended, t, traj)\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepared_bytes(name, source):
    text = source.decode("utf-8")
    if name in ("map_nav_src/r2r/agent.py", "map_nav_src/utils/ops.py"):
        text = text.replace("dtype=np.bool)", "dtype=np.bool_)")
    if name == "map_nav_src/r2r/agent.py":
        if text.count(ANCHOR) != 1:
            raise ValueError("Upstream decision hook anchor changed")
        text = text.replace(ANCHOR, HOOK)
    return text.encode("utf-8")


def verify(dest=DEFAULT_DEST):
    lock = json.loads(LOCK.read_text())
    for name, hashes in lock["files"].items():
        path = Path(dest) / name
        if not path.is_file() or sha256(path.read_bytes()) != hashes["prepared"]:
            raise ValueError(f"Pinned upstream source mismatch: {path}. Run prepare_duet.py.")
    return lock


def prepare(dest):
    lock = json.loads(LOCK.read_text())
    missing = any(not (dest / name).is_file() for name in lock["files"])
    if missing:
        url = f"https://codeload.github.com/cshizhe/VLN-DUET/tar.gz/{lock['commit']}"
        with urllib.request.urlopen(url, timeout=60) as response:
            archive = response.read()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # Read individual files; never extract archive links or arbitrary paths.
            for member in tar.getmembers():
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or not member.isfile() or ".." in parts:
                    continue
                relative = "/".join(parts[1:])
                if relative not in lock["files"]:
                    continue
                data = tar.extractfile(member).read()
                if sha256(data) != lock["files"][relative]["original"]:
                    raise ValueError(f"Archive hash mismatch: {relative}")
                target = dest / relative
                if target.exists():
                    if sha256(target.read_bytes()) not in lock["files"][relative].values():
                        raise ValueError(f"Refusing to overwrite modified source: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
    for name, hashes in lock["files"].items():
        target = dest / name
        data = target.read_bytes()
        if sha256(data) == hashes["prepared"]:
            continue
        if sha256(data) != hashes["original"]:
            raise ValueError(f"Refusing to overwrite modified source: {target}")
        target.write_bytes(prepared_bytes(name, data))
    verify(dest)
    print(f"Prepared DUET {lock['commit']} at {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.dest)
        print("Pinned source and evaluation code verified")
    else:
        prepare(args.dest)
