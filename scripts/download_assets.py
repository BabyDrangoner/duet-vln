#!/usr/bin/env python3
"""Download or verify the public DUET assets pinned in configs/assets-manifest.json."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def target_path(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset symlink escapes dataset root: {relative}")
    return path


def valid_file(path, row):
    return path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]


def fetch_file(root, row, verify_only=False):
    target = target_path(root, row["file"])
    if target.exists():
        if not valid_file(target, row):
            raise ValueError(f"Existing asset differs from the pinned bytes: {target}; preserve or remove it manually")
        print(f"Verified {row['file']}", flush=True)
        return {"file": row["file"], "status": "already_verified", "sha256": row["sha256"]}
    if verify_only:
        raise FileNotFoundError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".downloading")
    if partial.is_symlink():
        raise ValueError(f"Refusing partial-file symlink: {partial}")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= row["bytes"]:
                if valid_file(partial, row):
                    os.replace(partial, target)
                    return {"file": row["file"], "status": "recovered_verified", "sha256": row["sha256"]}
                partial.unlink()
                offset = 0
            request = urllib.request.Request(row["source_url"], headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=90) as response:
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Download returned HTML instead of a dataset file")
                resumed = bool(offset and response.status == 206)
                if response.status == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected Content-Range")
                with partial.open("ab" if resumed else "wb") as stream:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            if not valid_file(partial, row):
                # A short transfer can be resumed; complete but corrupt bytes cannot.
                if partial.stat().st_size >= row["bytes"]:
                    partial.unlink()
                raise ValueError(f"Download size or SHA-256 mismatch: {row['file']}")
            os.replace(partial, target)
            print(f"Downloaded and verified {row['file']}", flush=True)
            return {"file": row["file"], "status": "downloaded_verified", "sha256": row["sha256"]}
        except Exception as error:
            print(f"Attempt {attempt + 1}/3: {row['file']}: {type(error).__name__}: {error}", flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def restore_connectivity(root, manifest, verify_only=False):
    out = root / "R2R/connectivity"
    expected = manifest["files_sha256"]
    missing = set()
    for name, expected_hash in expected.items():
        path = target_path(out, name)
        if not path.exists():
            missing.add(name)
        elif not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f"Existing connectivity differs from pinned bytes: {path}")
    if missing and verify_only:
        raise FileNotFoundError(f"Missing {len(missing)} connectivity files in {out}")
    if missing:
        url = manifest["source_repo"].replace("https://github.com/", "https://codeload.github.com/") + "/tar.gz/" + manifest["commit"]
        with urllib.request.urlopen(url, timeout=90) as response:
            blob = response.read()
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                if not member.isfile() or len(parts) != 3 or parts[1] != "connectivity" or parts[2] not in missing:
                    continue
                name = parts[2]
                raw = archive.extractfile(member).read()
                if hashlib.sha256(raw).hexdigest() != expected[name]:
                    raise ValueError(f"Connectivity SHA-256 mismatch: {name}")
                target = target_path(out, name)
                partial = target.with_name(name + ".downloading")
                if partial.is_symlink():
                    raise ValueError(f"Refusing partial-file symlink: {partial}")
                with partial.open("wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(partial, target)
                missing.remove(name)
        if missing:
            raise ValueError(f"Source archive lacks {len(missing)} connectivity files")
    print(f"Verified {len(expected)} connectivity files", flush=True)
    return {"files": len(expected), "commit": manifest["commit"], "status": "verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root; data go into ROOT/datasets")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/assets-manifest.json")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--connectivity-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    root = args.root.expanduser().resolve()
    manifest = json.loads(args.manifest.read_text())
    status_dir = root / "outputs/runtime-wsl"
    status_dir.mkdir(parents=True, exist_ok=True)
    with (status_dir / "assets.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        if not args.connectivity_only:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                records = list(pool.map(lambda row: fetch_file(root / "datasets", row, args.verify_only), manifest["files"]))
        graph_status = restore_connectivity(root / "datasets", manifest["connectivity"], args.verify_only)
        report = {"status": "verified", "manifest_sha256": digest(args.manifest), "files": records, "connectivity": graph_status,
                  "connectivity_only": args.connectivity_only}
        report_path = status_dir / ("connectivity-verification.json" if args.connectivity_only else "assets-verification.json")
        temp = report_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temp, report_path)
        print(f"Verification report: {report_path}")


if __name__ == "__main__":
    main()
