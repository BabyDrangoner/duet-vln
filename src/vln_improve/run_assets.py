"""Immutable cache backups and small, credential-free source snapshots.

The caller establishes that ``backup_run`` is persistent storage. Read-back
hashes here verify filesystem contents, not an independent Drive API receipt.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, Sequence
import uuid

from .train import read_manifests


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _file(path: Path, relative: str) -> dict[str, Any]:
    return {"path": relative, "size": path.stat().st_size, "sha256": _sha(path)}


def _describe(cache_dirs: Sequence[str | Path]) -> dict[str, Any]:
    manifests = read_manifests(cache_dirs)
    caches = []
    for index, manifest in enumerate(manifests):
        paths = [manifest.root / "manifest.json", *manifest.shards]
        files = [_file(path, path.relative_to(manifest.root).as_posix()) for path in paths]
        caches.append({"directory": f"cache-{index:04d}", "files": files})
    return {"schema_version": 1, "caches": caches}


def _safe_file(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("backup contains an invalid relative path")
    parts = Path(name).parts
    if ".." in parts or "." in parts:
        raise ValueError("backup path escapes its directory")
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError("backup paths must not be symlinks")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("backup path escapes its directory")
    return current


def _verify_inputs(root: Path) -> tuple[dict[str, Any], list[Path]]:
    try:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("inputs backup is not a real directory")
        index_path = _safe_file(root, "index.json")
        marker = _safe_file(root, "COMMITTED").read_text().strip()
        if marker != _sha(index_path):
            raise ValueError("inputs backup commit checksum mismatch")
        index = json.loads(index_path.read_bytes())
        if not isinstance(index, dict) or type(index.get("schema_version")) is not int or index["schema_version"] != 1:
            raise ValueError("unsupported inputs backup schema")
        caches = index.get("caches")
        if not isinstance(caches, list) or not caches:
            raise ValueError("inputs backup has no caches")
        roots = []
        for number, cache in enumerate(caches):
            if not isinstance(cache, dict) or cache.get("directory") != f"cache-{number:04d}":
                raise ValueError("inputs backup cache order is invalid")
            cache_root = _safe_file(root, cache["directory"])
            files = cache.get("files")
            if not isinstance(files, list) or not files:
                raise ValueError("inputs backup file list is invalid")
            names = []
            for item in files:
                if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                    raise ValueError("inputs backup file metadata is invalid")
                if type(item["size"]) is not int or item["size"] < 0 or not isinstance(item["sha256"], str):
                    raise ValueError("inputs backup file checksum metadata is invalid")
                path = _safe_file(cache_root, item["path"])
                if not path.is_file() or _file(path, item["path"]) != item:
                    raise ValueError(f"inputs backup content checksum mismatch: {path}")
                names.append(item["path"])
            if len(names) != len(set(names)):
                raise ValueError("inputs backup has duplicate files")
            roots.append(cache_root)
        # Re-derive the entire index, including shard order and train_fit checks.
        if _describe(roots) != index:
            raise ValueError("inputs backup index disagrees with cache manifests")
        return index, roots
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"incomplete or corrupt inputs backup: {root}") from error


def _copy_inputs(roots: Sequence[Path], index: dict[str, Any], destination: Path) -> None:
    destination.mkdir()
    for source, cache in zip(roots, index["caches"]):
        for item in cache["files"]:
            target = destination / cache["directory"] / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with (source / item["path"]).open("rb") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            if _file(target, item["path"]) != item:
                raise ValueError("cache changed during backup or copy verification failed")
    content = _json(index)
    _write(destination / "index.json", content)
    _write(destination / "COMMITTED", (hashlib.sha256(content).hexdigest() + "\n").encode())
    _verify_inputs(destination)


def prepare_inputs(
    cache_dirs: Sequence[str | Path], local_run: str | Path, backup_run: str | Path,
) -> list[Path]:
    """Back up immutable train caches, or restore missing ones after VM loss.

    Existing source caches must match their original positions and bytes. An
    existing corrupt backup is never replaced, even when sources remain intact.
    Absolute source directory names are deliberately absent from the identity.
    """
    requested = [Path(path).expanduser().absolute() for path in cache_dirs]
    if not requested:
        raise ValueError("at least one cache path is required")
    if len({path.resolve() for path in requested}) != len(requested):
        raise ValueError("duplicate cache directory")
    backup_base = Path(backup_run).expanduser().resolve()
    local_base = Path(local_run).expanduser().resolve()
    backup = backup_base / "inputs"
    backup_base.mkdir(parents=True, exist_ok=True)
    if backup.exists() or backup.is_symlink():
        index, backed_up = _verify_inputs(backup)
        if len(requested) != len(index["caches"]):
            raise ValueError("cache list length changed since this run began")
        for number, source in enumerate(requested):
            if source.exists() or source.is_symlink():
                actual = _describe([source])["caches"][0]
                expected = index["caches"][number]
                if actual["files"] != expected["files"]:
                    raise ValueError(f"source cache changed at position {number}: {source}")
        if all(source.exists() for source in requested):
            # Also check shared feature/provenance consistency across originals.
            read_manifests(requested)
            return [source.resolve() for source in requested]
    else:
        if list(backup_base.glob(".inputs-*.tmp")):
            raise ValueError("uncommitted inputs backup exists; inspect it before retrying")
        index = _describe(requested)
        stage = backup_base / f".inputs-{uuid.uuid4().hex}.tmp"
        # Leave an interrupted stage available for diagnosis, never pretend it
        # is a complete backup on a subsequent run.
        _copy_inputs([source.resolve() for source in requested], index, stage)
        if backup.exists():
            raise ValueError("another process created this inputs backup")
        stage.rename(backup)
        _verify_inputs(backup)
        return [source.resolve() for source in requested]

    local_base.mkdir(parents=True, exist_ok=True)
    local = local_base / "inputs"
    if local.exists() or local.is_symlink():
        local_index, local_roots = _verify_inputs(local)
        if local_index != index:
            raise ValueError("local restored inputs differ from this run's backup")
        return local_roots
    stage = local_base / f".inputs-{uuid.uuid4().hex}.tmp"
    _copy_inputs(backed_up, index, stage)
    stage.rename(local)
    return _verify_inputs(local)[1]


_SOURCE_DIRS = ("src", "scripts", "configs", "config", "docs", "tests")
_SOURCE_SUFFIXES = {".py", ".sh", ".json", ".toml", ".yaml", ".yml", ".md", ".txt", ".lock", ".ini", ".cfg"}
_ROOT_NAMES = {
    "pyproject.toml", "README.md", "LICENSE", "LICENSE.txt", ".gitignore",
    "uv.lock", "poetry.lock", "requirements.txt", "requirements.lock",
    "setup_gpu.sh", "build_mattersim.sh", "download_assets.py", "full_model_smoke.py",
}
_REPRO_NAMES = {
    "artifacts/assets-manifest.json", "artifacts/requirements-freeze.txt",
    "artifacts/mattersim-compat.patch",
}
_FORBIDDEN = {".git", ".venv", "venv", "__pycache__", "datasets", "data", "outputs", "keys", "auth", "credentials", "secrets"}


def _allowed(path: Path) -> bool:
    if any(part.lower() in _FORBIDDEN or part.startswith(".") for part in path.parts):
        return False
    name = path.name.lower()
    stem = path.stem.lower()
    return not (
        stem in {"auth", "credentials", "secrets", "token", "tokens", "key", "keys"}
        or name in {"ssh_config", "known_hosts", "id_rsa", "id_ed25519"}
        or name.startswith(("client_secret", "service_account"))
    )


def snapshot_code(project_root: str | Path, backup_run: str | Path) -> dict[str, Any]:
    """Snapshot allowlisted source and return stable core-code/content hashes."""
    project = Path(project_root).expanduser().resolve(strict=True)
    sources: dict[str, bytes] = {}
    for dirname in _SOURCE_DIRS:
        directory = project / dirname
        if directory.is_symlink():
            raise ValueError("source directories must not be symlinks")
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(project)
            if not _allowed(relative) or path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"source snapshot refuses symlink: {relative}")
            if path.is_file():
                sources[relative.as_posix()] = path.read_bytes()
    for name in sorted(_ROOT_NAMES | _REPRO_NAMES):
        path = project / name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != project and parent.is_relative_to(project)):
            raise ValueError(f"source snapshot refuses symlink: {name}")
        if path.is_file():
            sources[name] = path.read_bytes()
    if not sources:
        raise ValueError("no allowlisted source files found")
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(sources.items())}
    core = {name: digest for name, digest in hashes.items() if name.startswith(("src/", "scripts/")) and name.endswith(".py")}
    if not core:
        raise ValueError("no core src/scripts code found")
    code_sha256 = hashlib.sha256(_json(core)).hexdigest()
    archive = io.BytesIO()
    with gzip.GzipFile(fileobj=archive, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as tar:
            for name, content in sorted(sources.items()):
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(content))
    content = archive.getvalue()
    archive_sha256 = hashlib.sha256(content).hexdigest()
    destination = Path(backup_run).expanduser().resolve() / "reproducibility"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"source-{archive_sha256}.tar.gz"
    metadata = {
        "schema_version": 1, "code_sha256": code_sha256,
        "archive_sha256": archive_sha256, "archive_size": len(content),
        "files": hashes, "verification": "filesystem-readback-sha256",
    }
    metadata_path = target.with_suffix(target.suffix + ".json")
    if target.exists() or target.is_symlink():
        if target.is_symlink() or _sha(target) != archive_sha256:
            raise ValueError("existing source archive is corrupt")
        if not metadata_path.is_file() or metadata_path.is_symlink() or metadata_path.read_bytes() != _json(metadata):
            raise ValueError("existing source archive metadata is incomplete or corrupt")
    else:
        temporary = destination / f".source-{uuid.uuid4().hex}.tmp"
        _write(temporary, content)
        if _sha(temporary) != archive_sha256:
            raise ValueError("source snapshot read-back checksum mismatch")
        temporary.rename(target)
        _write(metadata_path, _json(metadata))
        if _sha(target) != archive_sha256 or metadata_path.read_bytes() != _json(metadata):
            raise ValueError("source snapshot final verification failed")
    return {**metadata, "archive": str(target), "metadata": str(metadata_path)}
