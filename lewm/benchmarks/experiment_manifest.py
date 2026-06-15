"""Content-addressed experiment provenance for reproducible research runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict:
    """Return immutable identity metadata for one file artifact."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": stat.st_size,
    }


def _git_value(repository_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def git_record(repository_root: Path) -> dict:
    """Return commit and dirty-state provenance without mutating the repository."""

    status = _git_value(repository_root, "status", "--porcelain=v1")
    return {
        "repository_root": str(repository_root.resolve()),
        "commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "branch": _git_value(repository_root, "branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": [] if not status else status.splitlines(),
    }


def environment_record(packages: Sequence[str] = ("numpy", "Pillow", "torch")) -> dict:
    """Return a compact environment fingerprint for a research run."""

    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def build_experiment_manifest(
    *,
    experiment_id: str,
    repository_root: Path,
    inputs: Mapping[str, Path],
    artifacts: Mapping[str, Path] | None = None,
    config: Mapping | None = None,
    seeds: Sequence[int] = (),
    run_command: str | None = None,
) -> dict:
    """Build a machine-verifiable run manifest."""

    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    return {
        "schema": "lewm_experiment_manifest_v0",
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_record(repository_root),
        "environment": environment_record(),
        "run_command": run_command,
        "seeds": [int(seed) for seed in seeds],
        "config": dict(config or {}),
        "inputs": {
            name: artifact_record(path) for name, path in sorted(inputs.items())
        },
        "artifacts": {
            name: artifact_record(path)
            for name, path in sorted((artifacts or {}).items())
        },
    }


def verify_manifest_files(manifest: Mapping) -> dict:
    """Verify that manifest inputs and artifacts still match their hashes."""

    records = {
        f"{section}.{name}": record
        for section in ("inputs", "artifacts")
        for name, record in manifest.get(section, {}).items()
    }
    results = {}
    for name, record in records.items():
        path = Path(str(record["path"]))
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        results[name] = {
            "exists": exists,
            "expected_sha256": record["sha256"],
            "actual_sha256": actual,
            "matches": exists and actual == record["sha256"],
        }
    return {
        "schema": "lewm_experiment_manifest_verification_v0",
        "experiment_id": manifest.get("experiment_id"),
        "files": results,
        "passes": all(result["matches"] for result in results.values()),
    }


def write_json(path: Path, payload: Mapping) -> None:
    """Write stable human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
