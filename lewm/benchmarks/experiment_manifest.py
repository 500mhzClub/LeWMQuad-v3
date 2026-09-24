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
from typing import Any, Mapping, Sequence


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


def sha256_json(payload: Any) -> str:
    """Return a stable SHA-256 digest for one JSON-compatible value."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def _git_bytes(repository_root: Path, *args: str) -> bytes | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def git_record(repository_root: Path) -> dict:
    """Return commit and dirty-state provenance without mutating the repository."""

    status = _git_value(repository_root, "status", "--porcelain=v1")
    diff = _git_bytes(
        repository_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        ".",
    )
    return {
        "repository_root": str(repository_root.resolve()),
        "commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "branch": _git_value(repository_root, "branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": [] if not status else status.splitlines(),
        "status_sha256": (
            hashlib.sha256(status.encode()).hexdigest()
            if status is not None
            else None
        ),
        # The tracked working-tree and index delta is enough to recover the
        # code identity when paired with the commit. Untracked source inputs
        # must additionally be included in ``inputs`` so their content hashes
        # are recorded rather than only their names from status porcelain.
        "dirty_diff_sha256": (
            hashlib.sha256(diff).hexdigest() if diff is not None else None
        ),
        "dirty_diff_size_bytes": len(diff) if diff is not None else None,
    }


def _scene_split_record(scene_splits: Mapping[str, Sequence[str]]) -> dict:
    normalized = {
        str(split): [str(scene_id) for scene_id in scene_ids]
        for split, scene_ids in scene_splits.items()
    }
    owners: dict[str, str] = {}
    for split, scene_ids in normalized.items():
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError(f"duplicate scene ID within split {split!r}")
        for scene_id in scene_ids:
            previous = owners.setdefault(scene_id, split)
            if previous != split:
                raise ValueError(
                    f"scene {scene_id!r} appears in both {previous!r} and {split!r}"
                )
    return {
        "splits": normalized,
        "sha256": sha256_json(normalized),
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
    scene_splits: Mapping[str, Sequence[str]] | None = None,
    geometry_contract: Path | None = None,
    corpus_plan: Path | None = None,
    checkpoints: Mapping[str, Path] | None = None,
    runtime_contract: Mapping | None = None,
) -> dict:
    """Build a machine-verifiable run manifest."""

    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    config_record = dict(config or {})
    dependencies: dict[str, Any] = {}
    if geometry_contract is not None:
        dependencies["geometry_contract"] = artifact_record(geometry_contract)
    if corpus_plan is not None:
        dependencies["corpus_plan"] = artifact_record(corpus_plan)
    if checkpoints:
        dependencies["checkpoints"] = {
            name: artifact_record(path) for name, path in sorted(checkpoints.items())
        }
    return {
        "schema": "lewm_experiment_manifest_v1",
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_record(repository_root),
        "environment": environment_record(),
        "run_command": run_command,
        "seeds": [int(seed) for seed in seeds],
        "config": config_record,
        "config_sha256": sha256_json(config_record),
        "scene_splits": (
            _scene_split_record(scene_splits) if scene_splits is not None else None
        ),
        "runtime_contract": dict(runtime_contract or {}),
        "dependencies": dependencies,
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
    for name, record in manifest.get("dependencies", {}).items():
        if name == "checkpoints":
            records.update(
                {
                    f"dependencies.checkpoints.{checkpoint_name}": checkpoint_record
                    for checkpoint_name, checkpoint_record in record.items()
                }
            )
        else:
            records[f"dependencies.{name}"] = record
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
