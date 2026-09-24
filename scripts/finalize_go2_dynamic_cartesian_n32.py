#!/usr/bin/env python3
"""Validate one immutable dynamic-Cartesian N32 result without Torch or output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract  # noqa: E402


IMPLEMENTATION_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json"
).resolve()
CANONICAL_RESULT_PATHS = {
    seed: (
        REPOSITORY_ROOT
        / ".generated/go2_dynamic_cartesian_n32/v1"
        / f"seed_{seed}_result.json"
    ).resolve()
    for seed in contract.EXPECTED_SEEDS
}
CANONICAL_ATTEMPT_MARKER_PATHS = {
    seed: Path(contract.ATTEMPT_MARKER_PATHS[seed]).resolve()
    for seed in contract.EXPECTED_SEEDS
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_absolute_path(value: Path | str, *, name: str) -> Path:
    path = Path(value)
    absolute = Path(os.path.abspath(os.fspath(path)))
    if path != absolute:
        raise ValueError(f"{name} must be a canonical absolute path")
    return absolute


def _open_directory_no_symlinks(path: Path, *, name: str) -> int:
    path = _canonical_absolute_path(path, name=name)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file(path: Path, *, name: str) -> bytes:
    path = _canonical_absolute_path(path, name=name)
    parent = _open_directory_no_symlinks(path.parent, name=f"{name} parent")
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path.name, flags, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    data = b"".join(chunks)
    if identity_before != identity_after or len(data) != before.st_size:
        raise RuntimeError(f"{name} changed while it was read")
    return data


def _strict_json(data: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains nonfinite JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            data,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_expected_json(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[dict[str, Any], str]:
    data = _read_regular_file(path, name=name)
    observed = _sha256_bytes(data)
    if observed != str(expected_sha256):
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    value = _strict_json(data, name=name)
    return value, observed


def _manifest_source_entries(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("implementation manifest lacks its source map")
    entries = sources.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("implementation manifest source entries are malformed")
    normalized = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise ValueError(f"implementation source entry {index} is malformed")
        normalized.append(raw)
    return tuple(normalized)


def _verify_manifest_sources(manifest: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for entry in _manifest_source_entries(manifest):
        role = str(entry.get("role", ""))
        path = _canonical_absolute_path(
            Path(str(entry.get("path", ""))),
            name=f"implementation source {role} path",
        )
        expected = str(entry.get("sha256", ""))
        if not role or role in observed:
            raise ValueError("implementation source roles are empty or duplicated")
        data = _read_regular_file(path, name=f"implementation source {role}")
        digest = _sha256_bytes(data)
        if digest != expected:
            raise ValueError(f"implementation source changed: {role}")
        observed[role] = digest
        if role == "finalizer" and path != Path(__file__).resolve():
            raise ValueError("implementation manifest binds a different finalizer")
    if "finalizer" not in observed:
        raise ValueError("implementation manifest omits the finalizer")
    return observed


def _seed_for_result_path(path: Path) -> int:
    path = _canonical_absolute_path(path, name="result path")
    matches = [seed for seed, canonical in CANONICAL_RESULT_PATHS.items() if path == canonical]
    if len(matches) != 1:
        raise ValueError("authoritative result path is not canonical")
    return matches[0]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument(
        "--implementation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-implementation-manifest-sha256",
        required=True,
    )
    parser.add_argument("--attempt-marker", type=Path, required=True)
    parser.add_argument("--expected-attempt-marker-sha256", required=True)
    parser.add_argument("--seed-20260710-result", type=Path)
    parser.add_argument("--expected-seed-20260710-result-sha256")
    parser.add_argument("--seed-20260710-attempt-marker", type=Path)
    parser.add_argument("--expected-seed-20260710-attempt-marker-sha256")
    return parser.parse_args(argv)


def _validate_primary_arguments(
    args: argparse.Namespace,
    *,
    seed: int,
) -> None:
    primary = (
        args.seed_20260710_result,
        args.expected_seed_20260710_result_sha256,
        args.seed_20260710_attempt_marker,
        args.expected_seed_20260710_attempt_marker_sha256,
    )
    if seed == contract.EXPECTED_SEEDS[0]:
        if any(value is not None for value in primary):
            raise ValueError("the first seed rejects primary-result arguments")
    elif any(value is None for value in primary):
        raise ValueError("the second seed requires the immutable first result")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    implementation_path = _canonical_absolute_path(
        args.implementation_manifest,
        name="implementation manifest path",
    )
    if implementation_path != IMPLEMENTATION_MANIFEST_PATH:
        raise ValueError("implementation manifest path is not canonical")
    implementation, implementation_sha256 = _read_expected_json(
        implementation_path,
        expected_sha256=args.expected_implementation_manifest_sha256,
        name="N32 implementation manifest",
    )
    implementation = contract.validate_implementation_manifest(implementation)
    source_hashes_before = _verify_manifest_sources(implementation)

    result_path = _canonical_absolute_path(args.result, name="result path")
    seed = _seed_for_result_path(result_path)
    _validate_primary_arguments(args, seed=seed)
    attempt_marker_path = _canonical_absolute_path(
        args.attempt_marker, name="attempt-marker path"
    )
    if attempt_marker_path != CANONICAL_ATTEMPT_MARKER_PATHS[seed]:
        raise ValueError("authoritative attempt-marker path is not canonical")
    attempt_marker, attempt_marker_sha256 = _read_expected_json(
        attempt_marker_path,
        expected_sha256=args.expected_attempt_marker_sha256,
        name=f"N32 seed {seed} attempt marker",
    )
    result, result_sha256 = _read_expected_json(
        result_path,
        expected_sha256=args.expected_result_sha256,
        name=f"N32 seed {seed} result",
    )

    primary_sha256 = None
    primary_attempt_marker_sha256 = None
    pair_decision = None
    if seed == contract.EXPECTED_SEEDS[1]:
        assert args.seed_20260710_result is not None
        assert args.expected_seed_20260710_result_sha256 is not None
        assert args.seed_20260710_attempt_marker is not None
        assert args.expected_seed_20260710_attempt_marker_sha256 is not None
        primary_path = _canonical_absolute_path(
            args.seed_20260710_result,
            name="first-seed result path",
        )
        if primary_path != CANONICAL_RESULT_PATHS[contract.EXPECTED_SEEDS[0]]:
            raise ValueError("first-seed authorization path is not canonical")
        primary, primary_sha256 = _read_expected_json(
            primary_path,
            expected_sha256=args.expected_seed_20260710_result_sha256,
            name="N32 seed 20260710 result",
        )
        primary_attempt_marker_path = _canonical_absolute_path(
            args.seed_20260710_attempt_marker,
            name="first-seed attempt-marker path",
        )
        if primary_attempt_marker_path != CANONICAL_ATTEMPT_MARKER_PATHS[
            contract.EXPECTED_SEEDS[0]
        ]:
            raise ValueError("first-seed attempt-marker path is not canonical")
        primary_attempt_marker, primary_attempt_marker_sha256 = _read_expected_json(
            primary_attempt_marker_path,
            expected_sha256=args.expected_seed_20260710_attempt_marker_sha256,
            name="N32 seed 20260710 attempt marker",
        )
        primary_validated = contract.validate_authoritative_result(
            primary,
            expected_seed=contract.EXPECTED_SEEDS[0],
            implementation_manifest=implementation,
            implementation_manifest_file_sha256=implementation_sha256,
            attempt_marker=primary_attempt_marker,
            attempt_marker_file_sha256=primary_attempt_marker_sha256,
        )
        validated = contract.validate_authoritative_result(
            result,
            expected_seed=seed,
            implementation_manifest=implementation,
            implementation_manifest_file_sha256=implementation_sha256,
            attempt_marker=attempt_marker,
            attempt_marker_file_sha256=attempt_marker_sha256,
            primary_result=primary,
            primary_file_sha256=primary_sha256,
            primary_attempt_marker=primary_attempt_marker,
            primary_attempt_marker_file_sha256=primary_attempt_marker_sha256,
        )
        pair_decision = contract.validate_seed_pair(
            primary,
            result,
            implementation,
            implementation_sha256,
            primary_sha256,
            primary_attempt_marker,
            primary_attempt_marker_sha256,
            attempt_marker,
            attempt_marker_sha256,
        )
        if _sha256_bytes(_read_regular_file(primary_path, name="first-seed recheck")) != (
            primary_sha256
        ):
            raise RuntimeError("first-seed result changed during finalization")
        if _sha256_bytes(
            _read_regular_file(
                primary_attempt_marker_path,
                name="first-seed attempt-marker recheck",
            )
        ) != primary_attempt_marker_sha256:
            raise RuntimeError("first-seed attempt marker changed during finalization")
    else:
        validated = contract.validate_authoritative_result(
            result,
            expected_seed=seed,
            implementation_manifest=implementation,
            implementation_manifest_file_sha256=implementation_sha256,
            attempt_marker=attempt_marker,
            attempt_marker_file_sha256=attempt_marker_sha256,
        )

    if _sha256_bytes(_read_regular_file(result_path, name="result recheck")) != (
        result_sha256
    ):
        raise RuntimeError("N32 result changed during finalization")
    if _sha256_bytes(
        _read_regular_file(attempt_marker_path, name="attempt-marker recheck")
    ) != attempt_marker_sha256:
        raise RuntimeError("N32 attempt marker changed during finalization")
    if _sha256_bytes(
        _read_regular_file(implementation_path, name="implementation recheck")
    ) != implementation_sha256:
        raise RuntimeError("implementation manifest changed during finalization")
    if _verify_manifest_sources(implementation) != source_hashes_before:
        raise RuntimeError("implementation sources changed during finalization")

    decision = validated["decision"]
    summary = {
        "schema": "lewm_go2_dynamic_cartesian_n32_finalization_summary_v1",
        "validation_only": True,
        "output_file_created": False,
        "seed": seed,
        "result_path": str(result_path),
        "result_sha256": result_sha256,
        "result_content_sha256": validated["content_sha256"],
        "attempt_marker_path": str(attempt_marker_path),
        "attempt_marker_sha256": attempt_marker_sha256,
        "implementation_manifest_sha256": implementation_sha256,
        "seed_20260710_authorization_sha256": primary_sha256,
        "seed_20260710_attempt_marker_sha256": primary_attempt_marker_sha256,
        "classification": decision["classification"],
        "favorable": decision["favorable"],
        "shared_jepa_construction_licensed": bool(
            pair_decision is not None
            and pair_decision["shared_jepa_construction_licensed"]
        ),
    }
    print(
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
