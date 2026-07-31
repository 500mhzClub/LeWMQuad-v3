#!/usr/bin/env python3
"""Fail-closed launcher for the certified temporal patch-memory JEPA V1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from scripts import (
    execute_go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as executor,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_RELATIVE_PATH = executor.AUTHORITY_RELATIVE_PATH
CERTIFICATION_RELATIVE_PATH = executor.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PermissionError(f"{name} must be a regular non-symlink")
    raw = path.read_bytes()

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PermissionError(f"{name} repeats a JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=unique,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PermissionError(f"{name} contains {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{name} is not strict JSON") from error
    if type(value) is not dict or raw != _canonical_bytes(value) + b"\n":
        raise PermissionError(f"{name} is not canonical JSON")
    return value


def validate_certified_source_v1(
    source_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return executor.validate_certified_source_v1(source_root, authority)


def _safe_source_path(relative: str) -> Any:
    return executor._safe_certified_source_path(relative)


def validate_gpu_v1(torch: Any) -> dict[str, Any]:
    return executor.validate_gpu_v1(torch)


def launch_authorized_v1(authority_path: Path) -> Mapping[str, Any]:
    expected = ROOT / AUTHORITY_RELATIVE_PATH
    if Path(authority_path).resolve(strict=True) != expected.resolve(strict=True):
        raise PermissionError("launcher authority path changed")
    authority = executor.validate_future_execution_prerequisites_v1(
        _strict_json(expected, name="execution authority")
    )
    if ROOT.resolve(strict=True) != Path(
        authority["certified_source_root"]
    ).resolve(strict=True):
        raise PermissionError("launcher is outside the certified source root")
    return executor.execute_authorized_v1(ROOT, authority)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority",
        type=Path,
        default=ROOT / AUTHORITY_RELATIVE_PATH,
    )
    arguments = parser.parse_args()
    result = launch_authorized_v1(arguments.authority)
    print(_canonical_bytes(result).decode("ascii"))
    return (
        0
        if result.get("status") == "PASS_TEMPORAL_PERCEPTION_QUALIFIED"
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
