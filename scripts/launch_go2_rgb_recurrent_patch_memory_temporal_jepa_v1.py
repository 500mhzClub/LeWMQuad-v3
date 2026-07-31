#!/usr/bin/env python3
"""Fail-closed launcher for the certified temporal patch-memory JEPA V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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


def _validate_content_bound(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("content_sha256")) is not str:
        raise PermissionError(f"{name} lacks a content binding")
    core = dict(value)
    observed = core.pop("content_sha256")
    if observed != hashlib.sha256(_canonical_bytes(core)).hexdigest():
        raise PermissionError(f"{name} content binding changed")
    return dict(value)


def _binding(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise PermissionError(f"{name} binding is absent")
    required = {"path", "file_sha256", "byte_count"}
    if (
        not required.issubset(value)
        or type(value["path"]) is not str
        or type(value["file_sha256"]) is not str
        or len(value["file_sha256"]) != 64
        or type(value["byte_count"]) is not int
        or value["byte_count"] <= 0
    ):
        raise PermissionError(f"{name} binding changed")
    return dict(value)


def _safe_source_path(relative: str) -> PurePosixPath:
    value = PurePosixPath(relative)
    folded = tuple(part.casefold() for part in value.parts)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
        or value.suffix not in {".py", ".md", ".json"}
        or any(
            part in {".generated", "sealed", "heldout", "held_out"}
            or part.startswith(("sealed_", "heldout_", "held_out_"))
            for part in folded
        )
        or "sealed_test" in value.name.casefold()
    ):
        raise PermissionError(f"unsafe certified source path: {relative}")
    return value


def _validate_source_binding(source_root: Path, value: Any) -> None:
    binding = _binding(value, name="source")
    relative = _safe_source_path(binding["path"])
    path = source_root.joinpath(*relative.parts)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError(
            f"certified source is absent: {relative}"
        ) from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(source_root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError(f"certified source escaped: {relative}")
    raw = path.read_bytes()
    if (
        len(raw) != binding["byte_count"]
        or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
    ):
        raise PermissionError(f"certified source changed: {relative}")


def validate_certified_source_v1(
    source_root: Path,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(source_root).resolve(strict=True)
    certification_path = root / CERTIFICATION_RELATIVE_PATH
    certification = _validate_content_bound(
        _strict_json(
            certification_path,
            name="clean-export certification",
        ),
        name="clean-export certification",
    )
    identity = _binding(
        authority.get("clean_export_certification"),
        name="certification",
    )
    raw = certification_path.read_bytes()
    bindings = certification.get("source_bindings")
    if (
        identity["path"] != CERTIFICATION_RELATIVE_PATH
        or identity["byte_count"] != len(raw)
        or identity["file_sha256"] != hashlib.sha256(raw).hexdigest()
        or identity.get("content_sha256")
        != certification["content_sha256"]
        or certification.get("schema")
        != f"{executor.SCHEMA_PREFIX}_clean_export_certification_v1"
        or certification.get("status")
        != "PASS_NARROW_CLEAN_EXPORT_CERTIFIED"
        or certification.get("certified_source_root") != str(root)
        or certification.get("pinned_source_and_review_commit")
        != authority.get("pinned_source_and_review_commit")
        or type(bindings) is not list
        or not bindings
    ):
        raise PermissionError("clean-export certification identity changed")
    paths = [dict(item).get("path") for item in bindings]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PermissionError("certified source inventory order changed")
    expected_sha = hashlib.sha256(_canonical_bytes(bindings)).hexdigest()
    if certification.get("bindings_sha256") != expected_sha:
        raise PermissionError("certified source inventory binding changed")
    for binding in bindings:
        _validate_source_binding(root, binding)
    return {
        "status": "PASS_CERTIFIED_SOURCE_REHASH",
        "validated_path_count": len(bindings),
        "bindings_sha256": expected_sha,
        "certification_content_sha256": certification["content_sha256"],
    }


def validate_gpu_v1(torch: Any) -> dict[str, Any]:
    if not bool(torch.cuda.is_available()) or int(torch.cuda.device_count()) != 1:
        raise RuntimeError("temporal V1 requires exactly one visible AMD GPU")
    hip = getattr(getattr(torch, "version", None), "hip", None)
    name = str(torch.cuda.get_device_name(0))
    normalized = name.replace(" ", "").upper()
    if type(hip) is not str or not hip or "AMD" not in normalized or "R9700" not in normalized:
        raise RuntimeError("visible GPU is not the registered AMD R9700")
    return {
        "status": "PASS_EXACTLY_ONE_VISIBLE_AMD_R9700",
        "visible_device_count": 1,
        "visible_device_name": name,
        "torch_hip_version": hip,
        "tensor_allocation_count": 0,
        "dataset_open_count": 0,
        "checkpoint_open_count": 0,
    }


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
    validate_certified_source_v1(ROOT, authority)
    output = Path(authority["runtime_data_root"]) / authority["output_root"]
    if output.exists() or output.is_symlink():
        raise FileExistsError("one-shot output root is not absent")
    import torch

    validate_gpu_v1(torch)
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
