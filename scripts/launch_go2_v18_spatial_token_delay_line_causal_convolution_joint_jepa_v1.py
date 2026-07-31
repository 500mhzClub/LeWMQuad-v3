#!/usr/bin/env python3
"""Fail-closed launcher for the V18 causal delay-line memory JEPA V1."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence

from scripts import launch_go2_rgb_memory_role_factorized_joint_jepa_v1 as v5_launcher


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PREFIX = (
    "lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "update_zero_gate_timing_integrity_replacement_v2"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "update_zero_gate_timing_integrity_replacement_v2_execution_authorization_"
    "2026-07-31.json"
)
CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "update_zero_gate_timing_integrity_replacement_v2_clean_export_certification_"
    "2026-07-31.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "update_zero_gate_timing_integrity_replacement_v2/attempt_v1"
)
CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/"
    "LeWMQuad-v3-v18-spatial-token-delay-line-batch-schema-integrity-replacement-v2-source"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_v18_spatial_token_delay_line_causal_convolution_"
    "joint_jepa_v1"
)
MODEL_MODULE_NAME = (
    "lewm.models.v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1"
)
EXPERIMENT_ARM_NAME = (
    "v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "update_zero_gate_timing_integrity_replacement_v2"
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"{name} must be a regular non-symlink")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{name} is not strict JSON") from error
    if type(value) is not dict or raw != _canonical_json_bytes(value) + b"\n":
        raise PermissionError(f"{name} must be canonical JSON")
    return value


def _validate_content_bound(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("content_sha256")) is not str:
        raise PermissionError(f"{name} has no content binding")
    core = dict(value)
    observed = core.pop("content_sha256")
    if observed != hashlib.sha256(_canonical_json_bytes(core)).hexdigest():
        raise PermissionError(f"{name} content binding changed")
    return dict(value)


def _binding(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PermissionError(f"{name} binding is absent")
    result = dict(value)
    required = {"path", "file_sha256", "byte_count"}
    if (
        not required.issubset(result)
        or type(result["path"]) is not str
        or type(result["file_sha256"]) is not str
        or len(result["file_sha256"]) != 64
        or type(result["byte_count"]) is not int
        or result["byte_count"] <= 0
    ):
        raise PermissionError(f"{name} binding changed")
    return result


def _safe_source_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    folded = tuple(part.casefold() for part in path.parts)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "." in path.parts
        or path.suffix not in {".py", ".md", ".json"}
        or any(
            part in {"data", "heldout", "held_out", "sealed"}
            or part.startswith(("sealed_", "heldout_", "held_out_"))
            or part in {".generated", "runtime", "runtime_artifacts", "checkpoints"}
            for part in folded
        )
        or "sealed_test" in path.name.casefold()
    ):
        raise PermissionError(f"unsafe certified source path: {relative}")
    return path


def _validate_certified_source_binding(
    source_root: Path, value: Mapping[str, Any]
) -> Path:
    binding = _binding(value, name="certified source")
    relative = binding["path"]
    pure = _safe_source_path(relative)
    root = Path(source_root).resolve(strict=True)
    path = root.joinpath(*pure.parts)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError(f"certified source is absent: {relative}") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
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
    return resolved


def validate_source_certification_v1(
    source_root: Path, authority: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(source_root).resolve(strict=True)
    certification_path = root / CERTIFICATION_RELATIVE_PATH
    certification = _validate_content_bound(
        _strict_json(certification_path, name="clean-export certification"),
        name="clean-export certification",
    )
    identity = _binding(
        authority.get("clean_export_certification"),
        name="authority certification",
    )
    raw = certification_path.read_bytes()
    if (
        identity["path"] != CERTIFICATION_RELATIVE_PATH
        or identity["file_sha256"] != hashlib.sha256(raw).hexdigest()
        or identity["byte_count"] != len(raw)
        or identity.get("content_sha256") != certification["content_sha256"]
        or certification.get("certified_source_root") != str(root)
        or certification.get("pinned_source_and_review_commit")
        != authority.get("pinned_source_and_review_commit")
        or certification.get("status") != "PASS_NARROW_CLEAN_EXPORT_CERTIFIED"
    ):
        raise PermissionError("clean-export certification identity changed")
    bindings = certification.get("source_bindings")
    if type(bindings) is not list or not bindings:
        raise PermissionError("certified source inventory is absent")
    paths = [dict(value).get("path") for value in bindings]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise PermissionError("certified source inventory order changed")
    expected_sha = hashlib.sha256(_canonical_json_bytes(bindings)).hexdigest()
    if certification.get("bindings_sha256") != expected_sha:
        raise PermissionError("certified source inventory binding changed")
    for binding in bindings:
        _validate_certified_source_binding(root, binding)
    return {
        "schema": f"{SCHEMA_PREFIX}_source_validation_receipt_v1",
        "status": "PASS_CERTIFIED_SOURCE_REHASH",
        "validated_path_count": len(bindings),
        "bindings_sha256": expected_sha,
        "certified_source_bindings_sha256": expected_sha,
        "certified_export_binding_count": len(bindings),
        "certified_source_bindings": [dict(value) for value in bindings],
        "certification_content_sha256": certification["content_sha256"],
    }


# Reuse the reviewed V13/V25 data and GPU composer while replacing only its
# selected model, trainer, executor, and source validator.
_BASE = v5_launcher._BASE_LAUNCHER
_OVERRIDES = {
    "AUTHORITY_RELATIVE_PATH": AUTHORITY_RELATIVE_PATH,
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": CERTIFICATION_RELATIVE_PATH,
    "EXECUTOR_MODULE_NAME": EXECUTOR_MODULE_NAME,
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": SCHEMA_PREFIX,
    "EXPERIMENT_ARM_NAME": EXPERIMENT_ARM_NAME,
    "LAUNCHER_SCHEMA": f"{SCHEMA_PREFIX}_launcher_v1",
}
for _name, _value in _OVERRIDES.items():
    setattr(_BASE, _name, _value)
_BASE._validate_certified_source_binding_v13 = _validate_certified_source_binding


def _assert_runtime_adapter() -> None:
    if any(getattr(_BASE, name, None) != value for name, value in _OVERRIDES.items()):
        raise PermissionError("delay-line inherited runtime selectors changed")
    if _BASE._validate_certified_source_binding_v13 is not _validate_certified_source_binding:
        raise PermissionError("delay-line terminal source validator changed")


def _load_authority(path: Path) -> dict[str, Any]:
    expected = ROOT / AUTHORITY_RELATIVE_PATH
    try:
        resolved = Path(path).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("fixed delay-line authority is absent") from error
    if resolved != expected.absolute() or not resolved.is_relative_to(ROOT):
        raise PermissionError("delay-line authority path changed")
    return _strict_json(expected, name="delay-line execution authority")


def _terminal_exists(output_root: Path) -> bool:
    return any(
        (output_root / name).exists() or (output_root / name).is_symlink()
        for name in ("success.json", "failure.json")
    )


def validate_pre_reservation_gpu_visibility_v1(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reuse the reviewed single-GPU visibility guard before consuming the attempt."""

    return v5_launcher.validate_pre_reservation_gpu_visibility_v1(environment)


def execute_future_authorized_v1(
    *, repository_root: Path, authority: Mapping[str, Any]
) -> Mapping[str, Any]:
    _assert_runtime_adapter()
    repository = Path(repository_root).resolve(strict=True)
    if repository != ROOT.resolve(strict=True):
        raise PermissionError("delay-line execution must run from its certified export")
    fixed = _load_authority(ROOT / AUTHORITY_RELATIVE_PATH)
    if type(authority) is not dict or authority != fixed:
        raise PermissionError("supplied delay-line authority differs from fixed file")
    executor = importlib.import_module(EXECUTOR_MODULE_NAME)
    validated = executor.validate_future_execution_prerequisites_v1(fixed)
    validate_pre_reservation_gpu_visibility_v1()
    _BASE._validate_certified_source_root_v13(repository, validated)
    _BASE._activate_certified_source_root_v13(repository)
    source_evidence = validate_source_certification_v1(repository, validated)
    executor.validate_bound_sources_v13 = lambda root: validate_source_certification_v1(
        root, validated
    )
    _BASE._validate_runtime_data_root_v13(repository, validated)
    _BASE._ensure_output_parent_v13(repository, OUTPUT_ROOT_RELATIVE_PATH)
    reservation, recovery = executor.reserve_or_recover_attempt_v1(
        repository,
        validated,
        created_utc=_BASE._utc_now_v13(),
    )
    output_root = repository / OUTPUT_ROOT_RELATIVE_PATH
    runtime: Any = None
    stage = "post_reservation_runtime_composition"
    try:
        runtime = _BASE.compose_runtime_v13(
            repository_root=repository,
            authority=validated,
            reservation=reservation,
            source_evidence=source_evidence,
        )
        publisher = _BASE.V13WriteOncePublisher(output_root, executor)
        stage = "authorized_delay_line_engine"
        result = executor.run_future_authorized_engine_v1(
            authority=validated,
            reservation=reservation,
            recovery=recovery,
            runtime=runtime,
            publisher=publisher,
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("delay-line controller omitted terminal receipt")
        return dict(result)
    except BaseException as error:
        if not _terminal_exists(output_root):
            executor.terminalize_failure_v1(
                output_root,
                reservation,
                stage=stage,
                error=error,
                created_utc=_BASE._utc_now_v13(),
            )
        raise
    finally:
        if runtime is not None:
            runtime.close_v13()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-authority", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(
            json.dumps(
                {
                    "schema": f"{SCHEMA_PREFIX}_launcher_v1",
                    "status": "DENIED_NO_FUTURE_AUTHORITY",
                    "scientific_payload_opened": False,
                    "reservation_created": False,
                },
                sort_keys=True,
            )
        )
        return 4
    parsed = _parser().parse_args(arguments)
    result = execute_future_authorized_v1(
        repository_root=ROOT,
        authority=_load_authority(parsed.future_authority),
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if str(result.get("status", "")).startswith("PASS_") else 3


if __name__ == "__main__":
    raise SystemExit(main())
