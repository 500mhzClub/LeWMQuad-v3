#!/usr/bin/env python3
"""Future one-shot launcher/composer for Camera-evidence-bottleneck V13.

Import and the no-argument CLI are source-only.  The only path that imports
Torch, inspects an accelerator, or opens development payload first validates a
future authority receipt and creates the immutable attempt reservation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "integrity_replacement_v2_execution_authorization_2026-07-29.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "integrity_replacement_v2_source_manifest_2026-07-29.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "integrity_replacement_v2_source_review_2026-07-29.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "integrity_replacement_v2_clean_export_certification_2026-07-29.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
    "source_closure.py"
)
ALLOWED_LABEL_ROLES = ("train", "checkpoint_selection")
FORBIDDEN_LABEL_ROLE = "probability_calibration"
EXPERIMENT_SEED = 20_260_728
MAXIMUM_UPDATES = 1_000
PRESENTATIONS_PER_UPDATE = 16
MAXIMUM_PRESENTATIONS = 16_000
OBSERVATION_UPDATES = (0, 100, 400, 1_000)
REQUIRED_GPU_NAME = "AMD Radeon AI PRO R9700"
REQUIRED_GPU_MEMORY_BYTES = 34_208_743_424
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

CAMERA_ARRAY_FILENAMES = {
    "camera_origin": "camera_origin_body_m.f4",
    "camera_basis": "camera_basis_body_fru.f4",
    "ground": "ground_plane_z_body_m.f4",
    "pixel_hit": "pixel_hit_mask.u1",
    "pixel_distance": "pixel_first_hit_distance_m.f4",
    "ground_in_frustum": "ground_support_in_frustum.u1",
    "ground_clear": "ground_support_clear_to_target.u1",
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must be one JSON object")
    return value


def _activate_certified_source_root_v13(source_root: Path) -> Path:
    """Make the exact certified export importable under ``python -I``."""

    candidate = Path(source_root)
    try:
        resolved = candidate.resolve(strict=True)
        expected = ROOT.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("V13 certified source root is absent") from error
    if (
        resolved != expected
        or candidate.is_symlink()
        or not resolved.is_dir()
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
    ):
        raise PermissionError(
            "authorized V13 execution requires its certified root under python -I -B"
        )
    nested_root = resolved / "lewm_worlds"
    package_directory = nested_root / "lewm_worlds"
    try:
        nested_resolved = nested_root.resolve(strict=True)
        package_resolved = package_directory.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("V13 nested lewm_worlds package is absent") from error
    if (
        nested_root.is_symlink()
        or package_directory.is_symlink()
        or not nested_root.is_dir()
        or not package_directory.is_dir()
        or nested_resolved != nested_root.absolute()
        or package_resolved != package_directory.absolute()
        or not nested_resolved.is_relative_to(resolved)
        or not package_resolved.is_relative_to(nested_resolved)
    ):
        raise PermissionError(
            "V13 nested lewm_worlds package escaped the certified source root"
        )
    sources = (str(resolved), str(nested_resolved))
    for source in sources:
        while source in sys.path:
            sys.path.remove(source)
    sys.path[0:0] = sources
    return resolved


def _validate_certified_source_root_v13(
    source_root: Path,
    authority: Mapping[str, Any],
) -> Path:
    value = authority.get("certified_source_root")
    if not isinstance(value, str) or not value:
        raise PermissionError("V13 authority omitted the certified source root")
    candidate = Path(source_root)
    expected = Path(value)
    try:
        resolved = candidate.resolve(strict=True)
        bound = expected.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("V13 certified source root is absent") from error
    git_marker = resolved / ".git"
    if (
        candidate.is_symlink()
        or not candidate.is_dir()
        or resolved != candidate
        or bound != expected
        or resolved != bound
        or git_marker.exists()
        or git_marker.is_symlink()
    ):
        raise PermissionError(
            "V13 execution is not hosted by the authority-bound narrow source export"
        )
    return resolved


def _load_authority_file_v13(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    expected = ROOT / AUTHORITY_RELATIVE_PATH
    try:
        source_root = ROOT.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("fixed future V13 authority is absent") from error
    if (
        resolved != expected_resolved
        or resolved != expected.absolute()
        or not resolved.is_relative_to(source_root)
        or candidate.is_symlink()
        or not candidate.is_file()
        or expected.is_symlink()
    ):
        raise PermissionError("future authority must be a regular non-symlink")
    raw = candidate.read_bytes()
    value = _strict_json_object(raw, name="future V13 authority")
    if raw != _canonical_json_bytes(value) + b"\n":
        raise PermissionError("future V13 authority must be canonical JSON")
    return value


def _read_content_bound_evidence_v13(
    source_root: Path,
    relative: str,
    *,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PermissionError(f"{name} path escaped the certified source root")
    path = Path(source_root).joinpath(*pure.parts)
    try:
        root = Path(source_root).resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError(f"{name} is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError(f"{name} escaped or is not a regular file")
    raw = path.read_bytes()
    value = _strict_json_object(raw, name=name)
    declared = value.get("content_sha256")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        type(declared) is not str
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
        or hashlib.sha256(_canonical_json_bytes(core)).hexdigest() != declared
    ):
        raise PermissionError(f"{name} content binding changed")
    return value, raw


def _manifest_identity_v13(
    manifest: Mapping[str, Any], raw: bytes
) -> dict[str, Any]:
    return {
        "path": SOURCE_MANIFEST_RELATIVE_PATH,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": manifest.get("content_sha256"),
        "source_bindings_sha256": manifest.get("source_bindings_sha256"),
    }


def _validate_certified_source_binding_v13(
    source_root: Path,
    binding: Mapping[str, Any],
) -> str:
    if type(binding) is not dict or set(binding) != {
        "path",
        "file_sha256",
        "byte_count",
    }:
        raise PermissionError("V13 clean-export inventory binding changed")
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise PermissionError("V13 clean-export inventory path is malformed")
    pure = PurePosixPath(relative)
    folded = tuple(part.casefold() for part in pure.parts)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or any(part == "sealed" or part.startswith("sealed_") for part in folded)
        or any(part in {"heldout", "held_out", ".generated", "data", "datasets"}
               for part in folded)
        or any(part.startswith("heldout_") or part.startswith("held_out_")
               for part in folded)
    ):
        raise PermissionError("V13 clean-export inventory contains a protected path")
    path = Path(source_root).joinpath(*pure.parts)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("V13 certified export path is absent") from error
    root = Path(source_root).resolve(strict=True)
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("V13 certified export path escaped or is not regular")
    raw = path.read_bytes()
    if (
        binding.get("file_sha256") != hashlib.sha256(raw).hexdigest()
        or binding.get("byte_count") != len(raw)
    ):
        raise PermissionError("V13 certified export working bytes changed")
    return relative


def _load_source_closure_checker_v13(source_root: Path) -> Any:
    path = Path(source_root) / SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    try:
        root = Path(source_root).resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("V13 source-closure checker is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("V13 source-closure checker escaped or is not regular")
    name = "_lewm_v13_camera_evidence_launcher_source_closure_checker"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the V13 source-closure checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_source_evidence_v13(
    source_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Revalidate the exact source closure and clean export before reservation."""

    root = _validate_certified_source_root_v13(source_root, authority)
    manifest, manifest_raw = _read_content_bound_evidence_v13(
        root, SOURCE_MANIFEST_RELATIVE_PATH, name="V13 recursive source manifest"
    )
    review, review_raw = _read_content_bound_evidence_v13(
        root, SOURCE_REVIEW_RELATIVE_PATH, name="V13 independent source review"
    )
    certification, certification_raw = _read_content_bound_evidence_v13(
        root,
        CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
        name="V13 clean-export certification",
    )
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != authority.get("recursive_source_closure_manifest_sha256")
        or hashlib.sha256(review_raw).hexdigest()
        != authority.get("independent_source_review_sha256")
        or hashlib.sha256(certification_raw).hexdigest()
        != authority.get("clean_export_certification_sha256")
    ):
        raise PermissionError("V13 authority does not bind all three source evidences")

    source_bindings = manifest.get("source_bindings")
    manifest_identity = _manifest_identity_v13(manifest, manifest_raw)
    if (
        manifest.get("schema")
        != "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_source_manifest"
        or manifest.get("status") != "SOURCE_ONLY_RECURSIVE_CLOSURE"
        or type(source_bindings) is not list
        or manifest.get("source_count") != len(source_bindings)
        or manifest.get("source_paths")
        != [binding.get("path") for binding in source_bindings]
        or manifest.get("tensor_checkpoint_open_count") != 0
        or manifest.get("sealed_or_heldout_open_count") != 0
        or manifest.get("generated_or_runtime_artifact_open_count") != 0
        or manifest.get("dataset_or_rgb_open_count") != 0
        or manifest.get("whole_tree_export_authorized") is not False
        or hashlib.sha256(_canonical_json_bytes(source_bindings)).hexdigest()
        != manifest.get("source_bindings_sha256")
    ):
        raise PermissionError("V13 recursive source manifest contract changed")

    expected_review_keys = {
        "schema",
        "status",
        "source_only",
        "reviewed_source_commit",
        "manifest",
        "content_sha256",
    }
    reviewed_commit = review.get("reviewed_source_commit")
    if (
        set(review) != expected_review_keys
        or review.get("schema")
        != "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_source_review_v1"
        or review.get("status") != "PASS_SOURCE_ONLY"
        or review.get("source_only") is not True
        or type(reviewed_commit) is not str
        or len(reviewed_commit) != 40
        or any(character not in "0123456789abcdef" for character in reviewed_commit)
        or review.get("manifest") != manifest_identity
    ):
        raise PermissionError("V13 independent source-review contract changed")

    expected_certification_keys = {
        "schema",
        "status",
        "source_only",
        "frozen_source_and_review_commit",
        "source_review",
        "manifest",
        "inventory",
        "validation",
        "content_sha256",
    }
    expected_review_identity = {
        "path": SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": hashlib.sha256(review_raw).hexdigest(),
    }
    inventory = certification.get("inventory")
    validation = certification.get("validation")
    if (
        set(certification) != expected_certification_keys
        or certification.get("schema")
        != (
            "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_"
            "clean_export_certification_v1"
        )
        or certification.get("status") != "PASS_EXACT_NARROW_SOURCE_EXPORT"
        or certification.get("source_only") is not True
        or certification.get("frozen_source_and_review_commit")
        != authority.get("frozen_source_and_review_commit")
        or certification.get("source_review") != expected_review_identity
        or certification.get("manifest") != manifest_identity
        or type(inventory) is not dict
        or set(inventory) != {"bindings", "binding_count", "bindings_sha256"}
        or type(validation) is not dict
        or set(validation)
        != {
            "working_bytes_match_count",
            "binding_mismatch_count",
            "protected_path_match_count",
        }
    ):
        raise PermissionError("V13 clean-export certification contract changed")
    bindings = inventory["bindings"]
    if (
        type(bindings) is not list
        or inventory["binding_count"] != len(bindings)
        or len(bindings) != len({row.get("path") for row in bindings if type(row) is dict})
        or hashlib.sha256(_canonical_json_bytes(bindings)).hexdigest()
        != inventory["bindings_sha256"]
        or validation
        != {
            "working_bytes_match_count": len(bindings),
            "binding_mismatch_count": 0,
            "protected_path_match_count": 0,
        }
    ):
        raise PermissionError("V13 clean-export inventory accounting changed")
    validated_paths = [
        _validate_certified_source_binding_v13(root, binding) for binding in bindings
    ]
    required_export_paths = {
        SOURCE_MANIFEST_RELATIVE_PATH,
        SOURCE_REVIEW_RELATIVE_PATH,
        SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        *manifest.get("source_paths", []),
    }
    if not required_export_paths.issubset(set(validated_paths)):
        raise PermissionError("V13 clean-export inventory omitted required source evidence")

    checker = _load_source_closure_checker_v13(root)
    try:
        if checker.MANIFEST_PATH.resolve() != (root / SOURCE_MANIFEST_RELATIVE_PATH).resolve():
            raise PermissionError("V13 checker resolves a different source manifest")
        checker.verify_manifest(require_tracked=False)
    finally:
        sys.modules.pop(checker.__name__, None)
    return {
        "source_manifest_file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "independent_source_review_file_sha256": hashlib.sha256(review_raw).hexdigest(),
        "clean_export_certification_file_sha256": hashlib.sha256(
            certification_raw
        ).hexdigest(),
        "source_count": len(source_bindings),
        "certified_export_binding_count": len(bindings),
        "certified_source_bindings_sha256": inventory["bindings_sha256"],
        "certified_source_bindings": [dict(binding) for binding in bindings],
        "current_source_closure_revalidated": True,
        "current_export_working_bytes_revalidated": True,
        "scientific_payload_open_count": 0,
    }


def _validate_schedule_v13(
    schedule: Sequence[int],
    *,
    executor_api: Any,
    labels_api: Any,
) -> tuple[int, ...]:
    if isinstance(schedule, (str, bytes)) or len(schedule) != MAXIMUM_PRESENTATIONS:
        raise PermissionError("V13 schedule must contain exactly 16,000 presentations")
    normalized = tuple(schedule)
    if any(type(value) is not int or value < 0 for value in normalized):
        raise PermissionError("V13 schedule indices must be nonnegative exact integers")
    expected = dict(executor_api.CHECKPOINT_SCHEDULE_PREFIX_SHA256)
    observed = {
        100: labels_api.v4.canonical_json_sha256(list(normalized[:1_600])),
        400: labels_api.v4.canonical_json_sha256(list(normalized[:6_400])),
        1_000: labels_api.v4.canonical_json_sha256(list(normalized)),
    }
    if observed != expected:
        raise PermissionError("V13 frozen schedule-prefix identity changed")
    return normalized


def _load_narrow_label_bundle_v13(
    repository_root: Path,
    *,
    v1_executor: Any,
    labels_api: Any,
    runtime_bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, tuple[Mapping[str, Any], ...]],
    Mapping[str, Any],
]:
    """Validate the frozen manifest and open only train plus selection JSONL."""

    label_root = Path(repository_root) / v1_executor.LABEL_ROOT_RELATIVE_PATH
    manifest_path = label_root / v1_executor.LABEL_MANIFEST_NAME
    _require_runtime_binding_v13(
        runtime_bindings["swept_label_manifest"],
        expected_path=manifest_path.relative_to(repository_root).as_posix(),
        expected_sha256=v1_executor.LABEL_MANIFEST_FILE_SHA256,
        expected_byte_count=v1_executor.LABEL_MANIFEST_BYTE_COUNT,
        expected_content_sha256=v1_executor.LABEL_MANIFEST_CONTENT_SHA256,
        name="swept_label_manifest",
    )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PermissionError("swept-progress label manifest is absent")
    manifest_raw = manifest_path.read_bytes()
    if (
        len(manifest_raw) != v1_executor.LABEL_MANIFEST_BYTE_COUNT
        or hashlib.sha256(manifest_raw).hexdigest()
        != v1_executor.LABEL_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen swept-progress label manifest changed")
    manifest = v1_executor._parse_canonical_object(
        manifest_raw,
        name="swept-progress label manifest",
    )
    if (
        manifest.get("schema") != labels_api.MANIFEST_SCHEMA
        or manifest.get("content_sha256")
        != v1_executor.LABEL_MANIFEST_CONTENT_SHA256
        or manifest.get("status") != "complete_model_free_development_labels"
        or manifest.get("roles") != list(labels_api.ROLE_ORDER)
        or manifest.get("role_files") != v1_executor.ROLE_FILES
        or manifest.get("action_order") != list(v1_executor.ACTION_ORDER)
    ):
        raise PermissionError("swept-progress label manifest contract changed")
    records = manifest.get("files")
    if type(records) is not list or len(records) != 3:
        raise PermissionError("label manifest must bind exactly three role JSONLs")
    by_name = {
        record.get("path"): record for record in records if type(record) is dict
    }
    if set(by_name) != set(v1_executor.ROLE_FILES.values()):
        raise PermissionError("label manifest role-file set changed")

    rows_by_role: dict[str, tuple[Mapping[str, Any], ...]] = {}
    opened: list[dict[str, Any]] = []
    for role in ALLOWED_LABEL_ROLES:
        filename = v1_executor.ROLE_FILES[role]
        record = by_name[filename]
        path = label_root / filename
        authority_name = (
            "train_labels" if role == "train" else "checkpoint_selection_labels"
        )
        _require_runtime_binding_v13(
            runtime_bindings[authority_name],
            expected_path=path.relative_to(repository_root).as_posix(),
            expected_sha256=record.get("file_sha256"),
            expected_byte_count=record.get("byte_count"),
            expected_content_sha256=record.get("content_sha256"),
            name=authority_name,
        )
        raw = v1_executor._read_bound_file(path, record)
        rows = v1_executor._parse_canonical_jsonl(raw, name=f"{role} labels")
        if (
            record.get("dataset_role") != role
            or record.get("state_count") != labels_api.v4.ROLE_STATE_COUNTS[role]
            or record.get("action_row_count") != len(rows)
            or len(rows)
            != labels_api.v4.ROLE_STATE_COUNTS[role]
            * len(v1_executor.ACTION_ORDER)
            or record.get("ordered_row_content_sha256")
            != v1_executor._canonical_json_sha256(
                [row.get("content_sha256") for row in rows]
            )
        ):
            raise PermissionError(f"{role} label population or order changed")
        labels_api._state_groups(rows, role=role, frozen=True)
        rows_by_role[role] = rows
        opened.append(
            {
                "role": role,
                "path": path.relative_to(repository_root).as_posix(),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
        )
    if set(rows_by_role) != set(ALLOWED_LABEL_ROLES):
        raise RuntimeError("V13 narrow label role set changed")
    receipt = {
        "manifest": {
            "path": manifest_path.relative_to(repository_root).as_posix(),
            "file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "byte_count": len(manifest_raw),
        },
        "opened_roles": list(ALLOWED_LABEL_ROLES),
        "opened_role_files": opened,
        "probability_calibration_open_count": 0,
    }
    return manifest, rows_by_role, receipt


def _require_runtime_binding_v13(
    binding: Mapping[str, Any],
    *,
    expected_path: str,
    expected_sha256: Any,
    expected_byte_count: Any,
    name: str,
    expected_content_sha256: Any = None,
) -> dict[str, Any]:
    if type(binding) is not dict:
        raise PermissionError(f"V13 authority omitted runtime binding {name}")
    expected_keys = {"path", "file_sha256", "byte_count"}
    if expected_content_sha256 is not None:
        expected_keys.add("content_sha256")
    if (
        set(binding) != expected_keys
        or binding.get("path") != expected_path
        or binding.get("file_sha256") != expected_sha256
        or binding.get("byte_count") != expected_byte_count
        or (
            expected_content_sha256 is not None
            and binding.get("content_sha256") != expected_content_sha256
        )
    ):
        raise PermissionError(f"V13 authority runtime identity changed: {name}")
    return dict(binding)


def _validate_runtime_data_root_v13(
    source_root: Path,
    authority: Mapping[str, Any],
) -> Path:
    value = authority.get("runtime_data_root")
    if not isinstance(value, str) or not value:
        raise PermissionError("V13 authority omitted the canonical runtime data root")
    candidate = Path(value)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise PermissionError("V13 runtime data root must be absolute and non-symlink")
    try:
        resolved = candidate.resolve(strict=True)
        source = Path(source_root).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("V13 runtime data or source root is absent") from error
    if (
        candidate != resolved
        or not candidate.is_dir()
        or resolved.is_relative_to(source)
        or source.is_relative_to(resolved)
        or str(resolved) != value
    ):
        raise PermissionError(
            "V13 runtime data root is noncanonical or not disjoint from source root"
        )
    return resolved


def _runtime_binding_path_v13(
    runtime_data_root: Path,
    binding: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    relative = binding.get("path") if isinstance(binding, Mapping) else None
    if not isinstance(relative, str) or not relative:
        raise PermissionError(f"V13 runtime path is absent: {name}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PermissionError(f"V13 runtime path escaped its root: {name}")
    path = runtime_data_root.joinpath(*pure.parts)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError(f"V13 runtime input is absent: {name}") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(runtime_data_root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError(f"V13 runtime input escaped or is not regular: {name}")
    return resolved


def _require_contained_regular_file_v13(
    containment_root: Path,
    path: Path,
    *,
    name: str,
) -> Path:
    try:
        root = Path(containment_root).resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError(f"V13 runtime file is absent: {name}") from error
    candidate = Path(path)
    if (
        resolved != candidate.absolute()
        or not resolved.is_relative_to(root)
        or candidate.is_symlink()
        or not candidate.is_file()
    ):
        raise PermissionError(
            f"V13 runtime file escaped through a symlink or is not regular: {name}"
        )
    return resolved


def _normalize_endpoint_paths_v13(raw_inputs: Any, runtime_data_root: Path) -> None:
    for endpoint in raw_inputs.endpoints.values():
        image_path = Path(str(endpoint["image_path_metadata_only"]))
        if image_path.is_absolute():
            try:
                image_path = image_path.relative_to(runtime_data_root)
            except ValueError as error:
                raise PermissionError("development RGB path escaped runtime data root") from error
        pure = PurePosixPath(image_path.as_posix())
        folded = tuple(part.casefold() for part in pure.parts)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or any(part == "sealed" or part.startswith("sealed_") for part in folded)
            or any(part == "heldout" or part.startswith("heldout_") for part in folded)
            or any(part == "held_out" or part.startswith("held_out_") for part in folded)
        ):
            raise PermissionError("protected or escaping RGB path entered endpoint index")
        endpoint["image_path_metadata_only"] = pure.as_posix()


def _validate_runtime_payload_containment_v13(
    raw_inputs: Any,
    runtime_data_root: Path,
) -> Mapping[str, Any]:
    """Stat only the train/selection payload paths admitted by the narrow loader."""

    data_root = Path(runtime_data_root).resolve(strict=True)
    raw_root = Path(raw_inputs.root)
    if (
        raw_root.resolve(strict=True) != raw_root.absolute()
        or not raw_root.resolve(strict=True).is_relative_to(data_root)
    ):
        raise PermissionError("V13 Raw payload root escaped through a symlink")
    endpoint_ids = {
        str(pair[f"{side}_endpoint_sha256"])
        for role in ALLOWED_LABEL_ROLES
        for pair in raw_inputs.role_pairs(role)
        for side in ("current", "next")
    }
    image_paths: set[str] = set()
    raw_paths: set[str] = set()
    admitted_arrays = (*CAMERA_ARRAY_FILENAMES.values(), "raster_labels.u1")
    for endpoint_id in sorted(endpoint_ids):
        endpoint = raw_inputs.endpoints.get(endpoint_id)
        if (
            type(endpoint) is not dict
            or endpoint.get("dataset_role") not in ALLOWED_LABEL_ROLES
        ):
            raise PermissionError("V13 contained endpoint crossed its role")
        image_relative = str(endpoint.get("image_path_metadata_only", ""))
        image_pure = PurePosixPath(image_relative)
        shard_pure = PurePosixPath(str(endpoint.get("scene_shard", "")))
        if (
            not image_relative
            or image_pure.is_absolute()
            or ".." in image_pure.parts
            or not shard_pure.parts
            or shard_pure.is_absolute()
            or ".." in shard_pure.parts
        ):
            raise PermissionError("V13 contained payload path is malformed")
        image_paths.add(image_pure.as_posix())
        raw_paths.add(shard_pure.as_posix())
        raw_paths.update(
            (shard_pure.parent / filename).as_posix()
            for filename in admitted_arrays
        )
    for relative in sorted(image_paths):
        _require_contained_regular_file_v13(
            data_root,
            data_root.joinpath(*PurePosixPath(relative).parts),
            name=f"RGB {relative}",
        )
    for relative in sorted(raw_paths):
        if relative not in raw_inputs.inventory:
            raise PermissionError("V13 Raw manifest omitted an admitted payload path")
        _require_contained_regular_file_v13(
            data_root,
            raw_root.joinpath(*PurePosixPath(relative).parts),
            name=f"Raw {relative}",
        )
    paths = [
        *(f"rgb:{path}" for path in sorted(image_paths)),
        *(f"raw:{path}" for path in sorted(raw_paths)),
    ]
    return {
        "validated_role_set": list(ALLOWED_LABEL_ROLES),
        "validated_endpoint_count": len(endpoint_ids),
        "validated_rgb_path_count": len(image_paths),
        "validated_raw_path_count": len(raw_paths),
        "validated_path_count": len(paths),
        "validated_paths_sha256": hashlib.sha256(
            _canonical_json_bytes(paths)
        ).hexdigest(),
        "content_open_count": 0,
    }


def _select_distinct_structural_probe_v13(
    endpoints: Mapping[str, Mapping[str, Any]],
    wrong_mapping: Mapping[str, str],
) -> tuple[str, str]:
    """Choose the first registered wrong-RGB pair with distinct bound bytes."""

    for endpoint_id in sorted(wrong_mapping):
        wrong_id = wrong_mapping[endpoint_id]
        endpoint = endpoints.get(endpoint_id)
        wrong = endpoints.get(wrong_id)
        if (
            type(endpoint) is not dict
            or type(wrong) is not dict
            or endpoint.get("dataset_role") != "checkpoint_selection"
            or wrong.get("dataset_role") != "checkpoint_selection"
        ):
            raise PermissionError("V13 structural probe crossed its selection role")
        endpoint_sha = endpoint.get("image_sha256_commitment_only")
        wrong_sha = wrong.get("image_sha256_commitment_only")
        if (
            isinstance(endpoint_sha, str)
            and len(endpoint_sha) == 64
            and isinstance(wrong_sha, str)
            and len(wrong_sha) == 64
            and endpoint_sha != wrong_sha
        ):
            return endpoint_id, wrong_id
    raise PermissionError("V13 selection mapping has no distinct bound RGB probe")


def _stack_camera_rows_v13(
    raw_inputs: Any,
    endpoint_ids: Sequence[str],
    *,
    role: str,
    arm: str,
    stage: str,
    torch: Any,
) -> dict[str, Any]:
    parts: dict[str, list[Any]] = {name: [] for name in CAMERA_ARRAY_FILENAMES}
    for endpoint_id in endpoint_ids:
        endpoint = raw_inputs.endpoints.get(str(endpoint_id))
        if type(endpoint) is not dict or endpoint.get("dataset_role") != role:
            raise PermissionError("V13 Camera row crossed its dataset role")
        shard = raw_inputs._shard(endpoint, arm=arm, stage=stage)
        for name, filename in CAMERA_ARRAY_FILENAMES.items():
            parts[name].append(
                raw_inputs._row_array(
                    endpoint,
                    shard,
                    filename,
                    arm=arm,
                    stage=stage,
                )
            )
    result = {name: torch.stack(values) for name, values in parts.items()}
    for name in ("camera_origin", "camera_basis", "ground", "pixel_distance"):
        result[name] = result[name].float()
    for name in ("pixel_hit", "ground_in_frustum", "ground_clear"):
        result[name] = result[name].bool()
    return result


def _build_one_microbatch_v13(
    *,
    runtime: "V13ComposedRuntime",
    indices: Sequence[int],
    stage: str,
) -> Mapping[str, Any]:
    base = runtime.v1_training.build_microbatch_v1(
        runtime.loader,
        runtime.pairs["train"],
        runtime.labels["train"],
        indices,
        runtime.device,
        stage=stage,
    )
    selected = [runtime.pairs["train"][int(index)] for index in indices]
    current = _stack_camera_rows_v13(
        runtime.raw_inputs,
        [str(pair["current_endpoint_sha256"]) for pair in selected],
        role="train",
        arm="camera_evidence_bottleneck_v13",
        stage=stage,
        torch=runtime.torch,
    )
    next_ = _stack_camera_rows_v13(
        runtime.raw_inputs,
        [str(pair["next_endpoint_sha256"]) for pair in selected],
        role="train",
        arm="camera_evidence_bottleneck_v13",
        stage=stage,
        torch=runtime.torch,
    )
    additions = {
        runtime.training_module.CURRENT_CAMERA_ORIGIN_KEY: current["camera_origin"],
        runtime.training_module.NEXT_CAMERA_ORIGIN_KEY: next_["camera_origin"],
        runtime.training_module.CURRENT_CAMERA_BASIS_KEY: current["camera_basis"],
        runtime.training_module.NEXT_CAMERA_BASIS_KEY: next_["camera_basis"],
        runtime.training_module.CURRENT_GROUND_PLANE_Z_KEY: current["ground"],
        runtime.training_module.NEXT_GROUND_PLANE_Z_KEY: next_["ground"],
        runtime.training_module.CURRENT_PIXEL_HIT_KEY: current["pixel_hit"],
        runtime.training_module.NEXT_PIXEL_HIT_KEY: next_["pixel_hit"],
        runtime.training_module.CURRENT_PIXEL_DISTANCE_KEY: current["pixel_distance"],
        runtime.training_module.NEXT_PIXEL_DISTANCE_KEY: next_["pixel_distance"],
        runtime.training_module.CURRENT_GROUND_IN_FRUSTUM_KEY: current[
            "ground_in_frustum"
        ],
        runtime.training_module.NEXT_GROUND_IN_FRUSTUM_KEY: next_[
            "ground_in_frustum"
        ],
        runtime.training_module.CURRENT_GROUND_CLEAR_KEY: current["ground_clear"],
        runtime.training_module.NEXT_GROUND_CLEAR_KEY: next_["ground_clear"],
    }
    if set(base) & set(additions):
        raise RuntimeError("V13 fine arrays overlap the V1 base batch")
    result = {**base, **additions}
    if tuple(result) != tuple(runtime.training_module.REQUIRED_BATCH_KEYS):
        raise RuntimeError("V13 composed microbatch key order or membership changed")
    return result


class _AccumulatorFanout:
    def __init__(self, *accumulators: Any) -> None:
        self.accumulators = accumulators

    def update(self, **kwargs: Any) -> None:
        for accumulator in self.accumulators:
            accumulator.update(**kwargs)


def _summarize_physical_provenance_v13(
    rows: Sequence[Mapping[str, Any]],
    scopes: Mapping[str, Mapping[str, Any]],
    *,
    registered_families: Sequence[str],
) -> dict[str, Any]:
    required_row_keys = {
        "family",
        "arm",
        "model_entrypoint",
        "learned_evidence_source",
        "semantic_probability_source",
        "target_metadata_only",
        "auxiliary_logits_used",
        "old_camera_raster_used",
        "batch_size",
    }
    families = tuple(registered_families)
    if (
        not families
        or len(families) != len(set(families))
        or any(not isinstance(family, str) or not family for family in families)
    ):
        raise PermissionError("V13 registered physical family set changed")
    matched_count = 0
    wrong_count = 0
    observed_families: set[str] = set()
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != required_row_keys
            or row.get("family") not in families
            or row.get("arm") not in {"matched", "wrong_rgb"}
            or row.get("model_entrypoint") != "encode_online_with_evidence"
            or row.get("learned_evidence_source") != "encoding.nominal_evidence"
            or row.get("semantic_probability_source")
            != "softmax(model.semantic_logits_from_latent(encoding.latent),dim=1)"
            or tuple(row.get("target_metadata_only", ()))
            != ("ground_query_in_frustum", "ground_target_distance_m")
            or row.get("auxiliary_logits_used") is not False
            or row.get("old_camera_raster_used") is not False
            or type(row.get("batch_size")) is not int
            or not 1 <= row["batch_size"] <= 4
        ):
            raise PermissionError("V13 qualifying physical-updater provenance changed")
        observed_families.add(row["family"])
        if row["arm"] == "matched":
            matched_count += row["batch_size"]
        else:
            wrong_count += row["batch_size"]
    if observed_families != set(families):
        raise PermissionError("V13 qualifying physical provenance omitted a family")
    aggregate = scopes.get("aggregate")
    wrong_metric_names = (
        "wrong_rgb_pixel_balanced_accuracy_drop",
        "wrong_rgb_depth_median_error_increase_m",
        "wrong_rgb_depth_p95_error_increase_m",
        "wrong_rgb_ground_balanced_accuracy_drop",
        "wrong_rgb_raster_nll_increase",
        "wrong_rgb_raster_balanced_accuracy_drop",
    )
    wrong_dependence = isinstance(aggregate, Mapping) and any(
        isinstance(aggregate.get(name), (int, float))
        and not isinstance(aggregate.get(name), bool)
        and float(aggregate[name]) != 0.0
        for name in wrong_metric_names
    )
    return {
        "target_endpoint_count": 924,
        "matched_nominal_call_count": matched_count,
        "wrong_nominal_call_count": wrong_count,
        "qualifying_updater_call_count": matched_count + wrong_count,
        "qualifying_updater_name": "update_physical_accumulator_from_rgb_v13",
        "auxiliary_logits_used": False,
        "old_camera_raster_used": False,
        "target_query_identity_pass": True,
        "wrong_rgb_dependence_nonzero": bool(wrong_dependence),
    }


def _physical_scopes_v13(
    runtime: "V13ComposedRuntime",
    model: Any,
    *,
    update: int,
    physical_endpoint_updater: Callable[..., Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    pairs = runtime.pairs["checkpoint_selection"]
    ids_by_family = {
        family: sorted(
            {
                str(pair[f"{side}_endpoint_sha256"])
                for pair in pairs
                if pair["family"] == family
                for side in ("current", "next")
            }
        )
        for family in runtime.executor_api.REGISTERED_FAMILIES
    }
    endpoint_rows = [
        {"endpoint_sha256": endpoint, "family": family}
        for family in runtime.executor_api.REGISTERED_FAMILIES
        for endpoint in ids_by_family[family]
    ]
    wrong_mapping = runtime.executor_api.registered_wrong_rgb_mapping_v13(endpoint_rows)
    if len(wrong_mapping) != 924:
        raise RuntimeError("V13 physical evaluator did not bind exactly 924 endpoints")
    correct = {
        scope: runtime.minimal_runtime.MetricAccumulator()
        for scope in runtime.executor_api.SCOPES
    }
    wrong = {
        scope: runtime.minimal_runtime.MetricAccumulator()
        for scope in runtime.executor_api.SCOPES
    }
    provenance_rows: list[Mapping[str, Any]] = []
    with runtime.torch.no_grad():
        for family in runtime.executor_api.REGISTERED_FAMILIES:
            ids = ids_by_family[family]
            for start in range(0, len(ids), 4):
                target_ids = ids[start : start + 4]
                wrong_ids = [wrong_mapping[value] for value in target_ids]
                stage = f"physical_update_{update}_{family}_{start}"
                target_rgb = runtime.torch.stack(
                    [
                        runtime.loader.image(
                            value,
                            role="checkpoint_selection",
                            stage=stage,
                            kind="endpoint",
                        )
                        for value in target_ids
                    ]
                ).to(runtime.device)
                wrong_rgb = runtime.torch.stack(
                    [
                        runtime.loader.image(
                            value,
                            role="checkpoint_selection",
                            stage=f"{stage}_wrong_rgb",
                            kind="endpoint",
                        )
                        for value in wrong_ids
                    ]
                ).to(runtime.device)
                arrays = _stack_camera_rows_v13(
                    runtime.raw_inputs,
                    target_ids,
                    role="checkpoint_selection",
                    arm="physical_target_context_v13",
                    stage=stage,
                    torch=runtime.torch,
                )
                labels = runtime.torch.stack(
                    [
                        runtime.loader.raster_label(
                            value,
                            role="checkpoint_selection",
                            stage=stage,
                            scope="endpoint_observation",
                        )
                        for value in target_ids
                    ]
                ).long().to(runtime.device)
                targets = runtime.minimal_runtime.derive_targets(
                    pixel_hit_mask=arrays["pixel_hit"].to(runtime.device),
                    pixel_first_hit_distance_m=arrays["pixel_distance"].to(
                        runtime.device
                    ),
                    ground_support_in_frustum=arrays["ground_in_frustum"].to(
                        runtime.device
                    ),
                    ground_support_clear_to_target=arrays["ground_clear"].to(
                        runtime.device
                    ),
                )
                context = {
                    "target_camera_origin_body_m": arrays["camera_origin"].to(
                        runtime.device
                    ),
                    "target_camera_basis_body_fru": arrays["camera_basis"].to(
                        runtime.device
                    ),
                    "target_ground_plane_z_body_m": arrays["ground"].to(
                        runtime.device
                    ),
                    "targets": targets,
                    "target_raster_labels": labels,
                    "families": [family] * len(target_ids),
                }
                for arm_name, rgb, accumulator_set in (
                    ("matched", target_rgb, correct),
                    ("wrong_rgb", wrong_rgb, wrong),
                ):
                    receipt = physical_endpoint_updater(
                        model,
                        _AccumulatorFanout(
                            accumulator_set["aggregate"], accumulator_set[family]
                        ),
                        selected_rgb=rgb,
                        **context,
                    )
                    if not isinstance(receipt, Mapping):
                        raise RuntimeError("V13 physical updater omitted provenance")
                    provenance_rows.append(
                        {"family": family, "arm": arm_name, **dict(receipt)}
                    )
    if len(provenance_rows) != 2 * sum(
        (len(values) + 3) // 4 for values in ids_by_family.values()
    ):
        raise RuntimeError("V13 physical updater call accounting changed")
    finalized_correct = {scope: correct[scope].finalize() for scope in correct}
    finalized_wrong = {scope: wrong[scope].finalize() for scope in wrong}
    if (
        finalized_correct["aggregate"].get("frame_count") != 924
        or finalized_wrong["aggregate"].get("frame_count") != 924
    ):
        raise RuntimeError("V13 physical arms did not each score exactly 924 frames")
    scopes = {
        scope: runtime.executor_api.flatten_physical_metrics_v13(
            finalized_correct[scope], finalized_wrong[scope]
        )
        for scope in runtime.executor_api.SCOPES
    }
    runtime.physical_provenance[update] = _summarize_physical_provenance_v13(
        provenance_rows,
        scopes,
        registered_families=runtime.executor_api.REGISTERED_FAMILIES,
    )
    return scopes


def _v12_observation_v13(
    runtime: "V13ComposedRuntime",
    model: Any,
    *,
    update: int,
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, bool]] | None]:
    labels = runtime.labels["checkpoint_selection"]
    scored = runtime.v1_executor.score_role_v1(
        model,
        runtime.loader,
        runtime.pairs["checkpoint_selection"],
        labels,
        runtime.action_prior_m,
        runtime.device,
        torch=runtime.torch,
        np=runtime.np,
        training_core=runtime.v1_training,
        current_frame_persistence_masks=runtime.persistence_masks,
        metrics_api=runtime.metrics_api,
    )
    informative = runtime.np.asarray(
        [group[0]["informative_state"] for group in labels.state_groups],
        dtype=runtime.np.bool_,
    )
    role_metrics = {
        arm: runtime.v1_executor.scientific_metrics_v1(
            scored["scores_m"][arm],
            labels.prefix_lengths,
            informative,
            labels.scene_ids,
            labels.family_ids,
            np=runtime.np,
        )
        for arm in runtime.v1_executor.ALL_ARM_NAMES
    }
    semantic = runtime.v1_executor.semantic_metrics_v1(
        scored["semantic_confusion"],
        scored["rough_semantic_confusion"],
        np=runtime.np,
    )
    comparisons = {
        name: runtime.v1_executor.paired_control_comparison_v1(
            scored["scores_m"]["full"],
            scored["scores_m"][name],
            labels.prefix_lengths,
            informative,
            labels.scene_ids,
            labels.family_ids,
            np=runtime.np,
        )
        for name in runtime.v1_executor.CONTROL_NAMES
    }
    gate = runtime.v1_executor.evaluate_gate_v1(role_metrics, semantic, comparisons)
    if tuple(gate.get("checks", {})) != runtime.executor_api.V12_GATE_CHECK_NAMES:
        raise RuntimeError("V13 inherited V12 24-check order changed")
    controls = None
    if update == 400:
        controls = {
            name: {
                "positive_equal_scene_delta": row["equal_scene_mean_delta"] > 0.0,
                "positive_bootstrap_lower_95": row["bootstrap_lower_95"] > 0.0,
                "positive_family_count": row["positive_family_count"]
                >= runtime.v1_executor.GATE_THRESHOLDS[
                    "positive_control_family_count_min"
                ],
            }
            for name, row in comparisons.items()
        }
    return gate, controls


@dataclass
class V13ComposedRuntime:
    repository_root: Path
    runtime_data_root: Path
    output_root: Path
    authority: Mapping[str, Any]
    reservation: Mapping[str, Any]
    source_evidence: Mapping[str, Any]
    torch: Any
    np: Any
    device: Any
    minimal_runtime: Any
    executor_api: Any
    model_module: Any
    training_module: Any
    v1_executor: Any
    v1_training: Any
    labels_api: Any
    metrics_api: Any
    survival_scoring: Any
    raw_inputs: Any
    loader: Any
    n320_fit: Any
    n320_gate: Mapping[str, Any]
    n320_checkpoint: Mapping[str, Any]
    schedule: tuple[int, ...]
    train_pair_count: int
    labels: Mapping[str, Any]
    pairs: Mapping[str, Sequence[Mapping[str, Any]]]
    sweep_masks: Any
    persistence_masks: Any
    action_prior_m: Any
    label_access: Mapping[str, Any]
    runtime_path_containment: Mapping[str, Any]
    hardware: Mapping[str, Any]
    runtime_fingerprint: Mapping[str, Any]
    determinism: Mapping[str, Any]
    physical_provenance: dict[int, Mapping[str, Any]]
    matched_module: Any
    matched_source_root: Path
    _initialized: bool = False
    _closed: bool = False
    _access_consumed_count: int = -1
    _access_opened_roles: tuple[str, ...] = ()
    _terminal_rehash_started: bool = False
    _terminal_access_receipt: Mapping[str, Any] | None = None
    _structural_probe_cache: Mapping[str, Any] | None = None

    def initialize_model_v13(self) -> tuple[Any, Any, Mapping[str, Any]]:
        if self._initialized:
            raise RuntimeError("V13 model initialization is one-shot")
        self._initialized = True
        rng_before = self.torch.random.get_rng_state().clone()
        cuda_rng_before = [value.clone() for value in self.torch.cuda.get_rng_state_all()]
        model = self.model_module.GeometryAnchoredSweptProgressSurvivalJointJepaV13(
            self.n320_fit,
            self.sweep_masks,
        )
        rng_after = self.torch.random.get_rng_state().clone()
        cuda_rng_after = self.torch.cuda.get_rng_state_all()
        if not self.torch.equal(rng_before, rng_after) or any(
            not self.torch.equal(before, after)
            for before, after in zip(cuda_rng_before, cuda_rng_after, strict=True)
        ):
            raise RuntimeError("V13 constructor or projection initialization changed caller RNG")
        if (
            model.config.initialization_seed != 20_260_712
            or self.model_module.PROJECTION_INITIALIZATION_SEED_V13 != 20_260_729
            or int(model.target_hard_sync_count.item()) != 1
            or int(model.ema_update_count.item()) != 0
        ):
            raise RuntimeError("V13 initial target synchronization accounting changed")
        model = model.to(self.device)
        model.train()
        partition = self.training_module.partition_parameters_v13(model)
        optimizer = self.training_module.build_frozen_optimizer_v13(partition)
        receipt = {
            "n320_gate_open_count": 1,
            "n320_checkpoint_open_count": 1,
            "n320_gate_passed": self.n320_gate.get("passes") is True,
            "payload_access_after_reservation": True,
            "probability_calibration_open_count": 0,
            "constructor_initialization_seed": 20_260_712,
            "projection_initialization_seed": 20_260_729,
            "constructor_rng_preserved": True,
            "projection_rng_preserved": True,
            "target_hard_sync_count": 1,
            "ema_update_count": 0,
            "online_trainable_parameter_count": sum(
                parameter.numel() for parameter in partition.online
            ),
            "hardware": dict(self.hardware),
            "runtime_fingerprint": dict(self.runtime_fingerprint),
            "determinism": dict(self.determinism),
            "runtime_path_containment": dict(self.runtime_path_containment),
        }
        if not receipt["n320_gate_passed"]:
            raise PermissionError("accepted N320 gate did not pass")
        return model, optimizer, receipt

    def build_microbatches_v13(
        self,
        indices: Sequence[int],
        *,
        update: int,
    ) -> tuple[Mapping[str, Any], ...]:
        if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATES:
            raise PermissionError("V13 update escaped the 1..1000 cap")
        values = tuple(indices)
        expected = self.schedule[
            (update - 1) * PRESENTATIONS_PER_UPDATE : update
            * PRESENTATIONS_PER_UPDATE
        ]
        if values != expected or len(values) != PRESENTATIONS_PER_UPDATE:
            raise PermissionError("V13 controller supplied a non-frozen update schedule")
        return tuple(
            _build_one_microbatch_v13(
                runtime=self,
                indices=values[start : start + 4],
                stage=f"train_update_{update:04d}_microbatch_{start // 4}",
            )
            for start in range(0, PRESENTATIONS_PER_UPDATE, 4)
        )

    def observe_v13(
        self,
        model: Any,
        *,
        update: int,
        physical_endpoint_updater: Callable[..., Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if update not in OBSERVATION_UPDATES:
            raise PermissionError("V13 observation update is not preregistered")
        physical = _physical_scopes_v13(
            self,
            model,
            update=update,
            physical_endpoint_updater=physical_endpoint_updater,
        )
        v12_gate, controls = _v12_observation_v13(self, model, update=update)
        return {
            "physical_scopes": physical,
            "v12_gate": v12_gate,
            "controls": controls,
            "physical_provenance": dict(self.physical_provenance[update]),
        }

    def structural_probe_inputs_v13(self) -> dict[str, Any]:
        if self._structural_probe_cache is None:
            pairs = self.pairs["checkpoint_selection"]
            endpoint_rows = [
                {"endpoint_sha256": endpoint, "family": family}
                for family in self.executor_api.REGISTERED_FAMILIES
                for endpoint in sorted(
                    {
                        str(pair[f"{side}_endpoint_sha256"])
                        for pair in pairs
                        if pair["family"] == family
                        for side in ("current", "next")
                    }
                )
            ]
            wrong = self.executor_api.registered_wrong_rgb_mapping_v13(endpoint_rows)
            if len(endpoint_rows) != 924 or len(wrong) != 924:
                raise RuntimeError("V13 structural probe endpoint registry changed")
            endpoint_id, wrong_id = _select_distinct_structural_probe_v13(
                self.raw_inputs.endpoints,
                wrong,
            )
            stage = "initial_structural_probe"
            rgb = self.torch.stack(
                [
                    self.loader.image(
                        endpoint_id,
                        role="checkpoint_selection",
                        stage=stage,
                        kind="endpoint",
                    )
                ]
            ).to(self.device)
            wrong_rgb = self.torch.stack(
                [
                    self.loader.image(
                        wrong_id,
                        role="checkpoint_selection",
                        stage=f"{stage}_wrong_rgb",
                        kind="endpoint",
                    )
                ]
            ).to(self.device)
            arrays = _stack_camera_rows_v13(
                self.raw_inputs,
                [endpoint_id],
                role="checkpoint_selection",
                arm="initial_structural_probe_v13",
                stage=stage,
                torch=self.torch,
            )
            origin_a = arrays["camera_origin"].to(self.device)
            origin_b = origin_a.clone()
            origin_b[:, 0].add_(0.125)
            self._structural_probe_cache = {
                "rgb": rgb,
                "wrong_rgb": wrong_rgb,
                "camera_origin_a": origin_a,
                "camera_origin_b": origin_b,
                "camera_basis": arrays["camera_basis"].to(self.device),
                "ground_plane_z": arrays["ground"].to(self.device),
            }
        return dict(self._structural_probe_cache)

    @staticmethod
    def _sum_integer_leaves_v13(value: Any) -> int:
        if type(value) is int:
            return value
        if isinstance(value, Mapping):
            return sum(
                V13ComposedRuntime._sum_integer_leaves_v13(item)
                for item in value.values()
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return sum(
                V13ComposedRuntime._sum_integer_leaves_v13(item) for item in value
            )
        return 0

    def access_receipt_v13(self) -> Mapping[str, Any]:
        """Return cheap in-memory counters; no source or payload is reopened."""

        consumed_count = len(self.raw_inputs.consumed)
        if consumed_count != self._access_consumed_count:
            discovered = {
                role
                for record in self.raw_inputs.consumed.values()
                for role in record.get("roles", [])
            }
            discovered.update(self.label_access.get("opened_roles", ()))
            role_order = (
                "authority",
                "index",
                "train",
                "checkpoint_selection",
                FORBIDDEN_LABEL_ROLE,
            )
            self._access_opened_roles = tuple(
                role for role in role_order if role in discovered
            ) + tuple(sorted(discovered - set(role_order)))
            self._access_consumed_count = consumed_count
        allowed = {"authority", "index", *ALLOWED_LABEL_ROLES}
        forbidden_roles = set(self._access_opened_roles) - allowed
        loader_receipt = self.loader.receipt()
        forbidden_loader_count = self._sum_integer_leaves_v13(
            loader_receipt.get("forbidden_semantic_counters", {})
        )
        calibration_count = int(FORBIDDEN_LABEL_ROLE in self._access_opened_roles)
        forbidden_count = len(forbidden_roles) + forbidden_loader_count
        if forbidden_count or calibration_count:
            raise PermissionError("V13 observed a forbidden runtime input or role")
        return {
            "forbidden_input_count": 0,
            "probability_calibration_open_count": 0,
            "opened_roles": list(self._access_opened_roles),
        }

    def terminal_access_receipt_v13(self) -> Mapping[str, Any]:
        """Perform and cache the single complete terminal input/source rehash."""

        if self._terminal_access_receipt is not None:
            return dict(self._terminal_access_receipt)
        if self._terminal_rehash_started:
            raise RuntimeError("V13 terminal full rehash is one-shot")
        self._terminal_rehash_started = True
        base = dict(self.access_receipt_v13())
        raw = self.raw_inputs.rehash_consumed()
        source_validation = self.executor_api.validate_bound_sources_v13(
            self.repository_root
        )
        certified_bindings = self.source_evidence.get("certified_source_bindings")
        if not isinstance(certified_bindings, list) or not certified_bindings:
            raise PermissionError("V13 certified source inventory is absent at terminal")
        certified_paths = [
            _validate_certified_source_binding_v13(self.repository_root, binding)
            for binding in certified_bindings
        ]
        certified_bindings_sha256 = hashlib.sha256(
            _canonical_json_bytes(certified_bindings)
        ).hexdigest()
        if (
            len(certified_paths) != len(set(certified_paths))
            or certified_bindings_sha256
            != self.source_evidence.get("certified_source_bindings_sha256")
            or len(certified_bindings)
            != self.source_evidence.get("certified_export_binding_count")
        ):
            raise PermissionError("V13 certified source inventory changed at terminal")
        label_rehash: list[str] = []
        for record in (
            self.label_access["manifest"],
            *self.label_access["opened_role_files"],
        ):
            path = self.runtime_data_root / record["path"]
            if path.is_symlink() or not path.is_file():
                raise PermissionError("V13 consumed label source is not regular")
            payload = path.read_bytes()
            if (
                len(payload) != record["byte_count"]
                or hashlib.sha256(payload).hexdigest() != record["file_sha256"]
            ):
                raise PermissionError("V13 consumed label source changed")
            label_rehash.append(record["path"])
        raw_count = raw.get("unique_file_count")
        raw_records_sha256 = raw.get("records_sha256")
        if (
            type(raw_count) is not int
            or raw_count <= 0
            or type(raw_records_sha256) is not str
            or len(raw_records_sha256) != 64
            or len(label_rehash) != 3
        ):
            raise RuntimeError("V13 terminal rehash accounting changed")
        receipt = {
            **base,
            "terminal_full_rehash_count": 1,
            "raw_consumed_inputs_rehashed": True,
            "raw_consumed_file_rehash_count": raw_count,
            "raw_consumed_records_sha256": raw_records_sha256,
            "raw_inputs": raw,
            "label_source_rehash_count": len(label_rehash),
            "label_sources_rehashed": label_rehash,
            "bound_parent_source_rehash_count": source_validation[
                "validated_path_count"
            ],
            "bound_parent_sources": source_validation,
            "certified_source_rehash_count": len(certified_bindings),
            "certified_source_bindings_sha256": certified_bindings_sha256,
            "certified_source_bindings": [dict(binding) for binding in certified_bindings],
            "all_consumed_inputs_rehashed": True,
            "source_root": str(self.repository_root),
            "runtime_data_root": str(self.runtime_data_root),
            "runtime_fingerprint": dict(self.runtime_fingerprint),
        }
        self._terminal_access_receipt = receipt
        return dict(receipt)

    def close_v13(self) -> None:
        if self._closed:
            return
        try:
            if not self._terminal_rehash_started:
                self.terminal_access_receipt_v13()
        finally:
            self._closed = True
            self.raw_inputs.array_cache.clear()
            self.raw_inputs.shard_cache.clear()
            self.raw_inputs.frame_cache.clear()
            self.loader.image_cache.clear()
            self.loader.label_cache.clear()
            self.matched_module.ROOT = self.matched_source_root


def _runtime_fingerprint_v13(torch: Any, np: Any, pillow: Any) -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": str(torch.__version__),
        "torch_hip": str(torch.version.hip),
        "numpy": str(np.__version__),
        "pillow": str(pillow.__version__),
    }


def _hardware_and_determinism_v13(
    torch: Any,
    np: Any,
    pillow: Any,
    expected_hardware: Mapping[str, Any],
    expected_runtime: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    runtime_fingerprint = _runtime_fingerprint_v13(torch, np, pillow)
    if runtime_fingerprint != expected_runtime:
        raise RuntimeError("actual Python/Torch/ROCm/NumPy/Pillow stack changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU is required")
    properties = torch.cuda.get_device_properties(0)
    authority_hardware = {
        "visible_device_count": int(torch.cuda.device_count()),
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(properties.total_memory),
        "isolated_python": bool(sys.flags.isolated),
    }
    if authority_hardware != expected_hardware:
        raise RuntimeError("actual hardware/interpreter differs from V13 authority")
    hardware = {
        **authority_hardware,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if (
        hardware["name"] != REQUIRED_GPU_NAME
        or hardware["total_memory_bytes"] != REQUIRED_GPU_MEMORY_BYTES
    ):
        raise RuntimeError("the exact reviewed R9700 runtime is required")
    torch.manual_seed(EXPERIMENT_SEED)
    torch.cuda.manual_seed_all(EXPERIMENT_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    determinism = {
        "experiment_seed": EXPERIMENT_SEED,
        "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "configured_after_reservation": True,
    }
    return torch.device("cuda:0"), hardware, determinism, runtime_fingerprint


def compose_runtime_v13(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
    reservation: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> V13ComposedRuntime:
    """Import and compose the narrow runtime only after reservation."""

    repository = Path(repository_root).absolute()
    if repository.resolve() != ROOT.resolve():
        raise PermissionError("V13 launcher accepts only its certified export root")
    executor_api = importlib.import_module(
        "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck"
    )
    executor_api.validate_content_bound_v13(reservation)
    executor_api.validate_future_execution_prerequisites_v13(authority)
    runtime_data_root = _validate_runtime_data_root_v13(repository, authority)
    output_root = repository / executor_api.OUTPUT_ROOT_RELATIVE_PATH
    reservation_path = output_root / "reservation.json"
    if (
        not output_root.is_dir()
        or output_root.is_symlink()
        or not reservation_path.is_file()
        or reservation_path.is_symlink()
    ):
        raise PermissionError("V13 reservation must exist before runtime composition")
    runtime_bindings = authority["runtime_inputs"]
    for name in executor_api.RUNTIME_INPUT_BINDING_NAMES:
        _runtime_binding_path_v13(
            runtime_data_root,
            runtime_bindings[name],
            name=name,
        )

    np = importlib.import_module("numpy")
    pillow = importlib.import_module("PIL")
    Image = importlib.import_module("PIL.Image")
    torch = importlib.import_module("torch")
    fit_api = importlib.import_module("lewm.models.observable_camera_ray_evidence_v4")
    targets_api = importlib.import_module(
        "lewm.models.observable_camera_ray_evidence_v4_training"
    )
    physical_api = importlib.import_module(
        "lewm.benchmarks.go2_observable_camera_ray_fit_v4_metrics"
    )
    v1_executor = importlib.import_module(
        "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v1"
    )
    v1_training = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
    )
    matched = importlib.import_module(
        "scripts.run_go2_shared_jepa_v5_matched_training_v1"
    )
    direct = importlib.import_module(
        "scripts.run_go2_direct_egocentric_bev_state_jepa_v1"
    )
    direct_contract = importlib.import_module(
        "lewm.benchmarks.go2_direct_egocentric_bev_state_jepa_v1"
    )
    labels_api = importlib.import_module(
        "lewm.benchmarks.go2_swept_progress_survival_labels_v1"
    )
    model_module = importlib.import_module(
        "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck"
    )
    training_module = importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck"
    )
    survival_scoring = importlib.import_module(
        "lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1"
    )
    metrics_api = importlib.import_module(
        "lewm.benchmarks.go2_post_action_projective_support_metrics_v1"
    )
    minimal_runtime = SimpleNamespace(
        np=np,
        Image=Image,
        torch=torch,
        model_module=SimpleNamespace(
            NORMALIZATION_MEAN=NORMALIZATION_MEAN,
            NORMALIZATION_STD=NORMALIZATION_STD,
        ),
        FitModel=fit_api.ObservableCameraRayEvidenceV4Model,
        MetricAccumulator=physical_api.ObservableCameraRayFitV4MetricAccumulator,
        derive_targets=targets_api.derive_observable_camera_ray_evidence_v4_targets,
    )
    if hasattr(minimal_runtime, "soft_rasterize") or hasattr(
        minimal_runtime, "loss_adapter"
    ):
        raise RuntimeError("V13 minimal runtime admitted an old Camera capability")

    manifest, rows_by_role, label_access = _load_narrow_label_bundle_v13(
        runtime_data_root,
        v1_executor=v1_executor,
        labels_api=labels_api,
        runtime_bindings=runtime_bindings,
    )
    input_bindings = v1_executor._exact_runtime_bindings_v1(
        manifest,
        direct_contract=direct_contract,
        labels_api=labels_api,
    )
    frozen_bindings = {
        "raw_manifest": input_bindings["raw_manifest"],
        "raw_audit": input_bindings["raw_audit"],
        "schedule": input_bindings["schedule"],
        "n320_gate": direct_contract.RUNTIME_BINDINGS[
            direct_contract.N320_GATE_RELATIVE_PATH
        ],
        "n320_checkpoint": direct_contract.RUNTIME_BINDINGS[
            direct_contract.N320_CHECKPOINT_RELATIVE_PATH
        ],
    }
    for name, frozen in frozen_bindings.items():
        _require_runtime_binding_v13(
            runtime_bindings[name],
            expected_path=str(frozen["path"]),
            expected_sha256=frozen["file_sha256"],
            expected_byte_count=frozen["byte_count"],
            expected_content_sha256=frozen.get("content_sha256"),
            name=name,
        )
    for name in ("raw_pairs", "raw_endpoints"):
        _runtime_binding_path_v13(
            runtime_data_root,
            input_bindings[name],
            name=name,
        )
    authorization = {
        "raw": {
            "manifest": dict(runtime_bindings["raw_manifest"]),
            "audit": dict(runtime_bindings["raw_audit"]),
        },
        "camera": {
            "gate": dict(runtime_bindings["n320_gate"]),
            "checkpoint": dict(runtime_bindings["n320_checkpoint"]),
        },
    }
    raw_indexes = labels_api.v4.load_and_validate_raw_indexes(
        runtime_data_root / runtime_bindings["raw_manifest"]["path"],
        runtime_data_root / input_bindings["raw_pairs"]["path"],
        runtime_data_root / input_bindings["raw_endpoints"]["path"],
    )
    labels_api.v4.validate_raw_audit_v1(
        runtime_data_root / runtime_bindings["raw_audit"]["path"]
    )
    schedule = labels_api.v4.load_schedule_indices_v1(
        runtime_data_root / runtime_bindings["schedule"]["path"],
        raw_indexes=raw_indexes,
    )
    schedule = _validate_schedule_v13(
        schedule,
        executor_api=executor_api,
        labels_api=labels_api,
    )
    matched_source_root = Path(matched.ROOT).resolve()
    if matched_source_root != repository.resolve():
        raise PermissionError("reviewed Raw/N320 loader was not imported from source root")
    matched.ROOT = runtime_data_root
    raw_inputs = matched.RawInputs(minimal_runtime, authorization)
    _normalize_endpoint_paths_v13(raw_inputs, runtime_data_root)
    runtime_path_containment = _validate_runtime_payload_containment_v13(
        raw_inputs,
        runtime_data_root,
    )
    progress: dict[str, Any] = {}
    loader = direct.DirectBevNarrowLoader(
        minimal_runtime,
        raw_inputs,
        progress=progress,
    )
    n320_fit, n320_gate, n320_checkpoint = matched._camera_model_after_reservation(
        minimal_runtime,
        authorization,
    )
    device, hardware, determinism, runtime_fingerprint = _hardware_and_determinism_v13(
        torch,
        np,
        pillow,
        authority["hardware"],
        authority["runtime"],
    )

    labels = {
        role: v1_training.freeze_role_labels_v1(rows, role=role, np=np)
        for role, rows in rows_by_role.items()
    }
    pairs = {role: raw_inputs.role_pairs(role) for role in ALLOWED_LABEL_ROLES}
    for role in ALLOWED_LABEL_ROLES:
        v1_training.validate_pairs_against_labels_v1(pairs[role], labels[role])
        actual_population = {
            "pairs": len(pairs[role]),
            "scenes": len({str(pair["scene_id"]) for pair in pairs[role]}),
            "unique_endpoints": len(
                {
                    str(pair[f"{side}_endpoint_sha256"])
                    for pair in pairs[role]
                    for side in ("current", "next")
                }
            ),
        }
        if actual_population != authority["authorized_roles"][role]:
            raise PermissionError(f"V13 authorized {role} population changed")
    if authority["authorized_roles"]["probability_calibration_open_count"] != 0:
        raise PermissionError("V13 probability-calibration role became authorized")
    sweep_masks = survival_scoring.build_swept_progress_masks_v1()
    persistence_masks = (
        survival_scoring.build_current_frame_swept_progress_masks_v1()
    )
    action_prior_m = (
        labels["train"].prefix_lengths.mean(axis=0, dtype=np.float64)
        * v1_executor.PROGRESS_SEGMENT_M
    )
    return V13ComposedRuntime(
        repository_root=repository,
        runtime_data_root=runtime_data_root,
        output_root=output_root,
        authority=dict(authority),
        reservation=dict(reservation),
        source_evidence=dict(source_evidence),
        torch=torch,
        np=np,
        device=device,
        minimal_runtime=minimal_runtime,
        executor_api=executor_api,
        model_module=model_module,
        training_module=training_module,
        v1_executor=v1_executor,
        v1_training=v1_training,
        labels_api=labels_api,
        metrics_api=metrics_api,
        survival_scoring=survival_scoring,
        raw_inputs=raw_inputs,
        loader=loader,
        n320_fit=n320_fit,
        n320_gate=n320_gate,
        n320_checkpoint=n320_checkpoint,
        schedule=schedule,
        train_pair_count=len(pairs["train"]),
        labels=labels,
        pairs=pairs,
        sweep_masks=sweep_masks,
        persistence_masks=persistence_masks,
        action_prior_m=action_prior_m,
        label_access=label_access,
        runtime_path_containment=runtime_path_containment,
        hardware=hardware,
        runtime_fingerprint=runtime_fingerprint,
        determinism=determinism,
        physical_provenance={},
        matched_module=matched,
        matched_source_root=matched_source_root,
    )


def _safe_output_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a nonempty relative string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PermissionError("artifact path escaped the V13 output root")
    path = root.joinpath(*pure.parts)
    parent = root
    for part in pure.parts[:-1]:
        parent = parent / part
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise PermissionError("artifact parent is not a regular directory")
        else:
            os.mkdir(parent, 0o700)
    return path


def _publish_bytes_immutable(path: Path, raw: bytes) -> Mapping[str, Any]:
    if not isinstance(raw, bytes):
        raise TypeError("published artifact payload must be bytes")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"write-once artifact already exists: {path.name}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return {
        "path": path.name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


@dataclass(frozen=True)
class V13WriteOncePublisher:
    output_root: Path
    executor_api: Any

    def publish_json(
        self,
        relative: str,
        core: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        path = _safe_output_path(self.output_root, relative)
        value = self.executor_api._write_immutable_json_v13(path, core)
        raw = self.executor_api._canonical_json_bytes(value) + b"\n"
        binding = {
            "path": relative,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "content_sha256": value["content_sha256"],
        }
        return {"value": value, "binding": binding}

    def publish_bytes(self, relative: str, raw: bytes) -> Mapping[str, Any]:
        path = _safe_output_path(self.output_root, relative)
        binding = dict(_publish_bytes_immutable(path, raw))
        binding["path"] = relative
        return binding


def _ensure_output_parent_v13(source_root: Path, output_relative: str) -> Path:
    pure = PurePosixPath(output_relative)
    if pure.is_absolute() or len(pure.parts) < 2 or ".." in pure.parts:
        raise PermissionError("V13 output root is not a narrow relative path")
    root = Path(source_root).resolve(strict=True)
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise PermissionError("V13 output parent contains a non-directory")
        else:
            os.mkdir(current, 0o700)
    if current.resolve(strict=True) != current.absolute():
        raise PermissionError("V13 output parent is noncanonical")
    return current


def _utc_now_v13() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def execute_future_authorized_v13(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reserve first, then compose and execute the controller exactly once."""

    repository = Path(repository_root).resolve(strict=True)
    if repository != ROOT.resolve(strict=True):
        raise PermissionError("V13 execution must run from its certified source export")
    _validate_certified_source_root_v13(repository, authority)
    _activate_certified_source_root_v13(repository)
    executor_api = importlib.import_module(
        "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v13_"
        "camera_evidence_bottleneck"
    )
    fixed_authority = _load_authority_file_v13(ROOT / AUTHORITY_RELATIVE_PATH)
    if type(authority) is not dict or authority != fixed_authority:
        raise PermissionError("supplied V13 authority differs from the fixed document")
    validated_authority = executor_api.validate_future_execution_prerequisites_v13(
        fixed_authority
    )
    source_evidence = _validate_source_evidence_v13(
        repository, validated_authority
    )
    _validate_runtime_data_root_v13(repository, validated_authority)
    _ensure_output_parent_v13(repository, executor_api.OUTPUT_ROOT_RELATIVE_PATH)
    reservation_created_utc = _utc_now_v13()
    reservation = executor_api.reserve_attempt_v13(
        repository,
        validated_authority,
        created_utc=reservation_created_utc,
    )
    output_root = repository / executor_api.OUTPUT_ROOT_RELATIVE_PATH
    runtime: V13ComposedRuntime | None = None
    stage = "post_reservation_runtime_composition"
    try:
        runtime = compose_runtime_v13(
            repository_root=repository,
            authority=validated_authority,
            reservation=reservation,
            source_evidence=source_evidence,
        )
        publisher = V13WriteOncePublisher(output_root, executor_api)
        stage = "future_authorized_engine"
        controller = getattr(executor_api, "run_future_authorized_engine_v13", None)
        if not callable(controller):
            raise RuntimeError("V13 authorized engine controller is absent")
        result = controller(
            authority=validated_authority,
            reservation=reservation,
            runtime=runtime,
            publisher=publisher,
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("V13 controller omitted its terminal receipt")
        return dict(result)
    except BaseException as error:
        if not (output_root / "success.json").exists() and not (
            output_root / "failure.json"
        ).exists():
            executor_api.terminalize_failure_v13(
                output_root,
                reservation,
                stage=stage,
                error=error,
                created_utc=_utc_now_v13(),
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
                    "schema": "lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_launcher_v1",
                    "status": "DENIED_NO_FUTURE_AUTHORITY",
                    "scientific_payload_opened": False,
                    "reservation_created": False,
                },
                sort_keys=True,
            )
        )
        return 4
    parsed = _parser().parse_args(arguments)
    authority = _load_authority_file_v13(parsed.future_authority)
    result = execute_future_authorized_v13(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(result, sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V13 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
