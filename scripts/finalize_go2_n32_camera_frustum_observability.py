#!/usr/bin/env python3
"""Strictly validate the immutable N32 camera-frustum audit result.

This finalizer intentionally uses only the Python standard library.  It does
not import the audit runner, NumPy, Torch, dataset code, or model code.  The
result is representation-only: a valid result can authorize implementation of
the frozen camera-centered geometry, while every training, holdout, G2,
runtime, and promotion license remains false.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any, Mapping, Sequence
import zipfile


REPOSITORY_ROOT = Path(__file__).absolute().parents[1]
RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_n32_camera_frustum_observability_audit/v2/result.json"
)
BINDING_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_n32_camera_frustum_observability_audit_binding_2026-07-11.md"
)
IMPLEMENTATION_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.md"
)
MACHINE_IMPLEMENTATION_MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.json"
)
INCIDENT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_n32_camera_frustum_manifest_preparation_failure_2026-07-11.md"
)
FIT_PANEL_PATH = (
    REPOSITORY_ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
)
V4_ADJUDICATION_REPORT_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_categorical_radial_n32_v4_result_2026-07-11.md"
)
KNOWN_BIAS_PROOF_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_n32_known_bias_impossibility_2026-07-11.md"
)

EXECUTION_BINDING_SHA256 = (
    "c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9"
)
FIT_PANEL_FILE_SHA256 = (
    "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
)
FIT_PANEL_CONTENT_SHA256 = (
    "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
)
FIT_ROWS_SHA256 = "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
V4_ADJUDICATION_REPORT_SHA256 = (
    "dd0842d1c59b42a985eaf0843f0d6f6adc41286a2a1a2b4b1f95111a9c0efa50"
)
KNOWN_BIAS_PROOF_SHA256 = (
    "e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a"
)
INCIDENT_SHA256 = (
    "5c3fad3b8e296aed239c3573e263af766b52e391fb9fe86e0e31d26c94845db3"
)
INCIDENT_STATUS = "acknowledged_pre_authoritative_run"

RESULT_SCHEMA = "lewm_go2_n32_camera_frustum_observability_audit_result_v1"
GEOMETRY_SCHEMA = "lewm_go2_n32_camera_frustum_geometry_v1"
OLD_COLUMN_SPAN_SCHEMA = "lewm_go2_n32_old_body_column_span_audit_v1"
MAPPING_AUDIT_SCHEMA = "lewm_go2_n32_camera_centered_mapping_audit_v1"
LABEL_SUPPORT_SCHEMA = "lewm_go2_n32_camera_frustum_label_support_v1"
RAY_SEQUENCE_SCHEMA = "lewm_go2_n32_camera_frustum_ray_sequences_v1"
OBSERVABILITY_SUMMARY_SCHEMA = (
    "lewm_go2_n32_camera_frustum_label_observability_summary_v1"
)
AUTHORIZATION_SCHEMA = "lewm_go2_n32_camera_frustum_authorization_decision_v1"

FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
ENDPOINT_SIDES = ("current", "next")
CLASS_NAMES = ("unknown", "free", "occupied")
TRANSITION_NAMES = (
    "unknown_to_free",
    "unknown_to_occupied",
    "free_to_unknown",
    "free_to_occupied",
    "occupied_to_unknown",
    "occupied_to_free",
)
DISTANCE_BIN_NAMES = (
    "0.0_to_0.5",
    "0.5_to_1.0",
    "1.0_to_2.0",
    "2.0_to_3.0",
    "3.0_plus",
)
FRAME_IDENTITY_FIELDS = (
    "family",
    "scene_id",
    "global_row",
    "side",
    "image_sha256",
    "label_shard_sha256",
    "label_row",
)
EXPECTED_FRAME_COUNT = 320
EXPECTED_TRANSITION_COUNT = 160
EXPECTED_FAMILY_FRAME_COUNT = 64
EXPECTED_SIDE_FRAME_COUNT = 160
CELLS_PER_FRAME = 64 * 64
SUPPORTED_CELLS_PER_FRAME = 1990
UNSUPPORTED_CELLS_PER_FRAME = 2106
RAYS_PER_FRAME = 256
ELIGIBLE_RAYS_PER_FRAME = 252
SHORT_RAYS_PER_FRAME = 4
EXPECTED_LABEL_SHARD_COUNT = 20
EXPECTED_TOTAL_LABEL_BYTES = EXPECTED_FRAME_COUNT * CELLS_PER_FRAME
EXPECTED_TOTAL_RAY_COUNT = EXPECTED_FRAME_COUNT * RAYS_PER_FRAME
CAMERA_MOUNT_COMPOSITION_TOLERANCE = 1e-5
NOMINAL_CAMERA_MOUNT_BODY = {
    "parent_link": "camera_link",
    "rpy_body_rad": [0.0, 0.0, 0.0],
    "xyz_body_m": [0.326, 0.0, 0.043],
}

MAPPING_SHA256 = "2b8cfb9dcf2deeebe7304d64a4a79b1631eb658991108eb3c3149cccf7a7dd4e"
SUPPORT_MASK_SHA256 = (
    "026d7654864bea7ae0545bd6448f6def64519a3bedcbc7ea747e7b4b95f82b3a"
)
OLD_SPAN_TABLE_SHA256 = (
    "bfd10e1b1b7a4e1b2497682b41f8886ae610f43b493c85e5fe9133376bf72eaf"
)

TOP_LEVEL_FIELDS = {
    "schema",
    "created_at_utc",
    "scope",
    "execution_binding",
    "inputs",
    "preflight_access_incident",
    "human_implementation_manifest",
    "machine_implementation_manifest",
    "runtime_environment",
    "source_hashes",
    "geometry_contract",
    "old_body_column_span_audit",
    "mapping_audit",
    "frame_identity",
    "source_geometry_manifest",
    "label_shard_manifest",
    "selected_label_bytes",
    "frame_reports",
    "label_observability",
    "family_class_count_table",
    "reconstruction",
    "collision_veto",
    "box_parity",
    "camera_mount_composition",
    "provenance",
    "phase_ledgers",
    "expected_finalizer_ledger",
    "two_phase_access_reconciliation",
    "rendered_collision_target_ambiguity",
    "authorization_decision",
    "licenses",
    "content_sha256",
}
SOURCE_HASH_KEYS = {
    "binding",
    "audit_core",
    "audit_core_test",
    "audit_runner",
    "audit_runner_test",
    "audit_finalizer",
    "audit_finalizer_test",
    "label_semantics",
    "geometry_contract_semantics",
    "scene_manifest_semantics",
    "planning_grid_semantics",
}
FORBIDDEN_ACCESS_COUNTERS = (
    "rgb_byte_opens",
    "rgb_decodes",
    "holdout_label_or_geometry_opens",
    "selection_or_calibration_opens",
    "physical_nontrain_role_opens",
    "g2_opens",
    "runtime_opens",
    "sealed_opens",
    "model_checkpoint_or_output_opens",
    "generated_v4_result_opens",
    "seed_20260711_opens",
)

DENIAL_PRIMARY_REASONS = (
    "sealed",
    "g2",
    "seed_20260711",
    "generated_v4_result",
    "model",
    "runtime",
    "physical_nontrain",
    "selection_or_calibration",
    "holdout",
    "image_or_depth",
    "unregistered_role",
    "forbidden_modality",
    "path_alias_or_escape",
    "unallowlisted",
)
DENIAL_MODALITIES = (
    "markdown",
    "python_source",
    "json",
    "jsonl",
    "npz",
    "image",
    "video",
    "raster_array",
    "point_cloud",
    "model",
    "archive",
    "unknown",
)
ALLOWED_SEMANTIC_ROLES = frozenset(
    {
        *SOURCE_HASH_KEYS,
        "fit_panel",
        "human_implementation_manifest",
        "machine_implementation_manifest",
        "incident_record",
        "v4_adjudication_report",
        "known_bias_proof",
        "physical_geometry_contract",
        "fit_label_shard",
        "fit_render_summary",
        "render_audit_contract",
        "fit_frame_selection",
        "render_source_plan",
        "source_frames_jsonl",
        "source_scene_manifest",
        "renderer_source",
        "audit_output",
    }
)
ALLOWED_MODALITIES_BY_ROLE = {
    **{role: "python_source" for role in SOURCE_HASH_KEYS if role != "binding"},
    "binding": "markdown",
    "fit_panel": "json",
    "human_implementation_manifest": "markdown",
    "machine_implementation_manifest": "json",
    "incident_record": "markdown",
    "v4_adjudication_report": "markdown",
    "known_bias_proof": "markdown",
    "physical_geometry_contract": "json",
    "fit_label_shard": "npz",
    "fit_render_summary": "json",
    "render_audit_contract": "json",
    "fit_frame_selection": "json",
    "render_source_plan": "json",
    "source_frames_jsonl": "jsonl",
    "source_scene_manifest": "json",
    "renderer_source": "python_source",
    "audit_output": "json",
}
FORBIDDEN_IMAGE_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".exr",
        ".hdr",
        ".gif",
    }
)
FORBIDDEN_VIDEO_SUFFIXES = frozenset(
    {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }
)
FORBIDDEN_MODEL_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".onnx",
        ".pkl",
        ".pickle",
        ".joblib",
    }
)


def _infer_lexical_modality(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix == ".py":
        return "python_source"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".npz":
        return "npz"
    if suffix in FORBIDDEN_IMAGE_SUFFIXES:
        return "image"
    if suffix in FORBIDDEN_VIDEO_SUFFIXES:
        return "video"
    if suffix in FORBIDDEN_MODEL_SUFFIXES:
        return "model"
    if suffix in {".npy", ".bin", ".raw"}:
        return "raster_array"
    if suffix in {".pcd", ".ply", ".las", ".laz"}:
        return "point_cloud"
    if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z"}:
        return "archive"
    return "unknown"

NPY_MAGIC = b"\x93NUMPY"
REGISTERED_LABEL_ARRAYS = (
    "current_labels",
    "current_supervision_mask",
    "next_labels",
    "next_supervision_mask",
)
REGISTERED_AUX_ARRAYS = (
    "current_observed_mask",
    "next_observed_mask",
    "relative_se2_current_frame",
    "primitive",
    "current_image_path",
    "next_image_path",
    "current_image_sha256",
    "next_image_sha256",
)


class FinalizationError(ValueError):
    """Raised when an immutable result fails strict finalization."""


class _ForbiddenScientificImportFinder(importlib.abc.MetaPathFinder):
    """Make NumPy and Torch unavailable during authoritative finalization."""

    _lewm_camera_finalizer_blocker = True

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        del path, target
        if fullname.split(".", 1)[0] in {"numpy", "torch"}:
            raise ModuleNotFoundError(
                f"{fullname} is disabled during strict camera-audit finalization"
            )
        return None


def _activate_import_isolation() -> None:
    if any(name == "numpy" or name.startswith("numpy.") for name in sys.modules):
        _fail("NumPy was imported before strict finalization")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        _fail("Torch was imported before strict finalization")
    if not any(
        getattr(finder, "_lewm_camera_finalizer_blocker", False)
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _ForbiddenScientificImportFinder())
    for name in ("numpy", "torch"):
        try:
            __import__(name)
        except ModuleNotFoundError:
            continue
        _fail(f"{name} remained importable during strict finalization")


def canonical_json_sha256(value: object) -> str:
    """Hash UTF-8 compact canonical JSON, rejecting NaN and infinity."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FinalizationError("value is not strict canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    """Return the complete, non-circular implementation source map."""

    return {
        "binding": BINDING_PATH,
        "audit_core": REPOSITORY_ROOT
        / "lewm/benchmarks/go2_n32_camera_frustum_observability.py",
        "audit_core_test": REPOSITORY_ROOT
        / "lewm/tests/test_go2_n32_camera_frustum_observability.py",
        "audit_runner": REPOSITORY_ROOT
        / "scripts/audit_go2_n32_camera_frustum_observability.py",
        "audit_runner_test": REPOSITORY_ROOT
        / "lewm/tests/test_audit_go2_n32_camera_frustum_observability.py",
        "audit_finalizer": Path(__file__).absolute(),
        "audit_finalizer_test": REPOSITORY_ROOT
        / "lewm/tests/test_finalize_go2_n32_camera_frustum_observability.py",
        "label_semantics": REPOSITORY_ROOT
        / "lewm/datasets/go2_paired_navigation.py",
        "geometry_contract_semantics": REPOSITORY_ROOT
        / "lewm/planning/geometry_contract.py",
        "scene_manifest_semantics": REPOSITORY_ROOT
        / "lewm_worlds/lewm_worlds/manifest.py",
        "planning_grid_semantics": REPOSITORY_ROOT
        / "lewm_worlds/lewm_worlds/planning_grid.py",
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(_source_paths().items())
    }


def _fail(message: str) -> None:
    raise FinalizationError(message)


def _record(value: object, fields: set[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        extra = sorted(actual - fields)
        _fail(f"{context} keys differ: missing={missing}, extra={extra}")
    return value


def _mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{context} must be an object")
    return value


def _list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{context} must be a list")
    return value


def _string(value: object, *, context: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _fail(f"{context} must be a{' nonempty' if nonempty else ''} string")
    return value


def _sha256(value: object, *, context: str) -> str:
    text = _string(value, context=context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(f"{context} must be a lowercase SHA-256")
    return text


def _strict_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{context} must be a bool")
    return value


def _strict_int(
    value: object,
    *,
    context: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        _fail(f"{context} is outside its registered range")
    return value


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{context} must be finite")
    return result


def _equal_number(actual: object, expected: float, *, context: str) -> None:
    value = _finite_number(actual, context=context)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15):
        _fail(f"{context} changed from the frozen value")


def _require_equal(actual: object, expected: object, *, context: str) -> None:
    if actual != expected:
        _fail(f"{context} differs from the frozen contract")


def _strict_json_loads(raw: bytes, *, context: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{context} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        _fail(f"{context} contains forbidden JSON constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            _fail(f"{context} contains a nonfinite JSON number")
        return parsed

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{context} is not strict UTF-8 JSON") from exc


def _current_runtime_environment() -> dict[str, str]:
    """Return runtime identities without importing NumPy."""

    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise FinalizationError("the frozen NumPy distribution is unavailable") from exc
    return {
        "python_implementation_name": sys.implementation.name,
        "python_implementation_version": [
            int(sys.implementation.version.major),
            int(sys.implementation.version.minor),
            int(sys.implementation.version.micro),
            str(sys.implementation.version.releaselevel),
            int(sys.implementation.version.serial),
        ],
        "python_version": sys.version,
        "numpy_version": numpy_version,
    }


def new_finalizer_access_ledger() -> dict[str, Any]:
    """Create a phase-local ledger; synthetic denial tests use a fresh copy."""

    return {
        "phase": "strict_finalizer",
        "panel_metadata_byte_opens": 0,
        "document_hash_byte_opens": 0,
        "label_shard_hash_byte_opens": 0,
        "label_shard_npz_opens": 0,
        "registered_arrays_decompressed": 0,
        "materialized_label_rows": 0,
        "materialized_supervision_rows": 0,
        "selected_label_rows_read": 0,
        "selected_supervision_rows_read": 0,
        "unselected_row_values_inspected": 0,
        "unselected_row_metrics_computed": 0,
        "unselected_rows_retained": 0,
        "derivative_shard_or_cache_writes": 0,
        "source_geometry_hash_byte_opens": 0,
        "source_geometry_json_parses": 0,
        "source_geometry_jsonl_records": 0,
        "denied_attempts_total": 0,
        "unexpected_path_attempts": 0,
        "denied_primary_reasons": {
            reason: 0 for reason in DENIAL_PRIMARY_REASONS
        },
        "denied_modality_attempts": {
            modality: 0 for modality in DENIAL_MODALITIES
        },
        "denied_attempt_records": [],
        "label_shards": [],
        "source_geometry": [],
        **{name: 0 for name in FORBIDDEN_ACCESS_COUNTERS},
    }


def _lexical_denial_reason(
    path: Path,
    *,
    requested_role: str,
    declared_role: str,
    modality: str,
) -> str | None:
    """Apply frozen semantic precedence without resolving or statting a path."""

    lexical = str(path).replace("\\", "/").lower()
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", lexical) if token)
    token_set = set(tokens)
    requested_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(requested_role).lower())
        if token
    }
    declared_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(declared_role).lower())
        if token
    }
    role_tokens = requested_tokens | declared_tokens
    inferred_modality = _infer_lexical_modality(path)
    train_data_roles = {
        "fit_panel",
        "fit_label_shard",
        "fit_render_summary",
        "render_audit_contract",
        "fit_frame_selection",
        "render_source_plan",
        "source_frames_jsonl",
        "source_scene_manifest",
        "renderer_source",
        "physical_geometry_contract",
    }
    if (
        "sealed" in token_set
        or "sealed" in role_tokens
        or any(token.startswith("sealed") for token in tokens)
    ):
        return "sealed"
    if "g2" in token_set or "g2" in role_tokens or "g2payload" in token_set:
        return "g2"
    if (
        "seed20260711" in token_set
        or "seed20260711" in role_tokens
        or ("seed" in token_set and "20260711" in token_set)
        or ("seed" in role_tokens and "20260711" in role_tokens)
    ):
        return "seed_20260711"
    if (
        requested_role == "generated_v4_result"
        or (
            requested_role != "v4_adjudication_report"
            and "generated" in token_set
            and "v4" in token_set
            and "result" in token_set
        )
        or "generatedv4result" in token_set
        or "generatedv4result" in role_tokens
        or ("seed" in token_set and "20260710" in token_set and "result" in token_set)
    ):
        return "generated_v4_result"
    if (
        modality == "model"
        or inferred_modality == "model"
        or bool(
            role_tokens
            & {
                "model",
                "models",
                "checkpoint",
                "checkpoints",
                "activation",
                "logit",
                "probability",
                "prediction",
                "parameter",
                "parameters",
            }
        )
        or bool(
            token_set
            & {
                "model",
                "models",
                "checkpoint",
                "checkpoints",
                "activation",
                "logit",
                "probability",
                "prediction",
                "parameter",
                "parameters",
            }
        )
    ):
        return "model"
    if (
        bool(token_set & {"runtime", "development", "closedloop"})
        or bool(role_tokens & {"runtime", "development", "closedloop"})
        or ("closed" in token_set and "loop" in token_set)
        or ("closed" in role_tokens and "loop" in role_tokens)
    ):
        return "runtime"
    implementation_test_roles = {
        "audit_core_test",
        "audit_runner_test",
        "audit_finalizer_test",
    }
    if (
        "nontrain" in token_set
        or "nontrain" in role_tokens
        or (
            requested_role not in implementation_test_roles
            and bool(
                role_tokens & {"validation", "test", "testeasy", "testhard"}
            )
        )
    ):
        return "physical_nontrain"
    if "calibration" in token_set or "calib" in token_set or bool(
        role_tokens & {"calibration", "calib"}
    ):
        return "selection_or_calibration"
    if (
        ("selection" in token_set or "selection" in role_tokens)
        and requested_role != "fit_frame_selection"
    ):
        return "selection_or_calibration"
    if bool(token_set & {"holdout", "heldout"}) or bool(
        role_tokens & {"holdout", "heldout"}
    ):
        return "holdout"
    if (
        modality in {"image", "video", "raster_array", "point_cloud"}
        or inferred_modality in {"image", "video", "raster_array", "point_cloud"}
        or bool(token_set & {"rgb", "image", "images", "depth", "pixels", "pointcloud"})
        or bool(role_tokens & {"rgb", "image", "images", "depth", "pixels", "pointcloud"})
    ):
        return "image_or_depth"
    declared_role_allowed = declared_role == requested_role or (
        requested_role in train_data_roles and declared_role == "train"
    )
    if requested_role not in ALLOWED_SEMANTIC_ROLES or not declared_role_allowed:
        return "unregistered_role"
    if ALLOWED_MODALITIES_BY_ROLE.get(requested_role) != modality:
        return "forbidden_modality"
    if inferred_modality != modality:
        return "forbidden_modality"
    return None


def _record_denial(
    ledger: dict[str, Any],
    *,
    path: Path,
    resolved_path: Path | None,
    requested_role: str,
    declared_role: str,
    modality: str,
    reason: str,
) -> None:
    if reason not in DENIAL_PRIMARY_REASONS:
        _fail("unregistered denial reason")
    ledger["denied_attempts_total"] += 1
    ledger["unexpected_path_attempts"] += 1
    ledger["denied_primary_reasons"][reason] += 1
    modality_counts = ledger["denied_modality_attempts"]
    modality_bucket = modality if modality in DENIAL_MODALITIES else "unknown"
    modality_counts[modality_bucket] = int(modality_counts[modality_bucket]) + 1
    ledger["denied_attempt_records"].append(
        {
            "lexical_path": str(path),
            "resolved_path": None if resolved_path is None else str(resolved_path),
            "requested_role": requested_role,
            "declared_role": declared_role,
            "modality": modality,
            "primary_reason": reason,
        }
    )


def _authorize_path(
    path: Path,
    *,
    requested_role: str,
    declared_role: str,
    modality: str,
    allowlist: Mapping[Path, str],
    ledger: dict[str, Any],
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[Path, str]:
    """Authorize one exact input before any existence check or byte read."""

    lexical_reason = _lexical_denial_reason(
        path,
        requested_role=requested_role,
        declared_role=declared_role,
        modality=modality,
    )
    if lexical_reason is not None:
        _record_denial(
            ledger,
            path=path,
            resolved_path=None,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            reason=lexical_reason,
        )
        raise FinalizationError(f"input denied as {lexical_reason}")

    lexical_absolute = path.absolute()
    resolved = path.resolve(strict=False)
    root = repository_root.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        _record_denial(
            ledger,
            path=path,
            resolved_path=resolved,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            reason="path_alias_or_escape",
        )
        raise FinalizationError("input path escapes the repository")
    expected = allowlist.get(lexical_absolute)
    if expected is not None and lexical_absolute != resolved:
        _record_denial(
            ledger,
            path=path,
            resolved_path=resolved,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            reason="path_alias_or_escape",
        )
        raise FinalizationError("input path is a symlink or alias")
    if expected is None:
        reason = "path_alias_or_escape" if lexical_absolute != resolved else "unallowlisted"
        _record_denial(
            ledger,
            path=path,
            resolved_path=resolved,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            reason=reason,
        )
        raise FinalizationError(f"input denied as {reason}")
    digest = _sha256(expected, context="allowlisted input SHA-256")
    return resolved, digest


def _parse_npy_header_dict(header: str, *, context: str) -> dict[str, Any]:
    """Parse one literal NPY header while detecting duplicate dictionary keys."""

    if not header.startswith("{") or not header.isascii():
        _fail(f"{context} NPY header must be one ASCII dictionary")
    if any(character in header for character in "\r\n\t\v\f\x00"):
        _fail(f"{context} NPY header contains illegal whitespace")
    try:
        parsed_expression = ast.parse(header, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise FinalizationError(f"{context} has a malformed NPY header") from exc
    expression = parsed_expression.body
    if not isinstance(expression, ast.Dict):
        _fail(f"{context} NPY header must be a dictionary literal")
    body_end = expression.end_col_offset
    if body_end is None or any(character != " " for character in header[body_end:]):
        _fail(f"{context} NPY header has malformed space padding")
    pairs: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key_node, value_node in zip(expression.keys, expression.values):
        if key_node is None:
            _fail(f"{context} NPY header may not unpack dictionaries")
        try:
            key = ast.literal_eval(key_node)
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise FinalizationError(f"{context} NPY header is not literal") from exc
        if not isinstance(key, str):
            _fail(f"{context} NPY header keys must be strings")
        if key in seen:
            _fail(f"{context} NPY header contains duplicate key {key!r}")
        seen.add(key)
        pairs.append((key, value))
    result = dict(pairs)
    if set(result) != {"descr", "fortran_order", "shape"}:
        _fail(f"{context} NPY header keys changed")
    return result


def _decode_npy_array(
    payload: bytes,
    *,
    expected_kind: str,
    context: str,
) -> tuple[int, bytes, dict[str, Any]]:
    """Decode a canonical C-order uint8/bool ``[N,64,64]`` NPY array."""

    if len(payload) < 10 or payload[:6] != NPY_MAGIC:
        _fail(f"{context} is not an NPY payload")
    major, minor = payload[6], payload[7]
    if (major, minor) == (1, 0):
        length_size = 2
        header_length = struct.unpack_from("<H", payload, 8)[0]
        header_start = 10
        codec = "latin1"
    elif (major, minor) in {(2, 0), (3, 0)}:
        if len(payload) < 12:
            _fail(f"{context} has a truncated NPY prefix")
        length_size = 4
        header_length = struct.unpack_from("<I", payload, 8)[0]
        header_start = 12
        codec = "utf-8" if major == 3 else "latin1"
    else:
        _fail(f"{context} uses an unsupported NPY version")
    del length_size
    header_end = header_start + int(header_length)
    if header_end > len(payload) or header_length <= 0 or header_length > 4096:
        _fail(f"{context} has a malformed NPY header length")
    if header_end % 64 != 0:
        _fail(f"{context} NPY header alignment changed")
    header_bytes = payload[header_start:header_end]
    if not header_bytes.endswith(b"\n") or header_bytes.count(b"\n") != 1:
        _fail(f"{context} NPY header must end in one newline")
    try:
        header = header_bytes.decode(codec)
    except UnicodeDecodeError as exc:
        raise FinalizationError(f"{context} NPY header encoding is invalid") from exc
    parsed = _parse_npy_header_dict(header[:-1], context=context)
    descr = parsed["descr"]
    if not isinstance(descr, str):
        _fail(f"{context} NPY dtype descriptor must be a string")
    allowed_descr = {"labels": {"|u1", "<u1"}, "supervision": {"|b1"}}
    if expected_kind not in allowed_descr or descr not in allowed_descr[expected_kind]:
        _fail(f"{context} has an unsupported or object NPY dtype")
    if parsed["fortran_order"] is not False:
        _fail(f"{context} must be C-order")
    shape = parsed["shape"]
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in shape)
        or shape[0] <= 0
        or shape[1:] != (64, 64)
    ):
        _fail(f"{context} must have shape [N,64,64]")
    data = payload[header_end:]
    expected_bytes = int(shape[0]) * CELLS_PER_FRAME
    if len(data) != expected_bytes:
        _fail(f"{context} NPY payload length or trailing bytes changed")
    dtype_name = "uint8" if expected_kind == "labels" else "bool"
    return int(shape[0]), bytes(data), {
        "npy_version": [major, minor],
        "dtype": dtype_name,
        "shape": [int(shape[0]), 64, 64],
        "c_order": True,
        "storage_row_count": int(shape[0]),
    }


def _decode_fit_label_npz(
    payload: bytes,
    *,
    selections: Sequence[tuple[str, int]],
    context: str,
) -> tuple[list[tuple[bytes, bytes]], dict[str, Any]]:
    """Decode only the four registered arrays and copy exact selected rows."""

    required_members = {f"{name}.npy" for name in REGISTERED_LABEL_ARRAYS}
    exact_members = required_members | {
        f"{name}.npy" for name in REGISTERED_AUX_ARRAYS
    }
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), mode="r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise FinalizationError(f"{context} is not a valid NPZ container") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            _fail(f"{context} contains duplicate archive names")
        if set(names) != exact_members:
            _fail(f"{context} archive inventory differs from the frozen 12 names")
        # Auxiliary members are inventory commitments only. They remain
        # unopened and are not semantic inputs to this finalizer.
        by_name = {info.filename: info for info in infos}
        for info in infos:
            offset = int(info.header_offset)
            if offset < 0 or offset + 30 > len(payload):
                _fail(f"{context}:{info.filename} has a bad local ZIP offset")
            if payload[offset : offset + 4] != b"PK\x03\x04":
                _fail(f"{context}:{info.filename} lacks a local ZIP header")
            local_flags = struct.unpack_from("<H", payload, offset + 6)[0]
            local_compression = struct.unpack_from("<H", payload, offset + 8)[0]
            name_length = struct.unpack_from("<H", payload, offset + 26)[0]
            extra_length = struct.unpack_from("<H", payload, offset + 28)[0]
            name_start = offset + 30
            local_header_end = name_start + name_length + extra_length
            if local_header_end > len(payload):
                _fail(f"{context}:{info.filename} truncates its local ZIP header")
            encoding = "utf-8" if local_flags & 0x800 else "cp437"
            try:
                local_name = payload[name_start : name_start + name_length].decode(encoding)
            except UnicodeDecodeError as exc:
                raise FinalizationError(
                    f"{context}:{info.filename} has an invalid local ZIP name"
                ) from exc
            if local_name != info.filename:
                _fail(f"{context}:{info.filename} central/local names differ")
            if bool(local_flags & 0x1) != bool(info.flag_bits & 0x1) or local_flags & 0x1:
                _fail(f"{context}:{info.filename} has inconsistent encryption flags")
            if local_compression != info.compress_type:
                _fail(f"{context}:{info.filename} central/local compression differs")
            if info.is_dir():
                _fail(f"{context}:{info.filename} is not a plain file")
        decoded: dict[str, tuple[int, bytes, dict[str, Any]]] = {}
        for member in sorted(required_members):
            info = by_name[member]
            if info.flag_bits & 0x1:
                _fail(f"{context} contains an encrypted registered array")
            if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                _fail(f"{context} contains an unsafe archive member")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                _fail(f"{context}:{member} uses unsupported compression")
            try:
                member_payload = archive.read(info)
            except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
                raise FinalizationError(
                    f"{context} registered array decompression failed"
                ) from exc
            if len(member_payload) != info.file_size:
                _fail(f"{context} registered array size changed during decompression")
            stem = member[:-4]
            kind = "supervision" if stem.endswith("supervision_mask") else "labels"
            decoded[stem] = _decode_npy_array(
                member_payload,
                expected_kind=kind,
                context=f"{context}:{member}",
            )

    storage_row_counts = {
        decoded[f"{side}_{suffix}"][0]
        for side in ENDPOINT_SIDES
        for suffix in ("labels", "supervision_mask")
    }
    if len(storage_row_counts) != 1:
        _fail(f"{context} registered array storage row counts differ")
    storage_rows_per_array = next(iter(storage_row_counts))
    for side in ENDPOINT_SIDES:
        if decoded[f"{side}_labels"][0] != decoded[f"{side}_supervision_mask"][0]:
            _fail(f"{context} {side} label/supervision storage counts differ")
    side_rows = Counter((side, int(row)) for side, row in selections)
    if any(side not in ENDPOINT_SIDES or row < 0 for side, row in side_rows):
        _fail(f"{context} contains an invalid selected row")
    if any(count != 1 for count in side_rows.values()):
        _fail(f"{context} repeats a selected side/row")
    selected: list[tuple[bytes, bytes]] = []
    for side, raw_row in selections:
        row = int(raw_row)
        label_count, labels, _label_metadata = decoded[f"{side}_labels"]
        mask_count, masks, _mask_metadata = decoded[f"{side}_supervision_mask"]
        if label_count != mask_count:
            _fail(f"{context} label and supervision row counts differ")
        if row >= label_count:
            _fail(f"{context} selected row is outside the committed shard")
        start = row * CELLS_PER_FRAME
        end = start + CELLS_PER_FRAME
        selected_labels = bytes(labels[start:end])
        selected_masks = bytes(masks[start:end])
        if any(value > 2 for value in selected_labels):
            _fail(f"{context} selected target contains an unregistered class")
        if any(value not in (0, 1) for value in selected_masks):
            _fail(f"{context} selected mask contains a noncanonical bool byte")
        if any(value != 1 for value in selected_masks):
            _fail(f"{context} selected supervision is not the full bool grid")
        selected.append((selected_labels, selected_masks))
    label_rows = decoded["current_labels"][0] + decoded["next_labels"][0]
    supervision_rows = (
        decoded["current_supervision_mask"][0]
        + decoded["next_supervision_mask"][0]
    )
    if label_rows != supervision_rows:
        _fail(f"{context} materialized label/supervision row totals differ")
    return selected, {
        "registered_arrays_decompressed": len(required_members),
        "storage_rows_per_array": storage_rows_per_array,
        "arrays": [
            {"name": f"{name}.npy", **decoded[name][2]}
            for name in sorted(decoded)
        ],
        "materialized_label_rows": label_rows,
        "materialized_supervision_rows": supervision_rows,
        "selected_label_rows_read": len(selected),
        "selected_supervision_rows_read": len(selected),
    }


def _stdlib_camera_mapping() -> tuple[list[tuple[int, int]], bytes, bytes]:
    """Build the literal frozen mapping and its two byte encodings."""

    half_fov = math.radians(78.323 / 2.0)
    mapping: list[tuple[int, int]] = []
    mapping_bytes = bytearray()
    support_bytes = bytearray()
    for row in range(64):
        forward_body = -1.0 + (row + 0.5) * 0.1
        for column in range(64):
            left_body = -3.2 + (column + 0.5) * 0.1
            forward_camera = forward_body - 0.326
            range_m = math.hypot(forward_camera, left_body)
            bearing = math.atan2(left_body, forward_camera)
            supported = bool(
                forward_camera >= 0.05
                and 0.0 <= range_m < 6.4
                and -half_fov <= bearing <= half_fov
            )
            if supported:
                radial = int(math.floor(range_m / 0.1))
                angular_fraction = (bearing + half_fov) / (2.0 * half_fov)
                angular = min(255, int(math.floor(angular_fraction * 256.0)))
            else:
                radial = angular = -1
            mapping.append((radial, angular))
            mapping_bytes.extend(struct.pack("<hh", radial, angular))
            support_bytes.append(1 if supported else 0)
    return mapping, bytes(mapping_bytes), bytes(support_bytes)


def _stdlib_mapping_audit() -> dict[str, Any]:
    mapping, mapping_bytes, support_bytes = _stdlib_camera_mapping()
    used: dict[tuple[int, int], list[list[int]]] = {}
    invalid_negative = partial = out_of_range = 0
    for index, (radial, angular) in enumerate(mapping):
        row, column = divmod(index, 64)
        if (radial == -1) != (angular == -1):
            partial += 1
        if radial < -1 or angular < -1:
            invalid_negative += 1
        if radial >= 0 and angular >= 0:
            if radial >= 64 or angular >= 256:
                out_of_range += 1
            else:
                used.setdefault((radial, angular), []).append([row, column])
    collisions = [
        {
            "range_bin": radial,
            "angular_bin": angular,
            "multiplicity": len(locations),
            "cartesian_locations": locations,
        }
        for (radial, angular), locations in sorted(used.items())
        if len(locations) > 1
    ]
    supported = sum(support_bytes)
    passes = not (partial or invalid_negative or out_of_range or collisions)
    return {
        "schema": MAPPING_AUDIT_SCHEMA,
        "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "support_mask_sha256": hashlib.sha256(support_bytes).hexdigest(),
        "mapping_dtype": "int16",
        "signed_int16": True,
        "supported_cartesian_cell_count": supported,
        "unsupported_cartesian_cell_count": 4096 - supported,
        "unique_used_polar_bin_count": len(used),
        "unused_polar_bin_count": 64 * 256 - len(used),
        "partially_mapped_entry_count": partial,
        "invalid_negative_entry_count": invalid_negative,
        "out_of_range_entry_count": out_of_range,
        "nondeterministic_entry_count": 0,
        "expected_support_mismatch_count": 0,
        "collision_bin_count": len(collisions),
        "collision_extra_cartesian_count": sum(
            record["multiplicity"] - 1 for record in collisions
        ),
        "collisions": collisions,
        "deterministic": True,
        "all_mapped_indices_in_range": not (invalid_negative or out_of_range),
        "all_entries_complete": partial == 0,
        "support_matches_frozen_geometry": True,
        "injective": not collisions,
        "passes": passes,
    }


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        _fail("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stdlib_old_span_columns(*, require_vertical_anchor: bool) -> list[dict[str, Any]]:
    half_fov = math.radians(78.323 / 2.0)
    angular_width = 2.0 * half_fov / 256.0
    tan_vertical = math.tan(math.radians(62.8370386364) * 0.5)
    anchors = (-0.333, -0.133, 0.067, 0.267, 0.467)
    records: list[dict[str, Any]] = []
    for angular_bin in range(256):
        body_bearing = -half_fov + (angular_bin + 0.5) * angular_width
        cos_bearing = math.cos(body_bearing)
        sin_bearing = math.sin(body_bearing)
        selected = []
        for radial_bin in range(64):
            radius_m = (radial_bin + 0.5) * 0.1
            forward_camera = radius_m * cos_bearing - 0.326
            left_camera = radius_m * sin_bearing
            camera_bearing = math.atan2(left_camera, forward_camera)
            participates = bool(
                forward_camera >= 0.05
                and -half_fov <= camera_bearing <= half_fov
            )
            if participates and require_vertical_anchor:
                participates = any(
                    -1.0
                    <= -(anchor - 0.043) / (forward_camera * tan_vertical)
                    <= 1.0
                    for anchor in anchors
                )
            if participates:
                selected.append(camera_bearing)
        if len(selected) >= 2:
            minimum = min(selected)
            maximum = max(selected)
            span = maximum - minimum
            span_deg: float | None = math.degrees(span)
            span_bins: float | None = span / angular_width
        else:
            minimum = maximum = span = span_deg = span_bins = None
        records.append(
            {
                "body_angular_bin": angular_bin,
                "body_bearing_center_rad": body_bearing,
                "body_bearing_center_deg": math.degrees(body_bearing),
                "participating_range_count": len(selected),
                "minimum_camera_bearing_rad": minimum,
                "maximum_camera_bearing_rad": maximum,
                "span_rad": span,
                "span_deg": span_deg,
                "span_new_angular_bins": span_bins,
            }
        )
    return records


def _stdlib_span_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spans = [
        float(record["span_rad"])
        for record in records
        if record["span_rad"] is not None
    ]
    if not spans or any(not math.isfinite(value) for value in spans):
        _fail("old body-column span audit produced no finite spans")
    half_fov = math.radians(78.323 / 2.0)
    angular_width = 2.0 * half_fov / 256.0
    span_bins = [value / angular_width for value in spans]
    p50 = _linear_quantile(spans, 0.50)
    p95 = _linear_quantile(spans, 0.95)
    return {
        "column_count": len(records),
        "participating_sample_count": sum(
            int(record["participating_range_count"]) for record in records
        ),
        "columns_with_span_count": len(spans),
        "columns_with_fewer_than_two_participants_count": len(records) - len(spans),
        "span_rad": {"p50": p50, "p95": p95, "maximum": max(spans)},
        "span_deg": {
            "p50": math.degrees(p50),
            "p95": math.degrees(p95),
            "maximum": math.degrees(max(spans)),
        },
        "span_new_angular_bins": {
            "p50": _linear_quantile(span_bins, 0.50),
            "p95": _linear_quantile(span_bins, 0.95),
            "maximum": max(span_bins),
        },
        "columns_span_ge_1_new_bin": sum(value >= 1.0 for value in span_bins),
        "columns_span_ge_2_new_bins": sum(value >= 2.0 for value in span_bins),
        "columns_span_ge_4_new_bins": sum(value >= 4.0 for value in span_bins),
        "columns_span_ge_8_new_bins": sum(value >= 8.0 for value in span_bins),
        "quantile_method": "numpy_linear_float64",
    }


def _stdlib_old_body_column_span_audit() -> dict[str, Any]:
    half_fov = math.radians(78.323 / 2.0)
    angular_width = 2.0 * half_fov / 256.0
    primary = _stdlib_old_span_columns(require_vertical_anchor=True)
    horizontal = _stdlib_old_span_columns(require_vertical_anchor=False)
    table = {
        "primary_with_vertical_anchor": primary,
        "horizontal_only": horizontal,
    }
    return {
        "schema": OLD_COLUMN_SPAN_SCHEMA,
        "geometry": {
            "old_body_radius_centers_m": [
                float((index + 0.5) * 0.1) for index in range(64)
            ],
            "old_body_bearing_bin_count": 256,
            "old_body_bearing_range_rad": [-half_fov, half_fov],
            "new_angular_bin_width_rad": angular_width,
            "primary_vertical_anchor_rule": "at_least_one_registered_anchor_valid",
        },
        "primary": {
            "columns": primary,
            "summary": _stdlib_span_summary(primary),
        },
        "horizontal_only": {
            "columns": horizontal,
            "summary": _stdlib_span_summary(horizontal),
        },
        "old_column_span_table_sha256": canonical_json_sha256(table),
    }


def _stdlib_ray_locations(
    mapping: Sequence[tuple[int, int]],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    rays: list[list[tuple[int, int, int]]] = [[] for _ in range(256)]
    for index, (radial, angular) in enumerate(mapping):
        if radial >= 0:
            row, column = divmod(index, 64)
            rays[angular].append((radial, row, column))
    result = []
    for angular, locations in enumerate(rays):
        ordered = tuple(sorted(locations))
        ranges = [item[0] for item in ordered]
        if len(ranges) != len(set(ranges)):
            _fail(f"camera mapping ray {angular} has a range-bin tie")
        result.append(ordered)
    return tuple(result)


def _stdlib_ray_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    length_histogram = Counter(str(int(record["length"])) for record in records)
    transition_histogram = Counter(
        str(int(record["transition_count"])) for record in records
    )
    directed = Counter({name: 0 for name in TRANSITION_NAMES})
    for record in records:
        directed.update(record["directed_unequal_transition_counts"])
    sequence_count = len(records)
    eligible = sum(int(record["length"]) >= 2 for record in records)
    events = sum(int(record["transition_count"]) for record in records)
    return {
        "sequence_count": sequence_count,
        "length_histogram": dict(
            sorted(length_histogram.items(), key=lambda item: int(item[0]))
        ),
        "sequences_with_fewer_than_two_cells_count": sequence_count - eligible,
        "transition_rate_eligible_sequence_count": eligible,
        "class_transition_histogram": dict(
            sorted(transition_histogram.items(), key=lambda item: int(item[0]))
        ),
        "maximum_transitions_per_sequence": max(
            (int(record["transition_count"]) for record in records), default=0
        ),
        "directed_unequal_transition_counts": {
            name: int(directed[name]) for name in TRANSITION_NAMES
        },
        "transition_bucket_counts": {
            "0": sum(int(record["transition_count"]) == 0 for record in records),
            "1": sum(int(record["transition_count"]) == 1 for record in records),
            "2": sum(int(record["transition_count"]) == 2 for record in records),
            "3_plus": sum(
                int(record["transition_count"]) >= 3 for record in records
            ),
        },
        "transition_event_count": events,
        "transition_events_per_eligible_sequence": (
            float(events / eligible) if eligible else None
        ),
        "contains_known_after_unknown_count": sum(
            bool(record["contains_known_after_unknown"]) for record in records
        ),
        "contains_free_after_occupied_count": sum(
            bool(record["contains_free_after_occupied"]) for record in records
        ),
        "scalar_first_hit_irregular_count": sum(
            not bool(record["scalar_first_hit_regular"]) for record in records
        ),
        "scalar_first_hit_regular_count": sum(
            bool(record["scalar_first_hit_regular"]) for record in records
        ),
    }


def _stdlib_analyze_frame_labels(
    target: bytes,
    supervision: bytes,
    *,
    frame_key: Mapping[str, Any],
    mapping: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Recompute all label/support/ray evidence from one selected row."""

    if len(target) != CELLS_PER_FRAME or len(supervision) != CELLS_PER_FRAME:
        _fail("selected label or supervision row has the wrong byte length")
    if any(value > 2 for value in target):
        _fail("selected target contains an unregistered class")
    if any(value != 1 for value in supervision):
        _fail("selected supervision row is not the full finite grid")
    polar_mapping = list(mapping) if mapping is not None else _stdlib_camera_mapping()[0]
    if len(polar_mapping) != CELLS_PER_FRAME:
        _fail("camera mapping has the wrong cell count")
    support = [radial >= 0 and angular >= 0 for radial, angular in polar_mapping]
    by_class: dict[str, dict[str, int]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        total = sum(value == class_id for value in target)
        supported = sum(
            value == class_id and is_supported
            for value, is_supported in zip(target, support)
        )
        by_class[class_name] = {
            "total": total,
            "supported": supported,
            "unsupported": total - supported,
        }
    violations = []
    for index, (class_id, is_supported) in enumerate(zip(target, support)):
        if not is_supported and class_id in (1, 2):
            row, column = divmod(index, 64)
            violations.append(
                {
                    "frame_key": dict(frame_key),
                    "row": row,
                    "column": column,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                }
            )
    unsupported = sum(not value for value in support)
    unsupported_free = by_class["free"]["unsupported"]
    unsupported_occupied = by_class["occupied"]["unsupported"]
    unsupported_unknown = by_class["unknown"]["unsupported"]
    label_support = {
        "schema": LABEL_SUPPORT_SCHEMA,
        "total_supervised_label_count": CELLS_PER_FRAME,
        "supported_label_count": sum(support),
        "unsupported_label_count": unsupported,
        "class_counts": {
            name: by_class[name]["total"] for name in CLASS_NAMES
        },
        "by_class": by_class,
        "unsupported_free_count": unsupported_free,
        "unsupported_occupied_count": unsupported_occupied,
        "unsupported_unknown_count": unsupported_unknown,
        "unsupported_targets_are_all_unknown": unsupported_unknown == unsupported,
        "violations": violations,
        "passes": bool(
            unsupported_free == 0
            and unsupported_occupied == 0
            and unsupported_unknown == unsupported
        ),
    }

    ray_records = []
    ranks = {1: 0, 2: 1, 0: 2}
    for angular, locations in enumerate(_stdlib_ray_locations(polar_mapping)):
        ranges = [radial for radial, _row, _column in locations]
        classes = [target[row * 64 + column] for _radial, row, column in locations]
        collapsed: list[int] = []
        for value in classes:
            if not collapsed or value != collapsed[-1]:
                collapsed.append(value)
        directed = Counter({name: 0 for name in TRANSITION_NAMES})
        for source, destination in zip(collapsed, collapsed[1:]):
            directed[f"{CLASS_NAMES[source]}_to_{CLASS_NAMES[destination]}"] += 1
        unknown_positions = [index for index, value in enumerate(classes) if value == 0]
        known_positions = [index for index, value in enumerate(classes) if value != 0]
        occupied_positions = [index for index, value in enumerate(classes) if value == 2]
        free_positions = [index for index, value in enumerate(classes) if value == 1]
        ray_records.append(
            {
                "frame_key": dict(frame_key),
                "angular_bin": angular,
                "length": len(classes),
                "range_bins": ranges,
                "class_sequence": classes,
                "collapsed_class_sequence": collapsed,
                "transition_count": max(0, len(collapsed) - 1),
                "directed_unequal_transition_counts": {
                    name: int(directed[name]) for name in TRANSITION_NAMES
                },
                "contains_known_after_unknown": bool(
                    unknown_positions
                    and known_positions
                    and min(unknown_positions) < max(known_positions)
                ),
                "contains_free_after_occupied": bool(
                    occupied_positions
                    and free_positions
                    and min(occupied_positions) < max(free_positions)
                ),
                "scalar_first_hit_regular": all(
                    ranks[left] <= ranks[right]
                    for left, right in zip(classes, classes[1:])
                ),
            }
        )
    ray_summary = _stdlib_ray_summary(ray_records)
    return {
        "label_support": label_support,
        "ray_sequences": {
            "schema": RAY_SEQUENCE_SCHEMA,
            "summary": ray_summary,
            "sequence_summary_records_sha256": canonical_json_sha256(ray_records),
            "transition_table_sha256": canonical_json_sha256(ray_summary),
        },
        "ray_records": ray_records,
    }


def _validate_timestamp(value: object) -> None:
    text = _string(value, context="created_at_utc")
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise FinalizationError("created_at_utc must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail("created_at_utc must be an ISO-8601 UTC timestamp")


def _validate_scope(value: object) -> None:
    scope = _record(
        value,
        {
            "dataset_role",
            "transition_count",
            "frame_count",
            "families",
            "endpoint_sides",
            "learning_performed",
        },
        context="scope",
    )
    _require_equal(scope["dataset_role"], "train", context="scope.dataset_role")
    _require_equal(
        _strict_int(scope["transition_count"], context="scope.transition_count"),
        EXPECTED_TRANSITION_COUNT,
        context="scope.transition_count",
    )
    _require_equal(
        _strict_int(scope["frame_count"], context="scope.frame_count"),
        EXPECTED_FRAME_COUNT,
        context="scope.frame_count",
    )
    _require_equal(scope["families"], list(FAMILIES), context="scope.families")
    _require_equal(
        scope["endpoint_sides"], list(ENDPOINT_SIDES), context="scope.endpoint_sides"
    )
    if _strict_bool(scope["learning_performed"], context="scope.learning_performed"):
        _fail("the observability audit must not perform learning")


def _derive_panel_records(
    panel: object,
    *,
    expected_content_sha256: str = FIT_PANEL_CONTENT_SHA256,
    expected_rows_sha256: str = FIT_ROWS_SHA256,
    expected_transitions: int = EXPECTED_TRANSITION_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive the sole canonical frame/shard selection from strict panel JSON."""

    value = _mapping(panel, context="fit panel")
    _require_equal(
        value.get("schema"),
        "lewm_go2_physical_micro_overfit_panel_v1",
        context="fit panel schema",
    )
    core = dict(value)
    declared_content = _sha256(
        core.pop("content_sha256", None), context="fit panel content_sha256"
    )
    if (
        declared_content != expected_content_sha256
        or canonical_json_sha256(core) != declared_content
    ):
        _fail("fit panel canonical content hash mismatch")
    _require_equal(value.get("families"), list(FAMILIES), context="fit panel families")
    rows_per_family = _strict_int(
        value.get("rows_per_family_panel"),
        context="fit panel rows_per_family_panel",
        minimum=1,
    )
    if rows_per_family * len(FAMILIES) != expected_transitions:
        _fail("fit panel transition budget changed")
    panels = _mapping(value.get("panels"), context="fit panel panels")
    fit = _mapping(panels.get("fit"), context="fit panel fit partition")
    rows = _list(fit.get("rows"), context="fit panel fit rows")
    expected_frames = expected_transitions * len(ENDPOINT_SIDES)
    if len(rows) != expected_transitions:
        _fail("fit panel transition count changed")
    if _strict_int(fit.get("row_count"), context="fit panel fit row_count") != expected_transitions:
        _fail("fit panel declared transition count changed")
    if _strict_int(fit.get("frame_count"), context="fit panel fit frame_count") != expected_frames:
        _fail("fit panel declared frame count changed")
    declared_rows = _sha256(fit.get("rows_sha256"), context="fit panel rows_sha256")
    if declared_rows != expected_rows_sha256 or canonical_json_sha256(rows) != declared_rows:
        _fail("fit panel ordered row hash mismatch")

    family_transition_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(rows):
        row = _mapping(raw_row, context=f"fit panel rows[{row_index}]")
        family = _string(row.get("family"), context=f"fit row {row_index} family")
        if family not in FAMILIES:
            _fail("fit panel row has an unregistered family")
        if row.get("dataset_role") != "train":
            _fail("fit panel row is not physical train role")
        family_transition_counts[family] += 1
        scene_id = _string(row.get("scene_id"), context=f"fit row {row_index} scene_id")
        global_row = _strict_int(
            row.get("global_row"), context=f"fit row {row_index} global_row"
        )
        shard_path = _string(
            row.get("label_shard_path"), context=f"fit row {row_index} label_shard_path"
        )
        shard_hash = _sha256(
            row.get("label_shard_sha256"),
            context=f"fit row {row_index} label_shard_sha256",
        )
        label_row = _strict_int(
            row.get("label_shard_row"), context=f"fit row {row_index} label_shard_row"
        )
        for side in ENDPOINT_SIDES:
            image_hash = _sha256(
                row.get(f"{side}_image_sha256"),
                context=f"fit row {row_index} {side}_image_sha256",
            )
            records.append(
                {
                    "family": family,
                    "scene_id": scene_id,
                    "global_row": global_row,
                    "side": side,
                    "image_path_metadata_only": _string(
                        row.get(f"{side}_image_path"),
                        context=f"fit row {row_index} {side}_image_path",
                    ),
                    "image_sha256": image_hash,
                    "label_shard_path": shard_path,
                    "label_shard_sha256": shard_hash,
                    "label_row": label_row,
                    "frame_index": _strict_int(
                        row.get(f"{side}_frame_index"),
                        context=f"fit row {row_index} {side}_frame_index",
                    ),
                    "env_index": _strict_int(
                        row.get("env_index"), context=f"fit row {row_index} env_index"
                    ),
                    "timestamp_ns": _strict_int(
                        row.get(f"{side}_timestamp_ns"),
                        context=f"fit row {row_index} {side}_timestamp_ns",
                    ),
                    "episode_id": _string(
                        row.get("episode_id"), context=f"fit row {row_index} episode_id"
                    ),
                    "reset_count": _strict_int(
                        row.get("reset_count"), context=f"fit row {row_index} reset_count"
                    ),
                    "episode_step": _strict_int(
                        row.get(f"{side}_episode_step"),
                        context=f"fit row {row_index} {side}_episode_step",
                    ),
                }
            )
    expected_family_transitions = expected_transitions // len(FAMILIES)
    if family_transition_counts != Counter(
        {family: expected_family_transitions for family in FAMILIES}
    ):
        _fail("fit panel transition families are not balanced")
    family_rank = {family: index for index, family in enumerate(FAMILIES)}
    side_rank = {side: index for index, side in enumerate(ENDPOINT_SIDES)}
    records.sort(
        key=lambda record: (
            family_rank[str(record["family"])],
            str(record["scene_id"]),
            int(record["global_row"]),
            side_rank[str(record["side"])],
        )
    )
    ordering_keys = [
        (
            str(record["family"]),
            str(record["scene_id"]),
            int(record["global_row"]),
            str(record["side"]),
        )
        for record in records
    ]
    if len(set(ordering_keys)) != len(ordering_keys):
        _fail("fit panel contains a canonical frame-order tie")
    selected_storage_rows = [
        (
            str(record["label_shard_path"]),
            str(record["label_shard_sha256"]),
            str(record["side"]),
            int(record["label_row"]),
        )
        for record in records
    ]
    if len(set(selected_storage_rows)) != len(selected_storage_rows):
        _fail("fit panel reuses one selected shard side/row")
    identities = [
        tuple(record[field] for field in FRAME_IDENTITY_FIELDS) for record in records
    ]
    if len(records) != expected_frames or len(set(identities)) != expected_frames:
        _fail("fit panel does not derive exactly unique endpoint identities")
    if len({record["image_sha256"] for record in records}) != expected_frames:
        _fail("fit panel endpoint image commitments are not unique")
    family_counts = Counter(str(record["family"]) for record in records)
    side_counts = Counter(str(record["side"]) for record in records)
    family_side_counts = Counter(
        (str(record["family"]), str(record["side"])) for record in records
    )
    if family_counts != Counter(
        {family: expected_family_transitions * 2 for family in FAMILIES}
    ):
        _fail("fit panel does not derive the frozen per-family frame counts")
    if side_counts != Counter({side: expected_transitions for side in ENDPOINT_SIDES}):
        _fail("fit panel does not derive the frozen endpoint-side counts")
    if family_side_counts != Counter(
        {
            (family, side): expected_family_transitions
            for family in FAMILIES
            for side in ENDPOINT_SIDES
        }
    ):
        _fail("fit panel does not derive the frozen family-by-side counts")

    local_grid = _mapping(value.get("local_grid"), context="fit panel local_grid")
    if (
        local_grid.get("shape") != [64, 64]
        or _finite_number(local_grid.get("cell_size_m"), context="local_grid.cell_size_m") != 0.1
        or local_grid.get("forward_edge_range_m") != [-1.0, 5.4]
        or local_grid.get("left_edge_range_m") != [-3.2, 3.2]
    ):
        _fail("fit panel local-grid geometry changed")
    projection = _mapping(
        value.get("source_camera_projection"), context="fit panel source_camera_projection"
    )
    if not math.isclose(
        _finite_number(projection.get("horizontal_fov_deg"), context="projection.horizontal_fov_deg"),
        78.323,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        _finite_number(projection.get("near_m"), context="projection.near_m"),
        0.05,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        _fail("fit panel camera calibration changed")
    inputs = _mapping(value.get("inputs"), context="fit panel inputs")
    geometry = _mapping(inputs.get("geometry_contract"), context="fit panel geometry contract")
    render_audit = _mapping(
        inputs.get("render_audit_contract"), context="fit panel render-audit contract"
    )
    return records, {
        "local_grid": dict(local_grid),
        "camera_projection": dict(projection),
        "geometry_contract": {
            "path": _string(geometry.get("path"), context="geometry contract path"),
            "file_sha256": _sha256(
                geometry.get("file_sha256"), context="geometry contract file SHA-256"
            ),
            "semantic_sha256": _sha256(
                geometry.get("semantic_sha256"), context="geometry contract semantic SHA-256"
            ),
        },
        "render_audit_contract": {
            "path": _string(
                render_audit.get("path"), context="render-audit contract path"
            ),
            "file_sha256": _sha256(
                render_audit.get("file_sha256"),
                context="render-audit contract file SHA-256",
            ),
            "content_sha256": _sha256(
                render_audit.get("content_sha256"),
                context="render-audit contract content SHA-256",
            ),
        },
    }


def _expected_label_shard_entries(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    hashes: dict[str, str] = {}
    for record in records:
        path = str(record["label_shard_path"])
        digest = str(record["label_shard_sha256"])
        previous = hashes.setdefault(path, digest)
        if previous != digest:
            _fail("one panel-bound label shard has conflicting hashes")
        grouped.setdefault(path, []).append(record)
    entries = []
    for path in sorted(grouped):
        selected = grouped[path]
        selected_tuples = [
            [
                record["family"],
                record["scene_id"],
                record["global_row"],
                record["side"],
                record["label_row"],
            ]
            for record in selected
        ]
        entries.append(
            {
                "path": path,
                "sha256": hashes[path],
                "selected_tuples": selected_tuples,
                "selected_row_count": len(selected),
                "family_side_counts": {
                    family: {
                        side: sum(
                            record["family"] == family and record["side"] == side
                            for record in selected
                        )
                        for side in ENDPOINT_SIDES
                    }
                    for family in FAMILIES
                },
            }
        )
    return entries, grouped


def _independently_read_selected_shards(
    *,
    panel_records: Sequence[Mapping[str, Any]],
    expected_entries: Sequence[Mapping[str, Any]],
    machine_label_manifest: Mapping[str, Any],
    ledger: dict[str, Any],
) -> dict[tuple[Any, ...], tuple[bytes, bytes]]:
    machine_entries = _list(
        machine_label_manifest["entries"], context="machine label_shards.entries"
    )
    expected_machine = [dict(entry) for entry in expected_entries]
    if machine_entries != expected_machine:
        _fail("machine label-shard inventory differs from panel-derived commitments")
    if len(expected_entries) != EXPECTED_LABEL_SHARD_COUNT:
        _fail("panel-derived label-shard inventory is not exactly 20")
    records_by_path: dict[str, list[Mapping[str, Any]]] = {}
    for record in panel_records:
        records_by_path.setdefault(str(record["label_shard_path"]), []).append(record)
    allowlist = {
        Path(str(entry["path"])).absolute(): str(entry["sha256"])
        for entry in expected_entries
    }
    selected: dict[tuple[Any, ...], tuple[bytes, bytes]] = {}
    for entry in expected_entries:
        path = Path(str(entry["path"]))
        resolved, expected_hash = _authorize_path(
            path,
            requested_role="fit_label_shard",
            declared_role="train",
            modality="npz",
            allowlist=allowlist,
            ledger=ledger,
        )
        ledger["label_shard_hash_byte_opens"] += 1
        raw = resolved.read_bytes()
        observed_hash = hashlib.sha256(raw).hexdigest()
        if observed_hash != expected_hash:
            _fail("fit label shard raw-byte SHA-256 changed")
        records = records_by_path.get(str(entry["path"]), [])
        selections = [
            (str(record["side"]), int(record["label_row"])) for record in records
        ]
        ledger["label_shard_npz_opens"] += 1
        selected_values, counts = _decode_fit_label_npz(
            raw,
            selections=selections,
            context=f"fit label shard {resolved}",
        )
        for key in (
            "registered_arrays_decompressed",
            "materialized_label_rows",
            "materialized_supervision_rows",
            "selected_label_rows_read",
            "selected_supervision_rows_read",
        ):
            ledger[key] += int(counts[key])
        ledger["label_shards"].append(
            {
                "path": str(resolved),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
                **counts,
            }
        )
        if len(selected_values) != len(records):
            _fail("stdlib shard decoder returned the wrong selected-row count")
        for record, value in zip(records, selected_values):
            identity = tuple(record[field] for field in FRAME_IDENTITY_FIELDS)
            if identity in selected:
                _fail("independent selected label identity repeated")
            selected[identity] = value
        del raw, selected_values
    if len(selected) != EXPECTED_FRAME_COUNT:
        _fail("independent label selection did not produce exactly 320 frames")
    return selected


def _strict_jsonl_loads(raw: bytes, *, context: str) -> list[Mapping[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FinalizationError(f"{context} is not strict UTF-8 JSONL") from exc
    if text and not text.endswith("\n"):
        _fail(f"{context} must end with a newline")
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            _fail(f"{context} contains a blank JSONL record")
        value = _strict_json_loads(line.encode("utf-8"), context=f"{context}[{index}]")
        if not isinstance(value, Mapping):
            _fail(f"{context}[{index}] must contain an object")
        records.append(value)
    return records


def _source_path(value: object, *, context: str) -> str:
    raw = Path(_string(value, context=context))
    if ".." in raw.parts:
        _fail(f"{context} contains a forbidden parent-path alias")
    return str((raw if raw.is_absolute() else REPOSITORY_ROOT / raw).absolute())


def _finite_vector(value: object, *, size: int, context: str) -> list[float]:
    vector = _list(value, context=context)
    if len(vector) != size:
        _fail(f"{context} must contain exactly {size} values")
    return [
        _finite_number(component, context=f"{context}[{index}]")
        for index, component in enumerate(vector)
    ]


def _camera_mount_record(value: object, *, context: str) -> dict[str, Any]:
    record = _record(
        value,
        {"parent_link", "rpy_body_rad", "xyz_body_m"},
        context=context,
    )
    return {
        "parent_link": _string(record["parent_link"], context=f"{context}.parent_link"),
        "rpy_body_rad": _finite_vector(
            record["rpy_body_rad"], size=3, context=f"{context}.rpy_body_rad"
        ),
        "xyz_body_m": _finite_vector(
            record["xyz_body_m"], size=3, context=f"{context}.xyz_body_m"
        ),
    }


def _vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(component) * float(component) for component in vector))


def _angular_error(
    recorded: Sequence[float], expected: Sequence[float], *, context: str
) -> float:
    recorded_norm = _vector_norm(recorded)
    expected_norm = _vector_norm(expected)
    if recorded_norm <= 0.0 or expected_norm <= 0.0:
        _fail(f"{context} cannot be normalized")
    dot = sum(
        float(recorded[index]) * float(expected[index]) for index in range(3)
    ) / (recorded_norm * expected_norm)
    return math.acos(max(-1.0, min(1.0, dot)))


def _within_camera_mount_tolerance(value: float) -> bool:
    return float(value) <= CAMERA_MOUNT_COMPOSITION_TOLERANCE


def _camera_mount_composition_evidence(
    frame: Mapping[str, Any],
    *,
    plan_mount_value: object,
    context: str,
) -> dict[str, Any]:
    """Recompute one fixed-mount world pose from the unnormalized XYZW q."""

    base_pose = _mapping(frame.get("base_pose_world"), context=f"{context}.base_pose_world")
    base_position_record = _mapping(
        base_pose.get("position"), context=f"{context}.base_pose_world.position"
    )
    base_position = [
        _finite_number(
            base_position_record.get(axis),
            context=f"{context}.base_pose_world.position.{axis}",
        )
        for axis in ("x", "y", "z")
    ]
    quaternion = _finite_vector(
        frame.get("base_quat_world_xyzw"),
        size=4,
        context=f"{context}.base_quat_world_xyzw",
    )
    base_rpy = _mapping(frame.get("base_rpy_rad"), context=f"{context}.base_rpy_rad")
    stored_yaw = _finite_number(
        base_rpy.get("yaw"), context=f"{context}.base_rpy_rad.yaw"
    )
    plan_mount = _camera_mount_record(
        plan_mount_value, context=f"{context}.plan_camera_mount_body"
    )
    frame_mount = _camera_mount_record(
        frame.get("camera_mount_body"), context=f"{context}.frame_camera_mount_body"
    )
    camera_pose = _mapping(
        frame.get("camera_pose_world"), context=f"{context}.camera_pose_world"
    )
    recorded_position = _finite_vector(
        camera_pose.get("position"),
        size=3,
        context=f"{context}.camera_pose_world.position",
    )
    recorded_lookat = _finite_vector(
        camera_pose.get("lookat"),
        size=3,
        context=f"{context}.camera_pose_world.lookat",
    )
    recorded_up = _finite_vector(
        camera_pose.get("up"),
        size=3,
        context=f"{context}.camera_pose_world.up",
    )

    qx, qy, qz, qw = quaternion
    rotation = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )

    def rotate(vector: Sequence[float]) -> list[float]:
        return [
            sum(row[column] * float(vector[column]) for column in range(3))
            for row in rotation
        ]

    mount_xyz = NOMINAL_CAMERA_MOUNT_BODY["xyz_body_m"]
    expected_offset = rotate(mount_xyz)
    expected_position = [
        base_position[index] + expected_offset[index] for index in range(3)
    ]
    expected_forward = rotate((1.0, 0.0, 0.0))
    expected_up = rotate((0.0, 0.0, 1.0))
    expected_lookat = [
        expected_position[index] + expected_forward[index] for index in range(3)
    ]
    quaternion_norm = math.sqrt(
        qx * qx + qy * qy + qz * qz + qw * qw
    )
    quaternion_norm_residual = abs(quaternion_norm - 1.0)
    quaternion_yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    yaw_delta = stored_yaw - quaternion_yaw
    wrapped_yaw_residual = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
    position_residual = max(
        abs(recorded_position[index] - expected_position[index])
        for index in range(3)
    )
    lookat_residual = max(
        abs(recorded_lookat[index] - expected_lookat[index])
        for index in range(3)
    )
    up_residual = max(
        abs(recorded_up[index] - expected_up[index]) for index in range(3)
    )
    recorded_forward = [
        recorded_lookat[index] - recorded_position[index] for index in range(3)
    ]
    look_distance = _vector_norm(recorded_forward)
    look_distance_residual = abs(look_distance - 1.0)
    forward_angular_error = _angular_error(
        recorded_forward, expected_forward, context=f"{context}.forward"
    )
    up_angular_error = _angular_error(
        recorded_up, expected_up, context=f"{context}.up"
    )
    passes = bool(
        plan_mount == NOMINAL_CAMERA_MOUNT_BODY
        and frame_mount == NOMINAL_CAMERA_MOUNT_BODY
        and _within_camera_mount_tolerance(quaternion_norm_residual)
        and _within_camera_mount_tolerance(wrapped_yaw_residual)
        and _within_camera_mount_tolerance(position_residual)
        and _within_camera_mount_tolerance(lookat_residual)
        and _within_camera_mount_tolerance(up_residual)
        and _within_camera_mount_tolerance(look_distance_residual)
        and _within_camera_mount_tolerance(forward_angular_error)
        and _within_camera_mount_tolerance(up_angular_error)
    )
    return {
        "base_position_world": base_position,
        "base_quat_world_xyzw": quaternion,
        "stored_base_yaw_rad": stored_yaw,
        "plan_camera_mount_body": plan_mount,
        "frame_camera_mount_body": frame_mount,
        "recorded_camera_pose_world": {
            "position": recorded_position,
            "lookat": recorded_lookat,
            "up": recorded_up,
        },
        "expected_camera_pose_world": {
            "position": expected_position,
            "lookat": expected_lookat,
            "up": expected_up,
        },
        "quaternion_norm": quaternion_norm,
        "quaternion_norm_abs_residual": quaternion_norm_residual,
        "quaternion_yaw_rad": quaternion_yaw,
        "wrapped_yaw_abs_residual_rad": wrapped_yaw_residual,
        "position_max_abs_residual_m": position_residual,
        "lookat_max_abs_residual_m": lookat_residual,
        "up_max_abs_residual": up_residual,
        "look_distance_m": look_distance,
        "look_distance_abs_residual_m": look_distance_residual,
        "forward_angular_error_rad": forward_angular_error,
        "up_angular_error_rad": up_angular_error,
        "passes": passes,
    }


def _stdlib_render_object_records(scene_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: list[tuple[str, object]] = [
        ("wall", scene_manifest.get("walls", [])),
        ("obstacle", scene_manifest.get("obstacles", [])),
        ("landmark", scene_manifest.get("landmarks", [])),
    ]
    visual = scene_manifest.get("visual_randomization")
    distractors: object = []
    if isinstance(visual, Mapping):
        distractors = visual.get("distractor_objects", [])
    groups.append(("distractor", distractors))
    records = []
    for group, raw_boxes in groups:
        boxes = _list(raw_boxes, context=f"scene manifest {group} boxes")
        for index, raw_box in enumerate(boxes):
            box = _mapping(raw_box, context=f"scene manifest {group}[{index}]")
            center = _list(box.get("center_xyz_m"), context=f"{group}[{index}] center")
            size = _list(box.get("size_xyz_m"), context=f"{group}[{index}] size")
            if len(center) != 3 or len(size) != 3:
                _fail("scene box center/size must each contain three values")
            record = {
                "group": group,
                "object_id": _string(box.get("object_id"), context=f"{group} object_id"),
                "kind": _string(box.get("kind"), context=f"{group} kind"),
                "center_xyz_m": [
                    _finite_number(value, context=f"{group} center value") for value in center
                ],
                "size_xyz_m": [
                    _finite_number(value, context=f"{group} size value") for value in size
                ],
                "rpy_rad": [
                    _finite_number(box.get("roll_rad", 0.0), context=f"{group} roll"),
                    _finite_number(box.get("pitch_rad", 0.0), context=f"{group} pitch"),
                    _finite_number(box.get("yaw_rad", 0.0), context=f"{group} yaw"),
                ],
                "material_id": str(box.get("material_id", "")),
            }
            records.append(record)
    return sorted(records, key=lambda record: (record["group"], record["object_id"]))


def _validate_source_scene_provenance(
    *,
    machine_entries: Sequence[Mapping[str, Any]],
    parsed_by_path: Mapping[Path, Any],
    panel_records: Sequence[Mapping[str, Any]],
    selected_scene_families: Mapping[str, str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    required_roles = {
        "physical_geometry_contract",
        "render_audit_contract",
        "fit_render_summary",
        "fit_frame_selection",
        "render_source_plan",
        "source_frames_jsonl",
        "source_scene_manifest",
        "renderer_source",
    }
    assignments: dict[str, dict[str, Mapping[str, Any]]] = {
        scene_id: {} for scene_id in selected_scene_families
    }
    for entry in machine_entries:
        scene_id = str(entry["scene_id"])
        role = str(entry["semantic_role"])
        if role in assignments[scene_id]:
            _fail("one scene has multiple source assignments for one semantic role")
        assignments[scene_id][role] = entry
    records_by_scene = {
        scene_id: [
            record for record in panel_records if str(record["scene_id"]) == scene_id
        ]
        for scene_id in selected_scene_families
    }
    camera_evidence: dict[tuple[Any, ...], dict[str, Any]] = {}
    for scene_id in sorted(assignments):
        if set(assignments[scene_id]) != required_roles:
            _fail("source-geometry scene does not have the exact eight-role provenance set")
        roles = assignments[scene_id]

        def payload(role: str) -> Any:
            path = Path(str(roles[role]["path"])).absolute()
            if role == "renderer_source":
                return None
            if path not in parsed_by_path:
                _fail(f"source provenance role {role} was not strictly parsed")
            return parsed_by_path[path]

        summary = _mapping(payload("fit_render_summary"), context=f"summary {scene_id}")
        family = selected_scene_families[scene_id]
        if (
            summary.get("schema") != "lewm_rendered_vision_v04"
            or summary.get("render_status") != "complete"
            or str(summary.get("scene_id", "")) != scene_id
            or str(summary.get("family", "")) != family
            or bool(summary.get("g2_model_outputs_opened", False))
        ):
            _fail("render summary identity/status/G2 provenance changed")
        resolution = _list(summary.get("resolution_wh"), context="render summary resolution")
        if resolution != [224, 168] or not math.isclose(
            float(resolution[1]) / float(resolution[0]),
            3.0 / 4.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail("render summary native resolution/aspect changed")
        projection = _record(
            summary.get("camera_projection"),
            {
                "model",
                "renderer_fov_axis",
                "horizontal_fov_deg",
                "vertical_fov_deg",
                "near_m",
                "far_m",
                "runtime_rectification_required",
            },
            context="render summary camera_projection",
        )
        if (
            projection.get("model") != "pinhole"
            or projection.get("renderer_fov_axis") != "vertical"
            or projection.get("runtime_rectification_required") is not False
            or not math.isclose(
                _finite_number(projection.get("horizontal_fov_deg"), context="summary horizontal FOV"),
                78.323,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                _finite_number(projection.get("near_m"), context="summary near plane"),
                0.05,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or _finite_number(projection.get("far_m"), context="summary far plane")
            <= 0.05
        ):
            _fail("render summary projection contract changed")

        source_names = {
            "plan": "render_source_plan",
            "frames_jsonl": "source_frames_jsonl",
            "scene_manifest": "source_scene_manifest",
            "renderer_source": "renderer_source",
        }
        source = _record(
            summary.get("source"),
            set(source_names),
            context="render summary source",
        )
        for source_name, role in source_names.items():
            record = _record(
                source.get(source_name),
                {"path", "sha256"},
                context=f"summary source.{source_name}",
            )
            if (
                _source_path(record.get("path"), context=f"summary source.{source_name}.path")
                != str(roles[role]["path"])
                or _sha256(record.get("sha256"), context=f"summary source.{source_name}.sha256")
                != roles[role]["sha256"]
            ):
                _fail("render summary source commitment differs from machine inventory")

        rendered_raw = _list(summary.get("rendered_frames"), context="rendered frames")
        rendered = []
        for index, raw in enumerate(rendered_raw):
            record = _record(
                raw,
                {"frame_index", "env_index", "timestamp_ns", "image_sha256"},
                context=f"rendered_frames[{index}]",
            )
            rendered.append(
                {
                    "frame_index": _strict_int(record.get("frame_index"), context="rendered frame_index"),
                    "env_index": _strict_int(record.get("env_index"), context="rendered env_index"),
                    "timestamp_ns": _strict_int(record.get("timestamp_ns"), context="rendered timestamp_ns"),
                    "image_sha256": _sha256(record.get("image_sha256"), context="rendered image SHA-256"),
                }
            )
        rendered.sort(key=lambda record: (record["frame_index"], record["env_index"]))
        rendered_keys = [[record["frame_index"], record["env_index"]] for record in rendered]
        if (
            len({tuple(key) for key in rendered_keys}) != len(rendered_keys)
            or _strict_int(summary.get("frame_count"), context="summary frame_count") != len(rendered)
            or _sha256(summary.get("rendered_image_set_sha256"), context="rendered image-set SHA-256")
            != canonical_json_sha256(rendered)
        ):
            _fail("render summary rendered-frame set/hash changed")
        rendered_by_key = {
            (record["frame_index"], record["env_index"]): record for record in rendered
        }
        for panel_record in records_by_scene[scene_id]:
            key = (int(panel_record["frame_index"]), int(panel_record["env_index"]))
            rendered_record = rendered_by_key.get(key)
            if (
                rendered_record is None
                or rendered_record["timestamp_ns"] != int(panel_record["timestamp_ns"])
                or rendered_record["image_sha256"] != panel_record["image_sha256"]
            ):
                _fail("panel endpoint does not match render summary exactly once")

        selection_commitment = _record(
            summary.get("frame_selection"),
            {"path", "sha256", "frame_key_set_sha256"},
            context="summary frame_selection",
        )
        if (
            _source_path(selection_commitment.get("path"), context="selection path")
            != str(roles["fit_frame_selection"]["path"])
            or _sha256(selection_commitment.get("sha256"), context="selection SHA-256")
            != roles["fit_frame_selection"]["sha256"]
        ):
            _fail("summary frame-selection commitment changed")
        selection = _record(
            payload("fit_frame_selection"),
            {
                "schema",
                "scene_id",
                "scene_id_sha256",
                "dataset_role",
                "row_count",
                "frame_count",
                "frame_keys",
                "frame_key_set_sha256",
                "source_rows",
                "g2_images_opened",
                "g2_label_shards_opened",
                "content_sha256",
            },
            context="frame selection",
        )
        selection_core = dict(selection)
        selection_content_hash = _sha256(
            selection_core.pop("content_sha256", None), context="selection content SHA-256"
        )
        frame_keys = _list(selection.get("frame_keys"), context="selection frame_keys")
        normalized_keys = []
        for index, raw_key in enumerate(frame_keys):
            key = _list(raw_key, context=f"selection frame_keys[{index}]")
            if len(key) != 2:
                _fail("selection frame key must contain frame/env indices")
            normalized_keys.append(
                [
                    _strict_int(key[0], context="selection frame_index"),
                    _strict_int(key[1], context="selection env_index"),
                ]
            )
        panel_keys = sorted(
            {
                (int(record["frame_index"]), int(record["env_index"]))
                for record in records_by_scene[scene_id]
            }
        )
        if len(panel_keys) != len(records_by_scene[scene_id]):
            _fail("fit panel repeats one selected source-frame key")
        panel_key_lists = [list(key) for key in panel_keys]
        source_rows = _record(
            selection.get("source_rows"),
            {"path", "sha256"},
            context="selection source_rows",
        )
        _string(source_rows["path"], context="selection source_rows.path")
        _sha256(source_rows["sha256"], context="selection source_rows.sha256")
        selected_panel_rows = {
            (str(record["scene_id"]), int(record["global_row"]))
            for record in records_by_scene[scene_id]
        }
        if (
            selection.get("schema") != "lewm_go2_selected_render_frames_v1"
            or str(selection.get("scene_id", "")) != scene_id
            or _sha256(
                selection.get("scene_id_sha256"),
                context="selection scene_id SHA-256",
            )
            != hashlib.sha256(scene_id.encode("utf-8")).hexdigest()
            or selection.get("dataset_role") != "train"
            or selection.get("g2_images_opened") is not False
            or selection.get("g2_label_shards_opened") is not False
            or canonical_json_sha256(selection_core) != selection_content_hash
            or normalized_keys != sorted(normalized_keys)
            or len({tuple(key) for key in normalized_keys}) != len(normalized_keys)
            or normalized_keys != rendered_keys
            or not set(map(tuple, panel_key_lists)).issubset(
                set(map(tuple, normalized_keys))
            )
            or _strict_int(selection.get("row_count"), context="selection row_count")
            < len(selected_panel_rows)
            or _strict_int(selection.get("frame_count"), context="selection frame_count")
            != len(normalized_keys)
            or _sha256(selection.get("frame_key_set_sha256"), context="selection key-set SHA-256")
            != canonical_json_sha256(normalized_keys)
            or selection_commitment.get("frame_key_set_sha256")
            != canonical_json_sha256(normalized_keys)
        ):
            _fail("frame-selection/rendered-frame provenance changed")

        plan = _mapping(payload("render_source_plan"), context="render source plan")
        camera = _record(
            plan.get("camera"),
            {
                "native_resolution",
                "training_resolution",
                "fov_axis",
                "fov_deg",
                "near_m",
                "far_m",
                "encoding",
                "mount_body",
            },
            context="render source plan camera",
        )
        plan_mount = camera.get("mount_body")
        plan_horizontal = _finite_number(camera.get("fov_deg"), context="plan horizontal FOV")
        plan_near = _finite_number(camera.get("near_m"), context="plan near plane")
        plan_far = _finite_number(camera.get("far_m"), context="plan far plane")
        expected_vertical = math.degrees(
            2.0 * math.atan(math.tan(math.radians(plan_horizontal) * 0.5) * (168.0 / 224.0))
        )
        if (
            str(plan.get("schema", "")) != "lewm_render_replay_plan_v0"
            or str(plan.get("scene_id", "")) != scene_id
            or _source_path(plan.get("frames_jsonl"), context="plan frames_jsonl")
            != str(roles["source_frames_jsonl"]["path"])
            or camera.get("fov_axis") != "horizontal"
            or not math.isclose(plan_horizontal, float(projection["horizontal_fov_deg"]), rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(plan_near, float(projection["near_m"]), rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(
                plan_far,
                float(projection["far_m"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                _finite_number(projection.get("vertical_fov_deg"), context="summary vertical FOV"),
                expected_vertical,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            _fail("render plan/summary projection or frame source changed")

        scene_manifest = _mapping(payload("source_scene_manifest"), context="scene manifest")
        if (
            str(scene_manifest.get("scene_id", "")) != scene_id
            or str(scene_manifest.get("family", "")) != family
        ):
            _fail("source scene-manifest identity changed")
        frame_records = _list(payload("source_frames_jsonl"), context="source frames")
        source_by_key: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
        for frame in frame_records:
            key = (
                _strict_int(frame.get("frame_index"), context="source frame_index"),
                _strict_int(frame.get("env_index"), context="source env_index"),
            )
            source_by_key.setdefault(key, []).append(frame)
        if any(
            len(source_by_key.get(tuple(key), ())) != 1 for key in rendered_keys
        ):
            _fail("source frames do not contain every rendered/selected key exactly once")
        for rendered_record in rendered:
            key = (rendered_record["frame_index"], rendered_record["env_index"])
            frame = source_by_key[key][0]
            if _strict_int(frame.get("timestamp_ns"), context="source timestamp_ns") != rendered_record["timestamp_ns"]:
                _fail("source-frame timestamp differs from render summary")
        for panel_record in records_by_scene[scene_id]:
            key = (int(panel_record["frame_index"]), int(panel_record["env_index"]))
            frame = source_by_key[key][0]
            episode = _mapping(frame.get("episode"), context="source frame episode")
            if (
                str(episode.get("episode_id", "")) != str(panel_record["episode_id"])
                or _strict_int(episode.get("reset_count"), context="source reset_count")
                != int(panel_record["reset_count"])
                or _strict_int(episode.get("episode_step"), context="source episode_step")
                != int(panel_record["episode_step"])
            ):
                _fail("source-frame episode provenance differs from panel")
            base_pose = _mapping(frame.get("base_pose_world"), context="source base_pose_world")
            position = _record(
                base_pose.get("position"),
                {"x", "y", "z"},
                context="source base position",
            )
            rpy = _mapping(frame.get("base_rpy_rad"), context="source base_rpy_rad")
            if "yaw" not in rpy or set(rpy) - {"roll", "pitch", "yaw"}:
                _fail("source base_rpy_rad fields changed")
            camera_pose = _record(
                frame.get("camera_pose_world"),
                {"position", "lookat", "up"},
                context="source camera pose",
            )
            camera_vectors = {
                name: _list(camera_pose.get(name), context=f"camera {name}")
                for name in ("position", "lookat", "up")
            }
            if any(len(values) != 3 for values in camera_vectors.values()):
                _fail("source frame camera vectors must each contain three values")
            finite_values = [
                *[_finite_number(position.get(axis), context=f"base {axis}") for axis in ("x", "y", "z")],
                _finite_number(rpy.get("yaw"), context="base yaw"),
                *[
                    _finite_number(value, context="camera pose value")
                    for name in ("position", "lookat", "up")
                    for value in camera_vectors[name]
                ],
            ]
            if len(finite_values) != 13:
                _fail("source frame lacks exact finite base/camera pose geometry")
            identity = tuple(
                panel_record[field] for field in FRAME_IDENTITY_FIELDS
            )
            if identity in camera_evidence:
                _fail("one selected frame has duplicate camera-mount evidence")
            camera_evidence[identity] = _camera_mount_composition_evidence(
                frame,
                plan_mount_value=plan_mount,
                context=(
                    f"source camera mount {scene_id}:"
                    f"{panel_record['frame_index']}:{panel_record['env_index']}"
                ),
            )

        parity = _record(
            summary.get("object_parity"),
            {
                "schema",
                "rendered_groups",
                "collision_distractors_rendered",
                "full_box_roll_pitch_yaw_rendered",
                "rendered_object_count",
                "rendered_object_ids",
                "rendered_object_ids_sha256",
                "rendered_object_records_sha256",
            },
            context="summary object_parity",
        )
        expected_object_records = _stdlib_render_object_records(scene_manifest)
        expected_object_ids = sorted(
            str(record["object_id"]) for record in expected_object_records
        )
        object_ids = _list(parity.get("rendered_object_ids"), context="rendered object IDs")
        normalized_ids = [str(value) for value in object_ids]
        if (
            parity.get("schema") != "lewm_render_object_parity_v1"
            or parity.get("rendered_groups")
            != ["wall", "obstacle", "landmark", "distractor"]
            or parity.get("collision_distractors_rendered") is not True
            or parity.get("full_box_roll_pitch_yaw_rendered") is not True
            or normalized_ids != sorted(normalized_ids)
            or len(set(normalized_ids)) != len(normalized_ids)
            or normalized_ids != expected_object_ids
            or _strict_int(parity.get("rendered_object_count"), context="rendered object count")
            != len(expected_object_records)
            or _sha256(parity.get("rendered_object_ids_sha256"), context="rendered object IDs SHA-256")
            != canonical_json_sha256(normalized_ids)
        ):
            _fail("render object-parity provenance is incomplete or nonunique")
        _sha256(
            parity.get("rendered_object_records_sha256"),
            context="rendered object-record SHA-256",
        )
        if parity["rendered_object_records_sha256"] != canonical_json_sha256(
            expected_object_records
        ):
            _fail("render object-record hash differs from source scene manifest")
    expected_identities = {
        tuple(record[field] for field in FRAME_IDENTITY_FIELDS)
        for record in panel_records
    }
    if set(camera_evidence) != expected_identities:
        _fail("camera-mount evidence does not cover every selected frame exactly once")
    return camera_evidence


def _independently_verify_source_geometry(
    *,
    machine_source_manifest: Mapping[str, Any],
    machine_render_summary_manifest: Mapping[str, Any],
    geometry_commitment: Mapping[str, Any],
    render_audit_commitment: Mapping[str, Any],
    expected_result_entries: Sequence[Mapping[str, Any]],
    selected_scene_ids: set[str],
    selected_scene_families: Mapping[str, str],
    panel_records: Sequence[Mapping[str, Any]],
    ledger: dict[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    machine_entries = _list(
        machine_source_manifest["entries"],
        context="machine source_geometry.entries",
    )
    converted = [
        {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "semantic_role": entry["semantic_role"],
            "scene_id": entry["scene_id"],
        }
        for entry in machine_entries
    ]
    if converted != [dict(entry) for entry in expected_result_entries]:
        _fail("result source geometry differs from machine-authorized inventory")
    render_summary_entries = _list(
        machine_render_summary_manifest["entries"],
        context="machine render_summaries.entries",
    )
    source_summary_entries = [
        entry
        for entry in machine_entries
        if entry["semantic_role"] == "fit_render_summary"
    ]
    if render_summary_entries != source_summary_entries:
        _fail("ordered render-summary inventory differs from source geometry")
    expected_summary_assignments = {
        (
            str(entry["path"]),
            str(entry["sha256"]),
            str(entry["scene_id"]),
        )
        for entry in render_summary_entries
    }
    source_summary_assignments = {
        (
            str(entry["path"]),
            str(entry["sha256"]),
            str(entry["scene_id"]),
        )
        for entry in machine_entries
        if entry["semantic_role"] == "fit_render_summary"
    }
    if source_summary_assignments != expected_summary_assignments:
        _fail("render-summary inventory differs from source-geometry assignments")
    expected_physical_assignments = {
        (
            str(geometry_commitment["path"]),
            str(geometry_commitment["file_sha256"]),
            scene_id,
        )
        for scene_id in selected_scene_ids
    }
    actual_physical_assignments = {
        (str(entry["path"]), str(entry["sha256"]), str(entry["scene_id"]))
        for entry in machine_entries
        if entry["semantic_role"] == "physical_geometry_contract"
    }
    if actual_physical_assignments != expected_physical_assignments:
        _fail("physical-geometry assignments differ from panel/machine commitments")
    expected_render_audit_assignments = {
        (
            str(render_audit_commitment["path"]),
            str(render_audit_commitment["file_sha256"]),
            scene_id,
        )
        for scene_id in selected_scene_ids
    }
    actual_render_audit_assignments = {
        (str(entry["path"]), str(entry["sha256"]), str(entry["scene_id"]))
        for entry in machine_entries
        if entry["semantic_role"] == "render_audit_contract"
    }
    if actual_render_audit_assignments != expected_render_audit_assignments:
        _fail("render-audit assignments differ from the panel commitment")
    summary_by_scene: dict[str, str] = {}
    for path, _digest, scene_id in expected_summary_assignments:
        if scene_id in summary_by_scene:
            _fail("machine inventory assigns multiple summaries to one selected scene")
        summary_by_scene[scene_id] = path
    if set(summary_by_scene) != selected_scene_ids:
        _fail("render-summary scene set differs from the selected panel")
    for record in panel_records:
        image_path = Path(str(record["image_path_metadata_only"]))
        if image_path.parent.name != "rgb":
            _fail("panel image commitment is not under a render rgb directory")
        expected_summary_path = str(image_path.parent.parent / "summary.json")
        if summary_by_scene[str(record["scene_id"])] != expected_summary_path:
            _fail("panel image commitment disagrees with its render summary")
    commitments: dict[Path, str] = {}
    for entry in machine_entries:
        scene_id = str(entry["scene_id"])
        if scene_id not in selected_scene_ids:
            _fail("machine source geometry names a scene outside the fit panel")
        path = Path(str(entry["path"])).absolute()
        digest = str(entry["sha256"])
        previous = commitments.setdefault(path, digest)
        if previous != digest:
            _fail("one source-geometry path has conflicting machine hashes")
    allowlist = dict(commitments)
    observed_by_path: dict[Path, str] = {}
    parsed_by_path: dict[Path, Any] = {}
    parsed_paths: set[Path] = set()
    for entry in machine_entries:
        lexical = Path(str(entry["path"]))
        role = str(entry["semantic_role"])
        modality = ALLOWED_MODALITIES_BY_ROLE.get(role)
        if modality is None:
            _fail("machine source geometry has no registered modality")
        resolved, expected_hash = _authorize_path(
            lexical,
            requested_role=role,
            declared_role=role,
            modality=modality,
            allowlist=allowlist,
            ledger=ledger,
        )
        if resolved not in observed_by_path:
            ledger["source_geometry_hash_byte_opens"] += 1
            raw = resolved.read_bytes()
            observed = hashlib.sha256(raw).hexdigest()
            if observed != expected_hash:
                _fail("source-geometry raw-byte SHA-256 changed")
            observed_by_path[resolved] = observed
            if modality == "json":
                parsed = _strict_json_loads(raw, context=f"source geometry {resolved}")
                if not isinstance(parsed, Mapping):
                    _fail("source-geometry JSON root must be an object")
                ledger["source_geometry_json_parses"] += 1
                parsed_by_path[resolved] = parsed
            elif modality == "jsonl":
                records = _strict_jsonl_loads(raw, context=f"source geometry {resolved}")
                ledger["source_geometry_json_parses"] += 1
                ledger.setdefault("source_geometry_jsonl_records", 0)
                ledger["source_geometry_jsonl_records"] += len(records)
                parsed_by_path[resolved] = records
            elif modality != "python_source":
                _fail("source geometry uses a modality forbidden to the finalizer")
            parsed_paths.add(resolved)
            del raw
        parsed_value = parsed_by_path.get(resolved)
        scene_id = str(entry["scene_id"])
        if isinstance(parsed_value, Mapping):
            declared_scene = parsed_value.get("scene_id")
            if declared_scene is not None and str(declared_scene) != scene_id:
                _fail("source-geometry payload scene identity contradicts its assignment")
            if role == "fit_render_summary":
                if (
                    parsed_value.get("schema") != "lewm_rendered_vision_v04"
                    or parsed_value.get("render_status") != "complete"
                    or str(parsed_value.get("family", ""))
                    != selected_scene_families.get(scene_id)
                ):
                    _fail("fit render-summary provenance changed")
            elif role == "fit_frame_selection":
                if (
                    parsed_value.get("schema") != "lewm_go2_selected_render_frames_v1"
                    or parsed_value.get("dataset_role") != "train"
                    or parsed_value.get("g2_images_opened") is not False
                    or parsed_value.get("g2_label_shards_opened") is not False
                ):
                    _fail("fit frame-selection provenance changed")
            elif role == "physical_geometry_contract":
                if canonical_json_sha256(parsed_value) != geometry_commitment["semantic_sha256"]:
                    _fail("physical geometry semantic SHA-256 changed")
                camera = _mapping(parsed_value.get("camera"), context="physical geometry camera")
                configuration = _mapping(
                    parsed_value.get("configuration_space"),
                    context="physical geometry configuration_space",
                )
                if (
                    not math.isclose(
                        _finite_number(
                            configuration.get("oracle_cell_size_m"),
                            context="physical oracle cell size",
                        ),
                        0.05,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not isinstance(configuration.get("landmarks_are_obstacles"), bool)
                    or not isinstance(configuration.get("distractors_are_obstacles"), bool)
                    or not math.isclose(
                        _finite_number(camera.get("horizontal_fov_deg"), context="physical camera FOV"),
                        78.323,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(
                        _finite_number(camera.get("near_m"), context="physical camera near"),
                        0.05,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    _fail("physical geometry semantic fields changed")
            elif role == "render_audit_contract":
                audit_core = dict(parsed_value)
                embedded = audit_core.pop("content_sha256", None)
                if (
                    embedded != render_audit_commitment["content_sha256"]
                    or canonical_json_sha256(audit_core) != embedded
                ):
                    _fail("render-audit contract canonical content hash changed")
                projection = _mapping(
                    parsed_value.get("camera_projection"),
                    context="render-audit camera_projection",
                )
                objects = _mapping(
                    parsed_value.get("object_contract"),
                    context="render-audit object_contract",
                )
                contact = {
                    "g2_row_metadata_read": True,
                    "g2_image_bytes_hashed_for_integrity": True,
                    "g2_images_decoded_or_inspected": False,
                    "g2_image_content_metrics_computed": False,
                    "g2_label_shards_opened": False,
                    "g2_model_outputs_opened": False,
                }
                if (
                    parsed_value.get("schema") != "lewm_go2_selected_render_audit_v1"
                    or projection.get("resolution_wh") != [224, 168]
                    or not math.isclose(
                        _finite_number(projection.get("horizontal_fov_deg"), context="render-audit horizontal FOV"),
                        78.323,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        _finite_number(projection.get("vertical_fov_deg"), context="render-audit vertical FOV"),
                        62.837038636424516,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                    or not math.isclose(
                        _finite_number(projection.get("near_m"), context="render-audit near"),
                        0.05,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or projection.get("runtime_rectification_required") is not False
                    or objects.get("rendered_groups")
                    != ["wall", "obstacle", "landmark", "distractor"]
                    or objects.get("collision_distractors_rendered") is not True
                    or objects.get("full_box_roll_pitch_yaw_rendered") is not True
                    or any(parsed_value.get(name) is not expected for name, expected in contact.items())
                ):
                    _fail("render-audit projection/object/contact contract changed")
        ledger["source_geometry"].append(
            {
                "path": str(resolved),
                "semantic_role": role,
                "scene_id": str(entry["scene_id"]),
                "expected_sha256": expected_hash,
                "observed_sha256": observed_by_path[resolved],
                "open_count_for_unique_path": 1,
            }
        )
    if parsed_paths != set(commitments):
        _fail("not every unique machine source-geometry path was verified")
    return _validate_source_scene_provenance(
        machine_entries=machine_entries,
        parsed_by_path=parsed_by_path,
        panel_records=panel_records,
        selected_scene_families=selected_scene_families,
    )


def _known_input_records() -> dict[str, dict[str, str]]:
    return {
        "fit_panel": {
            "path": str(FIT_PANEL_PATH),
            "file_sha256": FIT_PANEL_FILE_SHA256,
            "content_sha256": FIT_PANEL_CONTENT_SHA256,
            "fit_rows_sha256": FIT_ROWS_SHA256,
        },
        "v4_adjudication_report": {
            "path": str(V4_ADJUDICATION_REPORT_PATH),
            "file_sha256": V4_ADJUDICATION_REPORT_SHA256,
        },
        "known_bias_proof": {
            "path": str(KNOWN_BIAS_PROOF_PATH),
            "file_sha256": KNOWN_BIAS_PROOF_SHA256,
        },
        "preflight_access_incident": {
            "path": str(INCIDENT_PATH),
            "file_sha256": INCIDENT_SHA256,
            "status": INCIDENT_STATUS,
        },
    }


def _validate_inputs(
    value: object,
    *,
    expected_inputs: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    inputs = _record(
        value,
        {
            "fit_panel",
            "human_implementation_manifest",
            "machine_implementation_manifest",
            "preflight_access_incident",
            "v4_adjudication_report",
            "known_bias_proof",
            "geometry_contract",
        },
        context="inputs",
    )
    for name, expected in _known_input_records().items():
        record = _record(inputs[name], set(expected), context=f"inputs.{name}")
        for key, registered in expected.items():
            if key.endswith("sha256"):
                _sha256(record[key], context=f"inputs.{name}.{key}")
            _require_equal(record[key], registered, context=f"inputs.{name}.{key}")

    human_manifest = _record(
        inputs["human_implementation_manifest"],
        {"path", "file_sha256"},
        context="inputs.human_implementation_manifest",
    )
    _require_equal(
        human_manifest["path"],
        str(IMPLEMENTATION_MANIFEST_PATH),
        context="inputs.human_implementation_manifest.path",
    )
    _sha256(
        human_manifest["file_sha256"],
        context="inputs.human_implementation_manifest.file_sha256",
    )
    machine_manifest = _record(
        inputs["machine_implementation_manifest"],
        {"path", "file_sha256", "content_sha256", "schema"},
        context="inputs.machine_implementation_manifest",
    )
    _require_equal(
        machine_manifest["path"],
        str(MACHINE_IMPLEMENTATION_MANIFEST_PATH),
        context="inputs.machine_implementation_manifest.path",
    )
    _require_equal(
        machine_manifest["schema"],
        "lewm_go2_n32_camera_frustum_observability_audit_implementation_manifest_v1",
        context="inputs.machine_implementation_manifest.schema",
    )
    _sha256(
        machine_manifest["file_sha256"],
        context="inputs.machine_implementation_manifest.file_sha256",
    )
    _sha256(
        machine_manifest["content_sha256"],
        context="inputs.machine_implementation_manifest.content_sha256",
    )
    geometry = _record(
        inputs["geometry_contract"],
        {"path", "file_sha256", "semantic_sha256"},
        context="inputs.geometry_contract",
    )
    geometry_path_text = _string(
        geometry["path"], context="inputs.geometry_contract.path"
    )
    geometry_path = Path(geometry_path_text)
    if not geometry_path.is_absolute():
        _fail("inputs.geometry_contract.path must be absolute")
    if expected_inputs is not None and geometry_path_text != str(
        expected_inputs["geometry_contract"]["path"]
    ):
        _fail("inputs.geometry_contract.path differs from the authorized record")
    _sha256(geometry["file_sha256"], context="inputs.geometry_contract.file_sha256")
    _sha256(geometry["semantic_sha256"], context="inputs.geometry_contract.semantic_sha256")

    if expected_inputs is not None and inputs != expected_inputs:
        _fail("inputs differ from the independently verified input records")
    return inputs


MACHINE_MANIFEST_SCHEMA = (
    "lewm_go2_n32_camera_frustum_observability_audit_implementation_manifest_v1"
)
MACHINE_MANIFEST_FIELDS = {
    "schema",
    "created_at_utc",
    "binding",
    "preflight_access_incident",
    "human_implementation_manifest",
    "authorized_inputs",
    "source_map",
    "runtime_environment",
    "verification_evidence",
    "exclusive_output",
    "preparation_access_ledger",
    "review",
    "authoritative_fit_audit_authorized",
    "content_sha256",
}
EXPECTED_AUDIT_TEST_COUNT = 145
REQUIRED_VERIFICATION_COMMANDS = (
    {
        "category": "pytest",
        "command": (
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
            "PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3:"
            "/home/andrewknowles/Workspace/LeWMQuad-v3/lewm_worlds "
            "/usr/bin/python3 -m pytest -q "
            "lewm/tests/test_go2_n32_camera_frustum_observability.py "
            "lewm/tests/test_go2_n32_camera_frustum_fit_evidence_stdlib.py "
            "lewm/tests/test_audit_go2_n32_camera_frustum_observability.py "
            "lewm/tests/test_finalize_go2_n32_camera_frustum_observability.py"
        ),
        "deterministic_result": {
            "kind": "passed_test_count",
            "count": EXPECTED_AUDIT_TEST_COUNT,
        },
    },
    {
        "category": "py_compile",
        "command": (
            "/home/andrewknowles/TinyQuadJEPA/bin/python -m py_compile "
            "lewm/benchmarks/go2_n32_camera_frustum_observability.py "
            "lewm/benchmarks/go2_n32_camera_frustum_fit_evidence_stdlib.py "
            "scripts/audit_go2_n32_camera_frustum_observability.py "
            "scripts/finalize_go2_n32_camera_frustum_observability.py"
        ),
        "deterministic_result": {
            "kind": "compiled_file_count",
            "count": 4,
        },
    },
    {
        "category": "import_isolation",
        "command": (
            "PYTHONNOUSERSITE=1 /usr/bin/python3 -c \"import sys; "
            "from scripts import finalize_go2_n32_camera_frustum_observability; "
            "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules\""
        ),
        "deterministic_result": {
            "kind": "forbidden_import_count",
            "count": 2,
        },
    },
    {
        "category": "diff_check",
        "command": "git diff --check",
        "deterministic_result": {
            "kind": "whitespace_error_count",
            "count": 0,
        },
    },
)


def _validate_runtime_environment(value: object, *, context: str) -> Mapping[str, Any]:
    record = _record(
        value,
        {
            "python_implementation_name",
            "python_implementation_version",
            "python_version",
            "numpy_version",
        },
        context=context,
    )
    for key in ("python_implementation_name", "python_version", "numpy_version"):
        _string(record[key], context=f"{context}.{key}")
    version = _list(
        record["python_implementation_version"],
        context=f"{context}.python_implementation_version",
    )
    if len(version) != 5:
        _fail(f"{context}.python_implementation_version must contain five fields")
    for index in (0, 1, 2, 4):
        _strict_int(version[index], context=f"{context}.python_implementation_version[{index}]")
    _string(version[3], context=f"{context}.python_implementation_version[3]")
    if dict(record) != _current_runtime_environment():
        _fail(f"{context} differs from the active frozen runtime")
    return record


def _validate_machine_source_map(
    value: object,
    *,
    expected_source_hashes: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, Any]:
    source_map = _record(
        value,
        {"entry_count", "entries", "source_map_sha256"},
        context="machine source_map",
    )
    entries = _list(source_map["entries"], context="machine source_map.entries")
    if _strict_int(
        source_map["entry_count"], context="machine source_map.entry_count"
    ) != len(SOURCE_HASH_KEYS) or len(entries) != len(SOURCE_HASH_KEYS):
        _fail("machine source_map must contain exactly eleven roles")
    expected_hashes = (
        _source_hashes()
        if expected_source_hashes is None
        else expected_source_hashes
    )
    frozen_paths = _source_paths()
    if set(expected_hashes) != SOURCE_HASH_KEYS or set(frozen_paths) != SOURCE_HASH_KEYS:
        _fail("machine source-map roles differ from the frozen eleven-role graph")
    for role in SOURCE_HASH_KEYS:
        if str(expected_hashes[role].get("path", "")) != str(frozen_paths[role]):
            _fail(f"machine source-map role {role} substituted a different path")
    expected_entries = [
        {
            "role": role,
            "path": str(frozen_paths[role]),
            "sha256": expected_hashes[role]["sha256"],
        }
        for role in sorted(SOURCE_HASH_KEYS)
    ]
    parsed = []
    for index, item in enumerate(entries):
        entry = _record(
            item,
            {"role", "path", "sha256"},
            context=f"machine source_map.entries[{index}]",
        )
        _string(entry["role"], context=f"machine source role {index}")
        _string(entry["path"], context=f"machine source path {index}")
        _sha256(entry["sha256"], context=f"machine source SHA-256 {index}")
        parsed.append(dict(entry))
    if parsed != expected_entries:
        _fail("machine source_map differs from the current exact eleven-role graph")
    declared = _sha256(
        source_map["source_map_sha256"], context="machine source_map SHA-256"
    )
    if declared != canonical_json_sha256(entries):
        _fail("machine source_map canonical hash mismatch")
    return source_map


def _validate_machine_input_leaf(
    value: object,
    *,
    context: str,
    expected_role: str,
    extra_fields: set[str] | None = None,
) -> Mapping[str, Any]:
    fields = {"semantic_role", "path", "file_sha256"} | (extra_fields or set())
    record = _record(value, fields, context=context)
    _require_equal(record["semantic_role"], expected_role, context=f"{context}.semantic_role")
    _string(record["path"], context=f"{context}.path")
    _sha256(record["file_sha256"], context=f"{context}.file_sha256")
    for key in extra_fields or set():
        if key.endswith("sha256"):
            _sha256(record[key], context=f"{context}.{key}")
        else:
            _string(record[key], context=f"{context}.{key}")
    return record


def _validate_machine_inventory(
    value: object,
    *,
    context: str,
    expected_count: int | None = None,
) -> Mapping[str, Any]:
    manifest = _record(
        value, {"entry_count", "entries", "manifest_sha256"}, context=context
    )
    entries = _list(manifest["entries"], context=f"{context}.entries")
    if _strict_int(manifest["entry_count"], context=f"{context}.entry_count") != len(entries):
        _fail(f"{context} entry count mismatch")
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            _fail(f"{context}.entries[{index}] must be an object")
        canonical_json_sha256(item)
    if expected_count is not None and len(entries) != expected_count:
        _fail(f"{context} must contain exactly {expected_count} entries")
    declared = _sha256(manifest["manifest_sha256"], context=f"{context}.manifest_sha256")
    if declared != canonical_json_sha256(entries):
        _fail(f"{context} manifest hash mismatch")
    return manifest


def _validate_machine_authorized_inputs(value: object) -> Mapping[str, Any]:
    inputs = _record(
        value,
        {
            "fit_panel",
            "v4_adjudication_report",
            "known_bias_proof",
            "physical_geometry_contract",
            "label_shards",
            "render_summaries",
            "source_geometry",
        },
        context="machine authorized_inputs",
    )
    fit_panel = _validate_machine_input_leaf(
        inputs["fit_panel"],
        context="machine authorized_inputs.fit_panel",
        expected_role="fit_panel",
        extra_fields={"content_sha256", "schema", "fit_rows_sha256"},
    )
    if dict(fit_panel) != {
        "semantic_role": "fit_panel",
        "path": str(FIT_PANEL_PATH),
        "file_sha256": FIT_PANEL_FILE_SHA256,
        "content_sha256": FIT_PANEL_CONTENT_SHA256,
        "schema": "lewm_go2_physical_micro_overfit_panel_v1",
        "fit_rows_sha256": FIT_ROWS_SHA256,
    }:
        _fail("machine fit-panel commitment differs from the frozen panel")
    v4_report = _validate_machine_input_leaf(
        inputs["v4_adjudication_report"],
        context="machine authorized_inputs.v4_adjudication_report",
        expected_role="v4_adjudication_report",
    )
    if dict(v4_report) != {
        "semantic_role": "v4_adjudication_report",
        "path": str(V4_ADJUDICATION_REPORT_PATH),
        "file_sha256": V4_ADJUDICATION_REPORT_SHA256,
    }:
        _fail("machine V4 adjudication commitment changed")
    bias_proof = _validate_machine_input_leaf(
        inputs["known_bias_proof"],
        context="machine authorized_inputs.known_bias_proof",
        expected_role="known_bias_proof",
    )
    if dict(bias_proof) != {
        "semantic_role": "known_bias_proof",
        "path": str(KNOWN_BIAS_PROOF_PATH),
        "file_sha256": KNOWN_BIAS_PROOF_SHA256,
    }:
        _fail("machine KNOWN-bias proof commitment changed")
    _validate_machine_input_leaf(
        inputs["physical_geometry_contract"],
        context="machine authorized_inputs.physical_geometry_contract",
        expected_role="physical_geometry_contract",
        extra_fields={"semantic_sha256", "schema"},
    )
    label_shards = _validate_machine_inventory(
        inputs["label_shards"],
        context="machine authorized_inputs.label_shards",
        expected_count=EXPECTED_LABEL_SHARD_COUNT,
    )
    render_summaries = _validate_machine_inventory(
        inputs["render_summaries"],
        context="machine authorized_inputs.render_summaries",
        expected_count=20,
    )
    # Source geometry contains several exact semantic roles, so validate its
    # common fields here and its role/scene assignment during reconciliation.
    source_geometry = _record(
        inputs["source_geometry"],
        {"entry_count", "entries", "manifest_sha256"},
        context="machine authorized_inputs.source_geometry",
    )
    entries = _list(
        source_geometry["entries"], context="machine authorized_inputs.source_geometry.entries"
    )
    if _strict_int(
        source_geometry["entry_count"],
        context="machine authorized_inputs.source_geometry.entry_count",
    ) != len(entries):
        _fail("machine source-geometry inventory count mismatch")
    parsed_entries = []
    for index, item in enumerate(entries):
        entry = _record(
            item,
            {"semantic_role", "path", "sha256", "scene_id"},
            context=f"machine authorized_inputs.source_geometry.entries[{index}]",
        )
        role = _string(entry["semantic_role"], context="machine source-geometry role")
        if role not in ALLOWED_SEMANTIC_ROLES or role in {
            "fit_panel",
            "fit_label_shard",
            "audit_output",
        }:
            _fail("machine source-geometry entry has an invalid semantic role")
        _string(entry["path"], context="machine source-geometry path")
        _sha256(entry["sha256"], context="machine source-geometry SHA-256")
        _string(entry["scene_id"], context="machine source-geometry scene_id")
        parsed_entries.append(dict(entry))
    if parsed_entries != sorted(
        parsed_entries,
        key=lambda entry: (
            str(entry["path"]), str(entry["semantic_role"]), str(entry["scene_id"])
        ),
    ):
        _fail("machine source-geometry inventory is not canonically ordered")
    if _sha256(
        source_geometry["manifest_sha256"],
        context="machine source-geometry manifest SHA-256",
    ) != canonical_json_sha256(entries):
        _fail("machine source-geometry manifest hash mismatch")
    return inputs


def _validate_machine_manifest(
    value: object,
    *,
    raw_file_sha256: str,
    expected_source_hashes: Mapping[str, Mapping[str, str]] | None = None,
) -> Mapping[str, Any]:
    manifest = _record(value, MACHINE_MANIFEST_FIELDS, context="machine manifest")
    _require_equal(manifest["schema"], MACHINE_MANIFEST_SCHEMA, context="machine manifest schema")
    _validate_timestamp(manifest["created_at_utc"])
    binding = _record(
        manifest["binding"], {"path", "file_sha256"}, context="machine manifest binding"
    )
    _require_equal(binding["path"], str(BINDING_PATH), context="machine manifest binding path")
    _require_equal(
        binding["file_sha256"], EXECUTION_BINDING_SHA256, context="machine manifest binding SHA-256"
    )
    incident = _record(
        manifest["preflight_access_incident"],
        {"path", "file_sha256", "status"},
        context="machine manifest incident",
    )
    _require_equal(
        dict(incident),
        _known_input_records()["preflight_access_incident"],
        context="machine manifest incident",
    )
    human = _record(
        manifest["human_implementation_manifest"],
        {"path", "file_sha256"},
        context="machine manifest human report",
    )
    _require_equal(human["path"], str(IMPLEMENTATION_MANIFEST_PATH), context="human report path")
    _sha256(human["file_sha256"], context="human report SHA-256")
    _validate_machine_authorized_inputs(manifest["authorized_inputs"])
    _validate_machine_source_map(
        manifest["source_map"], expected_source_hashes=expected_source_hashes
    )
    _validate_runtime_environment(manifest["runtime_environment"], context="machine runtime_environment")
    verification = _mapping(
        manifest["verification_evidence"], context="machine verification_evidence"
    )
    if verification.get("all_passed") is not True:
        _fail("machine verification evidence does not report all commands passed")
    commands = _list(
        verification.get("commands"), context="machine verification_evidence.commands"
    )
    if len(commands) != len(REQUIRED_VERIFICATION_COMMANDS):
        _fail("machine verification evidence must contain the exact four commands")
    for index, (command, expected) in enumerate(
        zip(commands, REQUIRED_VERIFICATION_COMMANDS)
    ):
        record = _record(
            command,
            {
                "category",
                "command",
                "exit_code",
                "deterministic_result",
                "captured_output_sha256",
            },
            context=f"machine verification command {index}",
        )
        _require_equal(
            record["category"],
            expected["category"],
            context=f"machine verification command {index}.category",
        )
        _require_equal(
            record["command"],
            expected["command"],
            context=f"machine verification command {index}.command",
        )
        if _strict_int(
            record["exit_code"], context=f"machine verification command {index}.exit_code"
        ) != 0:
            _fail("machine verification evidence contains a failed command")
        _sha256(
            record["captured_output_sha256"],
            context=f"machine verification command {index}.captured_output_sha256",
        )
        deterministic = _record(
            record["deterministic_result"],
            {"kind", "count"},
            context=f"machine verification command {index}.deterministic_result",
        )
        _string(
            deterministic["kind"],
            context=f"machine verification command {index}.deterministic_result.kind",
        )
        _strict_int(
            deterministic["count"],
            context=f"machine verification command {index}.deterministic_result.count",
        )
        if dict(deterministic) != expected["deterministic_result"]:
            _fail(
                f"machine verification command {index}.deterministic_result changed"
            )
    output = _record(
        manifest["exclusive_output"],
        {"path", "schema", "absent_before_authorization", "zero_output_state"},
        context="machine exclusive_output",
    )
    _require_equal(output["path"], str(RESULT_PATH), context="machine exclusive output path")
    _require_equal(output["schema"], RESULT_SCHEMA, context="machine exclusive output schema")
    if not _strict_bool(
        output["absent_before_authorization"],
        context="machine absent_before_authorization",
    ):
        _fail("machine manifest does not prove absent pre-run output")
    if not _strict_bool(
        output["zero_output_state"], context="machine zero_output_state"
    ):
        _fail("machine manifest does not prove zero pre-run output")
    preparation = _mapping(
        manifest["preparation_access_ledger"], context="machine preparation_access_ledger"
    )
    if preparation.get("passes") is not True or preparation.get("forbidden_counters_zero") is not True:
        _fail("machine preparation_access_ledger does not pass")
    _validate_fresh_ledger_zero_denials(
        preparation, context="machine preparation_access_ledger"
    )
    for field in (
        "label_shard_hash_byte_opens",
        "label_shard_npz_opens",
        "registered_arrays_decompressed",
        "materialized_label_rows",
        "materialized_supervision_rows",
        "selected_label_rows_read",
        "selected_supervision_rows_read",
    ):
        if _strict_int(preparation.get(field), context=f"machine preparation ledger {field}") != 0:
            _fail(f"machine preparation ledger {field} crossed the metadata boundary")
    source_entries = manifest["authorized_inputs"]["source_geometry"]["entries"]
    unique_source_paths = {str(entry["path"]) for entry in source_entries}
    expected_preparation_counts = {
        "panel_metadata_byte_opens": 1,
        "implementation_source_hash_byte_opens": 2 * len(SOURCE_HASH_KEYS),
        "document_hash_byte_opens": 4,
        "source_geometry_hash_byte_opens": 2 * len(unique_source_paths),
        "source_geometry_json_parses": sum(
            Path(path).suffix.lower() in {".json", ".jsonl"}
            for path in unique_source_paths
        ),
        "source_frame_records_selected": EXPECTED_FRAME_COUNT,
    }
    for field, expected in expected_preparation_counts.items():
        if _strict_int(preparation.get(field), context=f"machine preparation ledger {field}") != expected:
            _fail(f"machine preparation ledger {field} must equal {expected}")
    if _strict_int(
        preparation.get("source_geometry_jsonl_records"),
        context="machine preparation ledger source_geometry_jsonl_records",
    ) < EXPECTED_FRAME_COUNT:
        _fail("machine preparation ledger scanned too few source JSONL records")
    if preparation.get("per_shard_materialization") != []:
        _fail("machine preparation ledger materialized a label shard")
    review = _record(
        manifest["review"], {"reviewer_identity", "status"}, context="machine review"
    )
    _string(review["reviewer_identity"], context="machine reviewer_identity")
    _require_equal(
        review["status"], "reviewed_and_authorized", context="machine review status"
    )
    if not _strict_bool(
        manifest["authoritative_fit_audit_authorized"],
        context="machine authoritative_fit_audit_authorized",
    ):
        _fail("machine manifest does not authorize the fit audit")
    content_hash = _sha256(manifest["content_sha256"], context="machine manifest content_sha256")
    core = dict(manifest)
    del core["content_sha256"]
    if content_hash != canonical_json_sha256(core):
        _fail("machine manifest canonical content hash mismatch")
    _sha256(raw_file_sha256, context="machine manifest raw file SHA-256")
    return manifest


def _machine_declared_source_hashes(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    entries = _list(manifest["source_map"]["entries"], context="machine source_map entries")
    return {
        str(entry["role"]): {
            "path": str(entry["path"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in entries
    }


def _expected_result_inputs_from_machine(
    manifest: Mapping[str, Any], *, machine_file_sha256: str
) -> dict[str, Any]:
    authorized = manifest["authorized_inputs"]
    panel = authorized["fit_panel"]
    geometry = authorized["physical_geometry_contract"]
    return {
        "fit_panel": {
            "path": panel["path"],
            "file_sha256": panel["file_sha256"],
            "content_sha256": panel["content_sha256"],
            "fit_rows_sha256": panel["fit_rows_sha256"],
        },
        "human_implementation_manifest": dict(
            manifest["human_implementation_manifest"]
        ),
        "machine_implementation_manifest": {
            "path": str(MACHINE_IMPLEMENTATION_MANIFEST_PATH),
            "file_sha256": machine_file_sha256,
            "content_sha256": manifest["content_sha256"],
            "schema": manifest["schema"],
        },
        "preflight_access_incident": dict(manifest["preflight_access_incident"]),
        "v4_adjudication_report": {
            "path": authorized["v4_adjudication_report"]["path"],
            "file_sha256": authorized["v4_adjudication_report"]["file_sha256"],
        },
        "known_bias_proof": {
            "path": authorized["known_bias_proof"]["path"],
            "file_sha256": authorized["known_bias_proof"]["file_sha256"],
        },
        "geometry_contract": {
            "path": geometry["path"],
            "file_sha256": geometry["file_sha256"],
            "semantic_sha256": geometry["semantic_sha256"],
        },
    }


def _independently_verify_machine_and_documents(
    *,
    machine_manifest_sha256: str,
    ledger: dict[str, Any],
) -> tuple[Mapping[str, Any], dict[str, dict[str, str]], dict[str, Any]]:
    expected_machine_hash = _sha256(
        machine_manifest_sha256, context="explicit machine manifest SHA-256"
    )
    bootstrap_allowlist = {MACHINE_IMPLEMENTATION_MANIFEST_PATH: expected_machine_hash}
    machine_path, _expected = _authorize_path(
        MACHINE_IMPLEMENTATION_MANIFEST_PATH,
        requested_role="machine_implementation_manifest",
        declared_role="machine_implementation_manifest",
        modality="json",
        allowlist=bootstrap_allowlist,
        ledger=ledger,
    )
    ledger["document_hash_byte_opens"] += 1
    machine_raw = machine_path.read_bytes()
    observed_machine_hash = hashlib.sha256(machine_raw).hexdigest()
    if observed_machine_hash != expected_machine_hash:
        _fail("machine implementation manifest raw-byte SHA-256 mismatch")
    machine_payload = _strict_json_loads(machine_raw, context="machine implementation manifest")
    if not isinstance(machine_payload, Mapping):
        _fail("machine implementation manifest root must be an object")
    canonical_machine_raw = json.dumps(
        machine_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if machine_raw != canonical_machine_raw:
        _fail("machine implementation manifest is not canonical compact JSON bytes")
    declared_sources = _machine_declared_source_hashes(machine_payload)
    machine = _validate_machine_manifest(
        machine_payload,
        raw_file_sha256=observed_machine_hash,
        expected_source_hashes=declared_sources,
    )

    source_entries = machine["source_map"]["entries"]
    source_allowlist = {
        Path(str(entry["path"])).absolute(): str(entry["sha256"])
        for entry in source_entries
    }
    observed_sources: dict[str, dict[str, str]] = {}
    for entry in source_entries:
        role = str(entry["role"])
        modality = ALLOWED_MODALITIES_BY_ROLE.get(role)
        if modality is None:
            _fail("machine source-map role has no registered modality")
        path, expected_hash = _authorize_path(
            Path(str(entry["path"])),
            requested_role=role,
            declared_role=role,
            modality=modality,
            allowlist=source_allowlist,
            ledger=ledger,
        )
        ledger["document_hash_byte_opens"] += 1
        observed_hash = _sha256_file(path)
        if observed_hash != expected_hash:
            _fail(f"source-map role {role} raw-byte SHA-256 mismatch")
        observed_sources[role] = {"path": str(path), "sha256": observed_hash}
    _validate_machine_source_map(
        machine["source_map"], expected_source_hashes=observed_sources
    )

    document_specs = (
        (
            machine["preflight_access_incident"],
            "incident_record",
            "markdown",
        ),
        (
            machine["human_implementation_manifest"],
            "human_implementation_manifest",
            "markdown",
        ),
        (
            machine["authorized_inputs"]["v4_adjudication_report"],
            "v4_adjudication_report",
            "markdown",
        ),
        (
            machine["authorized_inputs"]["known_bias_proof"],
            "known_bias_proof",
            "markdown",
        ),
    )
    document_allowlist = {
        Path(str(record["path"])).absolute(): str(record["file_sha256"])
        for record, _role, _modality in document_specs
    }
    already_opened = {Path(record["path"]) for record in observed_sources.values()}
    for record, role, modality in document_specs:
        path, expected_hash = _authorize_path(
            Path(str(record["path"])),
            requested_role=role,
            declared_role=role,
            modality=modality,
            allowlist=document_allowlist,
            ledger=ledger,
        )
        if path not in already_opened:
            ledger["document_hash_byte_opens"] += 1
            if _sha256_file(path) != expected_hash:
                _fail(f"bound document {role} raw-byte SHA-256 mismatch")
            already_opened.add(path)
    expected_inputs = _expected_result_inputs_from_machine(
        machine, machine_file_sha256=observed_machine_hash
    )
    return machine, observed_sources, expected_inputs


def _independently_load_fit_panel(
    *,
    machine_manifest: Mapping[str, Any],
    ledger: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commitment = machine_manifest["authorized_inputs"]["fit_panel"]
    allowlist = {
        Path(str(commitment["path"])).absolute(): str(
            commitment["file_sha256"]
        )
    }
    path, expected_hash = _authorize_path(
        Path(str(commitment["path"])),
        requested_role="fit_panel",
        declared_role="train",
        modality="json",
        allowlist=allowlist,
        ledger=ledger,
    )
    ledger["panel_metadata_byte_opens"] += 1
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        _fail("fit panel raw-byte SHA-256 mismatch")
    panel = _strict_json_loads(raw, context="fit panel")
    records, metadata = _derive_panel_records(
        panel,
        expected_content_sha256=str(commitment["content_sha256"]),
        expected_rows_sha256=str(commitment["fit_rows_sha256"]),
    )
    return records, metadata


def _validate_execution_binding(value: object) -> None:
    record = _record(value, {"path", "sha256"}, context="execution_binding")
    _require_equal(record["path"], str(BINDING_PATH), context="execution_binding.path")
    _require_equal(
        _sha256(record["sha256"], context="execution_binding.sha256"),
        EXECUTION_BINDING_SHA256,
        context="execution_binding.sha256",
    )


def _validate_source_hashes(
    value: object,
    *,
    expected_source_hashes: Mapping[str, Any],
) -> Mapping[str, Any]:
    sources = _record(value, SOURCE_HASH_KEYS, context="source_hashes")
    for name in sorted(SOURCE_HASH_KEYS):
        record = _record(
            sources[name], {"path", "sha256"}, context=f"source_hashes.{name}"
        )
        _string(record["path"], context=f"source_hashes.{name}.path")
        _sha256(record["sha256"], context=f"source_hashes.{name}.sha256")
    if sources != expected_source_hashes:
        _fail("source_hashes do not match the current complete source map")
    return sources


def _expected_geometry_contract() -> dict[str, Any]:
    half_fov = math.radians(78.323 / 2.0)
    return {
        "schema": GEOMETRY_SCHEMA,
        "cartesian_shape": [64, 64],
        "cartesian_cell_size_m": 0.1,
        "cartesian_forward_min_edge_m": -1.0,
        "cartesian_left_min_edge_m": -3.2,
        "camera_xyz_body_m": [0.326, 0.0, 0.043],
        "camera_rpy_body_rad": [0.0, 0.0, 0.0],
        "camera_near_m": 0.05,
        "horizontal_fov_deg": 78.323,
        "half_horizontal_fov_rad": half_fov,
        "vertical_fov_deg": 62.8370386364,
        "vertical_anchor_z_body_m": [-0.333, -0.133, 0.067, 0.267, 0.467],
        "range_bin_count": 64,
        "range_bin_size_m": 0.1,
        "range_interval_m": [0.0, 6.4],
        "range_interval_convention": "left_closed_right_open",
        "angular_bin_count": 256,
        "angular_interval_rad": [-half_fov, half_fov],
        "angular_interval_convention": "closed_positive_edge_in_final_bin",
        "cartesian_sample": "cell_center",
        "unsupported_mapping_sentinel": [-1, -1],
        "mapping_dtype": "little_endian_signed_int16",
        "support_mask_dtype": "row_major_uint8",
        "class_order": list(CLASS_NAMES),
        "class_ids": [0, 1, 2],
        "family_order": list(FAMILIES),
        "endpoint_side_order": list(ENDPOINT_SIDES),
    }


def _validate_geometry_contract(value: object) -> None:
    expected = _expected_geometry_contract()
    record = _record(value, set(expected), context="geometry_contract")
    # Validate scalar types first so bool cannot pass equality against 0/1.
    for key in (
        "cartesian_cell_size_m",
        "cartesian_forward_min_edge_m",
        "cartesian_left_min_edge_m",
        "camera_near_m",
        "horizontal_fov_deg",
        "half_horizontal_fov_rad",
        "vertical_fov_deg",
        "range_bin_size_m",
    ):
        _finite_number(record[key], context=f"geometry_contract.{key}")
    for key in ("range_bin_count", "angular_bin_count"):
        _strict_int(record[key], context=f"geometry_contract.{key}")
    _require_equal(record, expected, context="geometry_contract")


SPAN_SUMMARY = {
    "column_count": 256,
    "participating_sample_count": 13204,
    "columns_with_span_count": 244,
    "columns_with_fewer_than_two_participants_count": 12,
    "span_rad": {
        "p50": 0.21925072264599688,
        "p95": 0.4024189060438979,
        "maximum": 0.4594157473075444,
    },
    "span_deg": {
        "p50": 12.562141062809003,
        "p95": 23.056904912586965,
        "maximum": 26.32258336257101,
    },
    "span_new_angular_bins": {
        "p50": 41.05956247946459,
        "p95": 75.36186889703234,
        "maximum": 86.03579205109838,
    },
    "columns_span_ge_1_new_bin": 242,
    "columns_span_ge_2_new_bins": 238,
    "columns_span_ge_4_new_bins": 232,
    "columns_span_ge_8_new_bins": 222,
    "quantile_method": "numpy_linear_float64",
}
SPAN_COLUMN_FIELDS = {
    "body_angular_bin",
    "body_bearing_center_rad",
    "body_bearing_center_deg",
    "participating_range_count",
    "minimum_camera_bearing_rad",
    "maximum_camera_bearing_rad",
    "span_rad",
    "span_deg",
    "span_new_angular_bins",
}


def _validate_span_columns(value: object, *, context: str) -> list[Any]:
    columns = _list(value, context=context)
    if len(columns) != 256:
        _fail(f"{context} must contain exactly 256 columns")
    angular_width = 2.0 * math.radians(78.323 / 2.0) / 256.0
    for index, item in enumerate(columns):
        column = _record(item, SPAN_COLUMN_FIELDS, context=f"{context}[{index}]")
        if _strict_int(
            column["body_angular_bin"], context=f"{context}[{index}].body_angular_bin"
        ) != index:
            _fail(f"{context} body angular bins are not in canonical order")
        count = _strict_int(
            column["participating_range_count"],
            context=f"{context}[{index}].participating_range_count",
            maximum=64,
        )
        bearing = _finite_number(
            column["body_bearing_center_rad"],
            context=f"{context}[{index}].body_bearing_center_rad",
        )
        degrees = _finite_number(
            column["body_bearing_center_deg"],
            context=f"{context}[{index}].body_bearing_center_deg",
        )
        if not math.isclose(degrees, math.degrees(bearing), rel_tol=0.0, abs_tol=1e-12):
            _fail(f"{context}[{index}] bearing degree conversion is inconsistent")
        span_fields = (
            "minimum_camera_bearing_rad",
            "maximum_camera_bearing_rad",
            "span_rad",
            "span_deg",
            "span_new_angular_bins",
        )
        if count < 2:
            if any(column[field] is not None for field in span_fields):
                _fail(f"{context}[{index}] must use null spans")
            continue
        values = {
            field: _finite_number(column[field], context=f"{context}[{index}].{field}")
            for field in span_fields
        }
        if values["minimum_camera_bearing_rad"] > values["maximum_camera_bearing_rad"]:
            _fail(f"{context}[{index}] has a reversed bearing range")
        span = values["maximum_camera_bearing_rad"] - values["minimum_camera_bearing_rad"]
        if not math.isclose(span, values["span_rad"], rel_tol=0.0, abs_tol=1e-15):
            _fail(f"{context}[{index}] span is inconsistent")
        if not math.isclose(
            math.degrees(span), values["span_deg"], rel_tol=0.0, abs_tol=1e-12
        ):
            _fail(f"{context}[{index}] span degree conversion is inconsistent")
        if not math.isclose(
            span / angular_width,
            values["span_new_angular_bins"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _fail(f"{context}[{index}] angular-bin span is inconsistent")
    return columns


def _validate_span_audit(value: object) -> None:
    audit = _record(
        value,
        {"schema", "geometry", "primary", "horizontal_only", "old_column_span_table_sha256"},
        context="old_body_column_span_audit",
    )
    _require_equal(audit["schema"], OLD_COLUMN_SPAN_SCHEMA, context="span schema")
    half_fov = math.radians(78.323 / 2.0)
    expected_geometry = {
        "old_body_radius_centers_m": [(index + 0.5) * 0.1 for index in range(64)],
        "old_body_bearing_bin_count": 256,
        "old_body_bearing_range_rad": [-half_fov, half_fov],
        "new_angular_bin_width_rad": 2.0 * half_fov / 256.0,
        "primary_vertical_anchor_rule": "at_least_one_registered_anchor_valid",
    }
    geometry = _record(
        audit["geometry"], set(expected_geometry), context="old_body_column_span_audit.geometry"
    )
    _strict_int(
        geometry["old_body_bearing_bin_count"],
        context="old_body_column_span_audit.geometry.old_body_bearing_bin_count",
    )
    _require_equal(geometry, expected_geometry, context="old body-column geometry")
    scopes: dict[str, list[Any]] = {}
    for name in ("primary", "horizontal_only"):
        scope = _record(
            audit[name], {"columns", "summary"}, context=f"old_body_column_span_audit.{name}"
        )
        columns = _validate_span_columns(
            scope["columns"], context=f"old_body_column_span_audit.{name}.columns"
        )
        summary = _record(
            scope["summary"], set(SPAN_SUMMARY), context=f"old_body_column_span_audit.{name}.summary"
        )
        for key in (
            "column_count",
            "participating_sample_count",
            "columns_with_span_count",
            "columns_with_fewer_than_two_participants_count",
            "columns_span_ge_1_new_bin",
            "columns_span_ge_2_new_bins",
            "columns_span_ge_4_new_bins",
            "columns_span_ge_8_new_bins",
        ):
            _strict_int(summary[key], context=f"old_body_column_span_audit.{name}.summary.{key}")
        _require_equal(summary, SPAN_SUMMARY, context=f"old body-column {name} summary")
        scopes[name] = columns
    if audit["primary"] != audit["horizontal_only"]:
        _fail("primary and horizontal-only span tables differ from the frozen result")
    declared = _sha256(
        audit["old_column_span_table_sha256"],
        context="old_body_column_span_audit.old_column_span_table_sha256",
    )
    table = {
        "primary_with_vertical_anchor": scopes["primary"],
        "horizontal_only": scopes["horizontal_only"],
    }
    if declared != canonical_json_sha256(table) or declared != OLD_SPAN_TABLE_SHA256:
        _fail("old body-column span table hash mismatch")
    if audit != _stdlib_old_body_column_span_audit():
        _fail("old body-column span audit differs from stdlib recomputation")


EXPECTED_MAPPING_AUDIT = {
    "schema": MAPPING_AUDIT_SCHEMA,
    "mapping_sha256": MAPPING_SHA256,
    "support_mask_sha256": SUPPORT_MASK_SHA256,
    "mapping_dtype": "int16",
    "signed_int16": True,
    "supported_cartesian_cell_count": 1990,
    "unsupported_cartesian_cell_count": 2106,
    "unique_used_polar_bin_count": 1990,
    "unused_polar_bin_count": 14394,
    "partially_mapped_entry_count": 0,
    "invalid_negative_entry_count": 0,
    "out_of_range_entry_count": 0,
    "nondeterministic_entry_count": 0,
    "expected_support_mismatch_count": 0,
    "collision_bin_count": 0,
    "collision_extra_cartesian_count": 0,
    "collisions": [],
    "deterministic": True,
    "all_mapped_indices_in_range": True,
    "all_entries_complete": True,
    "support_matches_frozen_geometry": True,
    "injective": True,
    "passes": True,
}


def _validate_mapping_audit(value: object) -> Mapping[str, Any]:
    independently_recomputed = _stdlib_mapping_audit()
    audit = _record(value, set(independently_recomputed), context="mapping_audit")
    for key, expected in independently_recomputed.items():
        if isinstance(expected, bool):
            _strict_bool(audit[key], context=f"mapping_audit.{key}")
        elif isinstance(expected, int):
            _strict_int(audit[key], context=f"mapping_audit.{key}")
    _sha256(audit["mapping_sha256"], context="mapping_audit.mapping_sha256")
    _sha256(audit["support_mask_sha256"], context="mapping_audit.support_mask_sha256")
    _require_equal(
        audit,
        independently_recomputed,
        context="mapping_audit stdlib recomputation",
    )
    return audit


def _validate_manifest(value: object, *, context: str, expected_count: int | None) -> Mapping[str, Any]:
    manifest = _record(
        value, {"entry_count", "entries", "manifest_sha256"}, context=context
    )
    entries = _list(manifest["entries"], context=f"{context}.entries")
    count = _strict_int(manifest["entry_count"], context=f"{context}.entry_count")
    if count != len(entries) or (expected_count is not None and count != expected_count):
        _fail(f"{context} entry count mismatch")
    paths: list[str] = []
    for index, item in enumerate(entries):
        record = _record(item, {"path", "sha256"}, context=f"{context}.entries[{index}]")
        path = _string(record["path"], context=f"{context}.entries[{index}].path")
        if not Path(path).is_absolute() or Path(path).resolve() != Path(path):
            _fail(f"{context}.entries[{index}].path must be absolute and resolved")
        _sha256(record["sha256"], context=f"{context}.entries[{index}].sha256")
        paths.append(path)
    if len(set(paths)) != len(paths):
        _fail(f"{context} contains duplicate paths")
    if paths != sorted(paths):
        _fail(f"{context} entries are not in canonical path order")
    declared = _sha256(manifest["manifest_sha256"], context=f"{context}.manifest_sha256")
    if declared != canonical_json_sha256(entries):
        _fail(f"{context} manifest hash mismatch")
    return manifest


def _validate_source_geometry_manifest(
    value: object,
    *,
    expected_entries: Sequence[Mapping[str, Any]],
    selected_scene_ids: set[str],
) -> Mapping[str, Any]:
    manifest = _record(
        value,
        {"entry_count", "entries", "manifest_sha256"},
        context="source_geometry_manifest",
    )
    entries = _list(manifest["entries"], context="source_geometry_manifest.entries")
    if _strict_int(
        manifest["entry_count"], context="source_geometry_manifest.entry_count"
    ) != len(entries) or not entries:
        _fail("source_geometry_manifest entry count mismatch or empty manifest")
    parsed = []
    assignments: set[tuple[str, str, str]] = set()
    for index, item in enumerate(entries):
        entry = _record(
            item,
            {"path", "sha256", "semantic_role", "scene_id"},
            context=f"source_geometry_manifest.entries[{index}]",
        )
        path = _string(entry["path"], context=f"source geometry path {index}")
        digest = _sha256(entry["sha256"], context=f"source geometry SHA-256 {index}")
        role = _string(entry["semantic_role"], context=f"source geometry role {index}")
        scene_id = _string(entry["scene_id"], context=f"source geometry scene {index}")
        if role not in ALLOWED_SEMANTIC_ROLES or role in {
            "fit_panel",
            "fit_label_shard",
            "audit_output",
        }:
            _fail("source_geometry_manifest contains an invalid semantic role")
        if scene_id not in selected_scene_ids:
            _fail("source_geometry_manifest names a scene outside the selected panel")
        assignment = (path, role, scene_id)
        if assignment in assignments:
            _fail("source_geometry_manifest contains a duplicate role/scene assignment")
        assignments.add(assignment)
        parsed.append(
            {
                "path": path,
                "sha256": digest,
                "semantic_role": role,
                "scene_id": scene_id,
            }
        )
    if parsed != sorted(
        parsed,
        key=lambda entry: (
            entry["path"], entry["semantic_role"], entry["scene_id"]
        ),
    ):
        _fail("source_geometry_manifest is not in canonical assignment order")
    if parsed != [dict(entry) for entry in expected_entries]:
        _fail("source_geometry_manifest differs from the machine-authorized inventory")
    declared = _sha256(
        manifest["manifest_sha256"], context="source_geometry_manifest.manifest_sha256"
    )
    if declared != canonical_json_sha256(entries):
        _fail("source_geometry_manifest canonical hash mismatch")
    return manifest


def _validate_label_shard_manifest(
    value: object,
    *,
    expected_entries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    manifest = _record(
        value,
        {"entry_count", "entries", "manifest_sha256"},
        context="label_shard_manifest",
    )
    entries = _list(manifest["entries"], context="label_shard_manifest.entries")
    if _strict_int(
        manifest["entry_count"], context="label_shard_manifest.entry_count"
    ) != EXPECTED_LABEL_SHARD_COUNT or len(entries) != EXPECTED_LABEL_SHARD_COUNT:
        _fail("label_shard_manifest must contain exactly 20 entries")
    parsed_entries = []
    seen_paths: set[str] = set()
    for index, item in enumerate(entries):
        entry = _record(
            item,
            {
                "path",
                "sha256",
                "selected_tuples",
                "selected_row_count",
                "family_side_counts",
            },
            context=f"label_shard_manifest.entries[{index}]",
        )
        path = _string(entry["path"], context=f"label shard entry {index} path")
        if path in seen_paths:
            _fail("label_shard_manifest contains duplicate paths")
        seen_paths.add(path)
        _sha256(entry["sha256"], context=f"label shard entry {index} SHA-256")
        tuples = _list(
            entry["selected_tuples"],
            context=f"label shard entry {index} selected_tuples",
        )
        for tuple_index, selected in enumerate(tuples):
            values = _list(
                selected,
                context=f"label shard entry {index} selected_tuples[{tuple_index}]",
            )
            if len(values) != 5:
                _fail("label shard selected tuple must contain exactly five fields")
            if values[0] not in FAMILIES or values[3] not in ENDPOINT_SIDES:
                _fail("label shard selected tuple has an unregistered family or side")
            _string(values[1], context="label shard selected tuple scene_id")
            _strict_int(values[2], context="label shard selected tuple global_row")
            _strict_int(values[4], context="label shard selected tuple label_row")
        if _strict_int(
            entry["selected_row_count"],
            context=f"label shard entry {index} selected_row_count",
        ) != len(tuples):
            _fail("label shard selected-row count mismatch")
        family_side_counts = _record(
            entry["family_side_counts"],
            set(FAMILIES),
            context=f"label shard entry {index} family_side_counts",
        )
        for family in FAMILIES:
            side_counts = _record(
                family_side_counts[family],
                set(ENDPOINT_SIDES),
                context=f"label shard entry {index} family_side_counts.{family}",
            )
            for side in ENDPOINT_SIDES:
                _strict_int(
                    side_counts[side],
                    context=f"label shard entry {index} family_side_counts.{family}.{side}",
                )
        parsed_entries.append(dict(entry))
    if [entry["path"] for entry in parsed_entries] != sorted(seen_paths):
        _fail("label_shard_manifest entries are not in canonical path order")
    if parsed_entries != [dict(entry) for entry in expected_entries]:
        _fail("label_shard_manifest differs from panel-derived shard commitments")
    declared = _sha256(
        manifest["manifest_sha256"], context="label_shard_manifest.manifest_sha256"
    )
    if declared != canonical_json_sha256(entries):
        _fail("label_shard_manifest manifest hash mismatch")
    return manifest


def _validate_record_key(value: object, *, context: str) -> Mapping[str, Any]:
    key = _record(value, set(FRAME_IDENTITY_FIELDS), context=context)
    family = _string(key["family"], context=f"{context}.family")
    if family not in FAMILIES:
        _fail(f"{context}.family is not registered")
    side = _string(key["side"], context=f"{context}.side")
    if side not in ENDPOINT_SIDES:
        _fail(f"{context}.side is not registered")
    _string(key["scene_id"], context=f"{context}.scene_id")
    _strict_int(key["global_row"], context=f"{context}.global_row")
    _strict_int(key["label_row"], context=f"{context}.label_row")
    _sha256(key["image_sha256"], context=f"{context}.image_sha256")
    _sha256(key["label_shard_sha256"], context=f"{context}.label_shard_sha256")
    return key


def _validate_class_counts(value: object, *, context: str) -> dict[str, int]:
    record = _record(value, set(CLASS_NAMES), context=context)
    return {
        name: _strict_int(record[name], context=f"{context}.{name}")
        for name in CLASS_NAMES
    }


LABEL_SUPPORT_FRAME_FIELDS = {
    "schema",
    "total_supervised_label_count",
    "supported_label_count",
    "unsupported_label_count",
    "class_counts",
    "by_class",
    "unsupported_free_count",
    "unsupported_occupied_count",
    "unsupported_unknown_count",
    "unsupported_targets_are_all_unknown",
    "violations",
    "passes",
}


def _validate_label_support(
    value: object,
    *,
    context: str,
    frame_count: int,
    expected_key: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    fields = set(LABEL_SUPPORT_FRAME_FIELDS)
    if frame_count != 1:
        fields.add("frame_count")
    support = _record(value, fields, context=context)
    _require_equal(support["schema"], LABEL_SUPPORT_SCHEMA, context=f"{context}.schema")
    if frame_count != 1 and _strict_int(
        support["frame_count"], context=f"{context}.frame_count"
    ) != frame_count:
        _fail(f"{context}.frame_count mismatch")
    total = _strict_int(
        support["total_supervised_label_count"],
        context=f"{context}.total_supervised_label_count",
    )
    supported = _strict_int(
        support["supported_label_count"], context=f"{context}.supported_label_count"
    )
    unsupported = _strict_int(
        support["unsupported_label_count"], context=f"{context}.unsupported_label_count"
    )
    if total != frame_count * CELLS_PER_FRAME:
        _fail(f"{context} total supervised label count mismatch")
    if supported != frame_count * SUPPORTED_CELLS_PER_FRAME:
        _fail(f"{context} supported label count mismatch")
    if unsupported != frame_count * UNSUPPORTED_CELLS_PER_FRAME:
        _fail(f"{context} unsupported label count mismatch")
    class_counts = _validate_class_counts(support["class_counts"], context=f"{context}.class_counts")
    if sum(class_counts.values()) != total:
        _fail(f"{context} class counts do not sum to total")
    by_class = _record(support["by_class"], set(CLASS_NAMES), context=f"{context}.by_class")
    parsed_by_class: dict[str, dict[str, int]] = {}
    for name in CLASS_NAMES:
        record = _record(
            by_class[name], {"total", "supported", "unsupported"}, context=f"{context}.by_class.{name}"
        )
        parsed = {
            field: _strict_int(record[field], context=f"{context}.by_class.{name}.{field}")
            for field in ("total", "supported", "unsupported")
        }
        if parsed["total"] != parsed["supported"] + parsed["unsupported"]:
            _fail(f"{context}.by_class.{name} does not reconcile")
        if parsed["total"] != class_counts[name]:
            _fail(f"{context}.by_class.{name} disagrees with class_counts")
        parsed_by_class[name] = parsed
    if sum(record["supported"] for record in parsed_by_class.values()) != supported:
        _fail(f"{context} supported by-class counts do not reconcile")
    if sum(record["unsupported"] for record in parsed_by_class.values()) != unsupported:
        _fail(f"{context} unsupported by-class counts do not reconcile")
    unsupported_free = _strict_int(
        support["unsupported_free_count"], context=f"{context}.unsupported_free_count"
    )
    unsupported_occupied = _strict_int(
        support["unsupported_occupied_count"], context=f"{context}.unsupported_occupied_count"
    )
    unsupported_unknown = _strict_int(
        support["unsupported_unknown_count"], context=f"{context}.unsupported_unknown_count"
    )
    if (
        unsupported_free != parsed_by_class["free"]["unsupported"]
        or unsupported_occupied != parsed_by_class["occupied"]["unsupported"]
        or unsupported_unknown != parsed_by_class["unknown"]["unsupported"]
    ):
        _fail(f"{context} unsupported class shortcuts do not reconcile")
    all_unknown = unsupported_unknown == unsupported
    if _strict_bool(
        support["unsupported_targets_are_all_unknown"],
        context=f"{context}.unsupported_targets_are_all_unknown",
    ) != all_unknown:
        _fail(f"{context} unsupported-target decision mismatch")
    passes = unsupported_free == 0 and unsupported_occupied == 0 and all_unknown
    if _strict_bool(support["passes"], context=f"{context}.passes") != passes:
        _fail(f"{context} pass decision mismatch")
    violations = _list(support["violations"], context=f"{context}.violations")
    if len(violations) != unsupported_free + unsupported_occupied:
        _fail(f"{context} violation count does not identify every known unsupported label")
    for index, item in enumerate(violations):
        violation = _record(
            item,
            {"frame_key", "row", "column", "class_id", "class_name"},
            context=f"{context}.violations[{index}]",
        )
        if expected_key is not None and violation["frame_key"] != expected_key:
            _fail(f"{context}.violations[{index}] frame key mismatch")
        _strict_int(violation["row"], context=f"{context}.violations[{index}].row", maximum=63)
        _strict_int(violation["column"], context=f"{context}.violations[{index}].column", maximum=63)
        class_id = _strict_int(
            violation["class_id"], context=f"{context}.violations[{index}].class_id", minimum=1, maximum=2
        )
        if violation["class_name"] != CLASS_NAMES[class_id]:
            _fail(f"{context}.violations[{index}] class identity mismatch")
    return support


RAY_SUMMARY_FIELDS = {
    "sequence_count",
    "length_histogram",
    "sequences_with_fewer_than_two_cells_count",
    "transition_rate_eligible_sequence_count",
    "class_transition_histogram",
    "maximum_transitions_per_sequence",
    "directed_unequal_transition_counts",
    "transition_bucket_counts",
    "transition_event_count",
    "transition_events_per_eligible_sequence",
    "contains_known_after_unknown_count",
    "contains_free_after_occupied_count",
    "scalar_first_hit_irregular_count",
    "scalar_first_hit_regular_count",
}
PER_FRAME_LENGTH_HISTOGRAM = {
    "0": 2,
    "1": 2,
    "3": 2,
    "4": 2,
    "5": 20,
    "6": 20,
    "7": 68,
    "8": 66,
    "9": 38,
    "10": 16,
    "11": 12,
    "12": 2,
    "14": 2,
    "16": 4,
}


def _count_histogram(value: object, *, context: str) -> dict[str, int]:
    record = _mapping(value, context=context)
    result: dict[str, int] = {}
    for key, raw in record.items():
        if not isinstance(key, str) or not key.isdigit() or str(int(key)) != key:
            _fail(f"{context} keys must be canonical nonnegative integers")
        result[key] = _strict_int(raw, context=f"{context}.{key}")
    return result


def _distance_counts(value: object, *, context: str) -> dict[str, int]:
    record = _record(value, set(DISTANCE_BIN_NAMES), context=context)
    return {
        name: _strict_int(record[name], context=f"{context}.{name}")
        for name in DISTANCE_BIN_NAMES
    }


def _validate_ray_summary(value: object, *, context: str, frame_count: int) -> Mapping[str, Any]:
    summary = _record(value, RAY_SUMMARY_FIELDS, context=context)
    sequence_count = _strict_int(summary["sequence_count"], context=f"{context}.sequence_count")
    if sequence_count != frame_count * RAYS_PER_FRAME:
        _fail(f"{context}.sequence_count mismatch")
    lengths = _count_histogram(summary["length_histogram"], context=f"{context}.length_histogram")
    expected_lengths = {
        key: value * frame_count for key, value in PER_FRAME_LENGTH_HISTOGRAM.items()
    }
    if lengths != expected_lengths:
        _fail(f"{context}.length_histogram changed from the frozen mapping")
    short = _strict_int(
        summary["sequences_with_fewer_than_two_cells_count"],
        context=f"{context}.sequences_with_fewer_than_two_cells_count",
    )
    eligible = _strict_int(
        summary["transition_rate_eligible_sequence_count"],
        context=f"{context}.transition_rate_eligible_sequence_count",
    )
    if short != frame_count * SHORT_RAYS_PER_FRAME or eligible != frame_count * ELIGIBLE_RAYS_PER_FRAME:
        _fail(f"{context} ray eligibility counts mismatch")
    transitions = _count_histogram(
        summary["class_transition_histogram"], context=f"{context}.class_transition_histogram"
    )
    if sum(transitions.values()) != sequence_count:
        _fail(f"{context}.class_transition_histogram does not sum to sequence_count")
    maximum = _strict_int(
        summary["maximum_transitions_per_sequence"],
        context=f"{context}.maximum_transitions_per_sequence",
    )
    if maximum != max((int(key) for key, count in transitions.items() if count), default=0):
        _fail(f"{context}.maximum_transitions_per_sequence mismatch")
    directed = _record(
        summary["directed_unequal_transition_counts"],
        set(TRANSITION_NAMES),
        context=f"{context}.directed_unequal_transition_counts",
    )
    directed_total = sum(
        _strict_int(directed[name], context=f"{context}.directed_unequal_transition_counts.{name}")
        for name in TRANSITION_NAMES
    )
    event_count = _strict_int(
        summary["transition_event_count"], context=f"{context}.transition_event_count"
    )
    if directed_total != event_count or event_count != sum(
        int(key) * count for key, count in transitions.items()
    ):
        _fail(f"{context} transition event counts do not reconcile")
    buckets = _record(
        summary["transition_bucket_counts"],
        {"0", "1", "2", "3_plus"},
        context=f"{context}.transition_bucket_counts",
    )
    parsed_buckets = {
        key: _strict_int(buckets[key], context=f"{context}.transition_bucket_counts.{key}")
        for key in ("0", "1", "2", "3_plus")
    }
    expected_buckets = {
        "0": transitions.get("0", 0),
        "1": transitions.get("1", 0),
        "2": transitions.get("2", 0),
        "3_plus": sum(count for key, count in transitions.items() if int(key) >= 3),
    }
    if parsed_buckets != expected_buckets or sum(parsed_buckets.values()) != sequence_count:
        _fail(f"{context}.transition_bucket_counts mismatch")
    rate = summary["transition_events_per_eligible_sequence"]
    if eligible:
        numeric_rate = _finite_number(rate, context=f"{context}.transition_events_per_eligible_sequence")
        if not math.isclose(numeric_rate, event_count / eligible, rel_tol=0.0, abs_tol=1e-15):
            _fail(f"{context}.transition_events_per_eligible_sequence mismatch")
    elif rate is not None:
        _fail(f"{context}.transition_events_per_eligible_sequence must be null")
    for key in (
        "contains_known_after_unknown_count",
        "contains_free_after_occupied_count",
        "scalar_first_hit_irregular_count",
        "scalar_first_hit_regular_count",
    ):
        _strict_int(summary[key], context=f"{context}.{key}", maximum=sequence_count)
    if (
        int(summary["scalar_first_hit_irregular_count"])
        + int(summary["scalar_first_hit_regular_count"])
        != sequence_count
    ):
        _fail(f"{context} scalar first-hit regularity counts do not reconcile")
    return summary


def _validate_frame_ray_sequences(value: object, *, context: str) -> Mapping[str, Any]:
    rays = _record(
        value,
        {"schema", "summary", "sequence_summary_records_sha256", "transition_table_sha256"},
        context=context,
    )
    _require_equal(rays["schema"], RAY_SEQUENCE_SCHEMA, context=f"{context}.schema")
    summary = _validate_ray_summary(rays["summary"], context=f"{context}.summary", frame_count=1)
    _sha256(
        rays["sequence_summary_records_sha256"],
        context=f"{context}.sequence_summary_records_sha256",
    )
    declared = _sha256(rays["transition_table_sha256"], context=f"{context}.transition_table_sha256")
    if declared != canonical_json_sha256(summary):
        _fail(f"{context}.transition_table_sha256 mismatch")
    return rays


CAMERA_MOUNT_EVIDENCE_FIELDS = {
    "base_position_world",
    "base_quat_world_xyzw",
    "stored_base_yaw_rad",
    "plan_camera_mount_body",
    "frame_camera_mount_body",
    "recorded_camera_pose_world",
    "expected_camera_pose_world",
    "quaternion_norm",
    "quaternion_norm_abs_residual",
    "quaternion_yaw_rad",
    "wrapped_yaw_abs_residual_rad",
    "position_max_abs_residual_m",
    "lookat_max_abs_residual_m",
    "up_max_abs_residual",
    "look_distance_m",
    "look_distance_abs_residual_m",
    "forward_angular_error_rad",
    "up_angular_error_rad",
    "passes",
}


def _camera_pose_record(value: object, *, context: str) -> dict[str, list[float]]:
    pose = _record(value, {"position", "lookat", "up"}, context=context)
    return {
        name: _finite_vector(pose[name], size=3, context=f"{context}.{name}")
        for name in ("position", "lookat", "up")
    }


def _validate_camera_mount_evidence(
    value: object,
    *,
    context: str,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = _record(value, CAMERA_MOUNT_EVIDENCE_FIELDS, context=context)
    normalized = {
        "base_position_world": _finite_vector(
            evidence["base_position_world"],
            size=3,
            context=f"{context}.base_position_world",
        ),
        "base_quat_world_xyzw": _finite_vector(
            evidence["base_quat_world_xyzw"],
            size=4,
            context=f"{context}.base_quat_world_xyzw",
        ),
        "stored_base_yaw_rad": _finite_number(
            evidence["stored_base_yaw_rad"],
            context=f"{context}.stored_base_yaw_rad",
        ),
        "plan_camera_mount_body": _camera_mount_record(
            evidence["plan_camera_mount_body"],
            context=f"{context}.plan_camera_mount_body",
        ),
        "frame_camera_mount_body": _camera_mount_record(
            evidence["frame_camera_mount_body"],
            context=f"{context}.frame_camera_mount_body",
        ),
        "recorded_camera_pose_world": _camera_pose_record(
            evidence["recorded_camera_pose_world"],
            context=f"{context}.recorded_camera_pose_world",
        ),
        "expected_camera_pose_world": _camera_pose_record(
            evidence["expected_camera_pose_world"],
            context=f"{context}.expected_camera_pose_world",
        ),
    }
    scalar_fields = (
        "quaternion_norm",
        "quaternion_norm_abs_residual",
        "quaternion_yaw_rad",
        "wrapped_yaw_abs_residual_rad",
        "position_max_abs_residual_m",
        "lookat_max_abs_residual_m",
        "up_max_abs_residual",
        "look_distance_m",
        "look_distance_abs_residual_m",
        "forward_angular_error_rad",
        "up_angular_error_rad",
    )
    for field in scalar_fields:
        normalized[field] = _finite_number(
            evidence[field], context=f"{context}.{field}"
        )
    for field in (
        "quaternion_norm",
        "quaternion_norm_abs_residual",
        "wrapped_yaw_abs_residual_rad",
        "position_max_abs_residual_m",
        "lookat_max_abs_residual_m",
        "up_max_abs_residual",
        "look_distance_m",
        "look_distance_abs_residual_m",
        "forward_angular_error_rad",
        "up_angular_error_rad",
    ):
        if normalized[field] < 0.0:
            _fail(f"{context}.{field} must be nonnegative")
    normalized["passes"] = _strict_bool(
        evidence["passes"], context=f"{context}.passes"
    )
    if expected is not None:
        if (
            normalized != dict(expected)
            or canonical_json_sha256(evidence) != canonical_json_sha256(expected)
        ):
            _fail(f"{context} differs from independent full-quaternion composition")
    return normalized


FRAME_REPORT_FIELDS = {
    "record_key",
    "camera_mount_composition",
    "label_support",
    "ray_sequences",
    "collision_veto_only_unknown_count",
    "collision_veto_only_unknown_identities",
    "collision_veto_only_unknown_distance_bin_counts",
    "attributed_to_matched_box_count",
    "depends_on_unmatched_collision_box_count",
    "attribution_partitions",
    "reconstruction_mismatch_cell_count",
    "reconstruction_mismatch_identities",
    "rendered_collision_overlap_xor_cell_count",
    "geometry_ambiguity_cell_count",
}
ATTRIBUTION_PARTITION_NAMES = (
    "matched_true_unmatched_false",
    "matched_false_unmatched_true",
    "matched_true_unmatched_true",
    "matched_false_unmatched_false",
)


def _validate_cell_coordinates(value: object, *, context: str) -> list[list[int]]:
    values = _list(value, context=context)
    parsed: list[list[int]] = []
    for index, item in enumerate(values):
        coordinate = _list(item, context=f"{context}[{index}]")
        if len(coordinate) != 2:
            _fail(f"{context}[{index}] must contain row and column")
        parsed.append(
            [
                _strict_int(
                    coordinate[0], context=f"{context}[{index}].row", maximum=63
                ),
                _strict_int(
                    coordinate[1], context=f"{context}[{index}].column", maximum=63
                ),
            ]
        )
    if parsed != sorted(parsed) or len({tuple(value) for value in parsed}) != len(parsed):
        _fail(f"{context} must be unique in row/column order")
    return parsed


def _validate_frame_attribution(
    report: Mapping[str, Any], *, context: str, veto_cells: Sequence[Sequence[int]]
) -> None:
    partitions = _record(
        report["attribution_partitions"],
        set(ATTRIBUTION_PARTITION_NAMES),
        context=f"{context}.attribution_partitions",
    )
    parsed: dict[str, list[list[int]]] = {}
    for name in ATTRIBUTION_PARTITION_NAMES:
        partition = _record(
            partitions[name],
            {"count", "identities", "identities_sha256"},
            context=f"{context}.attribution_partitions.{name}",
        )
        identities = _validate_cell_coordinates(
            partition["identities"],
            context=f"{context}.attribution_partitions.{name}.identities",
        )
        if _strict_int(
            partition["count"], context=f"{context}.attribution_partitions.{name}.count"
        ) != len(identities):
            _fail(f"{context} attribution partition count mismatch")
        if _sha256(
            partition["identities_sha256"],
            context=f"{context}.attribution_partitions.{name}.identities_sha256",
        ) != canonical_json_sha256(identities):
            _fail(f"{context} attribution partition hash mismatch")
        parsed[name] = identities
    flattened = [
        identity
        for name in ATTRIBUTION_PARTITION_NAMES
        for identity in parsed[name]
    ]
    if len({tuple(identity) for identity in flattened}) != len(flattened):
        _fail(f"{context} attribution partitions are not disjoint")
    if sorted(flattened) != [list(value) for value in veto_cells]:
        _fail(f"{context} attribution partitions are not exhaustive")
    matched = len(parsed["matched_true_unmatched_false"]) + len(
        parsed["matched_true_unmatched_true"]
    )
    unmatched = len(parsed["matched_false_unmatched_true"]) + len(
        parsed["matched_true_unmatched_true"]
    )
    if _strict_int(
        report["attributed_to_matched_box_count"],
        context=f"{context}.attributed_to_matched_box_count",
    ) != matched:
        _fail(f"{context} matched-box attribution count contradicts partitions")
    if _strict_int(
        report["depends_on_unmatched_collision_box_count"],
        context=f"{context}.depends_on_unmatched_collision_box_count",
    ) != unmatched:
        _fail(f"{context} unmatched-box dependence count contradicts partitions")


def _validate_frame_reports(
    value: object,
    *,
    label_manifest: Mapping[str, Any],
    expected_panel_records: Sequence[Mapping[str, Any]] | None = None,
    expected_camera_evidence: Mapping[tuple[Any, ...], Mapping[str, Any]] | None = None,
) -> list[Mapping[str, Any]]:
    reports = _list(value, context="frame_reports")
    if len(reports) != EXPECTED_FRAME_COUNT:
        _fail("frame_reports must contain exactly 320 records")
    shard_hashes = {entry["sha256"] for entry in label_manifest["entries"]}
    keys_seen: set[str] = set()
    families = Counter()
    sides = Counter()
    for index, item in enumerate(reports):
        report = _record(item, FRAME_REPORT_FIELDS, context=f"frame_reports[{index}]")
        key = _validate_record_key(report["record_key"], context=f"frame_reports[{index}].record_key")
        if expected_panel_records is not None:
            expected_key = {
                field: expected_panel_records[index][field]
                for field in FRAME_IDENTITY_FIELDS
            }
            if key != expected_key:
                _fail(
                    f"frame_reports[{index}] identity/order differs from the panel"
                )
        key_hash = canonical_json_sha256(key)
        if key_hash in keys_seen:
            _fail("frame_reports contain duplicate frame identities")
        keys_seen.add(key_hash)
        families[str(key["family"])] += 1
        sides[str(key["side"])] += 1
        if key["label_shard_sha256"] not in shard_hashes:
            _fail(f"frame_reports[{index}] names a shard outside label_shard_manifest")
        identity = tuple(key[field] for field in FRAME_IDENTITY_FIELDS)
        expected_camera = (
            None
            if expected_camera_evidence is None
            else expected_camera_evidence.get(identity)
        )
        if expected_camera_evidence is not None and expected_camera is None:
            _fail(f"frame_reports[{index}] lacks independent camera-mount evidence")
        _validate_camera_mount_evidence(
            report["camera_mount_composition"],
            context=f"frame_reports[{index}].camera_mount_composition",
            expected=expected_camera,
        )
        _validate_label_support(
            report["label_support"],
            context=f"frame_reports[{index}].label_support",
            frame_count=1,
            expected_key=key,
        )
        _validate_frame_ray_sequences(
            report["ray_sequences"], context=f"frame_reports[{index}].ray_sequences"
        )
        veto_count = _strict_int(
            report["collision_veto_only_unknown_count"],
            context=f"frame_reports[{index}].collision_veto_only_unknown_count",
            maximum=CELLS_PER_FRAME,
        )
        veto_cells = _validate_cell_coordinates(
            report["collision_veto_only_unknown_identities"],
            context=f"frame_reports[{index}].collision_veto_only_unknown_identities",
        )
        if len(veto_cells) != veto_count:
            _fail(f"frame_reports[{index}] veto identity count mismatch")
        distance_counts = _distance_counts(
            report["collision_veto_only_unknown_distance_bin_counts"],
            context=f"frame_reports[{index}].collision_veto_only_unknown_distance_bin_counts",
        )
        if sum(distance_counts.values()) != veto_count:
            _fail(f"frame_reports[{index}] veto distance counts do not reconcile")
        mismatch_cells = _validate_cell_coordinates(
            report["reconstruction_mismatch_identities"],
            context=f"frame_reports[{index}].reconstruction_mismatch_identities",
        )
        if len(mismatch_cells) != _strict_int(
            report["reconstruction_mismatch_cell_count"],
            context=f"frame_reports[{index}].reconstruction_mismatch_cell_count",
            maximum=CELLS_PER_FRAME,
        ):
            _fail(f"frame_reports[{index}] mismatch identity count mismatch")
        _validate_frame_attribution(
            report,
            context=f"frame_reports[{index}]",
            veto_cells=veto_cells,
        )
        for field in (
            "rendered_collision_overlap_xor_cell_count",
            "geometry_ambiguity_cell_count",
        ):
            _strict_int(
                report[field], context=f"frame_reports[{index}].{field}", maximum=CELLS_PER_FRAME
            )
    if families != Counter({family: EXPECTED_FAMILY_FRAME_COUNT for family in FAMILIES}):
        _fail("frame_reports family counts differ from the frozen 64-per-family panel")
    if sides != Counter({side: EXPECTED_SIDE_FRAME_COUNT for side in ENDPOINT_SIDES}):
        _fail("frame_reports endpoint-side counts differ from 160 per side")
    return reports


def _validate_camera_mount_composition_aggregate(
    value: object,
    *,
    reports: Sequence[Mapping[str, Any]],
    expected_camera_evidence: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> Mapping[str, Any]:
    aggregate = _record(
        value,
        {
            "frame_count",
            "pass_count",
            "failure_count",
            "passes",
            "ordered_frame_evidence_sha256",
        },
        context="camera_mount_composition",
    )
    ordered = []
    pass_count = 0
    for report in reports:
        key = dict(report["record_key"])
        identity = tuple(key[field] for field in FRAME_IDENTITY_FIELDS)
        expected = expected_camera_evidence.get(identity)
        if expected is None:
            _fail("aggregate camera-mount evidence lacks a selected frame")
        ordered.append(
            {"record_key": key, "camera_mount_composition": dict(expected)}
        )
        pass_count += int(bool(expected["passes"]))
    frame_count = len(ordered)
    failure_count = frame_count - pass_count
    if _strict_int(
        aggregate["frame_count"], context="camera_mount_composition.frame_count"
    ) != frame_count:
        _fail("camera_mount_composition.frame_count mismatch")
    if _strict_int(
        aggregate["pass_count"], context="camera_mount_composition.pass_count"
    ) != pass_count:
        _fail("camera_mount_composition.pass_count mismatch")
    if _strict_int(
        aggregate["failure_count"], context="camera_mount_composition.failure_count"
    ) != failure_count:
        _fail("camera_mount_composition.failure_count mismatch")
    passes = failure_count == 0 and frame_count == EXPECTED_FRAME_COUNT
    if _strict_bool(
        aggregate["passes"], context="camera_mount_composition.passes"
    ) != passes:
        _fail("camera_mount_composition.passes mismatch")
    if _sha256(
        aggregate["ordered_frame_evidence_sha256"],
        context="camera_mount_composition.ordered_frame_evidence_sha256",
    ) != canonical_json_sha256(ordered):
        _fail("camera_mount_composition ordered evidence hash mismatch")
    return aggregate


def _validate_frame_identity(value: object, *, reports: Sequence[Mapping[str, Any]]) -> None:
    identity = _record(value, {"count", "encoding_fields", "sha256"}, context="frame_identity")
    if _strict_int(identity["count"], context="frame_identity.count") != EXPECTED_FRAME_COUNT:
        _fail("frame_identity.count mismatch")
    _require_equal(
        identity["encoding_fields"], list(FRAME_IDENTITY_FIELDS), context="frame_identity.encoding_fields"
    )
    ordered = [
        [report["record_key"][field] for field in FRAME_IDENTITY_FIELDS]
        for report in reports
    ]
    declared = _sha256(identity["sha256"], context="frame_identity.sha256")
    if declared != canonical_json_sha256(ordered):
        _fail("frame_identity.sha256 mismatch")


def _validate_selected_label_bytes(value: object) -> None:
    record = _record(
        value, {"frame_count", "encoding", "byte_count", "sha256"}, context="selected_label_bytes"
    )
    if _strict_int(record["frame_count"], context="selected_label_bytes.frame_count") != EXPECTED_FRAME_COUNT:
        _fail("selected_label_bytes.frame_count mismatch")
    _require_equal(
        record["encoding"],
        "canonical_frame_order_contiguous_row_major_uint8_targets_only",
        context="selected_label_bytes.encoding",
    )
    if _strict_int(record["byte_count"], context="selected_label_bytes.byte_count") != EXPECTED_TOTAL_LABEL_BYTES:
        _fail("selected_label_bytes.byte_count mismatch")
    _sha256(record["sha256"], context="selected_label_bytes.sha256")


def _sum_histograms(values: Sequence[Mapping[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for value in values:
        total.update(value)
    return dict(sorted(total.items(), key=lambda item: int(item[0])))


def _expected_summed_ray_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [report["ray_sequences"]["summary"] for report in reports]
    sequence_count = sum(int(summary["sequence_count"]) for summary in summaries)
    eligible = sum(int(summary["transition_rate_eligible_sequence_count"]) for summary in summaries)
    event_count = sum(int(summary["transition_event_count"]) for summary in summaries)
    return {
        "sequence_count": sequence_count,
        "length_histogram": _sum_histograms([summary["length_histogram"] for summary in summaries]),
        "sequences_with_fewer_than_two_cells_count": sum(
            int(summary["sequences_with_fewer_than_two_cells_count"]) for summary in summaries
        ),
        "transition_rate_eligible_sequence_count": eligible,
        "class_transition_histogram": _sum_histograms(
            [summary["class_transition_histogram"] for summary in summaries]
        ),
        "maximum_transitions_per_sequence": max(
            (int(summary["maximum_transitions_per_sequence"]) for summary in summaries), default=0
        ),
        "directed_unequal_transition_counts": {
            name: sum(int(summary["directed_unequal_transition_counts"][name]) for summary in summaries)
            for name in TRANSITION_NAMES
        },
        "transition_bucket_counts": {
            name: sum(int(summary["transition_bucket_counts"][name]) for summary in summaries)
            for name in ("0", "1", "2", "3_plus")
        },
        "transition_event_count": event_count,
        "transition_events_per_eligible_sequence": event_count / eligible if eligible else None,
        "contains_known_after_unknown_count": sum(
            int(summary["contains_known_after_unknown_count"]) for summary in summaries
        ),
        "contains_free_after_occupied_count": sum(
            int(summary["contains_free_after_occupied_count"]) for summary in summaries
        ),
        "scalar_first_hit_irregular_count": sum(
            int(summary["scalar_first_hit_irregular_count"]) for summary in summaries
        ),
        "scalar_first_hit_regular_count": sum(
            int(summary["scalar_first_hit_regular_count"]) for summary in summaries
        ),
    }


def _expected_summed_label_support(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class = {
        name: {
            field: sum(int(report["label_support"]["by_class"][name][field]) for report in reports)
            for field in ("total", "supported", "unsupported")
        }
        for name in CLASS_NAMES
    }
    violations = [
        violation
        for report in reports
        for violation in report["label_support"]["violations"]
    ]
    total = sum(record["total"] for record in by_class.values())
    supported = sum(record["supported"] for record in by_class.values())
    unsupported = sum(record["unsupported"] for record in by_class.values())
    unsupported_free = by_class["free"]["unsupported"]
    unsupported_occupied = by_class["occupied"]["unsupported"]
    unsupported_unknown = by_class["unknown"]["unsupported"]
    return {
        "schema": LABEL_SUPPORT_SCHEMA,
        "frame_count": len(reports),
        "total_supervised_label_count": total,
        "supported_label_count": supported,
        "unsupported_label_count": unsupported,
        "class_counts": {name: record["total"] for name, record in by_class.items()},
        "by_class": by_class,
        "unsupported_free_count": unsupported_free,
        "unsupported_occupied_count": unsupported_occupied,
        "unsupported_unknown_count": unsupported_unknown,
        "unsupported_targets_are_all_unknown": unsupported_unknown == unsupported,
        "violations": violations,
        "passes": unsupported_free == 0 and unsupported_occupied == 0 and unsupported_unknown == unsupported,
    }


def _validate_observability_scope(
    value: object,
    *,
    context: str,
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    scope = _record(value, {"frame_count", "label_support", "ray_sequences"}, context=context)
    frame_count = len(reports)
    if _strict_int(scope["frame_count"], context=f"{context}.frame_count") != frame_count:
        _fail(f"{context}.frame_count mismatch")
    support = _validate_label_support(
        scope["label_support"], context=f"{context}.label_support", frame_count=frame_count
    )
    expected_support = _expected_summed_label_support(reports)
    if support != expected_support:
        _fail(f"{context}.label_support does not equal the frame-report sum")
    rays = _record(
        scope["ray_sequences"],
        {"schema", "summary", "sequence_summary_records_sha256"},
        context=f"{context}.ray_sequences",
    )
    _require_equal(rays["schema"], RAY_SEQUENCE_SCHEMA, context=f"{context}.ray_sequences.schema")
    summary = _validate_ray_summary(
        rays["summary"], context=f"{context}.ray_sequences.summary", frame_count=frame_count
    )
    if summary != _expected_summed_ray_summary(reports):
        _fail(f"{context}.ray_sequences.summary does not equal the frame-report sum")
    _sha256(
        rays["sequence_summary_records_sha256"],
        context=f"{context}.ray_sequences.sequence_summary_records_sha256",
    )
    return scope


def _validate_label_observability(
    value: object, *, reports: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    summary = _record(
        value,
        {
            "schema",
            "frame_count",
            "family_order",
            "endpoint_side_order",
            "aggregate",
            "families",
            "endpoint_sides",
            "fit_known_target_coverage_gate",
            "ordered_sequence_summary_records_sha256",
            "aggregate_transition_tables_sha256",
            "representative_scalar_first_hit_violations",
        },
        context="label_observability",
    )
    _require_equal(summary["schema"], OBSERVABILITY_SUMMARY_SCHEMA, context="label_observability.schema")
    if _strict_int(summary["frame_count"], context="label_observability.frame_count") != EXPECTED_FRAME_COUNT:
        _fail("label_observability.frame_count mismatch")
    _require_equal(summary["family_order"], list(FAMILIES), context="label_observability.family_order")
    _require_equal(
        summary["endpoint_side_order"], list(ENDPOINT_SIDES), context="label_observability.endpoint_side_order"
    )
    aggregate = _validate_observability_scope(
        summary["aggregate"], context="label_observability.aggregate", reports=reports
    )
    families = _record(summary["families"], set(FAMILIES), context="label_observability.families")
    family_scopes: dict[str, Mapping[str, Any]] = {}
    for family in FAMILIES:
        selected = [report for report in reports if report["record_key"]["family"] == family]
        family_scopes[family] = _validate_observability_scope(
            families[family], context=f"label_observability.families.{family}", reports=selected
        )
    sides = _record(summary["endpoint_sides"], set(ENDPOINT_SIDES), context="label_observability.endpoint_sides")
    side_scopes: dict[str, Mapping[str, Any]] = {}
    for side in ENDPOINT_SIDES:
        selected = [report for report in reports if report["record_key"]["side"] == side]
        side_scopes[side] = _validate_observability_scope(
            sides[side], context=f"label_observability.endpoint_sides.{side}", reports=selected
        )
    gate = _record(
        summary["fit_known_target_coverage_gate"],
        {"aggregate_passes", "family_passes", "requires_aggregate_and_all_families", "passes"},
        context="label_observability.fit_known_target_coverage_gate",
    )
    aggregate_passes = bool(aggregate["label_support"]["passes"])
    if _strict_bool(gate["aggregate_passes"], context="coverage_gate.aggregate_passes") != aggregate_passes:
        _fail("coverage gate aggregate decision mismatch")
    family_passes_record = _record(
        gate["family_passes"], set(FAMILIES), context="coverage_gate.family_passes"
    )
    family_passes = {
        family: bool(family_scopes[family]["label_support"]["passes"])
        for family in FAMILIES
    }
    for family in FAMILIES:
        _strict_bool(family_passes_record[family], context=f"coverage_gate.family_passes.{family}")
    if dict(family_passes_record) != family_passes:
        _fail("coverage gate family decisions mismatch")
    if not _strict_bool(
        gate["requires_aggregate_and_all_families"],
        context="coverage_gate.requires_aggregate_and_all_families",
    ):
        _fail("coverage gate conjunction rule changed")
    expected_passes = aggregate_passes and all(family_passes.values())
    if _strict_bool(gate["passes"], context="coverage_gate.passes") != expected_passes:
        _fail("coverage gate final decision mismatch")
    _sha256(
        summary["ordered_sequence_summary_records_sha256"],
        context="label_observability.ordered_sequence_summary_records_sha256",
    )
    transition_tables = {
        "aggregate": aggregate["ray_sequences"]["summary"],
        "families": {
            family: family_scopes[family]["ray_sequences"]["summary"] for family in FAMILIES
        },
        "endpoint_sides": {
            side: side_scopes[side]["ray_sequences"]["summary"] for side in ENDPOINT_SIDES
        },
    }
    declared_transition_hash = _sha256(
        summary["aggregate_transition_tables_sha256"],
        context="label_observability.aggregate_transition_tables_sha256",
    )
    if declared_transition_hash != canonical_json_sha256(transition_tables):
        _fail("label_observability aggregate transition-table hash mismatch")
    _validate_representative_ray_violations(
        summary["representative_scalar_first_hit_violations"],
        reports=reports,
        expected_total=int(
            aggregate["ray_sequences"]["summary"][
                "scalar_first_hit_irregular_count"
            ]
        ),
    )
    return summary


def _expected_family_class_count_rows(
    reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def row(scope: str, selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "scope": scope,
            "frame_count": len(selected),
            **{
                name: sum(
                    int(report["label_support"]["class_counts"][name])
                    for report in selected
                )
                for name in CLASS_NAMES
            },
        }

    return [
        row("aggregate", reports),
        *[
            row(
                family,
                [
                    report
                    for report in reports
                    if report["record_key"]["family"] == family
                ],
            )
            for family in FAMILIES
        ],
    ]


def _validate_family_class_count_table(
    value: object,
    *,
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    table = _record(
        value,
        {"family_order", "rows", "table_sha256"},
        context="family_class_count_table",
    )
    _require_equal(
        table["family_order"], list(FAMILIES), context="family class-count order"
    )
    rows = _list(table["rows"], context="family_class_count_table.rows")
    if len(rows) != 1 + len(FAMILIES):
        _fail("family class-count table must contain aggregate plus five families")
    for index, raw in enumerate(rows):
        record = _record(
            raw,
            {"scope", "frame_count", *CLASS_NAMES},
            context=f"family_class_count_table.rows[{index}]",
        )
        _string(record["scope"], context=f"family class-count scope {index}")
        frame_count = _strict_int(
            record["frame_count"], context=f"family class-count frame_count {index}"
        )
        class_total = sum(
            _strict_int(record[name], context=f"family class-count {index}.{name}")
            for name in CLASS_NAMES
        )
        if class_total != frame_count * CELLS_PER_FRAME:
            _fail("family class-count row does not reconcile to its frame count")
    expected = _expected_family_class_count_rows(reports)
    if rows != expected:
        _fail("family class-count table differs from per-frame evidence")
    declared = _sha256(
        table["table_sha256"], context="family_class_count_table.table_sha256"
    )
    if declared != canonical_json_sha256(rows):
        _fail("family class-count table canonical hash mismatch")
    return table


def _reconcile_selected_label_evidence(
    *,
    panel_records: Sequence[Mapping[str, Any]],
    selected_rows: Mapping[tuple[Any, ...], tuple[bytes, bytes]],
    reports: Sequence[Mapping[str, Any]],
    selected_label_bytes: Mapping[str, Any],
    label_observability: Mapping[str, Any],
    family_class_count_table: Mapping[str, Any],
) -> None:
    """Reject runner-supplied label evidence not derivable from selected bytes."""

    if len(panel_records) != len(reports) or len(selected_rows) != len(reports):
        _fail("independent selected-row evidence count mismatch")
    mapping = _stdlib_camera_mapping()[0]
    selected_digest = hashlib.sha256()
    computed: list[dict[str, Any]] = []
    for index, (record, report) in enumerate(zip(panel_records, reports)):
        identity = tuple(record[field] for field in FRAME_IDENTITY_FIELDS)
        row = selected_rows.get(identity)
        if row is None:
            _fail(f"panel frame {index} lacks an independently selected label row")
        target, supervision = row
        selected_digest.update(target)
        key = {field: record[field] for field in FRAME_IDENTITY_FIELDS}
        evidence = _stdlib_analyze_frame_labels(
            target,
            supervision,
            frame_key=key,
            mapping=mapping,
        )
        if report["label_support"] != evidence["label_support"]:
            _fail(f"frame_reports[{index}] label/support evidence is fabricated")
        if report["ray_sequences"] != evidence["ray_sequences"]:
            _fail(f"frame_reports[{index}] ray evidence is fabricated")
        computed.append(
            {
                "record": record,
                "report": report,
                "target": target,
                "ray_records": evidence["ray_records"],
            }
        )

    if _sha256(
        selected_label_bytes["sha256"], context="selected_label_bytes.sha256"
    ) != selected_digest.hexdigest():
        _fail("selected_label_bytes SHA-256 differs from independent shard selection")

    def rays_for(predicate: Any) -> list[Mapping[str, Any]]:
        return [
            ray
            for item in computed
            if predicate(item["record"])
            for ray in item["ray_records"]
        ]

    aggregate_rays = rays_for(lambda _record: True)
    scopes: list[tuple[str, Mapping[str, Any], list[Mapping[str, Any]]]] = [
        ("aggregate", label_observability["aggregate"], aggregate_rays)
    ]
    scopes.extend(
        (
            f"families.{family}",
            label_observability["families"][family],
            rays_for(lambda record, family=family: record["family"] == family),
        )
        for family in FAMILIES
    )
    scopes.extend(
        (
            f"endpoint_sides.{side}",
            label_observability["endpoint_sides"][side],
            rays_for(lambda record, side=side: record["side"] == side),
        )
        for side in ENDPOINT_SIDES
    )
    for context, scope, rays in scopes:
        expected_hash = canonical_json_sha256(rays)
        actual_hash = _sha256(
            scope["ray_sequences"]["sequence_summary_records_sha256"],
            context=f"label_observability.{context}.ray sequence hash",
        )
        if actual_hash != expected_hash:
            _fail(f"label_observability.{context} ray-record hash is fabricated")
        if scope["ray_sequences"]["summary"] != _stdlib_ray_summary(rays):
            _fail(f"label_observability.{context} ray summary is fabricated")
    ordered_hash = _sha256(
        label_observability["ordered_sequence_summary_records_sha256"],
        context="label_observability.ordered_sequence_summary_records_sha256",
    )
    if ordered_hash != canonical_json_sha256(aggregate_rays):
        _fail("ordered aggregate ray-record hash is fabricated")

    irregular = [
        {
            "frame_key": ray["frame_key"],
            "angular_bin": ray["angular_bin"],
            "range_bins": ray["range_bins"],
            "class_sequence": ray["class_sequence"],
        }
        for ray in aggregate_rays
        if not bool(ray["scalar_first_hit_regular"])
    ]
    representative = label_observability["representative_scalar_first_hit_violations"]
    if int(representative["total_violation_count"]) != len(irregular):
        _fail("representative ray total differs from independent selected labels")
    expected_representatives = irregular[: int(representative["limit"])]
    if representative["records"] != expected_representatives:
        _fail("representative ray records differ from independent selected labels")
    if representative["records_sha256"] != canonical_json_sha256(expected_representatives):
        _fail("representative ray hash differs from independent selected labels")

    expected_class_rows = []
    for scope_name, predicate in [
        ("aggregate", lambda _record: True),
        *[
            (family, lambda record, family=family: record["family"] == family)
            for family in FAMILIES
        ],
    ]:
        selected = [item for item in computed if predicate(item["record"])]
        expected_class_rows.append(
            {
                "scope": scope_name,
                "frame_count": len(selected),
                **{
                    name: sum(
                        target_value == class_id
                        for item in selected
                        for target_value in item["target"]
                    )
                    for class_id, name in enumerate(CLASS_NAMES)
                },
            }
        )
    if family_class_count_table["rows"] != expected_class_rows:
        _fail("family class-count table differs from independent selected labels")


def _validate_representative_ray_violations(
    value: object,
    *,
    reports: Sequence[Mapping[str, Any]],
    expected_total: int,
) -> None:
    representative = _record(
        value,
        {"selection", "limit", "total_violation_count", "records", "records_sha256"},
        context="label_observability.representative_scalar_first_hit_violations",
    )
    _require_equal(
        representative["selection"],
        "first_in_canonical_frame_then_angular_bin_order",
        context="representative ray selection",
    )
    limit = _strict_int(
        representative["limit"], context="representative ray limit", minimum=1
    )
    if limit != 32:
        _fail("representative ray limit changed from 32")
    total = _strict_int(
        representative["total_violation_count"],
        context="representative ray total_violation_count",
    )
    if total != expected_total:
        _fail("representative ray total does not equal scalar-irregular ray count")
    records = _list(representative["records"], context="representative ray records")
    if len(records) != min(limit, total):
        _fail("representative ray record count does not match limit and total")
    report_order = {
        canonical_json_sha256(report["record_key"]): index
        for index, report in enumerate(reports)
    }
    ordering: list[tuple[int, int]] = []
    for index, item in enumerate(records):
        record = _record(
            item,
            {"frame_key", "angular_bin", "range_bins", "class_sequence"},
            context=f"representative ray records[{index}]",
        )
        key = _validate_record_key(
            record["frame_key"], context=f"representative ray records[{index}].frame_key"
        )
        key_hash = canonical_json_sha256(key)
        if key_hash not in report_order:
            _fail(f"representative ray records[{index}] names an unknown fit frame")
        angular_bin = _strict_int(
            record["angular_bin"],
            context=f"representative ray records[{index}].angular_bin",
            maximum=255,
        )
        range_values = _list(
            record["range_bins"], context=f"representative ray records[{index}].range_bins"
        )
        class_values = _list(
            record["class_sequence"],
            context=f"representative ray records[{index}].class_sequence",
        )
        ranges = [
            _strict_int(
                value,
                context=f"representative ray records[{index}].range_bins[{position}]",
                maximum=63,
            )
            for position, value in enumerate(range_values)
        ]
        classes = [
            _strict_int(
                value,
                context=f"representative ray records[{index}].class_sequence[{position}]",
                maximum=2,
            )
            for position, value in enumerate(class_values)
        ]
        if len(ranges) != len(classes) or ranges != sorted(set(ranges)):
            _fail(f"representative ray records[{index}] sequence geometry is malformed")
        collapsed: list[int] = []
        for class_id in classes:
            if not collapsed or class_id != collapsed[-1]:
                collapsed.append(class_id)
        transition_count = max(0, len(collapsed) - 1)
        ranks = {1: 0, 2: 1, 0: 2}
        scalar_regular = all(
            ranks[left] <= ranks[right] for left, right in zip(classes, classes[1:])
        )
        if scalar_regular and transition_count < 3:
            _fail(f"representative ray records[{index}] is not a registered violation")
        ordering.append((report_order[key_hash], angular_bin))
    if ordering != sorted(ordering) or len(set(ordering)) != len(ordering):
        _fail("representative rays are not unique in canonical frame/angular order")
    declared = _sha256(
        representative["records_sha256"], context="representative ray records_sha256"
    )
    if declared != canonical_json_sha256(records):
        _fail("representative ray records hash mismatch")


def _validate_reconstruction(
    value: object, *, reports: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    record = _record(
        value,
        {
            "frame_count",
            "passes",
            "mismatch_frame_count",
            "mismatch_cell_count",
            "mismatch_identities",
            "mismatch_identities_sha256",
        },
        context="reconstruction",
    )
    frame_count = _strict_int(record["frame_count"], context="reconstruction.frame_count")
    if frame_count != EXPECTED_FRAME_COUNT:
        _fail("reconstruction.frame_count mismatch")
    mismatch_counts = [
        int(report["reconstruction_mismatch_cell_count"]) for report in reports
    ]
    mismatch_frames = sum(count > 0 for count in mismatch_counts)
    mismatch_cells = sum(mismatch_counts)
    if _strict_int(
        record["mismatch_frame_count"], context="reconstruction.mismatch_frame_count"
    ) != mismatch_frames:
        _fail("reconstruction.mismatch_frame_count does not reconcile to frame_reports")
    if _strict_int(
        record["mismatch_cell_count"], context="reconstruction.mismatch_cell_count"
    ) != mismatch_cells:
        _fail("reconstruction.mismatch_cell_count does not reconcile to frame_reports")
    passes = mismatch_cells == 0
    if _strict_bool(record["passes"], context="reconstruction.passes") != passes:
        _fail("reconstruction.passes disagrees with mismatch counts")
    identities_hash = _sha256(
        record["mismatch_identities_sha256"],
        context="reconstruction.mismatch_identities_sha256",
    )
    mismatch_identities = [
        [
            [report["record_key"][field] for field in FRAME_IDENTITY_FIELDS],
            int(coordinate[0]),
            int(coordinate[1]),
        ]
        for report in reports
        for coordinate in report["reconstruction_mismatch_identities"]
    ]
    if len(mismatch_identities) != mismatch_cells:
        _fail("reconstruction mismatch identities do not reconcile")
    if record["mismatch_identities"] != mismatch_identities:
        _fail("reconstruction persisted mismatch identities differ from frame reports")
    if identities_hash != canonical_json_sha256(mismatch_identities):
        _fail("reconstruction mismatch identity hash mismatch")
    return record


COLLISION_SCOPE_FIELDS = {
    "frame_count",
    "veto_only_unknown_count",
    "distance_bin_counts",
    "attributed_to_matched_box_count",
    "depends_on_unmatched_collision_box_count",
    "attribution_partitions",
}


def _validate_collision_scope(
    value: object,
    *,
    context: str,
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    scope = _record(value, COLLISION_SCOPE_FIELDS, context=context)
    frame_count = _strict_int(scope["frame_count"], context=f"{context}.frame_count")
    if frame_count != len(reports):
        _fail(f"{context}.frame_count mismatch")
    veto_count = _strict_int(
        scope["veto_only_unknown_count"], context=f"{context}.veto_only_unknown_count"
    )
    expected_veto_count = sum(
        int(report["collision_veto_only_unknown_count"]) for report in reports
    )
    if veto_count != expected_veto_count:
        _fail(f"{context}.veto_only_unknown_count does not reconcile to frame_reports")
    distance = _distance_counts(
        scope["distance_bin_counts"], context=f"{context}.distance_bin_counts"
    )
    expected_distance = {
        name: sum(
            int(report["collision_veto_only_unknown_distance_bin_counts"][name])
            for report in reports
        )
        for name in DISTANCE_BIN_NAMES
    }
    if distance != expected_distance or sum(distance.values()) != veto_count:
        _fail(f"{context}.distance_bin_counts do not reconcile")
    matched = _strict_int(
        scope["attributed_to_matched_box_count"],
        context=f"{context}.attributed_to_matched_box_count",
        maximum=veto_count,
    )
    unmatched = _strict_int(
        scope["depends_on_unmatched_collision_box_count"],
        context=f"{context}.depends_on_unmatched_collision_box_count",
        maximum=veto_count,
    )
    expected_matched = sum(
        int(report["attributed_to_matched_box_count"]) for report in reports
    )
    expected_unmatched = sum(
        int(report["depends_on_unmatched_collision_box_count"])
        for report in reports
    )
    if matched != expected_matched or unmatched != expected_unmatched:
        _fail(f"{context} attribution counts do not reconcile to frame reports")
    partitions = _record(
        scope["attribution_partitions"],
        set(ATTRIBUTION_PARTITION_NAMES),
        context=f"{context}.attribution_partitions",
    )
    for name in ATTRIBUTION_PARTITION_NAMES:
        partition = _record(
            partitions[name],
            {"count", "identities_sha256"},
            context=f"{context}.attribution_partitions.{name}",
        )
        identities = [
            [
                [report["record_key"][field] for field in FRAME_IDENTITY_FIELDS],
                int(coordinate[0]),
                int(coordinate[1]),
            ]
            for report in reports
            for coordinate in report["attribution_partitions"][name]["identities"]
        ]
        if _strict_int(
            partition["count"], context=f"{context}.attribution_partitions.{name}.count"
        ) != len(identities):
            _fail(f"{context} aggregate attribution partition count mismatch")
        if _sha256(
            partition["identities_sha256"],
            context=f"{context}.attribution_partitions.{name}.identities_sha256",
        ) != canonical_json_sha256(identities):
            _fail(f"{context} aggregate attribution partition hash mismatch")
    return scope


def _validate_collision_veto(
    value: object, *, reports: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    collision = _record(
        value,
        {
            "aggregate",
            "families",
            "attribution_partitions",
            "ordered_cell_identities",
            "ordered_cell_identities_sha256",
        },
        context="collision_veto",
    )
    aggregate = _validate_collision_scope(
        collision["aggregate"], context="collision_veto.aggregate", reports=reports
    )
    families = _record(
        collision["families"], set(FAMILIES), context="collision_veto.families"
    )
    family_scopes = {}
    for family in FAMILIES:
        selected = [report for report in reports if report["record_key"]["family"] == family]
        family_scopes[family] = _validate_collision_scope(
            families[family], context=f"collision_veto.families.{family}", reports=selected
        )
    for field in (
        "frame_count",
        "veto_only_unknown_count",
        "attributed_to_matched_box_count",
        "depends_on_unmatched_collision_box_count",
    ):
        if int(aggregate[field]) != sum(int(family_scopes[family][field]) for family in FAMILIES):
            _fail(f"collision_veto aggregate {field} does not equal family sum")
    for name in DISTANCE_BIN_NAMES:
        if int(aggregate["distance_bin_counts"][name]) != sum(
            int(family_scopes[family]["distance_bin_counts"][name]) for family in FAMILIES
        ):
            _fail(f"collision_veto aggregate distance bin {name} does not equal family sum")
    identities_hash = _sha256(
        collision["ordered_cell_identities_sha256"],
        context="collision_veto.ordered_cell_identities_sha256",
    )
    veto_identities = [
        [
            [report["record_key"][field] for field in FRAME_IDENTITY_FIELDS],
            int(coordinate[0]),
            int(coordinate[1]),
        ]
        for report in reports
        for coordinate in report["collision_veto_only_unknown_identities"]
    ]
    if len(veto_identities) != int(aggregate["veto_only_unknown_count"]):
        _fail("collision-veto identities do not reconcile")
    if collision["ordered_cell_identities"] != veto_identities:
        _fail("collision-veto persisted identities differ from frame reports")
    if identities_hash != canonical_json_sha256(veto_identities):
        _fail("collision-veto ordered identity hash mismatch")
    top_partitions = _record(
        collision["attribution_partitions"],
        set(ATTRIBUTION_PARTITION_NAMES),
        context="collision_veto.attribution_partitions",
    )
    partition_union: list[list[Any]] = []
    for name in ATTRIBUTION_PARTITION_NAMES:
        partition = _record(
            top_partitions[name],
            {"count", "identities", "identities_sha256"},
            context=f"collision_veto.attribution_partitions.{name}",
        )
        expected = [
            [
                [report["record_key"][field] for field in FRAME_IDENTITY_FIELDS],
                int(coordinate[0]),
                int(coordinate[1]),
            ]
            for report in reports
            for coordinate in report["attribution_partitions"][name]["identities"]
        ]
        if partition["identities"] != expected:
            _fail(f"collision-veto top partition {name} identities differ")
        if _strict_int(
            partition["count"], context=f"collision-veto partition {name} count"
        ) != len(expected):
            _fail(f"collision-veto top partition {name} count mismatch")
        if _sha256(
            partition["identities_sha256"],
            context=f"collision-veto partition {name} SHA-256",
        ) != canonical_json_sha256(expected):
            _fail(f"collision-veto top partition {name} hash mismatch")
        if partition["count"] != aggregate["attribution_partitions"][name]["count"]:
            _fail(f"collision-veto top/aggregate partition {name} counts differ")
        partition_union.extend(expected)
    if sorted(partition_union, key=canonical_json_sha256) != sorted(
        veto_identities, key=canonical_json_sha256
    ):
        _fail("collision-veto top attribution partitions do not exhaust veto identities")
    return collision


BOX_COUNT_FIELDS = {
    "rendered_box_count",
    "collision_box_count",
    "matched_box_count",
    "unmatched_rendered_box_count",
    "unmatched_collision_box_count",
    "collision_boxes_affecting_selected_target_without_rendered_match_count",
    "rendered_collision_overlap_xor_cell_count",
    "required_provenance_missing_count",
    "required_provenance_nonunique_count",
}
BOX_SCOPE_FIELDS = {"scene_count", *BOX_COUNT_FIELDS}
BOX_SCENE_FIELDS = {
    "scene_id",
    "family",
    "matched_multiplicities",
    "unmatched_rendered_boxes",
    "unmatched_collision_boxes",
    "collision_boxes_affecting_selected_target_without_rendered_match",
    *BOX_COUNT_FIELDS,
}


def _validate_box_counts(
    value: Mapping[str, Any], *, context: str
) -> dict[str, int]:
    counts = {
        field: _strict_int(value[field], context=f"{context}.{field}")
        for field in BOX_COUNT_FIELDS
    }
    if counts["matched_box_count"] + counts["unmatched_rendered_box_count"] != counts[
        "rendered_box_count"
    ]:
        _fail(f"{context} rendered box counts do not reconcile")
    if counts["matched_box_count"] + counts["unmatched_collision_box_count"] != counts[
        "collision_box_count"
    ]:
        _fail(f"{context} collision box counts do not reconcile")
    if counts[
        "collision_boxes_affecting_selected_target_without_rendered_match_count"
    ] > counts["unmatched_collision_box_count"]:
        _fail(f"{context} affected unmatched-box count exceeds unmatched collision boxes")
    return counts


def _validate_box_scope(
    value: object, *, context: str, scenes: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    scope = _record(value, BOX_SCOPE_FIELDS, context=context)
    if _strict_int(scope["scene_count"], context=f"{context}.scene_count") != len(scenes):
        _fail(f"{context}.scene_count mismatch")
    counts = _validate_box_counts(scope, context=context)
    for field in BOX_COUNT_FIELDS:
        if counts[field] != sum(int(scene[field]) for scene in scenes):
            _fail(f"{context}.{field} does not equal scene sum")
    return scope


def _validate_unmatched_box_records(
    value: object,
    *,
    context: str,
    box_count: int,
) -> list[dict[str, Any]]:
    values = _list(value, context=context)
    records: list[dict[str, Any]] = []
    for record_index, raw in enumerate(values):
        record = _record(
            raw,
            {"index", "canonical_geometry"},
            context=f"{context}[{record_index}]",
        )
        index = _strict_int(
            record["index"],
            context=f"{context}[{record_index}].index",
            maximum=max(0, box_count - 1),
        )
        geometry_raw = _list(
            record["canonical_geometry"],
            context=f"{context}[{record_index}].canonical_geometry",
        )
        if len(geometry_raw) != 15:
            _fail(f"{context}[{record_index}] geometry must contain 15 values")
        geometry = [
            _finite_number(
                component,
                context=(
                    f"{context}[{record_index}].canonical_geometry[{component_index}]"
                ),
            )
            for component_index, component in enumerate(geometry_raw)
        ]
        records.append({"index": index, "canonical_geometry": geometry})
    ordering = [
        (tuple(record["canonical_geometry"]), int(record["index"]))
        for record in records
    ]
    if ordering != sorted(ordering) or len(
        {int(record["index"]) for record in records}
    ) != len(records):
        _fail(f"{context} must be unique and canonically ordered")
    return records


def _validate_box_parity(
    value: object,
    *,
    reports: Sequence[Mapping[str, Any]],
    expected_scene_families: set[tuple[str, str]] | None = None,
) -> Mapping[str, Any]:
    parity = _record(
        value,
        {"aggregate", "families", "scenes", "ordered_box_parity_table_sha256"},
        context="box_parity",
    )
    scenes = _list(parity["scenes"], context="box_parity.scenes")
    scene_ids: list[str] = []
    scene_order: list[tuple[int, str]] = []
    validated_scenes: list[Mapping[str, Any]] = []
    for index, item in enumerate(scenes):
        scene = _record(item, BOX_SCENE_FIELDS, context=f"box_parity.scenes[{index}]")
        scene_id = _string(scene["scene_id"], context=f"box_parity.scenes[{index}].scene_id")
        family = _string(scene["family"], context=f"box_parity.scenes[{index}].family")
        if family not in FAMILIES:
            _fail(f"box_parity.scenes[{index}].family is not registered")
        _validate_box_counts(scene, context=f"box_parity.scenes[{index}]")
        multiplicities = _list(
            scene["matched_multiplicities"],
            context=f"box_parity.scenes[{index}].matched_multiplicities",
        )
        geometry_tuples: list[tuple[float, ...]] = []
        multiplicity_total = 0
        for match_index, match_value in enumerate(multiplicities):
            match = _record(
                match_value,
                {"canonical_geometry", "multiplicity"},
                context=f"box_parity.scenes[{index}].matched_multiplicities[{match_index}]",
            )
            geometry_raw = _list(
                match["canonical_geometry"],
                context=(
                    f"box_parity.scenes[{index}].matched_multiplicities"
                    f"[{match_index}].canonical_geometry"
                ),
            )
            if len(geometry_raw) != 15:
                _fail("matched box canonical geometry must contain 15 float64 values")
            geometry = tuple(
                _finite_number(
                    coordinate,
                    context=(
                        f"box_parity.scenes[{index}].matched_multiplicities"
                        f"[{match_index}].canonical_geometry[{coordinate_index}]"
                    ),
                )
                for coordinate_index, coordinate in enumerate(geometry_raw)
            )
            multiplicity = _strict_int(
                match["multiplicity"],
                context=f"box_parity.scenes[{index}].matched_multiplicities[{match_index}].multiplicity",
                minimum=1,
            )
            geometry_tuples.append(geometry)
            multiplicity_total += multiplicity
        if geometry_tuples != sorted(geometry_tuples) or len(set(geometry_tuples)) != len(
            geometry_tuples
        ):
            _fail(f"box_parity.scenes[{index}] matched multiplicities are not canonical")
        if multiplicity_total != int(scene["matched_box_count"]):
            _fail(f"box_parity.scenes[{index}] matched multiplicities do not reconcile")
        unmatched_rendered = _validate_unmatched_box_records(
            scene["unmatched_rendered_boxes"],
            context=f"box_parity.scenes[{index}].unmatched_rendered_boxes",
            box_count=int(scene["rendered_box_count"]),
        )
        unmatched_collision = _validate_unmatched_box_records(
            scene["unmatched_collision_boxes"],
            context=f"box_parity.scenes[{index}].unmatched_collision_boxes",
            box_count=int(scene["collision_box_count"]),
        )
        affecting = _validate_unmatched_box_records(
            scene[
                "collision_boxes_affecting_selected_target_without_rendered_match"
            ],
            context=(
                f"box_parity.scenes[{index}]"
                ".collision_boxes_affecting_selected_target_without_rendered_match"
            ),
            box_count=int(scene["collision_box_count"]),
        )
        if len(unmatched_rendered) != int(scene["unmatched_rendered_box_count"]):
            _fail("unmatched rendered-box identities do not reconcile")
        if len(unmatched_collision) != int(scene["unmatched_collision_box_count"]):
            _fail("unmatched collision-box identities do not reconcile")
        if len(affecting) != int(
            scene[
                "collision_boxes_affecting_selected_target_without_rendered_match_count"
            ]
        ):
            _fail("target-affecting unmatched-box identities do not reconcile")
        unmatched_by_index = {
            int(record["index"]): record for record in unmatched_collision
        }
        if any(
            unmatched_by_index.get(int(record["index"])) != record
            for record in affecting
        ):
            _fail("target-affecting box is not an exact unmatched collision box")
        scene_ids.append(scene_id)
        scene_order.append((FAMILIES.index(family), scene_id))
        validated_scenes.append(scene)
    if scene_order != sorted(scene_order) or len(set(scene_ids)) != len(scene_ids):
        _fail("box_parity scenes are not unique and canonically family/scene ordered")
    if expected_scene_families is not None:
        actual_scene_families = {
            (str(scene["scene_id"]), str(scene["family"]))
            for scene in validated_scenes
        }
        if actual_scene_families != expected_scene_families:
            _fail("box_parity scene/family set differs from the selected panel")
    scene_by_id = {str(scene["scene_id"]): scene for scene in validated_scenes}
    for report_index, report in enumerate(reports):
        scene = scene_by_id.get(str(report["record_key"]["scene_id"]))
        if scene is None:
            _fail("frame report has no box-parity scene")
        matched = int(report["attributed_to_matched_box_count"])
        unmatched = int(report["depends_on_unmatched_collision_box_count"])
        if matched and int(scene["matched_box_count"]) == 0:
            _fail("frame claims matched-box attribution in a scene with no matched box")
        if unmatched and int(scene["unmatched_collision_box_count"]) == 0:
            _fail("frame claims unmatched-box dependence in a scene with no unmatched box")
        if unmatched and int(
            scene[
                "collision_boxes_affecting_selected_target_without_rendered_match_count"
            ]
        ) == 0:
            _fail("frame unmatched dependence contradicts scene affected-box count")
        impossible = report["attribution_partitions"][
            "matched_false_unmatched_false"
        ]["count"]
        if int(impossible) != 0:
            _fail("a collision-veto cell cannot overlap neither matched nor unmatched boxes")
        if int(report["geometry_ambiguity_cell_count"]) < max(
            int(report["reconstruction_mismatch_cell_count"]), unmatched
        ):
            _fail(f"frame_reports[{report_index}] geometry ambiguity undercounts evidence")
    aggregate = _validate_box_scope(
        parity["aggregate"], context="box_parity.aggregate", scenes=validated_scenes
    )
    families = _record(parity["families"], set(FAMILIES), context="box_parity.families")
    for family in FAMILIES:
        selected = [scene for scene in validated_scenes if scene["family"] == family]
        _validate_box_scope(
            families[family], context=f"box_parity.families.{family}", scenes=selected
        )
    table_hash = _sha256(
        parity["ordered_box_parity_table_sha256"],
        context="box_parity.ordered_box_parity_table_sha256",
    )
    if table_hash != canonical_json_sha256(scenes):
        _fail("box_parity ordered table hash mismatch")
    expected_xor = sum(int(report["rendered_collision_overlap_xor_cell_count"]) for report in reports)
    if int(aggregate["rendered_collision_overlap_xor_cell_count"]) != expected_xor:
        _fail("box_parity XOR count does not reconcile to frame_reports")
    return parity


PROVENANCE_FIELDS = {
    "passes",
    "fit_panel_file_hash_pass",
    "fit_panel_content_hash_pass",
    "current_physical_dataset_role_train_only",
    "fit_frame_identity_unique",
    "one_to_one_frame_match",
    "source_hashes_pass",
    "source_geometry_allowlisted_before_parse",
    "source_geometry_rehashed_after_parse",
    "rendered_collision_provenance_complete",
    "fixed_camera_mount_composition_complete",
    "legacy_source_split_used_for_selection",
}


def _validate_provenance(
    value: object,
    *,
    box_parity: Mapping[str, Any],
    camera_mount_composition: Mapping[str, Any],
) -> Mapping[str, Any]:
    provenance = _record(value, PROVENANCE_FIELDS, context="provenance")
    parsed = {
        field: _strict_bool(provenance[field], context=f"provenance.{field}")
        for field in PROVENANCE_FIELDS
    }
    for field in (
        "fit_panel_file_hash_pass",
        "fit_panel_content_hash_pass",
        "current_physical_dataset_role_train_only",
        "fit_frame_identity_unique",
        "one_to_one_frame_match",
        "source_hashes_pass",
        "source_geometry_allowlisted_before_parse",
        "source_geometry_rehashed_after_parse",
    ):
        if not parsed[field]:
            _fail(f"provenance.{field} must pass for a structurally valid audit")
    if parsed["legacy_source_split_used_for_selection"]:
        _fail("legacy source split was forbidden for fit-row selection")
    aggregate = box_parity["aggregate"]
    complete = (
        int(aggregate["required_provenance_missing_count"]) == 0
        and int(aggregate["required_provenance_nonunique_count"]) == 0
    )
    if parsed["rendered_collision_provenance_complete"] != complete:
        _fail("rendered/collision provenance completeness decision mismatch")
    camera_complete = bool(camera_mount_composition["passes"])
    if parsed["fixed_camera_mount_composition_complete"] != camera_complete:
        _fail("fixed-camera mount provenance completeness decision mismatch")
    if parsed["passes"] != (complete and camera_complete):
        _fail("provenance.passes differs from the runner's registered conjunction")
    return provenance


def _authorization_decision(
    *,
    provenance_passes: bool,
    source_hashes_pass: bool,
    reconstruction_passes: bool,
    access_reconciliation_passes: bool,
    mapping_passes: bool,
    coverage_passes: bool,
    ambiguity: bool,
) -> dict[str, Any]:
    authorized = bool(
        provenance_passes
        and source_hashes_pass
        and reconstruction_passes
        and access_reconciliation_passes
        and mapping_passes
        and coverage_passes
    )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "provenance_passes": provenance_passes,
        "source_hashes_pass": source_hashes_pass,
        "reconstruction_passes": reconstruction_passes,
        "access_reconciliation_passes": access_reconciliation_passes,
        "camera_centered_mapping_passes": mapping_passes,
        "fit_known_target_coverage_passes": coverage_passes,
        "rendered_collision_target_ambiguity": ambiguity,
        "camera_frustum_representation_implementation_authorized": authorized,
        "target_amendment_required_before_model_output": ambiguity,
        "trained_model_output_authorized": False,
        "holdout_access_authorized": False,
        "seed_20260711_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }


def _validate_licenses(value: object) -> None:
    fields = {
        "n32_passed",
        "g2_passed",
        "trained_model_output_authorized",
        "holdout_access_authorized",
        "seed_20260711_authorized",
        "runtime_authorized",
        "promotion_authorized",
    }
    licenses = _record(value, fields, context="licenses")
    for field in fields:
        if _strict_bool(licenses[field], context=f"licenses.{field}"):
            _fail(f"licenses.{field} must remain false")


FINALIZER_LEDGER_SCALAR_FIELDS = {
    "panel_metadata_byte_opens",
    "document_hash_byte_opens",
    "label_shard_hash_byte_opens",
    "label_shard_npz_opens",
    "registered_arrays_decompressed",
    "materialized_label_rows",
    "materialized_supervision_rows",
    "selected_label_rows_read",
    "selected_supervision_rows_read",
    "unselected_row_values_inspected",
    "unselected_row_metrics_computed",
    "unselected_rows_retained",
    "derivative_shard_or_cache_writes",
    "source_geometry_hash_byte_opens",
    "source_geometry_json_parses",
    "source_geometry_jsonl_records",
    "denied_attempts_total",
    "unexpected_path_attempts",
    *FORBIDDEN_ACCESS_COUNTERS,
}


def _ledger_scalar_contract(ledger: Mapping[str, Any]) -> dict[str, int]:
    return {
        field: _strict_int(ledger.get(field, 0), context=f"finalizer ledger {field}")
        for field in sorted(FINALIZER_LEDGER_SCALAR_FIELDS)
    }


def _validate_fresh_ledger_zero_denials(
    ledger: Mapping[str, Any], *, context: str
) -> None:
    for field in (
        "unselected_row_values_inspected",
        "unselected_row_metrics_computed",
        "unselected_rows_retained",
        "derivative_shard_or_cache_writes",
        "denied_attempts_total",
        "unexpected_path_attempts",
        *FORBIDDEN_ACCESS_COUNTERS,
    ):
        if _strict_int(ledger.get(field), context=f"{context}.{field}") != 0:
            _fail(f"{context}.{field} must remain zero")
    primary = _record(
        ledger.get("denied_primary_reasons"),
        set(DENIAL_PRIMARY_REASONS),
        context=f"{context}.denied_primary_reasons",
    )
    primary_total = sum(
        _strict_int(primary[reason], context=f"{context}.denied_primary_reasons.{reason}")
        for reason in DENIAL_PRIMARY_REASONS
    )
    if primary_total != int(ledger.get("denied_attempts_total", -1)):
        _fail(f"{context} denial primary counters do not reconcile")
    modalities = _record(
        ledger.get("denied_modality_attempts"),
        set(DENIAL_MODALITIES),
        context=f"{context}.denied_modality_attempts",
    )
    modality_total = sum(
        _strict_int(count, context=f"{context}.denied_modality_attempts.{name}")
        for name, count in modalities.items()
    )
    if modality_total != int(ledger.get("denied_attempts_total", -1)):
        _fail(f"{context} modality denial counters do not reconcile")
    if _list(
        ledger.get("denied_attempt_records"),
        context=f"{context}.denied_attempt_records",
    ):
        _fail(f"{context}.denied_attempt_records must remain empty")


def _validate_result_phase_ledgers(
    *,
    phase_ledgers_value: object,
    expected_finalizer_value: object,
    reconciliation_value: object,
    machine_manifest: Mapping[str, Any],
    expected_label_shard_entries: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    phases = _record(
        phase_ledgers_value,
        {"preparation", "runner"},
        context="phase_ledgers",
    )
    preparation = _mapping(phases["preparation"], context="phase_ledgers.preparation")
    runner = _mapping(phases["runner"], context="phase_ledgers.runner")
    if preparation != machine_manifest["preparation_access_ledger"]:
        _fail("result preparation ledger differs from the reviewed machine manifest")
    _validate_fresh_ledger_zero_denials(
        preparation, context="phase_ledgers.preparation"
    )
    _validate_fresh_ledger_zero_denials(runner, context="phase_ledgers.runner")
    exact_runner_counts = {
        "panel_metadata_byte_opens": 1,
        "label_shard_hash_byte_opens": EXPECTED_LABEL_SHARD_COUNT,
        "label_shard_npz_opens": EXPECTED_LABEL_SHARD_COUNT,
        "registered_arrays_decompressed": EXPECTED_LABEL_SHARD_COUNT * 4,
        "selected_label_rows_read": EXPECTED_FRAME_COUNT,
        "selected_supervision_rows_read": EXPECTED_FRAME_COUNT,
        "source_frame_records_selected": EXPECTED_FRAME_COUNT,
        "implementation_source_hash_byte_opens": 2 * len(SOURCE_HASH_KEYS),
        "document_hash_byte_opens": 5,
    }
    for field, expected in exact_runner_counts.items():
        if _strict_int(runner.get(field), context=f"phase_ledgers.runner.{field}") != expected:
            _fail(f"phase_ledgers.runner.{field} must equal {expected}")
    label_rows = _strict_int(
        runner.get("materialized_label_rows"),
        context="phase_ledgers.runner.materialized_label_rows",
    )
    supervision_rows = _strict_int(
        runner.get("materialized_supervision_rows"),
        context="phase_ledgers.runner.materialized_supervision_rows",
    )
    if label_rows != supervision_rows or label_rows < EXPECTED_FRAME_COUNT:
        _fail("runner materialized row totals do not reconcile")
    source_entries = machine_manifest["authorized_inputs"]["source_geometry"]["entries"]
    unique_source_paths = {str(entry["path"]) for entry in source_entries}
    exact_runner_source_counts = {
        "source_geometry_hash_byte_opens": 2 * len(unique_source_paths),
        "source_geometry_json_parses": sum(
            Path(path).suffix.lower() in {".json", ".jsonl"}
            for path in unique_source_paths
        ),
    }
    for field, expected in exact_runner_source_counts.items():
        if _strict_int(runner.get(field), context=f"phase_ledgers.runner.{field}") != expected:
            _fail(f"phase_ledgers.runner.{field} must equal {expected}")
    runner_jsonl_records = _strict_int(
        runner.get("source_geometry_jsonl_records"),
        context="phase_ledgers.runner.source_geometry_jsonl_records",
    )
    preparation_jsonl_records = _strict_int(
        preparation.get("source_geometry_jsonl_records"),
        context="phase_ledgers.preparation.source_geometry_jsonl_records",
    )
    if preparation_jsonl_records != runner_jsonl_records:
        _fail("preparation and runner source JSONL record counts differ")

    expected_selected_by_path = {
        str(entry["path"]): int(entry["selected_row_count"])
        for entry in expected_label_shard_entries
    }
    shard_ledgers = _list(
        runner.get("per_shard_materialization"),
        context="phase_ledgers.runner.per_shard_materialization",
    )
    if len(shard_ledgers) != EXPECTED_LABEL_SHARD_COUNT:
        _fail("runner per-shard ledger must contain exactly 20 entries")
    parsed_shard_ledgers: list[dict[str, int | str]] = []
    for index, value in enumerate(shard_ledgers):
        record = _record(
            value,
            {
                "path",
                "storage_rows_per_array",
                "materialized_label_rows",
                "materialized_supervision_rows",
                "selected_endpoint_rows",
            },
            context=f"phase_ledgers.runner.per_shard_materialization[{index}]",
        )
        path = _string(
            record["path"],
            context=f"phase_ledgers.runner.per_shard_materialization[{index}].path",
        )
        storage_rows = _strict_int(
            record["storage_rows_per_array"],
            context=(
                f"phase_ledgers.runner.per_shard_materialization[{index}]"
                ".storage_rows_per_array"
            ),
            minimum=1,
        )
        materialized_labels = _strict_int(
            record["materialized_label_rows"],
            context=(
                f"phase_ledgers.runner.per_shard_materialization[{index}]"
                ".materialized_label_rows"
            ),
        )
        materialized_supervision = _strict_int(
            record["materialized_supervision_rows"],
            context=(
                f"phase_ledgers.runner.per_shard_materialization[{index}]"
                ".materialized_supervision_rows"
            ),
        )
        selected_endpoint_rows = _strict_int(
            record["selected_endpoint_rows"],
            context=(
                f"phase_ledgers.runner.per_shard_materialization[{index}]"
                ".selected_endpoint_rows"
            ),
            minimum=1,
        )
        if (
            materialized_labels != 2 * storage_rows
            or materialized_supervision != 2 * storage_rows
        ):
            _fail("runner per-shard materialization counts do not reconcile")
        if selected_endpoint_rows != expected_selected_by_path.get(path):
            _fail("runner per-shard selected count differs from the panel commitment")
        parsed_shard_ledgers.append(
            {
                "path": path,
                "storage_rows_per_array": storage_rows,
                "materialized_label_rows": materialized_labels,
                "materialized_supervision_rows": materialized_supervision,
                "selected_endpoint_rows": selected_endpoint_rows,
            }
        )
    if [record["path"] for record in parsed_shard_ledgers] != sorted(
        expected_selected_by_path
    ):
        _fail("runner per-shard materialization paths are not canonical and exact")
    if sum(
        int(record["materialized_label_rows"])
        for record in parsed_shard_ledgers
    ) != label_rows or sum(
        int(record["materialized_supervision_rows"])
        for record in parsed_shard_ledgers
    ) != supervision_rows:
        _fail("runner per-shard materialization totals do not reconcile")
    if sum(
        int(record["selected_endpoint_rows"]) for record in parsed_shard_ledgers
    ) != EXPECTED_FRAME_COUNT:
        _fail("runner per-shard selected-row total does not reconcile")

    expected_finalizer = _record(
        expected_finalizer_value,
        FINALIZER_LEDGER_SCALAR_FIELDS,
        context="expected_finalizer_ledger",
    )
    for field in FINALIZER_LEDGER_SCALAR_FIELDS:
        _strict_int(expected_finalizer[field], context=f"expected_finalizer_ledger.{field}")
    for field in (
        "unselected_row_values_inspected",
        "unselected_row_metrics_computed",
        "unselected_rows_retained",
        "derivative_shard_or_cache_writes",
        "denied_attempts_total",
        "unexpected_path_attempts",
        *FORBIDDEN_ACCESS_COUNTERS,
    ):
        if int(expected_finalizer[field]) != 0:
            _fail(f"expected_finalizer_ledger.{field} must remain zero")
    exact_finalizer_counts = {
        "panel_metadata_byte_opens": 1,
        "label_shard_hash_byte_opens": EXPECTED_LABEL_SHARD_COUNT,
        "label_shard_npz_opens": EXPECTED_LABEL_SHARD_COUNT,
        "registered_arrays_decompressed": EXPECTED_LABEL_SHARD_COUNT * 4,
        "selected_label_rows_read": EXPECTED_FRAME_COUNT,
        "selected_supervision_rows_read": EXPECTED_FRAME_COUNT,
    }
    for field, expected in exact_finalizer_counts.items():
        if int(expected_finalizer[field]) != expected:
            _fail(f"expected_finalizer_ledger.{field} must equal {expected}")
    expected_finalizer_cross_phase = {
        "materialized_label_rows": label_rows,
        "materialized_supervision_rows": supervision_rows,
        "source_geometry_hash_byte_opens": len(unique_source_paths),
        "source_geometry_json_parses": exact_runner_source_counts[
            "source_geometry_json_parses"
        ],
        "source_geometry_jsonl_records": _strict_int(
            runner_jsonl_records,
            context="phase_ledgers.runner.source_geometry_jsonl_records",
        ),
    }
    for field, expected in expected_finalizer_cross_phase.items():
        if int(expected_finalizer[field]) != expected:
            _fail(f"expected_finalizer_ledger.{field} does not reconcile to runner")

    reconciliation = _record(
        reconciliation_value,
        {
            "phase_names",
            "passes",
            "forbidden_counters_zero",
            "unexpected_paths_zero",
            "incident_separate",
            "expected_distinct_label_shards",
            "selected_label_rows_each",
            "selected_supervision_rows_each",
            "source_geometry_unique_path_count",
        },
        context="two_phase_access_reconciliation",
    )
    _require_equal(
        reconciliation["phase_names"],
        ["preparation", "runner"],
        context="two-phase names",
    )
    for field in (
        "passes",
        "forbidden_counters_zero",
        "unexpected_paths_zero",
        "incident_separate",
    ):
        if not _strict_bool(reconciliation[field], context=f"two-phase {field}"):
            _fail(f"two_phase_access_reconciliation.{field} must be true")
    expected_counts = {
        "expected_distinct_label_shards": EXPECTED_LABEL_SHARD_COUNT,
        "selected_label_rows_each": EXPECTED_FRAME_COUNT,
        "selected_supervision_rows_each": EXPECTED_FRAME_COUNT,
        "source_geometry_unique_path_count": int(
            expected_finalizer["source_geometry_hash_byte_opens"]
        ),
    }
    for field, expected in expected_counts.items():
        if _strict_int(reconciliation[field], context=f"two-phase {field}") != expected:
            _fail(f"two_phase_access_reconciliation.{field} mismatch")
    return phases, expected_finalizer, reconciliation


def _validate_measured_per_shard_materialization(
    *,
    phases: Mapping[str, Any],
    measured_finalizer_ledger: Mapping[str, Any],
) -> None:
    """Require exact per-shard shape/count agreement across both readers."""

    runner_records = _list(
        _mapping(phases["runner"], context="phase_ledgers.runner").get(
            "per_shard_materialization"
        ),
        context="phase_ledgers.runner.per_shard_materialization",
    )
    measured_records = _list(
        measured_finalizer_ledger.get("label_shards"),
        context="measured finalizer ledger.label_shards",
    )
    if len(measured_records) != EXPECTED_LABEL_SHARD_COUNT:
        _fail("measured finalizer ledger must contain exactly 20 shard records")
    expected_runner_records = []
    for index, value in enumerate(measured_records):
        record = _mapping(
            value, context=f"measured finalizer ledger.label_shards[{index}]"
        )
        expected_runner_records.append(
            {
                "path": _string(
                    record.get("path"),
                    context=f"measured finalizer shard {index}.path",
                ),
                "storage_rows_per_array": _strict_int(
                    record.get("storage_rows_per_array"),
                    context=(
                        f"measured finalizer shard {index}.storage_rows_per_array"
                    ),
                    minimum=1,
                ),
                "materialized_label_rows": _strict_int(
                    record.get("materialized_label_rows"),
                    context=f"measured finalizer shard {index}.materialized_label_rows",
                ),
                "materialized_supervision_rows": _strict_int(
                    record.get("materialized_supervision_rows"),
                    context=(
                        f"measured finalizer shard {index}"
                        ".materialized_supervision_rows"
                    ),
                ),
                "selected_endpoint_rows": _strict_int(
                    record.get("selected_label_rows_read"),
                    context=(
                        f"measured finalizer shard {index}.selected_label_rows_read"
                    ),
                    minimum=1,
                ),
            }
        )
        if _strict_int(
            record.get("selected_supervision_rows_read"),
            context=(
                f"measured finalizer shard {index}.selected_supervision_rows_read"
            ),
            minimum=1,
        ) != expected_runner_records[-1]["selected_endpoint_rows"]:
            _fail("measured finalizer selected shard-row counts differ")
    if runner_records != expected_runner_records:
        _fail("runner/finalizer per-shard materialization records differ")


def _three_phase_reconciliation(
    *,
    finalizer_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_fresh_ledger_zero_denials(finalizer_ledger, context="measured finalizer ledger")
    return {
        "phase_names": ["preparation", "runner", "finalizer"],
        "passes": True,
        "forbidden_counters_zero": True,
        "unexpected_paths_zero": True,
        "incident_separate": True,
        "expected_distinct_label_shards": EXPECTED_LABEL_SHARD_COUNT,
        "selected_label_rows_each": EXPECTED_FRAME_COUNT,
        "selected_supervision_rows_each": EXPECTED_FRAME_COUNT,
        "source_geometry_unique_path_count": int(
            finalizer_ledger["source_geometry_hash_byte_opens"]
        ),
    }


def validate_result(
    value: object,
    *,
    expected_source_hashes: Mapping[str, Any],
    expected_inputs: Mapping[str, Any],
    machine_manifest: Mapping[str, Any],
    panel_records: Sequence[Mapping[str, Any]],
    expected_label_shard_entries: Sequence[Mapping[str, Any]],
    expected_source_geometry_entries: Sequence[Mapping[str, Any]],
    selected_rows: Mapping[tuple[Any, ...], tuple[bytes, bytes]],
    expected_camera_evidence: Mapping[tuple[Any, ...], Mapping[str, Any]],
    measured_finalizer_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one decoded result and return its recomputed decision."""

    result = _record(value, TOP_LEVEL_FIELDS, context="result")
    _require_equal(result["schema"], RESULT_SCHEMA, context="result.schema")
    _validate_timestamp(result["created_at_utc"])
    _validate_scope(result["scope"])
    _validate_execution_binding(result["execution_binding"])
    inputs = _validate_inputs(result["inputs"], expected_inputs=expected_inputs)
    if result["preflight_access_incident"] != inputs["preflight_access_incident"]:
        _fail("top-level incident record differs from inputs")
    if result["human_implementation_manifest"] != inputs["human_implementation_manifest"]:
        _fail("top-level human manifest record differs from inputs")
    if result["machine_implementation_manifest"] != inputs["machine_implementation_manifest"]:
        _fail("top-level machine manifest record differs from inputs")
    _validate_runtime_environment(
        result["runtime_environment"], context="result.runtime_environment"
    )
    if result["runtime_environment"] != machine_manifest["runtime_environment"]:
        _fail("result runtime environment differs from machine manifest")
    sources = _validate_source_hashes(
        result["source_hashes"],
        expected_source_hashes=expected_source_hashes,
    )
    _validate_geometry_contract(result["geometry_contract"])
    _validate_span_audit(result["old_body_column_span_audit"])
    mapping_audit = _validate_mapping_audit(result["mapping_audit"])
    selected_scene_ids = {str(record["scene_id"]) for record in panel_records}
    source_manifest = _validate_source_geometry_manifest(
        result["source_geometry_manifest"],
        expected_entries=expected_source_geometry_entries,
        selected_scene_ids=selected_scene_ids,
    )
    label_manifest = _validate_label_shard_manifest(
        result["label_shard_manifest"],
        expected_entries=expected_label_shard_entries,
    )
    reports = _validate_frame_reports(
        result["frame_reports"],
        label_manifest=label_manifest,
        expected_panel_records=panel_records,
        expected_camera_evidence=expected_camera_evidence,
    )
    _validate_frame_identity(result["frame_identity"], reports=reports)
    _validate_selected_label_bytes(result["selected_label_bytes"])
    observability = _validate_label_observability(result["label_observability"], reports=reports)
    class_table = _validate_family_class_count_table(
        result["family_class_count_table"], reports=reports
    )
    _reconcile_selected_label_evidence(
        panel_records=panel_records,
        selected_rows=selected_rows,
        reports=reports,
        selected_label_bytes=result["selected_label_bytes"],
        label_observability=observability,
        family_class_count_table=class_table,
    )
    reconstruction = _validate_reconstruction(result["reconstruction"], reports=reports)
    collision = _validate_collision_veto(result["collision_veto"], reports=reports)
    expected_scene_families = {
        (str(record["scene_id"]), str(record["family"])) for record in panel_records
    }
    box_parity = _validate_box_parity(
        result["box_parity"],
        reports=reports,
        expected_scene_families=expected_scene_families,
    )
    camera_mount_composition = _validate_camera_mount_composition_aggregate(
        result["camera_mount_composition"],
        reports=reports,
        expected_camera_evidence=expected_camera_evidence,
    )
    provenance = _validate_provenance(
        result["provenance"],
        box_parity=box_parity,
        camera_mount_composition=camera_mount_composition,
    )
    _phases, expected_finalizer, access_reconciliation = _validate_result_phase_ledgers(
        phase_ledgers_value=result["phase_ledgers"],
        expected_finalizer_value=result["expected_finalizer_ledger"],
        reconciliation_value=result["two_phase_access_reconciliation"],
        machine_manifest=machine_manifest,
        expected_label_shard_entries=expected_label_shard_entries,
    )
    _validate_fresh_ledger_zero_denials(
        measured_finalizer_ledger, context="measured finalizer ledger"
    )
    if _ledger_scalar_contract(measured_finalizer_ledger) != dict(expected_finalizer):
        _fail("measured finalizer ledger differs from the pre-run expected contract")
    _validate_measured_per_shard_materialization(
        phases=_phases,
        measured_finalizer_ledger=measured_finalizer_ledger,
    )

    ambiguity_expected = bool(
        not bool(camera_mount_composition["passes"])
        or not bool(reconstruction["passes"])
        or int(collision["aggregate"]["depends_on_unmatched_collision_box_count"])
        or int(
            box_parity["aggregate"][
                "collision_boxes_affecting_selected_target_without_rendered_match_count"
            ]
        )
        or int(box_parity["aggregate"]["required_provenance_missing_count"])
        or int(box_parity["aggregate"]["required_provenance_nonunique_count"])
    )
    ambiguity = _strict_bool(
        result["rendered_collision_target_ambiguity"],
        context="rendered_collision_target_ambiguity",
    )
    if ambiguity != ambiguity_expected:
        _fail("rendered_collision_target_ambiguity does not match the frozen rule")
    expected_decision = _authorization_decision(
        provenance_passes=bool(provenance["passes"]),
        source_hashes_pass=bool(provenance["source_hashes_pass"]),
        reconstruction_passes=bool(reconstruction["passes"]),
        access_reconciliation_passes=bool(access_reconciliation["passes"]),
        mapping_passes=bool(mapping_audit["passes"]),
        coverage_passes=bool(
            observability["fit_known_target_coverage_gate"]["passes"]
        ),
        ambiguity=ambiguity,
    )
    decision = _record(
        result["authorization_decision"],
        set(expected_decision),
        context="authorization_decision",
    )
    for key, expected in expected_decision.items():
        if isinstance(expected, bool):
            _strict_bool(decision[key], context=f"authorization_decision.{key}")
    if dict(decision) != expected_decision:
        _fail("authorization_decision differs from the independently recomputed decision")
    _validate_licenses(result["licenses"])

    content_hash = _sha256(result["content_sha256"], context="content_sha256")
    core = dict(result)
    del core["content_sha256"]
    if content_hash != canonical_json_sha256(core):
        _fail("result canonical content hash mismatch")
    # Keep these references live until after hashing; this also documents that
    # both input and source maps were independently consumed.
    del inputs, sources
    return expected_decision


def _validate_result_path(path: Path) -> Path:
    lexical_absolute = path.absolute()
    if lexical_absolute != RESULT_PATH:
        _fail("result path is not the exclusive immutable canonical path")
    candidate = path.resolve(strict=False)
    if candidate != RESULT_PATH:
        _fail("result path is not the exclusive immutable canonical path")
    return candidate


def validate_result_file(
    path: Path = RESULT_PATH,
    *,
    binding_sha256: str,
    machine_manifest_sha256: str,
) -> dict[str, Any]:
    """Read and validate the canonical result without writing another artifact."""

    _activate_import_isolation()
    if _sha256(binding_sha256, context="explicit binding SHA-256") != EXECUTION_BINDING_SHA256:
        _fail("explicit binding SHA-256 differs from the amended freeze")
    resolved = _validate_result_path(path)
    ledger = new_finalizer_access_ledger()
    machine, source_start, expected_inputs = _independently_verify_machine_and_documents(
        machine_manifest_sha256=machine_manifest_sha256,
        ledger=ledger,
    )
    panel_records, panel_metadata = _independently_load_fit_panel(
        machine_manifest=machine,
        ledger=ledger,
    )
    if panel_metadata["geometry_contract"] != expected_inputs["geometry_contract"]:
        _fail("panel geometry commitment differs from the machine manifest")
    expected_label_entries, _grouped = _expected_label_shard_entries(panel_records)
    selected_rows = _independently_read_selected_shards(
        panel_records=panel_records,
        expected_entries=expected_label_entries,
        machine_label_manifest=machine["authorized_inputs"]["label_shards"],
        ledger=ledger,
    )
    machine_source_entries = machine["authorized_inputs"]["source_geometry"]["entries"]
    expected_source_entries = [
        {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "semantic_role": entry["semantic_role"],
            "scene_id": entry["scene_id"],
        }
        for entry in machine_source_entries
    ]
    selected_scene_families: dict[str, str] = {}
    for record in panel_records:
        scene_id = str(record["scene_id"])
        family = str(record["family"])
        previous = selected_scene_families.setdefault(scene_id, family)
        if previous != family:
            _fail("one panel scene has conflicting family assignments")
    expected_camera_evidence = _independently_verify_source_geometry(
        machine_source_manifest=machine["authorized_inputs"]["source_geometry"],
        machine_render_summary_manifest=machine["authorized_inputs"]["render_summaries"],
        geometry_commitment=machine["authorized_inputs"]["physical_geometry_contract"],
        render_audit_commitment=panel_metadata["render_audit_contract"],
        expected_result_entries=expected_source_entries,
        selected_scene_ids=set(selected_scene_families),
        selected_scene_families=selected_scene_families,
        panel_records=panel_records,
        ledger=ledger,
    )
    result_allowlist = {
        RESULT_PATH: hashlib.sha256(b"exclusive audit result path").hexdigest()
    }
    _authorize_path(
        resolved,
        requested_role="audit_output",
        declared_role="audit_output",
        modality="json",
        allowlist=result_allowlist,
        ledger=ledger,
    )
    raw = resolved.read_bytes()
    payload = _strict_json_loads(raw, context="result file")
    decision = validate_result(
        payload,
        expected_source_hashes=source_start,
        expected_inputs=expected_inputs,
        machine_manifest=machine,
        panel_records=panel_records,
        expected_label_shard_entries=expected_label_entries,
        expected_source_geometry_entries=expected_source_entries,
        selected_rows=selected_rows,
        expected_camera_evidence=expected_camera_evidence,
        measured_finalizer_ledger=ledger,
    )
    if resolved.read_bytes() != raw:
        _fail("immutable result changed during validation")
    three_phase = _three_phase_reconciliation(finalizer_ledger=ledger)
    return {
        "authorization_decision": decision,
        "finalizer_access_ledger": ledger,
        "three_phase_access_reconciliation": three_phase,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--machine-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        _validate_result_path(args.result)
        if _sha256(args.binding_sha256, context="explicit binding SHA-256") != EXECUTION_BINDING_SHA256:
            raise FinalizationError("explicit binding SHA-256 differs from amended freeze")
        _sha256(args.machine_manifest_sha256, context="explicit machine manifest SHA-256")
    except FinalizationError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = validate_result_file(
        args.result,
        binding_sha256=str(args.binding_sha256),
        machine_manifest_sha256=str(args.machine_manifest_sha256),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
