#!/usr/bin/env python3
"""Reviewed execution successor for observable camera-ray evidence V4 fitting.

The numerical/data path is the frozen V4 trainer. This successor changes only
the reviewed execution shell: prerequisite gates are verified by finalizer V2
and the successor review is propagated into spawned RGB decoder processes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence
import warnings

# Direct execution cannot carry the launcher's live in-memory capability. Keep
# this guard before NumPy, Torch, PIL, or any repository import.
if __name__ == "__main__":
    sys.stderr.write(
        "V4 trainer execution must use "
        "scripts/launch_go2_observable_camera_ray_fit_v4_v2.py\n"
    )
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import launch_go2_observable_camera_ray_fit_v4_v2 as preauth_launcher

import numpy as np
import torch
import torch.nn.functional as F

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (  # noqa: E402
    EVIDENCE_SCHEMA,
    ObservableCameraRayEvidenceV4,
    PIXEL_RAY_SHAPE,
    SOURCE_SHAPE,
    project_canonical_ground_support_v4,
    rasterize_observable_camera_ray_evidence_v4,
)
from lewm.benchmarks.go2_observable_camera_ray_fit_v4_metrics import (  # noqa: E402
    ObservableCameraRayFitV4MetricAccumulator,
)
from lewm.benchmarks import (  # noqa: E402
    go2_observable_camera_ray_fit_v4_ladder_gate as ladder_gate,
)
from scripts.finalize_go2_observable_camera_ray_fit_v4_ladder_v2 import (  # noqa: E402
    validate_canonical_seed_gate_for_execution,
    validate_canonical_stage_gate_for_execution,
)
from lewm.models.observable_camera_ray_evidence_v4 import (  # noqa: E402
    DEPTH_FAR_EDGE_M,
    IMAGE_SIZE,
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (  # noqa: E402
    balanced_ground_clear_bce_v4,
    derive_observable_camera_ray_evidence_v4_targets,
    hierarchical_raster_cross_entropy_v4,
    ordered_obstacle_first_hit_nll_breakdown_v4,
    soft_rasterize_observable_camera_ray_evidence_v4,
)


DATASET_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_dataset_v1"
SHARD_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_shard_v1"
INDEX_ROW_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_index_row_v1"
AUDIT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_audit_v1"
RGB_RECEIPT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_rgb_receipt_v1"
TRAINER_AUTHORIZATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_v2"
)
RESULT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_development_result_v2"
CANONICAL_DATASET_MANIFEST_PATH = preauth_launcher.CANONICAL_DATASET_PATH
CANONICAL_AUDIT_RECEIPT_PATH = preauth_launcher.CANONICAL_AUDIT_PATH
CANONICAL_TRAINER_AUTHORIZATION_PATH = preauth_launcher.CANONICAL_AUTHORIZATION_PATH
CANONICAL_REVIEW_RECORD_PATH = preauth_launcher.CANONICAL_REVIEW_RECORD_PATH
TRAINER_AUTHORIZATION_PATH = CANONICAL_TRAINER_AUTHORIZATION_PATH
UPSTREAM_IMPLEMENTATION_FILE_SHA256 = (
    preauth_launcher.UPSTREAM_IMPLEMENTATION_FILE_SHA256
)
UPSTREAM_IMPLEMENTATION_CONTENT_SHA256 = (
    preauth_launcher.UPSTREAM_IMPLEMENTATION_CONTENT_SHA256
)
UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256 = (
    preauth_launcher.UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256
)

SUPPORTED_FIT_SIZES = ladder_gate.LADDER_FIT_SIZES
EXPECTED_SEEDS = (20260710, 20260711)
SUBSET_NAMESPACE = "lewm_go2_observable_camera_ray_fit_v4_subset_v1"
SUBSET_RANK_SEPARATOR = b"\\0"
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MAX_RGB_WORKERS = 6
MIN_R9700_MEMORY_BYTES = 16 * 1024**3
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
DEFAULT_STEPS = dict(ladder_gate.DEFAULT_STEPS)
SCHEDULE_ALGORITHM = (
    "torch_cpu_generator_manual_seed_then_concatenated_randperm_cycles_"
    "take_steps_times_batch_v1"
)
V1_DEVELOPMENT_OUTPUT_ROOT = (
    ROOT / ".generated/go2_observable_camera_ray_fit_v4/development_fit_v1"
)
CANONICAL_DEVELOPMENT_OUTPUT_ROOT = (
    ROOT / ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
)
CANONICAL_ATTEMPT_ROOT = CANONICAL_DEVELOPMENT_OUTPUT_ROOT / "attempts"
CANONICAL_GATE_ROOT = CANONICAL_DEVELOPMENT_OUTPUT_ROOT / "gates"
ATTEMPT_RESERVATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_attempt_reservation_v2"
)
ATTEMPT_COMPLETION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_attempt_completion_v2"
)
ATTEMPT_FAILURE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_attempt_failure_v2"
LADDER_CONTRACT = "observable_camera_ray_fit_v4_ladder_v3"

PANEL_FILE_SHA256 = (
    "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
)
PANEL_CONTENT_SHA256 = (
    "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
)
FIT_ROWS_SHA256 = (
    "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
)
SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256 = (
    "93b59cc38338857f01160b1cc048071ab7f32d0d2cfc2ab0a30b64a0e5a40380"
)
SOURCE_AUTHORIZATION_MANIFEST_CONTENT_SHA256 = (
    "5811cc732ed4a0af53f70e099721c5d1854f49ccd8891cce8570ff9231ab70df"
)
SOURCE_AUTHORIZATION_BINDING_SHA256 = (
    "c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9"
)
SOURCE_GEOMETRY_MANIFEST_SHA256 = (
    "beddb29b9826d7a21968effea863d040a6cfc9849ab0b2a78c4105d28dbb37d2"
)
RENDER_SUMMARIES_MANIFEST_SHA256 = (
    "9fff0ee9933ee582e4452f15c58f44dae721379bc983f2046d0b87498d1d002f"
)
SIDECAR_MANIFEST_FILE_SHA256 = (
    "6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529"
)
SIDECAR_MANIFEST_CONTENT_SHA256 = (
    "6f1ef7d9ac0c55a42182c3e2c75909f00ab37fffa460aadb549d5cd60d278c1a"
)
SIDECAR_TRAIN_FILE_SHA256 = (
    "6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6"
)
SIDECAR_TRAIN_CONTENT_SHA256 = (
    "137f1286e85fbd3e4b45d1c9fb0337255ac735508d6ead57cd816e5134725fa2"
)
SIDECAR_TRAIN_ORDERED_GLOBAL_SHA256 = (
    "36ebaad06bb0370e8914a8012a68d7102e481e358d42e6e803f05d5459c0119c"
)
SIDECAR_TRAIN_ORDERED_IDENTITY_SHA256 = (
    "0ca28a66bc3c7a4def8ceb69a482322e18c808298b000860b0d227576c386c63"
)

ARRAY_LAYOUT = (
    ("camera_origin_body_f32.bin", "<f4", (3,)),
    ("camera_basis_body_fru_f32.bin", "<f4", (3, 3)),
    ("ground_plane_z_body_f32.bin", "<f4", (1,)),
    ("ground_support_in_frustum_u8.bin", "u1", (*SOURCE_SHAPE, 5)),
    ("ground_support_clear_to_target_u8.bin", "u1", (*SOURCE_SHAPE, 5)),
    ("pixel_hit_mask_u8.bin", "u1", PIXEL_RAY_SHAPE),
    ("pixel_first_hit_distance_f32.bin", "<f4", PIXEL_RAY_SHAPE),
)
EXACT_FORBIDDEN_BUILD_LEDGER_FIELDS = (
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
    "label_shard_hash_byte_opens",
    "label_shard_npz_opens",
    "selected_label_rows_read",
    "selected_supervision_rows_read",
    "registered_arrays_decompressed",
    "materialized_label_rows",
    "materialized_supervision_rows",
    "unselected_row_values_inspected",
    "unselected_row_metrics_computed",
    "unselected_rows_retained",
    "derivative_shard_or_cache_writes",
    "sidecar_checkpoint_selection_role_byte_opens",
    "sidecar_probability_calibration_role_byte_opens",
    "sidecar_g2_evaluation_role_byte_opens",
    "fit_label_payload_byte_opens",
)
PRIMARY_DENIAL_REASONS = (
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
EXACT_AUDIT_LEDGER_FIELDS = frozenset(
    {
        "denied_attempt_records",
        "denied_attempts_total",
        "denied_modality_attempts",
        "denied_primary_reasons",
        "derivative_shard_or_cache_writes",
        "document_hash_byte_opens",
        "g2_opens",
        "generated_v4_result_opens",
        "holdout_label_or_geometry_opens",
        "implementation_source_hash_byte_opens",
        "label_shard_hash_byte_opens",
        "label_shard_npz_opens",
        "materialized_label_rows",
        "materialized_supervision_rows",
        "model_checkpoint_or_output_opens",
        "panel_metadata_byte_opens",
        "per_shard_materialization",
        "physical_nontrain_role_opens",
        "registered_arrays_decompressed",
        "rgb_byte_opens",
        "rgb_decodes",
        "runtime_opens",
        "sealed_opens",
        "seed_20260711_opens",
        "selected_label_rows_read",
        "selected_supervision_rows_read",
        "selection_or_calibration_opens",
        "source_frame_records_selected",
        "source_geometry_hash_byte_opens",
        "source_geometry_json_parses",
        "source_geometry_jsonl_records",
        "unexpected_path_attempts",
        "unselected_row_metrics_computed",
        "unselected_row_values_inspected",
        "unselected_rows_retained",
    }
)

DETERMINISM_WARNING_KERNELS = (
    "grid_sampler_2d_backward_cuda",
    "scatter_add_cuda_kernel",
)
_DETERMINISM_WARNING_SUFFIX = (
    " does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file "
    "an issue at https://github.com/pytorch/pytorch/issues to help us prioritize "
    "adding deterministic support for this operation."
)
DETERMINISM_WARNING_WHITELIST = tuple(
    kernel + _DETERMINISM_WARNING_SUFFIX for kernel in DETERMINISM_WARNING_KERNELS
)
_PYTORCH_CONTEXT_TRAILER_PREFIX = (
    " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:"
)
_PYTORCH_CONTEXT_TRAILER_SUFFIX = ".)"

TRAINER_SOURCE_ROLES = {
    "evidence_contract_document": (
        "docs/lewm_go2_observable_camera_ray_evidence_v4_contract_2026-07-12.md"
    ),
    "upstream_builder_implementation_manifest": (
        "docs/lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_2026-07-12.json"
    ),
    "ladder_v3_failure_successor_amendment": (
        "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v3_failure_successor_amendment_2026-07-13.md"
    ),
    "evidence_core": "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
    "fit_metrics": "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
    "shared_encoder": "lewm/models/encoders.py",
    "evidence_model": "lewm/models/observable_camera_ray_evidence_v4.py",
    "training_mechanics": "lewm/models/observable_camera_ray_evidence_v4_training.py",
    "exact_dataset_auditor": "scripts/audit_go2_observable_camera_ray_fit_v4.py",
    "exact_dataset_builder": "scripts/build_go2_observable_camera_ray_fit_v4.py",
    "development_trainer": "scripts/train_go2_observable_camera_ray_fit_v4.py",
    "test_exact_dataset_auditor": (
        "lewm/tests/test_audit_go2_observable_camera_ray_fit_v4.py"
    ),
    "test_exact_dataset_builder": (
        "lewm/tests/test_build_go2_observable_camera_ray_fit_v4.py"
    ),
    "test_evidence_core": "lewm/tests/test_go2_observable_camera_ray_evidence_v4.py",
    "test_fit_metrics": "lewm/tests/test_go2_observable_camera_ray_fit_v4_metrics.py",
    "test_evidence_model": "lewm/tests/test_observable_camera_ray_evidence_v4_model.py",
    "test_training_mechanics": (
        "lewm/tests/test_observable_camera_ray_evidence_v4_training.py"
    ),
    "test_development_trainer": (
        "lewm/tests/test_train_go2_observable_camera_ray_fit_v4.py"
    ),
}


@dataclass(frozen=True)
class VerifiedV4Frame:
    index_row: Mapping[str, Any]
    evidence: ObservableCameraRayEvidenceV4
    target_raster_labels: np.ndarray
    rgb_path: Path
    image_sha256: str

    @property
    def frame_key(self) -> Mapping[str, Any]:
        return self.index_row["frame_key"]

    @property
    def family(self) -> str:
        return str(self.frame_key["family"])


@dataclass(frozen=True)
class LoadedExactInputs:
    frames: tuple[VerifiedV4Frame, ...]
    manifest: Mapping[str, Any]
    audit_receipt: Mapping[str, Any]
    rgb_receipt: Mapping[str, Any]
    trainer_authorization: Mapping[str, Any]
    subset_receipt: Mapping[str, Any]
    verified_files: tuple[tuple[Path, str], ...]


@dataclass(frozen=True)
class ExactAttemptReservation:
    directory: Path
    reservation: Mapping[str, Any]
    reservation_payload: bytes
    reservation_file_sha256: str

    @property
    def binding(self) -> dict[str, Any]:
        return {
            "path": "reservation.json",
            "file_sha256": self.reservation_file_sha256,
            "content_sha256": self.reservation["content_sha256"],
        }


@dataclass(frozen=True)
class V4Batch:
    image: torch.Tensor
    camera_origin_body_m: torch.Tensor
    camera_basis_body_fru: torch.Tensor
    ground_plane_z_body_m: torch.Tensor
    pixel_hit_mask: torch.Tensor
    pixel_first_hit_distance_m: torch.Tensor
    ground_support_in_frustum: torch.Tensor
    ground_support_clear_to_target: torch.Tensor
    target_raster_labels: torch.Tensor
    families: tuple[str, ...]

    def to(self, device: torch.device) -> "V4Batch":
        return V4Batch(
            image=self.image.to(device=device, non_blocking=False),
            camera_origin_body_m=self.camera_origin_body_m.to(device=device),
            camera_basis_body_fru=self.camera_basis_body_fru.to(device=device),
            ground_plane_z_body_m=self.ground_plane_z_body_m.to(device=device),
            pixel_hit_mask=self.pixel_hit_mask.to(device=device),
            pixel_first_hit_distance_m=self.pixel_first_hit_distance_m.to(device=device),
            ground_support_in_frustum=self.ground_support_in_frustum.to(device=device),
            ground_support_clear_to_target=self.ground_support_clear_to_target.to(device=device),
            target_raster_labels=self.target_raster_labels.to(device=device),
            families=self.families,
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, name: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc


def _read_regular_bytes(
    path: Path,
    *,
    name: str,
    allowed_root: Path | None = None,
) -> bytes:
    lexical = Path(path)
    if allowed_root is not None:
        root = Path(allowed_root).resolve(strict=True)
        resolved_parent = lexical.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"{name} escapes its allowed root") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise PermissionError(f"cannot open regular {name} without following links") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"{name} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError(f"{name} changed while it was read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeError(f"{name} byte count changed while it was read")
        return payload
    finally:
        os.close(descriptor)


def _strict_hashed_object(
    path: Path,
    expected_file_sha256: str,
    *,
    name: str,
    allowed_root: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    if not _is_sha256(expected_file_sha256):
        raise ValueError(f"caller {name} SHA-256 is malformed")
    raw = _read_regular_bytes(path, name=name, allowed_root=allowed_root)
    if _sha256_bytes(raw) != expected_file_sha256:
        raise ValueError(f"{name} raw file SHA-256 changed")
    value = _parse_json(raw, name=name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if raw != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical newline-terminated JSON")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or declared != canonical_json_sha256(core):
        raise ValueError(f"{name} content SHA-256 changed")
    return value, raw


def load_exact_prerequisite_gate_bindings(
    args: argparse.Namespace,
) -> dict[str, Any]:
    fit_size = int(args.fit_size)
    seed = int(args.seed)
    stage_binding = None
    if fit_size != SUPPORTED_FIT_SIZES[0]:
        stage_index = SUPPORTED_FIT_SIZES.index(fit_size)
        previous_size = SUPPORTED_FIT_SIZES[stage_index - 1]
        expected_path = CANONICAL_GATE_ROOT / f"stage_seed_{seed}_n{previous_size}.json"
        path = Path(args.previous_stage_gate)
        if str(path) != str(expected_path):
            raise PermissionError("V4 predecessor gate path is not canonical")
        stage_binding = validate_canonical_stage_gate_for_execution(
            path,
            str(args.previous_stage_gate_sha256),
            expected_seed=seed,
            expected_next_fit_size=fit_size,
        )
    seed_binding = None
    if seed == EXPECTED_SEEDS[1]:
        expected_path = CANONICAL_GATE_ROOT / "seed_20260710.json"
        path = Path(args.seed_20260710_gate)
        if str(path) != str(expected_path):
            raise PermissionError("V4 first-seed gate path is not canonical")
        seed_binding = validate_canonical_seed_gate_for_execution(
            path,
            str(args.seed_20260710_gate_sha256),
        )
    return {
        "previous_stage_gate": stage_binding,
        "seed_20260710_gate": seed_binding,
    }


def _resolve_declared_path(
    parent: Path,
    relative: object,
    *,
    name: str,
) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PermissionError(f"{name} must be a nonempty relative path")
    root = parent.resolve(strict=True)
    candidate = parent / relative
    resolved_parent = candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"{name} escapes its declared root") from exc
    return candidate


def _verify_zero_denial_receipt(ledger: Mapping[str, Any]) -> None:
    if (
        int(ledger.get("unexpected_path_attempts", -1)) != 0
        or int(ledger.get("denied_attempts_total", -1)) != 0
        or ledger.get("denied_attempt_records") != []
    ):
        raise PermissionError("dataset build recorded a denied access attempt")
    primary = ledger.get("denied_primary_reasons")
    modalities = ledger.get("denied_modality_attempts")
    if (
        not isinstance(primary, Mapping)
        or set(primary) != set(PRIMARY_DENIAL_REASONS)
        or any(int(value) != 0 for value in primary.values())
        or not isinstance(modalities, Mapping)
        or set(modalities) != set(DENIAL_MODALITIES)
        or any(int(value) != 0 for value in modalities.values())
    ):
        raise PermissionError("dataset build denial category receipt changed")


def validate_determinism_warnings(messages: Sequence[object]) -> dict[str, Any]:
    raw_messages = [str(message) for message in messages]
    normalized_messages = []
    normalization = []
    for raw in raw_messages:
        normalized = None
        source_line = None
        if raw in DETERMINISM_WARNING_WHITELIST:
            normalized = raw
        else:
            for allowed in DETERMINISM_WARNING_WHITELIST:
                prefix = allowed + _PYTORCH_CONTEXT_TRAILER_PREFIX
                if not raw.startswith(prefix) or not raw.endswith(
                    _PYTORCH_CONTEXT_TRAILER_SUFFIX
                ):
                    continue
                digits = raw[len(prefix) : -len(_PYTORCH_CONTEXT_TRAILER_SUFFIX)]
                if (
                    digits
                    and digits[0] != "0"
                    and all(character in "0123456789" for character in digits)
                ):
                    normalized = allowed
                    source_line = int(digits)
                break
        if normalized is None:
            raise RuntimeError(f"unexpected training warning: {raw}")
        normalized_messages.append(normalized)
        normalization.append(
            {
                "raw": raw,
                "normalized": normalized,
                "context_source_line": source_line,
                "trailer_removed": source_line is not None,
            }
        )
    return {
        "warning_count": len(raw_messages),
        "raw_messages": raw_messages,
        "normalized_messages": normalized_messages,
        "normalization": normalization,
        "whitelist": list(DETERMINISM_WARNING_WHITELIST),
        "kernel_inventory": list(DETERMINISM_WARNING_KERNELS),
        "kernel_counts": {
            kernel: sum(message.startswith(kernel) for message in normalized_messages)
            for kernel in DETERMINISM_WARNING_KERNELS
        },
    }


def deterministic_fit_subset(
    index_rows: Sequence[Mapping[str, Any]],
    fit_size: int,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    size = int(fit_size)
    if size not in SUPPORTED_FIT_SIZES:
        raise ValueError(f"fit size must be one of {SUPPORTED_FIT_SIZES}")
    if len(index_rows) != 320:
        raise ValueError("deterministic exact fit subsets require all 320 rows")
    ranked_by_family: dict[str, list[tuple[str, bytes, Mapping[str, Any]]]] = {
        family: [] for family in FAMILIES
    }
    seen = set()
    for row in index_rows:
        frame_key = row.get("frame_key")
        if not isinstance(frame_key, Mapping):
            raise ValueError("fit index row lacks a frame key")
        family = str(frame_key.get("family", ""))
        if family not in ranked_by_family:
            raise ValueError("fit index row has an unregistered family")
        key_bytes = _canonical_json_bytes(frame_key)
        if key_bytes in seen:
            raise ValueError("fit index repeats a frame key")
        seen.add(key_bytes)
        rank = _sha256_bytes(
            SUBSET_NAMESPACE.encode("ascii") + SUBSET_RANK_SEPARATOR + key_bytes
        )
        ranked_by_family[family].append((rank, key_bytes, row))
    if any(len(ranked_by_family[family]) != 64 for family in FAMILIES):
        raise ValueError("fit index rows are not balanced 64 per registered family")
    for values in ranked_by_family.values():
        values.sort(key=lambda value: (value[0], value[1]))
    balanced_order = tuple(
        ranked_by_family[family][ordinal][2]
        for ordinal in range(64)
        for family in FAMILIES
    )
    selected = balanced_order[:size]
    selected_key_hashes = [canonical_json_sha256(row["frame_key"]) for row in selected]
    family_counts = Counter(str(row["frame_key"]["family"]) for row in selected)
    receipt = {
        "namespace": SUBSET_NAMESPACE,
        "parent_frame_count": len(index_rows),
        "fit_size": size,
        "selection": (
            "registered_family_round_robin_then_namespaced_sha256_"
            "ascii_backslash_zero_rank_v1"
        ),
        "family_counts": {
            family: int(family_counts.get(family, 0)) for family in FAMILIES
        },
        "ordered_frame_key_sha256": selected_key_hashes,
        "content_sha256": canonical_json_sha256(selected_key_hashes),
    }
    return selected, receipt


def _validate_frozen_exact_subset_receipt(
    receipt: Mapping[str, Any],
    *,
    fit_size: int,
) -> None:
    expected_family_counts = ladder_gate.EXPECTED_FAMILY_COUNTS[fit_size]
    keys = receipt.get("ordered_frame_key_sha256")
    if (
        SUBSET_RANK_SEPARATOR.hex() != "5c30"
        or receipt.get("fit_size") != fit_size
        or receipt.get("family_counts") != expected_family_counts
        or receipt.get("content_sha256")
        != ladder_gate.EXPECTED_SUBSET_CONTENT_SHA256[fit_size]
        or not isinstance(keys, list)
        or len(keys) != fit_size
        or keys[0] != ladder_gate.EXPECTED_FIRST_FRAME_KEY_SHA256[fit_size]
        or keys[-1] != ladder_gate.EXPECTED_LAST_FRAME_KEY_SHA256[fit_size]
    ):
        raise PermissionError("V4 exact subset drifted from its frozen ranking receipt")


def _frame_target_sha256_v4(frame: VerifiedV4Frame) -> str:
    evidence = frame.evidence
    query = project_canonical_ground_support_v4(
        camera_origin_body_m=evidence.camera_origin_body_m,
        camera_basis_body_fru=evidence.camera_basis_body_fru,
        ground_plane_z_body_m=evidence.ground_plane_z_body_m,
    )
    hit = evidence.pixel_hit_mask & (
        evidence.pixel_first_hit_distance_m < DEPTH_FAR_EDGE_M
    )
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(frame.frame_key))
    arrays = (
        ("pixel_in_range_hit", hit, np.uint8),
        (
            "pixel_depth",
            evidence.pixel_first_hit_distance_m,
            np.dtype("<f4"),
        ),
        ("ground_valid", evidence.ground_support_in_frustum, np.uint8),
        (
            "ground_clear",
            evidence.ground_support_clear_to_target,
            np.uint8,
        ),
        ("ground_distance", query.target_distance_m, np.dtype("<f8")),
        ("raster", frame.target_raster_labels, np.uint8),
    )
    for name, value, dtype in arrays:
        raw = np.ascontiguousarray(value, dtype=dtype).tobytes(order="C")
        digest.update(name.encode("ascii"))
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def validate_exact_target_partition_v4(
    frames: Sequence[VerifiedV4Frame],
    *,
    fit_size: int,
) -> dict[str, Any]:
    if len(frames) != fit_size:
        raise ValueError("V4 target-partition frame count changed")
    target_hashes = [_frame_target_sha256_v4(frame) for frame in frames]
    target_digest = hashlib.sha256(b"lewm_go2_v4_target_partition_bytes_v1")
    for target_hash in target_hashes:
        target_digest.update(bytes.fromhex(target_hash))
    if (
        canonical_json_sha256(target_hashes)
        != ladder_gate.EXPECTED_ORDERED_PER_FRAME_TARGET_SHA256[fit_size]
        or target_digest.hexdigest()
        != ladder_gate.EXPECTED_ORDERED_TARGET_BYTES_SHA256[fit_size]
    ):
        raise PermissionError(
            f"V4 N{fit_size} target bytes differ from the frozen partition"
        )
    return ladder_gate.target_partition_binding_v4(fit_size)


def _validate_dataset_receipt(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    expected_top_fields = {
        "schema",
        "evidence_schema",
        "dataset_role",
        "frame_count",
        "scene_shard_count",
        "array_layout",
        "rgb_receipt",
        "shards",
        "input_provenance",
        "access_ledger",
        "parallel_contract",
        "publication",
        "licenses",
        "content_sha256",
    }
    if set(manifest) != expected_top_fields:
        raise ValueError("V4 dataset manifest fields changed")
    if (
        manifest.get("schema") != DATASET_SCHEMA
        or manifest.get("evidence_schema") != EVIDENCE_SCHEMA
        or manifest.get("dataset_role") != "train"
        or isinstance(manifest.get("frame_count"), bool)
        or int(manifest.get("frame_count", -1)) != 320
        or isinstance(manifest.get("scene_shard_count"), bool)
        or int(manifest.get("scene_shard_count", -1)) != 20
    ):
        raise ValueError("V4 exact dataset identity or scope changed")
    expected_layout = [
        {
            "path": name,
            "dtype": np.dtype(dtype).str,
            "trailing_shape": list(shape),
        }
        for name, dtype, shape in ARRAY_LAYOUT
    ]
    if manifest.get("array_layout") != expected_layout:
        raise ValueError("V4 dataset array layout changed")
    if manifest.get("parallel_contract") != {
        "worker_start_method": "spawn",
        "maximum_workers": 6,
        "native_threads_per_worker": 1,
        "canonical_merge": "scene_hash_then_canonical_frame_key",
        "worker_count_does_not_change_artifact_bytes": True,
        "per_worker_source_revalidation": True,
        "parent_source_revalidation_before_manifest": True,
    }:
        raise ValueError("V4 dataset parallel/source revalidation receipt changed")
    if manifest.get("publication") != "private_staging_hardlink_no_replace_manifest_last":
        raise ValueError("V4 dataset publication receipt changed")
    licenses = manifest.get("licenses")
    expected_license_keys = {
        "model_output_authorized",
        "holdout_authorized",
        "g2_authorized",
        "runtime_authorized",
        "promotion_authorized",
    }
    if (
        not isinstance(licenses, Mapping)
        or set(licenses) != expected_license_keys
        or any(value is not False for value in licenses.values())
    ):
        raise PermissionError("V4 dataset licenses must remain all false")

    provenance = manifest.get("input_provenance")
    expected_provenance_fields = {
        "implementation_manifest_file_sha256",
        "implementation_manifest_content_sha256",
        "source_authorization_manifest_file_sha256",
        "source_authorization_manifest_content_sha256",
        "source_hashes",
        "fit_panel_file_sha256",
        "fit_panel_content_sha256",
        "fit_rows_sha256",
        "source_geometry_manifest_sha256",
        "render_summaries_manifest_sha256",
        "sidecar_manifest_file_sha256",
        "sidecar_manifest_content_sha256",
        "sidecar_train_file_sha256",
        "sidecar_train_content_sha256",
        "sidecar_train_ordered_global_sha256",
        "sidecar_train_ordered_identity_sha256",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance_fields:
        raise ValueError("V4 dataset input provenance fields changed")
    expected_provenance = {
        "source_authorization_manifest_file_sha256": SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256,
        "source_authorization_manifest_content_sha256": SOURCE_AUTHORIZATION_MANIFEST_CONTENT_SHA256,
        "fit_panel_file_sha256": PANEL_FILE_SHA256,
        "fit_panel_content_sha256": PANEL_CONTENT_SHA256,
        "fit_rows_sha256": FIT_ROWS_SHA256,
        "source_geometry_manifest_sha256": SOURCE_GEOMETRY_MANIFEST_SHA256,
        "render_summaries_manifest_sha256": RENDER_SUMMARIES_MANIFEST_SHA256,
        "sidecar_manifest_file_sha256": SIDECAR_MANIFEST_FILE_SHA256,
        "sidecar_manifest_content_sha256": SIDECAR_MANIFEST_CONTENT_SHA256,
        "sidecar_train_file_sha256": SIDECAR_TRAIN_FILE_SHA256,
        "sidecar_train_content_sha256": SIDECAR_TRAIN_CONTENT_SHA256,
        "sidecar_train_ordered_global_sha256": SIDECAR_TRAIN_ORDERED_GLOBAL_SHA256,
        "sidecar_train_ordered_identity_sha256": SIDECAR_TRAIN_ORDERED_IDENTITY_SHA256,
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("V4 dataset frozen input commitment changed")
    if (
        provenance.get("implementation_manifest_file_sha256")
        != UPSTREAM_IMPLEMENTATION_FILE_SHA256
        or provenance.get("implementation_manifest_content_sha256")
        != UPSTREAM_IMPLEMENTATION_CONTENT_SHA256
    ):
        raise ValueError("V4 builder implementation receipt changed")
    source_hashes = provenance.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("V4 dataset lacks semantic source hashes")
    for role, record in source_hashes.items():
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not _is_sha256(record.get("sha256"))
        ):
            raise ValueError("V4 semantic source receipt is malformed")
    binding = source_hashes.get("binding")
    if not isinstance(binding, Mapping) or binding.get("sha256") != SOURCE_AUTHORIZATION_BINDING_SHA256:
        raise ValueError("V4 source authorization binding changed")

    ledger = manifest.get("access_ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError("V4 dataset lacks an access ledger")
    for field in EXACT_FORBIDDEN_BUILD_LEDGER_FIELDS:
        if int(ledger.get(field, -1)) != 0:
            raise PermissionError(f"V4 dataset build opened forbidden input: {field}")
    _verify_zero_denial_receipt(ledger)
    if (
        int(ledger.get("sidecar_manifest_byte_opens", -1)) != 1
        or int(ledger.get("sidecar_train_role_byte_opens", -1)) != 1
        or int(ledger.get("panel_metadata_byte_opens", -1)) != 1
        or int(ledger.get("source_frame_records_selected", -1)) != 320
        or int(ledger.get("source_geometry_hash_byte_opens", 0)) <= 0
        or int(ledger.get("source_geometry_json_parses", 0)) <= 0
        or int(ledger.get("source_geometry_jsonl_records", 0)) < 320
        or int(ledger.get("implementation_source_hash_byte_opens", 0)) <= 0
        or int(ledger.get("document_hash_byte_opens", 0)) <= 0
    ):
        raise ValueError("V4 dataset positive access receipt changed")
    return provenance


def preflight_exact_frozen_dataset_provenance(
    *,
    dataset_manifest_path: Path,
    dataset_manifest_file_sha256: str,
) -> Mapping[str, Any]:
    """Validate the frozen manifest provenance before payload or reservation work."""

    if (
        str(dataset_manifest_path) != str(CANONICAL_DATASET_MANIFEST_PATH)
        or dataset_manifest_file_sha256
        != preauth_launcher.DATASET_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("V4 provenance preflight input is not canonical")
    manifest, _raw = _strict_hashed_object(
        dataset_manifest_path,
        dataset_manifest_file_sha256,
        name="V4 provenance-preflight dataset manifest",
        allowed_root=ROOT,
    )
    return _validate_dataset_receipt(manifest)


def _validate_rgb_receipt(receipt: object) -> dict[bytes, Mapping[str, Any]]:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema",
        "dataset_role",
        "frame_count",
        "ordered_frame_keys_sha256",
        "entries_sha256",
        "rgb_byte_opens",
        "entries",
        "content_sha256",
    }:
        raise ValueError("V4 dataset lacks the exact RGB metadata receipt")
    core = dict(receipt)
    declared = core.pop("content_sha256", None)
    entries = receipt.get("entries")
    if (
        receipt.get("schema") != RGB_RECEIPT_SCHEMA
        or receipt.get("dataset_role") != "train"
        or receipt.get("rgb_byte_opens") != 0
        or declared != canonical_json_sha256(core)
        or not isinstance(entries, list)
        or isinstance(receipt.get("frame_count"), bool)
        or int(receipt.get("frame_count", -1)) != 320
        or len(entries) != 320
        or receipt.get("entries_sha256") != canonical_json_sha256(entries)
    ):
        raise ValueError("V4 RGB metadata receipt identity changed")
    by_key: dict[bytes, Mapping[str, Any]] = {}
    ordered_keys = []
    paths = set()
    digests = set()
    repository = ROOT.resolve(strict=True)
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "frame_key",
            "canonical_rgb_path",
            "rgb_file_sha256",
        }:
            raise ValueError("one V4 RGB receipt entry is malformed")
        frame_key = entry.get("frame_key")
        path_text = entry.get("canonical_rgb_path")
        digest = entry.get("rgb_file_sha256")
        if not isinstance(frame_key, dict) or frame_key.get("dataset_role") != "train":
            raise PermissionError("V4 RGB receipt contains a nontrain frame")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            raise PermissionError("V4 RGB receipt path must be absolute")
        if path_text != os.path.normpath(path_text) or ".." in Path(path_text).parts:
            raise PermissionError("V4 RGB receipt path is not canonical")
        try:
            Path(path_text).relative_to(repository)
        except ValueError as exc:
            raise PermissionError("V4 RGB receipt path escapes the repository") from exc
        if Path(path_text).parent.name != "rgb" or not _is_sha256(digest):
            raise PermissionError("V4 RGB receipt path/hash is not a train render commitment")
        if frame_key.get("image_sha256") != digest:
            raise ValueError("V4 RGB receipt disagrees with its frame key")
        key = _canonical_json_bytes(frame_key)
        if key in by_key or path_text in paths or digest in digests:
            raise ValueError("V4 RGB receipt is not injective")
        canonical_entry = {
            "frame_key": json.loads(key),
            "canonical_rgb_path": path_text,
            "rgb_file_sha256": digest,
        }
        if dict(entry) != canonical_entry:
            raise ValueError("V4 RGB receipt entry is not canonical")
        by_key[key] = canonical_entry
        ordered_keys.append(canonical_entry["frame_key"])
        paths.add(path_text)
        digests.add(digest)
    if list(by_key) != sorted(by_key):
        raise ValueError("V4 RGB receipt entries are not canonically ordered")
    if receipt.get("ordered_frame_keys_sha256") != canonical_json_sha256(ordered_keys):
        raise ValueError("V4 RGB receipt ordered keys changed")
    return by_key


def _validate_audit_receipt(
    receipt: Mapping[str, Any],
    *,
    dataset_manifest: Mapping[str, Any],
    dataset_manifest_file_sha256: str,
) -> None:
    if set(receipt) != {
        "schema",
        "dataset",
        "input_authorization",
        "scope",
        "audit",
        "access_ledger",
        "licenses",
        "content_sha256",
    } or receipt.get("schema") != AUDIT_SCHEMA:
        raise ValueError("V4 audit receipt identity changed")
    if receipt.get("dataset") != {
        "schema": DATASET_SCHEMA,
        "file_sha256": dataset_manifest_file_sha256,
        "content_sha256": dataset_manifest["content_sha256"],
        "dataset_role": "train",
        "frame_count": 320,
        "scene_shard_count": 20,
    }:
        raise ValueError("V4 audit receipt does not bind this dataset")
    scope = receipt.get("scope")
    if scope != {
        "exact_fit": True,
        "dataset_role": "train",
        "rgb_opened": False,
        "rgb_receipt_metadata_join_verified": True,
        "source_geometry_opened_by_auditor": False,
        "gpu_used": False,
    }:
        raise PermissionError("V4 audit scope crossed a forbidden role boundary")
    licenses = receipt.get("licenses")
    if not isinstance(licenses, Mapping) or set(licenses) != {
        "model_output_authorized",
        "holdout_authorized",
        "g2_authorized",
        "runtime_authorized",
        "promotion_authorized",
    } or any(value is not False for value in licenses.values()):
        raise PermissionError("V4 audit licenses must remain all false")
    audit = receipt.get("audit")
    if (
        not isinstance(audit, Mapping)
        or int(audit.get("frame_count", -1)) != 320
        or int(audit.get("cell_count", -1)) != 320 * 64 * 64
        or audit.get("internal_hash_and_raster_determinism_passes") is not True
        or not _is_sha256(audit.get("ordered_frame_keys_sha256"))
        or not _is_sha256(audit.get("ordered_evidence_sha256"))
    ):
        raise ValueError("V4 audit result receipt is incomplete")
    frame_reports = audit.get("frame_reports")
    if not isinstance(frame_reports, list) or len(frame_reports) != 320:
        raise ValueError("V4 audit frame-report receipt changed")
    expected_report_fields = {
        "schema",
        "frame_key",
        "evidence_content_sha256",
        "raster_content_sha256",
        "predicted_labels_sha256",
        "predicted_class_counts",
        "reference_labels_sha256",
        "supervised_cell_count",
        "mismatch_cell_count",
        "confusion_reference_rows",
    }
    expected_frame_key_fields = {
        "dataset_role",
        "family",
        "scene_id",
        "global_row",
        "side",
        "image_sha256",
        "label_shard_sha256",
        "label_row",
    }
    if any(
        not isinstance(report, Mapping)
        or set(report) != expected_report_fields
        or not isinstance(report.get("frame_key"), Mapping)
        or set(report["frame_key"]) != expected_frame_key_fields
        or report["frame_key"].get("dataset_role") != "train"
        or report.get("supervised_cell_count") != 64 * 64
        for report in frame_reports
    ):
        raise ValueError("V4 audit frame-report role/key schema changed")
    ledger = receipt.get("access_ledger")
    if not isinstance(ledger, Mapping) or set(ledger) != EXACT_AUDIT_LEDGER_FIELDS:
        raise ValueError("V4 audit access-ledger key set changed")
    for field in (
        "rgb_byte_opens",
        "rgb_decodes",
        "holdout_label_or_geometry_opens",
        "physical_nontrain_role_opens",
        "g2_opens",
        "runtime_opens",
        "sealed_opens",
        "model_checkpoint_or_output_opens",
        "generated_v4_result_opens",
    ):
        if int(ledger.get(field, -1)) != 0:
            raise PermissionError(f"V4 audit opened forbidden input: {field}")
    _verify_zero_denial_receipt(ledger)
    expected_positive = {
        "document_hash_byte_opens": 1,
        "implementation_source_hash_byte_opens": 11,
        "label_shard_hash_byte_opens": 20,
        "label_shard_npz_opens": 20,
        "materialized_label_rows": 2460,
        "materialized_supervision_rows": 2460,
        "panel_metadata_byte_opens": 1,
        "registered_arrays_decompressed": 80,
        "selected_label_rows_read": 320,
        "selected_supervision_rows_read": 320,
    }
    if any(ledger.get(field) != value for field, value in expected_positive.items()):
        raise ValueError("V4 audit positive access receipt changed")
    per_shard = ledger.get("per_shard_materialization")
    if not isinstance(per_shard, list) or len(per_shard) != 20 or any(
        not isinstance(record, Mapping)
        or set(record)
        != {
            "path",
            "storage_rows_per_array",
            "selected_endpoint_rows",
            "materialized_label_rows",
            "materialized_supervision_rows",
        }
        for record in per_shard
    ):
        raise ValueError("V4 audit per-shard materialization schema changed")
    paths = [str(record["path"]) for record in per_shard]
    if (
        paths != sorted(paths)
        or len(set(paths)) != 20
        or sum(int(record["selected_endpoint_rows"]) for record in per_shard) != 320
        or sum(int(record["materialized_label_rows"]) for record in per_shard) != 2460
        or sum(int(record["materialized_supervision_rows"]) for record in per_shard)
        != 2460
        or any(
            int(record["materialized_label_rows"])
            != 2 * int(record["storage_rows_per_array"])
            or int(record["materialized_supervision_rows"])
            != 2 * int(record["storage_rows_per_array"])
            for record in per_shard
        )
    ):
        raise ValueError("V4 audit 20/20/320/320 materialization receipt changed")


def _validate_trainer_authorization_snapshot_schema(
    authorization: Mapping[str, Any],
    *,
    review_record_file_sha256: str | None = None,
) -> tuple[tuple[Path, str], ...]:
    if set(authorization) != {
        "schema",
        "status",
        "dataset_binding",
        "audit_binding",
        "upstream_implementation",
        "fit_contract",
        "allowed_fit_sizes",
        "source_map",
        "authorization",
        "review_record",
        "content_sha256",
    } or authorization.get("schema") != TRAINER_AUTHORIZATION_SCHEMA:
        raise PermissionError("V4 trainer authorization identity changed")
    if authorization.get("status") not in {
        "pending_independent_review",
        "independent_review_passed_authorized",
    }:
        raise ValueError("V4 trainer authorization status is malformed")
    if authorization.get("fit_contract") != {
        "ladder_contract": ladder_gate.LADDER_CONTRACT,
        "development_output_root": (
            ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
        ),
        "ladder_v3_amendment_file_sha256": (
            ladder_gate.LADDER_V3_AMENDMENT_FILE_SHA256
        ),
        "v1_failure_lineage": {
            "reservation_file_sha256": ladder_gate.V1_FAILURE_LINEAGE[
                "reservation"
            ]["file_sha256"],
            "reservation_content_sha256": ladder_gate.V1_FAILURE_LINEAGE[
                "reservation"
            ]["content_sha256"],
            "failure_file_sha256": ladder_gate.V1_FAILURE_LINEAGE["failure"][
                "file_sha256"
            ],
            "failure_content_sha256": ladder_gate.V1_FAILURE_LINEAGE["failure"][
                "content_sha256"
            ],
        },
        "dataset_schema": DATASET_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
        "rgb_receipt_schema": RGB_RECEIPT_SCHEMA,
        "dataset_role": "train",
        "exact_frame_count": 320,
        "exact_scene_shard_count": 20,
        "fit_panel_file_sha256": PANEL_FILE_SHA256,
        "fit_panel_content_sha256": PANEL_CONTENT_SHA256,
        "fit_rows_sha256": FIT_ROWS_SHA256,
        "target_partition_freeze_file_sha256": (
            ladder_gate.TARGET_PARTITION_FREEZE_FILE_SHA256
        ),
        "target_partition_freeze_content_sha256": (
            ladder_gate.TARGET_PARTITION_FREEZE_CONTENT_SHA256
        ),
        "target_partition_verifier_file_sha256": (
            ladder_gate.TARGET_PARTITION_VERIFIER_FILE_SHA256
        ),
        "target_partition_amendment_file_sha256": (
            ladder_gate.TARGET_PARTITION_AMENDMENT_FILE_SHA256
        ),
        "target_partition_verified_dataset_file_count": 180,
    } or authorization.get("allowed_fit_sizes") != list(SUPPORTED_FIT_SIZES):
        raise ValueError("V4 trainer fit contract changed")
    if authorization.get("dataset_binding") != {
        "path": str(CANONICAL_DATASET_MANIFEST_PATH),
        "status": "reviewed_exact_artifact",
        "file_sha256": preauth_launcher.DATASET_MANIFEST_FILE_SHA256,
        "content_sha256": preauth_launcher.DATASET_MANIFEST_CONTENT_SHA256,
    } or authorization.get("audit_binding") != {
        "path": str(CANONICAL_AUDIT_RECEIPT_PATH),
        "status": "reviewed_exact_artifact",
        "file_sha256": preauth_launcher.AUDIT_RECEIPT_FILE_SHA256,
        "content_sha256": preauth_launcher.AUDIT_RECEIPT_CONTENT_SHA256,
    }:
        raise PermissionError("V4 trainer exact artifact bindings changed")
    if authorization.get("upstream_implementation") != {
        "path": str(preauth_launcher.UPSTREAM_IMPLEMENTATION_PATH),
        "file_sha256": UPSTREAM_IMPLEMENTATION_FILE_SHA256,
        "content_sha256": UPSTREAM_IMPLEMENTATION_CONTENT_SHA256,
        "source_map_sha256": UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256,
    }:
        raise PermissionError("V4 trainer upstream implementation binding changed")
    flags = authorization.get("authorization")
    expected_flags = {
        "development_fit",
        "development_checkpoint_creation_authorized",
        "checkpoint_use_authorized",
        "holdout_authorized",
        "g2_authorized",
        "runtime_authorized",
        "promotion_authorized",
    }
    if (
        not isinstance(flags, Mapping)
        or set(flags) != expected_flags
        or any(not isinstance(value, bool) for value in flags.values())
        or flags.get("checkpoint_use_authorized") is not False
        or any(
            flags.get(field) is not False
            for field in (
                "holdout_authorized",
                "g2_authorized",
                "runtime_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise PermissionError("V4 trainer authorization flags are malformed")
    source_map_sha256 = preauth_launcher._validate_source_map(
        authorization.get("source_map"), root=ROOT
    )
    review_binding = authorization.get("review_record")
    if not isinstance(review_binding, Mapping) or set(review_binding) != {
        "path",
        "file_sha256",
        "content_sha256",
        "status",
    } or review_binding.get("path") != str(CANONICAL_REVIEW_RECORD_PATH):
        raise ValueError("V4 trainer review-record binding is malformed")
    if (
        review_record_file_sha256 is not None
        and review_binding.get("file_sha256") != review_record_file_sha256
    ):
        raise PermissionError("V4 caller review-record hash changed")
    review, _raw = _strict_hashed_object(
        CANONICAL_REVIEW_RECORD_PATH,
        str(
            review_binding.get("file_sha256", "")
            if review_record_file_sha256 is None
            else review_record_file_sha256
        ),
        name="V4 trainer review record",
        allowed_root=ROOT,
    )
    preauth_launcher._validate_review_record(
        review,
        source_map_sha256=source_map_sha256,
    )
    if (
        review.get("content_sha256") != review_binding.get("content_sha256")
        or review.get("reviewed_source_map_sha256") not in {None, source_map_sha256}
    ):
        raise ValueError("V4 trainer review-record content changed")
    return tuple(
        (ROOT / str(entry["path"]), str(entry["sha256"]))
        for entry in authorization["source_map"]["entries"]
    )


def _validate_trainer_authorization(
    authorization: Mapping[str, Any],
    *,
    dataset_manifest_file_sha256: str,
    dataset_manifest_content_sha256: str,
    audit_receipt_file_sha256: str,
    audit_receipt_content_sha256: str,
    review_record_file_sha256: str,
) -> tuple[tuple[Path, str], ...]:
    sources = _validate_trainer_authorization_snapshot_schema(
        authorization,
        review_record_file_sha256=review_record_file_sha256,
    )
    if authorization.get("status") != "independent_review_passed_authorized":
        raise PermissionError("V4 trainer authorization is still pending review")
    if authorization.get("dataset_binding") != {
        "path": str(CANONICAL_DATASET_MANIFEST_PATH),
        "status": "reviewed_exact_artifact",
        "file_sha256": dataset_manifest_file_sha256,
        "content_sha256": dataset_manifest_content_sha256,
    } or authorization.get("audit_binding") != {
        "path": str(CANONICAL_AUDIT_RECEIPT_PATH),
        "status": "reviewed_exact_artifact",
        "file_sha256": audit_receipt_file_sha256,
        "content_sha256": audit_receipt_content_sha256,
    }:
        raise PermissionError("V4 trainer authorization does not bind exact inputs")
    if authorization.get("authorization") != {
        "development_fit": True,
        "development_checkpoint_creation_authorized": True,
        "checkpoint_use_authorized": False,
        "holdout_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }:
        raise PermissionError("V4 trainer development fit is not authorized")
    review_binding = authorization["review_record"]
    review, _raw = _strict_hashed_object(
        CANONICAL_REVIEW_RECORD_PATH,
        str(review_binding["file_sha256"]),
        name="V4 trainer review record",
        allowed_root=ROOT,
    )
    if (
        review_binding.get("status") != "independent_review_passed"
        or review.get("status") != "independent_review_passed"
        or review.get("decision") != "pass"
        or review.get("restricted_payload_opened") is not False
        or review.get("reviewed_source_map_sha256")
        != authorization["source_map"]["source_map_sha256"]
    ):
        raise PermissionError("V4 trainer lacks an independent review receipt")
    return sources


def _verify_file_commitments(
    commitments: Sequence[tuple[Path, str]],
    *,
    name: str,
) -> None:
    seen = set()
    for path, digest in commitments:
        lexical = str(path)
        if lexical in seen:
            continue
        seen.add(lexical)
        payload = _read_regular_bytes(path, name=name, allowed_root=ROOT)
        if _sha256_bytes(payload) != digest:
            raise ValueError(f"{name} SHA-256 changed: {path}")


def _parse_canonical_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{name} must be nonempty newline-terminated JSONL")
    rows = []
    for row_index, line in enumerate(payload.splitlines()):
        value = _parse_json(line, name=f"{name} row {row_index}")
        if not isinstance(value, dict) or line != _canonical_json_bytes(value):
            raise ValueError(f"{name} row {row_index} is not canonical")
        core = dict(value)
        declared = core.pop("content_sha256", None)
        if not _is_sha256(declared) or declared != canonical_json_sha256(core):
            raise ValueError(f"{name} row {row_index} content hash changed")
        rows.append(value)
    return rows


def _validate_index_row(row: Mapping[str, Any]) -> bytes:
    if set(row) != {
        "schema",
        "frame_key",
        "image_sha256_commitment_only",
        "sidecar_row_identity_sha256",
        "evidence_content_sha256",
        "raster_content_sha256",
        "content_sha256",
    } or row.get("schema") != INDEX_ROW_SCHEMA:
        raise ValueError("one V4 index row identity changed")
    frame_key = row.get("frame_key")
    if not isinstance(frame_key, dict) or set(frame_key) != {
        "dataset_role",
        "family",
        "scene_id",
        "global_row",
        "side",
        "image_sha256",
        "label_shard_sha256",
        "label_row",
    }:
        raise ValueError("one V4 frame key changed")
    if (
        frame_key.get("dataset_role") != "train"
        or frame_key.get("family") not in FAMILIES
        or not isinstance(frame_key.get("scene_id"), str)
        or not frame_key.get("scene_id")
        or isinstance(frame_key.get("global_row"), bool)
        or not isinstance(frame_key.get("global_row"), int)
        or frame_key.get("side") not in {"current", "next"}
        or not _is_sha256(frame_key.get("image_sha256"))
        or not _is_sha256(frame_key.get("label_shard_sha256"))
        or isinstance(frame_key.get("label_row"), bool)
        or not isinstance(frame_key.get("label_row"), int)
        or int(frame_key["label_row"]) < 0
    ):
        raise PermissionError("one V4 frame key is not current physical train")
    if (
        row.get("image_sha256_commitment_only") != frame_key["image_sha256"]
        or not _is_sha256(row.get("sidecar_row_identity_sha256"))
        or not _is_sha256(row.get("evidence_content_sha256"))
        or not _is_sha256(row.get("raster_content_sha256"))
    ):
        raise ValueError("one V4 index commitment changed")
    return _canonical_json_bytes(frame_key)


def _load_dataset_frames(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    audit_receipt_path: Path,
) -> tuple[
    tuple[tuple[Mapping[str, Any], ObservableCameraRayEvidenceV4, np.ndarray], ...],
    tuple[tuple[Path, str], ...],
]:
    dataset_root = manifest_path.parent.resolve(strict=True)
    shards_root = dataset_root / "shards"
    if shards_root.is_symlink() or not shards_root.is_dir():
        raise PermissionError("V4 shard root must be a real directory")
    shard_records = manifest.get("shards")
    if not isinstance(shard_records, list) or len(shard_records) != 20:
        raise ValueError("V4 shard inventory count changed")
    expected_shard_directories = set()
    verified_files: list[tuple[Path, str]] = []
    loaded_frames = []
    seen_keys = set()
    previous_scene_hash = ""
    for shard_record in shard_records:
        if not isinstance(shard_record, Mapping) or set(shard_record) != {
            "path",
            "scene_key_sha256",
            "frame_count",
            "content_sha256",
            "file_sha256",
        }:
            raise ValueError("one V4 top-level shard record is malformed")
        if (
            not _is_sha256(shard_record.get("scene_key_sha256"))
            or not _is_sha256(shard_record.get("content_sha256"))
            or not _is_sha256(shard_record.get("file_sha256"))
            or isinstance(shard_record.get("frame_count"), bool)
            or int(shard_record.get("frame_count", -1)) <= 0
        ):
            raise ValueError("one V4 shard commitment is malformed")
        scene_hash = str(shard_record["scene_key_sha256"])
        if scene_hash <= previous_scene_hash:
            raise ValueError("V4 shards are not strictly scene-hash ordered")
        previous_scene_hash = scene_hash
        shard_path = _resolve_declared_path(
            dataset_root, shard_record["path"], name="V4 shard manifest"
        )
        if shard_path.name != "shard.json" or shard_path.parent.parent != shards_root:
            raise PermissionError("V4 shard manifest path shape changed")
        expected_shard_directories.add(shard_path.parent.name)
        shard, _raw = _strict_hashed_object(
            shard_path,
            str(shard_record["file_sha256"]),
            name="V4 shard manifest",
            allowed_root=dataset_root,
        )
        verified_files.append((shard_path, str(shard_record["file_sha256"])))
        if set(shard) != {
            "schema",
            "scene_key_sha256",
            "frame_count",
            "ordered_frame_keys_sha256",
            "ordered_evidence_sha256",
            "files",
            "content_sha256",
        } or (
            shard.get("schema") != SHARD_SCHEMA
            or shard.get("scene_key_sha256") != scene_hash
            or shard.get("frame_count") != shard_record["frame_count"]
            or shard.get("content_sha256") != shard_record["content_sha256"]
            or not _is_sha256(shard.get("ordered_frame_keys_sha256"))
            or not _is_sha256(shard.get("ordered_evidence_sha256"))
        ):
            raise ValueError("one V4 shard manifest disagrees with its receipt")
        frame_count = int(shard["frame_count"])
        file_records_raw = shard.get("files")
        if not isinstance(file_records_raw, list):
            raise ValueError("one V4 shard lacks a file inventory")
        file_records: dict[str, Mapping[str, Any]] = {}
        for record in file_records_raw:
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "dtype",
                "shape",
                "byte_count",
                "file_sha256",
            }:
                raise ValueError("one V4 shard file record is malformed")
            name = record.get("path")
            if not isinstance(name, str) or name in file_records:
                raise ValueError("one V4 shard file path is repeated")
            file_records[name] = record
        expected_names = {name for name, _dtype, _shape in ARRAY_LAYOUT} | {"index.jsonl"}
        if set(file_records) != expected_names or list(file_records) != sorted(file_records):
            raise ValueError("one V4 shard file inventory changed")
        actual_entries = {entry.name for entry in shard_path.parent.iterdir()}
        if actual_entries != expected_names | {"shard.json"}:
            raise ValueError("one V4 published shard has undeclared entries")
        if any(entry.is_symlink() or not entry.is_file() for entry in shard_path.parent.iterdir()):
            raise PermissionError("one V4 published shard entry is not a regular file")

        payloads: dict[str, bytes] = {}
        for name, dtype_text, trailing_shape in ARRAY_LAYOUT:
            record = file_records[name]
            expected_shape = [frame_count, *trailing_shape]
            if (
                record.get("dtype") != np.dtype(dtype_text).str
                or record.get("shape") != expected_shape
                or isinstance(record.get("byte_count"), bool)
                or int(record.get("byte_count", -1))
                != int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(dtype_text).itemsize
                or not _is_sha256(record.get("file_sha256"))
            ):
                raise ValueError(f"V4 shard array contract changed: {name}")
            path = _resolve_declared_path(shard_path.parent, name, name="V4 shard array")
            payload = _read_regular_bytes(path, name="V4 shard array", allowed_root=dataset_root)
            if (
                len(payload) != int(record["byte_count"])
                or _sha256_bytes(payload) != record["file_sha256"]
            ):
                raise ValueError(f"V4 shard array bytes changed: {name}")
            payloads[name] = payload
            verified_files.append((path, str(record["file_sha256"])))

        index_record = file_records["index.jsonl"]
        if (
            index_record.get("dtype") != "canonical_jsonl"
            or index_record.get("shape") != [frame_count]
            or isinstance(index_record.get("byte_count"), bool)
            or int(index_record.get("byte_count", -1)) <= 0
            or not _is_sha256(index_record.get("file_sha256"))
        ):
            raise ValueError("V4 shard index file contract changed")
        index_path = _resolve_declared_path(
            shard_path.parent, "index.jsonl", name="V4 shard index"
        )
        index_payload = _read_regular_bytes(
            index_path, name="V4 shard index", allowed_root=dataset_root
        )
        if (
            len(index_payload) != int(index_record["byte_count"])
            or _sha256_bytes(index_payload) != index_record["file_sha256"]
        ):
            raise ValueError("V4 shard index bytes changed")
        verified_files.append((index_path, str(index_record["file_sha256"])))
        index_rows = _parse_canonical_jsonl(index_payload, name="V4 shard index")
        if len(index_rows) != frame_count:
            raise ValueError("V4 shard index row count changed")
        encoded_keys = [_validate_index_row(row) for row in index_rows]
        if encoded_keys != sorted(encoded_keys):
            raise ValueError("V4 shard frame keys are not canonically ordered")
        if canonical_json_sha256([row["frame_key"] for row in index_rows]) != shard[
            "ordered_frame_keys_sha256"
        ]:
            raise ValueError("V4 shard ordered frame-key hash changed")

        arrays = {
            name: np.frombuffer(payloads[name], dtype=np.dtype(dtype)).reshape(
                frame_count, *trailing_shape
            )
            for name, dtype, trailing_shape in ARRAY_LAYOUT
        }
        ordered_evidence = []
        for row_index, (row, encoded_key) in enumerate(zip(index_rows, encoded_keys)):
            if encoded_key in seen_keys:
                raise ValueError("V4 dataset repeats a frame key")
            seen_keys.add(encoded_key)
            evidence = ObservableCameraRayEvidenceV4(
                camera_origin_body_m=arrays["camera_origin_body_f32.bin"][row_index].copy(),
                camera_basis_body_fru=arrays[
                    "camera_basis_body_fru_f32.bin"
                ][row_index].copy(),
                ground_plane_z_body_m=float(
                    arrays["ground_plane_z_body_f32.bin"][row_index, 0]
                ),
                ground_support_in_frustum=arrays[
                    "ground_support_in_frustum_u8.bin"
                ][row_index].astype(bool, copy=True),
                ground_support_clear_to_target=arrays[
                    "ground_support_clear_to_target_u8.bin"
                ][row_index].astype(bool, copy=True),
                pixel_hit_mask=arrays["pixel_hit_mask_u8.bin"][row_index].astype(
                    bool, copy=True
                ),
                pixel_first_hit_distance_m=arrays[
                    "pixel_first_hit_distance_f32.bin"
                ][row_index].copy(),
            )
            raster = rasterize_observable_camera_ray_evidence_v4(evidence)
            if (
                evidence.content_sha256() != row["evidence_content_sha256"]
                or raster.content_sha256() != row["raster_content_sha256"]
            ):
                raise ValueError("one V4 evidence/raster content hash changed")
            ordered_evidence.append(evidence.content_sha256())
            loaded_frames.append((row, evidence, raster.output_labels.copy()))
        if canonical_json_sha256(ordered_evidence) != shard["ordered_evidence_sha256"]:
            raise ValueError("V4 shard ordered evidence hash changed")

    actual_shard_directories = {entry.name for entry in shards_root.iterdir()}
    if actual_shard_directories != expected_shard_directories or any(
        entry.is_symlink() or not entry.is_dir() for entry in shards_root.iterdir()
    ):
        raise ValueError("V4 published shard-directory inventory changed")
    expected_root_entries = {"manifest.json", "shards", "audit_result.json"}
    if audit_receipt_path.resolve(strict=True) != dataset_root / "audit_result.json":
        raise PermissionError("V4 audit receipt path is not the canonical dataset audit")
    if {entry.name for entry in dataset_root.iterdir()} != expected_root_entries:
        raise ValueError("V4 dataset root has undeclared published entries")
    if any(entry.is_symlink() for entry in dataset_root.iterdir()):
        raise PermissionError("V4 dataset root contains a symlink")
    if len(loaded_frames) != 320:
        raise ValueError("V4 aggregate frame count changed")
    family_counts = Counter(str(row["frame_key"]["family"]) for row, _evidence, _raster in loaded_frames)
    if family_counts != Counter({family: 64 for family in FAMILIES}):
        raise ValueError("V4 exact fit panel is not family balanced")
    return tuple(loaded_frames), tuple(verified_files)


def load_exact_inputs(
    *,
    dataset_manifest_path: Path,
    dataset_manifest_file_sha256: str,
    audit_receipt_path: Path,
    audit_receipt_file_sha256: str,
    trainer_authorization_path: Path,
    trainer_authorization_file_sha256: str,
    trainer_review_record_path: Path,
    trainer_review_record_file_sha256: str,
    fit_size: int,
) -> LoadedExactInputs:
    if (
        str(dataset_manifest_path) != str(CANONICAL_DATASET_MANIFEST_PATH)
        or str(audit_receipt_path) != str(CANONICAL_AUDIT_RECEIPT_PATH)
        or str(trainer_authorization_path)
        != str(CANONICAL_TRAINER_AUTHORIZATION_PATH)
        or str(trainer_review_record_path) != str(CANONICAL_REVIEW_RECORD_PATH)
        or dataset_manifest_file_sha256
        != preauth_launcher.DATASET_MANIFEST_FILE_SHA256
        or audit_receipt_file_sha256 != preauth_launcher.AUDIT_RECEIPT_FILE_SHA256
    ):
        raise PermissionError("V4 exact input path/hash arguments are not canonical")
    trainer_authorization, _authorization_raw = _strict_hashed_object(
        trainer_authorization_path,
        trainer_authorization_file_sha256,
        name="V4 trainer authorization",
        allowed_root=ROOT,
    )
    trainer_sources = _validate_trainer_authorization(
        trainer_authorization,
        dataset_manifest_file_sha256=dataset_manifest_file_sha256,
        dataset_manifest_content_sha256=preauth_launcher.DATASET_MANIFEST_CONTENT_SHA256,
        audit_receipt_file_sha256=audit_receipt_file_sha256,
        audit_receipt_content_sha256=preauth_launcher.AUDIT_RECEIPT_CONTENT_SHA256,
        review_record_file_sha256=trainer_review_record_file_sha256,
    )
    _verify_file_commitments(trainer_sources, name="V4 trainer source")
    upstream = preauth_launcher._strict_hashed_object(
        preauth_launcher.UPSTREAM_IMPLEMENTATION_PATH,
        UPSTREAM_IMPLEMENTATION_FILE_SHA256,
        name="V4 upstream implementation manifest",
        canonical_path=preauth_launcher.UPSTREAM_IMPLEMENTATION_PATH,
        require_canonical=False,
    )
    if (
        upstream.get("content_sha256") != UPSTREAM_IMPLEMENTATION_CONTENT_SHA256
        or upstream.get("source_map", {}).get("source_map_sha256")
        != UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256
    ):
        raise ValueError("V4 upstream implementation receipt changed")

    # Authorization, review, source closure, and upstream receipt have passed.
    # Exact dataset/audit receipts may now be opened.
    manifest, _manifest_raw = _strict_hashed_object(
        dataset_manifest_path,
        dataset_manifest_file_sha256,
        name="V4 dataset manifest",
        allowed_root=ROOT,
    )
    provenance = _validate_dataset_receipt(manifest)
    rgb_by_key = _validate_rgb_receipt(manifest["rgb_receipt"])
    audit_receipt, _audit_raw = _strict_hashed_object(
        audit_receipt_path,
        audit_receipt_file_sha256,
        name="V4 exact audit receipt",
        allowed_root=ROOT,
    )
    _validate_audit_receipt(
        audit_receipt,
        dataset_manifest=manifest,
        dataset_manifest_file_sha256=dataset_manifest_file_sha256,
    )
    if audit_receipt.get("input_authorization") != {
        "implementation_manifest_file_sha256": provenance[
            "implementation_manifest_file_sha256"
        ],
        "source_authorization_manifest_file_sha256": SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256,
    }:
        raise ValueError("V4 audit input authorization changed")
    if (
        manifest.get("content_sha256")
        != preauth_launcher.DATASET_MANIFEST_CONTENT_SHA256
        or audit_receipt.get("content_sha256")
        != preauth_launcher.AUDIT_RECEIPT_CONTENT_SHA256
    ):
        raise ValueError("V4 exact dataset/audit content binding changed")
    dataset_sources = tuple(
        (Path(str(record["path"])), str(record["sha256"]))
        for record in provenance["source_hashes"].values()
    )
    _verify_file_commitments(dataset_sources, name="V4 dataset semantic source")

    loaded, dataset_files = _load_dataset_frames(
        dataset_manifest_path,
        manifest,
        audit_receipt_path=audit_receipt_path,
    )
    index_rows = [row for row, _evidence, _raster in loaded]
    if set(rgb_by_key) != {_canonical_json_bytes(row["frame_key"]) for row in index_rows}:
        raise ValueError("V4 RGB receipt does not join exactly to dataset frames")
    for row in index_rows:
        key = _canonical_json_bytes(row["frame_key"])
        if rgb_by_key[key]["rgb_file_sha256"] != row["image_sha256_commitment_only"]:
            raise ValueError("V4 RGB receipt image commitment changed")
    audit = audit_receipt["audit"]
    if (
        audit["ordered_frame_keys_sha256"]
        != canonical_json_sha256([row["frame_key"] for row in index_rows])
        or audit["ordered_evidence_sha256"]
        != canonical_json_sha256([row["evidence_content_sha256"] for row in index_rows])
    ):
        raise ValueError("V4 audit ordered content does not bind loaded frames")

    selected_rows, subset_receipt = deterministic_fit_subset(index_rows, fit_size)
    _validate_frozen_exact_subset_receipt(subset_receipt, fit_size=fit_size)
    loaded_by_key = {
        _canonical_json_bytes(row["frame_key"]): (row, evidence, raster)
        for row, evidence, raster in loaded
    }
    selected_frames = []
    for selected_row in selected_rows:
        key = _canonical_json_bytes(selected_row["frame_key"])
        row, evidence, raster = loaded_by_key[key]
        rgb = rgb_by_key[key]
        selected_frames.append(
            VerifiedV4Frame(
                index_row=row,
                evidence=evidence,
                target_raster_labels=raster,
                rgb_path=Path(str(rgb["canonical_rgb_path"])),
                image_sha256=str(rgb["rgb_file_sha256"]),
            )
        )
    verified_files = (
        (dataset_manifest_path, dataset_manifest_file_sha256),
        (audit_receipt_path, audit_receipt_file_sha256),
        (trainer_authorization_path, trainer_authorization_file_sha256),
        (trainer_review_record_path, trainer_review_record_file_sha256),
        *dataset_sources,
        *trainer_sources,
        *dataset_files,
    )
    return LoadedExactInputs(
        frames=tuple(selected_frames),
        manifest=manifest,
        audit_receipt=audit_receipt,
        rgb_receipt=manifest["rgb_receipt"],
        trainer_authorization=trainer_authorization,
        subset_receipt=subset_receipt,
        verified_files=tuple(verified_files),
    )


def revalidate_exact_inputs_after_training(
    inputs: LoadedExactInputs,
    *,
    dataset_manifest_path: Path,
    dataset_manifest_file_sha256: str,
    audit_receipt_path: Path,
    audit_receipt_file_sha256: str,
    trainer_authorization_path: Path,
    trainer_authorization_file_sha256: str,
    trainer_review_record_path: Path,
    trainer_review_record_file_sha256: str,
) -> dict[str, int]:
    """Re-enumerate and rehash the complete exact input closure post-training."""

    authorization, _raw = _strict_hashed_object(
        trainer_authorization_path,
        trainer_authorization_file_sha256,
        name="post-training V4 trainer authorization",
        allowed_root=ROOT,
    )
    if str(trainer_review_record_path) != str(CANONICAL_REVIEW_RECORD_PATH):
        raise PermissionError("post-training V4 review-record path is not canonical")
    trainer_sources = _validate_trainer_authorization(
        authorization,
        dataset_manifest_file_sha256=dataset_manifest_file_sha256,
        dataset_manifest_content_sha256=preauth_launcher.DATASET_MANIFEST_CONTENT_SHA256,
        audit_receipt_file_sha256=audit_receipt_file_sha256,
        audit_receipt_content_sha256=preauth_launcher.AUDIT_RECEIPT_CONTENT_SHA256,
        review_record_file_sha256=trainer_review_record_file_sha256,
    )
    manifest, _raw = _strict_hashed_object(
        dataset_manifest_path,
        dataset_manifest_file_sha256,
        name="post-training V4 dataset manifest",
        allowed_root=ROOT,
    )
    provenance = _validate_dataset_receipt(manifest)
    _validate_rgb_receipt(manifest["rgb_receipt"])
    audit, _raw = _strict_hashed_object(
        audit_receipt_path,
        audit_receipt_file_sha256,
        name="post-training V4 audit receipt",
        allowed_root=ROOT,
    )
    _validate_audit_receipt(
        audit,
        dataset_manifest=manifest,
        dataset_manifest_file_sha256=dataset_manifest_file_sha256,
    )
    loaded, dataset_files = _load_dataset_frames(
        dataset_manifest_path,
        manifest,
        audit_receipt_path=audit_receipt_path,
    )
    loaded_keys = {_canonical_json_bytes(row["frame_key"]) for row, _evidence, _raster in loaded}
    selected_keys = {_canonical_json_bytes(frame.frame_key) for frame in inputs.frames}
    if not selected_keys.issubset(loaded_keys):
        raise ValueError("post-training V4 selected subset changed")
    dataset_sources = tuple(
        (Path(str(record["path"])), str(record["sha256"]))
        for record in provenance["source_hashes"].values()
    )
    _verify_file_commitments(dataset_sources, name="post-training dataset source")
    _verify_file_commitments(trainer_sources, name="post-training trainer source")
    _verify_file_commitments(dataset_files, name="post-training dataset file")
    return {
        "dataset_root_inventory_revalidations": 1,
        "shard_directory_inventory_revalidations": 20,
        "dataset_frame_revalidations": len(loaded),
        "dataset_file_rehashes": len(dataset_files),
        "trainer_source_rehashes": len(trainer_sources),
        "dataset_source_rehashes": len(dataset_sources),
    }


def _decode_rgb_job(
    path_text: str,
    expected_sha256: str,
    allowed_root_text: str,
    expected_trainer_source_sha256: str,
) -> np.ndarray:
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    source_payload = _read_regular_bytes(
        Path(__file__),
        name="V4 RGB worker trainer source",
        allowed_root=ROOT,
    )
    if _sha256_bytes(source_payload) != expected_trainer_source_sha256:
        raise PermissionError("V4 RGB worker trainer source changed before RGB access")
    from PIL import Image

    path = Path(path_text)
    payload = _read_regular_bytes(
        path,
        name="selected train RGB",
        allowed_root=Path(allowed_root_text),
    )
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError("selected train RGB SHA-256 changed before decode")
    with Image.open(BytesIO(payload)) as image:
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or not np.isfinite(array).all():
        raise ValueError("selected train RGB decode is malformed")
    mean = np.asarray(NORMALIZATION_MEAN, dtype=np.float32)[None, None, :]
    std = np.asarray(NORMALIZATION_STD, dtype=np.float32)[None, None, :]
    return np.ascontiguousarray(((array - mean) / std).transpose(2, 0, 1))


def decode_selected_rgb(
    frames: Sequence[VerifiedV4Frame],
    *,
    maximum_workers: int = MAX_RGB_WORKERS,
    allowed_rgb_root: Path = ROOT,
    expected_trainer_source_sha256: str | None = None,
    worker_authorization_file_sha256: str | None = None,
    worker_review_record_file_sha256: str | None = None,
    worker_successor_review_file_sha256: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not frames:
        raise ValueError("at least one selected V4 frame is required")
    worker_limit = int(maximum_workers)
    if isinstance(maximum_workers, bool) or not 1 <= worker_limit <= MAX_RGB_WORKERS:
        raise ValueError(f"RGB workers must lie in [1,{MAX_RGB_WORKERS}]")
    authorized_root = allowed_rgb_root.resolve(strict=True)
    trainer_source_sha256 = (
        _sha256_bytes(
            _read_regular_bytes(
                Path(__file__),
                name="V4 trainer source before RGB dispatch",
                allowed_root=ROOT,
            )
        )
        if expected_trainer_source_sha256 is None
        else str(expected_trainer_source_sha256)
    )
    if not _is_sha256(trainer_source_sha256):
        raise ValueError("V4 trainer source SHA-256 is malformed")
    jobs = [
        (
            str(frame.rgb_path),
            frame.image_sha256,
            str(authorized_root),
            trainer_source_sha256,
        )
        for frame in frames
    ]
    worker_count = min(worker_limit, len(jobs))
    if worker_count == 1:
        arrays = [_decode_rgb_job(*job) for job in jobs]
    else:
        if (
            not _is_sha256(worker_authorization_file_sha256)
            or not _is_sha256(worker_review_record_file_sha256)
            or not _is_sha256(worker_successor_review_file_sha256)
        ):
            raise PermissionError(
                "V4 spawned RGB decoding requires fixed canonical authority bindings"
            )
        live_launcher = next(
            (
                module
                for module in tuple(sys.modules.values())
                if getattr(module, "_rgb_worker_terminal", None) is not None
                and getattr(module, "__name__", None) in {"__main__", "__mp_main__"}
                and Path(str(getattr(module, "__file__", ""))).resolve()
                == preauth_launcher.TRAINER_PATH.parent
                / "launch_go2_observable_camera_ray_fit_v4_v2.py"
            ),
            None,
        )
        if live_launcher is None or preauth_launcher._module_code_sha256(
            live_launcher
        ) != preauth_launcher._module_code_sha256(preauth_launcher):
            raise PermissionError("V4 spawned RGB terminal differs from captured source")
        worker_terminal = live_launcher._rgb_worker_terminal
        context = multiprocessing.get_context("spawn")
        previous = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
        try:
            for name in THREAD_ENVIRONMENT:
                os.environ[name] = "1"
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                arrays = list(
                    executor.map(
                        worker_terminal,
                        (
                            (
                                job,
                                worker_authorization_file_sha256,
                                worker_review_record_file_sha256,
                                worker_successor_review_file_sha256,
                            )
                            for job in jobs
                        ),
                    )
                )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    image = torch.from_numpy(np.stack(arrays, axis=0).copy())
    return image, {
        "selected_rgb_count": len(frames),
        "nonselected_rgb_opens": 0,
        "rgb_hash_opens": len(frames),
        "rgb_decodes": len(frames),
        "worker_start_method": "inline" if worker_count == 1 else "spawn",
        "worker_count": worker_count,
        "native_threads_per_worker": 1,
    }


def _batch_from_indices(
    frames: Sequence[VerifiedV4Frame],
    images: torch.Tensor,
    target_indices: Sequence[int],
    *,
    image_indices: Sequence[int] | None = None,
) -> V4Batch:
    indices = tuple(int(value) for value in target_indices)
    if not indices or any(value < 0 or value >= len(frames) for value in indices):
        raise ValueError("V4 batch target index is out of range")
    mapped = indices if image_indices is None else tuple(int(value) for value in image_indices)
    if len(mapped) != len(indices) or any(value < 0 or value >= len(frames) for value in mapped):
        raise ValueError("V4 batch image index is out of range")
    if tuple(images.shape) != (len(frames), 3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("decoded V4 image tensor shape changed")
    selected = [frames[index] for index in indices]
    return V4Batch(
        image=images[list(mapped)].clone(),
        camera_origin_body_m=torch.from_numpy(
            np.stack([frame.evidence.camera_origin_body_m for frame in selected]).copy()
        ).float(),
        camera_basis_body_fru=torch.from_numpy(
            np.stack([frame.evidence.camera_basis_body_fru for frame in selected]).copy()
        ).float(),
        ground_plane_z_body_m=torch.tensor(
            [frame.evidence.ground_plane_z_body_m for frame in selected],
            dtype=torch.float32,
        ),
        pixel_hit_mask=torch.from_numpy(
            np.stack([frame.evidence.pixel_hit_mask for frame in selected]).copy()
        ).bool(),
        pixel_first_hit_distance_m=torch.from_numpy(
            np.stack(
                [frame.evidence.pixel_first_hit_distance_m for frame in selected]
            ).copy()
        ).float(),
        ground_support_in_frustum=torch.from_numpy(
            np.stack(
                [frame.evidence.ground_support_in_frustum for frame in selected]
            ).copy()
        ).bool(),
        ground_support_clear_to_target=torch.from_numpy(
            np.stack(
                [frame.evidence.ground_support_clear_to_target for frame in selected]
            ).copy()
        ).bool(),
        target_raster_labels=torch.from_numpy(
            np.stack([frame.target_raster_labels for frame in selected]).copy()
        ).long(),
        families=tuple(frame.family for frame in selected),
    )


def _skew_balanced_pixel_offset_loss(raw_output: Any, targets: Any) -> torch.Tensor:
    offsets = raw_output.pixel_within_bin_offset_m
    selected = offsets.gather(1, targets.pixel_hit_bin_index[:, None]).squeeze(1)
    group_losses = []
    for depth_bin in range(offsets.shape[1]):
        mask = targets.pixel_in_range_hit_mask & (
            targets.pixel_hit_bin_index == depth_bin
        )
        if bool(mask.any().item()):
            group_losses.append(
                F.smooth_l1_loss(
                    selected[mask],
                    targets.pixel_within_bin_offset_m[mask].to(
                        dtype=selected.dtype
                    ),
                    beta=0.01,
                    reduction="mean",
                )
            )
    return torch.stack(group_losses).mean() if group_losses else offsets.sum() * 0.0


def compute_four_equal_v4_losses(
    model: torch.nn.Module,
    batch: V4Batch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any, Any, Any]:
    targets = derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=batch.pixel_hit_mask,
        pixel_first_hit_distance_m=batch.pixel_first_hit_distance_m,
        ground_support_in_frustum=batch.ground_support_in_frustum,
        ground_support_clear_to_target=batch.ground_support_clear_to_target,
    )
    raw_output = model(
        batch.image,
        batch.camera_origin_body_m,
        batch.camera_basis_body_fru,
        batch.ground_plane_z_body_m,
    )
    if not torch.equal(raw_output.ground_query_in_frustum, targets.ground_in_frustum):
        raise ValueError("model calibration does not reproduce V4 ground visibility")
    ordered = ordered_obstacle_first_hit_nll_breakdown_v4(
        raw_output.pixel_first_hit_hazard_logits,
        targets,
    ).total
    offset = _skew_balanced_pixel_offset_loss(raw_output, targets)
    ground = balanced_ground_clear_bce_v4(
        raw_output.ground_clear_to_target_logits,
        targets,
        raw_output.ground_target_distance_m,
    )
    soft_raster = soft_rasterize_observable_camera_ray_evidence_v4(
        raw_output,
        camera_origin_body_m=batch.camera_origin_body_m,
        camera_basis_body_fru=batch.camera_basis_body_fru,
    )
    raster = hierarchical_raster_cross_entropy_v4(
        soft_raster,
        batch.target_raster_labels,
    ).total
    components = {
        "ordered_first_hit_nll": ordered,
        "target_bin_offset_smooth_l1": offset,
        "ground_clear_distance_state_balanced_bce": ground,
        "derived_raster_hierarchical_bce": raster,
    }
    total = 0.25 * sum(components.values())
    return total, components, raw_output, targets, soft_raster


def _deterministic_training_batches(
    *,
    frame_count: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    if frame_count <= 0 or batch_size <= 0 or steps <= 0:
        raise ValueError("training schedule sizes must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    stream: list[int] = []
    required = int(batch_size) * int(steps)
    while len(stream) < required:
        stream.extend(torch.randperm(frame_count, generator=generator).tolist())
    return tuple(
        tuple(stream[start : start + batch_size])
        for start in range(0, required, batch_size)
    )


def train_v4_fit(
    *,
    model: torch.nn.Module,
    frames: Sequence[VerifiedV4Frame],
    images: torch.Tensor,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning rate must be positive and finite")
    if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
        raise ValueError("weight decay must be non-negative and finite")
    if int(batch_size) > len(frames):
        raise ValueError("batch size cannot exceed selected fit size")
    schedule = _deterministic_training_batches(
        frame_count=len(frames),
        batch_size=int(batch_size),
        steps=int(steps),
        seed=int(seed),
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    initial = None
    final = None
    best_total = float("inf")
    log_interval = max(1, int(steps) // 20)
    trace = []
    for step_index, indices in enumerate(schedule):
        batch = _batch_from_indices(frames, images, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        total, components, _raw, _targets, _raster = compute_four_equal_v4_losses(
            model, batch
        )
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("V4 fit loss became non-finite")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("V4 fit gradient norm became non-finite")
        optimizer.step()
        snapshot = {
            "step": step_index + 1,
            "total": float(total.detach().item()),
            "components": {
                name: float(value.detach().item()) for name, value in components.items()
            },
            "gradient_norm_before_clip": float(gradient_norm.detach().item()),
        }
        if initial is None:
            initial = snapshot
        final = snapshot
        best_total = min(best_total, snapshot["total"])
        if step_index == 0 or step_index + 1 == len(schedule) or (
            (step_index + 1) % log_interval == 0
        ):
            trace.append(snapshot)
    return {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "optimizer": "AdamW",
        "precision": "float32",
        "autocast": False,
        "gradient_clip_norm": 1.0,
        "loss_weights": {
            "ordered_first_hit_nll": 0.25,
            "target_bin_offset_smooth_l1": 0.25,
            "ground_clear_distance_state_balanced_bce": 0.25,
            "derived_raster_hierarchical_bce": 0.25,
        },
        "initial": initial,
        "final": final,
        "best_total": best_total,
        "trace": trace,
        "schedule_algorithm": SCHEDULE_ALGORITHM,
        "schedule_sha256": canonical_json_sha256(schedule),
    }


def evaluate_v4_fit(
    *,
    model: torch.nn.Module,
    frames: Sequence[VerifiedV4Frame],
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
    wrong_rgb: bool,
) -> dict[str, Any]:
    if not frames or int(batch_size) <= 0:
        raise ValueError("evaluation requires frames and a positive batch size")
    model.eval()
    accumulator = ObservableCameraRayFitV4MetricAccumulator()
    loss_sums: Counter[str] = Counter()
    loss_count = 0
    mapping = tuple(
        ((index + 1) % len(frames)) if wrong_rgb else index
        for index in range(len(frames))
    )
    with torch.no_grad():
        for start in range(0, len(frames), int(batch_size)):
            target_indices = tuple(range(start, min(start + int(batch_size), len(frames))))
            image_indices = tuple(mapping[index] for index in target_indices)
            batch = _batch_from_indices(
                frames,
                images,
                target_indices,
                image_indices=image_indices,
            ).to(device)
            total, components, raw, targets, soft_raster = compute_four_equal_v4_losses(
                model, batch
            )
            frame_weight = len(target_indices)
            loss_sums["total"] += float(total.item()) * frame_weight
            for name, value in components.items():
                loss_sums[name] += float(value.item()) * frame_weight
            loss_count += frame_weight
            accumulator.update(
                raw_output=raw,
                targets=targets,
                soft_raster=soft_raster,
                target_raster_labels=batch.target_raster_labels,
                families=batch.families,
            )
    return {
        "control": (
            "wrong_rgb_with_target_calibration" if wrong_rgb else "matched_rgb"
        ),
        "wrong_rgb_degenerate_singleton": bool(wrong_rgb and len(frames) == 1),
        "image_index_mapping": list(mapping),
        "image_mapping_sha256": canonical_json_sha256(list(mapping)),
        "losses": {
            name: value / loss_count for name, value in sorted(loss_sums.items())
        },
        "metrics": accumulator.finalize(),
    }


def validate_gpu0_r9700_runtime(
    *,
    device_text: str,
    environ: Mapping[str, str] | None = None,
    torch_module: Any = torch,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    if device_text not in {"cuda", "cuda:0"}:
        raise PermissionError("V4 exact training is pinned to cuda:0")
    if environment.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("HIP_VISIBLE_DEVICES must expose only GPU 0")
    if "HSA_OVERRIDE_GFX_VERSION" in environment:
        raise PermissionError("HSA_OVERRIDE_GFX_VERSION must be unset")
    wrong_threads = {
        name: environment.get(name)
        for name in THREAD_ENVIRONMENT
        if environment.get(name) != "1"
    }
    if wrong_threads:
        raise PermissionError(f"native thread caps must all equal 1: {wrong_threads}")
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise RuntimeError("exact V4 training requires exactly one visible HIP device")
    name = str(torch_module.cuda.get_device_name(0))
    normalized = "".join(character for character in name.casefold() if character.isalnum())
    if "r9700" not in normalized or "raphael" in normalized:
        raise PermissionError("GPU 0 must be the R9700; Raphael iGPU is forbidden")
    properties = torch_module.cuda.get_device_properties(0)
    total_memory = int(properties.total_memory)
    if total_memory < MIN_R9700_MEMORY_BYTES:
        raise RuntimeError("R9700 visible memory is below the 16 GiB fit contract")
    return {
        "device": "cuda:0",
        "device_name": name,
        "visible_device_count": 1,
        "total_memory_bytes": total_memory,
        "hip_visible_devices": "0",
        "hsa_override_gfx_version_unset": True,
        "raphael_rejected": True,
        "minimum_memory_bytes": MIN_R9700_MEMORY_BYTES,
        "native_thread_environment": {name: "1" for name in THREAD_ENVIRONMENT},
    }


def configure_determinism(seed: int) -> dict[str, Any]:
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be non-negative")
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": value,
        "requested": "strict_deterministic_algorithms",
        "effective": "strict_where_supported_warn_on_exact_allowlisted_kernels",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_attempt_directory(*, seed: int, fit_size: int) -> Path:
    if int(seed) not in EXPECTED_SEEDS or int(fit_size) not in SUPPORTED_FIT_SIZES:
        raise ValueError("V4 attempt seed/rung is outside the frozen ladder")
    return CANONICAL_ATTEMPT_ROOT / f"seed_{int(seed)}" / f"n{int(fit_size)}"


def _ensure_real_directory(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PermissionError(f"{name} is not a real directory")


def validate_v1_failure_lineage(
    *,
    root: Path = V1_DEVELOPMENT_OUTPUT_ROOT,
    expected: Mapping[str, Any] = ladder_gate.V1_FAILURE_LINEAGE,
) -> dict[str, Any]:
    """Revalidate the immutable V1 failure without touching V2 outputs."""

    root = Path(root)
    if root.resolve(strict=True) == CANONICAL_DEVELOPMENT_OUTPUT_ROOT.resolve():
        raise PermissionError("V1 lineage and V2 output roots are not separated")
    _ensure_real_directory(root, name="V4 immutable V1 development root")
    reservation_binding = expected.get("reservation")
    failure_binding = expected.get("failure")
    if (
        expected != ladder_gate.V1_FAILURE_LINEAGE
        or not isinstance(reservation_binding, Mapping)
        or not isinstance(failure_binding, Mapping)
    ):
        raise PermissionError("V1 failure lineage commitment changed")
    reservation_path = root / str(reservation_binding["path"])
    failure_path = root / str(failure_binding["path"])
    attempt = reservation_path.parent
    if failure_path.parent != attempt:
        raise PermissionError("V1 terminal files do not share the frozen attempt")
    for directory, name in (
        (root / "attempts", "V1 attempts root"),
        (root / "attempts/seed_20260710", "V1 seed root"),
        (attempt, "V1 failed N5 attempt"),
    ):
        _ensure_real_directory(directory, name=name)
    if {entry.name for entry in root.iterdir()} != {"attempts"} or {
        entry.name for entry in (root / "attempts").iterdir()
    } != {"seed_20260710"} or {
        entry.name for entry in (root / "attempts/seed_20260710").iterdir()
    } != {"n5"}:
        raise PermissionError("V1 development-root lineage inventory changed")
    entries = list(attempt.iterdir())
    if (
        sorted(entry.name for entry in entries) != expected["terminal_inventory"]
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise PermissionError("V1 failed-attempt inventory changed")
    reservation, _raw = _strict_hashed_object(
        reservation_path,
        str(reservation_binding["file_sha256"]),
        name="immutable V1 reservation",
        allowed_root=root,
    )
    failure, _raw = _strict_hashed_object(
        failure_path,
        str(failure_binding["file_sha256"]),
        name="immutable V1 failure",
        allowed_root=root,
    )
    if (
        reservation.get("content_sha256")
        != reservation_binding["content_sha256"]
        or reservation.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_attempt_reservation_v1"
        or reservation.get("contract") != "observable_camera_ray_fit_v4_ladder_v1"
        or reservation.get("seed") != 20260710
        or reservation.get("fit_size") != 5
        or reservation.get("attempt_index") != 1
        or reservation.get("maximum_attempts") != 1
        or failure.get("content_sha256") != failure_binding["content_sha256"]
        or failure.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_attempt_failure_v1"
        or failure.get("status") != "failed"
        or failure.get("failure")
        != {"code": "execution_failure", "class": "runtime"}
        or failure.get("partial_artifacts_removed") is not True
        or failure.get("reservation")
        != {
            "path": "reservation.json",
            "file_sha256": reservation_binding["file_sha256"],
            "content_sha256": reservation_binding["content_sha256"],
        }
    ):
        raise PermissionError("V1 failure lineage semantics changed")
    return json.loads(_canonical_json_bytes(expected))


def reserve_exact_attempt(
    *,
    seed: int,
    fit_size: int,
    inputs: Mapping[str, Any],
    prerequisite_gates: Mapping[str, Any],
) -> ExactAttemptReservation:
    """Atomically consume the sole canonical attempt before GPU/RGB work."""

    predecessor_failure = validate_v1_failure_lineage()
    base = CANONICAL_DEVELOPMENT_OUTPUT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    _ensure_real_directory(base, name="V4 development output root")
    CANONICAL_ATTEMPT_ROOT.mkdir(exist_ok=True)
    _ensure_real_directory(CANONICAL_ATTEMPT_ROOT, name="V4 attempt root")
    seed_directory = CANONICAL_ATTEMPT_ROOT / f"seed_{int(seed)}"
    seed_directory.mkdir(exist_ok=True)
    _ensure_real_directory(seed_directory, name="V4 seed attempt directory")
    directory = _canonical_attempt_directory(seed=seed, fit_size=fit_size)
    os.mkdir(directory, 0o755)
    core = {
        "schema": ATTEMPT_RESERVATION_SCHEMA,
        "contract": LADDER_CONTRACT,
        "predecessor_failure": predecessor_failure,
        "seed": int(seed),
        "fit_size": int(fit_size),
        "attempt_index": 1,
        "maximum_attempts": 1,
        "scope": "one_frozen_attempt_per_seed_and_fit_size",
        "inputs": dict(inputs),
        "prerequisite_gates": dict(prerequisite_gates),
        "licenses": {
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    reservation = {**core, "content_sha256": canonical_json_sha256(core)}
    payload = _canonical_json_bytes(reservation) + b"\n"
    try:
        _write_exclusive(directory / "reservation.json", payload)
        _validate_publication_directory(
            directory,
            expected_payloads={"reservation.json": payload},
        )
        _fsync_directory(directory)
        _fsync_directory(seed_directory)
        _fsync_directory(CANONICAL_ATTEMPT_ROOT)
        _fsync_directory(base)
    except BaseException:
        # The directory itself is the atomic attempt claim. Never remove it:
        # an interrupted reservation still consumes the one reviewed attempt.
        raise
    return ExactAttemptReservation(
        directory=directory,
        reservation=reservation,
        reservation_payload=payload,
        reservation_file_sha256=_sha256_bytes(payload),
    )


def _sanitized_failure(error: BaseException) -> dict[str, str]:
    if isinstance(error, FloatingPointError):
        return {"code": "nonfinite_training_failure", "class": "numeric"}
    if isinstance(error, PermissionError):
        return {"code": "scope_or_authorization_failure", "class": "permission"}
    if isinstance(error, ValueError):
        return {"code": "structural_validation_failure", "class": "validation"}
    if isinstance(error, OSError):
        return {"code": "filesystem_or_device_failure", "class": "io"}
    if isinstance(error, KeyboardInterrupt):
        return {"code": "operator_interruption", "class": "interruption"}
    if isinstance(error, RuntimeError):
        return {"code": "execution_failure", "class": "runtime"}
    return {"code": "unexpected_internal_failure", "class": "internal"}


def fail_reserved_exact_attempt(
    reservation: ExactAttemptReservation,
    *,
    error: BaseException,
) -> dict[str, Any]:
    """Remove partial artifacts and terminate a reserved attempt as failed."""

    for name in ("completed.json", "result.json", "checkpoint.pt"):
        (reservation.directory / name).unlink(missing_ok=True)
    core = {
        "schema": ATTEMPT_FAILURE_SCHEMA,
        "status": "failed",
        "reservation": reservation.binding,
        "failure": _sanitized_failure(error),
        "partial_artifacts_removed": True,
        "licenses": {
            "development_checkpoint_creation_authorized": False,
            "checkpoint_use_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    failed = {**core, "content_sha256": canonical_json_sha256(core)}
    payload = _canonical_json_bytes(failed) + b"\n"
    _write_exclusive(reservation.directory / "failed.json", payload)
    _validate_publication_directory(
        reservation.directory,
        expected_payloads={
            "reservation.json": reservation.reservation_payload,
            "failed.json": payload,
        },
    )
    _fsync_directory(reservation.directory)
    _fsync_directory(reservation.directory.parent)
    return {
        "path": str(reservation.directory / "failed.json"),
        "file_sha256": _sha256_bytes(payload),
        "content_sha256": failed["content_sha256"],
    }


def _checkpoint_bytes(
    model: torch.nn.Module,
    *,
    metadata: Mapping[str, Any],
) -> tuple[bytes, str]:
    state = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in sorted(model.state_dict().items())
    }
    state_manifest = [
        {
            "name": name,
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": _sha256_bytes(value.numpy().tobytes(order="C")),
        }
        for name, value in state.items()
    ]
    normalized_metadata = json.loads(_canonical_json_bytes(metadata))
    semantic_core = {
        "schema": "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2",
        "model_class": "ObservableCameraRayEvidenceV4Model",
        "state_manifest": state_manifest,
        "metadata": normalized_metadata,
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
    }
    content_sha256 = canonical_json_sha256(semantic_core)
    stream = BytesIO()
    torch.save(
        {
            **semantic_core,
            "state_dict": state,
            "content_sha256": content_sha256,
        },
        stream,
    )
    return stream.getvalue(), content_sha256


def _validate_publication_directory(
    directory: Path,
    *,
    expected_payloads: Mapping[str, bytes],
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise PermissionError("V4 publication directory is not a real directory")
    entries = list(directory.iterdir())
    if {entry.name for entry in entries} != set(expected_payloads):
        raise ValueError("V4 publication inventory changed")
    for entry in entries:
        if entry.is_symlink() or not stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode):
            raise PermissionError("V4 publication entry is not a regular file")
        payload = _read_regular_bytes(
            entry,
            name="V4 publication entry",
            allowed_root=directory,
        )
        if payload != expected_payloads[entry.name]:
            raise ValueError(f"V4 publication bytes changed: {entry.name}")


def _inject_publication_failure(
    requested: str | None,
    point: str,
) -> None:
    if requested == point:
        raise RuntimeError(f"injected V4 publication failure: {point}")


def publish_reserved_exact_attempt(
    reservation: ExactAttemptReservation,
    *,
    checkpoint_payload: bytes,
    checkpoint_content_sha256: str,
    result: Mapping[str, Any],
    failure_injection: str | None = None,
) -> dict[str, Any]:
    """Publish checkpoint/result and write the immutable completion last."""

    result_payload = _canonical_json_bytes(result) + b"\n"
    checkpoint_file_sha256 = _sha256_bytes(checkpoint_payload)
    result_file_sha256 = _sha256_bytes(result_payload)
    _validate_publication_directory(
        reservation.directory,
        expected_payloads={"reservation.json": reservation.reservation_payload},
    )
    _write_exclusive(reservation.directory / "checkpoint.pt", checkpoint_payload)
    _inject_publication_failure(failure_injection, "after_checkpoint_write")
    _write_exclusive(reservation.directory / "result.json", result_payload)
    _inject_publication_failure(failure_injection, "after_result_write")
    partial_payloads = {
        "reservation.json": reservation.reservation_payload,
        "checkpoint.pt": checkpoint_payload,
        "result.json": result_payload,
    }
    _validate_publication_directory(
        reservation.directory,
        expected_payloads=partial_payloads,
    )
    _fsync_directory(reservation.directory)
    completion_core = {
        "schema": ATTEMPT_COMPLETION_SCHEMA,
        "status": "completed",
        "reservation": reservation.binding,
        "checkpoint": {
            "path": "checkpoint.pt",
            "file_sha256": checkpoint_file_sha256,
            "content_sha256": checkpoint_content_sha256,
            "byte_count": len(checkpoint_payload),
        },
        "result": {
            "path": "result.json",
            "file_sha256": result_file_sha256,
            "content_sha256": result["content_sha256"],
            "byte_count": len(result_payload),
        },
        "inventory": [
            "checkpoint.pt",
            "completed.json",
            "reservation.json",
            "result.json",
        ],
        "licenses": {
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    completion = {
        **completion_core,
        "content_sha256": canonical_json_sha256(completion_core),
    }
    completion_payload = _canonical_json_bytes(completion) + b"\n"
    _write_exclusive(reservation.directory / "completed.json", completion_payload)
    _inject_publication_failure(failure_injection, "after_completion_write")
    expected_payloads = {**partial_payloads, "completed.json": completion_payload}
    _validate_publication_directory(
        reservation.directory,
        expected_payloads=expected_payloads,
    )
    _fsync_directory(reservation.directory)
    _fsync_directory(reservation.directory.parent)
    _fsync_directory(CANONICAL_ATTEMPT_ROOT)
    _fsync_directory(CANONICAL_DEVELOPMENT_OUTPUT_ROOT)
    _inject_publication_failure(failure_injection, "after_directory_fsync")
    _validate_publication_directory(
        reservation.directory,
        expected_payloads=expected_payloads,
    )
    return {
        "directory": str(reservation.directory),
        "reservation": reservation.binding,
        "checkpoint_path": str(reservation.directory / "checkpoint.pt"),
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_content_sha256": checkpoint_content_sha256,
        "result_path": str(reservation.directory / "result.json"),
        "result_file_sha256": result_file_sha256,
        "completion": {
            "path": str(reservation.directory / "completed.json"),
            "file_sha256": _sha256_bytes(completion_payload),
            "content_sha256": completion["content_sha256"],
        },
        "completion_written_last": True,
        "post_fsync_inventory_revalidated": True,
    }


def publish_immutable_development_result(
    *,
    output_root: Path,
    run_name: str,
    checkpoint_payload: bytes,
    result: Mapping[str, Any],
    enforce_canonical_root: bool = True,
    failure_injection: str | None = None,
) -> dict[str, Any]:
    root = output_root.resolve()
    if enforce_canonical_root and root != CANONICAL_DEVELOPMENT_OUTPUT_ROOT.resolve():
        raise PermissionError("exact V4 development output root is frozen")
    if enforce_canonical_root:
        try:
            root.relative_to(ROOT.resolve(strict=True))
        except ValueError as exc:
            raise PermissionError("V4 development output escapes the repository") from exc
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("V4 development run name is malformed")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable V4 development run exists: {destination}")
    result_payload = _canonical_json_bytes(result) + b"\n"
    expected_payloads = {
        "checkpoint.pt": checkpoint_payload,
        "result.json": result_payload,
    }
    staging = Path(tempfile.mkdtemp(prefix=".v4_fit.", dir=root))
    published = False
    try:
        _write_exclusive(staging / "checkpoint.pt", checkpoint_payload)
        _write_exclusive(staging / "result.json", result_payload)
        _validate_publication_directory(staging, expected_payloads=expected_payloads)
        _inject_publication_failure(failure_injection, "after_stage_validation")
        _fsync_directory(staging)
        destination.mkdir(mode=0o755, exist_ok=False)
        os.link(staging / "checkpoint.pt", destination / "checkpoint.pt")
        _validate_publication_directory(
            destination,
            expected_payloads={"checkpoint.pt": checkpoint_payload},
        )
        _fsync_directory(destination)
        _inject_publication_failure(failure_injection, "after_checkpoint_link")
        # Result manifest is linked last.
        os.link(staging / "result.json", destination / "result.json")
        _inject_publication_failure(failure_injection, "after_result_link")
        _validate_publication_directory(destination, expected_payloads=expected_payloads)
        _fsync_directory(destination)
        _fsync_directory(root)
        _inject_publication_failure(
            failure_injection, "after_post_link_validation"
        )
        _validate_publication_directory(destination, expected_payloads=expected_payloads)
        published = True
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not published and destination.exists():
            shutil.rmtree(destination)
    return {
        "directory": str(destination),
        "checkpoint_path": str(destination / "checkpoint.pt"),
        "result_path": str(destination / "result.json"),
        "checkpoint_file_sha256": _sha256_bytes(checkpoint_payload),
        "result_file_sha256": _sha256_bytes(result_payload),
        "post_link_inventory_revalidated": True,
        "result_manifest_linked_last": True,
    }


def _synthetic_smoke_frame() -> tuple[VerifiedV4Frame, torch.Tensor]:
    origin = np.asarray((0.0, 0.0, 0.6), dtype=np.float32)
    basis = np.asarray(
        ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    ground_z = np.float32(-0.6)
    query = project_canonical_ground_support_v4(
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        ground_plane_z_body_m=float(ground_z),
    )
    in_frustum = query.in_frustum.copy()
    row, column, support = np.indices(in_frustum.shape)
    clear = in_frustum & ((row + 2 * column + support) % 3 != 0)
    pixel_hit = np.zeros(PIXEL_RAY_SHAPE, dtype=bool)
    pixel_distance = np.zeros(PIXEL_RAY_SHAPE, dtype=np.float32)
    flat_indices = np.arange(pixel_hit.size)[::97]
    pixel_hit.reshape(-1)[flat_indices] = True
    pixel_distance.reshape(-1)[flat_indices] = (
        0.25 + (flat_indices % 40).astype(np.float32) * 0.1
    )
    evidence = ObservableCameraRayEvidenceV4(
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        ground_plane_z_body_m=float(ground_z),
        ground_support_in_frustum=in_frustum,
        ground_support_clear_to_target=clear,
        pixel_hit_mask=pixel_hit,
        pixel_first_hit_distance_m=pixel_distance,
    )
    raster = rasterize_observable_camera_ray_evidence_v4(evidence)
    digest = "0" * 64
    frame_key = {
        "dataset_role": "train",
        "family": FAMILIES[0],
        "scene_id": "synthetic_smoke",
        "global_row": 0,
        "side": "current",
        "image_sha256": digest,
        "label_shard_sha256": "1" * 64,
        "label_row": 0,
    }
    index_core = {
        "schema": INDEX_ROW_SCHEMA,
        "frame_key": frame_key,
        "image_sha256_commitment_only": digest,
        "sidecar_row_identity_sha256": "2" * 64,
        "evidence_content_sha256": evidence.content_sha256(),
        "raster_content_sha256": raster.content_sha256(),
    }
    index_row = {**index_core, "content_sha256": canonical_json_sha256(index_core)}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260712)
    image = torch.rand((3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    mean = image.new_tensor(NORMALIZATION_MEAN)[:, None, None]
    std = image.new_tensor(NORMALIZATION_STD)[:, None, None]
    return (
        VerifiedV4Frame(
            index_row=index_row,
            evidence=evidence,
            target_raster_labels=raster.output_labels.copy(),
            rgb_path=ROOT / "synthetic_smoke_only.png",
            image_sha256=digest,
        ),
        (image - mean) / std,
    )


def run_smoke() -> dict[str, Any]:
    determinism = configure_determinism(20260712)
    frame, image = _synthetic_smoke_frame()
    model = ObservableCameraRayEvidenceV4Model(encoder_depth=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        training = train_v4_fit(
            model=model,
            frames=(frame,),
            images=image[None],
            device=torch.device("cpu"),
            steps=1,
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=0.0,
            seed=20260712,
        )
        matched = evaluate_v4_fit(
            model=model,
            frames=(frame,),
            images=image[None],
            device=torch.device("cpu"),
            batch_size=1,
            wrong_rgb=False,
        )
        wrong = evaluate_v4_fit(
            model=model,
            frames=(frame,),
            images=image[None],
            device=torch.device("cpu"),
            batch_size=1,
            wrong_rgb=True,
        )
    warning_receipt = validate_determinism_warnings([item.message for item in caught])
    core = {
        "schema": RESULT_SCHEMA,
        "mode": "synthetic_smoke",
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "exact_data_opened": False,
        "gpu_used": False,
        "heldout_opened": False,
        "g2_opened": False,
        "runtime_opened": False,
        "model_class": "ObservableCameraRayEvidenceV4Model",
        "encoder_depth": 0,
        "training": training,
        "evaluation": {"matched_rgb": matched, "wrong_rgb": wrong},
        "determinism": {**determinism, **warning_receipt},
        "licenses": {
            "model_output_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _require_captured_private_trainer(preauthorization: Mapping[str, Any]) -> None:
    review = preauthorization.get("successor_review")
    sources = review.get("successor_sources") if isinstance(review, Mapping) else None
    binding = (
        sources.get(preauth_launcher.SUCCESSOR_TRAINER_RELATIVE_PATH)
        if isinstance(sources, Mapping)
        else None
    )
    expected = binding.get("file_sha256") if isinstance(binding, Mapping) else None
    actual = _sha256_bytes(
        _read_regular_bytes(
            ROOT / preauth_launcher.SUCCESSOR_TRAINER_RELATIVE_PATH,
            name="V4 V2 captured trainer source",
            allowed_root=ROOT,
        )
    )
    if not sys.flags.isolated or not _is_sha256(expected) or actual != expected:
        raise PermissionError("V4 trainer library execution is unsupported")


def _run_captured_exact(
    args: argparse.Namespace,
    *,
    preauthorization: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_captured_private_trainer(preauthorization)
    preauth_launcher.validate_execution_args(args)
    _preauthorization = dict(preauthorization)
    fit_size = int(args.fit_size)
    steps = int(args.steps)
    preflight_exact_frozen_dataset_provenance(
        dataset_manifest_path=Path(args.dataset_manifest),
        dataset_manifest_file_sha256=str(args.dataset_manifest_sha256),
    )
    validate_v1_failure_lineage()
    prerequisites = load_exact_prerequisite_gate_bindings(args)
    inputs = load_exact_inputs(
        dataset_manifest_path=Path(args.dataset_manifest),
        dataset_manifest_file_sha256=str(args.dataset_manifest_sha256),
        audit_receipt_path=Path(args.audit_receipt),
        audit_receipt_file_sha256=str(args.audit_receipt_sha256),
        trainer_authorization_path=Path(args.trainer_authorization),
        trainer_authorization_file_sha256=str(args.trainer_authorization_sha256),
        trainer_review_record_path=Path(args.trainer_review_record),
        trainer_review_record_file_sha256=str(args.trainer_review_record_sha256),
        fit_size=fit_size,
    )
    target_partition = validate_exact_target_partition_v4(
        inputs.frames,
        fit_size=fit_size,
    )
    reservation_inputs = {
        "dataset_manifest_file_sha256": str(args.dataset_manifest_sha256),
        "dataset_manifest_content_sha256": inputs.manifest["content_sha256"],
        "audit_receipt_file_sha256": str(args.audit_receipt_sha256),
        "audit_receipt_content_sha256": inputs.audit_receipt["content_sha256"],
        "trainer_authorization_file_sha256": str(args.trainer_authorization_sha256),
        "trainer_authorization_content_sha256": inputs.trainer_authorization[
            "content_sha256"
        ],
        "trainer_review_record_file_sha256": str(args.trainer_review_record_sha256),
        "trainer_review_record_content_sha256": _preauthorization[
            "review_record_content_sha256"
        ],
        "rgb_receipt_content_sha256": inputs.rgb_receipt["content_sha256"],
        "subset_content_sha256": inputs.subset_receipt["content_sha256"],
        "target_partition": target_partition,
        "source_map_sha256": _preauthorization["source_map_sha256"],
    }
    reservation = reserve_exact_attempt(
        seed=int(args.seed),
        fit_size=fit_size,
        inputs=reservation_inputs,
        prerequisite_gates=prerequisites,
    )
    try:
        resource = validate_gpu0_r9700_runtime(device_text=str(args.device))
        determinism = configure_determinism(int(args.seed))
        images, rgb_access = decode_selected_rgb(
            inputs.frames,
            maximum_workers=int(args.rgb_workers),
            worker_authorization_file_sha256=str(
                args.trainer_authorization_sha256
            ),
            worker_review_record_file_sha256=str(
                args.trainer_review_record_sha256
            ),
            worker_successor_review_file_sha256=str(
                args.successor_review_sha256
            ),
            expected_trainer_source_sha256=str(
                _preauthorization["successor_review"]["successor_sources"]
                [preauth_launcher.SUCCESSOR_TRAINER_RELATIVE_PATH]["file_sha256"]
            ),
        )
        model = ObservableCameraRayEvidenceV4Model()
        device = torch.device("cuda:0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = train_v4_fit(
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                steps=steps,
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                seed=int(args.seed),
            )
            training = {
                **training,
                "evaluation_batch_size": int(args.eval_batch_size),
            }
            matched = evaluate_v4_fit(
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                batch_size=int(args.eval_batch_size),
                wrong_rgb=False,
            )
            wrong = evaluate_v4_fit(
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                batch_size=int(args.eval_batch_size),
                wrong_rgb=True,
            )
        warning_receipt = validate_determinism_warnings(
            [item.message for item in caught]
        )
        post_training_revalidation = revalidate_exact_inputs_after_training(
            inputs,
            dataset_manifest_path=Path(args.dataset_manifest),
            dataset_manifest_file_sha256=str(args.dataset_manifest_sha256),
            audit_receipt_path=Path(args.audit_receipt),
            audit_receipt_file_sha256=str(args.audit_receipt_sha256),
            trainer_authorization_path=Path(args.trainer_authorization),
            trainer_authorization_file_sha256=str(args.trainer_authorization_sha256),
            trainer_review_record_path=Path(args.trainer_review_record),
            trainer_review_record_file_sha256=str(args.trainer_review_record_sha256),
        )
        selected_rgb_commitments = tuple(
            (frame.rgb_path, frame.image_sha256) for frame in inputs.frames
        )
        _verify_file_commitments(selected_rgb_commitments, name="selected train RGB")
        checkpoint_metadata = {
            **reservation_inputs,
            "fit_size": fit_size,
            "seed": int(args.seed),
            "training_schedule_sha256": training["schedule_sha256"],
            "attempt_reservation": reservation.binding,
            "predecessor_failure": reservation.reservation["predecessor_failure"],
            "prerequisite_gates": prerequisites,
        }
        checkpoint_payload, checkpoint_content_sha256 = _checkpoint_bytes(
            model,
            metadata=checkpoint_metadata,
        )
        checkpoint_sha256 = _sha256_bytes(checkpoint_payload)
        checkpoint_byte_count = len(checkpoint_payload)
        result_inputs = {
            key: reservation_inputs[key]
            for key in (
                "dataset_manifest_file_sha256",
                "dataset_manifest_content_sha256",
                "audit_receipt_file_sha256",
                "audit_receipt_content_sha256",
                "trainer_authorization_file_sha256",
                "trainer_authorization_content_sha256",
                "trainer_review_record_file_sha256",
                "trainer_review_record_content_sha256",
                "rgb_receipt_content_sha256",
            )
        }
        result_inputs["target_partition_content_sha256"] = target_partition[
            "content_sha256"
        ]
        if prerequisites["previous_stage_gate"] is not None:
            result_inputs["previous_stage_gate"] = prerequisites[
                "previous_stage_gate"
            ]
        if prerequisites["seed_20260710_gate"] is not None:
            result_inputs["seed_20260710_gate"] = prerequisites[
                "seed_20260710_gate"
            ]
        core = {
            "schema": RESULT_SCHEMA,
            "mode": "exact_development_fit",
            "authoritative": False,
            "aggregation_eligible": False,
            "promotion_eligible": False,
            "dataset_role": "train",
            "fit_size": fit_size,
            "attempt": {
                "attempt_index": 1,
                "maximum_attempts": 1,
                "scope": "one_frozen_attempt_per_seed_and_fit_size",
                "reservation": reservation.binding,
                "predecessor_failure": reservation.reservation[
                    "predecessor_failure"
                ],
            },
            "subset": inputs.subset_receipt,
            "target_partition": target_partition,
            "inputs": result_inputs,
            "model": {
                "class": "ObservableCameraRayEvidenceV4Model",
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "checkpoint": {
                    "path": "checkpoint.pt",
                    "file_sha256": checkpoint_sha256,
                    "content_sha256": checkpoint_content_sha256,
                    "byte_count": checkpoint_byte_count,
                    "development_only": True,
                },
            },
            "training": training,
            "evaluation": {
                "matched_rgb": matched,
                "wrong_rgb_with_target_calibration": wrong,
            },
            "resource": resource,
            "determinism": {**determinism, **warning_receipt},
            "access_ledger": {
                **rgb_access,
                **post_training_revalidation,
                "selected_rgb_rehashes_before_publication": fit_size,
                "heldout_opens": 0,
                "g2_opens": 0,
                "runtime_opens": 0,
                "nonselected_rgb_opens": 0,
                "gpu1_uses": 0,
            },
            "licenses": {
                "development_checkpoint_creation_authorized": True,
                "checkpoint_use_authorized": False,
                "holdout_authorized": False,
                "g2_authorized": False,
                "runtime_authorized": False,
                "promotion_authorized": False,
            },
        }
        result = {**core, "content_sha256": canonical_json_sha256(core)}
        publication = publish_reserved_exact_attempt(
            reservation,
            checkpoint_payload=checkpoint_payload,
            checkpoint_content_sha256=checkpoint_content_sha256,
            result=result,
        )
        return result, publication
    except BaseException as error:
        try:
            fail_reserved_exact_attempt(reservation, error=error)
        except BaseException as terminal_error:
            raise RuntimeError("V4 attempt failed and terminal receipt could not be written") from terminal_error
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one synthetic CPU update without opening exact artifacts.",
    )
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--audit-receipt", type=Path)
    parser.add_argument("--audit-receipt-sha256")
    parser.add_argument("--trainer-authorization", type=Path)
    parser.add_argument("--trainer-authorization-sha256")
    parser.add_argument("--trainer-review-record", type=Path)
    parser.add_argument("--trainer-review-record-sha256")
    parser.add_argument("--successor-review", type=Path)
    parser.add_argument("--successor-review-sha256")
    parser.add_argument("--fit-size", type=int, choices=SUPPORTED_FIT_SIZES)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS)
    parser.add_argument("--rgb-workers", type=int, default=MAX_RGB_WORKERS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--previous-stage-gate", type=Path)
    parser.add_argument("--previous-stage-gate-sha256")
    parser.add_argument("--seed-20260710-gate", type=Path)
    parser.add_argument("--seed-20260710-gate-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        exact_values = (
            args.dataset_manifest,
            args.dataset_manifest_sha256,
            args.audit_receipt,
            args.audit_receipt_sha256,
            args.trainer_authorization,
            args.trainer_authorization_sha256,
            args.trainer_review_record,
            args.trainer_review_record_sha256,
            args.successor_review,
            args.successor_review_sha256,
            args.previous_stage_gate,
            args.previous_stage_gate_sha256,
            args.seed_20260710_gate,
            args.seed_20260710_gate_sha256,
        )
        if any(value is not None for value in exact_values):
            raise ValueError("synthetic smoke may not receive exact artifact paths")
        result = run_smoke()
        print((_canonical_json_bytes(result) + b"\n").decode("ascii"), end="")
        return 0
    raise PermissionError(
        "exact V4 fitting requires the launcher's live in-memory capability"
    )


def _captured_exact_cli(
    args: argparse.Namespace,
    preauthorization: Mapping[str, Any],
) -> int:
    result, publication = _run_captured_exact(
        args,
        preauthorization=preauthorization,
    )
    summary = {
        "schema": RESULT_SCHEMA,
        "content_sha256": result["content_sha256"],
        "fit_size": result["fit_size"],
        "authoritative": False,
        "publication": publication,
    }
    print((_canonical_json_bytes(summary) + b"\n").decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
