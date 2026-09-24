#!/usr/bin/env python3
"""Build the immutable train-fit observable camera-ray V4 dataset.

The default dry run uses only synthetic geometry.  Exact mode is deliberately
double gated by the already-frozen source-access manifest and a separately
reviewed V4 implementation manifest.  It may open current physical-train
metadata and the train attitude sidecar, but never RGB, fit labels, non-train
sidecars, checkpoints, model outputs, runtime data, G2, or sealed payloads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lewm_worlds") not in sys.path:
    sys.path.insert(0, str(ROOT / "lewm_worlds"))

PURE_EVIDENCE_PATH = (
    ROOT / "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py"
).resolve()
SOURCE_ACCESS_PATH = (
    ROOT / "scripts/audit_go2_n32_camera_frustum_observability.py"
).resolve()
DYNAMIC_PROJECTION_PATH = (
    ROOT / "lewm/benchmarks/go2_dynamic_cell_square_projection.py"
).resolve()
ATTITUDE_SIDECAR_PATH = (ROOT / "lewm/datasets/go2_attitude_sidecar.py").resolve()


def _load_neutral_module(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load neutral module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ray_v4 = _load_neutral_module(
    "go2_observable_camera_ray_evidence_v4_neutral", PURE_EVIDENCE_PATH
)
CAMERA_NEAR_M = ray_v4.CAMERA_NEAR_M
EVIDENCE_SCHEMA = ray_v4.EVIDENCE_SCHEMA
PIXEL_RAY_SHAPE = ray_v4.PIXEL_RAY_SHAPE
ObservableCameraRayEvidenceV4 = ray_v4.ObservableCameraRayEvidenceV4
calibrated_pixel_ray_directions_body_v4 = (
    ray_v4.calibrated_pixel_ray_directions_body_v4
)
project_canonical_ground_support_v4 = ray_v4.project_canonical_ground_support_v4
rasterize_observable_camera_ray_evidence_v4 = (
    ray_v4.rasterize_observable_camera_ray_evidence_v4
)


DATASET_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_dataset_v1"
SHARD_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_shard_v1"
INDEX_ROW_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_index_row_v1"
RGB_RECEIPT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_rgb_receipt_v1"
IMPLEMENTATION_MANIFEST_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_v1"
)
CANONICAL_OUTPUT = ROOT / ".generated/go2_observable_camera_ray_fit_v4/v1"
IMPLEMENTATION_MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_2026-07-12.json"
)

SOURCE_AUTHORIZATION_MANIFEST_PATH = (
    ROOT
    / (
        "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_"
        "implementation_manifest_2026-07-11.json"
    )
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
PANEL_FILE_SHA256 = (
    "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
)
PANEL_CONTENT_SHA256 = (
    "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
)
FIT_ROWS_SHA256 = (
    "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
)
SOURCE_GEOMETRY_MANIFEST_SHA256 = (
    "beddb29b9826d7a21968effea863d040a6cfc9849ab0b2a78c4105d28dbb37d2"
)
RENDER_SUMMARIES_MANIFEST_SHA256 = (
    "9fff0ee9933ee582e4452f15c58f44dae721379bc983f2046d0b87498d1d002f"
)
SIDECAR_MANIFEST_PATH = (
    ROOT / ".generated/go2_attitude_sidecar/dynamic_cartesian_v1/manifest.json"
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

MAX_WORKERS = 6
RAY_CHUNK_SIZE = 8192
GROUND_CLEARANCE_ABS_TOLERANCE_M = 1e-9
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ARRAY_LAYOUT = (
    ("camera_origin_body_f32.bin", "<f4", (3,)),
    ("camera_basis_body_fru_f32.bin", "<f4", (3, 3)),
    ("ground_plane_z_body_f32.bin", "<f4", (1,)),
    ("ground_support_in_frustum_u8.bin", "u1", (128, 128, 5)),
    ("ground_support_clear_to_target_u8.bin", "u1", (128, 128, 5)),
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
    "fit_label_payload_byte_opens",
    "sidecar_checkpoint_selection_role_byte_opens",
    "sidecar_probability_calibration_role_byte_opens",
    "sidecar_g2_evaluation_role_byte_opens",
)
EXACT_DENIAL_PRIMARY_REASONS = (
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
EXACT_DENIAL_MODALITIES = (
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
SOURCE_ACCESS_LEDGER_FIELDS = frozenset(
    {
        "panel_metadata_byte_opens",
        "label_shard_hash_byte_opens",
        "label_shard_npz_opens",
        "registered_arrays_decompressed",
        "materialized_label_rows",
        "materialized_supervision_rows",
        "per_shard_materialization",
        "selected_label_rows_read",
        "selected_supervision_rows_read",
        "unselected_row_values_inspected",
        "unselected_row_metrics_computed",
        "unselected_rows_retained",
        "derivative_shard_or_cache_writes",
        "source_geometry_hash_byte_opens",
        "source_geometry_json_parses",
        "source_geometry_jsonl_records",
        "source_frame_records_selected",
        "implementation_source_hash_byte_opens",
        "document_hash_byte_opens",
        "unexpected_path_attempts",
        "denied_attempts_total",
        "denied_primary_reasons",
        "denied_modality_attempts",
        "denied_attempt_records",
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
    }
)
EXACT_BUILD_LEDGER_FIELDS = SOURCE_ACCESS_LEDGER_FIELDS | {
    "sidecar_manifest_byte_opens",
    "sidecar_train_role_byte_opens",
    "sidecar_checkpoint_selection_role_byte_opens",
    "sidecar_probability_calibration_role_byte_opens",
    "sidecar_g2_evaluation_role_byte_opens",
    "fit_label_payload_byte_opens",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(path, _canonical_json_bytes(payload) + b"\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class RayBoxV4:
    """One rendered oriented box expressed in a frame's yaw-body basis."""

    center_body_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    rotation_body_from_box: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        center = np.asarray(self.center_body_m, dtype=np.float64)
        half = np.asarray(self.half_size_m, dtype=np.float64)
        rotation = np.asarray(self.rotation_body_from_box, dtype=np.float64)
        if center.shape != (3,) or half.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("ray box has malformed geometry")
        if not np.isfinite(center).all() or not np.isfinite(half).all():
            raise ValueError("ray box geometry must be finite")
        if np.any(half <= 0.0):
            raise ValueError("ray box half sizes must be positive")
        if not np.allclose(
            rotation @ rotation.T,
            np.eye(3),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("ray box rotation must be orthonormal")


@dataclass(frozen=True)
class FrameBuildInputV4:
    frame_key: Mapping[str, Any]
    camera_origin_body_m: tuple[float, float, float]
    camera_basis_body_fru: tuple[tuple[float, float, float], ...]
    ground_plane_z_body_m: float
    rendered_boxes_body: tuple[RayBoxV4, ...]
    image_path_metadata_only: str
    image_sha256: str
    sidecar_row_identity_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame_key, Mapping) or not self.frame_key:
            raise ValueError("frame key must be a nonempty mapping")
        if not isinstance(self.image_path_metadata_only, str) or not self.image_path_metadata_only:
            raise ValueError("frame RGB metadata path must be a nonempty string")
        if not _is_sha256(self.image_sha256) or not _is_sha256(
            self.sidecar_row_identity_sha256
        ):
            raise ValueError("frame image/sidecar identity hashes are malformed")


@dataclass(frozen=True)
class SceneBuildJobV4:
    scene_key: str
    frames: tuple[FrameBuildInputV4, ...]

    def __post_init__(self) -> None:
        if not self.scene_key or not self.frames:
            raise ValueError("scene job must have a key and frames")


def _ray_box_entry_distances(
    origin: np.ndarray,
    directions: np.ndarray,
    box: RayBoxV4,
) -> np.ndarray:
    center = np.asarray(box.center_body_m, dtype=np.float64)
    half = np.asarray(box.half_size_m, dtype=np.float64)
    rotation = np.asarray(box.rotation_body_from_box, dtype=np.float64)
    local_origin = rotation.T @ (origin - center)
    local_directions = directions @ rotation
    lower = np.full(directions.shape[0], -np.inf, dtype=np.float64)
    upper = np.full(directions.shape[0], np.inf, dtype=np.float64)
    valid = np.ones(directions.shape[0], dtype=bool)
    for axis in range(3):
        component = local_directions[:, axis]
        parallel = np.abs(component) <= 1e-12
        valid &= ~(parallel & (abs(local_origin[axis]) > half[axis] + 1e-12))
        first = np.full(component.shape, -np.inf, dtype=np.float64)
        second = np.full(component.shape, np.inf, dtype=np.float64)
        active = ~parallel
        first[active] = (-half[axis] - local_origin[axis]) / component[active]
        second[active] = (half[axis] - local_origin[axis]) / component[active]
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        lower = np.maximum(lower, low)
        upper = np.minimum(upper, high)
    entry = np.maximum(lower, 0.0)
    valid &= upper + 1e-12 >= entry
    return np.where(valid, entry, np.inf)


def _nearest_box_hits(
    origin: np.ndarray,
    directions: np.ndarray,
    boxes: Sequence[RayBoxV4],
) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("ray directions must have shape [N,3]")
    result = np.full(directions.shape[0], np.inf, dtype=np.float64)
    for start in range(0, directions.shape[0], RAY_CHUNK_SIZE):
        stop = min(start + RAY_CHUNK_SIZE, directions.shape[0])
        chunk = directions[start:stop]
        nearest = np.full(chunk.shape[0], np.inf, dtype=np.float64)
        for box in boxes:
            nearest = np.minimum(
                nearest, _ray_box_entry_distances(origin, chunk, box)
            )
        result[start:stop] = nearest
    return result


def _ground_support_clear(
    in_frustum: np.ndarray,
    first_hit_distance_m: np.ndarray,
    target_distance_m: np.ndarray,
) -> np.ndarray:
    in_view = np.asarray(in_frustum, dtype=bool)
    first = np.asarray(first_hit_distance_m, dtype=np.float64)
    target = np.asarray(target_distance_m, dtype=np.float64)
    if in_view.shape != first.shape or first.shape != target.shape:
        raise ValueError("ground support clearance arrays must have matching shapes")
    return in_view & (first >= target - GROUND_CLEARANCE_ABS_TOLERANCE_M)


def build_frame_evidence_v4(frame: FrameBuildInputV4) -> ObservableCameraRayEvidenceV4:
    origin_f32 = np.asarray(frame.camera_origin_body_m, dtype=np.float32)
    basis_f32 = np.asarray(frame.camera_basis_body_fru, dtype=np.float32)
    ground_z_f32 = np.float32(frame.ground_plane_z_body_m)
    origin = origin_f32.astype(np.float64)
    basis = basis_f32.astype(np.float64)
    ground = project_canonical_ground_support_v4(
        camera_origin_body_m=origin,
        camera_basis_body_fru=basis,
        ground_plane_z_body_m=float(ground_z_f32),
    )
    support_points = np.empty((*ground.in_frustum.shape, 3), dtype=np.float64)
    # Recover exact query points from calibrated direction and distance.  This
    # avoids maintaining a second grid convention in the builder.
    support_points[...] = ray_v4.canonical_ground_support_points_body_m(
        ground_z_body_m=float(ground_z_f32)
    )
    relative = support_points.reshape(-1, 3) - origin[None, :]
    target_distance = np.linalg.norm(relative, axis=1)
    directions = np.zeros_like(relative)
    active = target_distance > 1e-12
    directions[active] = relative[active] / target_distance[active, None]
    first_hit = _nearest_box_hits(origin, directions, frame.rendered_boxes_body)
    clear = _ground_support_clear(
        ground.in_frustum.reshape(-1), first_hit, target_distance
    )

    pixel_directions = calibrated_pixel_ray_directions_body_v4(basis)
    pixel_first = _nearest_box_hits(
        origin,
        pixel_directions.reshape(-1, 3),
        frame.rendered_boxes_body,
    ).reshape(PIXEL_RAY_SHAPE)
    hit_mask = np.isfinite(pixel_first) & (pixel_first > CAMERA_NEAR_M)
    hit_distance = np.zeros(PIXEL_RAY_SHAPE, dtype=np.float32)
    hit_distance[hit_mask] = pixel_first[hit_mask].astype(np.float32)
    return ObservableCameraRayEvidenceV4(
        camera_origin_body_m=origin_f32,
        camera_basis_body_fru=basis_f32,
        ground_plane_z_body_m=ground_z_f32,
        ground_support_in_frustum=ground.in_frustum,
        ground_support_clear_to_target=clear.reshape(ground.in_frustum.shape),
        pixel_hit_mask=hit_mask,
        pixel_first_hit_distance_m=hit_distance,
    )


def _frame_arrays(evidence: ObservableCameraRayEvidenceV4) -> tuple[np.ndarray, ...]:
    return (
        np.ascontiguousarray(evidence.camera_origin_body_m, dtype="<f4"),
        np.ascontiguousarray(evidence.camera_basis_body_fru, dtype="<f4"),
        np.ascontiguousarray(
            np.asarray(evidence.ground_plane_z_body_m), dtype="<f4"
        ),
        np.ascontiguousarray(evidence.ground_support_in_frustum, dtype=np.uint8),
        np.ascontiguousarray(
            evidence.ground_support_clear_to_target, dtype=np.uint8
        ),
        np.ascontiguousarray(evidence.pixel_hit_mask, dtype=np.uint8),
        np.ascontiguousarray(evidence.pixel_first_hit_distance_m, dtype="<f4"),
    )


def _write_scene_job(
    job: SceneBuildJobV4,
    staging_root: str,
    source_closure_entries: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if source_closure_entries is not None:
        _validate_source_entries(source_closure_entries)
    staging = Path(staging_root)
    scene_digest = hashlib.sha256(job.scene_key.encode("utf-8")).hexdigest()
    directory_name = f"{scene_digest[:16]}"
    directory = staging / directory_name
    directory.mkdir(parents=False, exist_ok=False)
    ordered_frames = tuple(
        sorted(job.frames, key=lambda item: _canonical_json_bytes(item.frame_key))
    )
    evidences = [build_frame_evidence_v4(frame) for frame in ordered_frames]
    frame_arrays = [_frame_arrays(evidence) for evidence in evidences]
    files: list[dict[str, Any]] = []
    for array_index, (filename, dtype, trailing_shape) in enumerate(ARRAY_LAYOUT):
        values = np.stack(
            [arrays[array_index] for arrays in frame_arrays], axis=0
        ).astype(np.dtype(dtype), copy=False)
        expected_shape = (len(ordered_frames), *trailing_shape)
        if values.shape != expected_shape:
            raise AssertionError(f"{filename} has shape {values.shape}, expected {expected_shape}")
        payload = values.tobytes(order="C")
        path = directory / filename
        _write_bytes_exclusive(path, payload)
        files.append(
            {
                "path": filename,
                "dtype": np.dtype(dtype).str,
                "shape": list(expected_shape),
                "byte_count": len(payload),
                "file_sha256": _sha256_bytes(payload),
            }
        )

    index_rows: list[dict[str, Any]] = []
    for frame, evidence in zip(ordered_frames, evidences):
        raster = rasterize_observable_camera_ray_evidence_v4(evidence)
        core = {
            "schema": INDEX_ROW_SCHEMA,
            "frame_key": json.loads(_canonical_json_bytes(frame.frame_key)),
            "image_sha256_commitment_only": frame.image_sha256,
            "sidecar_row_identity_sha256": frame.sidecar_row_identity_sha256,
            "evidence_content_sha256": evidence.content_sha256(),
            "raster_content_sha256": raster.content_sha256(),
        }
        index_rows.append(_with_content_sha256(core))
    index_payload = b"".join(
        _canonical_json_bytes(row) + b"\n" for row in index_rows
    )
    _write_bytes_exclusive(directory / "index.jsonl", index_payload)
    files.append(
        {
            "path": "index.jsonl",
            "dtype": "canonical_jsonl",
            "shape": [len(index_rows)],
            "byte_count": len(index_payload),
            "file_sha256": _sha256_bytes(index_payload),
        }
    )
    core = {
        "schema": SHARD_SCHEMA,
        "scene_key_sha256": scene_digest,
        "frame_count": len(ordered_frames),
        "ordered_frame_keys_sha256": canonical_json_sha256(
            [row["frame_key"] for row in index_rows]
        ),
        "ordered_evidence_sha256": canonical_json_sha256(
            [row["evidence_content_sha256"] for row in index_rows]
        ),
        "files": sorted(files, key=lambda item: item["path"]),
    }
    shard = _with_content_sha256(core)
    _write_json_exclusive(directory / "shard.json", shard)
    _fsync_directory(directory)
    _validate_shard_directory(directory, shard)
    if source_closure_entries is not None:
        _validate_source_entries(source_closure_entries)
    return {
        "directory_name": directory_name,
        "staging_path": str(directory),
        "shard": shard,
        "source_map_sha256": (
            None
            if source_closure_entries is None
            else canonical_json_sha256(source_closure_entries)
        ),
    }


def _validate_shard_directory(
    directory: Path,
    shard: Mapping[str, Any],
) -> None:
    expected_data_names = {name for name, _dtype, _shape in ARRAY_LAYOUT} | {
        "index.jsonl"
    }
    expected_names = expected_data_names | {"shard.json"}
    observed = {entry.name for entry in directory.iterdir()}
    if observed != expected_names:
        raise ValueError("staged shard file inventory changed")
    for name in sorted(observed):
        path = directory / name
        if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
            raise PermissionError("staged shard entries must be regular files")
    core = dict(shard)
    declared = core.pop("content_sha256", None)
    if declared != canonical_json_sha256(core):
        raise ValueError("staged shard content SHA-256 changed")
    expected_shard_bytes = _canonical_json_bytes(shard) + b"\n"
    if (directory / "shard.json").read_bytes() != expected_shard_bytes:
        raise ValueError("staged shard manifest bytes changed")
    files = shard.get("files")
    if not isinstance(files, list):
        raise ValueError("staged shard lacks its file inventory")
    records = {
        str(record.get("path", "")): record
        for record in files
        if isinstance(record, Mapping)
    }
    if set(records) != expected_data_names or len(records) != len(files):
        raise ValueError("staged shard declared file inventory changed")
    for name, record in records.items():
        payload = (directory / name).read_bytes()
        if (
            int(record.get("byte_count", -1)) != len(payload)
            or record.get("file_sha256") != _sha256_bytes(payload)
        ):
            raise ValueError(f"staged shard file changed: {name}")


def _validate_published_inventory(
    destination: Path,
    shard_directory_names: Sequence[str],
    *,
    manifest_present: bool,
) -> None:
    expected_root = {"shards"} | ({"manifest.json"} if manifest_present else set())
    observed_root = {entry.name for entry in destination.iterdir()}
    if observed_root != expected_root:
        raise ValueError("published dataset root inventory changed")
    shards_root = destination / "shards"
    if shards_root.is_symlink() or not stat.S_ISDIR(
        shards_root.stat(follow_symlinks=False).st_mode
    ):
        raise PermissionError("published shards root must be a regular directory")
    expected_shards = set(shard_directory_names)
    observed_shards = {entry.name for entry in shards_root.iterdir()}
    if observed_shards != expected_shards:
        raise ValueError("published shard-directory inventory changed")
    for name in sorted(observed_shards):
        path = shards_root / name
        if path.is_symlink() or not stat.S_ISDIR(
            path.stat(follow_symlinks=False).st_mode
        ):
            raise PermissionError("published shard entry must be a regular directory")
    if manifest_present:
        manifest_path = destination / "manifest.json"
        if manifest_path.is_symlink() or not stat.S_ISREG(
            manifest_path.stat(follow_symlinks=False).st_mode
        ):
            raise PermissionError("published manifest must be a regular file")


def _validate_jobs(jobs: Sequence[SceneBuildJobV4]) -> tuple[SceneBuildJobV4, ...]:
    ordered = tuple(sorted(jobs, key=lambda job: job.scene_key))
    if not ordered or len({job.scene_key for job in ordered}) != len(ordered):
        raise ValueError("scene jobs must be nonempty with unique keys")
    frame_keys = [
        _canonical_json_bytes(frame.frame_key)
        for job in ordered
        for frame in job.frames
    ]
    if len(set(frame_keys)) != len(frame_keys):
        raise ValueError("scene jobs repeat a frame key")
    return ordered


def _canonical_rgb_metadata_path(value: object) -> str:
    """Validate frozen RGB metadata lexically without touching RGB bytes."""

    if not isinstance(value, str) or not value:
        raise ValueError("RGB receipt path must be a nonempty string")
    path = Path(value)
    normalized = os.path.normpath(value)
    if not path.is_absolute() or value != normalized or ".." in path.parts:
        raise PermissionError("RGB receipt path is not canonical and absolute")
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise PermissionError("RGB receipt path escapes the repository") from exc
    if path.parent.name != "rgb":
        raise PermissionError("RGB receipt path is not under a render rgb directory")
    return value


def _validate_rgb_receipt(
    receipt: object,
    *,
    expected_frames: Mapping[bytes, str] | None = None,
) -> dict[bytes, dict[str, Any]]:
    if not isinstance(receipt, Mapping):
        raise ValueError("dataset lacks an RGB metadata receipt")
    expected_fields = {
        "schema",
        "dataset_role",
        "frame_count",
        "ordered_frame_keys_sha256",
        "entries_sha256",
        "rgb_byte_opens",
        "entries",
        "content_sha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("RGB metadata receipt fields changed")
    core = dict(receipt)
    declared = core.pop("content_sha256", None)
    rgb_byte_opens = receipt.get("rgb_byte_opens")
    if (
        receipt.get("schema") != RGB_RECEIPT_SCHEMA
        or receipt.get("dataset_role") != "train"
        or declared != canonical_json_sha256(core)
        or isinstance(rgb_byte_opens, bool)
        or not isinstance(rgb_byte_opens, int)
        or rgb_byte_opens != 0
    ):
        raise ValueError("RGB metadata receipt identity changed")
    entries = receipt.get("entries")
    frame_count = receipt.get("frame_count")
    if (
        not isinstance(entries, list)
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 0
    ):
        raise ValueError("RGB metadata receipt entries are malformed")
    if frame_count != len(entries):
        raise ValueError("RGB metadata receipt frame count changed")
    if receipt.get("entries_sha256") != canonical_json_sha256(entries):
        raise ValueError("RGB metadata receipt entries hash changed")

    by_key: dict[bytes, dict[str, Any]] = {}
    ordered_keys: list[dict[str, Any]] = []
    paths: set[str] = set()
    hashes: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "frame_key",
            "canonical_rgb_path",
            "rgb_file_sha256",
        }:
            raise ValueError("one RGB metadata receipt entry is malformed")
        frame_key = entry.get("frame_key")
        if not isinstance(frame_key, dict) or frame_key.get("dataset_role") != "train":
            raise PermissionError("RGB metadata receipt contains a nontrain frame")
        key = _canonical_json_bytes(frame_key)
        path = _canonical_rgb_metadata_path(entry.get("canonical_rgb_path"))
        digest = entry.get("rgb_file_sha256")
        if not _is_sha256(digest):
            raise ValueError("RGB metadata receipt hash is malformed")
        if "image_sha256" in frame_key and frame_key["image_sha256"] != digest:
            raise ValueError("RGB metadata receipt disagrees with its frame key")
        if key in by_key or path in paths or str(digest) in hashes:
            raise ValueError("RGB metadata receipt entries are not injective")
        normalized = {
            "frame_key": json.loads(key),
            "canonical_rgb_path": path,
            "rgb_file_sha256": str(digest),
        }
        if dict(entry) != normalized:
            raise ValueError("RGB metadata receipt entry is not canonical")
        by_key[key] = normalized
        ordered_keys.append(normalized["frame_key"])
        paths.add(path)
        hashes.add(str(digest))
    if list(by_key) != sorted(by_key):
        raise ValueError("RGB metadata receipt entries are not canonically ordered")
    if receipt.get("ordered_frame_keys_sha256") != canonical_json_sha256(
        ordered_keys
    ):
        raise ValueError("RGB metadata receipt ordered frame keys changed")
    if expected_frames is not None and (
        set(by_key) != set(expected_frames)
        or any(
            by_key[key]["rgb_file_sha256"] != expected_digest
            for key, expected_digest in expected_frames.items()
        )
    ):
        raise ValueError("RGB metadata receipt does not match dataset frames")
    return by_key


def _rgb_receipt_from_jobs(
    jobs: Sequence[SceneBuildJobV4],
) -> dict[str, Any]:
    entries = sorted(
        (
            {
                "frame_key": json.loads(_canonical_json_bytes(frame.frame_key)),
                "canonical_rgb_path": _canonical_rgb_metadata_path(
                    frame.image_path_metadata_only
                ),
                "rgb_file_sha256": frame.image_sha256,
            }
            for job in jobs
            for frame in job.frames
        ),
        key=lambda entry: _canonical_json_bytes(entry["frame_key"]),
    )
    core = {
        "schema": RGB_RECEIPT_SCHEMA,
        "dataset_role": "train",
        "frame_count": len(entries),
        "ordered_frame_keys_sha256": canonical_json_sha256(
            [entry["frame_key"] for entry in entries]
        ),
        "entries_sha256": canonical_json_sha256(entries),
        "rgb_byte_opens": 0,
        "entries": entries,
    }
    receipt = _with_content_sha256(core)
    expected = {
        _canonical_json_bytes(frame.frame_key): frame.image_sha256
        for job in jobs
        for frame in job.frames
    }
    _validate_rgb_receipt(receipt, expected_frames=expected)
    return receipt


def build_dataset_from_jobs(
    jobs: Sequence[SceneBuildJobV4],
    *,
    output_directory: Path,
    workers: int,
    input_provenance: Mapping[str, Any],
    access_ledger: Mapping[str, Any],
    source_closure_entries: Sequence[Mapping[str, str]] | None = None,
    required_output_root: Path | None = None,
) -> dict[str, Any]:
    """Build and publish deterministic shards; used by exact and synthetic paths."""

    ordered_jobs = _validate_jobs(jobs)
    if isinstance(workers, bool) or not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must lie in [1,{MAX_WORKERS}]")
    destination = output_directory.resolve()
    if required_output_root is not None:
        root = required_output_root.resolve(strict=True)
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise PermissionError("V4 output escapes the required repository root") from exc
    if destination.exists():
        raise FileExistsError(f"immutable V4 dataset already exists: {destination}")
    staging: Path | None = None
    destination_owned = False
    try:
        normalized_source_entries = (
            None
            if source_closure_entries is None
            else list(_validate_source_entries(list(source_closure_entries)))
        )
        authorized_source_hashes = input_provenance.get("source_hashes")
        if normalized_source_entries is not None:
            _validate_exact_parent_input_files(input_provenance)
            _validate_authorized_source_hashes(authorized_source_hashes)
            _validate_exact_build_ledger_schema(access_ledger)
        rgb_receipt = _rgb_receipt_from_jobs(ordered_jobs)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=".go2_ray_v4.", dir=destination.parent)
        ).resolve()
        context = multiprocessing.get_context("spawn")
        previous_thread_environment = {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        }
        try:
            for name in THREAD_ENVIRONMENT:
                os.environ[name] = "1"
            with ProcessPoolExecutor(
                max_workers=int(workers), mp_context=context
            ) as executor:
                futures = [
                    executor.submit(_write_scene_job, job, str(staging))
                    if normalized_source_entries is None
                    else executor.submit(
                        _write_scene_job,
                        job,
                        str(staging),
                        normalized_source_entries,
                    )
                    for job in ordered_jobs
                ]
                results = [future.result() for future in futures]
        finally:
            for name, previous in previous_thread_environment.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
        results.sort(key=lambda result: result["shard"]["scene_key_sha256"])
        if normalized_source_entries is not None:
            expected_source_map = canonical_json_sha256(normalized_source_entries)
            _validate_source_entries(normalized_source_entries)
            _validate_exact_parent_input_files(input_provenance)
            _validate_authorized_source_hashes(authorized_source_hashes)
            if any(
                result.get("source_map_sha256") != expected_source_map
                for result in results
            ):
                raise RuntimeError("one worker did not validate the frozen source map")
        for result in results:
            _validate_shard_directory(
                Path(result["staging_path"]), result["shard"]
            )

        destination.mkdir(mode=0o755, parents=False, exist_ok=False)
        destination_owned = True
        (destination / "shards").mkdir(mode=0o755, exist_ok=False)
        published: list[dict[str, Any]] = []
        published_directory_names: list[str] = []
        for ordinal, result in enumerate(results):
            source = Path(result["staging_path"])
            _validate_shard_directory(source, result["shard"])
            name = f"{ordinal:03d}_{result['directory_name']}"
            published_directory_names.append(name)
            target = destination / "shards" / name
            target.mkdir(mode=0o755, exist_ok=False)
            for source_file in sorted(source.iterdir(), key=lambda item: item.name):
                os.link(source_file, target / source_file.name)
            _fsync_directory(target)
            _validate_shard_directory(target, result["shard"])
            published.append(
                {
                    "path": f"shards/{name}/shard.json",
                    "scene_key_sha256": result["shard"]["scene_key_sha256"],
                    "frame_count": result["shard"]["frame_count"],
                    "content_sha256": result["shard"]["content_sha256"],
                    "file_sha256": _sha256_file(target / "shard.json"),
                }
            )

        if normalized_source_entries is not None:
            _validate_source_entries(normalized_source_entries)
            _validate_exact_parent_input_files(input_provenance)
            _validate_authorized_source_hashes(authorized_source_hashes)
        for ordinal, result in enumerate(results):
            name = f"{ordinal:03d}_{result['directory_name']}"
            _validate_shard_directory(
                destination / "shards" / name, result["shard"]
            )
        _validate_published_inventory(
            destination,
            published_directory_names,
            manifest_present=False,
        )

        frame_count = sum(int(item["frame_count"]) for item in published)
        core = {
            "schema": DATASET_SCHEMA,
            "evidence_schema": EVIDENCE_SCHEMA,
            "dataset_role": "train",
            "frame_count": frame_count,
            "scene_shard_count": len(published),
            "array_layout": [
                {
                    "path": name,
                    "dtype": np.dtype(dtype).str,
                    "trailing_shape": list(shape),
                }
                for name, dtype, shape in ARRAY_LAYOUT
            ],
            "rgb_receipt": rgb_receipt,
            "shards": published,
            "input_provenance": json.loads(_canonical_json_bytes(input_provenance)),
            "access_ledger": json.loads(_canonical_json_bytes(access_ledger)),
            "parallel_contract": {
                "worker_start_method": "spawn",
                "maximum_workers": MAX_WORKERS,
                "native_threads_per_worker": 1,
                "canonical_merge": "scene_hash_then_canonical_frame_key",
                "worker_count_does_not_change_artifact_bytes": True,
                "per_worker_source_revalidation": (
                    normalized_source_entries is not None
                ),
                "parent_source_revalidation_before_manifest": (
                    normalized_source_entries is not None
                ),
            },
            "publication": "private_staging_hardlink_no_replace_manifest_last",
            "licenses": {
                "model_output_authorized": False,
                "holdout_authorized": False,
                "g2_authorized": False,
                "runtime_authorized": False,
                "promotion_authorized": False,
            },
        }
        manifest = _with_content_sha256(core)
        _write_json_exclusive(destination / "manifest.json", manifest)
        _validate_published_inventory(
            destination,
            published_directory_names,
            manifest_present=True,
        )
        _fsync_directory(destination / "shards")
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
        return manifest
    except Exception:
        if destination_owned and destination.exists():
            shutil.rmtree(destination)
        raise
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _rotation_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)), dtype=np.float64
    )
    rotation_y = np.asarray(
        ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)), dtype=np.float64
    )
    rotation_z = np.asarray(
        ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
    )
    return rotation_z @ rotation_y @ rotation_x


def _box_in_yaw_body(
    raw_box: Any,
    *,
    base_position_world: Sequence[float],
    stored_yaw_rad: float,
) -> RayBoxV4:
    base = np.asarray(base_position_world, dtype=np.float64)
    center_world = np.asarray(raw_box.center_xyz_m, dtype=np.float64)
    size = np.asarray(raw_box.size_xyz_m, dtype=np.float64)
    yaw = float(stored_yaw_rad)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    body_from_world = np.asarray(
        ((cos_yaw, sin_yaw, 0.0), (-sin_yaw, cos_yaw, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    world_from_box = _rotation_xyz(
        float(getattr(raw_box, "roll_rad", 0.0)),
        float(getattr(raw_box, "pitch_rad", 0.0)),
        float(getattr(raw_box, "yaw_rad", 0.0)),
    )
    return RayBoxV4(
        center_body_m=tuple((body_from_world @ (center_world - base)).tolist()),
        half_size_m=tuple((0.5 * size).tolist()),
        rotation_body_from_box=tuple(
            tuple(row) for row in (body_from_world @ world_from_box).tolist()
        ),
    )


def _scene_sha256(scene_id: str) -> str:
    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()


def _assert_zero_access_denials(ledger: Mapping[str, Any]) -> None:
    records = ledger.get("denied_attempt_records")
    primary = ledger.get("denied_primary_reasons")
    modalities = ledger.get("denied_modality_attempts")
    if (
        not isinstance(records, list)
        or not isinstance(primary, Mapping)
        or set(primary) != set(EXACT_DENIAL_PRIMARY_REASONS)
        or not isinstance(modalities, Mapping)
        or set(modalities) != set(EXACT_DENIAL_MODALITIES)
    ):
        raise ValueError("exact access denial receipt schema changed")

    def strict_count(value: object, *, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"exact access denial count is malformed: {name}")
        return value

    total = strict_count(
        ledger.get("denied_attempts_total"), name="denied_attempts_total"
    )
    unexpected = strict_count(
        ledger.get("unexpected_path_attempts"), name="unexpected_path_attempts"
    )
    primary_total = sum(
        strict_count(value, name=f"denied_primary_reasons.{name}")
        for name, value in primary.items()
    )
    modality_total = sum(
        strict_count(value, name=f"denied_modality_attempts.{name}")
        for name, value in modalities.items()
    )
    if not (
        total == unexpected == primary_total == modality_total == len(records)
    ):
        raise ValueError("exact access denial totals and records disagree")
    if (
        unexpected != 0
        or total != 0
        or records
        or any(value != 0 for value in primary.values())
        or any(value != 0 for value in modalities.values())
    ):
        raise PermissionError("exact loader recorded a denied or unexpected access")


def _validate_exact_build_ledger_schema(ledger: object) -> Mapping[str, Any]:
    if not isinstance(ledger, Mapping) or set(ledger) != EXACT_BUILD_LEDGER_FIELDS:
        raise ValueError("exact build access-ledger fields changed")
    if ledger.get("per_shard_materialization") != []:
        raise PermissionError("exact builder retained label-shard materialization")
    _assert_zero_access_denials(ledger)
    nonscalar = {
        "per_shard_materialization",
        "denied_primary_reasons",
        "denied_modality_attempts",
        "denied_attempt_records",
    }
    for name in EXACT_BUILD_LEDGER_FIELDS - nonscalar:
        value = ledger[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"exact build access-ledger count changed: {name}")
    return ledger


def _validated_sidecar_source_attitude(
    source_frame: Mapping[str, Any],
    endpoint: Mapping[str, Any],
) -> tuple[tuple[float, ...], float]:
    composition = source_frame.get("camera_mount_composition")
    if not isinstance(composition, Mapping) or composition.get("passes") is not True:
        raise ValueError("selected source frame camera composition did not pass")
    source_quaternion = tuple(
        float(value) for value in source_frame.get("base_quat_world_xyzw", ())
    )
    sidecar_quaternion = tuple(
        float(value) for value in endpoint.get("base_quat_world_xyzw", ())
    )
    source_yaw = float(source_frame.get("base_rpy_rad", {}).get("yaw", math.nan))
    sidecar_yaw = float(endpoint.get("stored_base_yaw_rad", math.nan))
    composition_quaternion = tuple(
        float(value) for value in composition.get("base_quat_world_xyzw", ())
    )
    composition_yaw = float(composition.get("stored_base_yaw_rad", math.nan))
    if (
        len(source_quaternion) != 4
        or source_quaternion != sidecar_quaternion
        or source_quaternion != composition_quaternion
        or not math.isfinite(source_yaw)
        or source_yaw != sidecar_yaw
        or source_yaw != composition_yaw
    ):
        raise ValueError("sidecar attitude disagrees with selected source frame")
    return sidecar_quaternion, sidecar_yaw


def _normalized_camera_basis_fru(camera: Any) -> tuple[tuple[float, ...], ...]:
    forward = np.asarray(camera.forward_xyz, dtype=np.float64)
    up_hint = np.asarray(camera.up_xyz, dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    if not math.isfinite(forward_norm) or forward_norm <= 1e-12:
        raise ValueError("camera forward basis is degenerate")
    forward /= forward_norm
    right = np.cross(forward, up_hint)
    right_norm = float(np.linalg.norm(right))
    if not math.isfinite(right_norm) or right_norm <= 1e-12:
        raise ValueError("camera right basis is degenerate")
    right /= right_norm
    up = np.cross(right, forward)
    up_norm = float(np.linalg.norm(up))
    if not math.isfinite(up_norm) or up_norm <= 1e-12:
        raise ValueError("camera up basis is degenerate")
    up /= up_norm
    return tuple(tuple(float(value) for value in axis) for axis in (forward, right, up))


TRANSITIVE_SOURCE_ROLES = frozenset(
    {
        "contract",
        "pure_evidence",
        "builder",
        "auditor",
        "source_access_runner",
        "source_audit_core",
        "source_label_semantics",
        "source_geometry_contract_semantics",
        "source_scene_manifest_semantics",
        "source_planning_grid_semantics",
        "dynamic_projection",
        "attitude_sidecar",
        "pure_test",
        "builder_test",
        "auditor_test",
    }
)


def _validate_source_entries(entries: object) -> tuple[dict[str, str], ...]:
    if not isinstance(entries, list):
        raise ValueError("implementation source entries must be a list")
    observed_roles: set[str] = set()
    normalized: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "role", "sha256"}:
            raise ValueError("one implementation source entry is malformed")
        role = str(entry["role"])
        relative = Path(str(entry["path"]))
        if (
            role in observed_roles
            or role not in TRANSITIVE_SOURCE_ROLES
            or relative.is_absolute()
        ):
            raise ValueError("implementation source roles or paths changed")
        observed_roles.add(role)
        lexical = ROOT / relative
        if lexical.is_symlink():
            raise PermissionError("implementation source may not be a symlink")
        resolved = lexical.resolve(strict=True)
        try:
            resolved.relative_to(ROOT.resolve(strict=True))
        except ValueError as exc:
            raise PermissionError("implementation source escapes the repository") from exc
        digest = str(entry["sha256"])
        if not _is_sha256(digest) or _sha256_file(resolved) != digest:
            raise ValueError(f"implementation source changed: {role}")
        normalized.append(
            {"path": str(relative), "role": role, "sha256": digest}
        )
    if observed_roles != TRANSITIVE_SOURCE_ROLES:
        raise ValueError("implementation source role closure changed")
    return tuple(normalized)


def _validate_authorized_source_hashes(source_hashes: object) -> None:
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("authorized semantic source hashes are missing")
    for role, record in source_hashes.items():
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"authorized semantic source record changed: {role}")
        path = Path(str(record["path"]))
        if path.is_symlink():
            raise PermissionError("authorized semantic source may not be a symlink")
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(ROOT.resolve(strict=True))
        except ValueError as exc:
            raise PermissionError("authorized semantic source escapes repository") from exc
        if _sha256_file(resolved) != record["sha256"]:
            raise ValueError(f"authorized semantic source changed: {role}")


def _validate_exact_parent_input_files(provenance: Mapping[str, Any]) -> None:
    commitments = (
        (
            IMPLEMENTATION_MANIFEST_PATH,
            provenance.get("implementation_manifest_file_sha256"),
            "implementation manifest",
        ),
        (
            SOURCE_AUTHORIZATION_MANIFEST_PATH,
            provenance.get("source_authorization_manifest_file_sha256"),
            "source authorization manifest",
        ),
        (
            SIDECAR_MANIFEST_PATH,
            provenance.get("sidecar_manifest_file_sha256"),
            "attitude sidecar manifest",
        ),
    )
    for path, expected, label in commitments:
        if not _is_sha256(expected) or _sha256_file(path) != expected:
            raise ValueError(f"exact parent input changed: {label}")


def _load_reviewed_implementation_manifest(
    expected_file_sha256: str,
    *,
    required_authorization: str,
) -> dict[str, Any]:
    if not _is_sha256(expected_file_sha256):
        raise ValueError("implementation-manifest file SHA-256 is malformed")
    raw = IMPLEMENTATION_MANIFEST_PATH.read_bytes()
    if _sha256_bytes(raw) != expected_file_sha256:
        raise ValueError("implementation-manifest file SHA-256 changed")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("implementation manifest is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != IMPLEMENTATION_MANIFEST_SCHEMA:
        raise ValueError("implementation manifest schema changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if declared != canonical_json_sha256(core):
        raise ValueError("implementation manifest content SHA-256 changed")
    source_map = value.get("source_map")
    if not isinstance(source_map, Mapping):
        raise ValueError("implementation manifest lacks a source map")
    entries = source_map.get("entries")
    if (
        not isinstance(entries, list)
        or int(source_map.get("entry_count", -1)) != len(entries)
        or source_map.get("source_map_sha256") != canonical_json_sha256(entries)
    ):
        raise ValueError("implementation source map changed")
    _validate_source_entries(entries)
    frozen_inputs = value.get("frozen_inputs")
    expected_inputs = {
        "source_authorization_manifest_file_sha256": (
            SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256
        ),
        "source_authorization_manifest_content_sha256": (
            SOURCE_AUTHORIZATION_MANIFEST_CONTENT_SHA256
        ),
        "source_authorization_binding_sha256": SOURCE_AUTHORIZATION_BINDING_SHA256,
        "fit_panel_file_sha256": PANEL_FILE_SHA256,
        "fit_panel_content_sha256": PANEL_CONTENT_SHA256,
        "fit_rows_sha256": FIT_ROWS_SHA256,
        "source_geometry_manifest_sha256": SOURCE_GEOMETRY_MANIFEST_SHA256,
        "render_summaries_manifest_sha256": RENDER_SUMMARIES_MANIFEST_SHA256,
        "fit_label_shards_manifest_sha256": (
            "998ce5a768029c23c931fbbec730c1fe31b9ed1fe155494fc68f34a0c23d3d1b"
        ),
        "sidecar_manifest_file_sha256": SIDECAR_MANIFEST_FILE_SHA256,
        "sidecar_manifest_content_sha256": SIDECAR_MANIFEST_CONTENT_SHA256,
        "sidecar_train_file_sha256": SIDECAR_TRAIN_FILE_SHA256,
        "sidecar_train_content_sha256": SIDECAR_TRAIN_CONTENT_SHA256,
        "sidecar_train_ordered_global_sha256": SIDECAR_TRAIN_ORDERED_GLOBAL_SHA256,
        "sidecar_train_ordered_identity_sha256": (
            SIDECAR_TRAIN_ORDERED_IDENTITY_SHA256
        ),
    }
    if frozen_inputs != expected_inputs:
        raise ValueError("implementation frozen-input commitments changed")
    authorization_fields = {
        "build": "exact_fit_build_authorized_after_review",
        "audit": "exact_fit_audit_authorized_after_review",
    }
    if required_authorization not in authorization_fields:
        raise ValueError("required authorization must be build or audit")
    field = authorization_fields[required_authorization]
    if value.get(field) is not True:
        raise PermissionError(
            f"exact fit {required_authorization} is not authorized after review"
        )
    return value


def load_exact_fit_jobs(
    *,
    machine_manifest_sha256: str,
    implementation_manifest_sha256: str,
) -> tuple[tuple[SceneBuildJobV4, ...], dict[str, Any], dict[str, Any]]:
    """Load fit metadata/geometry and train attitude only; never fit labels/RGB."""

    implementation = _load_reviewed_implementation_manifest(
        implementation_manifest_sha256,
        required_authorization="build",
    )
    if machine_manifest_sha256 != SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256:
        raise PermissionError("source authorization manifest hash is not frozen")
    source_access = _load_neutral_module(
        "go2_n32_camera_frustum_source_access_v4_neutral", SOURCE_ACCESS_PATH
    )
    if (
        tuple(source_access.PRIMARY_DENIAL_REASONS)
        != EXACT_DENIAL_PRIMARY_REASONS
        or tuple(source_access.DENIAL_MODALITIES) != EXACT_DENIAL_MODALITIES
    ):
        raise ValueError("source access denial schema changed")

    ledger = source_access.new_access_ledger()
    if set(ledger) != SOURCE_ACCESS_LEDGER_FIELDS:
        raise ValueError("source access-ledger fields changed")
    spec = source_access.AuditSpec()
    source_hashes = source_access._source_hashes(spec.sources(), ledger=ledger)
    if source_hashes["binding"]["sha256"] != SOURCE_AUTHORIZATION_BINDING_SHA256:
        raise ValueError("source access binding changed")
    machine_manifest = source_access._load_machine_manifest(
        machine_manifest_sha256,
        source_hashes=source_hashes,
        ledger=ledger,
    )
    if machine_manifest.get("content_sha256") != SOURCE_AUTHORIZATION_MANIFEST_CONTENT_SHA256:
        raise ValueError("source authorization content changed")
    authorized = machine_manifest["authorized_inputs"]
    if (
        authorized["fit_panel"]["file_sha256"] != PANEL_FILE_SHA256
        or authorized["fit_panel"]["content_sha256"] != PANEL_CONTENT_SHA256
        or authorized["fit_panel"]["fit_rows_sha256"] != FIT_ROWS_SHA256
        or authorized["source_geometry"]["manifest_sha256"]
        != SOURCE_GEOMETRY_MANIFEST_SHA256
        or authorized["render_summaries"]["manifest_sha256"]
        != RENDER_SUMMARIES_MANIFEST_SHA256
    ):
        raise ValueError("source authorization inventory changed")
    source_access._load_authorized_semantics(source_hashes)
    dynamic_projection = source_access._load_authorized_module(
        "lewm.benchmarks.go2_dynamic_cell_square_projection",
        DYNAMIC_PROJECTION_PATH,
    )
    attitude_sidecar = source_access._load_authorized_module(
        "lewm.datasets.go2_attitude_sidecar",
        ATTITUDE_SIDECAR_PATH,
    )
    records, panel_metadata = source_access._load_panel(spec, ledger)
    source_frames, scenes, _unused_geometry, source_entries = (
        source_access._read_source_geometry(
            records,
            panel_metadata,
            spec=spec,
            ledger=ledger,
            authorized_source_entries=authorized["source_geometry"]["entries"],
        )
    )
    if authorized["source_geometry"] != source_access._canonical_manifest(
        source_entries
    ):
        raise ValueError("source geometry inventory changed")

    sidecar_rows = attitude_sidecar.load_attitude_sidecar_roles(
        SIDECAR_MANIFEST_PATH,
        roles=("train",),
        expected_manifest_sha256=SIDECAR_MANIFEST_FILE_SHA256,
        contract=attitude_sidecar.FROZEN_BUILD_CONTRACT,
    )["train"]
    if len(sidecar_rows) != 4262:
        raise ValueError("train sidecar row count changed")
    sidecar_by_global = {int(row["global_row"]): row for row in sidecar_rows}
    if len(sidecar_by_global) != len(sidecar_rows):
        raise ValueError("train sidecar global rows are not injective")

    by_scene: dict[str, list[FrameBuildInputV4]] = {}
    family_counts: dict[str, int] = {}
    for record in records:
        global_row = int(record["global_row"])
        side = str(record["side"])
        sidecar = sidecar_by_global.get(global_row)
        if sidecar is None or sidecar.get("dataset_role") != "train":
            raise PermissionError("fit frame lacks a train sidecar row")
        endpoint = sidecar.get(side)
        if not isinstance(endpoint, Mapping):
            raise ValueError("train sidecar lacks the selected endpoint")
        expected_endpoint = {
            "scene_id_sha256": _scene_sha256(str(record["scene_id"])),
            "env_index": int(record["env_index"]),
            f"{side}_frame_index": int(record["frame_index"]),
            f"{side}_timestamp_ns": int(record["timestamp_ns"]),
        }
        for key, expected in expected_endpoint.items():
            if sidecar.get(key) != expected:
                raise ValueError(f"train sidecar join mismatch for {key}")
        identity = tuple(source_access._frame_identity_values(record))
        source_frame = source_frames.get(identity)
        if source_frame is None:
            raise ValueError("fit source frame is missing after geometry load")
        position = source_frame["base_pose_world"]["position"]
        base_position = tuple(float(position[axis]) for axis in ("x", "y", "z"))
        sidecar_quaternion, stored_yaw = _validated_sidecar_source_attitude(
            source_frame, endpoint
        )
        camera = dynamic_projection.compose_yaw_aligned_camera(
            sidecar_quaternion, stored_yaw
        )
        basis = _normalized_camera_basis_fru(camera)
        scene_id = str(record["scene_id"])
        rendered_boxes = tuple(
            _box_in_yaw_body(
                raw_box,
                base_position_world=base_position,
                stored_yaw_rad=stored_yaw,
            )
            for raw_box in scenes[scene_id]["rendered_boxes"]
        )
        frame_key = {
            "dataset_role": "train",
            **source_access._frame_key(record),
        }
        by_scene.setdefault(scene_id, []).append(
            FrameBuildInputV4(
                frame_key=frame_key,
                camera_origin_body_m=tuple(camera.origin_xyz),
                camera_basis_body_fru=basis,
                ground_plane_z_body_m=-base_position[2],
                rendered_boxes_body=rendered_boxes,
                image_path_metadata_only=str(record["image_path_metadata_only"]),
                image_sha256=str(record["image_sha256"]),
                sidecar_row_identity_sha256=str(
                    sidecar["row_identity_sha256"]
                ),
            )
        )
        family = str(record["family"])
        family_counts[family] = family_counts.get(family, 0) + 1

    if len(records) != 320 or set(family_counts.values()) != {64}:
        raise ValueError("exact fit scope is not 320 frames balanced by family")
    jobs = tuple(
        SceneBuildJobV4(scene_key=scene_id, frames=tuple(frames))
        for scene_id, frames in sorted(by_scene.items())
    )
    if len(jobs) != 20:
        raise ValueError("exact fit scope must contain 20 scene shards")
    forbidden = list(source_access.FORBIDDEN_ACCESS_FIELDS)
    if any(int(ledger[name]) != 0 for name in forbidden):
        raise PermissionError("source loader crossed a forbidden role boundary")
    _assert_zero_access_denials(ledger)
    if (
        int(ledger["label_shard_hash_byte_opens"]) != 0
        or int(ledger["label_shard_npz_opens"]) != 0
        or int(ledger["rgb_byte_opens"]) != 0
    ):
        raise PermissionError("builder opened a fit label or RGB payload")
    input_provenance = {
        "implementation_manifest_file_sha256": implementation_manifest_sha256,
        "implementation_manifest_content_sha256": implementation[
            "content_sha256"
        ],
        "source_authorization_manifest_file_sha256": machine_manifest_sha256,
        "source_authorization_manifest_content_sha256": machine_manifest[
            "content_sha256"
        ],
        "source_hashes": source_hashes,
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
    builder_ledger = {
        **ledger,
        "sidecar_manifest_byte_opens": 1,
        "sidecar_train_role_byte_opens": 1,
        "sidecar_checkpoint_selection_role_byte_opens": 0,
        "sidecar_probability_calibration_role_byte_opens": 0,
        "sidecar_g2_evaluation_role_byte_opens": 0,
        "fit_label_payload_byte_opens": 0,
    }
    _validate_exact_build_ledger_schema(builder_ledger)
    _validate_authorized_source_hashes(source_hashes)
    return jobs, input_provenance, builder_ledger


def synthetic_scene_jobs(count: int = 6) -> tuple[SceneBuildJobV4, ...]:
    if isinstance(count, bool) or not 1 <= int(count) <= 12:
        raise ValueError("synthetic scene count must lie in [1,12]")
    basis = (
        (0.0, 0.0, -1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    jobs: list[SceneBuildJobV4] = []
    for index in range(int(count)):
        scene_key = f"synthetic_train_scene_{index:02d}"
        frame_key = {
            "dataset_role": "train",
            "family": "synthetic",
            "scene_id_sha256": _scene_sha256(scene_key),
            "global_row": index,
            "side": "current" if index % 2 == 0 else "next",
            "frame_index": index,
            "env_index": 0,
            "timestamp_ns": 1_000_000 + index,
        }
        angle = 0.07 * index
        rotation = _rotation_xyz(0.0, 0.0, angle)
        box = RayBoxV4(
            center_body_m=(2.2 + 0.03 * index, 0.1, 4.0),
            half_size_m=(0.35, 0.45, 0.5),
            rotation_body_from_box=tuple(tuple(row) for row in rotation.tolist()),
        )
        image_hash = hashlib.sha256(f"image:{index}".encode()).hexdigest()
        sidecar_hash = hashlib.sha256(f"sidecar:{index}".encode()).hexdigest()
        jobs.append(
            SceneBuildJobV4(
                scene_key=scene_key,
                frames=(
                    FrameBuildInputV4(
                        frame_key=frame_key,
                        camera_origin_body_m=(2.2, 0.0, 10.0),
                        camera_basis_body_fru=basis,
                        ground_plane_z_body_m=0.0,
                        rendered_boxes_body=(box,),
                        image_path_metadata_only=str(
                            ROOT
                            / ".synthetic/go2_observable_camera_ray_fit_v4"
                            / scene_key
                            / "rgb"
                            / f"frame_{index:06d}.png"
                        ),
                        image_sha256=image_hash,
                        sidecar_row_identity_sha256=sidecar_hash,
                    ),
                ),
            )
        )
    return tuple(jobs)


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): _sha256_file(path)
        for path in sorted(directory.rglob("*"), key=str)
        if path.is_file()
    }


def run_dry_run() -> dict[str, Any]:
    jobs = synthetic_scene_jobs()
    provenance = {
        "schema": "lewm_go2_observable_camera_ray_fit_v4_synthetic_input_v1",
        "payload_role": "synthetic_train",
    }
    ledger = {
        "synthetic_only": True,
        "rgb_byte_opens": 0,
        "fit_label_payload_byte_opens": 0,
        "nontrain_role_byte_opens": 0,
        "g2_byte_opens": 0,
        "model_or_checkpoint_byte_opens": 0,
    }
    with tempfile.TemporaryDirectory(prefix="go2_ray_v4_dry_run.") as temporary:
        root = Path(temporary)
        first_path = root / "one"
        second_path = root / "six"
        first = build_dataset_from_jobs(
            jobs,
            output_directory=first_path,
            workers=1,
            input_provenance=provenance,
            access_ledger=ledger,
        )
        second = build_dataset_from_jobs(
            jobs,
            output_directory=second_path,
            workers=MAX_WORKERS,
            input_provenance=provenance,
            access_ledger=ledger,
        )
        first_hashes = _artifact_hashes(first_path)
        second_hashes = _artifact_hashes(second_path)
        deterministic = bool(
            first == second
            and first["content_sha256"] == second["content_sha256"]
            and first_hashes == second_hashes
        )
        return {
            "schema": "lewm_go2_observable_camera_ray_fit_v4_dry_run_v1",
            "synthetic_only": True,
            "frame_count": first["frame_count"],
            "scene_shard_count": first["scene_shard_count"],
            "one_vs_six_worker_byte_identical": deterministic,
            "dataset_content_sha256": first["content_sha256"],
            "artifact_map_sha256": canonical_json_sha256(first_hashes),
            "gpu_used": False,
            "fit_payload_opened": False,
            "nontrain_payload_opened": False,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-exact-fit", action="store_true")
    parser.add_argument("--machine-manifest-sha256")
    parser.add_argument("--implementation-manifest-sha256")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args(argv)
    if args.dry_run:
        if (
            args.machine_manifest_sha256 is not None
            or args.implementation_manifest_sha256 is not None
        ):
            parser.error("dry-run forbids exact-fit authorization hashes")
    elif (
        args.machine_manifest_sha256 is None
        or args.implementation_manifest_sha256 is None
    ):
        parser.error(
            "exact fit requires both source and reviewed implementation manifest hashes"
        )
    if not 1 <= int(args.workers) <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1,{MAX_WORKERS}]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        result = run_dry_run()
        if not result["one_vs_six_worker_byte_identical"]:
            raise RuntimeError("one-worker and six-worker artifacts differ")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0

    implementation = _load_reviewed_implementation_manifest(
        str(args.implementation_manifest_sha256),
        required_authorization="build",
    )
    jobs, provenance, ledger = load_exact_fit_jobs(
        machine_manifest_sha256=str(args.machine_manifest_sha256),
        implementation_manifest_sha256=str(args.implementation_manifest_sha256),
    )
    result = build_dataset_from_jobs(
        jobs,
        output_directory=CANONICAL_OUTPUT,
        workers=int(args.workers),
        input_provenance=provenance,
        access_ledger=ledger,
        source_closure_entries=implementation["source_map"]["entries"],
        required_output_root=ROOT,
    )
    print(
        json.dumps(
            {
                "output": str((CANONICAL_OUTPUT / "manifest.json").resolve()),
                "content_sha256": result["content_sha256"],
                "frame_count": result["frame_count"],
                "scene_shard_count": result["scene_shard_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
