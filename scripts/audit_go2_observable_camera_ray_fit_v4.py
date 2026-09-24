#!/usr/bin/env python3
"""Audit an immutable observable camera-ray V4 fit dataset.

Synthetic mode opens no generated corpus data. Exact mode verifies all 320 V4
frames, then opens only the 20 registered current-physical-train fit label
shards to report the unchanged 64x64 N32 comparison. Legacy-target mismatch is
reported, never repaired with privileged geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
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

from scripts import build_go2_observable_camera_ray_fit_v4 as builder  # noqa: E402


EVIDENCE_SCHEMA = builder.EVIDENCE_SCHEMA
ObservableCameraRayEvidenceV4 = builder.ObservableCameraRayEvidenceV4
rasterize_observable_camera_ray_evidence_v4 = (
    builder.rasterize_observable_camera_ray_evidence_v4
)


AUDIT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_audit_v1"
FRAME_AUDIT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_frame_audit_v1"
CANONICAL_OUTPUT = builder.CANONICAL_OUTPUT / "audit_result.json"
CLASS_NAMES = ("unknown", "free", "occupied")
DISTANCE_BINS = (
    ("0.0_to_0.5", 0.0, 0.5),
    ("0.5_to_1.0", 0.5, 1.0),
    ("1.0_to_2.0", 1.0, 2.0),
    ("2.0_to_3.0", 2.0, 3.0),
    ("3.0_plus", 3.0, None),
)
EXPECTED_LABEL_SHARD_MANIFEST_SHA256 = (
    "998ce5a768029c23c931fbbec730c1fe31b9ed1fe155494fc68f34a0c23d3d1b"
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


def _strict_json(path: Path, expected_file_sha256: str, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise PermissionError(f"{name} must be a regular file")
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_file_sha256:
        raise ValueError(f"{name} file SHA-256 changed")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or declared != canonical_json_sha256(core):
        raise ValueError(f"{name} content SHA-256 changed")
    return value


def _resolve_inside(parent: Path, relative: str, *, name: str) -> Path:
    candidate = parent / relative
    if candidate.is_symlink():
        raise PermissionError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(parent.resolve(strict=True))
    except ValueError as exc:
        raise PermissionError(f"{name} escapes the dataset root") from exc
    return resolved


def _validate_dataset_inventory(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    dataset_root = manifest_path.parent
    if manifest_path.name != "manifest.json":
        raise ValueError("V4 dataset manifest must use its canonical filename")
    observed_root = {entry.name for entry in dataset_root.iterdir()}
    if observed_root != {"manifest.json", "shards"}:
        raise ValueError("V4 dataset root inventory changed")
    shards_root = dataset_root / "shards"
    if shards_root.is_symlink() or not stat.S_ISDIR(
        shards_root.stat(follow_symlinks=False).st_mode
    ):
        raise PermissionError("V4 shards root must be a regular directory")

    shard_records = manifest.get("shards")
    if not isinstance(shard_records, list):
        raise ValueError("V4 dataset shard records are malformed")
    expected_directories: set[str] = set()
    for record in shard_records:
        if not isinstance(record, Mapping):
            raise ValueError("V4 shard record is malformed")
        relative = Path(str(record.get("path", "")))
        if (
            relative.is_absolute()
            or len(relative.parts) != 3
            or relative.parts[0] != "shards"
            or relative.parts[1] in {"", ".", ".."}
            or relative.parts[2] != "shard.json"
        ):
            raise PermissionError("V4 shard path is not canonical")
        if relative.parts[1] in expected_directories:
            raise ValueError("V4 dataset repeats a shard directory")
        expected_directories.add(relative.parts[1])
    observed_directories = {entry.name for entry in shards_root.iterdir()}
    if observed_directories != expected_directories:
        raise ValueError("V4 shard-directory inventory changed")
    for name in sorted(observed_directories):
        path = shards_root / name
        if path.is_symlink() or not stat.S_ISDIR(
            path.stat(follow_symlinks=False).st_mode
        ):
            raise PermissionError("V4 shard entry must be a regular directory")


def _validate_shard_file_inventory(
    directory: Path,
    expected_names: set[str],
) -> None:
    observed = {entry.name for entry in directory.iterdir()}
    if observed != expected_names:
        raise ValueError("V4 shard filesystem inventory changed")
    for name in sorted(observed):
        path = directory / name
        if path.is_symlink() or not stat.S_ISREG(
            path.stat(follow_symlinks=False).st_mode
        ):
            raise PermissionError("V4 shard files must be regular files")


def _parse_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{name} must be nonempty newline-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} row {index} is invalid JSON") from exc
        if not isinstance(row, dict) or _canonical_json_bytes(row) != line:
            raise ValueError(f"{name} row {index} is not canonical")
        core = dict(row)
        declared = core.pop("content_sha256", None)
        if declared != canonical_json_sha256(core):
            raise ValueError(f"{name} row {index} content hash changed")
        rows.append(row)
    return rows


def _read_array(
    directory: Path,
    file_record: Mapping[str, Any],
    *,
    frame_count: int,
) -> np.ndarray:
    relative = str(file_record.get("path", ""))
    path = _resolve_inside(directory, relative, name=f"shard array {relative}")
    payload = path.read_bytes()
    if (
        len(payload) != int(file_record.get("byte_count", -1))
        or _sha256_bytes(payload) != file_record.get("file_sha256")
    ):
        raise ValueError(f"shard array {relative} bytes changed")
    dtype = np.dtype(str(file_record.get("dtype", "")))
    shape = tuple(file_record.get("shape", ()))
    if not shape or int(shape[0]) != frame_count:
        raise ValueError(f"shard array {relative} frame shape changed")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected_bytes != len(payload):
        raise ValueError(f"shard array {relative} byte count disagrees with shape")
    result = np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    result.setflags(write=False)
    return result


def load_and_verify_dataset(
    manifest_path: Path,
    *,
    expected_manifest_file_sha256: str,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], ObservableCameraRayEvidenceV4]]]:
    if manifest_path.is_symlink() or not stat.S_ISREG(
        manifest_path.stat(follow_symlinks=False).st_mode
    ):
        raise PermissionError("V4 dataset manifest must be a regular file")
    manifest_path = manifest_path.resolve(strict=True)
    manifest = _strict_json(
        manifest_path,
        expected_manifest_file_sha256,
        name="V4 dataset manifest",
    )
    if (
        manifest.get("schema") != builder.DATASET_SCHEMA
        or manifest.get("evidence_schema") != EVIDENCE_SCHEMA
        or manifest.get("dataset_role") != "train"
    ):
        raise ValueError("V4 dataset manifest identity changed")
    _validate_dataset_inventory(manifest_path, manifest)
    rgb_by_key = builder._validate_rgb_receipt(manifest.get("rgb_receipt"))
    dataset_root = manifest_path.parent
    frames: list[tuple[dict[str, Any], ObservableCameraRayEvidenceV4]] = []
    seen_keys: set[bytes] = set()
    for shard_record in manifest.get("shards", ()):
        if not isinstance(shard_record, Mapping):
            raise ValueError("V4 shard record is malformed")
        shard_path = _resolve_inside(
            dataset_root, str(shard_record.get("path", "")), name="V4 shard manifest"
        )
        shard = _strict_json(
            shard_path,
            str(shard_record.get("file_sha256", "")),
            name="V4 shard manifest",
        )
        if (
            shard.get("schema") != builder.SHARD_SCHEMA
            or shard.get("content_sha256") != shard_record.get("content_sha256")
        ):
            raise ValueError("V4 shard identity changed")
        frame_count = int(shard.get("frame_count", -1))
        declared_files = shard.get("files")
        if not isinstance(declared_files, list):
            raise ValueError("V4 shard declared file records are malformed")
        file_records: dict[str, Mapping[str, Any]] = {}
        for item in declared_files:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("path"), str)
                or not item["path"]
            ):
                raise ValueError("V4 shard declared file record is malformed")
            name = str(item["path"])
            if name in file_records:
                raise ValueError("V4 shard repeats a declared file record")
            file_records[name] = item
        expected_names = {name for name, _dtype, _shape in builder.ARRAY_LAYOUT}
        if set(file_records) != expected_names | {"index.jsonl"}:
            raise ValueError("V4 shard file inventory changed")
        _validate_shard_file_inventory(
            shard_path.parent,
            expected_names | {"index.jsonl", "shard.json"},
        )
        arrays = {
            name: _read_array(
                shard_path.parent, file_records[name], frame_count=frame_count
            )
            for name in sorted(expected_names)
        }
        index_record = file_records["index.jsonl"]
        index_path = _resolve_inside(
            shard_path.parent, "index.jsonl", name="V4 shard index"
        )
        index_payload = index_path.read_bytes()
        if (
            len(index_payload) != int(index_record.get("byte_count", -1))
            or _sha256_bytes(index_payload) != index_record.get("file_sha256")
        ):
            raise ValueError("V4 shard index bytes changed")
        index_rows = _parse_jsonl(index_payload, name="V4 shard index")
        if len(index_rows) != frame_count:
            raise ValueError("V4 shard index count changed")
        if canonical_json_sha256([row["frame_key"] for row in index_rows]) != shard.get(
            "ordered_frame_keys_sha256"
        ):
            raise ValueError("V4 shard ordered frame keys changed")

        for row_index, index_row in enumerate(index_rows):
            frame_key = index_row.get("frame_key")
            if not isinstance(frame_key, dict) or frame_key.get("dataset_role") != "train":
                raise PermissionError("V4 frame is not current physical train role")
            encoded_key = _canonical_json_bytes(frame_key)
            if encoded_key in seen_keys:
                raise ValueError("V4 dataset repeats a frame key")
            seen_keys.add(encoded_key)
            rgb_entry = rgb_by_key.get(encoded_key)
            if (
                rgb_entry is None
                or rgb_entry["rgb_file_sha256"]
                != index_row.get("image_sha256_commitment_only")
            ):
                raise ValueError("V4 RGB receipt does not join to its shard index")
            evidence = ObservableCameraRayEvidenceV4(
                camera_origin_body_m=arrays["camera_origin_body_f32.bin"][row_index],
                camera_basis_body_fru=arrays[
                    "camera_basis_body_fru_f32.bin"
                ][row_index],
                ground_plane_z_body_m=float(
                    arrays["ground_plane_z_body_f32.bin"][row_index, 0]
                ),
                ground_support_in_frustum=arrays[
                    "ground_support_in_frustum_u8.bin"
                ][row_index].astype(bool),
                ground_support_clear_to_target=arrays[
                    "ground_support_clear_to_target_u8.bin"
                ][row_index].astype(bool),
                pixel_hit_mask=arrays["pixel_hit_mask_u8.bin"][row_index].astype(
                    bool
                ),
                pixel_first_hit_distance_m=arrays[
                    "pixel_first_hit_distance_f32.bin"
                ][row_index],
            )
            raster = rasterize_observable_camera_ray_evidence_v4(evidence)
            if (
                evidence.content_sha256()
                != index_row.get("evidence_content_sha256")
                or raster.content_sha256()
                != index_row.get("raster_content_sha256")
            ):
                raise ValueError("V4 frame evidence/raster content hash changed")
            frames.append((index_row, evidence))
        _validate_shard_file_inventory(
            shard_path.parent,
            expected_names | {"index.jsonl", "shard.json"},
        )
        if _strict_json(
            shard_path,
            str(shard_record.get("file_sha256", "")),
            name="V4 shard manifest",
        ) != shard:
            raise ValueError("V4 shard manifest changed during audit")
    _validate_dataset_inventory(manifest_path, manifest)
    if _sha256_file(manifest_path) != expected_manifest_file_sha256:
        raise ValueError("V4 dataset manifest changed during audit")
    if (
        len(frames) != int(manifest.get("frame_count", -1))
        or len(manifest.get("shards", ()))
        != int(manifest.get("scene_shard_count", -1))
        or seen_keys != set(rgb_by_key)
    ):
        raise ValueError("V4 dataset aggregate counts changed")
    return manifest, frames


def _confusion(reference: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    return [
        [
            int(np.count_nonzero((reference == expected) & (predicted == actual)))
            for actual in range(3)
        ]
        for expected in range(3)
    ]


def _sum_confusions(values: Sequence[Sequence[Sequence[int]]]) -> list[list[int]]:
    total = np.zeros((3, 3), dtype=np.int64)
    for value in values:
        array = np.asarray(value, dtype=np.int64)
        if array.shape != (3, 3) or np.any(array < 0):
            raise ValueError("one frame confusion is malformed")
        total += array
    return total.tolist()


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(labels == class_index))
        for class_index, name in enumerate(CLASS_NAMES)
    }


def _class_metrics(confusion: Sequence[Sequence[int]]) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    result: dict[str, Any] = {}
    for class_index, name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[class_index, class_index])
        reference_count = int(matrix[class_index].sum())
        predicted_count = int(matrix[:, class_index].sum())
        result[name] = {
            "reference_count": reference_count,
            "predicted_count": predicted_count,
            "recall": (
                None if reference_count == 0 else true_positive / reference_count
            ),
            "precision": (
                None if predicted_count == 0 else true_positive / predicted_count
            ),
        }
    return result


def _distance_masks() -> dict[str, np.ndarray]:
    forward = -1.0 + (np.arange(64, dtype=np.float64) + 0.5) * 0.10
    left = -3.2 + (np.arange(64, dtype=np.float64) + 0.5) * 0.10
    forward_grid, left_grid = np.meshgrid(forward, left, indexing="ij")
    distance = np.hypot(forward_grid, left_grid)
    return {
        name: (distance >= lower)
        & (True if upper is None else distance < upper)
        for name, lower, upper in DISTANCE_BINS
    }


def audit_dataset_frames(
    frames: Sequence[tuple[Mapping[str, Any], ObservableCameraRayEvidenceV4]],
    *,
    reference_labels: Mapping[bytes, tuple[np.ndarray, np.ndarray]] | None,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    family_confusions: dict[str, list[list[list[int]]]] = {}
    distance_confusions: dict[str, list[list[list[int]]]] = {
        name: [] for name, _lower, _upper in DISTANCE_BINS
    }
    distance_masks = _distance_masks()
    for index_row, evidence in frames:
        frame_key = dict(index_row["frame_key"])
        raster = rasterize_observable_camera_ray_evidence_v4(evidence)
        predicted = raster.output_labels
        encoded_key = _canonical_json_bytes(frame_key)
        confusion = None
        mismatch_count = None
        reference_hash = None
        supervised_count = None
        if reference_labels is not None:
            pair = reference_labels.get(encoded_key)
            if pair is None:
                raise ValueError("one V4 frame lacks an exact reference label")
            target, supervision = pair
            target = np.asarray(target, dtype=np.uint8)
            supervision = np.asarray(supervision, dtype=bool)
            if target.shape != (64, 64) or supervision.shape != (64, 64):
                raise ValueError("one reference label has the wrong shape")
            if not supervision.all():
                raise ValueError("minimal N32 audit requires fully supervised frames")
            confusion = _confusion(target, predicted)
            mismatch_count = int(np.count_nonzero(target != predicted))
            reference_hash = hashlib.sha256(target.tobytes(order="C")).hexdigest()
            supervised_count = int(np.count_nonzero(supervision))
            family = str(frame_key.get("family", ""))
            family_confusions.setdefault(family, []).append(confusion)
            for name, mask in distance_masks.items():
                distance_confusions[name].append(_confusion(target[mask], predicted[mask]))
        reports.append(
            {
                "schema": FRAME_AUDIT_SCHEMA,
                "frame_key": frame_key,
                "evidence_content_sha256": evidence.content_sha256(),
                "raster_content_sha256": raster.content_sha256(),
                "predicted_labels_sha256": hashlib.sha256(
                    predicted.tobytes(order="C")
                ).hexdigest(),
                "predicted_class_counts": _class_counts(predicted),
                "reference_labels_sha256": reference_hash,
                "supervised_cell_count": supervised_count,
                "mismatch_cell_count": mismatch_count,
                "confusion_reference_rows": confusion,
            }
        )
    result: dict[str, Any] = {
        "frame_count": len(reports),
        "cell_count": len(reports) * 64 * 64,
        "ordered_frame_keys_sha256": canonical_json_sha256(
            [report["frame_key"] for report in reports]
        ),
        "ordered_evidence_sha256": canonical_json_sha256(
            [report["evidence_content_sha256"] for report in reports]
        ),
        "frame_reports": reports,
        "internal_hash_and_raster_determinism_passes": True,
    }
    if reference_labels is not None:
        confusions = [report["confusion_reference_rows"] for report in reports]
        total_confusion = _sum_confusions(confusions)  # type: ignore[arg-type]
        mismatches = sum(int(report["mismatch_cell_count"]) for report in reports)
        result["legacy_physical_v3_comparison"] = {
            "exact": mismatches == 0,
            "mismatch_cell_count": mismatches,
            "mismatch_frame_count": sum(
                int(report["mismatch_cell_count"]) > 0 for report in reports
            ),
            "confusion_reference_rows": total_confusion,
            "class_metrics": _class_metrics(total_confusion),
            "families": {
                family: {
                    "frame_count": len(values),
                    "confusion_reference_rows": _sum_confusions(values),
                    "class_metrics": _class_metrics(_sum_confusions(values)),
                }
                for family, values in sorted(family_confusions.items())
            },
            "distance_bins": {
                name: {
                    "cell_count": int(np.count_nonzero(distance_masks[name]))
                    * len(reports),
                    "confusion_reference_rows": _sum_confusions(values),
                    "class_metrics": _class_metrics(_sum_confusions(values)),
                }
                for name, values in distance_confusions.items()
            },
            "interpretation": (
                "diagnostic target-definition comparison; mismatch may not be "
                "repaired with physical-free or collision-geometry priors"
            ),
        }
    return result


def _validate_exact_dataset_receipt(
    manifest: Mapping[str, Any],
    *,
    implementation_manifest_file_sha256: str,
    implementation_manifest_content_sha256: str,
    source_authorization_manifest_file_sha256: str,
) -> None:
    frame_count = manifest.get("frame_count")
    shard_count = manifest.get("scene_shard_count")
    if (
        manifest.get("schema") != builder.DATASET_SCHEMA
        or manifest.get("evidence_schema") != builder.EVIDENCE_SCHEMA
        or manifest.get("dataset_role") != "train"
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count != 320
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count != 20
    ):
        raise ValueError("exact dataset receipt scope changed")
    rgb_receipt = builder._validate_rgb_receipt(manifest.get("rgb_receipt"))
    if len(rgb_receipt) != 320:
        raise ValueError("exact dataset RGB receipt scope changed")
    provenance = manifest.get("input_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("exact dataset receipt lacks input provenance")
    expected_provenance = {
        "implementation_manifest_file_sha256": implementation_manifest_file_sha256,
        "implementation_manifest_content_sha256": (
            implementation_manifest_content_sha256
        ),
        "source_authorization_manifest_file_sha256": (
            source_authorization_manifest_file_sha256
        ),
        "source_authorization_manifest_content_sha256": (
            builder.SOURCE_AUTHORIZATION_MANIFEST_CONTENT_SHA256
        ),
        "fit_panel_file_sha256": builder.PANEL_FILE_SHA256,
        "fit_panel_content_sha256": builder.PANEL_CONTENT_SHA256,
        "fit_rows_sha256": builder.FIT_ROWS_SHA256,
        "source_geometry_manifest_sha256": builder.SOURCE_GEOMETRY_MANIFEST_SHA256,
        "render_summaries_manifest_sha256": builder.RENDER_SUMMARIES_MANIFEST_SHA256,
        "sidecar_manifest_file_sha256": builder.SIDECAR_MANIFEST_FILE_SHA256,
        "sidecar_manifest_content_sha256": builder.SIDECAR_MANIFEST_CONTENT_SHA256,
        "sidecar_train_file_sha256": builder.SIDECAR_TRAIN_FILE_SHA256,
        "sidecar_train_content_sha256": builder.SIDECAR_TRAIN_CONTENT_SHA256,
        "sidecar_train_ordered_global_sha256": (
            builder.SIDECAR_TRAIN_ORDERED_GLOBAL_SHA256
        ),
        "sidecar_train_ordered_identity_sha256": (
            builder.SIDECAR_TRAIN_ORDERED_IDENTITY_SHA256
        ),
    }
    if set(provenance) != set(expected_provenance) | {"source_hashes"}:
        raise ValueError("exact dataset provenance fields changed")
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("exact dataset provenance commitment changed")
    source_hashes = provenance.get("source_hashes")
    binding_source = (
        source_hashes.get("binding") if isinstance(source_hashes, Mapping) else None
    )
    if (
        not isinstance(source_hashes, Mapping)
        or not isinstance(binding_source, Mapping)
        or binding_source.get("sha256")
        != builder.SOURCE_AUTHORIZATION_BINDING_SHA256
    ):
        raise ValueError("exact dataset source authorization receipt changed")
    ledger = manifest.get("access_ledger")
    builder._validate_exact_build_ledger_schema(ledger)
    assert isinstance(ledger, Mapping)
    for field in builder.EXACT_FORBIDDEN_BUILD_LEDGER_FIELDS:
        if int(ledger.get(field, -1)) != 0:
            raise PermissionError(f"exact dataset build ledger is nonzero: {field}")
    builder._assert_zero_access_denials(ledger)
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
        raise ValueError("exact dataset positive access receipt changed")
    expected_parallel = {
        "worker_start_method": "spawn",
        "maximum_workers": builder.MAX_WORKERS,
        "native_threads_per_worker": 1,
        "canonical_merge": "scene_hash_then_canonical_frame_key",
        "worker_count_does_not_change_artifact_bytes": True,
        "per_worker_source_revalidation": True,
        "parent_source_revalidation_before_manifest": True,
    }
    if manifest.get("parallel_contract") != expected_parallel:
        raise ValueError("exact dataset parallel/source-validation contract changed")
    if manifest.get("publication") != (
        "private_staging_hardlink_no_replace_manifest_last"
    ):
        raise ValueError("exact dataset publication contract changed")
    expected_layout = [
        {
            "path": name,
            "dtype": np.dtype(dtype).str,
            "trailing_shape": list(shape),
        }
        for name, dtype, shape in builder.ARRAY_LAYOUT
    ]
    if manifest.get("array_layout") != expected_layout:
        raise ValueError("exact dataset array layout changed")
    licenses = manifest.get("licenses")
    if not isinstance(licenses, Mapping) or set(licenses) != {
        "model_output_authorized",
        "holdout_authorized",
        "g2_authorized",
        "runtime_authorized",
        "promotion_authorized",
    } or any(value is not False for value in licenses.values()):
        raise PermissionError("exact dataset contains an unauthorized license")


def load_exact_fit_reference_labels(
    *,
    machine_manifest_sha256: str,
    implementation_manifest_sha256: str,
    dataset_manifest: Mapping[str, Any],
) -> tuple[dict[bytes, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    implementation = builder._load_reviewed_implementation_manifest(
        implementation_manifest_sha256,
        required_authorization="audit",
    )
    _validate_exact_dataset_receipt(
        dataset_manifest,
        implementation_manifest_file_sha256=implementation_manifest_sha256,
        implementation_manifest_content_sha256=implementation["content_sha256"],
        source_authorization_manifest_file_sha256=machine_manifest_sha256,
    )
    if machine_manifest_sha256 != builder.SOURCE_AUTHORIZATION_MANIFEST_FILE_SHA256:
        raise PermissionError("source authorization manifest hash is not frozen")
    source_access = builder._load_neutral_module(
        "go2_n32_camera_frustum_source_access_v4_audit_neutral",
        builder.SOURCE_ACCESS_PATH,
    )
    if (
        tuple(source_access.PRIMARY_DENIAL_REASONS)
        != builder.EXACT_DENIAL_PRIMARY_REASONS
        or tuple(source_access.DENIAL_MODALITIES)
        != builder.EXACT_DENIAL_MODALITIES
    ):
        raise ValueError("source access denial schema changed")

    ledger = source_access.new_access_ledger()
    if set(ledger) != builder.SOURCE_ACCESS_LEDGER_FIELDS:
        raise ValueError("source access-ledger fields changed")
    spec = source_access.AuditSpec()
    source_hashes = source_access._source_hashes(spec.sources(), ledger=ledger)
    if dataset_manifest["input_provenance"]["source_hashes"] != source_hashes:
        raise ValueError("dataset build semantic source receipt changed")
    machine_manifest = source_access._load_machine_manifest(
        machine_manifest_sha256,
        source_hashes=source_hashes,
        ledger=ledger,
    )
    source_access._load_authorized_semantics(source_hashes)
    records, _panel_metadata = source_access._load_panel(spec, ledger)
    rgb_receipt = builder._validate_rgb_receipt(dataset_manifest.get("rgb_receipt"))
    expected_rgb_keys: set[bytes] = set()
    for record in records:
        frame_key = {
            "dataset_role": "train",
            **source_access._frame_key(record),
        }
        encoded_key = _canonical_json_bytes(frame_key)
        expected_rgb_keys.add(encoded_key)
        receipt_entry = rgb_receipt.get(encoded_key)
        if (
            receipt_entry is None
            or receipt_entry["canonical_rgb_path"]
            != str(record["image_path_metadata_only"])
            or receipt_entry["rgb_file_sha256"] != str(record["image_sha256"])
        ):
            raise ValueError("dataset RGB receipt disagrees with frozen fit metadata")
    if expected_rgb_keys != set(rgb_receipt):
        raise ValueError("dataset RGB receipt frame scope changed")
    shard_entries, grouped = source_access._label_shard_manifest(
        records, spec=spec, ledger=ledger
    )
    if machine_manifest["authorized_inputs"]["label_shards"] != source_access._canonical_manifest(
        shard_entries
    ):
        raise ValueError("fit label-shard inventory changed")
    if (
        machine_manifest["authorized_inputs"]["label_shards"]["manifest_sha256"]
        != EXPECTED_LABEL_SHARD_MANIFEST_SHA256
    ):
        raise ValueError("fit label-shard manifest hash changed")
    selected = source_access._read_selected_labels_once(grouped, ledger=ledger)
    references: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        identity = tuple(source_access._frame_identity_values(record))
        pair = selected.get(identity)
        if pair is None:
            raise ValueError("one selected fit label is missing")
        frame_key = {
            "dataset_role": "train",
            **source_access._frame_key(record),
        }
        references[_canonical_json_bytes(frame_key)] = pair
    if len(references) != 320:
        raise ValueError("exact fit reference scope must contain 320 frames")
    forbidden_except_labels = tuple(source_access.FORBIDDEN_ACCESS_FIELDS)
    if any(int(ledger[name]) != 0 for name in forbidden_except_labels):
        raise PermissionError("label reader crossed a forbidden role boundary")
    builder._assert_zero_access_denials(ledger)
    if (
        int(ledger["rgb_byte_opens"]) != 0
        or int(ledger["source_geometry_hash_byte_opens"]) != 0
        or int(ledger["physical_nontrain_role_opens"]) != 0
        or int(ledger["g2_opens"]) != 0
    ):
        raise PermissionError("label reader opened geometry, RGB, or a non-fit role")
    if (
        int(ledger.get("label_shard_hash_byte_opens", -1)) != 20
        or int(ledger.get("label_shard_npz_opens", -1)) != 20
        or int(ledger.get("selected_label_rows_read", -1)) != 320
        or int(ledger.get("selected_supervision_rows_read", -1)) != 320
    ):
        raise ValueError("exact fit label access receipt changed")
    builder._validate_authorized_source_hashes(source_hashes)
    return references, ledger


def build_audit_result(
    *,
    dataset_manifest: Mapping[str, Any],
    frames: Sequence[tuple[Mapping[str, Any], ObservableCameraRayEvidenceV4]],
    reference_labels: Mapping[bytes, tuple[np.ndarray, np.ndarray]] | None,
    access_ledger: Mapping[str, Any],
    exact_fit: bool = False,
    implementation_manifest_file_sha256: str | None = None,
    source_authorization_manifest_file_sha256: str | None = None,
) -> dict[str, Any]:
    audit = audit_dataset_frames(frames, reference_labels=reference_labels)
    if exact_fit and reference_labels is None:
        raise ValueError("exact fit audit requires reference labels")
    if exact_fit:
        families = Counter(
            str(index_row["frame_key"].get("family", ""))
            for index_row, _evidence in frames
        )
        if len(frames) != 320 or set(families.values()) != {64}:
            raise ValueError("exact V4 audit is not the balanced 320-frame N32 panel")
    core = {
        "schema": AUDIT_SCHEMA,
        "dataset": {
            "schema": dataset_manifest["schema"],
            "file_sha256": _sha256_bytes(
                _canonical_json_bytes(dataset_manifest) + b"\n"
            ),
            "content_sha256": dataset_manifest["content_sha256"],
            "dataset_role": dataset_manifest["dataset_role"],
            "frame_count": dataset_manifest["frame_count"],
            "scene_shard_count": dataset_manifest["scene_shard_count"],
        },
        "input_authorization": {
            "implementation_manifest_file_sha256": (
                implementation_manifest_file_sha256
            ),
            "source_authorization_manifest_file_sha256": (
                source_authorization_manifest_file_sha256
            ),
        },
        "scope": {
            "exact_fit": exact_fit,
            "dataset_role": "train",
            "rgb_opened": False,
            "rgb_receipt_metadata_join_verified": True,
            "source_geometry_opened_by_auditor": False,
            "gpu_used": False,
        },
        "audit": audit,
        "access_ledger": json.loads(_canonical_json_bytes(access_ledger)),
        "licenses": {
            "model_output_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.resolve()
    if destination != CANONICAL_OUTPUT.resolve():
        raise PermissionError("V4 audit output path is frozen")
    try:
        destination.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise PermissionError("V4 audit output escapes the repository") from exc
    path = destination
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(_canonical_json_bytes(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def run_dry_run() -> dict[str, Any]:
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
        "source_geometry_byte_opens": 0,
    }
    with tempfile.TemporaryDirectory(prefix="go2_ray_v4_audit.") as temporary:
        directory = Path(temporary) / "dataset"
        built = builder.build_dataset_from_jobs(
            builder.synthetic_scene_jobs(),
            output_directory=directory,
            workers=builder.MAX_WORKERS,
            input_provenance=provenance,
            access_ledger=ledger,
        )
        manifest_path = directory / "manifest.json"
        manifest, frames = load_and_verify_dataset(
            manifest_path,
            expected_manifest_file_sha256=_sha256_file(manifest_path),
        )
        references = {
            _canonical_json_bytes(index_row["frame_key"]): (
                rasterize_observable_camera_ray_evidence_v4(evidence).output_labels,
                np.ones((64, 64), dtype=bool),
            )
            for index_row, evidence in frames
        }
        result = build_audit_result(
            dataset_manifest=manifest,
            frames=frames,
            reference_labels=references,
            access_ledger=ledger,
            exact_fit=False,
        )
        comparison = result["audit"]["legacy_physical_v3_comparison"]
        return {
            "schema": "lewm_go2_observable_camera_ray_fit_v4_audit_dry_run_v1",
            "synthetic_only": True,
            "dataset_content_sha256": built["content_sha256"],
            "audit_content_sha256": result["content_sha256"],
            "frame_count": result["audit"]["frame_count"],
            "internal_verification_passes": result["audit"][
                "internal_hash_and_raster_determinism_passes"
            ],
            "synthetic_reference_exact": comparison["exact"],
            "fit_payload_opened": False,
            "gpu_used": False,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-exact-fit", action="store_true")
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--machine-manifest-sha256")
    parser.add_argument("--implementation-manifest-sha256")
    args = parser.parse_args(argv)
    exact_values = (
        args.dataset_manifest_sha256,
        args.machine_manifest_sha256,
        args.implementation_manifest_sha256,
    )
    if args.dry_run and any(value is not None for value in exact_values):
        parser.error("dry-run forbids exact-fit authorization hashes")
    if args.run_exact_fit and any(value is None for value in exact_values):
        parser.error("exact fit requires dataset, source, and implementation hashes")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        result = run_dry_run()
        if not result["internal_verification_passes"] or not result[
            "synthetic_reference_exact"
        ]:
            raise RuntimeError("synthetic V4 audit failed")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0

    implementation = builder._load_reviewed_implementation_manifest(
        str(args.implementation_manifest_sha256),
        required_authorization="audit",
    )
    manifest_path = builder.CANONICAL_OUTPUT / "manifest.json"
    manifest, frames = load_and_verify_dataset(
        manifest_path,
        expected_manifest_file_sha256=str(args.dataset_manifest_sha256),
    )
    references, ledger = load_exact_fit_reference_labels(
        machine_manifest_sha256=str(args.machine_manifest_sha256),
        implementation_manifest_sha256=str(args.implementation_manifest_sha256),
        dataset_manifest=manifest,
    )
    result = build_audit_result(
        dataset_manifest=manifest,
        frames=frames,
        reference_labels=references,
        access_ledger=ledger,
        exact_fit=True,
        implementation_manifest_file_sha256=str(
            args.implementation_manifest_sha256
        ),
        source_authorization_manifest_file_sha256=str(
            args.machine_manifest_sha256
        ),
    )
    _write_json_exclusive(CANONICAL_OUTPUT, result)
    comparison = result["audit"]["legacy_physical_v3_comparison"]
    print(
        json.dumps(
            {
                "output": str(CANONICAL_OUTPUT.resolve()),
                "content_sha256": result["content_sha256"],
                "frame_count": result["audit"]["frame_count"],
                "legacy_target_exact": comparison["exact"],
                "legacy_target_mismatch_cell_count": comparison[
                    "mismatch_cell_count"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
