#!/usr/bin/env python3
"""Reproduce the V4 fit-ladder target partitions without RGB or model output."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (  # noqa: E402
    ObservableCameraRayEvidenceV4,
    project_canonical_ground_support_v4,
    rasterize_observable_camera_ray_evidence_v4,
)


DATASET_PATH = ROOT / ".generated/go2_observable_camera_ray_fit_v4/v1/manifest.json"
AUDIT_PATH = ROOT / ".generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json"
FREEZE_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json"
)
DATASET_FILE_SHA256 = "2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85"
DATASET_CONTENT_SHA256 = "9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812"
AUDIT_FILE_SHA256 = "2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c"
FREEZE_FILE_SHA256 = "4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a"
FREEZE_CONTENT_SHA256 = "8dd54d178e3c00a8622d89e4e371a115e1391f34588f667c20cd95b970fc68d2"
FIT_SIZES = (5, 16, 32, 320)
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
DISTANCE_EDGES_M = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, float("inf"))
DISTANCE_GROUPS = (
    "0.0_to_1.0",
    "1.0_to_2.0",
    "2.0_to_3.0",
    "3.0_to_4.0",
    "4.0_to_5.0",
    "5.0_plus",
)
SUBSET_NAMESPACE = b"lewm_go2_observable_camera_ray_fit_v4_subset_v1"
SUBSET_SEPARATOR = b"\\0"
DEPTH_FAR_EDGE_M = 6.45
ARRAY_LAYOUT = (
    ("camera_origin_body_f32.bin", "<f4", (3,)),
    ("camera_basis_body_fru_f32.bin", "<f4", (3, 3)),
    ("ground_plane_z_body_f32.bin", "<f4", (1,)),
    ("ground_support_in_frustum_u8.bin", "u1", (128, 128, 5)),
    ("ground_support_clear_to_target_u8.bin", "u1", (128, 128, 5)),
    ("pixel_hit_mask_u8.bin", "u1", (84, 112)),
    ("pixel_first_hit_distance_f32.bin", "<f4", (84, 112)),
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    claimed = payload.pop("content_sha256", None)
    if not isinstance(claimed, str):
        raise ValueError("hashed object lacks content_sha256")
    actual = canonical_sha256(payload)
    if actual != claimed:
        raise ValueError("canonical content SHA-256 mismatch")
    return actual


def load_hashed_json(path: Path, expected_file_sha256: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"expected a regular file: {path}")
    raw = path.read_bytes()
    if expected_file_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError(f"file SHA-256 mismatch: {path}")
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError(f"JSON root is not an object: {path}")
    content_sha256(value)
    return value


def resolve_child(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("declared path must be nonempty and relative")
    path = (root / relative).resolve(strict=True)
    if path != root and root not in path.parents:
        raise PermissionError("declared path escaped its root")
    if path.is_symlink() or not path.is_file():
        raise PermissionError("declared artifact is not a regular file")
    return path


def parse_index(path: Path, expected_sha256: str, frame_count: int) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("index SHA-256 mismatch")
    lines = raw.splitlines(keepends=True)
    if len(lines) != frame_count or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("index line framing changed")
    rows: list[dict[str, Any]] = []
    for line in lines:
        row = json.loads(line)
        if type(row) is not dict or line != canonical_bytes(row) + b"\n":
            raise ValueError("index row is not canonical JSON")
        content_sha256(row)
        rows.append(row)
    return rows


def load_verified_frames(
    manifest_path: Path,
    audit_path: Path,
) -> tuple[list[tuple[dict[str, Any], ObservableCameraRayEvidenceV4, np.ndarray]], int]:
    manifest = load_hashed_json(manifest_path, DATASET_FILE_SHA256)
    if manifest.get("content_sha256") != DATASET_CONTENT_SHA256:
        raise ValueError("dataset content identity changed")
    audit = load_hashed_json(audit_path, AUDIT_FILE_SHA256)
    dataset_binding = audit.get("dataset")
    if not isinstance(dataset_binding, Mapping) or dataset_binding != {
        "schema": "lewm_go2_observable_camera_ray_fit_v4_dataset_v1",
        "dataset_role": "train",
        "frame_count": 320,
        "scene_shard_count": 20,
        "file_sha256": DATASET_FILE_SHA256,
        "content_sha256": DATASET_CONTENT_SHA256,
    }:
        raise ValueError("audit is not bound to the dataset manifest")
    root = manifest_path.parent.resolve(strict=True)
    shards_root = (root / "shards").resolve(strict=True)
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("dataset must contain exactly 20 shards")
    frames: list[tuple[dict[str, Any], ObservableCameraRayEvidenceV4, np.ndarray]] = []
    verified_count = 0
    previous_scene = ""
    for top_record in records:
        if type(top_record) is not dict:
            raise ValueError("shard record is not an object")
        scene_key = str(top_record.get("scene_key_sha256", ""))
        if scene_key <= previous_scene:
            raise ValueError("shard scene order changed")
        previous_scene = scene_key
        shard_path = resolve_child(root, top_record.get("path"))
        if shard_path.parent.parent != shards_root:
            raise PermissionError("shard path shape changed")
        shard = load_hashed_json(shard_path, str(top_record.get("file_sha256")))
        verified_count += 1
        if (
            shard.get("content_sha256") != top_record.get("content_sha256")
            or shard.get("scene_key_sha256") != scene_key
            or shard.get("frame_count") != top_record.get("frame_count")
        ):
            raise ValueError("shard disagrees with the dataset manifest")
        frame_count = int(shard["frame_count"])
        files = shard.get("files")
        if not isinstance(files, list):
            raise ValueError("shard file inventory is missing")
        by_name = {str(record.get("path")): record for record in files if type(record) is dict}
        expected_names = {name for name, _dtype, _shape in ARRAY_LAYOUT} | {"index.jsonl"}
        if set(by_name) != expected_names or len(by_name) != len(files):
            raise ValueError("shard file inventory changed")
        arrays: dict[str, np.ndarray] = {}
        for name, dtype_text, trailing_shape in ARRAY_LAYOUT:
            record = by_name[name]
            expected_shape = [frame_count, *trailing_shape]
            dtype = np.dtype(dtype_text)
            expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * dtype.itemsize
            if (
                record.get("dtype") != dtype.str
                or record.get("shape") != expected_shape
                or record.get("byte_count") != expected_bytes
            ):
                raise ValueError(f"array contract changed: {name}")
            path = resolve_child(shard_path.parent, name)
            raw = path.read_bytes()
            if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != record.get("file_sha256"):
                raise ValueError(f"array bytes changed: {name}")
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(expected_shape)
            verified_count += 1
        index_record = by_name["index.jsonl"]
        index_path = resolve_child(shard_path.parent, "index.jsonl")
        rows = parse_index(index_path, str(index_record.get("file_sha256")), frame_count)
        verified_count += 1
        if canonical_sha256([row["frame_key"] for row in rows]) != shard.get("ordered_frame_keys_sha256"):
            raise ValueError("ordered frame-key commitment changed")
        evidence_hashes: list[str] = []
        for index, row in enumerate(rows):
            evidence = ObservableCameraRayEvidenceV4(
                camera_origin_body_m=arrays["camera_origin_body_f32.bin"][index],
                camera_basis_body_fru=arrays["camera_basis_body_fru_f32.bin"][index],
                ground_plane_z_body_m=float(arrays["ground_plane_z_body_f32.bin"][index, 0]),
                ground_support_in_frustum=arrays["ground_support_in_frustum_u8.bin"][index].astype(bool),
                ground_support_clear_to_target=arrays["ground_support_clear_to_target_u8.bin"][index].astype(bool),
                pixel_hit_mask=arrays["pixel_hit_mask_u8.bin"][index].astype(bool),
                pixel_first_hit_distance_m=arrays["pixel_first_hit_distance_f32.bin"][index],
            )
            raster = rasterize_observable_camera_ray_evidence_v4(evidence)
            if (
                evidence.content_sha256() != row.get("evidence_content_sha256")
                or raster.content_sha256() != row.get("raster_content_sha256")
            ):
                raise ValueError("frame evidence/raster commitment changed")
            evidence_hashes.append(evidence.content_sha256())
            frames.append((row, evidence, raster.output_labels))
        if canonical_sha256(evidence_hashes) != shard.get("ordered_evidence_sha256"):
            raise ValueError("ordered evidence commitment changed")
    if verified_count != 180 or len(frames) != 320:
        raise ValueError("verified dataset cardinality changed")
    families = Counter(str(row["frame_key"]["family"]) for row, _evidence, _raster in frames)
    if families != Counter({family: 64 for family in FAMILIES}):
        raise ValueError("dataset is not balanced 64 frames per family")
    return frames, verified_count


def balanced_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: dict[str, list[tuple[str, bytes, dict[str, Any]]]] = {family: [] for family in FAMILIES}
    for row in rows:
        frame_key = row["frame_key"]
        family = str(frame_key["family"])
        key = canonical_bytes(frame_key)
        rank = hashlib.sha256(SUBSET_NAMESPACE + SUBSET_SEPARATOR + key).hexdigest()
        ranked[family].append((rank, key, row))
    for family in FAMILIES:
        ranked[family].sort(key=lambda item: (item[0], item[1]))
    return [ranked[family][ordinal][2] for ordinal in range(64) for family in FAMILIES]


def frame_target_hash(
    row: Mapping[str, Any],
    evidence: ObservableCameraRayEvidenceV4,
    raster: np.ndarray,
    hit: np.ndarray,
    distance: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_bytes(row["frame_key"]))
    arrays = (
        ("pixel_in_range_hit", hit, np.uint8),
        ("pixel_depth", evidence.pixel_first_hit_distance_m, np.dtype("<f4")),
        ("ground_valid", evidence.ground_support_in_frustum, np.uint8),
        ("ground_clear", evidence.ground_support_clear_to_target, np.uint8),
        ("ground_distance", distance, np.dtype("<f8")),
        ("raster", raster, np.uint8),
    )
    for name, value, dtype in arrays:
        raw = np.ascontiguousarray(value, dtype=dtype).tobytes(order="C")
        digest.update(name.encode("ascii"))
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def derive_freeze(manifest_path: Path = DATASET_PATH, audit_path: Path = AUDIT_PATH) -> dict[str, Any]:
    frames, verified_count = load_verified_frames(manifest_path, audit_path)
    rows = [row for row, _evidence, _raster in frames]
    by_key = {canonical_bytes(row["frame_key"]): (row, evidence, raster) for row, evidence, raster in frames}
    ordered = balanced_order(rows)
    fit_sizes: dict[str, Any] = {}
    for fit_size in FIT_SIZES:
        selected = ordered[:fit_size]
        keys = [canonical_sha256(row["frame_key"]) for row in selected]
        signature = {
            "pixel": [0, 0],
            "depth_count": 0,
            "ground_overall": [0, 0],
            "ground_by_distance": {name: [0, 0] for name in DISTANCE_GROUPS},
            "ground_by_family": {family: [0, 0] for family in FAMILIES},
            "raster_target_counts": [0, 0, 0],
        }
        target_digest = hashlib.sha256(b"lewm_go2_v4_target_partition_bytes_v1")
        per_frame: list[str] = []
        for selected_row in selected:
            row, evidence, raster = by_key[canonical_bytes(selected_row["frame_key"])]
            hit = evidence.pixel_hit_mask & (evidence.pixel_first_hit_distance_m < DEPTH_FAR_EDGE_M)
            pixel = [int((~hit).sum()), int(hit.sum())]
            signature["pixel"][0] += pixel[0]
            signature["pixel"][1] += pixel[1]
            signature["depth_count"] += pixel[1]
            queries = project_canonical_ground_support_v4(
                camera_origin_body_m=evidence.camera_origin_body_m,
                camera_basis_body_fru=evidence.camera_basis_body_fru,
                ground_plane_z_body_m=evidence.ground_plane_z_body_m,
            )
            valid = evidence.ground_support_in_frustum
            clear = evidence.ground_support_clear_to_target
            if not np.array_equal(valid, queries.in_frustum):
                raise ValueError("ground calibration changed")
            ground = [int((valid & ~clear).sum()), int((valid & clear).sum())]
            for state in range(2):
                signature["ground_overall"][state] += ground[state]
                signature["ground_by_family"][str(row["frame_key"]["family"])][state] += ground[state]
            for name, low, high in zip(DISTANCE_GROUPS, DISTANCE_EDGES_M[:-1], DISTANCE_EDGES_M[1:]):
                group = valid & (queries.target_distance_m >= low) & (queries.target_distance_m < high)
                signature["ground_by_distance"][name][0] += int((group & ~clear).sum())
                signature["ground_by_distance"][name][1] += int((group & clear).sum())
            for target_class in range(3):
                signature["raster_target_counts"][target_class] += int((raster == target_class).sum())
            target_hash = frame_target_hash(row, evidence, raster, hit, queries.target_distance_m)
            per_frame.append(target_hash)
            target_digest.update(bytes.fromhex(target_hash))
        fit_sizes[str(fit_size)] = {
            "family_counts": {family: sum(row["frame_key"]["family"] == family for row in selected) for family in FAMILIES},
            "first_frame_key_sha256": keys[0],
            "fit_size": fit_size,
            "last_frame_key_sha256": keys[-1],
            "ordered_per_frame_target_sha256": canonical_sha256(per_frame),
            "ordered_target_bytes_sha256": target_digest.hexdigest(),
            "signature": signature,
            "signature_sha256": canonical_sha256(signature),
            "subset_content_sha256": canonical_sha256(keys),
        }
    manifest = load_hashed_json(manifest_path, DATASET_FILE_SHA256)
    result = {
        "audit_receipt_file_sha256": file_sha256(audit_path),
        "dataset_manifest_content_sha256": manifest["content_sha256"],
        "dataset_manifest_file_sha256": file_sha256(manifest_path),
        "derivation": "verified_audited_evidence_arrays_then_family_round_robin_ascii_backslash_zero_rank_v1",
        "fit_sizes": fit_sizes,
        "schema": "lewm_go2_observable_camera_ray_fit_v4_target_partitions_v1",
        "verified_dataset_file_count": verified_count,
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def verify_frozen_partitions() -> dict[str, Any]:
    if file_sha256(FREEZE_PATH) != FREEZE_FILE_SHA256:
        raise ValueError("target-partition freeze file changed")
    frozen = load_hashed_json(FREEZE_PATH, FREEZE_FILE_SHA256)
    if frozen.get("content_sha256") != FREEZE_CONTENT_SHA256:
        raise ValueError("target-partition freeze content changed")
    derived = derive_freeze()
    if derived != frozen:
        raise ValueError("derived target partitions differ from the frozen artifact")
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = verify_frozen_partitions()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
