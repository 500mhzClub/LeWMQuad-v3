"""Content-addressed camera provenance for rendered Go2 RGB sources."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SOURCE_CAMERA_CONTRACT_SCHEMA = "lewm_go2_source_camera_contract_v2"


class SourceCameraContractError(ValueError):
    """Raised when source-camera provenance is absent or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_source_camera_contract(path: Path) -> dict[str, Any]:
    """Load and fail closed on a source-bound camera contract."""

    resolved = path.resolve()
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise SourceCameraContractError("source camera contract must be an object")
    if payload.get("schema") != SOURCE_CAMERA_CONTRACT_SCHEMA:
        raise SourceCameraContractError("unsupported source camera contract schema")
    core = dict(payload)
    declared_content_sha = str(core.pop("content_sha256", ""))
    if declared_content_sha != _canonical_sha256(core):
        raise SourceCameraContractError("source camera contract content hash mismatch")
    source_index = payload.get("source_index")
    actual = payload.get("actual_source_projection")
    platform = payload.get("platform_projection_after_rectification")
    if not all(isinstance(value, Mapping) for value in (source_index, actual, platform)):
        raise SourceCameraContractError("source camera contract records are missing")
    if payload.get("g2_images_opened") is not False:
        raise SourceCameraContractError("source camera audit must not open G2 images")
    scene_count = int(payload.get("scene_count", 0))
    records = payload.get("source_records")
    if scene_count <= 0 or not isinstance(records, list) or len(records) != scene_count:
        raise SourceCameraContractError("source camera scene count is inconsistent")
    for name, value in (
        ("source horizontal FOV", actual.get("horizontal_fov_deg")),
        ("source vertical FOV", actual.get("vertical_fov_deg")),
        ("platform horizontal FOV", platform.get("horizontal_fov_deg")),
        ("platform vertical FOV", platform.get("vertical_fov_deg")),
    ):
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise SourceCameraContractError(f"{name} must be finite")
    return payload


def validate_source_camera_contract_for_index(
    contract: Mapping[str, Any],
    *,
    source_index_path: Path,
) -> None:
    """Require the audited source index to match the builder input bytewise."""

    record = contract.get("source_index")
    if not isinstance(record, Mapping):
        raise SourceCameraContractError("source camera contract has no source index")
    resolved = source_index_path.resolve()
    if str(record.get("sha256", "")) != _sha256_file(resolved):
        raise SourceCameraContractError("source camera contract targets another index")


def source_camera_contract_record(path: Path) -> dict[str, Any]:
    """Return the immutable identity stored by downstream artifacts."""

    resolved = path.resolve()
    payload = load_source_camera_contract(resolved)
    return {
        "schema": payload["schema"],
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "content_sha256": payload["content_sha256"],
        "scene_count": int(payload["scene_count"]),
    }


__all__ = [
    "SOURCE_CAMERA_CONTRACT_SCHEMA",
    "SourceCameraContractError",
    "load_source_camera_contract",
    "source_camera_contract_record",
    "validate_source_camera_contract_for_index",
]
