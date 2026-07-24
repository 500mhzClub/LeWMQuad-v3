"""Role-global, repository-fixed attempt reservation for Shared JEPA V5."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from lewm.models.shared_observable_camera_ray_jepa_v5_authority import (
    require_frozen_production_authority,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256_component(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 path component")
    return value


def role_global_namespace(
    *,
    gate: str,
    dataset_role_manifest_file_sha256: str,
    role_commitment_file_sha256: str,
    protocol_generation: str,
) -> str:
    if gate not in {"g2", "g3"}:
        raise ValueError("gate must be g2 or g3")
    _require_sha256_component(
        dataset_role_manifest_file_sha256,
        name="dataset role manifest hash",
    )
    _require_sha256_component(
        role_commitment_file_sha256,
        name="role commitment hash",
    )
    if type(protocol_generation) is not str or not protocol_generation:
        raise ValueError("protocol generation must be a nonempty string")
    return hashlib.sha256(
        _canonical_bytes(
            {
                "schema": "lewm_go2_shared_jepa_role_global_registry_namespace_v6",
                "gate": gate,
                "dataset_role_manifest_file_sha256": (
                    dataset_role_manifest_file_sha256
                ),
                "role_commitment_file_sha256": role_commitment_file_sha256,
                "protocol_generation": protocol_generation,
            }
        )
    ).hexdigest()


def _removed_acquire_canonical_attempt_tombstone(
    *,
    gate: str,
    namespace_sha256: str,
    reservation: Mapping[str, object],
) -> tuple[str, str]:
    """Removed: only the one-shot runner owns role-global reservation."""

    raise PermissionError("production registry mutation was removed; use the one-shot runner CLI")


__all__ = ["role_global_namespace"]
