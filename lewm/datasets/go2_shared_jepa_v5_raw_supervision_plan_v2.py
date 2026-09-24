"""Path-hardened successor for the shared-JEPA V5 raw-supervision plan.

V1 froze the scientific pair, endpoint, and source-inventory identities but
accepted a lexical in-repository path through a symlinked directory.  V2 keeps
those identities unchanged and validates each retained source path using only
filesystem metadata.  Referenced file bytes remain unopened; a later builder
must repeat no-follow validation when it actually opens a source.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan as _v1


ALL_ROLES = _v1.ALL_ROLES
DATASET_MANIFEST_FILE_SHA256 = _v1.DATASET_MANIFEST_FILE_SHA256
DATASET_ROWS_FILE_SHA256 = _v1.DATASET_ROWS_FILE_SHA256
DEVELOPMENT_ROLES = _v1.DEVELOPMENT_ROLES
DevelopmentRawSupervisionPlan = _v1.DevelopmentRawSupervisionPlan
DevelopmentSourceInventory = _v1.DevelopmentSourceInventory
ENDPOINT_SCHEMA = _v1.ENDPOINT_SCHEMA
PAIR_SCHEMA = _v1.PAIR_SCHEMA
PLAN_SCHEMA = _v1.PLAN_SCHEMA
PRIMITIVE_VOCABULARY = _v1.PRIMITIVE_VOCABULARY
ROLE_ASSIGNMENT_SHA256 = _v1.ROLE_ASSIGNMENT_SHA256
RawSupervisionPlanError = _v1.RawSupervisionPlanError
SOURCE_INDEX_FILE_SHA256 = _v1.SOURCE_INDEX_FILE_SHA256
SOURCE_INDEX_RELATIVE_PATH = _v1.SOURCE_INDEX_RELATIVE_PATH
SOURCE_INVENTORY_SHA256 = _v1.SOURCE_INVENTORY_SHA256
canonical_json_bytes = _v1.canonical_json_bytes
canonical_json_sha256 = _v1.canonical_json_sha256
endpoint_identity = _v1.endpoint_identity
plan_development_raw_supervision = _v1.plan_development_raw_supervision


_REFERENCED_FIELDS = (
    ("frames", "path"),
    ("scene_manifest", "path"),
    ("render_plan", "path"),
    ("render_summary", "path"),
)


def _canonical_absolute_path(value: object, *, name: str) -> tuple[str, Path]:
    if type(value) is not str or not value:
        raise RawSupervisionPlanError(f"{name} must be a nonempty string")
    raw = value
    path = Path(raw)
    if (
        "\x00" in raw
        or not path.is_absolute()
        or raw.startswith("//")
        or raw != os.path.normpath(raw)
        or raw != str(path)
        or ".." in path.parts
    ):
        raise RawSupervisionPlanError(f"{name} must be canonical and absolute")
    return raw, path


def _canonical_existing_directory(value: Path, *, name: str) -> Path:
    raw = os.fspath(value)
    _, path = _canonical_absolute_path(raw, name=name)
    try:
        resolved = path.resolve(strict=True)
        metadata = path.stat(follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RawSupervisionPlanError(f"{name} must exist") from error
    if resolved != path or path.is_symlink():
        raise RawSupervisionPlanError(f"{name} must not use a path alias")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RawSupervisionPlanError(f"{name} must be a directory")
    return path


def _validate_referenced_path(
    value: object,
    *,
    repo_root: Path,
    name: str,
) -> tuple[str, tuple[int, int]]:
    """Validate one source path without opening or reading the referenced file."""

    raw, path = _canonical_absolute_path(value, name=name)
    try:
        relative = path.relative_to(repo_root)
    except ValueError as error:
        raise PermissionError(f"{name} escapes the repository") from error
    if not relative.parts:
        raise RawSupervisionPlanError(f"{name} must identify a regular file")

    current = repo_root
    leaf_metadata: os.stat_result | None = None
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            metadata = current.stat(follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise RawSupervisionPlanError(f"{name} must exist") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RawSupervisionPlanError(
                f"{name} must not contain a symlinked component"
            )
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise RawSupervisionPlanError(
                f"{name} has a non-directory intermediate component"
            )
        leaf_metadata = metadata

    assert leaf_metadata is not None
    if not stat.S_ISREG(leaf_metadata.st_mode):
        raise RawSupervisionPlanError(f"{name} must identify a regular file")
    if leaf_metadata.st_nlink != 1:
        raise RawSupervisionPlanError(f"{name} must not identify a hard-link alias")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise PermissionError(f"{name} resolves outside the repository") from error
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RawSupervisionPlanError(f"{name} must exist") from error
    if resolved != path:
        raise RawSupervisionPlanError(f"{name} must not use a path alias")
    return raw, (int(leaf_metadata.st_dev), int(leaf_metadata.st_ino))


def _validate_inventory_paths(
    inventory: DevelopmentSourceInventory,
    *,
    repo_root: Path,
) -> None:
    seen_paths: set[str] = set()
    seen_files: set[tuple[int, int]] = set()
    for record in inventory.records:
        scene_id = record.get("scene_id")
        if type(scene_id) is not str or not scene_id:
            raise RawSupervisionPlanError("inventory scene_id must be a string")
        for section, field in _REFERENCED_FIELDS:
            nested = record.get(section)
            if not isinstance(nested, Mapping):
                raise RawSupervisionPlanError(
                    f"{scene_id} {section} metadata is absent"
                )
            raw, identity = _validate_referenced_path(
                nested.get(field),
                repo_root=repo_root,
                name=f"{scene_id}.{section}.{field}",
            )
            if raw in seen_paths or identity in seen_files:
                raise RawSupervisionPlanError(
                    "referenced metadata paths must have unique file identities"
                )
            seen_paths.add(raw)
            seen_files.add(identity)


def _validate_selected_source_paths(
    source_rows: Sequence[Mapping[str, Any]],
    inventory: DevelopmentSourceInventory,
    *,
    repo_root: Path,
) -> None:
    """Validate original lexical strings for retained development rows only."""

    selected_scenes = {str(record["scene_id"]) for record in inventory.records}
    field_map = (
        ("frames_jsonl_path", "frames"),
        ("scene_manifest_path", "scene_manifest"),
        ("render_plan_path", "render_plan"),
        ("render_summary_path", "render_summary"),
    )
    seen_paths: set[str] = set()
    seen_files: set[tuple[int, int]] = set()
    for row in source_rows:
        if row.get("scene_id") not in selected_scenes:
            continue
        scene_id = str(row["scene_id"])
        for source_field, section in field_map:
            raw, identity = _validate_referenced_path(
                row.get(source_field),
                repo_root=repo_root,
                name=f"{scene_id}.{section}.path",
            )
            if raw in seen_paths or identity in seen_files:
                raise RawSupervisionPlanError(
                    "referenced metadata paths must have unique file identities"
                )
            seen_paths.add(raw)
            seen_files.add(identity)


def _validate_selected_source_lexical_paths(
    plan: DevelopmentRawSupervisionPlan,
    source_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject unsafe spellings before V1 can normalize a retained path."""

    selected_scenes: set[str] = set()
    for endpoint in plan.endpoints:
        identity = endpoint.get("identity")
        if isinstance(identity, Mapping) and type(identity.get("scene_id")) is str:
            selected_scenes.add(str(identity["scene_id"]))
    for row in source_rows:
        if type(row) is not dict or row.get("scene_id") not in selected_scenes:
            continue
        scene_id = str(row["scene_id"])
        for field, section in (
            ("frames_jsonl_path", "frames"),
            ("scene_manifest_path", "scene_manifest"),
            ("render_plan_path", "render_plan"),
            ("render_summary_path", "render_summary"),
        ):
            _canonical_absolute_path(
                row.get(field),
                name=f"{scene_id}.{section}.path",
            )


def load_frozen_development_metadata(
    repo_root: Path,
) -> DevelopmentRawSupervisionPlan:
    """Load the frozen V1 scientific identities without changing their bytes."""

    root = _canonical_existing_directory(Path(repo_root), name="repo_root")
    return _v1.load_frozen_development_metadata(root)


def plan_development_source_inventory(
    plan: DevelopmentRawSupervisionPlan,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    enforce_frozen_hashes: bool = True,
) -> DevelopmentSourceInventory:
    """Produce the V1 inventory only after its retained paths pass V2 checks.

    V1 performs selection and all frozen identity checks without dereferencing
    a source path.  V2 then uses lstat/resolve metadata only on the retained
    development records.  No excluded G2 source path is inspected.
    """

    root = _canonical_existing_directory(Path(repo_root), name="repo_root")
    _validate_selected_source_lexical_paths(plan, source_rows)
    inventory = _v1.plan_development_source_inventory(
        plan,
        source_rows,
        repo_root=root,
        enforce_frozen_hashes=enforce_frozen_hashes,
    )
    _validate_selected_source_paths(source_rows, inventory, repo_root=root)
    _validate_inventory_paths(inventory, repo_root=root)
    return inventory


def _read_frozen_source_index(repo_root: Path) -> bytes:
    """Read the one allowed index after metadata-only no-alias validation."""

    source_path = repo_root / SOURCE_INDEX_RELATIVE_PATH
    _, identity = _validate_referenced_path(
        str(source_path),
        repo_root=repo_root,
        name="rendered source index",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino)) != identity
        ):
            raise RawSupervisionPlanError("rendered source index identity changed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        ) != (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_size),
            int(opened.st_mtime_ns),
        ):
            raise RawSupervisionPlanError("rendered source index changed while read")
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != SOURCE_INDEX_FILE_SHA256:
        raise RawSupervisionPlanError("rendered source index file SHA-256 changed")
    return payload


def _parse_source_index(payload: bytes) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise RawSupervisionPlanError(
            "rendered source index must be nonempty newline JSONL"
        )
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RawSupervisionPlanError(
                f"rendered source index:{number} is invalid JSON"
            ) from error
        if type(value) is not dict:
            raise RawSupervisionPlanError(
                f"rendered source index:{number} is not an object"
            )
        rows.append(value)
    return rows


def load_frozen_development_source_inventory(
    repo_root: Path,
    plan: DevelopmentRawSupervisionPlan | None = None,
) -> DevelopmentSourceInventory:
    root = _canonical_existing_directory(Path(repo_root), name="repo_root")
    resolved_plan = plan or load_frozen_development_metadata(root)
    rows = _parse_source_index(_read_frozen_source_index(root))
    return plan_development_source_inventory(
        resolved_plan,
        rows,
        repo_root=root,
    )


__all__ = [
    "ALL_ROLES",
    "DATASET_MANIFEST_FILE_SHA256",
    "DATASET_ROWS_FILE_SHA256",
    "DEVELOPMENT_ROLES",
    "DevelopmentRawSupervisionPlan",
    "DevelopmentSourceInventory",
    "ENDPOINT_SCHEMA",
    "PAIR_SCHEMA",
    "PLAN_SCHEMA",
    "PRIMITIVE_VOCABULARY",
    "ROLE_ASSIGNMENT_SHA256",
    "RawSupervisionPlanError",
    "SOURCE_INDEX_FILE_SHA256",
    "SOURCE_INVENTORY_SHA256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "endpoint_identity",
    "load_frozen_development_metadata",
    "load_frozen_development_source_inventory",
    "plan_development_raw_supervision",
    "plan_development_source_inventory",
]
