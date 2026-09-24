"""Complete fingerprint-continuity successor for V5 metadata planning.

V5 preserves V4's component-wise no-follow source-index walk and scientific
result.  It additionally retains the original complete fingerprint for every
opened directory component and the leaf, and rechecks every named entry and
open descriptor immediately before and after the only byte read.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Sequence

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v4 as _v4


ALL_ROLES = _v4.ALL_ROLES
DATASET_MANIFEST_FILE_SHA256 = _v4.DATASET_MANIFEST_FILE_SHA256
DATASET_ROWS_FILE_SHA256 = _v4.DATASET_ROWS_FILE_SHA256
DEVELOPMENT_ROLES = _v4.DEVELOPMENT_ROLES
DevelopmentRawSupervisionPlan = _v4.DevelopmentRawSupervisionPlan
DevelopmentSourceInventory = _v4.DevelopmentSourceInventory
ENDPOINT_SCHEMA = _v4.ENDPOINT_SCHEMA
PAIR_SCHEMA = _v4.PAIR_SCHEMA
PLAN_SCHEMA = _v4.PLAN_SCHEMA
PRIMITIVE_VOCABULARY = _v4.PRIMITIVE_VOCABULARY
ROLE_ASSIGNMENT_SHA256 = _v4.ROLE_ASSIGNMENT_SHA256
RawSupervisionPlanError = _v4.RawSupervisionPlanError
SOURCE_INDEX_FILE_SHA256 = _v4.SOURCE_INDEX_FILE_SHA256
SOURCE_INDEX_RELATIVE_PATH = _v4.SOURCE_INDEX_RELATIVE_PATH
SOURCE_INVENTORY_SHA256 = _v4.SOURCE_INVENTORY_SHA256
canonical_json_bytes = _v4.canonical_json_bytes
canonical_json_sha256 = _v4.canonical_json_sha256
endpoint_identity = _v4.endpoint_identity
plan_development_raw_supervision = _v4.plan_development_raw_supervision
plan_development_source_inventory = _v4.plan_development_source_inventory

_entry_identity = _v4._entry_identity
_file_fingerprint = _v4._file_fingerprint
_relative_components = _v4._relative_components
_directory_flags = _v4._directory_flags
_file_flags = _v4._file_flags
_lstat_at = _v4._lstat_at
_validate_open_directory = _v4._validate_open_directory
_validate_open_leaf = _v4._validate_open_leaf

Fingerprint = tuple[int, int, int, int, int, int, int]
DirectoryEntry = tuple[int, str, int, Fingerprint]


def _revalidate_open_chain(
    *,
    filesystem_root: Path,
    anchor_fd: int,
    anchor_fingerprint: Fingerprint,
    root: Path,
    root_fd: int,
    root_fingerprint: Fingerprint,
    root_directory_entries: Sequence[DirectoryEntry],
    directory_entries: Sequence[DirectoryEntry],
    leaf_parent_fd: int,
    leaf_name: str,
    leaf_fd: int,
    leaf_fingerprint: Fingerprint,
) -> None:
    """Require every original named-entry and descriptor fingerprint."""

    try:
        anchor_entry = filesystem_root.stat(follow_symlinks=False)
        root_metadata = root.stat(follow_symlinks=False)
        root_resolved = root.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RawSupervisionPlanError(
            "repository descriptor chain changed during source-index read"
        ) from error

    if (
        not stat.S_ISDIR(anchor_entry.st_mode)
        or _file_fingerprint(anchor_entry) != anchor_fingerprint
        or _file_fingerprint(os.fstat(anchor_fd)) != anchor_fingerprint
    ):
        raise RawSupervisionPlanError(
            "filesystem root directory changed during source-index read"
        )

    if (
        root_resolved != root
        or root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or _file_fingerprint(root_metadata) != root_fingerprint
        or _file_fingerprint(os.fstat(root_fd)) != root_fingerprint
    ):
        raise RawSupervisionPlanError(
            "repo_root directory changed during source-index read"
        )

    for parent_fd, component, child_fd, fingerprint in (
        *root_directory_entries,
        *directory_entries,
    ):
        entry = _lstat_at(
            parent_fd,
            component,
            purpose=f"descriptor directory {component!r}",
        )
        opened = os.fstat(child_fd)
        if (
            stat.S_ISLNK(entry.st_mode)
            or not stat.S_ISDIR(entry.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or _file_fingerprint(entry) != fingerprint
            or _file_fingerprint(opened) != fingerprint
        ):
            raise RawSupervisionPlanError(
                f"descriptor directory {component!r} changed during read"
            )

    leaf_entry = _lstat_at(
        leaf_parent_fd,
        leaf_name,
        purpose="rendered source index",
    )
    opened_leaf = os.fstat(leaf_fd)
    if (
        stat.S_ISLNK(leaf_entry.st_mode)
        or not stat.S_ISREG(leaf_entry.st_mode)
        or not stat.S_ISREG(opened_leaf.st_mode)
        or int(leaf_entry.st_nlink) != 1
        or int(opened_leaf.st_nlink) != 1
        or _file_fingerprint(leaf_entry) != leaf_fingerprint
        or _file_fingerprint(opened_leaf) != leaf_fingerprint
    ):
        raise RawSupervisionPlanError("rendered source index changed during read")


def _read_frozen_source_index(repo_root: Path) -> bytes:
    """Read the frozen index with complete pre/post fingerprint continuity."""

    root = _v4._v3._v2._canonical_existing_directory(
        Path(repo_root),
        name="repo_root",
    )
    components = _relative_components(
        SOURCE_INDEX_RELATIVE_PATH,
        name="rendered source index path",
    )
    if not components:
        raise RawSupervisionPlanError("rendered source index path is empty")

    root_chain_fds: list[int] = []
    root_directory_entries: list[DirectoryEntry] = []
    directory_fds: list[int] = []
    directory_entries: list[DirectoryEntry] = []
    leaf_fd: int | None = None
    payload = b""
    try:
        try:
            root_before = root.stat(follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise RawSupervisionPlanError(
                "repo_root changed before descriptor walk"
            ) from error
        root_fingerprint = _file_fingerprint(root_before)
        filesystem_root = Path(root.anchor)
        if not root.anchor or not filesystem_root.is_absolute():
            raise RawSupervisionPlanError("repo_root has no canonical filesystem root")
        try:
            anchor_before = filesystem_root.stat(follow_symlinks=False)
            anchor_fingerprint = _file_fingerprint(anchor_before)
            anchor_fd = os.open(filesystem_root, _directory_flags())
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise RawSupervisionPlanError(
                "filesystem root changed during descriptor open"
            ) from error
        root_chain_fds.append(anchor_fd)
        _validate_open_directory(
            os.fstat(anchor_fd),
            expected_identity=_entry_identity(anchor_before),
            expected_fingerprint=anchor_fingerprint,
            purpose="filesystem root",
        )

        parent_fd = anchor_fd
        for component in root.parts[1:]:
            before = _lstat_at(
                parent_fd,
                component,
                purpose=f"repo_root directory {component!r}",
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise RawSupervisionPlanError(
                    f"repo_root directory {component!r} must be a real directory"
                )
            expected_identity = _entry_identity(before)
            expected_fingerprint = _file_fingerprint(before)
            try:
                child_fd = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=parent_fd,
                )
            except (FileNotFoundError, NotADirectoryError, OSError) as error:
                raise RawSupervisionPlanError(
                    f"repo_root directory {component!r} changed during open"
                ) from error
            root_chain_fds.append(child_fd)
            _validate_open_directory(
                os.fstat(child_fd),
                expected_identity=expected_identity,
                expected_fingerprint=expected_fingerprint,
                purpose=f"repo_root directory {component!r}",
            )
            root_directory_entries.append(
                (parent_fd, component, child_fd, expected_fingerprint)
            )
            parent_fd = child_fd

        root_fd = parent_fd
        _validate_open_directory(
            os.fstat(root_fd),
            expected_identity=_entry_identity(root_before),
            expected_fingerprint=root_fingerprint,
            purpose="repo_root",
        )

        parent_fd = root_fd
        for component in components[:-1]:
            before = _lstat_at(
                parent_fd,
                component,
                purpose=f"source-index directory {component!r}",
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise RawSupervisionPlanError(
                    f"source-index directory {component!r} must be a real directory"
                )
            expected_identity = _entry_identity(before)
            expected_fingerprint = _file_fingerprint(before)
            try:
                child_fd = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=parent_fd,
                )
            except (FileNotFoundError, NotADirectoryError, OSError) as error:
                raise RawSupervisionPlanError(
                    f"source-index directory {component!r} changed during open"
                ) from error
            directory_fds.append(child_fd)
            _validate_open_directory(
                os.fstat(child_fd),
                expected_identity=expected_identity,
                expected_fingerprint=expected_fingerprint,
                purpose=f"source-index directory {component!r}",
            )
            directory_entries.append(
                (parent_fd, component, child_fd, expected_fingerprint)
            )
            parent_fd = child_fd

        leaf_name = components[-1]
        leaf_before = _lstat_at(
            parent_fd,
            leaf_name,
            purpose="rendered source index",
        )
        if stat.S_ISLNK(leaf_before.st_mode):
            raise RawSupervisionPlanError(
                "rendered source index must not be a symlink"
            )
        leaf_identity = _entry_identity(leaf_before)
        leaf_fingerprint = _file_fingerprint(leaf_before)
        _validate_open_leaf(
            leaf_before,
            expected_identity=leaf_identity,
            purpose="rendered source index",
        )
        try:
            leaf_fd = os.open(leaf_name, _file_flags(), dir_fd=parent_fd)
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise RawSupervisionPlanError(
                "rendered source index changed during descriptor open"
            ) from error
        opened_leaf = os.fstat(leaf_fd)
        _validate_open_leaf(
            opened_leaf,
            expected_identity=leaf_identity,
            purpose="rendered source index",
        )
        if _file_fingerprint(opened_leaf) != leaf_fingerprint:
            raise RawSupervisionPlanError(
                "rendered source index fingerprint changed during descriptor open"
            )

        validation = {
            "filesystem_root": filesystem_root,
            "anchor_fd": anchor_fd,
            "anchor_fingerprint": anchor_fingerprint,
            "root": root,
            "root_fd": root_fd,
            "root_fingerprint": root_fingerprint,
            "root_directory_entries": root_directory_entries,
            "directory_entries": directory_entries,
            "leaf_parent_fd": parent_fd,
            "leaf_name": leaf_name,
            "leaf_fd": leaf_fd,
            "leaf_fingerprint": leaf_fingerprint,
        }
        _revalidate_open_chain(**validation)

        chunks: list[bytes] = []
        while True:
            chunk = os.read(leaf_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        _revalidate_open_chain(**validation)
        payload = b"".join(chunks)
    finally:
        if leaf_fd is not None:
            os.close(leaf_fd)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)
        for descriptor in reversed(root_chain_fds):
            os.close(descriptor)

    if hashlib.sha256(payload).hexdigest() != SOURCE_INDEX_FILE_SHA256:
        raise RawSupervisionPlanError("rendered source index file SHA-256 changed")
    return payload


def load_frozen_development_metadata(
    repo_root: Path,
) -> DevelopmentRawSupervisionPlan:
    root = _v4._v3._v2._canonical_existing_directory(
        Path(repo_root),
        name="repo_root",
    )
    return _v4.load_frozen_development_metadata(root)


def load_frozen_development_source_inventory(
    repo_root: Path,
    plan: DevelopmentRawSupervisionPlan | None = None,
) -> DevelopmentSourceInventory:
    root = _v4._v3._v2._canonical_existing_directory(
        Path(repo_root),
        name="repo_root",
    )
    resolved_plan = plan or load_frozen_development_metadata(root)
    rows = _v4._v3._v2._parse_source_index(_read_frozen_source_index(root))
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
    "SOURCE_INDEX_RELATIVE_PATH",
    "SOURCE_INVENTORY_SHA256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "endpoint_identity",
    "load_frozen_development_metadata",
    "load_frozen_development_source_inventory",
    "plan_development_raw_supervision",
    "plan_development_source_inventory",
]
