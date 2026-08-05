"""Full-continuity successor for V5 raw-supervision metadata planning.

V4 preserves V3 scientific behavior while closing its two independent review
findings.  The repository root is acquired component-by-component from the
filesystem root, and the complete validated leaf fingerprint must equal the
opened descriptor before any byte read and after the read.  Referenced source
payloads remain metadata-only and unopened.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Sequence

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v3 as _v3


ALL_ROLES = _v3.ALL_ROLES
DATASET_MANIFEST_FILE_SHA256 = _v3.DATASET_MANIFEST_FILE_SHA256
DATASET_ROWS_FILE_SHA256 = _v3.DATASET_ROWS_FILE_SHA256
DEVELOPMENT_ROLES = _v3.DEVELOPMENT_ROLES
DevelopmentRawSupervisionPlan = _v3.DevelopmentRawSupervisionPlan
DevelopmentSourceInventory = _v3.DevelopmentSourceInventory
ENDPOINT_SCHEMA = _v3.ENDPOINT_SCHEMA
PAIR_SCHEMA = _v3.PAIR_SCHEMA
PLAN_SCHEMA = _v3.PLAN_SCHEMA
PRIMITIVE_VOCABULARY = _v3.PRIMITIVE_VOCABULARY
ROLE_ASSIGNMENT_SHA256 = _v3.ROLE_ASSIGNMENT_SHA256
RawSupervisionPlanError = _v3.RawSupervisionPlanError
SOURCE_INDEX_FILE_SHA256 = _v3.SOURCE_INDEX_FILE_SHA256
SOURCE_INDEX_RELATIVE_PATH = _v3.SOURCE_INDEX_RELATIVE_PATH
SOURCE_INVENTORY_SHA256 = _v3.SOURCE_INVENTORY_SHA256
canonical_json_bytes = _v3.canonical_json_bytes
canonical_json_sha256 = _v3.canonical_json_sha256
endpoint_identity = _v3.endpoint_identity
plan_development_raw_supervision = _v3.plan_development_raw_supervision
plan_development_source_inventory = _v3.plan_development_source_inventory


def _entry_identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _file_fingerprint(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _relative_components(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not str or not value:
        raise RawSupervisionPlanError(f"{name} must be a nonempty string")
    path = Path(value)
    if (
        "\x00" in value
        or path.is_absolute()
        or value != os.path.normpath(value)
        or value != str(path)
        or value.startswith("//")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RawSupervisionPlanError(f"{name} must be canonical and relative")
    return tuple(path.parts)


def _directory_flags() -> int:
    required = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise RawSupervisionPlanError(
            "descriptor-relative no-follow directory opens are unavailable"
        )
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | required


def _file_flags() -> int:
    if not getattr(os, "O_NOFOLLOW", 0):
        raise RawSupervisionPlanError("no-follow file opens are unavailable")
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | os.O_NOFOLLOW
    )


def _lstat_at(parent_fd: int, name: str, *, purpose: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RawSupervisionPlanError(f"{purpose} must exist without aliases") from error


def _validate_open_directory(
    metadata: os.stat_result,
    *,
    expected_identity: tuple[int, int],
    expected_fingerprint: tuple[int, int, int, int, int, int, int] | None = None,
    purpose: str,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise RawSupervisionPlanError(f"{purpose} must be a directory")
    if _entry_identity(metadata) != expected_identity:
        raise RawSupervisionPlanError(f"{purpose} identity changed during open")
    if (
        expected_fingerprint is not None
        and _file_fingerprint(metadata) != expected_fingerprint
    ):
        raise RawSupervisionPlanError(f"{purpose} fingerprint changed during open")


def _validate_open_leaf(
    metadata: os.stat_result,
    *,
    expected_identity: tuple[int, int],
    purpose: str,
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise RawSupervisionPlanError(f"{purpose} must be a regular file")
    if int(metadata.st_nlink) != 1:
        raise RawSupervisionPlanError(f"{purpose} must not have a hard-link alias")
    if _entry_identity(metadata) != expected_identity:
        raise RawSupervisionPlanError(f"{purpose} identity changed during open")


def _revalidate_open_chain(
    *,
    root: Path,
    root_fd: int,
    root_fingerprint: tuple[int, int, int, int, int, int, int],
    root_directory_entries: Sequence[tuple[int, str, int, tuple[int, int]]],
    directory_entries: Sequence[tuple[int, str, int, tuple[int, int]]],
    leaf_parent_fd: int,
    leaf_name: str,
    leaf_fingerprint: tuple[int, int, int, int, int, int, int],
) -> None:
    try:
        root_metadata = root.stat(follow_symlinks=False)
        root_resolved = root.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RawSupervisionPlanError("repo_root changed during source-index read") from error
    if (
        root_resolved != root
        or root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or _file_fingerprint(root_metadata) != root_fingerprint
        or _file_fingerprint(os.fstat(root_fd)) != root_fingerprint
    ):
        raise RawSupervisionPlanError("repo_root changed during source-index read")

    for parent_fd, component, child_fd, identity in (
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
            or _entry_identity(entry) != identity
            or _entry_identity(opened) != identity
            or not stat.S_ISDIR(opened.st_mode)
        ):
            raise RawSupervisionPlanError(
                f"descriptor directory {component!r} changed during read"
            )

    leaf_entry = _lstat_at(
        leaf_parent_fd,
        leaf_name,
        purpose="rendered source index",
    )
    if (
        stat.S_ISLNK(leaf_entry.st_mode)
        or not stat.S_ISREG(leaf_entry.st_mode)
        or int(leaf_entry.st_nlink) != 1
        or _file_fingerprint(leaf_entry) != leaf_fingerprint
    ):
        raise RawSupervisionPlanError("rendered source index changed during read")


def _read_frozen_source_index(repo_root: Path) -> bytes:
    """Read the source index through a filesystem-root-anchored descriptor walk."""

    root = _v3._v2._canonical_existing_directory(Path(repo_root), name="repo_root")
    components = _relative_components(
        SOURCE_INDEX_RELATIVE_PATH,
        name="rendered source index path",
    )
    if not components:
        raise RawSupervisionPlanError("rendered source index path is empty")

    root_chain_fds: list[int] = []
    root_directory_entries: list[tuple[int, str, int, tuple[int, int]]] = []
    directory_fds: list[int] = []
    directory_entries: list[tuple[int, str, int, tuple[int, int]]] = []
    leaf_fd: int | None = None
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
            anchor_fd = os.open(filesystem_root, _directory_flags())
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise RawSupervisionPlanError(
                "filesystem root changed during descriptor open"
            ) from error
        root_chain_fds.append(anchor_fd)
        _validate_open_directory(
            os.fstat(anchor_fd),
            expected_identity=_entry_identity(anchor_before),
            expected_fingerprint=_file_fingerprint(anchor_before),
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
                (parent_fd, component, child_fd, expected_identity)
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
            expected = _entry_identity(before)
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
            opened = os.fstat(child_fd)
            _validate_open_directory(
                opened,
                expected_identity=expected,
                expected_fingerprint=expected_fingerprint,
                purpose=f"source-index directory {component!r}",
            )
            directory_entries.append((parent_fd, component, child_fd, expected))
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
        opened_fingerprint = _file_fingerprint(opened_leaf)
        if opened_fingerprint != leaf_fingerprint:
            raise RawSupervisionPlanError(
                "rendered source index fingerprint changed during descriptor open"
            )
        _revalidate_open_chain(
            root=root,
            root_fd=root_fd,
            root_fingerprint=root_fingerprint,
            root_directory_entries=root_directory_entries,
            directory_entries=directory_entries,
            leaf_parent_fd=parent_fd,
            leaf_name=leaf_name,
            leaf_fingerprint=leaf_fingerprint,
        )

        chunks: list[bytes] = []
        while True:
            chunk = os.read(leaf_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(leaf_fd)
        _validate_open_leaf(
            after,
            expected_identity=leaf_identity,
            purpose="rendered source index",
        )
        if _file_fingerprint(after) != leaf_fingerprint:
            raise RawSupervisionPlanError("rendered source index changed while read")
        _revalidate_open_chain(
            root=root,
            root_fd=root_fd,
            root_fingerprint=root_fingerprint,
            root_directory_entries=root_directory_entries,
            directory_entries=directory_entries,
            leaf_parent_fd=parent_fd,
            leaf_name=leaf_name,
            leaf_fingerprint=leaf_fingerprint,
        )
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
    root = _v3._v2._canonical_existing_directory(Path(repo_root), name="repo_root")
    return _v3.load_frozen_development_metadata(root)


def load_frozen_development_source_inventory(
    repo_root: Path,
    plan: DevelopmentRawSupervisionPlan | None = None,
) -> DevelopmentSourceInventory:
    root = _v3._v2._canonical_existing_directory(Path(repo_root), name="repo_root")
    resolved_plan = plan or load_frozen_development_metadata(root)
    rows = _v3._v2._parse_source_index(_read_frozen_source_index(root))
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
