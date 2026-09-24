#!/usr/bin/env python3
"""Authority-free, one-process V6 executor for the frozen N5 experiment.

This is the sole production operation.  It performs source preflight, the
single filesystem claim, frozen training, independent metric verification, and
finalization in one isolated process.  It has no caller-held authority or
caller-controlled production path.  Importing it opens no data or output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import multiprocessing
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v6 as policy,
)

if __name__ == "__main__":
    ROOT = SCRIPT_ROOT
    BASE_TRAINER_RELATIVE_PATH = "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
    BASE_TRAINER_FILE_SHA256 = policy.frozen_source_bindings()[
        BASE_TRAINER_RELATIVE_PATH
    ]
    STAGING_PREFIX = ".n5.reservation-v6-"
    LEGACY_STAGING_NAME = ".n5.reservation-staging"
    PREDECESSOR_STAGING_PREFIXES = (
        ".n5.reservation-v4-",
        ".n5.reservation-v3-",
        ".n5.reservation-v2-",
    )
    LOCK_NAME = ".n5.reservation-v6.lock"
    STAGING_SCHEMA = "lewm_go2_n5_full_panel_v6_preclaim_staging_v1"


    def _stable_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )


    def _identity_security_fingerprint(
        metadata: os.stat_result,
    ) -> tuple[int, ...]:
        """Identity/security fields unaffected by unrelated child churn."""

        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_gid,
        )


    def _directory_flags() -> int:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        if not nofollow or not directory:
            raise PermissionError("V6 canonical directory walks require no-follow opens")
        return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


    @dataclass
    class DirectoryEntryV6:
        parent_fd: int
        name: str
        child_fd: int
        identity_security: tuple[int, ...]
        full_fingerprint: tuple[int, ...]
        exclusive: bool


    @dataclass
    class CanonicalDirectoryChainV6:
        anchor_fd: int
        anchor_identity_security: tuple[int, ...]
        descriptors: list[int]
        entries: list[DirectoryEntryV6]
        path_fds: dict[Path, int]
        closed: bool = False


    @dataclass(frozen=True)
    class OwnedArtifactV6:
        role: str
        parent_fd: int
        name: str
        fingerprint: tuple[int, ...]
        payload_sha256: str


    def _entry_metadata(parent_fd: int, name: str) -> os.stat_result:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)


    def _assert_directory_chain(chain: CanonicalDirectoryChainV6) -> None:
        if chain.closed:
            raise PermissionError("V6 canonical directory chain is closed")
        if (
            _identity_security_fingerprint(os.fstat(chain.anchor_fd))
            != chain.anchor_identity_security
        ):
            raise PermissionError("V6 filesystem-root descriptor changed")
        for entry in chain.entries:
            named = _entry_metadata(entry.parent_fd, entry.name)
            opened = os.fstat(entry.child_fd)
            named_identity = _identity_security_fingerprint(named)
            opened_identity = _identity_security_fingerprint(opened)
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or named_identity != entry.identity_security
                or opened_identity != entry.identity_security
                or named_identity != opened_identity
                or (
                    entry.exclusive
                    and _stable_fingerprint(named) != entry.full_fingerprint
                )
                or (
                    entry.exclusive
                    and _stable_fingerprint(opened) != entry.full_fingerprint
                )
            ):
                raise PermissionError("V6 canonical directory component changed")


    def _refresh_directory_chain(
        chain: CanonicalDirectoryChainV6,
        *,
        mutable_fds: set[int],
    ) -> None:
        if chain.closed:
            raise PermissionError("V6 canonical directory chain is closed")
        if (
            _identity_security_fingerprint(os.fstat(chain.anchor_fd))
            != chain.anchor_identity_security
        ):
            raise PermissionError("V6 filesystem-root descriptor changed")
        for entry in chain.entries:
            named = _entry_metadata(entry.parent_fd, entry.name)
            opened = os.fstat(entry.child_fd)
            named_identity = _identity_security_fingerprint(named)
            opened_identity = _identity_security_fingerprint(opened)
            named_fingerprint = _stable_fingerprint(named)
            opened_fingerprint = _stable_fingerprint(opened)
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or named_identity != entry.identity_security
                or opened_identity != entry.identity_security
                or named_fingerprint != opened_fingerprint
            ):
                raise PermissionError("V6 canonical directory identity changed")
            if not entry.exclusive:
                continue
            if entry.child_fd in mutable_fds:
                entry.full_fingerprint = opened_fingerprint
            elif opened_fingerprint != entry.full_fingerprint:
                raise PermissionError("V6 unexpected canonical directory mutation")


    def _open_canonical_directory_chain(final_path: Path) -> CanonicalDirectoryChainV6:
        final_path = Path(final_path)
        if (
            not final_path.is_absolute()
            or not final_path.is_relative_to(ROOT)
            or any(part in {"", ".", ".."} for part in final_path.parts[1:])
        ):
            raise PermissionError("V6 canonical directory target escaped the repository")
        filesystem_root = Path(final_path.anchor)
        anchor_before = filesystem_root.stat(follow_symlinks=False)
        anchor_identity_security = _identity_security_fingerprint(anchor_before)
        anchor_fd = os.open(filesystem_root, _directory_flags())
        descriptors = [anchor_fd]
        entries: list[DirectoryEntryV6] = []
        path_fds = {filesystem_root: anchor_fd}
        chain = CanonicalDirectoryChainV6(
            anchor_fd=anchor_fd,
            anchor_identity_security=anchor_identity_security,
            descriptors=descriptors,
            entries=entries,
            path_fds=path_fds,
        )
        try:
            if (
                _identity_security_fingerprint(os.fstat(anchor_fd))
                != anchor_identity_security
            ):
                raise PermissionError("V6 filesystem root changed during open")
            parent_fd = anchor_fd
            current_path = filesystem_root
            repository_depth = len(ROOT.parts) - 1
            for index, component in enumerate(final_path.parts[1:]):
                created = False
                try:
                    before = _entry_metadata(parent_fd, component)
                except FileNotFoundError:
                    if index < repository_depth:
                        raise PermissionError("V6 repository root component is missing")
                    os.mkdir(component, 0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    created = True
                    before = _entry_metadata(parent_fd, component)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise PermissionError("V6 canonical component is not a real directory")
                current_path = current_path / component
                exclusive = current_path == policy.CANONICAL_OUTPUT_ROOT or (
                    current_path.is_relative_to(policy.CANONICAL_OUTPUT_ROOT)
                )
                identity_security = _identity_security_fingerprint(before)
                full_fingerprint = _stable_fingerprint(before)
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(child_fd)
                opened = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or _identity_security_fingerprint(opened) != identity_security
                    or (exclusive and _stable_fingerprint(opened) != full_fingerprint)
                ):
                    raise PermissionError("V6 canonical component changed during open")
                if created and entries:
                    _refresh_directory_chain(chain, mutable_fds={parent_fd})
                entry = DirectoryEntryV6(
                    parent_fd,
                    component,
                    child_fd,
                    identity_security,
                    full_fingerprint,
                    exclusive,
                )
                entries.append(entry)
                path_fds[current_path] = child_fd
                parent_fd = child_fd
            _assert_directory_chain(chain)
            return chain
        except BaseException:
            chain.closed = True
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            raise


    def _open_chain_child(
        chain: CanonicalDirectoryChainV6,
        parent_path: Path,
        name: str,
    ) -> int:
        _assert_directory_chain(chain)
        parent_fd = chain.path_fds[parent_path]
        try:
            before = _entry_metadata(parent_fd, name)
        except FileNotFoundError:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            _refresh_directory_chain(chain, mutable_fds={parent_fd})
            before = _entry_metadata(parent_fd, name)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise PermissionError("V6 derived parent is not a real directory")
        identity_security = _identity_security_fingerprint(before)
        full_fingerprint = _stable_fingerprint(before)
        child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
        chain.descriptors.append(child_fd)
        opened = os.fstat(child_fd)
        if (
            _identity_security_fingerprint(opened) != identity_security
            or _stable_fingerprint(opened) != full_fingerprint
        ):
            raise PermissionError("V6 derived parent changed during open")
        child_path = parent_path / name
        chain.entries.append(
            DirectoryEntryV6(
                parent_fd,
                name,
                child_fd,
                identity_security,
                full_fingerprint,
                True,
            )
        )
        chain.path_fds[child_path] = child_fd
        _assert_directory_chain(chain)
        return child_fd


    def _close_directory_chain(chain: CanonicalDirectoryChainV6) -> None:
        if chain.closed:
            return
        chain.closed = True
        for descriptor in reversed(chain.descriptors):
            os.close(descriptor)


    @dataclass(frozen=True)
    class AttemptReservationV6:
        directory: Path
        value: Mapping[str, Any]
        raw: bytes
        file_sha256: str
        directory_fd: int = -1
        directory_identity: tuple[int, int] | None = None
        directory_fingerprint: tuple[int, ...] | None = None
        directory_chain: CanonicalDirectoryChainV6 | None = None
        owned_claim_artifacts: dict[str, OwnedArtifactV6] = field(
            default_factory=dict,
            compare=False,
        )
        owned_derived_artifacts: dict[str, OwnedArtifactV6] = field(
            default_factory=dict,
            compare=False,
        )

        @property
        def binding(self) -> dict[str, Any]:
            return policy.artifact_binding(
                "reservation.json",
                self.raw,
                content_sha256=str(self.value["content_sha256"]),
            )

        @property
        def seed_root_fd(self) -> int:
            if self.directory_chain is None:
                return -1
            return self.directory_chain.path_fds.get(self.directory.parent, -1)

        @property
        def output_root_fd(self) -> int:
            if self.directory_chain is None:
                return -1
            return self.directory_chain.path_fds.get(policy.CANONICAL_OUTPUT_ROOT, -1)


    def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise


    def _replace_bytes(path: Path, payload: bytes) -> None:
        temporary = path.parent / f".{path.name}.replace-{secrets.token_hex(12)}"
        _write_bytes_exclusive(temporary, payload, mode=0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)


    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


    def _assert_claim_fd_owned(reservation: AttemptReservationV6) -> None:
        if reservation.directory_fd < 0 or reservation.directory_identity is None:
            raise PermissionError("N5 full-panel V6 lacks an open claimed directory")
        opened = os.fstat(reservation.directory_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise PermissionError("N5 full-panel V6 claimed descriptor is not a directory")
        if (opened.st_dev, opened.st_ino) != reservation.directory_identity:
            raise PermissionError("N5 full-panel V6 claimed descriptor identity changed")


    def _assert_owned_claim(reservation: AttemptReservationV6) -> None:
        """Bind success stages to the retained canonical chain and claim FD."""

        _assert_claim_fd_owned(reservation)
        opened = os.fstat(reservation.directory_fd)
        if (
            reservation.directory_fingerprint is None
            or _stable_fingerprint(opened) != reservation.directory_fingerprint
        ):
            raise PermissionError("N5 full-panel V6 claimed directory changed")
        if reservation.directory_chain is None or reservation.seed_root_fd < 0:
            raise PermissionError("N5 full-panel V6 lacks its canonical directory chain")
        _assert_directory_chain(reservation.directory_chain)
        canonical = os.stat(
            reservation.directory.name,
            dir_fd=reservation.seed_root_fd,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(canonical.st_mode)
            or not stat.S_ISDIR(canonical.st_mode)
            or (canonical.st_dev, canonical.st_ino) != reservation.directory_identity
            or _stable_fingerprint(canonical) != reservation.directory_fingerprint
        ):
            raise PermissionError("N5 full-panel V6 canonical claim identity changed")
        expected_names = {"reservation.json"} | set(
            reservation.owned_claim_artifacts
        )
        if set(os.listdir(reservation.directory_fd)) != expected_names:
            raise PermissionError("N5 full-panel V6 claimed directory inventory changed")


    def _refresh_claim_directory(reservation: AttemptReservationV6) -> None:
        """Refresh only after an exact owned claim-file mutation."""

        _assert_claim_fd_owned(reservation)
        expected_names = {"reservation.json"} | set(
            reservation.owned_claim_artifacts
        )
        if set(os.listdir(reservation.directory_fd)) != expected_names:
            raise PermissionError("N5 full-panel V6 claimed directory inventory changed")
        current = os.fstat(reservation.directory_fd)
        previous = reservation.directory_fingerprint
        if previous is None:
            raise PermissionError("N5 full-panel V6 lacks its claim fingerprint")
        if _identity_security_fingerprint(current) != (
            previous[0],
            previous[1],
            previous[2],
            previous[4],
            previous[5],
        ):
            raise PermissionError("N5 full-panel V6 claimed directory security changed")
        object.__setattr__(
            reservation,
            "directory_fingerprint",
            _stable_fingerprint(current),
        )


    def _remove_created_leaf_if_owned(
        parent_fd: int,
        name: str,
        identity: tuple[int, int] | None,
    ) -> None:
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError):
            return
        if (
            identity is not None
            and (current.st_dev, current.st_ino) == identity
            and stat.S_ISREG(current.st_mode)
            and current.st_nlink == 1
        ):
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)


    def _write_claim_file_exclusive(
        reservation: AttemptReservationV6,
        name: str,
        payload: bytes,
        *,
        mode: int = 0o600,
        require_canonical: bool = True,
        role: str | None = None,
    ) -> None:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PermissionError("N5 full-panel V6 claim filename escaped")
        if require_canonical:
            _assert_owned_claim(reservation)
        else:
            _assert_claim_fd_owned(reservation)
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
            dir_fd=reservation.directory_fd,
        )
        created_identity: tuple[int, int] | None = None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError("N5 full-panel V6 claim leaf is not singly linked")
            created_identity = (metadata.st_dev, metadata.st_ino)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            _remove_created_leaf_if_owned(
                reservation.directory_fd,
                name,
                created_identity,
            )
            raise
        try:
            committed = os.stat(
                name,
                dir_fd=reservation.directory_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(committed.st_mode)
                or committed.st_nlink != 1
                or (committed.st_dev, committed.st_ino) != created_identity
            ):
                raise PermissionError("N5 full-panel V6 committed claim leaf changed")
        except BaseException:
            _remove_created_leaf_if_owned(
                reservation.directory_fd,
                name,
                created_identity,
            )
            raise
        reservation.owned_claim_artifacts[name] = OwnedArtifactV6(
            role=role or name,
            parent_fd=reservation.directory_fd,
            name=name,
            fingerprint=_stable_fingerprint(committed),
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
        if require_canonical:
            _refresh_claim_directory(reservation)
            _assert_owned_claim(reservation)
        else:
            _assert_claim_fd_owned(reservation)


    def _read_claim_file(
        reservation: AttemptReservationV6,
        name: str,
        *,
        require_canonical: bool = True,
    ) -> bytes:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PermissionError("N5 full-panel V6 claim filename escaped")
        if require_canonical:
            _assert_owned_claim(reservation)
        else:
            _assert_claim_fd_owned(reservation)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=reservation.directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise PermissionError("N5 full-panel V6 claim leaf is not singly linked")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            fingerprint = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if fingerprint(before) != fingerprint(after):
                raise RuntimeError("N5 full-panel V6 claim leaf changed while read")
        finally:
            os.close(descriptor)
        if require_canonical:
            _assert_owned_claim(reservation)
        else:
            _assert_claim_fd_owned(reservation)
        return b"".join(chunks)


    def _set_thread_caps() -> None:
        for name in policy.THREAD_ENVIRONMENT:
            os.environ[name] = "1"


    @contextmanager
    def _locked_seed_root(seed_root: Path) -> Iterator[None]:
        seed_root.mkdir(parents=True, exist_ok=True)
        if seed_root.is_symlink() or not seed_root.is_dir():
            raise PermissionError("N5 full-panel seed root is not a real directory")
        lock_path = seed_root / LOCK_NAME
        descriptor = os.open(
            lock_path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise PermissionError("N5 full-panel reservation lock is not private")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


    def _exact_input_binding(
        review_binding: Mapping[str, str],
    ) -> dict[str, str]:
        return {
            "dataset_manifest_file_sha256": policy.DATASET_MANIFEST_FILE_SHA256,
            "dataset_manifest_content_sha256": policy.DATASET_MANIFEST_CONTENT_SHA256,
            "audit_receipt_file_sha256": policy.AUDIT_RECEIPT_FILE_SHA256,
            "audit_receipt_content_sha256": policy.AUDIT_RECEIPT_CONTENT_SHA256,
            "trainer_authorization_file_sha256": policy.TRAINER_AUTHORIZATION_FILE_SHA256,
            "trainer_authorization_content_sha256": policy.TRAINER_AUTHORIZATION_CONTENT_SHA256,
            "trainer_review_file_sha256": policy.TRAINER_REVIEW_FILE_SHA256,
            "trainer_review_content_sha256": policy.TRAINER_REVIEW_CONTENT_SHA256,
            "rgb_receipt_content_sha256": policy.RGB_RECEIPT_CONTENT_SHA256,
            "subset_content_sha256": policy.SUBSET_CONTENT_SHA256,
            "target_partition_content_sha256": policy.TARGET_PARTITION_CONTENT_SHA256,
            "source_review_file_sha256": review_binding["file_sha256"],
            "source_review_content_sha256": review_binding["content_sha256"],
            "terminal_invalidation_file_sha256": policy.TERMINAL_INVALIDATION_FILE_SHA256,
            "terminal_invalidation_content_sha256": (
                policy.TERMINAL_INVALIDATION_CONTENT_SHA256
            ),
        }


    def _reservation_core(
        review_binding: Mapping[str, str],
        *,
        recovery_events: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "schema": policy.RESERVATION_SCHEMA,
            "status": "reserved",
            "attempt_index": 1,
            "maximum_attempts": 1,
            "scope": "one_exclusive_fresh_infrastructure_replacement_attempt",
            "seed": 20260710,
            "fit_size": 5,
            "experiment": policy.experiment_contract(),
            "authority_bindings": policy.authority_bindings(),
            "source_review": dict(review_binding),
            "inputs": _exact_input_binding(review_binding),
            "preclaim_recovery": [dict(item) for item in recovery_events],
            "licenses": {
                "development_checkpoint_creation_authorized": True,
                "checkpoint_use_authorized": False,
                "retry_authorized": False,
                "n16_execution_authorized": False,
                "second_seed_authorized": False,
                "holdout_authorized": False,
                "g2_authorized": False,
                "runtime_authorized": False,
                "promotion_authorized": False,
            },
        }


    def _reservation(
        review_binding: Mapping[str, str],
        *,
        attempt_path: Path,
        recovery_events: Sequence[Mapping[str, Any]],
    ) -> AttemptReservationV6:
        core = _reservation_core(
            review_binding,
            recovery_events=recovery_events,
        )
        value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
        raw = policy.canonical_json_bytes(value) + b"\n"
        return AttemptReservationV6(
            directory=attempt_path,
            value=value,
            raw=raw,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        )


    def _manifest_value(
        *,
        staging: Path,
        attempt_path: Path,
        reservation: AttemptReservationV6,
    ) -> dict[str, Any]:
        core = {
            "schema": STAGING_SCHEMA,
            "status": "complete_and_resume_eligible",
            "staging_name": staging.name,
            "attempt_path": str(attempt_path.resolve()),
            "reservation_file_sha256": reservation.file_sha256,
            "reservation_content_sha256": reservation.value["content_sha256"],
            "private_mode": "0700",
            "recovery_policy": {
                "incomplete": "remove_then_create_unique_staging",
                "complete": "rehash_then_resume_single_staging",
                "foreign": "inventory_then_remove_without_claim",
                "mutated": "inventory_then_remove_without_claim",
                "multiple_complete": "resume_lexical_first_remove_exact_duplicates",
            },
        }
        return {**core, "content_sha256": policy.canonical_json_sha256(core)}


    def _staging_inventory_digest(path: Path) -> tuple[list[str], str]:
        if path.is_symlink() or not path.is_dir():
            metadata = os.lstat(path)
            inventory = [f"non_directory:{stat.S_IFMT(metadata.st_mode)}:{metadata.st_size}"]
            return inventory, policy.canonical_json_sha256(inventory)
        inventory: list[str] = []
        try:
            children = sorted(path.iterdir(), key=lambda item: item.name)
        except OSError as error:
            metadata = os.lstat(path)
            inventory = [
                "unreadable_directory:"
                f"{stat.S_IMODE(metadata.st_mode):04o}:{metadata.st_uid}:"
                f"{type(error).__name__}"
            ]
            return inventory, policy.canonical_json_sha256(inventory)
        for child in children:
            metadata = child.lstat()
            if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= 1024 * 1024:
                raw = policy.read_regular_bytes(child, name=f"staging evidence {child.name}")
                inventory.append(
                    f"file:{child.name}:{len(raw)}:{hashlib.sha256(raw).hexdigest()}"
                )
            elif stat.S_ISDIR(metadata.st_mode):
                inventory.append(f"directory:{child.name}")
            elif stat.S_ISLNK(metadata.st_mode):
                inventory.append(f"symlink:{child.name}")
            else:
                inventory.append(f"other:{child.name}:{metadata.st_size}")
        return inventory, policy.canonical_json_sha256(inventory)


    def _validate_recoverable_reservation(
        value: Mapping[str, Any],
        *,
        review_binding: Mapping[str, str],
    ) -> bool:
        core = dict(value)
        declared = core.pop("content_sha256", None)
        if not policy.is_sha256(declared) or policy.canonical_json_sha256(core) != declared:
            return False
        recovery = core.pop("preclaim_recovery", None)
        if not isinstance(recovery, list) or any(
            not isinstance(item, Mapping) for item in recovery
        ):
            return False
        expected = _reservation_core(
            review_binding,
            recovery_events=(),
        )
        expected.pop("preclaim_recovery")
        return core == expected


    def _classify_staging(
        staging: Path,
        *,
        attempt_path: Path,
        review_binding: Mapping[str, str],
    ) -> tuple[dict[str, Any], AttemptReservationV6 | None]:
        inventory, inventory_sha = _staging_inventory_digest(staging)
        evidence = {
            "staging_name": staging.name,
            "inventory_sha256": inventory_sha,
            "classification": "foreign",
            "action": "remove_without_claim",
        }
        metadata = os.lstat(staging)
        if (
            staging.name == LEGACY_STAGING_NAME
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            if staging.name == LEGACY_STAGING_NAME:
                evidence["classification"] = "incomplete_legacy_v1"
            return evidence, None
        names = sorted(child.name for child in staging.iterdir())
        if names not in (
            ["reservation.json", "staging.json"],
            ["claim.json", "reservation.json"],
        ):
            evidence["classification"] = "incomplete" if len(names) < 2 else "mutated"
            return evidence, None
        manifest_name = "staging.json" if "staging.json" in names else "claim.json"
        try:
            reservation_value, reservation_raw = policy.load_hashed_json(
                staging / "reservation.json",
                hashlib.sha256(
                    policy.read_regular_bytes(
                        staging / "reservation.json",
                        name="recoverable reservation",
                    )
                ).hexdigest(),
                name="recoverable reservation",
            )
            manifest_raw = policy.read_regular_bytes(
                staging / manifest_name,
                name="recoverable staging manifest",
            )
            manifest = policy.parse_json(manifest_raw, name="recoverable staging manifest")
        except (OSError, ValueError, PermissionError, UnicodeError):
            evidence["classification"] = "mutated"
            return evidence, None
        manifest_core = dict(manifest)
        manifest_content = manifest_core.pop("content_sha256", None)
        expected_manifest_fields = {
            "schema",
            "status",
            "staging_name",
            "attempt_path",
            "reservation_file_sha256",
            "reservation_content_sha256",
            "private_mode",
            "recovery_policy",
        }
        valid = (
            set(manifest_core) == expected_manifest_fields
            and manifest.get("schema") == STAGING_SCHEMA
            and manifest.get("status") == "complete_and_resume_eligible"
            and manifest.get("staging_name") == staging.name
            and manifest.get("attempt_path") == str(attempt_path.resolve())
            and manifest.get("reservation_file_sha256")
            == hashlib.sha256(reservation_raw).hexdigest()
            and manifest.get("reservation_content_sha256")
            == reservation_value.get("content_sha256")
            and manifest.get("private_mode") == "0700"
            and policy.is_sha256(manifest_content)
            and policy.canonical_json_sha256(manifest_core) == manifest_content
            and _validate_recoverable_reservation(
                reservation_value,
                review_binding=review_binding,
            )
        )
        if not valid:
            evidence["classification"] = "mutated"
            return evidence, None
        evidence["classification"] = "complete"
        evidence["action"] = "resume_after_rehash"
        return evidence, AttemptReservationV6(
            directory=attempt_path,
            value=reservation_value,
            raw=reservation_raw,
            file_sha256=hashlib.sha256(reservation_raw).hexdigest(),
        )


    def _remove_staging(staging: Path, *, seed_root: Path) -> None:
        if staging.parent.resolve() != seed_root.resolve() or not (
            staging.name == LEGACY_STAGING_NAME
            or staging.name.startswith(STAGING_PREFIX)
            or any(
                staging.name.startswith(prefix)
                for prefix in PREDECESSOR_STAGING_PREFIXES
            )
        ):
            raise PermissionError("staging cleanup escaped its reviewed namespace")
        metadata = os.lstat(staging)
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_uid != os.getuid():
                raise PermissionError("foreign staging is not owned by the current user")
            os.chmod(staging, 0o700, follow_symlinks=False)
            shutil.rmtree(staging)
        else:
            staging.unlink()
        _fsync_directory(seed_root)


    def _new_staging(seed_root: Path) -> Path:
        for _ in range(128):
            staging = seed_root / f"{STAGING_PREFIX}{secrets.token_hex(16)}"
            try:
                os.mkdir(staging, 0o700)
                os.chmod(staging, 0o700, follow_symlinks=False)
                _fsync_directory(seed_root)
                return staging
            except FileExistsError:
                continue
        raise FileExistsError("could not allocate unique private reservation staging")


    def _prepare_new_staging(
        staging: Path,
        *,
        reservation: AttemptReservationV6,
        attempt_path: Path,
    ) -> None:
        _write_bytes_exclusive(staging / "reservation.json", reservation.raw, mode=0o600)
        manifest = _manifest_value(
            staging=staging,
            attempt_path=attempt_path,
            reservation=reservation,
        )
        _write_bytes_exclusive(
            staging / "staging.json",
            policy.canonical_json_bytes(manifest) + b"\n",
            mode=0o600,
        )
        _fsync_directory(staging)


    def _refresh_complete_staging(
        staging: Path,
        *,
        reservation: AttemptReservationV6,
        attempt_path: Path,
        recovery_events: Sequence[Mapping[str, Any]],
        review_binding: Mapping[str, str],
    ) -> AttemptReservationV6:
        updated = _reservation(
            review_binding,
            attempt_path=attempt_path,
            recovery_events=recovery_events,
        )
        _replace_bytes(staging / "reservation.json", updated.raw)
        manifest = _manifest_value(
            staging=staging,
            attempt_path=attempt_path,
            reservation=updated,
        )
        manifest_raw = policy.canonical_json_bytes(manifest) + b"\n"
        manifest_path = (
            staging / "staging.json"
            if (staging / "staging.json").exists()
            else staging / "claim.json"
        )
        _replace_bytes(manifest_path, manifest_raw)
        return updated


    def _reserve_exact_attempt(source_review_file_sha256: str) -> AttemptReservationV6:
        """Claim the one canonical attempt; no caller-controlled path exists."""

        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        review_binding = policy.source_review_binding(review, source_review_file_sha256)
        attempt_path = policy.CANONICAL_ATTEMPT_PATH
        seed_root = attempt_path.parent
        active_staging: Path | None = None
        owned_claim_identity: tuple[int, int] | None = None
        claimed_directory_fd: int | None = None
        claimed_chain: CanonicalDirectoryChainV6 | None = None
        claimed_reservation: AttemptReservationV6 | None = None
        try:
            with _locked_seed_root(seed_root):
                if attempt_path.exists() or attempt_path.is_symlink():
                    raise FileExistsError("the sole N5 full-panel attempt is already claimed")
                candidates = sorted(
                    (
                        child
                        for child in seed_root.iterdir()
                        if child.name == LEGACY_STAGING_NAME
                        or child.name.startswith(STAGING_PREFIX)
                        or any(
                            child.name.startswith(prefix)
                            for prefix in PREDECESSOR_STAGING_PREFIXES
                        )
                    ),
                    key=lambda child: child.name,
                )
                recovery_events: list[dict[str, Any]] = []
                complete: list[tuple[Path, AttemptReservationV6, dict[str, Any]]] = []
                for candidate in candidates:
                    evidence, recovered = _classify_staging(
                        candidate,
                        attempt_path=attempt_path,
                        review_binding=review_binding,
                    )
                    if recovered is None:
                        _remove_staging(candidate, seed_root=seed_root)
                        recovery_events.append(evidence)
                    else:
                        complete.append((candidate, recovered, evidence))
                if complete:
                    active_staging, reservation, selected = complete[0]
                    recovery_events.append(selected)
                    for duplicate, duplicate_reservation, evidence in complete[1:]:
                        _remove_staging(duplicate, seed_root=seed_root)
                        recovery_events.append(
                            {
                                **evidence,
                                "classification": "complete_equivalent_duplicate",
                                "action": "remove_equivalent_duplicate_without_claim",
                                "reservation_file_sha256": (
                                    duplicate_reservation.file_sha256
                                ),
                            }
                        )
                    reservation = _refresh_complete_staging(
                        active_staging,
                        reservation=reservation,
                        attempt_path=attempt_path,
                        recovery_events=recovery_events,
                        review_binding=review_binding,
                    )
                else:
                    active_staging = _new_staging(seed_root)
                    recovery_events.append(
                        {
                            "staging_name": active_staging.name,
                            "classification": "new_unique_private",
                            "action": "complete_then_atomic_claim",
                            "inventory_sha256": policy.canonical_json_sha256([]),
                        }
                    )
                    reservation = _reservation(
                        review_binding,
                        attempt_path=attempt_path,
                        recovery_events=recovery_events,
                    )
                    _prepare_new_staging(
                        active_staging,
                        reservation=reservation,
                        attempt_path=attempt_path,
                    )
                manifest_path = (
                    active_staging / "staging.json"
                    if (active_staging / "staging.json").exists()
                    else active_staging / "claim.json"
                )
                manifest_path.unlink()
                _fsync_directory(active_staging)
                claimed_chain = _open_canonical_directory_chain(seed_root)
                _open_chain_child(
                    claimed_chain,
                    policy.CANONICAL_OUTPUT_ROOT,
                    policy.CANONICAL_METRIC_RECEIPT_PATH.parent.name,
                )
                _open_chain_child(
                    claimed_chain,
                    policy.CANONICAL_OUTPUT_ROOT,
                    policy.CANONICAL_GATE_PATH.parent.name,
                )
                seed_root_fd = claimed_chain.path_fds[seed_root]
                claimed_directory_fd = os.open(
                    active_staging.name,
                    _directory_flags(),
                    dir_fd=seed_root_fd,
                )
                staging_metadata = os.fstat(claimed_directory_fd)
                named_staging_metadata = os.stat(
                    active_staging.name,
                    dir_fd=seed_root_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(staging_metadata.st_mode)
                    or (staging_metadata.st_dev, staging_metadata.st_ino)
                    != (named_staging_metadata.st_dev, named_staging_metadata.st_ino)
                ):
                    raise PermissionError("N5 full-panel V6 staging identity changed")
                owned_claim_identity = (staging_metadata.st_dev, staging_metadata.st_ino)
                claimed_reservation = AttemptReservationV6(
                    directory=attempt_path,
                    value=reservation.value,
                    raw=reservation.raw,
                    file_sha256=reservation.file_sha256,
                    directory_fd=claimed_directory_fd,
                    directory_identity=owned_claim_identity,
                    directory_fingerprint=_stable_fingerprint(staging_metadata),
                    directory_chain=claimed_chain,
                )
                os.rename(
                    active_staging.name,
                    attempt_path.name,
                    src_dir_fd=seed_root_fd,
                    dst_dir_fd=seed_root_fd,
                )
                os.fsync(seed_root_fd)
                active_staging = None
                _refresh_directory_chain(
                    claimed_chain,
                    mutable_fds={seed_root_fd},
                )
                _refresh_claim_directory(claimed_reservation)
                _assert_owned_claim(claimed_reservation)
                return claimed_reservation
        except BaseException as error:
            secondary_error: BaseException | None = None
            try:
                attempt_metadata = None
                if claimed_chain is not None and seed_root in claimed_chain.path_fds:
                    try:
                        attempt_metadata = os.stat(
                            attempt_path.name,
                            dir_fd=claimed_chain.path_fds[seed_root],
                            follow_symlinks=False,
                        )
                    except (FileNotFoundError, NotADirectoryError):
                        pass
                owns_canonical_claim = (
                    owned_claim_identity is not None
                    and attempt_metadata is not None
                    and (attempt_metadata.st_dev, attempt_metadata.st_ino)
                    == owned_claim_identity
                )
                if owns_canonical_claim and claimed_reservation is not None:
                    reservation_raw = _read_claim_file(
                        claimed_reservation,
                        "reservation.json",
                        require_canonical=False,
                    )
                    reservation_value = policy.parse_json(
                        reservation_raw,
                        name="claimed reservation during failure",
                    )
                    if (
                        reservation_value != claimed_reservation.value
                        or hashlib.sha256(reservation_raw).hexdigest()
                        != claimed_reservation.file_sha256
                    ):
                        raise RuntimeError("claimed reservation changed after rename")
                    _terminate_failure(
                        claimed_reservation,
                        error,
                        stage="reservation_claim",
                    )
                elif active_staging is not None and active_staging.exists():
                    _remove_staging(active_staging, seed_root=seed_root)
            except BaseException as cleanup_error:
                secondary_error = cleanup_error
            finally:
                if claimed_directory_fd is not None:
                    os.close(claimed_directory_fd)
                if claimed_chain is not None:
                    _close_directory_chain(claimed_chain)
            if secondary_error is not None:
                raise RuntimeError(
                    "reservation failure cleanup or terminalization failed"
                ) from secondary_error
            raise


    def decode_selected_rgb(
        frames: Sequence[Any],
        *,
        source_review_file_sha256: str,
        maximum_workers: int,
    ) -> tuple[Any, dict[str, Any]]:
        policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        worker_count = int(maximum_workers)
        if isinstance(maximum_workers, bool) or not 1 <= worker_count <= 5:
            raise ValueError("N5 full-panel RGB workers must lie in [1,5]")
        _set_thread_caps()
        jobs = [
            (
                str(frame.rgb_path),
                str(frame.image_sha256),
                str(ROOT),
                BASE_TRAINER_FILE_SHA256,
            )
            for frame in frames
        ]
        if len(jobs) != 5:
            raise ValueError("N5 full-panel decode requires exactly five selected RGBs")
        from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

        if worker_count == 1:
            arrays = [base._decode_rgb_job(*job) for job in jobs]
            start_method = "inline_source_revalidated"
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(worker_count, len(jobs)),
                mp_context=context,
            ) as executor:
                arrays = list(
                    executor.map(
                        base._decode_rgb_job,
                        (job[0] for job in jobs),
                        (job[1] for job in jobs),
                        (job[2] for job in jobs),
                        (job[3] for job in jobs),
                    )
                )
            start_method = "spawn"
        import numpy as np
        import torch

        images = torch.from_numpy(np.stack(arrays, axis=0).copy())
        return images, {
            "selected_rgb_count": 5,
            "nonselected_rgb_opens": 0,
            "rgb_hash_opens": 5,
            "rgb_decodes": 5,
            "worker_start_method": start_method,
            "worker_count": min(worker_count, 5),
            "native_threads_per_worker": 1,
        }


    def _failure_code(error: BaseException) -> dict[str, str]:
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


    def _cleanup_owned_artifacts(
        registry: dict[str, OwnedArtifactV6],
        *,
        selected: set[str] | None = None,
    ) -> list[dict[str, str]]:
        outcomes: list[dict[str, str]] = []
        for key, artifact in tuple(registry.items()):
            if selected is not None and key not in selected:
                continue
            try:
                current = os.stat(
                    artifact.name,
                    dir_fd=artifact.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                registry.pop(key, None)
                outcomes.append(
                    {"artifact": key, "role": artifact.role, "outcome": "already_absent"}
                )
                continue
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or _stable_fingerprint(current) != artifact.fingerprint
            ):
                outcomes.append(
                    {
                        "artifact": key,
                        "role": artifact.role,
                        "outcome": "ownership_mismatch_preserved_invalid",
                    }
                )
                continue
            os.unlink(artifact.name, dir_fd=artifact.parent_fd)
            os.fsync(artifact.parent_fd)
            registry.pop(key, None)
            outcomes.append(
                {"artifact": key, "role": artifact.role, "outcome": "removed_owned"}
            )
        return outcomes


    def _terminate_failure(
        reservation: AttemptReservationV6,
        error: BaseException,
        *,
        stage: str = "training",
    ) -> dict[str, Any]:
        _assert_claim_fd_owned(reservation)
        cleanup = _cleanup_owned_artifacts(reservation.owned_derived_artifacts)
        cleanup.extend(
            _cleanup_owned_artifacts(
                reservation.owned_claim_artifacts,
                selected={"checkpoint.pt", "result.json", "completed.json"},
            )
        )
        os.fsync(reservation.directory_fd)
        core = {
            "schema": policy.FAILURE_SCHEMA,
            "status": "failed",
            "reservation": reservation.binding,
            "failure_stage": stage,
            "failure": _failure_code(error),
            "artifact_cleanup": cleanup,
            "partial_artifacts_removed": all(
                item["outcome"] in {"removed_owned", "already_absent"}
                for item in cleanup
            ),
            "retry_authorized": False,
            "licenses": {
                "checkpoint_use_authorized": False,
                "n16_execution_authorized": False,
                "second_seed_authorized": False,
                "holdout_authorized": False,
                "g2_authorized": False,
                "runtime_authorized": False,
                "promotion_authorized": False,
            },
        }
        value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
        raw = policy.canonical_json_bytes(value) + b"\n"
        _write_claim_file_exclusive(
            reservation,
            "failed.json",
            raw,
            require_canonical=False,
            role="terminal_failure",
        )
        os.fsync(reservation.directory_fd)
        if reservation.seed_root_fd >= 0:
            os.fsync(reservation.seed_root_fd)
        _assert_claim_fd_owned(reservation)
        return policy.artifact_binding(
            "failed.json",
            raw,
            content_sha256=value["content_sha256"],
        )


    def _publish_success(
        reservation: AttemptReservationV6,
        *,
        checkpoint_raw: bytes,
        checkpoint_content_sha256: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        _assert_owned_claim(reservation)
        if not policy.is_sha256(checkpoint_content_sha256):
            raise ValueError("N5 full-panel V6 checkpoint content hash is malformed")
        policy.validate_result_structure(
            result,
            expected_source_review=reservation.value["source_review"],
        )
        checkpoint_binding = policy.artifact_binding(
            "checkpoint.pt",
            checkpoint_raw,
            content_sha256=checkpoint_content_sha256,
        )
        result_raw = policy.canonical_json_bytes(result) + b"\n"
        result_binding = policy.artifact_binding(
            "result.json",
            result_raw,
            content_sha256=str(result["content_sha256"]),
        )
        completion_core = {
            "schema": policy.COMPLETION_SCHEMA,
            "status": "completed",
            "reservation": reservation.binding,
            "checkpoint": checkpoint_binding,
            "result": result_binding,
            "inventory": [
                "checkpoint.pt",
                "completed.json",
                "reservation.json",
                "result.json",
            ],
            "retry_authorized": False,
            "licenses": {
                "checkpoint_use_authorized": False,
                "metric_verification_only_checkpoint_use_authorized": True,
                "n16_execution_authorized": False,
                "second_seed_authorized": False,
                "holdout_authorized": False,
                "g2_authorized": False,
                "runtime_authorized": False,
                "promotion_authorized": False,
            },
        }
        completion = {
            **completion_core,
            "content_sha256": policy.canonical_json_sha256(completion_core),
        }
        completion_raw = policy.canonical_json_bytes(completion) + b"\n"
        _write_claim_file_exclusive(
            reservation,
            "checkpoint.pt",
            checkpoint_raw,
            role="training_checkpoint",
        )
        _write_claim_file_exclusive(
            reservation,
            "result.json",
            result_raw,
            role="training_result",
        )
        _write_claim_file_exclusive(
            reservation,
            "completed.json",
            completion_raw,
            role="training_completion",
        )
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.seed_root_fd)
        _assert_owned_claim(reservation)
        return {
            "attempt_path": str(reservation.directory),
            "reservation": reservation.binding,
            "checkpoint": checkpoint_binding,
            "result": result_binding,
            "completion": policy.artifact_binding(
                "completed.json",
                completion_raw,
                content_sha256=completion["content_sha256"],
            ),
        }


    def _write_canonical_json(
        reservation: AttemptReservationV6,
        path: Path,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        _assert_owned_claim(reservation)
        path = Path(path)
        if path not in {policy.CANONICAL_METRIC_RECEIPT_PATH, policy.CANONICAL_GATE_PATH}:
            raise PermissionError("N5 full-panel V6 publication path is not canonical")
        expected_schema = (
            policy.METRIC_RECEIPT_SCHEMA
            if path == policy.CANONICAL_METRIC_RECEIPT_PATH
            else policy.GATE_SCHEMA
        )
        core = dict(value)
        declared = core.pop("content_sha256", None)
        if (
            value.get("schema") != expected_schema
            or not policy.is_sha256(declared)
            or policy.canonical_json_sha256(core) != declared
        ):
            raise ValueError("N5 full-panel V6 derived artifact is malformed")
        raw = policy.canonical_json_bytes(value) + b"\n"
        relative = path.relative_to(policy.CANONICAL_OUTPUT_ROOT)
        if len(relative.parts) != 2:
            raise PermissionError("N5 full-panel V6 publication depth changed")
        if reservation.directory_chain is None or reservation.output_root_fd < 0:
            raise PermissionError("N5 full-panel V6 lacks its retained output root")
        parent_path = policy.CANONICAL_OUTPUT_ROOT / relative.parts[0]
        parent_fd = reservation.directory_chain.path_fds.get(parent_path, -1)
        if parent_fd < 0:
            raise PermissionError("N5 full-panel V6 lacks its derived parent descriptor")
        leaf_fd = -1
        created_identity: tuple[int, int] | None = None
        try:
            leaf_fd = os.open(
                relative.parts[1],
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            metadata = os.fstat(leaf_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError("N5 full-panel V6 derived leaf is not singly linked")
            created_identity = (metadata.st_dev, metadata.st_ino)
            with os.fdopen(leaf_fd, "wb", closefd=True) as stream:
                leaf_fd = -1
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(parent_fd)
            os.fsync(reservation.output_root_fd)
        except BaseException:
            if leaf_fd >= 0:
                os.close(leaf_fd)
            _remove_created_leaf_if_owned(
                parent_fd,
                relative.parts[1],
                created_identity,
            )
            raise
        try:
            committed = os.stat(
                relative.parts[1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(committed.st_mode)
                or committed.st_nlink != 1
                or (committed.st_dev, committed.st_ino) != created_identity
            ):
                raise PermissionError("N5 full-panel V6 committed derived leaf changed")
        except BaseException:
            _remove_created_leaf_if_owned(
                parent_fd,
                relative.parts[1],
                created_identity,
            )
            raise
        relative_text = str(relative)
        reservation.owned_derived_artifacts[relative_text] = OwnedArtifactV6(
            role=("metric_receipt" if path == policy.CANONICAL_METRIC_RECEIPT_PATH else "gate"),
            parent_fd=parent_fd,
            name=relative.parts[1],
            fingerprint=_stable_fingerprint(committed),
            payload_sha256=hashlib.sha256(raw).hexdigest(),
        )
        _refresh_directory_chain(
            reservation.directory_chain,
            mutable_fds={parent_fd},
        )
        _assert_owned_claim(reservation)
        return policy.artifact_binding(
            relative_text,
            raw,
            content_sha256=str(value["content_sha256"]),
        )


    def _run_frozen_training(
        source_review_file_sha256: str,
        *,
        rgb_workers: int,
    ) -> tuple[dict[str, Any], AttemptReservationV6]:
        """Run frozen V1 numerical science behind an ephemeral local adapter."""

        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        review_binding = policy.source_review_binding(review, source_review_file_sha256)
        token = object()
        claimed_reservation: AttemptReservationV6 | None = None

        def require(value: object) -> object:
            if value is not token:
                raise PermissionError("retained science lacks its local V6 execution token")
            policy.preflight_static_authority()
            policy.preflight_source_review(
                policy.CANONICAL_SOURCE_REVIEW_PATH,
                source_review_file_sha256,
            )
            return value

        def source_binding(value: object) -> dict[str, str]:
            require(value)
            return dict(review_binding)

        def reserve(value: object) -> AttemptReservationV6:
            nonlocal claimed_reservation
            require(value)
            if claimed_reservation is not None:
                raise FileExistsError("the sole V6 attempt was already reserved")
            claimed_reservation = _reserve_exact_attempt(source_review_file_sha256)
            return claimed_reservation

        def decode(
            frames: Sequence[Any],
            *,
            authority: object,
            maximum_workers: int,
        ) -> tuple[Any, dict[str, Any]]:
            require(authority)
            return decode_selected_rgb(
                frames,
                source_review_file_sha256=source_review_file_sha256,
                maximum_workers=maximum_workers,
            )

        compatibility = SimpleNamespace(
            **{
                name: getattr(policy, name)
                for name in dir(policy)
                if not name.startswith("__")
            }
        )
        compatibility.EXPERIMENT = policy.experiment_contract()
        compatibility.AUTHORITY_BINDINGS = policy.authority_bindings()
        compatibility.LICENSES = policy.licenses()
        compatibility.FROZEN_SOURCE_BINDINGS = policy.frozen_source_bindings()
        compatibility.LOSS_WEIGHTS = {name: 0.25 for name in policy.LOSS_COMPONENTS}
        compatibility.require_verified_authority = require
        compatibility.source_review_binding = source_binding

        from scripts import train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained

        original = {
            "policy": retained.policy,
            "reserve": retained._reserve_attempt,
            "terminate": retained._terminate_failure,
            "publish": retained._publish_success,
            "decode": retained.decode_selected_rgb,
        }
        training_error: BaseException | None = None
        try:
            retained.policy = compatibility
            retained._reserve_attempt = reserve
            retained._terminate_failure = _terminate_failure
            retained._publish_success = _publish_success
            retained.decode_selected_rgb = decode
            summary = retained._run_training(token, rgb_workers=rgb_workers)
        except BaseException as error:
            training_error = error
            raise
        finally:
            retained.policy = original["policy"]
            retained._reserve_attempt = original["reserve"]
            retained._terminate_failure = original["terminate"]
            retained._publish_success = original["publish"]
            retained.decode_selected_rgb = original["decode"]
            if training_error is not None and claimed_reservation is not None:
                os.close(claimed_reservation.directory_fd)
                if claimed_reservation.directory_chain is not None:
                    _close_directory_chain(claimed_reservation.directory_chain)
        if claimed_reservation is None:
            raise RuntimeError("frozen V6 training returned without claiming its attempt")
        _assert_owned_claim(claimed_reservation)
        return {
            **summary,
            "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_launch_summary_v1",
        }, claimed_reservation


    def _artifact_args(
        reservation: AttemptReservationV6,
        source_review_file_sha256: str,
    ) -> argparse.Namespace:
        attempt = policy.CANONICAL_ATTEMPT_PATH
        bindings: dict[str, str] = {}
        for name in ("reservation.json", "result.json", "checkpoint.pt", "completed.json"):
            raw = _read_claim_file(reservation, name)
            bindings[name] = f"{attempt / name}:{hashlib.sha256(raw).hexdigest()}"
        return argparse.Namespace(
            source_review=policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_sha256=source_review_file_sha256,
            reservation=bindings["reservation.json"],
            result=bindings["result.json"],
            checkpoint=bindings["checkpoint.pt"],
            completion=bindings["completed.json"],
        )


    def _run_independent_verification(
        reservation: AttemptReservationV6,
        source_review_file_sha256: str,
    ) -> dict[str, Any]:
        _assert_owned_claim(reservation)
        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        review_binding = policy.source_review_binding(review, source_review_file_sha256)
        token = object()

        def require(value: object) -> object:
            if value is not token:
                raise PermissionError("retained verifier lacks its local V6 execution token")
            policy.preflight_static_authority()
            policy.preflight_source_review(
                policy.CANONICAL_SOURCE_REVIEW_PATH,
                source_review_file_sha256,
            )
            return value

        def source_binding(value: object) -> dict[str, str]:
            require(value)
            return dict(review_binding)

        def write_metric(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
            if Path(path) != policy.CANONICAL_METRIC_RECEIPT_PATH:
                raise PermissionError("retained verifier changed its V6 output path")
            return _write_canonical_json(reservation, Path(path), value)

        compatibility = SimpleNamespace(
            **{
                name: getattr(policy, name)
                for name in dir(policy)
                if not name.startswith("__")
            }
        )
        compatibility.EXPERIMENT = policy.experiment_contract()
        compatibility.AUTHORITY_BINDINGS = policy.authority_bindings()
        compatibility.LICENSES = policy.licenses()
        compatibility.FROZEN_SOURCE_BINDINGS = policy.frozen_source_bindings()
        compatibility.LOSS_WEIGHTS = {name: 0.25 for name in policy.LOSS_COMPONENTS}
        compatibility.require_verified_authority = require
        compatibility.source_review_binding = source_binding
        compatibility.write_exclusive = write_metric

        from scripts import verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as verifier

        original_policy = verifier.policy
        try:
            verifier.policy = compatibility
            args = _artifact_args(reservation, source_review_file_sha256)
            bundle = verifier._validate_attempt_bundle(token, args)
            receipt = verifier._compute_receipt(token, bundle)
            write_metric(policy.CANONICAL_METRIC_RECEIPT_PATH, receipt)
        finally:
            verifier.policy = original_policy
        _assert_owned_claim(reservation)
        return receipt


    def _run_finalization(
        reservation: AttemptReservationV6,
        source_review_file_sha256: str,
    ) -> dict[str, Any]:
        _assert_owned_claim(reservation)
        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        review_binding = policy.source_review_binding(review, source_review_file_sha256)
        token = object()

        def require(value: object) -> object:
            if value is not token:
                raise PermissionError("retained finalizer lacks its local V6 execution token")
            policy.preflight_static_authority()
            policy.preflight_source_review(
                policy.CANONICAL_SOURCE_REVIEW_PATH,
                source_review_file_sha256,
            )
            return value

        def verify(
            source_review_path: Path,
            supplied_sha256: str,
            *,
            require_unclaimed_output: bool = False,
        ) -> object:
            del require_unclaimed_output
            if Path(source_review_path) != policy.CANONICAL_SOURCE_REVIEW_PATH or supplied_sha256 != source_review_file_sha256:
                raise PermissionError("retained finalizer source review binding changed")
            require(token)
            return token

        def source_binding(value: object) -> dict[str, str]:
            require(value)
            return dict(review_binding)

        def write_gate(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
            if Path(path) != policy.CANONICAL_GATE_PATH:
                raise PermissionError("retained finalizer changed its V6 output path")
            return _write_canonical_json(reservation, Path(path), value)

        compatibility = SimpleNamespace(
            **{
                name: getattr(policy, name)
                for name in dir(policy)
                if not name.startswith("__")
            }
        )
        compatibility.EXPERIMENT = policy.experiment_contract()
        compatibility.AUTHORITY_BINDINGS = policy.authority_bindings()
        compatibility.LICENSES = policy.licenses()
        compatibility.FROZEN_SOURCE_BINDINGS = policy.frozen_source_bindings()
        compatibility.LOSS_WEIGHTS = {name: 0.25 for name in policy.LOSS_COMPONENTS}
        compatibility.require_verified_authority = require
        compatibility.verify_authority = verify
        compatibility.source_review_binding = source_binding
        compatibility.write_exclusive = write_gate

        from scripts import finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as finalizer
        from scripts import verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as verifier

        original_finalizer_policy = finalizer.policy
        original_verifier_policy = verifier.policy
        try:
            finalizer.policy = compatibility
            verifier.policy = compatibility
            args = _artifact_args(reservation, source_review_file_sha256)
            metric_raw = policy.read_regular_bytes(
                policy.CANONICAL_METRIC_RECEIPT_PATH,
                name="V6 metric verification",
            )
            args.metric_verification = (
                f"{policy.CANONICAL_METRIC_RECEIPT_PATH}:"
                f"{hashlib.sha256(metric_raw).hexdigest()}"
            )
            gate = finalizer.run(args)
        finally:
            finalizer.policy = original_finalizer_policy
            verifier.policy = original_verifier_policy
        _assert_owned_claim(reservation)
        return gate


    def execute_exact(
        source_review_file_sha256: str,
        *,
        rgb_workers: int,
    ) -> dict[str, Any]:
        """Own the complete canonical lifecycle; no stage authority escapes."""

        if not sys.flags.isolated:
            raise PermissionError("N5 full-panel V6 exact execution requires isolation")
        if isinstance(rgb_workers, bool) or not 1 <= int(rgb_workers) <= 5:
            raise ValueError("N5 full-panel RGB workers must lie in [1,5]")
        if (
            policy.CANONICAL_ATTEMPT_PATH.exists()
            or policy.CANONICAL_ATTEMPT_PATH.is_symlink()
        ):
            raise FileExistsError(
                "the sole N5 full-panel V6 recovery attempt is already claimed"
            )
        policy.preflight_static_authority()
        policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        training, reservation = _run_frozen_training(
            source_review_file_sha256,
            rgb_workers=int(rgb_workers),
        )
        stage = "verification"
        try:
            verification = _run_independent_verification(
                reservation,
                source_review_file_sha256,
            )
            stage = "finalization"
            gate = _run_finalization(reservation, source_review_file_sha256)
            _assert_owned_claim(reservation)
            return {
                "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v6_end_to_end_summary_v1",
                "seed": 20260710,
                "fit_size": 5,
                "training": training,
                "metric_verification_content_sha256": verification["content_sha256"],
                "gate_content_sha256": gate["content_sha256"],
                "passes": bool(gate["passes"]),
                "later_rung_execution_authorized": False,
            }
        except BaseException as error:
            try:
                _terminate_failure(reservation, error, stage=stage)
            except BaseException as terminal_error:
                raise RuntimeError(
                    f"N5 full-panel V6 {stage} failed and terminal receipt "
                    "could not be written"
                ) from terminal_error
            raise
        finally:
            os.close(reservation.directory_fd)
            if reservation.directory_chain is not None:
                _close_directory_chain(reservation.directory_chain)


    def run_cpu_contract_smoke() -> dict[str, Any]:
        """CPU-only retained schedule/arithmetic check; no data or model opens."""

        from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

        schedule = base._deterministic_training_batches(
            frame_count=5,
            batch_size=5,
            steps=400,
            seed=20260710,
        )
        components = {
            "ordered_first_hit_nll": 0.8,
            "target_bin_offset_smooth_l1": 0.02,
            "ground_clear_distance_state_balanced_bce": 0.04,
            "derived_raster_hierarchical_bce": 0.2,
        }
        losses = {
            **components,
            "total": 0.25 * sum(components[name] for name in policy.LOSS_COMPONENTS),
        }
        return {
            "schedule_sha256": base.canonical_json_sha256(schedule),
            "update_count": len(schedule),
            "frame_exposures": sum(len(batch) for batch in schedule),
            "every_update_is_full_panel": all(
                len(batch) == 5 and set(batch) == set(range(5)) for batch in schedule
            ),
            "losses": losses,
        }


    def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source-review-sha256")
        parser.add_argument("--rgb-workers", type=int, choices=range(1, 6), default=5)
        parser.add_argument("--cpu-contract-smoke", action="store_true")
        args = parser.parse_args(argv)
        if args.cpu_contract_smoke:
            if args.source_review_sha256 is not None:
                raise ValueError("CPU smoke does not accept a source review")
            return args
        if not policy.is_sha256(args.source_review_sha256):
            raise ValueError("N5 full-panel V6 source review SHA-256 is malformed")
        return args


    def _isolated_child(argv: Sequence[str]) -> int:
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["HIP_VISIBLE_DEVICES"] = "0"
        environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
        for name in policy.THREAD_ENVIRONMENT:
            environment[name] = "1"
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *argv],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        return int(completed.returncode)


    def dispatch(argv: Sequence[str] | None = None) -> int:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        if not sys.flags.isolated:
            return _isolated_child(raw_argv)
        args = parse_args(raw_argv)
        if args.cpu_contract_smoke:
            print(policy.canonical_json_bytes(run_cpu_contract_smoke()).decode("ascii"))
            return 0
        summary = execute_exact(
            args.source_review_sha256,
            rgb_workers=int(args.rgb_workers),
        )
        print(policy.canonical_json_bytes(summary).decode("ascii"))
        return 0

    raise SystemExit(dispatch())
