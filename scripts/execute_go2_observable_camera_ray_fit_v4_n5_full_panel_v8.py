#!/usr/bin/env python3
"""Authority-free V8 executor for the frozen N5 experiment.

This is the sole production operation.  It performs source preflight, the
single filesystem claim, frozen training, compute-only verification in a fresh
isolated child, and parent-owned publication/finalization. It has no
caller-held authority or caller-controlled production path. Importing it opens
no data or output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import multiprocessing
import os
from pathlib import Path
import secrets
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v8 as policy,
)

if __name__ == "__main__":
    ROOT = SCRIPT_ROOT
    BASE_TRAINER_RELATIVE_PATH = "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
    BASE_TRAINER_FILE_SHA256 = policy.frozen_source_bindings()[
        BASE_TRAINER_RELATIVE_PATH
    ]
    STAGING_PREFIX = ".n5.reservation-v8-"
    LOCK_NAME = ".n5.reservation-v8.lock"
    STAGING_SCHEMA = "lewm_go2_n5_full_panel_v8_preclaim_staging_v1"
    RECOVERY_POLICY = {
        "incomplete": "preserve_invalid_and_block_claim",
        "complete": "rehash_then_resume_single_staging",
        "foreign": "preserve_invalid_and_block_claim",
        "mutated": "preserve_invalid_and_block_claim",
        "multiple_complete": (
            "resume_lexical_first_remove_authority_equivalent_duplicates"
        ),
    }


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
            raise PermissionError("V8 canonical directory walks require no-follow opens")
        return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


    @dataclass
    class DirectoryEntryV8:
        parent_fd: int
        name: str
        child_fd: int
        identity_security: tuple[int, ...]
        exclusive: bool


    @dataclass(frozen=True)
    class DirectorySnapshotV8:
        fingerprint: tuple[int, ...]
        inventory: tuple[tuple[str, tuple[int, ...]], ...]


    @dataclass(frozen=True)
    class JournalEventV8:
        watch: int
        mask: int
        cookie: int
        name: str


    @dataclass
    class DirectoryWatchV8:
        watch: int
        generation: int
        directory_fd: int
        identity: tuple[int, int]
        label: str
        snapshot: DirectorySnapshotV8


    class OwnedDirectoryJournalV8:
        """Linux event provenance for closed mutations of owned directories."""

        IN_MODIFY = 0x00000002
        IN_ATTRIB = 0x00000004
        IN_CLOSE_WRITE = 0x00000008
        IN_MOVED_FROM = 0x00000040
        IN_MOVED_TO = 0x00000080
        IN_CREATE = 0x00000100
        IN_DELETE = 0x00000200
        IN_DELETE_SELF = 0x00000400
        IN_MOVE_SELF = 0x00000800
        IN_UNMOUNT = 0x00002000
        IN_Q_OVERFLOW = 0x00004000
        IN_IGNORED = 0x00008000
        IN_ONLYDIR = 0x01000000
        IN_ISDIR = 0x40000000
        RENAME_NOREPLACE = 1
        EVENT_STRUCT = struct.Struct("iIII")
        WATCH_MASK = (
            IN_MODIFY
            | IN_ATTRIB
            | IN_CLOSE_WRITE
            | IN_MOVED_FROM
            | IN_MOVED_TO
            | IN_CREATE
            | IN_DELETE
            | IN_DELETE_SELF
            | IN_MOVE_SELF
            | IN_UNMOUNT
            | IN_Q_OVERFLOW
            | IN_IGNORED
            | IN_ONLYDIR
        )

        def __init__(self) -> None:
            libc = ctypes.CDLL(None, use_errno=True)
            init = libc.inotify_init1
            init.argtypes = [ctypes.c_int]
            init.restype = ctypes.c_int
            descriptor = int(
                init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
            )
            if descriptor < 0:
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code))
            add = libc.inotify_add_watch
            add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            add.restype = ctypes.c_int
            remove = libc.inotify_rm_watch
            remove.argtypes = [ctypes.c_int, ctypes.c_int]
            remove.restype = ctypes.c_int
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is None:
                os.close(descriptor)
                raise PermissionError(
                    "V8 requires Linux renameat2 for no-replace publication"
                )
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            self._libc = libc
            self._add_watch = add
            self._remove_watch = remove
            self._renameat2 = renameat2
            self._fd = descriptor
            self._by_watch: dict[int, DirectoryWatchV8] = {}
            self._by_identity: dict[tuple[int, int], DirectoryWatchV8] = {}
            self._generation = 0
            self._poison: str | None = None
            self._closed = False

        @property
        def poisoned(self) -> bool:
            return self._poison is not None

        @property
        def poison_reason(self) -> str | None:
            return self._poison

        def _fail(self, reason: str) -> None:
            if self._poison is None:
                self._poison = reason
            raise PermissionError(f"V8 owned-directory journal rejected: {reason}")

        @staticmethod
        def _identity(directory_fd: int) -> tuple[int, int]:
            metadata = os.fstat(directory_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise PermissionError("V8 journal target is not a directory")
            return (metadata.st_dev, metadata.st_ino)

        def _rename_noreplace(
            self,
            source_fd: int,
            source: str,
            destination_fd: int,
            destination: str,
        ) -> None:
            result = int(
                self._renameat2(
                    source_fd,
                    os.fsencode(source),
                    destination_fd,
                    os.fsencode(destination),
                    self.RENAME_NOREPLACE,
                )
            )
            if result < 0:
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code), destination)

        @staticmethod
        def _snapshot(directory_fd: int) -> DirectorySnapshotV8:
            opened = os.fstat(directory_fd)
            if not stat.S_ISDIR(opened.st_mode):
                raise PermissionError("V8 journal directory descriptor changed")
            rows: list[tuple[str, tuple[int, ...]]] = []
            for name in sorted(os.listdir(directory_fd)):
                if Path(name).name != name or name in {"", ".", ".."}:
                    raise PermissionError("V8 journal observed a malformed child name")
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                rows.append((name, _stable_fingerprint(metadata)))
            return DirectorySnapshotV8(
                fingerprint=_stable_fingerprint(opened),
                inventory=tuple(rows),
            )

        def _events(self) -> list[JournalEventV8]:
            events: list[JournalEventV8] = []
            while True:
                try:
                    raw = os.read(self._fd, 1024 * 1024)
                except BlockingIOError:
                    break
                except OSError as error:
                    if error.errno == errno.EAGAIN:
                        break
                    raise
                if not raw:
                    break
                offset = 0
                while offset < len(raw):
                    if len(raw) - offset < self.EVENT_STRUCT.size:
                        self._fail("truncated inotify event")
                    watch, mask, cookie, length = self.EVENT_STRUCT.unpack_from(
                        raw, offset
                    )
                    offset += self.EVENT_STRUCT.size
                    end = offset + int(length)
                    if end > len(raw):
                        self._fail("truncated inotify name")
                    name_raw = raw[offset:end].split(b"\0", 1)[0]
                    offset = end
                    events.append(
                        JournalEventV8(
                            watch=int(watch),
                            mask=int(mask),
                            cookie=int(cookie),
                            name=os.fsdecode(name_raw),
                        )
                    )
            return events

        def _reject_special(self, events: Sequence[JournalEventV8]) -> None:
            for event in events:
                if event.mask & self.IN_Q_OVERFLOW:
                    self._fail("inotify queue overflow")
                if event.mask & self.IN_UNMOUNT:
                    self._fail("watched filesystem unmounted")
                if event.watch not in self._by_watch:
                    self._fail("unknown or reused inotify watch")

        def _idle(self) -> None:
            if self._closed:
                self._fail("journal is closed")
            if self._poison is not None:
                self._fail(self._poison)
            events = self._events()
            self._reject_special(events)
            if events:
                self._fail("event occurred outside an owned transaction")

        def _shared_parent_events(
            self,
            events: Sequence[JournalEventV8],
            *,
            watch: int,
            reserved_name: str,
            allow_ignored: bool = False,
        ) -> tuple[list[JournalEventV8], list[JournalEventV8]]:
            """Classify one-shot shared-parent events without trusting churn."""

            allowed_child_masks = {
                self.IN_MODIFY,
                self.IN_ATTRIB,
                self.IN_CLOSE_WRITE,
                self.IN_MOVED_FROM,
                self.IN_MOVED_TO,
                self.IN_CREATE,
                self.IN_DELETE,
            }
            reserved: list[JournalEventV8] = []
            ignored: list[JournalEventV8] = []
            for event in events:
                if event.mask & self.IN_Q_OVERFLOW:
                    self._fail("inotify queue overflow during root creation")
                if event.mask & self.IN_UNMOUNT:
                    self._fail("shared parent unmounted during root creation")
                if event.watch != watch:
                    self._fail("unknown watch during root creation")
                if event == JournalEventV8(watch, self.IN_IGNORED, 0, ""):
                    if not allow_ignored:
                        self._fail("shared parent watch was lost during root creation")
                    ignored.append(event)
                    continue
                if event.name == "" or Path(event.name).name != event.name:
                    self._fail("shared parent self or malformed event")
                base_mask = event.mask & ~self.IN_ISDIR
                expected_mask = base_mask | (
                    self.IN_ISDIR if event.mask & self.IN_ISDIR else 0
                )
                if (
                    event.mask != expected_mask
                    or base_mask not in allowed_child_masks
                ):
                    self._fail("unknown shared-parent event mask")
                if base_mask in {self.IN_MOVED_FROM, self.IN_MOVED_TO}:
                    if event.cookie == 0:
                        self._fail("shared-parent move cookie is missing")
                elif event.cookie != 0:
                    self._fail("unexpected shared-parent event cookie")
                if event.name == reserved_name:
                    reserved.append(event)
            return reserved, ignored

        def create_exclusive_root(
            self,
            parent_fd: int,
            name: str,
            *,
            mode: int = 0o700,
        ) -> tuple[int, ...]:
            """Prove creation of the exclusive root inside a shared parent.

            The shared parent is watched only for this operation. Events for
            unrelated child names are tolerated, while the V8 name must have
            exactly one CREATE|ISDIR event and the shared parent's identity and
            security fields must remain stable.
            """

            self._idle()
            parent_before = _identity_security_fingerprint(os.fstat(parent_fd))
            retained_fd = os.dup(parent_fd)
            watch = int(
                self._add_watch(
                    self._fd,
                    f"/proc/self/fd/{retained_fd}".encode("ascii"),
                    self.WATCH_MASK,
                )
            )
            if watch < 0:
                code = ctypes.get_errno()
                os.close(retained_fd)
                raise OSError(code, os.strerror(code))
            retired = False
            try:
                install_events = self._events()
                reserved, _ignored = self._shared_parent_events(
                    install_events,
                    watch=watch,
                    reserved_name=name,
                )
                if reserved:
                    self._fail("exclusive root changed while its watch was installed")
                if (
                    _identity_security_fingerprint(os.fstat(parent_fd))
                    != parent_before
                ):
                    self._fail("shared parent identity changed during watch installation")
                try:
                    os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError(name)
                os.mkdir(name, mode, dir_fd=parent_fd)
                os.fsync(parent_fd)
                created = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                parent_after = _identity_security_fingerprint(os.fstat(parent_fd))
                events = self._events()
                owned, _ignored = self._shared_parent_events(
                    events,
                    watch=watch,
                    reserved_name=name,
                )
                if owned != [
                    JournalEventV8(
                        watch,
                        self.IN_CREATE | self.IN_ISDIR,
                        0,
                        name,
                    )
                ]:
                    self._fail("exclusive-root creation event sequence changed")
                if (
                    parent_after != parent_before
                    or stat.S_ISLNK(created.st_mode)
                    or not stat.S_ISDIR(created.st_mode)
                    or stat.S_IMODE(created.st_mode) != mode
                    or created.st_uid != os.getuid()
                    or created.st_gid != os.getgid()
                ):
                    self._fail("exclusive-root creation identity changed")
                if self._remove_watch(self._fd, watch) < 0:
                    code = ctypes.get_errno()
                    self._fail(f"exclusive-root watch retirement failed: {code}")
                retired = True
                retirement_events = self._events()
                late_owned, ignored = self._shared_parent_events(
                    retirement_events,
                    watch=watch,
                    reserved_name=name,
                    allow_ignored=True,
                )
                if late_owned:
                    self._fail("exclusive root changed during watch retirement")
                if ignored != [
                    JournalEventV8(watch, self.IN_IGNORED, 0, "")
                ]:
                    self._fail("exclusive-root watch retirement changed")
                return _stable_fingerprint(created)
            finally:
                if not retired:
                    self._remove_watch(self._fd, watch)
                    self._events()
                os.close(retained_fd)

        def watch_directory(
            self,
            directory_fd: int,
            *,
            label: str,
            expected_snapshot: DirectorySnapshotV8 | None = None,
        ) -> None:
            self.assert_clean()
            identity = self._identity(directory_fd)
            existing = self._by_identity.get(identity)
            if existing is not None:
                if (
                    expected_snapshot is not None
                    and existing.snapshot != expected_snapshot
                ):
                    self._fail("existing directory watch baseline changed")
                return
            before_install = self._snapshot(directory_fd)
            if (
                expected_snapshot is not None
                and before_install != expected_snapshot
            ):
                self._fail("directory changed before its watch was installed")
            retained_fd = os.dup(directory_fd)
            path = f"/proc/self/fd/{retained_fd}".encode("ascii")
            watch = int(self._add_watch(self._fd, path, self.WATCH_MASK))
            if watch < 0:
                code = ctypes.get_errno()
                os.close(retained_fd)
                raise OSError(code, os.strerror(code))
            self._generation += 1
            state = DirectoryWatchV8(
                watch=watch,
                generation=self._generation,
                directory_fd=retained_fd,
                identity=identity,
                label=label,
                snapshot=before_install,
            )
            if watch in self._by_watch:
                os.close(retained_fd)
                self._fail("inotify watch descriptor reused before retirement")
            self._by_watch[watch] = state
            self._by_identity[identity] = state
            after_install = self._snapshot(retained_fd)
            events = self._events()
            self._reject_special(events)
            if events or after_install != before_install:
                self._fail("directory changed while its watch was installed")
            state.snapshot = after_install

        def _state(self, directory_fd: int) -> DirectoryWatchV8:
            identity = self._identity(directory_fd)
            state = self._by_identity.get(identity)
            if state is None:
                self._fail("exclusive directory lacks a retained watch")
            return state

        def baseline(self, directory_fd: int) -> DirectorySnapshotV8:
            return self._state(directory_fd).snapshot

        def _snapshots(self) -> dict[int, DirectorySnapshotV8]:
            return {
                state.watch: self._snapshot(state.directory_fd)
                for state in sorted(
                    self._by_watch.values(), key=lambda item: item.generation
                )
            }

        @staticmethod
        def _identity_security_from_snapshot(
            snapshot: DirectorySnapshotV8,
        ) -> tuple[int, ...]:
            value = snapshot.fingerprint
            return (value[0], value[1], value[2], value[4], value[5])

        def _validate_snapshots(
            self,
            before: Mapping[int, DirectorySnapshotV8],
            after: Mapping[int, DirectorySnapshotV8],
            *,
            direct_inventories: Mapping[
                int, tuple[tuple[str, tuple[int, ...]], ...]
            ],
            direct_fingerprint_changes: set[int],
        ) -> None:
            if set(after) != set(before) or set(after) != set(self._by_watch):
                self._fail("active watch set changed during a transaction")
            changed_directory_fingerprints = {
                self._by_watch[watch].identity: current.fingerprint
                for watch, current in after.items()
                if current.fingerprint != before[watch].fingerprint
            }
            for watch, prior in before.items():
                current = after[watch]
                if self._identity_security_from_snapshot(current) != (
                    self._identity_security_from_snapshot(prior)
                ):
                    self._fail("watched directory identity/security changed")
                if (
                    watch not in direct_fingerprint_changes
                    and current.fingerprint != prior.fingerprint
                ):
                    self._fail("undeclared watched directory fingerprint changed")
                expected_inventory = direct_inventories.get(watch)
                if expected_inventory is None:
                    propagated: list[tuple[str, tuple[int, ...]]] = []
                    for name, fingerprint in prior.inventory:
                        replacement = changed_directory_fingerprints.get(
                            (fingerprint[0], fingerprint[1])
                        )
                        propagated.append((name, replacement or fingerprint))
                    expected_inventory = tuple(propagated)
                if current.inventory != expected_inventory:
                    self._fail("undeclared watched directory inventory changed")

        def _commit_snapshots(
            self,
            snapshots: Mapping[int, DirectorySnapshotV8],
        ) -> None:
            for watch, snapshot in snapshots.items():
                self._by_watch[watch].snapshot = snapshot

        def assert_clean(self) -> None:
            self._idle()
            current = self._snapshots()
            for watch, snapshot in current.items():
                state = self._by_watch[watch]
                if snapshot != state.snapshot:
                    self._fail(f"uncommitted mutation in {state.label}")
            self._idle()

        def _begin(
            self,
            directory_fd: int,
        ) -> tuple[DirectoryWatchV8, dict[int, DirectorySnapshotV8]]:
            self.assert_clean()
            state = self._state(directory_fd)
            before = self._snapshots()
            for watch, snapshot in before.items():
                if snapshot != self._by_watch[watch].snapshot:
                    self._fail("pre-operation snapshot changed")
            self._idle()
            return state, before

        def _finish(
            self,
            state: DirectoryWatchV8,
            before: Mapping[int, DirectorySnapshotV8],
            *,
            expected_events: Sequence[JournalEventV8],
            expected_inventory: tuple[tuple[str, tuple[int, ...]], ...],
        ) -> DirectorySnapshotV8:
            after = self._snapshots()
            events = self._events()
            self._reject_special(events)
            if list(events) != list(expected_events):
                self._fail(
                    f"unexpected event sequence in {state.label}: {events!r}"
                )
            self._validate_snapshots(
                before,
                after,
                direct_inventories={state.watch: expected_inventory},
                direct_fingerprint_changes={state.watch},
            )
            # Commit the snapshot captured before the event drain. A later event
            # remains queued and can never be folded into this baseline.
            self._commit_snapshots(after)
            return after[state.watch]

        @staticmethod
        def _replace_inventory(
            before: DirectorySnapshotV8,
            *,
            remove: set[str] = frozenset(),
            add: Mapping[str, tuple[int, ...]] | None = None,
        ) -> tuple[tuple[str, tuple[int, ...]], ...]:
            rows = {name: fingerprint for name, fingerprint in before.inventory}
            for name in remove:
                rows.pop(name, None)
            if add:
                rows.update(add)
            return tuple(sorted(rows.items()))

        def mkdir(self, parent_fd: int, name: str, *, mode: int = 0o700) -> None:
            state, before = self._begin(parent_fd)
            target_before = before[state.watch]
            if name in dict(target_before.inventory):
                raise FileExistsError(name)
            os.mkdir(name, mode, dir_fd=parent_fd)
            os.fsync(parent_fd)
            child = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(child.st_mode)
                or stat.S_IMODE(child.st_mode) != mode
                or child.st_uid != os.getuid()
                or child.st_gid != os.getgid()
            ):
                self._fail("owned mkdir did not create a private directory")
            expected = self._replace_inventory(
                target_before,
                add={name: _stable_fingerprint(child)},
            )
            self._finish(
                state,
                before,
                expected_events=(
                    JournalEventV8(state.watch, self.IN_CREATE | self.IN_ISDIR, 0, name),
                ),
                expected_inventory=expected,
            )

        def create_file(
            self,
            parent_fd: int,
            name: str,
            payload: bytes,
            *,
            mode: int = 0o600,
        ) -> tuple[int, ...]:
            state, before = self._begin(parent_fd)
            target_before = before[state.watch]
            if name in dict(target_before.inventory):
                raise FileExistsError(name)
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != mode
                    or metadata.st_uid != os.getuid()
                    or metadata.st_gid != os.getgid()
                ):
                    self._fail("owned leaf is not private and singly linked")
                if payload:
                    view = memoryview(payload)
                    offset = 0
                    while offset < len(view):
                        written = os.write(descriptor, view[offset:])
                        if written <= 0:
                            raise OSError(errno.EIO, "short regular-file write")
                        offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(parent_fd)
            committed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            fingerprint = _stable_fingerprint(committed)
            expected_events = [
                JournalEventV8(state.watch, self.IN_CREATE, 0, name),
            ]
            if payload:
                expected_events.append(
                    JournalEventV8(state.watch, self.IN_MODIFY, 0, name)
                )
            expected_events.append(
                JournalEventV8(state.watch, self.IN_CLOSE_WRITE, 0, name)
            )
            expected = self._replace_inventory(
                target_before, add={name: fingerprint}
            )
            self._finish(
                state,
                before,
                expected_events=expected_events,
                expected_inventory=expected,
            )
            return fingerprint

        def replace_file(
            self,
            parent_fd: int,
            name: str,
            payload: bytes,
        ) -> tuple[int, ...]:
            temporary = f".{name}.replace-{secrets.token_hex(12)}"
            temporary_fingerprint = self.create_file(
                parent_fd, temporary, payload, mode=0o600
            )
            state, before = self._begin(parent_fd)
            target_before = before[state.watch]
            rows = dict(target_before.inventory)
            if rows.get(temporary) != temporary_fingerprint or name not in rows:
                self._fail("replace inputs changed")
            os.replace(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
            committed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            fingerprint = _stable_fingerprint(committed)
            after = self._snapshots()
            events = self._events()
            self._reject_special(events)
            if (
                len(events) != 2
                or events[0].watch != state.watch
                or events[0].mask != self.IN_MOVED_FROM
                or events[0].name != temporary
                or events[1].watch != state.watch
                or events[1].mask != self.IN_MOVED_TO
                or events[1].name != name
                or events[0].cookie == 0
                or events[0].cookie != events[1].cookie
            ):
                self._fail("replace move cookie or sequence changed")
            expected = self._replace_inventory(
                target_before,
                remove={temporary, name},
                add={name: fingerprint},
            )
            self._validate_snapshots(
                before,
                after,
                direct_inventories={state.watch: expected},
                direct_fingerprint_changes={state.watch},
            )
            self._commit_snapshots(after)
            return fingerprint

        def unlink(
            self,
            parent_fd: int,
            name: str,
            *,
            expected_fingerprint: tuple[int, ...],
        ) -> None:
            state, before = self._begin(parent_fd)
            target_before = before[state.watch]
            if dict(target_before.inventory).get(name) != expected_fingerprint:
                self._fail("owned unlink fingerprint changed")
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            expected = self._replace_inventory(target_before, remove={name})
            self._finish(
                state,
                before,
                expected_events=(
                    JournalEventV8(state.watch, self.IN_DELETE, 0, name),
                ),
                expected_inventory=expected,
            )

        def rmdir(
            self,
            parent_fd: int,
            name: str,
            *,
            directory_fd: int,
        ) -> None:
            """Remove one watched empty directory and retire its watch exactly."""

            parent, before = self._begin(parent_fd)
            child = self._state(directory_fd)
            parent_before = before[parent.watch]
            child_before = before[child.watch]
            source_fingerprint = dict(parent_before.inventory).get(name)
            if (
                source_fingerprint is None
                or (source_fingerprint[0], source_fingerprint[1]) != child.identity
                or source_fingerprint != child_before.fingerprint
                or parent.watch == child.watch
                or child_before.inventory
            ):
                self._fail("owned rmdir target changed or is not empty")
            # The caller transfers this descriptor to the deletion transaction.
            # Closing it leaves the journal's retained descriptor as the final
            # reference, so closing that descriptor below deterministically
            # emits DELETE_SELF followed by IGNORED for the same generation.
            os.close(directory_fd)
            os.rmdir(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            after = self._snapshots()
            events = self._events()
            self._reject_special(events)
            expected_parent = self._replace_inventory(
                parent_before,
                remove={name},
            )
            if events != [
                JournalEventV8(
                    parent.watch,
                    self.IN_DELETE | self.IN_ISDIR,
                    0,
                    name,
                )
            ]:
                self._fail("directory removal event sequence changed")
            self._validate_snapshots(
                before,
                after,
                direct_inventories={
                    parent.watch: expected_parent,
                    child.watch: (),
                },
                direct_fingerprint_changes={parent.watch, child.watch},
            )
            os.close(child.directory_fd)
            retired = self._events()
            if retired != [
                JournalEventV8(child.watch, self.IN_DELETE_SELF, 0, ""),
                JournalEventV8(child.watch, self.IN_IGNORED, 0, ""),
            ]:
                self._fail("owned watch retirement event sequence changed")
            self._commit_snapshots(after)
            self._by_watch.pop(child.watch, None)
            self._by_identity.pop(child.identity, None)

        def rename_directory(
            self,
            parent_fd: int,
            source: str,
            destination: str,
            *,
            directory_fd: int,
        ) -> DirectorySnapshotV8:
            parent, before = self._begin(parent_fd)
            child = self._state(directory_fd)
            before_parent = before[parent.watch]
            before_child = before[child.watch]
            source_fp = dict(before_parent.inventory).get(source)
            if (
                source_fp is None
                or destination in dict(before_parent.inventory)
                or parent.watch == child.watch
                or (source_fp[0], source_fp[1]) != child.identity
                or source_fp != before_child.fingerprint
            ):
                self._fail("rename source/destination changed")
            try:
                self._rename_noreplace(
                    parent_fd,
                    source,
                    parent_fd,
                    destination,
                )
            except OSError as error:
                try:
                    self._idle()
                except PermissionError as journal_error:
                    raise journal_error from error
                self._fail("no-replace directory publication failed")
            os.fsync(parent_fd)
            after = self._snapshots()
            events = self._events()
            self._reject_special(events)
            if (
                len(events) != 3
                or events[0].watch != parent.watch
                or events[0].mask != self.IN_MOVED_FROM | self.IN_ISDIR
                or events[0].name != source
                or events[1].watch != parent.watch
                or events[1].mask != self.IN_MOVED_TO | self.IN_ISDIR
                or events[1].name != destination
                or events[0].cookie == 0
                or events[0].cookie != events[1].cookie
                or events[2] != JournalEventV8(
                    child.watch, self.IN_MOVE_SELF, 0, ""
                )
            ):
                self._fail("directory rename event sequence changed")
            destination_metadata = os.stat(
                destination, dir_fd=parent_fd, follow_symlinks=False
            )
            expected_parent = self._replace_inventory(
                before_parent,
                remove={source},
                add={destination: _stable_fingerprint(destination_metadata)},
            )
            self._validate_snapshots(
                before,
                after,
                direct_inventories={
                    parent.watch: expected_parent,
                    child.watch: before_child.inventory,
                },
                direct_fingerprint_changes={parent.watch, child.watch},
            )
            self._commit_snapshots(after)
            return after[child.watch]

        def terminal_create_file(
            self,
            parent_fd: int,
            name: str,
            payload: bytes,
            *,
            mode: int = 0o600,
        ) -> tuple[int, ...]:
            """Identity-only terminal write; it never restores success."""

            if not self.poisoned:
                # If this journaled create poisons, its already-created leaf is
                # preserved invalid and the caller receives the rejection. It
                # cannot truthfully self-attest the poison discovered after its
                # payload was fixed.
                return self.create_file(parent_fd, name, payload, mode=mode)
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=parent_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != mode
                    or metadata.st_uid != os.getuid()
                    or metadata.st_gid != os.getgid()
                ):
                    raise PermissionError(
                        "terminal leaf is not private and singly linked"
                    )
                if payload:
                    view = memoryview(payload)
                    offset = 0
                    while offset < len(view):
                        written = os.write(descriptor, view[offset:])
                        if written <= 0:
                            raise OSError(errno.EIO, "short terminal write")
                        offset += written
                os.fsync(descriptor)
                metadata = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(parent_fd)
            return _stable_fingerprint(metadata)

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            for state in self._by_watch.values():
                try:
                    os.close(state.directory_fd)
                except OSError:
                    pass
            self._by_watch.clear()
            self._by_identity.clear()
            try:
                os.close(self._fd)
            except OSError:
                pass


    @dataclass
    class CanonicalDirectoryChainV8:
        anchor_fd: int
        anchor_identity_security: tuple[int, ...]
        descriptors: list[int]
        entries: list[DirectoryEntryV8]
        path_fds: dict[Path, int]
        journal: OwnedDirectoryJournalV8
        output_root_created: bool = False
        closed: bool = False


    @dataclass(frozen=True)
    class OwnedArtifactV8:
        role: str
        parent_fd: int
        name: str
        fingerprint: tuple[int, ...]
        payload_sha256: str


    @dataclass(frozen=True)
    class OwnedStagingV8:
        path: Path
        directory_fd: int
        identity: tuple[int, int]


    def _entry_metadata(parent_fd: int, name: str) -> os.stat_result:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)


    def _assert_directory_chain(chain: CanonicalDirectoryChainV8) -> None:
        if chain.closed:
            raise PermissionError("V8 canonical directory chain is closed")
        chain.journal.assert_clean()
        if (
            _identity_security_fingerprint(os.fstat(chain.anchor_fd))
            != chain.anchor_identity_security
        ):
            raise PermissionError("V8 filesystem-root descriptor changed")
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
                    and _stable_fingerprint(named)
                    != chain.journal.baseline(entry.child_fd).fingerprint
                )
                or (
                    entry.exclusive
                    and _stable_fingerprint(opened)
                    != chain.journal.baseline(entry.child_fd).fingerprint
                )
            ):
                raise PermissionError("V8 canonical directory component changed")


    def _open_canonical_directory_chain(final_path: Path) -> CanonicalDirectoryChainV8:
        final_path = Path(final_path)
        if (
            not final_path.is_absolute()
            or not final_path.is_relative_to(ROOT)
            or any(part in {"", ".", ".."} for part in final_path.parts[1:])
        ):
            raise PermissionError("V8 canonical directory target escaped the repository")
        filesystem_root = Path(final_path.anchor)
        anchor_before = filesystem_root.stat(follow_symlinks=False)
        anchor_identity_security = _identity_security_fingerprint(anchor_before)
        anchor_fd = os.open(filesystem_root, _directory_flags())
        descriptors = [anchor_fd]
        entries: list[DirectoryEntryV8] = []
        path_fds = {filesystem_root: anchor_fd}
        journal = OwnedDirectoryJournalV8()
        chain = CanonicalDirectoryChainV8(
            anchor_fd=anchor_fd,
            anchor_identity_security=anchor_identity_security,
            descriptors=descriptors,
            entries=entries,
            path_fds=path_fds,
            journal=journal,
        )
        try:
            if (
                _identity_security_fingerprint(os.fstat(anchor_fd))
                != anchor_identity_security
            ):
                raise PermissionError("V8 filesystem root changed during open")
            parent_fd = anchor_fd
            current_path = filesystem_root
            repository_depth = len(ROOT.parts) - 1
            for index, component in enumerate(final_path.parts[1:]):
                created = False
                created_fingerprint: tuple[int, ...] | None = None
                try:
                    before = _entry_metadata(parent_fd, component)
                except FileNotFoundError:
                    if index < repository_depth:
                        raise PermissionError("V8 repository root component is missing")
                    if current_path / component == policy.CANONICAL_OUTPUT_ROOT:
                        created_fingerprint = journal.create_exclusive_root(
                            parent_fd,
                            component,
                            mode=0o700,
                        )
                        chain.output_root_created = True
                    elif current_path == policy.CANONICAL_OUTPUT_ROOT or (
                        current_path.is_relative_to(policy.CANONICAL_OUTPUT_ROOT)
                    ):
                        if not chain.output_root_created:
                            raise PermissionError(
                                "existing V8 recovery tree is missing a structural directory"
                            )
                        journal.mkdir(parent_fd, component, mode=0o700)
                    else:
                        os.mkdir(component, 0o700, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                    created = True
                    before = _entry_metadata(parent_fd, component)
                    if (
                        created_fingerprint is not None
                        and _stable_fingerprint(before) != created_fingerprint
                    ):
                        raise PermissionError(
                            "V8 exclusive root changed before retained open"
                        )
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise PermissionError("V8 canonical component is not a real directory")
                current_path = current_path / component
                exclusive = current_path == policy.CANONICAL_OUTPUT_ROOT or (
                    current_path.is_relative_to(policy.CANONICAL_OUTPUT_ROOT)
                )
                identity_security = _identity_security_fingerprint(before)
                full_fingerprint = _stable_fingerprint(before)
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(child_fd)
                opened = os.fstat(child_fd)
                expected_snapshot = journal._snapshot(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or _identity_security_fingerprint(opened) != identity_security
                    or (exclusive and _stable_fingerprint(opened) != full_fingerprint)
                    or expected_snapshot.fingerprint != _stable_fingerprint(opened)
                    or (
                        exclusive
                        and (
                            stat.S_IMODE(opened.st_mode) != 0o700
                            or opened.st_uid != os.getuid()
                            or opened.st_gid != os.getgid()
                        )
                    )
                    or (created and expected_snapshot.inventory)
                ):
                    raise PermissionError("V8 canonical component changed during open")
                entry = DirectoryEntryV8(
                    parent_fd,
                    component,
                    child_fd,
                    identity_security,
                    exclusive,
                )
                entries.append(entry)
                path_fds[current_path] = child_fd
                if exclusive:
                    journal.watch_directory(
                        child_fd,
                        label=str(current_path),
                        expected_snapshot=expected_snapshot,
                    )
                parent_fd = child_fd
            _assert_directory_chain(chain)
            return chain
        except BaseException:
            chain.closed = True
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            journal.close()
            raise


    def _open_chain_child(
        chain: CanonicalDirectoryChainV8,
        parent_path: Path,
        name: str,
    ) -> int:
        _assert_directory_chain(chain)
        parent_fd = chain.path_fds[parent_path]
        try:
            before = _entry_metadata(parent_fd, name)
        except FileNotFoundError:
            if not chain.output_root_created:
                raise PermissionError(
                    "existing V8 recovery tree is missing a derived directory"
                )
            chain.journal.mkdir(parent_fd, name, mode=0o700)
            before = _entry_metadata(parent_fd, name)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise PermissionError("V8 derived parent is not a real directory")
        identity_security = _identity_security_fingerprint(before)
        full_fingerprint = _stable_fingerprint(before)
        child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
        chain.descriptors.append(child_fd)
        opened = os.fstat(child_fd)
        expected_snapshot = chain.journal._snapshot(child_fd)
        if (
            _identity_security_fingerprint(opened) != identity_security
            or _stable_fingerprint(opened) != full_fingerprint
            or expected_snapshot.fingerprint != _stable_fingerprint(opened)
            or stat.S_IMODE(opened.st_mode) != 0o700
            or opened.st_uid != os.getuid()
            or opened.st_gid != os.getgid()
        ):
            raise PermissionError("V8 derived parent changed during open")
        child_path = parent_path / name
        chain.entries.append(
            DirectoryEntryV8(
                parent_fd,
                name,
                child_fd,
                identity_security,
                True,
            )
        )
        chain.path_fds[child_path] = child_fd
        chain.journal.watch_directory(
            child_fd,
            label=str(child_path),
            expected_snapshot=expected_snapshot,
        )
        _assert_directory_chain(chain)
        return child_fd


    def _close_directory_chain(chain: CanonicalDirectoryChainV8) -> None:
        if chain.closed:
            return
        chain.closed = True
        chain.journal.close()
        for descriptor in reversed(chain.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


    @dataclass(frozen=True)
    class AttemptReservationV8:
        directory: Path
        value: Mapping[str, Any]
        raw: bytes
        file_sha256: str
        directory_fd: int = -1
        directory_identity: tuple[int, int] | None = None
        directory_fingerprint: tuple[int, ...] | None = None
        directory_chain: CanonicalDirectoryChainV8 | None = None
        owned_claim_artifacts: dict[str, OwnedArtifactV8] = field(
            default_factory=dict,
            compare=False,
        )
        owned_derived_artifacts: dict[str, OwnedArtifactV8] = field(
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

        @property
        def journal(self) -> OwnedDirectoryJournalV8:
            if self.directory_chain is None:
                raise PermissionError("N5 full-panel V8 lacks its transaction journal")
            return self.directory_chain.journal

    def _assert_claim_fd_owned(reservation: AttemptReservationV8) -> None:
        if reservation.directory_fd < 0 or reservation.directory_identity is None:
            raise PermissionError("N5 full-panel V8 lacks an open claimed directory")
        opened = os.fstat(reservation.directory_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise PermissionError("N5 full-panel V8 claimed descriptor is not a directory")
        if (opened.st_dev, opened.st_ino) != reservation.directory_identity:
            raise PermissionError("N5 full-panel V8 claimed descriptor identity changed")


    def _assert_owned_claim(reservation: AttemptReservationV8) -> None:
        """Bind success stages to the retained canonical chain and claim FD."""

        _assert_claim_fd_owned(reservation)
        opened = os.fstat(reservation.directory_fd)
        expected_snapshot = reservation.journal.baseline(reservation.directory_fd)
        if (
            _stable_fingerprint(opened) != expected_snapshot.fingerprint
        ):
            raise PermissionError("N5 full-panel V8 claimed directory changed")
        if reservation.directory_chain is None or reservation.seed_root_fd < 0:
            raise PermissionError("N5 full-panel V8 lacks its canonical directory chain")
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
            or _stable_fingerprint(canonical) != expected_snapshot.fingerprint
        ):
            raise PermissionError("N5 full-panel V8 canonical claim identity changed")
        expected_names = {"reservation.json"} | set(
            reservation.owned_claim_artifacts
        )
        if set(os.listdir(reservation.directory_fd)) != expected_names:
            raise PermissionError("N5 full-panel V8 claimed directory inventory changed")


    def _write_claim_file_exclusive(
        reservation: AttemptReservationV8,
        name: str,
        payload: bytes,
        *,
        mode: int = 0o600,
        require_canonical: bool = True,
        role: str | None = None,
    ) -> None:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PermissionError("N5 full-panel V8 claim filename escaped")
        if require_canonical:
            _assert_owned_claim(reservation)
        else:
            _assert_claim_fd_owned(reservation)
        if require_canonical:
            committed_fingerprint = reservation.journal.create_file(
                reservation.directory_fd,
                name,
                payload,
                mode=mode,
            )
        else:
            committed_fingerprint = reservation.journal.terminal_create_file(
                reservation.directory_fd,
                name,
                payload,
                mode=mode,
            )
        reservation.owned_claim_artifacts[name] = OwnedArtifactV8(
            role=role or name,
            parent_fd=reservation.directory_fd,
            name=name,
            fingerprint=committed_fingerprint,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )
        committed = os.stat(
            name,
            dir_fd=reservation.directory_fd,
            follow_symlinks=False,
        )
        if _stable_fingerprint(committed) != committed_fingerprint:
            raise PermissionError("N5 full-panel V8 committed claim leaf changed")
        if require_canonical:
            _assert_owned_claim(reservation)
        else:
            _assert_claim_fd_owned(reservation)


    def _read_claim_file(
        reservation: AttemptReservationV8,
        name: str,
        *,
        require_canonical: bool = True,
    ) -> bytes:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PermissionError("N5 full-panel V8 claim filename escaped")
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
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_uid != os.getuid()
                or before.st_gid != os.getgid()
            ):
                raise PermissionError("N5 full-panel V8 claim leaf is not singly linked")
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
                raise RuntimeError("N5 full-panel V8 claim leaf changed while read")
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
    def _locked_seed_root(
        seed_root_fd: int,
        journal: OwnedDirectoryJournalV8,
        *,
        allow_create: bool = False,
    ) -> Iterator[None]:
        journal.assert_clean()
        rows = dict(journal.baseline(seed_root_fd).inventory)
        if LOCK_NAME not in rows:
            if not allow_create:
                journal._fail("existing recovery tree lacks its lock leaf")
            lock_fingerprint = journal.create_file(
                seed_root_fd,
                LOCK_NAME,
                b"",
                mode=0o600,
            )
        else:
            lock_fingerprint = rows[LOCK_NAME]
        descriptor = os.open(
            LOCK_NAME,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=seed_root_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or metadata.st_gid != os.getgid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size != 0
                or _stable_fingerprint(metadata) != lock_fingerprint
            ):
                raise PermissionError("N5 full-panel reservation lock is not private")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            journal.assert_clean()
            yield
            journal.assert_clean()
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
            if not journal.poisoned:
                journal.assert_clean()


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
    ) -> AttemptReservationV8:
        core = _reservation_core(
            review_binding,
            recovery_events=recovery_events,
        )
        value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
        raw = policy.canonical_json_bytes(value) + b"\n"
        return AttemptReservationV8(
            directory=attempt_path,
            value=value,
            raw=raw,
            file_sha256=hashlib.sha256(raw).hexdigest(),
        )


    def _manifest_value(
        *,
        staging: Path,
        attempt_path: Path,
        reservation: AttemptReservationV8,
    ) -> dict[str, Any]:
        core = {
            "schema": STAGING_SCHEMA,
            "status": "complete_and_resume_eligible",
            "staging_name": staging.name,
            "attempt_path": str(attempt_path.resolve()),
            "reservation_file_sha256": reservation.file_sha256,
            "reservation_content_sha256": reservation.value["content_sha256"],
            "private_mode": "0700",
            "recovery_policy": dict(RECOVERY_POLICY),
        }
        return {**core, "content_sha256": policy.canonical_json_sha256(core)}


    def _read_regular_at(
        directory_fd: int,
        name: str,
        *,
        expected_fingerprint: tuple[int, ...],
        maximum_bytes: int = 1024 * 1024,
    ) -> bytes:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_uid != os.getuid()
                or before.st_gid != os.getgid()
                or before.st_size > maximum_bytes
                or _stable_fingerprint(before) != expected_fingerprint
            ):
                raise PermissionError("staging leaf identity changed")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if _stable_fingerprint(after) != expected_fingerprint:
                raise PermissionError("staging leaf changed while read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)


    def _staging_inventory_digest(
        staging: OwnedStagingV8,
        journal: OwnedDirectoryJournalV8,
    ) -> tuple[list[str], str]:
        opened = os.fstat(staging.directory_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != staging.identity
        ):
            raise PermissionError("staging descriptor identity changed")
        inventory: list[str] = []
        for name, fingerprint in journal.baseline(
            staging.directory_fd
        ).inventory:
            metadata = os.stat(
                name,
                dir_fd=staging.directory_fd,
                follow_symlinks=False,
            )
            if _stable_fingerprint(metadata) != fingerprint:
                raise PermissionError("staging inventory changed")
            if (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1
                and stat.S_IMODE(metadata.st_mode) == 0o600
                and metadata.st_uid == os.getuid()
                and metadata.st_gid == os.getgid()
                and metadata.st_size <= 1024 * 1024
            ):
                raw = _read_regular_at(
                    staging.directory_fd,
                    name,
                    expected_fingerprint=fingerprint,
                )
                inventory.append(
                    f"file:{name}:{len(raw)}:{hashlib.sha256(raw).hexdigest()}"
                )
            elif stat.S_ISREG(metadata.st_mode):
                inventory.append(
                    "foreign_file:"
                    f"{name}:{stat.S_IMODE(metadata.st_mode):04o}:"
                    f"{metadata.st_nlink}:{metadata.st_uid}:{metadata.st_gid}:"
                    f"{metadata.st_size}"
                )
            elif stat.S_ISDIR(metadata.st_mode):
                inventory.append(f"directory:{name}")
            elif stat.S_ISLNK(metadata.st_mode):
                inventory.append(f"symlink:{name}")
            else:
                inventory.append(f"other:{name}:{metadata.st_size}")
        journal.assert_clean()
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
        staging: OwnedStagingV8,
        journal: OwnedDirectoryJournalV8,
        *,
        attempt_path: Path,
        review_binding: Mapping[str, str],
    ) -> tuple[dict[str, Any], AttemptReservationV8 | None]:
        inventory, inventory_sha = _staging_inventory_digest(staging, journal)
        evidence = {
            "staging_name": staging.path.name,
            "inventory_sha256": inventory_sha,
            "classification": "foreign",
            "action": "preserve_invalid_and_block_claim",
        }
        metadata = os.fstat(staging.directory_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            return evidence, None
        rows = dict(journal.baseline(staging.directory_fd).inventory)
        names = sorted(rows)
        if names != ["reservation.json", "staging.json"]:
            evidence["classification"] = "incomplete" if len(names) < 2 else "mutated"
            return evidence, None
        try:
            reservation_raw = _read_regular_at(
                staging.directory_fd,
                "reservation.json",
                expected_fingerprint=rows["reservation.json"],
            )
            reservation_value = policy.parse_json(
                reservation_raw,
                name="recoverable reservation",
            )
            if reservation_raw != policy.canonical_json_bytes(reservation_value) + b"\n":
                raise ValueError("recoverable reservation is not canonical JSON")
            manifest_raw = _read_regular_at(
                staging.directory_fd,
                "staging.json",
                expected_fingerprint=rows["staging.json"],
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
            and manifest.get("staging_name") == staging.path.name
            and manifest.get("attempt_path") == str(attempt_path.resolve())
            and manifest.get("reservation_file_sha256")
            == hashlib.sha256(reservation_raw).hexdigest()
            and manifest.get("reservation_content_sha256")
            == reservation_value.get("content_sha256")
            and manifest.get("private_mode") == "0700"
            and manifest.get("recovery_policy") == RECOVERY_POLICY
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
        journal.assert_clean()
        return evidence, AttemptReservationV8(
            directory=attempt_path,
            value=reservation_value,
            raw=reservation_raw,
            file_sha256=hashlib.sha256(reservation_raw).hexdigest(),
        )


    def _is_staging_name(name: str) -> bool:
        """Match exactly the namespace emitted by `_new_staging`."""

        if not name.startswith(STAGING_PREFIX):
            return False
        suffix = name[len(STAGING_PREFIX) :]
        return len(suffix) == 32 and all(
            character in "0123456789abcdef" for character in suffix
        )


    def _recoverable_authority_core(value: Mapping[str, Any]) -> dict[str, Any]:
        """Discard only the prior recovery ledger and its derived hash."""

        core = dict(value)
        core.pop("content_sha256", None)
        core.pop("preclaim_recovery", None)
        return core


    def _complete_recoveries_are_equivalent(
        first: AttemptReservationV8,
        second: AttemptReservationV8,
    ) -> bool:
        return _recoverable_authority_core(
            first.value
        ) == _recoverable_authority_core(second.value)


    def _validate_recovery_tree_baseline(
        chain: CanonicalDirectoryChainV8,
        seed_root: Path,
    ) -> None:
        """Reject historical bytes outside the sole recoverable V8 shape."""

        chain.journal.assert_clean()
        relative_seed = seed_root.relative_to(policy.CANONICAL_OUTPUT_ROOT)
        if len(relative_seed.parts) != 2:
            chain.journal._fail("V8 recovery seed-root depth changed")
        attempts_path = policy.CANONICAL_OUTPUT_ROOT / relative_seed.parts[0]
        metric_path = policy.CANONICAL_METRIC_RECEIPT_PATH.parent
        gate_path = policy.CANONICAL_GATE_PATH.parent
        required_paths = {
            policy.CANONICAL_OUTPUT_ROOT,
            attempts_path,
            seed_root,
            metric_path,
            gate_path,
        }
        if not required_paths <= set(chain.path_fds):
            chain.journal._fail("V8 recovery tree lacks retained descriptors")

        def names(path: Path) -> set[str]:
            return {
                name
                for name, _fingerprint in chain.journal.baseline(
                    chain.path_fds[path]
                ).inventory
            }

        output_expected = {
            relative_seed.parts[0],
            metric_path.name,
            gate_path.name,
        }
        if names(policy.CANONICAL_OUTPUT_ROOT) != output_expected:
            chain.journal._fail(
                "exclusive output root has an unproved recovery inventory"
            )
        if names(attempts_path) != {relative_seed.parts[1]}:
            chain.journal._fail(
                "exclusive attempts directory has an unproved recovery inventory"
            )
        if names(metric_path) or names(gate_path):
            chain.journal._fail(
                "unclaimed derived directory has an unproved recovery inventory"
            )
        seed_names = names(seed_root)
        unknown_seed_names = {
            name
            for name in seed_names
            if name != LOCK_NAME and not _is_staging_name(name)
        }
        if unknown_seed_names:
            chain.journal._fail(
                "seed root has an unproved recovery inventory"
            )
        if not chain.output_root_created and LOCK_NAME not in seed_names:
            chain.journal._fail(
                "existing recovery tree is missing its exact lock leaf"
            )
        chain.journal.assert_clean()


    def _open_staging(
        seed_root: Path,
        seed_root_fd: int,
        journal: OwnedDirectoryJournalV8,
        name: str,
    ) -> OwnedStagingV8:
        if Path(name).name != name or not _is_staging_name(name):
            raise PermissionError("staging open escaped its reviewed namespace")
        fingerprint = dict(journal.baseline(seed_root_fd).inventory).get(name)
        if fingerprint is None:
            raise FileNotFoundError(name)
        if not stat.S_ISDIR(fingerprint[2]) or fingerprint[4] != os.getuid():
            raise PermissionError("linked or foreign staging is preserved invalid")
        descriptor = os.open(name, _directory_flags(), dir_fd=seed_root_fd)
        try:
            metadata = os.fstat(descriptor)
            if (
                _stable_fingerprint(metadata) != fingerprint
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != os.getuid()
                or metadata.st_gid != os.getgid()
            ):
                raise PermissionError("private staging identity or mode changed")
            expected_snapshot = journal._snapshot(descriptor)
            if expected_snapshot.fingerprint != fingerprint:
                raise PermissionError("private staging changed before watch installation")
            journal.watch_directory(
                descriptor,
                label=str(seed_root / name),
                expected_snapshot=expected_snapshot,
            )
            journal.assert_clean()
            return OwnedStagingV8(
                path=seed_root / name,
                directory_fd=descriptor,
                identity=(metadata.st_dev, metadata.st_ino),
            )
        except BaseException:
            os.close(descriptor)
            raise


    def _remove_staging_tree(
        parent_fd: int,
        name: str,
        directory_fd: int,
        journal: OwnedDirectoryJournalV8,
    ) -> None:
        snapshot = journal.baseline(directory_fd)
        for child_name, expected_fingerprint in tuple(snapshot.inventory):
            current = os.stat(
                child_name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if _stable_fingerprint(current) != expected_fingerprint:
                raise PermissionError("staging cleanup child changed")
            if (
                stat.S_ISREG(current.st_mode)
                and current.st_nlink == 1
                and current.st_uid == os.getuid()
            ):
                journal.unlink(
                    directory_fd,
                    child_name,
                    expected_fingerprint=expected_fingerprint,
                )
                continue
            if stat.S_ISDIR(current.st_mode) and current.st_uid == os.getuid():
                child_fd = os.open(
                    child_name,
                    _directory_flags(),
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if _stable_fingerprint(opened) != expected_fingerprint:
                        raise PermissionError("staging cleanup directory changed")
                    expected_snapshot = journal._snapshot(child_fd)
                    if expected_snapshot.fingerprint != expected_fingerprint:
                        raise PermissionError(
                            "staging cleanup directory changed before watch installation"
                        )
                    journal.watch_directory(
                        child_fd,
                        label=f"staging/{child_name}",
                        expected_snapshot=expected_snapshot,
                    )
                    _remove_staging_tree(
                        directory_fd,
                        child_name,
                        child_fd,
                        journal,
                    )
                except BaseException:
                    try:
                        os.close(child_fd)
                    except OSError:
                        pass
                    raise
                continue
            raise PermissionError(
                "linked, changed, or foreign staging child is preserved invalid"
            )
        journal.rmdir(parent_fd, name, directory_fd=directory_fd)


    def _remove_staging(
        staging: OwnedStagingV8,
        *,
        seed_root: Path,
        seed_root_fd: int,
        journal: OwnedDirectoryJournalV8,
    ) -> None:
        if staging.path.parent != seed_root or not _is_staging_name(staging.path.name):
            raise PermissionError("staging cleanup escaped its reviewed namespace")
        opened = os.fstat(staging.directory_fd)
        if (
            (opened.st_dev, opened.st_ino) != staging.identity
            or opened.st_uid != os.getuid()
        ):
            raise PermissionError("foreign staging is preserved invalid")
        try:
            _remove_staging_tree(
                seed_root_fd,
                staging.path.name,
                staging.directory_fd,
                journal,
            )
        except BaseException:
            try:
                os.close(staging.directory_fd)
            except OSError:
                pass
            raise


    def _new_staging(
        seed_root: Path,
        seed_root_fd: int,
        journal: OwnedDirectoryJournalV8,
    ) -> OwnedStagingV8:
        for _ in range(128):
            name = f"{STAGING_PREFIX}{secrets.token_hex(16)}"
            try:
                journal.mkdir(seed_root_fd, name, mode=0o700)
            except FileExistsError:
                continue
            return _open_staging(seed_root, seed_root_fd, journal, name)
        raise FileExistsError("could not allocate unique private reservation staging")


    def _prepare_new_staging(
        staging: OwnedStagingV8,
        journal: OwnedDirectoryJournalV8,
        *,
        reservation: AttemptReservationV8,
        attempt_path: Path,
    ) -> None:
        journal.create_file(
            staging.directory_fd,
            "reservation.json",
            reservation.raw,
            mode=0o600,
        )
        manifest = _manifest_value(
            staging=staging.path,
            attempt_path=attempt_path,
            reservation=reservation,
        )
        journal.create_file(
            staging.directory_fd,
            "staging.json",
            policy.canonical_json_bytes(manifest) + b"\n",
            mode=0o600,
        )


    def _update_complete_staging(
        staging: OwnedStagingV8,
        journal: OwnedDirectoryJournalV8,
        *,
        reservation: AttemptReservationV8,
        attempt_path: Path,
        recovery_events: Sequence[Mapping[str, Any]],
        review_binding: Mapping[str, str],
    ) -> AttemptReservationV8:
        del reservation
        updated = _reservation(
            review_binding,
            attempt_path=attempt_path,
            recovery_events=recovery_events,
        )
        journal.replace_file(
            staging.directory_fd,
            "reservation.json",
            updated.raw,
        )
        manifest = _manifest_value(
            staging=staging.path,
            attempt_path=attempt_path,
            reservation=updated,
        )
        manifest_raw = policy.canonical_json_bytes(manifest) + b"\n"
        names = dict(journal.baseline(staging.directory_fd).inventory)
        if set(names) != {"reservation.json", "staging.json"}:
            journal._fail("complete staging manifest grammar changed")
        journal.replace_file(staging.directory_fd, "staging.json", manifest_raw)
        return updated


    def _reserve_exact_attempt(source_review_file_sha256: str) -> AttemptReservationV8:
        """Claim the one canonical attempt; no caller-controlled path exists."""

        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        review_binding = policy.source_review_binding(review, source_review_file_sha256)
        attempt_path = policy.CANONICAL_ATTEMPT_PATH
        seed_root = attempt_path.parent
        active_staging: OwnedStagingV8 | None = None
        owned_claim_identity: tuple[int, int] | None = None
        claimed_directory_fd: int | None = None
        claimed_chain: CanonicalDirectoryChainV8 | None = None
        claimed_reservation: AttemptReservationV8 | None = None
        recovery_staging_fds: set[int] = set()
        try:
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
            _validate_recovery_tree_baseline(claimed_chain, seed_root)
            seed_root_fd = claimed_chain.path_fds[seed_root]
            journal = claimed_chain.journal
            with _locked_seed_root(
                seed_root_fd,
                journal,
                allow_create=claimed_chain.output_root_created,
            ):
                seed_inventory = dict(journal.baseline(seed_root_fd).inventory)
                if attempt_path.name in seed_inventory:
                    raise FileExistsError("the sole N5 full-panel attempt is already claimed")
                candidate_names = sorted(
                    name for name in seed_inventory if _is_staging_name(name)
                )
                recovery_events: list[dict[str, Any]] = []
                complete: list[
                    tuple[OwnedStagingV8, AttemptReservationV8, dict[str, Any]]
                ] = []
                for candidate_name in candidate_names:
                    candidate = _open_staging(
                        seed_root,
                        seed_root_fd,
                        journal,
                        candidate_name,
                    )
                    recovery_staging_fds.add(candidate.directory_fd)
                    evidence, recovered = _classify_staging(
                        candidate,
                        journal,
                        attempt_path=attempt_path,
                        review_binding=review_binding,
                    )
                    journal.assert_clean()
                    if recovered is None:
                        evidence = {
                            **evidence,
                            "action": "preserve_invalid_and_block_claim",
                        }
                        recovery_events.append(evidence)
                        journal._fail(
                            "foreign, incomplete, or mutated staging is "
                            "preserved invalid"
                        )
                    else:
                        complete.append((candidate, recovered, evidence))
                if complete:
                    active_staging, reservation, selected = complete[0]
                    recovery_events.append(selected)
                    if any(
                        not _complete_recoveries_are_equivalent(
                            reservation,
                            duplicate_reservation,
                        )
                        for _duplicate, duplicate_reservation, _evidence in complete[1:]
                    ):
                        journal._fail(
                            "conflicting complete recovery stagings are preserved and "
                            "block claim"
                        )
                    for duplicate, duplicate_reservation, evidence in complete[1:]:
                        _remove_staging(
                            duplicate,
                            seed_root=seed_root,
                            seed_root_fd=seed_root_fd,
                            journal=journal,
                        )
                        recovery_staging_fds.discard(duplicate.directory_fd)
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
                    reservation = _update_complete_staging(
                        active_staging,
                        journal,
                        reservation=reservation,
                        attempt_path=attempt_path,
                        recovery_events=recovery_events,
                        review_binding=review_binding,
                    )
                else:
                    if not claimed_chain.output_root_created:
                        journal._fail(
                            "existing recovery tree lacks one complete staging"
                        )
                    active_staging = _new_staging(
                        seed_root,
                        seed_root_fd,
                        journal,
                    )
                    recovery_staging_fds.add(active_staging.directory_fd)
                    recovery_events.append(
                        {
                            "staging_name": active_staging.path.name,
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
                        journal,
                        reservation=reservation,
                        attempt_path=attempt_path,
                    )
                manifest_inventory = dict(
                    journal.baseline(active_staging.directory_fd).inventory
                )
                journal.unlink(
                    active_staging.directory_fd,
                    "staging.json",
                    expected_fingerprint=manifest_inventory["staging.json"],
                )
                claimed_directory_fd = active_staging.directory_fd
                recovery_staging_fds.discard(claimed_directory_fd)
                staging_metadata = os.fstat(claimed_directory_fd)
                named_staging_metadata = os.stat(
                    active_staging.path.name,
                    dir_fd=seed_root_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(staging_metadata.st_mode)
                    or (staging_metadata.st_dev, staging_metadata.st_ino)
                    != (named_staging_metadata.st_dev, named_staging_metadata.st_ino)
                ):
                    raise PermissionError("N5 full-panel V8 staging identity changed")
                owned_claim_identity = (staging_metadata.st_dev, staging_metadata.st_ino)
                claimed_reservation = AttemptReservationV8(
                    directory=attempt_path,
                    value=reservation.value,
                    raw=reservation.raw,
                    file_sha256=reservation.file_sha256,
                    directory_fd=claimed_directory_fd,
                    directory_identity=owned_claim_identity,
                    directory_fingerprint=_stable_fingerprint(staging_metadata),
                    directory_chain=claimed_chain,
                )
                journal.rename_directory(
                    seed_root_fd,
                    active_staging.path.name,
                    attempt_path.name,
                    directory_fd=claimed_directory_fd,
                )
                active_staging = None
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
                elif active_staging is not None:
                    if claimed_chain is None or claimed_chain.journal.poisoned:
                        raise PermissionError(
                            "poisoned preclaim staging is preserved invalid"
                        )
                    _remove_staging(
                        active_staging,
                        seed_root=seed_root,
                        seed_root_fd=claimed_chain.path_fds[seed_root],
                        journal=claimed_chain.journal,
                    )
                    recovery_staging_fds.discard(active_staging.directory_fd)
                    active_staging = None
                    claimed_directory_fd = None
            except BaseException as cleanup_error:
                secondary_error = cleanup_error
            finally:
                for staging_fd in tuple(recovery_staging_fds):
                    try:
                        os.close(staging_fd)
                    except OSError:
                        pass
                recovery_staging_fds.clear()
                try:
                    if claimed_directory_fd is not None:
                        try:
                            os.close(claimed_directory_fd)
                        except OSError:
                            pass
                finally:
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
        registry: dict[str, OwnedArtifactV8],
        *,
        journal: OwnedDirectoryJournalV8,
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
                outcomes.append(
                    {
                        "artifact": key,
                        "role": artifact.role,
                        "outcome": "missing_owned_artifact_invalid",
                    }
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
            if journal.poisoned:
                outcomes.append(
                    {
                        "artifact": key,
                        "role": artifact.role,
                        "outcome": "journal_integrity_failed_preserved_invalid",
                    }
                )
                continue
            try:
                journal.unlink(
                    artifact.parent_fd,
                    artifact.name,
                    expected_fingerprint=artifact.fingerprint,
                )
            except PermissionError:
                outcomes.append(
                    {
                        "artifact": key,
                        "role": artifact.role,
                        "outcome": "journal_integrity_failed_preserved_invalid",
                    }
                )
                continue
            registry.pop(key, None)
            outcomes.append(
                {"artifact": key, "role": artifact.role, "outcome": "removed_owned"}
            )
        return outcomes


    def _terminate_failure(
        reservation: AttemptReservationV8,
        error: BaseException,
        *,
        stage: str = "training",
    ) -> dict[str, Any]:
        _assert_claim_fd_owned(reservation)
        cleanup = _cleanup_owned_artifacts(
            reservation.owned_derived_artifacts,
            journal=reservation.journal,
        )
        cleanup.extend(
            _cleanup_owned_artifacts(
                reservation.owned_claim_artifacts,
                journal=reservation.journal,
                selected={"checkpoint.pt", "result.json", "completed.json"},
            )
        )
        if not reservation.journal.poisoned:
            try:
                reservation.journal.assert_clean()
            except PermissionError:
                pass
        os.fsync(reservation.directory_fd)
        journal_integrity = (
            "failed" if reservation.journal.poisoned else "intact"
        )
        core = {
            "schema": policy.FAILURE_SCHEMA,
            "status": "failed",
            "reservation": reservation.binding,
            "failure_stage": stage,
            "failure": _failure_code(error),
            "artifact_cleanup": cleanup,
            "owned_directory_journal": {
                "integrity": journal_integrity,
                "poison_reason": reservation.journal.poison_reason,
                "success_eligibility_restored": False,
            },
            "partial_artifacts_removed": all(
                item["outcome"] == "removed_owned"
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
        reservation: AttemptReservationV8,
        *,
        checkpoint_raw: bytes,
        checkpoint_content_sha256: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        _assert_owned_claim(reservation)
        if not policy.is_sha256(checkpoint_content_sha256):
            raise ValueError("N5 full-panel V8 checkpoint content hash is malformed")
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
        reservation: AttemptReservationV8,
        path: Path,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        _assert_owned_claim(reservation)
        path = Path(path)
        if path not in {policy.CANONICAL_METRIC_RECEIPT_PATH, policy.CANONICAL_GATE_PATH}:
            raise PermissionError("N5 full-panel V8 publication path is not canonical")
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
            raise ValueError("N5 full-panel V8 derived artifact is malformed")
        raw = policy.canonical_json_bytes(value) + b"\n"
        relative = path.relative_to(policy.CANONICAL_OUTPUT_ROOT)
        if len(relative.parts) != 2:
            raise PermissionError("N5 full-panel V8 publication depth changed")
        if reservation.directory_chain is None or reservation.output_root_fd < 0:
            raise PermissionError("N5 full-panel V8 lacks its retained output root")
        parent_path = policy.CANONICAL_OUTPUT_ROOT / relative.parts[0]
        parent_fd = reservation.directory_chain.path_fds.get(parent_path, -1)
        if parent_fd < 0:
            raise PermissionError("N5 full-panel V8 lacks its derived parent descriptor")
        relative_text = str(relative)
        committed_fingerprint = reservation.journal.create_file(
            parent_fd,
            relative.parts[1],
            raw,
            mode=0o600,
        )
        reservation.owned_derived_artifacts[relative_text] = OwnedArtifactV8(
            role=("metric_receipt" if path == policy.CANONICAL_METRIC_RECEIPT_PATH else "gate"),
            parent_fd=parent_fd,
            name=relative.parts[1],
            fingerprint=committed_fingerprint,
            payload_sha256=hashlib.sha256(raw).hexdigest(),
        )
        os.fsync(reservation.output_root_fd)
        committed = os.stat(
            relative.parts[1],
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if _stable_fingerprint(committed) != committed_fingerprint:
            raise PermissionError("N5 full-panel V8 committed derived leaf changed")
        _assert_owned_claim(reservation)
        return policy.artifact_binding(
            relative_text,
            raw,
            content_sha256=str(value["content_sha256"]),
        )


    def _read_owned_derived_artifact(
        reservation: AttemptReservationV8,
        path: Path,
    ) -> bytes:
        _assert_owned_claim(reservation)
        relative = str(Path(path).relative_to(policy.CANONICAL_OUTPUT_ROOT))
        artifact = reservation.owned_derived_artifacts.get(relative)
        if artifact is None:
            raise PermissionError("N5 full-panel V8 derived artifact is not owned")
        raw = _read_regular_at(
            artifact.parent_fd,
            artifact.name,
            expected_fingerprint=artifact.fingerprint,
        )
        if hashlib.sha256(raw).hexdigest() != artifact.payload_sha256:
            raise PermissionError("N5 full-panel V8 derived payload changed")
        _assert_owned_claim(reservation)
        return raw


    def _run_frozen_training(
        source_review_file_sha256: str,
        *,
        rgb_workers: int,
    ) -> tuple[dict[str, Any], AttemptReservationV8]:
        """Run frozen V1 numerical science behind an ephemeral local adapter."""

        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        review_binding = policy.source_review_binding(review, source_review_file_sha256)
        token = object()
        claimed_reservation: AttemptReservationV8 | None = None

        def require(value: object) -> object:
            if value is not token:
                raise PermissionError("retained science lacks its local V8 execution token")
            policy.preflight_static_authority()
            policy.preflight_source_review(
                policy.CANONICAL_SOURCE_REVIEW_PATH,
                source_review_file_sha256,
            )
            return value

        def source_binding(value: object) -> dict[str, str]:
            require(value)
            return dict(review_binding)

        def reserve(value: object) -> AttemptReservationV8:
            nonlocal claimed_reservation
            require(value)
            if claimed_reservation is not None:
                raise FileExistsError("the sole V8 attempt was already reserved")
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
        retained_returned = False
        try:
            retained.policy = compatibility
            retained._reserve_attempt = reserve
            retained._terminate_failure = _terminate_failure
            retained._publish_success = _publish_success
            retained.decode_selected_rgb = decode
            summary = retained._run_training(token, rgb_workers=rgb_workers)
            retained_returned = True
            if claimed_reservation is None:
                raise RuntimeError(
                    "frozen V8 training returned without claiming its attempt"
                )
            _assert_owned_claim(claimed_reservation)
            launch_summary = {
                **summary,
                "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v8_launch_summary_v1",
            }
            return launch_summary, claimed_reservation
        except BaseException as error:
            training_error = error
            if retained_returned and claimed_reservation is not None:
                try:
                    _terminate_failure(
                        claimed_reservation,
                        error,
                        stage="training",
                    )
                except BaseException as terminal_error:
                    raise RuntimeError(
                        "post-training ownership failed and terminal receipt "
                        "could not be written"
                    ) from terminal_error
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


    def _artifact_args(
        reservation: AttemptReservationV8,
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


    def _self_hashed_value(
        value: Mapping[str, Any],
        *,
        name: str,
    ) -> dict[str, Any]:
        result = dict(value)
        core = dict(result)
        declared = core.pop("content_sha256", None)
        if (
            not policy.is_sha256(declared)
            or policy.canonical_json_sha256(core) != declared
        ):
            raise ValueError(f"{name} self-hash changed")
        return result


    def _child_environment_projection(
        environment: Mapping[str, str],
    ) -> dict[str, Any]:
        return {
            "hip_visible_devices": environment.get("HIP_VISIBLE_DEVICES"),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "rocr_visible_devices": environment.get("ROCR_VISIBLE_DEVICES"),
            "gpu_device_ordinal": environment.get("GPU_DEVICE_ORDINAL"),
            "hsa_visible_devices": environment.get("HSA_VISIBLE_DEVICES"),
            "hsa_override_gfx_version": environment.get(
                "HSA_OVERRIDE_GFX_VERSION"
            ),
            "pythonhome": environment.get("PYTHONHOME"),
            "pythonpath": environment.get("PYTHONPATH"),
            "pythonstartup": environment.get("PYTHONSTARTUP"),
            "pythonuserbase": environment.get("PYTHONUSERBASE"),
            "python_no_user_site": environment.get("PYTHONNOUSERSITE"),
            "native_thread_environment": {
                name: environment.get(name) for name in policy.THREAD_ENVIRONMENT
            },
        }


    def _expected_child_environment() -> dict[str, Any]:
        return {
            "hip_visible_devices": "0",
            "cuda_visible_devices": None,
            "rocr_visible_devices": None,
            "gpu_device_ordinal": None,
            "hsa_visible_devices": None,
            "hsa_override_gfx_version": None,
            "pythonhome": None,
            "pythonpath": None,
            "pythonstartup": None,
            "pythonuserbase": None,
            "python_no_user_site": "1",
            "native_thread_environment": {
                name: "1" for name in policy.THREAD_ENVIRONMENT
            },
        }


    def _verification_child_environment() -> dict[str, str]:
        environment = dict(os.environ)
        for name in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "CUDA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_VISIBLE_DEVICES",
            "HSA_OVERRIDE_GFX_VERSION",
        ):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["HIP_VISIBLE_DEVICES"] = "0"
        for name in policy.THREAD_ENVIRONMENT:
            environment[name] = "1"
        if _child_environment_projection(environment) != _expected_child_environment():
            raise PermissionError("V8 verifier child environment sanitization failed")
        return environment


    def _artifact_argument_map(args: argparse.Namespace) -> dict[str, str]:
        return {
            name: str(getattr(args, name))
            for name in ("reservation", "result", "checkpoint", "completion")
        }


    def _verification_request(
        reservation: AttemptReservationV8,
        source_review_file_sha256: str,
    ) -> tuple[dict[str, Any], bytes]:
        if not sys.flags.isolated:
            raise PermissionError("V8 verifier parent must already be isolated")
        _assert_owned_claim(reservation)
        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        source_review = policy.source_review_binding(
            review, source_review_file_sha256
        )
        arguments = _artifact_argument_map(
            _artifact_args(reservation, source_review_file_sha256)
        )
        core = {
            "schema": policy.VERIFICATION_REQUEST_SCHEMA,
            "nonce": secrets.token_hex(32),
            "source_review": source_review,
            "sources": {
                name: dict(binding)
                for name, binding in review["successor_sources"].items()
            },
            "artifacts": arguments,
            "process": {
                "parent_pid": os.getpid(),
                "expected_executable": str(Path(sys.executable).resolve()),
                "expected_executor": str(Path(__file__).resolve()),
                "expected_child_mode": "verification_child",
                "expected_isolated": True,
                "expected_no_bytecode": True,
            },
            "environment": _expected_child_environment(),
            "contract": policy.isolated_verifier_contract(),
        }
        request = {**core, "content_sha256": policy.canonical_json_sha256(core)}
        raw = policy.canonical_json_bytes(request) + b"\n"
        if len(raw) > policy.VERIFICATION_MAX_REQUEST_BYTES:
            raise ValueError("V8 verifier request exceeds its frozen size bound")
        _assert_owned_claim(reservation)
        return request, raw


    def _validate_verification_request(
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = _self_hashed_value(request, name="V8 verifier request")
        expected_fields = {
            "schema",
            "nonce",
            "source_review",
            "sources",
            "artifacts",
            "process",
            "environment",
            "contract",
            "content_sha256",
        }
        if (
            set(value) != expected_fields
            or value.get("schema") != policy.VERIFICATION_REQUEST_SCHEMA
            or not policy.is_sha256(value.get("nonce"))
            or value.get("contract") != policy.isolated_verifier_contract()
            or value.get("environment") != _expected_child_environment()
        ):
            raise PermissionError("V8 verifier request contract changed")
        process = value.get("process")
        if (
            not isinstance(process, Mapping)
            or set(process)
            != {
                "parent_pid",
                "expected_executable",
                "expected_executor",
                "expected_child_mode",
                "expected_isolated",
                "expected_no_bytecode",
            }
            or isinstance(process.get("parent_pid"), bool)
            or not isinstance(process.get("parent_pid"), int)
            or process.get("parent_pid", 0) <= 1
            or process.get("expected_executable")
            != str(Path(sys.executable).resolve())
            or process.get("expected_executor") != str(Path(__file__).resolve())
            or process.get("expected_child_mode") != "verification_child"
            or process.get("expected_isolated") is not True
            or process.get("expected_no_bytecode") is not True
        ):
            raise PermissionError("V8 verifier request process binding changed")
        source_review = value.get("source_review")
        if not isinstance(source_review, Mapping):
            raise PermissionError("V8 verifier request source review is malformed")
        review, _ = policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            str(source_review.get("file_sha256")),
        )
        if (
            source_review != policy.source_review_binding(
                review, str(source_review.get("file_sha256"))
            )
            or value.get("sources") != review.get("successor_sources")
        ):
            raise PermissionError("V8 verifier request source binding changed")
        artifacts = value.get("artifacts")
        expected_paths = {
            "reservation": policy.CANONICAL_ATTEMPT_PATH / "reservation.json",
            "result": policy.CANONICAL_ATTEMPT_PATH / "result.json",
            "checkpoint": policy.CANONICAL_ATTEMPT_PATH / "checkpoint.pt",
            "completion": policy.CANONICAL_ATTEMPT_PATH / "completed.json",
        }
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_paths):
            raise PermissionError("V8 verifier request artifact map changed")
        for role, expected_path in expected_paths.items():
            bound = artifacts.get(role)
            if not isinstance(bound, str):
                raise PermissionError("V8 verifier artifact binding is malformed")
            path, digest = policy.parse_bound_path(bound)
            if path != expected_path.resolve(strict=True) or not policy.is_sha256(digest):
                raise PermissionError("V8 verifier artifact binding changed")
        policy.preflight_static_authority()
        return value


    def _compute_verification_receipt_child(
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = _validate_verification_request(request)
        if (
            not sys.flags.isolated
            or not sys.flags.dont_write_bytecode
            or os.getppid() != value["process"]["parent_pid"]
            or _child_environment_projection(os.environ)
            != _expected_child_environment()
        ):
            raise PermissionError("V8 verifier child process boundary changed")
        source_review = value["source_review"]
        source_review_file_sha256 = str(source_review["file_sha256"])
        token = object()

        def require(candidate: object) -> object:
            if candidate is not token:
                raise PermissionError(
                    "retained verifier lacks its local V8 child token"
                )
            policy.preflight_static_authority()
            policy.preflight_source_review(
                policy.CANONICAL_SOURCE_REVIEW_PATH,
                source_review_file_sha256,
            )
            return candidate

        def source_binding(candidate: object) -> dict[str, str]:
            require(candidate)
            return dict(source_review)

        def forbid_publication(*_args: Any, **_kwargs: Any) -> None:
            raise PermissionError("V8 verifier child cannot publish")

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
        compatibility.LOSS_WEIGHTS = {
            name: 0.25 for name in policy.LOSS_COMPONENTS
        }
        compatibility.require_verified_authority = require
        compatibility.source_review_binding = source_binding
        compatibility.write_exclusive = forbid_publication

        from scripts import (
            verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as verifier,
        )

        original_policy = verifier.policy
        try:
            verifier.policy = compatibility
            args = argparse.Namespace(
                source_review=policy.CANONICAL_SOURCE_REVIEW_PATH,
                source_review_sha256=source_review_file_sha256,
                **dict(value["artifacts"]),
            )
            bundle = verifier._validate_attempt_bundle(token, args)
            receipt = verifier._compute_receipt(token, bundle)
        finally:
            verifier.policy = original_policy
        return receipt


    def _verification_response(
        request: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_value = _validate_verification_request(request)
        core = {
            "schema": policy.VERIFICATION_RESPONSE_SCHEMA,
            "status": "verified_compute_only",
            "nonce": request_value["nonce"],
            "request_content_sha256": request_value["content_sha256"],
            "process": {
                "child_pid": os.getpid(),
                "parent_pid": os.getppid(),
                "executable": str(Path(sys.executable).resolve()),
                "executor": str(Path(__file__).resolve()),
                "mode": "verification_child",
                "isolated": bool(sys.flags.isolated),
                "no_bytecode": bool(sys.flags.dont_write_bytecode),
            },
            "environment": _child_environment_projection(os.environ),
            "sources": request_value["sources"],
            "artifacts": request_value["artifacts"],
            "receipt": dict(receipt),
            "receipt_content_sha256": receipt.get("content_sha256"),
            "publication_performed": False,
        }
        return {**core, "content_sha256": policy.canonical_json_sha256(core)}


    def _load_claim_json(
        reservation: AttemptReservationV8,
        name: str,
    ) -> tuple[dict[str, Any], bytes]:
        raw = _read_claim_file(reservation, name)
        value = policy.parse_json(raw, name=f"V8 parent {name}")
        if raw != policy.canonical_json_bytes(value) + b"\n":
            raise ValueError(f"V8 parent {name} is not canonical JSON")
        return _self_hashed_value(value, name=f"V8 parent {name}"), raw


    def _validate_child_metric_receipt(
        reservation: AttemptReservationV8,
        request: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = _self_hashed_value(receipt, name="V8 child metric receipt")
        expected_fields = {
            "schema",
            "authoritative",
            "aggregation_eligible",
            "promotion_eligible",
            "dataset_role",
            "seed",
            "fit_size",
            "authority_bindings",
            "source_review",
            "artifacts",
            "result_content_sha256",
            "target_partition",
            "target_partition_signature",
            "target_partition_signature_sha256",
            "recomputed_evaluation",
            "recomputed_evaluation_sha256",
            "numeric_gate",
            "verification",
            "resource",
            "access_ledger",
            "licenses",
            "content_sha256",
        }
        reservation_value, reservation_raw = _load_claim_json(
            reservation, "reservation.json"
        )
        result, result_raw = _load_claim_json(reservation, "result.json")
        completion, completion_raw = _load_claim_json(
            reservation, "completed.json"
        )
        checkpoint_raw = _read_claim_file(reservation, "checkpoint.pt")
        source_review = dict(request["source_review"])
        policy.validate_reservation_structure(
            reservation_value,
            expected_source_review=source_review,
        )
        policy.validate_result_structure(
            result,
            expected_source_review=source_review,
        )
        checkpoint_content = result["model"]["checkpoint"]["content_sha256"]
        expected_artifacts = {
            "reservation": policy.artifact_binding(
                "reservation.json",
                reservation_raw,
                content_sha256=reservation_value["content_sha256"],
            ),
            "result": policy.artifact_binding(
                "result.json",
                result_raw,
                content_sha256=result["content_sha256"],
            ),
            "checkpoint": policy.artifact_binding(
                "checkpoint.pt",
                checkpoint_raw,
                content_sha256=checkpoint_content,
            ),
            "completion": policy.artifact_binding(
                "completed.json",
                completion_raw,
                content_sha256=completion["content_sha256"],
            ),
        }
        for role, binding in expected_artifacts.items():
            _path, digest = policy.parse_bound_path(str(request["artifacts"][role]))
            if digest != binding["file_sha256"]:
                raise PermissionError("V8 child request artifact changed")
        expected_verification = {
            "checkpoint_loaded": True,
            "checkpoint_state_manifest_rehashed": True,
            "checkpoint_final_update_binding_validated": True,
            "fresh_model_loaded_for_inference": True,
            "selected_train_targets_loaded": True,
            "selected_matched_rgb_loaded": True,
            "wrong_rgb_mapping_rerun": True,
            "evaluation_losses_recomputed": True,
            "evaluation_loss_arithmetic_validated": True,
            "all_confusions_recomputed": True,
            "depth_quantiles_and_sorted_commitments_recomputed": True,
            "raster_nll_recomputed": True,
            "family_metrics_recomputed": True,
            "frozen_thresholds_recomputed": True,
            "result_metrics_reused": False,
            "metric_repair_applied": False,
            "threshold_weakened": False,
        }
        expected_access = {
            "selected_rgb_count": 5,
            "selected_rgb_hash_opens": 5,
            "selected_rgb_decodes": 5,
            "checkpoint_opens": 1,
            "heldout_opens": 0,
            "g2_opens": 0,
            "selection_opens": 0,
            "calibration_opens": 0,
            "runtime_opens": 0,
            "hardware_opens": 0,
            "production_opens": 0,
            "gpu1_uses": 0,
        }
        expected_licenses = {
            "checkpoint_use_authorized_for_metric_verification_only": True,
            "development_checkpoint_use_authorized": False,
            "new_model_output_authorized": False,
            "retry_authorized": False,
            "n16_execution_authorized": False,
            "second_seed_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        }
        evaluation = value.get("recomputed_evaluation")
        policy.validate_evaluation_structure(evaluation)
        if (
            set(value) != expected_fields
            or value.get("schema") != policy.METRIC_RECEIPT_SCHEMA
            or value.get("authoritative") is not False
            or value.get("aggregation_eligible") is not False
            or value.get("promotion_eligible") is not False
            or value.get("dataset_role") != "train"
            or value.get("seed") != 20260710
            or value.get("fit_size") != 5
            or value.get("authority_bindings") != policy.authority_bindings()
            or value.get("source_review") != source_review
            or value.get("artifacts") != expected_artifacts
            or value.get("result_content_sha256") != result["content_sha256"]
            or value.get("target_partition") != result["target_partition"]
            or evaluation != result["evaluation"]
            or value.get("recomputed_evaluation_sha256")
            != policy.canonical_json_sha256(evaluation)
            or value.get("verification") != expected_verification
            or value.get("access_ledger") != expected_access
            or value.get("licenses") != expected_licenses
            or value.get("resource") != result["resource"]
        ):
            raise PermissionError("V8 child metric receipt provenance changed")
        from lewm.benchmarks import (
            go2_observable_camera_ray_fit_v4_ladder_gate as frozen,
        )

        matched, wrong, signature = frozen._validated_metric_evaluation(
            evaluation, fit_size=5
        )
        numeric = frozen._gate_stage(
            {"fit_size": 5, "matched": matched, "wrong": wrong}
        )
        if (
            value.get("target_partition_signature") != signature
            or value.get("target_partition_signature_sha256")
            != policy.canonical_json_sha256(signature)
            or value.get("numeric_gate") != numeric
        ):
            raise ValueError("V8 child frozen metric decision changed")
        _assert_owned_claim(reservation)
        return value


    def _validate_verification_response(
        reservation: AttemptReservationV8,
        request: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = _self_hashed_value(response, name="V8 verifier response")
        expected_fields = {
            "schema",
            "status",
            "nonce",
            "request_content_sha256",
            "process",
            "environment",
            "sources",
            "artifacts",
            "receipt",
            "receipt_content_sha256",
            "publication_performed",
            "content_sha256",
        }
        process = value.get("process")
        if (
            set(value) != expected_fields
            or value.get("schema") != policy.VERIFICATION_RESPONSE_SCHEMA
            or value.get("status") != "verified_compute_only"
            or value.get("nonce") != request.get("nonce")
            or value.get("request_content_sha256")
            != request.get("content_sha256")
            or value.get("environment") != _expected_child_environment()
            or value.get("sources") != request.get("sources")
            or value.get("artifacts") != request.get("artifacts")
            or value.get("publication_performed") is not False
            or not isinstance(process, Mapping)
            or set(process)
            != {
                "child_pid",
                "parent_pid",
                "executable",
                "executor",
                "mode",
                "isolated",
                "no_bytecode",
            }
            or isinstance(process.get("child_pid"), bool)
            or not isinstance(process.get("child_pid"), int)
            or process.get("child_pid", 0) <= 1
            or process.get("child_pid") == os.getpid()
            or process.get("parent_pid") != os.getpid()
            or process.get("executable") != str(Path(sys.executable).resolve())
            or process.get("executor") != str(Path(__file__).resolve())
            or process.get("mode") != "verification_child"
            or process.get("isolated") is not True
            or process.get("no_bytecode") is not True
        ):
            raise PermissionError("V8 verifier response binding changed")
        receipt = value.get("receipt")
        if not isinstance(receipt, Mapping):
            raise ValueError("V8 verifier response lacks a receipt")
        checked = _validate_child_metric_receipt(
            reservation,
            request,
            receipt,
        )
        if value.get("receipt_content_sha256") != checked["content_sha256"]:
            raise PermissionError("V8 verifier response receipt binding changed")
        return checked


    def _run_independent_verification(
        reservation: AttemptReservationV8,
        source_review_file_sha256: str,
    ) -> dict[str, Any]:
        request, request_raw = _verification_request(
            reservation, source_review_file_sha256
        )
        command = [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            "--verification-child",
        ]
        try:
            completed = subprocess.run(
                command,
                input=request_raw,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=ROOT,
                env=_verification_child_environment(),
                check=False,
                timeout=policy.VERIFICATION_TIMEOUT_SECONDS,
                close_fds=True,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("V8 verifier child timed out; no fallback") from error
        if completed.returncode != 0:
            raise RuntimeError(
                "V8 verifier child failed or was signaled; no fallback"
            )
        if completed.stderr != b"":
            raise RuntimeError("V8 verifier child emitted stderr; no fallback")
        response_raw = bytes(completed.stdout)
        if (
            not response_raw
            or len(response_raw) > policy.VERIFICATION_MAX_RESPONSE_BYTES
        ):
            raise ValueError("V8 verifier child response size changed")
        response = policy.parse_json(response_raw, name="V8 verifier child response")
        if response_raw != policy.canonical_json_bytes(response) + b"\n":
            raise ValueError("V8 verifier child response is not one canonical JSON value")
        receipt = _validate_verification_response(
            reservation,
            request,
            response,
        )
        _write_canonical_json(
            reservation,
            policy.CANONICAL_METRIC_RECEIPT_PATH,
            receipt,
        )
        _assert_owned_claim(reservation)
        return receipt


    def _verification_child_main() -> int:
        if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
            raise PermissionError("V8 verifier child requires -I -B")
        raw = sys.stdin.buffer.read(policy.VERIFICATION_MAX_REQUEST_BYTES + 1)
        if not raw or len(raw) > policy.VERIFICATION_MAX_REQUEST_BYTES:
            raise ValueError("V8 verifier child request size changed")
        request = policy.parse_json(raw, name="V8 verifier child request")
        if raw != policy.canonical_json_bytes(request) + b"\n":
            raise ValueError("V8 verifier child request is not canonical JSON")
        receipt = _compute_verification_receipt_child(request)
        response = _verification_response(request, receipt)
        response_raw = policy.canonical_json_bytes(response) + b"\n"
        if len(response_raw) > policy.VERIFICATION_MAX_RESPONSE_BYTES:
            raise ValueError("V8 verifier child response exceeds its size bound")
        sys.stdout.buffer.write(response_raw)
        sys.stdout.buffer.flush()
        return 0


    def _run_finalization(
        reservation: AttemptReservationV8,
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
                raise PermissionError("retained finalizer lacks its local V8 execution token")
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
                raise PermissionError("retained finalizer changed its V8 output path")
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
            metric_raw = _read_owned_derived_artifact(
                reservation,
                policy.CANONICAL_METRIC_RECEIPT_PATH,
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
            raise PermissionError("N5 full-panel V8 exact execution requires isolation")
        if isinstance(rgb_workers, bool) or not 1 <= int(rgb_workers) <= 5:
            raise ValueError("N5 full-panel RGB workers must lie in [1,5]")
        if (
            policy.CANONICAL_ATTEMPT_PATH.exists()
            or policy.CANONICAL_ATTEMPT_PATH.is_symlink()
        ):
            raise FileExistsError(
                "the sole N5 full-panel V8 recovery attempt is already claimed"
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
                "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v8_end_to_end_summary_v1",
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
                    f"N5 full-panel V8 {stage} failed and terminal receipt "
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
            raise ValueError("N5 full-panel V8 source review SHA-256 is malformed")
        return args


    def _isolated_child(argv: Sequence[str]) -> int:
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        if "--cpu-contract-smoke" in argv:
            for name in (
                "HIP_VISIBLE_DEVICES",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
                "HSA_VISIBLE_DEVICES",
            ):
                environment[name] = ""
        else:
            environment["HIP_VISIBLE_DEVICES"] = "0"
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
                "HSA_VISIBLE_DEVICES",
            ):
                environment.pop(name, None)
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
        if raw_argv == ["--verification-child"]:
            if not sys.flags.isolated:
                raise PermissionError(
                    "V8 verifier child cannot self-authorize isolation"
                )
            return _verification_child_main()
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
