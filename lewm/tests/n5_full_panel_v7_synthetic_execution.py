"""Production-ineligible V7 filesystem lifecycle used only by tests.

The synthetic lifecycle has fixed paths below a caller-supplied safe temporary
root. It models descriptor-relative declared transactions and permanent journal
poison, but it is deliberately not imported by the production executor.
Kernel inotify history is tested against the production journal itself.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_OUTPUT = (
    REPOSITORY_ROOT
    / ".generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v7"
).resolve()
STAGING_PREFIX = ".n5.synthetic-v7-"
LOCK_NAME = ".n5.synthetic-v7.lock"
SERIALIZER_SUFFIX = ".n5.synthetic-v7.serializer"
SCHEMA = "lewm_go2_n5_full_panel_v7_synthetic_reservation_v1"
MANIFEST_SCHEMA = "lewm_go2_n5_full_panel_v7_synthetic_staging_v1"


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
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


def _identity_security(metadata: os.stat_result) -> tuple[int, ...]:
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
        raise PermissionError("synthetic V7 requires no-follow directory opens")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _is_staging_name(name: str) -> bool:
    if not name.startswith(STAGING_PREFIX):
        return False
    suffix = name[len(STAGING_PREFIX) :]
    return len(suffix) == 32 and all(
        character in "0123456789abcdef" for character in suffix
    )


@dataclass(frozen=True)
class _SnapshotV7:
    fingerprint: tuple[int, ...]
    inventory: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass
class _WatchV7:
    directory_fd: int
    identity: tuple[int, int]
    label: str
    generation: int
    snapshot: _SnapshotV7


class _DeclaredJournalV7:
    """A deterministic test journal for declared filesystem state deltas."""

    def __init__(self) -> None:
        self._watches: dict[tuple[int, int], _WatchV7] = {}
        self._generation = 0
        self._poison: str | None = None
        self._closed = False

    @property
    def poisoned(self) -> bool:
        return self._poison is not None

    @property
    def poison_reason(self) -> str | None:
        return self._poison

    def poison_event(self, reason: str) -> None:
        if self._poison is None:
            self._poison = str(reason)

    def _fail(self, reason: str) -> None:
        self.poison_event(reason)
        raise PermissionError(f"synthetic V7 journal rejected: {reason}")

    @staticmethod
    def _identity(directory_fd: int) -> tuple[int, int]:
        metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError("synthetic watch target is not a directory")
        return (metadata.st_dev, metadata.st_ino)

    @staticmethod
    def _snapshot(directory_fd: int) -> _SnapshotV7:
        opened = os.fstat(directory_fd)
        rows = tuple(
            (
                name,
                _fingerprint(
                    os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                ),
            )
            for name in sorted(os.listdir(directory_fd))
        )
        return _SnapshotV7(_fingerprint(opened), rows)

    def watch(self, directory_fd: int, *, label: str) -> None:
        self.assert_clean()
        identity = self._identity(directory_fd)
        if identity in self._watches:
            return
        retained = os.dup(directory_fd)
        self._generation += 1
        self._watches[identity] = _WatchV7(
            retained,
            identity,
            label,
            self._generation,
            self._snapshot(retained),
        )

    def state(self, directory_fd: int) -> _WatchV7:
        state = self._watches.get(self._identity(directory_fd))
        if state is None:
            self._fail("exclusive directory lacks a retained generation")
        return state

    def baseline(self, directory_fd: int) -> _SnapshotV7:
        return self.state(directory_fd).snapshot

    def assert_clean(self) -> None:
        if self._closed:
            self._fail("journal is closed")
        if self._poison is not None:
            self._fail(self._poison)
        for state in self._watches.values():
            if self._snapshot(state.directory_fd) != state.snapshot:
                self._fail(f"undeclared mutation in {state.label}")

    def _snapshots(self) -> dict[tuple[int, int], _SnapshotV7]:
        return {
            identity: self._snapshot(state.directory_fd)
            for identity, state in self._watches.items()
        }

    def _begin(
        self, directory_fd: int
    ) -> tuple[_WatchV7, dict[tuple[int, int], _SnapshotV7]]:
        self.assert_clean()
        state = self.state(directory_fd)
        return state, self._snapshots()

    @staticmethod
    def _delta(
        before: _SnapshotV7,
        *,
        remove: set[str] = frozenset(),
        add: Mapping[str, tuple[int, ...]] | None = None,
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        rows = dict(before.inventory)
        for name in remove:
            rows.pop(name, None)
        if add:
            rows.update(add)
        return tuple(sorted(rows.items()))

    def _finish(
        self,
        before: Mapping[tuple[int, int], _SnapshotV7],
        *,
        direct_inventories: Mapping[
            tuple[int, int], tuple[tuple[str, tuple[int, ...]], ...]
        ],
        direct_fingerprint_changes: set[tuple[int, int]],
    ) -> None:
        after = self._snapshots()
        if set(before) != set(after):
            self._fail("watch generation set changed during transaction")
        changed = {
            identity: value.fingerprint
            for identity, value in after.items()
            if value.fingerprint != before[identity].fingerprint
        }
        for identity, prior in before.items():
            current = after[identity]
            if _identity_security_from_fingerprint(current.fingerprint) != (
                _identity_security_from_fingerprint(prior.fingerprint)
            ):
                self._fail("directory identity/security changed")
            if (
                identity not in direct_fingerprint_changes
                and current.fingerprint != prior.fingerprint
            ):
                self._fail("undeclared directory fingerprint change")
            expected = direct_inventories.get(identity)
            if expected is None:
                expected = tuple(
                    (
                        name,
                        changed.get((fingerprint[0], fingerprint[1]), fingerprint),
                    )
                    for name, fingerprint in prior.inventory
                )
            if current.inventory != expected:
                self._fail("post-state is not the declared delta")
        for identity, snapshot in after.items():
            self._watches[identity].snapshot = snapshot

    def mkdir(self, parent_fd: int, name: str) -> None:
        state, before = self._begin(parent_fd)
        prior = before[state.identity]
        if name in dict(prior.inventory):
            raise FileExistsError(name)
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        self._finish(
            before,
            direct_inventories={
                state.identity: self._delta(
                    prior, add={name: _fingerprint(metadata)}
                )
            },
            direct_fingerprint_changes={state.identity},
        )

    def create(self, parent_fd: int, name: str, raw: bytes) -> tuple[int, ...]:
        state, before = self._begin(parent_fd)
        prior = before[state.identity]
        if name in dict(prior.inventory):
            raise FileExistsError(name)
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                self._fail("created leaf is not singly linked")
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short synthetic write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
        committed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        fingerprint = _fingerprint(committed)
        self._finish(
            before,
            direct_inventories={
                state.identity: self._delta(prior, add={name: fingerprint})
            },
            direct_fingerprint_changes={state.identity},
        )
        return fingerprint

    def replace(self, parent_fd: int, name: str, raw: bytes) -> tuple[int, ...]:
        temporary = f".{name}.replace-{secrets.token_hex(12)}"
        self.create(parent_fd, temporary, raw)
        state, before = self._begin(parent_fd)
        prior = before[state.identity]
        rows = dict(prior.inventory)
        if name not in rows or temporary not in rows:
            self._fail("replace inputs changed")
        os.replace(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        fingerprint = _fingerprint(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
        self._finish(
            before,
            direct_inventories={
                state.identity: self._delta(
                    prior,
                    remove={temporary, name},
                    add={name: fingerprint},
                )
            },
            direct_fingerprint_changes={state.identity},
        )
        return fingerprint

    def unlink(
        self,
        parent_fd: int,
        name: str,
        *,
        expected_fingerprint: tuple[int, ...],
    ) -> None:
        state, before = self._begin(parent_fd)
        prior = before[state.identity]
        if dict(prior.inventory).get(name) != expected_fingerprint:
            self._fail("unlink fingerprint changed")
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        self._finish(
            before,
            direct_inventories={
                state.identity: self._delta(prior, remove={name})
            },
            direct_fingerprint_changes={state.identity},
        )

    def rename_directory(
        self,
        parent_fd: int,
        source: str,
        destination: str,
        *,
        directory_fd: int,
    ) -> None:
        parent, before = self._begin(parent_fd)
        child = self.state(directory_fd)
        parent_prior = before[parent.identity]
        child_prior = before[child.identity]
        rows = dict(parent_prior.inventory)
        if source not in rows or destination in rows:
            self._fail("rename inputs changed")
        os.rename(
            source,
            destination,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
        destination_fp = _fingerprint(
            os.stat(destination, dir_fd=parent_fd, follow_symlinks=False)
        )
        self._finish(
            before,
            direct_inventories={
                parent.identity: self._delta(
                    parent_prior,
                    remove={source},
                    add={destination: destination_fp},
                ),
                child.identity: child_prior.inventory,
            },
            direct_fingerprint_changes={parent.identity, child.identity},
        )

    def rmdir(self, parent_fd: int, name: str, *, directory_fd: int) -> None:
        parent, before = self._begin(parent_fd)
        child = self.state(directory_fd)
        parent_prior = before[parent.identity]
        child_prior = before[child.identity]
        if child_prior.inventory:
            self._fail("rmdir target is not empty")
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        self._finish(
            before,
            direct_inventories={
                parent.identity: self._delta(parent_prior, remove={name}),
                child.identity: (),
            },
            direct_fingerprint_changes={parent.identity, child.identity},
        )
        self._watches.pop(child.identity)
        os.close(child.directory_fd)
        os.close(directory_fd)

    def terminal_create(
        self, parent_fd: int, name: str, raw: bytes
    ) -> tuple[int, ...]:
        if not self.poisoned:
            return self.create(parent_fd, name, raw)
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short synthetic terminal write")
                offset += written
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
        return _fingerprint(metadata)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for state in self._watches.values():
            os.close(state.directory_fd)
        self._watches.clear()


def _identity_security_from_fingerprint(
    fingerprint: tuple[int, ...],
) -> tuple[int, ...]:
    return (
        fingerprint[0],
        fingerprint[1],
        fingerprint[2],
        fingerprint[4],
        fingerprint[5],
    )


@dataclass
class _AncestorEntryV7:
    parent_fd: int
    name: str
    child_fd: int
    identity_security: tuple[int, ...]


@dataclass
class _AncestorChainV7:
    descriptors: list[int]
    entries: list[_AncestorEntryV7]
    closed: bool = False


def _open_ancestor_chain(path: Path) -> _AncestorChainV7:
    filesystem_root = Path(path.anchor)
    anchor_fd = os.open(filesystem_root, _directory_flags())
    chain = _AncestorChainV7([anchor_fd], [])
    parent_fd = anchor_fd
    try:
        for component in path.parts[1:]:
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                raise PermissionError("synthetic ancestor is not a real directory")
            identity = _identity_security(named)
            child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
            chain.descriptors.append(child_fd)
            if _identity_security(os.fstat(child_fd)) != identity:
                raise PermissionError("synthetic ancestor changed during open")
            chain.entries.append(
                _AncestorEntryV7(parent_fd, component, child_fd, identity)
            )
            parent_fd = child_fd
        return chain
    except BaseException:
        _close_ancestor_chain(chain)
        raise


def _assert_ancestor_chain(chain: _AncestorChainV7) -> None:
    if chain.closed:
        raise PermissionError("synthetic ancestor chain is closed")
    for entry in chain.entries:
        named = os.stat(entry.name, dir_fd=entry.parent_fd, follow_symlinks=False)
        opened = os.fstat(entry.child_fd)
        if (
            stat.S_ISLNK(named.st_mode)
            or _identity_security(named) != entry.identity_security
            or _identity_security(opened) != entry.identity_security
        ):
            raise PermissionError("synthetic canonical directory identity changed")


def _close_ancestor_chain(chain: _AncestorChainV7) -> None:
    if chain.closed:
        return
    chain.closed = True
    for descriptor in reversed(chain.descriptors):
        os.close(descriptor)


def _safe_root(root: Path) -> Path:
    resolved = Path(root).resolve()
    if (
        resolved == REPOSITORY_ROOT
        or resolved.is_relative_to(REPOSITORY_ROOT)
        or REPOSITORY_ROOT.is_relative_to(resolved)
        or resolved == CANONICAL_OUTPUT
        or resolved.is_relative_to(CANONICAL_OUTPUT)
        or CANONICAL_OUTPUT.is_relative_to(resolved)
    ):
        raise PermissionError("synthetic executor cannot target production or repository")
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    canonical = resolved.resolve(strict=True)
    metadata = canonical.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
    ):
        raise PermissionError("synthetic exclusive root is not private")
    return canonical


@dataclass(frozen=True)
class _OwnedArtifactV7:
    role: str
    parent_fd: int
    name: str
    fingerprint: tuple[int, ...]


@dataclass(frozen=True)
class SyntheticReservationV7:
    directory: Path
    value: Mapping[str, Any]
    raw: bytes
    directory_fd: int
    directory_identity: tuple[int, int]
    root_fd: int
    seed_root_fd: int
    derived_fd: int
    journal: _DeclaredJournalV7
    ancestor_chain: _AncestorChainV7
    owned_directory_fds: tuple[int, ...]
    owned_claim_artifacts: dict[str, _OwnedArtifactV7] = field(
        default_factory=dict, compare=False
    )
    owned_derived_artifacts: dict[str, _OwnedArtifactV7] = field(
        default_factory=dict, compare=False
    )


class SyntheticExecutionV7:
    """Fixed-path, temporary-root-only lifecycle model."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        object.__setattr__(self, "_root", _safe_root(root))

    @property
    def root(self) -> Path:
        return Path(object.__getattribute__(self, "_root"))

    @property
    def attempt(self) -> Path:
        return self.root / "attempts/seed_20260710/n5"

    def __copy__(self) -> "SyntheticExecutionV7":
        return type(self)(self.root)

    def __deepcopy__(self, memo: object) -> "SyntheticExecutionV7":
        del memo
        return type(self)(self.root)

    @contextmanager
    def _serializer(self) -> Iterator[None]:
        path = self.root.parent / f".{self.root.name}{SERIALIZER_SUFFIX}"
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _open_child(
        parent_fd: int,
        parent_path: Path,
        name: str,
        journal: _DeclaredJournalV7,
    ) -> tuple[Path, int]:
        if name not in dict(journal.baseline(parent_fd).inventory):
            journal.mkdir(parent_fd, name)
        child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
        metadata = os.fstat(child_fd)
        if (
            stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
            or metadata.st_gid != os.getgid()
        ):
            os.close(child_fd)
            journal._fail("synthetic exclusive child is not private")
        journal.watch(child_fd, label=str(parent_path / name))
        return parent_path / name, child_fd

    def _open_lifecycle(
        self,
    ) -> tuple[
        _AncestorChainV7,
        _DeclaredJournalV7,
        int,
        int,
        int,
        list[int],
        bool,
    ]:
        chain = _open_ancestor_chain(self.root)
        root_fd = chain.entries[-1].child_fd
        journal = _DeclaredJournalV7()
        journal.watch(root_fd, label=str(self.root))
        initial_root_names = set(dict(journal.baseline(root_fd).inventory))
        fresh_scaffold = not initial_root_names
        if not fresh_scaffold and initial_root_names != {"attempts", "derived"}:
            journal._fail("synthetic recovery inventory is unproved")
        owned_descriptors: list[int] = []
        attempts, attempts_fd = self._open_child(
            root_fd, self.root, "attempts", journal
        )
        owned_descriptors.append(attempts_fd)
        if (
            not fresh_scaffold
            and set(dict(journal.baseline(attempts_fd).inventory))
            != {"seed_20260710"}
        ):
            journal._fail("synthetic recovery inventory is unproved")
        seed, seed_fd = self._open_child(
            attempts_fd, attempts, "seed_20260710", journal
        )
        owned_descriptors.append(seed_fd)
        _derived, derived_fd = self._open_child(
            root_fd, self.root, "derived", journal
        )
        owned_descriptors.append(derived_fd)
        root_names = set(dict(journal.baseline(root_fd).inventory))
        attempts_names = set(dict(journal.baseline(attempts_fd).inventory))
        seed_names = set(dict(journal.baseline(seed_fd).inventory))
        derived_names = set(dict(journal.baseline(derived_fd).inventory))
        unknown_seed = {
            name
            for name in seed_names
            if name not in {LOCK_NAME, self.attempt.name}
            and not _is_staging_name(name)
        }
        if (
            root_names != {"attempts", "derived"}
            or attempts_names != {"seed_20260710"}
            or unknown_seed
            or (self.attempt.name not in seed_names and derived_names)
        ):
            journal._fail("synthetic recovery inventory is unproved")
        if LOCK_NAME not in dict(journal.baseline(seed_fd).inventory):
            if not fresh_scaffold:
                journal._fail("synthetic existing recovery tree lacks its lock")
            journal.create(seed_fd, LOCK_NAME, b"")
        journal.assert_clean()
        _assert_ancestor_chain(chain)
        return (
            chain,
            journal,
            root_fd,
            seed_fd,
            derived_fd,
            owned_descriptors,
            fresh_scaffold,
        )

    def _core(self, recovery: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "status": "reserved",
            "attempt_path": str(self.attempt.resolve()),
            "attempt_index": 1,
            "maximum_attempts": 1,
            "production_eligible": False,
            "recovery": [dict(item) for item in recovery],
        }

    def _reservation_value(
        self, recovery: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], bytes]:
        core = self._core(recovery)
        value = {**core, "content_sha256": _digest(core)}
        return value, _json(value) + b"\n"

    def _manifest(self, staging_name: str, reservation_raw: bytes) -> bytes:
        core = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "staging_name": staging_name,
            "attempt_path": str(self.attempt.resolve()),
            "reservation_file_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        }
        return _json({**core, "content_sha256": _digest(core)}) + b"\n"

    @staticmethod
    def _recoverable_authority_core(value: Mapping[str, Any]) -> dict[str, Any]:
        core = dict(value)
        core.pop("content_sha256", None)
        core.pop("recovery", None)
        return core

    @staticmethod
    def _read_at(directory_fd: int, name: str) -> bytes:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            if _fingerprint(os.fstat(descriptor)) != _fingerprint(before):
                raise PermissionError("synthetic leaf changed while read")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    def _classify(
        self,
        staging_fd: int,
        staging_name: str,
        journal: _DeclaredJournalV7,
    ) -> tuple[dict[str, str], tuple[dict[str, Any], bytes] | None]:
        rows = dict(journal.baseline(staging_fd).inventory)
        evidence = {
            "staging_name": staging_name,
            "inventory_sha256": _digest(sorted(rows)),
            "classification": "incomplete",
            "action": "preserve_invalid_and_block_claim",
        }
        if set(rows) != {"reservation.json", "staging.json"}:
            return evidence, None
        try:
            for leaf_name in ("reservation.json", "staging.json"):
                metadata = os.stat(
                    leaf_name,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_uid != os.getuid()
                    or metadata.st_gid != os.getgid()
                    or metadata.st_size > 1024 * 1024
                ):
                    raise PermissionError("synthetic recovery leaf is not private")
            reservation_raw = self._read_at(staging_fd, "reservation.json")
            manifest_raw = self._read_at(staging_fd, "staging.json")
            reservation = json.loads(reservation_raw)
            manifest = json.loads(manifest_raw)
            reservation_core = dict(reservation)
            reservation_hash = reservation_core.pop("content_sha256")
            manifest_core = dict(manifest)
            manifest_hash = manifest_core.pop("content_sha256")
            recovery = reservation_core.pop("recovery", None)
            expected_reservation = self._core(())
            expected_reservation.pop("recovery")
            valid = (
                isinstance(recovery, list)
                and all(isinstance(item, Mapping) for item in recovery)
                and reservation_hash
                == _digest({**reservation_core, "recovery": recovery})
                and reservation_core == expected_reservation
                and manifest_hash == _digest(manifest_core)
                and reservation.get("schema") == SCHEMA
                and reservation.get("production_eligible") is False
                and reservation.get("attempt_path") == str(self.attempt.resolve())
                and manifest.get("schema") == MANIFEST_SCHEMA
                and manifest.get("staging_name") == staging_name
                and manifest.get("attempt_path") == str(self.attempt.resolve())
                and manifest.get("reservation_file_sha256")
                == hashlib.sha256(reservation_raw).hexdigest()
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            valid = False
        journal.assert_clean()
        if not valid:
            evidence["classification"] = "mutated"
            return evidence, None
        evidence["classification"] = "complete"
        evidence["action"] = "resume_after_rehash"
        return evidence, (reservation, reservation_raw)

    @staticmethod
    def _remove_flat_staging(
        seed_fd: int,
        staging_fd: int,
        staging_name: str,
        journal: _DeclaredJournalV7,
    ) -> None:
        for name, fingerprint in tuple(journal.baseline(staging_fd).inventory):
            metadata = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            if (
                _fingerprint(metadata) != fingerprint
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
            ):
                raise PermissionError("synthetic foreign staging is preserved invalid")
            journal.unlink(
                staging_fd, name, expected_fingerprint=fingerprint
            )
        journal.rmdir(seed_fd, staging_name, directory_fd=staging_fd)

    def _new_staging(
        self, seed_fd: int, journal: _DeclaredJournalV7
    ) -> tuple[str, int]:
        for _ in range(128):
            name = f"{STAGING_PREFIX}{secrets.token_hex(16)}"
            try:
                journal.mkdir(seed_fd, name)
            except FileExistsError:
                continue
            descriptor = os.open(name, _directory_flags(), dir_fd=seed_fd)
            journal.watch(descriptor, label=f"synthetic/{name}")
            return name, descriptor
        raise FileExistsError("synthetic staging namespace exhausted")

    @classmethod
    def _assert_claim_fd(cls, reservation: SyntheticReservationV7) -> None:
        metadata = os.fstat(reservation.directory_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != reservation.directory_identity
        ):
            raise PermissionError("synthetic claimed descriptor identity changed")

    @classmethod
    def _assert_claim(cls, reservation: SyntheticReservationV7) -> None:
        cls._assert_claim_fd(reservation)
        _assert_ancestor_chain(reservation.ancestor_chain)
        reservation.journal.assert_clean()
        named = os.stat(
            reservation.directory.name,
            dir_fd=reservation.seed_root_fd,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(named.st_mode)
            or (named.st_dev, named.st_ino) != reservation.directory_identity
        ):
            raise PermissionError("synthetic canonical claim identity changed")
        expected = {"reservation.json"} | set(reservation.owned_claim_artifacts)
        if set(os.listdir(reservation.directory_fd)) != expected:
            raise PermissionError("synthetic claimed directory inventory changed")

    @classmethod
    def _write_claim(
        cls,
        reservation: SyntheticReservationV7,
        name: str,
        raw: bytes,
        *,
        terminal: bool = False,
        role: str,
    ) -> None:
        if terminal:
            cls._assert_claim_fd(reservation)
            fingerprint = reservation.journal.terminal_create(
                reservation.directory_fd, name, raw
            )
        else:
            cls._assert_claim(reservation)
            fingerprint = reservation.journal.create(
                reservation.directory_fd, name, raw
            )
        reservation.owned_claim_artifacts[name] = _OwnedArtifactV7(
            role, reservation.directory_fd, name, fingerprint
        )
        if not terminal:
            cls._assert_claim(reservation)

    @classmethod
    def _write_derived(
        cls, reservation: SyntheticReservationV7, name: str, raw: bytes
    ) -> None:
        cls._assert_claim(reservation)
        fingerprint = reservation.journal.create(
            reservation.derived_fd, name, raw
        )
        reservation.owned_derived_artifacts[name] = _OwnedArtifactV7(
            "derived", reservation.derived_fd, name, fingerprint
        )
        cls._assert_claim(reservation)

    def prepare_complete_staging(self) -> Path:
        with self._serializer():
            (
                chain,
                journal,
                _root_fd,
                seed_fd,
                _derived_fd,
                descriptors,
                _fresh_scaffold,
            ) = self._open_lifecycle()
            try:
                name, staging_fd = self._new_staging(seed_fd, journal)
                recovery = [
                    {
                        "staging_name": name,
                        "classification": "new",
                        "action": "atomic_claim",
                        "inventory_sha256": _digest([]),
                    }
                ]
                _value, raw = self._reservation_value(recovery)
                journal.create(staging_fd, "reservation.json", raw)
                journal.create(staging_fd, "staging.json", self._manifest(name, raw))
                journal.assert_clean()
                os.close(staging_fd)
                return self.attempt.parent / name
            finally:
                journal.close()
                for descriptor in reversed(descriptors):
                    os.close(descriptor)
                _close_ancestor_chain(chain)

    def claim(
        self, *, failure_injection: str | None = None
    ) -> SyntheticReservationV7:
        with self._serializer():
            (
                chain,
                journal,
                root_fd,
                seed_fd,
                derived_fd,
                descriptors,
                fresh_scaffold,
            ) = self._open_lifecycle()
            active_fd: int | None = None
            active_name: str | None = None
            claimed: SyntheticReservationV7 | None = None
            candidate_fds: set[int] = set()
            try:
                if self.attempt.name in dict(journal.baseline(seed_fd).inventory):
                    raise FileExistsError("synthetic sole attempt is already claimed")
                candidates = sorted(
                    name
                    for name in dict(journal.baseline(seed_fd).inventory)
                    if _is_staging_name(name)
                )
                recovery: list[dict[str, Any]] = []
                complete: list[
                    tuple[str, int, tuple[dict[str, Any], bytes], dict[str, str]]
                ] = []
                for name in candidates:
                    descriptor = os.open(name, _directory_flags(), dir_fd=seed_fd)
                    candidate_fds.add(descriptor)
                    journal.watch(descriptor, label=f"synthetic/{name}")
                    evidence, recovered = self._classify(
                        descriptor, name, journal
                    )
                    if recovered is None:
                        os.close(descriptor)
                        candidate_fds.discard(descriptor)
                        journal._fail(
                            "synthetic foreign, incomplete, or mutated staging "
                            "is preserved invalid"
                        )
                    else:
                        complete.append((name, descriptor, recovered, evidence))
                if complete:
                    active_name, active_fd, recovered, selected = complete[0]
                    recovery.append(selected)
                    if any(
                        self._recoverable_authority_core(recovered[0])
                        != self._recoverable_authority_core(duplicate[2][0])
                        for duplicate in complete[1:]
                    ):
                        journal._fail(
                            "synthetic conflicting complete stagings are preserved "
                            "and block claim"
                        )
                    for (
                        duplicate_name,
                        duplicate_fd,
                        duplicate_recovered,
                        evidence,
                    ) in complete[1:]:
                        self._remove_flat_staging(
                            seed_fd,
                            duplicate_fd,
                            duplicate_name,
                            journal,
                        )
                        candidate_fds.discard(duplicate_fd)
                        recovery.append(
                            {
                                **evidence,
                                "classification": "complete_equivalent_duplicate",
                                "action": "remove_equivalent_duplicate_without_claim",
                                "reservation_file_sha256": hashlib.sha256(
                                    duplicate_recovered[1]
                                ).hexdigest(),
                            }
                        )
                    value, raw = self._reservation_value(recovery)
                    journal.replace(active_fd, "reservation.json", raw)
                    journal.replace(
                        active_fd,
                        "staging.json",
                        self._manifest(active_name, raw),
                    )
                else:
                    if not fresh_scaffold:
                        journal._fail(
                            "synthetic existing recovery tree lacks one complete staging"
                        )
                    active_name, active_fd = self._new_staging(seed_fd, journal)
                    candidate_fds.add(active_fd)
                    recovery.append(
                        {
                            "staging_name": active_name,
                            "classification": "new",
                            "action": "atomic_claim",
                            "inventory_sha256": _digest([]),
                        }
                    )
                    value, raw = self._reservation_value(recovery)
                    journal.create(active_fd, "reservation.json", raw)
                    journal.create(
                        active_fd,
                        "staging.json",
                        self._manifest(active_name, raw),
                    )
                if failure_injection == "before_atomic_claim":
                    raise RuntimeError("injected before atomic claim")
                manifest_fp = dict(journal.baseline(active_fd).inventory)[
                    "staging.json"
                ]
                journal.unlink(
                    active_fd,
                    "staging.json",
                    expected_fingerprint=manifest_fp,
                )
                metadata = os.fstat(active_fd)
                claimed = SyntheticReservationV7(
                    self.attempt,
                    value,
                    raw,
                    active_fd,
                    (metadata.st_dev, metadata.st_ino),
                    root_fd,
                    seed_fd,
                    derived_fd,
                    journal,
                    chain,
                    tuple(descriptors),
                )
                journal.rename_directory(
                    seed_fd,
                    active_name,
                    self.attempt.name,
                    directory_fd=active_fd,
                )
                candidate_fds.discard(active_fd)
                active_name = None
                active_fd = None
                self._assert_claim(claimed)
                if failure_injection == "after_atomic_claim":
                    raise RuntimeError("injected after atomic claim")
                return claimed
            except BaseException as error:
                if claimed is not None and self.attempt.exists():
                    self.terminate(claimed, error, stage="reservation_claim")
                elif active_name is not None and active_fd is not None:
                    self._remove_flat_staging(
                        seed_fd, active_fd, active_name, journal
                    )
                    candidate_fds.discard(active_fd)
                    active_fd = None
                if active_fd is not None:
                    os.close(active_fd)
                    candidate_fds.discard(active_fd)
                for candidate_fd in tuple(candidate_fds):
                    try:
                        os.close(candidate_fd)
                    except OSError:
                        pass
                candidate_fds.clear()
                journal.close()
                for descriptor in reversed(descriptors):
                    os.close(descriptor)
                _close_ancestor_chain(chain)
                raise

    @staticmethod
    def _cleanup_owned(
        reservation: SyntheticReservationV7,
        registry: dict[str, _OwnedArtifactV7],
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
                outcomes.append(
                    {"artifact": key, "outcome": "missing_owned_artifact_invalid"}
                )
                continue
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or _fingerprint(current) != artifact.fingerprint
            ):
                outcomes.append(
                    {
                        "artifact": key,
                        "outcome": "ownership_mismatch_preserved_invalid",
                    }
                )
                continue
            if reservation.journal.poisoned:
                outcomes.append(
                    {
                        "artifact": key,
                        "outcome": "journal_integrity_failed_preserved_invalid",
                    }
                )
                continue
            try:
                reservation.journal.unlink(
                    artifact.parent_fd,
                    artifact.name,
                    expected_fingerprint=artifact.fingerprint,
                )
            except PermissionError:
                outcomes.append(
                    {
                        "artifact": key,
                        "outcome": "journal_integrity_failed_preserved_invalid",
                    }
                )
                continue
            registry.pop(key)
            outcomes.append({"artifact": key, "outcome": "removed_owned"})
        return outcomes

    @classmethod
    def terminate(
        cls,
        reservation: SyntheticReservationV7,
        error: BaseException,
        *,
        stage: str,
    ) -> dict[str, Any]:
        cls._assert_claim_fd(reservation)
        cleanup = cls._cleanup_owned(
            reservation, reservation.owned_derived_artifacts
        )
        cleanup.extend(
            cls._cleanup_owned(
                reservation,
                reservation.owned_claim_artifacts,
                selected={"checkpoint.pt", "result.json", "completed.json"},
            )
        )
        if not reservation.journal.poisoned:
            try:
                reservation.journal.assert_clean()
            except PermissionError:
                pass
        core = {
            "schema": "lewm_go2_n5_full_panel_v7_synthetic_failure_v1",
            "status": "failed",
            "failure_stage": stage,
            "error": type(error).__name__,
            "artifact_cleanup": cleanup,
            "owned_directory_journal": {
                "integrity": (
                    "failed" if reservation.journal.poisoned else "intact"
                ),
                "poison_reason": reservation.journal.poison_reason,
                "success_eligibility_restored": False,
            },
            "retry_authorized": False,
        }
        value = {**core, "content_sha256": _digest(core)}
        cls._write_claim(
            reservation,
            "failed.json",
            _json(value) + b"\n",
            terminal=True,
            role="terminal_failure",
        )
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.seed_root_fd)
        return value

    @classmethod
    def publish_claim_artifact(
        cls, reservation: SyntheticReservationV7, name: str, raw: bytes
    ) -> None:
        cls._write_claim(
            reservation,
            name,
            raw,
            terminal=False,
            role="partial_claim",
        )

    @classmethod
    def publish_derived_artifact(
        cls, reservation: SyntheticReservationV7, name: str, raw: bytes
    ) -> None:
        cls._write_derived(reservation, name, raw)

    def publish(self, reservation: SyntheticReservationV7, raw: bytes) -> None:
        if reservation.directory != self.attempt:
            raise PermissionError("synthetic reservation belongs to another operation")
        self._write_claim(
            reservation,
            "completed.json",
            raw,
            terminal=False,
            role="training_completion",
        )
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.seed_root_fd)

    @staticmethod
    def close(reservation: SyntheticReservationV7) -> None:
        try:
            os.close(reservation.directory_fd)
        except OSError:
            pass
        reservation.journal.close()
        # root_fd belongs to the ancestor chain; these are lifecycle opens.
        for descriptor in reversed(reservation.owned_directory_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
        # The attempts descriptor is retained only by the journal and chain walk.
        _close_ancestor_chain(reservation.ancestor_chain)
