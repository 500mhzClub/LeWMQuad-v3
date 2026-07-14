"""Production-ineligible filesystem model for V6 lifecycle tests.

This module intentionally lives under ``lewm.tests``.  Every root is rejected
if it contains, equals, or lies inside the repository or canonical output.
It shares no code, state, token, object, or path argument with the production
executor.
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
import shutil
import stat
from typing import Any, Iterator, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_OUTPUT = (
    REPOSITORY_ROOT
    / ".generated/go2_observable_camera_ray_fit_v4/n5_full_panel_recovery_v6"
).resolve()
STAGING_PREFIX = ".n5.synthetic-v6-"
LOCK_NAME = ".n5.synthetic-v6.lock"
SCHEMA = "lewm_go2_n5_full_panel_v6_synthetic_reservation_v1"
MANIFEST_SCHEMA = "lewm_go2_n5_full_panel_v6_synthetic_staging_v1"


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
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


@dataclass
class _DirectoryEntryV6:
    parent_fd: int
    name: str
    child_fd: int
    identity_security: tuple[int, ...]
    full_fingerprint: tuple[int, ...]
    exclusive: bool


@dataclass
class _DirectoryChainV6:
    anchor_fd: int
    anchor_identity_security: tuple[int, ...]
    descriptors: list[int]
    entries: list[_DirectoryEntryV6]
    path_fds: dict[Path, int]
    closed: bool = False


@dataclass(frozen=True)
class _OwnedArtifactV6:
    role: str
    parent_fd: int
    name: str
    fingerprint: tuple[int, ...]


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


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PermissionError("synthetic evidence is not a regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError("synthetic evidence changed while read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _write(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _replace(path: Path, raw: bytes) -> None:
    temporary = path.parent / f".{path.name}.replace-{secrets.token_hex(12)}"
    _write(temporary, raw)
    os.replace(temporary, path)
    _fsync(path.parent)


def _fsync(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _entry_metadata(parent_fd: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)


def _assert_chain(chain: _DirectoryChainV6) -> None:
    if chain.closed:
        raise PermissionError("synthetic V6 canonical directory chain is closed")
    if _identity_security(os.fstat(chain.anchor_fd)) != chain.anchor_identity_security:
        raise PermissionError("synthetic V6 filesystem-root descriptor changed")
    for entry in chain.entries:
        named = _entry_metadata(entry.parent_fd, entry.name)
        opened = os.fstat(entry.child_fd)
        named_identity = _identity_security(named)
        opened_identity = _identity_security(opened)
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or named_identity != entry.identity_security
            or opened_identity != entry.identity_security
            or named_identity != opened_identity
            or (entry.exclusive and _fingerprint(named) != entry.full_fingerprint)
            or (entry.exclusive and _fingerprint(opened) != entry.full_fingerprint)
        ):
            raise PermissionError("synthetic V6 canonical directory identity changed")


def _refresh_chain(chain: _DirectoryChainV6, mutable_fds: set[int]) -> None:
    if chain.closed:
        raise PermissionError("synthetic V6 canonical directory chain is closed")
    if _identity_security(os.fstat(chain.anchor_fd)) != chain.anchor_identity_security:
        raise PermissionError("synthetic V6 filesystem-root descriptor changed")
    for entry in chain.entries:
        named = _entry_metadata(entry.parent_fd, entry.name)
        opened = os.fstat(entry.child_fd)
        named_identity = _identity_security(named)
        opened_identity = _identity_security(opened)
        named_fingerprint = _fingerprint(named)
        opened_fingerprint = _fingerprint(opened)
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or named_identity != entry.identity_security
            or opened_identity != entry.identity_security
            or named_fingerprint != opened_fingerprint
        ):
            raise PermissionError("synthetic V6 canonical directory identity changed")
        if not entry.exclusive:
            continue
        if entry.child_fd in mutable_fds:
            entry.full_fingerprint = opened_fingerprint
        elif opened_fingerprint != entry.full_fingerprint:
            raise PermissionError("synthetic V6 unexpected canonical directory mutation")


def _open_chain(final_path: Path, *, exclusive_root: Path) -> _DirectoryChainV6:
    final_path = Path(final_path)
    if not final_path.is_absolute() or any(
        component in {"", ".", ".."} for component in final_path.parts[1:]
    ):
        raise PermissionError("synthetic V6 canonical path is malformed")
    filesystem_root = Path(final_path.anchor)
    anchor_before = filesystem_root.stat(follow_symlinks=False)
    anchor_identity_security = _identity_security(anchor_before)
    anchor_fd = os.open(filesystem_root, _directory_flags())
    chain = _DirectoryChainV6(
        anchor_fd=anchor_fd,
        anchor_identity_security=anchor_identity_security,
        descriptors=[anchor_fd],
        entries=[],
        path_fds={filesystem_root: anchor_fd},
    )
    try:
        if _identity_security(os.fstat(anchor_fd)) != anchor_identity_security:
            raise PermissionError("synthetic V6 filesystem root changed during open")
        parent_fd = anchor_fd
        current_path = filesystem_root
        for component in final_path.parts[1:]:
            before = _entry_metadata(parent_fd, component)
            current_path = current_path / component
            exclusive = current_path == exclusive_root or current_path.is_relative_to(
                exclusive_root
            )
            identity_security = _identity_security(before)
            full_fingerprint = _fingerprint(before)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise PermissionError("synthetic V6 canonical component is not a directory")
            child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
            chain.descriptors.append(child_fd)
            opened = os.fstat(child_fd)
            if (
                _identity_security(opened) != identity_security
                or (exclusive and _fingerprint(opened) != full_fingerprint)
            ):
                raise PermissionError("synthetic V6 canonical component changed during open")
            chain.entries.append(
                _DirectoryEntryV6(
                    parent_fd,
                    component,
                    child_fd,
                    identity_security,
                    full_fingerprint,
                    exclusive,
                )
            )
            chain.path_fds[current_path] = child_fd
            parent_fd = child_fd
        _assert_chain(chain)
        return chain
    except BaseException:
        _close_chain(chain)
        raise


def _open_chain_child(
    chain: _DirectoryChainV6,
    parent_path: Path,
    name: str,
) -> int:
    _assert_chain(chain)
    parent_fd = chain.path_fds[parent_path]
    try:
        before = _entry_metadata(parent_fd, name)
    except FileNotFoundError:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _refresh_chain(chain, {parent_fd})
        before = _entry_metadata(parent_fd, name)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise PermissionError("synthetic V6 derived parent is not a directory")
    identity_security = _identity_security(before)
    full_fingerprint = _fingerprint(before)
    child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    chain.descriptors.append(child_fd)
    if (
        _identity_security(os.fstat(child_fd)) != identity_security
        or _fingerprint(os.fstat(child_fd)) != full_fingerprint
    ):
        raise PermissionError("synthetic V6 derived parent changed during open")
    child_path = parent_path / name
    chain.entries.append(
        _DirectoryEntryV6(
            parent_fd,
            name,
            child_fd,
            identity_security,
            full_fingerprint,
            True,
        )
    )
    chain.path_fds[child_path] = child_fd
    _assert_chain(chain)
    return child_fd


def _close_chain(chain: _DirectoryChainV6) -> None:
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
    return resolved.resolve(strict=True)


@dataclass(frozen=True)
class SyntheticReservationV6:
    directory: Path
    value: Mapping[str, Any]
    raw: bytes
    directory_fd: int = -1
    directory_identity: tuple[int, int] | None = None
    directory_fingerprint: tuple[int, ...] | None = None
    directory_chain: _DirectoryChainV6 | None = None
    owned_claim_artifacts: dict[str, _OwnedArtifactV6] = field(
        default_factory=dict,
        compare=False,
    )
    owned_derived_artifacts: dict[str, _OwnedArtifactV6] = field(
        default_factory=dict,
        compare=False,
    )

    @property
    def seed_root_fd(self) -> int:
        if self.directory_chain is None:
            return -1
        return self.directory_chain.path_fds.get(self.directory.parent, -1)

    @property
    def derived_fd(self) -> int:
        if self.directory_chain is None:
            return -1
        return self.directory_chain.path_fds.get(
            self.directory.parents[2] / "derived",
            -1,
        )


class SyntheticExecutionV6:
    """A test-only operation whose fixed attempt is relative to a safe root."""

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        object.__setattr__(self, "_root", _safe_root(root))

    @property
    def root(self) -> Path:
        return Path(object.__getattribute__(self, "_root"))

    @property
    def attempt(self) -> Path:
        return self.root / "attempts/seed_20260710/n5"

    def __copy__(self) -> "SyntheticExecutionV6":
        return type(self)(self.root)

    def __deepcopy__(self, memo: object) -> "SyntheticExecutionV6":
        del memo
        return type(self)(self.root)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        seed_root = self.attempt.parent
        seed_root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            seed_root / LOCK_NAME,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise PermissionError("synthetic reservation lock is not private")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

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

    def _reservation(self, recovery: Sequence[Mapping[str, Any]]) -> SyntheticReservationV6:
        core = self._core(recovery)
        value = {**core, "content_sha256": _digest(core)}
        return SyntheticReservationV6(self.attempt, value, _json(value) + b"\n")

    @staticmethod
    def _assert_claim_fd(reservation: SyntheticReservationV6) -> None:
        if reservation.directory_fd < 0 or reservation.directory_identity is None:
            raise PermissionError("synthetic V6 reservation has no open claim")
        opened = os.fstat(reservation.directory_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != reservation.directory_identity
        ):
            raise PermissionError("synthetic V6 claimed descriptor identity changed")

    @classmethod
    def _assert_claim(cls, reservation: SyntheticReservationV6) -> None:
        cls._assert_claim_fd(reservation)
        opened = os.fstat(reservation.directory_fd)
        if (
            reservation.directory_fingerprint is None
            or _fingerprint(opened) != reservation.directory_fingerprint
        ):
            raise PermissionError("synthetic V6 claimed directory changed")
        if reservation.directory_chain is None or reservation.seed_root_fd < 0:
            raise PermissionError("synthetic V6 reservation lacks canonical ancestry")
        _assert_chain(reservation.directory_chain)
        named = os.stat(
            reservation.directory.name,
            dir_fd=reservation.seed_root_fd,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (named.st_dev, named.st_ino) != reservation.directory_identity
            or _fingerprint(named) != reservation.directory_fingerprint
        ):
            raise PermissionError("synthetic V6 canonical claim identity changed")
        expected = {"reservation.json"} | set(reservation.owned_claim_artifacts)
        if set(os.listdir(reservation.directory_fd)) != expected:
            raise PermissionError("synthetic V6 claimed directory inventory changed")

    @classmethod
    def _refresh_claim(cls, reservation: SyntheticReservationV6) -> None:
        cls._assert_claim_fd(reservation)
        expected = {"reservation.json"} | set(reservation.owned_claim_artifacts)
        if set(os.listdir(reservation.directory_fd)) != expected:
            raise PermissionError("synthetic V6 claimed directory inventory changed")
        current = os.fstat(reservation.directory_fd)
        previous = reservation.directory_fingerprint
        if previous is None or _identity_security(current) != (
            previous[0],
            previous[1],
            previous[2],
            previous[4],
            previous[5],
        ):
            raise PermissionError("synthetic V6 claimed directory security changed")
        object.__setattr__(reservation, "directory_fingerprint", _fingerprint(current))

    @classmethod
    def _write_claim(
        cls,
        reservation: SyntheticReservationV6,
        name: str,
        raw: bytes,
        *,
        require_canonical: bool = True,
        role: str | None = None,
    ) -> None:
        if require_canonical:
            cls._assert_claim(reservation)
        else:
            cls._assert_claim_fd(reservation)
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=reservation.directory_fd,
        )
        identity: tuple[int, int] | None = None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError("synthetic V6 claim leaf is not singly linked")
            identity = (metadata.st_dev, metadata.st_ino)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        committed = os.stat(
            name,
            dir_fd=reservation.directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(committed.st_mode)
            or committed.st_nlink != 1
            or (committed.st_dev, committed.st_ino) != identity
        ):
            raise PermissionError("synthetic V6 committed claim leaf changed")
        reservation.owned_claim_artifacts[name] = _OwnedArtifactV6(
            role=role or name,
            parent_fd=reservation.directory_fd,
            name=name,
            fingerprint=_fingerprint(committed),
        )
        if require_canonical:
            cls._refresh_claim(reservation)
            cls._assert_claim(reservation)
        else:
            cls._assert_claim_fd(reservation)

    @classmethod
    def _write_derived(
        cls,
        reservation: SyntheticReservationV6,
        name: str,
        raw: bytes,
    ) -> None:
        cls._assert_claim(reservation)
        if Path(name).name != name or reservation.derived_fd < 0:
            raise PermissionError("synthetic V6 derived artifact escaped")
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=reservation.derived_fd,
        )
        identity: tuple[int, int] | None = None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError("synthetic V6 derived leaf is not singly linked")
            identity = (metadata.st_dev, metadata.st_ino)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        os.fsync(reservation.derived_fd)
        committed = os.stat(name, dir_fd=reservation.derived_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(committed.st_mode)
            or committed.st_nlink != 1
            or (committed.st_dev, committed.st_ino) != identity
        ):
            raise PermissionError("synthetic V6 committed derived leaf changed")
        reservation.owned_derived_artifacts[name] = _OwnedArtifactV6(
            role="derived",
            parent_fd=reservation.derived_fd,
            name=name,
            fingerprint=_fingerprint(committed),
        )
        if reservation.directory_chain is None:
            raise PermissionError("synthetic V6 reservation lacks canonical ancestry")
        _refresh_chain(reservation.directory_chain, {reservation.derived_fd})
        cls._assert_claim(reservation)

    @staticmethod
    def _cleanup_owned(
        registry: dict[str, _OwnedArtifactV6],
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
                outcomes.append({"artifact": key, "outcome": "already_absent"})
                continue
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or _fingerprint(current) != artifact.fingerprint
            ):
                outcomes.append(
                    {"artifact": key, "outcome": "ownership_mismatch_preserved_invalid"}
                )
                continue
            os.unlink(artifact.name, dir_fd=artifact.parent_fd)
            os.fsync(artifact.parent_fd)
            registry.pop(key, None)
            outcomes.append({"artifact": key, "outcome": "removed_owned"})
        return outcomes

    @classmethod
    def terminate(
        cls,
        reservation: SyntheticReservationV6,
        error: BaseException,
        *,
        stage: str,
    ) -> dict[str, Any]:
        cls._assert_claim_fd(reservation)
        cleanup = cls._cleanup_owned(reservation.owned_derived_artifacts)
        cleanup.extend(
            cls._cleanup_owned(
                reservation.owned_claim_artifacts,
                selected={"checkpoint.pt", "result.json", "completed.json"},
            )
        )
        core = {
            "schema": "lewm_go2_n5_full_panel_v6_synthetic_failure_v1",
            "status": "failed",
            "failure_stage": stage,
            "error": type(error).__name__,
            "artifact_cleanup": cleanup,
            "retry_authorized": False,
        }
        value = {**core, "content_sha256": _digest(core)}
        cls._write_claim(
            reservation,
            "failed.json",
            _json(value) + b"\n",
            require_canonical=False,
            role="terminal_failure",
        )
        os.fsync(reservation.directory_fd)
        if reservation.seed_root_fd >= 0:
            os.fsync(reservation.seed_root_fd)
        return value

    @classmethod
    def publish_claim_artifact(
        cls,
        reservation: SyntheticReservationV6,
        name: str,
        raw: bytes,
    ) -> None:
        cls._write_claim(reservation, name, raw, role="partial_claim")

    @classmethod
    def publish_derived_artifact(
        cls,
        reservation: SyntheticReservationV6,
        name: str,
        raw: bytes,
    ) -> None:
        cls._write_derived(reservation, name, raw)

    def publish(self, reservation: SyntheticReservationV6, raw: bytes) -> None:
        self._assert_claim(reservation)
        if reservation.directory != self.attempt:
            raise PermissionError("synthetic V6 reservation belongs to another operation")
        self._write_claim(
            reservation,
            "completed.json",
            raw,
            role="training_completion",
        )
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.seed_root_fd)

    @staticmethod
    def close(reservation: SyntheticReservationV6) -> None:
        os.close(reservation.directory_fd)
        if reservation.directory_chain is not None:
            _close_chain(reservation.directory_chain)

    def _manifest(self, staging: Path, reservation: SyntheticReservationV6) -> bytes:
        core = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "staging_name": staging.name,
            "attempt_path": str(self.attempt.resolve()),
            "reservation_file_sha256": hashlib.sha256(reservation.raw).hexdigest(),
        }
        return _json({**core, "content_sha256": _digest(core)}) + b"\n"

    def _new_staging(self) -> Path:
        for _ in range(128):
            path = self.attempt.parent / f"{STAGING_PREFIX}{secrets.token_hex(16)}"
            try:
                os.mkdir(path, 0o700)
                os.chmod(path, 0o700, follow_symlinks=False)
                _fsync(path.parent)
                return path
            except FileExistsError:
                continue
        raise FileExistsError("synthetic staging namespace exhausted")

    def _inventory(self, staging: Path) -> str:
        rows: list[str] = []
        if staging.is_symlink() or not staging.is_dir():
            metadata = os.lstat(staging)
            rows.append(f"other:{stat.S_IFMT(metadata.st_mode)}:{metadata.st_size}")
        else:
            for child in sorted(staging.iterdir(), key=lambda item: item.name):
                metadata = child.lstat()
                if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= 1024 * 1024:
                    raw = _read(child)
                    rows.append(f"file:{child.name}:{len(raw)}:{hashlib.sha256(raw).hexdigest()}")
                else:
                    rows.append(f"other:{child.name}:{metadata.st_size}")
        return _digest(rows)

    def _classify(self, staging: Path) -> tuple[dict[str, str], SyntheticReservationV6 | None]:
        evidence = {
            "staging_name": staging.name,
            "inventory_sha256": self._inventory(staging),
            "classification": "foreign",
            "action": "remove_without_claim",
        }
        metadata = os.lstat(staging)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            return evidence, None
        if sorted(path.name for path in staging.iterdir()) != ["reservation.json", "staging.json"]:
            evidence["classification"] = "incomplete"
            return evidence, None
        try:
            reservation_raw = _read(staging / "reservation.json")
            manifest_raw = _read(staging / "staging.json")
            reservation = json.loads(reservation_raw)
            manifest = json.loads(manifest_raw)
            reservation_core = dict(reservation)
            reservation_hash = reservation_core.pop("content_sha256")
            manifest_core = dict(manifest)
            manifest_hash = manifest_core.pop("content_sha256")
            valid = (
                reservation_hash == _digest(reservation_core)
                and manifest_hash == _digest(manifest_core)
                and reservation.get("schema") == SCHEMA
                and reservation.get("attempt_path") == str(self.attempt.resolve())
                and reservation.get("production_eligible") is False
                and manifest.get("schema") == MANIFEST_SCHEMA
                and manifest.get("staging_name") == staging.name
                and manifest.get("attempt_path") == str(self.attempt.resolve())
                and manifest.get("reservation_file_sha256") == hashlib.sha256(reservation_raw).hexdigest()
            )
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            valid = False
        if not valid:
            evidence["classification"] = "mutated"
            return evidence, None
        evidence["classification"] = "complete"
        evidence["action"] = "resume_after_rehash"
        return evidence, SyntheticReservationV6(self.attempt, reservation, reservation_raw)

    def _remove(self, staging: Path) -> None:
        if staging.parent.resolve() != self.attempt.parent.resolve() or not staging.name.startswith(STAGING_PREFIX):
            raise PermissionError("synthetic cleanup escaped its namespace")
        metadata = os.lstat(staging)
        if stat.S_ISDIR(metadata.st_mode):
            os.chmod(staging, 0o700, follow_symlinks=False)
            shutil.rmtree(staging)
        else:
            staging.unlink()
        _fsync(staging.parent)

    def prepare_complete_staging(self) -> Path:
        """Model uncatchable death after staging fsync but before rename."""

        self.attempt.parent.mkdir(parents=True, exist_ok=True)
        staging = self._new_staging()
        reservation = self._reservation(
            ({"staging_name": staging.name, "classification": "new", "action": "atomic_claim", "inventory_sha256": _digest([])},)
        )
        _write(staging / "reservation.json", reservation.raw)
        _write(staging / "staging.json", self._manifest(staging, reservation))
        _fsync(staging)
        return staging

    def claim(self, *, failure_injection: str | None = None) -> SyntheticReservationV6:
        active: Path | None = None
        identity: tuple[int, int] | None = None
        directory_fd: int | None = None
        directory_chain: _DirectoryChainV6 | None = None
        claimed: SyntheticReservationV6 | None = None
        try:
            with self._lock():
                if self.attempt.exists() or self.attempt.is_symlink():
                    raise FileExistsError("synthetic sole attempt is already claimed")
                recovery: list[dict[str, str]] = []
                complete: list[tuple[Path, SyntheticReservationV6, dict[str, str]]] = []
                for candidate in sorted(
                    (path for path in self.attempt.parent.iterdir() if path.name.startswith(STAGING_PREFIX)),
                    key=lambda path: path.name,
                ):
                    evidence, recovered = self._classify(candidate)
                    if recovered is None:
                        self._remove(candidate)
                        recovery.append(evidence)
                    else:
                        complete.append((candidate, recovered, evidence))
                if complete:
                    active, _old, selected = complete[0]
                    recovery.append(selected)
                    for duplicate, _reservation, evidence in complete[1:]:
                        self._remove(duplicate)
                        recovery.append({**evidence, "classification": "complete_duplicate", "action": "remove_duplicate"})
                    reservation = self._reservation(recovery)
                    _replace(active / "reservation.json", reservation.raw)
                    _replace(active / "staging.json", self._manifest(active, reservation))
                else:
                    active = self._new_staging()
                    recovery.append({"staging_name": active.name, "classification": "new", "action": "atomic_claim", "inventory_sha256": _digest([])})
                    reservation = self._reservation(recovery)
                    _write(active / "reservation.json", reservation.raw)
                    _write(active / "staging.json", self._manifest(active, reservation))
                    _fsync(active)
                if failure_injection == "before_atomic_claim":
                    raise RuntimeError("injected before atomic claim")
                (active / "staging.json").unlink()
                _fsync(active)
                directory_chain = _open_chain(
                    self.attempt.parent,
                    exclusive_root=self.root,
                )
                _open_chain_child(directory_chain, self.root, "derived")
                seed_root_fd = directory_chain.path_fds[self.attempt.parent]
                directory_fd = os.open(
                    active.name,
                    _directory_flags(),
                    dir_fd=seed_root_fd,
                )
                metadata = os.fstat(directory_fd)
                named = os.stat(
                    active.name,
                    dir_fd=seed_root_fd,
                    follow_symlinks=False,
                )
                if _fingerprint(named) != _fingerprint(metadata):
                    raise PermissionError("synthetic V6 staging identity changed")
                identity = (metadata.st_dev, metadata.st_ino)
                claimed = SyntheticReservationV6(
                    directory=reservation.directory,
                    value=reservation.value,
                    raw=reservation.raw,
                    directory_fd=directory_fd,
                    directory_identity=identity,
                    directory_fingerprint=_fingerprint(metadata),
                    directory_chain=directory_chain,
                )
                os.rename(
                    active.name,
                    self.attempt.name,
                    src_dir_fd=seed_root_fd,
                    dst_dir_fd=seed_root_fd,
                )
                os.fsync(seed_root_fd)
                active = None
                _refresh_chain(directory_chain, {seed_root_fd})
                self._refresh_claim(claimed)
                self._assert_claim(claimed)
                if failure_injection == "after_atomic_claim":
                    raise RuntimeError("injected after atomic claim")
                return claimed
        except BaseException as error:
            metadata = None
            if directory_chain is not None and self.attempt.parent in directory_chain.path_fds:
                try:
                    metadata = os.stat(
                        self.attempt.name,
                        dir_fd=directory_chain.path_fds[self.attempt.parent],
                        follow_symlinks=False,
                    )
                except (FileNotFoundError, NotADirectoryError):
                    pass
            owns = metadata is not None and identity == (metadata.st_dev, metadata.st_ino)
            if owns and claimed is not None:
                self.terminate(claimed, error, stage="reservation_claim")
            elif active is not None and active.exists():
                self._remove(active)
            if directory_fd is not None:
                os.close(directory_fd)
            if directory_chain is not None:
                _close_chain(directory_chain)
            raise


__all__ = ["SyntheticExecutionV6", "SyntheticReservationV6"]
