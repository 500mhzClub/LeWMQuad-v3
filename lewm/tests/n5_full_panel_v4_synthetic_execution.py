"""Production-ineligible filesystem model for V4 lifecycle tests.

This module intentionally lives under ``lewm.tests``.  Every root is rejected
if it contains, equals, or lies inside the repository or canonical output.
It shares no code, state, token, object, or path argument with the production
executor.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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
    REPOSITORY_ROOT / ".generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1"
).resolve()
STAGING_PREFIX = ".n5.synthetic-v4-"
LOCK_NAME = ".n5.synthetic-v4.lock"
SCHEMA = "lewm_go2_n5_full_panel_v4_synthetic_reservation_v1"
MANIFEST_SCHEMA = "lewm_go2_n5_full_panel_v4_synthetic_staging_v1"


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
class SyntheticReservationV4:
    directory: Path
    value: Mapping[str, Any]
    raw: bytes
    directory_fd: int = -1
    directory_identity: tuple[int, int] | None = None


class SyntheticExecutionV4:
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

    def __copy__(self) -> "SyntheticExecutionV4":
        return type(self)(self.root)

    def __deepcopy__(self, memo: object) -> "SyntheticExecutionV4":
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

    def _reservation(self, recovery: Sequence[Mapping[str, Any]]) -> SyntheticReservationV4:
        core = self._core(recovery)
        value = {**core, "content_sha256": _digest(core)}
        return SyntheticReservationV4(self.attempt, value, _json(value) + b"\n")

    @staticmethod
    def _assert_claim(reservation: SyntheticReservationV4) -> None:
        if reservation.directory_fd < 0 or reservation.directory_identity is None:
            raise PermissionError("synthetic V4 reservation has no open claim")
        opened = os.fstat(reservation.directory_fd)
        named = os.stat(reservation.directory, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (opened.st_dev, opened.st_ino) != reservation.directory_identity
            or (named.st_dev, named.st_ino) != reservation.directory_identity
        ):
            raise PermissionError("synthetic V4 canonical claim identity changed")

    @classmethod
    def _write_claim(
        cls,
        reservation: SyntheticReservationV4,
        name: str,
        raw: bytes,
    ) -> None:
        cls._assert_claim(reservation)
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
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError("synthetic V4 claim leaf is not singly linked")
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = -1
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        cls._assert_claim(reservation)

    def publish(self, reservation: SyntheticReservationV4, raw: bytes) -> None:
        self._assert_claim(reservation)
        if reservation.directory != self.attempt:
            raise PermissionError("synthetic V4 reservation belongs to another operation")
        self._write_claim(reservation, "completed.json", raw)
        os.fsync(reservation.directory_fd)
        _fsync(self.attempt.parent)

    @staticmethod
    def close(reservation: SyntheticReservationV4) -> None:
        os.close(reservation.directory_fd)

    def _manifest(self, staging: Path, reservation: SyntheticReservationV4) -> bytes:
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

    def _classify(self, staging: Path) -> tuple[dict[str, str], SyntheticReservationV4 | None]:
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
        return evidence, SyntheticReservationV4(self.attempt, reservation, reservation_raw)

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

    def claim(self, *, failure_injection: str | None = None) -> SyntheticReservationV4:
        active: Path | None = None
        identity: tuple[int, int] | None = None
        directory_fd: int | None = None
        claimed: SyntheticReservationV4 | None = None
        try:
            with self._lock():
                if self.attempt.exists() or self.attempt.is_symlink():
                    raise FileExistsError("synthetic sole attempt is already claimed")
                recovery: list[dict[str, str]] = []
                complete: list[tuple[Path, SyntheticReservationV4, dict[str, str]]] = []
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
                directory_fd = os.open(
                    active,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                )
                metadata = os.fstat(directory_fd)
                identity = (metadata.st_dev, metadata.st_ino)
                claimed = SyntheticReservationV4(
                    directory=reservation.directory,
                    value=reservation.value,
                    raw=reservation.raw,
                    directory_fd=directory_fd,
                    directory_identity=identity,
                )
                os.rename(active, self.attempt)
                _fsync(self.attempt.parent)
                active = None
                self._assert_claim(claimed)
                if failure_injection == "after_atomic_claim":
                    raise RuntimeError("injected after atomic claim")
                return claimed
        except BaseException as error:
            metadata = (
                os.stat(self.attempt, follow_symlinks=False)
                if self.attempt.is_dir() and not self.attempt.is_symlink()
                else None
            )
            owns = metadata is not None and identity == (metadata.st_dev, metadata.st_ino)
            if owns and claimed is not None:
                core = {"schema": "lewm_go2_n5_full_panel_v4_synthetic_failure_v1", "status": "failed", "error": type(error).__name__, "retry_authorized": False}
                self._write_claim(
                    claimed,
                    "failed.json",
                    _json({**core, "content_sha256": _digest(core)}) + b"\n",
                )
                os.fsync(claimed.directory_fd)
                _fsync(self.attempt.parent)
            elif active is not None and active.exists():
                self._remove(active)
            if directory_fd is not None:
                os.close(directory_fd)
            raise


__all__ = ["SyntheticExecutionV4", "SyntheticReservationV4"]
