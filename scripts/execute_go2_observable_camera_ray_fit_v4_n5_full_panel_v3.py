#!/usr/bin/env python3
"""Authority-free, one-process V3 executor for the frozen N5 experiment.

This is the sole production operation.  It performs source preflight, the
single filesystem claim, frozen training, independent metric verification, and
finalization in one isolated process.  It has no caller-held authority or
caller-controlled production path.  Importing it opens no data or output.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
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

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v3 as policy,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINER_RELATIVE_PATH = "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
BASE_TRAINER_FILE_SHA256 = policy.frozen_source_bindings()[
    BASE_TRAINER_RELATIVE_PATH
]
STAGING_PREFIX = ".n5.reservation-v3-"
LEGACY_STAGING_NAME = ".n5.reservation-staging"
V2_STAGING_PREFIX = ".n5.reservation-v2-"
LOCK_NAME = ".n5.reservation-v3.lock"
STAGING_SCHEMA = "lewm_go2_n5_full_panel_v3_preclaim_staging_v1"


@dataclass(frozen=True)
class AttemptReservation:
    directory: Path
    value: Mapping[str, Any]
    raw: bytes
    file_sha256: str

    @property
    def binding(self) -> dict[str, Any]:
        return policy.artifact_binding(
            "reservation.json",
            self.raw,
            content_sha256=str(self.value["content_sha256"]),
        )


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
        "scope": "one_exclusive_fresh_full_panel_attempt",
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
) -> AttemptReservation:
    core = _reservation_core(
        review_binding,
        recovery_events=recovery_events,
    )
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    return AttemptReservation(
        directory=attempt_path,
        value=value,
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _manifest_value(
    *,
    staging: Path,
    attempt_path: Path,
    reservation: AttemptReservation,
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
) -> tuple[dict[str, Any], AttemptReservation | None]:
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
    return evidence, AttemptReservation(
        directory=attempt_path,
        value=reservation_value,
        raw=reservation_raw,
        file_sha256=hashlib.sha256(reservation_raw).hexdigest(),
    )


def _remove_staging(staging: Path, *, seed_root: Path) -> None:
    if staging.parent.resolve() != seed_root.resolve() or not (
        staging.name == LEGACY_STAGING_NAME
        or staging.name.startswith(STAGING_PREFIX)
        or staging.name.startswith(V2_STAGING_PREFIX)
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


def _new_staging(seed_root: Path, attempt_name: str) -> Path:
    for _ in range(128):
        staging = seed_root / f".{attempt_name}.reservation-v3-{secrets.token_hex(16)}"
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
    reservation: AttemptReservation,
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
    reservation: AttemptReservation,
    attempt_path: Path,
    recovery_events: Sequence[Mapping[str, Any]],
    review_binding: Mapping[str, str],
) -> AttemptReservation:
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


def _reserve_exact_attempt(source_review_file_sha256: str) -> AttemptReservation:
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
                    or child.name.startswith(V2_STAGING_PREFIX)
                ),
                key=lambda child: child.name,
            )
            recovery_events: list[dict[str, Any]] = []
            complete: list[tuple[Path, AttemptReservation, dict[str, Any]]] = []
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
                active_staging = _new_staging(seed_root, attempt_path.name)
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
            staging_metadata = os.stat(active_staging, follow_symlinks=False)
            owned_claim_identity = (staging_metadata.st_dev, staging_metadata.st_ino)
            os.rename(active_staging, attempt_path)
            _fsync_directory(seed_root)
            active_staging = None
            return reservation
    except BaseException as error:
        attempt_metadata = (
            os.stat(attempt_path, follow_symlinks=False)
            if attempt_path.is_dir() and not attempt_path.is_symlink()
            else None
        )
        owns_canonical_claim = (
            owned_claim_identity is not None
            and attempt_metadata is not None
            and (attempt_metadata.st_dev, attempt_metadata.st_ino)
            == owned_claim_identity
        )
        if owns_canonical_claim and (attempt_path / "reservation.json").is_file():
            reservation_raw = policy.read_regular_bytes(
                attempt_path / "reservation.json",
                name="claimed reservation during failure",
            )
            reservation_value = policy.parse_json(
                reservation_raw,
                name="claimed reservation during failure",
            )
            claimed_reservation = AttemptReservation(
                directory=attempt_path,
                value=reservation_value,
                raw=reservation_raw,
                file_sha256=hashlib.sha256(reservation_raw).hexdigest(),
            )
            try:
                _terminate_failure(claimed_reservation, error)
            except BaseException as terminal_error:
                raise RuntimeError(
                    "reservation claim failed and terminal receipt could not be written"
                ) from terminal_error
        elif active_staging is not None and active_staging.exists():
            _remove_staging(active_staging, seed_root=seed_root)
        raise


def _decode_worker(
    payload: tuple[tuple[str, str, str, str], str, str],
) -> Any:
    job, review_path_text, review_sha256 = payload
    _set_thread_caps()
    policy.preflight_static_authority()
    policy.preflight_source_review(Path(review_path_text), review_sha256)
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    return base._decode_rgb_job(*job)


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
    payloads = [
        (job, str(policy.CANONICAL_SOURCE_REVIEW_PATH), source_review_file_sha256)
        for job in jobs
    ]
    if worker_count == 1:
        arrays = [_decode_worker(payload) for payload in payloads]
        start_method = "inline_source_revalidated"
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(payloads)),
            mp_context=context,
        ) as executor:
            arrays = list(executor.map(_decode_worker, payloads))
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


def _terminate_failure(
    reservation: AttemptReservation,
    error: BaseException,
) -> dict[str, Any]:
    for name in ("checkpoint.pt", "result.json", "completed.json"):
        (reservation.directory / name).unlink(missing_ok=True)
    _fsync_directory(reservation.directory)
    core = {
        "schema": policy.FAILURE_SCHEMA,
        "status": "failed",
        "reservation": reservation.binding,
        "failure": _failure_code(error),
        "partial_artifacts_removed": True,
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
    _write_bytes_exclusive(reservation.directory / "failed.json", raw)
    _fsync_directory(reservation.directory)
    _fsync_directory(reservation.directory.parent)
    return policy.artifact_binding(
        "failed.json",
        raw,
        content_sha256=value["content_sha256"],
    )


def _publish_success(
    reservation: AttemptReservation,
    *,
    checkpoint_raw: bytes,
    checkpoint_content_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
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
    _write_bytes_exclusive(reservation.directory / "checkpoint.pt", checkpoint_raw)
    _write_bytes_exclusive(reservation.directory / "result.json", result_raw)
    _write_bytes_exclusive(reservation.directory / "completed.json", completion_raw)
    _fsync_directory(reservation.directory)
    _fsync_directory(reservation.directory.parent)
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


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    if path not in {policy.CANONICAL_METRIC_RECEIPT_PATH, policy.CANONICAL_GATE_PATH}:
        raise PermissionError("N5 full-panel V3 publication path is not canonical")
    if policy.CANONICAL_OUTPUT_ROOT.is_symlink() or not policy.CANONICAL_OUTPUT_ROOT.is_dir():
        raise PermissionError("N5 full-panel V3 output root is not a real directory")
    path.parent.mkdir(exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PermissionError("N5 full-panel V3 publication parent is not real")
    raw = policy.canonical_json_bytes(value) + b"\n"
    _write_bytes_exclusive(path, raw)
    _fsync_directory(path.parent)
    return policy.artifact_binding(
        str(path.relative_to(policy.CANONICAL_OUTPUT_ROOT)),
        raw,
        content_sha256=str(value["content_sha256"]),
    )


def _run_frozen_training(
    source_review_file_sha256: str,
    *,
    rgb_workers: int,
) -> dict[str, Any]:
    """Run frozen V1 numerical science behind an ephemeral local adapter."""

    review, _ = policy.preflight_source_review(
        policy.CANONICAL_SOURCE_REVIEW_PATH,
        source_review_file_sha256,
    )
    review_binding = policy.source_review_binding(review, source_review_file_sha256)
    token = object()

    def require(value: object) -> object:
        if value is not token:
            raise PermissionError("retained science lacks its local V3 execution token")
        policy.preflight_static_authority()
        policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        return value

    def source_binding(value: object) -> dict[str, str]:
        require(value)
        return dict(review_binding)

    def reserve(value: object) -> AttemptReservation:
        require(value)
        return _reserve_exact_attempt(source_review_file_sha256)

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
    try:
        retained.policy = compatibility
        retained._reserve_attempt = reserve
        retained._terminate_failure = _terminate_failure
        retained._publish_success = _publish_success
        retained.decode_selected_rgb = decode
        summary = retained._run_training(token, rgb_workers=rgb_workers)
    finally:
        retained.policy = original["policy"]
        retained._reserve_attempt = original["reserve"]
        retained._terminate_failure = original["terminate"]
        retained._publish_success = original["publish"]
        retained.decode_selected_rgb = original["decode"]
    return {
        **summary,
        "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_launch_summary_v1",
    }


def _artifact_args(source_review_file_sha256: str) -> argparse.Namespace:
    attempt = policy.CANONICAL_ATTEMPT_PATH
    bindings: dict[str, str] = {}
    for name in ("reservation.json", "result.json", "checkpoint.pt", "completed.json"):
        raw = policy.read_regular_bytes(attempt / name, name=f"V3 {name}")
        bindings[name] = f"{attempt / name}:{hashlib.sha256(raw).hexdigest()}"
    return argparse.Namespace(
        source_review=policy.CANONICAL_SOURCE_REVIEW_PATH,
        source_review_sha256=source_review_file_sha256,
        reservation=bindings["reservation.json"],
        result=bindings["result.json"],
        checkpoint=bindings["checkpoint.pt"],
        completion=bindings["completed.json"],
    )


def _run_independent_verification(source_review_file_sha256: str) -> dict[str, Any]:
    review, _ = policy.preflight_source_review(
        policy.CANONICAL_SOURCE_REVIEW_PATH,
        source_review_file_sha256,
    )
    review_binding = policy.source_review_binding(review, source_review_file_sha256)
    token = object()

    def require(value: object) -> object:
        if value is not token:
            raise PermissionError("retained verifier lacks its local V3 execution token")
        policy.preflight_static_authority()
        policy.preflight_source_review(
            policy.CANONICAL_SOURCE_REVIEW_PATH,
            source_review_file_sha256,
        )
        return value

    def source_binding(value: object) -> dict[str, str]:
        require(value)
        return dict(review_binding)

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
    compatibility.write_exclusive = _write_canonical_json

    from scripts import verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as verifier

    original_policy = verifier.policy
    try:
        verifier.policy = compatibility
        args = _artifact_args(source_review_file_sha256)
        bundle = verifier._validate_attempt_bundle(token, args)
        receipt = verifier._compute_receipt(token, bundle)
        _write_canonical_json(policy.CANONICAL_METRIC_RECEIPT_PATH, receipt)
    finally:
        verifier.policy = original_policy
    return receipt


def _run_finalization(source_review_file_sha256: str) -> dict[str, Any]:
    review, _ = policy.preflight_source_review(
        policy.CANONICAL_SOURCE_REVIEW_PATH,
        source_review_file_sha256,
    )
    review_binding = policy.source_review_binding(review, source_review_file_sha256)
    token = object()

    def require(value: object) -> object:
        if value is not token:
            raise PermissionError("retained finalizer lacks its local V3 execution token")
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
        if Path(source_review_path).resolve(strict=True) != policy.CANONICAL_SOURCE_REVIEW_PATH.resolve(strict=True) or supplied_sha256 != source_review_file_sha256:
            raise PermissionError("retained finalizer source review binding changed")
        require(token)
        return token

    def source_binding(value: object) -> dict[str, str]:
        require(value)
        return dict(review_binding)

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
    compatibility.write_exclusive = _write_canonical_json

    from scripts import finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as finalizer
    from scripts import verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as verifier

    original_finalizer_policy = finalizer.policy
    original_verifier_policy = verifier.policy
    try:
        finalizer.policy = compatibility
        verifier.policy = compatibility
        args = _artifact_args(source_review_file_sha256)
        metric_raw = policy.read_regular_bytes(
            policy.CANONICAL_METRIC_RECEIPT_PATH,
            name="V3 metric verification",
        )
        args.metric_verification = (
            f"{policy.CANONICAL_METRIC_RECEIPT_PATH}:"
            f"{hashlib.sha256(metric_raw).hexdigest()}"
        )
        gate = finalizer.run(args)
    finally:
        finalizer.policy = original_finalizer_policy
        verifier.policy = original_verifier_policy
    return gate


def execute_exact(
    source_review_file_sha256: str,
    *,
    rgb_workers: int,
) -> dict[str, Any]:
    """Own the complete canonical lifecycle; no stage authority escapes."""

    if not sys.flags.isolated:
        raise PermissionError("N5 full-panel V3 exact execution requires isolation")
    if isinstance(rgb_workers, bool) or not 1 <= int(rgb_workers) <= 5:
        raise ValueError("N5 full-panel RGB workers must lie in [1,5]")
    if policy.CANONICAL_ATTEMPT_PATH.exists() or policy.CANONICAL_ATTEMPT_PATH.is_symlink():
        raise FileExistsError("the sole N5 full-panel attempt is already claimed")
    policy.preflight_static_authority()
    policy.preflight_source_review(
        policy.CANONICAL_SOURCE_REVIEW_PATH,
        source_review_file_sha256,
    )
    training = _run_frozen_training(
        source_review_file_sha256,
        rgb_workers=int(rgb_workers),
    )
    verification = _run_independent_verification(source_review_file_sha256)
    gate = _run_finalization(source_review_file_sha256)
    return {
        "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v3_end_to_end_summary_v1",
        "seed": 20260710,
        "fit_size": 5,
        "training": training,
        "metric_verification_content_sha256": verification["content_sha256"],
        "gate_content_sha256": gate["content_sha256"],
        "passes": bool(gate["passes"]),
        "later_rung_execution_authorized": False,
    }


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
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--rgb-workers", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args(argv)
    if not policy.is_sha256(args.source_review_sha256):
        raise ValueError("N5 full-panel V3 source review SHA-256 is malformed")
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


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated:
        return _isolated_child(raw_argv)
    args = parse_args(raw_argv)
    summary = execute_exact(
        args.source_review_sha256,
        rgb_workers=int(args.rgb_workers),
    )
    print(policy.canonical_json_bytes(summary).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
