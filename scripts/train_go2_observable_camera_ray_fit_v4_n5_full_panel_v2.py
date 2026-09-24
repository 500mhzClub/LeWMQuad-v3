#!/usr/bin/env python3
"""V2 one-shot trainer boundary for the frozen N5 full-panel experiment.

The numerical training/evaluation implementation is retained byte-for-byte from
V1.  This module replaces its authority, reservation, crash recovery, and
directory durability boundary before delegating to that frozen science.
"""
from __future__ import annotations

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
import sys
from typing import Any, Iterator, Mapping, Sequence

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as policy,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINER_RELATIVE_PATH = "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
BASE_TRAINER_FILE_SHA256 = policy.FROZEN_SOURCE_BINDINGS[
    BASE_TRAINER_RELATIVE_PATH
]
STAGING_PREFIX = ".n5.reservation-v2-"
LEGACY_STAGING_NAME = ".n5.reservation-staging"
LOCK_NAME = ".n5.reservation-v2.lock"
STAGING_SCHEMA = "lewm_go2_n5_full_panel_v2_preclaim_staging_v1"


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


def _review_binding(
    authority: policy.VerifiedAuthorityV2,
    *,
    test_capability: policy.TestAuthorityCapabilityV2 | None = None,
) -> dict[str, str]:
    return policy.source_review_binding(authority, test_capability=test_capability)


def _exact_input_binding(
    authority: policy.VerifiedAuthorityV2,
    *,
    test_capability: policy.TestAuthorityCapabilityV2 | None = None,
) -> dict[str, str]:
    review = _review_binding(authority, test_capability=test_capability)
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
        "source_review_file_sha256": review["file_sha256"],
        "source_review_content_sha256": review["content_sha256"],
        "terminal_invalidation_file_sha256": policy.TERMINAL_INVALIDATION_FILE_SHA256,
        "terminal_invalidation_content_sha256": (
            policy.TERMINAL_INVALIDATION_CONTENT_SHA256
        ),
    }


def _reservation_core(
    authority: policy.VerifiedAuthorityV2,
    *,
    recovery_events: Sequence[Mapping[str, Any]],
    test_capability: policy.TestAuthorityCapabilityV2 | None = None,
) -> dict[str, Any]:
    if test_capability is None:
        policy.require_verified_authority(
            authority,
            purpose="exact_run",
            target_path=policy.CANONICAL_ATTEMPT_PATH,
            allowed_states=("claiming",),
        )
    else:
        test_capability.validate(
            authority,
            target_path=authority.target_path,
            allowed_states=("claiming",),
        )
    return {
        "schema": policy.RESERVATION_SCHEMA,
        "status": "reserved",
        "attempt_index": 1,
        "maximum_attempts": 1,
        "scope": "one_exclusive_fresh_full_panel_attempt",
        "seed": 20260710,
        "fit_size": 5,
        "experiment": policy.EXPERIMENT,
        "authority_bindings": policy.AUTHORITY_BINDINGS,
        "source_review": _review_binding(
            authority,
            test_capability=test_capability,
        ),
        "inputs": _exact_input_binding(
            authority,
            test_capability=test_capability,
        ),
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
    authority: policy.VerifiedAuthorityV2,
    *,
    attempt_path: Path,
    recovery_events: Sequence[Mapping[str, Any]],
    test_capability: policy.TestAuthorityCapabilityV2 | None,
) -> AttemptReservation:
    core = _reservation_core(
        authority,
        recovery_events=recovery_events,
        test_capability=test_capability,
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
    authority: policy.VerifiedAuthorityV2,
    test_capability: policy.TestAuthorityCapabilityV2 | None,
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
        authority,
        recovery_events=(),
        test_capability=test_capability,
    )
    expected.pop("preclaim_recovery")
    return core == expected


def _classify_staging(
    staging: Path,
    *,
    attempt_path: Path,
    authority: policy.VerifiedAuthorityV2,
    test_capability: policy.TestAuthorityCapabilityV2 | None,
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
            authority=authority,
            test_capability=test_capability,
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
        staging.name == LEGACY_STAGING_NAME or staging.name.startswith(STAGING_PREFIX)
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
        staging = seed_root / f".{attempt_name}.reservation-v2-{secrets.token_hex(16)}"
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
    authority: policy.VerifiedAuthorityV2,
    test_capability: policy.TestAuthorityCapabilityV2 | None,
) -> AttemptReservation:
    updated = _reservation(
        authority,
        attempt_path=attempt_path,
        recovery_events=recovery_events,
        test_capability=test_capability,
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


def _transition_bound(
    authority: policy.VerifiedAuthorityV2,
    *,
    attempt_path: Path,
    from_states: Sequence[str],
    to_state: str,
    test_capability: policy.TestAuthorityCapabilityV2 | None,
) -> None:
    if test_capability is None:
        policy.transition_authority(
            authority,
            purpose="exact_run",
            target_path=attempt_path,
            from_states=from_states,
            to_state=to_state,
        )
    else:
        test_capability.transition(
            authority,
            target_path=attempt_path,
            from_states=from_states,
            to_state=to_state,
        )


def _reserve_bound_attempt(
    authority: policy.VerifiedAuthorityV2,
    *,
    attempt_path: Path,
    test_capability: policy.TestAuthorityCapabilityV2 | None,
    failure_injection: str | None,
) -> AttemptReservation:
    attempt_path = Path(attempt_path).resolve()
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
                ),
                key=lambda child: child.name,
            )
            recovery_events: list[dict[str, Any]] = []
            complete: list[tuple[Path, AttemptReservation, dict[str, Any]]] = []
            for candidate in candidates:
                evidence, recovered = _classify_staging(
                    candidate,
                    attempt_path=attempt_path,
                    authority=authority,
                    test_capability=test_capability,
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
                    authority=authority,
                    test_capability=test_capability,
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
                    authority,
                    attempt_path=attempt_path,
                    recovery_events=recovery_events,
                    test_capability=test_capability,
                )
                _prepare_new_staging(
                    active_staging,
                    reservation=reservation,
                    attempt_path=attempt_path,
                )
            if failure_injection == "before_atomic_claim":
                raise RuntimeError("injected failure before atomic reservation claim")
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
            _transition_bound(
                authority,
                attempt_path=attempt_path,
                from_states=("claiming",),
                to_state="claimed",
                test_capability=test_capability,
            )
            if failure_injection == "after_atomic_claim":
                raise RuntimeError("injected failure after atomic reservation claim")
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
        try:
            _transition_bound(
                authority,
                attempt_path=attempt_path,
                from_states=("claiming", "claimed"),
                to_state="terminal",
                test_capability=test_capability,
            )
        except PermissionError:
            pass
        raise


def _reserve_attempt(
    authority: policy.VerifiedAuthorityV2,
    *,
    failure_injection: str | None = None,
) -> AttemptReservation:
    """Production claim has no caller-controlled path surface."""

    policy.transition_authority(
        authority,
        purpose="exact_run",
        target_path=policy.CANONICAL_ATTEMPT_PATH,
        from_states=("active",),
        to_state="claiming",
    )
    return _reserve_bound_attempt(
        authority,
        attempt_path=policy.CANONICAL_ATTEMPT_PATH,
        test_capability=None,
        failure_injection=failure_injection,
    )


def _reserve_attempt_for_test(
    authority: policy.VerifiedAuthorityV2,
    *,
    test_capability: policy.TestAuthorityCapabilityV2,
    attempt_path: Path,
    failure_injection: str | None = None,
) -> AttemptReservation:
    """Irreversibly test-only path injection, rejected by production functions."""

    bound = Path(attempt_path).resolve()
    test_capability.transition(
        authority,
        target_path=bound,
        from_states=("active",),
        to_state="claiming",
    )
    return _reserve_bound_attempt(
        authority,
        attempt_path=bound,
        test_capability=test_capability,
        failure_injection=failure_injection,
    )


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
    authority: policy.VerifiedAuthorityV2,
    maximum_workers: int,
) -> tuple[Any, dict[str, Any]]:
    policy.require_verified_authority(
        authority,
        purpose="exact_run",
        target_path=policy.CANONICAL_ATTEMPT_PATH,
        allowed_states=("claimed",),
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
    review = _review_binding(authority)
    payloads = [
        (job, str(policy.CANONICAL_SOURCE_REVIEW_PATH), review["file_sha256"])
        for job in jobs
    ]
    if worker_count == 1:
        arrays = [_decode_worker(payload) for payload in payloads]
        start_method = "inline_authority_revalidated"
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


def _bind_retained_science() -> Any:
    from scripts import (
        train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained,
    )

    retained.policy = policy
    retained._reserve_attempt = _reserve_attempt
    retained._terminate_failure = _terminate_failure
    retained._publish_success = _publish_success
    retained.decode_selected_rgb = decode_selected_rgb
    return retained


def _run_training(
    authority: policy.VerifiedAuthorityV2,
    *,
    rgb_workers: int,
) -> dict[str, Any]:
    policy.require_verified_authority(
        authority,
        purpose="exact_run",
        target_path=policy.CANONICAL_ATTEMPT_PATH,
        allowed_states=("active",),
    )
    retained = _bind_retained_science()
    summary = retained._run_training(authority, rgb_workers=rgb_workers)
    return {
        **summary,
        "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_launch_summary_v1",
    }


def run_exact(
    authority: policy.VerifiedAuthorityV2,
    *,
    rgb_workers: int,
) -> dict[str, Any]:
    if not sys.flags.isolated:
        raise PermissionError("N5 full-panel V2 exact training requires isolated launcher")
    if isinstance(rgb_workers, bool) or not 1 <= int(rgb_workers) <= 5:
        raise ValueError("N5 full-panel RGB workers must lie in [1,5]")
    policy.transition_authority(
        authority,
        purpose="exact_run",
        target_path=policy.CANONICAL_ATTEMPT_PATH,
        from_states=("issued",),
        to_state="active",
    )
    return _run_training(authority, rgb_workers=int(rgb_workers))


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


def main() -> int:
    raise PermissionError(
        "N5 full-panel V2 trainer cannot execute directly; use reviewed launcher"
    )


if __name__ == "__main__":
    raise SystemExit(main())
