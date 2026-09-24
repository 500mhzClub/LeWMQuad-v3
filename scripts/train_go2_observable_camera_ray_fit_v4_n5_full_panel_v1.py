#!/usr/bin/env python3
"""Authorized trainer for the single V4 N5 full-panel development attempt.

The module has no heavy imports at import time. Exact execution is available
only through the isolated launcher after a different-agent source review.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import warnings

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as policy,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINER_RELATIVE_PATH = "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
BASE_TRAINER_FILE_SHA256 = policy.FROZEN_SOURCE_BINDINGS[
    BASE_TRAINER_RELATIVE_PATH
]


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


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _set_thread_caps() -> None:
    for name in policy.THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def _review_binding(authority: policy.VerifiedAuthority) -> dict[str, str]:
    return policy.source_review_binding(authority)


def _exact_input_binding(
    authority: policy.VerifiedAuthority,
) -> dict[str, str]:
    review = _review_binding(authority)
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
    authority: policy.VerifiedAuthority,
) -> dict[str, Any]:
    policy.require_verified_authority(authority)
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
        "source_review": _review_binding(authority),
        "inputs": _exact_input_binding(authority),
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


def _reserve_attempt(
    authority: policy.VerifiedAuthority,
    *,
    attempt_path: Path = policy.CANONICAL_ATTEMPT_PATH,
    failure_injection: str | None = None,
) -> AttemptReservation:
    """Atomically claim the one attempt with a complete reservation already inside."""

    policy.require_verified_authority(authority)
    attempt_path = Path(attempt_path)
    seed_root = attempt_path.parent
    seed_root.mkdir(parents=True, exist_ok=True)
    if seed_root.is_symlink() or not seed_root.is_dir():
        raise PermissionError("N5 full-panel seed root is not a real directory")
    if attempt_path.exists():
        raise FileExistsError("the sole N5 full-panel attempt is already claimed")
    staging = seed_root / f".{attempt_path.name}.reservation-staging"
    os.mkdir(staging, 0o755)
    core = _reservation_core(authority)
    value = {**core, "content_sha256": policy.canonical_json_sha256(core)}
    raw = policy.canonical_json_bytes(value) + b"\n"
    reservation = AttemptReservation(
        directory=attempt_path,
        value=value,
        raw=raw,
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    claimed = False
    try:
        _write_bytes_exclusive(staging / "reservation.json", raw)
        _fsync_directory(staging)
        if failure_injection == "before_atomic_claim":
            raise RuntimeError("injected failure before atomic reservation claim")
        os.rename(staging, attempt_path)
        claimed = True
        if failure_injection == "after_atomic_claim":
            raise RuntimeError("injected failure after atomic reservation claim")
        _fsync_directory(seed_root)
        return reservation
    except BaseException as error:
        if claimed:
            try:
                _terminate_failure(reservation, error)
            except BaseException as terminal_error:
                raise RuntimeError(
                    "reservation claim failed and terminal receipt could not be written"
                ) from terminal_error
        elif staging.exists():
            shutil.rmtree(staging)
        raise


def _decode_worker(
    payload: tuple[tuple[str, str, str, str], str, str],
) -> Any:
    """Spawned CPU decoder that rebinds authority before opening one RGB."""

    job, review_path_text, review_sha256 = payload
    _set_thread_caps()
    policy.preflight_static_authority()
    policy.preflight_source_review(Path(review_path_text), review_sha256)
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    return base._decode_rgb_job(*job)


def decode_selected_rgb(
    frames: Sequence[Any],
    *,
    authority: policy.VerifiedAuthority,
    maximum_workers: int,
) -> tuple[Any, dict[str, Any]]:
    policy.require_verified_authority(authority)
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


def evaluate_full_panel_v1(
    *,
    model: Any,
    frames: Sequence[Any],
    images: Any,
    device: Any,
    wrong_rgb: bool,
) -> dict[str, Any]:
    """Batch-one evaluation with a structurally exact aggregate loss total."""

    import torch
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    if len(frames) != 5:
        raise ValueError("N5 full-panel evaluation requires exactly five frames")
    model.eval()
    accumulator = base.ObservableCameraRayFitV4MetricAccumulator()
    component_sums: Counter[str] = Counter()
    mapping = tuple(
        ((index + 1) % len(frames)) if wrong_rgb else index
        for index in range(len(frames))
    )
    with torch.no_grad():
        for target_index in range(len(frames)):
            batch = base._batch_from_indices(
                frames,
                images,
                (target_index,),
                image_indices=(mapping[target_index],),
            ).to(device)
            _total, components, raw, targets, soft_raster = (
                base.compute_four_equal_v4_losses(model, batch)
            )
            for name in policy.LOSS_COMPONENTS:
                component_sums[name] += float(components[name].item())
            accumulator.update(
                raw_output=raw,
                targets=targets,
                soft_raster=soft_raster,
                target_raster_labels=batch.target_raster_labels,
                families=batch.families,
            )
    component_means = {
        name: component_sums[name] / len(frames) for name in policy.LOSS_COMPONENTS
    }
    total = 0.25 * sum(component_means[name] for name in policy.LOSS_COMPONENTS)
    losses = {**component_means, "total": total}
    return {
        "control": (
            "wrong_rgb_with_target_calibration" if wrong_rgb else "matched_rgb"
        ),
        "wrong_rgb_degenerate_singleton": False,
        "image_index_mapping": list(mapping),
        "image_mapping_sha256": policy.canonical_json_sha256(list(mapping)),
        "losses": losses,
        "metrics": accumulator.finalize(),
    }


def revalidate_selected_rgb_before_publication(
    base: Any,
    frames: Sequence[Any],
) -> int:
    commitments = tuple((frame.rgb_path, frame.image_sha256) for frame in frames)
    if len(commitments) != 5:
        raise ValueError("N5 full-panel publication requires five RGB commitments")
    base._verify_file_commitments(
        commitments,
        name="N5 full-panel selected train RGB before publication",
    )
    return len(commitments)


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


def _run_training(
    authority: policy.VerifiedAuthority,
    *,
    rgb_workers: int,
) -> dict[str, Any]:
    """Run exact work after launcher authority; caller owns failure termination."""

    policy.require_verified_authority(authority)
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    dataset_path = ROOT / policy.DATASET_MANIFEST_RELATIVE_PATH
    audit_path = ROOT / policy.AUDIT_RECEIPT_RELATIVE_PATH
    trainer_authorization_path = ROOT / policy.TRAINER_AUTHORIZATION_RELATIVE_PATH
    trainer_review_path = ROOT / policy.TRAINER_REVIEW_RELATIVE_PATH
    base.preflight_exact_frozen_dataset_provenance(
        dataset_manifest_path=dataset_path,
        dataset_manifest_file_sha256=policy.DATASET_MANIFEST_FILE_SHA256,
    )
    inputs = base.load_exact_inputs(
        dataset_manifest_path=dataset_path,
        dataset_manifest_file_sha256=policy.DATASET_MANIFEST_FILE_SHA256,
        audit_receipt_path=audit_path,
        audit_receipt_file_sha256=policy.AUDIT_RECEIPT_FILE_SHA256,
        trainer_authorization_path=trainer_authorization_path,
        trainer_authorization_file_sha256=policy.TRAINER_AUTHORIZATION_FILE_SHA256,
        trainer_review_record_path=trainer_review_path,
        trainer_review_record_file_sha256=policy.TRAINER_REVIEW_FILE_SHA256,
        fit_size=5,
    )
    target_partition = base.validate_exact_target_partition_v4(
        inputs.frames,
        fit_size=5,
    )
    if (
        inputs.subset_receipt.get("content_sha256") != policy.SUBSET_CONTENT_SHA256
        or target_partition.get("content_sha256")
        != policy.TARGET_PARTITION_CONTENT_SHA256
        or len(inputs.frames) != 5
    ):
        raise PermissionError("N5 full-panel selected subset or target changed")
    reservation = _reserve_attempt(authority)
    try:
        resource = base.validate_gpu0_r9700_runtime(device_text="cuda:0")
        determinism = base.configure_determinism(20260710)
        images, rgb_access = decode_selected_rgb(
            inputs.frames,
            authority=authority,
            maximum_workers=rgb_workers,
        )
        model = base.ObservableCameraRayEvidenceV4Model()
        device = base.torch.device("cuda:0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            training = base.train_v4_fit(
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                steps=400,
                batch_size=5,
                learning_rate=1e-4,
                weight_decay=1e-4,
                seed=20260710,
            )
            training = {
                **training,
                "evaluation_batch_size": 1,
                "checkpoint_selection": "final_update_only",
                "frame_exposures": 2000,
                "fresh_model_initialization": True,
            }
            if training.get("schedule_sha256") != policy.EXPECTED_SCHEDULE_SHA256:
                raise ValueError("N5 full-panel seeded schedule changed")
            matched = evaluate_full_panel_v1(
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                wrong_rgb=False,
            )
            wrong = evaluate_full_panel_v1(
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                wrong_rgb=True,
            )
        warning_receipt = base.validate_determinism_warnings(
            [item.message for item in caught]
        )
        post_training = base.revalidate_exact_inputs_after_training(
            inputs,
            dataset_manifest_path=dataset_path,
            dataset_manifest_file_sha256=policy.DATASET_MANIFEST_FILE_SHA256,
            audit_receipt_path=audit_path,
            audit_receipt_file_sha256=policy.AUDIT_RECEIPT_FILE_SHA256,
            trainer_authorization_path=trainer_authorization_path,
            trainer_authorization_file_sha256=policy.TRAINER_AUTHORIZATION_FILE_SHA256,
            trainer_review_record_path=trainer_review_path,
            trainer_review_record_file_sha256=policy.TRAINER_REVIEW_FILE_SHA256,
        )
        selected_rgb_rehashes = revalidate_selected_rgb_before_publication(
            base,
            inputs.frames,
        )
        checkpoint_metadata = {
            "experiment": policy.EXPERIMENT,
            "authority_bindings": policy.AUTHORITY_BINDINGS,
            "source_review": _review_binding(authority),
            "inputs": _exact_input_binding(authority),
            "attempt_reservation": reservation.binding,
            "training_schedule_sha256": training["schedule_sha256"],
            "checkpoint_selection": "final_update_only",
        }
        checkpoint_raw, checkpoint_content_sha256 = base._checkpoint_bytes(
            model,
            metadata=checkpoint_metadata,
        )
        checkpoint_sha256 = hashlib.sha256(checkpoint_raw).hexdigest()
        checkpoint_binding = {
            "path": "checkpoint.pt",
            "file_sha256": checkpoint_sha256,
            "content_sha256": checkpoint_content_sha256,
            "byte_count": len(checkpoint_raw),
            "development_only": True,
        }
        result_core = {
            "schema": policy.RESULT_SCHEMA,
            "mode": "exact_train_only_n5_full_panel_development_fit",
            "authoritative": False,
            "aggregation_eligible": False,
            "promotion_eligible": False,
            "dataset_role": "train",
            "seed": 20260710,
            "fit_size": 5,
            "experiment": policy.EXPERIMENT,
            "authority_bindings": policy.AUTHORITY_BINDINGS,
            "source_review": _review_binding(authority),
            "attempt": {
                "attempt_index": 1,
                "maximum_attempts": 1,
                "scope": "one_exclusive_fresh_full_panel_attempt",
                "reservation": reservation.binding,
            },
            "subset": inputs.subset_receipt,
            "target_partition": target_partition,
            "inputs": _exact_input_binding(authority),
            "model": {
                "class": "ObservableCameraRayEvidenceV4Model",
                "fresh_initialization": True,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "checkpoint": checkpoint_binding,
            },
            "training": training,
            "evaluation": {
                "matched_rgb": matched,
                "wrong_rgb_with_target_calibration": wrong,
            },
            "resource": resource,
            "determinism": {**determinism, **warning_receipt},
            "access_ledger": {
                **rgb_access,
                **post_training,
                "selected_rgb_rehashes_before_publication": selected_rgb_rehashes,
                "heldout_opens": 0,
                "g2_opens": 0,
                "selection_opens": 0,
                "calibration_opens": 0,
                "runtime_opens": 0,
                "hardware_opens": 0,
                "production_opens": 0,
                "gpu1_uses": 0,
            },
            "licenses": {
                "development_checkpoint_creation_authorized": True,
                "checkpoint_use_authorized": False,
                "retry_authorized": False,
                "n16_execution_authorized": False,
                "second_seed_authorized": False,
                "v5_training_authorized": False,
                "holdout_authorized": False,
                "g2_authorized": False,
                "selection_authorized": False,
                "calibration_change_authorized": False,
                "runtime_authorized": False,
                "hardware_authorized": False,
                "production_authorized": False,
                "promotion_authorized": False,
            },
        }
        result = {
            **result_core,
            "content_sha256": policy.canonical_json_sha256(result_core),
        }
        policy.validate_result_structure(
            result,
            expected_source_review=_review_binding(authority),
        )
        publication = _publish_success(
            reservation,
            checkpoint_raw=checkpoint_raw,
            checkpoint_content_sha256=checkpoint_content_sha256,
            result=result,
        )
        return {
            "schema": "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_launch_summary_v1",
            "authoritative": False,
            "seed": 20260710,
            "fit_size": 5,
            "result_content_sha256": result["content_sha256"],
            "publication": publication,
            "metric_verification_required": True,
            "later_rung_execution_authorized": False,
        }
    except BaseException as error:
        try:
            _terminate_failure(reservation, error)
        except BaseException as terminal_error:
            raise RuntimeError(
                "N5 full-panel attempt failed and terminal receipt could not be written"
            ) from terminal_error
        raise


def run_exact(
    authority: policy.VerifiedAuthority,
    *,
    rgb_workers: int,
) -> dict[str, Any]:
    if not sys.flags.isolated:
        raise PermissionError("N5 full-panel exact training requires isolated launcher")
    policy.require_verified_authority(authority)
    if isinstance(rgb_workers, bool) or not 1 <= int(rgb_workers) <= 5:
        raise ValueError("N5 full-panel RGB workers must lie in [1,5]")
    return _run_training(authority, rgb_workers=int(rgb_workers))


def run_cpu_contract_smoke() -> dict[str, Any]:
    """CPU-only schedule and structural-arithmetic smoke; no model or data opens."""

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
        "N5 full-panel trainer cannot execute directly; use the reviewed launcher"
    )


if __name__ == "__main__":
    raise SystemExit(main())
