#!/usr/bin/env python3
"""Verifier successor for frozen V4 development-ladder checkpoints.

This additive successor exists only because the reviewed V1 verifier reached
the frozen trainer's launcher-specific multiprocessing guard before RGB decode.
It may reopen the exact completed N5 attempt and verify later frozen rungs only
after a different-agent review. It cannot train, alter a rung, access held-out
data, or promote a checkpoint. Every rung deliberately uses the frozen
trainer's inline decoder instead of inventing a second spawn authority.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
import uuid
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_authorization_2026-07-12.json"
).resolve()
CANONICAL_DEVELOPMENT_ROOT = (
    ROOT / ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
)
CANONICAL_ATTEMPT_ROOT = CANONICAL_DEVELOPMENT_ROOT / "attempts"
CANONICAL_RECEIPT_ROOT = CANONICAL_DEVELOPMENT_ROOT / "metric_verifications"
CANONICAL_N5_ATTEMPT = CANONICAL_ATTEMPT_ROOT / "seed_20260710" / "n5"
CANONICAL_SUCCESSOR_REVIEW_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_finalizer_v2_independent_review_2026-07-13.json"
).resolve()
SUCCESSOR_REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_execution_successor_review_v2"
)
SUCCESSOR_RELATIVE_PATH = (
    "scripts/verify_go2_observable_camera_ray_fit_v4_metrics_v2.py"
)
SUCCESSOR_LOGICAL_NAME = (
    "scripts.verify_go2_observable_camera_ray_fit_v4_metrics_v2"
)
SUCCESSOR_FINALIZER_RELATIVE_PATH = (
    "scripts/finalize_go2_observable_camera_ray_fit_v4_ladder_v2.py"
)
SUCCESSOR_TRAINER_RELATIVE_PATH = (
    "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
)
SUCCESSOR_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_observable_camera_ray_fit_v4_v2.py"
)
SUCCESSOR_SOURCE_PATHS = (
    SUCCESSOR_RELATIVE_PATH,
    SUCCESSOR_FINALIZER_RELATIVE_PATH,
    SUCCESSOR_TRAINER_RELATIVE_PATH,
    SUCCESSOR_LAUNCHER_RELATIVE_PATH,
)
PREDECESSOR_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py"
)
PREDECESSOR_VERIFIER_FILE_SHA256 = (
    "235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f"
)
FAILURE_RECORD_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_prepublication_failure_2026-07-13.md"
)
FAILURE_RECORD_FILE_SHA256 = (
    "d99fc34ca6584348a3a67939722928287affa925b18ed895ef23f6e1e3954842"
)
N5_ARTIFACT_BINDINGS = {
    "reservation.json": {
        "file_sha256": "f5926ee9006df8d163a2d1a17882d82124608ddce319ea0fb5e80fcfe2c2a8aa",
        "content_sha256": "699b4e95ed05cb13a79fe6af8507fae5d987af9ff1977b0e4684f32742aa4943",
    },
    "checkpoint.pt": {
        "file_sha256": "f1739c742f9c19d5e17753da504a547254eb6e1997bb1ac4eca8b188bbf1dcf0",
        "content_sha256": "589060417903167bbf9ce7605c906b25cd802edd73b79ec607c77403c6df305a",
        "byte_count": 13_778_252,
    },
    "result.json": {
        "file_sha256": "39030bb7928a6b078b03156dc9e14fb206c60c73ab2acac88bfd307c5a65bbfa",
        "content_sha256": "8c38e13f411a5cd9b03362cb5ac98379875065f284a75ac894706944ff252b61",
        "byte_count": 27_102_689,
    },
    "completed.json": {
        "file_sha256": "4fb9b5629f039ac16692ec6e171a8188f3bf8b7d052ac8cde26b8ac86c10f6af",
        "content_sha256": "48022dca829a73b7cbd3b665ac7679807825a9aefd56a48e752ae07e6eaa336f",
    },
}
AUTHORIZATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_metric_verifier_authorization_v1"
)
METRIC_VERIFICATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_metric_verification_v2"
)
AUTHORIZATION_FILE_SHA256 = (
    "091d26f6be0372c003528be370028e6f431bcdef9770ce3855d8b1cf4045a3cf"
)
AUTHORIZATION_CONTENT_SHA256 = (
    "c4090f47b417d5766f5d5100615b2f1c3891a8340e2813ad089bf894beeb98d2"
)
TARGET_PARTITION_FREEZE_FILE_SHA256 = (
    "4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a"
)
TARGET_PARTITION_FREEZE_CONTENT_SHA256 = (
    "8dd54d178e3c00a8622d89e4e371a115e1391f34588f667c20cd95b970fc68d2"
)
TARGET_PARTITION_VERIFIER_FILE_SHA256 = (
    "4624dd761901808c72b37eb256b360e3db61c9b8f61337879547ed38836a3eed"
)
TARGET_PARTITION_AMENDMENT_FILE_SHA256 = (
    "1e65f8884b1b8e0ad2219ddad54f79f9fabae514bfcaa048b29c8113b076ac1f"
)
TARGET_PARTITION_BOUNDARY = {
    "freeze_file_sha256": TARGET_PARTITION_FREEZE_FILE_SHA256,
    "freeze_content_sha256": TARGET_PARTITION_FREEZE_CONTENT_SHA256,
    "verifier_file_sha256": TARGET_PARTITION_VERIFIER_FILE_SHA256,
    "amendment_file_sha256": TARGET_PARTITION_AMENDMENT_FILE_SHA256,
    "fit_sizes": [5, 16, 32, 320],
    "verified_dataset_file_count": 180,
}
CANONICAL_TRAINER_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json"
).resolve()
LAUNCHER_LOGICAL_NAME = "scripts.launch_go2_observable_camera_ray_fit_v4"
LAUNCHER_RELATIVE_PATH = "scripts/launch_go2_observable_camera_ray_fit_v4.py"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_regular_bytes(path: Path, expected_sha256: str, *, name: str) -> bytes:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"caller {name} SHA-256 is malformed")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"{name} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
        )
        if identity(before) != identity(after):
            raise RuntimeError(f"{name} changed while read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"{name} caller SHA-256 changed")
    return raw


def _strict_json(path: Path, expected_sha256: str, *, name: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, expected_sha256, name=name)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical newline-terminated JSON")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or declared != _sha256_bytes(_canonical_json_bytes(core)):
        raise ValueError(f"{name} content SHA-256 changed")
    return value, raw


def _require_path(actual: Path, expected: Path, *, name: str) -> None:
    if str(actual) != str(expected) or actual.resolve(strict=True) != expected:
        raise PermissionError(f"{name} path is not canonical")


def canonical_metric_receipt_path(seed: int, fit_size: int) -> Path:
    if seed not in (20260710, 20260711) or fit_size not in (5, 16, 32, 320):
        raise ValueError("V4 V2 metric verifier seed/rung is outside the frozen ladder")
    return CANONICAL_RECEIPT_ROOT / f"seed_{seed}_n{fit_size}.json"


def preflight_metric_verifier_authorization(
    path: Path,
    file_sha256: str,
) -> dict[str, Any]:
    """Validate the separate license before any protected or heavy import."""

    _require_path(path, CANONICAL_AUTHORIZATION_PATH, name="V4 metric authorization")
    if file_sha256 != AUTHORIZATION_FILE_SHA256:
        raise ValueError("V4 metric authorization frozen file SHA-256 changed")
    authorization, _raw = _strict_json(
        path,
        file_sha256,
        name="V4 metric authorization",
    )
    expected_fields = {
        "schema",
        "status",
        "authoritative",
        "scope",
        "target_partition_boundary",
        "review",
        "licenses",
        "content_sha256",
    }
    review = authorization.get("review")
    licenses = authorization.get("licenses")
    if set(authorization) != expected_fields or (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("content_sha256") != AUTHORIZATION_CONTENT_SHA256
        or authorization.get("status") != "authorized_after_independent_review"
        or authorization.get("authoritative") is not False
        or authorization.get("scope")
        != "exact_train_only_checkpoint_metric_reverification"
        or authorization.get("target_partition_boundary")
        != TARGET_PARTITION_BOUNDARY
        or review
        != {
            "independent_reviewer": review.get("independent_reviewer")
            if isinstance(review, Mapping)
            else None,
            "review_completed": True,
            "source_closure_approved": True,
            "target_partition_constants_approved": True,
        }
        or not isinstance(review.get("independent_reviewer"), str)
        or not review.get("independent_reviewer")
        or licenses
        != {
            "authorizes_verification_only_checkpoint_use": True,
            "authorizes_selected_train_target_access": True,
            "authorizes_selected_train_rgb_access": True,
            "authorizes_model_inference": True,
            "authorizes_metric_receipt_creation": True,
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        }
    ):
        raise PermissionError("V4 metric verifier is not independently authorized")
    return authorization


def _authorization_binding(authorization: Mapping[str, Any]) -> dict[str, str]:
    return {
        "file_sha256": AUTHORIZATION_FILE_SHA256,
        "content_sha256": str(authorization["content_sha256"]),
    }


def preflight_successor_review(
    path: Path,
    file_sha256: str,
) -> dict[str, Any]:
    """Require a different-agent review before any attempt or heavy import."""

    _require_path(path, CANONICAL_SUCCESSOR_REVIEW_PATH, name="V4 V2 verifier review")
    review, _raw = _strict_json(path, file_sha256, name="V4 V2 verifier review")
    successor_sources = review.get("successor_sources")
    predecessor = review.get("predecessor_verifier")
    failure = review.get("failed_invocation")
    policy = review.get("execution_policy")
    licenses = review.get("licenses")
    reviewer = review.get("reviewer")
    expected_fields = {
        "schema",
        "status",
        "implementation_author",
        "reviewer",
        "review_completed",
        "source_closure_approved",
        "n5_reopen_approved",
        "successor_sources",
        "predecessor_verifier",
        "failed_invocation",
        "n5_artifacts",
        "execution_policy",
        "licenses",
        "content_sha256",
    }
    if set(review) != expected_fields or (
        review.get("schema") != SUCCESSOR_REVIEW_SCHEMA
        or review.get("status") != "different_agent_review_passed_frozen_ladder"
        or review.get("implementation_author") != "/root/g5_perf_closure"
        or not isinstance(reviewer, str)
        or not reviewer.startswith("/root/")
        or reviewer == review.get("implementation_author")
        or review.get("review_completed") is not True
        or review.get("source_closure_approved") is not True
        or review.get("n5_reopen_approved") is not True
        or not isinstance(successor_sources, Mapping)
        or set(successor_sources) != set(SUCCESSOR_SOURCE_PATHS)
        or any(
            not isinstance(successor_sources.get(relative), Mapping)
            or successor_sources[relative].get("path") != relative
            or not _is_sha256(successor_sources[relative].get("file_sha256"))
            for relative in SUCCESSOR_SOURCE_PATHS
        )
    ):
        raise PermissionError("V4 V2 verifier lacks a different-agent source review")
    if predecessor != {
        "path": PREDECESSOR_VERIFIER_RELATIVE_PATH,
        "file_sha256": PREDECESSOR_VERIFIER_FILE_SHA256,
    } or failure != {
        "path": FAILURE_RECORD_RELATIVE_PATH,
        "file_sha256": FAILURE_RECORD_FILE_SHA256,
        "exception": "PermissionError: V4 spawned RGB terminal differs from captured source",
        "phase": "captured_trainer_decode_selected_rgb_before_receipt",
    }:
        raise PermissionError("V4 V2 verifier failure lineage changed")
    if review.get("n5_artifacts") != N5_ARTIFACT_BINDINGS or policy != {
        "seeds": [20260710, 20260711],
        "fit_sizes": [5, 16, 32, 320],
        "selected_rgb_count": "fit_size",
        "verifier_rgb_worker_start_method": "inline",
        "verifier_rgb_worker_count": 1,
        "training_rgb_worker_start_method": "spawn",
        "training_rgb_worker_count_max": 6,
        "native_threads_per_worker": 1,
        "gpu_device": "cuda:0",
        "gpu_name": "AMD Radeon AI PRO R9700",
        "frozen_steps": {"5": 1000, "16": 1200, "32": 1600, "320": 3200},
        "later_rung_prerequisite_validator": SUCCESSOR_FINALIZER_RELATIVE_PATH,
        "second_seed_prerequisite_validator": SUCCESSOR_FINALIZER_RELATIVE_PATH,
        "training_forbidden_in_verifier": True,
        "receipt_exclusive_create": True,
    } or licenses != {
        "authorizes_frozen_ladder_metric_reverification": True,
        "authorizes_stage_finalization": True,
        "authorizes_reviewed_execution_successor": True,
        "requires_frozen_trainer_authorization": True,
        "authorizes_training_configuration_change": False,
        "authorizes_unreviewed_training": False,
        "authorizes_holdout": False,
        "authorizes_g2": False,
        "authorizes_runtime": False,
        "authorizes_promotion": False,
    }:
        raise PermissionError("V4 V2 verifier review scope changed")
    checked = tuple(
        (
            relative,
            str(successor_sources[relative]["file_sha256"]),
        )
        for relative in SUCCESSOR_SOURCE_PATHS
    ) + (
        (PREDECESSOR_VERIFIER_RELATIVE_PATH, PREDECESSOR_VERIFIER_FILE_SHA256),
        (FAILURE_RECORD_RELATIVE_PATH, FAILURE_RECORD_FILE_SHA256),
    )
    for relative, expected in checked:
        _read_regular_bytes(ROOT / relative, expected, name=f"V4 V2 bound source {relative}")
    return review


def build_metric_verification_receipt(
    *,
    gate_module: Any,
    authorization: Mapping[str, Any],
    seed: int,
    fit_size: int,
    result_content_sha256: str,
    checkpoint: Mapping[str, Any],
    target_partition_reproduction: Mapping[str, Any],
    recomputed_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if not __name__.startswith("_lewm_v4_ca_"):
        raise PermissionError("V4 metric receipt library computation is unsupported")
    expected_partition = gate_module.target_partition_binding_v4(fit_size)
    if dict(target_partition_reproduction) != {
        "target_partition": expected_partition,
        "reproduced_before_checkpoint_inference": True,
    }:
        raise ValueError("V4 target-partition reproduction binding changed")
    matched, wrong, signature = gate_module._validated_metric_evaluation(
        recomputed_evaluation,
        fit_size=fit_size,
    )
    numeric = gate_module._gate_stage(
        {"fit_size": fit_size, "matched": matched, "wrong": wrong}
    )
    core = {
        "schema": METRIC_VERIFICATION_SCHEMA,
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "dataset_role": "train",
        "seed": seed,
        "fit_size": fit_size,
        "result_content_sha256": result_content_sha256,
        "checkpoint": {
            key: checkpoint[key]
            for key in ("file_sha256", "content_sha256", "byte_count")
        },
        "metric_verifier_authorization": _authorization_binding(authorization),
        "target_partition_reproduction": dict(target_partition_reproduction),
        "target_partition_signature": signature,
        "target_partition_signature_sha256": gate_module.canonical_json_sha256(
            signature
        ),
        "recomputed_evaluation": dict(recomputed_evaluation),
        "recomputed_evaluation_sha256": gate_module.canonical_json_sha256(
            recomputed_evaluation
        ),
        "numeric_gate": numeric,
        "verification": {
            "checkpoint_loaded": True,
            "selected_train_targets_loaded": True,
            "selected_matched_rgb_loaded": True,
            "wrong_rgb_mapping_rerun": True,
            "all_losses_recomputed": True,
            "all_confusions_recomputed": True,
            "depth_quantiles_and_evidence_recomputed": True,
            "raster_nll_recomputed": True,
            "family_metrics_recomputed": True,
            "gate_decisions_recomputed": True,
            "ordered_target_bytes_reproduced_before_checkpoint_inference": True,
        },
        "licenses": {
            "checkpoint_use_authorized_for_metric_verification_only": True,
            "authorizes_development_checkpoint_use": False,
            "authorizes_new_model_output": False,
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        },
    }
    return {**core, "content_sha256": gate_module.canonical_json_sha256(core)}


def _compute_exact_receipt(
    *,
    authorization: Mapping[str, Any],
    seed: int,
    fit_size: int,
    reservation_path: Path,
    reservation_file_sha256: str,
    result_path: Path,
    result_file_sha256: str,
    checkpoint_path: Path,
    checkpoint_file_sha256: str,
    completion_path: Path,
    completion_file_sha256: str,
    trainer_authorization_path: Path,
    trainer_authorization_file_sha256: str,
    trainer_review_path: Path,
    trainer_review_file_sha256: str,
) -> dict[str, Any]:
    logical_self = SUCCESSOR_LOGICAL_NAME
    if (
        __name__ == logical_self
        or not __name__.startswith("_lewm_v4_ca_")
        or globals().get("__verified_logical_name__") != logical_self
    ):
        raise PermissionError("V4 V2 metric verifier library computation is unsupported")
    supplied = {
        "reservation.json": reservation_file_sha256,
        "checkpoint.pt": checkpoint_file_sha256,
        "result.json": result_file_sha256,
        "completed.json": completion_file_sha256,
    }
    if (
        seed not in (20260710, 20260711)
        or fit_size not in (5, 16, 32, 320)
        or not all(_is_sha256(value) for value in supplied.values())
    ):
        raise PermissionError("V4 V2 verifier frozen ladder binding changed")
    is_reopened_n5 = seed == 20260710 and fit_size == 5
    if is_reopened_n5 and any(
        supplied[name] != binding["file_sha256"]
        for name, binding in N5_ARTIFACT_BINDINGS.items()
    ):
        raise PermissionError("V4 V2 verifier immutable N5 binding changed")
    from scripts import launch_go2_observable_camera_ray_fit_v4 as launcher
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_ladder_gate as gate,
    )
    from scripts import verify_go2_observable_camera_ray_fit_v4_target_partitions as partition_verifier
    from lewm.models.observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4Model,
    )
    from scripts import finalize_go2_observable_camera_ray_fit_v4_ladder as finalizer
    from scripts import train_go2_observable_camera_ray_fit_v4 as trainer

    preauth = launcher.preflight_exact_authorization(
        dataset_path=launcher.CANONICAL_DATASET_PATH,
        dataset_file_sha256=launcher.DATASET_MANIFEST_FILE_SHA256,
        audit_path=launcher.CANONICAL_AUDIT_PATH,
        audit_file_sha256=launcher.AUDIT_RECEIPT_FILE_SHA256,
        authorization_path=trainer_authorization_path,
        authorization_file_sha256=trainer_authorization_file_sha256,
        review_record_path=trainer_review_path,
        review_record_file_sha256=trainer_review_file_sha256,
    )
    attempt = CANONICAL_ATTEMPT_ROOT / f"seed_{seed}" / f"n{fit_size}"
    _require_path(reservation_path, attempt / "reservation.json", name="V4 reservation")
    _require_path(result_path, attempt / "result.json", name="V4 result")
    _require_path(checkpoint_path, attempt / "checkpoint.pt", name="V4 checkpoint")
    _require_path(completion_path, attempt / "completed.json", name="V4 completion")
    if attempt.is_symlink() or not attempt.is_dir():
        raise PermissionError("V4 completed attempt directory changed")
    attempt_entries = tuple(attempt.iterdir())
    if {entry.name for entry in attempt_entries} != set(N5_ARTIFACT_BINDINGS) or any(
        entry.is_symlink() or not entry.is_file() for entry in attempt_entries
    ):
        raise PermissionError("V4 completed attempt inventory changed")
    reproduced_partitions = partition_verifier.verify_frozen_partitions()
    target_partition = gate.target_partition_binding_v4(fit_size)
    reproduced_entry = reproduced_partitions.get("fit_sizes", {}).get(str(fit_size))
    if (
        reproduced_partitions.get("content_sha256")
        != TARGET_PARTITION_FREEZE_CONTENT_SHA256
        or reproduced_partitions.get("verified_dataset_file_count") != 180
        or set(reproduced_partitions.get("fit_sizes", {}))
        != {"5", "16", "32", "320"}
        or not isinstance(reproduced_entry, Mapping)
        or reproduced_entry.get("fit_size") != fit_size
        or reproduced_entry.get("family_counts")
        != target_partition["family_counts"]
        or reproduced_entry.get("first_frame_key_sha256")
        != target_partition["first_frame_key_sha256"]
        or reproduced_entry.get("last_frame_key_sha256")
        != target_partition["last_frame_key_sha256"]
        or reproduced_entry.get("subset_content_sha256")
        != target_partition["subset_content_sha256"]
        or reproduced_entry.get("signature_sha256")
        != target_partition["signature_sha256"]
        or reproduced_entry.get("ordered_per_frame_target_sha256")
        != target_partition["ordered_per_frame_target_sha256"]
        or reproduced_entry.get("ordered_target_bytes_sha256")
        != target_partition["ordered_target_bytes_sha256"]
    ):
        raise ValueError("V4 reproduced target partition changed")
    target_partition_reproduction = {
        "target_partition": target_partition,
        "reproduced_before_checkpoint_inference": True,
    }
    # Every heavy import remains below both independent authorization checks.
    import torch

    reservation, _reservation_raw = _strict_json(
        reservation_path, reservation_file_sha256, name="V4 reservation"
    )
    result, _result_raw = _strict_json(
        result_path, result_file_sha256, name="V4 result"
    )
    completion, _completion_raw = _strict_json(
        completion_path, completion_file_sha256, name="V4 completion"
    )
    checkpoint_raw = _read_regular_bytes(
        checkpoint_path,
        checkpoint_file_sha256,
        name="V4 checkpoint",
    )
    checkpoint = result.get("model", {}).get("checkpoint", {})
    if (
        reservation.get("schema") != gate.ATTEMPT_RESERVATION_SCHEMA
        or reservation.get("contract") != gate.LADDER_CONTRACT
        or reservation.get("predecessor_failure") != gate.V1_FAILURE_LINEAGE
        or reservation.get("seed") != seed
        or reservation.get("fit_size") != fit_size
        or result.get("determinism", {}).get("seed") != seed
        or result.get("fit_size") != fit_size
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint)
        != {
            "path",
            "file_sha256",
            "content_sha256",
            "byte_count",
            "development_only",
        }
        or checkpoint.get("path") != "checkpoint.pt"
        or checkpoint.get("file_sha256") != checkpoint_file_sha256
        or not _is_sha256(checkpoint.get("content_sha256"))
        or checkpoint.get("byte_count") != len(checkpoint_raw)
        or checkpoint.get("development_only") is not True
    ):
        raise ValueError("V4 metric result seed/rung changed")
    if is_reopened_n5 and (
        reservation.get("content_sha256")
        != N5_ARTIFACT_BINDINGS["reservation.json"]["content_sha256"]
        or result.get("content_sha256")
        != N5_ARTIFACT_BINDINGS["result.json"]["content_sha256"]
        or completion.get("content_sha256")
        != N5_ARTIFACT_BINDINGS["completed.json"]["content_sha256"]
        or checkpoint.get("content_sha256")
        != N5_ARTIFACT_BINDINGS["checkpoint.pt"]["content_sha256"]
        or len(checkpoint_raw)
        != N5_ARTIFACT_BINDINGS["checkpoint.pt"]["byte_count"]
    ):
        raise PermissionError("V4 immutable N5 semantic binding changed")
    expected_completion = {
        "schema": "lewm_go2_observable_camera_ray_fit_v4_attempt_completion_v2",
        "status": "completed",
        "inventory": [
            "checkpoint.pt",
            "completed.json",
            "reservation.json",
            "result.json",
        ],
        "reservation": {
            "path": "reservation.json",
            "file_sha256": reservation_file_sha256,
            "content_sha256": reservation["content_sha256"],
        },
        "checkpoint": {
            "path": "checkpoint.pt",
            "file_sha256": checkpoint_file_sha256,
            "content_sha256": checkpoint["content_sha256"],
            "byte_count": len(checkpoint_raw),
        },
        "result": {
            "path": "result.json",
            "file_sha256": result_file_sha256,
            "content_sha256": result["content_sha256"],
            "byte_count": len(_result_raw),
        },
        "licenses": {
            "checkpoint_use_authorized": False,
            "development_checkpoint_creation_authorized": True,
            "g2_authorized": False,
            "holdout_authorized": False,
            "promotion_authorized": False,
            "runtime_authorized": False,
        },
        "content_sha256": completion["content_sha256"],
    }
    if completion != expected_completion:
        raise PermissionError("V4 completed attempt binding changed")
    expected_metadata = {
        **reservation["inputs"],
        "fit_size": fit_size,
        "seed": seed,
        "training_schedule_sha256": result["training"]["schedule_sha256"],
        "attempt_reservation": result["attempt"]["reservation"],
        "predecessor_failure": reservation["predecessor_failure"],
        "prerequisite_gates": reservation["prerequisite_gates"],
    }
    finalizer._validate_checkpoint(
        checkpoint_raw,
        expected_content_sha256=checkpoint["content_sha256"],
        expected_metadata=expected_metadata,
    )
    inputs = trainer.load_exact_inputs(
        dataset_manifest_path=launcher.CANONICAL_DATASET_PATH,
        dataset_manifest_file_sha256=launcher.DATASET_MANIFEST_FILE_SHA256,
        audit_receipt_path=launcher.CANONICAL_AUDIT_PATH,
        audit_receipt_file_sha256=launcher.AUDIT_RECEIPT_FILE_SHA256,
        trainer_authorization_path=trainer_authorization_path,
        trainer_authorization_file_sha256=trainer_authorization_file_sha256,
        trainer_review_record_path=trainer_review_path,
        trainer_review_record_file_sha256=trainer_review_file_sha256,
        fit_size=fit_size,
    )
    if dict(inputs.subset_receipt) != result.get("subset"):
        raise ValueError("V4 metric selected target subset changed")
    if len(inputs.frames) != fit_size:
        raise PermissionError("V4 V2 verifier selected RGB count changed")
    trainer_source_sha256 = next(
        str(entry["sha256"])
        for entry in preauth["source_map"]["entries"]
        if entry["path"] == "scripts/train_go2_observable_camera_ray_fit_v4.py"
    )
    trainer.validate_gpu0_r9700_runtime(device_text="cuda:0")
    images, _rgb_access = trainer.decode_selected_rgb(
        inputs.frames,
        maximum_workers=1,
        expected_trainer_source_sha256=trainer_source_sha256,
    )
    if _rgb_access != {
        "selected_rgb_count": fit_size,
        "nonselected_rgb_opens": 0,
        "rgb_hash_opens": fit_size,
        "rgb_decodes": fit_size,
        "worker_start_method": "inline",
        "worker_count": 1,
        "native_threads_per_worker": 1,
    }:
        raise PermissionError("V4 V2 verifier inline RGB access changed")
    payload = torch.load(BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
    model = ObservableCameraRayEvidenceV4Model()
    model.load_state_dict(dict(payload["state_dict"]), strict=True)
    device = torch.device("cuda:0")
    model.to(device)
    matched = trainer.evaluate_v4_fit(
        model=model,
        frames=inputs.frames,
        images=images,
        device=device,
        batch_size=1,
        wrong_rgb=False,
    )
    wrong = trainer.evaluate_v4_fit(
        model=model,
        frames=inputs.frames,
        images=images,
        device=device,
        batch_size=1,
        wrong_rgb=True,
    )
    evaluation = {
        "matched_rgb": matched,
        "wrong_rgb_with_target_calibration": wrong,
    }
    if _canonical_json_bytes(evaluation) != _canonical_json_bytes(result["evaluation"]):
        raise ValueError("V4 reported metrics differ from the exact inference rerun")
    receipt = build_metric_verification_receipt(
        gate_module=gate,
        authorization=authorization,
        seed=seed,
        fit_size=fit_size,
        result_content_sha256=result["content_sha256"],
        checkpoint=checkpoint,
        target_partition_reproduction=target_partition_reproduction,
        recomputed_evaluation=evaluation,
    )
    return receipt


def recompute_exact_metric_verification(**kwargs: Any) -> dict[str, Any]:
    authorization = preflight_metric_verifier_authorization(
        Path(kwargs.pop("metric_authorization_path")),
        str(kwargs.pop("metric_authorization_file_sha256")),
    )
    if not __name__.startswith("_lewm_v4_ca_"):
        raise PermissionError("V4 metric verifier library computation is unsupported")
    return _compute_exact_receipt(authorization=authorization, **kwargs)


def reverify_canonical_metric_receipt(
    *,
    receipt_path: Path,
    receipt_file_sha256: str,
    **kwargs: Any,
) -> dict[str, Any]:
    authorization = preflight_metric_verifier_authorization(
        Path(kwargs.pop("metric_authorization_path")),
        str(kwargs.pop("metric_authorization_file_sha256")),
    )
    if not __name__.startswith("_lewm_v4_ca_"):
        raise PermissionError("V4 metric verifier library computation is unsupported")
    seed = int(kwargs["seed"])
    fit_size = int(kwargs["fit_size"])
    _require_path(
        receipt_path,
        canonical_metric_receipt_path(seed, fit_size),
        name="V4 metric verification receipt",
    )
    receipt, raw = _strict_json(
        receipt_path,
        receipt_file_sha256,
        name="V4 metric verification receipt",
    )
    recomputed = _compute_exact_receipt(authorization=authorization, **kwargs)
    if raw != _canonical_json_bytes(recomputed) + b"\n":
        raise ValueError("V4 metric verification receipt is not reproducible")
    return receipt


def _captured_metric_cli(
    args: argparse.Namespace,
) -> int:
    if not __name__.startswith("_lewm_v4_ca_"):
        raise PermissionError("V4 V2 metric verifier CLI dispatch is unsupported")
    authorization = preflight_metric_verifier_authorization(
        args.metric_authorization,
        args.metric_authorization_sha256,
    )
    receipt = _compute_exact_receipt(
        authorization=authorization,
        trainer_authorization_path=args.trainer_authorization,
        trainer_authorization_file_sha256=args.trainer_authorization_sha256,
        trainer_review_path=args.trainer_review_record,
        trainer_review_file_sha256=args.trainer_review_record_sha256,
        reservation_path=args.reservation,
        reservation_file_sha256=args.reservation_sha256,
        result_path=args.result,
        result_file_sha256=args.result_sha256,
        checkpoint_path=args.checkpoint,
        checkpoint_file_sha256=args.checkpoint_sha256,
        completion_path=args.completion,
        completion_file_sha256=args.completion_sha256,
        seed=args.seed,
        fit_size=args.fit_size,
    )
    output = canonical_metric_receipt_path(args.seed, args.fit_size)
    publication = _write_exclusive(output, receipt)
    print((_canonical_json_bytes(publication) + b"\n").decode("ascii"), end="")
    return 0


def _dispatch_captured_metric_cli(
    args: argparse.Namespace,
) -> int:
    if __name__ != "__main__":
        raise PermissionError("V4 V2 metric verifier library execution is unsupported")
    successor_review = preflight_successor_review(
        args.successor_review,
        args.successor_review_sha256,
    )
    preflight_metric_verifier_authorization(
        args.metric_authorization,
        args.metric_authorization_sha256,
    )
    _require_path(
        args.trainer_authorization,
        CANONICAL_TRAINER_AUTHORIZATION_PATH,
        name="V4 trainer authorization",
    )
    trainer_authorization, _trainer_raw = _strict_json(
        args.trainer_authorization,
        args.trainer_authorization_sha256,
        name="V4 trainer authorization",
    )
    source_map = trainer_authorization.get("source_map")
    entries = source_map.get("entries") if isinstance(source_map, Mapping) else None
    if not isinstance(entries, list):
        raise PermissionError("V4 trainer source map is unavailable")
    normalized = []
    launcher_sha = None
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "role", "sha256"}:
            raise ValueError("V4 trainer source-map entry is malformed")
        normalized.append(dict(entry))
        if entry.get("path") == LAUNCHER_RELATIVE_PATH:
            launcher_sha = entry.get("sha256")
    if (
        not _is_sha256(launcher_sha)
        or source_map.get("source_map_sha256")
        != _sha256_bytes(_canonical_json_bytes(normalized))
    ):
        raise ValueError("V4 trainer source-map binding changed")
    launcher_path = (ROOT / LAUNCHER_RELATIVE_PATH).resolve(strict=True)
    launcher_raw = _read_regular_bytes(
        launcher_path,
        str(launcher_sha),
        name="V4 launcher source",
    )
    launcher_name = f"_lewm_v4_ca_launcher_{launcher_sha}_{uuid.uuid4().hex}"
    launcher = types.ModuleType(launcher_name)
    launcher.__file__ = str(launcher_path)
    launcher.__cached__ = None
    launcher.__verified_logical_name__ = LAUNCHER_LOGICAL_NAME
    launcher.__verified_source_sha256__ = launcher_sha
    sys.modules[launcher_name] = launcher
    try:
        exec(
            compile(
                launcher_raw,
                f"v4ca://{launcher_sha}/{LAUNCHER_RELATIVE_PATH}",
                "exec",
            ),
            launcher.__dict__,
        )
    except BaseException:
        sys.modules.pop(launcher_name, None)
        raise
    preauth = launcher.preflight_exact_authorization(
        dataset_path=launcher.CANONICAL_DATASET_PATH,
        dataset_file_sha256=launcher.DATASET_MANIFEST_FILE_SHA256,
        audit_path=launcher.CANONICAL_AUDIT_PATH,
        audit_file_sha256=launcher.AUDIT_RECEIPT_FILE_SHA256,
        authorization_path=args.trainer_authorization,
        authorization_file_sha256=args.trainer_authorization_sha256,
        review_record_path=args.trainer_review_record,
        review_record_file_sha256=args.trainer_review_record_sha256,
    )
    # Both independent authorizations have passed. Only now is the executable
    # graph captured and its loader constructed, entirely inside this terminal.
    import builtins
    import importlib
    import importlib.abc
    import importlib.util

    runtime_relatives = (
        "lewm/__init__.py",
        "lewm/benchmarks/__init__.py",
        "lewm/benchmarks/counterfactual.py",
        "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py",
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py",
        "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py",
        "lewm/models/__init__.py",
        "lewm/models/encoders.py",
        "lewm/models/lewm.py",
        "lewm/models/observable_camera_ray_evidence_v4.py",
        "lewm/models/observable_camera_ray_evidence_v4_training.py",
        "lewm/models/phase2d_spatial_lewm.py",
        "lewm/models/predictor.py",
        "lewm/models/primitive_affordance.py",
        "lewm/models/sigreg.py",
        "lewm/models/source_action_utility.py",
        "lewm/models/spatial_lewm.py",
        "lewm/models/spatial_predictor.py",
        "scripts/finalize_go2_observable_camera_ray_fit_v4_ladder.py",
        "scripts/launch_go2_observable_camera_ray_fit_v4.py",
        "scripts/train_go2_observable_camera_ray_fit_v4.py",
        "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py",
        SUCCESSOR_RELATIVE_PATH,
        "scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py",
    )

    def logical_name(relative: str) -> str:
        value = relative[:-3].replace("/", ".")
        return value[:-9] if value.endswith(".__init__") else value

    entry_hashes = {
        str(row["path"]): str(row["sha256"])
        for row in preauth["source_map"]["entries"]
    }
    entry_hashes[SUCCESSOR_RELATIVE_PATH] = str(
        successor_review["successor_sources"][SUCCESSOR_RELATIVE_PATH][
            "file_sha256"
        ]
    )
    captured: dict[str, tuple[Path, bytes, str]] = {}
    for relative in runtime_relatives:
        path = (ROOT / relative).resolve(strict=True)
        source = _read_regular_bytes(
            path,
            entry_hashes.get(relative, ""),
            name=f"V4 metric runtime source {relative}",
        )
        captured[logical_name(relative)] = (
            path,
            source,
            _sha256_bytes(source),
        )
    preloaded = sorted(name for name in captured if name in sys.modules)
    if preloaded:
        raise PermissionError(f"V4 metric runtime modules were preloaded: {preloaded}")
    allowed_roots = frozenset({*sys.stdlib_module_names, "PIL", "numpy", "torch"})
    namespace = f"_lewm_v4_ca_{preauth['source_map_sha256'][:12]}_{uuid.uuid4().hex}"

    class Loader(importlib.abc.Loader):
        def __init__(self, logical: str, record: tuple[Path, bytes, str]) -> None:
            self.logical = logical
            self.path, self.source, self.digest = record

        def create_module(self, spec: Any) -> None:
            return None

        def exec_module(self, module: types.ModuleType) -> None:
            module.__file__ = str(self.path)
            module.__cached__ = None
            module.__verified_source_sha256__ = self.digest
            module.__verified_logical_name__ = self.logical
            verified_builtins = dict(vars(builtins))
            verified_builtins["__import__"] = finder.verified_import
            module.__builtins__ = verified_builtins
            filename = f"v4ca://{self.digest}/{self.logical.replace('.', '/')}"
            exec(compile(self.source, filename, "exec", dont_inherit=True), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def synthetic(self, logical: str) -> str:
            return f"{namespace}.{logical}"

        def verified_import(
            self,
            name: str,
            globals: Mapping[str, Any] | None = None,
            locals: Mapping[str, Any] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> Any:
            if level:
                return builtins.__import__(name, globals, locals, fromlist, level)
            tracked = name in captured or any(key.startswith(f"{name}.") for key in captured)
            if not tracked:
                if name.split(".", 1)[0] not in allowed_roots:
                    raise ImportError(f"V4 metric runtime import is not whitelisted: {name}")
                return builtins.__import__(name, globals, locals, fromlist, level)
            translated = self.synthetic(name)
            builtins.__import__(translated, globals, locals, fromlist, 0)
            if fromlist:
                return sys.modules[translated]
            return sys.modules[self.synthetic(name.split(".", 1)[0])]

        def find_spec(self, fullname: str, path: object = None, target: object = None) -> Any:
            prefix = f"{namespace}."
            logical = fullname[len(prefix) :] if fullname.startswith(prefix) else None
            record = captured.get(logical or "")
            if record is None:
                return None
            return importlib.util.spec_from_loader(
                fullname,
                Loader(str(logical), record),
                origin=str(record[0]),
                is_package=record[0].name == "__init__.py",
            )

    finder = Finder()
    root = types.ModuleType(namespace)
    root.__path__ = []
    root.__package__ = namespace
    sys.modules[namespace] = root
    scripts_name = finder.synthetic("scripts")
    scripts_package = types.ModuleType(scripts_name)
    scripts_package.__path__ = []
    scripts_package.__package__ = scripts_name
    sys.modules[scripts_name] = scripts_package
    sys.meta_path.insert(0, finder)
    fingerprints: dict[str, str] = {}

    def load(logical: str) -> types.ModuleType:
        module = importlib.import_module(finder.synthetic(logical))
        path, source, digest = captured[logical]
        if (
            module is not sys.modules.get(finder.synthetic(logical))
            or getattr(module, "__verified_logical_name__", None) != logical
            or getattr(module, "__verified_source_sha256__", None) != digest
            or Path(str(getattr(module, "__file__", ""))).resolve(strict=True) != path
            or _sha256_bytes(source) != digest
        ):
            raise PermissionError(f"V4 metric loaded module identity changed: {logical}")
        fingerprint = launcher._module_code_sha256(module)
        if fingerprints.setdefault(logical, fingerprint) != fingerprint:
            raise PermissionError(f"V4 metric loaded module code changed: {logical}")
        return module

    logical_self = SUCCESSOR_LOGICAL_NAME
    try:
        private = load(logical_self)
        live = sys.modules.get(__name__)
        if not isinstance(live, types.ModuleType) or (
            launcher._module_code_sha256(live)
            != launcher._module_code_sha256(private)
        ):
            raise PermissionError("V4 live V2 metric verifier differs from captured source")
        result = int(private._captured_metric_cli(args))
        for logical in tuple(fingerprints):
            load(logical)
        return result
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(f"{namespace}."):
                sys.modules.pop(name, None)
        sys.modules.pop(launcher_name, None)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if not __name__.startswith("_lewm_v4_ca_"):
        raise PermissionError("V4 metric receipt library publication is unsupported")
    expected = canonical_metric_receipt_path(int(value["seed"]), int(value["fit_size"]))
    if (
        str(path) != str(expected)
        or path.resolve() != expected
        or CANONICAL_RECEIPT_ROOT.is_symlink()
    ):
        raise PermissionError("V4 metric receipt output path is not canonical")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PermissionError("V4 metric receipt root is not a real directory")
    payload = _canonical_json_bytes(value) + b"\n"
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "file_sha256": _sha256_bytes(payload),
        "content_sha256": value["content_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-review", type=Path, required=True)
    parser.add_argument("--successor-review-sha256", required=True)
    parser.add_argument("--metric-authorization", type=Path, required=True)
    parser.add_argument("--metric-authorization-sha256", required=True)
    parser.add_argument("--trainer-authorization", type=Path, required=True)
    parser.add_argument("--trainer-authorization-sha256", required=True)
    parser.add_argument("--trainer-review-record", type=Path, required=True)
    parser.add_argument("--trainer-review-record-sha256", required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--reservation-sha256", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(20260710, 20260711), required=True)
    parser.add_argument("--fit-size", type=int, choices=(5, 16, 32, 320), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not sys.flags.isolated:
        if __name__ != "__main__":
            raise PermissionError("V4 V2 metric verifier library execution is unsupported")
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        return int(
            subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *raw_argv],
                cwd=ROOT,
                env=environment,
                check=False,
            ).returncode
        )
    args = parse_args(raw_argv)
    return _dispatch_captured_metric_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
