#!/usr/bin/env python3
"""Reviewed stdlib-only execution successor for exact V4 development fitting.

No repository, NumPy, Torch, PIL, dataset, audit, or RGB module is imported
until both the frozen trainer authority and the additive successor-source
review pass. The numerical and data contracts remain frozen.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types
from types import MappingProxyType
import uuid
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = (
    ROOT / "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
).resolve()
CANONICAL_DATASET_PATH = (
    ROOT / ".generated/go2_observable_camera_ray_fit_v4/v1/manifest.json"
).resolve()
CANONICAL_AUDIT_PATH = (
    ROOT / ".generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json"
).resolve()
CANONICAL_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json"
).resolve()
CANONICAL_REVIEW_RECORD_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json"
).resolve()
UPSTREAM_IMPLEMENTATION_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_2026-07-12.json"
).resolve()
CANONICAL_SUCCESSOR_REVIEW_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_finalizer_v2_independent_review_2026-07-13.json"
).resolve()

DATASET_MANIFEST_FILE_SHA256 = (
    "2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85"
)
DATASET_MANIFEST_CONTENT_SHA256 = (
    "9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812"
)
AUDIT_RECEIPT_FILE_SHA256 = (
    "2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c"
)
AUDIT_RECEIPT_CONTENT_SHA256 = (
    "a922114b7e42552043a487bae527c35fb511804d4e8683c5a3f64a2bf499cf76"
)
UPSTREAM_IMPLEMENTATION_FILE_SHA256 = (
    "aa882ae7cc7b038028acf73e4addc049e030a7d3fe7fd1ceb0ff9ded1e464e0e"
)
UPSTREAM_IMPLEMENTATION_CONTENT_SHA256 = (
    "17440ae679d1e730f8f37b2fe62de9bef5029e69198b4969d6ff8990bd38d90b"
)
UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256 = (
    "a22989bcd64c2e79fbb2a06743622fd7ca14332d0b4715f2c982af083a2061bd"
)

AUTHORIZATION_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_v2"
REVIEW_RECORD_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_trainer_review_v1"
VERIFIED_CONTEXT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_verified_context_v1"
SUCCESSOR_REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_execution_successor_review_v2"
)
SUCCESSOR_VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_observable_camera_ray_fit_v4_metrics_v2.py"
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
    SUCCESSOR_VERIFIER_RELATIVE_PATH,
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
SUCCESSOR_EXECUTION_POLICY = {
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
}
SUCCESSOR_LICENSES = {
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
}
SUPPORTED_FIT_SIZES = (5, 16, 32, 320)
EXPECTED_SEEDS = (20260710, 20260711)
DEFAULT_STEPS = {5: 1000, 16: 1200, 32: 1600, 320: 3200}
MAX_RGB_WORKERS = 6
EXPECTED_FIT_CONTRACT = {
    "ladder_contract": "observable_camera_ray_fit_v4_ladder_v3",
    "development_output_root": (
        ".generated/go2_observable_camera_ray_fit_v4/development_fit_v2"
    ),
    "ladder_v3_amendment_file_sha256": (
        "86718d072fe151b9419318c204d4130147e098150d4fd80557f9d5865dc8f9f3"
    ),
    "v1_failure_lineage": {
        "reservation_file_sha256": (
            "115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e"
        ),
        "reservation_content_sha256": (
            "ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd"
        ),
        "failure_file_sha256": (
            "6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48"
        ),
        "failure_content_sha256": (
            "7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4"
        ),
    },
    "dataset_schema": "lewm_go2_observable_camera_ray_fit_v4_dataset_v1",
    "audit_schema": "lewm_go2_observable_camera_ray_fit_v4_audit_v1",
    "rgb_receipt_schema": "lewm_go2_observable_camera_ray_fit_v4_rgb_receipt_v1",
    "dataset_role": "train",
    "exact_frame_count": 320,
    "exact_scene_shard_count": 20,
    "fit_panel_file_sha256": (
        "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
    ),
    "fit_panel_content_sha256": (
        "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
    ),
    "fit_rows_sha256": (
        "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
    ),
    "target_partition_freeze_file_sha256": (
        "4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a"
    ),
    "target_partition_freeze_content_sha256": (
        "8dd54d178e3c00a8622d89e4e371a115e1391f34588f667c20cd95b970fc68d2"
    ),
    "target_partition_verifier_file_sha256": (
        "4624dd761901808c72b37eb256b360e3db61c9b8f61337879547ed38836a3eed"
    ),
    "target_partition_amendment_file_sha256": (
        "1e65f8884b1b8e0ad2219ddad54f79f9fabae514bfcaa048b29c8113b076ac1f"
    ),
    "target_partition_verified_dataset_file_count": 180,
}

REQUIRED_SOURCE_ROLES = {
    "docs/lewm_go2_observable_camera_ray_evidence_v4_contract_2026-07-12.md": "evidence_contract_document",
    "docs/lewm_go2_observable_camera_ray_fit_v4_implementation_manifest_2026-07-12.json": "upstream_builder_implementation_manifest",
    "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_gate_2026-07-12.md": "ladder_gate_contract_document",
    "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v2_partition_amendment_2026-07-12.md": "target_partition_amendment",
    "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v3_failure_successor_amendment_2026-07-13.md": "ladder_v3_failure_successor_amendment",
    "docs/lewm_go2_observable_camera_ray_fit_v4_metric_verifier_authorization_2026-07-12.json": "metric_verifier_authorization_policy",
    "docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json": "target_partition_freeze",
    "lewm/__init__.py": "lewm_package_initializer",
    "lewm/benchmarks/__init__.py": "benchmark_package_initializer",
    "lewm/benchmarks/counterfactual.py": "benchmark_initializer_counterfactual_dependency",
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py": "evidence_core",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py": "ladder_gate_core",
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py": "fit_metrics",
    "lewm/models/__init__.py": "model_package_initializer",
    "lewm/models/encoders.py": "shared_encoder",
    "lewm/models/lewm.py": "model_initializer_lewm_dependency",
    "lewm/models/observable_camera_ray_evidence_v4.py": "evidence_model",
    "lewm/models/observable_camera_ray_evidence_v4_training.py": "training_mechanics",
    "lewm/models/phase2d_spatial_lewm.py": "model_initializer_phase2d_dependency",
    "lewm/models/predictor.py": "model_initializer_predictor_dependency",
    "lewm/models/primitive_affordance.py": "model_initializer_affordance_dependency",
    "lewm/models/sigreg.py": "model_initializer_sigreg_dependency",
    "lewm/models/source_action_utility.py": "model_initializer_source_utility_dependency",
    "lewm/models/spatial_lewm.py": "model_initializer_spatial_lewm_dependency",
    "lewm/models/spatial_predictor.py": "model_initializer_spatial_predictor_dependency",
    "scripts/audit_go2_observable_camera_ray_fit_v4.py": "exact_dataset_auditor",
    "scripts/build_go2_observable_camera_ray_fit_v4.py": "exact_dataset_builder",
    "scripts/finalize_go2_observable_camera_ray_fit_v4_ladder.py": "ladder_gate_finalizer",
    "scripts/launch_go2_observable_camera_ray_fit_v4.py": "stdlib_preauthorization_launcher",
    "scripts/train_go2_observable_camera_ray_fit_v4.py": "development_trainer",
    "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py": "metric_verifier_finalizer",
    "scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py": "target_partition_verifier",
    "lewm/tests/test_audit_go2_observable_camera_ray_fit_v4.py": "test_exact_dataset_auditor",
    "lewm/tests/test_build_go2_observable_camera_ray_fit_v4.py": "test_exact_dataset_builder",
    "lewm/tests/test_finalize_go2_observable_camera_ray_fit_v4_ladder.py": "test_ladder_gate_finalizer",
    "lewm/tests/test_go2_observable_camera_ray_evidence_v4.py": "test_evidence_core",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_ladder_gate.py": "test_ladder_gate_core",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_metrics.py": "test_fit_metrics",
    "lewm/tests/test_launch_go2_observable_camera_ray_fit_v4.py": "test_stdlib_preauthorization_launcher",
    "lewm/tests/test_observable_camera_ray_evidence_v4_model.py": "test_evidence_model",
    "lewm/tests/test_observable_camera_ray_evidence_v4_training.py": "test_training_mechanics",
    "lewm/tests/test_train_go2_observable_camera_ray_fit_v4.py": "test_development_trainer",
    "lewm/tests/test_verify_go2_observable_camera_ray_fit_v4_metrics.py": "test_metric_verifier_finalizer",
}
REQUIRED_SOURCE_PATHS = frozenset(REQUIRED_SOURCE_ROLES)

def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _regular_bytes(path: Path, *, name: str) -> bytes:
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
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"{name} changed while read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _strict_hashed_object(
    path: Path,
    expected_file_sha256: str,
    *,
    name: str,
    canonical_path: Path,
    require_canonical: bool = True,
) -> dict[str, Any]:
    if str(path) != str(canonical_path) or path.resolve(strict=True) != canonical_path:
        raise PermissionError(f"{name} path is not canonical")
    if not _is_sha256(expected_file_sha256):
        raise ValueError(f"caller {name} SHA-256 is malformed")
    raw = _regular_bytes(path, name=name)
    if _sha256_bytes(raw) != expected_file_sha256:
        raise ValueError(f"{name} caller SHA-256 changed")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if require_canonical and raw != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical newline-terminated JSON")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or declared != _sha256_bytes(_canonical_json_bytes(core)):
        raise ValueError(f"{name} content SHA-256 changed")
    return value


def preflight_successor_review(path: Path, file_sha256: str) -> dict[str, Any]:
    """Bind the additive execution sources before protected or heavy imports."""

    review = _strict_hashed_object(
        path,
        file_sha256,
        name="V4 V2 execution successor review",
        canonical_path=CANONICAL_SUCCESSOR_REVIEW_PATH,
    )
    sources = review.get("successor_sources")
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
    if (
        set(review) != expected_fields
        or review.get("schema") != SUCCESSOR_REVIEW_SCHEMA
        or review.get("status") != "different_agent_review_passed_frozen_ladder"
        or review.get("implementation_author") != "/root/g5_perf_closure"
        or not isinstance(reviewer, str)
        or not reviewer.startswith("/root/")
        or reviewer == review.get("implementation_author")
        or review.get("review_completed") is not True
        or review.get("source_closure_approved") is not True
        or review.get("n5_reopen_approved") is not True
        or not isinstance(sources, Mapping)
        or set(sources) != set(SUCCESSOR_SOURCE_PATHS)
        or review.get("predecessor_verifier")
        != {
            "path": PREDECESSOR_VERIFIER_RELATIVE_PATH,
            "file_sha256": PREDECESSOR_VERIFIER_FILE_SHA256,
        }
        or review.get("failed_invocation")
        != {
            "path": FAILURE_RECORD_RELATIVE_PATH,
            "file_sha256": FAILURE_RECORD_FILE_SHA256,
            "exception": (
                "PermissionError: V4 spawned RGB terminal differs from captured source"
            ),
            "phase": "captured_trainer_decode_selected_rgb_before_receipt",
        }
        or review.get("n5_artifacts") != N5_ARTIFACT_BINDINGS
        or review.get("execution_policy") != SUCCESSOR_EXECUTION_POLICY
        or review.get("licenses") != SUCCESSOR_LICENSES
    ):
        raise PermissionError(
            "V4 V2 execution successor lacks a different-agent source review"
        )
    for relative in SUCCESSOR_SOURCE_PATHS:
        binding = sources.get(relative)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != relative
            or not _is_sha256(binding.get("file_sha256"))
        ):
            raise PermissionError("V4 V2 successor source binding changed")
        source = _regular_bytes(
            ROOT / relative,
            name=f"V4 V2 successor source {relative}",
        )
        if _sha256_bytes(source) != binding["file_sha256"]:
            raise PermissionError(f"V4 V2 successor source changed: {relative}")
    for relative, digest in (
        (PREDECESSOR_VERIFIER_RELATIVE_PATH, PREDECESSOR_VERIFIER_FILE_SHA256),
        (FAILURE_RECORD_RELATIVE_PATH, FAILURE_RECORD_FILE_SHA256),
    ):
        source = _regular_bytes(
            ROOT / relative,
            name=f"V4 V2 predecessor lineage {relative}",
        )
        if _sha256_bytes(source) != digest:
            raise PermissionError(f"V4 V2 predecessor lineage changed: {relative}")
    return review


def _validate_source_map(
    source_map: object,
    *,
    root: Path,
) -> str:
    if not isinstance(source_map, Mapping) or set(source_map) != {
        "algorithm",
        "entry_count",
        "entries",
        "source_map_sha256",
    } or source_map.get("algorithm") != "canonical_json_sha256_entries_v1":
        raise ValueError("trainer source-map contract changed")
    entries = source_map.get("entries")
    if not isinstance(entries, list) or source_map.get("entry_count") != len(entries):
        raise ValueError("trainer source-map count changed")
    paths = []
    normalized = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "role", "sha256"}:
            raise ValueError("one trainer source-map entry is malformed")
        relative = entry.get("path")
        role = entry.get("role")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or not isinstance(role, str)
            or not role
            or not _is_sha256(digest)
        ):
            raise ValueError("one trainer source-map commitment is malformed")
        lexical = root / relative
        resolved = lexical.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PermissionError("trainer source path escapes repository") from exc
        if _sha256_bytes(_regular_bytes(lexical, name="trainer source")) != digest:
            raise ValueError(f"trainer source SHA-256 changed: {relative}")
        paths.append(relative)
        normalized.append(dict(entry))
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        raise ValueError("trainer source map is not canonical/injective")
    if set(paths) != REQUIRED_SOURCE_PATHS:
        missing = sorted(REQUIRED_SOURCE_PATHS - set(paths))
        extra = sorted(set(paths) - REQUIRED_SOURCE_PATHS)
        raise PermissionError(f"trainer transitive source closure changed: missing={missing} extra={extra}")
    actual_roles = {str(entry["path"]): str(entry["role"]) for entry in normalized}
    if actual_roles != REQUIRED_SOURCE_ROLES:
        raise PermissionError("trainer source role-to-path closure changed")
    digest = _sha256_bytes(_canonical_json_bytes(normalized))
    if source_map.get("source_map_sha256") != digest:
        raise ValueError("trainer source-map SHA-256 changed")
    return digest


def _validate_review_record(
    review: Mapping[str, Any],
    *,
    source_map_sha256: str,
) -> None:
    if set(review) != {
        "schema",
        "status",
        "decision",
        "reviewer",
        "reviewed_source_map_sha256",
        "restricted_payload_opened",
        "findings",
        "content_sha256",
    } or review.get("schema") != REVIEW_RECORD_SCHEMA:
        raise ValueError("V4 trainer review-record schema changed")
    findings = review.get("findings")
    if not isinstance(findings, list) or not all(
        isinstance(item, str) and item for item in findings
    ):
        raise ValueError("V4 trainer review findings are malformed")
    status = review.get("status")
    if status == "pending_second_independent_review":
        if (
            review.get("decision") != "pending"
            or review.get("reviewer") is not None
            or review.get("reviewed_source_map_sha256") is not None
            or review.get("restricted_payload_opened") is not False
        ):
            raise ValueError("pending V4 trainer review record is malformed")
        return
    if status != "independent_review_passed" or (
        review.get("decision") != "pass"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"]
        or review.get("reviewed_source_map_sha256") != source_map_sha256
        or review.get("restricted_payload_opened") is not False
    ):
        raise PermissionError("V4 trainer review record does not grant a PASS")


def preflight_exact_authorization(
    *,
    dataset_path: Path,
    dataset_file_sha256: str,
    audit_path: Path,
    audit_file_sha256: str,
    authorization_path: Path,
    authorization_file_sha256: str,
    review_record_path: Path,
    review_record_file_sha256: str,
    root: Path = ROOT,
    canonical_dataset_path: Path = CANONICAL_DATASET_PATH,
    canonical_audit_path: Path = CANONICAL_AUDIT_PATH,
    canonical_authorization_path: Path = CANONICAL_AUTHORIZATION_PATH,
    canonical_review_record_path: Path = CANONICAL_REVIEW_RECORD_PATH,
    upstream_implementation_path: Path = UPSTREAM_IMPLEMENTATION_PATH,
) -> dict[str, Any]:
    """Validate authorization/source/review before dataset or audit is opened."""

    if not sys.flags.isolated:
        raise PermissionError("V4 authorization preflight is one-shot only")

    authorization = _strict_hashed_object(
        authorization_path,
        authorization_file_sha256,
        name="V4 trainer authorization",
        canonical_path=canonical_authorization_path,
    )
    if set(authorization) != {
        "schema",
        "status",
        "dataset_binding",
        "audit_binding",
        "upstream_implementation",
        "fit_contract",
        "allowed_fit_sizes",
        "source_map",
        "authorization",
        "review_record",
        "content_sha256",
    } or authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise PermissionError("V4 trainer authorization schema changed")
    if (
        authorization.get("fit_contract") != EXPECTED_FIT_CONTRACT
        or authorization.get("allowed_fit_sizes") != list(SUPPORTED_FIT_SIZES)
    ):
        raise PermissionError("V4 trainer fit contract changed")
    source_map_sha256 = _validate_source_map(authorization["source_map"], root=root)
    expected_dataset = {
        "path": str(canonical_dataset_path),
        "file_sha256": DATASET_MANIFEST_FILE_SHA256,
        "content_sha256": DATASET_MANIFEST_CONTENT_SHA256,
        "status": "reviewed_exact_artifact",
    }
    expected_audit = {
        "path": str(canonical_audit_path),
        "file_sha256": AUDIT_RECEIPT_FILE_SHA256,
        "content_sha256": AUDIT_RECEIPT_CONTENT_SHA256,
        "status": "reviewed_exact_artifact",
    }
    if authorization.get("dataset_binding") != expected_dataset or authorization.get(
        "audit_binding"
    ) != expected_audit:
        raise PermissionError("V4 trainer authorization exact bindings changed")
    if authorization.get("upstream_implementation") != {
        "path": str(upstream_implementation_path),
        "file_sha256": UPSTREAM_IMPLEMENTATION_FILE_SHA256,
        "content_sha256": UPSTREAM_IMPLEMENTATION_CONTENT_SHA256,
        "source_map_sha256": UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256,
    }:
        raise PermissionError("V4 trainer upstream implementation binding changed")
    review_binding = authorization.get("review_record")
    if not isinstance(review_binding, Mapping) or set(review_binding) != {
        "path",
        "file_sha256",
        "content_sha256",
        "status",
    } or review_binding.get("path") != str(canonical_review_record_path):
        raise PermissionError("V4 trainer review-record binding changed")
    if (
        str(review_record_path) != str(canonical_review_record_path)
        or review_record_path.resolve(strict=True) != canonical_review_record_path
        or review_record_file_sha256 != review_binding.get("file_sha256")
    ):
        raise PermissionError("V4 trainer caller review-record binding changed")

    # The exact upstream implementation is part of the authorization closure,
    # so verify it before opening review, dataset, or audit receipts.
    upstream = _strict_hashed_object(
        upstream_implementation_path,
        UPSTREAM_IMPLEMENTATION_FILE_SHA256,
        name="V4 upstream implementation manifest",
        canonical_path=upstream_implementation_path,
        require_canonical=False,
    )
    if (
        upstream.get("content_sha256") != UPSTREAM_IMPLEMENTATION_CONTENT_SHA256
        or upstream.get("source_map", {}).get("source_map_sha256")
        != UPSTREAM_IMPLEMENTATION_SOURCE_MAP_SHA256
    ):
        raise ValueError("V4 upstream implementation receipt changed")

    review = _strict_hashed_object(
        review_record_path,
        review_record_file_sha256,
        name="V4 trainer review record",
        canonical_path=canonical_review_record_path,
    )
    if review.get("content_sha256") != review_binding.get("content_sha256"):
        raise ValueError("V4 trainer review-record content binding changed")
    _validate_review_record(review, source_map_sha256=source_map_sha256)
    flags = authorization.get("authorization")
    if (
        authorization.get("status") != "independent_review_passed_authorized"
        or review_binding.get("status") != "independent_review_passed"
        or review.get("status") != "independent_review_passed"
        or review.get("decision") != "pass"
        or review.get("reviewed_source_map_sha256") != source_map_sha256
        or review.get("restricted_payload_opened") is not False
        or not isinstance(flags, Mapping)
        or flags
        != {
            "development_fit": True,
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        }
    ):
        raise PermissionError("V4 trainer remains unauthorized pending independent PASS")

    # Only after authorization and review pass may exact receipts be opened.
    dataset = _strict_hashed_object(
        dataset_path,
        dataset_file_sha256,
        name="V4 dataset manifest",
        canonical_path=canonical_dataset_path,
    )
    audit = _strict_hashed_object(
        audit_path,
        audit_file_sha256,
        name="V4 audit receipt",
        canonical_path=canonical_audit_path,
    )
    if (
        dataset.get("content_sha256") != DATASET_MANIFEST_CONTENT_SHA256
        or audit.get("content_sha256") != AUDIT_RECEIPT_CONTENT_SHA256
    ):
        raise ValueError("one exact V4 receipt content binding changed")
    return {
        "schema": VERIFIED_CONTEXT_SCHEMA,
        "authorization_path": str(authorization_path),
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_content_sha256": authorization["content_sha256"],
        "source_map_sha256": source_map_sha256,
        "review_record_path": str(canonical_review_record_path),
        "review_record_file_sha256": review_record_file_sha256,
        "review_record_content_sha256": review_binding["content_sha256"],
        "dataset_path": str(dataset_path),
        "dataset_file_sha256": dataset_file_sha256,
        "dataset_content_sha256": dataset["content_sha256"],
        "audit_path": str(audit_path),
        "audit_file_sha256": audit_file_sha256,
        "audit_content_sha256": audit["content_sha256"],
        "source_map": authorization["source_map"],
    }


def _stable_code_constant(value: object) -> object:
    if isinstance(value, types.CodeType):
        return {"code": _stable_code_record(value)}
    if isinstance(value, tuple):
        return {"tuple": [_stable_code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_stable_code_constant(item) for item in value]
        return {"frozenset": sorted(items, key=_canonical_json_bytes)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, complex):
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if value is Ellipsis:
        return {"singleton": "ellipsis"}
    if value is None:
        return {"singleton": "none"}
    if isinstance(value, (bool, int, str)):
        return {type(value).__name__: value}
    raise TypeError(f"unsupported V4 code constant: {type(value).__name__}")


def _stable_code_record(code: types.CodeType) -> dict[str, object]:
    return {
        "argcount": code.co_argcount,
        "cellvars": list(code.co_cellvars),
        "code": code.co_code.hex(),
        "consts": [_stable_code_constant(value) for value in code.co_consts],
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
        "firstlineno": code.co_firstlineno,
        "flags": code.co_flags,
        "freevars": list(code.co_freevars),
        "kwonlyargcount": code.co_kwonlyargcount,
        "linetable": getattr(code, "co_linetable", b"").hex(),
        "name": code.co_name,
        "names": list(code.co_names),
        "nlocals": code.co_nlocals,
        "posonlyargcount": getattr(code, "co_posonlyargcount", 0),
        "qualname": getattr(code, "co_qualname", code.co_name),
        "stacksize": code.co_stacksize,
        "varnames": list(code.co_varnames),
    }


def _module_code_sha256(module: types.ModuleType) -> str:
    digest = hashlib.sha256(b"lewm_v4_loaded_module_code_v1")
    for name, value in sorted(vars(module).items()):
        codes = []
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            codes.append((name, value.__code__))
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for child_name, child in sorted(vars(value).items()):
                function = (
                    child.__func__
                    if isinstance(child, (classmethod, staticmethod))
                    else child
                )
                if inspect.isfunction(function):
                    codes.append((f"{name}.{child_name}", function.__code__))
        for code_name, code in codes:
            encoded_name = code_name.encode("utf-8")
            payload = _canonical_json_bytes(_stable_code_record(code))
            digest.update(len(encoded_name).to_bytes(8, "little"))
            digest.update(encoded_name)
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def _rgb_worker_terminal(
    payload: tuple[tuple[str, str, str, str], str, str, str],
) -> Any:
    """Decode one authorized RGB inside an isolated, locally captured runtime."""

    if not sys.flags.isolated or __name__ not in {"__main__", "__mp_main__"}:
        raise PermissionError("V4 RGB worker terminal requires an isolated spawn")
    (
        job,
        authorization_file_sha256,
        review_record_file_sha256,
        successor_review_file_sha256,
    ) = payload
    successor_review = preflight_successor_review(
        CANONICAL_SUCCESSOR_REVIEW_PATH,
        successor_review_file_sha256,
    )
    receipt = preflight_exact_authorization(
        dataset_path=CANONICAL_DATASET_PATH,
        dataset_file_sha256=DATASET_MANIFEST_FILE_SHA256,
        audit_path=CANONICAL_AUDIT_PATH,
        audit_file_sha256=AUDIT_RECEIPT_FILE_SHA256,
        authorization_path=CANONICAL_AUTHORIZATION_PATH,
        authorization_file_sha256=authorization_file_sha256,
        review_record_path=CANONICAL_REVIEW_RECORD_PATH,
        review_record_file_sha256=review_record_file_sha256,
    )

    # Runtime machinery is deliberately lexical and is created only after the
    # complete canonical authorization above has passed.
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
        SUCCESSOR_FINALIZER_RELATIVE_PATH,
        "scripts/launch_go2_observable_camera_ray_fit_v4.py",
        SUCCESSOR_LAUNCHER_RELATIVE_PATH,
        "scripts/train_go2_observable_camera_ray_fit_v4.py",
        SUCCESSOR_TRAINER_RELATIVE_PATH,
        "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py",
        SUCCESSOR_VERIFIER_RELATIVE_PATH,
        "scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py",
    )

    def logical_name(relative: str) -> str:
        value = relative[:-3].replace("/", ".")
        return value[:-9] if value.endswith(".__init__") else value

    source_map = receipt["source_map"]
    entries = {str(row["path"]): str(row["sha256"]) for row in source_map["entries"]}
    entries.update(
        {
            relative: str(
                successor_review["successor_sources"][relative]["file_sha256"]
            )
            for relative in SUCCESSOR_SOURCE_PATHS
        }
    )
    captured: dict[str, tuple[Path, bytes, str]] = {}
    for relative in runtime_relatives:
        path = (ROOT / relative).resolve(strict=True)
        source = _regular_bytes(path, name=f"V4 RGB runtime source {relative}")
        digest = _sha256_bytes(source)
        if entries.get(relative) != digest:
            raise ValueError(f"V4 RGB runtime source changed: {relative}")
        captured[logical_name(relative)] = (path, source, digest)
    preloaded = sorted(name for name in captured if name in sys.modules)
    if preloaded:
        raise PermissionError(f"V4 RGB runtime modules were preloaded: {preloaded}")
    allowed_roots = frozenset({*sys.stdlib_module_names, "PIL", "numpy", "torch"})
    namespace = f"_lewm_v4_ca_{receipt['source_map_sha256'][:12]}_{uuid.uuid4().hex}"

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
                    raise ImportError(f"V4 RGB runtime import is not whitelisted: {name}")
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
        if logical not in captured:
            raise PermissionError(f"V4 RGB runtime source is not bound: {logical}")
        module = importlib.import_module(finder.synthetic(logical))
        path, source, digest = captured[logical]
        if (
            module is not sys.modules.get(finder.synthetic(logical))
            or getattr(module, "__verified_logical_name__", None) != logical
            or getattr(module, "__verified_source_sha256__", None) != digest
            or Path(str(getattr(module, "__file__", ""))).resolve(strict=True) != path
            or _sha256_bytes(source) != digest
        ):
            raise PermissionError(f"V4 RGB loaded module identity changed: {logical}")
        fingerprint = _module_code_sha256(module)
        if fingerprints.setdefault(logical, fingerprint) != fingerprint:
            raise PermissionError(f"V4 RGB loaded module code changed: {logical}")
        return module

    try:
        captured_launcher = load(
            "scripts.launch_go2_observable_camera_ray_fit_v4_v2"
        )
        live_launcher = sys.modules.get(__name__)
        if not isinstance(live_launcher, types.ModuleType) or (
            _module_code_sha256(live_launcher) != _module_code_sha256(captured_launcher)
        ):
            raise PermissionError("V4 live RGB terminal differs from captured source")
        trainer = load("scripts.train_go2_observable_camera_ray_fit_v4_v2")
        result = trainer._decode_rgb_job(*job)
        for logical in tuple(fingerprints):
            load(logical)
        return result
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(f"{namespace}."):
                sys.modules.pop(name, None)


def _execute_captured_trainer(
    args: argparse.Namespace,
) -> int:
    if not sys.flags.isolated:
        raise PermissionError("V4 captured trainer execution is one-shot only")
    successor_review = preflight_successor_review(
        Path(args.successor_review),
        str(args.successor_review_sha256),
    )
    receipt = preflight_exact_authorization(
        dataset_path=Path(args.dataset_manifest),
        dataset_file_sha256=str(args.dataset_manifest_sha256),
        audit_path=Path(args.audit_receipt),
        audit_file_sha256=str(args.audit_receipt_sha256),
        authorization_path=Path(args.trainer_authorization),
        authorization_file_sha256=str(args.trainer_authorization_sha256),
        review_record_path=Path(args.trainer_review_record),
        review_record_file_sha256=str(args.trainer_review_record_sha256),
    )
    authorization_file_sha256 = str(args.trainer_authorization_sha256)
    if receipt.get("authorization_file_sha256") != authorization_file_sha256:
        raise PermissionError("V4 canonical authorization receipt changed")
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
        SUCCESSOR_FINALIZER_RELATIVE_PATH,
        "scripts/launch_go2_observable_camera_ray_fit_v4.py",
        SUCCESSOR_LAUNCHER_RELATIVE_PATH,
        "scripts/train_go2_observable_camera_ray_fit_v4.py",
        SUCCESSOR_TRAINER_RELATIVE_PATH,
        "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py",
        SUCCESSOR_VERIFIER_RELATIVE_PATH,
        "scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py",
    )

    def logical_name(relative: str) -> str:
        value = relative[:-3].replace("/", ".")
        return value[:-9] if value.endswith(".__init__") else value

    source_map = receipt["source_map"]
    entries = {str(row["path"]): str(row["sha256"]) for row in source_map["entries"]}
    entries.update(
        {
            relative: str(
                successor_review["successor_sources"][relative]["file_sha256"]
            )
            for relative in SUCCESSOR_SOURCE_PATHS
        }
    )
    captured: dict[str, tuple[Path, bytes, str]] = {}
    for relative in runtime_relatives:
        path = (ROOT / relative).resolve(strict=True)
        source = _regular_bytes(path, name=f"V4 trainer runtime source {relative}")
        digest = _sha256_bytes(source)
        if entries.get(relative) != digest:
            raise ValueError(f"V4 trainer runtime source changed: {relative}")
        captured[logical_name(relative)] = (path, source, digest)
    preloaded = sorted(name for name in captured if name in sys.modules)
    if preloaded:
        raise PermissionError(f"V4 trainer runtime modules were preloaded: {preloaded}")
    allowed_roots = frozenset({*sys.stdlib_module_names, "PIL", "numpy", "torch"})
    namespace = f"_lewm_v4_ca_{receipt['source_map_sha256'][:12]}_{uuid.uuid4().hex}"

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
                    raise ImportError(f"V4 trainer runtime import is not whitelisted: {name}")
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
        if logical not in captured:
            raise PermissionError(f"V4 trainer runtime source is not bound: {logical}")
        module = importlib.import_module(finder.synthetic(logical))
        path, source, digest = captured[logical]
        if (
            module is not sys.modules.get(finder.synthetic(logical))
            or getattr(module, "__verified_logical_name__", None) != logical
            or getattr(module, "__verified_source_sha256__", None) != digest
            or Path(str(getattr(module, "__file__", ""))).resolve(strict=True) != path
            or _sha256_bytes(source) != digest
        ):
            raise PermissionError(f"V4 trainer loaded module identity changed: {logical}")
        fingerprint = _module_code_sha256(module)
        if fingerprints.setdefault(logical, fingerprint) != fingerprint:
            raise PermissionError(f"V4 trainer loaded module code changed: {logical}")
        return module

    try:
        captured_launcher = load(
            "scripts.launch_go2_observable_camera_ray_fit_v4_v2"
        )
        live_launcher = sys.modules.get(__name__)
        if not isinstance(live_launcher, types.ModuleType) or (
            _module_code_sha256(live_launcher) != _module_code_sha256(captured_launcher)
        ):
            raise PermissionError("V4 live launcher differs from captured source")
        trainer = load("scripts.train_go2_observable_camera_ray_fit_v4_v2")
        captured_preauthorization = {
            **dict(receipt),
            "successor_review": successor_review,
            "successor_review_file_sha256": str(args.successor_review_sha256),
        }
        result = int(trainer._captured_exact_cli(args, captured_preauthorization))
        for logical in tuple(fingerprints):
            load(logical)
        return result
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in tuple(sys.modules):
            if name == namespace or name.startswith(f"{namespace}."):
                sys.modules.pop(name, None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
    parser.add_argument("--trainer-authorization", type=Path, required=True)
    parser.add_argument("--trainer-authorization-sha256", required=True)
    parser.add_argument("--trainer-review-record", type=Path, required=True)
    parser.add_argument("--trainer-review-record-sha256", required=True)
    parser.add_argument("--successor-review", type=Path, required=True)
    parser.add_argument("--successor-review-sha256", required=True)
    parser.add_argument("--fit-size", type=int, choices=SUPPORTED_FIT_SIZES, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument("--rgb-workers", type=int, default=MAX_RGB_WORKERS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--previous-stage-gate", type=Path)
    parser.add_argument("--previous-stage-gate-sha256")
    parser.add_argument("--seed-20260710-gate", type=Path)
    parser.add_argument("--seed-20260710-gate-sha256")
    return validate_execution_args(parser.parse_args(argv))


def validate_execution_args(args: argparse.Namespace) -> argparse.Namespace:
    if (
        str(args.successor_review) != str(CANONICAL_SUCCESSOR_REVIEW_PATH)
        or not _is_sha256(args.successor_review_sha256)
        or args.steps != DEFAULT_STEPS[args.fit_size]
        or args.batch_size != 1
        or args.eval_batch_size != 1
        or args.learning_rate != 1e-4
        or args.weight_decay != 1e-4
        or args.rgb_workers != MAX_RGB_WORKERS
        or args.device != "cuda:0"
    ):
        raise PermissionError("V4 exact execution configuration is not frozen")
    previous_pair = (
        args.previous_stage_gate,
        args.previous_stage_gate_sha256,
    )
    seed_pair = (args.seed_20260710_gate, args.seed_20260710_gate_sha256)
    if any(value is None for value in previous_pair) != all(
        value is None for value in previous_pair
    ):
        raise ValueError("previous-stage gate path and SHA-256 must be supplied together")
    if any(value is None for value in seed_pair) != all(
        value is None for value in seed_pair
    ):
        raise ValueError("seed-20260710 gate path and SHA-256 must be supplied together")
    if args.fit_size == SUPPORTED_FIT_SIZES[0] and any(
        value is not None for value in previous_pair
    ):
        raise PermissionError("N5 may not bind a previous-stage gate")
    if args.fit_size != SUPPORTED_FIT_SIZES[0] and any(
        value is None for value in previous_pair
    ):
        raise PermissionError("larger V4 rungs require a caller-hashed predecessor gate")
    if args.seed == EXPECTED_SEEDS[0] and any(value is not None for value in seed_pair):
        raise PermissionError("the first V4 seed may not bind itself")
    if args.seed == EXPECTED_SEEDS[1] and any(value is None for value in seed_pair):
        raise PermissionError("the second V4 seed requires the completed first-seed gate")
    return args


def execution_binding(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "fit_size": int(args.fit_size),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "evaluation_batch_size": int(args.eval_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "rgb_workers": int(args.rgb_workers),
        "device": str(args.device),
        "successor_review": {
            "path": str(args.successor_review),
            "file_sha256": str(args.successor_review_sha256),
        },
        "previous_stage_gate": (
            None
            if args.previous_stage_gate is None
            else {
                "path": str(args.previous_stage_gate),
                "file_sha256": str(args.previous_stage_gate_sha256),
            }
        ),
        "seed_20260710_gate": (
            None
            if args.seed_20260710_gate is None
            else {
                "path": str(args.seed_20260710_gate),
                "file_sha256": str(args.seed_20260710_gate_sha256),
            }
        ),
    }


def _run_isolated_child(argv: Sequence[str]) -> int:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
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
        return _run_isolated_child(raw_argv)
    args = parse_args(raw_argv)
    return _execute_captured_trainer(args)


if __name__ == "__main__":
    raise SystemExit(main())
