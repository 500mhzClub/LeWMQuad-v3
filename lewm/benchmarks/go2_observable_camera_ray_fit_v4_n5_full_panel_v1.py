"""Fail-closed authority and artifact contract for the V4 N5 full-panel attempt.

This module is intentionally stdlib-only. Importing it does not open train data,
RGB, a checkpoint, a model module, an accelerator runtime, or an output path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/v4_execution_successor_review"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_"
    "successor_preregistration_2026-07-13.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "0ad13e3897c70f90df6705538f4d86262ec53d3e096618a69563acdf63567c01"
)
TRIGGER_AMENDMENT_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_"
    "structural_trigger_amendment_2026-07-13.md"
)
TRIGGER_AMENDMENT_FILE_SHA256 = (
    "1e08aac0ace734d2cbcce9e965b10a7031a94764dd7b47114d38e33944990262"
)
TERMINAL_INVALIDATION_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "prepublication_structural_invalidation_2026-07-13.json"
)
TERMINAL_INVALIDATION_FILE_SHA256 = (
    "1744a50badd6c9f5c1ef4c8c3cbd05f8c0fc8acff4fbbf066e40e1f7de24f560"
)
TERMINAL_INVALIDATION_CONTENT_SHA256 = (
    "7bdaae6ebb13b7d90290dfe07f5d48f403d29cad977f4a56c9ac7b8cfbcb8602"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "independent_review_2026-07-13.json"
)
CANONICAL_SOURCE_REVIEW_PATH = (ROOT / SOURCE_REVIEW_RELATIVE_PATH).resolve()

POLICY_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py"
)
TRAINER_RELATIVE_PATH = (
    "scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py"
)
VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py"
)
FINALIZER_RELATIVE_PATH = (
    "scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py"
)
SUCCESSOR_SOURCE_PATHS = (
    POLICY_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    TRAINER_RELATIVE_PATH,
    VERIFIER_RELATIVE_PATH,
    FINALIZER_RELATIVE_PATH,
)

FROZEN_SOURCE_BINDINGS = {
    "scripts/launch_go2_observable_camera_ray_fit_v4_v2.py": (
        "65c58e36cb97d155a58ec1cbc93a1f2f42a75e62f049b5d8e874481a435a614b"
    ),
    "scripts/train_go2_observable_camera_ray_fit_v4_v2.py": (
        "c9d22fb38acdf5fd3099271661dc65bb9cea989426a3b6021ad28649d6dd74d3"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py": (
        "aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py": (
        "6a0e40f9dcb496831553dc5bbc6d1efcdf6d82676d6f18aa20e417f8de4fa6a0"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py": (
        "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85"
    ),
    "lewm/models/observable_camera_ray_evidence_v4.py": (
        "6238f7fb2b9c0c5201c9d7ebb5343ceef72fa97b423dddb466465b6c594cc882"
    ),
    "lewm/models/observable_camera_ray_evidence_v4_training.py": (
        "c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json": (
        "4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a"
    ),
    "scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py": (
        "4624dd761901808c72b37eb256b360e3db61c9b8f61337879547ed38836a3eed"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v2_partition_amendment_2026-07-12.md": (
        "1e65f8884b1b8e0ad2219ddad54f79f9fabae514bfcaa048b29c8113b076ac1f"
    ),
}

DATASET_MANIFEST_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/v1/manifest.json"
)
DATASET_MANIFEST_FILE_SHA256 = (
    "2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85"
)
DATASET_MANIFEST_CONTENT_SHA256 = (
    "9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812"
)
AUDIT_RECEIPT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json"
)
AUDIT_RECEIPT_FILE_SHA256 = (
    "2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c"
)
AUDIT_RECEIPT_CONTENT_SHA256 = (
    "a922114b7e42552043a487bae527c35fb511804d4e8683c5a3f64a2bf499cf76"
)
TRAINER_AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "trainer_authorization_bound_2026-07-12.json"
)
TRAINER_AUTHORIZATION_FILE_SHA256 = (
    "d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802"
)
TRAINER_AUTHORIZATION_CONTENT_SHA256 = (
    "18a285e80252d41de7daadba918a00223d8770b71c533f74807e0ace5444ac1e"
)
TRAINER_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_"
    "trainer_review_record_2026-07-12.json"
)
TRAINER_REVIEW_FILE_SHA256 = (
    "c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea"
)
TRAINER_REVIEW_CONTENT_SHA256 = (
    "ab55270986268c5a326eeb6ba191cd9a0531112b1b742812d2cbd549f67158be"
)
RGB_RECEIPT_CONTENT_SHA256 = (
    "d763d7ae294e4e5a9e5f2352672913bc06411388d92abe1fb0f5090dfc41d5c3"
)
SUBSET_CONTENT_SHA256 = (
    "3595dff9d24dbb44f3e73086fce3be4ec53eb8659684738defa8591c4a375f15"
)
TARGET_PARTITION_CONTENT_SHA256 = (
    "ac9d6e1c91ca58c1182fa5e05d3189a6dc319013c3dc07e2f229f88c55cca429"
)

OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/n5_full_panel_v1"
)
CANONICAL_OUTPUT_ROOT = (ROOT / OUTPUT_ROOT_RELATIVE_PATH).resolve()
CANONICAL_ATTEMPT_PATH = (
    CANONICAL_OUTPUT_ROOT / "attempts/seed_20260710/n5"
).resolve()
CANONICAL_METRIC_RECEIPT_PATH = (
    CANONICAL_OUTPUT_ROOT / "metric_verifications/seed_20260710_n5.json"
).resolve()
CANONICAL_GATE_PATH = (
    CANONICAL_OUTPUT_ROOT / "gates/seed_20260710_n5.json"
).resolve()

SCHEDULE_ALGORITHM = (
    "torch_cpu_generator_manual_seed_then_concatenated_randperm_cycles_"
    "take_steps_times_batch_v1"
)
EXPECTED_SCHEDULE_SHA256 = (
    "62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634"
)
LOSS_COMPONENTS = (
    "ordered_first_hit_nll",
    "target_bin_offset_smooth_l1",
    "ground_clear_distance_state_balanced_bce",
    "derived_raster_hierarchical_bce",
)
LOSS_WEIGHTS = {name: 0.25 for name in LOSS_COMPONENTS}
LOSS_ABSOLUTE_TOLERANCE = 1e-9
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
EXPERIMENT = {
    "seed": 20260710,
    "fit_size": 5,
    "fresh_model_initialization": True,
    "model_class": "ObservableCameraRayEvidenceV4Model",
    "optimizer": "AdamW",
    "optimizer_updates": 400,
    "training_batch_size": 5,
    "frame_exposures": 2000,
    "evaluation_batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "precision": "float32",
    "autocast": False,
    "gradient_clip_norm": 1.0,
    "loss_weights": LOSS_WEIGHTS,
    "schedule_algorithm": SCHEDULE_ALGORITHM,
    "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    "checkpoint_selection": "final_update_only",
    "evaluation_controls": [
        "matched_rgb",
        "wrong_rgb_with_target_calibration",
    ],
    "device": "cuda:0",
    "device_name": "AMD Radeon AI PRO R9700",
    "raphael_igpu_forbidden": True,
    "rgb_worker_count_max": 5,
    "native_threads_per_process": 1,
    "attempt_count": 1,
    "output_path": str(CANONICAL_ATTEMPT_PATH),
}
AUTHORITY_BINDINGS = {
    "preregistration": {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
    },
    "structural_trigger_amendment": {
        "path": TRIGGER_AMENDMENT_RELATIVE_PATH,
        "file_sha256": TRIGGER_AMENDMENT_FILE_SHA256,
    },
    "terminal_invalidation": {
        "path": TERMINAL_INVALIDATION_RELATIVE_PATH,
        "file_sha256": TERMINAL_INVALIDATION_FILE_SHA256,
        "content_sha256": TERMINAL_INVALIDATION_CONTENT_SHA256,
    },
}
LICENSES = {
    "authorizes_one_fresh_n5_full_panel_attempt": True,
    "authorizes_metric_verification_only_checkpoint_use": True,
    "authorizes_stage_finalization": True,
    "authorizes_retry": False,
    "authorizes_n16_execution": False,
    "authorizes_second_seed": False,
    "authorizes_v5_training": False,
    "authorizes_g2": False,
    "authorizes_holdout": False,
    "authorizes_selection": False,
    "authorizes_calibration_change": False,
    "authorizes_runtime": False,
    "authorizes_hardware": False,
    "authorizes_production": False,
    "authorizes_promotion": False,
}

SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_source_review_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_reservation_v1"
)
RESULT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_result_v1"
COMPLETION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_completion_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_failure_v1"
)
METRIC_RECEIPT_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_metric_verification_v1"
)
GATE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_gate_v1"

_AUTHORITY_MARKER = object()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, *, name: str) -> dict[str, Any]:
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def read_regular_bytes(path: Path, *, name: str) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"{name} is not a regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"{name} is not a regular file")
        chunks: list[bytes] = []
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


def read_hashed_bytes(path: Path, expected_sha256: str, *, name: str) -> bytes:
    if not is_sha256(expected_sha256):
        raise ValueError(f"{name} caller SHA-256 is malformed")
    raw = read_regular_bytes(path, name=name)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{name} file SHA-256 changed")
    return raw


def load_hashed_json(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
    require_canonical: bool = True,
) -> tuple[dict[str, Any], bytes]:
    raw = read_hashed_bytes(path, expected_sha256, name=name)
    value = parse_json(raw, name=name)
    if require_canonical and raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not canonical JSON plus newline")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content SHA-256 changed")
    return value, raw


def _verify_file(relative: str, expected_sha256: str, *, name: str) -> None:
    read_hashed_bytes(ROOT / relative, expected_sha256, name=name)


def preflight_static_authority() -> dict[str, Any]:
    """Bind trigger records and frozen sources before any protected operation."""

    _verify_file(
        PREREGISTRATION_RELATIVE_PATH,
        PREREGISTRATION_FILE_SHA256,
        name="N5 full-panel preregistration",
    )
    _verify_file(
        TRIGGER_AMENDMENT_RELATIVE_PATH,
        TRIGGER_AMENDMENT_FILE_SHA256,
        name="N5 full-panel structural-trigger amendment",
    )
    invalidation, _raw = load_hashed_json(
        ROOT / TERMINAL_INVALIDATION_RELATIVE_PATH,
        TERMINAL_INVALIDATION_FILE_SHA256,
        name="N5 terminal structural invalidation",
        require_canonical=False,
    )
    authority = invalidation.get("authority")
    primary = invalidation.get("primary_structural_invalidation")
    if (
        invalidation.get("content_sha256") != TERMINAL_INVALIDATION_CONTENT_SHA256
        or invalidation.get("status")
        != "terminal_prepublication_structural_invalidation"
        or invalidation.get("scope", {}).get("decision")
        != "immutable_n5_is_structurally_invalid_for_canonical_finalization"
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
        or primary.get("full_immutable_result_validation")
        != {
            "passed": False,
            "exception": "ValueError: V4 matched evaluation losses are inconsistent",
        }
    ):
        raise PermissionError("N5 structural-trigger authority changed")
    for relative, digest in FROZEN_SOURCE_BINDINGS.items():
        _verify_file(relative, digest, name=f"frozen full-panel dependency {relative}")
    return {
        "authority_bindings": AUTHORITY_BINDINGS,
        "terminal_invalidation_status": invalidation["status"],
        "frozen_source_bindings": FROZEN_SOURCE_BINDINGS,
    }


def expected_source_review_core(
    *,
    reviewer: str,
    successor_sources: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema": SOURCE_REVIEW_SCHEMA,
        "status": "different_agent_review_passed_exact_full_panel_v1",
        "implementation_author": IMPLEMENTATION_AUTHOR,
        "reviewer": reviewer,
        "review_completed": True,
        "source_closure_approved": True,
        "exact_attempt_authorized": True,
        "successor_sources": dict(successor_sources),
        "frozen_source_bindings": FROZEN_SOURCE_BINDINGS,
        "authority_bindings": AUTHORITY_BINDINGS,
        "experiment": EXPERIMENT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "licenses": LICENSES,
    }


def preflight_source_review(
    path: Path,
    file_sha256: str,
    *,
    canonical_path: Path = CANONICAL_SOURCE_REVIEW_PATH,
) -> tuple[dict[str, Any], bytes]:
    path = Path(path).resolve(strict=True)
    if path != Path(canonical_path).resolve(strict=True):
        raise PermissionError("N5 full-panel source review path is not canonical")
    review, raw = load_hashed_json(
        path,
        file_sha256,
        name="N5 full-panel different-agent source review",
    )
    reviewer = review.get("reviewer")
    sources = review.get("successor_sources")
    if (
        not isinstance(reviewer, str)
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or not isinstance(sources, Mapping)
        or set(sources) != set(SUCCESSOR_SOURCE_PATHS)
    ):
        raise PermissionError("N5 full-panel review is not by a different agent")
    expected_sources: dict[str, dict[str, str]] = {}
    for relative in SUCCESSOR_SOURCE_PATHS:
        binding = sources.get(relative)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != relative
            or not is_sha256(binding.get("file_sha256"))
        ):
            raise PermissionError("N5 full-panel successor source binding changed")
        raw_source = read_regular_bytes(
            ROOT / relative,
            name=f"N5 full-panel successor source {relative}",
        )
        if hashlib.sha256(raw_source).hexdigest() != binding["file_sha256"]:
            raise PermissionError(f"N5 full-panel successor source changed: {relative}")
        expected_sources[relative] = dict(binding)
    expected_core = expected_source_review_core(
        reviewer=reviewer,
        successor_sources=expected_sources,
    )
    core = dict(review)
    declared = core.pop("content_sha256")
    if core != expected_core or canonical_json_sha256(core) != declared:
        raise PermissionError("N5 full-panel source review contract changed")
    return review, raw


@dataclass(frozen=True)
class VerifiedAuthority:
    static: Mapping[str, Any]
    source_review: Mapping[str, Any]
    source_review_file_sha256: str
    source_review_content_sha256: str
    _marker: object


def verify_authority(
    source_review_path: Path,
    source_review_file_sha256: str,
    *,
    canonical_review_path: Path = CANONICAL_SOURCE_REVIEW_PATH,
    require_unclaimed_output: bool = True,
) -> VerifiedAuthority:
    static = preflight_static_authority()
    review, _raw = preflight_source_review(
        source_review_path,
        source_review_file_sha256,
        canonical_path=canonical_review_path,
    )
    if require_unclaimed_output and CANONICAL_ATTEMPT_PATH.exists():
        raise FileExistsError("the sole N5 full-panel attempt is already claimed")
    return VerifiedAuthority(
        static=static,
        source_review=review,
        source_review_file_sha256=source_review_file_sha256,
        source_review_content_sha256=str(review["content_sha256"]),
        _marker=_AUTHORITY_MARKER,
    )


def require_verified_authority(value: object) -> VerifiedAuthority:
    if not isinstance(value, VerifiedAuthority) or value._marker is not _AUTHORITY_MARKER:
        raise PermissionError("N5 full-panel protected work lacks verified authority")
    return value


def source_review_binding(authority: VerifiedAuthority) -> dict[str, str]:
    require_verified_authority(authority)
    return {
        "path": SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": authority.source_review_file_sha256,
        "content_sha256": authority.source_review_content_sha256,
    }


def artifact_binding(
    relative_path: str,
    raw: bytes,
    *,
    content_sha256: str | None = None,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": relative_path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    if content_sha256 is not None:
        binding["content_sha256"] = content_sha256
    return binding


def _validate_content_hash(value: Mapping[str, Any], *, name: str) -> None:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content SHA-256 changed")


def _validate_loss_record(value: object, *, name: str) -> dict[str, float]:
    expected = set(LOSS_COMPONENTS) | {"total"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} loss fields changed")
    normalized: dict[str, float] = {}
    for key in (*LOSS_COMPONENTS, "total"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} loss {key} is not numeric")
        normalized[key] = float(item)
        if not math.isfinite(normalized[key]) or normalized[key] < 0.0:
            raise ValueError(f"{name} loss {key} is invalid")
    expected_total = 0.25 * sum(normalized[key] for key in LOSS_COMPONENTS)
    if not math.isclose(
        normalized["total"],
        expected_total,
        rel_tol=0.0,
        abs_tol=LOSS_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"{name} evaluation losses are inconsistent")
    return normalized


def validate_evaluation_structure(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "matched_rgb",
        "wrong_rgb_with_target_calibration",
    }:
        raise ValueError("N5 full-panel evaluation controls changed")
    expected = {
        "matched_rgb": ("matched_rgb", [0, 1, 2, 3, 4], False),
        "wrong_rgb_with_target_calibration": (
            "wrong_rgb_with_target_calibration",
            [1, 2, 3, 4, 0],
            False,
        ),
    }
    normalized: dict[str, Any] = {}
    for key, (control, mapping, degenerate) in expected.items():
        row = value[key]
        if not isinstance(row, Mapping) or set(row) != {
            "control",
            "wrong_rgb_degenerate_singleton",
            "image_index_mapping",
            "image_mapping_sha256",
            "losses",
            "metrics",
        }:
            raise ValueError(f"N5 full-panel {key} evaluation fields changed")
        if (
            row.get("control") != control
            or row.get("image_index_mapping") != mapping
            or row.get("image_mapping_sha256") != canonical_json_sha256(mapping)
            or row.get("wrong_rgb_degenerate_singleton") is not degenerate
            or not isinstance(row.get("metrics"), Mapping)
        ):
            raise ValueError(f"N5 full-panel {key} control changed")
        _validate_loss_record(row["losses"], name=f"N5 full-panel {key}")
        normalized[key] = dict(row)
    return normalized


def _expected_exact_inputs(source_review: Mapping[str, str]) -> dict[str, str]:
    return {
        "dataset_manifest_file_sha256": DATASET_MANIFEST_FILE_SHA256,
        "dataset_manifest_content_sha256": DATASET_MANIFEST_CONTENT_SHA256,
        "audit_receipt_file_sha256": AUDIT_RECEIPT_FILE_SHA256,
        "audit_receipt_content_sha256": AUDIT_RECEIPT_CONTENT_SHA256,
        "trainer_authorization_file_sha256": TRAINER_AUTHORIZATION_FILE_SHA256,
        "trainer_authorization_content_sha256": TRAINER_AUTHORIZATION_CONTENT_SHA256,
        "trainer_review_file_sha256": TRAINER_REVIEW_FILE_SHA256,
        "trainer_review_content_sha256": TRAINER_REVIEW_CONTENT_SHA256,
        "rgb_receipt_content_sha256": RGB_RECEIPT_CONTENT_SHA256,
        "subset_content_sha256": SUBSET_CONTENT_SHA256,
        "target_partition_content_sha256": TARGET_PARTITION_CONTENT_SHA256,
        "source_review_file_sha256": source_review["file_sha256"],
        "source_review_content_sha256": source_review["content_sha256"],
        "terminal_invalidation_file_sha256": TERMINAL_INVALIDATION_FILE_SHA256,
        "terminal_invalidation_content_sha256": TERMINAL_INVALIDATION_CONTENT_SHA256,
    }


def validate_reservation_structure(
    reservation: Mapping[str, Any],
    *,
    expected_source_review: Mapping[str, str],
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "status",
        "attempt_index",
        "maximum_attempts",
        "scope",
        "seed",
        "fit_size",
        "experiment",
        "authority_bindings",
        "source_review",
        "inputs",
        "licenses",
        "content_sha256",
    }
    expected_licenses = {
        "development_checkpoint_creation_authorized": True,
        "checkpoint_use_authorized": False,
        "retry_authorized": False,
        "n16_execution_authorized": False,
        "second_seed_authorized": False,
        "holdout_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }
    if set(reservation) != expected_fields or reservation.get("schema") != RESERVATION_SCHEMA:
        raise ValueError("N5 full-panel reservation schema changed")
    _validate_content_hash(reservation, name="N5 full-panel reservation")
    if (
        reservation.get("status") != "reserved"
        or reservation.get("attempt_index") != 1
        or reservation.get("maximum_attempts") != 1
        or reservation.get("scope") != "one_exclusive_fresh_full_panel_attempt"
        or reservation.get("seed") != 20260710
        or reservation.get("fit_size") != 5
        or reservation.get("experiment") != EXPERIMENT
        or reservation.get("authority_bindings") != AUTHORITY_BINDINGS
        or reservation.get("source_review") != dict(expected_source_review)
        or reservation.get("inputs")
        != _expected_exact_inputs(expected_source_review)
        or reservation.get("licenses") != expected_licenses
    ):
        raise PermissionError("N5 full-panel reservation scope/licenses changed")
    return dict(reservation)


def validate_result_structure(
    result: Mapping[str, Any],
    *,
    expected_source_review: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "mode",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "dataset_role",
        "seed",
        "fit_size",
        "experiment",
        "authority_bindings",
        "source_review",
        "attempt",
        "subset",
        "target_partition",
        "inputs",
        "model",
        "training",
        "evaluation",
        "resource",
        "determinism",
        "access_ledger",
        "licenses",
        "content_sha256",
    }
    if set(result) != expected_fields or result.get("schema") != RESULT_SCHEMA:
        raise ValueError("N5 full-panel result schema changed")
    _validate_content_hash(result, name="N5 full-panel result")
    if (
        result.get("mode") != "exact_train_only_n5_full_panel_development_fit"
        or result.get("authoritative") is not False
        or result.get("aggregation_eligible") is not False
        or result.get("promotion_eligible") is not False
        or result.get("dataset_role") != "train"
        or result.get("seed") != 20260710
        or result.get("fit_size") != 5
        or result.get("experiment") != EXPERIMENT
        or result.get("authority_bindings") != AUTHORITY_BINDINGS
    ):
        raise PermissionError("N5 full-panel result crossed its frozen scope")
    review = result.get("source_review")
    if not isinstance(review, Mapping) or set(review) != {
        "path",
        "file_sha256",
        "content_sha256",
    } or review.get("path") != SOURCE_REVIEW_RELATIVE_PATH or not all(
        is_sha256(review.get(key)) for key in ("file_sha256", "content_sha256")
    ):
        raise PermissionError("N5 full-panel result source-review binding changed")
    if expected_source_review is not None and dict(review) != dict(expected_source_review):
        raise PermissionError("N5 full-panel result binds another source review")
    attempt = result.get("attempt")
    if not isinstance(attempt, Mapping) or set(attempt) != {
        "attempt_index",
        "maximum_attempts",
        "scope",
        "reservation",
    } or (
        attempt.get("attempt_index") != 1
        or attempt.get("maximum_attempts") != 1
        or attempt.get("scope") != "one_exclusive_fresh_full_panel_attempt"
        or not isinstance(attempt.get("reservation"), Mapping)
    ):
        raise PermissionError("N5 full-panel one-attempt contract changed")
    subset = result.get("subset")
    target = result.get("target_partition")
    if (
        not isinstance(subset, Mapping)
        or subset.get("fit_size") != 5
        or subset.get("content_sha256") != SUBSET_CONTENT_SHA256
        or not isinstance(target, Mapping)
        or target.get("fit_size") != 5
        or target.get("content_sha256") != TARGET_PARTITION_CONTENT_SHA256
    ):
        raise PermissionError("N5 full-panel frozen subset/target changed")
    inputs = result.get("inputs")
    expected_inputs = _expected_exact_inputs(review)
    if inputs != expected_inputs:
        raise PermissionError("N5 full-panel exact input bindings changed")
    training = result.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("N5 full-panel training record is absent")
    required_training = {
        "steps": 400,
        "batch_size": 5,
        "evaluation_batch_size": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "precision": "float32",
        "autocast": False,
        "gradient_clip_norm": 1.0,
        "loss_weights": LOSS_WEIGHTS,
        "schedule_algorithm": SCHEDULE_ALGORITHM,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "checkpoint_selection": "final_update_only",
        "frame_exposures": 2000,
        "fresh_model_initialization": True,
    }
    if any(training.get(key) != expected for key, expected in required_training.items()):
        raise PermissionError("N5 full-panel optimizer/exposure contract changed")
    if not isinstance(training.get("final"), Mapping) or training["final"].get("step") != 400:
        raise PermissionError("N5 full-panel result did not select the final update")
    validate_evaluation_structure(result.get("evaluation"))
    model = result.get("model")
    checkpoint = model.get("checkpoint") if isinstance(model, Mapping) else None
    if (
        not isinstance(model, Mapping)
        or model.get("class") != "ObservableCameraRayEvidenceV4Model"
        or model.get("fresh_initialization") is not True
        or not isinstance(model.get("parameter_count"), int)
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != "checkpoint.pt"
        or checkpoint.get("development_only") is not True
        or not is_sha256(checkpoint.get("file_sha256"))
        or not is_sha256(checkpoint.get("content_sha256"))
        or not isinstance(checkpoint.get("byte_count"), int)
    ):
        raise PermissionError("N5 full-panel checkpoint/model binding changed")
    resource = result.get("resource")
    if (
        not isinstance(resource, Mapping)
        or resource.get("device") != "cuda:0"
        or resource.get("visible_device_count") != 1
        or resource.get("hip_visible_devices") != "0"
        or resource.get("raphael_rejected") is not True
        or resource.get("hsa_override_gfx_version_unset") is not True
        or "r9700" not in "".join(
            character
            for character in str(resource.get("device_name", "")).casefold()
            if character.isalnum()
        )
    ):
        raise PermissionError("N5 full-panel GPU0 R9700 binding changed")
    ledger = result.get("access_ledger")
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("selected_rgb_count") != 5
        or ledger.get("rgb_decodes") != 5
        or not 1 <= int(ledger.get("worker_count", 0)) <= 5
        or ledger.get("native_threads_per_worker") != 1
        or any(
            ledger.get(key) != 0
            for key in (
                "nonselected_rgb_opens",
                "heldout_opens",
                "g2_opens",
                "selection_opens",
                "calibration_opens",
                "runtime_opens",
                "hardware_opens",
                "production_opens",
                "gpu1_uses",
            )
        )
    ):
        raise PermissionError("N5 full-panel access ledger crossed scope")
    expected_licenses = {
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
    }
    if result.get("licenses") != expected_licenses:
        raise PermissionError("N5 full-panel result licenses changed")
    return dict(result)


def parse_bound_path(value: str) -> tuple[Path, str]:
    path_text, separator, digest = value.rpartition(":")
    if not separator or not path_text or not is_sha256(digest):
        raise ValueError("artifact binding must be PATH:SHA256")
    return Path(path_text).resolve(strict=True), digest


def write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    if path not in {CANONICAL_METRIC_RECEIPT_PATH, CANONICAL_GATE_PATH}:
        raise PermissionError("N5 full-panel output path is not canonical")
    if CANONICAL_OUTPUT_ROOT.is_symlink() or not CANONICAL_OUTPUT_ROOT.is_dir():
        raise PermissionError("N5 full-panel output root is not a real directory")
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PermissionError("N5 full-panel output parent is not a real directory")
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
    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return artifact_binding(
        str(path.relative_to(CANONICAL_OUTPUT_ROOT)),
        payload,
        content_sha256=str(value["content_sha256"]),
    )


__all__ = [
    "AUDIT_RECEIPT_CONTENT_SHA256",
    "AUDIT_RECEIPT_FILE_SHA256",
    "AUDIT_RECEIPT_RELATIVE_PATH",
    "AUTHORITY_BINDINGS",
    "CANONICAL_ATTEMPT_PATH",
    "CANONICAL_GATE_PATH",
    "CANONICAL_METRIC_RECEIPT_PATH",
    "CANONICAL_OUTPUT_ROOT",
    "CANONICAL_SOURCE_REVIEW_PATH",
    "COMPLETION_SCHEMA",
    "DATASET_MANIFEST_CONTENT_SHA256",
    "DATASET_MANIFEST_FILE_SHA256",
    "DATASET_MANIFEST_RELATIVE_PATH",
    "EXPERIMENT",
    "EXPECTED_SCHEDULE_SHA256",
    "FAILURE_SCHEMA",
    "FINALIZER_RELATIVE_PATH",
    "FROZEN_SOURCE_BINDINGS",
    "GATE_SCHEMA",
    "IMPLEMENTATION_AUTHOR",
    "LAUNCHER_RELATIVE_PATH",
    "LICENSES",
    "LOSS_COMPONENTS",
    "LOSS_WEIGHTS",
    "METRIC_RECEIPT_SCHEMA",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "POLICY_RELATIVE_PATH",
    "RESERVATION_SCHEMA",
    "RESULT_SCHEMA",
    "RGB_RECEIPT_CONTENT_SHA256",
    "SCHEDULE_ALGORITHM",
    "SOURCE_REVIEW_RELATIVE_PATH",
    "SOURCE_REVIEW_SCHEMA",
    "SUBSET_CONTENT_SHA256",
    "SUCCESSOR_SOURCE_PATHS",
    "TARGET_PARTITION_CONTENT_SHA256",
    "TERMINAL_INVALIDATION_CONTENT_SHA256",
    "TERMINAL_INVALIDATION_FILE_SHA256",
    "THREAD_ENVIRONMENT",
    "TRAINER_AUTHORIZATION_CONTENT_SHA256",
    "TRAINER_AUTHORIZATION_FILE_SHA256",
    "TRAINER_AUTHORIZATION_RELATIVE_PATH",
    "TRAINER_RELATIVE_PATH",
    "TRAINER_REVIEW_CONTENT_SHA256",
    "TRAINER_REVIEW_FILE_SHA256",
    "TRAINER_REVIEW_RELATIVE_PATH",
    "VERIFIER_RELATIVE_PATH",
    "VerifiedAuthority",
    "artifact_binding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "expected_source_review_core",
    "is_sha256",
    "load_hashed_json",
    "parse_bound_path",
    "parse_json",
    "preflight_source_review",
    "preflight_static_authority",
    "read_hashed_bytes",
    "read_regular_bytes",
    "require_verified_authority",
    "source_review_binding",
    "validate_evaluation_structure",
    "validate_reservation_structure",
    "validate_result_structure",
    "verify_authority",
    "write_exclusive",
]
