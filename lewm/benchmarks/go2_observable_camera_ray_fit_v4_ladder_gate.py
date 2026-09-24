"""Pre-output numeric gate for the observable camera-ray V4 fit ladder.

This module is pure result validation.  It does not load a model, checkpoint,
dataset, RGB, heldout role, G2, or runtime artifact.  Structural/provenance
violations raise; valid results that miss a predeclared numeric threshold are
returned with an explicit failure inventory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence


FIT_RESULT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_development_result_v2"
METRIC_VERIFICATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_metric_verification_v2"
)
SEED_GATE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_seed_gate_v2"
STAGE_GATE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_stage_gate_v2"
TWO_SEED_GATE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_two_seed_gate_v2"
ATTEMPT_RESERVATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_attempt_reservation_v2"
)
ATTEMPT_COMPLETION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_attempt_completion_v2"
)
LADDER_CONTRACT = "observable_camera_ray_fit_v4_ladder_v3"
ROOT = Path(__file__).resolve().parents[2]
TARGET_PARTITION_FREEZE_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json"
)
TARGET_PARTITION_VERIFIER_PATH = (
    ROOT / "scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py"
)
TARGET_PARTITION_AMENDMENT_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v2_partition_amendment_2026-07-12.md"
)
LADDER_V3_AMENDMENT_PATH = (
    ROOT
    / "docs/lewm_go2_observable_camera_ray_fit_v4_ladder_v3_failure_successor_amendment_2026-07-13.md"
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
LADDER_V3_AMENDMENT_FILE_SHA256 = (
    "86718d072fe151b9419318c204d4130147e098150d4fd80557f9d5865dc8f9f3"
)
V1_FAILURE_LINEAGE = {
    "root": ".generated/go2_observable_camera_ray_fit_v4/development_fit_v1",
    "seed": 20260710,
    "fit_size": 5,
    "reservation": {
        "path": "attempts/seed_20260710/n5/reservation.json",
        "file_sha256": "115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e",
        "content_sha256": "ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd",
    },
    "failure": {
        "path": "attempts/seed_20260710/n5/failed.json",
        "file_sha256": "6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48",
        "content_sha256": "7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4",
    },
    "terminal_inventory": ["failed.json", "reservation.json"],
    "checkpoint_present": False,
    "result_present": False,
    "completion_present": False,
    "metric_verification_present": False,
}
DETERMINISM_WARNING_KERNELS = (
    "grid_sampler_2d_backward_cuda",
    "scatter_add_cuda_kernel",
)
_DETERMINISM_WARNING_SUFFIX = (
    " does not have a deterministic implementation, but you set "
    "'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file "
    "an issue at https://github.com/pytorch/pytorch/issues to help us prioritize "
    "adding deterministic support for this operation."
)
DETERMINISM_WARNING_WHITELIST = tuple(
    kernel + _DETERMINISM_WARNING_SUFFIX for kernel in DETERMINISM_WARNING_KERNELS
)
_PYTORCH_CONTEXT_TRAILER_PREFIX = (
    " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:"
)
_PYTORCH_CONTEXT_TRAILER_SUFFIX = ".)"


def _source_boundary_bytes(path: Path, expected_sha256: str, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"V4 {name} is not a regular source-boundary file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"V4 {name} is not a regular file")
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
            raise RuntimeError(f"V4 {name} changed while read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"V4 {name} file SHA-256 changed")
    return raw


def _load_target_partition_freeze() -> dict[str, Any]:
    _source_boundary_bytes(
        LADDER_V3_AMENDMENT_PATH,
        LADDER_V3_AMENDMENT_FILE_SHA256,
        name="ladder-v3 failure-successor amendment",
    )
    _source_boundary_bytes(
        TARGET_PARTITION_VERIFIER_PATH,
        TARGET_PARTITION_VERIFIER_FILE_SHA256,
        name="target-partition verifier",
    )
    _source_boundary_bytes(
        TARGET_PARTITION_AMENDMENT_PATH,
        TARGET_PARTITION_AMENDMENT_FILE_SHA256,
        name="partition amendment",
    )
    raw = _source_boundary_bytes(
        TARGET_PARTITION_FREEZE_PATH,
        TARGET_PARTITION_FREEZE_FILE_SHA256,
        name="target-partition freeze",
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("V4 target-partition freeze is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("V4 target-partition freeze root changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    encoded = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")
    if (
        declared != TARGET_PARTITION_FREEZE_CONTENT_SHA256
        or hashlib.sha256(encoded).hexdigest() != declared
        or value.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_target_partitions_v1"
        or value.get("verified_dataset_file_count") != 180
        or set(value.get("fit_sizes", {})) != {"5", "16", "32", "320"}
    ):
        raise ValueError("V4 target-partition freeze identity changed")
    return value


_TARGET_PARTITION_FREEZE = _load_target_partition_freeze()
LADDER_FIT_SIZES = (5, 16, 32, 320)
EXPECTED_SEEDS = (20260710, 20260711)
DEFAULT_STEPS = {5: 1000, 16: 1200, 32: 1600, 320: 3200}
TRAIN_BATCH_SIZE = 1
EVALUATION_BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_PARAMETER_COUNT = 3_105_513
EXPECTED_TRAINER_SOURCE_REHASHES = 43
SCHEDULE_ALGORITHM = (
    "torch_cpu_generator_manual_seed_then_concatenated_randperm_cycles_"
    "take_steps_times_batch_v1"
)
EXPECTED_SCHEDULE_SHA256 = {
    20260710: {
        5: "840d392b2bd20205a11355b7e22c2c58bd2214d6c3d488e9738136298b668b93",
        16: "36c1aec1a098c8d2f2d118683d98bc85d3916e5a977954e5358f8849a4827e65",
        32: "bf99215e92aa22b78c8932841768ac8de1730a94eea29e9377501a166fd5a4e4",
        320: "5ee928ebcb4a50c8f7db45d26433e1601a929932336804d39928133d28725857",
    },
    20260711: {
        5: "4cde96eaa1531a3fdc1030c4fe14c57eee65d8c4e52a4f338b77c1c6fff2b168",
        16: "9dcd773c40493c5e2d92d2f6d876bf29144c63bf7e7e57e67033fca0843e2d86",
        32: "6b7a4ff4709170da2e3093d0b16ac80d337782f00f268faf3694485802211440",
        320: "9206b5fe2ac0fc995a466609ebb9323e3a2dec7831021905e2e5164d34c8a2ee",
    },
}
EXPECTED_RASTER_CLASS_COUNTS = {
    size: dict(
        zip(
            ("unknown", "free", "occupied"),
            _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)]["signature"][
                "raster_target_counts"
            ],
            strict=True,
        )
    )
    for size in LADDER_FIT_SIZES
}
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
GROUND_DISTANCE_GROUPS = (
    "0.0_to_1.0",
    "1.0_to_2.0",
    "2.0_to_3.0",
    "3.0_to_4.0",
    "4.0_to_5.0",
    "5.0_plus",
)
RASTER_CLASSES = ("unknown", "free", "occupied")

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
RGB_RECEIPT_CONTENT_SHA256 = (
    "d763d7ae294e4e5a9e5f2352672913bc06411388d92abe1fb0f5090dfc41d5c3"
)
EXPECTED_SUBSET_CONTENT_SHA256 = {
    size: str(
        _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)][
            "subset_content_sha256"
        ]
    )
    for size in LADDER_FIT_SIZES
}
EXPECTED_TARGET_PARTITION_SIGNATURES: dict[int, Mapping[str, Any]] = {
    size: json.loads(
        json.dumps(
            _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)]["signature"]
        )
    )
    for size in LADDER_FIT_SIZES
}
EXPECTED_TARGET_PARTITION_SIGNATURE_SHA256 = {
    size: str(
        _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)]["signature_sha256"]
    )
    for size in LADDER_FIT_SIZES
}
EXPECTED_ORDERED_PER_FRAME_TARGET_SHA256 = {
    size: str(
        _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)][
            "ordered_per_frame_target_sha256"
        ]
    )
    for size in LADDER_FIT_SIZES
}
EXPECTED_ORDERED_TARGET_BYTES_SHA256 = {
    size: str(
        _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)][
            "ordered_target_bytes_sha256"
        ]
    )
    for size in LADDER_FIT_SIZES
}
EXPECTED_FIRST_FRAME_KEY_SHA256 = {
    size: str(
        _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)][
            "first_frame_key_sha256"
        ]
    )
    for size in LADDER_FIT_SIZES
}
EXPECTED_LAST_FRAME_KEY_SHA256 = {
    size: str(
        _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)][
            "last_frame_key_sha256"
        ]
    )
    for size in LADDER_FIT_SIZES
}
EXPECTED_FAMILY_COUNTS = {
    size: {
        str(family): int(count)
        for family, count in _TARGET_PARTITION_FREEZE["fit_sizes"][str(size)][
            "family_counts"
        ].items()
    }
    for size in LADDER_FIT_SIZES
}


def _expected_family_counts(fit_size: int) -> dict[str, int]:
    if fit_size not in EXPECTED_FAMILY_COUNTS:
        raise ValueError("V4 fit size is outside the frozen target partitions")
    return dict(EXPECTED_FAMILY_COUNTS[fit_size])


@dataclass(frozen=True)
class FitThresholdV4:
    pixel_hit_balanced_accuracy_min: float
    pixel_hit_depth_median_error_m_max: float
    pixel_hit_depth_p95_error_m_max: float
    ground_overall_balanced_accuracy_min: float
    ground_distance_balanced_accuracy_min: float
    ground_family_balanced_accuracy_min: float
    raster_nll_max: float
    raster_balanced_accuracy_min: float
    raster_class_recall_min: float
    wrong_pixel_balanced_accuracy_drop_min: float | None
    wrong_depth_median_error_increase_m_min: float | None
    wrong_depth_p95_error_increase_m_min: float | None
    wrong_ground_balanced_accuracy_drop_min: float | None
    wrong_raster_nll_increase_min: float | None
    wrong_raster_balanced_accuracy_drop_min: float | None


FIT_THRESHOLDS: dict[int, FitThresholdV4] = {
    5: FitThresholdV4(
        pixel_hit_balanced_accuracy_min=0.99,
        pixel_hit_depth_median_error_m_max=0.06,
        pixel_hit_depth_p95_error_m_max=0.15,
        ground_overall_balanced_accuracy_min=0.99,
        ground_distance_balanced_accuracy_min=0.97,
        ground_family_balanced_accuracy_min=0.97,
        raster_nll_max=0.06,
        raster_balanced_accuracy_min=0.99,
        raster_class_recall_min=0.97,
        wrong_pixel_balanced_accuracy_drop_min=0.08,
        wrong_depth_median_error_increase_m_min=0.08,
        wrong_depth_p95_error_increase_m_min=0.12,
        wrong_ground_balanced_accuracy_drop_min=0.08,
        wrong_raster_nll_increase_min=0.08,
        wrong_raster_balanced_accuracy_drop_min=0.08,
    ),
    16: FitThresholdV4(
        pixel_hit_balanced_accuracy_min=0.97,
        pixel_hit_depth_median_error_m_max=0.08,
        pixel_hit_depth_p95_error_m_max=0.20,
        ground_overall_balanced_accuracy_min=0.97,
        ground_distance_balanced_accuracy_min=0.94,
        ground_family_balanced_accuracy_min=0.94,
        raster_nll_max=0.10,
        raster_balanced_accuracy_min=0.97,
        raster_class_recall_min=0.95,
        wrong_pixel_balanced_accuracy_drop_min=0.10,
        wrong_depth_median_error_increase_m_min=0.10,
        wrong_depth_p95_error_increase_m_min=0.15,
        wrong_ground_balanced_accuracy_drop_min=0.10,
        wrong_raster_nll_increase_min=0.10,
        wrong_raster_balanced_accuracy_drop_min=0.10,
    ),
    32: FitThresholdV4(
        pixel_hit_balanced_accuracy_min=0.95,
        pixel_hit_depth_median_error_m_max=0.10,
        pixel_hit_depth_p95_error_m_max=0.25,
        ground_overall_balanced_accuracy_min=0.95,
        ground_distance_balanced_accuracy_min=0.92,
        ground_family_balanced_accuracy_min=0.92,
        raster_nll_max=0.15,
        raster_balanced_accuracy_min=0.95,
        raster_class_recall_min=0.92,
        wrong_pixel_balanced_accuracy_drop_min=0.11,
        wrong_depth_median_error_increase_m_min=0.11,
        wrong_depth_p95_error_increase_m_min=0.18,
        wrong_ground_balanced_accuracy_drop_min=0.11,
        wrong_raster_nll_increase_min=0.11,
        wrong_raster_balanced_accuracy_drop_min=0.11,
    ),
    320: FitThresholdV4(
        pixel_hit_balanced_accuracy_min=0.95,
        pixel_hit_depth_median_error_m_max=0.10,
        pixel_hit_depth_p95_error_m_max=0.25,
        ground_overall_balanced_accuracy_min=0.95,
        ground_distance_balanced_accuracy_min=0.92,
        ground_family_balanced_accuracy_min=0.92,
        raster_nll_max=0.15,
        raster_balanced_accuracy_min=0.95,
        raster_class_recall_min=0.95,
        wrong_pixel_balanced_accuracy_drop_min=0.12,
        wrong_depth_median_error_increase_m_min=0.12,
        wrong_depth_p95_error_increase_m_min=0.20,
        wrong_ground_balanced_accuracy_drop_min=0.12,
        wrong_raster_nll_increase_min=0.12,
        wrong_raster_balanced_accuracy_drop_min=0.12,
    ),
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def target_partition_binding_v4(fit_size: int) -> dict[str, Any]:
    if isinstance(fit_size, bool) or fit_size not in LADDER_FIT_SIZES:
        raise ValueError("V4 target-partition fit size is outside the ladder")
    entry = _TARGET_PARTITION_FREEZE["fit_sizes"][str(fit_size)]
    core = {
        "schema": "lewm_go2_observable_camera_ray_fit_v4_target_partition_binding_v1",
        "fit_size": fit_size,
        "freeze_file_sha256": TARGET_PARTITION_FREEZE_FILE_SHA256,
        "freeze_content_sha256": TARGET_PARTITION_FREEZE_CONTENT_SHA256,
        "verifier_file_sha256": TARGET_PARTITION_VERIFIER_FILE_SHA256,
        "amendment_file_sha256": TARGET_PARTITION_AMENDMENT_FILE_SHA256,
        "verified_dataset_file_count": 180,
        "family_counts": dict(entry["family_counts"]),
        "first_frame_key_sha256": entry["first_frame_key_sha256"],
        "last_frame_key_sha256": entry["last_frame_key_sha256"],
        "subset_content_sha256": entry["subset_content_sha256"],
        "signature_sha256": entry["signature_sha256"],
        "ordered_per_frame_target_sha256": entry[
            "ordered_per_frame_target_sha256"
        ],
        "ordered_target_bytes_sha256": entry["ordered_target_bytes_sha256"],
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def validate_target_partition_binding_v4(
    value: object,
    *,
    fit_size: int,
) -> dict[str, Any]:
    expected = target_partition_binding_v4(fit_size)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ValueError(f"V4 N{fit_size} target-partition binding changed")
    return expected


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def _validate_binary_metrics(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "confusion_target_rows_predicted_columns",
        "negative_recall",
        "positive_recall",
        "balanced_accuracy",
        "count",
    }:
        raise ValueError(f"{name} binary metric fields changed")
    confusion = value["confusion_target_rows_predicted_columns"]
    if not isinstance(confusion, list) or len(confusion) != 2 or any(
        not isinstance(row, list) or len(row) != 2 for row in confusion
    ):
        raise ValueError(f"{name} confusion must be 2x2")
    matrix = [
        [
            _nonnegative_integer(cell, name=f"{name} confusion cell")
            for cell in row
        ]
        for row in confusion
    ]
    count = _nonnegative_integer(value["count"], name=f"{name} count")
    if sum(sum(row) for row in matrix) != count:
        raise ValueError(f"{name} confusion does not sum to count")
    expected_recalls: list[float | None] = []
    for state in range(2):
        reference_count = sum(matrix[state])
        expected_recalls.append(
            None if reference_count == 0 else matrix[state][state] / reference_count
        )
    for key, expected in zip(("negative_recall", "positive_recall"), expected_recalls):
        observed = value[key]
        if expected is None:
            if observed is not None:
                raise ValueError(f"{name} absent recall must be null")
        elif not _close(_finite_number(observed, name=f"{name} {key}"), expected):
            raise ValueError(f"{name} recall disagrees with confusion")
    present = [item for item in expected_recalls if item is not None]
    expected_balanced = None if not present else sum(present) / len(present)
    observed_balanced = value["balanced_accuracy"]
    if expected_balanced is None:
        if observed_balanced is not None:
            raise ValueError(f"{name} absent balanced accuracy must be null")
    elif not _close(
        _finite_number(observed_balanced, name=f"{name} balanced accuracy"),
        expected_balanced,
    ):
        raise ValueError(f"{name} balanced accuracy disagrees with confusion")
    return {
        **dict(value),
        "balanced_accuracy": expected_balanced,
        "count": count,
    }


def _validate_raster_metrics(
    value: object,
    *,
    name: str,
    fit_size: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "nll",
        "nll_sum",
        "confusion_target_rows_predicted_columns",
        "class_recalls",
        "balanced_accuracy",
        "count",
    }:
        raise ValueError(f"{name} raster metric fields changed")
    count = _nonnegative_integer(value["count"], name=f"{name} count")
    if count != fit_size * 64 * 64:
        raise ValueError(f"{name} raster count changed")
    confusion = value["confusion_target_rows_predicted_columns"]
    if not isinstance(confusion, list) or len(confusion) != 3 or any(
        not isinstance(row, list) or len(row) != 3 for row in confusion
    ):
        raise ValueError(f"{name} raster confusion must be 3x3")
    matrix = [
        [
            _nonnegative_integer(cell, name=f"{name} confusion cell")
            for cell in row
        ]
        for row in confusion
    ]
    if sum(sum(row) for row in matrix) != count:
        raise ValueError(f"{name} raster confusion does not sum to count")
    recalls = value["class_recalls"]
    if not isinstance(recalls, Mapping) or set(recalls) != set(RASTER_CLASSES):
        raise ValueError(f"{name} raster class recalls changed")
    expected_class_counts = EXPECTED_RASTER_CLASS_COUNTS[fit_size]
    expected_recalls: list[float | None] = []
    for class_index, class_name in enumerate(RASTER_CLASSES):
        reference_count = sum(matrix[class_index])
        if reference_count != expected_class_counts[class_name]:
            raise ValueError(f"{name} raster target class counts changed")
        if reference_count == 0:
            if recalls[class_name] is not None:
                raise ValueError(f"{name} absent raster class recall must be null")
            expected_recalls.append(None)
            continue
        expected = matrix[class_index][class_index] / reference_count
        if not _close(
            _finite_number(recalls[class_name], name=f"{name} {class_name} recall"),
            expected,
        ):
            raise ValueError(f"{name} raster recall disagrees with confusion")
        expected_recalls.append(expected)
    present_recalls = [recall for recall in expected_recalls if recall is not None]
    balanced = sum(present_recalls) / len(present_recalls)
    if not _close(
        _finite_number(value["balanced_accuracy"], name=f"{name} raster BA"),
        balanced,
    ):
        raise ValueError(f"{name} raster balanced accuracy disagrees with confusion")
    nll = _finite_number(value["nll"], name=f"{name} raster NLL")
    nll_sum = _finite_number(value["nll_sum"], name=f"{name} raster NLL sum")
    if nll < 0.0 or nll_sum < 0.0 or not _close(nll, nll_sum / count):
        raise ValueError(f"{name} raster NLL cannot be negative")
    return {
        **dict(value),
        "nll": nll,
        "nll_sum": nll_sum,
        "balanced_accuracy": balanced,
        "class_recalls": {
            class_name: expected_recalls[index]
            for index, class_name in enumerate(RASTER_CLASSES)
        },
    }


def _validate_metric_bundle(
    value: object,
    *,
    fit_size: int,
    expected_families: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "frame_count",
        "pixel_hit_no_hit",
        "pixel_hit_depth",
        "ground_clear",
        "derived_raster",
    }:
        raise ValueError(f"{name} metric bundle fields changed")
    if value["frame_count"] != fit_size:
        raise ValueError(f"{name} metric frame count changed")
    pixel = _validate_binary_metrics(
        value["pixel_hit_no_hit"], name=f"{name} pixel hit/no-hit"
    )
    if pixel["count"] != fit_size * 84 * 112:
        raise ValueError(f"{name} pixel-ray count changed")
    depth = value["pixel_hit_depth"]
    if not isinstance(depth, Mapping) or set(depth) != {
        "count",
        "median_absolute_error_m",
        "p95_absolute_error_m",
        "absolute_error_evidence",
    }:
        raise ValueError(f"{name} pixel-depth fields changed")
    depth_count = _nonnegative_integer(depth["count"], name=f"{name} depth count")
    pixel_positive_count = sum(
        pixel["confusion_target_rows_predicted_columns"][1]
    )
    if not 0 < depth_count <= pixel["count"] or depth_count != pixel_positive_count:
        raise ValueError(f"{name} pixel-depth target count changed")
    median = _finite_number(
        depth["median_absolute_error_m"], name=f"{name} depth median"
    )
    p95 = _finite_number(depth["p95_absolute_error_m"], name=f"{name} depth p95")
    if median < 0.0 or p95 < median:
        raise ValueError(f"{name} pixel-depth errors are malformed")
    evidence = depth.get("absolute_error_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "dtype",
        "quantile_method",
        "sorted_values_sha256",
        "median",
        "p95",
    } or (
        evidence.get("dtype") != "little_endian_float64"
        or evidence.get("quantile_method") != "linear_interpolation_n_minus_1_v1"
        or not _is_sha256(evidence.get("sorted_values_sha256"))
    ):
        raise ValueError(f"{name} pixel-depth evidence changed")

    def validate_quantile(record: object, quantile: float, observed: float) -> None:
        if not isinstance(record, Mapping) or set(record) != {
            "quantile",
            "lower_index",
            "upper_index",
            "upper_weight",
            "lower_value_m",
            "upper_value_m",
        }:
            raise ValueError(f"{name} depth quantile evidence changed")
        position = (depth_count - 1) * quantile
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        weight = position - lower_index
        lower_value = _finite_number(
            record.get("lower_value_m"), name=f"{name} depth lower value"
        )
        upper_value = _finite_number(
            record.get("upper_value_m"), name=f"{name} depth upper value"
        )
        if (
            not _close(_finite_number(record.get("quantile"), name="quantile"), quantile)
            or record.get("lower_index") != lower_index
            or record.get("upper_index") != upper_index
            or not _close(
                _finite_number(record.get("upper_weight"), name="quantile weight"),
                weight,
            )
            or lower_value < 0.0
            or upper_value < lower_value
            or not _close(observed, lower_value * (1.0 - weight) + upper_value * weight)
        ):
            raise ValueError(f"{name} depth quantile evidence is inconsistent")

    validate_quantile(evidence.get("median"), 0.5, median)
    validate_quantile(evidence.get("p95"), 0.95, p95)

    ground = value["ground_clear"]
    if not isinstance(ground, Mapping) or set(ground) != {
        "overall",
        "by_distance_m",
        "by_family",
    }:
        raise ValueError(f"{name} ground metric fields changed")
    overall = _validate_binary_metrics(ground["overall"], name=f"{name} ground")
    if overall["count"] <= 0:
        raise ValueError(f"{name} ground metric has no valid supports")
    by_distance_raw = ground["by_distance_m"]
    if not isinstance(by_distance_raw, Mapping) or set(by_distance_raw) != set(
        GROUND_DISTANCE_GROUPS
    ):
        raise ValueError(f"{name} ground distance groups changed")
    by_distance = {
        group: _validate_binary_metrics(
            by_distance_raw[group], name=f"{name} ground distance {group}"
        )
        for group in GROUND_DISTANCE_GROUPS
    }
    if sum(item["count"] for item in by_distance.values()) != overall["count"]:
        raise ValueError(f"{name} ground distance counts do not partition overall")
    for target_state in range(2):
        if sum(
            sum(item["confusion_target_rows_predicted_columns"][target_state])
            for item in by_distance.values()
        ) != sum(overall["confusion_target_rows_predicted_columns"][target_state]):
            raise ValueError(f"{name} ground distance target classes do not partition")
    if sum(item["count"] > 0 for item in by_distance.values()) < 2:
        raise ValueError(f"{name} ground distance support is too narrow")
    by_family_raw = ground["by_family"]
    if not isinstance(by_family_raw, Mapping) or set(by_family_raw) != expected_families:
        raise ValueError(f"{name} ground family groups changed")
    by_family = {
        family: _validate_binary_metrics(
            by_family_raw[family], name=f"{name} ground family {family}"
        )
        for family in sorted(expected_families)
    }
    if any(item["count"] <= 0 for item in by_family.values()) or sum(
        item["count"] for item in by_family.values()
    ) != overall["count"]:
        raise ValueError(f"{name} ground family counts do not partition overall")
    for target_state in range(2):
        if sum(
            sum(item["confusion_target_rows_predicted_columns"][target_state])
            for item in by_family.values()
        ) != sum(overall["confusion_target_rows_predicted_columns"][target_state]):
            raise ValueError(f"{name} ground family target classes do not partition")
    raster = _validate_raster_metrics(
        value["derived_raster"],
        name=f"{name} derived raster",
        fit_size=fit_size,
    )
    return {
        "pixel_hit_no_hit": pixel,
        "pixel_hit_depth": {
            "count": depth_count,
            "median_absolute_error_m": median,
            "p95_absolute_error_m": p95,
            "absolute_error_evidence": dict(evidence),
        },
        "ground_clear": {
            "overall": overall,
            "by_distance_m": by_distance,
            "by_family": by_family,
        },
        "derived_raster": raster,
    }


def _target_partition_signature(metrics: Mapping[str, Any]) -> dict[str, Any]:
    def rows(binary: Mapping[str, Any]) -> list[int]:
        return [
            sum(binary["confusion_target_rows_predicted_columns"][state])
            for state in range(2)
        ]

    return {
        "pixel": rows(metrics["pixel_hit_no_hit"]),
        "depth_count": metrics["pixel_hit_depth"]["count"],
        "ground_overall": rows(metrics["ground_clear"]["overall"]),
        "ground_by_distance": {
            group: rows(value)
            for group, value in metrics["ground_clear"]["by_distance_m"].items()
        },
        "ground_by_family": {
            family: rows(value)
            for family, value in metrics["ground_clear"]["by_family"].items()
        },
        "raster_target_counts": [
            sum(row)
            for row in metrics["derived_raster"][
                "confusion_target_rows_predicted_columns"
            ]
        ],
    }


def _validate_frozen_target_partition_signature(
    value: object,
    *,
    fit_size: int,
) -> dict[str, Any]:
    expected = EXPECTED_TARGET_PARTITION_SIGNATURES.get(fit_size)
    if expected is None:
        raise PermissionError(
            f"V4 N{fit_size} target-partition constants are not independently frozen"
        )
    normalized = json.loads(_canonical_json_bytes(value).decode("ascii"))
    if normalized != expected:
        raise ValueError(f"V4 N{fit_size} target-partition signature changed")
    return normalized


def _validated_metric_evaluation(
    evaluation: object,
    *,
    fit_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "matched_rgb",
        "wrong_rgb_with_target_calibration",
    }:
        raise ValueError("V4 metric-verification evaluation fields changed")
    matched = evaluation["matched_rgb"]
    wrong = evaluation["wrong_rgb_with_target_calibration"]
    expected_families = {
        family
        for family, count in _expected_family_counts(fit_size).items()
        if count > 0
    }
    for value, control, wrong_flag, name in (
        (matched, "matched_rgb", False, "matched"),
        (
            wrong,
            "wrong_rgb_with_target_calibration",
            True,
            "wrong-RGB",
        ),
    ):
        if not isinstance(value, Mapping) or set(value) != {
            "control",
            "wrong_rgb_degenerate_singleton",
            "image_index_mapping",
            "image_mapping_sha256",
            "losses",
            "metrics",
        } or (
            value.get("control") != control
            or value.get("wrong_rgb_degenerate_singleton") is not False
        ):
            raise ValueError(f"V4 {name} verification control changed")
        mapping = value.get("image_index_mapping")
        expected_mapping = [
            ((index + 1) % fit_size) if wrong_flag else index
            for index in range(fit_size)
        ]
        if (
            mapping != expected_mapping
            or value.get("image_mapping_sha256")
            != canonical_json_sha256(expected_mapping)
        ):
            raise ValueError(f"V4 {name} image mapping changed")
        _validate_evaluation_losses(value.get("losses"), name=name)
    matched_metrics = _validate_metric_bundle(
        matched["metrics"],
        fit_size=fit_size,
        expected_families=expected_families,
        name="V4 verified matched",
    )
    wrong_metrics = _validate_metric_bundle(
        wrong["metrics"],
        fit_size=fit_size,
        expected_families=expected_families,
        name="V4 verified wrong-RGB",
    )
    signature = _target_partition_signature(matched_metrics)
    if _target_partition_signature(wrong_metrics) != signature:
        raise ValueError("V4 verified matched/wrong target partitions differ")
    return matched_metrics, wrong_metrics, _validate_frozen_target_partition_signature(
        signature,
        fit_size=fit_size,
    )


def _validate_evaluation_losses(value: object, *, name: str) -> dict[str, float]:
    expected_components = {
        "ordered_first_hit_nll",
        "target_bin_offset_smooth_l1",
        "ground_clear_distance_state_balanced_bce",
        "derived_raster_hierarchical_bce",
    }
    if not isinstance(value, Mapping) or set(value) != expected_components | {"total"}:
        raise ValueError(f"V4 {name} evaluation loss fields changed")
    normalized = {
        key: _finite_number(item, name=f"V4 {name} loss {key}")
        for key, item in value.items()
    }
    if any(item < 0.0 for item in normalized.values()) or not _close(
        normalized["total"],
        0.25 * sum(normalized[key] for key in expected_components),
    ):
        raise ValueError(f"V4 {name} evaluation losses are inconsistent")
    return normalized


def _validate_result(
    result: Mapping[str, Any],
    *,
    expected_seed: int,
    previous_stage_binding: Mapping[str, Any] | None,
    seed_20260710_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "mode",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "dataset_role",
        "fit_size",
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
    if set(result) != expected_fields or result.get("schema") != FIT_RESULT_SCHEMA:
        raise ValueError("V4 development fit result schema changed")
    core = dict(result)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or declared != canonical_json_sha256(core):
        raise ValueError("V4 development fit result content hash changed")
    if (
        result.get("mode") != "exact_development_fit"
        or result.get("dataset_role") != "train"
        or result.get("authoritative") is not False
        or result.get("aggregation_eligible") is not False
        or result.get("promotion_eligible") is not False
    ):
        raise PermissionError("V4 fit result crossed its development-only scope")
    fit_size = result.get("fit_size")
    if isinstance(fit_size, bool) or fit_size not in LADDER_FIT_SIZES:
        raise ValueError("V4 result fit size is outside the ladder")
    if expected_seed not in EXPECTED_SEEDS:
        raise ValueError("V4 result seed is outside the frozen sequence")
    attempt = result.get("attempt")
    if not isinstance(attempt, Mapping) or set(attempt) != {
        "attempt_index",
        "maximum_attempts",
        "scope",
        "reservation",
        "predecessor_failure",
    } or (
        attempt.get("attempt_index") != 1
        or attempt.get("maximum_attempts") != 1
        or attempt.get("scope") != "one_frozen_attempt_per_seed_and_fit_size"
        or attempt.get("predecessor_failure") != V1_FAILURE_LINEAGE
    ):
        raise PermissionError("V4 result violates the one-attempt contract")
    reservation_binding = attempt.get("reservation")
    if not isinstance(reservation_binding, Mapping) or set(reservation_binding) != {
        "path",
        "file_sha256",
        "content_sha256",
    } or (
        reservation_binding.get("path") != "reservation.json"
        or not _is_sha256(reservation_binding.get("file_sha256"))
        or not _is_sha256(reservation_binding.get("content_sha256"))
    ):
        raise PermissionError("V4 result attempt reservation binding changed")

    inputs = result.get("inputs")
    expected_input_fields = {
        "dataset_manifest_file_sha256",
        "dataset_manifest_content_sha256",
        "audit_receipt_file_sha256",
        "audit_receipt_content_sha256",
        "trainer_authorization_file_sha256",
        "trainer_authorization_content_sha256",
        "trainer_review_record_file_sha256",
        "trainer_review_record_content_sha256",
        "rgb_receipt_content_sha256",
        "target_partition_content_sha256",
    }
    if fit_size != LADDER_FIT_SIZES[0]:
        expected_input_fields.add("previous_stage_gate")
    if expected_seed == 20260711:
        expected_input_fields.add("seed_20260710_gate")
    if not isinstance(inputs, Mapping) or set(inputs) != expected_input_fields:
        raise ValueError("V4 result input receipt fields changed")
    if (
        inputs.get("dataset_manifest_file_sha256") != DATASET_MANIFEST_FILE_SHA256
        or inputs.get("dataset_manifest_content_sha256")
        != DATASET_MANIFEST_CONTENT_SHA256
        or inputs.get("audit_receipt_file_sha256") != AUDIT_RECEIPT_FILE_SHA256
        or inputs.get("audit_receipt_content_sha256")
        != AUDIT_RECEIPT_CONTENT_SHA256
        or not _is_sha256(inputs.get("trainer_authorization_file_sha256"))
        or not _is_sha256(inputs.get("trainer_authorization_content_sha256"))
        or not _is_sha256(inputs.get("trainer_review_record_file_sha256"))
        or not _is_sha256(inputs.get("trainer_review_record_content_sha256"))
        or inputs.get("rgb_receipt_content_sha256") != RGB_RECEIPT_CONTENT_SHA256
        or inputs.get("target_partition_content_sha256")
        != target_partition_binding_v4(fit_size)["content_sha256"]
    ):
        raise ValueError("V4 result does not bind the reviewed exact inputs")
    if fit_size == LADDER_FIT_SIZES[0]:
        if previous_stage_binding is not None:
            raise ValueError("the first fit rung may not have a previous-stage gate")
    elif previous_stage_binding is None or inputs.get("previous_stage_gate") != dict(
        previous_stage_binding
    ):
        raise PermissionError("larger V4 rung lacks its passing previous-stage gate")
    if expected_seed == 20260710:
        if seed_20260710_binding is not None:
            raise ValueError("seed 20260710 may not bind itself as a prerequisite")
    elif seed_20260710_binding is None or inputs.get("seed_20260710_gate") != dict(
        seed_20260710_binding
    ):
        raise PermissionError("seed 20260711 lacks the passing seed-20260710 gate")

    subset = result.get("subset")
    if not isinstance(subset, Mapping) or set(subset) != {
        "namespace",
        "parent_frame_count",
        "fit_size",
        "selection",
        "family_counts",
        "ordered_frame_key_sha256",
        "content_sha256",
    } or (
        subset.get("namespace") != "lewm_go2_observable_camera_ray_fit_v4_subset_v1"
        or subset.get("parent_frame_count") != 320
        or subset.get("fit_size") != fit_size
        or subset.get("selection")
        != (
            "registered_family_round_robin_then_namespaced_sha256_"
            "ascii_backslash_zero_rank_v1"
        )
    ):
        raise ValueError("V4 deterministic subset receipt changed")
    family_counts = subset.get("family_counts")
    if not isinstance(family_counts, Mapping) or set(family_counts) != set(FAMILIES):
        raise ValueError("V4 subset family counts changed")
    normalized_family_counts = {
        family: _nonnegative_integer(
            family_counts[family], name=f"V4 subset family {family} count"
        )
        for family in FAMILIES
    }
    if sum(normalized_family_counts.values()) != fit_size or (
        max(normalized_family_counts.values()) - min(normalized_family_counts.values())
        > 1
    ):
        raise ValueError("V4 subset is not round-robin family balanced")
    keys = subset.get("ordered_frame_key_sha256")
    if not isinstance(keys, list) or len(keys) != fit_size or not all(
        _is_sha256(value) for value in keys
    ) or len(set(keys)) != fit_size:
        raise ValueError("V4 subset ordered frame-key receipt changed")
    if (
        subset.get("content_sha256") != EXPECTED_SUBSET_CONTENT_SHA256[fit_size]
        or subset.get("content_sha256") != canonical_json_sha256(keys)
        or subset.get("family_counts") != EXPECTED_FAMILY_COUNTS[fit_size]
        or keys[0] != EXPECTED_FIRST_FRAME_KEY_SHA256[fit_size]
        or keys[-1] != EXPECTED_LAST_FRAME_KEY_SHA256[fit_size]
    ):
        raise ValueError("V4 subset content hash changed")
    target_partition = validate_target_partition_binding_v4(
        result.get("target_partition"),
        fit_size=fit_size,
    )
    if (
        target_partition["subset_content_sha256"]
        != subset["content_sha256"]
        or target_partition["family_counts"] != subset["family_counts"]
    ):
        raise ValueError("V4 result target partition differs from its subset")

    licenses = result.get("licenses")
    if not isinstance(licenses, Mapping) or set(licenses) != {
        "development_checkpoint_creation_authorized",
        "checkpoint_use_authorized",
        "holdout_authorized",
        "g2_authorized",
        "runtime_authorized",
        "promotion_authorized",
    } or licenses.get("development_checkpoint_creation_authorized") is not True or any(
        licenses.get(field) is not False
        for field in (
            "checkpoint_use_authorized",
            "holdout_authorized",
            "g2_authorized",
            "runtime_authorized",
            "promotion_authorized",
        )
    ):
        raise PermissionError("V4 result checkpoint creation/use licenses changed")
    access = result.get("access_ledger")
    expected_access_fields = {
        "selected_rgb_count",
        "nonselected_rgb_opens",
        "rgb_hash_opens",
        "rgb_decodes",
        "worker_start_method",
        "worker_count",
        "native_threads_per_worker",
        "selected_rgb_rehashes_before_publication",
        "heldout_opens",
        "g2_opens",
        "runtime_opens",
        "gpu1_uses",
        "dataset_root_inventory_revalidations",
        "shard_directory_inventory_revalidations",
        "dataset_frame_revalidations",
        "dataset_file_rehashes",
        "trainer_source_rehashes",
        "dataset_source_rehashes",
    }
    expected_workers = min(6, fit_size)
    if not isinstance(access, Mapping) or set(access) != expected_access_fields or (
        access.get("selected_rgb_count") != fit_size
        or access.get("rgb_hash_opens") != fit_size
        or access.get("rgb_decodes") != fit_size
        or access.get("selected_rgb_rehashes_before_publication") != fit_size
        or access.get("worker_count") != expected_workers
        or access.get("worker_start_method")
        != "spawn"
        or access.get("native_threads_per_worker") != 1
        or access.get("dataset_root_inventory_revalidations") != 1
        or access.get("shard_directory_inventory_revalidations") != 20
        or access.get("dataset_frame_revalidations") != 320
        or not isinstance(access.get("dataset_file_rehashes"), int)
        or access["dataset_file_rehashes"] <= 0
        or access.get("trainer_source_rehashes")
        != EXPECTED_TRAINER_SOURCE_REHASHES
        or access.get("dataset_source_rehashes") != 11
        or any(
            access.get(field) != 0
            for field in (
                "heldout_opens",
                "g2_opens",
                "runtime_opens",
                "nonselected_rgb_opens",
                "gpu1_uses",
            )
        )
    ):
        raise PermissionError("V4 result access ledger crossed a forbidden role")
    resource = result.get("resource")
    if not isinstance(resource, Mapping) or set(resource) != {
        "device",
        "device_name",
        "visible_device_count",
        "total_memory_bytes",
        "hip_visible_devices",
        "hsa_override_gfx_version_unset",
        "raphael_rejected",
        "minimum_memory_bytes",
        "native_thread_environment",
    } or (
        resource.get("device") != "cuda:0"
        or resource.get("visible_device_count") != 1
        or resource.get("hip_visible_devices") != "0"
        or resource.get("raphael_rejected") is not True
        or resource.get("hsa_override_gfx_version_unset") is not True
        or resource.get("minimum_memory_bytes") != 16 * 1024**3
        or not isinstance(resource.get("total_memory_bytes"), int)
        or resource["total_memory_bytes"] < resource["minimum_memory_bytes"]
        or "r9700" not in str(resource.get("device_name", "")).casefold()
        or "raphael" in str(resource.get("device_name", "")).casefold()
        or resource.get("native_thread_environment")
        != {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    ):
        raise PermissionError("V4 result did not use the registered R9700 GPU 0 scope")
    model = result.get("model")
    if not isinstance(model, Mapping) or set(model) != {
        "class",
        "parameter_count",
        "checkpoint",
    } or (
        model.get("class") != "ObservableCameraRayEvidenceV4Model"
        or model.get("parameter_count") != MODEL_PARAMETER_COUNT
    ):
        raise ValueError("V4 result model contract changed")
    checkpoint = model.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "file_sha256",
        "byte_count",
        "content_sha256",
        "development_only",
    } or (
        checkpoint.get("path") != "checkpoint.pt"
        or not _is_sha256(checkpoint.get("file_sha256"))
        or not _is_sha256(checkpoint.get("content_sha256"))
        or not isinstance(checkpoint.get("byte_count"), int)
        or checkpoint["byte_count"] <= 0
        or checkpoint.get("development_only") is not True
    ):
        raise ValueError("V4 development checkpoint receipt changed")
    training = result.get("training")
    expected_training_fields = {
        "steps",
        "batch_size",
        "evaluation_batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "precision",
        "autocast",
        "gradient_clip_norm",
        "loss_weights",
        "initial",
        "final",
        "best_total",
        "trace",
        "schedule_algorithm",
        "schedule_sha256",
    }
    if not isinstance(training, Mapping) or set(training) != expected_training_fields or (
        training.get("steps") != DEFAULT_STEPS[fit_size]
        or training.get("batch_size") != TRAIN_BATCH_SIZE
        or training.get("evaluation_batch_size") != EVALUATION_BATCH_SIZE
        or training.get("learning_rate") != LEARNING_RATE
        or training.get("weight_decay") != WEIGHT_DECAY
        or training.get("optimizer") != "AdamW"
        or training.get("loss_weights")
        != {
            "ordered_first_hit_nll": 0.25,
            "target_bin_offset_smooth_l1": 0.25,
            "ground_clear_distance_state_balanced_bce": 0.25,
            "derived_raster_hierarchical_bce": 0.25,
        }
        or training.get("precision") != "float32"
        or training.get("autocast") is not False
        or training.get("gradient_clip_norm") != 1.0
        or training.get("schedule_algorithm") != SCHEDULE_ALGORITHM
        or training.get("schedule_sha256")
        != EXPECTED_SCHEDULE_SHA256[expected_seed][fit_size]
    ):
        raise ValueError("V4 result training/loss contract changed")
    if not isinstance(training.get("initial"), Mapping) or training["initial"].get(
        "step"
    ) != 1 or not isinstance(training.get("final"), Mapping) or training["final"].get(
        "step"
    ) != DEFAULT_STEPS[fit_size]:
        raise ValueError("V4 result initial/final exposure receipt changed")
    trace = training.get("trace")
    if not isinstance(trace, list) or not trace or any(
        not isinstance(item, Mapping) for item in trace
    ):
        raise ValueError("V4 result training trace is malformed")
    if trace[0].get("step") != 1 or trace[-1].get("step") != DEFAULT_STEPS[
        fit_size
    ] or any(
        int(left["step"]) >= int(right["step"])
        for left, right in zip(trace[:-1], trace[1:])
    ):
        raise ValueError("V4 result training trace exposure changed")
    if not math.isfinite(
        _finite_number(training.get("best_total"), name="V4 best training loss")
    ):
        raise ValueError("V4 best training loss is non-finite")
    determinism = result.get("determinism")
    expected_determinism_fields = {
        "seed",
        "requested",
        "effective",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "torch_num_threads",
        "torch_num_interop_threads",
        "warning_count",
        "raw_messages",
        "normalized_messages",
        "normalization",
        "whitelist",
        "kernel_inventory",
        "kernel_counts",
    }
    if not isinstance(determinism, Mapping) or set(determinism) != expected_determinism_fields or (
        determinism.get("seed") != expected_seed
        or determinism.get("requested") != "strict_deterministic_algorithms"
        or determinism.get("effective")
        != "strict_where_supported_warn_on_exact_allowlisted_kernels"
        or determinism.get("cudnn_benchmark") is not False
        or determinism.get("cudnn_deterministic") is not True
        or determinism.get("torch_num_threads") != 1
        or determinism.get("torch_num_interop_threads") != 1
    ):
        raise ValueError("V4 result seed/determinism contract changed")
    raw_messages = determinism.get("raw_messages")
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, str) for message in raw_messages
    ):
        raise ValueError("V4 raw determinism warning evidence changed")
    expected_normalized = []
    expected_records = []
    for raw in raw_messages:
        normalized = None
        source_line = None
        if raw in DETERMINISM_WARNING_WHITELIST:
            normalized = raw
        else:
            for allowed in DETERMINISM_WARNING_WHITELIST:
                prefix = allowed + _PYTORCH_CONTEXT_TRAILER_PREFIX
                if not raw.startswith(prefix) or not raw.endswith(
                    _PYTORCH_CONTEXT_TRAILER_SUFFIX
                ):
                    continue
                digits = raw[len(prefix) : -len(_PYTORCH_CONTEXT_TRAILER_SUFFIX)]
                if (
                    digits
                    and digits[0] != "0"
                    and all(character in "0123456789" for character in digits)
                ):
                    normalized = allowed
                    source_line = int(digits)
                break
        if normalized is None:
            raise ValueError("V4 result contains an unallowlisted determinism warning")
        expected_normalized.append(normalized)
        expected_records.append(
            {
                "raw": raw,
                "normalized": normalized,
                "context_source_line": source_line,
                "trailer_removed": source_line is not None,
            }
        )
    expected_counts = {
        kernel: sum(message.startswith(kernel) for message in expected_normalized)
        for kernel in DETERMINISM_WARNING_KERNELS
    }
    if (
        determinism.get("warning_count") != len(raw_messages)
        or determinism.get("normalized_messages") != expected_normalized
        or determinism.get("normalization") != expected_records
        or determinism.get("whitelist") != list(DETERMINISM_WARNING_WHITELIST)
        or determinism.get("kernel_inventory") != list(DETERMINISM_WARNING_KERNELS)
        or determinism.get("kernel_counts") != expected_counts
    ):
        raise ValueError("V4 normalized determinism warning evidence changed")

    expected_families = {
        family for family, count in normalized_family_counts.items() if count > 0
    }
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, Mapping) or set(evaluation) != {
        "matched_rgb",
        "wrong_rgb_with_target_calibration",
    }:
        raise ValueError("V4 result evaluation controls changed")
    matched = evaluation["matched_rgb"]
    wrong = evaluation["wrong_rgb_with_target_calibration"]
    for name, value, expected_control in (
        ("matched", matched, "matched_rgb"),
        ("wrong RGB", wrong, "wrong_rgb_with_target_calibration"),
    ):
        if not isinstance(value, Mapping) or set(value) != {
            "control",
            "wrong_rgb_degenerate_singleton",
            "image_index_mapping",
            "image_mapping_sha256",
            "losses",
            "metrics",
        } or value.get("control") != expected_control or not _is_sha256(
            value.get("image_mapping_sha256")
        ):
            raise ValueError(f"V4 {name} evaluation receipt changed")
        mapping = value.get("image_index_mapping")
        expected_mapping = list(range(fit_size)) if expected_control == "matched_rgb" else [
            (index + 1) % fit_size for index in range(fit_size)
        ]
        if (
            mapping != expected_mapping
            or value.get("image_mapping_sha256") != canonical_json_sha256(mapping)
        ):
            raise ValueError(f"V4 {name} image assignment changed")
        _validate_evaluation_losses(value.get("losses"), name=name)
    if matched.get("wrong_rgb_degenerate_singleton") is not False or wrong.get(
        "wrong_rgb_degenerate_singleton"
    ) is not False:
        raise ValueError("V4 wrong-RGB degeneracy receipt changed")
    if fit_size > 1 and matched["image_mapping_sha256"] == wrong["image_mapping_sha256"]:
        raise ValueError("V4 wrong-RGB control did not change image assignments")
    matched_metrics = _validate_metric_bundle(
        matched["metrics"],
        fit_size=fit_size,
        expected_families=expected_families,
        name="matched RGB",
    )
    wrong_metrics = _validate_metric_bundle(
        wrong["metrics"],
        fit_size=fit_size,
        expected_families=expected_families,
        name="wrong RGB",
    )
    if _target_partition_signature(matched_metrics) != _target_partition_signature(
        wrong_metrics
    ):
        raise ValueError("V4 matched/wrong controls do not share targets")
    return {
        "seed": expected_seed,
        "fit_size": fit_size,
        "content_sha256": declared,
        "inputs": dict(inputs),
        "checkpoint": dict(checkpoint),
        "reservation": dict(reservation_binding),
        "target_partition": dict(target_partition),
        "subset_keys": list(keys),
        "matched": matched_metrics,
        "wrong": wrong_metrics,
    }


def _record_minimum(
    checks: list[dict[str, Any]],
    *,
    name: str,
    value: float,
    threshold: float,
) -> None:
    checks.append(
        {
            "name": name,
            "comparison": "greater_than_or_equal",
            "value": value,
            "threshold": threshold,
            "passes": value >= threshold,
        }
    )


def _record_maximum(
    checks: list[dict[str, Any]],
    *,
    name: str,
    value: float,
    threshold: float,
) -> None:
    checks.append(
        {
            "name": name,
            "comparison": "less_than_or_equal",
            "value": value,
            "threshold": threshold,
            "passes": value <= threshold,
        }
    )


def _gate_stage(validated: Mapping[str, Any]) -> dict[str, Any]:
    fit_size = int(validated["fit_size"])
    threshold = FIT_THRESHOLDS[fit_size]
    matched = validated["matched"]
    wrong = validated["wrong"]
    checks: list[dict[str, Any]] = []
    _record_minimum(
        checks,
        name="matched.pixel_hit_balanced_accuracy",
        value=matched["pixel_hit_no_hit"]["balanced_accuracy"],
        threshold=threshold.pixel_hit_balanced_accuracy_min,
    )
    _record_maximum(
        checks,
        name="matched.pixel_hit_depth_median_absolute_error_m",
        value=matched["pixel_hit_depth"]["median_absolute_error_m"],
        threshold=threshold.pixel_hit_depth_median_error_m_max,
    )
    _record_maximum(
        checks,
        name="matched.pixel_hit_depth_p95_absolute_error_m",
        value=matched["pixel_hit_depth"]["p95_absolute_error_m"],
        threshold=threshold.pixel_hit_depth_p95_error_m_max,
    )
    _record_minimum(
        checks,
        name="matched.ground_overall_balanced_accuracy",
        value=matched["ground_clear"]["overall"]["balanced_accuracy"],
        threshold=threshold.ground_overall_balanced_accuracy_min,
    )
    for group, metric in matched["ground_clear"]["by_distance_m"].items():
        if metric["count"]:
            _record_minimum(
                checks,
                name=f"matched.ground_distance.{group}.balanced_accuracy",
                value=metric["balanced_accuracy"],
                threshold=threshold.ground_distance_balanced_accuracy_min,
            )
    for family, metric in matched["ground_clear"]["by_family"].items():
        _record_minimum(
            checks,
            name=f"matched.ground_family.{family}.balanced_accuracy",
            value=metric["balanced_accuracy"],
            threshold=threshold.ground_family_balanced_accuracy_min,
        )
    _record_maximum(
        checks,
        name="matched.raster_nll",
        value=matched["derived_raster"]["nll"],
        threshold=threshold.raster_nll_max,
    )
    _record_minimum(
        checks,
        name="matched.raster_balanced_accuracy",
        value=matched["derived_raster"]["balanced_accuracy"],
        threshold=threshold.raster_balanced_accuracy_min,
    )
    for class_name, recall in matched["derived_raster"]["class_recalls"].items():
        if recall is None:
            continue
        _record_minimum(
            checks,
            name=f"matched.raster_recall.{class_name}",
            value=recall,
            threshold=threshold.raster_class_recall_min,
        )

    dependence_assessable = fit_size > 1
    if dependence_assessable:
        dependence_values = (
            (
                "wrong_rgb.pixel_balanced_accuracy_drop",
                matched["pixel_hit_no_hit"]["balanced_accuracy"]
                - wrong["pixel_hit_no_hit"]["balanced_accuracy"],
                threshold.wrong_pixel_balanced_accuracy_drop_min,
            ),
            (
                "wrong_rgb.depth_median_error_increase_m",
                wrong["pixel_hit_depth"]["median_absolute_error_m"]
                - matched["pixel_hit_depth"]["median_absolute_error_m"],
                threshold.wrong_depth_median_error_increase_m_min,
            ),
            (
                "wrong_rgb.depth_p95_error_increase_m",
                wrong["pixel_hit_depth"]["p95_absolute_error_m"]
                - matched["pixel_hit_depth"]["p95_absolute_error_m"],
                threshold.wrong_depth_p95_error_increase_m_min,
            ),
            (
                "wrong_rgb.ground_balanced_accuracy_drop",
                matched["ground_clear"]["overall"]["balanced_accuracy"]
                - wrong["ground_clear"]["overall"]["balanced_accuracy"],
                threshold.wrong_ground_balanced_accuracy_drop_min,
            ),
            (
                "wrong_rgb.raster_nll_increase",
                wrong["derived_raster"]["nll"]
                - matched["derived_raster"]["nll"],
                threshold.wrong_raster_nll_increase_min,
            ),
            (
                "wrong_rgb.raster_balanced_accuracy_drop",
                matched["derived_raster"]["balanced_accuracy"]
                - wrong["derived_raster"]["balanced_accuracy"],
                threshold.wrong_raster_balanced_accuracy_drop_min,
            ),
        )
        for name, value, minimum in dependence_values:
            if minimum is None:
                raise AssertionError("wrong-RGB threshold missing for assessable stage")
            _record_minimum(checks, name=name, value=value, threshold=minimum)
    failures = [check for check in checks if not check["passes"]]
    return {
        "fit_size": fit_size,
        "thresholds": asdict(threshold),
        "wrong_rgb_dependence_assessable": dependence_assessable,
        "check_count": len(checks),
        "checks": checks,
        "failure_count": len(failures),
        "failed_checks": failures,
        "passes": not failures,
    }


def _gate_file_sha256(gate: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(gate) + b"\n").hexdigest()


def validate_metric_verification_receipt_v4(
    receipt: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_fit_size: int,
    expected_result_content_sha256: str,
    expected_checkpoint: Mapping[str, Any],
    expected_evaluation: Mapping[str, Any],
    expected_target_partition: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a separately licensed full-inference metric rerun receipt."""

    fields = {
        "schema",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "dataset_role",
        "seed",
        "fit_size",
        "result_content_sha256",
        "checkpoint",
        "metric_verifier_authorization",
        "target_partition_reproduction",
        "target_partition_signature",
        "target_partition_signature_sha256",
        "recomputed_evaluation",
        "recomputed_evaluation_sha256",
        "numeric_gate",
        "verification",
        "licenses",
    }
    _validate_gate_content(
        receipt,
        schema=METRIC_VERIFICATION_SCHEMA,
        name="V4 metric verification",
        expected_fields=fields,
    )
    checkpoint = receipt.get("checkpoint")
    authorization = receipt.get("metric_verifier_authorization")
    evaluation = receipt.get("recomputed_evaluation")
    matched, wrong, signature = _validated_metric_evaluation(
        evaluation,
        fit_size=expected_fit_size,
    )
    numeric = _gate_stage(
        {"fit_size": expected_fit_size, "matched": matched, "wrong": wrong}
    )
    expected_checkpoint_fields = {
        "file_sha256",
        "content_sha256",
        "byte_count",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_checkpoint_fields:
        raise ValueError("V4 metric checkpoint binding changed")
    expected_checkpoint_core = {
        key: expected_checkpoint[key] for key in expected_checkpoint_fields
    }
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "file_sha256",
        "content_sha256",
    } or not all(_is_sha256(value) for value in authorization.values()):
        raise ValueError("V4 metric verifier authorization binding changed")
    verification = receipt.get("verification")
    licenses = receipt.get("licenses")
    expected_partition = validate_target_partition_binding_v4(
        expected_target_partition,
        fit_size=expected_fit_size,
    )
    partition_reproduction = receipt.get("target_partition_reproduction")
    if (
        receipt.get("authoritative") is not False
        or receipt.get("aggregation_eligible") is not False
        or receipt.get("promotion_eligible") is not False
        or receipt.get("dataset_role") != "train"
        or receipt.get("seed") != expected_seed
        or receipt.get("fit_size") != expected_fit_size
        or receipt.get("result_content_sha256") != expected_result_content_sha256
        or checkpoint != expected_checkpoint_core
        or evaluation != expected_evaluation
        or receipt.get("recomputed_evaluation_sha256")
        != canonical_json_sha256(evaluation)
        or receipt.get("target_partition_signature") != signature
        or receipt.get("target_partition_signature_sha256")
        != canonical_json_sha256(signature)
        or partition_reproduction
        != {
            "target_partition": expected_partition,
            "reproduced_before_checkpoint_inference": True,
        }
        or receipt.get("numeric_gate") != numeric
        or verification
        != {
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
        }
        or licenses
        != {
            "checkpoint_use_authorized_for_metric_verification_only": True,
            "authorizes_development_checkpoint_use": False,
            "authorizes_new_model_output": False,
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        }
    ):
        raise PermissionError("V4 metric verification provenance/scope changed")
    return dict(receipt)


def _validate_gate_content(
    gate: Mapping[str, Any],
    *,
    schema: str,
    name: str,
    expected_fields: set[str],
) -> None:
    if set(gate) != expected_fields | {"content_sha256"} or gate.get("schema") != schema:
        raise ValueError(f"{name} schema changed")
    core = dict(gate)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or declared != canonical_json_sha256(core):
        raise ValueError(f"{name} content hash changed")


def _validate_numeric_gate(value: object, *, fit_size: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "fit_size",
        "thresholds",
        "wrong_rgb_dependence_assessable",
        "check_count",
        "checks",
        "failure_count",
        "failed_checks",
        "passes",
    } or (
        value.get("fit_size") != fit_size
        or value.get("thresholds") != asdict(FIT_THRESHOLDS[fit_size])
        or value.get("wrong_rgb_dependence_assessable") is not (fit_size > 1)
    ):
        raise ValueError("V4 stage numeric-gate contract changed")
    checks = value.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("V4 stage numeric checks are malformed")
    names = []
    normalized = []
    for check in checks:
        if not isinstance(check, Mapping) or set(check) != {
            "name",
            "comparison",
            "value",
            "threshold",
            "passes",
        }:
            raise ValueError("one V4 stage numeric check is malformed")
        name = check.get("name")
        comparison = check.get("comparison")
        observed = _finite_number(check.get("value"), name="V4 gate observed value")
        threshold = _finite_number(
            check.get("threshold"), name="V4 gate threshold value"
        )
        if not isinstance(name, str) or not name or comparison not in {
            "greater_than_or_equal",
            "less_than_or_equal",
        }:
            raise ValueError("one V4 stage numeric check identity changed")
        expected_pass = (
            observed >= threshold
            if comparison == "greater_than_or_equal"
            else observed <= threshold
        )
        if check.get("passes") is not expected_pass:
            raise ValueError("one V4 stage numeric verdict is inconsistent")
        names.append(name)
        normalized.append(dict(check))
    if len(names) != len(set(names)):
        raise ValueError("V4 stage numeric checks repeat a name")
    failures = [check for check in normalized if not check["passes"]]
    passes = not failures
    if (
        value.get("check_count") != len(normalized)
        or value.get("failure_count") != len(failures)
        or value.get("failed_checks") != failures
        or value.get("passes") is not passes
    ):
        raise ValueError("V4 stage numeric summary is inconsistent")
    return passes


def _validate_file_binding(
    value: object,
    *,
    path: str,
    name: str,
    require_byte_count: bool = False,
) -> dict[str, Any]:
    expected = {"path", "file_sha256", "content_sha256"}
    if require_byte_count:
        expected.add("byte_count")
    if not isinstance(value, Mapping) or set(value) != expected or (
        value.get("path") != path
        or not _is_sha256(value.get("file_sha256"))
        or not _is_sha256(value.get("content_sha256"))
        or (
            require_byte_count
            and (
                not isinstance(value.get("byte_count"), int)
                or isinstance(value.get("byte_count"), bool)
                or value["byte_count"] <= 0
            )
        )
    ):
        raise ValueError(f"{name} binding changed")
    return dict(value)


def _validate_stage_artifacts(
    value: object,
    *,
    seed: int,
    fit_size: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "attempt_directory",
        "reservation",
        "result",
        "checkpoint",
        "completion",
        "metric_verification",
    } or value.get("attempt_directory") != f"attempts/seed_{seed}/n{fit_size}":
        raise ValueError("V4 stage canonical attempt binding changed")
    return {
        "attempt_directory": value["attempt_directory"],
        "reservation": _validate_file_binding(
            value["reservation"], path="reservation.json", name="V4 reservation"
        ),
        "result": _validate_file_binding(
            value["result"], path="result.json", name="V4 result"
        ),
        "checkpoint": _validate_file_binding(
            value["checkpoint"],
            path="checkpoint.pt",
            name="V4 checkpoint",
            require_byte_count=True,
        ),
        "completion": _validate_file_binding(
            value["completion"], path="completed.json", name="V4 completion"
        ),
        "metric_verification": _validate_file_binding(
            value["metric_verification"],
            path=f"metric_verifications/seed_{seed}_n{fit_size}.json",
            name="V4 metric verification",
        ),
    }


def _validate_stage_binding_record(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "file_sha256",
        "content_sha256",
        "seed",
        "fit_size",
        "passes",
        "development_checkpoint_use_authorized",
        "next_rung_execution_authorized",
    } or (
        value.get("schema") != STAGE_GATE_SCHEMA
        or not _is_sha256(value.get("file_sha256"))
        or not _is_sha256(value.get("content_sha256"))
        or value.get("seed") not in EXPECTED_SEEDS
        or value.get("fit_size") not in LADDER_FIT_SIZES
        or not isinstance(value.get("passes"), bool)
        or value.get("development_checkpoint_use_authorized") is not False
        or not isinstance(value.get("next_rung_execution_authorized"), bool)
    ):
        raise ValueError(f"{name} binding changed")
    return dict(value)


def _validate_seed_binding_record(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "file_sha256",
        "content_sha256",
        "seed",
        "ladder_passes",
        "seed_20260711_n5_execution_authorized",
    } or (
        value.get("schema") != SEED_GATE_SCHEMA
        or not _is_sha256(value.get("file_sha256"))
        or not _is_sha256(value.get("content_sha256"))
        or value.get("seed") not in EXPECTED_SEEDS
        or not isinstance(value.get("ladder_passes"), bool)
        or not isinstance(value.get("seed_20260711_n5_execution_authorized"), bool)
    ):
        raise ValueError(f"{name} binding changed")
    return dict(value)


def _validate_stage_gate_schema(gate: Mapping[str, Any]) -> None:
    fields = {
        "schema",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "seed",
        "fit_size",
        "input_result_content_sha256",
        "artifacts",
        "reviewed_inputs",
        "subset",
        "target_partition",
        "previous_stage_gate",
        "seed_20260710_gate",
        "checkpoint_created",
        "checkpoint_file_sha256",
        "development_checkpoint_use_authorized",
        "next_fit_size",
        "next_rung_execution_authorized",
        "numeric_gate",
        "passes",
        "licenses",
    }
    _validate_gate_content(
        gate,
        schema=STAGE_GATE_SCHEMA,
        name="V4 stage gate",
        expected_fields=fields,
    )
    seed = gate.get("seed")
    fit_size = gate.get("fit_size")
    if seed not in EXPECTED_SEEDS or fit_size not in LADDER_FIT_SIZES:
        raise ValueError("V4 stage gate seed/rung changed")
    stage_index = LADDER_FIT_SIZES.index(fit_size)
    next_fit_size = (
        None
        if stage_index + 1 == len(LADDER_FIT_SIZES)
        else LADDER_FIT_SIZES[stage_index + 1]
    )
    passes = _validate_numeric_gate(gate.get("numeric_gate"), fit_size=fit_size)
    artifacts = _validate_stage_artifacts(
        gate.get("artifacts"), seed=seed, fit_size=fit_size
    )
    reviewed = gate.get("reviewed_inputs")
    if not isinstance(reviewed, Mapping) or set(reviewed) != {
        "trainer_authorization_file_sha256",
        "trainer_authorization_content_sha256",
        "trainer_review_record_file_sha256",
        "trainer_review_record_content_sha256",
        "rgb_receipt_content_sha256",
        "metric_verifier_authorization_file_sha256",
        "metric_verifier_authorization_content_sha256",
    } or (
        not all(_is_sha256(item) for item in reviewed.values())
        or reviewed.get("rgb_receipt_content_sha256") != RGB_RECEIPT_CONTENT_SHA256
    ):
        raise ValueError("V4 stage reviewed input binding changed")
    subset = gate.get("subset")
    if not isinstance(subset, Mapping) or set(subset) != {
        "namespace",
        "parent_frame_count",
        "fit_size",
        "selection",
        "family_counts",
        "ordered_frame_key_sha256",
        "content_sha256",
    } or (
        subset.get("fit_size") != fit_size
        or subset.get("content_sha256") != EXPECTED_SUBSET_CONTENT_SHA256[fit_size]
        or not isinstance(subset.get("ordered_frame_key_sha256"), list)
        or len(subset["ordered_frame_key_sha256"]) != fit_size
        or not all(_is_sha256(item) for item in subset["ordered_frame_key_sha256"])
        or subset.get("content_sha256")
        != canonical_json_sha256(subset["ordered_frame_key_sha256"])
    ):
        raise ValueError("V4 stage subset binding changed")
    target_partition = validate_target_partition_binding_v4(
        gate.get("target_partition"),
        fit_size=fit_size,
    )
    if target_partition["subset_content_sha256"] != subset["content_sha256"]:
        raise ValueError("V4 stage target-partition subset binding changed")
    next_authorized = bool(passes and next_fit_size is not None)
    previous = gate.get("previous_stage_gate")
    if stage_index == 0:
        if previous is not None:
            raise ValueError("V4 N5 stage gate may not bind a predecessor")
    else:
        previous = _validate_stage_binding_record(
            previous, name="V4 previous-stage"
        )
        if (
            previous["seed"] != seed
            or previous["fit_size"] != LADDER_FIT_SIZES[stage_index - 1]
            or previous["passes"] is not True
            or previous["next_rung_execution_authorized"] is not True
        ):
            raise PermissionError("V4 stage gate predecessor chain changed")
    first_seed = gate.get("seed_20260710_gate")
    if seed == 20260710:
        if first_seed is not None:
            raise ValueError("first-seed V4 stage may not bind itself")
    else:
        first_seed = _validate_seed_binding_record(
            first_seed, name="V4 first-seed"
        )
        if (
            first_seed["seed"] != 20260710
            or first_seed["ladder_passes"] is not True
            or first_seed["seed_20260711_n5_execution_authorized"] is not True
        ):
            raise PermissionError("second-seed V4 stage lost its seed prerequisite")
    if (
        gate.get("authoritative") is not False
        or gate.get("aggregation_eligible") is not False
        or gate.get("promotion_eligible") is not False
        or not _is_sha256(gate.get("input_result_content_sha256"))
        or gate.get("checkpoint_created") is not True
        or not _is_sha256(gate.get("checkpoint_file_sha256"))
        or gate.get("checkpoint_file_sha256")
        != artifacts["checkpoint"]["file_sha256"]
        or gate.get("input_result_content_sha256")
        != artifacts["result"]["content_sha256"]
        or gate.get("development_checkpoint_use_authorized") is not False
        or gate.get("next_fit_size") != next_fit_size
        or gate.get("next_rung_execution_authorized") is not next_authorized
        or gate.get("passes") is not passes
        or gate.get("licenses")
        != {
            "development_checkpoint_use_authorized": False,
            "next_rung_execution_authorized": next_authorized,
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        }
    ):
        raise PermissionError("V4 stage gate license/provenance contract changed")


def _stage_gate_binding(gate: Mapping[str, Any]) -> dict[str, Any]:
    _validate_stage_gate_schema(gate)
    return {
        "schema": STAGE_GATE_SCHEMA,
        "file_sha256": _gate_file_sha256(gate),
        "content_sha256": gate["content_sha256"],
        "seed": gate["seed"],
        "fit_size": gate["fit_size"],
        "passes": gate["passes"],
        "development_checkpoint_use_authorized": gate[
            "development_checkpoint_use_authorized"
        ],
        "next_rung_execution_authorized": gate["next_rung_execution_authorized"],
    }


def _validate_stage_execution_fields(
    gate: Mapping[str, Any],
    *,
    gate_file_sha256: str,
    expected_seed: int,
    expected_next_fit_size: int,
) -> dict[str, Any]:
    binding = _stage_gate_binding(gate)
    expected_previous_index = LADDER_FIT_SIZES.index(expected_next_fit_size) - 1
    if expected_previous_index < 0:
        raise ValueError("N5 cannot consume a previous-stage gate")
    if gate_file_sha256 != binding["file_sha256"]:
        raise ValueError("caller V4 stage-gate file hash changed")
    if (
        binding["seed"] != expected_seed
        or binding["fit_size"] != LADDER_FIT_SIZES[expected_previous_index]
        or binding["passes"] is not True
        or binding["next_rung_execution_authorized"] is not True
        or gate.get("next_fit_size") != expected_next_fit_size
    ):
        raise PermissionError("V4 stage gate does not authorize this immediate rung")
    return binding


def _optimizer_exposure_contract(seed: int) -> dict[str, Any]:
    return {
        "steps": {str(size): DEFAULT_STEPS[size] for size in LADDER_FIT_SIZES},
        "train_batch_size": TRAIN_BATCH_SIZE,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "schedule_algorithm": SCHEDULE_ALGORITHM,
        "schedule_sha256": {
            str(size): EXPECTED_SCHEDULE_SHA256[seed][size]
            for size in LADDER_FIT_SIZES
        },
    }


def _validate_seed_gate_schema(gate: Mapping[str, Any]) -> None:
    fields = {
        "schema",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "seed",
        "dataset_manifest_file_sha256",
        "dataset_manifest_content_sha256",
        "audit_receipt_file_sha256",
        "audit_receipt_content_sha256",
        "fit_sizes",
        "optimizer_exposure_contract",
        "threshold_contract_sha256",
        "stage_gate_file_sha256",
        "stage_gate_content_sha256",
        "reviewed_inputs",
        "seed_20260710_gate",
        "failure_count",
        "ladder_passes",
        "seed_20260711_n5_execution_authorized",
        "licenses",
    }
    _validate_gate_content(
        gate,
        schema=SEED_GATE_SCHEMA,
        name="V4 seed gate",
        expected_fields=fields,
    )
    seed = gate.get("seed")
    threshold_payload = {
        str(size): asdict(FIT_THRESHOLDS[size]) for size in LADDER_FIT_SIZES
    }
    stage_files = gate.get("stage_gate_file_sha256")
    stage_contents = gate.get("stage_gate_content_sha256")
    reviewed_inputs = gate.get("reviewed_inputs")
    if (
        seed not in EXPECTED_SEEDS
        or gate.get("authoritative") is not False
        or gate.get("aggregation_eligible") is not False
        or gate.get("promotion_eligible") is not False
        or gate.get("dataset_manifest_file_sha256") != DATASET_MANIFEST_FILE_SHA256
        or gate.get("dataset_manifest_content_sha256") != DATASET_MANIFEST_CONTENT_SHA256
        or gate.get("audit_receipt_file_sha256") != AUDIT_RECEIPT_FILE_SHA256
        or gate.get("audit_receipt_content_sha256") != AUDIT_RECEIPT_CONTENT_SHA256
        or gate.get("fit_sizes") != list(LADDER_FIT_SIZES)
        or gate.get("optimizer_exposure_contract") != _optimizer_exposure_contract(seed)
        or gate.get("threshold_contract_sha256")
        != canonical_json_sha256(threshold_payload)
        or not isinstance(stage_files, list)
        or len(stage_files) != len(LADDER_FIT_SIZES)
        or not all(_is_sha256(item) for item in stage_files)
        or len(set(stage_files)) != len(stage_files)
        or not isinstance(stage_contents, list)
        or len(stage_contents) != len(LADDER_FIT_SIZES)
        or not all(_is_sha256(item) for item in stage_contents)
        or len(set(stage_contents)) != len(stage_contents)
        or not isinstance(reviewed_inputs, Mapping)
        or set(reviewed_inputs)
        != {
            "trainer_authorization_file_sha256",
            "trainer_authorization_content_sha256",
            "trainer_review_record_file_sha256",
            "trainer_review_record_content_sha256",
            "rgb_receipt_content_sha256",
            "metric_verifier_authorization_file_sha256",
            "metric_verifier_authorization_content_sha256",
        }
        or not all(_is_sha256(item) for item in reviewed_inputs.values())
        or reviewed_inputs.get("rgb_receipt_content_sha256")
        != RGB_RECEIPT_CONTENT_SHA256
    ):
        raise ValueError("V4 seed gate frozen ladder contract changed")
    ladder_passes = gate.get("ladder_passes")
    if not isinstance(ladder_passes, bool) or gate.get("failure_count") not in {
        0,
        1,
        2,
        3,
        4,
        5,
    } or ladder_passes is not (gate.get("failure_count") == 0):
        raise ValueError("V4 seed gate failure summary changed")
    first_seed_binding = gate.get("seed_20260710_gate")
    if seed == 20260710:
        if first_seed_binding is not None:
            raise ValueError("first V4 seed gate may not bind itself")
    elif not isinstance(first_seed_binding, Mapping) or set(first_seed_binding) != {
        "schema",
        "file_sha256",
        "content_sha256",
        "seed",
        "ladder_passes",
        "seed_20260711_n5_execution_authorized",
    } or (
        first_seed_binding.get("schema") != SEED_GATE_SCHEMA
        or first_seed_binding.get("seed") != 20260710
        or first_seed_binding.get("ladder_passes") is not True
        or first_seed_binding.get("seed_20260711_n5_execution_authorized") is not True
        or not _is_sha256(first_seed_binding.get("file_sha256"))
        or not _is_sha256(first_seed_binding.get("content_sha256"))
    ):
        raise PermissionError("second V4 seed gate lost its first-seed binding")
    seed2_authorized = bool(seed == 20260710 and ladder_passes)
    if (
        gate.get("seed_20260711_n5_execution_authorized") is not seed2_authorized
        or gate.get("licenses")
        != {
            "authorizes_development_checkpoint_creation": False,
            "authorizes_checkpoint_use": False,
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        }
    ):
        raise PermissionError("V4 seed gate licenses changed")


def _seed_gate_binding(gate: Mapping[str, Any]) -> dict[str, Any]:
    _validate_seed_gate_schema(gate)
    return {
        "schema": SEED_GATE_SCHEMA,
        "file_sha256": _gate_file_sha256(gate),
        "content_sha256": gate["content_sha256"],
        "seed": gate["seed"],
        "ladder_passes": gate["ladder_passes"],
        "seed_20260711_n5_execution_authorized": gate[
            "seed_20260711_n5_execution_authorized"
        ],
    }


def _validate_seed_execution_fields(
    gate: Mapping[str, Any],
    *,
    gate_file_sha256: str,
) -> dict[str, Any]:
    binding = _seed_gate_binding(gate)
    if gate_file_sha256 != binding["file_sha256"]:
        raise ValueError("caller V4 seed-gate file hash changed")
    if (
        binding["seed"] != 20260710
        or binding["ladder_passes"] is not True
        or binding["seed_20260711_n5_execution_authorized"] is not True
    ):
        raise PermissionError("V4 first-seed gate does not authorize seed 20260711")
    return binding


def finalize_development_fit_stage_v4(
    result: Mapping[str, Any],
    *,
    expected_seed: int,
    artifact_binding: Mapping[str, Any],
    metric_verification_receipt: Mapping[str, Any],
    previous_stage_gate: Mapping[str, Any] | None = None,
    seed_20260710_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize one rung; creation and later checkpoint use remain distinct."""

    previous_binding = (
        None if previous_stage_gate is None else _stage_gate_binding(previous_stage_gate)
    )
    seed_binding = (
        None if seed_20260710_gate is None else _seed_gate_binding(seed_20260710_gate)
    )
    if previous_binding is not None and (
        previous_binding["passes"] is not True
        or previous_binding["next_rung_execution_authorized"] is not True
        or previous_binding["seed"] != expected_seed
    ):
        raise PermissionError("previous V4 stage did not license the larger rung")
    if expected_seed == 20260711 and (
        seed_binding is None
        or seed_binding["seed"] != 20260710
        or seed_binding["ladder_passes"] is not True
        or seed_binding["seed_20260711_n5_execution_authorized"] is not True
    ):
        raise PermissionError("seed 20260711 is forbidden before seed 20260710 passes")
    validated = _validate_result(
        result,
        expected_seed=expected_seed,
        previous_stage_binding=previous_binding,
        seed_20260710_binding=seed_binding,
    )
    numeric = _gate_stage(validated)
    fit_size = int(validated["fit_size"])
    metric_verification = validate_metric_verification_receipt_v4(
        metric_verification_receipt,
        expected_seed=expected_seed,
        expected_fit_size=fit_size,
        expected_result_content_sha256=validated["content_sha256"],
        expected_checkpoint=validated["checkpoint"],
        expected_evaluation=result["evaluation"],
        expected_target_partition=result["target_partition"],
    )
    artifacts = _validate_stage_artifacts(
        artifact_binding,
        seed=expected_seed,
        fit_size=fit_size,
    )
    if (
        artifacts["reservation"] != validated["reservation"]
        or artifacts["result"]["content_sha256"] != validated["content_sha256"]
        or artifacts["checkpoint"]["file_sha256"]
        != validated["checkpoint"]["file_sha256"]
        or artifacts["checkpoint"]["content_sha256"]
        != validated["checkpoint"]["content_sha256"]
        or artifacts["checkpoint"]["byte_count"]
        != validated["checkpoint"]["byte_count"]
        or artifacts["metric_verification"]["content_sha256"]
        != metric_verification["content_sha256"]
    ):
        raise PermissionError("V4 stage artifacts do not bind the validated result")
    stage_index = LADDER_FIT_SIZES.index(fit_size)
    if stage_index == 0:
        if previous_binding is not None:
            raise ValueError("the first V4 rung may not bind a predecessor")
    elif (
        previous_binding is None
        or previous_binding["fit_size"] != LADDER_FIT_SIZES[stage_index - 1]
    ):
        raise PermissionError("V4 rung did not bind its immediate predecessor")
    next_fit_size = (
        None
        if stage_index + 1 == len(LADDER_FIT_SIZES)
        else LADDER_FIT_SIZES[stage_index + 1]
    )
    passes = bool(numeric["passes"])
    core = {
        "schema": STAGE_GATE_SCHEMA,
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "seed": expected_seed,
        "fit_size": fit_size,
        "input_result_content_sha256": validated["content_sha256"],
        "artifacts": artifacts,
        "reviewed_inputs": {
            key: validated["inputs"][key]
            for key in (
                "trainer_authorization_file_sha256",
                "trainer_authorization_content_sha256",
                "trainer_review_record_file_sha256",
                "trainer_review_record_content_sha256",
                "rgb_receipt_content_sha256",
            )
        }
        | {
            "metric_verifier_authorization_file_sha256": metric_verification[
                "metric_verifier_authorization"
            ]["file_sha256"],
            "metric_verifier_authorization_content_sha256": metric_verification[
                "metric_verifier_authorization"
            ]["content_sha256"],
        },
        "subset": dict(result["subset"]),
        "target_partition": dict(validated["target_partition"]),
        "previous_stage_gate": previous_binding,
        "seed_20260710_gate": seed_binding,
        "checkpoint_created": True,
        "checkpoint_file_sha256": validated["checkpoint"]["file_sha256"],
        "development_checkpoint_use_authorized": False,
        "next_fit_size": next_fit_size,
        "next_rung_execution_authorized": bool(passes and next_fit_size is not None),
        "numeric_gate": numeric,
        "passes": passes,
        "licenses": {
            "development_checkpoint_use_authorized": False,
            "next_rung_execution_authorized": bool(passes and next_fit_size is not None),
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def finalize_development_fit_seed_v4(
    stage_gates: Sequence[Mapping[str, Any]],
    *,
    expected_seed: int,
    seed_20260710_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Finalize the four canonical passing stage gates for one frozen seed."""

    if not isinstance(stage_gates, Sequence) or isinstance(stage_gates, (str, bytes)):
        raise TypeError("V4 seed stage gates must be a sequence")
    if len(stage_gates) != len(LADDER_FIT_SIZES):
        raise ValueError("V4 seed ladder requires N=5/16/32/320 stage gates")
    expected_first_binding = (
        None if seed_20260710_gate is None else _seed_gate_binding(seed_20260710_gate)
    )
    if expected_seed == 20260710:
        if expected_first_binding is not None:
            raise ValueError("first V4 seed ladder may not bind itself")
    elif (
        expected_seed != 20260711
        or expected_first_binding is None
        or expected_first_binding["seed"] != 20260710
        or expected_first_binding["ladder_passes"] is not True
        or expected_first_binding["seed_20260711_n5_execution_authorized"] is not True
    ):
        raise PermissionError("second V4 seed ladder lacks the completed first seed")
    bindings = []
    previous_binding = None
    previous_keys: list[str] | None = None
    common_reviewed_inputs = None
    for expected_size, stage in zip(LADDER_FIT_SIZES, stage_gates):
        binding = _stage_gate_binding(stage)
        if (
            binding["seed"] != expected_seed
            or binding["fit_size"] != expected_size
            or binding["passes"] is not True
            or stage.get("previous_stage_gate") != previous_binding
            or stage.get("seed_20260710_gate") != expected_first_binding
        ):
            raise PermissionError("V4 seed ladder stage chain changed")
        keys = list(stage["subset"]["ordered_frame_key_sha256"])
        if previous_keys is not None and keys[: len(previous_keys)] != previous_keys:
            raise ValueError("V4 ladder fit subsets are not nested")
        previous_keys = keys
        reviewed_inputs = dict(stage["reviewed_inputs"])
        if common_reviewed_inputs is None:
            common_reviewed_inputs = reviewed_inputs
        elif reviewed_inputs != common_reviewed_inputs:
            raise ValueError("V4 seed ladder changed reviewed input bindings")
        bindings.append(binding)
        previous_binding = binding
    threshold_payload = {
        str(size): asdict(FIT_THRESHOLDS[size]) for size in LADDER_FIT_SIZES
    }
    core = {
        "schema": SEED_GATE_SCHEMA,
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "seed": expected_seed,
        "dataset_manifest_file_sha256": DATASET_MANIFEST_FILE_SHA256,
        "dataset_manifest_content_sha256": DATASET_MANIFEST_CONTENT_SHA256,
        "audit_receipt_file_sha256": AUDIT_RECEIPT_FILE_SHA256,
        "audit_receipt_content_sha256": AUDIT_RECEIPT_CONTENT_SHA256,
        "fit_sizes": list(LADDER_FIT_SIZES),
        "optimizer_exposure_contract": _optimizer_exposure_contract(expected_seed),
        "threshold_contract_sha256": canonical_json_sha256(threshold_payload),
        "stage_gate_file_sha256": [binding["file_sha256"] for binding in bindings],
        "stage_gate_content_sha256": [
            binding["content_sha256"] for binding in bindings
        ],
        "reviewed_inputs": dict(common_reviewed_inputs),
        "seed_20260710_gate": expected_first_binding,
        "failure_count": 0,
        "ladder_passes": True,
        "seed_20260711_n5_execution_authorized": bool(
            expected_seed == 20260710
        ),
        "licenses": {
            "authorizes_development_checkpoint_creation": False,
            "authorizes_checkpoint_use": False,
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def finalize_development_fit_two_seed_v4(
    seed_20260710_gate: Mapping[str, Any],
    seed_20260711_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize both seeds only after their sequential seed gates exist."""

    first = _seed_gate_binding(seed_20260710_gate)
    second = _seed_gate_binding(seed_20260711_gate)
    if first["seed"] != 20260710 or first["ladder_passes"] is not True:
        raise PermissionError("seed 20260710 did not pass before seed 20260711")
    if second["seed"] != 20260711 or second["ladder_passes"] is not True:
        raise PermissionError("seed 20260711 ladder did not pass")
    if seed_20260711_gate.get("seed_20260710_gate") != first:
        raise PermissionError("seed 20260711 gate does not persist the first-seed binding")
    core = {
        "schema": TWO_SEED_GATE_SCHEMA,
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "seeds": list(EXPECTED_SEEDS),
        "seed_gate_file_sha256": [first["file_sha256"], second["file_sha256"]],
        "seed_gate_content_sha256": [
            first["content_sha256"],
            second["content_sha256"],
        ],
        "both_seed_ladders_pass": True,
        "licenses": {
            "authorizes_new_model_output": False,
            "authorizes_checkpoint_use": False,
            "authorizes_holdout": False,
            "authorizes_g2": False,
            "authorizes_runtime": False,
            "authorizes_promotion": False,
        },
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


__all__ = [
    "AUDIT_RECEIPT_CONTENT_SHA256",
    "AUDIT_RECEIPT_FILE_SHA256",
    "ATTEMPT_COMPLETION_SCHEMA",
    "ATTEMPT_RESERVATION_SCHEMA",
    "DATASET_MANIFEST_CONTENT_SHA256",
    "DATASET_MANIFEST_FILE_SHA256",
    "DETERMINISM_WARNING_KERNELS",
    "DETERMINISM_WARNING_WHITELIST",
    "DEFAULT_STEPS",
    "EVALUATION_BATCH_SIZE",
    "EXPECTED_RASTER_CLASS_COUNTS",
    "EXPECTED_FAMILY_COUNTS",
    "EXPECTED_FIRST_FRAME_KEY_SHA256",
    "EXPECTED_LAST_FRAME_KEY_SHA256",
    "EXPECTED_ORDERED_PER_FRAME_TARGET_SHA256",
    "EXPECTED_ORDERED_TARGET_BYTES_SHA256",
    "EXPECTED_TARGET_PARTITION_SIGNATURES",
    "EXPECTED_TARGET_PARTITION_SIGNATURE_SHA256",
    "EXPECTED_TRAINER_SOURCE_REHASHES",
    "EXPECTED_SUBSET_CONTENT_SHA256",
    "EXPECTED_SCHEDULE_SHA256",
    "EXPECTED_SEEDS",
    "FIT_THRESHOLDS",
    "LADDER_CONTRACT",
    "LADDER_FIT_SIZES",
    "LADDER_V3_AMENDMENT_FILE_SHA256",
    "LEARNING_RATE",
    "MODEL_PARAMETER_COUNT",
    "METRIC_VERIFICATION_SCHEMA",
    "RGB_RECEIPT_CONTENT_SHA256",
    "SCHEDULE_ALGORITHM",
    "SEED_GATE_SCHEMA",
    "STAGE_GATE_SCHEMA",
    "TARGET_PARTITION_AMENDMENT_FILE_SHA256",
    "TARGET_PARTITION_FREEZE_CONTENT_SHA256",
    "TARGET_PARTITION_FREEZE_FILE_SHA256",
    "TARGET_PARTITION_VERIFIER_FILE_SHA256",
    "TRAIN_BATCH_SIZE",
    "TWO_SEED_GATE_SCHEMA",
    "V1_FAILURE_LINEAGE",
    "WEIGHT_DECAY",
    "FitThresholdV4",
    "canonical_json_sha256",
    "finalize_development_fit_seed_v4",
    "finalize_development_fit_stage_v4",
    "finalize_development_fit_two_seed_v4",
    "target_partition_binding_v4",
    "validate_metric_verification_receipt_v4",
    "validate_target_partition_binding_v4",
]
