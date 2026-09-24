#!/usr/bin/env python3
"""Receipt-only checker for the existing-pool three-arm experiment.

The checker opens only ``result.json``, the two JSON audit receipts, and the
24 explicitly bound JSON measurement receipts.  Existing-pool leaves, the six
directly bound fresh-pack payloads, RGB, and model snapshots are identity-only
values: their paths and sizes are checked with lstat, but their bytes are never
opened by this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping, Sequence


RESULT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "result_v1"
)
METRICS_SCHEMA = "lewm_go2_world_model_existing_pool_three_arm_metrics_v1"
OVERLAP_AUDIT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_overlap_audit_v1"
)
SHUFFLE_AUDIT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_candidate_action_derangement_v1"
)
REPORT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "receipt_check_v1"
)
RESULT_STATUS = "COMPLETE_PENDING_TERMINAL_REVIEW"
ATTEMPT_ID = (
    "world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1"
)
ARM_NAMES = ("conditioned", "blind", "shuffled")
MEASUREMENT_UPDATES = tuple(range(0, 701, 100))
REGISTERED_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM = (
    "python_random_mt19937_getrandbits52_open01_neg_log1p_shared_family_scene_weights_v1"
)
ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION = (
    "bayesian_positive_weight_cluster_5th_percentile_not_frequentist_coverage"
)
ACTION_IDENTIFICATION_BOOTSTRAP_SEED = 20_260_803
ACTION_IDENTIFICATION_BOOTSTRAP_REPLICATES = 10_000
ACTION_IDENTIFICATION_BOOTSTRAP_LOWER_INDEX = 500
ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES = 2
EXPECTED_MISSING_ORDERED_TRAIN_TRIPLES = (
    (0, 4, 3),
    (5, 2, 3),
    (5, 7, 8),
    (6, 2, 1),
    (7, 3, 5),
    (7, 6, 0),
    (8, 7, 0),
)
OVERLAP_GATE_SCOPE = (
    "role_scene_disjointness_and_visible_action_and_adjacent_pair_support_only; "
    "triple_support_entropy_and_mutual_information_are_diagnostic"
)
OVERLAP_AUDIT_KEYS = (
    "schema",
    "status",
    "passed",
    "row_count",
    "role_row_counts",
    "role_scene_counts",
    "checks",
    "failed_checks",
    "diagnostic_checks",
    "failed_diagnostic_checks",
    "role_scene_overlap_count",
    "role_scene_overlap",
    "train_support",
    "entropy",
    "mutual_information_bits",
    "scene_diagnostics",
    "gate_scope",
)
ACTION_IDENTIFICATION_RECEIPT_KEYS = (
    "bootstrap_algorithm",
    "bootstrap_interpretation",
    "bootstrap_seed",
    "bootstrap_replicates",
    "bootstrap_lower_index",
    "family_action_supporting_scene_counts",
    "minimum_family_action_supporting_scene_count",
    "balanced_accuracy",
    "balanced_accuracy_one_sided_95_lower_bound",
    "balanced_chance",
    "exact_tie_count",
    "exact_tie_rate",
    "unique_winner_count",
    "unique_winner_accuracy",
    "hardest_wrong_action_margin",
    "hardest_wrong_action_margin_one_sided_95_lower_bound",
)
PACK_ARTIFACT_RELATIVE_PATHS = {
    "train": {
        "frames": "pack/train_frames.u8",
        "actions": "pack/train_actions.npy",
        "metadata": "pack/train_meta.json",
    },
    "val": {
        "frames": "pack/val_frames.u8",
        "actions": "pack/val_actions.npy",
        "metadata": "pack/val_meta.json",
    },
}
CORE_OUTPUT_PATHS = frozenset(
    {"pack/manifest.json", "overlap_audit.json", "shuffle_audit.json"}
    | {
        relative
        for artifacts in PACK_ARTIFACT_RELATIVE_PATHS.values()
        for relative in artifacts.values()
    }
    | {
        f"arms/{arm}/measurements/update_{update:06d}.json"
        for arm in ARM_NAMES
        for update in MEASUREMENT_UPDATES
    }
    | {
        f"arms/{arm}/snapshots/update_{update:06d}.pt"
        for arm in ARM_NAMES
        for update in MEASUREMENT_UPDATES
    }
)
MAX_JSON_RECEIPT_BYTES = 128 * 1024 * 1024

_SHA256 = frozenset("0123456789abcdef")
_FORBIDDEN_OPEN_COMPONENTS = frozenset(
    {
        "checkpoint",
        "checkpoints",
        "depth",
        "pack",
        "rgb",
        "snapshot",
        "snapshots",
    }
)
_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "authority_binding",
        "plan_binding",
        "review_binding",
        "source_commit",
        "attempt",
        "caps",
        "runtime",
        "input_bindings",
        "predecessor_terminal_failure_binding",
        "pack_binding",
        "pack_artifact_bindings",
        "overlap_audit_binding",
        "shuffle_audit_binding",
        "arms",
        "joint_decision",
        "accounting",
        "forbidden_access",
        "checkpoint_bindings",
    }
)


class ThreeArmReceiptError(RuntimeError):
    """Raised when JSON receipts do not prove the frozen result contract."""


def _fail(message: str) -> None:
    raise ThreeArmReceiptError(message)


def _reject_protected_path(path: Path, *, label: str) -> None:
    lowered = tuple(part.lower() for part in Path(path).parts)
    if any(
        part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        or part in {"heldout", "held_out", "held-out"}
        or part.startswith("heldout_")
        or part.startswith("held_out_")
        or part.startswith("held-out-")
        for part in lowered
    ):
        _fail(f"{label} path is custody-protected")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, label: str) -> Any:
    """Decode strict UTF-8 JSON with duplicate/non-finite rejection."""

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ThreeArmReceiptError(
                    f"non-finite JSON value in {label}: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThreeArmReceiptError(f"invalid JSON in {label}") from exc


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_regular_file(path: Path, *, expected_bytes: int, label: str) -> bytes:
    """Read one caller-sized regular file without following a final symlink."""

    selected = Path(path)
    _reject_protected_path(selected, label=label)
    if expected_bytes < 1 or expected_bytes > MAX_JSON_RECEIPT_BYTES:
        _fail(f"{label} byte count is outside the JSON-receipt ceiling")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = selected.stat(follow_symlinks=False)
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise ThreeArmReceiptError(f"cannot open bound {label}: {selected}") from exc
    try:
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode):
            _fail(f"bound {label} is not a regular file")
        if before_fd.st_size != expected_bytes:
            _fail(f"bound {label} byte count changed")
        chunks: list[bytes] = []
        remaining = expected_bytes
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"bound {label} ended before its declared byte count")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"bound {label} exceeds its declared byte count")
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = selected.stat(follow_symlinks=False)
    if not (
        _fingerprint(before_path)
        == _fingerprint(before_fd)
        == _fingerprint(after_fd)
        == _fingerprint(after_path)
    ):
        _fail(f"bound {label} changed while being read")
    return b"".join(chunks)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _plain_dict(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be a plain JSON object")
    return value


def _plain_list(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be a plain JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        _fail(
            f"{label} keys changed: missing={sorted(required - observed)}, "
            f"unexpected={sorted(observed - required)}"
        )


def binding_shape(value: Any, *, label: str) -> dict[str, Any]:
    binding = _plain_dict(value, label=label)
    _exact_keys(
        binding,
        ("path", "file_sha256", "byte_count"),
        label=label,
    )
    if type(binding["path"]) is not str or not binding["path"]:
        _fail(f"{label}.path must be a non-empty string")
    _reject_protected_path(Path(binding["path"]), label=label)
    if not _is_sha256(binding["file_sha256"]):
        _fail(f"{label}.file_sha256 must be lowercase SHA-256")
    if (
        type(binding["byte_count"]) is not int
        or binding["byte_count"] < 1
    ):
        _fail(f"{label}.byte_count must be a positive integer")
    return dict(binding)


def file_binding(path: Path) -> dict[str, Any]:
    selected = Path(path)
    _reject_protected_path(selected, label="bound file")
    if selected.is_symlink() or not selected.is_file():
        _fail(f"bound file is absent, non-regular, or a symlink: {selected}")
    size = selected.stat().st_size
    raw = _read_regular_file(selected, expected_bytes=size, label="bound file")
    return {
        "path": str(selected.resolve()),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _resolve_openable_receipt(
    binding: Mapping[str, Any],
    *,
    receipt_root: Path,
    expected_relative: str,
    label: str,
) -> Path:
    declared = Path(str(binding["path"]))
    candidate = declared if declared.is_absolute() else receipt_root / declared
    _reject_protected_path(candidate, label=label)
    lowered = tuple(part.lower() for part in candidate.parts)
    if any(part in _FORBIDDEN_OPEN_COMPONENTS for part in lowered):
        _fail(f"{label} names a payload path forbidden to this checker")
    if candidate.suffix != ".json":
        _fail(f"{label} is not a JSON receipt")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(receipt_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ThreeArmReceiptError(f"{label} escapes the receipt root") from exc
    if relative.as_posix() != PurePosixPath(expected_relative).as_posix():
        _fail(
            f"{label} path changed: expected {expected_relative!r}, "
            f"observed {relative.as_posix()!r}"
        )
    return resolved


def _read_bound_json(
    value: Any,
    *,
    receipt_root: Path,
    expected_relative: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = binding_shape(value, label=f"{label} binding")
    path = _resolve_openable_receipt(
        binding,
        receipt_root=receipt_root,
        expected_relative=expected_relative,
        label=label,
    )
    raw = _read_regular_file(
        path,
        expected_bytes=int(binding["byte_count"]),
        label=label,
    )
    if hashlib.sha256(raw).hexdigest() != binding["file_sha256"]:
        _fail(f"{label} SHA-256 changed")
    document = strict_json_bytes(raw, label=label)
    return _plain_dict(document, label=label), binding


def _require_finite_json(value: Any, *, label: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite number")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _require_finite_json(child, label=f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{label} contains a non-string key")
            _require_finite_json(child, label=f"{label}.{key}")
        return
    _fail(f"{label} contains a non-JSON value")


def _require_number(
    value: Any, *, label: str, minimum: float | None = None
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(f"{label} must be a finite JSON number")
    number = float(value)
    if minimum is not None and number < minimum:
        _fail(f"{label} must be >= {minimum}")
    return number


def _require_int(value: Any, *, label: str, expected: int | None = None) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative JSON integer")
    if expected is not None and value != expected:
        _fail(f"{label} must equal {expected}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        _fail(f"{label} must be lowercase SHA-256")
    return str(value)


def _validate_attempt(value: Any, *, receipt_root: Path) -> dict[str, Any]:
    attempt = _plain_dict(value, label="result.attempt")
    _exact_keys(
        attempt,
        (
            "id",
            "root",
            "maximum_attempts",
            "must_be_absent",
            "reservation_consumes_attempt",
            "retry",
            "resume",
            "overwrite",
            "refill",
            "reservation",
        ),
        label="result.attempt",
    )
    reservation = _plain_dict(
        attempt["reservation"], label="result.attempt.reservation"
    )
    _exact_keys(
        reservation,
        (
            "binding",
            "supervisor_nonce",
            "status",
            "maximum_attempts",
            "retry",
            "resume",
            "overwrite",
            "refill",
        ),
        label="result.attempt.reservation",
    )
    reservation_binding = binding_shape(
        reservation["binding"], label="result.attempt.reservation.binding"
    )
    _require_inert_relative_path(
        reservation_binding,
        receipt_root=receipt_root,
        expected_relative="reservation.json",
        label="result.attempt.reservation.binding",
    )
    if (
        attempt.get("id") != ATTEMPT_ID
        or attempt.get("root") != str(receipt_root.resolve(strict=True))
        or attempt.get("maximum_attempts") != 1
        or attempt.get("must_be_absent") is not True
        or attempt.get("reservation_consumes_attempt") is not True
        or attempt.get("retry") is not False
        or attempt.get("resume") is not False
        or attempt.get("overwrite") is not False
        or attempt.get("refill") is not False
        or reservation.get("status") != "RESERVED_ATTEMPT_CONSUMED"
        or reservation.get("maximum_attempts") != 1
        or reservation.get("retry") is not False
        or reservation.get("resume") is not False
        or reservation.get("overwrite") is not False
        or reservation.get("refill") is not False
        or not _is_sha256(reservation.get("supervisor_nonce"))
    ):
        _fail(
            "result attempt is not the exact fresh V3 consumed, non-retriable "
            "attempt"
        )
    return attempt


def _validate_caps_and_runtime(
    caps_value: Any, runtime_value: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    caps = _plain_dict(caps_value, label="result.caps")
    _exact_keys(
        caps,
        ("maximum_wall_seconds", "maximum_gpu_seconds", "maximum_training_updates"),
        label="result.caps",
    )
    wall_cap = _require_number(
        caps["maximum_wall_seconds"],
        label="result.caps.maximum_wall_seconds",
        minimum=0.0,
    )
    gpu_cap = _require_number(
        caps["maximum_gpu_seconds"],
        label="result.caps.maximum_gpu_seconds",
        minimum=0.0,
    )
    if wall_cap != 43_200.0 or gpu_cap != 36_000.0:
        _fail("result caps differ from the exact preregistered wall/GPU values")
    _require_int(
        caps["maximum_training_updates"],
        label="result.caps.maximum_training_updates",
        expected=700,
    )
    runtime = _plain_dict(runtime_value, label="result.runtime")
    _exact_keys(runtime, ("authorized", "observed"), label="result.runtime")
    _plain_dict(runtime["authorized"], label="result.runtime.authorized")
    observed = _plain_dict(runtime["observed"], label="result.runtime.observed")
    required = {
        "device_name": "AMD Radeon AI PRO R9700",
        "device_arch": "gfx1201",
        "torch_version": "2.9.1+rocm7.2.1.gitff65f5bc",
        "torch_hip": "7.2.53211-e1a6bc5663",
        "numpy_version": "1.26.4",
        "pillow_version": "11.3.0",
    }
    if any(observed.get(key) != expected for key, expected in required.items()):
        _fail("result observed runtime identity changed")
    gpu_elapsed = _require_number(
        observed.get("gpu_phase_elapsed_seconds"),
        label="result.runtime.observed.gpu_phase_elapsed_seconds",
        minimum=0.0,
    )
    wall_elapsed = _require_number(
        observed.get("wall_elapsed_seconds"),
        label="result.runtime.observed.wall_elapsed_seconds",
        minimum=0.0,
    )
    if gpu_elapsed > gpu_cap or wall_elapsed > wall_cap:
        _fail("result observed runtime exceeded an authority cap")
    inventory = _plain_list(
        observed.get("output_inventory"),
        label="result.runtime.observed.output_inventory",
    )
    if (
        len(inventory) != len(CORE_OUTPUT_PATHS)
        or any(type(item) is not str or not item for item in inventory)
        or len(set(inventory)) != len(inventory)
        or set(inventory) != CORE_OUTPUT_PATHS
    ):
        _fail("result observed output inventory is incomplete or duplicated")
    _require_finite_json(observed, label="result.runtime.observed")
    return caps, runtime


def _validate_audit(
    document: Mapping[str, Any],
    *,
    suffix: str,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    label: str,
) -> None:
    schema = document.get("schema")
    if type(schema) is not str or not schema.endswith(suffix):
        _fail(f"{label} schema is invalid")
    if (
        document.get("status") != "PASS"
        or document.get("passed") is not True
        or document.get("authority_binding") != authority_binding
        or document.get("plan_binding") != plan_binding
    ):
        _fail(f"{label} is not an exact linked PASS")
    _exact_keys(
        document,
        ("schema", "status", "passed", "authority_binding", "plan_binding", "audit"),
        label=label,
    )
    audit = _plain_dict(document.get("audit"), label=f"{label}.audit")
    if suffix == "_overlap_audit_v1":
        _exact_keys(
            audit,
            OVERLAP_AUDIT_KEYS,
            label=f"{label}.audit",
        )
        checks = {
            "role_scene_disjointness": True,
            "train_all_actions_supported": True,
            "train_all_ordered_pairs_supported": True,
        }
        diagnostic_checks = {"train_all_ordered_triples_supported": False}
        if (
            audit["schema"] != OVERLAP_AUDIT_SCHEMA
            or audit["status"] != "PASS"
            or audit["passed"] is not True
            or audit["checks"] != checks
            or audit["failed_checks"] != []
            or audit["diagnostic_checks"] != diagnostic_checks
            or audit["failed_diagnostic_checks"]
            != ["train_all_ordered_triples_supported"]
            or audit["role_row_counts"] != {"train": 16_000, "val": 2_048}
            or audit["role_scene_counts"] != {"train": 1_000, "val": 150}
            or audit["role_scene_overlap_count"] != 0
            or audit["role_scene_overlap"] != []
            or audit["gate_scope"] != OVERLAP_GATE_SCOPE
        ):
            _fail(f"{label} exact support/split checks failed")
        _require_int(
            audit["row_count"], label=f"{label}.audit.row_count", expected=18_048
        )
        support = _plain_dict(
            audit["train_support"], label=f"{label}.audit.train_support"
        )
        _exact_keys(
            support,
            (
                "visible_action_positions",
                "action_count",
                "action_count_by_position",
                "ordered_pair_count",
                "ordered_pair_count_by_position",
                "ordered_triple_count",
                "missing_action_ids_by_position",
                "missing_action_ids",
                "missing_ordered_pairs_by_position",
                "missing_ordered_pairs",
                "missing_ordered_triples",
            ),
            label=f"{label}.audit.train_support",
        )
        if support != {
            "visible_action_positions": [0, 1, 2],
            "action_count": 9,
            "action_count_by_position": {"a0": 9, "a1": 9, "a2": 9},
            "ordered_pair_count": 81,
            "ordered_pair_count_by_position": {"a0_a1": 81, "a1_a2": 81},
            "ordered_triple_count": 722,
            "missing_action_ids_by_position": {"a0": [], "a1": [], "a2": []},
            "missing_action_ids": [],
            "missing_ordered_pairs_by_position": {"a0_a1": [], "a1_a2": []},
            "missing_ordered_pairs": [],
            "missing_ordered_triples": [
                list(values) for values in EXPECTED_MISSING_ORDERED_TRAIN_TRIPLES
            ],
        }:
            _fail(f"{label} train action support is incomplete")
        for key in ("entropy", "mutual_information_bits", "scene_diagnostics"):
            diagnostic = _plain_dict(audit[key], label=f"{label}.audit.{key}")
            if not diagnostic:
                _fail(f"{label}.audit.{key} is empty")
            _require_finite_json(diagnostic, label=f"{label}.audit.{key}")
    else:
        _exact_keys(
            audit,
            (
                "schema",
                "status",
                "passed",
                "algorithm",
                "candidate_action_position",
                "changed_action_positions",
                "row_count",
                "role_family_group_count",
                "group_selected_offsets",
                "group_methods",
                "mapping_sha256",
                "checks",
                "fixed_donor_identity_count",
                "same_scene_donor_count",
                "fixed_candidate_action_count",
                "mapping_rows",
            ),
            label=f"{label}.audit",
        )
        checks = {
            "donor_map_is_global_bijection": True,
            "donor_identity_zero_fixed_points": True,
            "different_scene_donors": True,
            "candidate_a2_zero_fixed_points": True,
            "role_family_action_marginals_exact": True,
        }
        if (
            audit["schema"] != SHUFFLE_AUDIT_SCHEMA
            or audit["status"] != "PASS"
            or audit["passed"] is not True
            or audit["algorithm"]
            != "role_family_local_cyclic_then_exact_bipartite_derangement_v1"
            or audit["candidate_action_position"] != 2
            or audit["changed_action_positions"] != [2]
            or audit["checks"] != checks
        ):
            _fail(f"{label} registered algorithm/checks changed")
        _require_int(audit["row_count"], label=f"{label}.audit.row_count", expected=16_000)
        _require_int(
            audit["role_family_group_count"],
            label=f"{label}.audit.role_family_group_count",
            expected=8,
        )
        for key in (
            "fixed_donor_identity_count",
            "same_scene_donor_count",
            "fixed_candidate_action_count",
        ):
            _require_int(audit[key], label=f"{label}.audit.{key}", expected=0)
        offsets = _plain_dict(
            audit["group_selected_offsets"],
            label=f"{label}.audit.group_selected_offsets",
        )
        methods = _plain_dict(
            audit["group_methods"],
            label=f"{label}.audit.group_methods",
        )
        if set(offsets) != set(methods) or len(offsets) != 8:
            _fail(f"{label} group offset inventory changed")
        for group_name, method in methods.items():
            offset = offsets[group_name]
            if method == "dual_hash_ranked_cyclic_search":
                if type(offset) is not int or offset < 0:
                    _fail(f"{label} cyclic group offset changed")
            elif method == "exact_hopcroft_karp_dense_complement":
                if offset is not None:
                    _fail(f"{label} exact-matching group offset must be null")
            else:
                _fail(f"{label} unregistered group derangement method")
        mapping_rows = _plain_list(
            audit["mapping_rows"], label=f"{label}.audit.mapping_rows"
        )
        if len(mapping_rows) != 16_000:
            _fail(f"{label} mapping row count changed")
        donor_positions: list[int] = []
        families: set[str] = set()
        for index, row_value in enumerate(mapping_rows):
            row = _plain_dict(row_value, label=f"{label}.audit.mapping_rows[{index}]")
            _exact_keys(
                row,
                (
                    "row_position",
                    "row_index",
                    "role",
                    "family",
                    "scene_id",
                    "factual_candidate_action_id",
                    "donor_position",
                    "donor_index",
                    "donor_scene_id",
                    "deranged_candidate_action_id",
                ),
                label=f"{label}.audit.mapping_rows[{index}]",
            )
            donor = _require_int(
                row["donor_position"],
                label=f"{label}.audit.mapping_rows[{index}].donor_position",
            )
            factual = _require_int(
                row["factual_candidate_action_id"],
                label=f"{label}.audit.mapping_rows[{index}].factual_candidate_action_id",
            )
            deranged = _require_int(
                row["deranged_candidate_action_id"],
                label=f"{label}.audit.mapping_rows[{index}].deranged_candidate_action_id",
            )
            if (
                row["row_position"] != index
                or row["role"] != "train"
                or type(row["family"]) is not str
                or not row["family"]
                or donor == index
                or donor >= 16_000
                or factual >= 9
                or deranged >= 9
                or factual == deranged
                or row["scene_id"] == row["donor_scene_id"]
            ):
                _fail(f"{label} mapping invariant failed at row {index}")
            donor_positions.append(donor)
            families.add(row["family"])
        expected_groups = {f"train:{family}" for family in families}
        if (
            sorted(donor_positions) != list(range(16_000))
            or len(families) != 8
            or set(offsets) != expected_groups
        ):
            _fail(f"{label} mapping is not the exact eight-family bijection")
        for index, donor in enumerate(donor_positions):
            recipient = mapping_rows[index]
            donor_row = mapping_rows[donor]
            if (
                donor_row["family"] != recipient["family"]
                or donor_row["role"] != recipient["role"]
                or donor_row["row_index"] != recipient["donor_index"]
                or donor_row["scene_id"] != recipient["donor_scene_id"]
                or donor_row["factual_candidate_action_id"]
                != recipient["deranged_candidate_action_id"]
            ):
                _fail(f"{label} donor linkage changed at row {index}")
        mapping_sha256 = hashlib.sha256(
            json.dumps(
                mapping_rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if audit["mapping_sha256"] != mapping_sha256:
            _fail(f"{label} mapping SHA-256 does not recompute")


def _validate_measurement(
    document: Mapping[str, Any],
    *,
    arm: str,
    update: int,
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    label = f"{arm} update {update} measurement"
    _exact_keys(
        document,
        (
            "schema",
            "status",
            "arm",
            "update",
            "authority_binding",
            "plan_binding",
            "encoder_sha256",
            "target_sha256",
            "panel",
            "validation",
            "training",
            "optimization",
            "integrity",
        ),
        label=label,
    )
    if (
        document.get("schema") != METRICS_SCHEMA
        or document.get("status") != "COMPLETE"
        or document.get("arm") != arm
        or document.get("update") != update
        or document.get("authority_binding") != authority_binding
        or document.get("plan_binding") != plan_binding
    ):
        _fail(f"{label} identity or completion contract changed")
    for key in ("encoder_sha256", "target_sha256"):
        _require_sha256(document.get(key), label=f"{label}.{key}")
    panel = _plain_dict(document["panel"], label=f"{label}.panel")
    _exact_keys(
        panel,
        ("kind", "row_count", "row_indices_sha256"),
        label=f"{label}.panel",
    )
    if panel["kind"] != "scene_disjoint_factual_validation":
        _fail(f"{label}.panel.kind changed")
    _require_int(panel["row_count"], label=f"{label}.panel.row_count", expected=2_048)
    _require_sha256(
        panel["row_indices_sha256"], label=f"{label}.panel.row_indices_sha256"
    )
    validation = _plain_dict(
        document.get("validation"), label=f"{label}.validation"
    )
    _exact_keys(
        validation,
        (
            "row_count",
            "factual_energy",
            "cross_arm",
            "controls",
            "action_identification",
            "representation",
        ),
        label=f"{label}.validation",
    )
    _require_int(
        validation["row_count"],
        label=f"{label}.validation.row_count",
        expected=2_048,
    )
    factual = _plain_dict(
        validation["factual_energy"], label=f"{label}.validation.factual_energy"
    )
    _exact_keys(
        factual,
        (
            "mean",
            "family_equal_mean",
            "action_equal_mean",
            "family_count",
            "action_count",
            "scene_count",
        ),
        label=f"{label}.validation.factual_energy",
    )
    for key in ("mean", "family_equal_mean", "action_equal_mean"):
        number = _require_number(
            factual[key], label=f"{label}.validation.factual_energy.{key}", minimum=0.0
        )
        if number <= 0.0:
            _fail(f"{label}.validation.factual_energy.{key} must be positive")
    for key, expected in (("family_count", 8), ("action_count", 9), ("scene_count", 150)):
        _require_int(
            factual[key],
            label=f"{label}.validation.factual_energy.{key}",
            expected=expected,
        )
    cross_arm = _plain_dict(
        validation["cross_arm"], label=f"{label}.validation.cross_arm"
    )
    _exact_keys(
        cross_arm,
        (
            "conditioned_vs_blind_log_energy_advantage",
            "conditioned_vs_blind_one_sided_95_lower_bound",
            "conditioned_vs_shuffled_log_energy_advantage",
            "conditioned_vs_shuffled_one_sided_95_lower_bound",
            "scene_cluster_count",
        ),
        label=f"{label}.validation.cross_arm",
    )
    for key in (
        "conditioned_vs_blind_log_energy_advantage",
        "conditioned_vs_blind_one_sided_95_lower_bound",
        "conditioned_vs_shuffled_log_energy_advantage",
        "conditioned_vs_shuffled_one_sided_95_lower_bound",
    ):
        _require_number(cross_arm[key], label=f"{label}.validation.cross_arm.{key}")
    if (
        cross_arm["conditioned_vs_blind_one_sided_95_lower_bound"]
        > cross_arm["conditioned_vs_blind_log_energy_advantage"]
        or cross_arm["conditioned_vs_shuffled_one_sided_95_lower_bound"]
        > cross_arm["conditioned_vs_shuffled_log_energy_advantage"]
    ):
        _fail(f"{label} cross-arm lower bound exceeds its point estimate")
    _require_int(
        cross_arm["scene_cluster_count"],
        label=f"{label}.validation.cross_arm.scene_cluster_count",
        expected=150,
    )
    controls = _plain_dict(
        validation["controls"], label=f"{label}.validation.controls"
    )
    _exact_keys(
        controls,
        (
            "persistence_log_energy_advantage",
            "persistence_one_sided_95_lower_bound",
            "wrong_history_log_energy_advantage",
            "wrong_history_one_sided_95_lower_bound",
        ),
        label=f"{label}.validation.controls",
    )
    for key, value in controls.items():
        _require_number(value, label=f"{label}.validation.controls.{key}")
    if (
        controls["persistence_one_sided_95_lower_bound"]
        > controls["persistence_log_energy_advantage"]
        or controls["wrong_history_one_sided_95_lower_bound"]
        > controls["wrong_history_log_energy_advantage"]
    ):
        _fail(f"{label} control lower bound exceeds its point estimate")
    action = _plain_dict(
        validation["action_identification"],
        label=f"{label}.validation.action_identification",
    )
    _exact_keys(
        action,
        ACTION_IDENTIFICATION_RECEIPT_KEYS,
        label=f"{label}.validation.action_identification",
    )
    if (
        action["bootstrap_algorithm"] != ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM
        or action["bootstrap_interpretation"]
        != ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
    ):
        _fail(f"{label} nine-way bootstrap algorithm/interpretation changed")
    _require_int(
        action["bootstrap_seed"],
        label=f"{label}.validation.action_identification.bootstrap_seed",
        expected=ACTION_IDENTIFICATION_BOOTSTRAP_SEED,
    )
    _require_int(
        action["bootstrap_replicates"],
        label=f"{label}.validation.action_identification.bootstrap_replicates",
        expected=ACTION_IDENTIFICATION_BOOTSTRAP_REPLICATES,
    )
    _require_int(
        action["bootstrap_lower_index"],
        label=f"{label}.validation.action_identification.bootstrap_lower_index",
        expected=ACTION_IDENTIFICATION_BOOTSTRAP_LOWER_INDEX,
    )
    support = _plain_dict(
        action["family_action_supporting_scene_counts"],
        label=(
            f"{label}.validation.action_identification."
            "family_action_supporting_scene_counts"
        ),
    )
    _exact_keys(
        support,
        REGISTERED_FAMILIES,
        label=(
            f"{label}.validation.action_identification."
            "family_action_supporting_scene_counts"
        ),
    )
    support_counts: list[int] = []
    for family in REGISTERED_FAMILIES:
        counts = _plain_list(
            support[family],
            label=(
                f"{label}.validation.action_identification."
                f"family_action_supporting_scene_counts.{family}"
            ),
        )
        if len(counts) != 9:
            _fail(f"{label} family/action support must have exactly nine actions")
        for action_id, count in enumerate(counts):
            observed_count = _require_int(
                count,
                label=(
                    f"{label}.validation.action_identification."
                    f"family_action_supporting_scene_counts.{family}[{action_id}]"
                ),
            )
            if observed_count < 1:
                _fail(f"{label} family/action supporting-scene count is not positive")
            support_counts.append(observed_count)
    declared_minimum = _require_int(
        action["minimum_family_action_supporting_scene_count"],
        label=(
            f"{label}.validation.action_identification."
            "minimum_family_action_supporting_scene_count"
        ),
    )
    recomputed_minimum = min(support_counts)
    if declared_minimum != recomputed_minimum:
        _fail(f"{label} minimum family/action supporting-scene count is inconsistent")
    if recomputed_minimum < ACTION_IDENTIFICATION_MIN_SUPPORTING_SCENES:
        _fail(f"{label} family/action supporting-scene minimum is below two")
    for key in (
        "balanced_accuracy",
        "balanced_accuracy_one_sided_95_lower_bound",
        "exact_tie_rate",
        "unique_winner_accuracy",
    ):
        number = _require_number(
            action[key],
            label=f"{label}.validation.action_identification.{key}",
            minimum=0.0,
        )
        if number > 1.0:
            _fail(f"{label}.validation.action_identification.{key} exceeds one")
    if not math.isclose(
        _require_number(
            action["balanced_chance"],
            label=f"{label}.validation.action_identification.balanced_chance",
        ),
        1.0 / 9.0,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        _fail(f"{label} nine-way balanced chance changed")
    tie_count = _require_int(
        action["exact_tie_count"],
        label=f"{label}.validation.action_identification.exact_tie_count",
    )
    unique_count = _require_int(
        action["unique_winner_count"],
        label=f"{label}.validation.action_identification.unique_winner_count",
    )
    if tie_count + unique_count != 2_048:
        _fail(f"{label} tie/unique-winner accounting changed")
    if not math.isclose(
        float(action["exact_tie_rate"]),
        tie_count / 2_048,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        _fail(f"{label} tie rate does not match tie count")
    if arm == "blind" and (
        tie_count != 2_048
        or unique_count != 0
        or action["exact_tie_rate"] != 1.0
        or action["unique_winner_accuracy"] != 0.0
    ):
        _fail(f"{label} candidate-blind nine-way queries are not exact ties")
    for key in (
        "hardest_wrong_action_margin",
        "hardest_wrong_action_margin_one_sided_95_lower_bound",
    ):
        _require_number(
            action[key], label=f"{label}.validation.action_identification.{key}"
        )
    if (
        action["balanced_accuracy_one_sided_95_lower_bound"]
        > action["balanced_accuracy"]
        or action[
            "hardest_wrong_action_margin_one_sided_95_lower_bound"
        ]
        > action["hardest_wrong_action_margin"]
    ):
        _fail(f"{label} action lower bound exceeds its point estimate")
    representation = _plain_dict(
        validation["representation"], label=f"{label}.validation.representation"
    )
    _exact_keys(
        representation,
        (
            "prediction_effective_rank",
            "target_effective_rank",
            "prediction_to_target_rank_ratio",
        ),
        label=f"{label}.validation.representation",
    )
    for key, value in representation.items():
        _require_number(
            value, label=f"{label}.validation.representation.{key}", minimum=0.0
        )
    if representation["target_effective_rank"] <= 0.0 or not math.isclose(
        float(representation["prediction_to_target_rank_ratio"]),
        float(representation["prediction_effective_rank"])
        / float(representation["target_effective_rank"]),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        _fail(f"{label} prediction/target rank ratio does not recompute")
    training = document["training"]
    if update < 700:
        if training is not None:
            _fail(f"{label}.training must be null before update 700")
    else:
        training = _plain_dict(training, label=f"{label}.training")
        _exact_keys(
            training,
            (
                "row_count",
                "family_count",
                "factual_mean_energy",
                "conditioned_vs_blind_family_equal_log_energy_advantage",
                "conditioned_vs_shuffled_family_equal_log_energy_advantage",
                "backward_calls",
                "optimizer_steps",
            ),
            label=f"{label}.training",
        )
        _require_int(training["row_count"], label=f"{label}.training.row_count", expected=16_000)
        _require_int(training["family_count"], label=f"{label}.training.family_count", expected=8)
        _require_number(
            training["factual_mean_energy"],
            label=f"{label}.training.factual_mean_energy",
            minimum=0.0,
        )
        for key in (
            "conditioned_vs_blind_family_equal_log_energy_advantage",
            "conditioned_vs_shuffled_family_equal_log_energy_advantage",
        ):
            _require_number(training[key], label=f"{label}.training.{key}")
        _require_int(training["backward_calls"], label=f"{label}.training.backward_calls", expected=0)
        _require_int(training["optimizer_steps"], label=f"{label}.training.optimizer_steps", expected=0)
    optimization = _plain_dict(
        document["optimization"], label=f"{label}.optimization"
    )
    _exact_keys(
        optimization,
        (
            "completed_updates",
            "optimizer_steps",
            "loss",
            "learning_rate_fraction",
            "predictor_learning_rate",
            "memory_learning_rate",
            "warmup_updates",
            "schedule_horizon_updates",
        ),
        label=f"{label}.optimization",
    )
    _require_int(
        optimization["completed_updates"],
        label=f"{label}.optimization.completed_updates",
        expected=update,
    )
    _require_int(
        optimization["optimizer_steps"],
        label=f"{label}.optimization.optimizer_steps",
        expected=update,
    )
    if update == 0 and optimization["loss"] is None:
        pass
    else:
        _require_number(
            optimization["loss"],
            label=f"{label}.optimization.loss",
            minimum=0.0,
        )
    for key in (
        "learning_rate_fraction",
        "predictor_learning_rate",
        "memory_learning_rate",
    ):
        _require_number(
            optimization[key],
            label=f"{label}.optimization.{key}",
            minimum=0.0,
        )
    _require_int(
        optimization["warmup_updates"],
        label=f"{label}.optimization.warmup_updates",
        expected=150,
    )
    _require_int(
        optimization["schedule_horizon_updates"],
        label=f"{label}.optimization.schedule_horizon_updates",
        expected=3_000,
    )
    integrity = _plain_dict(document["integrity"], label=f"{label}.integrity")
    _exact_keys(
        integrity,
        (
            "candidate_blind_treatment_exact",
            "shuffled_derangement_exact",
            "factual_evaluation_exact",
            "frozen_substrate_exact",
            "no_gradient_during_evaluation",
            "finite",
        ),
        label=f"{label}.integrity",
    )
    if any(value is not True for value in integrity.values()):
        _fail(f"{label} integrity assertion failed")
    return (
        str(document["encoder_sha256"]),
        str(document["target_sha256"]),
        {
            "panel_sha256": panel["row_indices_sha256"],
            "cross_arm": dict(cross_arm),
            "controls": dict(controls),
            "action_identification": dict(action),
            "representation": dict(representation),
            "training": None if training is None else dict(training),
        },
    )


_DECISION_PRECEDENCE = [
    "INCONCLUSIVE_CONTRACT_FAILURE",
    "LOCALIZE_TRAIN_FIT_FAILURE",
    "LOCALIZE_GENERALIZATION_OR_CONFOUNDING",
    "LOCALIZE_ACTION_ALIGNMENT_FAILURE",
    "LOCALIZE_PREDICTOR_NOT_USEFUL",
    "PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY",
]


def _expected_decision_status(evidence: Mapping[str, Any]) -> str:
    train = evidence["train_fit_update_700"]
    if (
        float(train["conditioned_vs_blind_family_equal_log_energy_advantage"])
        <= 0.0
        or float(
            train[
                "conditioned_vs_shuffled_family_equal_log_energy_advantage"
            ]
        )
        <= 0.0
    ):
        return "LOCALIZE_TRAIN_FIT_FAILURE"
    tail = evidence["validation_tail"]
    if any(
        float(row["conditioned_vs_blind_log_energy_advantage"]) <= 0.0
        or float(row["conditioned_vs_shuffled_log_energy_advantage"]) <= 0.0
        for row in tail
    ) or (
        float(tail[-1]["conditioned_vs_blind_one_sided_95_lower_bound"])
        <= 0.0
        or float(
            tail[-1]["conditioned_vs_shuffled_one_sided_95_lower_bound"]
        )
        <= 0.0
    ):
        return "LOCALIZE_GENERALIZATION_OR_CONFOUNDING"
    final = evidence["conditioned_update_700"]
    if (
        float(final["balanced_accuracy_one_sided_95_lower_bound"])
        <= 1.0 / 9.0
        or float(
            final[
                "hardest_wrong_action_margin_one_sided_95_lower_bound"
            ]
        )
        <= 0.0
    ):
        return "LOCALIZE_ACTION_ALIGNMENT_FAILURE"
    collapsed = sum(
        float(row["prediction_to_target_rank_ratio"]) < 0.25 for row in tail
    )
    if (
        float(final["persistence_one_sided_95_lower_bound"]) <= 0.0
        or float(final["wrong_history_one_sided_95_lower_bound"]) <= 0.0
        or collapsed >= 2
    ):
        return "LOCALIZE_PREDICTOR_NOT_USEFUL"
    return "PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY"


def _validate_joint_decision(
    value: Any,
    *,
    conditioned: Mapping[int, Mapping[str, Any]],
    encoder_sha256: str,
    target_sha256: str,
) -> str:
    joint = _plain_dict(value, label="result.joint_decision")
    _exact_keys(
        joint,
        (
            "status",
            "citable_as_scientific_evidence",
            "scientific_claim_authorized",
            "treatment",
            "schedule",
            "frozen_substrate",
            "evidence",
            "gate_precedence",
        ),
        label="result.joint_decision",
    )
    if (
        joint["citable_as_scientific_evidence"] is not False
        or joint["scientific_claim_authorized"] is not False
        or joint["gate_precedence"] != _DECISION_PRECEDENCE
    ):
        _fail("joint decision authority/precedence contract changed")
    treatment = _plain_dict(joint["treatment"], label="joint_decision.treatment")
    _exact_keys(
        treatment,
        (
            "conditioned_action_gains",
            "blind_action_gains",
            "shuffled_action_gains",
            "blind_preserves_factual_history",
            "shuffled_changes_only_training_candidate",
            "shuffled_validation_uses_factual_candidate",
            "requested_executed_equivalence_claimed",
        ),
        label="joint_decision.treatment",
    )
    if treatment != {
        "conditioned_action_gains": [1, 1, 1],
        "blind_action_gains": [1, 1, 0],
        "shuffled_action_gains": [1, 1, 1],
        "blind_preserves_factual_history": True,
        "shuffled_changes_only_training_candidate": True,
        "shuffled_validation_uses_factual_candidate": True,
        "requested_executed_equivalence_claimed": False,
    }:
        _fail("registered arm treatment changed")
    schedule = _plain_dict(joint["schedule"], label="joint_decision.schedule")
    _exact_keys(
        schedule,
        (
            "seed",
            "updates",
            "sequence_batch",
            "microbatch",
            "train_rows",
            "validation_rows",
            "warmup_updates",
            "schedule_horizon_updates",
            "observation_updates",
            "early_stopping",
            "checkpoint_selection",
        ),
        label="joint_decision.schedule",
    )
    if schedule != {
        "seed": 20_260_731,
        "updates": 700,
        "sequence_batch": 256,
        "microbatch": 32,
        "train_rows": 16_000,
        "validation_rows": 2_048,
        "warmup_updates": 150,
        "schedule_horizon_updates": 3_000,
        "observation_updates": list(MEASUREMENT_UPDATES),
        "early_stopping": False,
        "checkpoint_selection": False,
    }:
        _fail("registered schedule changed")
    frozen = _plain_dict(
        joint["frozen_substrate"], label="joint_decision.frozen_substrate"
    )
    _exact_keys(
        frozen,
        (
            "encoder_initial_sha256",
            "encoder_final_sha256",
            "target_initial_sha256",
            "target_final_sha256",
            "requires_grad",
            "evaluation_mode",
            "gradient_tensor_count",
            "ema_update_count",
        ),
        label="joint_decision.frozen_substrate",
    )
    if (
        frozen["encoder_initial_sha256"] != encoder_sha256
        or frozen["encoder_final_sha256"] != encoder_sha256
        or frozen["target_initial_sha256"] != target_sha256
        or frozen["target_final_sha256"] != target_sha256
        or frozen["requires_grad"] is not False
        or frozen["evaluation_mode"] is not True
        or frozen["gradient_tensor_count"] != 0
        or frozen["ema_update_count"] != 0
    ):
        _fail("frozen substrate identity or state changed")
    evidence = _plain_dict(joint["evidence"], label="joint_decision.evidence")
    _exact_keys(
        evidence,
        ("train_fit_update_700", "validation_tail", "conditioned_update_700"),
        label="joint_decision.evidence",
    )
    train = _plain_dict(
        evidence["train_fit_update_700"],
        label="joint_decision.evidence.train_fit_update_700",
    )
    train_keys = (
        "conditioned_vs_blind_family_equal_log_energy_advantage",
        "conditioned_vs_shuffled_family_equal_log_energy_advantage",
    )
    _exact_keys(train, train_keys, label="joint_decision.evidence.train_fit_update_700")
    for key in train_keys:
        _require_number(train[key], label=f"joint_decision.evidence.train_fit_update_700.{key}")
    measurement_train = conditioned[700]["training"]
    if measurement_train is None or any(
        train[key] != measurement_train[key] for key in train_keys
    ):
        _fail("joint training-fit evidence differs from update-700 measurement")
    tail = _plain_list(
        evidence["validation_tail"], label="joint_decision.evidence.validation_tail"
    )
    if len(tail) != 3:
        _fail("joint validation tail must contain updates 500/600/700")
    tail_keys = (
        "update",
        "conditioned_vs_blind_log_energy_advantage",
        "conditioned_vs_blind_one_sided_95_lower_bound",
        "conditioned_vs_shuffled_log_energy_advantage",
        "conditioned_vs_shuffled_one_sided_95_lower_bound",
        "prediction_to_target_rank_ratio",
    )
    for expected_update, row_value in zip((500, 600, 700), tail, strict=True):
        row = _plain_dict(
            row_value,
            label=f"joint_decision.evidence.validation_tail[{expected_update}]",
        )
        _exact_keys(
            row,
            tail_keys,
            label=f"joint_decision.evidence.validation_tail[{expected_update}]",
        )
        if row["update"] != expected_update:
            _fail("joint validation tail update order changed")
        source = conditioned[expected_update]
        cross = source["cross_arm"]
        representation = source["representation"]
        for key in tail_keys[1:-1]:
            _require_number(row[key], label=f"joint validation tail {expected_update}.{key}")
            if row[key] != cross[key]:
                _fail("joint cross-arm evidence differs from measurement")
        _require_number(
            row["prediction_to_target_rank_ratio"],
            label=f"joint validation tail {expected_update}.prediction_to_target_rank_ratio",
            minimum=0.0,
        )
        if (
            row["prediction_to_target_rank_ratio"]
            != representation["prediction_to_target_rank_ratio"]
        ):
            _fail("joint rank evidence differs from conditioned measurement")
    final = _plain_dict(
        evidence["conditioned_update_700"],
        label="joint_decision.evidence.conditioned_update_700",
    )
    final_keys = (
        "balanced_accuracy_one_sided_95_lower_bound",
        "hardest_wrong_action_margin_one_sided_95_lower_bound",
        "persistence_one_sided_95_lower_bound",
        "wrong_history_one_sided_95_lower_bound",
    )
    _exact_keys(final, final_keys, label="joint_decision.evidence.conditioned_update_700")
    action = conditioned[700]["action_identification"]
    controls = conditioned[700]["controls"]
    sources = {
        final_keys[0]: action[final_keys[0]],
        final_keys[1]: action[final_keys[1]],
        final_keys[2]: controls[final_keys[2]],
        final_keys[3]: controls[final_keys[3]],
    }
    for key, source in sources.items():
        _require_number(final[key], label=f"joint conditioned update 700.{key}")
        if final[key] != source:
            _fail("joint conditioned evidence differs from measurement")
    expected = _expected_decision_status(evidence)
    if joint["status"] != expected:
        _fail(
            f"joint decision precedence disagrees with evidence: "
            f"declared={joint['status']!r} expected={expected!r}"
        )
    return expected


def _validate_accounting(value: Any) -> None:
    accounting = _plain_dict(value, label="result.accounting")
    required = {
        "bound_h6_rows": 18_048,
        "initial_rgb_leaf_opens": 72_192,
        "verification_rgb_leaf_reopens": 192,
        "total_rgb_leaf_opens": 72_384,
        "forbidden_future_rgb_leaf_opens": 0,
        "packed_frame_bytes": 2_716_729_344,
        "training_schedule_row_presentations": 179_200,
        "sequence_presentations_per_arm": 179_200,
        "total_arm_head_sequence_presentations": 537_600,
        "shared_online_context_frame_encodings": 537_600,
        "shared_future_target_frame_encodings": 179_200,
        "actual_training_frame_encodings": 716_800,
        "optimizer_steps_per_arm": 700,
        "total_optimizer_steps": 2_100,
        "target_ema_steps": 0,
        "validation_row_panels_per_arm": 16_384,
        "shared_validation_frame_encodings": 65_536,
        "nine_way_arm_candidate_row_queries": 442_368,
        "validation_backward_calls": 0,
        "validation_optimizer_steps": 0,
        "train_fit_rows": 16_000,
        "train_fit_shared_frame_encodings": 64_000,
        "train_fit_arm_factual_row_queries": 48_000,
        "train_fit_backward_calls": 0,
        "train_fit_optimizer_steps": 0,
        "total_shared_frame_encodings": 846_336,
        "measurement_receipts": 24,
        "snapshot_bindings": 24,
        "sealed_open_count": 0,
        "heldout_open_count": 0,
        "network_access_count": 0,
    }
    missing = set(required) - set(accounting)
    if missing:
        _fail(f"result.accounting omits registered fields: {sorted(missing)}")
    for key, expected in required.items():
        _require_int(accounting[key], label=f"result.accounting.{key}", expected=expected)
    if accounting.get("training_consumed_pack_only") is not True:
        _fail("result.accounting does not prove pack-only training")
    _require_finite_json(accounting, label="result.accounting")


def _validate_inert_binding_map(value: Any, *, label: str) -> dict[str, Any]:
    mapping = _plain_dict(value, label=label)
    if not mapping:
        _fail(f"{label} must not be empty")
    for name, item in mapping.items():
        if type(name) is not str or not name:
            _fail(f"{label} has an invalid name")
        binding_shape(item, label=f"{label}.{name}")
    return mapping


def _require_inert_relative_path(
    binding: Mapping[str, Any],
    *,
    receipt_root: Path,
    expected_relative: str,
    label: str,
) -> None:
    """Validate an inert binding's location without opening its payload."""

    declared = PurePosixPath(str(binding["path"]))
    expected = PurePosixPath(expected_relative)
    if declared.is_absolute():
        try:
            declared = PurePosixPath(
                Path(str(binding["path"])).resolve(strict=False).relative_to(
                    receipt_root.resolve(strict=True)
                ).as_posix()
            )
        except ValueError as exc:
            raise ThreeArmReceiptError(f"{label} escapes attempt root") from exc
    if declared != expected:
        _fail(f"{label} path changed")
    candidate = receipt_root.joinpath(*expected.parts)
    cursor = receipt_root.resolve(strict=True)
    for part in expected.parts:
        cursor = cursor / part
        try:
            metadata = cursor.stat(follow_symlinks=False)
        except OSError as exc:
            raise ThreeArmReceiptError(
                f"{label} inert payload is absent"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            _fail(f"{label} inert payload path contains a symlink")
    metadata = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} inert payload is not a regular file")
    if metadata.st_size != int(binding["byte_count"]):
        _fail(f"{label} inert payload byte count changed")


def _validate_pack_artifact_bindings(
    value: Any,
    *,
    receipt_root: Path,
) -> dict[str, Any]:
    """Validate six direct pack bindings by lstat and size, never by opening."""

    roles = _plain_dict(value, label="result.pack_artifact_bindings")
    _exact_keys(
        roles,
        PACK_ARTIFACT_RELATIVE_PATHS,
        label="result.pack_artifact_bindings",
    )
    normalized: dict[str, Any] = {}
    for role, expected_artifacts in PACK_ARTIFACT_RELATIVE_PATHS.items():
        artifacts = _plain_dict(
            roles[role], label=f"result.pack_artifact_bindings.{role}"
        )
        _exact_keys(
            artifacts,
            expected_artifacts,
            label=f"result.pack_artifact_bindings.{role}",
        )
        normalized[role] = {}
        for name, expected_relative in expected_artifacts.items():
            label = f"result.pack_artifact_bindings.{role}.{name}"
            binding = binding_shape(artifacts[name], label=label)
            _require_inert_relative_path(
                binding,
                receipt_root=receipt_root,
                expected_relative=expected_relative,
                label=label,
            )
            normalized[role][name] = binding
    return normalized


def validate_result(
    result: Mapping[str, Any],
    *,
    result_binding: Mapping[str, Any],
    receipt_root: Path,
) -> dict[str, Any]:
    """Validate the result and every permitted JSON receipt."""

    _exact_keys(result, _RESULT_KEYS, label="result")
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != RESULT_STATUS:
        _fail("result schema/status is not terminal-review pending")
    commit = result.get("source_commit")
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in _SHA256 for character in commit)
    ):
        _fail("result.source_commit must be full lowercase Git hex")
    authority_binding = binding_shape(
        result["authority_binding"], label="result.authority_binding"
    )
    plan_binding = binding_shape(result["plan_binding"], label="result.plan_binding")
    binding_shape(result["review_binding"], label="result.review_binding")
    _validate_attempt(result["attempt"], receipt_root=receipt_root)
    _validate_caps_and_runtime(result["caps"], result["runtime"])
    _validate_inert_binding_map(result["input_bindings"], label="result.input_bindings")
    predecessor_failure_binding = binding_shape(
        result["predecessor_terminal_failure_binding"],
        label="result.predecessor_terminal_failure_binding",
    )
    pack_binding = binding_shape(result["pack_binding"], label="result.pack_binding")
    _require_inert_relative_path(
        pack_binding,
        receipt_root=receipt_root,
        expected_relative="pack/manifest.json",
        label="result.pack_binding",
    )
    _validate_pack_artifact_bindings(
        result["pack_artifact_bindings"],
        receipt_root=receipt_root,
    )

    opened: list[dict[str, Any]] = []
    overlap, overlap_binding = _read_bound_json(
        result["overlap_audit_binding"],
        receipt_root=receipt_root,
        expected_relative="overlap_audit.json",
        label="overlap audit",
    )
    opened.append(overlap_binding)
    _validate_audit(
        overlap,
        suffix="_overlap_audit_v1",
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        label="overlap audit",
    )
    shuffle, shuffle_binding = _read_bound_json(
        result["shuffle_audit_binding"],
        receipt_root=receipt_root,
        expected_relative="shuffle_audit.json",
        label="shuffle audit",
    )
    opened.append(shuffle_binding)
    _validate_audit(
        shuffle,
        suffix="_candidate_action_derangement_v1",
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        label="shuffle audit",
    )

    arms = _plain_dict(result["arms"], label="result.arms")
    _exact_keys(arms, ARM_NAMES, label="result.arms")
    encoder_identities: set[str] = set()
    target_identities: set[str] = set()
    panel_identities: set[str] = set()
    cross_arm_by_update: dict[int, set[str]] = {
        update: set() for update in MEASUREMENT_UPDATES
    }
    training_update_700: set[str] = set()
    conditioned_evidence: dict[int, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        arm_result = _plain_dict(arms[arm], label=f"result.arms.{arm}")
        if arm_result.get("status") != "COMPLETE":
            _fail(f"result.arms.{arm} did not complete")
        bindings = _plain_list(
            arm_result.get("measurement_bindings"),
            label=f"result.arms.{arm}.measurement_bindings",
        )
        if len(bindings) != len(MEASUREMENT_UPDATES):
            _fail(f"result.arms.{arm} must bind exactly eight measurements")
        for update, binding in zip(MEASUREMENT_UPDATES, bindings, strict=True):
            relative = f"arms/{arm}/measurements/update_{update:06d}.json"
            document, opened_binding = _read_bound_json(
                binding,
                receipt_root=receipt_root,
                expected_relative=relative,
                label=f"{arm} update {update} measurement",
            )
            opened.append(opened_binding)
            encoder_sha256, target_sha256, summary = _validate_measurement(
                document,
                arm=arm,
                update=update,
                authority_binding=authority_binding,
                plan_binding=plan_binding,
            )
            encoder_identities.add(encoder_sha256)
            target_identities.add(target_sha256)
            panel_identities.add(str(summary["panel_sha256"]))
            cross_arm_by_update[update].add(
                json.dumps(
                    summary["cross_arm"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            if update == 700:
                training_update_700.add(
                    json.dumps(
                        {
                            key: summary["training"][key]
                            for key in (
                                "conditioned_vs_blind_family_equal_log_energy_advantage",
                                "conditioned_vs_shuffled_family_equal_log_energy_advantage",
                            )
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
            if arm == "conditioned":
                conditioned_evidence[update] = summary
    if len(encoder_identities) != 1 or len(target_identities) != 1:
        _fail("frozen encoder/target identity changed across measurements or arms")
    if len(panel_identities) != 1:
        _fail("validation row identity changed across measurements or arms")
    if any(len(values) != 1 for values in cross_arm_by_update.values()):
        _fail("cross-arm evidence is not identical across arm receipts")
    if len(training_update_700) != 1:
        _fail("update-700 train-fit evidence is not identical across arm receipts")

    checkpoints = _plain_dict(
        result["checkpoint_bindings"], label="result.checkpoint_bindings"
    )
    _exact_keys(checkpoints, ARM_NAMES, label="result.checkpoint_bindings")
    for arm in ARM_NAMES:
        items = _plain_list(checkpoints[arm], label=f"checkpoint_bindings.{arm}")
        if len(items) != len(MEASUREMENT_UPDATES):
            _fail(f"checkpoint_bindings.{arm} must contain exactly eight snapshots")
        for update, item in zip(MEASUREMENT_UPDATES, items, strict=True):
            inert = binding_shape(item, label=f"checkpoint {arm} update {update}")
            _require_inert_relative_path(
                inert,
                receipt_root=receipt_root,
                expected_relative=f"arms/{arm}/snapshots/update_{update:06d}.pt",
                label=f"checkpoint {arm} update {update}",
            )

    forbidden = _plain_dict(result["forbidden_access"], label="result.forbidden_access")
    required_false = (
        "sealed_material_opened",
        "heldout_material_opened",
        "network_access_used",
        "validation_used_for_gradient_updates",
        "existing_pool_modified",
    )
    if any(forbidden.get(key) is not False for key in required_false):
        _fail("result forbidden-access assertions are incomplete or unsafe")
    decision_status = _validate_joint_decision(
        result["joint_decision"],
        conditioned=conditioned_evidence,
        encoder_sha256=next(iter(encoder_identities)),
        target_sha256=next(iter(target_identities)),
    )
    _validate_accounting(result["accounting"])
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "phase": "existing_pool_three_arm_v1_integrity_replacement_v3",
        "purpose": (
            "existing_pool_three_arm_v1_factual_learning_"
            "integrity_replacement_v3"
        ),
        "predecessor_terminal_failure_binding": predecessor_failure_binding,
        "manifest_binding": dict(result_binding),
        "opened_json_receipt_count": len(opened),
        "opened_json_receipt_bindings": opened,
        "arms": list(ARM_NAMES),
        "measurement_updates": list(MEASUREMENT_UPDATES),
        "joint_decision_status": decision_status,
        "pack_payloads_opened": False,
        "input_data_opened": False,
        "runtime_payloads_opened": False,
        "rgb_bytes_opened": False,
        "checkpoints_opened": False,
        "sealed_material_opened": False,
        "citable_as_scientific_evidence": False,
        "scientific_verdict_emitted": False,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_binding(path)


def check_manifest(
    manifest_path: Path,
    *,
    expected_file_sha256: str,
    expected_byte_count: int,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if manifest_path.name != "result.json":
        _fail("manifest must be the exact result.json receipt")
    binding = file_binding(manifest_path)
    if binding["file_sha256"] != expected_file_sha256:
        _fail("caller-bound result SHA-256 changed")
    if binding["byte_count"] != expected_byte_count:
        _fail("caller-bound result byte count changed")
    root = manifest_path.resolve().parent
    if output_path.resolve(strict=False).parent != root or output_path.name != "receipt_check.json":
        _fail("checker output must be fresh receipt_check.json in the attempt root")
    raw = _read_regular_file(
        manifest_path.resolve(),
        expected_bytes=expected_byte_count,
        label="result manifest",
    )
    result = _plain_dict(
        strict_json_bytes(raw, label="result manifest"), label="result manifest"
    )
    report = validate_result(result, result_binding=binding, receipt_root=root)
    report_binding = _write_json_exclusive(output_path, report)
    return report, report_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--expected-byte-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_byte_count < 1:
        parser.error("expected byte count must be positive")
    if not _is_sha256(args.expected_file_sha256):
        parser.error("expected file SHA-256 must be lowercase hexadecimal")
    report, report_binding = check_manifest(
        args.manifest,
        expected_file_sha256=args.expected_file_sha256,
        expected_byte_count=args.expected_byte_count,
        output_path=args.output,
    )
    print(
        json.dumps(
            {"status": report["status"], "receipt_check": report_binding},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
