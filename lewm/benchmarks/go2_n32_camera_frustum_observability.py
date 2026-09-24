"""Pure geometry and label diagnostics for the N32 camera-frustum audit.

This module performs no file I/O and imports no dataset or model code.  The
audit runner owns the fit-only access boundary and supplies validated target
arrays and canonical frame identities.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


EXECUTION_BINDING_SHA256 = (
    "c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9"
)
RESULT_SCHEMA = "lewm_go2_n32_camera_frustum_observability_audit_result_v1"
GEOMETRY_SCHEMA = "lewm_go2_n32_camera_frustum_geometry_v1"
OLD_COLUMN_SPAN_SCHEMA = "lewm_go2_n32_old_body_column_span_audit_v1"
MAPPING_AUDIT_SCHEMA = "lewm_go2_n32_camera_centered_mapping_audit_v1"
FRAME_ANALYSIS_SCHEMA = "lewm_go2_n32_camera_frustum_frame_analysis_v1"
LABEL_SUPPORT_SCHEMA = "lewm_go2_n32_camera_frustum_label_support_v1"
RAY_SEQUENCE_SCHEMA = "lewm_go2_n32_camera_frustum_ray_sequences_v1"
OBSERVABILITY_SUMMARY_SCHEMA = (
    "lewm_go2_n32_camera_frustum_label_observability_summary_v1"
)
AUTHORIZATION_SCHEMA = "lewm_go2_n32_camera_frustum_authorization_decision_v1"

FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
ENDPOINT_SIDES = ("current", "next")
UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2
CLASS_NAMES = ("unknown", "free", "occupied")
CLASS_IDS = (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)

CARTESIAN_SHAPE = (64, 64)
CARTESIAN_CELL_SIZE_M = 0.10
CARTESIAN_FORWARD_MIN_EDGE_M = -1.0
CARTESIAN_LEFT_MIN_EDGE_M = -3.2
CAMERA_XYZ_BODY_M = (0.326, 0.0, 0.043)
CAMERA_NEAR_M = 0.05
HORIZONTAL_FOV_DEG = 78.323
HALF_FOV_RAD = math.radians(HORIZONTAL_FOV_DEG / 2.0)
VERTICAL_FOV_DEG = 62.8370386364
VERTICAL_ANCHOR_Z_BODY_M = (-0.333, -0.133, 0.067, 0.267, 0.467)
RANGE_BIN_COUNT = 64
RANGE_BIN_SIZE_M = 0.10
RANGE_LIMIT_M = RANGE_BIN_COUNT * RANGE_BIN_SIZE_M
ANGULAR_BIN_COUNT = 256
ANGULAR_BIN_WIDTH_RAD = 2.0 * HALF_FOV_RAD / ANGULAR_BIN_COUNT
POLAR_BIN_COUNT = RANGE_BIN_COUNT * ANGULAR_BIN_COUNT

_DIRECTED_TRANSITIONS = (
    (UNKNOWN_CLASS, FREE_CLASS),
    (UNKNOWN_CLASS, OCCUPIED_CLASS),
    (FREE_CLASS, UNKNOWN_CLASS),
    (FREE_CLASS, OCCUPIED_CLASS),
    (OCCUPIED_CLASS, UNKNOWN_CLASS),
    (OCCUPIED_CLASS, FREE_CLASS),
)
_TRANSITION_NAMES = tuple(
    f"{CLASS_NAMES[source]}_to_{CLASS_NAMES[destination]}"
    for source, destination in _DIRECTED_TRANSITIONS
)


def canonical_json_sha256(value: object) -> str:
    """Hash strict compact canonical JSON."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_copy(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain strict JSON values") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return result


def geometry_contract() -> dict[str, Any]:
    """Return the exact frozen coordinate and class contract."""

    return {
        "schema": GEOMETRY_SCHEMA,
        "cartesian_shape": list(CARTESIAN_SHAPE),
        "cartesian_cell_size_m": CARTESIAN_CELL_SIZE_M,
        "cartesian_forward_min_edge_m": CARTESIAN_FORWARD_MIN_EDGE_M,
        "cartesian_left_min_edge_m": CARTESIAN_LEFT_MIN_EDGE_M,
        "camera_xyz_body_m": list(CAMERA_XYZ_BODY_M),
        "camera_rpy_body_rad": [0.0, 0.0, 0.0],
        "camera_near_m": CAMERA_NEAR_M,
        "horizontal_fov_deg": HORIZONTAL_FOV_DEG,
        "half_horizontal_fov_rad": HALF_FOV_RAD,
        "vertical_fov_deg": VERTICAL_FOV_DEG,
        "vertical_anchor_z_body_m": list(VERTICAL_ANCHOR_Z_BODY_M),
        "range_bin_count": RANGE_BIN_COUNT,
        "range_bin_size_m": RANGE_BIN_SIZE_M,
        "range_interval_m": [0.0, RANGE_LIMIT_M],
        "range_interval_convention": "left_closed_right_open",
        "angular_bin_count": ANGULAR_BIN_COUNT,
        "angular_interval_rad": [-HALF_FOV_RAD, HALF_FOV_RAD],
        "angular_interval_convention": "closed_positive_edge_in_final_bin",
        "cartesian_sample": "cell_center",
        "unsupported_mapping_sentinel": [-1, -1],
        "mapping_dtype": "little_endian_signed_int16",
        "support_mask_dtype": "row_major_uint8",
        "class_order": list(CLASS_NAMES),
        "class_ids": list(CLASS_IDS),
        "family_order": list(FAMILIES),
        "endpoint_side_order": list(ENDPOINT_SIDES),
    }


def _cell_center(row: int, column: int) -> tuple[float, float]:
    if not 0 <= int(row) < CARTESIAN_SHAPE[0]:
        raise IndexError("Cartesian row is outside [0,64)")
    if not 0 <= int(column) < CARTESIAN_SHAPE[1]:
        raise IndexError("Cartesian column is outside [0,64)")
    forward_m = CARTESIAN_FORWARD_MIN_EDGE_M + (
        int(row) + 0.5
    ) * CARTESIAN_CELL_SIZE_M
    left_m = CARTESIAN_LEFT_MIN_EDGE_M + (
        int(column) + 0.5
    ) * CARTESIAN_CELL_SIZE_M
    return forward_m, left_m


def camera_point_to_bin(forward_body_m: float, left_body_m: float) -> tuple[int, int] | None:
    """Map one body-frame point through the frozen camera-centered lattice."""

    forward = float(forward_body_m)
    left = float(left_body_m)
    if not math.isfinite(forward) or not math.isfinite(left):
        raise ValueError("camera mapping requires finite point coordinates")
    forward_camera = forward - CAMERA_XYZ_BODY_M[0]
    left_camera = left
    range_m = math.hypot(forward_camera, left_camera)
    bearing_rad = math.atan2(left_camera, forward_camera)
    if forward_camera < CAMERA_NEAR_M:
        return None
    if not 0.0 <= range_m < RANGE_LIMIT_M:
        return None
    if not -HALF_FOV_RAD <= bearing_rad <= HALF_FOV_RAD:
        return None
    range_bin = int(math.floor(range_m / RANGE_BIN_SIZE_M))
    angular_fraction = (bearing_rad + HALF_FOV_RAD) / (2.0 * HALF_FOV_RAD)
    angular_bin = min(
        ANGULAR_BIN_COUNT - 1,
        int(math.floor(angular_fraction * ANGULAR_BIN_COUNT)),
    )
    return range_bin, angular_bin


def build_camera_centered_mapping() -> np.ndarray:
    """Build exact ``[64,64,2]`` int16 row-major Cartesian mapping."""

    mapping = np.full((*CARTESIAN_SHAPE, 2), -1, dtype=np.int16)
    for row in range(CARTESIAN_SHAPE[0]):
        for column in range(CARTESIAN_SHAPE[1]):
            polar_bin = camera_point_to_bin(*_cell_center(row, column))
            if polar_bin is not None:
                mapping[row, column] = polar_bin
    return mapping


def _mapping_array(mapping: np.ndarray) -> np.ndarray:
    array = np.asarray(mapping)
    if array.shape != (*CARTESIAN_SHAPE, 2):
        raise ValueError(f"mapping must have shape {(*CARTESIAN_SHAPE, 2)}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("mapping must use an integer dtype")
    values = np.asarray(array, dtype=np.int64)
    if np.any(values < np.iinfo(np.int16).min) or np.any(
        values > np.iinfo(np.int16).max
    ):
        raise ValueError("mapping values must fit signed int16")
    return array


def camera_centered_support_mask(mapping: np.ndarray | None = None) -> np.ndarray:
    """Return the exact support mask implied by a mapping."""

    array = build_camera_centered_mapping() if mapping is None else _mapping_array(mapping)
    return np.all(np.asarray(array, dtype=np.int64) >= 0, axis=-1)


def _mapping_sha256(mapping: np.ndarray) -> str:
    canonical = np.ascontiguousarray(mapping, dtype="<i2")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _support_sha256(mask: np.ndarray) -> str:
    canonical = np.ascontiguousarray(mask, dtype=np.uint8)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def audit_camera_centered_mapping(
    mapping: np.ndarray | None = None,
) -> dict[str, Any]:
    """Report every frozen mapping invariant without hiding mutations."""

    supplied = build_camera_centered_mapping() if mapping is None else _mapping_array(mapping)
    values = np.asarray(supplied, dtype=np.int64)
    expected = build_camera_centered_mapping()
    expected_values = np.asarray(expected, dtype=np.int64)
    radial = values[..., 0]
    angular = values[..., 1]
    partial = (radial == -1) != (angular == -1)
    invalid_negative = (radial < -1) | (angular < -1)
    claims_mapping = (radial >= 0) & (angular >= 0)
    out_of_range = claims_mapping & (
        (radial >= RANGE_BIN_COUNT) | (angular >= ANGULAR_BIN_COUNT)
    )
    valid_mapping = claims_mapping & ~out_of_range
    support = claims_mapping

    flat_bins = (
        radial[valid_mapping] * ANGULAR_BIN_COUNT + angular[valid_mapping]
    ).astype(np.int64, copy=False)
    unique_bins, counts = np.unique(flat_bins, return_counts=True)
    collisions = []
    for flat_bin, multiplicity in zip(unique_bins, counts):
        if int(multiplicity) <= 1:
            continue
        range_bin, angular_bin = divmod(int(flat_bin), ANGULAR_BIN_COUNT)
        locations = np.argwhere(
            valid_mapping
            & (radial == range_bin)
            & (angular == angular_bin)
        )
        collisions.append(
            {
                "range_bin": range_bin,
                "angular_bin": angular_bin,
                "multiplicity": int(multiplicity),
                "cartesian_locations": locations.astype(int).tolist(),
            }
        )

    deterministic_mismatch = np.any(values != expected_values, axis=-1)
    expected_support = np.all(expected_values >= 0, axis=-1)
    supported_count = int(np.count_nonzero(support))
    collision_extra = sum(record["multiplicity"] - 1 for record in collisions)
    signed_int16 = supplied.dtype == np.dtype(np.int16)
    deterministic = not bool(np.any(deterministic_mismatch))
    in_range = not bool(np.any(out_of_range | invalid_negative))
    complete_entries = not bool(np.any(partial))
    injective = not collisions
    support_matches = bool(np.array_equal(support, expected_support))
    passes = bool(
        signed_int16
        and deterministic
        and in_range
        and complete_entries
        and injective
        and support_matches
    )
    return {
        "schema": MAPPING_AUDIT_SCHEMA,
        "mapping_sha256": _mapping_sha256(supplied),
        "support_mask_sha256": _support_sha256(support),
        "mapping_dtype": str(supplied.dtype),
        "signed_int16": signed_int16,
        "supported_cartesian_cell_count": supported_count,
        "unsupported_cartesian_cell_count": int(np.size(support) - supported_count),
        "unique_used_polar_bin_count": int(unique_bins.size),
        "unused_polar_bin_count": int(POLAR_BIN_COUNT - unique_bins.size),
        "partially_mapped_entry_count": int(np.count_nonzero(partial)),
        "invalid_negative_entry_count": int(np.count_nonzero(invalid_negative)),
        "out_of_range_entry_count": int(np.count_nonzero(out_of_range)),
        "nondeterministic_entry_count": int(np.count_nonzero(deterministic_mismatch)),
        "expected_support_mismatch_count": int(
            np.count_nonzero(support != expected_support)
        ),
        "collision_bin_count": len(collisions),
        "collision_extra_cartesian_count": int(collision_extra),
        "collisions": collisions,
        "deterministic": deterministic,
        "all_mapped_indices_in_range": in_range,
        "all_entries_complete": complete_entries,
        "support_matches_frozen_geometry": support_matches,
        "injective": injective,
        "passes": passes,
    }


def _old_column_records(*, require_vertical_anchor: bool) -> list[dict[str, Any]]:
    radial_centers = tuple(
        (index + 0.5) * RANGE_BIN_SIZE_M for index in range(RANGE_BIN_COUNT)
    )
    body_bearings = tuple(
        -HALF_FOV_RAD + (index + 0.5) * ANGULAR_BIN_WIDTH_RAD
        for index in range(ANGULAR_BIN_COUNT)
    )
    tan_vertical = math.tan(math.radians(VERTICAL_FOV_DEG) * 0.5)
    records = []
    for angular_bin, body_bearing in enumerate(body_bearings):
        selected = []
        cos_bearing = math.cos(body_bearing)
        sin_bearing = math.sin(body_bearing)
        for radius_m in radial_centers:
            forward_camera = radius_m * cos_bearing - CAMERA_XYZ_BODY_M[0]
            left_camera = radius_m * sin_bearing
            camera_bearing = math.atan2(left_camera, forward_camera)
            participates = (
                forward_camera >= CAMERA_NEAR_M
                and -HALF_FOV_RAD <= camera_bearing <= HALF_FOV_RAD
            )
            if participates and require_vertical_anchor:
                participates = any(
                    -1.0
                    <= -(anchor_z - CAMERA_XYZ_BODY_M[2])
                    / (forward_camera * tan_vertical)
                    <= 1.0
                    for anchor_z in VERTICAL_ANCHOR_Z_BODY_M
                )
            if participates:
                selected.append(camera_bearing)
        if len(selected) >= 2:
            minimum = min(selected)
            maximum = max(selected)
            span_rad = maximum - minimum
            span_deg = math.degrees(span_rad)
            span_bins = span_rad / ANGULAR_BIN_WIDTH_RAD
        else:
            minimum = maximum = span_rad = span_deg = span_bins = None
        records.append(
            {
                "body_angular_bin": angular_bin,
                "body_bearing_center_rad": body_bearing,
                "body_bearing_center_deg": math.degrees(body_bearing),
                "participating_range_count": len(selected),
                "minimum_camera_bearing_rad": minimum,
                "maximum_camera_bearing_rad": maximum,
                "span_rad": span_rad,
                "span_deg": span_deg,
                "span_new_angular_bins": span_bins,
            }
        )
    return records


def _span_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spans = np.asarray(
        [record["span_rad"] for record in records if record["span_rad"] is not None],
        dtype=np.float64,
    )
    if not spans.size or not np.isfinite(spans).all():
        raise ValueError("old-column span audit produced no finite spans")
    span_bins = spans / ANGULAR_BIN_WIDTH_RAD
    return {
        "column_count": len(records),
        "participating_sample_count": sum(
            int(record["participating_range_count"]) for record in records
        ),
        "columns_with_span_count": int(spans.size),
        "columns_with_fewer_than_two_participants_count": int(len(records) - spans.size),
        "span_rad": {
            "p50": float(np.quantile(spans, 0.50, method="linear")),
            "p95": float(np.quantile(spans, 0.95, method="linear")),
            "maximum": float(np.max(spans)),
        },
        "span_deg": {
            "p50": math.degrees(float(np.quantile(spans, 0.50, method="linear"))),
            "p95": math.degrees(float(np.quantile(spans, 0.95, method="linear"))),
            "maximum": math.degrees(float(np.max(spans))),
        },
        "span_new_angular_bins": {
            "p50": float(np.quantile(span_bins, 0.50, method="linear")),
            "p95": float(np.quantile(span_bins, 0.95, method="linear")),
            "maximum": float(np.max(span_bins)),
        },
        "columns_span_ge_1_new_bin": int(np.count_nonzero(span_bins >= 1.0)),
        "columns_span_ge_2_new_bins": int(np.count_nonzero(span_bins >= 2.0)),
        "columns_span_ge_4_new_bins": int(np.count_nonzero(span_bins >= 4.0)),
        "columns_span_ge_8_new_bins": int(np.count_nonzero(span_bins >= 8.0)),
        "quantile_method": "numpy_linear_float64",
    }


def old_body_column_span_audit() -> dict[str, Any]:
    """Quantify camera-bearing variation inside every old body-bearing column."""

    primary_columns = _old_column_records(require_vertical_anchor=True)
    horizontal_columns = _old_column_records(require_vertical_anchor=False)
    span_table = {
        "primary_with_vertical_anchor": primary_columns,
        "horizontal_only": horizontal_columns,
    }
    return {
        "schema": OLD_COLUMN_SPAN_SCHEMA,
        "geometry": {
            "old_body_radius_centers_m": [
                float((index + 0.5) * RANGE_BIN_SIZE_M)
                for index in range(RANGE_BIN_COUNT)
            ],
            "old_body_bearing_bin_count": ANGULAR_BIN_COUNT,
            "old_body_bearing_range_rad": [-HALF_FOV_RAD, HALF_FOV_RAD],
            "new_angular_bin_width_rad": ANGULAR_BIN_WIDTH_RAD,
            "primary_vertical_anchor_rule": "at_least_one_registered_anchor_valid",
        },
        "primary": {
            "columns": primary_columns,
            "summary": _span_summary(primary_columns),
        },
        "horizontal_only": {
            "columns": horizontal_columns,
            "summary": _span_summary(horizontal_columns),
        },
        "old_column_span_table_sha256": canonical_json_sha256(span_table),
    }


def _validated_target(target: np.ndarray) -> np.ndarray:
    array = np.asarray(target)
    if array.shape != CARTESIAN_SHAPE:
        raise ValueError(f"target must have shape {CARTESIAN_SHAPE}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("target must use an integer dtype")
    if np.any((array < UNKNOWN_CLASS) | (array > OCCUPIED_CLASS)):
        raise ValueError("target contains values outside registered classes 0/1/2")
    return np.asarray(array, dtype=np.uint8)


def _validate_full_supervision_mask(mask: np.ndarray) -> None:
    array = np.asarray(mask)
    if array.shape != CARTESIAN_SHAPE:
        raise ValueError(f"supervision mask must have shape {CARTESIAN_SHAPE}")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("supervision mask must be numeric or boolean")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError("supervision mask must be finite")
    if not np.all(array == 1):
        raise ValueError("supervision mask must supervise the full 64 x 64 grid")


def _count_classes(target: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.count_nonzero(mask & (target == class_id)))
        for name, class_id in zip(CLASS_NAMES, CLASS_IDS)
    }


def _label_support_report(
    target: np.ndarray,
    support: np.ndarray,
    *,
    frame_key: Mapping[str, Any],
) -> dict[str, Any]:
    supported = np.asarray(support, dtype=bool)
    unsupported = ~supported
    by_class = {}
    for class_name, class_id in zip(CLASS_NAMES, CLASS_IDS):
        class_mask = target == class_id
        by_class[class_name] = {
            "total": int(np.count_nonzero(class_mask)),
            "supported": int(np.count_nonzero(class_mask & supported)),
            "unsupported": int(np.count_nonzero(class_mask & unsupported)),
        }
    violations = []
    for row, column in np.argwhere(
        unsupported & ((target == FREE_CLASS) | (target == OCCUPIED_CLASS))
    ):
        class_id = int(target[row, column])
        violations.append(
            {
                "frame_key": dict(frame_key),
                "row": int(row),
                "column": int(column),
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            }
        )
    unsupported_free = by_class["free"]["unsupported"]
    unsupported_occupied = by_class["occupied"]["unsupported"]
    unsupported_unknown = by_class["unknown"]["unsupported"]
    unsupported_count = int(np.count_nonzero(unsupported))
    passes = unsupported_free == 0 and unsupported_occupied == 0 and (
        unsupported_unknown == unsupported_count
    )
    return {
        "schema": LABEL_SUPPORT_SCHEMA,
        "total_supervised_label_count": int(target.size),
        "supported_label_count": int(np.count_nonzero(supported)),
        "unsupported_label_count": unsupported_count,
        "class_counts": _count_classes(target, np.ones_like(supported, dtype=bool)),
        "by_class": by_class,
        "unsupported_free_count": unsupported_free,
        "unsupported_occupied_count": unsupported_occupied,
        "unsupported_unknown_count": unsupported_unknown,
        "unsupported_targets_are_all_unknown": unsupported_unknown == unsupported_count,
        "violations": violations,
        "passes": passes,
    }


def _transition_name(source: int, destination: int) -> str:
    return f"{CLASS_NAMES[source]}_to_{CLASS_NAMES[destination]}"


def _collapse_classes(classes: Sequence[int]) -> list[int]:
    result: list[int] = []
    for value in classes:
        class_id = int(value)
        if not result or class_id != result[-1]:
            result.append(class_id)
    return result


def _ray_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    length_histogram = Counter(str(int(record["length"])) for record in records)
    transition_histogram = Counter(
        str(int(record["transition_count"])) for record in records
    )
    transition_counts = Counter({name: 0 for name in _TRANSITION_NAMES})
    for record in records:
        transition_counts.update(record["directed_unequal_transition_counts"])
    sequence_count = len(records)
    eligible = sum(int(record["length"]) >= 2 for record in records)
    transition_event_count = sum(
        int(record["transition_count"]) for record in records
    )
    bucket_counts = {
        "0": sum(int(record["transition_count"]) == 0 for record in records),
        "1": sum(int(record["transition_count"]) == 1 for record in records),
        "2": sum(int(record["transition_count"]) == 2 for record in records),
        "3_plus": sum(int(record["transition_count"]) >= 3 for record in records),
    }
    return {
        "sequence_count": sequence_count,
        "length_histogram": dict(sorted(length_histogram.items(), key=lambda item: int(item[0]))),
        "sequences_with_fewer_than_two_cells_count": sequence_count - eligible,
        "transition_rate_eligible_sequence_count": eligible,
        "class_transition_histogram": dict(
            sorted(transition_histogram.items(), key=lambda item: int(item[0]))
        ),
        "maximum_transitions_per_sequence": max(
            (int(record["transition_count"]) for record in records), default=0
        ),
        "directed_unequal_transition_counts": {
            name: int(transition_counts[name]) for name in _TRANSITION_NAMES
        },
        "transition_bucket_counts": bucket_counts,
        "transition_event_count": transition_event_count,
        "transition_events_per_eligible_sequence": (
            float(transition_event_count / eligible) if eligible else None
        ),
        "contains_known_after_unknown_count": sum(
            bool(record["contains_known_after_unknown"]) for record in records
        ),
        "contains_free_after_occupied_count": sum(
            bool(record["contains_free_after_occupied"]) for record in records
        ),
        "scalar_first_hit_irregular_count": sum(
            not bool(record["scalar_first_hit_regular"]) for record in records
        ),
        "scalar_first_hit_regular_count": sum(
            bool(record["scalar_first_hit_regular"]) for record in records
        ),
    }


def _ray_sequence_records(
    target: np.ndarray,
    mapping: np.ndarray,
    *,
    frame_key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = []
    for angular_bin in range(ANGULAR_BIN_COUNT):
        locations = np.argwhere(mapping[..., 1] == angular_bin)
        ordered = sorted(
            (
                int(mapping[row, column, 0]),
                int(row),
                int(column),
            )
            for row, column in locations
        )
        range_bins = [record[0] for record in ordered]
        if len(range_bins) != len(set(range_bins)):
            raise ValueError("ray sequence contains a range-bin tie")
        classes = [int(target[row, column]) for _, row, column in ordered]
        collapsed = _collapse_classes(classes)
        directed = Counter({name: 0 for name in _TRANSITION_NAMES})
        for source, destination in zip(collapsed, collapsed[1:]):
            directed[_transition_name(source, destination)] += 1
        unknown_positions = [index for index, value in enumerate(classes) if value == UNKNOWN_CLASS]
        known_positions = [index for index, value in enumerate(classes) if value != UNKNOWN_CLASS]
        occupied_positions = [index for index, value in enumerate(classes) if value == OCCUPIED_CLASS]
        free_positions = [index for index, value in enumerate(classes) if value == FREE_CLASS]
        contains_known_after_unknown = bool(
            unknown_positions
            and known_positions
            and min(unknown_positions) < max(known_positions)
        )
        contains_free_after_occupied = bool(
            occupied_positions
            and free_positions
            and min(occupied_positions) < max(free_positions)
        )
        regular_ranks = {
            FREE_CLASS: 0,
            OCCUPIED_CLASS: 1,
            UNKNOWN_CLASS: 2,
        }
        scalar_regular = all(
            regular_ranks[source] <= regular_ranks[destination]
            for source, destination in zip(classes, classes[1:])
        )
        records.append(
            {
                "frame_key": dict(frame_key),
                "angular_bin": angular_bin,
                "length": len(classes),
                "range_bins": range_bins,
                "class_sequence": classes,
                "collapsed_class_sequence": collapsed,
                "transition_count": max(0, len(collapsed) - 1),
                "directed_unequal_transition_counts": {
                    name: int(directed[name]) for name in _TRANSITION_NAMES
                },
                "contains_known_after_unknown": contains_known_after_unknown,
                "contains_free_after_occupied": contains_free_after_occupied,
                "scalar_first_hit_regular": scalar_regular,
            }
        )
    return records


def analyze_frame_labels(
    target: np.ndarray,
    supervision_mask: np.ndarray,
    *,
    frame_key: Mapping[str, Any],
    family: str,
    endpoint_side: str,
    mapping: np.ndarray | None = None,
) -> dict[str, Any]:
    """Analyze one selected fit target without opening any external bytes."""

    if family not in FAMILIES:
        raise ValueError("frame family is not registered")
    if endpoint_side not in ENDPOINT_SIDES:
        raise ValueError("frame endpoint side is not registered")
    key = _canonical_json_copy(frame_key, name="frame_key")
    labels = _validated_target(target)
    _validate_full_supervision_mask(supervision_mask)
    polar_mapping = build_camera_centered_mapping() if mapping is None else _mapping_array(mapping)
    mapping_audit = audit_camera_centered_mapping(polar_mapping)
    if not mapping_audit["passes"]:
        raise ValueError("frame analysis requires the exact passing camera mapping")
    support = camera_centered_support_mask(polar_mapping)
    label_support = _label_support_report(labels, support, frame_key=key)
    ray_records = _ray_sequence_records(
        labels,
        np.asarray(polar_mapping, dtype=np.int64),
        frame_key=key,
    )
    transition_table = _ray_summary(ray_records)
    return {
        "schema": FRAME_ANALYSIS_SCHEMA,
        "frame_key": key,
        "family": family,
        "endpoint_side": endpoint_side,
        "label_support": label_support,
        "ray_sequences": {
            "schema": RAY_SEQUENCE_SCHEMA,
            "records": ray_records,
            "summary": transition_table,
            "sequence_summary_records_sha256": canonical_json_sha256(ray_records),
            "transition_table_sha256": canonical_json_sha256(transition_table),
        },
    }


def _sum_label_support(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class = {
        name: {"total": 0, "supported": 0, "unsupported": 0}
        for name in CLASS_NAMES
    }
    violations: list[dict[str, Any]] = []
    for report in reports:
        support = report["label_support"]
        for name in CLASS_NAMES:
            for field in ("total", "supported", "unsupported"):
                by_class[name][field] += int(support["by_class"][name][field])
        violations.extend(support["violations"])
    total = sum(record["total"] for record in by_class.values())
    supported = sum(record["supported"] for record in by_class.values())
    unsupported = sum(record["unsupported"] for record in by_class.values())
    class_counts = {name: record["total"] for name, record in by_class.items()}
    unsupported_free = by_class["free"]["unsupported"]
    unsupported_occupied = by_class["occupied"]["unsupported"]
    unsupported_unknown = by_class["unknown"]["unsupported"]
    passes = (
        unsupported_free == 0
        and unsupported_occupied == 0
        and unsupported_unknown == unsupported
    )
    return {
        "schema": LABEL_SUPPORT_SCHEMA,
        "frame_count": len(reports),
        "total_supervised_label_count": total,
        "supported_label_count": supported,
        "unsupported_label_count": unsupported,
        "class_counts": class_counts,
        "by_class": by_class,
        "unsupported_free_count": unsupported_free,
        "unsupported_occupied_count": unsupported_occupied,
        "unsupported_unknown_count": unsupported_unknown,
        "unsupported_targets_are_all_unknown": unsupported_unknown == unsupported,
        "violations": violations,
        "passes": passes,
    }


def _sum_ray_records(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [
        record
        for report in reports
        for record in report["ray_sequences"]["records"]
    ]
    return {
        "schema": RAY_SEQUENCE_SCHEMA,
        "summary": _ray_summary(records),
        "sequence_summary_records_sha256": canonical_json_sha256(records),
    }


def _scope_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "frame_count": len(reports),
        "label_support": _sum_label_support(reports),
        "ray_sequences": _sum_ray_records(reports),
    }


def aggregate_label_observability(
    frame_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate canonical frame reports by frozen family and endpoint side."""

    reports = tuple(frame_reports)
    if not reports:
        raise ValueError("label observability requires at least one frame report")
    canonical_keys = []
    for report in reports:
        if not isinstance(report, Mapping) or report.get("schema") != FRAME_ANALYSIS_SCHEMA:
            raise ValueError("label observability received a malformed frame report")
        if report.get("family") not in FAMILIES:
            raise ValueError("frame report family is not registered")
        if report.get("endpoint_side") not in ENDPOINT_SIDES:
            raise ValueError("frame report endpoint side is not registered")
        records = report.get("ray_sequences", {}).get("records")
        if not isinstance(records, list) or len(records) != ANGULAR_BIN_COUNT:
            raise ValueError("frame report must contain exactly 256 ray records")
        key = _canonical_json_copy(report.get("frame_key"), name="frame_key")
        canonical_keys.append(canonical_json_sha256(key))
    if len(set(canonical_keys)) != len(canonical_keys):
        raise ValueError("label observability frame keys must be unique")

    family_reports = {
        family: tuple(report for report in reports if report["family"] == family)
        for family in FAMILIES
    }
    side_reports = {
        side: tuple(report for report in reports if report["endpoint_side"] == side)
        for side in ENDPOINT_SIDES
    }
    if any(not values for values in family_reports.values()):
        raise ValueError("label observability requires every registered family")
    if any(not values for values in side_reports.values()):
        raise ValueError("label observability requires both endpoint sides")

    aggregate = _scope_summary(reports)
    families = {
        family: _scope_summary(family_reports[family]) for family in FAMILIES
    }
    endpoint_sides = {
        side: _scope_summary(side_reports[side]) for side in ENDPOINT_SIDES
    }
    transition_tables = {
        "aggregate": aggregate["ray_sequences"]["summary"],
        "families": {
            family: families[family]["ray_sequences"]["summary"]
            for family in FAMILIES
        },
        "endpoint_sides": {
            side: endpoint_sides[side]["ray_sequences"]["summary"]
            for side in ENDPOINT_SIDES
        },
    }
    family_gate = {
        family: bool(families[family]["label_support"]["passes"])
        for family in FAMILIES
    }
    coverage_passes = bool(aggregate["label_support"]["passes"]) and all(
        family_gate.values()
    )
    all_ray_records = [
        record
        for report in reports
        for record in report["ray_sequences"]["records"]
    ]
    return {
        "schema": OBSERVABILITY_SUMMARY_SCHEMA,
        "frame_count": len(reports),
        "family_order": list(FAMILIES),
        "endpoint_side_order": list(ENDPOINT_SIDES),
        "aggregate": aggregate,
        "families": families,
        "endpoint_sides": endpoint_sides,
        "fit_known_target_coverage_gate": {
            "aggregate_passes": bool(aggregate["label_support"]["passes"]),
            "family_passes": family_gate,
            "requires_aggregate_and_all_families": True,
            "passes": coverage_passes,
        },
        "ordered_sequence_summary_records_sha256": canonical_json_sha256(
            all_ray_records
        ),
        "aggregate_transition_tables_sha256": canonical_json_sha256(
            transition_tables
        ),
    }


def _strict_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def authorization_decision(
    *,
    provenance_passes: bool,
    source_hashes_pass: bool,
    reconstruction_passes: bool,
    access_reconciliation_passes: bool,
    mapping_audit: Mapping[str, Any],
    label_observability: Mapping[str, Any],
    rendered_collision_target_ambiguity: bool,
) -> dict[str, Any]:
    """Apply the frozen representation-only authorization decision."""

    provenance = _strict_bool(provenance_passes, name="provenance_passes")
    sources = _strict_bool(source_hashes_pass, name="source_hashes_pass")
    reconstruction = _strict_bool(
        reconstruction_passes, name="reconstruction_passes"
    )
    access = _strict_bool(
        access_reconciliation_passes, name="access_reconciliation_passes"
    )
    ambiguity = _strict_bool(
        rendered_collision_target_ambiguity,
        name="rendered_collision_target_ambiguity",
    )
    if not isinstance(mapping_audit, Mapping) or mapping_audit.get("schema") != MAPPING_AUDIT_SCHEMA:
        raise ValueError("authorization requires the registered mapping audit")
    if not isinstance(mapping_audit.get("passes"), bool):
        raise ValueError("mapping audit lacks a strict pass decision")
    if (
        not isinstance(label_observability, Mapping)
        or label_observability.get("schema") != OBSERVABILITY_SUMMARY_SCHEMA
    ):
        raise ValueError("authorization requires the registered observability summary")
    coverage_gate = label_observability.get("fit_known_target_coverage_gate")
    if not isinstance(coverage_gate, Mapping) or not isinstance(
        coverage_gate.get("passes"), bool
    ):
        raise ValueError("observability summary lacks a strict coverage decision")
    mapping_passes = bool(mapping_audit["passes"])
    coverage_passes = bool(coverage_gate["passes"])
    authorized = bool(
        provenance
        and sources
        and reconstruction
        and access
        and mapping_passes
        and coverage_passes
    )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "provenance_passes": provenance,
        "source_hashes_pass": sources,
        "reconstruction_passes": reconstruction,
        "access_reconciliation_passes": access,
        "camera_centered_mapping_passes": mapping_passes,
        "fit_known_target_coverage_passes": coverage_passes,
        "rendered_collision_target_ambiguity": ambiguity,
        "camera_frustum_representation_implementation_authorized": authorized,
        "target_amendment_required_before_model_output": ambiguity,
        "trained_model_output_authorized": False,
        "holdout_access_authorized": False,
        "seed_20260711_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }


__all__ = [
    "ANGULAR_BIN_COUNT",
    "AUTHORIZATION_SCHEMA",
    "CAMERA_NEAR_M",
    "CAMERA_XYZ_BODY_M",
    "CARTESIAN_SHAPE",
    "CLASS_IDS",
    "CLASS_NAMES",
    "ENDPOINT_SIDES",
    "EXECUTION_BINDING_SHA256",
    "FAMILIES",
    "FREE_CLASS",
    "GEOMETRY_SCHEMA",
    "HALF_FOV_RAD",
    "HORIZONTAL_FOV_DEG",
    "LABEL_SUPPORT_SCHEMA",
    "MAPPING_AUDIT_SCHEMA",
    "OBSERVABILITY_SUMMARY_SCHEMA",
    "OCCUPIED_CLASS",
    "OLD_COLUMN_SPAN_SCHEMA",
    "RANGE_BIN_COUNT",
    "RESULT_SCHEMA",
    "UNKNOWN_CLASS",
    "aggregate_label_observability",
    "analyze_frame_labels",
    "audit_camera_centered_mapping",
    "authorization_decision",
    "build_camera_centered_mapping",
    "camera_centered_support_mask",
    "camera_point_to_bin",
    "canonical_json_sha256",
    "geometry_contract",
    "old_body_column_span_audit",
]
