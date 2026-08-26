"""Pure metrics/reporting support for minimum multi-origin range coverage V1.

This module is intentionally independent of corpus and simulator I/O.  It
adds only the operations which differ from the predecessor single-origin
qualification: label-free training-layout selection, multi-origin condition
accounting, error-class accounting, and complete-set/per-state timing.  The
contact, calibration, two-ply, H3, and immutable gate functions are aliases to
the predecessor implementations, so their scientific semantics cannot drift.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import resource
import time
from typing import Any

import numpy as np

from lewm.safety import body_centric_range_coverage_metrics_v1 as _BASE
from lewm.safety import body_centric_range_coverage_reporting_v1 as _REPORTING


# Exact predecessor API reuse.  These are assignments, not reimplementations.
contact_predictions = _BASE.contact_predictions
contact_confusion = _BASE.contact_confusion
contact_summary = _BASE.contact_summary
current_successor_contact_summaries = _BASE.current_successor_contact_summaries
grouped_contact_summaries = _BASE.grouped_contact_summaries
enumerate_threshold_frontier = _BASE.enumerate_threshold_frontier
reduce_two_ply_state = _BASE.reduce_two_ply_state
reduce_two_ply_states = _BASE.reduce_two_ply_states
safe_action_count_summary = _BASE.safe_action_count_summary
evaluate_true_future_gate = _REPORTING.evaluate_true_future_gate


PHYSICS_STEPS = 50
PROTECTED_LINKS = 13
REGION_IDS = (
    "TRUNK",
    "FRONT_LIMBS",
    "REAR_LIMBS",
    "HIPS_AND_THIGHS",
    "CALVES",
)
DUAL_LAYOUT_IDS = (
    "HEAD_STOCK__REAR_TOP_TRUNK",
    "HEAD_STOCK__LEFT_UPPER_FLANK",
    "HEAD_STOCK__RIGHT_UPPER_FLANK",
)
TRIPLE_LAYOUT_IDS = (
    "HEAD_STOCK__REAR_TOP_TRUNK__LEFT_UPPER_FLANK",
    "HEAD_STOCK__REAR_TOP_TRUNK__RIGHT_UPPER_FLANK",
    "HEAD_STOCK__LEFT_UPPER_FLANK__RIGHT_UPPER_FLANK",
)
LAYOUT_IDS = DUAL_LAYOUT_IDS + TRIPLE_LAYOUT_IDS

REGRESSION_CONDITION_IDS = (
    "REALISTIC_PLATFORM_SCAN",
    "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
)
DUAL_CONDITION_IDS = (
    "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
    "DUAL_DENSE_L2_FOV_UPPER_BOUND",
    "DUAL_REALISTIC_L2_SCAN",
)
TRIPLE_CONDITION_IDS = (
    "THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND",
    "THREE_DENSE_L2_FOV_UPPER_BOUND",
    "THREE_REALISTIC_L2_SCAN",
)
ALL_FOUR_CONDITION_ID = "ALL_FOUR_DENSE_SPHERICAL_DIAGNOSTIC"
MULTI_ORIGIN_CONDITION_IDS = DUAL_CONDITION_IDS + TRIPLE_CONDITION_IDS + (
    ALL_FOUR_CONDITION_ID,
)
CONDITION_IDS = REGRESSION_CONDITION_IDS + MULTI_ORIGIN_CONDITION_IDS

PRIMARY_CLASSIFICATIONS = (
    "DUAL_ORIGIN_REALISTIC_RANGE_SIGNAL",
    "THREE_ORIGIN_REALISTIC_RANGE_SIGNAL",
    "DUAL_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK",
    "THREE_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK",
    "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO",
)
FOUR_OR_MORE_DIAGNOSTIC = "FOUR_OR_MORE_ORIGINS_THEORETICALLY_REQUIRED"
REPLANNING_INTERFACE_CLASSIFICATION = "REPLANNING_INTERFACE_UNRESOLVED"

ERROR_CLASSES = (
    "INSUFFICIENT_ORIGIN_COUNT",
    "VERTICAL_FOV_LIMITATION",
    "SCAN_PATTERN_SPARSITY",
    "SCAN_TIMING_LIMITATION",
    "ROBOT_SELF_OCCLUSION",
    "NEAR_BLIND_REGION",
    "MOUNT_POSITION_LIMITATION",
    "POINT_FUSION_ERROR",
    "UNRESOLVED",
)

BENCHMARK_CLASSIFICATIONS = (
    "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL",
    "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY",
    "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO",
)


class MultiOriginMetricsError(ValueError):
    """An incomplete, ambiguous, or non-finite pure-metric input."""


def _boolean_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return array.astype(bool, copy=False)
    if array.dtype.kind not in "iu" or not np.isin(array, (0, 1)).all():
        raise MultiOriginMetricsError(f"{name} must contain only booleans")
    return array.astype(bool)


def union_origin_support(
    origin_support: Mapping[str, Any], origin_ids: Sequence[str]
) -> np.ndarray:
    """Return the exact Boolean union for one prospectively fixed layout."""

    if not origin_ids or len(set(origin_ids)) != len(origin_ids):
        raise MultiOriginMetricsError("origin_ids must be nonempty and unique")
    unknown = set(origin_ids) - set(origin_support)
    if unknown:
        raise MultiOriginMetricsError(f"origin support missing {sorted(unknown)}")
    result: np.ndarray | None = None
    for origin_id in origin_ids:
        value = _boolean_array(
            origin_support[origin_id], name=f"origin_support.{origin_id}"
        )
        if result is None:
            result = value.copy()
        elif value.shape != result.shape:
            raise MultiOriginMetricsError("all origin support arrays must share a shape")
        else:
            np.logical_or(result, value, out=result)
    assert result is not None
    return result


def fuse_origin_clearance(
    *,
    origin_clearance_m: Mapping[str, Any],
    origin_support: Mapping[str, Any],
    origin_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Fuse already-reduced origin evidence without treating absence as free.

    A union point cloud's minimum clearance is the minimum of the supported
    per-origin minima.  When no origin supports a query, clearance is ``inf``
    and ``unsupported`` is true, matching the predecessor contact convention.
    """

    support = union_origin_support(origin_support, origin_ids)
    stacked: list[np.ndarray] = []
    for origin_id in origin_ids:
        if origin_id not in origin_clearance_m:
            raise MultiOriginMetricsError(f"origin clearance missing {origin_id}")
        value = np.asarray(origin_clearance_m[origin_id], dtype=np.float64)
        local_support = _boolean_array(
            origin_support[origin_id], name=f"origin_support.{origin_id}"
        )
        if value.shape != support.shape:
            raise MultiOriginMetricsError("clearance and support shapes must match")
        if np.isnan(value[local_support]).any():
            raise MultiOriginMetricsError("supported clearance must not contain NaN")
        stacked.append(np.where(local_support, value, np.inf))
    clearance = np.minimum.reduce(stacked)
    unsupported = ~support | ~np.isfinite(clearance)
    clearance = np.where(unsupported, np.inf, clearance)
    return {
        "support": support,
        "unsupported": unsupported,
        "minimum_clearance_m": clearance,
    }


def reduce_prematerialized_per_link_contact(
    minimum_clearance_m: Any,
    support: Any,
    threshold_m: float,
    *,
    expected_steps: int = PHYSICS_STEPS,
    expected_links: int = PROTECTED_LINKS,
    require_float32: bool = True,
) -> dict[str, Any]:
    """Reduce one prematerialized protected sweep through the exact contact API."""

    clearance = np.asarray(minimum_clearance_m)
    observed = _boolean_array(support, name="support")
    expected_shape = (expected_steps, expected_links)
    if clearance.shape != expected_shape or observed.shape != expected_shape:
        raise MultiOriginMetricsError(
            f"per-link clearance/support must have shape {expected_shape}"
        )
    if require_float32 and clearance.dtype != np.dtype(np.float32):
        raise MultiOriginMetricsError("benchmark per-link clearance must be float32")
    numeric = clearance.astype(np.float64, copy=False)
    if np.isnan(numeric[observed]).any() or np.isinf(numeric[observed]).any():
        raise MultiOriginMetricsError("supported per-link clearance must be finite")
    supported_values = numeric[observed]
    global_minimum = (
        float(supported_values.min()) if len(supported_values) else math.inf
    )
    unsupported = not bool(observed.all())
    predicted = bool(
        contact_predictions(
            np.asarray([global_minimum]),
            threshold_m,
            unsupported=np.asarray([unsupported]),
        )[0]
    )
    return {
        "minimum_clearance_m": global_minimum,
        "unsupported": unsupported,
        "predicted_contact": predicted,
        "supported_witnesses": int(observed.sum()),
        "witnesses": int(observed.size),
        "numeric_dtype": str(clearance.dtype),
    }


def _validate_region_masks(
    masks: Mapping[str, Any], link_count: int
) -> dict[str, np.ndarray]:
    if set(masks) != set(REGION_IDS):
        raise MultiOriginMetricsError(
            f"region_link_masks must contain exactly {REGION_IDS}"
        )
    output: dict[str, np.ndarray] = {}
    for region_id in REGION_IDS:
        mask = _boolean_array(masks[region_id], name=f"region_link_masks.{region_id}")
        if mask.shape != (link_count,) or not mask.any():
            raise MultiOriginMetricsError(
                f"region {region_id} must select at least one of {link_count} links"
            )
        output[region_id] = mask
    if not np.logical_or.reduce(list(output.values())).all():
        raise MultiOriginMetricsError("every protected link must belong to a region")
    return output


def summarize_training_layout_support(
    *,
    layout_id: str,
    support: Any,
    nominal_fov: Any,
    self_occluded: Any,
    region_link_masks: Mapping[str, Any],
    transition_ids: Sequence[str],
    expected_steps: int = PHYSICS_STEPS,
    expected_links: int = PROTECTED_LINKS,
) -> dict[str, Any]:
    """Summarize the frozen label-free training-only layout score.

    Support uses the complete ``transition x physics-step x protected-link``
    closest-surface witness denominator.  The self-occlusion rate alone uses
    nominal-FOV target queries as its denominator.  No contact, action, route,
    or held-out field is accepted by this array-only interface.
    """

    if layout_id not in LAYOUT_IDS:
        raise MultiOriginMetricsError(f"unknown prospectively fixed layout {layout_id}")
    observed = _boolean_array(support, name="support")
    nominal = _boolean_array(nominal_fov, name="nominal_fov")
    occluded = _boolean_array(self_occluded, name="self_occluded")
    if observed.ndim != 3:
        raise MultiOriginMetricsError("support must have transition x step x link shape")
    expected_shape = (len(transition_ids), expected_steps, expected_links)
    if observed.shape != expected_shape:
        raise MultiOriginMetricsError(
            f"support shape {observed.shape} does not match {expected_shape}"
        )
    if nominal.shape != observed.shape or occluded.shape != observed.shape:
        raise MultiOriginMetricsError("nominal_fov/self_occluded must match support")
    if not transition_ids or len(set(map(str, transition_ids))) != len(transition_ids):
        raise MultiOriginMetricsError("transition_ids must be nonempty and unique")
    if np.any(occluded & ~nominal):
        raise MultiOriginMetricsError("self-occluded queries must be inside nominal FOV")
    if np.any(observed & ~nominal):
        raise MultiOriginMetricsError("direct support must be inside nominal FOV")
    if np.any(observed & occluded):
        raise MultiOriginMetricsError("a query cannot be both supported and self-occluded")
    nominal_count = int(nominal.sum())
    if nominal_count == 0:
        raise MultiOriginMetricsError("layout has no nominal-FOV target queries")
    masks = _validate_region_masks(region_link_masks, expected_links)
    region_support = {
        region_id: float(observed[..., masks[region_id]].mean())
        for region_id in REGION_IDS
    }
    transition_support = observed.mean(axis=(1, 2), dtype=np.float64)
    self_occlusion_fraction = float((occluded & nominal).sum() / nominal_count)
    transition_id_sha256 = hashlib.sha256(
        canonical_json_bytes(sorted(map(str, transition_ids)))
    ).hexdigest()
    return {
        "schema": "minimum_multi_origin_training_layout_support_v1",
        "role": "training",
        "label_free": True,
        "layout_id": layout_id,
        "origin_count": layout_id.count("__") + 1,
        "training_transition_count": len(transition_ids),
        "physics_steps": expected_steps,
        "protected_links": expected_links,
        "witness_queries": int(observed.size),
        "supported_witness_queries": int(observed.sum()),
        "nominal_fov_target_queries": nominal_count,
        "self_occluded_target_queries": int((occluded & nominal).sum()),
        "transition_ids_sha256": transition_id_sha256,
        "body_region_support": region_support,
        "min_body_region_support": min(region_support.values()),
        "p5_transition_support": float(
            np.percentile(transition_support, 5, method="linear")
        ),
        "rear_limb_support": region_support["REAR_LIMBS"],
        "calf_support": region_support["CALVES"],
        "overall_mean_support": float(observed.mean()),
        "self_occlusion_fraction": self_occlusion_fraction,
    }


_SCORE_FIELDS = (
    "min_body_region_support",
    "p5_transition_support",
    "rear_limb_support",
    "calf_support",
    "overall_mean_support",
    "self_occlusion_fraction",
)

_SCORE_ALIASES = {
    "min_body_region_support": (
        "min_body_region_support",
        "minimum_body_region_support",
    ),
    "p5_transition_support": (
        "p5_transition_support",
        "transition_support_p05",
    ),
    "rear_limb_support": ("rear_limb_support",),
    "calf_support": ("calf_support",),
    "overall_mean_support": ("overall_mean_support",),
    "self_occlusion_fraction": (
        "self_occlusion_fraction",
        "self_occluded_fraction",
    ),
}

_OUTCOME_KEY_FRAGMENTS = (
    "contact",
    "safe_action",
    "oracle",
    "route",
    "h3",
    "regret",
    "heldout",
    "held_out",
    "calibration",
)


def _reject_outcome_fields(value: Any, *, location: str = "score") -> None:
    """Reject outcome-bearing additions to the label-free selector payload."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _OUTCOME_KEY_FRAGMENTS):
                raise MultiOriginMetricsError(
                    f"{location}.{key} is forbidden in label-free layout selection"
                )
            _reject_outcome_fields(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_outcome_fields(item, location=f"{location}[{index}]")


def _score_key(score: Mapping[str, Any]) -> tuple[float, ...]:
    values: list[float] = []
    for field in _SCORE_FIELDS:
        aliases = _SCORE_ALIASES[field]
        present = [alias for alias in aliases if alias in score]
        if len(present) != 1:
            raise MultiOriginMetricsError(
                f"layout score requires exactly one alias for {field}: {aliases}"
            )
        value = score[present[0]]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise MultiOriginMetricsError(f"layout score lacks numeric {field}")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise MultiOriginMetricsError(f"layout score {field} must lie in [0,1]")
        values.append(numeric)
    return (*values[:5], -values[5])


def select_training_layout(
    scores: Sequence[Mapping[str, Any]], *, expected_layout_ids: Sequence[str]
) -> dict[str, Any]:
    """Select a dual or triple layout by the frozen training-only key."""

    expected = tuple(expected_layout_ids)
    if expected not in (DUAL_LAYOUT_IDS, TRIPLE_LAYOUT_IDS):
        raise MultiOriginMetricsError("expected_layout_ids must be a frozen dual/triple set")
    by_id: dict[str, Mapping[str, Any]] = {}
    for score in scores:
        _reject_outcome_fields(score)
        if "role" in score and score.get("role") not in ("training", "development_training"):
            raise MultiOriginMetricsError("layout selection requires training-role scores")
        if "label_free" in score and score.get("label_free") is not True:
            raise MultiOriginMetricsError("layout selection requires label-free scores")
        layout_id = str(score.get("layout_id", ""))
        if layout_id in by_id:
            raise MultiOriginMetricsError(f"duplicate layout score {layout_id}")
        _score_key(score)
        by_id[layout_id] = score
    if set(by_id) != set(expected):
        raise MultiOriginMetricsError("layout scores do not match the frozen candidate set")
    # Earlier fixed IDs win only after all six scientific criteria tie.
    fixed_rank = {layout_id: index for index, layout_id in enumerate(expected)}
    selected_id = max(
        expected,
        key=lambda layout_id: (_score_key(by_id[layout_id]), -fixed_rank[layout_id]),
    )
    candidate_rows = []
    for layout_id in expected:
        row = dict(by_id[layout_id])
        row["scientific_lexicographic_key"] = list(_score_key(row))
        row["fixed_layout_rank"] = fixed_rank[layout_id]
        candidate_rows.append(row)
    return {
        "schema": "minimum_multi_origin_training_layout_selection_v1",
        "selection_role": "training",
        "label_free": True,
        "candidate_layout_ids": list(expected),
        "selection_key": [
            "max_min_body_region_support",
            "max_p5_transition_support",
            "max_rear_limb_support",
            "max_calf_support",
            "max_overall_mean_support",
            "min_self_occlusion_fraction",
            "fixed_layout_id_order",
        ],
        # Both names are retained for the streaming evaluator and standalone
        # reporting API; they are necessarily identical.
        "layout_id": selected_id,
        "selected_layout_id": selected_id,
        "selected_score": dict(by_id[selected_id]),
        "candidates": candidate_rows,
    }


def evaluate_multi_origin_condition_gates(
    condition_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply the predecessor immutable gate to each executed condition."""

    unknown = set(condition_metrics) - set(CONDITION_IDS)
    if unknown or not condition_metrics:
        raise MultiOriginMetricsError(f"unknown or empty condition set: {sorted(unknown)}")
    return {
        condition_id: evaluate_true_future_gate(condition_metrics[condition_id])
        for condition_id in CONDITION_IDS
        if condition_id in condition_metrics
    }


def _gate_pass(value: Any, condition_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("pass"), bool):
        return bool(value["pass"])
    raise MultiOriginMetricsError(f"gate {condition_id} lacks Boolean pass")


def classify_multi_origin_conditions(
    condition_gate_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate conditional execution and select the frozen primary class."""

    unknown = set(condition_gate_results) - set(CONDITION_IDS)
    if unknown:
        raise MultiOriginMetricsError(f"unknown condition gates: {sorted(unknown)}")
    executed = set(condition_gate_results)
    regression_executed = executed & set(REGRESSION_CONDITION_IDS)
    if regression_executed and regression_executed != set(REGRESSION_CONDITION_IDS):
        raise MultiOriginMetricsError("single-origin regressions must be supplied together")
    multi_origin_executed = executed - set(REGRESSION_CONDITION_IDS)
    if not set(DUAL_CONDITION_IDS).issubset(multi_origin_executed):
        raise MultiOriginMetricsError("all three selected-dual conditions must execute")
    passed = {
        condition_id: _gate_pass(condition_gate_results[condition_id], condition_id)
        for condition_id in executed
    }
    dual_realistic = passed["DUAL_REALISTIC_L2_SCAN"]
    if dual_realistic:
        if multi_origin_executed != set(DUAL_CONDITION_IDS):
            raise MultiOriginMetricsError(
                "triple/all-four conditions must not execute after dual realistic passes"
            )
    else:
        if not set(TRIPLE_CONDITION_IDS).issubset(multi_origin_executed):
            raise MultiOriginMetricsError(
                "all three selected-triple conditions are required after dual realistic fails"
            )
        triple_spherical = passed["THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"]
        expected = set(DUAL_CONDITION_IDS + TRIPLE_CONDITION_IDS)
        if not triple_spherical:
            expected.add(ALL_FOUR_CONDITION_ID)
        if multi_origin_executed != expected:
            rule = "required" if not triple_spherical else "forbidden"
            raise MultiOriginMetricsError(
                f"all-four diagnostic is {rule} under the triple-spherical result"
            )

    if dual_realistic:
        primary = "DUAL_ORIGIN_REALISTIC_RANGE_SIGNAL"
    elif passed["THREE_REALISTIC_L2_SCAN"]:
        primary = "THREE_ORIGIN_REALISTIC_RANGE_SIGNAL"
    elif (
        passed["DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"]
        or passed["DUAL_DENSE_L2_FOV_UPPER_BOUND"]
    ):
        primary = "DUAL_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK"
    elif (
        passed["THREE_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND"]
        or passed["THREE_DENSE_L2_FOV_UPPER_BOUND"]
    ):
        primary = "THREE_ORIGIN_MOUNT_SIGNAL_SCAN_OR_FOV_BOTTLENECK"
    else:
        primary = "MULTI_ORIGIN_UP_TO_THREE_RANGE_COVERAGE_NO_GO"

    secondary: list[str] = []
    if passed.get(ALL_FOUR_CONDITION_ID, False) and not any(
        passed.get(condition_id, False) for condition_id in TRIPLE_CONDITION_IDS
    ):
        secondary.append(FOUR_OR_MORE_DIAGNOSTIC)
    return {
        "schema": "minimum_multi_origin_condition_classification_v1",
        "executed_condition_ids": [
            condition_id for condition_id in CONDITION_IDS if condition_id in executed
        ],
        "condition_pass": {
            condition_id: passed[condition_id]
            for condition_id in CONDITION_IDS
            if condition_id in passed
        },
        "conditional_execution_valid": True,
        "primary_classification": primary,
        "secondary_classifications": secondary,
        "replanning_interface_classification": REPLANNING_INTERFACE_CLASSIFICATION,
    }


def derive_secondary_classifications(
    *,
    conditional_classification: Mapping[str, Any],
    planning_time_observability_limitation: bool,
    coverage_error_counts: Mapping[str, int],
    assumed_sensor_contract: bool,
) -> list[str]:
    """Derive the frozen secondary labels from explicit completed evidence."""

    for name, value in (
        ("planning_time_observability_limitation", planning_time_observability_limitation),
        ("assumed_sensor_contract", assumed_sensor_contract),
    ):
        if not isinstance(value, bool):
            raise MultiOriginMetricsError(f"{name} must be Boolean")
    existing = conditional_classification.get("secondary_classifications")
    if not isinstance(existing, Sequence) or isinstance(existing, (str, bytes)):
        raise MultiOriginMetricsError(
            "conditional_classification lacks secondary_classifications"
        )
    unknown_existing = set(existing) - {FOUR_OR_MORE_DIAGNOSTIC}
    if unknown_existing:
        raise MultiOriginMetricsError(
            f"unexpected conditional secondary labels: {sorted(unknown_existing)}"
        )
    unknown_counts = set(coverage_error_counts) - set(ERROR_CLASSES)
    if unknown_counts:
        raise MultiOriginMetricsError(
            f"unknown coverage error counts: {sorted(unknown_counts)}"
        )
    counts: dict[str, int] = {}
    for error_class in ERROR_CLASSES:
        value = coverage_error_counts.get(error_class, 0)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise MultiOriginMetricsError(
                f"coverage error count {error_class} must be a non-negative integer"
            )
        counts[error_class] = int(value)

    supported: set[str] = set(existing)
    if planning_time_observability_limitation:
        supported.add("PLANNING_TIME_MULTI_ORIGIN_OBSERVABILITY_LIMITATION")
    if counts["ROBOT_SELF_OCCLUSION"]:
        supported.add("ROBOT_BODY_SELF_OCCLUSION")
    if counts["VERTICAL_FOV_LIMITATION"]:
        supported.add("VERTICAL_FOV_LIMITATION")
    if counts["SCAN_PATTERN_SPARSITY"] or counts["SCAN_TIMING_LIMITATION"]:
        supported.add("SCAN_DENSITY_OR_TIMING_LIMITATION")
    if assumed_sensor_contract:
        supported.add("ASSUMED_SENSOR_CONTRACT")
    supported.add(REPLANNING_INTERFACE_CLASSIFICATION)
    frozen_order = (
        FOUR_OR_MORE_DIAGNOSTIC,
        "PLANNING_TIME_MULTI_ORIGIN_OBSERVABILITY_LIMITATION",
        "ROBOT_BODY_SELF_OCCLUSION",
        "VERTICAL_FOV_LIMITATION",
        "SCAN_DENSITY_OR_TIMING_LIMITATION",
        "ASSUMED_SENSOR_CONTRACT",
        REPLANNING_INTERFACE_CLASSIFICATION,
    )
    return [label for label in frozen_order if label in supported]


ERROR_ATTRIBUTION_HIERARCHY = (
    "POINT_FUSION_ERROR",
    "NEAR_BLIND_REGION",
    "ROBOT_SELF_OCCLUSION",
    "VERTICAL_FOV_LIMITATION",
    "SCAN_TIMING_LIMITATION",
    "SCAN_PATTERN_SPARSITY",
    "INSUFFICIENT_ORIGIN_COUNT",
    "MOUNT_POSITION_LIMITATION",
    "UNRESOLVED",
)

_ERROR_EVIDENCE_FIELDS = {
    "POINT_FUSION_ERROR": "point_fusion_error",
    "NEAR_BLIND_REGION": "near_blind_region",
    "ROBOT_SELF_OCCLUSION": "robot_self_occlusion",
    "VERTICAL_FOV_LIMITATION": "vertical_fov_limitation",
    "SCAN_TIMING_LIMITATION": "scan_timing_limitation",
    "SCAN_PATTERN_SPARSITY": "scan_pattern_sparsity",
    "INSUFFICIENT_ORIGIN_COUNT": "insufficient_origin_count",
    "MOUNT_POSITION_LIMITATION": "mount_position_limitation",
}


def classify_multi_origin_error(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed attribution: one supported cause or ``UNRESOLVED``.

    The evaluator owns the geometric counterfactuals.  This pure helper does
    not infer a cause from outcome labels: it accepts their explicit Boolean
    conclusions and refuses to choose arbitrarily when several remain true.
    """

    unknown = set(evidence) - set(_ERROR_EVIDENCE_FIELDS.values())
    if unknown:
        raise MultiOriginMetricsError(f"unknown error evidence fields: {sorted(unknown)}")
    supported: list[str] = []
    normalized: dict[str, bool] = {}
    for error_class, field in _ERROR_EVIDENCE_FIELDS.items():
        value = evidence.get(field, False)
        if not isinstance(value, (bool, np.bool_)):
            raise MultiOriginMetricsError(f"error evidence {field} must be Boolean")
        normalized[field] = bool(value)
        if value:
            supported.append(error_class)
    classification = supported[0] if len(supported) == 1 else "UNRESOLVED"
    return {
        "error_class": classification,
        "supported_error_classes": supported,
        "ambiguous": len(supported) > 1,
        "prospective_hierarchy": list(ERROR_ATTRIBUTION_HIERARCHY),
        "evidence": normalized,
    }


def classify_multi_origin_errors(
    evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [classify_multi_origin_error(row) for row in evidence_rows]
    counts = {
        error_class: sum(row["error_class"] == error_class for row in rows)
        for error_class in ERROR_CLASSES
    }
    return {
        "schema": "minimum_multi_origin_coverage_error_attribution_v1",
        "rows": rows,
        "counts": counts,
        "complete": sum(counts.values()) == len(rows),
    }


def classify_complete_set_benchmark(result: Mapping[str, Any]) -> str:
    """Classify exact unrounded complete-set P99/maximum latency."""

    try:
        p99 = float(result["p99_ms"])
        maximum = float(result["max_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MultiOriginMetricsError("benchmark requires numeric p99_ms/max_ms") from exc
    if not math.isfinite(p99) or not math.isfinite(maximum) or min(p99, maximum) < 0:
        raise MultiOriginMetricsError("benchmark latency must be finite and non-negative")
    if p99 <= 50.0 and maximum <= 80.0:
        return "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL"
    if p99 <= 80.0 and maximum <= 100.0:
        return "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_POSITIVE_TENDENCY"
    return "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_NO_GO"


def _latency_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples_ms, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise MultiOriginMetricsError("latency samples must be finite and nonempty")
    result = {
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p90_ms": float(np.percentile(values, 90, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "p99_ms": float(np.percentile(values, 99, method="linear")),
        "max_ms": float(values.max()),
        "misses_50ms": int((values > 50.0).sum()),
        "misses_80ms": int((values > 80.0).sum()),
        "misses_100ms": int((values > 100.0).sum()),
    }
    result["classification"] = classify_complete_set_benchmark(result)
    return result


def benchmark_complete_and_per_state_decisions(
    states: Sequence[Mapping[str, Any]],
    *,
    reducer: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] = reduce_two_ply_states,
    per_state_reducer: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    warmups: int = 30,
    iterations: int = 1000,
    required_families: Sequence[str] | None = None,
    enforce_contract_minimums: bool = True,
) -> dict[str, Any]:
    """Benchmark pooled rotating complete-state candidate-set reductions.

    Callers evaluating prematerialized float32 per-link tensors should pass the
    exact ``reducer`` which performs their per-link threshold reduction before
    delegating to the predecessor two-ply functions.  Every timed iteration
    contains exactly one representative state's complete current/next-action
    set.  Classification uses the pooled samples and global maximum; an
    all-held-out-states wall time is deliberately not measured or classified.
    """

    if not states or warmups < 0 or iterations <= 0:
        raise MultiOriginMetricsError("states/iterations must be nonempty and valid")
    if iterations < len(states):
        raise MultiOriginMetricsError(
            "timed iterations must include every representative state at least once"
        )
    if not isinstance(enforce_contract_minimums, bool):
        raise MultiOriginMetricsError("enforce_contract_minimums must be Boolean")
    if enforce_contract_minimums and (warmups < 30 or iterations < 1000):
        raise MultiOriginMetricsError(
            "scientific benchmark requires at least 30 warmups and 1000 timed iterations"
        )
    state_ids = [str(state["state_id"]) for state in states]
    if len(set(state_ids)) != len(state_ids):
        raise MultiOriginMetricsError("benchmark state IDs must be unique")
    families = [str(state["family"]) for state in states]
    if required_families is not None and not set(required_families).issubset(families):
        raise MultiOriginMetricsError("representative states do not cover every required family")

    if per_state_reducer is None:
        def timed_reducer(state: Mapping[str, Any]) -> Mapping[str, Any]:
            return reducer((state,))
    else:
        timed_reducer = per_state_reducer

    for iteration in range(warmups):
        timed_reducer(states[iteration % len(states)])
    pooled_samples: list[float] = []
    per_state_samples: dict[str, list[float]] = {state_id: [] for state_id in state_ids}
    for iteration in range(iterations):
        index = iteration % len(states)
        state_id = state_ids[index]
        state = states[index]
        started = time.perf_counter_ns()
        timed_reducer(state)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        pooled_samples.append(elapsed_ms)
        per_state_samples[state_id].append(elapsed_ms)

    pooled = _latency_summary(pooled_samples)
    per_state = {
        state_id: {
            "family": families[index],
            "timed_samples": len(per_state_samples[state_id]),
            **_latency_summary(per_state_samples[state_id]),
        }
        for index, state_id in enumerate(state_ids)
    }
    return {
        "schema": "minimum_multi_origin_complete_and_per_state_benchmark_v1",
        "device": "CPU",
        "numeric_binding": (
            "float32 materialized physics-step/protected-link clearance and support "
            "witnesses"
        ),
        "includes_ray_generation": False,
        "includes_future_trajectory_acquisition": False,
        "timed_sample_unit": (
            "one representative held-out state's complete current/next-action candidate set"
        ),
        "all_states_aggregate_wall_time_classified": False,
        "includes": [
            "per-link reduction from materialized step/link witnesses",
            "all unique current actions",
            "all next-action sets",
            "current per-link prematerialized reduction",
            "contact threshold decisions",
            "safe-action counting",
            "H3 route selection",
        ],
        "states": len(states),
        "families": sorted(set(families)),
        "warmups": warmups,
        "iterations": iterations,
        "contract_minimums_met": warmups >= 30 and iterations >= 1000,
        # ``complete_set`` is the frozen output-schema name.  The longer alias
        # remains byte-for-byte equivalent for callers written before the
        # executable schema was finalized.
        "complete_set": pooled,
        "pooled_complete_state_samples": pooled,
        "per_state": per_state,
        "classification": pooled["classification"],
        "replanning_interface_classification": REPLANNING_INTERFACE_CLASSIFICATION,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_vram_bytes": 0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise MultiOriginMetricsError("canonical report keys must be strings")
            output[key] = _jsonable(value[key])
        return output
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MultiOriginMetricsError("canonical reports forbid NaN/Infinity")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise MultiOriginMetricsError(f"unsupported canonical report value {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes without platform-dependent whitespace."""

    return (
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_report(**sections: Any) -> dict[str, Any]:
    """Compose sorted reporting sections and bind their canonical content."""

    payload = {
        "schema": "minimum_multi_origin_body_range_coverage_metrics_report_v1",
        "sections": {key: _jsonable(sections[key]) for key in sorted(sections)},
    }
    return {**payload, "content_sha256": canonical_sha256(payload)}


def run_fixtures() -> dict[str, Any]:
    """Run small deterministic pure-metric fixtures for the freeze gate."""

    tied_scores = [
        {
            "layout_id": layout_id,
            "minimum_body_region_support": 0.5,
            "transition_support_p05": 0.4,
            "rear_limb_support": 0.6,
            "calf_support": 0.6,
            "overall_mean_support": 0.7,
            "self_occluded_fraction": 0.2,
        }
        for layout_id in DUAL_LAYOUT_IDS
    ]
    selected = select_training_layout(
        tuple(reversed(tied_scores)), expected_layout_ids=DUAL_LAYOUT_IDS
    )
    condition = classify_multi_origin_conditions(
        {
            "DUAL_DENSE_SPHERICAL_ORIGIN_UPPER_BOUND": False,
            "DUAL_DENSE_L2_FOV_UPPER_BOUND": False,
            "DUAL_REALISTIC_L2_SCAN": True,
        }
    )
    fixtures = {
        "predecessor_exact_aliases": bool(
            contact_predictions is _BASE.contact_predictions
            and enumerate_threshold_frontier is _BASE.enumerate_threshold_frontier
            and reduce_two_ply_states is _BASE.reduce_two_ply_states
            and evaluate_true_future_gate is _REPORTING.evaluate_true_future_gate
        ),
        "threshold_tie_and_unsupported_are_contact": bool(
            contact_predictions(
                [0.2, 0.200001, np.inf],
                0.2,
                unsupported=[False, False, False],
            ).tolist()
            == [True, False, True]
        ),
        "origin_union": bool(
            union_origin_support(
                {
                    "a": np.asarray([True, False]),
                    "b": np.asarray([False, True]),
                },
                ("a", "b"),
            ).all()
        ),
        "fixed_layout_tie": selected["layout_id"] == DUAL_LAYOUT_IDS[0],
        "conditional_dual_stop": (
            condition["primary_classification"]
            == "DUAL_ORIGIN_REALISTIC_RANGE_SIGNAL"
        ),
        "compute_boundary": (
            classify_complete_set_benchmark({"p99_ms": 50.0, "max_ms": 80.0})
            == "MULTI_ORIGIN_SET_REDUCTION_COMPUTE_SIGNAL"
        ),
        "ambiguous_attribution_fails_closed": (
            classify_multi_origin_error(
                {"scan_timing_limitation": True, "robot_self_occlusion": True}
            )["error_class"]
            == "UNRESOLVED"
        ),
    }
    payload = {
        "schema": "minimum_multi_origin_body_range_coverage_metrics_fixture_v1",
        "fixtures": fixtures,
        "pass": all(fixtures.values()),
    }
    return {**payload, "content_digest": canonical_sha256(payload)}
