"""Pure gate, classification, and attribution reporting for range coverage V1.

The functions in this module consume already-computed scalar summaries.  They
perform no file I/O, corpus access, simulation, sensor materialisation, model
inference, or training.  Their purpose is to keep the immutable scientific
gate and the prospective interpretation hierarchy out of the execution
script.

``evaluate_true_future_gate`` expects this compact metric shape (aliases used
by :mod:`body_centric_range_coverage_metrics_v1` are also accepted)::

    {
        "current_contact": {"auc": ...},
        "successor_contact": {"auc": ...},
        "combined_contact": {"recall": ..., "fnr": ...},
        "safe_action_count": {
            "zero_vs_nonzero_accuracy": ...,
            "false_nonzero_rate": ...,
        },
        "viability": {
            "oracle_viable_states": 20,
            "states_retaining_admitted_action": ...,
            "oracle_nonviable_states": 4,
            "correct_abstentions": ...,
            "selected_immediate_contacts": ...,
            "selected_oracle_nonviable_successors": ...,
            "oracle_progress_fraction": ...,
            "normalized_regret": ...,
            "best_admissible_top3": ...,
        },
        "per_family": {
            "family": {
                "combined_contact": {
                    "positives": ..., "negatives": ...,
                    "recall": ..., "auc": ...,
                },
                "oracle_viable_states": ...,
                "states_retaining_admitted_action": ...,
                "oracle_nonviable_states": ...,
                "correct_abstentions": ...,
            },
        },
    }

Undefined aggregate metrics fail their check.  Missing or malformed metrics
raise :class:`ReportingError`; this distinguishes an invalid result receipt
from an honestly executed condition that misses a gate.
"""

from __future__ import annotations

import copy
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from lewm.safety import body_centric_range_coverage_qualification_v1_contract as CONTRACT


class ReportingError(ValueError):
    """Raised when a reporting input is incomplete or internally invalid."""


_REGION_SECONDARY = {
    "FRONT_LIMB": "FRONT_LIMB_VISIBILITY_FAILURE",
    "REAR_LIMB": "REAR_LIMB_VISIBILITY_FAILURE",
    "CALF": "CALF_VISIBILITY_FAILURE",
    "TRUNK": "TRUNK_VISIBILITY_FAILURE",
}


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportingError(f"{location} must be a mapping")
    return value


def _section(
    source: Mapping[str, Any], aliases: Sequence[str], location: str
) -> Mapping[str, Any]:
    for key in aliases:
        if key in source:
            return _mapping(source[key], f"{location}.{key}")
    raise ReportingError(f"{location} lacks required section {tuple(aliases)}")


def _value(source: Mapping[str, Any], aliases: Sequence[str], location: str) -> Any:
    for key in aliases:
        if key in source:
            return source[key]
    raise ReportingError(f"{location} lacks required metric {tuple(aliases)}")


def _number_or_none(value: Any, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReportingError(f"{location} must be a finite real number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ReportingError(f"{location} must be finite")
    return result


def _rate_or_none(value: Any, location: str) -> float | None:
    result = _number_or_none(value, location)
    if result is not None and not 0.0 <= result <= 1.0:
        raise ReportingError(f"{location} must lie in [0,1]")
    return result


def _nonnegative_or_none(value: Any, location: str) -> float | None:
    result = _number_or_none(value, location)
    if result is not None and result < 0.0:
        raise ReportingError(f"{location} must be non-negative")
    return result


def _count(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
        raise ReportingError(f"{location} must be an exact non-negative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ReportingError(f"{location} must be an exact non-negative integer")
    return int(numeric)


def _check(
    *, name: str, value: float | int | None, operator: str, threshold: float | int
) -> dict[str, Any]:
    if value is None:
        passed = False
    elif operator == ">=":
        passed = value >= threshold
    elif operator == "<=":
        passed = value <= threshold
    elif operator == "==":
        passed = value == threshold
    else:  # pragma: no cover - all operators are defined locally.
        raise AssertionError(operator)
    return {
        "name": name,
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "pass": bool(passed),
    }


def _family_contact(family: Mapping[str, Any], location: str) -> Mapping[str, Any]:
    if "combined_contact" in family:
        return _mapping(family["combined_contact"], f"{location}.combined_contact")
    if "contact" in family:
        return _mapping(family["contact"], f"{location}.contact")
    # A flat representation is permitted but still requires all four fields.
    return family


def evaluate_family_collapse(
    per_family: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the prospectively frozen no-family-collapse definition."""

    families = _mapping(per_family, "per_family")
    if not families:
        raise ReportingError("per_family must contain every held-out family")
    rows: dict[str, Any] = {}
    for raw_name in sorted(families, key=str):
        name = str(raw_name)
        family = _mapping(families[raw_name], f"per_family.{name}")
        contact = _family_contact(family, f"per_family.{name}")
        positives = _count(
            _value(contact, ("positives", "contact_positives"), f"per_family.{name}.contact"),
            f"per_family.{name}.contact.positives",
        )
        negatives = _count(
            _value(contact, ("negatives", "contact_negatives"), f"per_family.{name}.contact"),
            f"per_family.{name}.contact.negatives",
        )
        recall = _rate_or_none(
            _value(contact, ("recall", "combined_recall"), f"per_family.{name}.contact"),
            f"per_family.{name}.contact.recall",
        )
        auc = _rate_or_none(
            _value(contact, ("auc", "combined_auc"), f"per_family.{name}.contact"),
            f"per_family.{name}.contact.auc",
        )
        viable = _count(
            _value(family, ("oracle_viable_states",), f"per_family.{name}"),
            f"per_family.{name}.oracle_viable_states",
        )
        retained = _count(
            _value(
                family,
                ("states_retaining_admitted_action", "viable_states_retaining_action"),
                f"per_family.{name}",
            ),
            f"per_family.{name}.states_retaining_admitted_action",
        )
        nonviable = _count(
            _value(family, ("oracle_nonviable_states",), f"per_family.{name}"),
            f"per_family.{name}.oracle_nonviable_states",
        )
        abstentions = _count(
            _value(family, ("correct_abstentions",), f"per_family.{name}"),
            f"per_family.{name}.correct_abstentions",
        )
        if retained > viable:
            raise ReportingError(f"per_family.{name} retained count exceeds viable count")
        if abstentions > nonviable:
            raise ReportingError(f"per_family.{name} abstentions exceed nonviable count")

        checks = {
            "positive_contact_recall": {
                "applicable": positives > 0,
                **_check(
                    name="positive_contact_recall",
                    value=recall,
                    operator=">=",
                    threshold=0.8,
                ),
            },
            "two_class_contact_auc": {
                "applicable": positives > 0 and negatives > 0,
                **_check(
                    name="two_class_contact_auc",
                    value=auc,
                    operator=">=",
                    threshold=0.75,
                ),
            },
            "viable_action_retention": {
                "applicable": viable > 0,
                **_check(
                    name="viable_action_retention",
                    value=retained,
                    operator=">=",
                    threshold=1,
                ),
            },
            "all_nonviable_abstain": {
                "applicable": nonviable > 0,
                **_check(
                    name="all_nonviable_abstain",
                    value=abstentions,
                    operator="==",
                    threshold=nonviable,
                ),
            },
        }
        # Non-applicable checks are true by definition, while an applicable
        # undefined AUC/recall remains a fail-closed failure.
        for check in checks.values():
            if not check["applicable"]:
                check["pass"] = True
        passed = all(check["pass"] for check in checks.values())
        rows[name] = {
            "positives": positives,
            "negatives": negatives,
            "oracle_viable_states": viable,
            "states_retaining_admitted_action": retained,
            "oracle_nonviable_states": nonviable,
            "correct_abstentions": abstentions,
            "checks": checks,
            "pass": passed,
        }
    return {
        "definition": copy.deepcopy(
            CONTRACT.CONTRACT["immutable_gate"]["no_family_collapse"]["definition"]
        ),
        "families": rows,
        "pass": all(row["pass"] for row in rows.values()),
    }


def evaluate_true_future_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply every immutable true-future condition gate, without rounding."""

    source = _mapping(metrics, "metrics")
    current = _section(source, ("current_contact", "current"), "metrics")
    successor = _section(source, ("successor_contact", "successor"), "metrics")
    combined = _section(source, ("combined_contact", "combined"), "metrics")
    safe_count = _section(source, ("safe_action_count",), "metrics")
    viability = (
        _mapping(source["viability"], "metrics.viability")
        if "viability" in source
        else _mapping(source.get("two_ply", source), "metrics.two_ply")
    )

    current_auc = _rate_or_none(_value(current, ("auc",), "current_contact"), "current_contact.auc")
    successor_auc = _rate_or_none(
        _value(successor, ("auc",), "successor_contact"), "successor_contact.auc"
    )
    combined_recall = _rate_or_none(
        _value(combined, ("recall",), "combined_contact"), "combined_contact.recall"
    )
    combined_fnr = _rate_or_none(
        _value(combined, ("fnr", "false_negative_rate"), "combined_contact"),
        "combined_contact.fnr",
    )
    zero_nonzero = _rate_or_none(
        _value(
            safe_count,
            ("zero_nonzero_accuracy", "zero_vs_nonzero_accuracy"),
            "safe_action_count",
        ),
        "safe_action_count.zero_nonzero_accuracy",
    )
    false_nonzero = _rate_or_none(
        _value(safe_count, ("false_nonzero_rate",), "safe_action_count"),
        "safe_action_count.false_nonzero_rate",
    )

    viable = _count(
        _value(viability, ("oracle_viable_states",), "viability"),
        "viability.oracle_viable_states",
    )
    retained = _count(
        _value(
            viability,
            ("states_retaining_admitted_action", "viable_states_retaining_action"),
            "viability",
        ),
        "viability.states_retaining_admitted_action",
    )
    nonviable = _count(
        _value(viability, ("oracle_nonviable_states",), "viability"),
        "viability.oracle_nonviable_states",
    )
    abstentions = _count(
        _value(viability, ("correct_abstentions",), "viability"),
        "viability.correct_abstentions",
    )
    immediate_contacts = _count(
        _value(viability, ("selected_immediate_contacts",), "viability"),
        "viability.selected_immediate_contacts",
    )
    nonviable_successors = _count(
        _value(
            viability,
            ("selected_oracle_nonviable_successors", "selected_nonviable_successors"),
            "viability",
        ),
        "viability.selected_oracle_nonviable_successors",
    )
    progress = _nonnegative_or_none(
        _value(
            viability,
            (
                "h3_route_progress_fraction_of_exact_geometry",
                "oracle_progress_fraction",
                "exact_geometry_progress_fraction",
            ),
            "viability",
        ),
        "viability.h3_route_progress_fraction_of_exact_geometry",
    )
    regret = _nonnegative_or_none(
        _value(
            viability,
            ("normalized_regret", "normalized_viability_constrained_regret"),
            "viability",
        ),
        "viability.normalized_regret",
    )
    top3 = _rate_or_none(
        _value(viability, ("best_admissible_top3", "best_admissible_top_3"), "viability"),
        "viability.best_admissible_top3",
    )
    if retained > viable:
        raise ReportingError("retained viable-state count exceeds oracle-viable count")
    if abstentions > nonviable:
        raise ReportingError("correct abstentions exceed oracle-nonviable count")

    family = evaluate_family_collapse(
        _mapping(source.get("per_family", viability.get("per_family")), "metrics.per_family")
    )
    checks = {
        "current_contact_auc": _check(
            name="current_contact_auc", value=current_auc, operator=">=", threshold=0.9
        ),
        "successor_contact_auc": _check(
            name="successor_contact_auc", value=successor_auc, operator=">=", threshold=0.9
        ),
        "combined_contact_recall": _check(
            name="combined_contact_recall", value=combined_recall, operator=">=", threshold=0.95
        ),
        "combined_contact_fnr": _check(
            name="combined_contact_fnr", value=combined_fnr, operator="<=", threshold=0.05
        ),
        "safe_action_zero_nonzero_accuracy": _check(
            name="safe_action_zero_nonzero_accuracy",
            value=zero_nonzero,
            operator=">=",
            threshold=0.9,
        ),
        "safe_action_false_nonzero_rate": _check(
            name="safe_action_false_nonzero_rate",
            value=false_nonzero,
            operator="<=",
            threshold=0.05,
        ),
        "oracle_viable_state_denominator": _check(
            name="oracle_viable_state_denominator", value=viable, operator="==", threshold=20
        ),
        "oracle_viable_states_retaining_action": _check(
            name="oracle_viable_states_retaining_action",
            value=retained,
            operator=">=",
            threshold=18,
        ),
        "oracle_nonviable_state_denominator": _check(
            name="oracle_nonviable_state_denominator", value=nonviable, operator="==", threshold=4
        ),
        "oracle_nonviable_correct_abstentions": _check(
            name="oracle_nonviable_correct_abstentions",
            value=abstentions,
            operator="==",
            threshold=4,
        ),
        "selected_immediate_contacts": _check(
            name="selected_immediate_contacts", value=immediate_contacts, operator="==", threshold=0
        ),
        "selected_nonviable_successors": _check(
            name="selected_nonviable_successors",
            value=nonviable_successors,
            operator="==",
            threshold=0,
        ),
        "h3_route_progress_fraction": _check(
            name="h3_route_progress_fraction", value=progress, operator=">=", threshold=0.8
        ),
        "normalized_regret": _check(
            name="normalized_regret", value=regret, operator="<=", threshold=0.2
        ),
        "best_admissible_top3": _check(
            name="best_admissible_top3", value=top3, operator=">=", threshold=0.75
        ),
        "no_family_collapse": {
            "name": "no_family_collapse",
            "value": family["pass"],
            "operator": "==",
            "threshold": True,
            "pass": family["pass"],
        },
    }
    failed = [name for name, row in checks.items() if not row["pass"]]
    return {
        "schema": "body_centric_range_coverage_true_future_gate_v1",
        "mode": "TRUE_FUTURE_OBSERVABILITY_CLOUD",
        "checks": checks,
        "family_collapse": family,
        "failed_checks": failed,
        "pass": not failed,
    }


def evaluate_condition_gates(
    condition_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Apply the same immutable true-future gate to all four conditions."""

    metrics = _mapping(condition_metrics, "condition_metrics")
    if set(metrics) != set(CONTRACT.CONDITION_IDS):
        raise ReportingError(
            "condition_metrics must contain exactly the four prospectively frozen conditions"
        )
    return {
        condition_id: evaluate_true_future_gate(metrics[condition_id])
        for condition_id in CONTRACT.CONDITION_IDS
    }


def _pass_value(value: Any, location: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("pass"), bool):
        return bool(value["pass"])
    raise ReportingError(f"{location} must be a bool or a gate result containing bool pass")


def select_primary_classification(
    condition_gate_results: Mapping[str, Any],
    *,
    realistic_platform_sensor_resolved: bool,
) -> str:
    """Select exactly one primary classification under the frozen B/C/D order."""

    gates = _mapping(condition_gate_results, "condition_gate_results")
    required = set(CONTRACT.CONDITION_IDS)
    if set(gates) != required:
        raise ReportingError("condition gate results must contain exactly conditions A/B/C/D")
    if not isinstance(realistic_platform_sensor_resolved, bool):
        raise ReportingError("realistic_platform_sensor_resolved must be bool")
    passed = {
        condition_id: _pass_value(gates[condition_id], f"condition_gate_results.{condition_id}")
        for condition_id in CONTRACT.CONDITION_IDS
    }
    if not realistic_platform_sensor_resolved:
        return "RANGE_SENSOR_CONTRACT_UNRESOLVED"
    if passed["REALISTIC_PLATFORM_SCAN"]:
        return "PLATFORM_RANGE_COVERAGE_SIGNAL"
    if passed["DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT"]:
        return "SCAN_DENSITY_OR_TIMING_BOTTLENECK"
    if passed["DENSE_BODY_CENTRIC_SINGLE_ORIGIN"]:
        return "BODY_CENTRIC_MOUNT_REQUIRED"
    return "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO"


def derive_secondary_classifications(
    *,
    true_future_gate_results: Mapping[str, Any],
    planning_time_gate_results: Mapping[str, Any],
    sensor_binding_class: str,
    per_region_metrics: Mapping[str, Mapping[str, Any]],
    coverage_error_counts: Mapping[str, int],
) -> list[str]:
    """Return only secondary labels directly supported by frozen summaries.

    A body-region visibility failure is supported when that region has either
    a coverage-attributed error or positive contact rows with combined recall
    below the immutable 0.95 aggregate contact target.
    """

    future = _mapping(true_future_gate_results, "true_future_gate_results")
    causal = _mapping(planning_time_gate_results, "planning_time_gate_results")
    expected = set(CONTRACT.CONDITION_IDS)
    if set(future) != expected or set(causal) != expected:
        raise ReportingError("future and causal gate results must each contain exactly A/B/C/D")
    future_pass = {
        condition_id: _pass_value(future[condition_id], f"future.{condition_id}")
        for condition_id in CONTRACT.CONDITION_IDS
    }
    causal_pass = {
        condition_id: _pass_value(causal[condition_id], f"causal.{condition_id}")
        for condition_id in CONTRACT.CONDITION_IDS
    }
    if not isinstance(sensor_binding_class, str) or not sensor_binding_class:
        raise ReportingError("sensor_binding_class must be a nonempty string")

    counts = _mapping(coverage_error_counts, "coverage_error_counts")
    normalized_counts: dict[str, int] = {}
    for label in CONTRACT.COVERAGE_ERROR_CLASSES:
        normalized_counts[label] = _count(counts.get(label, 0), f"coverage_error_counts.{label}")
    unknown_counts = set(counts) - set(CONTRACT.COVERAGE_ERROR_CLASSES)
    if unknown_counts:
        raise ReportingError(f"unknown coverage error classes: {sorted(unknown_counts)}")

    regions = _mapping(per_region_metrics, "per_region_metrics")
    unknown_regions = set(regions) - set(_REGION_SECONDARY)
    if unknown_regions:
        raise ReportingError(f"unknown canonical body regions: {sorted(unknown_regions)}")

    supported: set[str] = set()
    if not future_pass["CURRENT_SPARSE_RANGE_BASELINE"]:
        supported.add("FOUR_CHANNEL_RANGE_BASELINE_INSUFFICIENT")
    if sensor_binding_class == "ASSUMED_GO2_HEAD_LIDAR_L2":
        supported.add("ASSUMED_SENSOR_CONTRACT")
    if normalized_counts["ROBOT_SELF_OCCLUSION"] > 0:
        supported.add("ROBOT_BODY_SELF_OCCLUSION")

    strongest_passing = next(
        (
            condition_id
            for condition_id in (
                "REALISTIC_PLATFORM_SCAN",
                "DENSE_FOV_UPPER_BOUND_AT_PLATFORM_MOUNT",
                "DENSE_BODY_CENTRIC_SINGLE_ORIGIN",
            )
            if future_pass[condition_id]
        ),
        None,
    )
    if strongest_passing is not None and not causal_pass[strongest_passing]:
        supported.add("PLANNING_TIME_RANGE_OBSERVABILITY_LIMITATION")

    for region, classification in _REGION_SECONDARY.items():
        if region not in regions:
            continue
        row = _mapping(regions[region], f"per_region_metrics.{region}")
        positives = _count(
            _value(row, ("contact_positives", "positives"), f"per_region_metrics.{region}"),
            f"per_region_metrics.{region}.contact_positives",
        )
        recall = _rate_or_none(
            _value(
                row,
                ("combined_contact_recall", "contact_recall", "recall"),
                f"per_region_metrics.{region}",
            ),
            f"per_region_metrics.{region}.combined_contact_recall",
        )
        errors = _count(
            row.get("coverage_error_count", 0),
            f"per_region_metrics.{region}.coverage_error_count",
        )
        if errors > 0 or (positives > 0 and (recall is None or recall < 0.95)):
            supported.add(classification)

    return [
        classification
        for classification in CONTRACT.SECONDARY_CLASSIFICATIONS
        if classification in supported
    ]


_ATTRIBUTION_BOOLEAN_FIELDS = (
    "point_accumulation_error",
    "platform_inside_near_blind_region",
    "body_inside_near_blind_region",
    "platform_robot_self_occluded",
    "body_robot_self_occluded",
    "platform_nominal_vertical_fov",
    "platform_nominal_horizontal_fov",
    "realistic_spatial_ray_support",
    "realistic_temporal_ray_support",
    "dense_platform_support",
    "dense_body_support",
)


def _attribution_evidence(value: Mapping[str, Any]) -> dict[str, bool]:
    source = _mapping(value, "evidence")
    output: dict[str, bool] = {}
    for field in _ATTRIBUTION_BOOLEAN_FIELDS:
        item = source.get(field)
        if not isinstance(item, bool):
            raise ReportingError(f"evidence.{field} must be bool")
        output[field] = item
    return output


def classify_coverage_error(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one coverage error using the frozen B/C/D evidence hierarchy.

    ``realistic_spatial_ray_support`` means at least one frozen B ray direction
    reaches the relevant surface when time is ignored.
    ``realistic_temporal_ray_support`` additionally requires such a ray during
    the oracle event interval.  Thus spatial absence is sparsity, while spatial
    presence with temporal absence is timing.  Dense-platform and dense-body
    support are the corresponding C and D counterfactuals.
    """

    row = _attribution_evidence(evidence)
    if row["realistic_temporal_ray_support"] and not row["realistic_spatial_ray_support"]:
        raise ReportingError("temporal ray support cannot exist without spatial ray support")

    decision_trace: list[str] = []

    def choose(label: str, reason: str) -> dict[str, Any]:
        decision_trace.append(reason)
        return {
            "error_class": label,
            "reason": reason,
            "decision_trace": decision_trace,
            "evidence": row,
        }

    decision_trace.append("POINT_ACCUMULATION_ERROR checked first")
    if row["point_accumulation_error"]:
        return choose(
            "POINT_ACCUMULATION_ERROR",
            "a valid temporally relevant return exists but the accumulated/reprojected cloud loses it",
        )

    decision_trace.append("near-blind evidence checked at platform and body origins")
    if row["platform_inside_near_blind_region"] or (
        not row["dense_platform_support"] and row["body_inside_near_blind_region"]
    ):
        return choose("NEAR_BLIND_REGION", "the responsible first hit lies inside a frozen blind region")

    decision_trace.append("exact robot self-occlusion checked at platform and body origins")
    if row["platform_robot_self_occluded"] or (
        not row["dense_platform_support"] and row["body_robot_self_occluded"]
    ):
        return choose("ROBOT_SELF_OCCLUSION", "exact articulated robot geometry blocks the environment")

    decision_trace.append("nominal vertical FOV checked before horizontal FOV")
    if not row["platform_nominal_vertical_fov"]:
        return choose("VERTICAL_COVERAGE_LIMITATION", "responsible geometry lies outside nominal elevation FOV")
    if not row["platform_nominal_horizontal_fov"]:
        return choose("HORIZONTAL_COVERAGE_LIMITATION", "responsible geometry lies outside nominal azimuth FOV")

    decision_trace.append("condition C dense platform support compared with finite condition B")
    if row["dense_platform_support"]:
        if not row["realistic_spatial_ray_support"]:
            return choose(
                "SCAN_PATTERN_SPARSITY",
                "dense condition C observes the geometry but no finite B ray direction supports it",
            )
        if not row["realistic_temporal_ray_support"]:
            return choose(
                "SCAN_TIMING_LIMITATION",
                "a finite B ray direction exists, but none intersects during the oracle event interval",
            )
        return choose(
            "UNRESOLVED",
            "B has spatial and temporal ray support and C supports the geometry without an accumulation fault",
        )

    decision_trace.append("condition D body-origin support compared with failed condition C")
    if row["dense_body_support"]:
        return choose(
            "PLATFORM_MOUNT_LIMITATION",
            "dense platform condition C fails while dense body-origin condition D observes the geometry",
        )
    return choose(
        "SINGLE_ORIGIN_LIMITATION",
        "even dense full-sphere condition D cannot support the responsible protected geometry",
    )


def classify_coverage_errors(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach deterministic attribution to rows without mutating caller data."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ReportingError("rows must be a sequence of mappings")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        source_mapping = _mapping(source, f"rows[{index}]")
        evidence = source_mapping.get("attribution_evidence", source_mapping)
        attribution = classify_coverage_error(_mapping(evidence, f"rows[{index}].attribution_evidence"))
        result = copy.deepcopy(dict(source_mapping))
        result["coverage_error_class"] = attribution["error_class"]
        result["coverage_error_reason"] = attribution["reason"]
        result["coverage_error_decision_trace"] = attribution["decision_trace"]
        output.append(result)
    return output


__all__ = [
    "ReportingError",
    "classify_coverage_error",
    "classify_coverage_errors",
    "derive_secondary_classifications",
    "evaluate_condition_gates",
    "evaluate_family_collapse",
    "evaluate_true_future_gate",
    "select_primary_classification",
]
