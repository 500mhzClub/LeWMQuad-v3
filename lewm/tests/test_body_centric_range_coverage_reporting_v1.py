from __future__ import annotations

import copy

import pytest

from lewm.safety import body_centric_range_coverage_reporting_v1 as R
from lewm.safety import body_centric_range_coverage_qualification_v1_contract as C


def _family(
    *,
    positives: int = 10,
    negatives: int = 10,
    recall: float | None = 0.95,
    auc: float | None = 0.9,
    viable: int = 10,
    retained: int = 9,
    nonviable: int = 2,
    abstentions: int = 2,
) -> dict:
    return {
        "combined_contact": {
            "positives": positives,
            "negatives": negatives,
            "recall": recall,
            "auc": auc,
        },
        "oracle_viable_states": viable,
        "states_retaining_admitted_action": retained,
        "oracle_nonviable_states": nonviable,
        "correct_abstentions": abstentions,
    }


def _passing_metrics() -> dict:
    return {
        "current_contact": {"auc": 0.9},
        "successor_contact": {"auc": 0.9},
        "combined_contact": {"recall": 0.95, "fnr": 0.05},
        "safe_action_count": {
            "zero_vs_nonzero_accuracy": 0.9,
            "false_nonzero_rate": 0.05,
        },
        "viability": {
            "oracle_viable_states": 20,
            "states_retaining_admitted_action": 18,
            "oracle_nonviable_states": 4,
            "correct_abstentions": 4,
            "selected_immediate_contacts": 0,
            "selected_oracle_nonviable_successors": 0,
            "oracle_progress_fraction": 0.8,
            "normalized_regret": 0.2,
            "best_admissible_top3": 0.75,
        },
        "per_family": {
            "family-a": _family(),
            "family-b": _family(),
        },
    }


def _set_path(value: dict, path: str, replacement: object) -> None:
    target = value
    pieces = path.split(".")
    for piece in pieces[:-1]:
        target = target[piece]
    target[pieces[-1]] = replacement


def test_true_future_gate_passes_all_exact_boundary_values() -> None:
    result = R.evaluate_true_future_gate(_passing_metrics())
    assert result["pass"] is True
    assert result["failed_checks"] == []
    assert result["mode"] == "TRUE_FUTURE_OBSERVABILITY_CLOUD"
    assert len(result["checks"]) == 16
    assert all(row["pass"] for row in result["checks"].values())
    assert result["family_collapse"]["pass"] is True


@pytest.mark.parametrize(
    ("path", "bad_value", "failed_check"),
    [
        ("current_contact.auc", 0.899, "current_contact_auc"),
        ("successor_contact.auc", 0.899, "successor_contact_auc"),
        ("combined_contact.recall", 0.949, "combined_contact_recall"),
        ("combined_contact.fnr", 0.051, "combined_contact_fnr"),
        (
            "safe_action_count.zero_vs_nonzero_accuracy",
            0.899,
            "safe_action_zero_nonzero_accuracy",
        ),
        (
            "safe_action_count.false_nonzero_rate",
            0.051,
            "safe_action_false_nonzero_rate",
        ),
        ("viability.oracle_viable_states", 19, "oracle_viable_state_denominator"),
        (
            "viability.states_retaining_admitted_action",
            17,
            "oracle_viable_states_retaining_action",
        ),
        ("viability.oracle_nonviable_states", 5, "oracle_nonviable_state_denominator"),
        ("viability.correct_abstentions", 3, "oracle_nonviable_correct_abstentions"),
        ("viability.selected_immediate_contacts", 1, "selected_immediate_contacts"),
        (
            "viability.selected_oracle_nonviable_successors",
            1,
            "selected_nonviable_successors",
        ),
        ("viability.oracle_progress_fraction", 0.799, "h3_route_progress_fraction"),
        ("viability.normalized_regret", 0.201, "normalized_regret"),
        ("viability.best_admissible_top3", 0.749, "best_admissible_top3"),
        ("per_family.family-a.combined_contact.recall", 0.79, "no_family_collapse"),
    ],
)
def test_each_immutable_gate_requirement_fails_closed(
    path: str, bad_value: object, failed_check: str
) -> None:
    metrics = _passing_metrics()
    _set_path(metrics, path, bad_value)
    # Keep semantic counts internally possible when the denominator is changed.
    if path == "viability.oracle_viable_states":
        metrics["viability"]["states_retaining_admitted_action"] = 18
    if path == "viability.oracle_nonviable_states":
        metrics["viability"]["correct_abstentions"] = 4
    result = R.evaluate_true_future_gate(metrics)
    assert result["pass"] is False
    assert failed_check in result["failed_checks"]


def test_undefined_auc_is_an_honest_gate_failure_not_a_pass() -> None:
    metrics = _passing_metrics()
    metrics["current_contact"]["auc"] = None
    result = R.evaluate_true_future_gate(metrics)
    assert result["pass"] is False
    assert result["checks"]["current_contact_auc"]["value"] is None


def test_progress_and_regret_are_not_artificially_capped_at_one() -> None:
    metrics = _passing_metrics()
    metrics["viability"]["oracle_progress_fraction"] = 1.1
    metrics["viability"]["normalized_regret"] = 1.1
    result = R.evaluate_true_future_gate(metrics)
    assert result["checks"]["h3_route_progress_fraction"]["pass"] is True
    assert result["checks"]["normalized_regret"]["pass"] is False


def test_signed_backward_route_progress_is_preserved_as_a_gate_failure() -> None:
    metrics = _passing_metrics()
    metrics["viability"]["oracle_progress_fraction"] = -0.1
    result = R.evaluate_true_future_gate(metrics)
    assert result["pass"] is False
    check = result["checks"]["h3_route_progress_fraction"]
    assert check["value"] == -0.1
    assert check["operator"] == ">="
    assert check["threshold"] == 0.8
    assert check["pass"] is False
    assert "h3_route_progress_fraction" in result["failed_checks"]


@pytest.mark.parametrize(
    ("family_patch", "failed_family_check"),
    [
        ({"combined_contact": {"positives": 10, "negatives": 10, "recall": 0.79, "auc": 0.9}}, "positive_contact_recall"),
        ({"combined_contact": {"positives": 10, "negatives": 10, "recall": 0.95, "auc": 0.74}}, "two_class_contact_auc"),
        ({"oracle_viable_states": 1, "states_retaining_admitted_action": 0}, "viable_action_retention"),
        ({"oracle_nonviable_states": 1, "correct_abstentions": 0}, "all_nonviable_abstain"),
    ],
)
def test_every_family_collapse_clause_is_enforced(
    family_patch: dict, failed_family_check: str
) -> None:
    family = _family()
    for key, value in family_patch.items():
        if key == "combined_contact":
            family[key] = value
        else:
            family[key] = value
    result = R.evaluate_family_collapse({"family": family})
    assert result["pass"] is False
    assert result["families"]["family"]["checks"][failed_family_check]["pass"] is False


def _gates(*, a: bool = False, b: bool = False, c: bool = False, d: bool = False) -> dict:
    return dict(zip(C.CONDITION_IDS, (a, b, c, d), strict=True))


@pytest.mark.parametrize(
    ("gates", "resolved", "expected"),
    [
        (_gates(b=True), True, "PLATFORM_RANGE_COVERAGE_SIGNAL"),
        (_gates(c=True), True, "SCAN_DENSITY_OR_TIMING_BOTTLENECK"),
        (_gates(d=True), True, "BODY_CENTRIC_MOUNT_REQUIRED"),
        (_gates(), True, "SINGLE_ORIGIN_RANGE_COVERAGE_NO_GO"),
        (_gates(c=True, d=True), False, "RANGE_SENSOR_CONTRACT_UNRESOLVED"),
    ],
)
def test_every_primary_classification(
    gates: dict, resolved: bool, expected: str
) -> None:
    assert R.select_primary_classification(
        gates, realistic_platform_sensor_resolved=resolved
    ) == expected


def test_primary_hierarchy_prefers_b_then_c_then_d() -> None:
    assert R.select_primary_classification(
        _gates(a=True, b=True, c=True, d=True),
        realistic_platform_sensor_resolved=True,
    ) == "PLATFORM_RANGE_COVERAGE_SIGNAL"
    assert R.select_primary_classification(
        _gates(c=True, d=True), realistic_platform_sensor_resolved=True
    ) == "SCAN_DENSITY_OR_TIMING_BOTTLENECK"


def test_secondary_classifications_require_direct_metric_support() -> None:
    future = _gates(a=False, b=True, c=True, d=True)
    causal = _gates(a=False, b=False, c=True, d=True)
    regions = {
        "FRONT_LIMB": {"contact_positives": 2, "combined_contact_recall": 0.94},
        "REAR_LIMB": {
            "contact_positives": 0,
            "combined_contact_recall": None,
            "coverage_error_count": 1,
        },
        "CALF": {"contact_positives": 1, "combined_contact_recall": 0.0},
        "TRUNK": {"contact_positives": 1, "combined_contact_recall": 0.94},
    }
    counts = {label: 0 for label in C.COVERAGE_ERROR_CLASSES}
    counts["ROBOT_SELF_OCCLUSION"] = 1
    assert R.derive_secondary_classifications(
        true_future_gate_results=future,
        planning_time_gate_results=causal,
        sensor_binding_class="ASSUMED_GO2_HEAD_LIDAR_L2",
        per_region_metrics=regions,
        coverage_error_counts=counts,
    ) == list(C.SECONDARY_CLASSIFICATIONS)


def test_secondary_classifications_are_empty_without_support() -> None:
    gates = _gates(a=True, b=True, c=True, d=True)
    counts = {label: 0 for label in C.COVERAGE_ERROR_CLASSES}
    assert R.derive_secondary_classifications(
        true_future_gate_results=gates,
        planning_time_gate_results=gates,
        sensor_binding_class="PLANNED_PLATFORM_RANGE_SENSOR",
        per_region_metrics={},
        coverage_error_counts=counts,
    ) == []


def _error_evidence(**updates: bool) -> dict:
    value = {
        "point_accumulation_error": False,
        "platform_inside_near_blind_region": False,
        "body_inside_near_blind_region": False,
        "platform_robot_self_occluded": False,
        "body_robot_self_occluded": False,
        "platform_nominal_vertical_fov": True,
        "platform_nominal_horizontal_fov": True,
        "realistic_spatial_ray_support": True,
        "realistic_temporal_ray_support": True,
        "dense_platform_support": True,
        "dense_body_support": True,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"point_accumulation_error": True}, "POINT_ACCUMULATION_ERROR"),
        ({"platform_inside_near_blind_region": True}, "NEAR_BLIND_REGION"),
        ({"platform_robot_self_occluded": True}, "ROBOT_SELF_OCCLUSION"),
        ({"platform_nominal_vertical_fov": False}, "VERTICAL_COVERAGE_LIMITATION"),
        ({"platform_nominal_horizontal_fov": False}, "HORIZONTAL_COVERAGE_LIMITATION"),
        ({"realistic_spatial_ray_support": False, "realistic_temporal_ray_support": False}, "SCAN_PATTERN_SPARSITY"),
        ({"realistic_temporal_ray_support": False}, "SCAN_TIMING_LIMITATION"),
        ({"dense_platform_support": False}, "PLATFORM_MOUNT_LIMITATION"),
        ({"dense_platform_support": False, "dense_body_support": False}, "SINGLE_ORIGIN_LIMITATION"),
        ({}, "UNRESOLVED"),
    ],
)
def test_every_coverage_error_attribution_label(
    updates: dict[str, bool], expected: str
) -> None:
    result = R.classify_coverage_error(_error_evidence(**updates))
    assert result["error_class"] == expected
    assert result["reason"]
    assert result["decision_trace"]
    assert result["error_class"] in C.COVERAGE_ERROR_CLASSES


def test_attribution_hierarchy_is_prospective_and_conservative() -> None:
    evidence = _error_evidence(
        point_accumulation_error=True,
        platform_inside_near_blind_region=True,
        platform_robot_self_occluded=True,
        platform_nominal_vertical_fov=False,
        dense_platform_support=False,
        dense_body_support=False,
    )
    assert R.classify_coverage_error(evidence)["error_class"] == "POINT_ACCUMULATION_ERROR"

    with pytest.raises(R.ReportingError, match="without spatial"):
        R.classify_coverage_error(
            _error_evidence(
                realistic_spatial_ray_support=False,
                realistic_temporal_ray_support=True,
            )
        )


def test_batch_attribution_does_not_mutate_rows() -> None:
    rows = [{"state_id": "fixture", "attribution_evidence": _error_evidence(realistic_temporal_ray_support=False)}]
    before = copy.deepcopy(rows)
    output = R.classify_coverage_errors(rows)
    assert rows == before
    assert output[0]["coverage_error_class"] == "SCAN_TIMING_LIMITATION"


def test_condition_gate_wrapper_requires_and_preserves_frozen_order() -> None:
    result = R.evaluate_condition_gates(
        {condition_id: _passing_metrics() for condition_id in reversed(C.CONDITION_IDS)}
    )
    assert tuple(result) == C.CONDITION_IDS
    assert all(row["pass"] for row in result.values())
    with pytest.raises(R.ReportingError, match="exactly the four"):
        R.evaluate_condition_gates({C.CONDITION_IDS[0]: _passing_metrics()})
