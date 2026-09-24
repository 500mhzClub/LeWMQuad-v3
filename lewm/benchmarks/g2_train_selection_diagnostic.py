"""Scene-balanced train/selection diagnostics for the Go2 G2 model.

This module contains only report shaping and interpretation helpers.  It does
not open dataset shards.  The standalone diagnostic script is responsible for
restricting evaluation to the ``train`` and ``checkpoint_selection`` roles.
"""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, median
from typing import Any, Mapping, Sequence


ALLOWED_DIAGNOSTIC_ROLES = frozenset({"train", "checkpoint_selection"})
CONFIGURATION_OCCUPANCY_TARGET_SPACE = "body_inflated_configuration_space"
PHYSICAL_OCCUPANCY_TARGET_SPACE = "observable_physical_occupancy"

OCCUPANCY_CHECKS = (
    "planner_admitted_free_precision_ge_0_99",
    "obstacle_recall_within_2m_ge_0_95",
    "obstacle_exclusion_within_2m_ge_0_95",
    "useful_traversable_recall_ge_0_90",
)
PHYSICAL_OCCUPANCY_CHECKS = (
    "admitted_observable_physical_free_precision_ge_0_99",
    "directly_observable_physical_obstacle_recall_within_2m_ge_0_95",
    "useful_observable_physical_free_recall_ge_0_90",
)
REPRESENTATION_PREDICTION_CHECKS = (
    "observed_predictor_beats_warped_persistence",
    "observed_target_change_is_nontrivial",
    "changed_predictor_beats_warped_persistence",
    "changed_target_change_is_nontrivial",
    "observed_real_action_beats_zero_by_0_10",
    "observed_real_action_beats_shuffled_by_0_10",
    "changed_real_action_beats_zero_by_0_10",
    "changed_real_action_beats_shuffled_by_0_10",
    "wrong_commanded_delta_is_worse_on_changed_cells",
    "target_representation_not_collapsed",
    "target_effective_rank_ge_4",
)
ROUTING_CHECKS = (
    "predicted_routes_no_worse_than_oracle",
    "nonzero_oracle_route_panel",
    "predicted_route_success_rate_ge_0_90",
    "predicted_route_length_recall_ge_0_90",
)

# ``direction`` describes improvement, and lets the comparison report express
# every gap with the same sign: positive means selection is worse than train.
METRIC_SPECS: dict[str, tuple[tuple[str, ...], str]] = {
    "total_loss": (("losses", "loss"), "lower"),
    "jepa_loss": (("losses", "jepa_loss"), "lower"),
    "occupancy_loss": (("losses", "occupancy_loss"), "lower"),
    "planner_admitted_free_precision": (
        ("traversability", "planner_admitted_free_precision"),
        "higher",
    ),
    "obstacle_detection_recall_within_range": (
        ("traversability", "obstacle_detection_recall_within_range"),
        "higher",
    ),
    "obstacle_exclusion_recall_within_range": (
        ("traversability", "obstacle_exclusion_recall_within_range"),
        "higher",
    ),
    "useful_traversable_recall": (
        ("traversability", "useful_traversable_recall"),
        "higher",
    ),
    "free_probability_ece": (
        ("traversability", "free_probability_ece"),
        "lower",
    ),
    "free_probability_brier": (
        ("traversability", "free_probability_brier"),
        "lower",
    ),
    "unknown_admission_rate": (
        ("traversability", "unknown_admission_rate"),
        "lower",
    ),
    "observed_prediction_to_persistence_ratio": (
        (
            "predictive_controls",
            "panels",
            "observed",
            "prediction_to_warped_persistence_ratio",
        ),
        "lower",
    ),
    "changed_prediction_to_persistence_ratio": (
        (
            "predictive_controls",
            "panels",
            "changed",
            "prediction_to_warped_persistence_ratio",
        ),
        "lower",
    ),
    "observed_shuffled_action_advantage": (
        (
            "predictive_controls",
            "panels",
            "observed",
            "shuffled_action_advantage_over_target_change",
        ),
        "higher",
    ),
    "changed_shuffled_action_advantage": (
        (
            "predictive_controls",
            "panels",
            "changed",
            "shuffled_action_advantage_over_target_change",
        ),
        "higher",
    ),
    "target_effective_rank": (
        ("predictive_controls", "target_cross_sample_effective_rank_mean"),
        "higher",
    ),
    "route_success_rate": (("routing", "route_success_rate"), "higher"),
    "mean_route_length_recall": (
        ("routing", "mean_route_length_recall"),
        "higher",
    ),
    "planned_path_collision_rate": (
        ("routing", "planned_path_collision_rate"),
        "lower",
    ),
}
PHYSICAL_METRIC_SPECS = {
    name: value
    for name, value in METRIC_SPECS.items()
    if name
    not in {
        "planner_admitted_free_precision",
        "obstacle_detection_recall_within_range",
        "obstacle_exclusion_recall_within_range",
        "useful_traversable_recall",
        "unknown_admission_rate",
        "free_probability_brier",
        "free_probability_ece",
        "route_success_rate",
        "mean_route_length_recall",
        "planned_path_collision_rate",
    }
}
PHYSICAL_METRIC_SPECS.update(
    {
        "admitted_observable_physical_free_precision": (
            (
                "physical_evidence",
                "admitted_observable_physical_free_precision",
            ),
            "higher",
        ),
        "directly_observable_physical_obstacle_recall_within_2m": (
            (
                "physical_evidence",
                "directly_observable_physical_obstacle_recall_within_2m",
            ),
            "higher",
        ),
        "observable_physical_obstacle_exclusion_recall_within_2m": (
            (
                "physical_evidence",
                "observable_physical_obstacle_exclusion_recall_within_2m",
            ),
            "higher",
        ),
        "useful_observable_physical_free_recall": (
            ("physical_evidence", "useful_observable_physical_free_recall"),
            "higher",
        ),
        "unknown_evidence_admission_rate": (
            ("physical_evidence", "unknown_evidence_admission_rate"),
            "lower",
        ),
        "free_probability_brier": (
            ("physical_evidence", "free_probability_brier"),
            "lower",
        ),
        "free_probability_ece": (
            ("physical_evidence", "free_probability_ece"),
            "lower",
        ),
    }
)


def _metric_specs_for_target(
    occupancy_target_space: str,
) -> Mapping[str, tuple[tuple[str, ...], str]]:
    if occupancy_target_space == CONFIGURATION_OCCUPANCY_TARGET_SPACE:
        return METRIC_SPECS
    if occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE:
        return PHYSICAL_METRIC_SPECS
    raise ValueError(f"unsupported occupancy target space: {occupancy_target_space!r}")


def _nested_number(payload: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"evaluation metrics are missing {'.'.join(path)}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"evaluation metric {'.'.join(path)} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"evaluation metric {'.'.join(path)} is not finite")
    return result


def select_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    scene_roles: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Return rows for the two permitted roles without opening any row artifact."""

    selected = {role: [] for role in sorted(ALLOWED_DIAGNOSTIC_ROLES)}
    for raw_row in rows:
        row = dict(raw_row)
        scene_id = str(row.get("scene_id", ""))
        if scene_id not in scene_roles:
            raise ValueError(f"row scene has no role assignment: {scene_id!r}")
        role = str(scene_roles[scene_id])
        direct_role = row.get("dataset_role")
        if direct_role is not None and str(direct_role) != role:
            raise ValueError(
                f"row role disagrees with scene role for {scene_id!r}: "
                f"{direct_role!r} != {role!r}"
            )
        if role in ALLOWED_DIAGNOSTIC_ROLES:
            selected[role].append(row)
    for role, role_rows in selected.items():
        if not role_rows:
            raise ValueError(f"diagnostic role has no rows: {role}")
    return selected


def compact_scene_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the fixed set of paper-relevant metrics from one evaluation."""

    g2 = metrics.get("g2")
    if not isinstance(g2, Mapping) or not isinstance(g2.get("checks"), Mapping):
        raise ValueError("evaluation metrics lack G2 checks")
    checks = {str(key): bool(value) for key, value in g2["checks"].items()}
    rows = metrics.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("evaluation metrics must contain a positive integer row count")
    occupancy_target_space = str(metrics.get("occupancy_target_space", ""))
    metric_specs = _metric_specs_for_target(occupancy_target_space)
    return {
        "rows": int(rows),
        "occupancy_target_space": occupancy_target_space,
        "g2_passes": bool(g2.get("passes", False)),
        "g2_checks_passed": sum(checks.values()),
        "g2_checks_total": len(checks),
        "g2_checks": checks,
        "metrics": {
            name: _nested_number(metrics, path)
            for name, (path, _direction) in metric_specs.items()
        },
    }


def _component_check_summary(
    metrics: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    g2 = metrics.get("g2")
    checks = g2.get("checks") if isinstance(g2, Mapping) else None
    if not isinstance(checks, Mapping):
        raise ValueError("evaluation metrics lack G2 checks")
    missing = sorted(set(names) - set(checks))
    if missing:
        raise ValueError(f"evaluation metrics lack component checks: {missing}")
    selected = {name: bool(checks[name]) for name in names}
    return {
        "passes": all(selected.values()),
        "checks_passed": sum(selected.values()),
        "check_count": len(selected),
        "checks": selected,
    }


def decompose_role_evaluations(
    calibrated_frozen: Mapping[str, Any],
    uncalibrated_frozen: Mapping[str, Any],
    calibrated_role_local: Mapping[str, Any],
    uncalibrated_role_local: Mapping[str, Any],
    *,
    occupancy_target_space: str,
) -> dict[str, Any]:
    """Separate representation, calibration, threshold, and routing evidence."""

    physical_target = occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE
    _metric_specs_for_target(occupancy_target_space)
    for view in (
        calibrated_frozen,
        uncalibrated_frozen,
        calibrated_role_local,
        uncalibrated_role_local,
    ):
        if view.get("occupancy_target_space") != occupancy_target_space:
            raise ValueError("evaluation view occupancy target-space mismatch")
    calibrated_contract = calibrated_frozen.get("calibration")
    uncalibrated_contract = uncalibrated_frozen.get("calibration")
    calibrated_local_contract = calibrated_role_local.get("calibration")
    uncalibrated_local_contract = uncalibrated_role_local.get("calibration")
    if not all(
        isinstance(item, Mapping)
        for item in (
            calibrated_contract,
            uncalibrated_contract,
            calibrated_local_contract,
            uncalibrated_local_contract,
        )
    ):
        raise ValueError("evaluation views lack calibration contracts")
    if not bool(calibrated_contract["applied"]):
        raise ValueError("frozen promotion view must apply checkpoint calibration")
    if bool(uncalibrated_contract["applied"]):
        raise ValueError("uncalibrated diagnostic view applied calibration")
    if not bool(calibrated_local_contract["applied"]):
        raise ValueError("role-local threshold view must apply checkpoint calibration")
    if bool(uncalibrated_local_contract["applied"]):
        raise ValueError("raw role-local threshold view applied calibration")
    if calibrated_frozen.get("threshold_selection") is not None:
        raise ValueError("frozen promotion view must not reselect thresholds")
    calibrated_local_selection = calibrated_role_local.get("threshold_selection")
    uncalibrated_local_selection = uncalibrated_role_local.get(
        "threshold_selection"
    )
    if not isinstance(calibrated_local_selection, Mapping) or not isinstance(
        uncalibrated_local_selection, Mapping
    ):
        raise ValueError("role-local diagnostic views lack threshold sweeps")

    calibrated_traversability = calibrated_frozen.get("traversability")
    uncalibrated_traversability = uncalibrated_frozen.get("traversability")
    calibrated_local_traversability = calibrated_role_local.get("traversability")
    uncalibrated_local_traversability = uncalibrated_role_local.get(
        "traversability"
    )
    if not all(
        isinstance(item, Mapping)
        for item in (
            calibrated_traversability,
            uncalibrated_traversability,
            calibrated_local_traversability,
            uncalibrated_local_traversability,
        )
    ):
        raise ValueError("evaluation views lack traversability metrics")

    quality_names = ("free_probability_brier", "free_probability_ece")
    uncalibrated_quality = {
        name: _nested_number(uncalibrated_traversability, (name,))
        for name in quality_names
    }
    calibrated_quality = {
        name: _nested_number(calibrated_traversability, (name,))
        for name in quality_names
    }
    quality_delta = {
        name: calibrated_quality[name] - uncalibrated_quality[name]
        for name in quality_names
    }

    occupancy_checks = PHYSICAL_OCCUPANCY_CHECKS if physical_target else OCCUPANCY_CHECKS
    frozen_occupancy = _component_check_summary(calibrated_frozen, occupancy_checks)
    calibrated_local_occupancy = _component_check_summary(
        calibrated_role_local, occupancy_checks
    )
    uncalibrated_local_occupancy = _component_check_summary(
        uncalibrated_role_local, occupancy_checks
    )

    def candidate_counts(selection: Mapping[str, Any]) -> tuple[int, int]:
        candidate_count = int(selection["candidate_count"])
        passing_candidates = int(selection["passing_candidate_count"])
        if candidate_count <= 0 or not 0 <= passing_candidates <= candidate_count:
            raise ValueError("invalid role-local threshold candidate counts")
        return candidate_count, passing_candidates

    calibrated_candidate_count, calibrated_passing_candidates = candidate_counts(
        calibrated_local_selection
    )
    uncalibrated_candidate_count, uncalibrated_passing_candidates = (
        candidate_counts(uncalibrated_local_selection)
    )
    if calibrated_candidate_count != uncalibrated_candidate_count:
        raise ValueError("calibrated/raw threshold sweeps use different candidate grids")

    if uncalibrated_local_occupancy["passes"]:
        raw_head_read = "raw_head_attains_full_occupancy_gate_with_role_local_thresholds"
    elif uncalibrated_passing_candidates > 0:
        raw_head_read = (
            "raw_head_has_registered_grid_safety_candidates_but_full_gate_fails"
        )
    else:
        raw_head_read = "raw_head_has_no_registered_grid_safety_candidate"

    if (
        uncalibrated_local_occupancy["passes"]
        and not calibrated_local_occupancy["passes"]
    ):
        calibration_attainability_read = (
            "checkpoint_calibration_destroys_full_gate_attainability"
        )
    elif (
        uncalibrated_passing_candidates > 0
        and calibrated_passing_candidates == 0
    ):
        calibration_attainability_read = (
            "checkpoint_calibration_destroys_safety_candidate_attainability"
        )
    elif (
        not uncalibrated_local_occupancy["passes"]
        and calibrated_local_occupancy["passes"]
    ):
        calibration_attainability_read = (
            "checkpoint_calibration_restores_full_gate_attainability"
        )
    elif (
        uncalibrated_passing_candidates == 0
        and calibrated_passing_candidates > 0
    ):
        calibration_attainability_read = (
            "checkpoint_calibration_restores_safety_candidate_attainability"
        )
    else:
        calibration_attainability_read = (
            "checkpoint_calibration_preserves_attainability_category"
        )

    if frozen_occupancy["passes"]:
        threshold_read = "frozen_promotion_thresholds_attain_occupancy_gate"
    elif (
        uncalibrated_local_occupancy["passes"]
        and not calibrated_local_occupancy["passes"]
    ):
        threshold_read = "raw_head_attains_gate_but_checkpoint_calibration_destroys_it"
    elif (
        calibrated_local_occupancy["passes"]
        and not uncalibrated_local_occupancy["passes"]
    ):
        threshold_read = "checkpoint_calibration_restores_role_local_gate_attainability"
    elif calibrated_local_occupancy["passes"]:
        threshold_read = "role_local_thresholds_attain_gate_but_frozen_do_not"
    elif calibrated_passing_candidates > 0:
        threshold_read = "role_local_safety_candidates_exist_but_full_gate_fails"
    elif uncalibrated_passing_candidates > 0:
        threshold_read = (
            "raw_head_has_safety_candidates_but_calibration_removes_them"
        )
    else:
        threshold_read = "raw_head_has_no_role_local_safety_candidate_in_registered_grid"

    losses = calibrated_frozen.get("losses")
    predictive = calibrated_frozen.get("predictive_controls")
    frozen_routing_metrics = calibrated_frozen.get("routing")
    calibrated_local_routing_metrics = calibrated_role_local.get("routing")
    uncalibrated_local_routing_metrics = uncalibrated_role_local.get("routing")
    if not all(
        isinstance(item, Mapping)
        for item in (
            losses,
            predictive,
            frozen_routing_metrics,
            calibrated_local_routing_metrics,
            uncalibrated_local_routing_metrics,
        )
    ):
        raise ValueError("evaluation views lack component metrics")
    if physical_target:
        for routing in (
            frozen_routing_metrics,
            calibrated_local_routing_metrics,
            uncalibrated_local_routing_metrics,
        ):
            if (
                routing.get("applicability") != "not_applicable"
                or routing.get("excluded_from_gate") is not True
                or routing.get("valid_for_target_space") is not False
            ):
                raise ValueError("physical evaluation exposed applicable routing")
        representation = {
            "applicability": "diagnostic_only",
            "included_in_head_g2": False,
        }
        routing_result = {
            "applicability": "not_applicable",
            "included_in_head_g2": False,
            "deferred_to": "G3_post_memory_multi_view_fusion",
        }
    else:
        representation = _component_check_summary(
            calibrated_frozen, REPRESENTATION_PREDICTION_CHECKS
        )
        routing_frozen = _component_check_summary(calibrated_frozen, ROUTING_CHECKS)
        calibrated_routing_local = _component_check_summary(
            calibrated_role_local, ROUTING_CHECKS
        )
        uncalibrated_routing_local = _component_check_summary(
            uncalibrated_role_local, ROUTING_CHECKS
        )
        routing_result = {
            "frozen_promotion": {
                "gate": routing_frozen,
                "metrics": dict(frozen_routing_metrics),
            },
            "calibrated_role_local_diagnostic": {
                "gate": calibrated_routing_local,
                "metrics": dict(calibrated_local_routing_metrics),
            },
            "uncalibrated_role_local_diagnostic": {
                "gate": uncalibrated_routing_local,
                "metrics": dict(uncalibrated_local_routing_metrics),
            },
        }
    return {
        "occupancy_target_space": occupancy_target_space,
        "representation_prediction": {
            "gate": representation,
            "losses": dict(losses),
            "predictive_controls": dict(predictive),
        },
        "occupancy_probability_quality": {
            "scope": "known_truth_free_vs_occupied",
            "uncalibrated": uncalibrated_quality,
            "checkpoint_calibrated": calibrated_quality,
            "calibrated_minus_uncalibrated_negative_is_improvement": quality_delta,
            "checkpoint_calibration_improves_or_preserves_both": all(
                delta <= 0.0 for delta in quality_delta.values()
            ),
        },
        "threshold_attainability": {
            "frozen_promotion": {
                "thresholds": dict(calibrated_frozen["thresholds"]),
                "occupancy_gate": frozen_occupancy,
                "traversability": dict(calibrated_traversability),
            },
            "calibrated_role_local_diagnostic": {
                "thresholds": dict(calibrated_role_local["thresholds"]),
                "occupancy_gate": calibrated_local_occupancy,
                "candidate_count": calibrated_candidate_count,
                "safety_passing_candidate_count": calibrated_passing_candidates,
                "traversability": dict(calibrated_local_traversability),
            },
            "uncalibrated_role_local_diagnostic": {
                "thresholds": dict(uncalibrated_role_local["thresholds"]),
                "occupancy_gate": uncalibrated_local_occupancy,
                "candidate_count": uncalibrated_candidate_count,
                "safety_passing_candidate_count": uncalibrated_passing_candidates,
                "traversability": dict(uncalibrated_local_traversability),
            },
            "bounded_read": threshold_read,
            "raw_head_bounded_read": raw_head_read,
            "calibration_effect_on_attainability": calibration_attainability_read,
            "caveat": (
                "Role-local thresholds are diagnostic upper bounds fitted and "
                "scored on the same role; they are not promotion thresholds."
            ),
        },
        "routing": routing_result,
    }


def scene_balanced_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize scene records with equal weight per scene."""

    if not records:
        raise ValueError("at least one scene record is required")
    metric_values: dict[str, list[float]] = defaultdict(list)
    scene_passes: list[float] = []
    check_fractions: list[float] = []
    row_count = 0
    occupancy_target_space: str | None = None
    metric_specs: Mapping[str, tuple[tuple[str, ...], str]] | None = None
    for record in records:
        compact = record.get("evaluation")
        if not isinstance(compact, Mapping):
            raise ValueError("each scene record requires an evaluation object")
        row_count += int(compact["rows"])
        record_target = str(compact.get("occupancy_target_space", ""))
        if occupancy_target_space is None:
            occupancy_target_space = record_target
            metric_specs = _metric_specs_for_target(record_target)
        elif record_target != occupancy_target_space:
            raise ValueError("scene records mix occupancy target spaces")
        scene_passes.append(float(bool(compact["g2_passes"])))
        total = int(compact["g2_checks_total"])
        check_fractions.append(float(compact["g2_checks_passed"]) / max(1, total))
        values = compact.get("metrics")
        if not isinstance(values, Mapping):
            raise ValueError("scene evaluation lacks compact metrics")
        assert metric_specs is not None
        for name in metric_specs:
            value = float(values[name])
            if not math.isfinite(value):
                raise ValueError(f"scene metric is not finite: {name}")
            metric_values[name].append(value)
    return {
        "occupancy_target_space": occupancy_target_space,
        "scene_count": len(records),
        "row_count": row_count,
        "scene_g2_pass_fraction": mean(scene_passes),
        "scene_g2_check_fraction_mean": mean(check_fractions),
        "metrics": {
            name: {
                "direction": metric_specs[name][1],
                "mean": mean(values),
                "median": median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for name, values in sorted(metric_values.items())
        },
    }


def family_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return per-family scene-balanced summaries and a family-balanced mean."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        family = str(record.get("family", ""))
        if not family:
            raise ValueError("scene record lacks family")
        grouped[family].append(record)
    per_family = {
        family: scene_balanced_summary(items)
        for family, items in sorted(grouped.items())
    }
    targets = {
        str(summary["occupancy_target_space"]) for summary in per_family.values()
    }
    if len(targets) != 1:
        raise ValueError("family summaries mix occupancy target spaces")
    occupancy_target_space = next(iter(targets))
    metric_specs = _metric_specs_for_target(occupancy_target_space)
    return {
        "occupancy_target_space": occupancy_target_space,
        "family_count": len(per_family),
        "per_family": per_family,
        "family_balanced_metric_means": {
            name: mean(
                summary["metrics"][name]["mean"] for summary in per_family.values()
            )
            for name in metric_specs
        },
    }


def _linear_slope(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    x_mean = mean(point[0] for point in points)
    y_mean = mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _y in points)
    if denominator <= 0.0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def learning_curve_inputs(report: Mapping[str, Any]) -> dict[str, Any]:
    """Extract explicit optimization/generalization inputs from a trainer report."""

    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("training report has no nonempty history")
    records: list[dict[str, float | int]] = []
    for raw in history:
        if not isinstance(raw, Mapping):
            raise ValueError("training history rows must be objects")
        epoch = int(raw["epoch"])
        train = raw.get("train")
        selection = raw.get("checkpoint_selection")
        if not isinstance(train, Mapping) or not isinstance(selection, Mapping):
            raise ValueError("training history row lacks train/selection metrics")
        checks = selection.get("g2", {}).get("checks")
        if not isinstance(checks, Mapping):
            raise ValueError("selection history lacks G2 checks")
        records.append(
            {
                "epoch": epoch,
                "train_loss": _nested_number(train, ("loss",)),
                "train_occupancy_loss": _nested_number(
                    train, ("occupancy_loss",)
                ),
                "train_jepa_loss": _nested_number(train, ("jepa_loss",)),
                "selection_loss": _nested_number(selection, ("losses", "loss")),
                "selection_occupancy_loss": _nested_number(
                    selection, ("losses", "occupancy_loss")
                ),
                "selection_jepa_loss": _nested_number(
                    selection, ("losses", "jepa_loss")
                ),
                "selection_checks_passed": sum(bool(value) for value in checks.values()),
                "selection_free_precision": _nested_number(
                    selection,
                    ("traversability", "planner_admitted_free_precision"),
                ),
                "selection_traversable_recall": _nested_number(
                    selection,
                    ("traversability", "useful_traversable_recall"),
                ),
                "selection_obstacle_recall": _nested_number(
                    selection,
                    ("traversability", "obstacle_detection_recall_within_range"),
                ),
            }
        )
    epochs = [int(item["epoch"]) for item in records]
    if epochs != sorted(set(epochs)):
        raise ValueError("training history epochs must be unique and increasing")
    late_count = min(len(records), max(5, math.ceil(len(records) * 0.25)))
    late = records[-late_count:]

    def slope(name: str) -> float | None:
        return _linear_slope(
            [(float(item["epoch"]), float(item[name])) for item in late]
        )

    first, last = records[0], records[-1]
    best_epoch = int(report.get("best_epoch", 0))
    best = next(
        (record for record in records if int(record["epoch"]) == best_epoch),
        None,
    )
    if best is None:
        raise ValueError("training report best_epoch is absent from history")
    initial_loss = float(first["train_loss"])
    final_loss = float(last["train_loss"])
    return {
        "epoch_count": len(records),
        "best_epoch": best_epoch,
        "epochs_after_selected_checkpoint": int(last["epoch"]) - best_epoch,
        "selected_checkpoint_record": dict(best),
        "last_epoch_record": dict(last),
        "late_window_epoch_count": late_count,
        "train_loss_first": initial_loss,
        "train_loss_last": final_loss,
        "train_loss_fractional_change": (
            (final_loss - initial_loss) / initial_loss if initial_loss else None
        ),
        "selection_checks_passed_first": int(first["selection_checks_passed"]),
        "selection_checks_passed_last": int(last["selection_checks_passed"]),
        "selection_checks_passed_maximum": max(
            int(item["selection_checks_passed"]) for item in records
        ),
        "late_slopes_per_epoch": {
            "train_loss": slope("train_loss"),
            "train_occupancy_loss": slope("train_occupancy_loss"),
            "train_jepa_loss": slope("train_jepa_loss"),
            "selection_loss": slope("selection_loss"),
            "selection_occupancy_loss": slope("selection_occupancy_loss"),
            "selection_jepa_loss": slope("selection_jepa_loss"),
            "selection_checks_passed": slope("selection_checks_passed"),
            "selection_free_precision": slope("selection_free_precision"),
            "selection_traversable_recall": slope("selection_traversable_recall"),
            "selection_obstacle_recall": slope("selection_obstacle_recall"),
        },
        "caveat": (
            "Train losses are within-epoch optimization averages. Selection curves "
            "use uncalibrated logits and reselect thresholds on the selection role "
            "at every epoch, while the saved runtime uses calibration-role-fitted "
            "calibration and thresholds. These curves are not directly comparable "
            "to the final frozen operating point and cannot by themselves distinguish "
            "finite data from optimization, capacity, objective, or inductive bias."
        ),
    }


def compare_roles(
    train_pooled: Mapping[str, Any],
    selection_pooled: Mapping[str, Any],
    train_scene_summary: Mapping[str, Any],
    selection_scene_summary: Mapping[str, Any],
    *,
    curve_inputs: Mapping[str, Any] | None,
    train_decomposition: Mapping[str, Any] | None = None,
    selection_decomposition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build bounded train-vs-selection inputs without overclaiming causality."""

    train_compact = compact_scene_metrics(train_pooled)
    selection_compact = compact_scene_metrics(selection_pooled)
    occupancy_target_space = str(train_compact["occupancy_target_space"])
    if (
        selection_compact["occupancy_target_space"] != occupancy_target_space
        or train_scene_summary.get("occupancy_target_space")
        != occupancy_target_space
        or selection_scene_summary.get("occupancy_target_space")
        != occupancy_target_space
    ):
        raise ValueError("role comparison mixes occupancy target spaces")
    metric_specs = _metric_specs_for_target(occupancy_target_space)
    gaps: dict[str, Any] = {}
    for name, (_path, direction) in metric_specs.items():
        train_value = float(train_scene_summary["metrics"][name]["mean"])
        selection_value = float(selection_scene_summary["metrics"][name]["mean"])
        raw_gap = selection_value - train_value
        directional_gap = -raw_gap if direction == "higher" else raw_gap
        gaps[name] = {
            "direction": direction,
            "train_scene_mean": train_value,
            "selection_scene_mean": selection_value,
            "selection_minus_train": raw_gap,
            "directional_gap_positive_means_selection_worse": directional_gap,
        }

    if (train_decomposition is None) != (selection_decomposition is None):
        raise ValueError("train and selection decompositions must be supplied together")
    component_inputs = None
    if train_decomposition is not None and selection_decomposition is not None:
        component_paths = {
            "occupancy_at_frozen_thresholds": (
                "threshold_attainability",
                "frozen_promotion",
                "occupancy_gate",
            ),
        }
        if occupancy_target_space == CONFIGURATION_OCCUPANCY_TARGET_SPACE:
            component_paths.update(
                {
                    "representation_prediction": (
                        "representation_prediction",
                        "gate",
                    ),
                    "routing_at_frozen_thresholds": (
                        "routing",
                        "frozen_promotion",
                        "gate",
                    ),
                }
            )

        def component_gate(
            payload: Mapping[str, Any], path: Sequence[str]
        ) -> Mapping[str, Any]:
            value: Any = payload
            for key in path:
                if not isinstance(value, Mapping) or key not in value:
                    raise ValueError(f"role decomposition lacks {'.'.join(path)}")
                value = value[key]
            if not isinstance(value, Mapping) or "passes" not in value:
                raise ValueError(f"role decomposition gate is invalid: {'.'.join(path)}")
            return value

        component_inputs = {
            name: {
                "train": dict(component_gate(train_decomposition, path)),
                "checkpoint_selection": dict(
                    component_gate(selection_decomposition, path)
                ),
            }
            for name, path in component_paths.items()
        }

    gate_label = (
        "physical-evidence head contract"
        if occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE
        else "runtime contract"
    )
    if not train_compact["g2_passes"]:
        bounded_read = (
            "train_role_physical_head_failure_blocks_generalization_attribution"
            if occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE
            else "train_role_runtime_contract_failure_blocks_generalization_attribution"
        )
        explanation = (
            f"The checkpoint fails the pooled frozen {gate_label} on training "
            "scenes. The component views identify where it fails, but this comparison "
            "cannot call that an optimization failure or isolate finite data from "
            "capacity, objective, calibration transfer, threshold transfer, or "
            "observability limits."
        )
    elif not selection_compact["g2_passes"]:
        bounded_read = "selection_role_generalization_gap_observed"
        explanation = (
            "The checkpoint passes pooled training evaluation but fails checkpoint "
            "selection under the same frozen operating point. This is a descriptive "
            "development-role gap, but selection was used adaptively to choose the "
            "checkpoint and the gap does not distinguish finite data from "
            "regularization or inductive bias."
        )
    else:
        bounded_read = (
            "no_train_selection_physical_head_failure_observed"
            if occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE
            else "no_train_selection_runtime_failure_observed"
        )
        explanation = (
            "Both permitted roles pass the pooled gate; this diagnostic does not "
            "identify the cause of any later untouched-role failure."
        )

    return {
        "occupancy_target_space": occupancy_target_space,
        "bounded_read": bounded_read,
        "explanation": explanation,
        "pooled_gate_inputs": {
            "train_passes": train_compact["g2_passes"],
            "selection_passes": selection_compact["g2_passes"],
            "train_checks_passed": train_compact["g2_checks_passed"],
            "selection_checks_passed": selection_compact["g2_checks_passed"],
            "check_count": train_compact["g2_checks_total"],
        },
        "scene_gate_inputs": {
            "train_scene_pass_fraction": train_scene_summary[
                "scene_g2_pass_fraction"
            ],
            "selection_scene_pass_fraction": selection_scene_summary[
                "scene_g2_pass_fraction"
            ],
            "train_scene_check_fraction_mean": train_scene_summary[
                "scene_g2_check_fraction_mean"
            ],
            "selection_scene_check_fraction_mean": selection_scene_summary[
                "scene_g2_check_fraction_mean"
            ],
            "train_all_scenes_pass": float(
                train_scene_summary["scene_g2_pass_fraction"]
            )
            == 1.0,
            "selection_all_scenes_pass": float(
                selection_scene_summary["scene_g2_pass_fraction"]
            )
            == 1.0,
            "pooled_and_all_scene_gate_disagree": {
                "train": bool(train_compact["g2_passes"])
                != (
                    float(train_scene_summary["scene_g2_pass_fraction"])
                    == 1.0
                ),
                "checkpoint_selection": bool(selection_compact["g2_passes"])
                != (
                    float(selection_scene_summary["scene_g2_pass_fraction"])
                    == 1.0
                ),
            },
        },
        "scene_balanced_metric_gaps": gaps,
        "component_gate_inputs": component_inputs,
        "learning_curve_inputs": None if curve_inputs is None else dict(curve_inputs),
        "non_conclusions": [
            "This report does not evaluate or infer untouched G2 performance.",
            "A falling train loss does not prove that adding rows will fix the model.",
            "A train-selection gap does not distinguish data volume from model bias.",
            "Role-local diagnostic thresholds are not valid promotion thresholds.",
            "Checkpoint-selection performance is adaptively selected and optimistic.",
        ],
    }


__all__ = [
    "ALLOWED_DIAGNOSTIC_ROLES",
    "METRIC_SPECS",
    "OCCUPANCY_CHECKS",
    "REPRESENTATION_PREDICTION_CHECKS",
    "ROUTING_CHECKS",
    "compact_scene_metrics",
    "compare_roles",
    "decompose_role_evaluations",
    "family_summaries",
    "learning_curve_inputs",
    "scene_balanced_summary",
    "select_diagnostic_rows",
]
