"""Pure contracts and decisions for categorical-radial N32 execution."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from lewm.benchmarks.go2_physical_micro_overfit import (
    FAMILIES,
    fit_gate,
)


EXECUTION_BINDING_SHA256 = (
    "42c2ce88ac78f045b92fdd2b33ad5b77a0801de0af2e05c79d3bb518ca188241"
)
RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_result_v1"
SMOKE_RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_smoke_result_v1"
TWO_SEED_RESULT_SCHEMA = "lewm_go2_categorical_radial_n32_two_seed_result_v1"
FIT_GATE_SCHEMA = "lewm_go2_categorical_radial_n32_fit_gate_v1"
TERMINAL_FIT_SCHEMA = "lewm_go2_categorical_radial_n32_terminal_fit_gate_v1"
REFERENCE_SCHEMA = "lewm_go2_categorical_radial_n32_patch7_reference_v1"
HOLDOUT_CHECK_SCHEMA = "lewm_go2_categorical_radial_n32_holdout_checks_v1"
PER_SEED_DECISION_SCHEMA = "lewm_go2_categorical_radial_n32_seed_decision_v1"
HOLDOUT_PANELS = ("same_scene_holdout", "cross_scene_holdout")
CONDITIONS = (
    "correct_rgb",
    "role_global_shuffled_rgb",
    "same_scene_wrong_view_rgb",
)
CLASS_NAMES = ("unknown", "free", "occupied")
PATCH7_FINAL_STATE_SHA256 = (
    "fba4e91b333d57a813fb94edb13b215064d03da2830aae9d0ae4b34685cd38c1"
)
REFERENCE_MACRO_ASSERTIONS = {
    "same_scene_holdout": {
        "hierarchical_nll": 0.3219876256599372,
        "far_free_recall": 0.46708481911812594,
    },
    "cross_scene_holdout": {
        "hierarchical_nll": 0.4054638461731662,
        "far_free_recall": 0.4665871805991353,
    },
}


def _required_metric(metrics: Mapping[str, Any], *path: str) -> float:
    value: Any = metrics
    for name in path:
        if not isinstance(value, Mapping) or name not in value:
            raise ValueError(f"metrics lack {'/'.join(path)}")
        value = value[name]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"metrics contain invalid {'/'.join(path)}")
    return float(value)


def _conditions(report: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    conditions = report.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
        raise ValueError("N32 report conditions are incomplete")
    if not all(isinstance(conditions[name], Mapping) for name in CONDITIONS):
        raise ValueError("N32 report condition metrics are malformed")
    return conditions


def _gate_for_conditions(
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    correct = conditions["correct_rgb"]
    role_global_nll = _required_metric(
        conditions["role_global_shuffled_rgb"],
        "raw_hierarchical_balanced_nll",
    )
    same_scene_nll = _required_metric(
        conditions["same_scene_wrong_view_rgb"],
        "raw_hierarchical_balanced_nll",
    )
    return fit_gate(
        correct,
        cross_scene_shuffled_nll=role_global_nll,
        same_scene_shuffled_nll=same_scene_nll,
    )


def fit_panel_gate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the aggregate and canonical five-family fit gates."""

    aggregate_gate = _gate_for_conditions(_conditions(report))
    families = report.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError("N32 fit report must contain exactly five families")
    family_gates = {
        family: _gate_for_conditions(_conditions(families[family]))
        for family in FAMILIES
    }
    passes = bool(aggregate_gate["passes"]) and all(
        bool(family_gates[family]["passes"]) for family in FAMILIES
    )
    return {
        "schema": FIT_GATE_SCHEMA,
        "aggregate": aggregate_gate,
        "families": family_gates,
        "family_order": list(FAMILIES),
        "requires_aggregate_and_all_families": True,
        "passes": passes,
    }


def all_family_and_aggregate_fit_pass(report: Mapping[str, Any]) -> bool:
    """Return true only when the pooled and all five family gates pass."""

    return bool(fit_panel_gate_report(report)["passes"])


def terminal_fit_gate_summary(
    curve: Sequence[Mapping[str, Any]],
    max_steps: int,
    eval_interval: int,
) -> dict[str, Any]:
    """Recompute the fixed final-three decision over a complete fit curve."""

    max_steps = int(max_steps)
    eval_interval = int(eval_interval)
    if max_steps <= 0 or eval_interval <= 0 or max_steps % eval_interval:
        raise ValueError("N32 stage budget must be positive and exactly divisible")
    expected_steps = list(range(eval_interval, max_steps + 1, eval_interval))
    if len(expected_steps) < 3:
        raise ValueError("N32 terminal decision requires three evaluations")
    actual_steps = [int(point.get("step", -1)) for point in curve]
    if actual_steps != expected_steps:
        raise ValueError("N32 fit curve cadence or fixed budget is incomplete")
    evaluation_passes = []
    for point in curve:
        report = point.get("fit_panel", point.get("fit"))
        if not isinstance(report, Mapping):
            raise ValueError("N32 curve point lacks its fit-panel report")
        evaluation_passes.append(all_family_and_aggregate_fit_pass(report))
    terminal_steps = expected_steps[-3:]
    terminal_passes = evaluation_passes[-3:]
    first_single = next(
        (step for step, passes in zip(expected_steps, evaluation_passes) if passes),
        None,
    )
    consecutive = 0
    first_three = None
    for step, passes in zip(expected_steps, evaluation_passes):
        consecutive = consecutive + 1 if passes else 0
        if consecutive >= 3 and first_three is None:
            first_three = step
    return {
        "schema": TERMINAL_FIT_SCHEMA,
        "maximum_steps": max_steps,
        "evaluation_interval": eval_interval,
        "evaluation_steps": expected_steps,
        "evaluation_passes": evaluation_passes,
        "terminal_evaluation_steps": terminal_steps,
        "terminal_evaluation_passes": terminal_passes,
        "requires_exact_final_three": True,
        "first_single_fit_gate_step": first_single,
        "first_three_consecutive_fit_gate_step": first_three,
        "passes": all(terminal_passes),
    }


def _family_correct_metrics(
    panel: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    families = panel.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError("N32 panel must contain exactly five families")
    result = {}
    for family in FAMILIES:
        record = families[family]
        if not isinstance(record, Mapping):
            raise ValueError(f"malformed N32 family record: {family}")
        conditions = record.get("conditions")
        if not isinstance(conditions, Mapping):
            raise ValueError(f"N32 family lacks conditions: {family}")
        metrics = conditions.get("correct_rgb")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"N32 family lacks correct RGB metrics: {family}")
        result[family] = metrics
    return result


def _ordered_mean(values: Sequence[float]) -> float:
    if len(values) != len(FAMILIES):
        raise ValueError("N32 macro requires exactly five ordered family values")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def extract_faithful_patch7_family_reference(
    patch7_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract and validate the immutable faithful patch7 family comparator."""

    try:
        stage = patch7_result["stages"]["production_faithful"]["patch7_16x16"]
    except (KeyError, TypeError) as exc:
        raise ValueError("patch7 result lacks its faithful comparator") from exc
    if str(stage.get("final_state_sha256", "")) != PATCH7_FINAL_STATE_SHA256:
        raise ValueError("faithful patch7 final-state SHA-256 mismatch")
    final_panels = stage.get("final_panels")
    if not isinstance(final_panels, Mapping):
        raise ValueError("faithful patch7 result lacks final panels")
    panels = {}
    macro_assertions = {}
    for panel_name in HOLDOUT_PANELS:
        panel = final_panels.get(panel_name)
        if not isinstance(panel, Mapping):
            raise ValueError(f"faithful patch7 lacks {panel_name}")
        metrics = _family_correct_metrics(panel)
        nll = _ordered_mean(
            [
                _required_metric(metrics[family], "raw_hierarchical_balanced_nll")
                for family in FAMILIES
            ]
        )
        far = _ordered_mean(
            [
                _required_metric(
                    metrics[family], "distance_free_recall", "3.0_plus"
                )
                for family in FAMILIES
            ]
        )
        asserted = REFERENCE_MACRO_ASSERTIONS[panel_name]
        if not math.isclose(
            nll,
            asserted["hierarchical_nll"],
            rel_tol=0.0,
            abs_tol=5e-16,
        ) or not math.isclose(
            far,
            asserted["far_free_recall"],
            rel_tol=0.0,
            abs_tol=5e-16,
        ):
            raise ValueError(f"faithful patch7 {panel_name} macro assertion failed")
        panels[panel_name] = panel
        macro_assertions[panel_name] = {
            "hierarchical_nll": nll,
            "far_free_recall": far,
        }
    support = patch7_result.get("post_selection_support_audit")
    if not isinstance(support, Mapping) or set(support) != {
        "fit",
        *HOLDOUT_PANELS,
    }:
        raise ValueError("patch7 post-selection support audit is incomplete")
    return {
        "schema": REFERENCE_SCHEMA,
        "source_stage": "production_faithful",
        "source_arm": "patch7_16x16",
        "final_state_sha256": PATCH7_FINAL_STATE_SHA256,
        "panels": panels,
        "macro_assertions": macro_assertions,
    }


def categorical_holdout_checks(
    candidate_panel: Mapping[str, Any],
    reference_panel: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare candidate/reference family metrics in frozen family order."""

    panel_name = str(candidate_panel.get("panel", ""))
    reference_name = str(reference_panel.get("panel", panel_name))
    if panel_name not in HOLDOUT_PANELS or reference_name != panel_name:
        raise ValueError("categorical holdout panel name is invalid or mismatched")
    candidate = _family_correct_metrics(candidate_panel)
    reference = _family_correct_metrics(reference_panel)
    per_family = {}
    for family in FAMILIES:
        candidate_nll = _required_metric(
            candidate[family], "raw_hierarchical_balanced_nll"
        )
        reference_nll = _required_metric(
            reference[family], "raw_hierarchical_balanced_nll"
        )
        candidate_far = _required_metric(
            candidate[family], "distance_free_recall", "3.0_plus"
        )
        reference_far = _required_metric(
            reference[family], "distance_free_recall", "3.0_plus"
        )
        class_deltas = {
            name: _required_metric(candidate[family], "class_recall", name)
            - _required_metric(reference[family], "class_recall", name)
            for name in CLASS_NAMES
        }
        per_family[family] = {
            "candidate_hierarchical_nll": candidate_nll,
            "reference_hierarchical_nll": reference_nll,
            "candidate_far_free_recall": candidate_far,
            "reference_far_free_recall": reference_far,
            "candidate_minus_reference_class_recall": class_deltas,
            "strictly_lower_nll_and_higher_far_free": (
                candidate_nll < reference_nll and candidate_far > reference_far
            ),
        }
    candidate_macro_nll = _ordered_mean(
        [per_family[family]["candidate_hierarchical_nll"] for family in FAMILIES]
    )
    reference_macro_nll = _ordered_mean(
        [per_family[family]["reference_hierarchical_nll"] for family in FAMILIES]
    )
    ratio = (
        None
        if reference_macro_nll <= 0.0
        else candidate_macro_nll / reference_macro_nll
    )
    far_delta = _ordered_mean(
        [
            per_family[family]["candidate_far_free_recall"]
            - per_family[family]["reference_far_free_recall"]
            for family in FAMILIES
        ]
    )
    class_macro_deltas = {
        name: _ordered_mean(
            [
                per_family[family]["candidate_minus_reference_class_recall"][name]
                for family in FAMILIES
            ]
        )
        for name in CLASS_NAMES
    }
    favorable_count = sum(
        bool(per_family[family]["strictly_lower_nll_and_higher_far_free"])
        for family in FAMILIES
    )
    required_favorable = 5 if panel_name == "cross_scene_holdout" else 4
    checks = {
        "equal_weight_family_macro_nll_ratio_le_0_80": (
            ratio is not None and ratio <= 0.80
        ),
        "equal_weight_family_macro_far_free_delta_ge_0_10": far_delta >= 0.10,
        "every_macro_class_recall_delta_ge_neg_0_01": (
            min(class_macro_deltas.values()) >= -0.01
        ),
        "no_family_class_recall_delta_lt_neg_0_01": min(
            delta
            for family in FAMILIES
            for delta in per_family[family][
                "candidate_minus_reference_class_recall"
            ].values()
        )
        >= -0.01,
        f"strict_family_nll_and_far_improvement_ge_{required_favorable}_of_5": (
            favorable_count >= required_favorable
        ),
    }
    return {
        "schema": HOLDOUT_CHECK_SCHEMA,
        "panel": panel_name,
        "passes": all(checks.values()),
        "checks": checks,
        "family_order": list(FAMILIES),
        "family_macro_weighting": "equal_weight_across_five_families",
        "macro": {
            "candidate_hierarchical_nll": candidate_macro_nll,
            "reference_hierarchical_nll": reference_macro_nll,
            "candidate_minus_reference_far_free_recall": far_delta,
            "candidate_minus_reference_class_recall": class_macro_deltas,
        },
        "candidate_to_reference_macro_hierarchical_nll_ratio": ratio,
        "strictly_favorable_family_count": favorable_count,
        "strictly_favorable_family_requirement": required_favorable,
        "ties_count_as_failure": True,
        "per_family": per_family,
    }


def _terminal_pass(branch: Mapping[str, Any], name: str) -> bool:
    terminal = branch.get("terminal_fit_gate", branch)
    if not isinstance(terminal, Mapping) or not isinstance(
        terminal.get("passes"), bool
    ):
        raise ValueError(f"{name} lacks a terminal fit decision")
    return bool(terminal["passes"])


def per_seed_decision(
    faithful: Mapping[str, Any],
    ceiling: Mapping[str, Any] | None,
    holdouts: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Adjudicate one seed while never self-licensing full training."""

    faithful_pass = _terminal_pass(faithful, "production_faithful")
    if faithful_pass and ceiling is not None:
        raise ValueError("ceiling is forbidden after a faithful fit pass")
    if not faithful_pass and ceiling is None:
        raise ValueError("ceiling is mandatory after a faithful fit failure")
    ceiling_pass = (
        None
        if ceiling is None
        else _terminal_pass(ceiling, "ceiling_optimizer")
    )
    qualifying_stage = (
        "production_faithful"
        if faithful_pass
        else "ceiling_optimizer"
        if ceiling_pass
        else None
    )
    if qualifying_stage is None:
        if holdouts not in (None, {}):
            raise ValueError("holdouts are forbidden when both fit branches fail")
        holdout_passes = None
        favorable = False
        classification = "fit_gate_failed"
    else:
        if not isinstance(holdouts, Mapping) or set(holdouts) != set(
            HOLDOUT_PANELS
        ):
            raise ValueError("both holdouts are mandatory after a fit pass")
        holdout_passes = {
            panel: bool(holdouts[panel].get("passes", False))
            for panel in HOLDOUT_PANELS
        }
        favorable = all(holdout_passes.values())
        classification = (
            "favorable" if favorable else "fit_pass_holdout_gate_failed"
        )
    return {
        "schema": PER_SEED_DECISION_SCHEMA,
        "production_faithful_fit_passes": faithful_pass,
        "ceiling_optimizer_invoked": ceiling is not None,
        "ceiling_optimizer_fit_passes": ceiling_pass,
        "qualifying_optimizer_stage": qualifying_stage,
        "holdout_passes": holdout_passes,
        "classification": classification,
        "favorable": favorable,
        "aggregation_eligible": True,
        "categorical_radial_full_train_candidate_licensed": False,
        "promotion_licensed": False,
    }


__all__ = [
    "CLASS_NAMES",
    "CONDITIONS",
    "EXECUTION_BINDING_SHA256",
    "FAMILIES",
    "FIT_GATE_SCHEMA",
    "HOLDOUT_CHECK_SCHEMA",
    "HOLDOUT_PANELS",
    "PATCH7_FINAL_STATE_SHA256",
    "PER_SEED_DECISION_SCHEMA",
    "REFERENCE_MACRO_ASSERTIONS",
    "REFERENCE_SCHEMA",
    "RESULT_SCHEMA",
    "SMOKE_RESULT_SCHEMA",
    "TERMINAL_FIT_SCHEMA",
    "TWO_SEED_RESULT_SCHEMA",
    "all_family_and_aggregate_fit_pass",
    "categorical_holdout_checks",
    "extract_faithful_patch7_family_reference",
    "fit_panel_gate_report",
    "per_seed_decision",
    "terminal_fit_gate_summary",
]
