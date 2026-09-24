from __future__ import annotations

import copy

import pytest

from lewm.benchmarks.go2_categorical_radial_n32 import (
    EXECUTION_BINDING_SHA256,
    FAMILIES,
    HOLDOUT_PANELS,
    PATCH7_FINAL_STATE_SHA256,
    REFERENCE_MACRO_ASSERTIONS,
    all_family_and_aggregate_fit_pass,
    categorical_holdout_checks,
    extract_faithful_patch7_family_reference,
    fit_panel_gate_report,
    per_seed_decision,
    terminal_fit_gate_summary,
)


def _metrics(
    *,
    nll: float = 0.01,
    recall: float = 1.0,
    far: float = 1.0,
    balanced_accuracy: float = 1.0,
) -> dict:
    return {
        "raw_hierarchical_balanced_nll": nll,
        "unknown_known_balanced_accuracy": balanced_accuracy,
        "free_occupied_balanced_accuracy": balanced_accuracy,
        "class_recall": {
            "unknown": recall,
            "free": recall,
            "occupied": recall,
        },
        "distance_free_recall": {
            "1.0_to_2.0": far,
            "2.0_to_3.0": far,
            "3.0_plus": far,
        },
    }


def _conditions(metrics: dict | None = None, *, control_nll: float = 0.5) -> dict:
    return {
        "correct_rgb": _metrics() if metrics is None else metrics,
        "role_global_shuffled_rgb": {
            "raw_hierarchical_balanced_nll": control_nll
        },
        "same_scene_wrong_view_rgb": {
            "raw_hierarchical_balanced_nll": control_nll
        },
    }


def _fit_report() -> dict:
    return {
        "panel": "fit",
        "conditions": _conditions(),
        "families": {
            family: {"conditions": _conditions()} for family in FAMILIES
        },
    }


def _holdout_panel(
    panel: str,
    *,
    nll: float,
    far: float,
    recall: float = 0.8,
) -> dict:
    return {
        "panel": panel,
        "conditions": _conditions(_metrics(nll=nll, far=far, recall=recall)),
        "families": {
            family: {
                "conditions": {
                    "correct_rgb": _metrics(
                        nll=nll,
                        far=far,
                        recall=recall,
                    )
                }
            }
            for family in FAMILIES
        },
    }


def test_binding_and_all_family_fit_gate() -> None:
    assert EXECUTION_BINDING_SHA256 == (
        "42c2ce88ac78f045b92fdd2b33ad5b77a0801de0af2e05c79d3bb518ca188241"
    )
    report = _fit_report()
    gate = fit_panel_gate_report(report)
    assert gate["family_order"] == list(FAMILIES)
    assert gate["passes"] is True
    assert all_family_and_aggregate_fit_pass(report) is True


def test_fit_requires_aggregate_and_every_family() -> None:
    family_failure = _fit_report()
    family_failure["families"][FAMILIES[-1]]["conditions"]["correct_rgb"][
        "class_recall"
    ]["occupied"] = 0.979
    gate = fit_panel_gate_report(family_failure)
    assert gate["aggregate"]["passes"] is True
    assert gate["families"][FAMILIES[-1]]["passes"] is False
    assert gate["passes"] is False

    aggregate_failure = _fit_report()
    aggregate_failure["conditions"]["correct_rgb"][
        "raw_hierarchical_balanced_nll"
    ] = 0.031
    assert all_family_and_aggregate_fit_pass(aggregate_failure) is False


def test_fit_gate_keeps_registered_inclusive_thresholds() -> None:
    report = _fit_report()
    for record in [report, *report["families"].values()]:
        correct = record["conditions"]["correct_rgb"]
        correct["raw_hierarchical_balanced_nll"] = 0.03
        correct["unknown_known_balanced_accuracy"] = 0.99
        correct["free_occupied_balanced_accuracy"] = 0.99
        correct["class_recall"] = dict.fromkeys(
            ("unknown", "free", "occupied"), 0.98
        )
        correct["distance_free_recall"] = dict.fromkeys(
            ("1.0_to_2.0", "2.0_to_3.0", "3.0_plus"), 0.95
        )
        record["conditions"]["role_global_shuffled_rgb"][
            "raw_hierarchical_balanced_nll"
        ] = 0.28
        record["conditions"]["same_scene_wrong_view_rgb"][
            "raw_hierarchical_balanced_nll"
        ] = 0.28
    assert all_family_and_aggregate_fit_pass(report) is True


def test_terminal_summary_uses_exact_final_three_and_complete_cadence() -> None:
    passing = _fit_report()
    failing = copy.deepcopy(passing)
    failing["conditions"]["correct_rgb"]["class_recall"]["free"] = 0.5
    curve = [
        {"step": step, "fit_panel": copy.deepcopy(passing)}
        for step in range(1, 7)
    ]
    curve[-1]["fit_panel"] = failing
    summary = terminal_fit_gate_summary(curve, 6, 1)
    assert summary["first_three_consecutive_fit_gate_step"] == 3
    assert summary["terminal_evaluation_steps"] == [4, 5, 6]
    assert summary["terminal_evaluation_passes"] == [True, True, False]
    assert summary["passes"] is False

    curve[-1]["fit_panel"] = copy.deepcopy(passing)
    assert terminal_fit_gate_summary(curve, 6, 1)["passes"] is True
    with pytest.raises(ValueError, match="cadence"):
        terminal_fit_gate_summary(curve[:-1], 6, 1)


def test_reference_extraction_uses_only_faithful_patch7_family_metrics() -> None:
    panels = {}
    for panel in HOLDOUT_PANELS:
        asserted = REFERENCE_MACRO_ASSERTIONS[panel]
        panels[panel] = _holdout_panel(
            panel,
            nll=asserted["hierarchical_nll"],
            far=asserted["far_free_recall"],
        )
    payload = {
        "stages": {
            "production_faithful": {
                "patch7_16x16": {
                    "final_state_sha256": PATCH7_FINAL_STATE_SHA256,
                    "final_panels": panels,
                }
            },
            "ceiling_optimizer": {
                "patch7_16x16": {"final_panels": {"ignored": True}}
            },
        },
        "post_selection_support_audit": {
            "fit": {},
            "same_scene_holdout": {},
            "cross_scene_holdout": {},
        },
    }
    reference = extract_faithful_patch7_family_reference(payload)
    assert reference["source_stage"] == "production_faithful"
    assert reference["source_arm"] == "patch7_16x16"
    assert tuple(reference["panels"]) == HOLDOUT_PANELS


def test_holdout_macro_is_ratio_of_means_and_uses_canonical_order() -> None:
    reference = _holdout_panel("cross_scene_holdout", nll=0.5, far=0.5)
    candidate = _holdout_panel("cross_scene_holdout", nll=0.35, far=0.65)
    candidate_nlls = (0.1, 0.2, 0.3, 0.4, 0.5)
    reference_nlls = (0.2, 0.3, 0.5, 0.7, 0.8)
    for family, candidate_nll, reference_nll in zip(
        FAMILIES,
        candidate_nlls,
        reference_nlls,
    ):
        candidate["families"][family]["conditions"]["correct_rgb"][
            "raw_hierarchical_balanced_nll"
        ] = candidate_nll
        reference["families"][family]["conditions"]["correct_rgb"][
            "raw_hierarchical_balanced_nll"
        ] = reference_nll
    checks = categorical_holdout_checks(candidate, reference)
    expected_ratio = (sum(candidate_nlls) / 5) / (sum(reference_nlls) / 5)
    mean_of_ratios = sum(
        candidate_nll / reference_nll
        for candidate_nll, reference_nll in zip(
            candidate_nlls,
            reference_nlls,
        )
    ) / 5
    assert checks["candidate_to_reference_macro_hierarchical_nll_ratio"] == (
        pytest.approx(expected_ratio)
    )
    assert expected_ratio != pytest.approx(mean_of_ratios)
    assert checks["family_order"] == list(FAMILIES)
    assert checks["passes"] is True


def test_strict_family_ties_fail_cross_but_four_of_five_pass_same_scene() -> None:
    for panel, expected_pass in (
        ("cross_scene_holdout", False),
        ("same_scene_holdout", True),
    ):
        reference = _holdout_panel(panel, nll=0.5, far=0.5)
        candidate = _holdout_panel(panel, nll=0.35, far=0.65)
        tied = candidate["families"][FAMILIES[0]]["conditions"]["correct_rgb"]
        tied["raw_hierarchical_balanced_nll"] = 0.5
        checks = categorical_holdout_checks(candidate, reference)
        assert checks["strictly_favorable_family_count"] == 4
        assert checks["passes"] is expected_pass
        assert checks["ties_count_as_failure"] is True


def test_any_individual_family_class_regression_below_limit_fails() -> None:
    reference = _holdout_panel("cross_scene_holdout", nll=0.5, far=0.5)
    candidate = _holdout_panel("cross_scene_holdout", nll=0.35, far=0.65)
    candidate["families"][FAMILIES[0]]["conditions"]["correct_rgb"][
        "class_recall"
    ]["occupied"] = 0.789
    checks = categorical_holdout_checks(candidate, reference)
    assert checks["checks"][
        "no_family_class_recall_delta_lt_neg_0_01"
    ] is False
    assert checks["passes"] is False


def _terminal(passes: bool) -> dict:
    return {"terminal_fit_gate": {"passes": passes}}


def _holdout_decisions(passes: bool) -> dict:
    return {panel: {"passes": passes} for panel in HOLDOUT_PANELS}


def test_per_seed_faithful_and_ceiling_branches() -> None:
    faithful = per_seed_decision(_terminal(True), None, _holdout_decisions(True))
    assert faithful["favorable"] is True
    assert faithful["qualifying_optimizer_stage"] == "production_faithful"
    assert faithful["categorical_radial_full_train_candidate_licensed"] is False

    ceiling = per_seed_decision(
        _terminal(False),
        _terminal(True),
        _holdout_decisions(True),
    )
    assert ceiling["favorable"] is True
    assert ceiling["qualifying_optimizer_stage"] == "ceiling_optimizer"

    failed = per_seed_decision(_terminal(False), _terminal(False), None)
    assert failed["favorable"] is False
    assert failed["classification"] == "fit_gate_failed"


def test_per_seed_enforces_conditional_branch_and_holdout_structure() -> None:
    with pytest.raises(ValueError, match="ceiling is forbidden"):
        per_seed_decision(_terminal(True), _terminal(False), _holdout_decisions(True))
    with pytest.raises(ValueError, match="ceiling is mandatory"):
        per_seed_decision(_terminal(False), None, None)
    with pytest.raises(ValueError, match="holdouts are forbidden"):
        per_seed_decision(
            _terminal(False),
            _terminal(False),
            _holdout_decisions(False),
        )
    with pytest.raises(ValueError, match="both holdouts"):
        per_seed_decision(
            _terminal(True),
            None,
            {"same_scene_holdout": {"passes": True}},
        )
