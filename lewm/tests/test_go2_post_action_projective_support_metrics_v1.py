from __future__ import annotations

import math
import random
from dataclasses import replace

import numpy as np
import pytest

from lewm.benchmarks.go2_post_action_projective_support_metrics_v1 import (
    ACTION_VOCABULARY,
    ARM_NAMES,
    BOOTSTRAP_LOWER_INDEX,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    IntegrityMetrics,
    REGISTERED_FAMILIES,
    SemanticRetentionMetrics,
    action_prior_probabilities,
    calibrate_arms,
    evaluate_arm,
    evaluate_conjunctive_gate,
    expected_calibration_error,
    oracle_metric_pipeline_preflight,
    paired_scene_bootstrap,
    primary_subset_mask,
    safe_prefix_lengths,
    select_calibration_threshold,
    select_non_hold_actions,
    semantic_retention_from_confusions,
    station_classification_metrics,
    threshold_candidates,
    wrong_rgb_endpoint_mapping,
)


def test_threshold_candidates_and_frozen_lexicographic_selection() -> None:
    scores = np.array([0.8, 0.8, 0.9, 0.7, 0.1])
    labels = np.array([1, 1, 1, 0, 0])
    candidates = threshold_candidates(scores)
    assert np.array_equal(
        candidates,
        np.array([0.0, 0.1, 0.7, 0.8, 0.9, np.nextafter(0.9, math.inf)]),
    )

    selected = select_calibration_threshold(scores, labels)
    assert selected.eligible
    assert selected.threshold == 0.8
    assert selected.metrics is not None
    assert selected.metrics.safe_recall == 1.0
    assert selected.metrics.unsafe_recall == 1.0

    no_threshold = select_calibration_threshold(
        np.array([0.9, 0.8]), np.array([0, 1])
    )
    assert not no_threshold.eligible
    assert no_threshold.threshold is None


def test_ece_has_exact_fifteen_bin_edges_and_row_weighting() -> None:
    scores = np.array([0.0, 1 / 15, np.nextafter(1 / 15, 0.0), 1.0])
    labels = np.array([0, 1, 1, 1])
    expected = (
        2 / 4 * abs(np.mean([0.0, np.nextafter(1 / 15, 0.0)]) - 0.5)
        + 1 / 4 * abs(1 / 15 - 1.0)
        + 1 / 4 * abs(1.0 - 1.0)
    )
    assert expected_calibration_error(scores, labels) == pytest.approx(expected)
    metrics = station_classification_metrics(scores, labels, 1 / 15)
    assert metrics.brier_score == pytest.approx(
        np.mean(np.square(scores - labels))
    )


def test_prefix_action_ties_and_primary_subset_are_target_defined() -> None:
    labels = np.zeros((3, 9, 11), dtype=bool)
    labels[0, 0, :4] = True
    labels[0, 1, :2] = True
    labels[1, 0, :3] = True
    labels[1, 1, :3] = True
    labels[2, 6, :11] = True  # HOLD never makes a state informative.
    feasible = np.ones((3, 9), dtype=bool)
    feasible[1, 7] = False

    prefixes = safe_prefix_lengths(labels)
    assert prefixes[0, :2].tolist() == [4, 2]
    assert select_non_hold_actions(prefixes).tolist() == [0, 0, 0]
    assert primary_subset_mask(labels, feasible, np.ones_like(feasible)).tolist() == [True, False, False]


def _evaluation_fixture() -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    labels = np.zeros((16, 9, 11), dtype=bool)
    for index in range(16):
        labels[index, 0, :11] = True
        labels[index, 1, :6] = True
    probabilities = np.where(labels, 0.9, 0.1)
    scenes = [f"scene-{index // 2}" for index in range(16)]
    families = [REGISTERED_FAMILIES[index // 2] for index in range(16)]
    feasible = np.ones((16, 9), dtype=bool)
    return probabilities, labels, scenes, families, feasible


def test_evaluation_uses_equal_scene_means_and_fixed_action_order() -> None:
    probabilities, labels, scenes, families, feasible = _evaluation_fixture()
    # Make scene 0 choose action 1, all other scenes choose the oracle action 0.
    probabilities[:2, 0] = 0.1
    evaluation = evaluate_arm(
        probabilities,
        labels,
        0.5,
        scenes,
        families,
        feasible,
        feasible,
    )
    assert evaluation.primary_mask.all()
    assert evaluation.selected_action_indices[:2].tolist() == [1, 1]
    assert evaluation.selected_action_indices[2:].tolist() == [0] * 14
    assert evaluation.scenes["scene-0"].mean_primary_utility == pytest.approx(6 / 11)
    assert evaluation.overall.mean_primary_utility == pytest.approx((7 + 6 / 11) / 8)
    assert evaluation.overall.selected_action_shares[0] == pytest.approx(7 / 8)
    assert evaluation.overall.oracle_action_shares[0] == 1.0


def test_outside_subset_reports_exact_feasible_and_zero_oracle_utilities() -> None:
    labels = np.zeros((2, 9, 11), dtype=bool)
    labels[0, 0, :4] = True
    labels[0, 1, :2] = True
    probabilities = np.full((2, 9, 11), 0.1, dtype=np.float64)
    probabilities[0, 1, :] = 0.9
    feasible = np.ones((2, 9), dtype=bool)
    feasible[0, 7] = False  # Excludes state 0 without invalidating selected action 1.

    evaluation = evaluate_arm(
        probabilities,
        labels,
        0.5,
        ["scene-0", "scene-0"],
        [REGISTERED_FAMILIES[0], REGISTERED_FAMILIES[0]],
        feasible,
        np.ones_like(feasible),
    )

    assert evaluation.primary_mask.tolist() == [False, False]
    assert evaluation.outside_subset_state_count == 2
    assert evaluation.outside_subset_selected_infeasible_count == 0
    assert evaluation.outside_subset_state_indices == (0, 1)
    assert evaluation.outside_subset_utility_values == (0.5, None)
    assert evaluation.outside_subset_mean_utility == pytest.approx(0.5)


@pytest.mark.parametrize("infeasible_scope", ["immediate", "blind_bridge"])
def test_outside_subset_selected_infeasible_utility_is_exactly_minus_one(
    infeasible_scope: str,
) -> None:
    labels = np.zeros((1, 9, 11), dtype=bool)
    labels[0, 0, :4] = True
    labels[0, 1, :2] = True
    probabilities = np.full((1, 9, 11), 0.1, dtype=np.float64)
    probabilities[0, 1, :] = 0.9
    immediate = np.ones((1, 9), dtype=bool)
    blind_bridge = np.ones((1, 9), dtype=bool)
    (immediate if infeasible_scope == "immediate" else blind_bridge)[0, 1] = False

    evaluation = evaluate_arm(
        probabilities,
        labels,
        0.5,
        ["scene-0"],
        [REGISTERED_FAMILIES[0]],
        immediate,
        blind_bridge,
    )

    assert evaluation.primary_mask.tolist() == [False]
    assert evaluation.outside_subset_state_count == 1
    assert evaluation.outside_subset_selected_infeasible_count == 1
    assert evaluation.outside_subset_state_indices == (0,)
    assert evaluation.outside_subset_utility_values == (-1.0,)
    assert evaluation.outside_subset_mean_utility == -1.0


def test_paired_bootstrap_is_exact_seeded_index_249() -> None:
    full = {f"scene-{index}": 0.9 for index in range(8)}
    control = {f"scene-{index}": index / 10 for index in range(8)}
    result = paired_scene_bootstrap(full, control)
    deltas = [full[key] - control[key] for key in sorted(full)]
    rng = random.Random(BOOTSTRAP_SEED)
    reference = sorted(
        math.fsum(deltas[rng.randrange(8)] for _ in range(8)) / 8
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    assert result.lower_95 == reference[BOOTSTRAP_LOWER_INDEX]
    assert result.point_delta == pytest.approx(np.mean(deltas))
    with pytest.raises(ValueError, match="exactly eight"):
        paired_scene_bootstrap({"only": 1.0}, {"only": 0.0})


def _passing_integrity() -> IntegrityMetrics:
    return IntegrityMetrics(
        exact_accounting=True,
        outputs_and_gradients_finite=True,
        target_gradients_zero=True,
        target_optimizer_membership_zero=True,
        online_gradients_nonzero_every_update=True,
        predictor_forward_count=4_000,
        predictor_objective_count=4_000,
        backward_count=4_000,
        predictor_optimizer_update_count=1_000,
        forbidden_input_count=0,
        bypass_count=0,
        forbidden_open_count=0,
        current_latents_nonconstant=True,
        paired_latents_nonconstant=True,
        current_and_paired_latents_nonidentical=True,
        one_step_zero_support_witnessed=True,
        all_corridor_masks_nonempty=True,
        corridor_masks_inside_support=True,
    )


def _passing_semantic() -> SemanticRetentionMetrics:
    return SemanticRetentionMetrics(0.8, 0.85, 0.7, 0.9, 0.65)


def test_conjunctive_gate_passes_only_full_over_all_controls() -> None:
    full_scores, labels, scenes, families, feasible = _evaluation_fixture()
    # Controls confidently select the shorter action 1; full selects oracle action 0.
    control_scores = np.full_like(full_scores, 0.1)
    control_scores[:, 1, :6] = 0.9
    evaluation_scores = {"full": full_scores}
    evaluation_scores.update({name: control_scores for name in ARM_NAMES[1:]})

    calibration_labels = np.tile(np.array([1, 0], dtype=bool), (16, 9, 6))[:, :, :11]
    calibration_scores = np.where(calibration_labels, 0.9, 0.1)
    suite = calibrate_arms(
        {name: calibration_scores for name in ARM_NAMES}, calibration_labels
    )
    assert suite.comparable
    evaluations = {
        name: evaluate_arm(
            evaluation_scores[name],
            labels,
            suite.arms[name].threshold,
            scenes,
            families,
            feasible,
            feasible,
            arm_name=name,
        )
        for name in ARM_NAMES
    }
    decision = evaluate_conjunctive_gate(
        suite, evaluations, _passing_integrity(), _passing_semantic()
    )
    assert decision.status == "PASS"
    assert decision.passed
    assert all(comparison.positive_family_count == 8 for comparison in decision.comparisons.values())
    assert all(comparison.bootstrap.lower_95 > 0 for comparison in decision.comparisons.values())
    failed = evaluate_conjunctive_gate(
        suite,
        evaluations,
        replace(_passing_integrity(), backward_count=3_999),
        _passing_semantic(),
    )
    assert failed.status == "FAIL"
    assert failed.failed_checks == ("integrity_backward_count",)

    wrong_family_ids = tuple(f"wrong-{index // 2}" for index in range(16))
    wrong_evaluations = {
        name: replace(
            evaluation,
            family_ids=wrong_family_ids,
            families={
                f"wrong-{index}": evaluation.families[family]
                for index, family in enumerate(REGISTERED_FAMILIES)
            },
        )
        for name, evaluation in evaluations.items()
    }
    wrong_families = evaluate_conjunctive_gate(
        suite, wrong_evaluations, _passing_integrity(), _passing_semantic()
    )
    assert not wrong_families.passed
    assert "selection_has_exact_registered_families" in wrong_families.failed_checks


def test_failed_control_calibration_is_explicitly_terminal_noncomparable() -> None:
    labels = np.array([1, 0], dtype=bool)
    good = np.array([0.9, 0.1])
    bad = np.array([0.8, 0.9])
    scores = {name: good for name in ARM_NAMES}
    scores["wrong_rgb"] = bad
    suite = calibrate_arms(scores, labels)
    assert suite.failed_arms == ("wrong_rgb",)
    decision = evaluate_conjunctive_gate(
        suite, {}, _passing_integrity(), _passing_semantic()
    )
    assert decision.status == "TERMINAL_NON_COMPARABLE_CONTROL_CALIBRATION"
    assert not decision.passed
    assert decision.failed_calibration_arms == ("wrong_rgb",)


def test_semantic_retention_uses_unknown_free_occupied_true_rows() -> None:
    overall = np.array(((90, 5, 5), (3, 85, 12), (10, 20, 70)))
    rough = np.array(((0, 0, 0), (0, 0, 0), (15, 20, 65)))
    metrics = semantic_retention_from_confusions(overall, rough)
    assert metrics.unknown_recall == 0.90
    assert metrics.free_recall == 0.85
    assert metrics.occupied_recall == 0.70
    assert metrics.balanced_accuracy == pytest.approx((0.90 + 0.85 + 0.70) / 3)
    assert metrics.rough_family_occupied_recall == 0.65
    with pytest.raises(ValueError, match="UNKNOWN, FREE, and OCCUPIED"):
        semantic_retention_from_confusions(np.zeros((3, 3), dtype=int), rough)


def test_train_action_prior_requires_non_hold_safe_and_unsafe_support() -> None:
    labels = np.zeros((2, 9, 11), dtype=bool)
    labels[0] = True
    labels[:, 6] = True  # HOLD is not part of the preregistered support check.
    prior = action_prior_probabilities(labels)
    assert prior.shape == (9, 11)
    assert prior.dtype == np.float64
    assert np.all(prior[np.array((0, 1, 2, 3, 4, 5, 7, 8))] == 0.5)
    assert np.all(prior[6] == 1.0)
    labels[:, 0, 3] = True
    with pytest.raises(ValueError, match="safe and unsafe support"):
        action_prior_probabilities(labels)


def test_wrong_rgb_mapping_is_role_scene_local_cyclic_and_hash_bound() -> None:
    mapping = wrong_rgb_endpoint_mapping(
        (
            ("checkpoint_selection", "scene-z", "b"),
            ("probability_calibration", "scene-a", "2"),
            ("checkpoint_selection", "scene-z", "a"),
            ("probability_calibration", "scene-a", "1"),
            ("checkpoint_selection", "scene-z", "c"),
            ("checkpoint_selection", "scene-z", "a"),
        )
    )
    assert mapping.rows == (
        ("checkpoint_selection", "scene-z", "a", "b"),
        ("checkpoint_selection", "scene-z", "b", "c"),
        ("checkpoint_selection", "scene-z", "c", "a"),
        ("probability_calibration", "scene-a", "1", "2"),
        ("probability_calibration", "scene-a", "2", "1"),
    )
    assert mapping.by_endpoint[("checkpoint_selection", "scene-z", "c")] == "a"
    assert mapping.mapping_sha256 == (
        "df64054e9ada2db10b762135c69ef31668f906a13150a3c2a56696f80699ddbf"
    )
    with pytest.raises(ValueError, match="fewer than two"):
        wrong_rgb_endpoint_mapping((("checkpoint_selection", "only", "a"),))


def test_oracle_metric_pipeline_preflight_proves_exact_metrics() -> None:
    _, selection_labels, scenes, families, feasible = _evaluation_fixture()
    calibration_labels = np.zeros((16, 9, 11), dtype=bool)
    calibration_labels[::2] = True
    result = oracle_metric_pipeline_preflight(
        calibration_labels,
        selection_labels,
        scenes,
        families,
        feasible,
        feasible,
        calibration_family_ids=families,
    )
    assert result.passed
    assert result.failed_checks == ()
    assert result.calibration.threshold == 1.0
    assert result.selection is not None
    assert result.selection.overall.mean_primary_utility == 1.0
    assert result.bootstrap is not None
    assert result.bootstrap.point_delta == result.bootstrap.lower_95 == 1.0
    assert all(result.checks.values())

    infeasible = feasible.copy()
    infeasible[:2, 0] = False
    failed = oracle_metric_pipeline_preflight(
        calibration_labels,
        selection_labels,
        scenes,
        families,
        infeasible,
        feasible,
    )
    assert not failed.passed
    assert "selection_utility_exact_one" in failed.failed_checks


def test_action_vocabulary_is_frozen() -> None:
    assert ACTION_VOCABULARY == (
        "arc_left",
        "arc_right",
        "backward",
        "forward_fast",
        "forward_medium",
        "forward_slow",
        "hold",
        "yaw_left",
        "yaw_right",
    )
