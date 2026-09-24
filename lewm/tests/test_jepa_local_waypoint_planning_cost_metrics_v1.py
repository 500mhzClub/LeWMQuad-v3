"""Synthetic-only tests for the JEPA local-waypoint planning metrics V1."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from lewm.safety import jepa_local_waypoint_planning_cost_metrics_v1 as metrics


def _row(
    candidate_index: int,
    p_d: float,
    p_theta: float = 0.0,
    *,
    completed: bool = False,
    contact: bool = False,
    viable: bool = True,
    successor_viable: bool | None = None,
    stuck: bool = False,
    family: str = "large_enclosed_maze",
    role: str = "heldout",
) -> dict:
    return {
        "candidate_index": candidate_index,
        "p_d": p_d,
        "p_theta": p_theta,
        "completed": completed,
        "p_d_h1": p_d,
        "p_theta_h1": p_theta,
        "completed_h1": completed,
        "p_d_h2": p_d,
        "p_theta_h2": p_theta,
        "completed_h2": completed,
        "p_d_h3": p_d,
        "p_theta_h3": p_theta,
        "completed_h3": completed,
        "oracle_contact": contact,
        "oracle_viability_admissible": viable,
        "successor_viable": viable if successor_viable is None else successor_viable,
        "descriptive_contact_h2": False,
        "descriptive_contact_h3": False,
        "stuck": stuck,
        "family": family,
        "role": role,
    }


def _gate_source(
    source_id: str,
    *,
    pairwise: float | None = 0.70,
    spearman: float | None = 0.60,
    regret: float | None = 0.25,
    top3: float | None = 0.75,
    selected_progress: float = 8.0,
    best_progress: float = 10.0,
    no_collapse: bool = True,
    contacts: int = 0,
    nonviable: int = 0,
) -> dict:
    oracle_aggregate = {
        "pairwise_accuracy": pairwise,
        "spearman_rho": spearman,
        "normalized_regret": regret,
        "best_route_top3_rate": top3,
        "selected_route_progress_m_sum": selected_progress,
        "oracle_best_route_progress_m_sum": best_progress,
        "selected_progress_ratio": selected_progress / max(abs(best_progress), 1e-9),
    }
    family = {
        family_id: {"complete_family_collapse": not no_collapse}
        for family_id in metrics.FAMILY_IDS
    }
    return {
        "source_id": source_id,
        "populations": {
            metrics.ORACLE_VIABILITY_ADMISSIBLE: {
                "aggregate": oracle_aggregate,
                "no_family_complete_collapse": no_collapse,
                "per_family": family,
                "per_state": [],
            },
            metrics.ALL_CANDIDATES: {
                "aggregate": {
                    "selected_oracle_contacts": contacts,
                    "selected_oracle_nonviable": nonviable,
                }
            },
        },
    }


def _comparison_source(
    source_id: str,
    *,
    selected_progress: float,
    regret: float | None,
    contacts: int = 0,
    nonviable: int = 0,
    collapsed: bool = False,
) -> dict:
    per_state = []
    for number, family in enumerate(metrics.FAMILY_IDS):
        per_state.append(
            {
                "state_id": f"state-{number}",
                "family": family,
                "selected_candidate_index": (
                    2
                    if source_id == metrics.TWO_STEP_PREDICTED_LATENT_COST
                    else 1
                ),
                "selected_route_progress_m": selected_progress,
                "oracle_best_route_progress_m": 1.0,
                "normalized_regret": regret,
                "pairwise_accuracy": 0.8,
                "best_route_top3": True,
                "best_route_rank": 1,
            }
        )
    return {
        "source_id": source_id,
        "populations": {
            metrics.ORACLE_VIABILITY_ADMISSIBLE: {
                "per_state": per_state,
                "per_family": {
                    family: {"complete_family_collapse": collapsed}
                    for family in metrics.FAMILY_IDS
                },
            },
            metrics.ALL_CANDIDATES: {
                "aggregate": {
                    "selected_oracle_contacts": contacts,
                    "selected_oracle_nonviable": nonviable,
                }
            },
        },
    }


def test_frozen_ids_thresholds_and_seed_are_exact() -> None:
    assert metrics.SOURCE_IDS == (
        "TRUE_FUTURE",
        "ONE_STEP_PREDICTED",
        "TWO_STEP_PREDICTED",
    )
    assert metrics.POPULATION_IDS == (
        "ALL_CANDIDATES",
        "ORACLE_CONTACT_FREE",
        "ORACLE_VIABILITY_ADMISSIBLE",
    )
    assert metrics.COMPARATOR_IDS == (
        "KINEMATIC_ROUTE_BASELINE",
        "RANDOM",
        "TRUE_FUTURE_LATENT_COST",
        "ONE_STEP_PREDICTED_LATENT_COST",
        "TWO_STEP_PREDICTED_LATENT_COST",
    )
    assert metrics.BOOTSTRAP_DRAWS == 10_000
    assert metrics.BOOTSTRAP_SEED == 2026080901
    assert metrics.DISTANCE_PREFERENCE_MARGIN_M == 0.03
    assert metrics.HEADING_PREFERENCE_MARGIN_RAD == math.radians(5.0)


def test_raw_token_cost_layer_normalizes_then_l2_normalizes() -> None:
    candidate = np.asarray(
        [[[1, -1, 0], [1, 1, -2]], [[-1, 1, 0], [2, -1, -1]]],
        dtype=np.float16,
    )
    goal = np.asarray([[1, -1, 0], [1, 1, -2]], dtype=np.float16)
    value = metrics.tokenwise_normalized_cosine_mean_cost(candidate, goal)
    np.testing.assert_allclose(value, [0.0, 1.25], atol=2e-6, rtol=0.0)
    assert metrics.tokenwise_normalized_cosine_mean_cost(goal, goal) == pytest.approx(
        0.0, abs=2e-7
    )
    orthogonal = np.asarray([[1, 1, -2], [1, 1, -2]], np.float16)
    axis = np.asarray([[1, -1, 0], [1, -1, 0]], np.float16)
    assert metrics.tokenwise_normalized_cosine_mean_cost(axis, orthogonal) == pytest.approx(
        1.0, abs=2e-6
    )
    assert metrics.tokenwise_normalized_cosine_mean_cost(axis, -axis) == pytest.approx(
        2.0, abs=2e-6
    )


def test_raw_token_cost_matches_torch_synthetic_reference() -> None:
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torch.nn.functional")
    candidate = np.asarray(
        [[[3, -2, 4, 1], [2, 8, -1, 3]], [[1, 4, 7, -5], [9, 2, -3, 6]]],
        dtype=np.float16,
    )
    goal = np.asarray([[2, -3, 5, 7], [-2, 6, 4, 1]], dtype=np.float16)
    actual = metrics.tokenwise_normalized_cosine_mean_cost(candidate, goal)
    left = functional.layer_norm(torch.as_tensor(candidate).float(), (4,), eps=1e-5)
    right = functional.layer_norm(torch.as_tensor(goal).float(), (4,), eps=1e-5)
    left = functional.normalize(left, dim=-1, eps=1e-12)
    right = functional.normalize(right, dim=-1, eps=1e-12)
    expected = (1.0 - (left * right).sum(dim=-1)).double().mean(dim=-1).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=0.0)


@pytest.mark.parametrize(
    "candidate,goal",
    [
        (np.ones((2, 3), np.float32), np.ones((2, 3), np.float16)),
        (np.ones((2, 3), np.float16), np.ones((3, 3), np.float16)),
        (
            np.asarray([[1.0, math.nan, 2.0]], np.float16),
            np.asarray([[1.0, 2.0, 3.0]], np.float16),
        ),
    ],
)
def test_raw_token_cost_fails_closed(candidate, goal) -> None:
    with pytest.raises(metrics.PlanningCostMetricsError):
        metrics.tokenwise_normalized_cosine_mean_cost(candidate, goal)


def test_zero_variance_tokens_follow_f_normalize_epsilon_clamp() -> None:
    zero_after_layernorm = np.ones((2, 3), dtype=np.float16)
    nonzero = np.asarray([[1, -1, 0], [1, 1, -2]], dtype=np.float16)
    assert metrics.tokenwise_normalized_cosine_mean_cost(
        zero_after_layernorm, nonzero
    ) == pytest.approx(1.0)
    assert metrics.tokenwise_normalized_cosine_mean_cost(
        zero_after_layernorm, zero_after_layernorm
    ) == pytest.approx(1.0)


def test_monotonic_diagnostic_reports_registered_deltas() -> None:
    result = metrics.trajectory_cost_monotonicity([1.0, 0.8, 0.8, 0.9])
    np.testing.assert_allclose(result["adjacent_progress_deltas"], [0.2, 0.0, -0.1])
    assert result["violation_count"] == 1
    assert result["strict_improvement_count"] == 1
    assert result["end_to_end_progress_delta"] == pytest.approx(0.1)
    assert result["pass_nonincreasing"] is False


def test_realised_route_preference_uses_completion_distance_heading_only() -> None:
    base = _row(0, 0.0, 0.0, contact=True, viable=False, stuck=True)
    complete = _row(1, -1.0, -1.0, completed=True)
    assert metrics.route_preference(complete, base) == 1
    distance_tie = _row(2, metrics.DISTANCE_PREFERENCE_MARGIN_M, 0.0)
    assert metrics.route_preference(distance_tie, _row(3, 0.0, 0.0)) == 0
    heading_tie = _row(4, 0.0, metrics.HEADING_PREFERENCE_MARGIN_RAD)
    assert metrics.route_preference(heading_tie, _row(5, 0.0, 0.0)) == 0
    assert metrics.route_preference(_row(6, 0.031, -10), base) == 1
    assert metrics.route_preference(_row(7, 0.0, math.radians(5.1)), base) == 1
    # Safety/stuck differences do not alter an otherwise tied route outcome.
    assert metrics.route_preference(base, _row(8, 0.0, 0.0)) == 0


def test_kinematic_integrator_uses_all_15_commands_and_zero_vy() -> None:
    actions = np.zeros((3, 5, 3), dtype=np.float64)
    actions[..., 0] = 1.0
    outcome = metrics.kinematic_nominal_outcome(actions, [2.0, 0.0])
    assert outcome["integrated_commands"] == 15
    assert outcome["elapsed_s"] == pytest.approx(1.5)
    assert outcome["x_m"] == pytest.approx(1.5)
    assert outcome["nominal_p_d"] == pytest.approx(1.5)
    assert "completed" not in outcome
    actions[0, 0, 1] = 1e-9
    with pytest.raises(metrics.PlanningCostMetricsError, match="vy"):
        metrics.kinematic_nominal_outcome(actions, [2.0, 0.0])


def test_kinematic_order_has_no_heading_deadband() -> None:
    rows = [
        {"candidate_index": 4, "nominal_p_d": 1.0, "nominal_p_theta": 0.0},
        {"candidate_index": 2, "nominal_p_d": 0.98, "nominal_p_theta": 1e-6},
        {"candidate_index": 1, "nominal_p_d": 0.5, "nominal_p_theta": 10.0},
    ]
    assert metrics.kinematic_route_order(rows) == [1, 0, 2]
    np.testing.assert_array_equal(metrics.kinematic_rank_costs(rows), [1.0, 0.0, 2.0])


def test_population_filters_and_nesting_fail_closed() -> None:
    rows = [
        _row(0, 0.0, contact=False, viable=True),
        _row(1, 0.0, contact=False, viable=False),
        _row(2, 0.0, contact=True, viable=False),
    ]
    assert [row["candidate_index"] for row in metrics.filter_population(rows, metrics.ALL_CANDIDATES)] == [0, 1, 2]
    assert [row["candidate_index"] for row in metrics.filter_population(rows, metrics.ORACLE_CONTACT_FREE)] == [0, 1]
    assert [row["candidate_index"] for row in metrics.filter_population(rows, metrics.ORACLE_VIABILITY_ADMISSIBLE)] == [0]
    with pytest.raises(metrics.PlanningCostMetricsError, match="must equal"):
        metrics.filter_population([_row(0, 0.0, contact=True, viable=True)], metrics.ALL_CANDIDATES)


def test_successor_fanout_and_admissibility_identity_fail_closed() -> None:
    zero = _row(1, 0.0, viable=False, successor_viable=False)
    zero.pop("successor_viable")
    zero["successor_safe_action_count"] = 0
    assert metrics.filter_population(
        [zero], metrics.ORACLE_VIABILITY_ADMISSIBLE
    ) == []
    valid = _row(0, 0.0)
    valid.pop("successor_viable")
    valid["successor_safe_action_count"] = 9
    assert metrics.filter_population([valid], metrics.ORACLE_VIABILITY_ADMISSIBLE)
    invalid_count = {**valid, "successor_safe_action_count": 10}
    with pytest.raises(metrics.PlanningCostMetricsError, match=r"\[0,9\]"):
        metrics.filter_population([invalid_count], metrics.ALL_CANDIDATES)
    with pytest.raises(metrics.PlanningCostMetricsError, match=r"\[0,9\]"):
        metrics.filter_population(
            [{**zero, "successor_safe_action_count": -1}], metrics.ALL_CANDIDATES
        )
    inconsistent = _row(0, 0.0, contact=False, viable=False, successor_viable=True)
    with pytest.raises(metrics.PlanningCostMetricsError, match="must equal"):
        metrics.filter_population([inconsistent], metrics.ALL_CANDIDATES)


def test_route_ordering_metrics_are_complete_and_tie_aware() -> None:
    rows = [_row(3, 0.3), _row(1, 0.2), _row(2, 0.1)]
    result = metrics.route_ordering_metrics(rows, [0.1, 0.2, 0.3])
    assert result["pairwise_accuracy"] == 1.0
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["kendall_tau_b"] == pytest.approx(1.0)
    assert result["best_route_top1"] is True
    assert result["best_route_top3"] is True
    assert result["mean_reciprocal_rank"] == 1.0
    assert result["mean_best_route_rank"] == 1.0
    assert result["cost_spread"] == pytest.approx(0.2)
    assert result["cost_tie_count"] == 0
    tied = metrics.route_ordering_metrics(rows, [0.1, 0.1, 0.1])
    assert tied["pairwise_accuracy"] == 0.5
    assert tied["cost_tie_count"] == 3
    assert tied["cost_tie_rate"] == 1.0
    assert tied["spearman_rho"] is None
    assert tied["selected_candidate_index"] == 1

    near_tied = metrics.route_ordering_metrics(
        [_row(5, 0.0), _row(1, 0.2)],
        [0.1, 0.1 + 5.0e-13],
    )
    assert near_tied["cost_tie_count"] == 1
    assert near_tied["selected_candidate_index"] == 1
    assert near_tied["cost_order"] == [1, 5]


def test_margin_borda_and_oracle_ties_are_exact() -> None:
    rows = [_row(2, 0.0), _row(1, 0.0), _row(3, 0.2)]
    np.testing.assert_allclose(metrics.margin_borda_utility(rows), [0.25, 0.25, 1.0])
    assert metrics.realised_route_order(rows) == [2, 1, 0]
    tied = metrics.route_ordering_metrics(rows[:2], [0.2, 0.1])
    assert tied["ordered_pairs"] == 0
    assert tied["pairwise_accuracy"] is None


def test_state_selection_outcomes_regret_and_empty_abstention() -> None:
    rows = [
        _row(0, 0.3, contact=False, viable=True),
        _row(1, 0.2, contact=True, viable=False, stuck=True),
    ]
    selected = metrics.evaluate_state_population(
        rows,
        [1.0, 0.0],
        state_id="s",
        family="large_enclosed_maze",
        role="heldout",
        source_id=metrics.TRUE_FUTURE,
        population_id=metrics.ALL_CANDIDATES,
    )
    assert selected["selected_candidate_index"] == 1
    assert selected["selected_immediate_contact"] is True
    assert selected["selected_successor_nonviable"] is True
    assert selected["selected_stuck"] is True
    assert selected["selected_p_d"] == 0.2
    assert selected["selected_p_theta"] == 0.0
    assert selected["normalized_regret"] == pytest.approx(1.0)
    empty = metrics.evaluate_state_population(
        [_row(0, 0.3, viable=False)],
        [0.0],
        state_id="empty",
        family="large_enclosed_maze",
        role="heldout",
        source_id=metrics.TRUE_FUTURE,
        population_id=metrics.ORACLE_VIABILITY_ADMISSIBLE,
    )
    assert empty["abstention"] is True
    assert empty["correct_abstention"] is True
    assert empty["false_abstention"] is False
    assert empty["no_oracle_viability_admissible_action"] is True


def test_immediate_contact_and_successor_nonviability_are_independent() -> None:
    contact_but_successor_viable = _row(
        0,
        0.1,
        contact=True,
        viable=False,
        successor_viable=True,
    )
    result = metrics.evaluate_state_population(
        [contact_but_successor_viable],
        [0.0],
        state_id="separate",
        family="large_enclosed_maze",
        role="heldout",
        source_id=metrics.TRUE_FUTURE,
        population_id=metrics.ALL_CANDIDATES,
    )
    assert result["selected_immediate_contact_h1"] is True
    assert result["selected_nonviable"] is False


def test_equal_progress_range_has_zero_regret_but_tiny_nonzero_fails() -> None:
    equal = [_row(0, 0.2), _row(1, 0.2)]
    result = metrics.evaluate_state_population(
        equal,
        [1.0, 0.0],
        state_id="equal",
        family="large_enclosed_maze",
        role="heldout",
        source_id=metrics.TRUE_FUTURE,
        population_id=metrics.ALL_CANDIDATES,
    )
    assert result["normalized_regret"] == 0.0
    tiny_equal_to_route_best = metrics.evaluate_state_population(
        [_row(0, 0.2), _row(1, 0.2 + 5e-13)],
        [0.0, 1.0],
        state_id="tiny-best",
        family="large_enclosed_maze",
        role="heldout",
        source_id=metrics.TRUE_FUTURE,
        population_id=metrics.ALL_CANDIDATES,
    )
    assert tiny_equal_to_route_best["selected_candidate_index"] == 0
    assert tiny_equal_to_route_best["normalized_regret"] == 0.0
    with pytest.raises(
        metrics.PlanningCostMetricsError,
        match="requires selected p_d to equal route-best p_d",
    ):
        metrics.evaluate_state_population(
            [_row(0, 0.2), _row(1, 0.2 + 5e-13)],
            [1.0, 0.0],
            state_id="tiny",
            family="large_enclosed_maze",
            role="heldout",
            source_id=metrics.TRUE_FUTURE,
            population_id=metrics.ALL_CANDIDATES,
        )


def test_source_summary_is_complete_per_family_and_role() -> None:
    candidates = {}
    costs = {}
    roles = ("fit", "calibration", "heldout", "heldout")
    for number, (family, role) in enumerate(zip(metrics.FAMILY_IDS, roles)):
        state = f"s{number}"
        candidates[state] = [
            _row(0, 0.2, family=family, role=role),
            _row(1, 0.0, family=family, role=role),
        ]
        costs[state] = [0.0, 1.0]
    summary = metrics.summarize_source(candidates, costs, source_id=metrics.TRUE_FUTURE)
    for population in metrics.POPULATION_IDS:
        group = summary["populations"][population]
        assert set(group["per_family"]) == set(metrics.FAMILY_IDS)
        assert set(group["per_role"]) == {"fit", "calibration", "heldout"}
        assert group["aggregate"]["candidates"] == 8
        assert group["aggregate"]["selected_stuck"] == 0
        assert group["no_family_complete_collapse"] is True


def test_source_summary_preserves_frozen_manifest_order_not_lexicographic_order() -> None:
    # Lexicographic sorting would incorrectly place purpose-10 before purpose-2.
    candidates = {
        "purpose-2": [
            _row(0, 0.2, family="synthetic_family"),
            _row(1, 0.0, family="synthetic_family"),
        ],
        "purpose-10": [
            _row(0, 0.2, family="synthetic_family"),
            _row(1, 0.0, family="synthetic_family"),
        ],
    }
    costs = {"purpose-2": [0.0, 1.0], "purpose-10": [0.0, 1.0]}
    summary = metrics.summarize_source(
        candidates,
        costs,
        source_id=metrics.TRUE_FUTURE,
        expected_families=("synthetic_family",),
    )
    for population in metrics.POPULATION_IDS:
        assert [
            row["state_id"]
            for row in summary["populations"][population]["per_state"]
        ] == ["purpose-2", "purpose-10"]
    with pytest.raises(metrics.PlanningCostMetricsError, match="manifest order"):
        metrics.summarize_source(
            candidates,
            {"purpose-10": [0.0, 1.0], "purpose-2": [0.0, 1.0]},
            source_id=metrics.TRUE_FUTURE,
            expected_families=("synthetic_family",),
        )


def test_complete_family_collapse_requires_support_and_one_positive_signal() -> None:
    base = {
        "state_id": "s",
        "ordered_pairs": 1,
        "pairwise_correct_credit": 0.5,
        "abstained": False,
        "best_route_top3": False,
        "best_route_top1": False,
        "spearman_rho": None,
        "kendall_tau_b": None,
        "mean_reciprocal_rank": 0.5,
        "mean_best_route_rank": 2.0,
        "normalized_regret": 1.0,
        "selected_route_progress_m": 0.0,
        "selected_heading_progress_rad": 0.0,
        "selected_combined_utility": 0.0,
        "oracle_best_route_progress_m": 1.0,
        "correct_abstention": False,
        "false_abstention": False,
        "selected_completed": False,
        "selected_oracle_contact": False,
        "selected_descriptive_contact_h2": False,
        "selected_descriptive_contact_h3": False,
        "selected_oracle_nonviable": False,
        "selected_stuck": False,
        "selected_candidate_index": 0,
        "candidates": 2,
        "cost_spread": 1.0,
        "cost_pair_count": 1,
        "cost_tie_count": 0,
    }
    assert metrics.aggregate_state_metrics([base])["complete_family_collapse"] is True
    assert metrics.aggregate_state_metrics([{**base, "pairwise_correct_credit": 0.500001}])["complete_family_collapse"] is False
    assert metrics.aggregate_state_metrics([{**base, "best_route_top3": True}])["complete_family_collapse"] is False
    assert metrics.aggregate_state_metrics([{**base, "selected_route_progress_m": 0.01}])["complete_family_collapse"] is False


def _manual_bootstrap(values: np.ndarray, draws: int, seed: int, comparison_id: str) -> np.ndarray:
    prefix = (
        metrics.STATE_BOOTSTRAP_NAMESPACE.encode()
        + b"\x00"
        + seed.to_bytes(8, "big")
        + b"\x00"
        + comparison_id.encode()
        + b"\x00"
    )
    output = []
    for replicate in range(draws):
        sample = []
        for draw in range(len(values)):
            digest = hashlib.sha256(
                prefix + replicate.to_bytes(4, "big") + draw.to_bytes(4, "big")
            ).digest()
            sample.append(values[int.from_bytes(digest[:8], "big") % len(values)])
        output.append(np.mean(sample))
    return np.asarray(output)


def test_hash_bootstrap_and_type7_are_exact_and_deterministic() -> None:
    candidate = {"b": 1.0, "a": 3.0, "c": 8.0}
    comparator = {"b": 0.0, "a": 1.0, "c": 5.0}
    families = {"b": "f1", "a": "f1", "c": "f2"}
    comparison_id = "SYNTHETIC_COMPARISON"
    result = metrics.paired_state_bootstrap(
        candidate,
        comparator,
        families,
        comparison_id=comparison_id,
        draws=11,
    )
    expected = _manual_bootstrap(np.asarray([1.0, 2.0, 3.0]), 11, metrics.BOOTSTRAP_SEED, comparison_id)
    assert result["state_order"] == ["b", "a", "c"]
    assert result["bootstrap_lower_95"] == metrics.type7_quantile(expected, 0.025)
    assert result["bootstrap_upper_95"] == metrics.type7_quantile(expected, 0.975)
    assert metrics.canonical_json_bytes(result) == metrics.canonical_json_bytes(
        metrics.paired_state_bootstrap(
            candidate,
            comparator,
            families,
            comparison_id=comparison_id,
            draws=11,
        )
    )
    assert metrics.type7_quantile([0.0, 10.0], 0.25) == 2.5


def test_equal_family_bootstrap_does_not_weight_large_family_more() -> None:
    candidate = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
    comparator = {key: 0.0 for key in candidate}
    families = {"a": "hard", "b": "easy", "c": "easy", "d": "easy"}
    result = metrics.paired_family_state_bootstrap(
        candidate, comparator, families, draws=5
    )
    assert result["point"] == pytest.approx(0.5)


def test_paired_materiality_is_paired_and_requires_safety_and_support() -> None:
    candidate = _comparison_source(
        metrics.TWO_STEP_PREDICTED_LATENT_COST,
        selected_progress=0.8,
        regret=0.1,
    )
    comparator = _comparison_source(
        metrics.KINEMATIC_ROUTE_BASELINE,
        selected_progress=0.6,
        regret=0.3,
    )
    result = metrics.paired_source_comparison(candidate, comparator, draws=9)
    assert result["mean_route_progress_gain_m"]["point"] == pytest.approx(0.2)
    assert result["normalized_regret_reduction"]["point"] == pytest.approx(0.2)
    assert result["hard_family_progress_ratio_gain"]["point"] == pytest.approx(0.2)
    assert [row["state_id"] for row in result["per_state"]] == [
        "state-0",
        "state-1",
        "state-2",
        "state-3",
    ]
    first = result["per_state"][0]
    assert first["candidate_selected_candidate_index"] == 2
    assert first["comparator_selected_candidate_index"] == 1
    assert first["selected_progress_delta_m"] == pytest.approx(0.2)
    assert first["normalized_regret_reduction"] == pytest.approx(0.2)
    assert first["pairwise_accuracy_delta"] == 0.0
    assert first["best_route_top3_delta"] == 0
    assert first["best_route_rank_improvement"] == 0.0
    assert result["material_improvement"] is True
    unsafe = _comparison_source(
        metrics.TWO_STEP_PREDICTED_LATENT_COST,
        selected_progress=0.8,
        regret=0.1,
        contacts=1,
    )
    blocked = metrics.paired_source_comparison(unsafe, comparator, draws=9)
    assert any(blocked["raw_material_triggers"].values())
    assert blocked["required_safety_and_support_preserved"] is False
    assert blocked["material_improvement"] is False


@pytest.mark.parametrize(
    "comparator_id",
    [
        metrics.KINEMATIC_ROUTE_BASELINE,
        metrics.RANDOM,
        metrics.TRUE_FUTURE_LATENT_COST,
        metrics.ONE_STEP_PREDICTED_LATENT_COST,
    ],
)
def test_all_four_paired_comparisons_emit_complete_per_state_rows(
    comparator_id: str,
) -> None:
    candidate = _comparison_source(
        metrics.TWO_STEP_PREDICTED_LATENT_COST,
        selected_progress=0.8,
        regret=0.1,
    )
    comparator = _comparison_source(
        comparator_id,
        selected_progress=0.6,
        regret=0.3,
    )
    result = metrics.paired_source_comparison(candidate, comparator, draws=5)
    assert len(result["per_state"]) == 4
    required = {
        "state_id",
        "family",
        "candidate_selected_candidate_index",
        "comparator_selected_candidate_index",
        "candidate_selected_progress_m",
        "comparator_selected_progress_m",
        "selected_progress_delta_m",
        "candidate_normalized_regret",
        "comparator_normalized_regret",
        "normalized_regret_reduction",
        "candidate_pairwise_accuracy",
        "comparator_pairwise_accuracy",
        "pairwise_accuracy_delta",
        "candidate_best_route_top3",
        "comparator_best_route_top3",
        "best_route_top3_delta",
        "candidate_best_route_rank",
        "comparator_best_route_rank",
        "best_route_rank_improvement",
    }
    assert all(set(row) == required for row in result["per_state"])


def test_paired_regret_can_be_unavailable_for_empty_populations() -> None:
    candidate = _comparison_source(
        metrics.TWO_STEP_PREDICTED_LATENT_COST,
        selected_progress=0.8,
        regret=None,
    )
    comparator = _comparison_source(
        metrics.KINEMATIC_ROUTE_BASELINE,
        selected_progress=0.6,
        regret=None,
    )
    result = metrics.paired_source_comparison(candidate, comparator, draws=7)
    assert result["normalized_regret_reduction"]["point"] is None
    assert result["material_criteria"]["normalized_regret"] is False


def test_true_future_gate_boundaries_and_undefined_failure() -> None:
    passing = metrics.evaluate_true_future_gate(_gate_source(metrics.TRUE_FUTURE))
    assert passing["pass"] is True
    assert passing["classification"] == "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL"
    failing = metrics.evaluate_true_future_gate(
        _gate_source(metrics.TRUE_FUTURE, spearman=None)
    )
    assert failing["pass"] is False
    assert failing["classification"] == "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO"


def test_absolute_screens_and_full_predicted_gate_are_separate() -> None:
    true = _gate_source(metrics.TRUE_FUTURE, selected_progress=10.0, best_progress=10.0)
    one = _gate_source(
        metrics.ONE_STEP_PREDICTED,
        pairwise=0.65,
        regret=0.30,
        selected_progress=7.5,
        best_progress=10.0,
        contacts=1,
        nonviable=1,
    )
    two = _gate_source(
        metrics.TWO_STEP_PREDICTED,
        pairwise=0.66,
        regret=0.29,
        selected_progress=7.6,
        best_progress=10.0,
        contacts=1,
        nonviable=1,
    )
    result = metrics.evaluate_predicted_gate(
        true_future_source=true, one_step_source=one, two_step_source=two
    )
    assert result["pass"] is True
    assert result["nonclassifying_absolute_screens"]["ONE_STEP_PREDICTED"]["pass"] is True
    assert result["nonclassifying_absolute_screens"]["TWO_STEP_PREDICTED"]["pass"] is True
    tied_pairwise = _gate_source(
        metrics.TWO_STEP_PREDICTED,
        pairwise=0.65,
        regret=0.29,
        selected_progress=7.6,
        best_progress=10.0,
        contacts=1,
        nonviable=1,
    )
    failed = metrics.evaluate_predicted_gate(
        true_future_source=true, one_step_source=one, two_step_source=tied_pairwise
    )
    assert failed["nonclassifying_absolute_screens"]["TWO_STEP_PREDICTED"]["pass"] is True
    assert failed["pass"] is False


@pytest.mark.parametrize(
    "true_pass,predicted_pass,jepa,kinematic,expected,next_id",
    [
        (False, True, True, False, "RAW_LATENT_GOAL_COST_NO_GO", "PLAN_AWARE_MONOTONE_JEPA_COST_V1"),
        (True, True, False, True, "KINEMATIC_BASELINE_DOMINANT", "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1"),
        (True, True, True, True, "TWO_STEP_JEPA_PLANNING_COST_SIGNAL", "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1"),
        (True, True, False, False, "TWO_STEP_JEPA_PLANNING_COST_SIGNAL", "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1"),
        (True, False, True, False, "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO", "PLAN_AWARE_MONOTONE_JEPA_COST_V1"),
    ],
)
def test_classification_precedence_and_next_mapping(
    true_pass, predicted_pass, jepa, kinematic, expected, next_id
) -> None:
    predicted_gate = {
        "pass": predicted_pass,
        "nonclassifying_absolute_screens": {
            "ONE_STEP_PREDICTED": {"pass": True, "source_id": "ONE_STEP_PREDICTED"},
            "TWO_STEP_PREDICTED": {"pass": predicted_pass, "source_id": "TWO_STEP_PREDICTED"},
        },
    }
    result = metrics.classify_qualification(
        true_future_gate={
            "pass": true_pass,
            "classification": (
                "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL"
                if true_pass
                else "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO"
            ),
        },
        predicted_gate=predicted_gate,
        jepa_vs_kinematic={"material_improvement": jepa},
        kinematic_vs_jepa={"material_improvement": kinematic},
    )
    assert result["primary_classification"] == expected
    assert result["next_experiment"] == next_id
    assert result["secondary_classifications"] == (
        [metrics.SECONDARY_INCREMENTAL_VALUE] if jepa else []
    )
    assert result["true_future_gate_classification"] == (
        "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL"
        if true_pass
        else "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO"
    )
    assert result["two_step_gate_passed"] is predicted_pass
    assert result["two_step_gate_signal_or_null"] == (
        "TWO_STEP_JEPA_PLANNING_COST_SIGNAL" if predicted_pass else None
    )
    assert set(result["predicted_base_screens"]) == {
        "ONE_STEP_PREDICTED",
        "TWO_STEP_PREDICTED",
    }
    assert result["diagnostic_flags"] == {
        "both_predicted_base_screens_failed": False,
        "ONE_STEP_BASE_SCREEN_ONLY": not predicted_pass,
        "two_step_base_screen_passed_but_full_gate_failed": False,
    }


def test_classification_descriptive_flags_are_derived_from_base_screens() -> None:
    result = metrics.classify_qualification(
        true_future_gate={
            "pass": True,
            "classification": "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL",
        },
        predicted_gate={
            "pass": False,
            "nonclassifying_absolute_screens": {
                "ONE_STEP_PREDICTED": {"pass": False},
                "TWO_STEP_PREDICTED": {"pass": True},
            },
        },
        jepa_vs_kinematic={"material_improvement": False},
        kinematic_vs_jepa={"material_improvement": False},
    )
    assert result["diagnostic_flags"] == {
        "both_predicted_base_screens_failed": False,
        "ONE_STEP_BASE_SCREEN_ONLY": False,
        "two_step_base_screen_passed_but_full_gate_failed": True,
    }


def test_classification_fails_closed_without_predicted_base_screens() -> None:
    with pytest.raises(
        metrics.PlanningCostMetricsError,
        match="lacks both frozen absolute base-preservation screens",
    ):
        metrics.classify_qualification(
            true_future_gate={"pass": True},
            predicted_gate={"pass": False},
            jepa_vs_kinematic={"material_improvement": False},
            kinematic_vs_jepa={"material_improvement": False},
        )


def test_latent_progress_diagnostics_cover_horizons_family_role_and_monotonicity() -> None:
    rows = []
    for state_number, (family, role) in enumerate(
        [("large_enclosed_maze", "fit"), ("loop_alias_stress", "heldout")]
    ):
        for candidate_index, progress in enumerate([0.0, 0.2]):
            rows.append(
                {
                    **_row(candidate_index, progress, progress, family=family, role=role),
                    "p_d_h1": 0.2 - progress,
                    "p_theta_h1": 0.2 - progress,
                    "state_id": f"s{state_number}",
                    "cost_current": 1.0,
                    "cost_h1": 0.9 - progress,
                    "cost_h2": 0.8 - progress,
                    "cost_h3": 0.7 - progress,
                }
            )
    result = metrics.latent_progress_diagnostics(rows, source_id=metrics.TRUE_FUTURE)
    assert set(result["horizons"]) == {"H1", "H2", "H3"}
    assert set(result["horizons"]["H3"]["per_family"]) == {
        "large_enclosed_maze",
        "loop_alias_stress",
    }
    assert set(result["horizons"]["H3"]["per_role"]) == {"fit", "heldout"}
    assert result["monotonicity"]["all"]["overall_monotonic_trajectory_fraction"] == 1.0
    assert result["horizons"]["H1"]["all"][
        "against_realized_distance_progress"
    ]["spearman_rho"] == pytest.approx(-1.0)
    assert result["horizons"]["H3"]["all"][
        "against_realized_distance_progress"
    ]["spearman_rho"] == pytest.approx(1.0)
    assert result["gate"] is False
    assert all(row["overall_monotonic_trajectory"] for row in result["per_candidate"])


def test_monotonic_diagnostics_distinguish_ties_from_strict_decreases() -> None:
    row = {
        **_row(0, 0.1),
        "state_id": "tie",
        "cost_current": 1.0,
        "cost_h1": 0.8,
        "cost_h2": 0.8,
        "cost_h3": 0.7,
    }
    result = metrics.latent_progress_diagnostics([row], source_id=metrics.TRUE_FUTURE)
    aggregate = result["monotonicity"]["all"]
    assert aggregate["h1_to_h2_nonincrease_fraction"] == 1.0
    assert aggregate["h1_to_h2_strict_decrease_fraction"] == 0.0
    assert aggregate["overall_monotonic_trajectory_fraction"] == 1.0
    assert aggregate["overall_strict_decrease_trajectory_fraction"] == 0.0
    assert aggregate["current_to_h3_nonincrease_fraction"] == 1.0
    assert aggregate["current_to_h3_strict_decrease_fraction"] == 1.0
    assert result["per_candidate"][0]["current_to_h3_nonincrease"] is True
    assert result["per_candidate"][0]["current_to_h3_strict_decrease"] is True


def test_current_to_h3_endpoint_decrease_is_distinct_from_stepwise_monotonicity() -> None:
    row = {
        **_row(0, 0.1),
        "state_id": "nonmonotone-endpoint-improvement",
        "cost_current": 1.0,
        "cost_h1": 0.8,
        "cost_h2": 0.9,
        "cost_h3": 0.7,
    }
    result = metrics.latent_progress_diagnostics([row], source_id=metrics.TRUE_FUTURE)
    aggregate = result["monotonicity"]["all"]
    assert aggregate["overall_monotonic_trajectory_fraction"] == 0.0
    assert aggregate["current_to_h3_nonincrease_fraction"] == 1.0
    assert aggregate["current_to_h3_strict_decrease_fraction"] == 1.0
    candidate = result["per_candidate"][0]
    assert candidate["overall_monotonic_trajectory"] is False
    assert candidate["current_to_h3_nonincrease"] is True
    assert candidate["current_to_h3_strict_decrease"] is True


def test_all_candidate_tendency_diagnostics_are_outcome_separated() -> None:
    rows = {
        "s": [
            _row(0, 0.2, contact=False, viable=True, stuck=False),
            _row(1, 0.0, contact=True, viable=False, stuck=True),
        ]
    }
    result = metrics.all_candidate_tendency_diagnostics(
        rows, {"s": [0.1, 0.9]}, source_id=metrics.TWO_STEP_PREDICTED
    )
    assert result["all"]["immediate_contact_h1"]["downranking_accuracy"] == 1.0
    assert result["all"]["successor_nonviable"]["downranking_accuracy"] == 1.0
    assert result["all"]["stuck"]["downranking_accuracy"] == 1.0
    assert result["all"]["no_progress"]["downranking_accuracy"] == 1.0
    ranks = result["all"]["immediate_contact_h1"]["cost_rank_by_class"]
    assert ranks["contact"]["mean_cost_rank"] == 2.0
    assert ranks["contact_free"]["median_cost_rank"] == 1.0
    assert result["diagnostic_only"] is True and result["gate"] is False


def test_random_order_uses_exact_registered_bytes() -> None:
    state_id = "state-μ"
    indices = [4, 1, 9]
    prefix = (
        metrics.RANDOM_NAMESPACE.encode()
        + b"\x00"
        + metrics.BOOTSTRAP_SEED.to_bytes(8, "big")
        + b"\x00"
        + state_id.encode()
        + b"\x00"
    )
    expected = sorted(
        indices,
        key=lambda value: (
            hashlib.sha256(prefix + value.to_bytes(4, "big")).digest(),
            value,
        ),
    )
    assert metrics.deterministic_random_order(state_id, indices) == expected


def test_canonical_json_is_byte_stable_and_rejects_nonfinite() -> None:
    left = {"b": np.asarray([2, 1]), "a": np.float64(0.5)}
    right = {"a": 0.5, "b": [2, 1]}
    assert metrics.canonical_json_bytes(left) == metrics.canonical_json_bytes(right)
    assert metrics.canonical_sha256(left) == metrics.canonical_sha256(right)
    with pytest.raises(metrics.PlanningCostMetricsError):
        metrics.canonical_json_bytes({"bad": math.inf})
