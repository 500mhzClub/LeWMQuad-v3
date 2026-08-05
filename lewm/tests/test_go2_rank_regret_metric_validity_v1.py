"""Focused tests for the rank-regret metric-validity study V1 contract."""

from __future__ import annotations

import numpy as np
import pytest

from lewm.benchmarks import go2_observability_ceiling_assay_v1 as assay
from lewm.benchmarks import go2_rank_regret_metric_validity_v1 as validity

from lewm.tests.test_go2_observability_ceiling_assay_v1 import _group


# ------------------------------------------------------------ correlation ---


def test_spearman_is_exact_on_a_perfect_monotone_pair():
    assert validity.spearman_v1([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert validity.spearman_v1([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_uses_average_ranks_for_ties():
    # Two tied values must share the averaged rank rather than an arbitrary order.
    assert validity.spearman_v1([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)


def test_correlation_requires_at_least_three_points():
    with pytest.raises(validity.MetricValidityError):
        validity.spearman_v1([1, 2], [3, 4])


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate():
    x = [0.0, 0.02, 0.06, 0.07, 0.08, 0.13, 0.14]
    y = [0.9, 0.8, 0.65, 0.5, 0.35, 0.0, 0.0]
    first = validity.bootstrap_correlation_v1(x, y)
    second = validity.bootstrap_correlation_v1(x, y)
    assert first == second
    assert first["ci_lower"] <= first["spearman"] <= first["ci_upper"]


# -------------------------------------------------------------- part b1 ---


def test_part_b1_uses_exactly_the_seven_bound_policies():
    result = validity.part_b1_v1()
    assert len(result["policies"]) == 7
    names = {row["policy"] for row in result["policies"]}
    assert "dino_true_successor" in names and "oracle_mpc" in names
    # The bound oracle has exactly zero geometric regret by construction.
    oracle = next(r for r in result["policies"] if r["policy"] == "oracle_mpc")
    assert oracle["geometric_regret_m"] == 0.0


# ------------------------------------------------------ geometric regret ---


def test_geometric_regret_is_zero_for_the_best_branch_and_positive_otherwise():
    progress = [0.0, 0.5, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0]
    group = _group(list(range(9)), progress=progress)
    best = validity.geometric_regret_v1([group], [1])
    assert best["geometric_regret_m"] == pytest.approx(0.0)
    worse = validity.geometric_regret_v1([group], [3])
    assert worse["geometric_regret_m"] == pytest.approx(0.4)


def test_geometric_regret_rejects_a_mismatched_selection_count():
    group = _group(list(range(9)))
    with pytest.raises(validity.MetricValidityError):
        validity.geometric_regret_v1([group], [0, 1])


def test_progress_matrix_shape_matches_the_action_grid():
    group = _group(list(range(9)))
    assert validity.progress_matrix_v1([group]).shape == (1, assay.ACTION_COUNT)


# ------------------------------------------------------- rule-based arms ---


def test_rule_based_scores_have_the_action_grid_shape():
    group = _group(list(range(9)))
    for rule in ("geometric_endpoint", "bearing", "hold"):
        scores = validity.rule_based_scores_v1([group], rule=rule)
        assert scores.shape == (1, assay.ACTION_COUNT)
        assert np.isfinite(scores).all()


def test_unknown_rule_is_rejected():
    with pytest.raises(validity.MetricValidityError):
        validity.rule_based_scores_v1([_group(list(range(9)))], rule="nope")


# ---------------------------------------------------------------- decide ---


def _links(rho_closed, closed_ci, rho_rank, rank_ci):
    return (
        {"correlation": {"spearman": rho_closed, "ci_lower": closed_ci[0], "ci_upper": closed_ci[1]}},
        {"correlation": {"spearman": rho_rank, "ci_lower": rank_ci[0], "ci_upper": rank_ci[1]}},
    )


def test_valid_proxy_requires_both_links():
    b1, b2 = _links(-0.96, (-1.0, -0.7), 0.85, (0.5, 0.98))
    assert validity.decide_v1(b1, b2)["terminal"] == validity.VALID_PROXY


def test_rank_link_failure_promotes_the_geometric_metric():
    b1, b2 = _links(-0.96, (-1.0, -0.7), 0.10, (-0.4, 0.6))
    decision = validity.decide_v1(b1, b2)
    assert decision["terminal"] == validity.INVALID_AT_RANK_LINK


def test_closed_loop_link_failure_dominates():
    b1, b2 = _links(-0.10, (-0.6, 0.4), 0.95, (0.8, 1.0))
    decision = validity.decide_v1(b1, b2)
    assert decision["terminal"] == validity.INVALID_AT_CLOSED_LOOP_LINK


def test_ambiguous_when_neither_pass_nor_failure_conditions_hold():
    b1, b2 = _links(-0.75, (-0.95, -0.2), 0.5, (0.35, 0.8))
    assert validity.decide_v1(b1, b2)["terminal"] == validity.AMBIGUOUS


def test_every_outcome_states_that_the_prior_stops_stand():
    for args in (
        (-0.96, (-1.0, -0.7), 0.85, (0.5, 0.98)),
        (-0.96, (-1.0, -0.7), 0.10, (-0.4, 0.6)),
        (-0.10, (-0.6, 0.4), 0.95, (0.8, 1.0)),
    ):
        b1, b2 = _links(*args)
        assert "stand" in validity.decide_v1(b1, b2)["stops_unchanged"]
