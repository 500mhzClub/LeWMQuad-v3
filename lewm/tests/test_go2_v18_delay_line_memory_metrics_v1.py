from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from lewm.benchmarks.go2_v18_delay_line_memory_metrics_v1 import (
    REGISTERED_FAMILIES,
    ObservationMetrics,
    RuntimeSafeguards,
    SubstrateMetrics,
    evaluate_temporal_metrics,
    participation_rank,
    terminal_qualification_decision,
    update250_futility_decision,
    update500_continuation_decision,
)


def _evaluate_from_lifts(
    persistence_lift: tuple[float, float, float, float],
    action_lift: tuple[float, float, float, float],
    history_lift: tuple[float, float, float, float],
    *,
    history_by_family: tuple[float, ...] | None = None,
):
    scenes: list[str] = []
    families: list[str] = []
    p_rows: list[tuple[float, ...]] = []
    a_rows: list[tuple[float, ...]] = []
    h_rows: list[tuple[float, ...]] = []
    for family_index, family in enumerate(REGISTERED_FAMILIES):
        for _ in range(2):
            scenes.append(f"scene-{family_index}")
            families.append(family)
            p_rows.append(persistence_lift)
            a_rows.append(action_lift)
            if history_by_family is None:
                h_rows.append(history_lift)
            else:
                h_rows.append(
                    (
                        history_lift[0],
                        history_lift[1],
                        history_lift[2],
                        history_by_family[family_index],
                    )
                )
    persistence = torch.ones(len(scenes), 4, dtype=torch.float64)
    p = torch.tensor(p_rows, dtype=torch.float64)
    a = torch.tensor(a_rows, dtype=torch.float64)
    h = torch.tensor(h_rows, dtype=torch.float64)
    real = persistence * (1.0 - p)
    wrong = real + a * persistence
    best_corrupt = real + h * persistence
    return evaluate_temporal_metrics(
        real,
        persistence,
        wrong,
        best_corrupt + 0.2,
        best_corrupt + 0.1,
        best_corrupt,
        scenes,
        families,
        bootstrap_replicates=64,
    )


def _passing_rank():
    return participation_rank(torch.eye(16, dtype=torch.float64))


def _safeguards(**changes):
    return replace(
        RuntimeSafeguards(
            integrity_pass=True,
            perception_safeguards_pass=True,
            gradient_accounting_pass=True,
            target_noncollapsed=True,
            online_noncollapsed=True,
        ),
        **changes,
    )


def _substrate(**changes):
    return replace(
        SubstrateMetrics(
            place_chance_multiple=2.1,
            place_scene_count_above_chance=7,
            target_place_rank=2.4,
            target_place_rank_retention=0.9,
            physical_passed_margin_count=60,
            physical_causal_control_pass_count=12,
        ),
        **changes,
    )


def _observation(update: int, temporal, **changes):
    return replace(
        ObservationMetrics(
            update=update,
            temporal=temporal,
            memory_state=_passing_rank(),
            safeguards=_safeguards(),
            substrate=_substrate(),
        ),
        **changes,
    )


def test_temporal_formulas_use_best_corrupted_history_and_exact_horizons() -> None:
    metrics = _evaluate_from_lifts(
        (0.1, 0.2, 0.3, 0.4),
        (0.05, 0.06, 0.07, 0.08),
        (0.01, 0.02, 0.03, 0.04),
    )
    assert metrics.score.macro == pytest.approx((0.9, 0.8, 0.7, 0.6))
    assert metrics.persistence_lift.macro == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert metrics.action_lift.macro == pytest.approx((0.05, 0.06, 0.07, 0.08))
    assert metrics.history_lift.macro == pytest.approx((0.01, 0.02, 0.03, 0.04))
    assert metrics.history_lift.positive_family_count == (8, 8, 8, 8)
    assert metrics.row_count == 16
    assert metrics.scene_count == 8
    assert metrics.family_count == 8


def test_aggregation_is_equal_scene_then_equal_family_not_row_weighted() -> None:
    scenes: list[str] = []
    families: list[str] = []
    p_values: list[float] = []
    for family_index, family in enumerate(REGISTERED_FAMILIES):
        if family_index == 0:
            scenes.extend(["heavy"] * 10 + ["light"])
            families.extend([family] * 11)
            p_values.extend([0.1] * 10 + [0.5])
        else:
            scenes.append(f"scene-{family_index}")
            families.append(family)
            p_values.append(0.2)
    p = torch.tensor(p_values, dtype=torch.float64)[:, None].expand(-1, 4)
    persistence = torch.ones_like(p)
    real = persistence - p
    metrics = evaluate_temporal_metrics(
        real,
        persistence,
        real + 0.1,
        real + 0.3,
        real + 0.2,
        real + 0.1,
        scenes,
        families,
        bootstrap_replicates=64,
    )
    assert metrics.persistence_lift.per_scene["heavy"][0] == pytest.approx(0.1)
    assert metrics.persistence_lift.per_scene["light"][0] == pytest.approx(0.5)
    assert metrics.persistence_lift.per_family[REGISTERED_FAMILIES[0]][0] == pytest.approx(0.3)
    assert metrics.persistence_lift.macro[0] == pytest.approx(
        (0.3 + 7 * 0.2) / 8
    )


def test_scene_bootstrap_is_deterministic_and_stratified_by_family() -> None:
    first = _evaluate_from_lifts(
        (0.1, 0.1, 0.1, 0.1),
        (0.1, 0.1, 0.1, 0.1),
        (0.1, 0.1, 0.1, 0.1),
    )
    second = _evaluate_from_lifts(
        (0.1, 0.1, 0.1, 0.1),
        (0.1, 0.1, 0.1, 0.1),
        (0.1, 0.1, 0.1, 0.1),
    )
    assert first.persistence_lift.bootstrap_lower_95 == (
        second.persistence_lift.bootstrap_lower_95
    )
    assert first.persistence_lift.bootstrap_lower_95 == pytest.approx(
        (0.1, 0.1, 0.1, 0.1)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("zero_persistence", "strictly positive"),
        ("negative_energy", "finite nonnegative"),
        ("wrong_shape", "shape"),
        ("missing_family", "exactly the eight"),
        ("scene_in_two_families", "multiple families"),
    ],
)
def test_temporal_input_contract_rejects_invalid_values(
    mutation: str, message: str
) -> None:
    n = 8
    energy = torch.ones(n, 4)
    energies = [energy.clone() for _ in range(6)]
    scenes = [f"scene-{index}" for index in range(n)]
    families = list(REGISTERED_FAMILIES)
    if mutation == "zero_persistence":
        energies[1][0, 0] = 0.0
    elif mutation == "negative_energy":
        energies[2][0, 0] = -1.0
    elif mutation == "wrong_shape":
        energies[5] = torch.ones(n, 3)
    elif mutation == "missing_family":
        families[-1] = families[0]
    else:
        scenes[-1] = scenes[0]
    with pytest.raises(ValueError, match=message):
        evaluate_temporal_metrics(
            *energies,
            scenes,
            families,
            bootstrap_replicates=64,
        )


def test_participation_rank_distinguishes_full_rank_low_rank_and_nonfinite() -> None:
    healthy = participation_rank(torch.eye(16, dtype=torch.float64))
    assert healthy.effective_rank == pytest.approx(15.0)
    assert healthy.participation_rank_ratio == pytest.approx(15 / 16)
    assert healthy.near_zero_fraction == 0.0
    assert healthy.noncollapsed

    axis = torch.linspace(-1.0, 1.0, 32)[:, None]
    collapsed = participation_rank(axis.expand(-1, 16))
    assert collapsed.effective_rank == pytest.approx(1.0)
    assert collapsed.participation_rank_ratio == pytest.approx(1 / 16)
    assert not collapsed.noncollapsed

    nonfinite_rows = torch.eye(4)
    nonfinite_rows[0, 0] = float("nan")
    nonfinite = participation_rank(nonfinite_rows)
    assert not nonfinite.finite
    assert not nonfinite.nonzero_scale
    assert not nonfinite.noncollapsed


def test_update250_stops_only_for_structure_or_joint_two_observation_futility() -> None:
    dead = _evaluate_from_lifts(
        (-0.1, -0.1, -0.1, -0.1),
        (-0.1, -0.1, -0.1, -0.1),
        (-0.1, -0.1, -0.1, -0.1),
    )
    decision = update250_futility_decision(
        _observation(100, dead), _observation(250, dead)
    )
    assert not decision.passed
    assert decision.status == "STOP_UPDATE250_SCIENTIFIC_FUTILITY"
    assert decision.failed_checks == (
        "not_jointly_futile_at_updates_100_and_250",
    )

    emerging = _evaluate_from_lifts(
        (0.01, 0.01, 0.01, 0.01),
        (-0.01, -0.01, -0.01, -0.01),
        (-0.01, -0.01, -0.01, -0.01),
    )
    continued = update250_futility_decision(
        _observation(100, dead), _observation(250, emerging)
    )
    assert continued.passed
    assert continued.status == "CONTINUE_TO_UPDATE500"

    bad_place = update250_futility_decision(
        _observation(100, emerging),
        _observation(
            250,
            emerging,
            substrate=_substrate(place_chance_multiple=1.49),
        ),
    )
    assert not bad_place.passed
    assert bad_place.status == "STOP_UPDATE250_INTEGRITY_OR_COLLAPSE"
    assert "place_at_least_1p5x_chance" in bad_place.failed_checks


def test_update500_requires_robust_persistence_action_and_four_history_families() -> None:
    passing = _evaluate_from_lifts(
        (0.02, 0.03, 0.04, 0.05),
        (0.01, 0.01, 0.01, 0.02),
        (0.0, 0.0, 0.0, 0.01),
        history_by_family=(0.02, 0.02, 0.02, 0.02, -0.01, -0.01, -0.01, -0.01),
    )
    decision = update500_continuation_decision(_observation(500, passing))
    assert decision.passed
    assert decision.status == "CONTINUE_TO_UPDATE1000"
    assert decision.observed["h4_history_positive_family_count"] == 4

    failing = _evaluate_from_lifts(
        (0.02, 0.03, 0.04, 0.05),
        (0.01, 0.01, 0.01, 0.02),
        (0.0, 0.0, 0.0, 0.01),
        history_by_family=(0.02, 0.02, 0.02, -0.01, -0.01, -0.01, -0.01, -0.01),
    )
    stopped = update500_continuation_decision(_observation(500, failing))
    assert not stopped.passed
    assert stopped.status == "STOP_UPDATE500_CONTINUATION_GATE"
    assert stopped.failed_checks == ("history_positive_in_four_families",)


def test_terminal_selects_minimum_mean_score_among_eligible_and_passes() -> None:
    update500 = _evaluate_from_lifts(
        (0.11, 0.11, 0.11, 0.11),
        (0.06, 0.06, 0.06, 0.06),
        (0.01, 0.01, 0.01, 0.04),
    )
    update750 = _evaluate_from_lifts(
        (0.15, 0.15, 0.15, 0.15),
        (0.06, 0.06, 0.06, 0.06),
        (0.01, 0.01, 0.01, 0.04),
    )
    update1000 = _evaluate_from_lifts(
        (0.12, 0.12, 0.12, 0.12),
        (0.06, 0.06, 0.06, 0.06),
        (0.01, 0.01, 0.01, 0.04),
    )
    decision = terminal_qualification_decision(
        (
            _observation(500, update500),
            _observation(750, update750),
            _observation(1000, update1000),
        )
    )
    assert decision.passed
    assert decision.status == "PASS_SHORT_HORIZON_CAUSAL_MEMORY_SUBSTRATE"
    assert decision.selected_update == 750
    assert decision.observed["selected_mean_score"] == pytest.approx(0.85)
    assert not decision.failed_checks


def test_terminal_excludes_ineligible_lower_score_and_reports_exact_failure() -> None:
    ineligible_best_score = _evaluate_from_lifts(
        (0.20, 0.20, 0.20, 0.20),
        (0.06, 0.06, 0.06, 0.06),
        (0.01, 0.01, 0.01, -0.01),
    )
    eligible_but_small_history = _evaluate_from_lifts(
        (0.12, 0.12, 0.12, 0.12),
        (0.06, 0.06, 0.06, 0.06),
        (0.01, 0.01, 0.01, 0.02),
    )
    worse = _evaluate_from_lifts(
        (0.11, 0.11, 0.11, 0.11),
        (0.06, 0.06, 0.06, 0.06),
        (0.01, 0.01, 0.01, 0.04),
    )
    decision = terminal_qualification_decision(
        (
            _observation(500, ineligible_best_score),
            _observation(750, eligible_but_small_history),
            _observation(1000, worse),
        )
    )
    assert not decision.passed
    assert decision.selected_update == 750
    assert decision.status == "FAIL_TERMINAL_QUALIFICATION"
    assert decision.failed_checks == ("h4_history_lift_at_least_0p03",)


def test_terminal_requires_all_three_registered_selection_observations() -> None:
    metrics = _evaluate_from_lifts(
        (0.12, 0.12, 0.12, 0.12),
        (0.06, 0.06, 0.06, 0.06),
        (0.01, 0.01, 0.01, 0.04),
    )
    with pytest.raises(ValueError, match="500, 750, and 1000"):
        terminal_qualification_decision(
            (_observation(500, metrics), _observation(1000, metrics))
        )
