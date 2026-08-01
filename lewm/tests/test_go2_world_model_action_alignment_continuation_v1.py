from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_action_alignment_continuation_v1 as metric
from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as three_arm


@dataclass(frozen=True)
class Row:
    index: int
    role: str
    family: str
    scene_id: str
    rgb: tuple[str, str, str, str, str, str]
    actions: tuple[int, int, int, int, int, int]


def rows() -> tuple[Row, ...]:
    result = []
    for family_id, family in enumerate(three_arm.REGISTERED_FAMILIES):
        for action in range(three_arm.ACTION_COUNT):
            for scene_copy in range(2):
                result.append(
                    Row(
                        index=len(result),
                        role="val",
                        family=family,
                        scene_id=(
                            f"family-{family_id}-action-{action}-scene-{scene_copy}"
                        ),
                        rgb=("a", "b", "c", "d", "e", "f"),
                        actions=(0, 1, action, 0, 0, 0),
                    )
                )
    return tuple(result)


def candidate_panel(
    selected_rows: tuple[Row, ...], margins_by_action: tuple[float, ...]
) -> np.ndarray:
    panel = np.empty((len(selected_rows), three_arm.ACTION_COUNT), dtype=np.float64)
    for index, row in enumerate(selected_rows):
        panel[index] = 1.0 + margins_by_action[row.actions[2]]
        panel[index, row.actions[2]] = 1.0
    return panel


def margin_vector(*, hardest_action: int, hardest_margin: float) -> tuple[float, ...]:
    return tuple(
        hardest_margin if action == hardest_action else 0.01
        for action in range(three_arm.ACTION_COUNT)
    )


def decision_kwargs(
    *,
    treatment_u700: tuple[float, ...],
    treatment_u900: tuple[float, ...],
    baseline_u700: tuple[float, ...] | None = None,
    baseline_u900: tuple[float, ...] | None = None,
) -> dict[str, object]:
    selected = rows()
    count = len(selected)
    factual = np.ones(count)
    control = np.full(count, 1.2)
    return {
        "baseline_candidate_energy_u700": candidate_panel(
            selected, baseline_u700 or treatment_u700
        ),
        "baseline_candidate_energy_u900": candidate_panel(
            selected, baseline_u900 or treatment_u900
        ),
        "treatment_candidate_energy_u700": candidate_panel(selected, treatment_u700),
        "treatment_factual_energy_u700": factual,
        "treatment_persistence_energy_u700": control,
        "treatment_wrong_history_energy_u700": control,
        "treatment_candidate_energy_u900": candidate_panel(selected, treatment_u900),
        "treatment_factual_energy_u900": factual,
        "treatment_persistence_energy_u900": control,
        "treatment_wrong_history_energy_u900": control,
        "validation_rows": selected,
        "treatment_rank_ratio_by_update": {700: 0.3, 800: 0.3, 900: 0.3},
        "contract_checks": {"exact": True},
        "train_fit_checks": {"finite": True},
    }


def test_absolute_gain_uses_shared_checkpoint_weights():
    selected = rows()
    u700 = margin_vector(hardest_action=8, hardest_margin=-0.01)
    u900 = margin_vector(hardest_action=8, hardest_margin=-0.008)
    result = metric.paired_absolute_hardest_margin_gain(
        treatment_candidate_energy_u700=candidate_panel(selected, u700),
        treatment_candidate_energy_u900=candidate_panel(selected, u900),
        validation_rows=selected,
    )
    assert result["bootstrap_seed"] == 20_260_812
    assert result["bootstrap_replicates"] == 10_000
    assert result["bootstrap_lower_index"] == 500
    assert result["bootstrap_median_index"] == 5_000
    assert result["bootstrap_upper_index"] == 9_499
    assert result["point"] == pytest.approx(0.002, abs=1e-15)
    assert result["one_sided_95_lower_quantile"] == pytest.approx(0.002, abs=1e-15)
    assert result["median_quantile"] == pytest.approx(0.002, abs=1e-15)
    assert result["one_sided_95_upper_quantile"] == pytest.approx(0.002, abs=1e-15)
    assert result["absolute_gain_threshold"] == 0.001298360001376009
    assert result["recovery_threshold_affects_decision"] is False


@pytest.mark.parametrize(
    (
        "contract",
        "retention",
        "repaired",
        "persistence_failures",
        "persistence_lower",
        "point",
        "lower",
        "upper",
        "expected",
    ),
    [
        (False, True, True, 0, 1.0, 1.0, 1.0, 1.0, "FAIL_CONTRACT_CLOSE_ALIGNMENT_BRANCH"),
        (True, False, True, 0, 1.0, 1.0, 1.0, 1.0, "FAIL_RETENTION_CLOSE_ALIGNMENT_BRANCH"),
        (
            True,
            True,
            True,
            5,
            -1.0,
            0.0,
            -1.0,
            -1.0,
            "PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PERSISTENCE_SYSTEMIC",
        ),
        (
            True,
            True,
            True,
            1,
            -1.0,
            0.0,
            -1.0,
            -1.0,
            "PASS_ACTION_ALIGNMENT_PROXY_REPAIR_PLANNING_WITH_PROXY_CAVEAT",
        ),
        (
            True,
            True,
            True,
            0,
            1.0,
            0.0,
            -1.0,
            -1.0,
            "PASS_EXPLORATORY_ACTION_ALIGNMENT_AND_PREDICTOR_USEFULNESS_PROXY",
        ),
        (
            True,
            True,
            False,
            0,
            1.0,
            metric.ABSOLUTE_GAIN_THRESHOLD,
            1e-9,
            1.0,
            "MEANINGFUL_ABSOLUTE_PROGRESS_INCOMPLETE_CONTINUE_SAME_MECHANISM",
        ),
        (
            True,
            True,
            False,
            0,
            1.0,
            metric.ABSOLUTE_GAIN_THRESHOLD - 1e-12,
            1e-9,
            1.0,
            "POSITIVE_BUT_INSUFFICIENT_RATE_CLOSE_ALIGNMENT_BRANCH",
        ),
        (
            True,
            True,
            False,
            0,
            1.0,
            1e-3,
            0.0,
            1.0,
            "INCONCLUSIVE_ABSOLUTE_CHANGE_CLOSE_ALIGNMENT_BRANCH",
        ),
        (
            True,
            True,
            False,
            0,
            1.0,
            0.0,
            -1.0,
            1.0,
            "STALLED_OR_HARMFUL_CLOSE_ALIGNMENT_BRANCH",
        ),
        (
            True,
            True,
            False,
            0,
            1.0,
            1e-3,
            -1.0,
            0.0,
            "STALLED_OR_HARMFUL_CLOSE_ALIGNMENT_BRANCH",
        ),
    ],
)
def test_terminal_decision_precedence(
    contract,
    retention,
    repaired,
    persistence_failures,
    persistence_lower,
    point,
    lower,
    upper,
    expected,
):
    status, _branch, _next = metric.classify_continuation_outcome(
        contract_passed=contract,
        retention_passed=retention,
        action_alignment_repaired=repaired,
        persistence_lower_failure_count=persistence_failures,
        aggregate_persistence_lower=persistence_lower,
        absolute_gain_point=point,
        absolute_gain_lower=lower,
        absolute_gain_upper=upper,
    )
    assert status == expected


def test_degrading_baseline_cannot_rescue_absolute_treatment_stall():
    treatment_u700 = margin_vector(hardest_action=8, hardest_margin=-0.01)
    treatment_u900 = margin_vector(hardest_action=8, hardest_margin=-0.011)
    baseline_u700 = margin_vector(hardest_action=8, hardest_margin=-0.02)
    baseline_u900 = margin_vector(hardest_action=8, hardest_margin=-0.20)
    decision = metric.decide_alignment_continuation(
        **decision_kwargs(
            treatment_u700=treatment_u700,
            treatment_u900=treatment_u900,
            baseline_u700=baseline_u700,
            baseline_u900=baseline_u900,
        )
    )
    relative = decision["concurrent_baseline_relative_delta_diagnostic_only"]
    assert relative["by_update"]["u900"]["point"] > 0.1
    assert relative["point_change_u900_minus_u700"] > 0.1
    assert relative["decision_relevant"] is False
    assert decision["absolute_treatment_hardest_margin_gain"]["point"] < 0.0
    assert decision["status"] == "STALLED_OR_HARMFUL_CLOSE_ALIGNMENT_BRANCH"
    assert decision["selected_next_step"] == (
        "NO_FURTHER_ALIGNMENT_TRAINING_OR_PLANNING_GATE"
    )
    assert decision["authorizes_further_alignment_training"] is False


def test_pace_level_absolute_gain_permits_only_separate_same_mechanism_preregistration():
    selected = rows()
    treatment_u700 = margin_vector(hardest_action=8, hardest_margin=-0.01)
    treatment_u900 = margin_vector(
        hardest_action=8,
        hardest_margin=-0.01 + metric.ABSOLUTE_GAIN_THRESHOLD + 1.0e-6,
    )
    decision = metric.decide_alignment_continuation(
        **decision_kwargs(
            treatment_u700=treatment_u700,
            treatment_u900=treatment_u900,
        )
    )
    assert selected
    assert decision["status"] == (
        "MEANINGFUL_ABSOLUTE_PROGRESS_INCOMPLETE_CONTINUE_SAME_MECHANISM"
    )
    assert decision["selected_next_step"] == (
        "PREREGISTER_NEXT_FIXED_SAME_MECHANISM_BLOCK"
    )
    assert decision["permits_separate_same_mechanism_preregistration"] is True
    assert decision["authorizes_next_execution"] is False
    assert decision["authorizes_further_alignment_training"] is False


def test_relative_gain_cannot_override_preserved_action_regression():
    treatment_u700 = margin_vector(hardest_action=8, hardest_margin=-0.02)
    treatment_u900_values = list(
        margin_vector(hardest_action=8, hardest_margin=-0.005)
    )
    treatment_u900_values[0] = -0.001
    treatment_u900 = tuple(treatment_u900_values)
    baseline_u700 = margin_vector(hardest_action=8, hardest_margin=-0.03)
    baseline_u900 = margin_vector(hardest_action=8, hardest_margin=-0.20)
    decision = metric.decide_alignment_continuation(
        **decision_kwargs(
            treatment_u700=treatment_u700,
            treatment_u900=treatment_u900,
            baseline_u700=baseline_u700,
            baseline_u900=baseline_u900,
        )
    )
    assert decision["absolute_treatment_hardest_margin_gain"]["point"] > (
        metric.ABSOLUTE_GAIN_THRESHOLD
    )
    assert (
        decision["concurrent_baseline_relative_delta_diagnostic_only"]["by_update"]
        ["u900"]["point"]
        > 0.1
    )
    assert decision["retention"]["preservation_checks_by_action_id"]["0"] is False
    assert decision["status"] == "FAIL_RETENTION_CLOSE_ALIGNMENT_BRANCH"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract_passed", 1, "contract passed must be boolean"),
        ("persistence_lower_failure_count", True, "persistence failure count"),
        ("aggregate_persistence_lower", float("nan"), "aggregate persistence lower"),
        ("absolute_gain_point", "1", "absolute gain point"),
    ],
)
def test_classifier_rejects_nonexact_or_nonfinite_inputs(field, value, message):
    kwargs = {
        "contract_passed": True,
        "retention_passed": True,
        "action_alignment_repaired": False,
        "persistence_lower_failure_count": 0,
        "aggregate_persistence_lower": 1.0,
        "absolute_gain_point": 1.0,
        "absolute_gain_lower": 1.0,
        "absolute_gain_upper": 1.0,
    }
    kwargs[field] = value
    with pytest.raises(metric.AlignmentContinuationMetricError, match=message):
        metric.classify_continuation_outcome(**kwargs)


def test_decision_rejects_vacuous_checks_and_bad_rank_contract():
    kwargs = decision_kwargs(
        treatment_u700=margin_vector(hardest_action=8, hardest_margin=-0.01),
        treatment_u900=margin_vector(hardest_action=8, hardest_margin=-0.008),
    )
    kwargs["contract_checks"] = {}
    with pytest.raises(metric.AlignmentContinuationMetricError, match="nonempty mapping"):
        metric.decide_alignment_continuation(**kwargs)

    kwargs["contract_checks"] = {"exact": True}
    kwargs["treatment_rank_ratio_by_update"] = {700: 0.3, 900: 0.3}
    with pytest.raises(metric.AlignmentContinuationMetricError, match="exactly u700/u800/u900"):
        metric.decide_alignment_continuation(**kwargs)


def test_contract_failure_is_reported_inside_retention_bundle_too():
    kwargs = decision_kwargs(
        treatment_u700=margin_vector(hardest_action=8, hardest_margin=-0.01),
        treatment_u900=margin_vector(hardest_action=8, hardest_margin=-0.008),
    )
    kwargs["contract_checks"] = {"exact": False}
    decision = metric.decide_alignment_continuation(**kwargs)
    assert decision["status"] == "FAIL_CONTRACT_CLOSE_ALIGNMENT_BRANCH"
    assert decision["contract"]["passed"] is False
    assert decision["retention"]["checks"]["all_contract_checks"] is False
    assert decision["retention"]["passed"] is False


def test_absolute_metric_rejects_text_boolean_nan_and_missing_action_inputs():
    selected = rows()
    valid = candidate_panel(
        selected, margin_vector(hardest_action=8, hardest_margin=-0.01)
    )
    bad_text = valid.astype(str)
    with pytest.raises(metric.AlignmentContinuationMetricError, match="not boolean or text"):
        metric.paired_absolute_hardest_margin_gain(
            treatment_candidate_energy_u700=bad_text,
            treatment_candidate_energy_u900=valid,
            validation_rows=selected,
        )
    bad_nan = valid.copy()
    bad_nan[0, 0] = np.nan
    with pytest.raises(metric.AlignmentContinuationMetricError, match="finite, nonnegative"):
        metric.paired_absolute_hardest_margin_gain(
            treatment_candidate_energy_u700=bad_nan,
            treatment_candidate_energy_u900=valid,
            validation_rows=selected,
        )
    missing_action_rows = tuple(row for row in selected if row.actions[2] != 8)
    reindexed = tuple(
        Row(
            index=index,
            role=row.role,
            family=row.family,
            scene_id=row.scene_id,
            rgb=row.rgb,
            actions=row.actions,
        )
        for index, row in enumerate(missing_action_rows)
    )
    with pytest.raises(metric.AlignmentContinuationMetricError, match="action is absent"):
        metric.paired_absolute_hardest_margin_gain(
            treatment_candidate_energy_u700=valid[: len(reindexed)],
            treatment_candidate_energy_u900=valid[: len(reindexed)],
            validation_rows=reindexed,
        )
