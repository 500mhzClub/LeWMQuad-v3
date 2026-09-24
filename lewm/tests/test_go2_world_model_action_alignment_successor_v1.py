from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_action_alignment_successor_v1 as metric
from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as three_arm


@dataclass(frozen=True)
class Row:
    index: int
    role: str
    family: str
    scene_id: str
    rgb: tuple[str, str, str, str, str, str]
    actions: tuple[int, int, int]


def rows() -> tuple[Row, ...]:
    result = []
    for family_id, family in enumerate(three_arm.REGISTERED_FAMILIES):
        for action in range(three_arm.ACTION_COUNT):
            for scene_copy in range(2):
                result.append(
                    Row(
                        index=len(result), role="val", family=family,
                        scene_id=f"family-{family_id}-action-{action}-scene-{scene_copy}",
                        rgb=("a", "b", "c", "d", "e", "f"),
                        actions=(0, 1, action, 0, 0, 0),
                    )
                )
    return tuple(result)


def candidate_panel(selected_rows: tuple[Row, ...], margin: float) -> np.ndarray:
    panel = np.full((len(selected_rows), three_arm.ACTION_COUNT), 1.0 + margin)
    for index, row in enumerate(selected_rows):
        panel[index, row.actions[2]] = 1.0
    return panel


@pytest.mark.parametrize(
    ("repair", "retention", "point", "lower", "upper", "status"),
    [
        (True, True, 0.0, -1.0, -1.0, "PASS_EXPLORATORY_ACTION_ALIGNMENT_PROXY_REPAIR"),
        (True, False, 1.0, 1.0, 1.0, "FAIL_RETENTION_CLOSE_ALIGNMENT_BRANCH"),
        (False, True, metric.MEANINGFUL_POINT_THRESHOLD, 1e-9, 1.0, "MEANINGFUL_ALIGNMENT_IMPROVEMENT_INCOMPLETE"),
        (False, True, 0.0, -1.0, metric.STALL_UPPER_THRESHOLD - 1e-9, "STALLED_CLOSE_ALIGNMENT_BRANCH"),
        (False, True, 0.0, -1.0, metric.STALL_UPPER_THRESHOLD, "INCONCLUSIVE_ALIGNMENT_COMPARISON"),
    ],
)
def test_frozen_decision_precedence(repair, retention, point, lower, upper, status):
    observed, _branch, _next = metric.classify_alignment_outcome(
        alignment_repaired=repair,
        retention_passed=retention,
        delta_point=point,
        delta_lower=lower,
        delta_upper=upper,
        repaired_next_step="NEXT",
    )
    assert observed == status


def test_paired_delta_uses_shared_weights_for_constant_scene_effect():
    selected = rows()
    result = metric.paired_minimum_action_margin_delta(
        baseline_candidate_energy=candidate_panel(selected, -0.01),
        treatment_candidate_energy=candidate_panel(selected, 0.01),
        validation_rows=selected,
    )
    assert result["point"] == pytest.approx(0.02, abs=1e-15)
    assert result["one_sided_95_lower_quantile"] == pytest.approx(0.02, abs=1e-15)
    assert result["one_sided_95_upper_quantile"] == pytest.approx(0.02, abs=1e-15)


def test_absolute_proxy_repair_routes_persistence_after_retention():
    selected = rows()
    count = len(selected)
    factual = np.ones(count)
    control = np.full(count, 1.2)
    decision = metric.decide_alignment_successor(
        baseline_candidate_energy=candidate_panel(selected, -0.01),
        baseline_factual_energy=factual,
        baseline_persistence_energy=control,
        baseline_wrong_history_energy=control,
        treatment_candidate_energy=candidate_panel(selected, 0.01),
        treatment_factual_energy=factual,
        treatment_persistence_energy=control,
        treatment_wrong_history_energy=control,
        validation_rows=selected,
        treatment_rank_ratio_by_update={500: 0.3, 600: 0.3, 700: 0.3},
        contract_checks={"exact": True},
        train_fit_checks={"finite": True},
    )
    assert decision["status"] == "PASS_EXPLORATORY_ACTION_ALIGNMENT_PROXY_REPAIR"
    assert decision["latent_proxy_thresholds"]["passed"] is True
    assert decision["selected_next_step"] == "PROCEED_TO_PLANNING_USEFULNESS_GATE"
    assert decision["citable_as_original_factual_learnability_claim"] is False
