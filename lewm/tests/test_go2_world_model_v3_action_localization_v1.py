from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as three_arm
from lewm.benchmarks.go2_world_model_v3_action_localization_v1 import (
    ActionLocalizationError,
    _routing_decision,
    localize_action_and_controls,
)


def _fixture() -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    for family in three_arm.REGISTERED_FAMILIES:
        for scene_number in range(2):
            scene = f"{family}_localization_{scene_number}"
            for action in range(three_arm.ACTION_COUNT):
                rows.append(
                    {
                        "index": len(rows),
                        "role": "val",
                        "family": family,
                        "scene_id": scene,
                        "actions": [0, 1, action, 3, 4, 5],
                    }
                )
    count = len(rows)
    actions = np.asarray([int(row["actions"][2]) for row in rows])
    candidate = np.full((count, three_arm.ACTION_COUNT), 2.0, dtype=np.float64)
    candidate[np.arange(count), actions] = 1.0
    backward_rows = actions == 2
    candidate[backward_rows, 0] = 0.8
    factual = candidate[np.arange(count), actions].copy()
    persistence = np.full(count, 1.5, dtype=np.float64)
    persistence[backward_rows] = 0.9
    wrong_history = np.full(count, 2.0, dtype=np.float64)
    return rows, candidate, factual, persistence, wrong_history


def test_localization_identifies_one_bad_action_and_one_bad_persistence_class() -> None:
    rows, candidate, factual, persistence, wrong_history = _fixture()
    result = localize_action_and_controls(
        candidate_energies=candidate,
        factual_energy=factual,
        persistence_energy=persistence,
        wrong_history_energy=wrong_history,
        validation_rows=rows,
    )

    assert result["status"] == "PASS_READ_ONLY_LOCALIZATION"
    assert result["row_count"] == 144
    assert result["scene_count"] == 16
    assert result["factual_candidate_energy_max_abs_error"] == 0.0
    topology = result["failure_topology"]
    assert topology["registered_hardest_action_id"] == 2
    assert topology["registered_hardest_action_name"] == "backward"
    assert topology["alignment_point_failure_action_ids"] == [2]
    assert topology["alignment_point_failure_scope"] == "localized"
    assert topology["persistence_point_failure_action_ids"] == [2]
    assert topology["persistence_point_failure_scope"] == "localized"
    assert result["routing_decision"]["alignment_route"] == (
        "TEST_GLOBAL_ALIGNMENT_HYPOTHESIS"
    )
    assert result["routing_decision"]["selected_next_step"] == (
        "EXPLICIT_ACTION_ALIGNMENT_OBJECTIVE_VS_MATCHED_BASELINE"
    )

    identification = result["action_identification"]
    assert identification["confusion_matrix"][2][0] == 16
    assert identification["factual_action_counts"] == [16] * 9
    assert identification["scene_family_margin_by_action"][2] == pytest.approx(-0.2)

    margin = result["action_margin_localization"]["per_action"][2]
    assert margin["family_equal_scene_macro_point"] == pytest.approx(-0.2)
    assert margin["one_sided_95_lower_quantile"] == pytest.approx(-0.2)
    assert margin["minimum_supporting_scene_count"] == 2
    persistence_row = result["persistence_localization"]["per_action"][2]
    assert persistence_row["family_equal_scene_macro_point"] == pytest.approx(
        math.log(0.9)
    )
    assert all(
        item["one_sided_95_lower_quantile"] > 0.0
        for item in result["wrong_history_localization"]["per_action"]
    )
    serialized = json.dumps(result, sort_keys=True)
    for family in three_arm.REGISTERED_FAMILIES:
        for scene_number in range(2):
            assert f"{family}_localization_{scene_number}" not in serialized
    assert "conditioned_energy_by_scene" not in serialized
    assert "control_energy_by_scene" not in serialized
    assert "log_advantage_by_scene" not in serialized


def test_localization_is_deterministic_and_rejects_factual_column_mismatch() -> None:
    rows, candidate, factual, persistence, wrong_history = _fixture()
    first = localize_action_and_controls(
        candidate_energies=candidate,
        factual_energy=factual,
        persistence_energy=persistence,
        wrong_history_energy=wrong_history,
        validation_rows=rows,
    )
    second = localize_action_and_controls(
        candidate_energies=candidate,
        factual_energy=factual,
        persistence_energy=persistence,
        wrong_history_energy=wrong_history,
        validation_rows=rows,
    )
    assert first == second

    changed = factual.copy()
    changed[0] += 1.0e-9
    with pytest.raises(ActionLocalizationError, match="factual candidate-energy"):
        localize_action_and_controls(
            candidate_energies=candidate,
            factual_energy=changed,
            persistence_energy=persistence,
            wrong_history_energy=wrong_history,
            validation_rows=rows,
        )


def test_localization_rejects_nonvalidation_rows_and_nonpositive_energy() -> None:
    rows, candidate, factual, persistence, wrong_history = _fixture()
    rows[0] = {**rows[0], "role": "train"}
    with pytest.raises(ActionLocalizationError, match="validation-only"):
        localize_action_and_controls(
            candidate_energies=candidate,
            factual_energy=factual,
            persistence_energy=persistence,
            wrong_history_energy=wrong_history,
            validation_rows=rows,
        )

    rows[0] = {**rows[0], "role": "val"}
    persistence[0] = 0.0
    with pytest.raises(ActionLocalizationError, match="finite positive vector"):
        localize_action_and_controls(
            candidate_energies=candidate,
            factual_energy=factual,
            persistence_energy=persistence,
            wrong_history_energy=wrong_history,
            validation_rows=rows,
        )


def _route_rows(
    *,
    point_failures: tuple[int, ...] = (),
    lower_failures: tuple[int, ...] = (),
) -> list[dict[str, float | int]]:
    return [
        {
            "action_id": action,
            "family_equal_scene_macro_point": (
                -0.1 if action in point_failures else 0.1
            ),
            "one_sided_95_lower_quantile": (
                -0.05 if action in lower_failures else 0.05
            ),
        }
        for action in range(9)
    ]


def test_routing_boundaries_are_finite_and_noncausal() -> None:
    persistence_clear = _route_rows()
    reweight = _routing_decision(
        _route_rows(point_failures=(5,), lower_failures=(5,)),
        persistence_clear,
        alignment_shared_minimum_lower=-0.05,
        aggregate_persistence_lower=0.1,
    )
    assert reweight["alignment_route"] == "TEST_ACTION_REWEIGHTING_HYPOTHESIS"
    assert reweight["train_count_evidence_kind"].startswith("unique_frozen_train_row")

    broad = _routing_decision(
        _route_rows(point_failures=(2,), lower_failures=(2,)),
        persistence_clear,
        alignment_shared_minimum_lower=-0.05,
        aggregate_persistence_lower=0.1,
    )
    assert broad["alignment_route"] == "TEST_GLOBAL_ALIGNMENT_HYPOTHESIS"

    uncertainty = _routing_decision(
        _route_rows(),
        persistence_clear,
        alignment_shared_minimum_lower=-1.0e-6,
        aggregate_persistence_lower=0.1,
    )
    assert uncertainty["alignment_route"] == "UNCERTAINTY_LIMITED"
    assert not uncertainty["alignment_repaired"]

    systemic = _routing_decision(
        _route_rows(),
        _route_rows(lower_failures=(0, 1, 2, 3, 4)),
        alignment_shared_minimum_lower=0.01,
        aggregate_persistence_lower=-0.1,
    )
    assert systemic["alignment_route"] == "ALIGNMENT_PASSED"
    assert systemic["persistence_route"] == "PERSISTENCE_SYSTEMIC"
    assert systemic["selected_next_step"] == (
        "PERSISTENCE_RESIDUAL_VS_MATCHED_BASELINE"
    )

    localized = _routing_decision(
        _route_rows(),
        _route_rows(lower_failures=(0, 1)),
        alignment_shared_minimum_lower=0.01,
        aggregate_persistence_lower=-0.01,
    )
    assert localized["persistence_route"] == (
        "PERSISTENCE_LOCALIZED_OR_AGGREGATE_UNREPAIRED"
    )
    assert localized["selected_next_step"] == (
        "PLANNING_USEFULNESS_GATE_WITH_PROXY_CAVEAT"
    )

    passed = _routing_decision(
        _route_rows(),
        persistence_clear,
        alignment_shared_minimum_lower=0.01,
        aggregate_persistence_lower=0.01,
    )
    assert passed["persistence_route"] == "PERSISTENCE_PASSED"
    assert passed["selected_next_step"] == "PROCEED_TO_PLANNING_USEFULNESS_GATE"


def test_nonuniform_scene_macro_tie_rank_and_pairwise_orientation() -> None:
    rows: list[dict[str, object]] = []
    action_ids: list[int] = []
    scene_numbers: list[int] = []
    within_scene_rows: list[int] = []
    for family in three_arm.REGISTERED_FAMILIES:
        for scene_number, rows_per_action in ((0, 2), (1, 1)):
            for action in range(9):
                for within_scene in range(rows_per_action):
                    rows.append(
                        {
                            "index": len(rows),
                            "role": "val",
                            "family": family,
                            "scene_id": f"{family}_nonuniform_{scene_number}",
                            "actions": [0, 1, action, 3, 4, 5],
                        }
                    )
                    action_ids.append(action)
                    scene_numbers.append(scene_number)
                    within_scene_rows.append(within_scene)
    actions = np.asarray(action_ids)
    candidate = np.full((len(rows), 9), 3.0, dtype=np.float64)
    candidate[np.arange(len(rows)), actions] = 1.0
    for index, action in enumerate(actions):
        if action == 0:
            if scene_numbers[index] == 0:
                candidate[index, 1] = 0.5 if within_scene_rows[index] == 0 else 1.5
            else:
                candidate[index, 1] = 1.4
        elif action == 1:
            candidate[index, 0] = 1.0
    factual = candidate[np.arange(len(rows)), actions]
    result = localize_action_and_controls(
        candidate_energies=candidate,
        factual_energy=factual,
        persistence_energy=np.full(len(rows), 2.0),
        wrong_history_energy=np.full(len(rows), 2.5),
        validation_rows=rows,
    )

    # Action 0's scene points are 0.0 and 0.4, hence scene-equal 0.2;
    # the row-weighted value would instead be 0.1333....
    assert result["action_margin_localization"]["per_action"][0][
        "family_equal_scene_macro_point"
    ] == pytest.approx(0.2)
    matrix = result["pairwise_family_equal_scene_macro_margin_matrix"]["values"]
    assert matrix[0][1] == pytest.approx(0.2)
    assert matrix[1][0] == pytest.approx(0.0)
    action_one = result["action_diagnostics"][1]
    assert action_one["factual_rank_histogram_rank_1_through_9"][1] == 24
    assert action_one["row_weighted_factual_mean_reciprocal_rank"] == pytest.approx(
        0.5
    )
    serialized = json.dumps(result, sort_keys=True)
    assert "_nonuniform_" not in serialized
