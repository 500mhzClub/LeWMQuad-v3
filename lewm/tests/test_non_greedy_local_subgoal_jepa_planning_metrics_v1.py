from __future__ import annotations

import copy

import pytest

from lewm.safety import non_greedy_local_subgoal_jepa_planning_metrics_v1 as M
from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as C


def _score(model_id: str, source_id: str, candidate: int) -> float:
    target = 12.0 - candidate
    if (model_id, source_id) == (
        C.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER,
        "TRUE_FUTURE",
    ):
        return target
    if source_id in ("TRUE_FUTURE_DERANGED_CANDIDATE", "TRUE_FUTURE_DERANGED_TIME"):
        return -target
    return -target


def _row(
    *,
    stage_id: str,
    state_id: str,
    model_id: str,
    source_id: str,
    candidate: int,
    score: float,
    immediate_contact: bool = False,
    committed_prefix_contact: bool = False,
    successor_viable: bool = True,
    dead_end: bool = False,
    completed: bool = False,
) -> dict:
    family = C.FAMILY_IDS[int(state_id.split("-")[1])]
    return {
        "stage_id": stage_id,
        "state_id": state_id,
        "family": family,
        "split_role": "DEVELOPMENT_HELDOUT",
        "candidate_index": candidate,
        "model_id": model_id,
        "source_id": source_id,
        "score": float(score),
        "geodesic_progress_m": float(12 - candidate),
        "remaining_geodesic_m": float(candidate + 1),
        "heading_error_to_next_shortest_segment_rad": float(candidate) / 100.0,
        "euclidean_progress_m": float(candidate - 6) / 10.0,
        "oracle_admissible": not committed_prefix_contact and successor_viable,
        "immediate_contact": immediate_contact,
        "committed_prefix_contact": committed_prefix_contact,
        "successor_viable": successor_viable,
        "stuck": False,
        "dead_end": dead_end,
        "completed": completed,
    }


def _stage_a_rows(*, true_scores_reversed: bool = False) -> list[dict]:
    rows = []
    for state_id in C.DEVELOPMENT_HELDOUT_STATE_IDS:
        for model_id, source_id in C.STAGE_A_MODEL_SOURCE_PAIRS:
            for candidate in range(12):
                score = _score(model_id, source_id, candidate)
                if true_scores_reversed and source_id == "TRUE_FUTURE":
                    score = -score
                rows.append(
                    _row(
                        stage_id=M.STAGE_A,
                        state_id=state_id,
                        model_id=model_id,
                        source_id=source_id,
                        candidate=candidate,
                        score=score,
                    )
                )
    return rows


def _r1_score(candidate: int) -> float:
    # Adjacent swaps preserve an absolute pass while RR has registered gains.
    order = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
    return float(12 - order.index(candidate))


def _stage_b_rows(*, rr_mode: str = "perfect") -> list[dict]:
    rows = []
    for state_id in C.DEVELOPMENT_HELDOUT_STATE_IDS:
        for model_id, source_id in C.STAGE_B_MODEL_SOURCE_PAIRS:
            for candidate in range(12):
                if source_id == "R1":
                    score = _r1_score(candidate)
                elif rr_mode == "perfect":
                    score = float(12 - candidate)
                elif rr_mode == "same":
                    score = _r1_score(candidate)
                else:
                    score = float(candidate)
                rows.append(
                    _row(
                        stage_id=M.CONDITIONAL_STAGE_B,
                        state_id=state_id,
                        model_id=model_id,
                        source_id=source_id,
                        candidate=candidate,
                        score=score,
                    )
                )
    return rows


def test_score_row_authority_exact_pairs_and_cardinalities() -> None:
    authority = M.score_row_authority()
    assert authority["schema"].endswith("score_row_authority.v1")
    stage_a = authority["stages"]["stage_a"]
    stage_b = authority["stages"]["conditional_stage_b"]
    assert len(stage_a["required_fields"]) == 19
    assert stage_a["finite_numeric_fields"] == list(M.FINITE_NUMERIC_FIELDS)
    assert stage_a["candidate_count"] == stage_b["candidate_count"] == 12
    assert len(stage_a["state_ids"]) == len(stage_b["state_ids"]) == 16
    assert stage_a["model_source_pairs"] == [
        {"model_id": model_id, "source_id": source_id}
        for model_id, source_id in C.STAGE_A_MODEL_SOURCE_PAIRS
    ]
    assert stage_b["model_source_pairs"] == [
        {"model_id": model_id, "source_id": source_id}
        for model_id, source_id in C.STAGE_B_MODEL_SOURCE_PAIRS
    ]


def test_target_order_uses_progress_remaining_heading_index_not_completion() -> None:
    state_id = C.DEVELOPMENT_HELDOUT_STATE_IDS[0]
    rows = [
        _row(
            stage_id=M.STAGE_A,
            state_id=state_id,
            model_id=C.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER,
            source_id="TRUE_FUTURE",
            candidate=candidate,
            score=float(12 - candidate),
            completed=candidate == 7,
        )
        for candidate in range(12)
    ]
    # Exact progress tie: candidate 1 wins by lower remaining, then heading.
    rows[1]["geodesic_progress_m"] = rows[0]["geodesic_progress_m"]
    rows[1]["remaining_geodesic_m"] = 0.5
    rows[1]["heading_error_to_next_shortest_segment_rad"] = 0.2
    rows[0]["remaining_geodesic_m"] = 0.5
    rows[0]["heading_error_to_next_shortest_segment_rad"] = 0.3
    value = M.state_metrics(rows)
    assert value["oracle_best_candidate_index"] == 1
    assert value["oracle_best_candidate_index"] != 7


def test_population_reporting_and_all_candidate_adverse_counts() -> None:
    rows = [
        _row(
            stage_id=M.STAGE_A,
            state_id=C.DEVELOPMENT_HELDOUT_STATE_IDS[0],
            model_id=C.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER,
            source_id="TRUE_FUTURE",
            candidate=candidate,
            score=float(12 - candidate),
            immediate_contact=candidate == 0,
            committed_prefix_contact=candidate == 0,
            dead_end=candidate == 0,
        )
        for candidate in range(12)
    ]
    all_candidates = M.state_metrics(rows, population_id="ALL_CANDIDATES")
    contact_free = M.state_metrics(rows, population_id="ORACLE_CONTACT_FREE")
    viable = M.state_metrics(rows, population_id="ORACLE_VIABILITY_ADMISSIBLE")
    assert all_candidates["admissible_candidate_count"] == 12
    assert contact_free["admissible_candidate_count"] == viable["admissible_candidate_count"] == 11
    assert all_candidates["raw_selected_contact"] is True
    assert all_candidates["raw_selected_dead_end"] is True


def test_stage_a_only_reducer_authorizes_but_final_reducer_requires_it() -> None:
    rows = _stage_a_rows()
    provisional = M.recompute_stage_a_metrics_and_gate(rows)
    assert provisional["decision"]["stage_a_authorizes_predictor_substitution"] is True
    assert provisional["stage_a"]["gate"]["classification"] == (
        "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_SIGNAL"
    )
    with pytest.raises(M.NonGreedyMetricsError, match="presence contradicts"):
        M.recompute_metrics_and_gates(rows, None)


def test_stage_a_absolute_failure_precedes_baseline_classification() -> None:
    rows = _stage_a_rows(true_scores_reversed=True)
    value = M.recompute_metrics_and_gates(rows, None)
    assert value["final_classification"] == "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_NO_SIGNAL"
    assert value["decision"]["stage_a_authorizes_predictor_substitution"] is False


def test_full_two_step_signal_and_exact_decision_mapping() -> None:
    value = M.recompute_metrics_and_gates(_stage_a_rows(), _stage_b_rows())
    assert value["final_classification"] == "NON_GREEDY_TWO_STEP_JEPA_PLANNING_SIGNAL"
    assert value["decision"]["next_experiment"] == (
        "ORACLE_ADMISSIBLE_NON_GREEDY_CLOSED_LOOP_JEPA_MPC_V1"
    )
    assert value["conditional_stage_b"]["gate"]["r1_absolute"]["pass"] is True
    assert value["conditional_stage_b"]["gate"]["rr_absolute"]["pass"] is True


def test_stage_b_no_rollout_advantage_when_r1_qualifies() -> None:
    value = M.recompute_metrics_and_gates(_stage_a_rows(), _stage_b_rows(rr_mode="same"))
    assert value["final_classification"] == (
        "NON_GREEDY_JEPA_PLANNING_SIGNAL_NO_ROLLOUT_ADVANTAGE"
    )


def test_row_cube_and_oracle_tamper_fail_closed() -> None:
    rows = _stage_a_rows(true_scores_reversed=True)
    with pytest.raises(M.NonGreedyMetricsError, match="incomplete"):
        M.recompute_metrics_and_gates(rows[:-1], None)
    tampered = copy.deepcopy(rows)
    tampered[0]["oracle_admissible"] = False
    with pytest.raises(M.NonGreedyMetricsError, match="admissibility"):
        M.recompute_metrics_and_gates(tampered, None)
    tampered = copy.deepcopy(rows)
    tampered[0]["score"] = float("nan")
    with pytest.raises(M.NonGreedyMetricsError, match="finite"):
        M.recompute_metrics_and_gates(tampered, None)
    tampered = copy.deepcopy(rows)
    tampered[0]["immediate_contact"] = True
    with pytest.raises(M.NonGreedyMetricsError, match="immediate contact"):
        M.recompute_metrics_and_gates(tampered, None)


def test_family_floor_is_not_merely_nonconstant_scores() -> None:
    rows = _stage_a_rows(true_scores_reversed=True)
    value = M.recompute_metrics_and_gates(rows, None)
    true_key = f"{C.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER}::TRUE_FUTURE"
    true_metrics = value["stage_a"]["sources"][true_key]
    assert true_metrics["aggregate"]["complete_score_collapse"] is False
    assert true_metrics["aggregate"]["no_family_complete_collapse"] is True
    assert true_metrics["aggregate"]["family_performance_floor_pass"] is False
    checks = value["stage_a"]["gate"]["absolute_checks"]
    assert checks["no_family_complete_collapse"] is True
    assert checks["family_performance_floor"] is False


def test_tied_stage_a_scores_fail_null_correlation_gate_without_exception() -> None:
    rows = _stage_a_rows()
    for row in rows:
        if row["source_id"] == "TRUE_FUTURE":
            row["score"] = 0.0
    value = M.recompute_metrics_and_gates(rows, None)
    true_key = f"{C.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER}::TRUE_FUTURE"
    aggregate = value["stage_a"]["sources"][true_key]["aggregate"]
    gate = value["stage_a"]["gate"]
    assert aggregate["spearman"] is None
    assert aggregate["complete_score_collapse"] is True
    assert aggregate["no_family_complete_collapse"] is False
    assert aggregate["family_performance_floor_pass"] is False
    assert gate["absolute_checks"]["spearman"] is False
    assert gate["absolute_checks"]["no_family_complete_collapse"] is False
    assert gate["absolute_checks"]["family_performance_floor"] is False
    assert gate["pass"] is False
    assert value["final_classification"] == "NON_GREEDY_TRUE_FUTURE_JEPA_ROUTE_NO_SIGNAL"


def test_no_admissible_stage_a_population_fails_all_null_checks_without_exception() -> None:
    rows = _stage_a_rows()
    for row in rows:
        row["committed_prefix_contact"] = True
        row["oracle_admissible"] = False
    value = M.recompute_metrics_and_gates(rows, None)
    true_key = f"{C.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER}::TRUE_FUTURE"
    aggregate = value["stage_a"]["sources"][true_key]["aggregate"]
    gate = value["stage_a"]["gate"]
    assert aggregate["states_with_admissible_candidates"] == 0
    assert aggregate["pairwise_accuracy"] is None
    assert aggregate["spearman"] is None
    assert aggregate["normalized_regret"] is None
    assert aggregate["family_performance_floor_pass"] is False
    assert not any(gate["absolute_checks"].values())
    assert gate["incremental_over_kinematic"]["comparison"] == {
        "pairwise_accuracy_gain": None,
        "normalized_regret_reduction": None,
        "oracle_progress_fraction_gain": None,
    }
    assert gate["pass"] is False


@pytest.mark.parametrize("edge", ["tied", "no_admissible"])
def test_stage_b_null_or_collapsed_aggregates_fail_gates_without_exception(
    edge: str,
) -> None:
    stage_b = _stage_b_rows()
    for row in stage_b:
        if edge == "tied":
            row["score"] = 0.0
        else:
            row["committed_prefix_contact"] = True
            row["oracle_admissible"] = False
    value = M.recompute_metrics_and_gates(_stage_a_rows(), stage_b)
    gate = value["conditional_stage_b"]["gate"]
    assert gate["r1_absolute"]["pass"] is False
    assert gate["rr_absolute"]["pass"] is False
    assert gate["r1_absolute"]["checks"]["family_performance_floor"] is False
    assert gate["rr_absolute"]["checks"]["family_performance_floor"] is False
    assert gate["pass"] is False
    assert value["final_classification"] == (
        "NON_GREEDY_TRUE_FUTURE_SIGNAL_PREDICTOR_NO_GO"
    )
    if edge == "no_admissible":
        assert gate["r1_absolute"]["true_selected_progress_fraction"] is None
        assert gate["rr_over_r1"]["comparison"]["pairwise_accuracy_gain"] is None
