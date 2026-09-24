from __future__ import annotations

import copy
import hashlib

import pytest

from lewm.safety import plan_aware_monotone_jepa_cost_metrics_v1 as M


def _candidate(
    index: int,
    *,
    family: str,
    completed: bool,
    progress: float,
    heading: float,
    contact: bool = False,
    successor_viable: bool = True,
    stuck: bool = False,
) -> dict:
    return {
        "candidate_index": index,
        "family": family,
        "role": "heldout",
        "completed": completed,
        "p_d": progress,
        "p_theta": heading,
        "immediate_contact_h1": contact,
        "successor_viable": successor_viable,
        "oracle_viability_admissible": (not contact) and successor_viable,
        "stuck": stuck,
    }


def _panel():
    candidates = {}
    scores = {}
    for state_index, family in enumerate(M.FAMILY_IDS):
        state = f"state-{state_index}"
        candidates[state] = [
            _candidate(
                0,
                family=family,
                completed=True,
                progress=1.0,
                heading=0.30,
            ),
            _candidate(
                1,
                family=family,
                completed=False,
                progress=0.60,
                heading=0.20,
            ),
            _candidate(
                2,
                family=family,
                completed=False,
                progress=0.30,
                heading=0.10,
                contact=True,
            ),
            _candidate(
                3,
                family=family,
                completed=False,
                progress=0.10,
                heading=0.00,
                successor_viable=False,
                stuck=True,
            ),
        ]
        scores[state] = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
    return candidates, scores


def _aggregate(
    *,
    pairwise: float,
    progress_ratio: float,
    regret: float,
    top3: float,
    selected_sum: float,
    best_sum: float = 4.0,
    contacts: int = 0,
    nonviable: int = 0,
    stuck: int = 0,
    states: int = 4,
) -> dict:
    return {
        "states": states,
        "pairwise_accuracy": pairwise,
        "spearman_rho": pairwise,
        "kendall_tau_b": pairwise - 0.05,
        "normalized_regret": regret,
        "best_route_top3_rate": top3,
        "selected_route_progress_m_sum": selected_sum,
        "selected_route_progress_m_mean": selected_sum / states,
        "oracle_best_route_progress_m_sum": best_sum,
        "selected_progress_ratio": progress_ratio,
        "selected_immediate_contacts_h1": contacts,
        "selected_nonviable_successors": nonviable,
        "selected_stuck": stuck,
        "complete_family_collapse": False,
    }


def _source(
    source_id: str,
    *,
    pairwise: float = 0.80,
    progress_ratio: float = 0.90,
    regret: float = 0.10,
    top3: float = 0.80,
    selected_sum: float | None = None,
    contacts: int = 0,
    nonviable: int = 0,
    stuck: int = 0,
) -> dict:
    selected = 4.0 * progress_ratio if selected_sum is None else selected_sum
    populations = {}
    for population in M.POPULATION_IDS:
        aggregate = _aggregate(
            pairwise=pairwise,
            progress_ratio=progress_ratio,
            regret=regret,
            top3=top3,
            selected_sum=selected,
            contacts=contacts if population == M.ALL_CANDIDATES else 0,
            nonviable=nonviable if population == M.ALL_CANDIDATES else 0,
            stuck=stuck if population == M.ALL_CANDIDATES else 0,
        )
        per_family = {
            family: _aggregate(
                pairwise=pairwise,
                progress_ratio=progress_ratio,
                regret=regret,
                top3=top3,
                selected_sum=selected / 4.0,
                best_sum=1.0,
                states=1,
            )
            for family in M.FAMILY_IDS
        }
        per_state = [
            {
                "state_id": f"state-{index}",
                "family": family,
                "pairwise_accuracy": pairwise,
                "normalized_regret": regret,
                "best_route_top3": top3 >= 0.75,
                "selected_route_progress_m": selected / 4.0,
                "oracle_best_route_progress_m": 1.0,
            }
            for index, family in enumerate(M.FAMILY_IDS)
        ]
        populations[population] = {
            "aggregate": aggregate,
            "per_family": per_family,
            "per_state": per_state,
            "no_family_complete_collapse": True,
            "collapsed_families": [],
        }
    return {"source_id": source_id, "populations": populations}


def test_route_preference_reuses_completion_distance_heading_and_not_safety() -> None:
    left = _candidate(
        0,
        family=M.FAMILY_IDS[0],
        completed=True,
        progress=-1.0,
        heading=-1.0,
        contact=True,
        successor_viable=False,
    )
    right = _candidate(
        1,
        family=M.FAMILY_IDS[0],
        completed=False,
        progress=2.0,
        heading=2.0,
    )
    assert M.route_only_preference(left, right) == 1

    left["completed"] = right["completed"] = False
    left["p_d"], right["p_d"] = 1.0, 0.90
    assert M.route_only_preference(left, right) == 1
    left["p_d"], right["p_d"] = 1.0, 0.98
    left["p_theta"], right["p_theta"] = 0.0, 0.20
    assert M.route_only_preference(left, right) == -1

    # Safety fields never change the route order.
    factual = M.route_only_preference(left, right)
    left["immediate_contact_h1"] = not left["immediate_contact_h1"]
    left["successor_viable"] = not left["successor_viable"]
    left["stuck"] = not left["stuck"]
    assert M.route_only_preference(left, right) == factual


def test_higher_is_better_metrics_and_three_population_summary() -> None:
    candidates, scores = _panel()
    summary = M.summarize_scores(candidates, scores, "TRUE_FUTURE")
    assert summary["score_direction"] == "higher_is_better"
    assert set(summary["populations"]) == set(M.POPULATION_IDS)
    for population in M.POPULATION_IDS:
        result = summary["populations"][population]
        assert result["aggregate"]["pairwise_accuracy"] == 1.0
        assert result["aggregate"]["best_route_top3_rate"] == 1.0
        assert result["aggregate"]["selected_progress_ratio"] == 1.0
        assert result["aggregate"]["selected_combined_route_utility_mean"] == 1.0
        assert result["aggregate"]["selected_combined_route_utility_sum"] == 4.0
        assert result["aggregate"]["selected_combined_route_utility_count"] == 4
        assert all(
            row["selected_combined_route_utility"] == 1.0
            for row in result["per_state"]
        )
        assert result["no_family_complete_collapse"] is True
        assert len(result["per_state"]) == 4
        assert set(result["per_family"]) == set(M.FAMILY_IDS)
    assert (
        summary["populations"][M.ALL_CANDIDATES]["aggregate"][
            "selected_immediate_contacts_h1"
        ]
        == 0
    )
    downranking = summary["descriptive_adverse_downranking"]
    assert downranking["descriptive_only"] is True
    assert downranking["used_as_score_or_route_target"] is False
    assert downranking["used_as_classification_gate"] is False
    assert downranking["outcomes"]["immediate_contact"]["overall"] == {
        "pair_count": 12,
        "correct_credit": 8.0,
        "pairwise_accuracy": 2.0 / 3.0,
    }
    assert all(
        row == {
            "pair_count": 3,
            "correct_credit": 2.0,
            "pairwise_accuracy": 2.0 / 3.0,
        }
        for row in downranking["outcomes"]["immediate_contact"][
            "per_family"
        ].values()
    )
    for outcome in ("successor_nonviable", "stuck"):
        assert downranking["outcomes"][outcome]["overall"] == {
            "pair_count": 12,
            "correct_credit": 12.0,
            "pairwise_accuracy": 1.0,
        }


def test_selected_utility_mean_uses_only_nonabstaining_defined_states() -> None:
    candidates, scores = _panel()
    first_state = next(iter(candidates))
    for row in candidates[first_state]:
        row["immediate_contact_h1"] = True
        row["oracle_viability_admissible"] = False
    aggregate = M.summarize_scores(candidates, scores, "SYNTHETIC")["populations"][
        M.ORACLE_CONTACT_FREE
    ]["aggregate"]
    assert aggregate["states"] == 4
    assert aggregate["abstentions"] == 1
    assert aggregate["selected_combined_route_utility_count"] == 3
    assert aggregate["selected_combined_route_utility_sum"] == 3.0
    assert aggregate["selected_combined_route_utility_mean"] == 1.0


def test_adverse_downranking_uses_only_within_state_cross_group_pairs_and_half_ties() -> None:
    family = M.FAMILY_IDS[0]
    candidates = {
        "state-0": [
            _candidate(
                0,
                family=family,
                completed=False,
                progress=0.0,
                heading=0.0,
                contact=True,
            ),
            _candidate(
                1,
                family=family,
                completed=False,
                progress=0.0,
                heading=0.0,
            ),
            _candidate(
                2,
                family=family,
                completed=False,
                progress=0.0,
                heading=0.0,
            ),
        ]
    }
    metrics = M.adverse_cross_group_downranking_metrics(
        candidates,
        {"state-0": {0: 1.0, 1: 1.0, 2: 0.0}},
        expected_families=[family],
    )
    contact = metrics["outcomes"]["immediate_contact"]["overall"]
    assert contact == {
        "pair_count": 2,
        "correct_credit": 0.5,
        "pairwise_accuracy": 0.25,
    }
    assert metrics["outcomes"]["successor_nonviable"]["overall"][
        "pairwise_accuracy"
    ] is None


def test_constant_scores_are_complete_family_collapse_despite_tie_broken_selection() -> None:
    candidates, _scores = _panel()
    tied_scores = {
        state_id: {row["candidate_index"]: 0.0 for row in rows}
        for state_id, rows in candidates.items()
    }
    summary = M.summarize_scores(candidates, tied_scores, "TIED")
    for population in M.POPULATION_IDS:
        population_summary = summary["populations"][population]
        assert population_summary["no_family_complete_collapse"] is False
        assert set(population_summary["collapsed_families"]) == set(M.FAMILY_IDS)
        for family in M.FAMILY_IDS:
            aggregate = population_summary["per_family"][family]
            assert aggregate["all_comparable_score_pairs_tied"] is True
            assert aggregate[
                "all_nonabstaining_score_spreads_within_tolerance"
            ] is True
            assert aggregate["complete_family_collapse"] is True


def test_constant_true_scores_fail_gate_and_classify_no_signal_without_exception() -> None:
    candidates, _ = _panel()
    constant = {
        state_id: {row["candidate_index"]: 0.0 for row in rows}
        for state_id, rows in candidates.items()
    }
    true = M.summarize_scores(candidates, constant, "TRUE_CONSTANT")
    population = true["populations"][M.ORACLE_VIABILITY_ADMISSIBLE]
    assert population["aggregate"]["spearman_rho"] is None
    assert population["no_family_complete_collapse"] is False
    derangement = M.evaluate_future_derangement_materiality(true, true)
    assert derangement["pass"] is False
    gate = M.evaluate_true_future_gate(true, derangement)
    assert gate["observed_metrics"]["spearman_rho"] is None
    assert gate["criteria"]["spearman_rho"] is False
    assert gate["criteria"]["no_family_complete_collapse"] is False
    assert gate["pass"] is False
    primary = M.classify_primary(
        true_gate=gate,
        predicted_gate={"pass": False},
        true_incremental_gate={"pass": False},
        all_predicted_substitutions_fail_materially=True,
    )
    assert primary["classification"] == "PLAN_AWARE_JEPA_COST_NO_SIGNAL"


def test_unavailable_metrics_fail_stage_b_and_attribution_gates_without_abort() -> None:
    def nullable(source_id: str) -> dict:
        source = _source(source_id)
        for population in source["populations"].values():
            aggregate = population["aggregate"]
            for key in (
                "pairwise_accuracy",
                "spearman_rho",
                "kendall_tau_b",
                "normalized_regret",
                "best_route_top3_rate",
            ):
                aggregate[key] = None
        return source

    true = nullable("TRUE")
    r1 = nullable("R1")
    rr = nullable("RR")
    p1 = nullable("P1")
    pr = nullable("PR")
    predicted = M.evaluate_predicted_gate(
        true_gate={"pass": True},
        true_source=true,
        one_step_source=r1,
        rollout_source=rr,
    )
    assert predicted["observed_metrics"]["rollout_pairwise_accuracy"] is None
    assert predicted["criteria"]["rollout_pairwise_accuracy"] is False
    assert predicted["pass"] is False
    assert predicted["classification"] == "TWO_STEP_PLAN_AWARE_JEPA_COST_NO_SIGNAL"

    incremental = M.evaluate_incremental_value(
        true, kinematic_source=nullable("KIN"), no_latent_source=nullable("BASE")
    )
    assert incremental["pass"] is False
    assert all(
        comparison["principal_deltas"]["spearman_gain"] is None
        for comparison in incremental["comparisons"].values()
    )

    proprio = M.evaluate_proprio_gate(
        rgb_one_step=r1,
        rgb_rollout=rr,
        proprio_one_step=p1,
        proprio_rollout=pr,
    )
    assert proprio["PR_minus_RR"]["pairwise_accuracy_gain"] is None
    assert proprio["factorial_contrasts"]["J_BP_minus_BR"][
        "spearman_gain"
    ] is None
    assert proprio["pass"] is False
    assert proprio["classification"] == (
        "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED"
    )

    substitution = M.evaluate_substitution_gate(
        proprio_contribution_gate=proprio,
        matched_proprio_rollout=pr,
        visual_deranged=nullable("VISUAL_D"),
        proprio_deranged=nullable("PROPRIO_D"),
        control_deranged=nullable("CONTROL_D"),
        candidate_action_sensitivity_evidence={
            "authority": "synthetic",
            "raw_evidence": [],
            "passed": True,
        },
    )
    assert substitution["damage"]["visual"]["pairwise_accuracy_loss"] is None
    assert substitution["pass"] is False
    assert substitution["classification"] == (
        "PROPRIOCEPTIVE_SUBSTITUTION_NOT_SUPPORTED"
    )


def test_score_ties_receive_half_pairwise_credit_and_candidate_tie_break() -> None:
    family = M.FAMILY_IDS[0]
    rows = [
        _candidate(2, family=family, completed=False, progress=1.0, heading=0.0),
        _candidate(1, family=family, completed=False, progress=0.0, heading=0.0),
    ]
    metrics = M.route_ordering_metrics(rows, [0.5, 0.5])
    assert metrics["pairwise_accuracy"] == 0.5
    assert metrics["score_tie_rate"] == 1.0
    assert metrics["selected_candidate_index"] == 1


def test_ordering_metrics_use_population_borda_and_tie_aware_best_set() -> None:
    family = M.FAMILY_IDS[0]
    # The frozen direct comparator cycles: 0>1 on heading, 1>2 on heading,
    # and 2>0 on distance.  Population-conditioned Borda therefore ties all
    # three at 0.5; direct pair outcomes must not leak into evaluation targets.
    rows = [
        _candidate(0, family=family, completed=False, progress=0.00, heading=0.4),
        _candidate(1, family=family, completed=False, progress=0.02, heading=0.2),
        _candidate(2, family=family, completed=False, progress=0.04, heading=0.0),
    ]
    assert M.route_only_preference(rows[0], rows[1]) == 1
    assert M.route_only_preference(rows[1], rows[2]) == 1
    assert M.route_only_preference(rows[2], rows[0]) == 1
    scores = [3.0, 2.0, 1.0]
    ordering = M.route_ordering_metrics(rows, scores)
    assert ordering["route_borda_utility"] == {"0": 0.5, "1": 0.5, "2": 0.5}
    assert ordering["ordered_pairs"] == 0
    assert ordering["pairwise_accuracy"] is None
    assert ordering["oracle_best_candidate_indices"] == [0, 1, 2]
    assert ordering["oracle_best_candidate_index"] == 0
    assert ordering["ideal_route_borda_order"] == [0, 1, 2]
    # Any member of the tied best set receives top-k/rank credit.
    assert ordering["best_route_top1"] is True
    assert ordering["best_route_top3"] is True
    assert ordering["mrr"] == 1.0
    assert ordering["mean_best_route_rank"] == 1.0

    state = M.evaluate_state_population(
        rows,
        scores,
        state_id="state-cycle",
        family=family,
        role="heldout",
        source_id="SYNTHETIC",
        population_id=M.ALL_CANDIDATES,
    )
    # Borda top-1 and max-progress regret are intentionally distinct.
    assert state["best_route_top1"] is True
    assert state["selected_route_progress_m"] == 0.0
    assert state["oracle_best_route_progress_m"] == 0.04
    assert state["normalized_regret"] == 1.0


def test_future_derangement_is_deterministic_bijective_and_moves_h1_h3_together() -> None:
    digest = hashlib.sha256(b"contract").hexdigest()
    trajectories = {index: [f"{index}-H1", f"{index}-H2", f"{index}-H3"] for index in range(12)}
    first, receipt = M.derange_future_trajectories(
        trajectories, contract_digest=digest, state_id="state-7"
    )
    second, repeated = M.derange_future_trajectories(
        dict(reversed(list(trajectories.items()))),
        contract_digest=digest,
        state_id="state-7",
    )
    assert first == second
    assert receipt["destination_to_donor"] == repeated["destination_to_donor"]
    assert receipt["bijection"] is True
    assert receipt["fixed_point_count"] == 0
    for destination, donor in receipt["destination_to_donor"].items():
        assert first[int(destination)] == tuple(trajectories[int(donor)])
    with pytest.raises(M.PlanAwareMetricsError, match="H1--H3"):
        M.derange_future_trajectories(
            {0: [1, 2], 1: [3, 4]}, contract_digest=digest, state_id="state"
        )


def test_derangement_and_true_gate_thresholds_are_conjunctive() -> None:
    matched = _source(
        "TRUE", pairwise=0.80, progress_ratio=0.85, regret=0.15, top3=0.80
    )
    deranged = _source(
        "DERANGED", pairwise=0.74, progress_ratio=0.80, regret=0.16, top3=0.70
    )
    material = M.evaluate_future_derangement_materiality(matched, deranged)
    assert material["pass"] is True
    assert material["damage"]["pairwise_accuracy_loss"] == pytest.approx(0.06)
    assert material["damage"]["selected_progress_m_loss"] == pytest.approx(0.05)
    assert material["damage"]["selected_progress_ratio_loss"] == pytest.approx(0.05)
    assert material["damage"]["normalized_regret_worsening"] == pytest.approx(0.01)
    assert material["damage"]["best_route_top3_loss"] == pytest.approx(0.10)
    assert set(material["triggers"]) == {
        "pairwise_accuracy_loss",
        "selected_progress_ratio_loss",
        "normalized_regret_worsening",
    }
    gate = M.evaluate_true_future_gate(matched, material)
    assert gate["pass"] is True

    submaterial_mixed_direction = _source(
        "DERANGED",
        pairwise=0.74,
        progress_ratio=0.949,  # improves by 0.099, below the material threshold
        regret=0.16,
        top3=0.70,
    )
    allowed = M.evaluate_future_derangement_materiality(
        matched, submaterial_mixed_direction
    )
    assert allowed["damage"]["no_material_principal_reversal"] is True
    assert allowed["pass"] is True

    material_reversal = _source(
        "DERANGED",
        pairwise=0.74,
        progress_ratio=0.95,  # improves by 0.10: material opposing reversal
        regret=0.16,
        top3=0.70,
    )
    blocked = M.evaluate_future_derangement_materiality(matched, material_reversal)
    assert blocked["damage"]["no_material_principal_reversal"] is False
    assert blocked["damage"][
        "selected_progress_ratio_material_opposing_reversal"
    ] is True
    assert blocked["pass"] is False
    assert M.evaluate_true_future_gate(matched, blocked)["pass"] is False


def test_incremental_value_must_pass_against_both_baselines() -> None:
    true = _source("TRUE", progress_ratio=0.92, regret=0.10, selected_sum=3.68)
    kinematic = _source("KIN", progress_ratio=0.87, regret=0.14, selected_sum=3.48)
    no_latent = _source("NO_LATENT", progress_ratio=0.86, regret=0.15, selected_sum=3.44)
    gate = M.evaluate_incremental_value(
        true, kinematic_source=kinematic, no_latent_source=no_latent
    )
    assert gate["pass"] is True
    assert gate["classification"] == "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE"

    too_close = _source("NO_LATENT", progress_ratio=0.915, regret=0.12, selected_sum=3.66)
    failed = M.evaluate_incremental_value(
        true, kinematic_source=kinematic, no_latent_source=too_close
    )
    assert failed["pass"] is False


def test_predicted_gate_requires_absolute_signal_retention_improvement_and_adverse_parity() -> None:
    true = _source("TRUE", progress_ratio=1.0, regret=0.05, selected_sum=4.0)
    r1 = _source(
        "R1", pairwise=0.70, progress_ratio=0.84, regret=0.23, top3=0.75
    )
    rr = _source(
        "RR", pairwise=0.72, progress_ratio=0.90, regret=0.20, top3=0.75
    )
    gate = M.evaluate_predicted_gate(
        true_gate={"pass": True},
        true_source=true,
        one_step_source=r1,
        rollout_source=rr,
    )
    assert gate["pass"] is True
    assert gate["classification"] == "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL"

    adverse = _source(
        "RR",
        pairwise=0.72,
        progress_ratio=0.90,
        regret=0.20,
        top3=0.75,
        contacts=1,
    )
    assert M.evaluate_predicted_gate(
        true_gate={"pass": True},
        true_source=true,
        one_step_source=r1,
        rollout_source=adverse,
    )["pass"] is False

    stuck_adverse = _source(
        "RR",
        pairwise=0.72,
        progress_ratio=0.90,
        regret=0.20,
        top3=0.75,
        stuck=1,
    )
    assert M.evaluate_predicted_gate(
        true_gate={"pass": True},
        true_source=true,
        one_step_source=r1,
        rollout_source=stuck_adverse,
    )["pass"] is False


def test_proprio_gate_and_factorial_contrasts() -> None:
    r1 = _source("R1", pairwise=0.68, progress_ratio=0.82, regret=0.24, top3=0.70)
    rr = _source("RR", pairwise=0.72, progress_ratio=0.90, regret=0.20, top3=0.75)
    p1 = _source("P1", pairwise=0.69, progress_ratio=0.84, regret=0.23, top3=0.70)
    pr = _source("PR", pairwise=0.78, progress_ratio=0.96, regret=0.18, top3=0.80)
    result = M.evaluate_proprio_gate(
        rgb_one_step=r1,
        rgb_rollout=rr,
        proprio_one_step=p1,
        proprio_rollout=pr,
    )
    assert result["pass"] is True
    assert result["classification"] == "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION"
    assert result["trigger_count"] == 2
    contrasts = result["factorial_contrasts"]
    br = contrasts["BR_rgb_rollout_minus_rgb_one_step"]
    bp = contrasts["BP_proprio_rollout_minus_proprio_one_step"]
    interaction = contrasts["J_BP_minus_BR"]
    assert interaction["pairwise_accuracy_gain"] == pytest.approx(
        bp["pairwise_accuracy_gain"] - br["pairwise_accuracy_gain"]
    )


def test_substitution_tendency_requires_proprio_specific_damage_and_action_authority() -> None:
    matched = _source("PR", pairwise=0.80, progress_ratio=0.95, regret=0.15)
    visual = _source("VISUAL_D", pairwise=0.79, progress_ratio=0.93, regret=0.16)
    proprio = _source("PROPRIO_D", pairwise=0.65, progress_ratio=0.78, regret=0.27)
    control = _source("CONTROL_D", pairwise=0.76, progress_ratio=0.90, regret=0.18)
    result = M.evaluate_substitution_gate(
        proprio_contribution_gate={"pass": True},
        matched_proprio_rollout=matched,
        visual_deranged=visual,
        proprio_deranged=proprio,
        control_deranged=control,
        candidate_action_sensitivity_evidence={
            "passed": True,
            "authority": "frozen-prior-diagnostic",
            "raw_evidence": {"receipt": "sha256:abc"},
        },
    )
    assert result["pass"] is True
    assert result["classification"] == "PROPRIOCEPTIVE_SUBSTITUTION_TENDENCY"
    assert result["criteria"]["not_explained_by_control_derangement_alone"] is True

    with pytest.raises(M.PlanAwareMetricsError, match="authority"):
        M.evaluate_substitution_gate(
            proprio_contribution_gate={"pass": True},
            matched_proprio_rollout=matched,
            visual_deranged=visual,
            proprio_deranged=proprio,
            control_deranged=control,
            candidate_action_sensitivity_evidence={
                "passed": True,
                "raw_evidence": {},
            },
        )


def test_substitution_attribution_distinguishes_visual_and_multimodal_dependence() -> None:
    matched = _source("PR", pairwise=0.85, progress_ratio=0.98, regret=0.10)
    proprio = _source("PD", pairwise=0.70, progress_ratio=0.80, regret=0.23)
    control = _source("CD", pairwise=0.80, progress_ratio=0.93, regret=0.14)
    action = {"passed": True, "authority": "frozen", "raw_evidence": {}}

    multimodal = M.evaluate_substitution_gate(
        proprio_contribution_gate={"pass": True},
        matched_proprio_rollout=matched,
        visual_deranged=copy.deepcopy(proprio),
        proprio_deranged=proprio,
        control_deranged=control,
        candidate_action_sensitivity_evidence=action,
    )
    assert multimodal["pass"] is False
    assert multimodal["dependence_attribution"] == "MULTIMODAL_ROUTE_DEPENDENCE"

    visual_worse = _source("VD", pairwise=0.60, progress_ratio=0.65, regret=0.35)
    visual = M.evaluate_substitution_gate(
        proprio_contribution_gate={"pass": True},
        matched_proprio_rollout=matched,
        visual_deranged=visual_worse,
        proprio_deranged=proprio,
        control_deranged=control,
        candidate_action_sensitivity_evidence=action,
    )
    assert visual["dependence_attribution"] == "VISUAL_ROUTE_DEPENDENCE"


def test_paired_bootstrap_is_deterministic_and_preserves_positive_direction() -> None:
    candidate = _source("A", pairwise=0.80, progress_ratio=0.90, regret=0.10)
    comparator = _source("B", pairwise=0.70, progress_ratio=0.80, regret=0.20)
    first = M.paired_principal_bootstrap(candidate, comparator, draws=64, seed=17)
    second = M.paired_principal_bootstrap(candidate, comparator, draws=64, seed=17)
    assert first == second
    assert first["metrics"]["selected_progress_gain_m"]["point"] > 0.0
    assert first["metrics"]["normalized_regret_reduction"]["point"] > 0.0


@pytest.mark.parametrize(
    ("true", "incremental", "predicted", "all_fail", "expected"),
    [
        (False, False, False, True, "PLAN_AWARE_JEPA_COST_NO_SIGNAL"),
        (True, False, False, True, "KINEMATIC_BASELINE_DOMINANT"),
        (True, True, True, False, "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL"),
        (
            True,
            True,
            False,
            True,
            "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO",
        ),
    ],
)
def test_primary_classification_precedence(
    true: bool, incremental: bool, predicted: bool, all_fail: bool, expected: str
) -> None:
    result = M.classify_primary(
        true_gate={"pass": true},
        predicted_gate={"pass": predicted},
        true_incremental_gate={"pass": incremental},
        all_predicted_substitutions_fail_materially=all_fail,
    )
    assert result["classification"] == expected


def test_primary_classification_fails_closed_when_predicted_status_is_unresolved() -> None:
    with pytest.raises(M.PlanAwareMetricsError, match="unresolved"):
        M.classify_primary(
            true_gate={"pass": True},
            predicted_gate={"pass": False},
            true_incremental_gate={"pass": True},
            all_predicted_substitutions_fail_materially=False,
        )
