from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess

import pytest
import torch
import torch.nn as nn

from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as C


def _eligible_rows() -> list[dict[str, str]]:
    return [
        {
            "family": family,
            "scene_id": f"{family.lower()}-{ordinal:03d}",
            "scene_manifest_sha256": hashlib.sha256(
                f"{family}:{ordinal}".encode()
            ).hexdigest(),
        }
        for family in C.FAMILY_IDS
        for ordinal in range(27)
    ]


def test_exact_panel_bank_and_authority_constants() -> None:
    assert C.STATE_COUNT == 96
    assert C.FAMILY_IDS == ("WALL_DETOUR", "U_ESCAPE", "DEAD_END_LURE", "OFFSET_PASSAGE")
    assert C.SPLIT_STATE_COUNTS == {"FIT": 64, "CALIBRATION": 16, "DEVELOPMENT_HELDOUT": 16}
    assert [name for name, _ in C.CANDIDATE_BANK] == [
        "straight_fast", "straight_medium", "straight_slow", "arc_left", "arc_right",
        "turn_left", "turn_right", "turn_left_then_go", "turn_right_then_go",
        "go_then_turn_left", "reverse_then_turn", "hold",
    ]
    assert C.HORIZON_IDS == ("H1", "H2", "H3")
    assert C.PANEL_SELECTION_SEED == 2026083100
    assert C.BASE_FEATURE_DIM == 131
    assert C.QUERY_FEATURE_DIM == 66
    assert C.REGISTERED_PARAMETER_COUNT == 204289
    assert C.PANEL_CANDIDATE_BLOCK_SIZE == 48
    assert C.PANEL_MAXIMUM_CANDIDATES_PER_FAMILY == 1536


def test_panel_continuation_geometry_and_descriptive_outcomes_are_frozen() -> None:
    contract = C.build_contract()
    assert contract["panel"]["candidate_block_size"] == 48
    assert contract["panel"]["maximum_candidates_per_family"] == 1536
    assert contract["panel"]["continuation_rule"] == C.PANEL_CONTINUATION_RULE
    assert "complete eligible population" in C.PANEL_CONTINUATION_RULE
    assert contract["panel_geometry"] == C.PANEL_GEOMETRY_AUTHORITY
    assert C.PANEL_GEOMETRY_AUTHORITY == {
        "world_half_extent_m": 3.0,
        "robot_radius_m": 0.22,
        "occupancy_grid_resolution_m": 0.05,
        "occupancy_geodesic": (
            "exact_8_neighbor_with_no_diagonal_corner_cutting_and_"
            "edge_cost_aware_shortest_path_descent"
        ),
        "command_tick_seconds": 0.1,
        "physics_step_seconds": 0.002,
        "horizon_seconds": {"H1": 0.5, "H2": 1.0, "H3": 1.5},
        "slew_limits_per_command_tick": {
            "delta_vx_max_mps": 0.25,
            "delta_vy_max_mps": 0.0,
            "delta_yaw_rate_max_radps": 0.35,
        },
        "renderer": {
            "obstacles_visible": True,
            "width_pixels": 224,
            "height_pixels": 168,
            "horizontal_fov_degrees": 92.0,
            "goal_marker_visible": False,
            "wall_material": "uniform_neutral_geometry_only",
        },
    }
    assert contract["nonvisual_counterbalance"] == (
        C.NONVISUAL_COUNTERBALANCE_AUTHORITY
    )
    assert C.NONVISUAL_COUNTERBALANCE_AUTHORITY[
        "required_unique_base_feature_signatures"
    ] == 1
    assert contract["descriptive_outcomes"] == C.DESCRIPTIVE_OUTCOME_AUTHORITY
    assert C.DESCRIPTIVE_OUTCOME_AUTHORITY["completion"]["completion_radius_m"] == 0.25
    assert C.DESCRIPTIVE_OUTCOME_AUTHORITY["stuck"] == {
        "formula": (
            "endpoint displacement < displacement_threshold_m and "
            "(absolute applied vx > applied_abs_vx_threshold_mps or absolute applied "
            "yaw rate > applied_abs_yaw_rate_threshold_radps)"
        ),
        "displacement_threshold_m": 0.015,
        "applied_abs_vx_threshold_mps": 0.05,
        "applied_abs_yaw_rate_threshold_radps": 0.1,
    }
    assert C.DESCRIPTIVE_OUTCOME_AUTHORITY["dead_end"] == {
        "formula": (
            "endpoint geodesic distance >= start geodesic distance - "
            "geodesic_no_progress_tolerance_m while start Euclidean distance - "
            "endpoint Euclidean distance > euclidean_closer_threshold_m"
        ),
        "geodesic_no_progress_tolerance_m": 1.0e-9,
        "euclidean_closer_threshold_m": 1.0e-6,
    }


def test_family_performance_floor_is_explicit_contract_authority() -> None:
    assert C.FAMILY_PERFORMANCE_FLOOR == {
        "population": "ORACLE_VIABILITY_ADMISSIBLE",
        "pairwise_accuracy_minimum_each_family": 0.50,
        "oracle_progress_fraction_minimum_each_family": 0.50,
        "score_not_completely_collapsed_each_family": True,
        "formula": (
            "every one of the four families has defined pairwise accuracy >= 0.50, "
            "defined oracle-progress fraction >= 0.50, and at least one state with "
            "non-tied admissible scores"
        ),
    }
    contract = C.build_contract()
    assert contract["gates"]["family_performance_floor"] == C.FAMILY_PERFORMANCE_FLOOR
    assert contract["gates"]["stage_a"]["absolute"]["family_performance_floor"] is True
    assert contract["gates"]["stage_b"]["rr_absolute"]["family_performance_floor"] is True


def test_recovery_five_and_claim_boundary_are_exact() -> None:
    assert len(C.RECOVERY_FILE_BINDINGS) == 5
    assert C.RECOVERY_FILE_BINDINGS["recovered_development_result.json"]["sha256"] == (
        "510661e5f7da88db7a7c17f06e328cfeb9c9e41e596ea761bf0d10642d0c0b1a"
    )
    assert C.RECOVERY_DECISION_AUTHORITY["artifacts_reused_for_new_science"] == 0
    assert C.CLAIM_AUTHORITY["positive_wording"] == "JEPA route selection under oracle admissibility."
    assert C.CLAIM_AUTHORITY["prohibited_wording"] == "JEPA safety."
    assert C.CLAIM_AUTHORITY["deployment_safety_claim"] is False


def test_deterministic_split_uses_complete_population_then_16_4_4() -> None:
    rows = _eligible_rows()
    first = C.deterministic_split_manifest(rows)
    second = C.deterministic_split_manifest(list(reversed(rows)))
    assert first == second
    assert first["role_counts"] == C.SPLIT_STATE_COUNTS
    assert len(first["states"]) == 96
    for family in C.FAMILY_IDS:
        family_rows = [row for row in first["states"] if row["family"] == family]
        assert [row["split_role"] for row in family_rows] == ["FIT"] * 16 + [
            "CALIBRATION"
        ] * 4 + ["DEVELOPMENT_HELDOUT"] * 4
        assert {row["family_population_count"] for row in family_rows} == {27}


def test_split_rejects_duplicate_or_incomplete_population() -> None:
    rows = _eligible_rows()
    with pytest.raises(C.NonGreedyContractError, match="duplicate"):
        C.deterministic_split_manifest([*rows, copy.deepcopy(rows[0])])
    with pytest.raises(C.NonGreedyContractError, match="fewer"):
        C.deterministic_split_manifest(rows[:-4])


def test_contract_round_trip_and_tamper_rejection() -> None:
    contract = C.build_contract()
    assert C.validate_contract(contract) == contract
    assert C.contract_bytes() == C.canonical_json_bytes(contract)
    tampered = copy.deepcopy(contract)
    tampered["panel"]["state_count"] = 95
    with pytest.raises(C.NonGreedyContractError):
        C.validate_contract(tampered)


def test_import_constructs_no_model_instance() -> None:
    assert not any(isinstance(value, nn.Module) for value in vars(C).values())


def test_pure_contract_and_metrics_import_without_torch() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from lewm.safety import non_greedy_local_subgoal_jepa_planning_metrics_v1 as m; "
        "print(m.score_row_authority()['schema'])"
    )
    completed = subprocess.run(
        ["/usr/bin/python3", "-E", "-s", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == (
        "non_greedy_local_subgoal_jepa_planning_v1.score_row_authority.v1"
    )


def test_three_rankers_are_byte_matched_and_under_cap() -> None:
    before = torch.random.get_rng_state().clone()
    models = C.build_matched_rankers()
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    C.assert_matched_initialization(models)
    assert set(models) == set(C.MODEL_IDS)
    assert {C.parameter_count(model) for model in models.values()} == {204289}
    assert all(C.parameter_count(model) < C.PARAMETER_CAP_EXCLUSIVE for model in models.values())


def test_ranker_conditions_use_exact_absence_and_token_shapes() -> None:
    models = C.build_matched_rankers()
    base = torch.zeros(1, C.BASE_FEATURE_DIM)
    query = torch.zeros(1, C.QUERY_FEATURE_DIM)
    anchor = torch.tensor([0.25])
    current = torch.zeros(1, C.TOKENS_PER_FRAME, C.TOKEN_DIM)
    future = torch.zeros(1, 3, C.TOKENS_PER_FRAME, C.TOKEN_DIM)
    no_latent = models[C.NO_LATENT_NON_GREEDY_RANKER](
        base_features=base, query_features=query, kinematic_anchor=anchor
    )
    current_only = models[C.CURRENT_VISUAL_REACTIVE_RANKER](
        base_features=base,
        query_features=query,
        kinematic_anchor=anchor,
        current_tokens=current,
    )
    true_future = models[C.TRUE_FUTURE_JEPA_TRAJECTORY_RANKER](
        base_features=base,
        query_features=query,
        kinematic_anchor=anchor,
        current_tokens=current,
        future_tokens=future,
    )
    assert no_latent.score.shape == current_only.score.shape == true_future.score.shape == (1,)
    assert torch.count_nonzero(no_latent.timepoint_summaries) == 0
    assert torch.count_nonzero(current_only.timepoint_summaries[:, 1:]) == 0
    with pytest.raises(C.NonGreedyContractError, match="forbids"):
        models[C.NO_LATENT_NON_GREEDY_RANKER](
            base_features=base,
            query_features=query,
            kinematic_anchor=anchor,
            current_tokens=current,
        )


def test_registered_pure_loss_is_finite_and_differentiable() -> None:
    scores = torch.tensor([2.0, 1.0, 0.0, 0.0, 1.0, 2.0], requires_grad=True)
    targets = torch.tensor([2.0, 1.0, 0.0, 2.0, 1.0, 0.0])
    states = torch.tensor([0, 0, 0, 1, 1, 1])
    residuals = scores * 0.1
    loss = C.matched_ranker_loss(
        scores=scores,
        target_utilities=targets,
        state_indices=states,
        residuals=residuals,
    )
    assert loss["pair_count"].item() == 6
    assert loss["state_count"].item() == 2
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_output_and_next_decision_literals() -> None:
    assert C.RUNTIME_OUTPUT_PATHS["independent_regeneration_receipt"] == (
        "independent_regeneration_receipt.json"
    )
    assert C.RUNTIME_OUTPUT_PATHS["result"] == "result.json"
    assert C.RUNTIME_OUTPUT_PATHS["no_latent_checkpoint"].endswith("epoch_060.pt")
    assert "RENAME_NOREPLACE" not in C.OUTPUT_AUTHORITY["persistence"]
    assert C.NEXT_DECISION_BY_CLASSIFICATION[
        "NON_GREEDY_TWO_STEP_JEPA_PLANNING_SIGNAL"
    ] == "ORACLE_ADMISSIBLE_NON_GREEDY_CLOSED_LOOP_JEPA_MPC_V1"
