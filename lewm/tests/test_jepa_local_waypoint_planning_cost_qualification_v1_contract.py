from __future__ import annotations

import ast
import base64
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

from lewm.safety import jepa_local_waypoint_planning_cost_qualification_v1_contract as contract


def test_canonical_json_and_self_digests_are_stable() -> None:
    assert contract.canonical_json_bytes({"z": 2, "a": [True, None, 1.25]}) == (
        b'{"a":[true,null,1.25],"z":2}'
    )
    first = contract.build_contract()
    second = contract.build_contract()
    assert first == second == contract.CONTRACT
    declared = first.pop("contract_sha256")
    assert declared == contract.CONTRACT_SHA256
    assert contract.canonical_json_sha256(first) == declared
    assert contract.contract_receipt_bytes().endswith(b"\n")
    assert hashlib.sha256(contract.contract_receipt_bytes()).hexdigest() == (
        contract.CONTRACT_RECEIPT_SHA256
    )

    schema = contract.build_output_schema()
    schema_digest = schema.pop("output_schema_sha256")
    assert schema_digest == contract.OUTPUT_SCHEMA_SHA256
    assert contract.canonical_json_sha256(schema) == schema_digest

    with pytest.raises(contract.ContractError, match="non-finite"):
        contract.canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(contract.ContractError, match="non-string"):
        contract.canonical_json_bytes({1: "bad"})
    with pytest.raises(contract.ContractError, match="unsupported JSON type"):
        contract.canonical_json_bytes({"bad": (1, 2)})


def test_exact_user_enums_seed_checkpoints_and_panel_are_frozen() -> None:
    value = contract.build_contract()
    assert value["experiment_id"] == "JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1"
    assert value["starting_head"] == "b29eae1929725a4cc26a35d95662b545daee4553"
    assert value["claims_boundary"]["one_seed"] == 2026080901
    assert tuple(value["sources"]) == contract.SOURCE_IDS
    assert tuple(value["populations"]["ids"]) == contract.POPULATION_IDS
    assert tuple(value["comparators"]) == contract.COMPARATOR_IDS
    assert tuple(value["paired_comparison_ids"]) == contract.PAIRED_COMPARISON_IDS
    assert tuple(value["classification"]["primary_exactly_one"]) == (
        contract.PRIMARY_CLASSIFICATIONS
    )

    panel = value["frozen_panel"]
    assert panel["states"] == 48
    assert panel["candidates_per_state"] == 12
    assert panel["candidate_rows"] == 576
    assert panel["split"]["fit_states"] == 32
    assert panel["split"]["calibration_states"] == 8
    assert panel["split"]["heldout_states"] == 8
    assert tuple(panel["families"]) == contract.FAMILY_IDS
    assert panel["identity_role_candidate_and_outcome_changes"] == "forbidden"

    predictors = value["predictor_bindings"]
    assert predictors["one_step"]["sha256"] == (
        "20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a"
    )
    assert predictors["two_step"]["sha256"] == (
        "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4"
    )
    assert predictors["architecture"]["flattened_action_dimension"] == 10
    assert predictors["architecture"]["control_history_shape"] == [3, 5, 2]
    assert predictors["architecture"]["active_channels"] == [
        "vx_body_mps",
        "yaw_rate_radps",
    ]
    assert predictors["architecture"]["vy"].endswith("forbidden in evaluation")


def test_context_and_mixed_scale_control_history_are_exact() -> None:
    predictor = contract.build_contract()["predictor_bindings"]
    reconstruction = predictor["input_reconstruction"]
    assert reconstruction["context_offsets_source_frames"] == [-480, -240, 0]
    assert reconstruction["context_offsets_command_ticks"] == [-10, -5, 0]
    assert reconstruction["context_offsets_elapsed_s"] == [-1.0, -0.5, 0.0]
    assert reconstruction["warmup_block_boundaries"] == [38, 39, 40]
    history = reconstruction["observed_control_history"]
    assert history["slot_0"] == [
        "block37_tick5",
        "block38_tick1",
        "block38_tick2",
        "block38_tick3",
        "block38_tick4",
    ]
    assert history["slot_1"][0] == "block38_tick5"
    assert history["slot_2"][-1] == "block40_tick4"
    assert "using blocks38/39/40 directly is forbidden" in history["required_validation"]
    semantics = predictor["action_and_control_semantics"]
    assert "normalised" in semantics["initial_control_history"]
    assert "RAW" in semantics["autoregressive_append"]
    assert "not corrected" in semantics["mixed_scale_disclosure"]
    assert semantics["normalisation_stats"]["file_sha256"] == (
        "9380b4c6d9b59099e43bba9898e1417c273f88075d1ed122401cbb3272e18f94"
    )
    assert semantics["normalisation_stats"]["stats_sha256"] == (
        "f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab"
    )


def test_both_predictors_use_the_same_autoregressive_unroll() -> None:
    semantics = contract.build_contract()["source_semantics"]
    assert "P.unroll(..., max_h=3)" in semantics["ONE_STEP_PREDICTED"]
    assert "P.unroll(..., max_h=3)" in semantics["TWO_STEP_PREDICTED"]
    assert "neither receives an independent true context" in semantics[
        "predicted_source_common_rule"
    ]


def test_goal_view_is_candidate_independent_and_route_heading_is_exact() -> None:
    goal = contract.build_contract()["goal_view"]
    assert goal["count"] == "exactly one candidate-independent goal view per frozen state"
    assert "waypoint_path_cells[2]" in goal["position_world"]["x_y"]
    assert goal["position_world"]["z"] == "snapshot base z"
    assert goal["orientation_world_rpy_rad"]["roll"] == 0.0
    assert goal["orientation_world_rpy_rad"]["pitch"] == 0.0
    assert goal["candidate_dependent_inputs"] == []
    assert "path[2]" in goal["preconditions"][1]
    assert "need not be free or transit-safe" in goal["preconditions"][2]
    assert goal["renderer"]["goal_render_semantics"] == (
        contract.GOAL_VIEW_RENDER_SEMANTICS
    )
    assert goal["renderer"]["physical_executability_claim"] is False
    assert goal["source_semantic_validation"]["path1_position_substitution"] == (
        "forbidden"
    )
    assert goal["source_semantic_validation"]["nav_blocked_is_diagnostic_not_failure"]
    assert goal["goal_cell_classification_counts"] == (
        contract.GOAL_CELL_CLASSIFICATION_COUNTS
    )
    assert "only 22/48" in goal["manifest_waypoint_fields"]
    assert goal["renderer"]["robot_visibility"].startswith("no-robot")

    assert contract.route_heading_yaw([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert contract.route_heading_yaw([0.0, 0.0], [0.0, 1.0]) == pytest.approx(
        math.pi / 2.0
    )
    with pytest.raises(contract.ContractError, match="identical centres"):
        contract.route_heading_yaw([1.0, 1.0], [1.0, 1.0])


def test_goal_cell_static_classification_matches_all_48_frozen_states() -> None:
    root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(root / "lewm_worlds"))
    from lewm_worlds.manifest import parse_scene_manifest_dict
    from lewm_worlds.scene_graph import SceneGraph

    manifest = json.loads(
        (root / ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json")
        .read_text(encoding="utf-8")
    )
    counts = {
        "states": 0,
        "endpoint_reachable": 0,
        "nav_blocked": 0,
        "beacon_endpoint": 0,
        "low_clearance_transit_blocked": 0,
        "unblocked": 0,
    }
    blocked = {"beacon_endpoint": [], "low_clearance_transit_blocked": []}
    optional_waypoint_fields = 0
    for state in manifest["state_candidates"]:
        graph = SceneGraph(
            parse_scene_manifest_dict(
                json.loads(
                    (Path(state["scene_dir"]) / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        )
        path = [int(value) for value in state["waypoint_path_cells"]]
        assert len(path) >= 3
        assert all(cell in range(graph.n_nodes) for cell in path[:3])
        assert path[1] in graph.neighbors(path[0])
        assert path[2] in graph.neighbors(path[1])
        reachable = graph.bfs_distance(
            path[0], path[2], transit_blocked=graph.nav_blocked_cells
        )
        assert reachable is not None
        counts["states"] += 1
        counts["endpoint_reachable"] += 1
        if path[2] in graph.beacon_cells_set:
            classification = "beacon_endpoint"
        elif path[2] in graph.nav_blocked_cells:
            classification = "low_clearance_transit_blocked"
        else:
            classification = "unblocked"
        counts[classification] += 1
        if classification != "unblocked":
            counts["nav_blocked"] += 1
            blocked[classification].append(state["state_id"])
        if state.get("waypoint_xy") is not None:
            optional_waypoint_fields += 1
            assert list(state["waypoint_xy"]) == [
                float(value) for value in graph.cell_center(path[2])
            ]

    assert counts == contract.GOAL_CELL_CLASSIFICATION_COUNTS
    assert blocked == contract.GOAL_CELL_BLOCKED_STATE_IDS
    assert optional_waypoint_fields == 22


def test_tokenwise_cosine_cost_is_token_aligned_float64_mean() -> None:
    assert contract.tokenwise_cosine_cost(
        [[1.0, -1.0, 0.0]],
        [[1.0, -1.0, 0.0]],
        expected_tokens=1,
        expected_width=3,
    ) == pytest.approx(5.960464477539063e-08, abs=1e-15)
    assert contract.tokenwise_cosine_cost(
        [[1.0, -1.0, 0.0]],
        [[1.0, 1.0, -2.0]],
        expected_tokens=1,
        expected_width=3,
    ) == pytest.approx(1.0, abs=2e-7)
    assert contract.tokenwise_cosine_cost(
        [[1.0, -1.0, 0.0]],
        [[-1.0, 1.0, 0.0]],
        expected_tokens=1,
        expected_width=3,
    ) == pytest.approx(1.9999999403953552, abs=1e-15)
    assert contract.tokenwise_cosine_cost(
        [[1.0, 0.0], [0.0, 2.0]],
        [[1.0, 0.0], [0.0, -3.0]],
        expected_tokens=2,
        expected_width=2,
    ) == pytest.approx(1.0)
    assert contract.tokenwise_cosine_cost(
        [[0.0, 0.0]], [[1.0, -1.0]], expected_tokens=1, expected_width=2
    ) == pytest.approx(1.0)


def test_primary_cost_and_monotonic_diagnostic_are_untuned() -> None:
    cost = contract.build_contract()["cost"]
    assert cost["primary_horizon"] == "H3"
    assert cost["token_count"] == 768
    assert cost["token_width"] == 1024
    assert "eps=1e-12" in cost["per_token_normalisation"]
    assert "default eps=1e-5" in cost["layer_normalisation"]
    assert cost["weights"].startswith("none")
    assert "tokenwise_normalized_cosine_mean_cost" in cost["canonical_reducer"]
    assert "fixture scaffolding only" in cost["contract_reference_helper"]
    assert cost["torch_vs_canonical_numpy_fixture_absolute_tolerance"] == 1e-6
    assert cost["monotonic_diagnostics"]["sequence"] == ["CURRENT", "H1", "H2", "H3"]
    assert cost["monotonic_diagnostics"]["gate"] is False


def test_route_ranking_ties_utility_and_regret_are_exact() -> None:
    ranking = contract.build_contract()["ranking_metrics"]
    authority = ranking["route_outcome_authority"]
    assert authority["horizon"] == "H3"
    assert authority["safety_fields_in_route_preference"] == []
    assert authority["ordered_tuple"][0].startswith("realised completed")
    ties = ranking["rank_ties"]
    assert ties["predicted_cost_tie_abs_lte"] == 1e-12
    assert ties["predicted_cost_tie_pairwise_credit"] == 0.5
    assert "exclude" in ties["oracle_unordered_pair"]
    assert "spearmanr(-cost" in ties["spearman"]
    assert "tau-b" in ties["kendall"]
    assert ranking["combined_route_utility"]["formula"] == (
        "(wins + 0.5*unordered_or_tied_pairs)/(N-1)"
    )
    assert "best p_d - selected p_d" in ranking["normalised_regret"]
    assert ranking["selected_progress_gate_ratio"].endswith("signed")
    assert ranking["aggregation"]["spearman"] == (
        "arithmetic mean of finite per-state coefficients"
    )
    assert "sum per-state ordered-pair correct credit" in ranking["aggregation"][
        "pairwise_accuracy"
    ]


def _family_metric(
    *, pairwise: float = 0.5, top3: float = 0.0, progress: float = 0.0
) -> dict[str, float | int]:
    return {
        "evaluable_nonabstaining_states": 1,
        "ordered_pairs": 1,
        "pairwise_accuracy": pairwise,
        "best_route_top3_rate": top3,
        "selected_route_progress_sum_m": progress,
    }


def test_family_complete_collapse_is_fail_closed_and_exact() -> None:
    assert contract.family_complete_collapse(_family_metric()) is True
    assert contract.family_complete_collapse(_family_metric(pairwise=0.500001)) is False
    assert contract.family_complete_collapse(_family_metric(top3=0.1)) is False
    assert contract.family_complete_collapse(_family_metric(progress=0.01)) is False
    missing_pair = _family_metric()
    missing_pair["ordered_pairs"] = 0
    assert contract.family_complete_collapse(missing_pair) is True

    all_good = {family: _family_metric(progress=0.01) for family in contract.FAMILY_IDS}
    assert contract.no_family_complete_collapse(all_good) is True
    del all_good[contract.FAMILY_IDS[0]]
    with pytest.raises(contract.ContractError, match="exactly the four"):
        contract.no_family_complete_collapse(all_good)


def test_true_future_and_two_step_gates_are_exact() -> None:
    gates = contract.build_contract()["gates"]
    assert gates["true_future"] == {
        "source": "TRUE_FUTURE",
        "population": "ORACLE_VIABILITY_ADMISSIBLE",
        "pairwise_accuracy_gte": 0.70,
        "spearman_gte": 0.60,
        "normalised_regret_lte": 0.25,
        "best_route_top3_gte": 0.75,
        "selected_progress_fraction_of_oracle_best_gte": 0.80,
        "no_family_complete_collapse": True,
        "classification_if_pass": "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL",
        "classification_if_fail": "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
    }
    predicted = gates["two_step_predicted"]
    assert predicted["prerequisite_true_future_gate"] is True
    assert predicted["pairwise_accuracy_gte"] == 0.65
    assert predicted["normalised_regret_lte"] == 0.30
    assert predicted["selected_progress_fraction_of_true_future_selected_gte"] == 0.75
    assert predicted["all_candidates_contact_selections_lte_one_step"] is True
    assert predicted["all_candidates_nonviable_selections_lte_one_step"] is True
    assert predicted["pairwise_accuracy_gt_one_step"] is True
    assert tuple(contract.TRUE_FUTURE_GATE_CLASSIFICATIONS) == (
        "TRUE_FUTURE_LATENT_GOAL_COST_SIGNAL",
        "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
    )
    assert predicted["classification_if_pass"] == "TWO_STEP_JEPA_PLANNING_COST_SIGNAL"
    assert predicted["classification_if_fail"] is None


def test_materiality_is_prospective_and_safety_preserving() -> None:
    base = {
        "progress_delta_m": 0.05,
        "progress_bootstrap_ci95_lower_m": 0.001,
        "normalized_regret_improvement": 0.0,
        "regret_bootstrap_ci95_lower": -0.1,
        "hard_family_progress_ratio_delta": 0.0,
        "hard_family_bootstrap_ci95_lower": -0.1,
        "contact_selection_increase": 0,
        "nonviable_selection_increase": 0,
        "hard_family_collapse": False,
    }
    assert contract.derive_incremental_route_value_secondary(**base) is True
    below = dict(base, progress_delta_m=0.049)
    assert contract.derive_incremental_route_value_secondary(**below) is False
    zero_ci = dict(base, progress_bootstrap_ci95_lower_m=0.0)
    assert contract.derive_incremental_route_value_secondary(**zero_ci) is False
    unsafe = dict(base, contact_selection_increase=1)
    assert contract.derive_incremental_route_value_secondary(**unsafe) is False
    hard = dict(
        base,
        progress_delta_m=0.0,
        progress_bootstrap_ci95_lower_m=-1.0,
        hard_family_progress_ratio_delta=0.10,
        hard_family_bootstrap_ci95_lower=0.001,
    )
    assert contract.derive_incremental_route_value_secondary(**hard) is True
    frozen = contract.build_contract()["paired_materiality"]
    assert frozen["population"] == "ORACLE_VIABILITY_ADMISSIBLE"
    assert frozen["triggers_any"]["hard_family_selected_progress_ratio"]["families"] == [
        "large_enclosed_maze",
        "loop_alias_stress",
    ]


@pytest.mark.parametrize(
    ("true_gate", "two_gate", "kinematic", "incremental", "expected"),
    [
        (False, False, False, False, "RAW_LATENT_GOAL_COST_NO_GO"),
        (False, True, True, True, "RAW_LATENT_GOAL_COST_NO_GO"),
        (True, True, True, False, "KINEMATIC_BASELINE_DOMINANT"),
        (True, True, False, False, "TWO_STEP_JEPA_PLANNING_COST_SIGNAL"),
        (True, True, True, True, "TWO_STEP_JEPA_PLANNING_COST_SIGNAL"),
        (True, False, False, False, "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_PLANNING_NO_GO"),
    ],
)
def test_primary_classification_precedence(
    true_gate: bool,
    two_gate: bool,
    kinematic: bool,
    incremental: bool,
    expected: str,
) -> None:
    assert contract.derive_primary_classification(
        true_future_gate_passed=true_gate,
        two_step_gate_passed=two_gate,
        kinematic_baseline_materially_superior=kinematic,
        jepa_incremental_route_value=incremental,
    ) == expected


def test_candidate_viability_materialisation_is_complete_and_outcome_independent() -> None:
    semantics = contract.build_contract()["candidate_and_viability_semantics"]
    assert semantics["decision_block"] == "five 100 ms ticks; commit one block and then replan"
    assert len(semantics["unique_first_block_primitives"]) == 9
    assert semantics["macro_candidates"] == 12
    materialisation = semantics["unconditional_materialisation"]
    assert materialisation["current_blocks_per_state"] == 9
    assert materialisation["successor_blocks_per_state"] == 81
    assert materialisation["blocks"] == 4320
    assert materialisation["physics_frames"] == 1080000
    assert materialisation["current_contact_bitset_shape"] == [9, 250]
    assert materialisation["successor_contact_bitset_shape"] == [9, 9, 250]
    assert materialisation["include_successors_of_contacting_prefixes"] is True
    assert semantics["row_identities"]["successor_viable"] == (
        "successor_safe_action_count > 0"
    )
    assert "not immediate_contact_h1" in semantics["row_identities"][
        "oracle_viability_admissible"
    ]


def test_environment_process_split_roles_and_storage_are_frozen() -> None:
    value = contract.build_contract()
    execution = value["execution"]
    assert execution["roles"]["primary_gate"] == "heldout (8 frozen states)"
    assert execution["roles"]["threshold_or_model_calibration"] == "none"
    cpu = execution["environments"]["cpu_replay_render_oracle"]
    assert cpu["interpreter"] == ".generated/venvs/genesis_render_vulkan/bin/python"
    assert cpu["interpreter_binary_binding"] == contract.INTERPRETER_BINARY_BINDING
    assert cpu["genesis"] == "0.3.14"
    assert cpu["workers"] == "exactly os.cpu_count(); expected 32 at preflight"
    gpu = execution["environments"]["encoder_predictor"]
    assert gpu["interpreter"] == "/home/andrewknowles/TinyQuadJEPA/bin/python"
    assert gpu["interpreter_binary_binding"] == contract.INTERPRETER_BINARY_BINDING
    assert gpu["torch"] == "2.10.0.dev20250926+rocm6.3"
    assert cpu["pillow"] == "11.3.0"
    assert cpu["pyyaml"] == "6.0.3"
    assert gpu["pyyaml"] == "6.0.3"
    assert set(cpu["foundational_packages"]) == {
        "torch",
        "numpy",
        "scipy",
        "pillow",
        "pyyaml",
    }
    assert set(gpu["foundational_packages"]) == set(
        cpu["foundational_packages"]
    )
    boundary = execution["environments"]["foundational_package_closure_policy"]
    assert boundary["record_or_recursive_file_byte_closure"] is False
    assert boundary["residual_limitation"] == (
        "same-version foundational package mutation is not byte-closed"
    )
    assert "R9700" in gpu["device"]
    assert "no simulator import" in execution["environments"]["process_separation"]
    digest_rule = value["predictor_bindings"]["inference_custody"][
        "parameter_state_digest_algorithm"
    ]
    assert digest_rule["models"] == ["encoder", "one_step", "two_step"]
    assert digest_rule["coverage"] == "complete state_dict: parameters and buffers"
    assert digest_rule["hash"] == "SHA-256"

    storage = value["storage"]
    assert storage["workspace_filesystem_minimum_free_gb"] == 20
    assert storage["output_filesystem_minimum_free_gb"] == 50
    assert storage["temporary_storage_ceiling_gb"] == 20
    assert storage["final_storage_ceiling_gb"] == 12
    assert "raw predicted float16" in storage["full_latent_persistence"]


def test_requirements_boundary_and_prohibitions_are_preserved() -> None:
    value = contract.build_contract()
    custody = value["requirements_custody"]
    assert custody["classifications"] == list(contract.REQUIREMENTS_CLASSIFICATIONS)
    assert len(custody["classifications"]) == 11
    assert "DISTRIBUTED_BODY_SENSING_CANDIDATE" not in custody["classifications"]
    assert custody["statement"] == (
        "Deployment hard-contact requirements, consequences and recovery criteria remain "
        "unresolved. No further deployment-safety scope reduction, sensor qualification "
        "or learned hard-safety model is authorised."
    )
    assert custody["next_decision"] == "REQUIREMENTS_ACQUISITION_REQUIRED"
    assert custody["authoritative_result"] == {
        "path": "docs/lewm_protected_contact_scope_requirements_review_v1_result.json",
        "sha256": "c348d5e2d265a118922ae138c549d8a6e48e4e900a45b70d919de4aa530e4027",
        "content_digest": "148a1757f4b8d55291ab38010a2dd0701e4606d61cd4c656de78cff571dac948",
        "result_commit": "b29eae1929725a4cc26a35d95662b545daee4553",
    }
    assert custody["scope_or_label_change"] is False
    assert value["prohibitions"]["training_steps"] == 0
    assert value["prohibitions"]["untouched_g2_access"] is False
    assert value["prohibitions"]["memory"] is False
    assert value["prohibitions"]["experimental_candidate_selecting_navigation"] is False
    controller = value["execution"]["controller_execution_custody"]
    assert controller["frozen_ppo_controller_total_blocks"] == 6240
    assert controller["frozen_ppo_controller_total_physics_frames"] == 1560000
    assert controller["reconstruction_route_teacher_ppo_blocks"] == 1920
    assert controller["fixed_oracle_fanout_ppo_blocks"] == 4320
    assert (
        controller[
            "experimental_candidate_selecting_jepa_mpc_navigation_planner_executions"
        ]
        == 0
    )


def test_all_accidental_exposures_are_disclosed_and_excluded() -> None:
    custody = contract.build_contract()["preexecution_custody"]
    assert custody["status"] == "ACCIDENTAL_EXPOSURES_DISCLOSED_AND_EXCLUDED"
    assert custody["authorized_outcome_fields_for_contract_derivation"] == []
    exposures = custody["accidental_exposures"]
    assert len(exposures) == 3
    assert all(row["values_retained_or_used"] is False for row in exposures)
    assert exposures[0]["exposure"] == "detailed held-out-state outcome table"
    assert "Stage-A diagnostic" in exposures[1]["exposure"]
    assert "unsafe/safe H3" in exposures[2]["exposure"]
    assertions = custody["required_preexecution_assertions"]
    assert assertions["outcome_values_used_for_contract_derivation"] == 0
    assert assertions["untouched_g2_reads"] == 0
    assert assertions["training_steps"] == 0


def test_output_schema_persists_rows_raw_latents_and_reproduction() -> None:
    schema = contract.build_output_schema()
    files = schema["files"]
    phase_file_ids = (
        "preexecution_receipt",
        "environment_receipt",
        "gpu_environment_receipt",
        "gpu_child_preflight_receipt",
        "gpu_child_materialize_receipt",
        "gpu_inference_receipt",
        "cpu_runtime_input_inventory",
        "context_reconstruction_index",
        "dense_route_replay_input_index",
        "oracle_admissibility_fanout_index",
        "latent_tensor_index",
        "goal_view_index",
        "aggregate_metrics",
        "persistence_receipt",
    )
    phase_keys = {
        "schema",
        "experiment_id",
        "source_freeze_commit",
        "contract_sha256",
        "output_schema_sha256",
    }
    for file_id in phase_file_ids:
        assert phase_keys <= set(files[file_id]["required_keys"])
    assert {
        "canonical_output_fresh",
        "canonical_output_root",
        "hidden_attempt_root",
        "cpu_runtime_input_inventory_binding",
        "goal_view_execution_amendment_binding",
        "current_token_execution_amendment_binding",
        "current_token_authority_policy",
        "gpu_child_execution_receipts",
    } <= set(files["preexecution_receipt"]["required_keys"])
    preexecution = files["preexecution_receipt"]
    assert preexecution["preexecution_custody_required_keys"] == [
        "contract_disclosure",
        "live_validation",
    ]
    assert preexecution["preexecution_contract_disclosure_exact"] == (
        contract.build_contract()["preexecution_custody"]
    )
    assert len(
        preexecution["preexecution_contract_disclosure_exact"][
            "accidental_exposures"
        ]
    ) == 3
    assert preexecution["preexecution_live_validation_exact"] == {
        "outcome_barrier_lifted_only_after_freeze_commit": True,
        "outcome_rows_read": 0,
        "checkpoint_tensors_opened": 0,
        "predictor_inference_calls": 0,
        "fixture_pass": True,
    }
    assert preexecution["goal_view_execution_amendment_binding_exact"] == (
        contract.GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
    )
    assert preexecution["current_token_execution_amendment_binding_exact"] == (
        contract.CURRENT_TOKEN_EXECUTION_AMENDMENT_BINDING
    )
    assert preexecution["current_token_authority_policy_exact"] == (
        contract.CURRENT_TOKEN_AUTHORITY_POLICY
    )
    assert preexecution["gpu_child_execution_receipts_required_phase_ids"] == [
        "PREFLIGHT"
    ]
    assert preexecution["gpu_child_execution_receipt_binding_required_keys"] == [
        "path",
        "sha256",
        "bytes",
        "content_digest",
    ]
    assert preexecution["goal_view_static_validation_exact"] == (
        contract.GOAL_VIEW_STATIC_VALIDATION_SUCCESS
    )
    assert preexecution["cpu_worker_environment_exact"] == {
        "MALLOC_ARENA_MAX": "1",
        "workers": 32,
        "dynamic_worker_fallback": False,
        "purpose": "allocator fragmentation mitigation only",
        "scientific_semantics_change": False,
    }
    assert preexecution["execution_watchdog_config_exact"][
        "cpu_worker_environment"
    ] == preexecution["cpu_worker_environment_exact"]
    goal_index = files["goal_view_index"]
    assert goal_index["goal_cell_classification_counts_exact"] == (
        contract.GOAL_CELL_CLASSIFICATION_COUNTS
    )
    assert goal_index["goal_cell_classification_validation_exact"] == (
        contract.GOAL_CELL_CLASSIFICATION_VALIDATION_SUCCESS
    )
    assert goal_index["goal_render_semantics_exact"] == (
        contract.GOAL_VIEW_RENDER_SEMANTICS
    )
    for key in (
        "goal_cell_preconditions",
        "goal_cell_endpoint_reachable",
        "goal_cell_nav_blocked",
        "goal_cell_block_classification",
        "goal_cell_is_beacon_endpoint",
        "goal_cell_is_low_clearance_transit_blocked",
        "goal_render_semantics",
    ):
        assert key in goal_index["record_required_keys"]
    assert goal_index["goal_cell_precondition_required_keys"] == [
        "path_cells",
        "path_cell_centers_world_xy",
        "goal_cell",
        "goal_path_cell_ids_valid",
        "goal_path_consecutive_edge_pairs",
        "goal_path_consecutive_edges_traversable",
        "goal_cell_endpoint_reachable",
        "goal_cell_endpoint_bfs_hops",
        "goal_cell_nav_blocked",
        "goal_cell_block_classification",
        "goal_cell_is_beacon_endpoint",
        "goal_cell_is_low_clearance_transit_blocked",
        "goal_render_semantics",
        "pass",
    ]
    assert files["candidate_evidence"]["path"].endswith(".jsonl.gz")
    assert files["selection_evidence"]["path"].endswith(".jsonl.gz")
    assert files["paired_effect_evidence"]["path"].endswith(".jsonl.gz")
    candidate_keys = files["candidate_evidence"]["required_keys"]
    assert "context_control_history_raw" in candidate_keys
    assert "context_control_history_normalized" in candidate_keys
    assert "action_blocks_raw_3x10" in candidate_keys
    assert "requested_action_blocks_raw_3x5x3" in candidate_keys
    assert "applied_action_blocks_raw_3x5x3" in candidate_keys
    assert "realised_route_fields_by_horizon" in candidate_keys
    assert "oracle_route_fields_h3_primary" in candidate_keys
    assert "dense_route_replay_state_record_ref" in candidate_keys
    assert "successor_safe_action_count" in candidate_keys
    assert "successor_viable" in candidate_keys
    assert "oracle_viability_admissible" in candidate_keys
    assert "nonviable" not in candidate_keys
    selection_keys = files["selection_evidence"]["required_keys"]
    assert "pairwise_correct_credit" in selection_keys
    assert "correct_ordered_pair_count" not in selection_keys
    assert "cost_tie_pair_count" in selection_keys
    assert "cost_tie_pair_rate" in selection_keys
    result_keys = files["result"]["required_keys"]
    assert "source_freeze_commit" in result_keys
    assert "primary_classification" in result_keys
    assert "secondary_classifications" in result_keys
    assert "requirements_custody" in result_keys
    assert "goal_view_execution_amendment_binding" in result_keys
    assert "current_token_execution_amendment_binding" in result_keys
    assert "current_token_authority_policy" in result_keys
    assert "gpu_child_execution_receipts" in result_keys
    assert "goal_pose_semantics" in result_keys
    assert "goal_cell_classification_counts" in result_keys
    assert "goal_cell_classification_validation" in result_keys
    result_schema = files["result"]
    assert result_schema["materialisation_count_required_keys"] == [
        "states",
        "candidates",
        "reconstruction_prefix_blocks",
        "reconstruction_physics_frames",
        "oracle_fanout_blocks",
        "oracle_fanout_physics_frames",
        "total_simulator_blocks",
        "total_simulator_physics_frames",
        "snapshot_reproductions",
        "current_blocks",
        "successor_blocks",
        "total_oracle_blocks",
        "physics_frames",
        "new_encoder_frames",
        "new_encoder_batches",
        "current_authority_payload_copies",
        "current_token_reencodes",
        "latent_tensors",
        "predicted_tensors",
        "candidate_evidence_rows",
        "selection_evidence_rows",
        "paired_effect_evidence_rows",
    ]
    assert set(result_schema["runtime_required_keys"]) == {
        "preflight",
        "cpu_materialization",
        "gpu_materialization",
        "evaluation_reduction",
        "total",
    }
    assert "peak_vram_bytes" in result_schema["storage_required_keys"]
    gpu_keys = files["gpu_inference_receipt"]["required_keys"]
    assert "interpreter_entrypoint" in gpu_keys
    assert "source_closure_binding" in gpu_keys
    assert "gpu_environment_receipt_binding" in gpu_keys
    assert "encoder_source_repository_binding" in gpu_keys
    assert "preprocessing_digest" in gpu_keys
    assert "parameter_state_digest_before" in gpu_keys
    assert "parameter_state_digest_after" in gpu_keys
    assert "training_steps" in gpu_keys
    assert "future_input_fields" in gpu_keys
    assert "current_token_execution_amendment_binding" in gpu_keys
    assert "current_token_authority_policy" in gpu_keys
    assert "gpu_child_execution_receipts" not in gpu_keys
    assert files["gpu_inference_receipt"]["interpreter_entrypoint_exact"] == (
        "/home/andrewknowles/TinyQuadJEPA/bin/python"
    )
    assert files["gpu_inference_receipt"]["interpreter_binding_exact"] == (
        contract.INTERPRETER_BINARY_BINDING
    )
    environment_keys = files["environment_receipt"]["required_keys"]
    assert "interpreter_bindings" in environment_keys
    assert "process_separation" in environment_keys
    replay_index = files["dense_route_replay_input_index"]
    assert replay_index["path"] == "materialization/dense_route_replay_input_index.json"
    assert replay_index["cardinality"] == {
        "states": 48,
        "branches_per_state": 12,
        "h3_tick_count": 15,
        "horizon_tick_boundaries": [5, 10, 15],
    }
    assert replay_index["evidence_receipt"]["sha256"] == (
        "a547ac544a869a6ef75a4798b22875291e55604f9e53ceaea24a790db09df7e1"
    )
    paired = files["paired_effect_evidence"]
    assert paired["comparison_ids"] == list(contract.PAIRED_COMPARISON_IDS)
    assert paired["expected_rows"] == 32
    assert paired["population"] == "ORACLE_VIABILITY_ADMISSIBLE"
    assert paired["role"] == "heldout"
    assert paired["effect_conventions"]["normalised_regret_improvement"].startswith(
        "right minus left"
    )
    assert "dense_route_replay_input_index_binding" in files["result"]["required_keys"]
    assert "dense_route_replay_input_index_binding" in files["persistence_receipt"][
        "required_keys"
    ]
    persistence_schema = files["persistence_receipt"]
    assert "goal_view_execution_amendment_binding" in persistence_schema[
        "required_keys"
    ]
    assert "current_token_execution_amendment_binding" in persistence_schema[
        "required_keys"
    ]
    assert "current_token_authority_policy" in persistence_schema["required_keys"]
    assert "gpu_child_execution_receipts" in persistence_schema["required_keys"]
    assert "artifact_manifest_exclusions" in persistence_schema["required_keys"]
    assert persistence_schema["artifact_manifest_exclusions_exact"] == [
        "receipts/persistence.json",
        "result.json",
        "receipts/RUNNING.json",
    ]
    assert persistence_schema["report_integrity"].startswith(
        "report.md is included"
    )
    assert schema["storage_ceilings_gb"] == {"temporary": 20, "final": 12}
    assert "reuses no scientific phase or shard" in schema["atomic_publication"][
        "partial_run"
    ]
    assert "Persist all predicted" in schema["raw_tensor_policy"]
    assert schema["aggregate_reproduction"]["tensor_to_cost_row"]["predictor_inference"] is False
    assert schema["files"]["latent_tensor_index"]["expected_counts"]["total"] == 5424
    assert schema["files"]["latent_tensor_index"][
        "physical_encoder_materialisation_counts_exact"
    ] == {
        "context_slots_0_and_1": 96,
        "goal": 48,
        "current": 0,
        "total_new_encoder_frames": 144,
        "batch_size": 16,
        "batches": 9,
        "authority_current_payload_copies": 48,
    }
    latent_required = schema["files"]["latent_tensor_index"]["required_keys"]
    assert "unique_tensor_payload_count" in latent_required
    assert "external_current_index_binding" in latent_required
    for phase in ("PREFLIGHT", "MATERIALIZE"):
        spec = files[f"gpu_child_{phase.lower()}_receipt"]
        assert spec["phase_exact"] == phase
        assert spec["required_keys"] == list(
            contract.GPU_CHILD_EXECUTION_RECEIPT_REQUIRED_KEYS
        )
        assert spec["stream_required_keys"] == list(
            contract.GPU_CHILD_STREAM_REQUIRED_KEYS
        )
        assert spec["stream_tail_max_bytes"] == 8192
    assert schema["files"]["oracle_admissibility_fanout_index"]["array_requirements"][
        "successor_contact_bitset"
    ]["shape"] == [9, 9, 250]
    aggregate_required = schema["files"]["aggregate_metrics"]["required_keys"]
    for key in (
        "per_role",
        "latent_progress_diagnostics",
        "all_candidate_tendency_diagnostics",
        "gates",
        "classification",
    ):
        assert key in aggregate_required
    classification_schema = schema["files"]["aggregate_metrics"]
    assert classification_schema["source_or_comparator_ids"] == list(
        contract.COMPARATOR_IDS
    )
    assert classification_schema["population_ids"] == list(contract.POPULATION_IDS)
    assert classification_schema["role_ids"] == ["fit", "calibration", "heldout"]
    assert classification_schema["family_ids"] == list(contract.FAMILY_IDS)
    assert classification_schema["classification_predicted_base_screen_keys"] == [
        "ONE_STEP_PREDICTED",
        "TWO_STEP_PREDICTED",
    ]
    assert classification_schema["classification_diagnostic_flag_keys"] == [
        "both_predicted_base_screens_failed",
        "ONE_STEP_BASE_SCREEN_ONLY",
        "two_step_base_screen_passed_but_full_gate_failed",
    ]


def test_dense_route_replay_binding_is_complete_and_outcome_independent() -> None:
    authority = contract.build_contract()["latent_bindings"][
        "dense_route_replay_outcome_authority"
    ]
    assert authority["state_files"] == 48
    assert authority["branches_per_state"] == 12
    assert authority["h3_tick_count"] == 15
    assert authority["horizon_tick_boundaries"] == [5, 10, 15]
    assert authority["evidence_receipt"]["bytes"] == 1484
    assert authority["outcome_values_open_before_freeze"] is False
    assert "content_digest" in authority["required_pre_reduction_validation"]


def test_fixture_is_self_digesting_payload_free_and_complete() -> None:
    fixture = contract.build_fixture_receipt()
    contract.validate_fixture_receipt(fixture)
    assert fixture["outcome_rows_read"] == 0
    assert fixture["checkpoint_tensors_opened"] == 0
    assert fixture["predictor_inference_calls"] == 0
    assert fixture["fixtures"]["token_cosine"]["orthogonal_after_layer_norm"]["cost"] == 1.0
    assert fixture["fixtures"]["token_cosine"]["canonical_reducer"].endswith(
        "tokenwise_normalized_cosine_mean_cost"
    )
    assert fixture["fixtures"]["token_cosine"]["opposite"]["cost"] == 2.0
    assert fixture["fixtures"]["goal_heading"]["identical_cells"] == "FAIL_CLOSED"
    assert fixture["fixtures"]["bootstrap"]["replicates"] == 10000
    assert fixture["fixtures"]["bootstrap"]["seed"] == 2026080901
    assert fixture["executed_checks"]
    assert all(fixture["executed_checks"].values())
    assert fixture["pass"] is True

    tampered = copy.deepcopy(fixture)
    tampered["pass"] = False
    with pytest.raises(contract.ContractError, match="content_digest mismatch"):
        contract.validate_fixture_receipt(tampered)


def test_goal_view_amendment_binds_failed_attempt_and_is_prospective() -> None:
    root = Path(__file__).resolve().parents[2]
    value = contract.build_goal_view_execution_amendment()
    contract.validate_goal_view_execution_amendment(value)
    binding = contract.GOAL_VIEW_EXECUTION_AMENDMENT_BINDING
    tracked = root / binding["path"]
    assert tracked.read_bytes() == contract.goal_view_execution_amendment_receipt_bytes()
    assert len(tracked.read_bytes()) == binding["bytes"]
    assert hashlib.sha256(tracked.read_bytes()).hexdigest() == binding["sha256"]
    assert value["content_digest"] == binding["content_digest"]
    assert value["original_freeze"]["commit"] == contract.ORIGINAL_FREEZE_COMMIT
    assert value["static_diagnosis"]["route_outcome_rows_read_or_used"] == 0
    assert value["failed_attempt"]["scientific_phase_or_shard_reuse"] is False
    assert value["failed_attempt"]["scientific_result_published"] is False
    assert value["failed_attempt"]["aggregate_metrics_or_gates_computed"] is False
    assert value["failed_attempt"]["gpu_predictor_inference_executed"] is False
    summary = value["failed_attempt"]["artifact_summary"]
    assert (
        summary["worker_logs"]
        + summary["partial_context_rgb_files"]
        + summary["preexecution_or_environment_receipts"]
        + summary["cpu_runtime_input_inventory_receipts"]
        + summary["failure_or_running_marker_receipts"]
        == 41
    )

    for artifact in ("contract", "output_schema", "fixture", "source_closure"):
        row = value["original_freeze"][artifact]
        payload = subprocess.run(
            ["git", "show", f"{contract.ORIGINAL_FREEZE_COMMIT}:{row['path']}"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]

    archive = Path(value["failed_attempt"]["archive_path"])
    rows = []
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(archive).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    inventory = value["failed_attempt"]["archive_inventory"]
    canonical = contract.canonical_json_bytes(rows)
    assert len(rows) == inventory["record_count"] == 41
    assert sum(row["bytes"] for row in rows) == inventory["total_bytes"]
    assert len(canonical) == inventory["canonical_records_bytes"]
    assert hashlib.sha256(canonical).hexdigest() == inventory["aggregate_sha256"]
    failure = value["failed_attempt"]["failure_receipt"]
    failure_payload = (archive / failure["path"]).read_bytes()
    assert len(failure_payload) == failure["bytes"]
    assert hashlib.sha256(failure_payload).hexdigest() == failure["sha256"]
    assert json.loads(failure_payload)["content_digest"] == failure["content_digest"]


def test_current_token_amendment_binds_second_failure_and_exact_new_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    value = contract.build_current_token_execution_amendment()
    contract.validate_current_token_execution_amendment(value)
    binding = contract.CURRENT_TOKEN_EXECUTION_AMENDMENT_BINDING
    tracked = root / binding["path"]
    assert tracked.read_bytes() == (
        contract.current_token_execution_amendment_receipt_bytes()
    )
    assert len(tracked.read_bytes()) == binding["bytes"]
    assert hashlib.sha256(tracked.read_bytes()).hexdigest() == binding["sha256"]
    assert value["content_digest"] == binding["content_digest"]
    assert value["prior_source_freeze"]["commit"] == (
        contract.GOAL_VIEW_CORRECTION_COMMIT
    )

    policy = value["amended_current_token_semantics"]
    encoded = policy["new_encoder_frames"]
    logical = policy["logical_tensor_counts"]
    assert encoded == {
        "context_slots_0_and_1": 96,
        "goal": 48,
        "current": 0,
        "total": 144,
        "batch_size": 16,
        "batches": 9,
        "batch_order": (
            "ascending RGB SHA-256, then kind, numeric state identity and context slot"
        ),
        "dynamic_batch_fallback": False,
    }
    assert logical["total"] == 5424
    assert sum(value for key, value in logical.items() if key != "total") == 5424
    payload = policy["current_payload_materialisation"]
    assert payload["authority_payload_copies"] == 48
    assert payload["copies_per_state"] == 1
    assert payload["duplicate_physical_current_payloads"] == 0
    assert "NPY container" in payload["copy_semantics"]
    assert "payload bytes equal" in payload["copy_semantics"]
    assert policy["current_reencoding_for_equality_gate"] == "forbidden"
    assert policy["current_rgb_byte_equality_gate_retained"] is True
    assert policy["scientific_cost_gate_metric_or_classification_change"] is False

    diagnosis = value["static_diagnosis"]
    assert diagnosis["current_rgb_exact_matches"] == 48
    assert diagnosis["reencoded_current_token_exact_matches"] == 0
    assert value["failed_attempt"]["predictor_checkpoint_files_hashed"] == 2
    assert value["failed_attempt"][
        "predictor_checkpoint_tensor_deserializations"
    ] == 0
    assert diagnosis["predictor_checkpoint_tensors_opened"] == 0
    assert diagnosis["predictor_inference_calls"] == 0
    assert diagnosis["flat_cosine_range"] == [
        0.9999043258031383,
        0.9999475581155232,
    ]
    assert diagnosis["flat_cosine_mean"] == 0.9999380251025588
    assert diagnosis["per_token_mean_cosine_mean"] == 0.9999401111347598
    assert diagnosis["mae_range"] == [
        0.012470918548312207,
        0.017233230190110287,
    ]
    assert diagnosis["rmse_range"] == [
        0.01791116887331469,
        0.024418504702133508,
    ]

    for artifact in (
        "contract",
        "output_schema",
        "fixture",
        "goal_view_amendment",
        "source_closure",
    ):
        row = value["prior_source_freeze"][artifact]
        git_payload = subprocess.run(
            [
                "git",
                "show",
                f"{contract.GOAL_VIEW_CORRECTION_COMMIT}:{row['path']}",
            ],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert len(git_payload) == row["bytes"]
        assert hashlib.sha256(git_payload).hexdigest() == row["sha256"]

    archive = Path(value["failed_attempt"]["archive_path"])
    rows = []
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        file_payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(archive).as_posix(),
                "sha256": hashlib.sha256(file_payload).hexdigest(),
                "bytes": len(file_payload),
            }
        )
    inventory = value["failed_attempt"]["archive_inventory"]
    canonical = contract.canonical_json_bytes(rows)
    assert len(rows) == inventory["record_count"] == 537
    assert sum(row["bytes"] for row in rows) == inventory["total_bytes"]
    assert len(canonical) == inventory["canonical_records_bytes"]
    assert hashlib.sha256(canonical).hexdigest() == inventory["aggregate_sha256"]
    assert value["failed_attempt"]["artifact_summary"]["total"] == 537
    assert value["failed_attempt"]["scientific_phase_or_shard_reuse"] is False
    assert value["failed_attempt"]["aggregate_metrics_or_gates_computed"] is False
    assert value["failed_attempt"]["predictor_inference_executed"] is False

    durable = value["durable_gpu_child_error_capture"]
    assert durable["phase_ids"] == ["PREFLIGHT", "MATERIALIZE"]
    assert set(durable["receipts"]) == {"PREFLIGHT", "MATERIALIZE"}
    assert durable["terminal_check_capture"][
        "canonical_success_receipt_or_log"
    ] is False


def test_write_and_load_helpers_are_immutable(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    schema_path = tmp_path / "schema.json"
    fixture_path = tmp_path / "fixture.json"
    amendment_path = tmp_path / "amendment.json"
    current_amendment_path = tmp_path / "current-amendment.json"
    contract.write_contract(contract_path)
    contract.write_output_schema(schema_path)
    contract.write_fixture_receipt(fixture_path)
    contract.write_goal_view_execution_amendment(amendment_path)
    contract.write_current_token_execution_amendment(current_amendment_path)
    assert contract.load_and_validate_contract(contract_path) == contract.build_contract()
    assert contract.load_and_validate_output_schema(schema_path) == contract.build_output_schema()
    assert contract.load_and_validate_fixture_receipt(fixture_path) == (
        contract.build_fixture_receipt()
    )
    assert contract.load_and_validate_goal_view_execution_amendment(
        amendment_path
    ) == contract.build_goal_view_execution_amendment()
    assert contract.load_and_validate_current_token_execution_amendment(
        current_amendment_path
    ) == contract.build_current_token_execution_amendment()
    assert json.loads(contract_path.read_bytes())["experiment_id"] == contract.EXPERIMENT_ID
    contract_path.write_bytes(b"{}\n")
    with pytest.raises(contract.ContractError, match="refusing to overwrite"):
        contract.write_contract(contract_path)


def test_static_source_bindings_match_the_clean_starting_tree() -> None:
    root = Path(__file__).resolve().parents[2]
    for row in contract.STATIC_FILE_BINDINGS.values():
        payload = (root / row["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]


def test_cpu_runtime_input_bindings_are_exact_without_deserialising_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    binding = contract.CPU_RUNTIME_INPUT_BINDINGS
    assert contract.CPU_FOUNDATIONAL_PACKAGE_BINDINGS["torch"]["package_root"] == (
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/venvs/"
        "genesis_render_vulkan/lib/python3.12/site-packages/torch"
    )
    assert binding["foundational_packages"] == (
        contract.CPU_FOUNDATIONAL_PACKAGE_BINDINGS
    )
    assert binding["foundational_package_closure_policy"] == (
        contract.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
    )
    for row in binding["foundational_packages"].values():
        assert Path(row["package_root"]).is_dir()
        assert row["import_name"]
        assert row["distribution"]
        assert row["version"]
    for key in ("platform_manifest", "primitive_registry"):
        row = binding[key]
        payload = (root / row["path"]).read_bytes()
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    for row in binding["policy_artifacts"].values():
        payload = (root / row["path"]).read_bytes()
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]

    urdf = binding["genesis_builtin_urdf"]
    urdf_path = Path(urdf["resolved_path"])
    payload = urdf_path.read_bytes()
    assert len(payload) == urdf["bytes"]
    assert hashlib.sha256(payload).hexdigest() == urdf["sha256"]
    mesh_root = urdf_path.parent.parent / "dae"
    assert len(urdf["referenced_meshes"]) == 7
    for row in urdf["referenced_meshes"]:
        mesh = (urdf_path.parent / row["relative_path"]).resolve()
        assert mesh == mesh_root / row["name"]
        payload = mesh.read_bytes()
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]

    textures = binding["textures"]
    assert textures["environment_variable_required_state"] == "ABSENT"
    assert len(textures["records"]) == 12
    for row in textures["records"]:
        payload = (root / row["path"]).read_bytes()
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
    assert binding["box_obj_cache"]["execution_status"] == (
        "NOT_REACHED_BY_FROZEN_HISTORICAL_RENDERER"
    )
    assert binding["box_obj_cache"]["files_opened_or_used"] == 0

    for package in binding["cpu_packages"].values():
        closure = package["record_closure"]
        record_path = Path(closure["record_path"])
        record_payload = record_path.read_bytes()
        assert len(record_payload) == closure["record_bytes"]
        assert hashlib.sha256(record_payload).hexdigest() == closure["record_sha256"]
        base = record_path.parent.parent
        present = []
        absent = []
        declared_hash_entries = 0
        with record_path.open(newline="", encoding="utf-8") as handle:
            record_rows = list(csv.reader(handle))
        for relative, declared, declared_size in record_rows:
            path = (base / relative).resolve()
            if not path.is_file():
                assert declared == ""
                absent.append(relative)
                continue
            payload = path.read_bytes()
            if declared:
                declared_hash_entries += 1
                algorithm, encoded = declared.split("=", 1)
                observed = base64.urlsafe_b64encode(
                    hashlib.new(algorithm, payload).digest()
                ).decode("ascii").rstrip("=")
                assert observed == encoded
            if declared_size:
                assert len(payload) == int(declared_size)
            present.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
        present.sort(key=lambda row: row["path"])
        absent.sort()
        assert len(record_rows) == closure["record_entries"]
        assert declared_hash_entries == closure["declared_hash_entries"]
        assert len(present) == closure["present_files"]
        assert len(absent) == closure["absent_unhashed_files"]
        assert sum(row["bytes"] for row in present) == closure["present_file_bytes"]
        assert contract.canonical_json_sha256(present) == closure[
            "present_file_aggregate_sha256"
        ]
        assert contract.canonical_json_sha256(absent) == closure[
            "absent_unhashed_path_list_sha256"
        ]


def test_dense_replay_prefreeze_byte_inventory_matches_without_json_parse() -> None:
    root = Path(__file__).resolve().parents[2]
    binding = contract.DENSE_ROUTE_REPLAY_INPUT_BINDINGS
    rows = []
    for state_index in range(48):
        state_id = f"purpose-{state_index}"
        relative = binding["path_rule"].format(state_id=state_id)
        payload = (root / relative).read_bytes()
        rows.append(
            {
                "state_id": state_id,
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    assert len(rows) == binding["record_count"] == 48
    assert sum(row["bytes"] for row in rows) == binding["total_bytes"]
    canonical = contract.canonical_json_bytes(rows)
    assert len(canonical) == binding["canonical_records_bytes"]
    assert hashlib.sha256(canonical).hexdigest() == binding[
        "canonical_sorted_path_sha_bytes_aggregate_sha256"
    ]
    assert binding["outcome_fields_parsed_before_freeze"] == []


def test_scene_prefreeze_byte_inventory_matches_before_scene_parse() -> None:
    root = Path(__file__).resolve().parents[2]
    binding = contract.SCENE_INPUT_BYTE_INVENTORY_BINDING
    state_manifest = json.loads(
        (root / ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json")
        .read_text(encoding="utf-8")
    )
    rows = []
    for state in state_manifest["state_candidates"]:
        scene_dir = Path(state["scene_dir"])
        if scene_dir.is_absolute():
            scene_dir = scene_dir.resolve().relative_to(root.resolve())
        for kind, filename in (
            ("manifest", "manifest.json"),
            ("genesis_scene", "genesis_scene.json"),
        ):
            relative = str(scene_dir / filename)
            payload = (root / relative).read_bytes()
            rows.append(
                {
                    "state_id": state["state_id"],
                    "scene_id": state["scene_id"],
                    "kind": kind,
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
    assert len(rows) == binding["record_count"] == 96
    assert sum(row["bytes"] for row in rows) == binding["total_bytes"]
    canonical = contract.canonical_json_bytes(rows)
    assert len(canonical) == binding["canonical_records_bytes"]
    assert hashlib.sha256(canonical).hexdigest() == binding[
        "canonical_sorted_path_sha_bytes_aggregate_sha256"
    ]
    assert binding["outcome_fields_parsed_before_freeze"] == []


def test_cpu_runtime_inventory_and_raw_continuation_are_schema_bound() -> None:
    schema = contract.build_output_schema()["files"]
    inventory = schema["cpu_runtime_input_inventory"]
    assert inventory["path"] == "materialization/cpu_runtime_input_inventory.json"
    assert inventory["counts_exact"] == {
        "states": 48,
        "scene_records": 48,
        "manifest_files": 48,
        "genesis_scene_files": 48,
        "textures": 12,
        "genesis_referenced_meshes": 7,
    }
    assert inventory["box_obj_cache_exact"]["files_opened_or_used"] == 0
    assert "prefreeze_scene_byte_inventory_binding" in inventory["required_keys"]
    assert "prefreeze_scene_byte_inventory_records" in inventory["required_keys"]
    assert "prefreeze_scene_byte_inventory_validation" in inventory["required_keys"]
    assert inventory["renderer_structure_schema_audit_exact"][
        "effective_scene_geometry"
    ] == "FLOOR_PLANE_ONLY"
    assert inventory["outcome_fields_read_exact"] == []
    for file_id in (
        "context_reconstruction_index",
        "oracle_admissibility_fanout_index",
        "result",
        "persistence_receipt",
    ):
        assert "cpu_runtime_input_inventory_binding" in schema[file_id]["required_keys"]

    fanout = schema["oracle_admissibility_fanout_index"]
    assert "raw_continuation_policy_validation" in fanout["required_keys"]
    assert "raw_continuation_snapshots" in fanout["record_required_keys"]
    assert fanout["raw_continuation_snapshots_per_state"] == 9
    assert fanout["cpu_watchdog_status_exact"] == (
        contract.CPU_WATCHDOG_STATUS_SUCCESS
    )
    assert fanout["raw_continuation_snapshot_required_keys"] == [
        "current_primitive_index",
        "current_primitive_id",
        "capture_mode",
        "snapshot_digest",
        "terminal_flags",
        "consecutive_tipped_blocks",
        "production_reset_checks_suppressed",
        "evaluation_only",
        "restored_for_successors",
    ]
    dense = schema["dense_route_replay_input_index"]
    assert {
        "prefreeze_byte_inventory_binding",
        "prefreeze_byte_inventory_records",
        "prefreeze_byte_inventory_validation",
    } <= set(dense["required_keys"])

    result = schema["result"]
    assert "historical_renderer_limitations" in result["required_keys"]
    assert "reconstruction_prefix_custody" in result["required_keys"]
    assert "controller_execution_custody" in result["required_keys"]
    assert result["materialisation_count_required_keys"][:9] == [
        "states",
        "candidates",
        "reconstruction_prefix_blocks",
        "reconstruction_physics_frames",
        "oracle_fanout_blocks",
        "oracle_fanout_physics_frames",
        "total_simulator_blocks",
        "total_simulator_physics_frames",
        "snapshot_reproductions",
    ]
    gpu_environment = schema["gpu_environment_receipt"]
    assert gpu_environment["path"] == "receipts/gpu_environment.json"
    assert gpu_environment["preinference_exact"] == {
        "genesis_imported": False,
        "checkpoint_tensor_open_count": 0,
        "predictor_inference_calls": 0,
        "pass": True,
    }
    assert gpu_environment["interpreter_binding_exact"] == (
        contract.INTERPRETER_BINARY_BINDING
    )
    assert schema["environment_receipt"]["interpreter_bindings_exact"] == {
        "cpu": contract.INTERPRETER_BINARY_BINDING,
        "gpu": contract.INTERPRETER_BINARY_BINDING,
    }
    assert contract.build_contract()["claims_boundary"][
        "historical_renderer_limitations"
    ] == contract.HISTORICAL_RENDERER_LIMITATIONS
    assert contract.HISTORICAL_RENDERER_LIMITATIONS == {
        "effective_scene_geometry": "FLOOR_PLANE_ONLY",
        "input_schema": "genesis_scene.json with structural geometry under objects",
        "builder_schema": (
            "scripts.render_replay_v03.build_scene reads walls, obstacles and landmarks"
        ),
        "structural_walls_obstacles_landmarks_rendered": False,
        "current_true_future_byte_compatibility_preserved": True,
        "explicit_wall_visual_reasoning_claim": False,
        "interpretation": "HISTORICAL_RENDERER_LATENT_ROUTE_RANKING_ONLY",
    }
    assert schema["context_reconstruction_index"][
        "reconstruction_prefix_custody_exact"
    ] == contract.RECONSTRUCTION_PREFIX_CUSTODY
    assert result["reconstruction_prefix_custody_exact"] == (
        contract.RECONSTRUCTION_PREFIX_CUSTODY
    )
    assert result["controller_execution_custody_exact"] == (
        contract.CONTROLLER_EXECUTION_CUSTODY
    )
    assert result["execution_watchdog_status_exact"] == (
        contract.EXECUTION_WATCHDOG_STATUS_SUCCESS
    )
    assert contract.RECONSTRUCTION_PREFIX_CUSTODY[
        "frozen_state_reconstruction_route_teacher_ppo_blocks"
    ] == 1920
    assert schema["result"]["required_keys"]
    assert contract.build_output_schema()["prohibition_counter_ids"] == list(
        contract.PROHIBITION_COUNTER_IDS
    )
    assert "closed_loop_navigation_runs" not in contract.PROHIBITION_COUNTER_IDS


def test_upstream_encoder_repository_binding_is_exact_and_clean() -> None:
    binding = contract.build_contract()["latent_bindings"]["encoder"][
        "source_repository"
    ]
    repository = Path(binding["path"])
    observed_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    observed_status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    backbones = repository / binding["backbones_path"]
    assert observed_commit == binding["git_commit"]
    assert observed_status == ""
    assert backbones.stat().st_size == binding["backbones_bytes"]
    assert hashlib.sha256(backbones.read_bytes()).hexdigest() == binding[
        "backbones_sha256"
    ]


def test_source_closure_builder_never_traverses_generated_or_outcomes() -> None:
    root = Path(__file__).resolve().parents[2]
    receipt = contract.build_source_closure(root, require_complete=False)
    contract.validate_content_digest(receipt)
    assert receipt["outcome_or_result_payloads_parsed"] == []
    assert receipt["custody_only_result_files_hashed_without_parsing"] == [
        "docs/lewm_protected_contact_scope_requirements_review_v1_result.json"
    ]
    assert receipt["generated_cache_paths_traversed"] == []
    paths = {row["path"] for row in receipt["rows"]}
    assert (
        "lewm/safety/jepa_local_waypoint_planning_cost_qualification_v1_contract.py"
        in paths
    )
    assert str(contract.TRACKED_GOAL_VIEW_AMENDMENT_PATH) in paths
    assert str(contract.TRACKED_CURRENT_TOKEN_AMENDMENT_PATH) in paths
    assert not any("route_intent_v2_result" in path for path in paths)
    with pytest.raises(contract.ContractError, match="duplicate"):
        contract.build_source_closure(
            root,
            additional_paths=[
                "lewm/safety/jepa_local_waypoint_planning_cost_qualification_v1_contract.py"
            ],
            require_complete=False,
        )


def test_predictor_transitive_local_imports_are_source_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    closed = set(contract.SOURCE_CLOSURE_DEFAULT_PATHS)
    predictor_runtime_modules = (
        "scripts/run_jepa_local_waypoint_planning_cost_inference_v1.py",
        "scripts/dev_proprio_predictor_v1.py",
        "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    )
    discovered: set[str] = set()
    for relative in predictor_runtime_modules:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            module = node.module or ""
            if module == "scripts":
                for alias in node.names:
                    candidate = f"scripts/{alias.name}.py"
                    if (root / candidate).is_file():
                        discovered.add(candidate)
            elif module.startswith("scripts."):
                candidate = module.replace(".", "/") + ".py"
                if (root / candidate).is_file():
                    discovered.add(candidate)
    assert {
        "scripts/build_dev_v03_proprio_action_manifest_v1.py",
        "scripts/dev_checkpoint_v1.py",
    } <= discovered
    assert discovered <= closed


def test_cpu_replay_render_execution_chain_is_source_closed() -> None:
    """Keep the selected live CPU/render graph inside the frozen source closure.

    The replay stack has optional and dormant imports (including training-only
    model paths), so this deliberately binds the concrete execution modules
    selected by the qualification rather than recursively traversing every
    import in the repository.
    """

    root = Path(__file__).resolve().parents[2]
    closed = set(contract.SOURCE_CLOSURE_DEFAULT_PATHS)
    required = {
        "lewm/__init__.py",
        "lewm/safety/__init__.py",
        "lewm/oracle/__init__.py",
        "scripts/render_replay_v03.py",
        "lewm_genesis/lewm_genesis/__init__.py",
        "lewm_genesis/lewm_genesis/batch_renderer.py",
        "lewm_genesis/lewm_genesis/camera_safety.py",
        "lewm_genesis/lewm_genesis/collectors/__init__.py",
        "lewm_genesis/lewm_genesis/collectors/base.py",
        "lewm_genesis/lewm_genesis/collectors/frontier.py",
        "lewm_genesis/lewm_genesis/collectors/ou_noise.py",
        "lewm_genesis/lewm_genesis/collectors/primitive_curriculum.py",
        "lewm_genesis/lewm_genesis/collectors/recovery.py",
        "lewm_genesis/lewm_genesis/collectors/route_teacher.py",
        "lewm_genesis/lewm_genesis/go2_adapter.py",
        "lewm_genesis/lewm_genesis/lewm_contract.py",
        "lewm_genesis/lewm_genesis/parity_checks.py",
        "lewm_genesis/lewm_genesis/render_replay.py",
        "lewm_genesis/lewm_genesis/rollout.py",
        "lewm_genesis/lewm_genesis/ros_msg_adapter.py",
        "lewm_genesis/lewm_genesis/scene_builder.py",
        "lewm_genesis/lewm_genesis/scene_loader.py",
        "lewm_genesis/lewm_genesis/textures.py",
        "lewm_worlds/lewm_worlds/__init__.py",
        "lewm_worlds/lewm_worlds/corpus.py",
        "lewm_worlds/lewm_worlds/exporters/__init__.py",
        "lewm_worlds/lewm_worlds/exporters/to_gazebo_sdf.py",
        "lewm_worlds/lewm_worlds/exporters/to_genesis.py",
        "lewm_worlds/lewm_worlds/families.py",
        "lewm_worlds/lewm_worlds/labels/__init__.py",
        "lewm_worlds/lewm_worlds/labels/derived.py",
        "lewm_worlds/lewm_worlds/labels/topology.py",
        "lewm_worlds/lewm_worlds/manifest.py",
        "lewm_worlds/lewm_worlds/planning_grid.py",
        "lewm_worlds/lewm_worlds/randomization.py",
        "lewm_worlds/lewm_worlds/scene_graph.py",
        "lewm_worlds/lewm_worlds/scene_validation.py",
        "lewm_worlds/lewm_worlds/splits.py",
    }
    assert required <= closed
    assert all((root / relative).is_file() for relative in required)

    selected_modules = {
        "scripts/evaluate_jepa_local_waypoint_planning_cost_qualification_v1.py",
        "scripts/run_go2_oracle_branch_pilot_v1_2.py",
        "lewm/oracle/go2_textured_v03_renderer.py",
        "lewm_genesis/lewm_genesis/rollout.py",
        "lewm_genesis/lewm_genesis/scene_builder.py",
        "lewm_genesis/lewm_genesis/scene_loader.py",
    }
    imported_local_modules: set[str] = set()
    for relative in selected_modules:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            module = node.module or ""
            candidates: list[str] = []
            if module in {"lewm_genesis", "lewm_worlds", "scripts"}:
                candidates.extend(
                    f"{module.replace('.', '/')}/{alias.name}.py"
                    for alias in node.names
                )
            elif module.startswith(("lewm_genesis.", "lewm_worlds.", "scripts.")):
                candidates.append(module.replace(".", "/") + ".py")
            for candidate in candidates:
                repository_candidate = candidate
                if candidate.startswith("lewm_genesis/"):
                    repository_candidate = "lewm_genesis/" + candidate
                elif candidate.startswith("lewm_worlds/"):
                    repository_candidate = "lewm_worlds/" + candidate
                if (root / repository_candidate).is_file():
                    imported_local_modules.add(repository_candidate)
    assert {
        "scripts/render_replay_v03.py",
        "lewm_genesis/lewm_genesis/render_replay.py",
        "lewm_genesis/lewm_genesis/scene_builder.py",
        "lewm_genesis/lewm_genesis/scene_loader.py",
        "lewm_worlds/lewm_worlds/labels/derived.py",
        "lewm_worlds/lewm_worlds/planning_grid.py",
        "lewm_worlds/lewm_worlds/randomization.py",
        "lewm_worlds/lewm_worlds/manifest.py",
        "lewm_worlds/lewm_worlds/scene_graph.py",
    } <= imported_local_modules
    assert imported_local_modules <= closed
