from __future__ import annotations

from collections import Counter
import copy
import json
import math

import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as C


def _segment_distance(a: list[list[float]], b: list[list[float]]) -> float:
    # Contract geometry places selected and competing openings on distinct
    # rectangle sides. Endpoint distance is a sufficient prospective witness
    # because none of those axis-frame segments crosses another.
    return min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for left in a for right in b
    )


def test_contract_counts_subjects_payloads_and_digest_roundtrip() -> None:
    assert C.EXPERIMENT_ID == "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1"
    assert C.CONTRACT_FREEZE_COMMIT_SUBJECT == "Freeze physical graph edge handoff qualification"
    assert C.RESULT_COMMIT_SUBJECT == "Evaluate physical graph edge handoff qualification"
    assert C.ROLE_COUNTS == {"DEVELOPMENT": 48, "DEVELOPMENT_HELDOUT": 16}
    assert C.BRANCH_COUNT == 768
    assert C.TEACHER_TRACE_COUNT == 256
    assert C.RESET_FIXTURE_PHYSICS_SAMPLES == 750
    assert C.CANDIDATE_TRACE_COUNT == 960
    assert C.RESET_TRACE_PAIR_COMPARISON_AUTHORITY["completion_termination_reason"] == "H3_COMPLETE"
    assert C.NPZ_PAYLOAD_AUTHORITY["teacher_traces.npz"]["trace_offsets"]["shape"] == [257]
    assert C.NPZ_PAYLOAD_AUTHORITY["candidate_traces.npz"]["trace_offsets"]["shape"] == [961]
    assert C.OUTPUT_LEAF_COUNT == 23
    assert len(C.RUNTIME_OUTPUT_PATHS) == 23
    assert len(C.TRACKED_SOURCE_PATHS) == len(set(C.TRACKED_SOURCE_PATHS)) == 13
    assert len(C.SOURCE_CLOSURE_PATHS) == len(set(C.SOURCE_CLOSURE_PATHS))
    assert not any(path.startswith("docs/") for path in C.SOURCE_CLOSURE_PATHS)
    assert len(C.SOURCE_DEPENDENCY_PATHS) == 58
    assert len(C.SOURCE_CLOSURE_PATHS) == 66
    assert "lewm_genesis/lewm_genesis/ros_msg_adapter.py" in C.SOURCE_DEPENDENCY_PATHS
    assert "lewm/__init__.py" in C.SOURCE_DEPENDENCY_PATHS
    contract = C.build_contract()
    assert contract["source_parent_commit"] == "3a784118b461d693d5dbf07035b3f85ee1553598"
    assert contract["source_baseline_commit"] == "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
    assert C.validate_contract(contract) == contract
    tampered = copy.deepcopy(contract)
    tampered["panel"]["state_count"] = 63
    with pytest.raises(C.PhysicalGraphEdgeHandoffContractError):
        C.validate_contract(tampered)
    runtime = C.build_runtime_contract("1" * 40)
    assert runtime["source_baseline_commit"] == C.SOURCE_BASELINE_COMMIT
    assert C.validate_runtime_contract(runtime, source_freeze_commit="1" * 40) == runtime
    persisted_runtime = json.loads(C.canonical_json_bytes(runtime))
    assert C.validate_runtime_contract(
        persisted_runtime, source_freeze_commit="1" * 40
    ) == persisted_runtime
    persisted_contract = json.loads(C.canonical_json_bytes(contract))
    assert C.validate_contract(persisted_contract) == persisted_contract


def test_prospective_pool_is_balanced_unique_and_role_free() -> None:
    specs = C.build_prospective_pool_specs()
    assert len(specs) == 256
    assert len({row["candidate_spec_id"] for row in specs}) == 256
    assert len({row["canonical_spec_sha256"] for row in specs}) == 256
    assert all(row["role"] is None for row in specs)
    assert Counter(row["family"] for row in specs) == Counter({family: 64 for family in C.FAMILY_IDS})
    assert Counter((row["family"], row["stratum_index"]) for row in specs) == Counter(
        {(family, stratum): 4 for family in C.FAMILY_IDS for stratum in range(16)}
    )
    # Variants alter relative geometry, not only global translation.
    for family in C.FAMILY_IDS:
        group = [row for row in specs if row["family"] == family and row["stratum_index"] == 0]
        relative = {
            (
                round(row["variant_adjustments"]["port_distance_delta_m"], 12),
                round(row["variant_adjustments"]["spawn_lateral_delta_m"], 12),
                round(row["variant_adjustments"]["spawn_yaw_delta_rad"], 12),
                round(row["variant_adjustments"]["corridor_length_delta_m"], 12),
                round(row["variant_adjustments"]["opening_lateral_delta_m"], 12),
            )
            for row in group
        }
        assert len(relative) == 4


def test_chamber_ports_are_nonoverlapping_and_body_direction_is_real() -> None:
    specs = C.build_candidate_specs()
    for spec in specs:
        geometry = spec["geometry"]
        witness = geometry["geometry_validity_witness"]
        assert witness["robot_start_inside_source"] is True
        assert witness["selected_port_endpoints_on_source_boundary"] is True
        assert witness["selected_and_competing_port_segments_are_on_distinct_boundary_sides"] is True
        assert witness["selected_corridor_outside_source_except_boundary"] is True
        assert witness["spawn_footprint_proxy_wall_clearance_m"] > 0.18
        selected = geometry["selected_directed_edge"]["opening_segment_world"]
        for competing in geometry["competing_directed_edges"]:
            assert _segment_distance(selected, competing["opening_segment_world"]) > 0.0
        bearing = witness["body_frame_selected_port_bearing_rad"]
        if spec["route_direction"] == "LEFT":
            assert bearing > 0.0
        elif spec["route_direction"] == "RIGHT":
            assert bearing < 0.0
    witnesses = {
        family: {
            row["geometry"]["family_geometry_witness"]
            for row in specs if row["family"] == family
        }
        for family in C.FAMILY_IDS
    }
    assert all(len(values) == 1 for values in witnesses.values())
    assert len({next(iter(values)) for values in witnesses.values()}) == 4


def test_camera_ranker_snapshot_and_runtime_authority_are_explicit() -> None:
    camera = C.GEOMETRY_AUTHORITY["camera"]
    assert camera["relative_position_body_m"] == [0.326, 0.0, 0.043]
    assert camera["horizontal_fov_deg"] == pytest.approx(78.323)
    assert camera["native_resolution_wh"] == [640, 480]
    assert camera["persisted_resolution_hw"] == [168, 224]
    assert "atan2(dy_m,dx_m)" in C.RANKER_INPUT_AUTHORITY["ranker_goal_projection"]
    snapshot = C.NPZ_PAYLOAD_AUTHORITY["state_snapshots.npz"]
    assert snapshot["snapshot_payload_bytes"]["slice_digest_domain"] == (
        "sha256_of_exact_serialized_snapshot_bytes"
    )
    assert snapshot["controller_observation"]["shape"] == [64, 45]
    assert snapshot["policy_last_action"]["shape"] == [64, 12]
    assert snapshot["command_history"]["shape"] == [64, 15, 3]
    assert snapshot["control_history"]["shape"] == [64, 15, 2]
    assert snapshot["low_level_policy_state"]["shape"] == [64, 12]
    assert C.LOW_LEVEL_CONTROLLER_AUTHORITY["rollout_foot_contact_source"] == "zero"
    assert C.COMMAND_TRACKING_AUTHORITY["command_ticks"] == 15
    assert C.COMMAND_TRACKING_AUTHORITY["physics_samples_per_command_tick"] == 50
    assert C.COMMAND_TRACKING_AUTHORITY["discard_initial_physics_samples_per_command_tick"] == 20
    preprocessing = C.CANONICAL_ENCODING_AUTHORITY["preprocessing"]
    assert preprocessing["input_shape"] == [168, 224, 3]
    assert preprocessing["output_shape"] == [3, 384, 512]
    assert preprocessing["output_dtype"] == "float32"
    assert "preprocess_array is not an available API" in preprocessing["transport"]
    assert C.CANONICAL_ENCODING_AUTHORITY["external_encoder_source"]["worktree_clean_including_untracked"] is True
    compatibility = C.RESET_RUNTIME_BINDING["runtime_compatibility"]
    assert compatibility["solver_runtime_package"] == "quadrants"
    assert compatibility["solver_runtime_version"] == "0.6.2"
    assert compatibility["compatibility_function"] == "_collect_solver_fields_compat"
    reset = C.RESET_RUNTIME_BINDING["graph_free_spawn_reset_compatibility"]
    assert reset["function"] == "_reset_robot_to_fixed_spawn_compat"
    assert reset["settling"]["physics_samples"] == 750
    assert C.DIRECT_RUNTIME_POLICY["required_environment_before_simulator_creation"]["PYTHONHASHSEED"] == "0"
    physical_runtime = C.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    assert physical_runtime["qualification_runtime_count"] == 256
    assert physical_runtime["selected_snapshot_runtime_count"] == 64
    assert physical_runtime["torch_version"] == "2.12.0+rocm7.2"
    assert C.RUNTIME_ENVIRONMENT_AUTHORITY["encoder"]["device_name"] == "AMD Radeon AI PRO R9700"
    assert C.RUNTIME_ENVIRONMENT_AUTHORITY["ranker"]["device"] == "cpu"
    assert len(C.EXTERNAL_ARTIFACT_BINDINGS) == 7
