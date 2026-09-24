from __future__ import annotations

import copy

import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v1_contract as C
from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as M


def _classification_input(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "teacher_correct_execution_count": 16,
        "coverage_rate": 1.0,
        "ranker_correct_edge_top1_rate": 0.80,
        "ranker_correct_edge_top3_rate": 0.95,
        "ranker_selected_correct_edge_execution_rate": 0.80,
        "ranker_normalized_port_regret": 0.20,
        "oracle_selected_correct_edge_execution_rate": 1.0,
        "oracle_covered_state_correct_execution_rate": 1.0,
        "repeatability_rate": 1.0,
        "command_tracking_pass": True,
        "minimum_family_correct_execution_count": 1,
        "selected_target_id": "TARGET_NODE_CENTRE",
        "selected_target_passes_handoff_gate": True,
        "selected_target_materially_outperforms_node_centre": False,
    }
    row.update(updates)
    return row


def test_reducer_and_trace_authorities_are_complete_and_self_digested() -> None:
    authority = M.reducer_authority()
    C.validate_content_digest(authority)
    assert authority["documents"]["teacher_trace_index"]["count"] == 256
    assert authority["ledgers"]["candidate_fanout"]["count"] == 768
    assert authority["ledgers"]["heldout_ranker_scores"]["count"] == 64
    assert authority["ledgers"]["repeated_execution"]["count"] == 64
    assert authority["reset_fixture_physics_samples"] == 750
    assert authority["branch_physics_samples"] == 750
    trace = M.physical_trace_reduction_authority()
    C.validate_content_digest(trace)
    assert len(trace["specs"]) == 256
    assert trace["endpoint_indices"] == {"H1": 249, "H2": 499, "H3": 749}
    assert trace["sustained_beyond_samples"] == 100
    assert "linearly interpolate" in trace["crossing_velocity"]
    assert trace["command_tracking_authority"] == C.COMMAND_TRACKING_AUTHORITY
    assert trace["command_tracking_authority"]["physics_samples_per_command_tick"] == 50
    assert trace["command_tracking_authority"]["discard_initial_physics_samples_per_command_tick"] == 20


def test_shared_polygon_and_transverse_crossing_helpers_have_exact_boundaries() -> None:
    polygon = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    assert M.point_in_polygon_inclusive([0.0, 0.0], polygon)
    assert M.point_in_polygon_inclusive([1.0, 0.0], polygon)
    assert not M.point_in_polygon_inclusive([1.01, 0.0], polygon)
    segment = [[0.0, -0.5], [0.0, 0.5]]
    crossing = M.transverse_port_crossing(
        [-0.1, 0.25], [0.1, 0.25], segment, [1.0, 0.0]
    )
    assert crossing is not None
    assert crossing["crossing_fraction"] == pytest.approx(0.5)
    assert crossing["crossing_point_world_xy"] == pytest.approx([0.0, 0.25])
    assert crossing["directed_normal_displacement_m"] == pytest.approx(0.2)
    assert crossing["lateral_fraction"] == pytest.approx(0.75)
    assert M.transverse_port_crossing(
        [0.1, 0.25], [-0.1, 0.25], segment, [1.0, 0.0]
    ) is None
    assert M.transverse_port_crossing(
        [-0.1, 0.75], [0.1, 0.75], segment, [1.0, 0.0]
    ) is None


def test_primary_classification_boundaries_and_composite_precedence() -> None:
    signal = M.classify_physical_handoff_aggregates(_classification_input())
    assert signal["primary_classification"] == "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL"

    target = M.classify_physical_handoff_aggregates(_classification_input(
        selected_target_id="DIRECTED_EDGE_PORT",
        selected_target_materially_outperforms_node_centre=True,
    ))
    assert target["primary_classification"] == "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT"

    # Fifteen covered states is >=.90. Conditional oracle success, not the
    # unconditional 15/16 rate, controls low-level attribution.
    adequate_15 = M.classify_physical_handoff_aggregates(_classification_input(
        coverage_rate=15 / 16,
        oracle_selected_correct_edge_execution_rate=15 / 16,
    ))
    assert adequate_15["primary_classification"] == "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL"

    coverage = M.classify_physical_handoff_aggregates(_classification_input(
        coverage_rate=14 / 16,
        oracle_selected_correct_edge_execution_rate=14 / 16,
        selected_target_passes_handoff_gate=False,
    ))
    assert coverage["primary_classification"] == "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO"

    ranker = M.classify_physical_handoff_aggregates(_classification_input(
        ranker_correct_edge_top1_rate=0.69,
        selected_target_passes_handoff_gate=False,
    ))
    assert ranker["primary_classification"] == "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO"

    low_level = M.classify_physical_handoff_aggregates(_classification_input(
        repeatability_rate=0.94,
        selected_target_passes_handoff_gate=False,
    ))
    assert low_level["primary_classification"] == "LOW_LEVEL_PREFIX_EXECUTION_NO_GO"

    composite = M.classify_physical_handoff_aggregates(_classification_input(
        coverage_rate=14 / 16,
        oracle_selected_correct_edge_execution_rate=14 / 16,
        command_tracking_pass=False,
        selected_target_passes_handoff_gate=False,
    ))
    assert composite["primary_classification"] == "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO"
    assert composite["active_components_in_precedence_order"] == [
        "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO", "LOW_LEVEL_PREFIX_EXECUTION_NO_GO"
    ]
    assert composite["next_experiment"] == "GRAPH_EDGE_LOCAL_ACTION_BANK_SUCCESSOR_V1"

    with pytest.raises(M.PhysicalGraphEdgeHandoffMetricsError, match="full-handoff"):
        M.classify_physical_handoff_aggregates(_classification_input(
            selected_target_passes_handoff_gate=False,
        ))


def _branch(index: int, *, correct: bool = False) -> dict[str, object]:
    return {
        "candidate_index": index,
        "oracle_admissible": correct,
        "entered_correct_edge": correct,
        "successor_viable": True,
        "positive_port_progress": correct,
        "physics_contact": False,
        "stuck": False,
        "entered_wrong_edge": False,
        "no_edge": not correct,
        "port_progress_m": 1.0 if correct else 0.0,
        "lateral_error_m": 0.0,
        "angular_error_rad": 0.0,
    }


def test_correct_candidate_requires_positive_progress_and_pairwise_ties_half_credit() -> None:
    branch = _branch(0, correct=True)
    assert M._correct_candidate(branch) is True
    branch["positive_port_progress"] = False
    assert M._correct_candidate(branch) is False

    fanout = [_branch(index, correct=index == 0) for index in range(12)]
    scores = [1.0, 1.0, *([0.0] * 10)]
    ranking = M._rank(scores)
    expected_pairwise = 10.5 / 11.0
    row = {
        "candidate_ids": list(C.CANDIDATE_IDS),
        "scores": scores,
        "ranking": ranking,
        "eligible_correct_edge_candidate_indices": [0],
        "selected_candidate_index": 0,
        "correct_edge_top1": True,
        "correct_edge_top3": True,
        "correct_edge_mrr": 1.0,
        "selected_correct_edge_execution": True,
        "selected_port_progress_m": 1.0,
        "oracle_best_port_progress_m": 1.0,
        "minimum_admissible_port_progress_m": 1.0,
        "normalized_port_regret": 0.0,
        "pairwise_correct_edge_ordering": expected_pairwise,
        "selected_wrong_edge": False,
        "selected_no_edge": False,
        "selected_lateral_error_m": 0.0,
        "selected_angular_error_rad": 0.0,
        "selected_contact": False,
        "selected_stuck": False,
        "selected_successor_viable": True,
    }
    projection = M._selection_projection(row, fanout)
    assert projection["pairwise_correct_edge_ordering"] == pytest.approx(expected_pairwise)


def _npz_inspections() -> list[dict[str, object]]:
    symbol_values = {"B": 64, "T": 512, "U": 64, "P": 960 * 750}
    result: list[dict[str, object]] = []
    for path, members in C.NPZ_PAYLOAD_AUTHORITY.items():
        observed_members: dict[str, object] = {}
        if path == "state_snapshots.npz":
            offsets = list(range(65))
        elif path == "teacher_traces.npz":
            offsets = [2 * index for index in range(257)]
        elif path == "candidate_traces.npz":
            offsets = [750 * index for index in range(961)]
        else:
            offsets = []
        for member_name, spec in members.items():
            shape = [symbol_values.get(value, value) for value in spec["shape"]]
            mode = spec["hash_mode"]
            if mode == "whole":
                count = 1
            elif mode == "rows_axis0":
                count = shape[0]
            else:
                count = len(offsets) - 1
            observed_members[member_name] = {
                "descr": spec["descr"],
                "digest_dtype": spec["digest_dtype"],
                "shape": shape,
                "c_contiguous": True,
                "object_dtype": False,
                "member_sha256": "1" * 64,
                "row_or_slice_sha256s": ["2" * 64] * count,
                "offset_values": offsets if member_name.endswith("offsets") else None,
            }
        result.append({
            "path": path, "bytes": 1, "sha256": "3" * 64,
            "members": observed_members,
        })
    return result


def test_npz_authority_enforces_all_960_full_horizon_trace_slices() -> None:
    inspections = _npz_inspections()
    validated = M.validate_npz_inspections(inspections)
    assert validated["candidate_traces.npz"]["members"]["trace_offsets"]["shape"] == [961]
    tampered = copy.deepcopy(inspections)
    candidate = next(row for row in tampered if row["path"] == "candidate_traces.npz")
    candidate["members"]["trace_offsets"]["offset_values"][1] = 749
    with pytest.raises(M.PhysicalGraphEdgeHandoffMetricsError, match="duration"):
        M.validate_npz_inspections(tampered)


def test_port_helpers_freeze_tolerance_and_exact_selected_competing_tie() -> None:
    segment = [[0.0, -0.5], [0.0, 0.5]]
    normal = [1.0, 0.0]
    crossing = M.transverse_port_crossing(
        [-0.1, 0.5 + 0.5 * C.NUMERICAL_TOLERANCES["se2_position_m"]],
        [0.1, 0.5 + 0.5 * C.NUMERICAL_TOLERANCES["se2_position_m"]],
        segment,
        normal,
    )
    assert crossing is not None
    assert crossing["lateral_fraction"] == 1.0
    first = M.first_registered_port_crossing(
        [[-0.1, 0.0], [0.1, 0.0]],
        {
            "edge_id": "selected-edge",
            "opening_segment_world": segment,
            "opening_normal_world": normal,
        },
        [{
            "edge_id": "competing-edge-0",
            "opening_segment_world": segment,
            "opening_normal_world": normal,
        }],
    )
    assert first is not None
    assert first["edge_id"] == "competing-edge-0"
    assert first["is_selected_edge"] is False


def _real_visual_runtime(role: str) -> dict[str, object]:
    authority = C.RUNTIME_ENVIRONMENT_AUTHORITY[role]
    return {
        "stage_id": authority["stage_id"],
        "python_executable": authority["real_python_executable"],
        "python_version": authority["python_version"],
        "torch_version": authority["torch_version"],
        "torch_hip_version": authority["torch_hip_version"],
        "visible_device_count": authority["visible_device_count"],
        "device": authority["device"],
        "device_name": authority["device_name"],
        "device_capability": authority["device_capability"],
        "backend": authority["backend"],
        "fake_runtime": False,
        "model_role": authority["model_role"],
        "checkpoint_sha256": authority["checkpoint_sha256"],
        "model_source_path": authority["model_source_path"],
        "model_source_sha256": authority["model_source_sha256"],
        "external_repository_path": authority["external_repository_path"],
        "external_repository_commit": authority["external_repository_commit"],
        "external_worktree_clean": authority["external_worktree_clean"],
    }


def test_observed_runtime_projections_are_exact_and_cross_digest_bound() -> None:
    authority = C.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    core = {
        "stage_id": authority["stage_id"],
        "python_executable": authority["real_python_executable"],
        "python_version": authority["python_version"],
        "torch_version": authority["torch_version"],
        "torch_hip_version": authority["torch_hip_version"],
        "genesis_version": authority["genesis_version"],
        "quadrants_version": authority["quadrants_version"],
        "visible_device_count": authority["visible_device_count"],
        "device": authority["device"],
        "backend": authority["backend"],
        "deterministic_environment": copy.deepcopy(
            C.DIRECT_RUNTIME_POLICY["required_environment_before_simulator_creation"]
        ),
        "fake_runtime": False,
    }
    digest = M.runtime_environment_sha256(core)
    physical = {
        **core,
        "runtime_core_sha256": digest,
        "qualification_runtime_sha256s": [digest] * C.TEACHER_TRACE_COUNT,
        "selected_snapshot_runtime_sha256s": [digest] * C.STATE_COUNT,
    }
    assert M.validate_physical_runtime_environment(physical) == physical
    assert M.validate_visual_runtime_environment(
        _real_visual_runtime("encoder"), runtime_role="encoder"
    )["fake_runtime"] is False
    assert M.validate_visual_runtime_environment(
        _real_visual_runtime("ranker"), runtime_role="ranker"
    )["device"] == "cpu"
    tampered = copy.deepcopy(physical)
    tampered["qualification_runtime_sha256s"][17] = "0" * 64
    with pytest.raises(M.PhysicalGraphEdgeHandoffMetricsError, match="divergent shard"):
        M.validate_physical_runtime_environment(tampered)
    wrong_ranker = _real_visual_runtime("ranker")
    wrong_ranker["torch_version"] = "drift"
    with pytest.raises(M.PhysicalGraphEdgeHandoffMetricsError, match="torch_version"):
        M.validate_visual_runtime_environment(wrong_ranker, runtime_role="ranker")
