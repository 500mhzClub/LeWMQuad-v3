from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v4_contract as C
from lewm.safety import physical_graph_edge_handoff_qualification_v4_metrics as M


ZERO_FLAGS = {name: False for name in C.TERMINATION_FLAG_ORDER}
IDENTITY = {
    "artifact_file_sha256": "1" * 64,
    "snapshot_semantic_digest_v1": "2" * 64,
    "snapshot_behavioural_digest_v1": "3" * 64,
}
SPECS = C.build_candidate_specs()


def _criteria(**changes: bool) -> dict[str, bool]:
    result = {name: True for name in C.TEACHER_QUALIFICATION_COMPONENT_IDS}
    result.update(changes)
    return result


def _state(index: int, kind: str = "qualified") -> dict:
    spec = SPECS[index]
    kwargs = dict(
        stage_reached="COMPLETE",
        initial_termination_flags=ZERO_FLAGS,
        probe_trial_termination_flags=[ZERO_FLAGS, ZERO_FLAGS],
        teacher_termination_flags=ZERO_FLAGS,
        probe_tip_sample_indices=[None, None],
        teacher_criteria=_criteria(),
        executable_snapshot_exists=True,
        teacher_executed=True,
        snapshot_identity=IDENTITY,
        diagnostics_inventory=[],
        payload_member_inventory=["snapshot_payload_bytes"],
    )
    if kind == "initial_tipped":
        kwargs.update(
            stage_reached="INITIAL_BOUNDARY",
            initial_termination_flags={**ZERO_FLAGS, "tipped": True},
            probe_trial_termination_flags=None,
            probe_tip_sample_indices=None,
            teacher_termination_flags=None,
            teacher_criteria=None,
            executable_snapshot_exists=False,
            teacher_executed=False,
            snapshot_identity=None,
            diagnostics_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
            payload_member_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
        )
    elif kind == "probe_tipped":
        kwargs.update(
            stage_reached="RESTORATION_PROBE",
            probe_trial_termination_flags=[
                {**ZERO_FLAGS, "tipped": True}, {**ZERO_FLAGS, "tipped": True}
            ],
            probe_tip_sample_indices=[99, 99],
            teacher_termination_flags=None,
            teacher_criteria=None,
            teacher_executed=False,
            snapshot_identity={**IDENTITY, "snapshot_behavioural_digest_v1": None},
        )
    elif kind == "contact":
        kwargs["stage_reached"] = "TEACHER_EXECUTION"
        kwargs["teacher_criteria"] = _criteria(teacher_trace_contact_free=False)
    elif kind == "crossing":
        kwargs["stage_reached"] = "TEACHER_EXECUTION"
        kwargs["teacher_criteria"] = _criteria(teacher_crossed_directed_port=False)
    elif kind == "leave":
        kwargs["stage_reached"] = "TEACHER_EXECUTION"
        kwargs["teacher_criteria"] = _criteria(teacher_left_source_region=False)
    elif kind == "progress":
        kwargs["stage_reached"] = "TEACHER_EXECUTION"
        kwargs["teacher_criteria"] = _criteria(teacher_positive_route_progress=False)
    elif kind == "unsafe":
        kwargs["stage_reached"] = "TEACHER_EXECUTION"
        kwargs["teacher_termination_flags"] = {**ZERO_FLAGS, "fall": True}
    elif kind == "material_corrupt":
        kwargs.update(
            stage_reached="INITIAL_BOUNDARY", probe_trial_termination_flags=None,
            probe_tip_sample_indices=None,
            teacher_termination_flags=None, teacher_criteria=None,
            executable_snapshot_exists=False, teacher_executed=False,
            snapshot_identity=None, materialisation_corrupt=True,
        )
    elif kind == "nondeterministic":
        kwargs.update(
            stage_reached="RESTORATION_PROBE",
            probe_trial_termination_flags=[ZERO_FLAGS, {**ZERO_FLAGS, "tipped": True}],
            teacher_termination_flags=None, teacher_criteria=None,
            teacher_executed=False, state_nondeterministic=True,
            snapshot_identity={**IDENTITY, "snapshot_behavioural_digest_v1": None},
        )
    elif kind == "unresolved":
        kwargs["stage_reached"] = "TEACHER_EXECUTION"
        kwargs["teacher_criteria"] = _criteria(goal_reachable=False)
    elif kind != "qualified":
        raise AssertionError(kind)
    return M.build_state_disposition_record(spec, **kwargs)


def _official(state: dict) -> dict:
    index = state["pool_index"]
    return M.build_qualification_disposition_row(
        state,
        material_metadata_binding={
            "path": f"qualification/pool-{index:03d}/metadata.json",
            "bytes": 100 + index,
            "sha256": hashlib.sha256(f"metadata-{index}".encode()).hexdigest(),
        },
        material_payload_binding={
            "path": f"qualification/pool-{index:03d}/payload.npz",
            "bytes": 200 + index,
            "sha256": hashlib.sha256(f"payload-{index}".encode()).hexdigest(),
        },
        persisted_array_evidence_sha256=hashlib.sha256(f"arrays-{index}".encode()).hexdigest(),
    )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("qualified", "QUALIFIED"),
        ("initial_tipped", "INITIAL_BOUNDARY_TIPPED"),
        ("probe_tipped", "RESTORATION_PROBE_TIPPED"),
        ("contact", "TEACHER_PHYSICS_CONTACT"),
        ("crossing", "TEACHER_CROSSING_INVALID"),
        ("leave", "TEACHER_DID_NOT_LEAVE_SOURCE"),
        ("progress", "TEACHER_NO_POSITIVE_PROGRESS"),
        ("unsafe", "TEACHER_TERMINATED_UNSAFELY"),
        ("material_corrupt", "STATE_MATERIALISATION_CORRUPT"),
        ("nondeterministic", "STATE_NONDETERMINISTIC"),
        ("unresolved", "UNRESOLVED_STATE_FAILURE"),
    ],
)
def test_state_disposition_taxonomy(kind: str, expected: str) -> None:
    row = _state(0, kind)
    assert row["disposition"] == expected
    assert row["qualified"] is (expected == "QUALIFIED")
    assert row["hard_stop"] is (expected in C.HARD_STOP_DISPOSITIONS)
    assert row["continuation_authorized"] is (expected not in C.HARD_STOP_DISPOSITIONS)
    assert M.validate_state_disposition_record(row, expected_pool_index=0) == row


def test_precedence_preserves_all_failed_criteria() -> None:
    disposition, failed = M.classify_state_disposition(
        initial_termination_flags=ZERO_FLAGS,
        probe_trial_termination_flags=[ZERO_FLAGS, ZERO_FLAGS],
        teacher_termination_flags=ZERO_FLAGS,
        teacher_criteria=_criteria(
            teacher_trace_contact_free=False,
            teacher_crossed_directed_port=False,
            teacher_left_source_region=False,
            teacher_positive_route_progress=False,
        ),
    )
    assert disposition == "TEACHER_PHYSICS_CONTACT"
    assert failed == [
        "teacher_trace_contact_free", "teacher_left_source_region",
        "teacher_crossed_directed_port", "teacher_positive_route_progress",
    ]

    disposition, failed = M.classify_state_disposition(
        initial_termination_flags={
            "fall": True, "out_of_bounds": False, "tipped": True, "nan": True,
        },
    )
    assert disposition == "INITIAL_BOUNDARY_TIPPED"
    assert failed == ["initial_boundary_tipped", "unresolved_state_failure"]


def test_nondeterministic_probe_is_hard_stop_not_tipped_rejection() -> None:
    row = _state(1, "nondeterministic")
    assert row["disposition"] == "STATE_NONDETERMINISTIC"
    assert "restoration_probe_tipped" in row["failed_criteria"]
    assert row["continuation_authorized"] is False


def test_stage_boundary_rejects_future_evidence_and_checks_initial_unresolved() -> None:
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.build_state_disposition_record(
            SPECS[0], stage_reached="INITIAL_BOUNDARY",
            initial_termination_flags={**ZERO_FLAGS, "tipped": True},
            probe_trial_termination_flags=[ZERO_FLAGS, ZERO_FLAGS],
            executable_snapshot_exists=False, teacher_executed=False,
            diagnostics_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
            payload_member_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
        )
    unresolved = M.build_state_disposition_record(
        SPECS[0], stage_reached="INITIAL_BOUNDARY",
        initial_termination_flags={**ZERO_FLAGS, "fall": True},
        executable_snapshot_exists=False, teacher_executed=False,
        unresolved_state_failure=True,
        diagnostics_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
        payload_member_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
    )
    assert unresolved["disposition"] == "UNRESOLVED_STATE_FAILURE"
    tampered = copy.deepcopy(unresolved)
    tampered.pop("content_digest")
    tampered["payload_member_inventory"] = []
    tampered = C.attach_content_digest(tampered)
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_state_disposition_record(tampered)


def test_repeated_rejection_record_generation_is_byte_identical() -> None:
    one = _state(3, "initial_tipped")
    two = _state(3, "initial_tipped")
    assert C.canonical_json_bytes(one) == C.canonical_json_bytes(two)


@pytest.mark.parametrize("family", C.V3.FAMILY_IDS)
def test_non_scientific_family_fixture_reaches_normal_classification(family: str) -> None:
    index = next(i for i, spec in enumerate(SPECS) if spec["family"] == family)
    row = _state(index, "qualified")
    assert row["family"] == family
    assert row["disposition"] == "QUALIFIED"


def _all_official(kind_by_index: dict[int, str] | None = None) -> list[dict]:
    kinds = {} if kind_by_index is None else kind_by_index
    return [_official(_state(index, kinds.get(index, "qualified"))) for index in range(256)]


def test_complete_adequate_panel_selects_exactly_64_and_roles_remain_frozen() -> None:
    rows = _all_official()
    panel = M.build_panel_adequacy(rows)
    assert panel["adequate"] is True
    assert panel["status"] == "ADEQUATE"
    assert panel["panel_state_count"] == 64
    assert len(panel["selected_pool_indices"]) == 64
    assert panel["family_role_counts_if_adequate"] == C.V3.FAMILY_ROLE_COUNTS
    assert panel["offset_opening_supplies_any_qualified_state"] is True
    assert M.validate_panel_adequacy(panel, rows) == panel


def test_inadequate_panel_is_legitimate_terminal_and_opens_no_downstream() -> None:
    rows = _all_official({index: "initial_tipped" for index in range(128, 132)})
    panel = M.build_panel_adequacy(rows)
    assert panel["adequate"] is False
    assert panel["status"] == C.PANEL_INADEQUATE_DISPOSITION
    assert panel["shortfall_strata"] == [{"family": "OFFSET_OPENING", "stratum_index": 0}]
    assert panel["selected_pool_indices"] == []
    assert panel["panel_state_count"] == 0
    assert panel["next_decision"] == C.NEXT_DECISION_PANEL_INADEQUATE
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.build_panel_adequacy(rows, downstream_outcomes_opened=1)


def test_hard_stop_cannot_be_relabelled_as_panel_inadequacy() -> None:
    rows = _all_official({0: "material_corrupt"})
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4HardStop):
        M.build_panel_adequacy(rows)


def test_jsonl_is_canonical_complete_and_tamper_evident() -> None:
    rows = _all_official()
    payload = M.build_qualification_state_dispositions_jsonl(rows)
    assert len(payload.splitlines()) == 256
    assert M.validate_qualification_state_dispositions_jsonl(payload) == rows
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_qualification_state_dispositions_jsonl(payload[:-1])


def _tipped_trace(samples: int = 11) -> dict:
    result: dict = {}
    for member, authority in C.V4_PROBE_TRACE_MEMBER_AUTHORITY.items():
        shape = [samples, *authority["shape"][1:]]
        dtype = np.dtype(authority["descr"])
        result[member] = np.zeros(shape, dtype=dtype)
    result["timestamp_s"] = np.arange(samples, dtype=np.float64) * 0.002
    result["requested_command"][:, 0] = C.V3.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND[0]
    result.update(
        {
            "termination_reason": "TIPPED",
            "stuck": False,
            "termination_flags": {**ZERO_FLAGS, "tipped": True},
            "tip_sample_index": samples - 1,
        }
    )
    return result


def test_variable_length_tipped_probe_comparison_and_mutation() -> None:
    left = _tipped_trace()
    right = copy.deepcopy(left)
    assert M.compare_tipped_behavioural_probe_traces(left, right)["pass"] is True
    right["base_twist_world"][0, 0] = 1.0
    assert M.compare_tipped_behavioural_probe_traces(left, right)["pass"] is False
    shorter = _tipped_trace(10)
    assert M.compare_tipped_behavioural_probe_traces(left, shorter)["pass"] is False


def test_matching_non_tipped_partial_probe_is_unresolved_not_tipped() -> None:
    left = _tipped_trace()
    left["termination_reason"] = "OUT_OF_BOUNDS"
    left["termination_flags"] = {**ZERO_FLAGS, "out_of_bounds": True}
    right = copy.deepcopy(left)
    comparison = M.compare_terminated_behavioural_probe_traces(left, right)
    assert comparison["pass"] is True
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.compare_tipped_behavioural_probe_traces(left, right)
    right["termination_flags"] = {**ZERO_FLAGS, "fall": True}
    right["termination_reason"] = "FALL"
    assert M.compare_terminated_behavioural_probe_traces(left, right)["pass"] is False


def test_nan_terminal_probe_requires_final_exact_ieee_pattern() -> None:
    left = _tipped_trace()
    left["termination_reason"] = "NAN"
    left["termination_flags"] = {**ZERO_FLAGS, "nan": True}
    left["base_pose_world"][-1, 0] = np.asarray(
        [0x7FF8000000000001], dtype=np.uint64
    ).view(np.float64)[0]
    right = copy.deepcopy(left)
    equal = M.compare_terminated_behavioural_probe_traces(left, right)
    assert equal["nonfinite_masks_equal"] is True
    assert equal["nonfinite_bit_patterns_equal"] is True
    assert equal["pass"] is True

    different_payload = copy.deepcopy(right)
    different_payload["base_pose_world"][-1, 0] = np.asarray(
        [0x7FF8000000000002], dtype=np.uint64
    ).view(np.float64)[0]
    mismatch = M.compare_terminated_behavioural_probe_traces(
        left, different_payload
    )
    assert mismatch["nonfinite_masks_equal"] is True
    assert mismatch["nonfinite_bit_patterns_equal"] is False
    assert mismatch["pass"] is False

    early = copy.deepcopy(left)
    early["base_pose_world"][-2, 0] = np.nan
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.compare_terminated_behavioural_probe_traces(early, right)
    false_flag = copy.deepcopy(left)
    false_flag["termination_flags"] = ZERO_FLAGS
    false_flag["termination_reason"] = C.PARTIAL_PROBE_NO_FLAG_TERMINATION_REASON
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.compare_terminated_behavioural_probe_traces(false_flag, right)


def test_initial_tipped_material_payload_binds_exact_pose_flags_dtypes_and_values() -> None:
    state = _state(0, "initial_tipped")
    spec = SPECS[0]
    x, y, yaw = spec["geometry"]["spawn_se2_world"]
    arrays = {
        "intended_base_pose_world": np.asarray([x, y, 0.375, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float64),
        "base_pose_world": np.asarray(
            [x, y, 0.375, math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)],
            dtype=np.float64,
        ),
        "base_twist_world": np.zeros(6, dtype=np.float64),
        "joint_position": np.zeros(12, dtype=np.float64),
        "joint_velocity": np.zeros(12, dtype=np.float64),
        "previous_applied_command": np.zeros(3, dtype=np.float64),
        "physics_contact": np.zeros(1, dtype=np.uint8),
        "sim_time_ns": np.zeros(1, dtype=np.int64),
        "episode_step": np.zeros(1, dtype=np.int64),
        "command_ticks": np.zeros(1, dtype=np.int64),
        "policy_steps": np.zeros(1, dtype=np.int64),
        "termination_flags": np.asarray([0, 0, 1, 0], dtype=np.uint8),
    }
    evidence = M.build_persisted_array_evidence(
        shard_kind="qualification", shard_id="pool-000",
        payload_file={"path": "qualification/pool-000/payload.npz", "bytes": 123, "sha256": "a" * 64},
        arrays=arrays, reopened_arrays=copy.deepcopy(arrays),
    )
    assert M.validate_state_material_payload(state, evidence, reopened_arrays=arrays)["state_disposition"] == state
    mutated = copy.deepcopy(arrays)
    mutated["intended_base_pose_world"][0] += 0.01
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_state_material_payload(state, evidence, reopened_arrays=mutated)


def test_initial_nan_is_persisted_unresolved_with_exact_raw_boundary() -> None:
    spec = SPECS[0]
    x, y, yaw = spec["geometry"]["spawn_se2_world"]
    state = M.build_state_disposition_record(
        spec,
        stage_reached="INITIAL_BOUNDARY",
        initial_termination_flags={**ZERO_FLAGS, "nan": True},
        executable_snapshot_exists=False,
        teacher_executed=False,
        unresolved_state_failure=True,
        diagnostics_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
        payload_member_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
    )
    arrays = {
        "intended_base_pose_world": np.asarray(
            [x, y, 0.375, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
            dtype=np.float64,
        ),
        "base_pose_world": np.asarray(
            [np.nan, y, 0.375, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
            dtype=np.float64,
        ),
        "base_twist_world": np.zeros(6, dtype=np.float64),
        "joint_position": np.zeros(12, dtype=np.float64),
        "joint_velocity": np.zeros(12, dtype=np.float64),
        "previous_applied_command": np.zeros(3, dtype=np.float64),
        "physics_contact": np.zeros(1, dtype=np.uint8),
        "sim_time_ns": np.zeros(1, dtype=np.int64),
        "episode_step": np.zeros(1, dtype=np.int64),
        "command_ticks": np.zeros(1, dtype=np.int64),
        "policy_steps": np.zeros(1, dtype=np.int64),
        "termination_flags": np.asarray([0, 0, 0, 1], dtype=np.uint8),
    }
    evidence = M.build_persisted_array_evidence(
        shard_kind="qualification", shard_id="pool-000",
        payload_file={
            "path": "qualification/pool-000/payload.npz",
            "bytes": 123, "sha256": "a" * 64,
        },
        arrays=arrays, reopened_arrays=copy.deepcopy(arrays),
    )
    assert M.validate_state_material_payload(
        state, evidence, reopened_arrays=arrays
    )["state_disposition"] == state
    observed_state_nan = copy.deepcopy(arrays)
    observed_state_nan["base_twist_world"][0] = np.inf
    observed_state_nan["joint_velocity"][0] = np.nan
    observed_state_evidence = M.build_persisted_array_evidence(
        shard_kind="qualification", shard_id="pool-000",
        payload_file={
            "path": "qualification/pool-000/payload.npz",
            "bytes": 123, "sha256": "b" * 64,
        },
        arrays=observed_state_nan,
        reopened_arrays=copy.deepcopy(observed_state_nan),
    )
    assert M.validate_state_material_payload(
        state, observed_state_evidence, reopened_arrays=observed_state_nan,
    )["state_disposition"] == state
    off_path = copy.deepcopy(arrays)
    off_path["previous_applied_command"][0] = np.inf
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_state_material_payload(
            state, evidence, reopened_arrays=off_path
        )


def test_nan_flag_does_not_erase_cooccurring_frozen_boundary_flags() -> None:
    pose = np.asarray(
        [np.nan, 0.0, 0.1, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        dtype=np.float64,
    )
    assert M._teacher_termination_flags_from_pose(pose) == {
        "fall": True,
        "out_of_bounds": False,
        "tipped": True,
        "nan": True,
    }


@pytest.mark.parametrize(
    ("point", "segment", "expected_hex"),
    [
        (
            [0.23387508547129157, 0.05499406737182661],
            [
                [0.3887954107122013, -0.148793657899766],
                [-0.06509439358982835, 0.4482691910120259],
            ],
            "-0x1.e7795b3662ac9p-4",
        ),
        (
            [0.2431894711552432, -0.22328484498284906],
            [
                [-0.08892477942359078, -0.6601597840877913],
                [0.6372989074596567, 0.29514077417107576],
            ],
            "-0x1.a397ef76cb898p-5",
        ),
        (
            [0.28336348910215864, -0.14553272122046151],
            [
                [-0.09368400649916991, -0.6415144545950767],
                [0.6325396803840775, 0.31378610366379034],
            ],
            "0x1.794668f9195b0p-6",
        ),
    ],
)
def test_binary64_planar_projection_matches_frozen_numpy2_producer_bytes(
    point: list[float], segment: list[list[float]], expected_hex: str,
) -> None:
    one = M.canonical_v1_planar_segment_projection(point, segment)
    two = M.canonical_v1_planar_segment_projection(point, segment)
    assert one == two
    assert tuple(one) == C.V4_BINARY64_PLANAR_SEGMENT_PROJECTION_FIELDS
    assert one["lateral_coordinate_m"].hex() == expected_hex


def test_binary64_projection_covers_dependent_teacher_record_fields() -> None:
    segment = [
        [0.3887954107122013, -0.148793657899766],
        [-0.06509439358982835, 0.4482691910120259],
    ]
    crossing = M.canonical_v1_planar_segment_projection(
        [0.23387508547129157, 0.05499406737182661], segment,
    )
    assert crossing["segment_width_m"] == 0.7500000000000001
    assert (
        crossing["lateral_coordinate_m"] / crossing["segment_width_m"] + 0.5
    ).hex() == "0x1.5d8236eddf1bep-2"
    endpoint = M.canonical_v1_planar_segment_projection(
        [0.2846214771270752, 0.177300825715065],
        [
            [0.4173925601615939, -0.1839959249665702],
            [-0.036497244140435786, 0.4130669239452217],
        ],
    )
    assert abs(endpoint["lateral_coordinate_m"]).hex() == (
        "0x1.cc7902d115d36p-8"
    )
    assert M.frozen_binary64_fma(-0.09474369918430334, 0.7960837985490559,
                                 0.07202457691010508 * -0.6051864057360394).hex() == (
        "-0x1.e7795b3662ac9p-4"
    )
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.canonical_v1_planar_segment_projection([math.nan, 0.0], segment)


def _nan_terminal_teacher_fixture(
    sample_count: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    spec = SPECS[0]
    arrays: dict[str, np.ndarray] = {}
    byte_members = {
        "physics_contact", "source_region_member", "edge_region_member",
        "target_region_member",
    }
    tails = {
        "base_pose_world": (7,), "base_twist_world": (6,),
        "joint_position": (12,), "joint_velocity": (12,),
        "applied_command": (3,), "requested_command": (3,),
    }
    for member in C.TEACHER_TRACE_MEMBER_ORDER:
        arrays[f"teacher__{member}"] = np.zeros(
            (sample_count, *tails.get(member, ())),
            dtype=np.uint8 if member in byte_members else np.float64,
        )
    arrays["teacher__timestamp_s"][:] = (
        np.arange(1, sample_count + 1, dtype=np.float64) * C.TEACHER_TRACE_DT_S
    )
    spawn_x, spawn_y, spawn_yaw = spec["geometry"]["spawn_se2_world"]
    anchor = np.asarray(
        [
            spawn_x, spawn_y, 0.375, 0.0, 0.0,
            math.sin(spawn_yaw / 2.0), math.cos(spawn_yaw / 2.0),
        ],
        dtype=np.float64,
    )
    arrays["teacher__base_pose_world"][:] = anchor
    finite_stop = sample_count - 1
    tolerance = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    polygons = {
        "source_region_member": spec["geometry"]["source_node"][
            "boundary_polygon_world"
        ],
        "edge_region_member": spec["geometry"]["selected_directed_edge"][
            "edge_region_polygon_world"
        ],
        "target_region_member": spec["geometry"]["target_node"][
            "boundary_polygon_world"
        ],
    }
    finite_memberships: dict[str, np.ndarray] = {}
    for member, polygon in polygons.items():
        membership = int(
            M.point_in_polygon_inclusive(
                anchor[:2].tolist(), polygon, tolerance_m=tolerance,
            )
        )
        arrays[f"teacher__{member}"][:finite_stop] = membership
        finite_memberships[member] = np.full(
            max(finite_stop, 1), membership, dtype=np.uint8,
        )
    arrays["teacher__base_pose_world"][-1, 0] = np.asarray(
        [0x7FF8000000000001], dtype=np.uint64,
    ).view(np.float64)[0]
    if sample_count == 1:
        arrays["snapshot__base_pose_world"] = anchor.copy()
    left_source = bool((finite_memberships["source_region_member"] == 0).any())
    criteria = {
        "goal_reachable": True,
        "teacher_trace_contact_free": True,
        "teacher_left_source_region": left_source,
        "teacher_crossed_directed_port": False,
        "teacher_positive_route_progress": False,
        "teacher_no_competing_port": True,
        "teacher_normal_positive": False,
        "teacher_within_lateral_bounds": False,
        "teacher_dwell_satisfied": False,
        "directed_port_defined": False,
        "current_rgb_valid": True,
        "graph_edge_physically_executable": True,
    }
    state = M.build_state_disposition_record(
        spec,
        stage_reached="TEACHER_EXECUTION",
        initial_termination_flags=ZERO_FLAGS,
        probe_trial_termination_flags=[ZERO_FLAGS, ZERO_FLAGS],
        teacher_termination_flags={**ZERO_FLAGS, "nan": True},
        probe_tip_sample_indices=[None, None],
        teacher_criteria=criteria,
        executable_snapshot_exists=True,
        teacher_executed=True,
        snapshot_identity=IDENTITY,
        diagnostics_inventory=[],
        payload_member_inventory=sorted(arrays),
    )
    metadata = {
        "pool_index": 0,
        "candidate_spec": spec,
        "graph": M._expected_registered_graph(
            spec, positive_route_progress=False,
            competing_port_entered=False,
        ),
        "state_disposition": state,
    }
    return metadata, arrays


@pytest.mark.parametrize("sample_count", [1, 2])
def test_nan_terminal_teacher_reduces_only_finite_prefix_or_snapshot_anchor(
    sample_count: int,
) -> None:
    metadata, arrays = _nan_terminal_teacher_fixture(sample_count)
    reduced = M.reduce_qualification_teacher_trace(
        metadata, arrays, expected_pool_index=0,
    )
    assert reduced["termination_flags"] == {**ZERO_FLAGS, "nan": True}
    assert reduced["disposition"] == "TEACHER_TERMINATED_UNSAFELY"
    assert reduced["route_progress_m"] == 0.0
    assert reduced["crossing"] is None
    assert reduced["teacher_record_raw_projection"]["successor_viable"] is False

    preterminal = {name: value.copy() for name, value in arrays.items()}
    if sample_count == 2:
        preterminal["teacher__joint_position"][0, 0] = np.nan
        with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
            M.reduce_qualification_teacher_trace(metadata, preterminal)
    unlisted = {name: value.copy() for name, value in arrays.items()}
    unlisted["teacher__requested_command"][-1, 0] = np.nan
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.reduce_qualification_teacher_trace(metadata, unlisted)


def test_nan_terminal_teacher_membership_uses_finite_xy_or_undefined_zero() -> None:
    metadata, arrays = _nan_terminal_teacher_fixture(2)
    spec = metadata["candidate_spec"]
    terminal = arrays["teacher__base_pose_world"][-1]
    terminal[0] = arrays["teacher__base_pose_world"][0, 0]
    terminal[2] = np.asarray(
        [0x7FF80000000000A5], dtype=np.uint64,
    ).view(np.float64)[0]
    tolerance = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    polygons = {
        "source_region_member": spec["geometry"]["source_node"][
            "boundary_polygon_world"
        ],
        "edge_region_member": spec["geometry"]["selected_directed_edge"][
            "edge_region_polygon_world"
        ],
        "target_region_member": spec["geometry"]["target_node"][
            "boundary_polygon_world"
        ],
    }
    for member, polygon in polygons.items():
        arrays[f"teacher__{member}"][-1] = int(
            M.point_in_polygon_inclusive(
                terminal[:2].tolist(), polygon, tolerance_m=tolerance,
            )
        )
    assert M.reduce_qualification_teacher_trace(metadata, arrays)[
        "termination_flags"
    ]["nan"] is True

    tampered = {name: value.copy() for name, value in arrays.items()}
    tampered["teacher__source_region_member"][-1] ^= np.uint8(1)
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.reduce_qualification_teacher_trace(metadata, tampered)


def test_identity_adapter_changes_only_official_version_tokens() -> None:
    value = C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v3.example.v1",
            "experiment_id": C.V3_EXPERIMENT_ID,
            "state_id": "pgehq-v1-state-00-00",
            "salt": "pgehq-v1",
        }
    )
    v4 = M.project_v3_evidence_to_v4(value)
    assert v4["experiment_id"] == C.EXPERIMENT_ID
    assert "qualification_v4" in v4["schema"]
    assert v4["state_id"] == value["state_id"] and v4["salt"] == value["salt"]
    assert M.project_v4_evidence_to_v3(v4) == value
    bad = copy.deepcopy(value)
    bad["salt"] = "tampered"
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.project_v3_evidence_to_v4(bad)


def test_v4_runtime_external_and_predecessor_projections_are_native() -> None:
    binding = C.HISTORICAL_CUSTODY_RECEIPT_BINDING
    runtime = C.build_runtime_contract("b" * 40, binding)
    assert M.external_artifact_bindings(runtime) == runtime[
        "external_artifact_bindings"
    ]
    assert M.predecessor_context_binding(runtime) == runtime[
        "scientific_contract"
    ]["v2_context_binding"]
    assert M.external_artifact_bindings(C.build_contract()) == list(
        C.EXTERNAL_ARTIFACT_BINDINGS
    )
    assert M.predecessor_context_binding(C.build_contract()) == (
        C.V3.V2.V1.V2_CONTEXT_BINDING
    )


def test_invariance_receipt_requires_all_ten_and_four_family_fixtures() -> None:
    results = M.build_regression_results({name: True for name in C.ALL_V4_REGRESSION_IDS})
    assert len(results) == 14
    receipt = M.build_scientific_invariance_receipt(C.build_contract(), results)
    assert receipt["pass"] is True
    assert receipt["final_evaluation_eligible"] is False
    assert M.validate_scientific_invariance_receipt(receipt) == receipt
    bad = copy.deepcopy(results)
    bad[0]["passed"] = False
    failed = M.build_scientific_invariance_receipt(C.build_contract(), bad)
    assert failed["pass"] is False
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_scientific_invariance_receipt(failed)


def test_reducer_authority_is_v4_native_and_has_no_v3_terminal_surface() -> None:
    authority = M.reducer_authority()
    forbidden = {
        "successful_output_leaf_count", "successful_output_leaves",
        "first_eight_reproduction_authority",
        "reproduction_mismatch_output_leaves", "new_documents",
        "v3_additional_recompute_evidence_keys", "v3_recompute_evidence_keys",
        "v1_v2_custody_and_nonreuse_authority",
        "external_historical_custody_receipt_authority",
        "qualification_shard_augmentation_authority",
        "historical_snapshot_deserializer_authority", "regression_gate_authority",
    }
    assert forbidden.isdisjoint(authority)
    assert authority["success_output_leaf_count"] == 27
    assert authority["success_output_leaves"] == list(C.SUCCESS_OUTPUT_LEAVES)
    assert authority["panel_inadequate_output_leaf_count"] == 9
    assert authority["panel_inadequate_output_leaves"] == list(
        C.PANEL_INADEQUATE_OUTPUT_LEAVES
    )
    assert authority["result_publication_authority"] == C.RESULT_PUBLICATION_AUTHORITY
    assert authority["panel_teacher_subset_authority"] == C.V4_PANEL_TEACHER_SUBSET_AUTHORITY
    assert authority["raw_teacher_reduction_authority"] == C.V4_RAW_TEACHER_REDUCTION_AUTHORITY
    assert authority["pre_panel_engineering_correction_authority"] == (
        C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
    )
    assert authority["teacher_selection_authority"] == C.TEACHER_SELECTION_AUTHORITY
    assert authority["qualification_runtime_authority"] == (
        C.QUALIFICATION_RUNTIME_AUTHORITY
    )
    assert authority["documents"]["teacher_trace_index"]["count"] == (
        "actual teacher_executed state count in [64,256]"
    )
    dynamic = authority["npz_authorities"]["teacher_traces.npz"]
    assert dynamic["dynamic_authority_builder"] == "teacher_trace_npz_authority"
    assert dynamic["placeholder_or_padded_rows_forbidden"] is True


def test_sparse_teacher_and_raw_material_apis_are_public() -> None:
    expected = {
        "teacher_trace_npz_authority", "reduce_qualification_teacher_trace",
        "validate_qualification_teacher_raw_evidence",
        "validate_teacher_selection", "validate_v4_success_evidence",
        "validate_panel_context", "validate_encoding_receipt",
        "build_qualification_runtime_environment",
        "validate_qualification_runtime_environment",
        "canonical_v1_planar_segment_projection", "frozen_binary64_fma",
    }
    assert expected.issubset(M.__all__)
    assert M.teacher_trace_npz_authority(64) == C.teacher_trace_npz_authority(64)
    assert C.teacher_trace_npz_authority(64)["trace_offsets"]["shape"] == [65]
    assert C.teacher_trace_npz_authority(256)["trace_offsets"]["shape"] == [257]


def test_external_custody_root_mapping_survives_canonical_save_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = {
        "v1_source_freeze_commit": C.V1_SOURCE_FREEZE_COMMIT,
        "v2_source_freeze_commit": C.V2_SOURCE_FREEZE_COMMIT,
        "v3_source_freeze_commit": C.V3_SOURCE_FREEZE_COMMIT,
        "v1_freeze_subject": C.V3.V2.V1.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "v2_freeze_subject": C.V3.V2.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "v3_freeze_subject": C.V3.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "v4_source_parent_commit": C.SOURCE_PARENT_COMMIT,
        "current_head_commit": C.SOURCE_PARENT_COMMIT,
        "repository_worktree_clean_at_receipt_emission": False,
        "historical_tracked_sources_match_frozen_commits": True,
        "v4_development_only": True,
        "v4_permanently_ineligible_for_final_evaluation": True,
        "sealed_paths_accessed": 0,
        "tracked_ignore_bypass_used": False,
    }
    prior = {
        "v1_custody_receipt_binding": C.V1_RECEIPT_BINDING,
        "v1_v2_custody_receipt_binding": C.V1_V2_RECEIPT_BINDING,
        "v2_regeneration_receipt_present": False,
        "v3_regeneration_receipt_present": False,
    }
    immutability = {
        field: "READ_ONLY" if field == "audit_mode" else True
        for field in C.HISTORICAL_CUSTODY_IMMUTABILITY_FIELDS
    }
    nonreuse = {
        field: False if field == "historical_runtime_artifact_or_shard_reused" else 0
        for field in C.HISTORICAL_CUSTODY_NONREUSE_FIELDS
    }
    roots = {
        key: {"files": [{"device": index, "inode": index}]}
        for index, key in enumerate(C.HISTORICAL_ROOT_KEYS, 1)
    }
    receipt = {
        "schema": C.HISTORICAL_CUSTODY_RECEIPT_SCHEMA,
        "experiment_id": C.EXPERIMENT_ID,
        "generated_before_v4_simulator_creation": True,
        "repository": repository,
        "prior_receipt_bindings": prior,
        "roots": roots,
        "v3_partial_boundary": C.V3_PARTIAL_BOUNDARY_EXPECTATION,
        "v3_failure_log_bindings": C.V3_FAILURE_EXPECTATION,
        "v3_terminal_interpretation": C.V3_TERMINAL_INTERPRETATION,
        "immutability": immutability,
        "nonreuse": nonreuse,
    }
    monkeypatch.setattr(M, "_historical_root", lambda value, _key: value)
    reloaded = json.loads(C.canonical_json_bytes(receipt))
    assert tuple(reloaded["roots"]) != C.HISTORICAL_ROOT_KEYS
    assert M.validate_external_v1_v2_v3_custody_receipt(reloaded) == reloaded


def _qualification_runtime_fixture(*, fake: bool = False) -> tuple[dict, list[dict]]:
    binding = C.HISTORICAL_CUSTODY_RECEIPT_BINDING
    runtime = C.build_runtime_contract("b" * 40, binding)
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
            C.DIRECT_RUNTIME_POLICY[
                "required_environment_before_simulator_creation"
            ]
        ),
        "fake_runtime": fake,
    }
    metadata = [
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v4."
                "teacher_pool_terminal.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "pool_index": index,
            "source_freeze_commit": runtime["source_freeze_commit"],
            "runtime_contract_content_digest": runtime["content_digest"],
            "stage_runtime": copy.deepcopy(core),
            "backend_runtime": copy.deepcopy(
                C.QUALIFICATION_BACKEND_RUNTIME_AUTHORITY
            ),
            "teacher_executed": False,
        }
        for index in range(C.V3.PROSPECTIVE_POOL_COUNT)
    ]
    return runtime, metadata


def test_qualification_runtime_projection_binds_all_256_and_fails_closed() -> None:
    runtime, metadata = _qualification_runtime_fixture()
    projection = M.build_qualification_runtime_environment(metadata, runtime)
    assert projection["qualification_shard_count"] == 256
    assert projection["fake_runtime"] is False
    assert projection["models_trained"] == 0
    assert len(set(projection["qualification_stage_runtime_sha256s"])) == 1
    assert M.validate_qualification_runtime_environment(
        projection, runtime, metadata_rows=metadata
    ) == projection
    material = [
        {
            "stage_runtime_sha256": projection[
                "qualification_stage_runtime_sha256s"
            ][index],
            "backend_runtime_sha256": projection[
                "qualification_backend_runtime_sha256s"
            ][index],
            "backend_runtime_core_sha256": projection[
                "backend_runtime_core_sha256"
            ],
        }
        for index in range(256)
    ]
    assert M.validate_qualification_runtime_environment(
        projection, runtime, material_shard_validations=material
    ) == projection

    bad = copy.deepcopy(metadata)
    bad[128]["stage_runtime"]["backend"] = "drift"
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.build_qualification_runtime_environment(bad, runtime)
    bad = copy.deepcopy(metadata)
    bad[129]["backend_runtime"]["policy_device"] = "accelerator"
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.build_qualification_runtime_environment(bad, runtime)
    bad = copy.deepcopy(metadata)
    bad[17]["source_freeze_commit"] = "c" * 40
    with pytest.raises(
        M.PhysicalGraphEdgeHandoffV4MetricsError,
        match="source/runtime contract binding drift",
    ):
        M.build_qualification_runtime_environment(bad, runtime)
    bad = copy.deepcopy(metadata)
    bad[18]["runtime_contract_content_digest"] = "c" * 64
    with pytest.raises(
        M.PhysicalGraphEdgeHandoffV4MetricsError,
        match="source/runtime contract binding drift",
    ):
        M.build_qualification_runtime_environment(bad, runtime)
    bad = copy.deepcopy(metadata)
    bad[19]["source_freeze_commit"] = C.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
    with pytest.raises(
        M.PhysicalGraphEdgeHandoffV4MetricsError,
        match="invalidated pre-correction runtime",
    ):
        M.build_qualification_runtime_environment(bad, runtime)
    bad = copy.deepcopy(metadata)
    bad[20]["runtime_contract_content_digest"] = (
        C.INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
    )
    with pytest.raises(
        M.PhysicalGraphEdgeHandoffV4MetricsError,
        match="invalidated pre-correction runtime",
    ):
        M.build_qualification_runtime_environment(bad, runtime)
    bad_projection = copy.deepcopy(projection)
    bad_projection.pop("content_digest")
    bad_projection["models_trained"] = 1
    bad_projection = C.attach_content_digest(bad_projection)
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_qualification_runtime_environment(bad_projection, runtime)
    bad_external = copy.deepcopy(projection)
    bad_external.pop("content_digest")
    bad_external["external_artifact_bindings_sha256"] = "e" * 64
    bad_external = C.attach_content_digest(bad_external)
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_qualification_runtime_environment(bad_external, runtime)
    bad_material = copy.deepcopy(material)
    bad_material[7]["backend_runtime_sha256"] = "0" * 64
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_qualification_runtime_environment(
            projection, runtime, material_shard_validations=bad_material
        )


def test_fake_qualification_runtime_requires_explicit_test_allowance() -> None:
    runtime, metadata = _qualification_runtime_fixture(fake=True)
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.build_qualification_runtime_environment(metadata, runtime)
    projection = M.build_qualification_runtime_environment(
        metadata, runtime, allow_fake_runtime=True
    )
    assert projection["fake_runtime"] is True
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.validate_qualification_runtime_environment(projection, runtime)
    assert M.validate_qualification_runtime_environment(
        projection, runtime, metadata_rows=metadata, allow_fake_runtime=True
    ) == projection


def test_panel_inadequate_result_and_report_carry_qualification_runtime() -> None:
    runtime, metadata = _qualification_runtime_fixture()
    qualification = M.build_qualification_runtime_environment(metadata, runtime)
    runtime_environments = {
        "qualification": copy.deepcopy(qualification),
        "physical": None,
        "encoder": None,
        "ranker": None,
        "any_fake_runtime": False,
    }
    metrics = C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v4.metrics.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "development_only": True,
            "final_evaluation_eligible": False,
            "scientific_result_produced": True,
            "terminal_disposition": C.PANEL_INADEQUATE_DISPOSITION,
            "primary_classification": C.PANEL_INADEQUATE_DISPOSITION,
            "secondary_classifications": [],
            "next_experiment": C.NEXT_DECISION_PANEL_INADEQUATE,
            "v4_state_dispositions": {
                "disposition_counts": {"INITIAL_BOUNDARY_TIPPED": 256}
            },
            "v4_panel_adequacy": {
                "adequate": False, "qualified_count": 0,
                "qualification_record_count": 256,
                "adequate_stratum_count": 0,
                "offset_opening_supplies_any_qualified_state": False,
                "shortfall_causes": [],
            },
            "v4_historical_custody": {},
            "v4_scientific_invariance": {},
            "downstream_scientific_metrics": None,
            "qualification_runtime_environment": qualification,
            "runtime_environments": runtime_environments,
            "models_trained": 0,
            "prohibited_components_trained_or_implemented": [],
        }
    )
    result = M.build_result_document(
        metrics, runtime, metrics_sha256="c" * 64,
        independent_reducer_receipt_sha256="d" * 64,
        runtime_seconds=1.0, scientific_storage_bytes=1,
    )
    assert result["runtime_environments"] == runtime_environments
    assert result["qualification_runtime_environment"] == qualification
    assert result["models_trained"] == 0
    report = M.build_result_report(result, metrics)
    assert "Qualification runtime environment:" in report
    assert C.V3_TERMINAL_DIAGNOSIS in report

    bad = copy.deepcopy(metrics)
    bad.pop("content_digest")
    bad["runtime_environments"]["qualification"] = None
    bad = C.attach_content_digest(bad)
    with pytest.raises(M.PhysicalGraphEdgeHandoffV4MetricsError):
        M.build_result_document(
            bad, runtime, metrics_sha256="c" * 64,
            independent_reducer_receipt_sha256="d" * 64,
            runtime_seconds=1.0, scientific_storage_bytes=1,
        )
