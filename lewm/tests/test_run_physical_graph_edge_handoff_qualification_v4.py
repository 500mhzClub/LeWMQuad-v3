from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v4_contract as C
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import run_physical_graph_edge_handoff_qualification_v2 as V2
from scripts import run_physical_graph_edge_handoff_qualification_v3 as V3
from scripts import run_physical_graph_edge_handoff_qualification_v4 as R


ZERO_FLAGS = {name: False for name in C.TERMINATION_FLAG_ORDER}


def _synthetic_runtime_contract() -> dict:
    return C.build_runtime_contract(
        "a" * 40, C.HISTORICAL_CUSTODY_RECEIPT_BINDING
    )


def _synthetic_qualification_backend_runtime() -> dict:
    runtime = copy.deepcopy(C.QUALIFICATION_BACKEND_RUNTIME_AUTHORITY)
    runtime["backend"] = "synthetic_test_only"
    runtime["policy_device"] = "cpu"
    return runtime


def _emit_temp_historical_receipt(
    evaluator: object,
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    receipt = evaluator.build_historical_custody_receipt(metrics_module=R.METRICS)
    raw = R.canonical_bytes(receipt)
    monkeypatch.setattr(
        C,
        "HISTORICAL_CUSTODY_RECEIPT_BINDING",
        {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    evaluator.emit_historical_custody_receipt(
        path, receipt=receipt, metrics_module=R.METRICS
    )
    return evaluator.validate_existing_historical_custody_receipt(
        path, metrics_module=R.METRICS
    )


def _identity(*, tipped_probe: bool = False) -> dict[str, str | None]:
    return {
        "artifact_file_sha256": "1" * 64,
        "snapshot_semantic_digest_v1": "2" * 64,
        "snapshot_behavioural_digest_v1": None if tipped_probe else "3" * 64,
    }


def _criteria(**changes: bool) -> dict[str, bool]:
    result = {name: True for name in C.TEACHER_QUALIFICATION_COMPONENT_IDS}
    result.update(changes)
    return result


def _record(index: int, kind: str) -> dict:
    spec = C.build_candidate_specs()[index]
    evidence: dict = {
        "initial_termination_flags": ZERO_FLAGS,
        "probe_trial_termination_flags": [ZERO_FLAGS, ZERO_FLAGS],
        "teacher_termination_flags": ZERO_FLAGS,
        "probe_tip_sample_indices": [None, None],
        "teacher_criteria": _criteria(),
        "snapshot_identity": _identity(),
        "diagnostics_inventory": [],
        "payload_member_inventory": ["snapshot_payload_bytes"],
    }
    disposition = "QUALIFIED"
    stage = "COMPLETE"
    executable = True
    teacher = True
    if kind == "initial_tipped":
        disposition = "INITIAL_BOUNDARY_TIPPED"
        stage = "INITIAL_BOUNDARY"
        executable = teacher = False
        evidence.update(
            {
                "initial_termination_flags": {**ZERO_FLAGS, "tipped": True},
                "probe_trial_termination_flags": None,
                "teacher_termination_flags": None,
                "probe_tip_sample_indices": None,
                "teacher_criteria": None,
                "snapshot_identity": None,
                "diagnostics_inventory": sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
                "payload_member_inventory": sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
            }
        )
    elif kind == "probe_tipped":
        disposition = "RESTORATION_PROBE_TIPPED"
        stage = "RESTORATION_PROBE"
        teacher = False
        evidence.update(
            {
                "probe_trial_termination_flags": [
                    {**ZERO_FLAGS, "tipped": True},
                    {**ZERO_FLAGS, "tipped": True},
                ],
                "probe_tip_sample_indices": [4, 4],
                "teacher_termination_flags": None,
                "teacher_criteria": None,
                "snapshot_identity": _identity(tipped_probe=True),
            }
        )
    elif kind == "contact":
        disposition = "TEACHER_PHYSICS_CONTACT"
        stage = "TEACHER_EXECUTION"
        evidence["teacher_criteria"] = _criteria(teacher_trace_contact_free=False)
    elif kind == "crossing":
        disposition = "TEACHER_CROSSING_INVALID"
        stage = "TEACHER_EXECUTION"
        evidence["teacher_criteria"] = _criteria(teacher_crossed_directed_port=False)
    elif kind == "leave":
        disposition = "TEACHER_DID_NOT_LEAVE_SOURCE"
        stage = "TEACHER_EXECUTION"
        evidence["teacher_criteria"] = _criteria(teacher_left_source_region=False)
    elif kind == "progress":
        disposition = "TEACHER_NO_POSITIVE_PROGRESS"
        stage = "TEACHER_EXECUTION"
        evidence["teacher_criteria"] = _criteria(teacher_positive_route_progress=False)
    elif kind == "unsafe":
        disposition = "TEACHER_TERMINATED_UNSAFELY"
        stage = "TEACHER_EXECUTION"
        evidence["teacher_termination_flags"] = {**ZERO_FLAGS, "fall": True}
    elif kind != "qualified":
        raise AssertionError(kind)
    return R._terminal_record_metadata(
        pool_index=index,
        spec=spec,
        disposition=disposition,
        stage_reached=stage,
        executable_snapshot_exists=executable,
        teacher_executed=teacher,
        reason=disposition,
        stage_runtime={},
        backend_runtime={},
        evidence=evidence,
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
    ],
)
def test_runner_builds_exact_continuing_state_dispositions(
    kind: str, expected: str
) -> None:
    row = _record(0, kind)
    assert row["disposition"] == expected
    assert row["hard_stop"] is False
    assert row["continuation_authorized"] is True


def test_valid_initial_state_reaches_complete_teacher_classification() -> None:
    row = _record(1, "qualified")
    assert row["initial_termination_flags"] == ZERO_FLAGS
    assert row["executable_snapshot_exists"] is True
    assert row["teacher_executed"] is True
    assert row["stage_reached"] == "COMPLETE"


def test_initial_tipped_is_sampled_without_snapshot_or_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = C.build_candidate_specs()[0]

    class Session:
        flags = {**ZERO_FLAGS, "tipped": True}
        _runtime = {"backend": "fixture"}

        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                runner=SimpleNamespace(
                    _last_executed=np.zeros((1, 3), dtype=np.float32),
                    _sim_time_ns=200_000_000,
                    episode_states=[SimpleNamespace(episode_step=2)],
                ),
                ticks_executed=2,
                policy_steps=10,
            )
            self.capture_calls = 0
            self.teacher_calls = 0

        def begin_and_settle(self) -> None:
            return None

        def _sample(self, requested: object, applied: object, timestamp: float) -> dict:
            return {
                "base_pose_world": np.asarray(
                    [0, 0, 0.2, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
                ),
                "base_twist_world": np.arange(6, dtype=np.float64),
                "joint_position": np.arange(12, dtype=np.float64),
                "joint_velocity": np.arange(12, dtype=np.float64) / 10,
                "physics_contact": True,
            }

        def capture_snapshot(self) -> object:
            self.capture_calls += 1
            raise AssertionError("tipped boundary must not be captured")

        def execute_teacher_with_disposition(self) -> object:
            self.teacher_calls += 1
            raise AssertionError("tipped boundary must not execute teacher")

    session = Session()
    backend = object.__new__(R.GenesisGo2PhysicalBackend)
    backend._spec_sha_by_id = {
        spec["candidate_spec_id"]: spec["canonical_spec_sha256"]
    }
    monkeypatch.setattr(backend, "_session", lambda _spec: session)
    monkeypatch.setattr(R, "_termination_flags", lambda value: dict(value.flags))
    packet = backend._qualify(spec, require_registered=True)
    assert packet["mode"] == "INITIAL_REJECTION"
    assert packet["disposition"] == "INITIAL_BOUNDARY_TIPPED"
    assert session.capture_calls == 0 and session.teacher_calls == 0
    assert set(packet["arrays"]) == set(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)
    assert packet["arrays"]["termination_flags"].tolist() == [0, 0, 1, 0]
    assert packet["diagnostics"]["intended_pose_representation"] == (
        "xyz_plus_quaternion_xyzw"
    )


def test_initial_nan_boundary_retains_observed_state_and_rejects_false_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = C.build_candidate_specs()[0]
    pose_nan = _nan64(0xA5)

    class Session:
        flags = {**ZERO_FLAGS, "nan": True}
        _runtime = {"backend": "fixture"}

        def __init__(self) -> None:
            self.ctx = SimpleNamespace(
                runner=SimpleNamespace(
                    _last_executed=np.zeros((1, 3), dtype=np.float64),
                    _sim_time_ns=200_000_000,
                    episode_states=[SimpleNamespace(episode_step=2)],
                ),
                ticks_executed=2,
                policy_steps=10,
            )
            self.capture_calls = 0
            self.teacher_calls = 0

        def begin_and_settle(self) -> None:
            return None

        def _sample(self, requested: object, applied: object, timestamp: float) -> dict:
            return {
                "base_pose_world": np.asarray(
                    [pose_nan, 0, 0.35, 0, 0, 0, 1], dtype=np.float64
                ),
                "base_twist_world": np.asarray(
                    [_nan64(0xB6), 0, 0, 0, 0, 0], dtype=np.float64
                ),
                "joint_position": np.asarray(
                    [_nan64(0xC7), *np.zeros(11)], dtype=np.float64
                ),
                "joint_velocity": np.asarray(
                    [_nan64(0xD8), *np.zeros(11)], dtype=np.float64
                ),
                "physics_contact": False,
            }

        def capture_snapshot(self) -> object:
            self.capture_calls += 1
            raise AssertionError("nan boundary must not be captured")

        def execute_teacher_with_disposition(self) -> object:
            self.teacher_calls += 1
            raise AssertionError("nan boundary must not execute teacher")

    session = Session()
    backend = object.__new__(R.GenesisGo2PhysicalBackend)
    backend._spec_sha_by_id = {
        spec["candidate_spec_id"]: spec["canonical_spec_sha256"]
    }
    monkeypatch.setattr(backend, "_session", lambda _spec: session)
    monkeypatch.setattr(R, "_termination_flags", lambda value: dict(value.flags))
    packet = backend._qualify(spec, require_registered=True)
    assert packet["mode"] == "INITIAL_REJECTION"
    assert packet["disposition"] == "UNRESOLVED_STATE_FAILURE"
    assert session.capture_calls == 0 and session.teacher_calls == 0
    observed_bits = packet["arrays"]["base_pose_world"].view(np.uint64)[0]
    assert observed_bits == np.asarray([pose_nan]).view(np.uint64)[0]
    assert np.isnan(packet["arrays"]["base_twist_world"][0])
    assert np.isnan(packet["arrays"]["joint_position"][0])
    assert np.isnan(packet["arrays"]["joint_velocity"][0])

    session.flags = dict(ZERO_FLAGS)
    with pytest.raises(R.ExperimentError, match="nan flag/base-pose"):
        R._boundary_arrays_and_metadata(session, spec)

    session.flags = {**ZERO_FLAGS, "nan": True}
    session.ctx.runner._last_executed[0, 0] = _nan64(0xE9)
    with pytest.raises(R.ExperimentError, match="outside terminal authority"):
        R._boundary_arrays_and_metadata(session, spec)


def test_nan_flags_cooccur_and_tipped_or_teacher_unsafe_precedence_is_stable() -> None:
    pose = np.asarray(
        [
            _nan64(0xA5),
            0.0,
            0.1,
            np.sqrt(0.5),
            0.0,
            0.0,
            np.sqrt(0.5),
        ],
        dtype=np.float64,
    )
    flags = R.METRICS._teacher_termination_flags_from_pose(pose)
    assert flags == {"fall": True, "out_of_bounds": False, "tipped": True, "nan": True}
    initial = R.METRICS.build_state_disposition_record(
        C.build_candidate_specs()[0],
        stage_reached="INITIAL_BOUNDARY",
        initial_termination_flags=flags,
        executable_snapshot_exists=False,
        teacher_executed=False,
        diagnostics_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
        payload_member_inventory=sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY),
    )
    assert initial["disposition"] == "INITIAL_BOUNDARY_TIPPED"
    teacher_disposition, _components = R._teacher_disposition(
        contact_free=True,
        crossing={"normal_dot_displacement_m": 1.0},
        left_source=True,
        positive_progress=True,
        competing_port_entered=False,
        goal_reachable=True,
        physically_executable=True,
        termination_flags=flags,
    )
    assert teacher_disposition == "TEACHER_TERMINATED_UNSAFELY"


def test_v4_sample_preserves_nonfinite_xy_and_only_marks_membership_undefined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose_nan = _nan64(0xA5)

    class Robot:
        def get_pos(self) -> np.ndarray:
            return np.asarray([[pose_nan, 0.0, 0.35]], dtype=np.float64)

        def get_quat(self) -> np.ndarray:
            return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

        def get_vel(self) -> np.ndarray:
            return np.zeros((1, 3), dtype=np.float64)

        def get_ang(self) -> np.ndarray:
            return np.zeros((1, 3), dtype=np.float64)

        def get_dofs_position(self, _indices: object) -> np.ndarray:
            return np.zeros((1, 12), dtype=np.float64)

        def get_dofs_velocity(self, _indices: object) -> np.ndarray:
            return np.zeros((1, 12), dtype=np.float64)

    session = object.__new__(R._V4GenesisPhysicalSession)
    session.ctx = SimpleNamespace(
        runner=SimpleNamespace(
            _as_np=lambda value: value,
            _leg_dof_idx=np.arange(12),
        ),
        build=SimpleNamespace(robot=Robot()),
        policy=SimpleNamespace(_last_actions=np.zeros((1, 12))),
    )
    session._last_controller_observation = np.zeros(45, dtype=np.float64)
    session._v4_capture_mode = None
    session._v4_capture_rows = []
    session._disallowed_contact = lambda: False
    monkeypatch.setattr(
        V3._V3GenesisPhysicalSession,
        "_sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            V1.ExperimentError("pure polygon-membership authority failed")
        ),
    )
    row = session._sample([0, 0, 0], [0, 0, 0], 0.002)
    assert row["base_pose_world"].view(np.uint64)[0] == np.asarray(
        [pose_nan]
    ).view(np.uint64)[0]
    assert all(
        int(row[member]) == 0
        for member in (
            "source_region_member",
            "correct_edge_region_member",
            "edge_region_member",
            "wrong_edge_region_member",
            "target_region_member",
        )
    )


def test_v4_sample_retains_derived_memberships_when_only_quaternion_is_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "base_pose_world": np.asarray(
            [0.0, 0.0, 0.35, _nan64(0xA5), 0.0, 0.0, 1.0]
        ),
        "source_region_member": np.uint8(1),
        "edge_region_member": np.uint8(0),
        "correct_edge_region_member": np.uint8(0),
        "wrong_edge_region_member": np.uint8(0),
        "target_region_member": np.uint8(0),
    }
    session = object.__new__(R._V4GenesisPhysicalSession)
    session._v4_capture_mode = None
    session._v4_capture_rows = []
    monkeypatch.setattr(
        V3._V3GenesisPhysicalSession,
        "_sample",
        lambda *_args, **_kwargs: copy.deepcopy(expected),
    )
    row = session._sample([0, 0, 0], [0, 0, 0], 0.002)
    assert row["source_region_member"] == 1
    assert np.isnan(row["base_pose_world"][3])


def _partial_probe_trace(samples: int = 11) -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {}
    for member, authority in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items():
        shape = [samples, *authority["shape"][1:]]
        trace[member] = np.zeros(shape, dtype=np.dtype(authority["descr"]))
    trace["timestamp_s"] = (np.arange(samples, dtype=np.float64) + 1) * 0.002
    trace["requested_command"][:] = np.asarray(
        C.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND, dtype=np.float64
    )
    trace["post_slew_applied_command"][:] = np.asarray(
        C.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND, dtype=np.float64
    )
    trace["base_pose_world"][:, 2] = 0.35
    trace["base_pose_world"][:, 6] = 1.0
    return trace


def _mark_terminal_pose_tipped(trace: dict[str, np.ndarray]) -> None:
    trace["base_pose_world"][-1, 3:] = np.asarray(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float64
    )


def _nan64(payload: int = 0xA5) -> np.float64:
    bits = np.asarray([0x7FF8000000000000 | payload], dtype=np.uint64)
    return bits.view(np.float64)[0]


def test_probe_tipped_trace_is_inclusive_and_has_no_final_snapshot() -> None:
    trace = _partial_probe_trace(11)
    _mark_terminal_pose_tipped(trace)
    flags = {**ZERO_FLAGS, "tipped": True}
    comparison = R._probe_pair_comparison(
        trace, copy.deepcopy(trace), completed=False, left_flags=flags, right_flags=flags
    )
    trials = [
        {
            "trial_index": index,
            "completed": False,
            "trace": copy.deepcopy(trace),
            "termination_flags": flags,
            "final_snapshot_semantic_evidence": None,
            "snapshot_behavioural_digest_v1": None,
            "trace_member_manifests": R._trace_member_manifests(trace),
            "trial_pair_comparison": comparison,
        }
        for index in range(2)
    ]
    arrays, metadata = R._probe_arrays_and_metadata(
        semantics={
            "semantic_payload_bytes": b"semantic",
            "artifact_file_sha256": "1" * 64,
            "snapshot_semantic_digest_v1": "2" * 64,
            "semantic_evidence": {"fixture": True},
        },
        trials=trials,
    )
    assert comparison["pass"] is True
    assert metadata["behavioural_probe"]["trials"][0]["tip_sample_index"] == 10
    assert metadata["behavioural_probe"]["trials"][0]["termination_reason"] == "TIPPED"
    assert metadata["snapshot_identity"]["snapshot_behavioural_digest_v1"] is None
    assert not any("final_snapshot_semantic_digest_bytes" in name for name in arrays)


def test_non_tipped_partial_probe_is_not_mislabeled_and_tamper_is_rejected() -> None:
    trace = _partial_probe_trace(7)
    fall = {**ZERO_FLAGS, "fall": True}
    comparison = R._probe_pair_comparison(
        trace, copy.deepcopy(trace), completed=False, left_flags=fall, right_flags=fall
    )
    assert comparison["pass"] is True
    trials = [
        {
            "trial_index": index,
            "completed": False,
            "trace": copy.deepcopy(trace),
            "termination_flags": fall,
            "final_snapshot_semantic_evidence": None,
            "snapshot_behavioural_digest_v1": None,
            "trace_member_manifests": R._trace_member_manifests(trace),
            "trial_pair_comparison": comparison,
        }
        for index in range(2)
    ]
    _arrays, metadata = R._probe_arrays_and_metadata(
        semantics={
            "semantic_payload_bytes": b"semantic",
            "artifact_file_sha256": "1" * 64,
            "snapshot_semantic_digest_v1": "2" * 64,
            "semantic_evidence": {"fixture": True},
        },
        trials=trials,
    )
    assert metadata["behavioural_probe"]["trials"][0]["tipped"] is False
    assert metadata["behavioural_probe"]["trials"][0]["termination_reason"] == "FALL"
    mutated = copy.deepcopy(trace)
    mutated["joint_position"][0, 0] = 1.0e-3
    mismatch = R._probe_pair_comparison(
        trace, mutated, completed=False, left_flags=fall, right_flags=fall
    )
    assert mismatch["pass"] is False


def test_nan_probe_retains_only_terminal_state_nonfinite_bits() -> None:
    trace = _partial_probe_trace(7)
    trace["base_pose_world"][-1, 0] = _nan64(0xA5)
    trace["base_twist_world"][-1, 1] = _nan64(0xB6)
    trace["joint_position"][-1, 2] = _nan64(0xC7)
    trace["joint_velocity"][-1, 3] = _nan64(0xD8)
    flags = {**ZERO_FLAGS, "nan": True}
    comparison = R._probe_pair_comparison(
        trace,
        copy.deepcopy(trace),
        completed=False,
        left_flags=flags,
        right_flags=flags,
    )
    assert comparison["pass"] is True
    assert comparison["nonfinite_masks_equal"] is True
    assert comparison["nonfinite_bit_patterns_equal"] is True

    changed_payload = copy.deepcopy(trace)
    changed_payload["base_pose_world"][-1, 0] = _nan64(0xA6)
    mismatch = R._probe_pair_comparison(
        trace,
        changed_payload,
        completed=False,
        left_flags=flags,
        right_flags=flags,
    )
    assert mismatch["pass"] is False
    assert mismatch["nonfinite_masks_equal"] is True
    assert mismatch["nonfinite_bit_patterns_equal"] is False

    false_flag = {**ZERO_FLAGS}
    with pytest.raises(R.ExperimentError, match="nan flag/base-pose"):
        R._normalise_partial_probe_trace(
            trace, termination_flags=false_flag
        )
    preterminal = copy.deepcopy(trace)
    preterminal["base_pose_world"][-2, 0] = _nan64(0xE9)
    with pytest.raises(R.ExperimentError, match="before the terminal sample"):
        R._normalise_partial_probe_trace(
            preterminal, termination_flags=flags
        )
    unlisted = copy.deepcopy(trace)
    unlisted["policy_output"][-1, 0] = _nan64(0xFA)
    with pytest.raises(R.ExperimentError, match="outside terminal authority"):
        R._normalise_partial_probe_trace(
            unlisted, termination_flags=flags
        )


def _partial_teacher_trace(samples: int = 1) -> dict[str, np.ndarray]:
    trace: dict[str, np.ndarray] = {}
    byte_members = {
        "physics_contact",
        "source_region_member",
        "edge_region_member",
        "correct_edge_region_member",
        "wrong_edge_region_member",
        "target_region_member",
    }
    tails = {
        "base_pose_world": (7,),
        "base_twist_world": (6,),
        "joint_position": (12,),
        "joint_velocity": (12,),
        "applied_command": (3,),
        "requested_command": (3,),
        "post_slew_applied_command": (3,),
    }
    for member in V1.TEACHER_TRACE_MEMBERS:
        dtype = np.uint8 if member in byte_members else np.float64
        trace[member] = np.zeros((samples, *tails.get(member, ())), dtype=dtype)
    trace["timestamp_s"][:] = (np.arange(samples) + 1) * R.TRACE_DT_S
    trace["base_pose_world"][:, 2] = 0.35
    trace["base_pose_world"][:, 6] = 1.0
    return trace


def test_single_sample_unsafe_teacher_trace_is_valid_rejection_evidence() -> None:
    trace = _partial_teacher_trace()
    trace["base_pose_world"][0, 2] = 0.1
    normalized = R._normalise_partial_teacher_trace(
        trace, termination_flags={**ZERO_FLAGS, "fall": True}
    )
    assert len(normalized["timestamp_s"]) == 1


def test_nan_teacher_trace_uses_finite_prefix_or_snapshot_anchor() -> None:
    spec = C.build_candidate_specs()[0]
    flags = {**ZERO_FLAGS, "nan": True}
    trace = _partial_teacher_trace(3)
    trace["base_pose_world"][-1, 0] = _nan64(0xA5)
    trace["base_twist_world"][-1, 1] = _nan64(0xB6)
    trace["joint_position"][-1, 2] = _nan64(0xC7)
    trace["joint_velocity"][-1, 3] = _nan64(0xD8)
    normalized = R._normalise_partial_teacher_trace(
        trace, termination_flags=flags
    )
    anchor = np.asarray(R._intended_base_pose_world(spec), dtype=np.float64)
    science = R._teacher_science_trace(
        normalized,
        spec=spec,
        snapshot_base_pose_world=anchor,
        termination_flags=flags,
    )
    assert np.array_equal(science["base_pose_world"], trace["base_pose_world"][:-1])

    first_sample = {member: value[-1:].copy() for member, value in trace.items()}
    fallback = R._teacher_science_trace(
        R._normalise_partial_teacher_trace(
            first_sample, termination_flags=flags
        ),
        spec=spec,
        snapshot_base_pose_world=anchor,
        termination_flags=flags,
    )
    assert np.array_equal(fallback["base_pose_world"], anchor.reshape(1, 7))
    assert fallback["source_region_member"].shape == (1,)

    preterminal = copy.deepcopy(trace)
    preterminal["base_pose_world"][-2, 0] = _nan64(0xE9)
    with pytest.raises(R.ExperimentError, match="before the terminal sample"):
        R._normalise_partial_teacher_trace(
            preterminal, termination_flags=flags
        )
    off_list = copy.deepcopy(trace)
    off_list["requested_command"][-1, 0] = _nan64(0xFA)
    with pytest.raises(R.ExperimentError, match="outside terminal authority"):
        R._normalise_partial_teacher_trace(
            off_list, termination_flags=flags
        )


def _initial_tipped_arrays(spec: dict) -> dict[str, np.ndarray]:
    arrays = {
        "intended_base_pose_world": np.asarray(
            R._intended_base_pose_world(spec), dtype=np.float64
        ),
        "base_pose_world": np.asarray(
            [0, 0, 0.2, np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
            dtype=np.float64,
        ),
        "base_twist_world": np.zeros(6, dtype=np.float64),
        "joint_position": np.zeros(12, dtype=np.float64),
        "joint_velocity": np.zeros(12, dtype=np.float64),
        "previous_applied_command": np.zeros(3, dtype=np.float64),
        "physics_contact": np.asarray([0], dtype=np.uint8),
        "sim_time_ns": np.asarray([1], dtype=np.int64),
        "episode_step": np.asarray([0], dtype=np.int64),
        "command_ticks": np.asarray([0], dtype=np.int64),
        "policy_steps": np.asarray([0], dtype=np.int64),
        "termination_flags": np.asarray([0, 0, 1, 0], dtype=np.uint8),
    }
    return {name: np.ascontiguousarray(value) for name, value in arrays.items()}


def test_initial_tipped_production_writer_reopens_exact_float64_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = C.build_candidate_specs()[0]
    runtime_contract = _synthetic_runtime_contract()
    arrays = _initial_tipped_arrays(spec)
    state = _record(0, "initial_tipped")
    metadata = {
        "schema": "physical_graph_edge_handoff_qualification_v4.teacher_pool_terminal.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "source_freeze_commit": runtime_contract["source_freeze_commit"],
        "runtime_contract_content_digest": runtime_contract["content_digest"],
        "pool_index": 0,
        "candidate_spec": spec,
        "candidate_spec_sha256": spec["canonical_spec_sha256"],
        "disposition": "INITIAL_BOUNDARY_TIPPED",
        "qualified": False,
        "rejection_reason": "INITIAL_BOUNDARY_TIPPED",
        "rejection_components": ["INITIAL_BOUNDARY_TIPPED"],
        "stage_reached": "INITIAL_BOUNDARY",
        "executable_snapshot_exists": False,
        "teacher_executed": False,
        "boundary_evidence": {
            "intended_pose_representation": "xyz_plus_quaternion_xyzw",
            "termination_flag_order": list(C.TERMINATION_FLAG_ORDER),
            "available_simulator_diagnostics": sorted(arrays),
            "previous_applied_command_sha256": V2.persisted_array_sha256(
                arrays["previous_applied_command"]
            ),
            "previous_applied_command_dtype": "<f8",
            "previous_applied_command_shape": [3],
        },
        "snapshot": None,
        "graph": None,
        "teacher": None,
        "state_disposition": state,
        "stage_runtime": {},
        "backend_runtime": {},
        "reset_or_candidate_outcome_opened": False,
    }
    material = tmp_path / "material"
    (material / "qualification").mkdir(parents=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    reopened = R._write_and_reopen_terminal(
        pool_index=0, spec=spec, metadata=metadata, arrays=arrays
    )
    _metadata, payload = R._load_material_shard(material / "qualification/pool-000")
    assert reopened["disposition"] == "INITIAL_BOUNDARY_TIPPED"
    assert reopened["source_freeze_commit"] == runtime_contract["source_freeze_commit"]
    assert reopened["runtime_contract_content_digest"] == runtime_contract[
        "content_digest"
    ]
    assert payload["previous_applied_command"].dtype.str == "<f8"
    assert V2.persisted_array_sha256(payload["previous_applied_command"]) == (
        "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
    )

    missing = copy.deepcopy(reopened)
    missing.pop("content_digest")
    missing.pop("source_freeze_commit")
    missing = C.attach_content_digest(missing)
    with pytest.raises(
        R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError,
        match="qualification material metadata field drift",
    ):
        R.METRICS.validate_qualification_material_shard(
            missing, reopened_arrays=payload, expected_pool_index=0
        )

    assert C.INVALIDATED_V4_SOURCE_FREEZE_COMMIT == (
        "cb8c1a225550dc0af16626aa5446ad3ac2aaa80a"
    )
    assert C.INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST == (
        "9b844e34ab4693e68fe229831edc82ac72d3fec89fcd2bae75d27ef4efc3af56"
    )
    for field, invalidated in (
        ("source_freeze_commit", C.INVALIDATED_V4_SOURCE_FREEZE_COMMIT),
        (
            "runtime_contract_content_digest",
            C.INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST,
        ),
    ):
        stale = copy.deepcopy(reopened)
        stale.pop("content_digest")
        stale[field] = invalidated
        stale = C.attach_content_digest(stale)
        with pytest.raises(
            R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError,
            match="binds invalidated pre-correction runtime",
        ):
            R.METRICS.validate_qualification_material_shard(
                stale, reopened_arrays=payload, expected_pool_index=0
            )

    for field, wrong in (
        ("source_freeze_commit", "b" * 40),
        ("runtime_contract_content_digest", "c" * 64),
    ):
        mismatched = copy.deepcopy(reopened)
        mismatched[field] = wrong
        with pytest.raises(
            R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError,
            match="source/runtime contract binding drift",
        ):
            R.METRICS._validate_qualification_terminal_runtime(
                mismatched,
                runtime_contract,
                expected_pool_index=0,
                allow_fake_runtime=True,
            )


def test_materialisation_corruption_is_a_hard_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = C.build_candidate_specs()[0]
    monkeypatch.setattr(
        R,
        "_write_material_shard",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fixture corruption")),
    )
    with pytest.raises(R.ExperimentError, match="STATE_MATERIALISATION_CORRUPT"):
        R._write_and_reopen_terminal(
            pool_index=0, spec=spec, metadata={}, arrays={"x": np.zeros(1)}
        )


def _snapshot_and_complete_probe_fixture(spec: dict, teacher: dict) -> tuple:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _fake_snapshot,
    )
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v3 import (
        _snapshot_payload,
        _trace,
    )

    snapshot = _fake_snapshot(spec, teacher)
    snapshot["payload_bytes"] = _snapshot_payload()
    semantics = V3._fresh_snapshot_semantics(snapshot["payload_bytes"])
    probe_trace = _trace(
        final_digest=semantics["snapshot_semantic_digest_v1"]
    )
    comparison = V3.METRICS.compare_behavioural_probe_traces(
        probe_trace, copy.deepcopy(probe_trace)
    )
    trials = [
        {
            "trial_index": index,
            "completed": True,
            "trace": copy.deepcopy(probe_trace),
            "termination_flags": ZERO_FLAGS,
            "final_snapshot_semantic_evidence": semantics["semantic_evidence"],
            "snapshot_behavioural_digest_v1": V3.METRICS.snapshot_behavioural_digest(
                probe_trace
            ),
            "trace_member_manifests": R._trace_member_manifests(probe_trace),
            "trial_pair_comparison": comparison,
        }
        for index in range(2)
    ]
    probe_arrays, probe_metadata = R._probe_arrays_and_metadata(
        semantics=semantics, trials=trials
    )
    return snapshot, probe_arrays, probe_metadata


def _patch_one_pool_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    official = tmp_path / "official"
    material = tmp_path / "material"
    official.mkdir()
    for leaf in ("qualification", "fanout"):
        (material / leaf).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(R, "OUTPUT_ROOT", official)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_stage_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        R,
        "_require_initialized",
        lambda: (_synthetic_runtime_contract(), {"specs": C.build_candidate_specs()}),
    )
    monkeypatch.setattr(R, "_bind_physical_runtime", lambda stage, backend: {})
    return official, material


def test_malformed_backend_terminal_mode_is_materialisation_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = C.build_candidate_specs()[0]

    class Backend:
        @staticmethod
        def qualify(_candidate_spec: dict) -> dict:
            return {
                "mode": "UNKNOWN",
                "candidate_spec_id": spec["candidate_spec_id"],
                "runtime_evidence": {},
            }

    _official, material = _patch_one_pool_stage(tmp_path, monkeypatch)
    with pytest.raises(R.ExperimentError, match="STATE_MATERIALISATION_CORRUPT"):
        R.qualify_pool_state_stage(0, backend=Backend(), fake_runtime=True)
    assert not (material / "qualification/pool-000").exists()


def test_missing_backend_runtime_is_materialisation_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = C.build_candidate_specs()[0]

    class Backend:
        @staticmethod
        def qualify(_candidate_spec: dict) -> dict:
            return {
                "mode": "INITIAL_REJECTION",
                "candidate_spec_id": spec["candidate_spec_id"],
                "disposition": "UNRESOLVED_STATE_FAILURE",
                "reason": "fixture missing runtime",
                "arrays": _initial_tipped_arrays(spec),
                "diagnostics": {},
            }

    _official, material = _patch_one_pool_stage(tmp_path, monkeypatch)
    with pytest.raises(R.ExperimentError, match="STATE_MATERIALISATION_CORRUPT"):
        R.qualify_pool_state_stage(0, backend=Backend(), fake_runtime=True)
    assert not (material / "qualification/pool-000").exists()


def test_initial_nan_persists_one_unresolved_terminal_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = C.build_candidate_specs()[0]
    arrays = _initial_tipped_arrays(spec)
    arrays["base_pose_world"] = np.asarray(
        [_nan64(0xA5), 0, 0.35, 0, 0, 0, 1], dtype=np.float64
    )
    arrays["base_twist_world"][0] = _nan64(0xB6)
    arrays["joint_position"][0] = _nan64(0xC7)
    arrays["joint_velocity"][0] = _nan64(0xD8)
    arrays["termination_flags"] = np.asarray([0, 0, 0, 1], dtype=np.uint8)
    diagnostics = {
        "intended_pose_representation": "xyz_plus_quaternion_xyzw",
        "termination_flag_order": list(C.TERMINATION_FLAG_ORDER),
        "available_simulator_diagnostics": sorted(arrays),
        "previous_applied_command_sha256": V2.persisted_array_sha256(
            arrays["previous_applied_command"]
        ),
        "previous_applied_command_dtype": "<f8",
        "previous_applied_command_shape": [3],
    }

    class Backend:
        def qualify(self, candidate_spec: dict) -> dict:
            assert candidate_spec == spec
            return {
                "mode": "INITIAL_REJECTION",
                "candidate_spec_id": spec["candidate_spec_id"],
                "disposition": "UNRESOLVED_STATE_FAILURE",
                "reason": "fixture initial nan boundary",
                "stage_reached": "INITIAL_BOUNDARY",
                "arrays": arrays,
                "diagnostics": diagnostics,
                "runtime_evidence": {},
            }

    _official, material = _patch_one_pool_stage(tmp_path, monkeypatch)
    metadata = R.qualify_pool_state_stage(
        0, backend=Backend(), fake_runtime=True
    )
    reopened, payload = R._load_material_shard(
        material / "qualification/pool-000"
    )
    assert metadata["disposition"] == "UNRESOLVED_STATE_FAILURE"
    assert metadata["source_freeze_commit"] == _synthetic_runtime_contract()[
        "source_freeze_commit"
    ]
    assert metadata["runtime_contract_content_digest"] == (
        _synthetic_runtime_contract()["content_digest"]
    )
    assert reopened["stage_reached"] == "INITIAL_BOUNDARY"
    assert reopened["teacher"] is None and reopened["snapshot"] is None
    assert payload["base_pose_world"].view(np.uint64)[0] == (
        arrays["base_pose_world"].view(np.uint64)[0]
    )


def test_complete_teacher_terminal_passes_production_writer_and_outer_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakePhysicalBackend as V1FakeBackend,
        _fake_snapshot,
    )
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v3 import (
        _snapshot_payload,
        _trace,
    )

    spec = C.build_candidate_specs()[0]
    teacher_backend = V1FakeBackend()
    teacher = teacher_backend._teacher(spec)
    snapshot = _fake_snapshot(spec, teacher)
    snapshot["payload_bytes"] = _snapshot_payload()
    semantics = V3._fresh_snapshot_semantics(snapshot["payload_bytes"])
    probe_trace = _trace(
        final_digest=semantics["snapshot_semantic_digest_v1"]
    )
    comparison = V3.METRICS.compare_behavioural_probe_traces(
        probe_trace, copy.deepcopy(probe_trace)
    )
    trials = [
        {
            "trial_index": index,
            "completed": True,
            "trace": copy.deepcopy(probe_trace),
            "termination_flags": ZERO_FLAGS,
            "final_snapshot_semantic_evidence": semantics["semantic_evidence"],
            "snapshot_behavioural_digest_v1": V3.METRICS.snapshot_behavioural_digest(
                probe_trace
            ),
            "trace_member_manifests": R._trace_member_manifests(probe_trace),
            "trial_pair_comparison": comparison,
        }
        for index in range(2)
    ]
    probe_arrays, probe_metadata = R._probe_arrays_and_metadata(
        semantics=semantics, trials=trials
    )
    snapshot_sha = hashlib.sha256(snapshot["payload_bytes"]).hexdigest()
    inherited_crossing = V1.canonical_port_crossing
    injected_legacy_lateral_values: list[float] = []

    def version_sensitive_crossing(*args: object, **kwargs: object) -> dict:
        value = copy.deepcopy(inherited_crossing(*args, **kwargs))
        canonical = R.METRICS.canonical_v1_planar_segment_projection(
            value["point_world"],
            spec["geometry"]["selected_directed_edge"]["opening_segment_world"],
        )["lateral_coordinate_m"]
        value["lateral_coordinate_m"] = float(np.nextafter(canonical, np.inf))
        injected_legacy_lateral_values.append(value["lateral_coordinate_m"])
        return value

    monkeypatch.setattr(V1, "canonical_port_crossing", version_sensitive_crossing)

    class Backend:
        def qualify(self, candidate_spec: dict) -> dict:
            assert candidate_spec == spec
            graph = teacher_backend.qualify(spec)["graph"]
            graph["teacher_positive_route_progress"] = (
                V1._teacher_route_progress_m(
                    teacher["base_pose_world"],
                    spec["geometry"]["selected_directed_edge"][
                        "opening_segment_world"
                    ],
                )
                > 0.0
            )
            graph["teacher_competing_port_entered"] = (
                V1._first_competing_crossing(
                    teacher["base_pose_world"],
                    spec["geometry"]["competing_directed_edges"],
                )
                is not None
            )
            return {
                "mode": "TEACHER",
                "candidate_spec_id": spec["candidate_spec_id"],
                "initial_decision_state_sha256": snapshot_sha,
                "graph": graph,
                "teacher_trace": teacher,
                "teacher_termination_flags": ZERO_FLAGS,
                "teacher_completed_without_termination": True,
                "rgb": np.zeros((168, 224, 3), dtype=np.uint8),
                "snapshot": snapshot,
                "probe_arrays": probe_arrays,
                "probe_metadata": probe_metadata,
                "contact_instrumentation": {
                    "api": "robot.get_contacts",
                    "sample_period_s": R.TRACE_DT_S,
                    "forbidden_net_force_api_used": False,
                    "ontology_sha256": C.CONTACT_AUTHORITY["ontology_sha256"],
                },
                "runtime_evidence": {
                    "snapshot_captured_before_teacher": True,
                    "teacher_restored_from_serialized_snapshot": True,
                    "teacher_snapshot_sha256": snapshot_sha,
                },
            }

    official = tmp_path / "official"
    material = tmp_path / "material"
    official.mkdir()
    for leaf in ("qualification", "fanout"):
        (material / leaf).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(R, "OUTPUT_ROOT", official)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_stage_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        R,
        "_require_initialized",
        lambda: (_synthetic_runtime_contract(), {"specs": C.build_candidate_specs()}),
    )
    monkeypatch.setattr(R, "_bind_physical_runtime", lambda stage, backend: {})
    metadata = R.qualify_pool_state_stage(0, backend=Backend(), fake_runtime=True)
    assert metadata["disposition"] == "QUALIFIED"
    assert metadata["source_freeze_commit"] == _synthetic_runtime_contract()[
        "source_freeze_commit"
    ]
    assert metadata["runtime_contract_content_digest"] == (
        _synthetic_runtime_contract()["content_digest"]
    )
    assert len(injected_legacy_lateral_values) == 1
    canonical_crossing_lateral = (
        R.METRICS.canonical_v1_planar_segment_projection(
            metadata["teacher"]["crossing"]["point_world"],
            spec["geometry"]["selected_directed_edge"]["opening_segment_world"],
        )["lateral_coordinate_m"]
    )
    assert (
        metadata["teacher"]["crossing"]["lateral_coordinate_m"]
        == canonical_crossing_lateral
    )
    assert canonical_crossing_lateral != injected_legacy_lateral_values[0]
    reopened, arrays = R._load_material_shard(material / "qualification/pool-000")
    assert C.validate_content_digest(reopened) == reopened
    assert R.METRICS.validate_qualification_material_shard(
        reopened, reopened_arrays=arrays, expected_pool_index=0
    )["metadata"]["qualified"] is True

    def refresh_raw_bindings(
        document: dict, mutated_arrays: dict[str, np.ndarray]
    ) -> dict:
        """Keep custody manifests coherent so raw-science tampering is reached."""

        result = copy.deepcopy(document)
        result.pop("content_digest")
        old_evidence = result["persisted_array_evidence"]
        result["persisted_array_evidence"] = (
            R.METRICS.build_persisted_array_evidence(
                shard_kind=old_evidence["shard_kind"],
                shard_id=old_evidence["shard_id"],
                payload_file=old_evidence["payload_file"],
                arrays=mutated_arrays,
                reopened_arrays=mutated_arrays,
            )
        )
        result["payload"] = {
            "role": "material_shard_payload",
            **result["persisted_array_evidence"]["payload_file"],
            "kind": "npz",
        }
        return C.attach_content_digest(result)

    # A trajectory can be internally rehashed yet must not retain the old
    # crossing/progress summary or QUALIFIED disposition.  This reaches the
    # raw adequacy reducer instead of failing at the custody hash layer.
    pose_arrays = {name: value.copy() for name, value in arrays.items()}
    pose_arrays["teacher__base_pose_world"][:, :2] = (
        pose_arrays["teacher__base_pose_world"][0, :2]
    )
    pose_metadata = copy.deepcopy(reopened)
    pose_metadata["teacher"]["trace_digests"]["base_pose_world"] = (
        V1.canonical_array_sha256(pose_arrays["teacher__base_pose_world"])
    )
    pose_metadata = refresh_raw_bindings(pose_metadata, pose_arrays)
    with pytest.raises(
        R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError
    ):
        R.METRICS.validate_qualification_material_shard(
            pose_metadata, reopened_arrays=pose_arrays, expected_pool_index=0
        )

    # Contact is likewise rederived from the physics-rate trace rather than
    # trusted from the material summary.
    contact_arrays = {name: value.copy() for name, value in arrays.items()}
    contact_arrays["teacher__physics_contact"][0] = 1
    contact_metadata = copy.deepcopy(reopened)
    contact_metadata["teacher"]["trace_digests"]["physics_contact"] = (
        V1.canonical_array_sha256(contact_arrays["teacher__physics_contact"])
    )
    contact_metadata = refresh_raw_bindings(contact_metadata, contact_arrays)
    with pytest.raises(
        R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError
    ):
        R.METRICS.validate_qualification_material_shard(
            contact_metadata, reopened_arrays=contact_arrays, expected_pool_index=0
        )

    summary_metadata = copy.deepcopy(reopened)
    summary_metadata.pop("content_digest")
    summary_metadata["teacher"]["positive_route_progress"] = False
    summary_metadata = C.attach_content_digest(summary_metadata)
    with pytest.raises(
        R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError
    ):
        R.METRICS.validate_qualification_material_shard(
            summary_metadata, reopened_arrays=arrays, expected_pool_index=0
        )

    for ulp_count in (1, 2):
        lateral_metadata = copy.deepcopy(reopened)
        lateral_metadata.pop("content_digest")
        lateral = lateral_metadata["teacher"]["crossing"][
            "lateral_coordinate_m"
        ]
        for _ in range(ulp_count):
            lateral = float(np.nextafter(lateral, np.inf))
        lateral_metadata["teacher"]["crossing"][
            "lateral_coordinate_m"
        ] = lateral
        lateral_metadata = C.attach_content_digest(lateral_metadata)
        with pytest.raises(
            R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError,
            match="teacher summary differs from independent raw reduction",
        ):
            R.METRICS.validate_qualification_material_shard(
                lateral_metadata, reopened_arrays=arrays, expected_pool_index=0
            )

    disposition_metadata = copy.deepcopy(reopened)
    disposition_metadata.pop("content_digest")
    disposition_metadata.update(
        {
            "disposition": "TEACHER_NO_POSITIVE_PROGRESS",
            "qualified": False,
            "rejection_reason": "TEACHER_NO_POSITIVE_PROGRESS",
            "rejection_components": ["TEACHER_NO_POSITIVE_PROGRESS"],
        }
    )
    disposition_metadata = C.attach_content_digest(disposition_metadata)
    with pytest.raises(
        R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError
    ):
        R.METRICS.validate_qualification_material_shard(
            disposition_metadata, reopened_arrays=arrays, expected_pool_index=0
        )

    behavioural_metadata = copy.deepcopy(reopened)
    behavioural_metadata.pop("content_digest")
    behavioural_metadata["behavioural_probe"]["trials"][1][
        "snapshot_behavioural_digest_v1"
    ] = "f" * 64
    behavioural_metadata = C.attach_content_digest(behavioural_metadata)
    with pytest.raises(
        R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError
    ):
        R.METRICS.validate_qualification_material_shard(
            behavioural_metadata, reopened_arrays=arrays, expected_pool_index=0
        )


@pytest.mark.parametrize(
    ("first_sample_nan", "nan_component"),
    [(False, 0), (True, 0), (False, 3), (True, 3)],
)
def test_teacher_nan_persists_unsafe_terminal_and_uses_finite_science_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_sample_nan: bool,
    nan_component: int,
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakePhysicalBackend as V1FakeBackend,
    )

    spec = C.build_candidate_specs()[0]
    teacher_backend = V1FakeBackend()
    original_teacher = teacher_backend._teacher(spec)
    snapshot, probe_arrays, probe_metadata = _snapshot_and_complete_probe_fixture(
        spec, original_teacher
    )
    teacher = {
        member: np.ascontiguousarray(
            value[-1:].copy() if first_sample_nan else value.copy()
        )
        for member, value in original_teacher.items()
    }
    teacher["base_pose_world"][-1, nan_component] = _nan64(0xA5)
    teacher["base_twist_world"][-1, 1] = _nan64(0xB6)
    teacher["joint_position"][-1, 2] = _nan64(0xC7)
    teacher["joint_velocity"][-1, 3] = _nan64(0xD8)
    if nan_component in {0, 1}:
        for member in (
            "source_region_member",
            "edge_region_member",
            "target_region_member",
        ):
            teacher[member][-1] = 0
    flags = {**ZERO_FLAGS, "nan": True}
    science = R._teacher_science_trace(
        teacher,
        spec=spec,
        snapshot_base_pose_world=snapshot["base_pose_world"],
        termination_flags=flags,
    )
    graph = teacher_backend.qualify(spec)["graph"]
    graph["teacher_positive_route_progress"] = (
        V1._teacher_route_progress_m(
            science["base_pose_world"],
            spec["geometry"]["selected_directed_edge"]["opening_segment_world"],
        )
        > 0.0
    )
    graph["teacher_competing_port_entered"] = (
        V1._first_competing_crossing(
            science["base_pose_world"],
            spec["geometry"]["competing_directed_edges"],
        )
        is not None
    )
    snapshot_sha = hashlib.sha256(snapshot["payload_bytes"]).hexdigest()

    class Backend:
        def qualify(self, candidate_spec: dict) -> dict:
            assert candidate_spec == spec
            return {
                "mode": "TEACHER",
                "candidate_spec_id": spec["candidate_spec_id"],
                "initial_decision_state_sha256": snapshot_sha,
                "graph": graph,
                "teacher_trace": teacher,
                "teacher_termination_flags": flags,
                "teacher_completed_without_termination": False,
                "rgb": np.zeros((168, 224, 3), dtype=np.uint8),
                "snapshot": snapshot,
                "probe_arrays": probe_arrays,
                "probe_metadata": probe_metadata,
                "contact_instrumentation": {
                    "api": "robot.get_contacts",
                    "sample_period_s": R.TRACE_DT_S,
                    "forbidden_net_force_api_used": False,
                    "ontology_sha256": C.CONTACT_AUTHORITY["ontology_sha256"],
                },
                "runtime_evidence": {
                    "snapshot_captured_before_teacher": True,
                    "teacher_restored_from_serialized_snapshot": True,
                    "teacher_snapshot_sha256": snapshot_sha,
                },
            }

    official, material = _patch_one_pool_stage(tmp_path, monkeypatch)
    metadata = R.qualify_pool_state_stage(
        0, backend=Backend(), fake_runtime=True
    )
    reopened, arrays = R._load_material_shard(
        material / "qualification/pool-000"
    )
    reduction = R.METRICS.validate_qualification_teacher_raw_evidence(
        reopened, arrays, expected_pool_index=0
    )
    assert metadata["disposition"] == "TEACHER_TERMINATED_UNSAFELY"
    assert reopened["teacher"]["termination_flags"] == flags
    assert reduction["sample_count"] == len(teacher["timestamp_s"])
    assert reduction["teacher_record_raw_projection"]["successor_viable"] is False
    if first_sample_nan:
        assert reduction["route_progress_m"] == 0.0

    teacher_arrays, spans = V1._concat_traces(
        [
            {
                member: arrays[f"teacher__{member}"]
                for member in V1.TEACHER_TRACE_MEMBERS
            }
        ],
        V1.TEACHER_TRACE_MEMBERS,
    )
    teacher_path = official / "teacher_traces.npz"
    V1.atomic_npz(teacher_path, **teacher_arrays)
    record = R._v4_teacher_record(
        reopened,
        arrays,
        trace_index=0,
        pool_index=0,
        span=spans[0],
        selected=False,
        teacher_file=teacher_path,
    )
    assert record["qualification_pool_index"] == 0
    assert record["successor_viable"] is False
    assert C.canonical_json_bytes(record)


def test_probe_tipped_terminal_retains_initial_snapshot_and_partial_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakePhysicalBackend as V1FakeBackend,
        _fake_snapshot,
    )
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v3 import (
        _snapshot_payload,
    )

    spec = C.build_candidate_specs()[0]
    teacher = V1FakeBackend()._teacher(spec)
    snapshot = _fake_snapshot(spec, teacher)
    snapshot["payload_bytes"] = _snapshot_payload()
    snapshot_arrays, snapshot_metadata = R._normalise_snapshot(snapshot)
    semantics = V3._fresh_snapshot_semantics(snapshot["payload_bytes"])
    trace = _partial_probe_trace(11)
    _mark_terminal_pose_tipped(trace)
    flags = {**ZERO_FLAGS, "tipped": True}
    comparison = R._probe_pair_comparison(
        trace, copy.deepcopy(trace), completed=False, left_flags=flags, right_flags=flags
    )
    trials = [
        {
            "trial_index": index,
            "completed": False,
            "trace": copy.deepcopy(trace),
            "termination_flags": flags,
            "final_snapshot_semantic_evidence": None,
            "snapshot_behavioural_digest_v1": None,
            "trace_member_manifests": R._trace_member_manifests(trace),
            "trial_pair_comparison": comparison,
        }
        for index in range(2)
    ]
    probe_arrays, probe_metadata = R._probe_arrays_and_metadata(
        semantics=semantics, trials=trials
    )

    class Backend:
        def qualify(self, candidate_spec: dict) -> dict:
            assert candidate_spec == spec
            return {
                "mode": "PROBE_REJECTION",
                "candidate_spec_id": spec["candidate_spec_id"],
                "disposition": "RESTORATION_PROBE_TIPPED",
                "reason": "deterministic fixture probe tipped",
                "stage_reached": "RESTORATION_PROBE",
                "arrays": {
                    "rgb": np.zeros((168, 224, 3), dtype=np.uint8),
                    **snapshot_arrays,
                    **probe_arrays,
                },
                "snapshot": snapshot_metadata,
                "probe_metadata": probe_metadata,
                "runtime_evidence": {},
            }

    official = tmp_path / "official"
    material = tmp_path / "material"
    official.mkdir()
    for leaf in ("qualification", "fanout"):
        (material / leaf).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(R, "OUTPUT_ROOT", official)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_stage_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        R,
        "_require_initialized",
        lambda: (_synthetic_runtime_contract(), {"specs": C.build_candidate_specs()}),
    )
    monkeypatch.setattr(R, "_bind_physical_runtime", lambda stage, backend: {})
    metadata = R.qualify_pool_state_stage(0, backend=Backend(), fake_runtime=True)
    assert metadata["disposition"] == "RESTORATION_PROBE_TIPPED"
    assert metadata["source_freeze_commit"] == _synthetic_runtime_contract()[
        "source_freeze_commit"
    ]
    assert metadata["runtime_contract_content_digest"] == (
        _synthetic_runtime_contract()["content_digest"]
    )
    assert metadata["snapshot_identity"]["snapshot_behavioural_digest_v1"] is None
    reopened, arrays = R._load_material_shard(material / "qualification/pool-000")
    validated = R.METRICS.validate_qualification_material_shard(
        reopened, reopened_arrays=arrays, expected_pool_index=0
    )
    assert validated["metadata"]["teacher_executed"] is False
    assert not any("final_snapshot_semantic_digest_bytes" in name for name in arrays)


def test_probe_nan_persists_one_unresolved_terminal_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakePhysicalBackend as V1FakeBackend,
    )

    spec = C.build_candidate_specs()[0]
    teacher = V1FakeBackend()._teacher(spec)
    snapshot, _complete_arrays, _complete_metadata = (
        _snapshot_and_complete_probe_fixture(spec, teacher)
    )
    snapshot_arrays, snapshot_metadata = R._normalise_snapshot(snapshot)
    semantics = V3._fresh_snapshot_semantics(snapshot["payload_bytes"])
    trace = _partial_probe_trace(11)
    trace["base_pose_world"][-1, 0] = _nan64(0xA5)
    trace["base_twist_world"][-1, 1] = _nan64(0xB6)
    trace["joint_position"][-1, 2] = _nan64(0xC7)
    trace["joint_velocity"][-1, 3] = _nan64(0xD8)
    flags = {**ZERO_FLAGS, "nan": True}
    comparison = R._probe_pair_comparison(
        trace,
        copy.deepcopy(trace),
        completed=False,
        left_flags=flags,
        right_flags=flags,
    )
    trials = [
        {
            "trial_index": index,
            "completed": False,
            "trace": copy.deepcopy(trace),
            "termination_flags": flags,
            "final_snapshot_semantic_evidence": None,
            "snapshot_behavioural_digest_v1": None,
            "trace_member_manifests": R._trace_member_manifests(trace),
            "trial_pair_comparison": comparison,
        }
        for index in range(2)
    ]
    probe_arrays, probe_metadata = R._probe_arrays_and_metadata(
        semantics=semantics, trials=trials
    )

    class Backend:
        def qualify(self, candidate_spec: dict) -> dict:
            assert candidate_spec == spec
            return {
                "mode": "PROBE_REJECTION",
                "candidate_spec_id": spec["candidate_spec_id"],
                "disposition": "UNRESOLVED_STATE_FAILURE",
                "reason": "fixture restoration probe nan",
                "stage_reached": "RESTORATION_PROBE",
                "arrays": {
                    "rgb": np.zeros((168, 224, 3), dtype=np.uint8),
                    **snapshot_arrays,
                    **probe_arrays,
                },
                "snapshot": snapshot_metadata,
                "probe_metadata": probe_metadata,
                "runtime_evidence": {},
            }

    _official, material = _patch_one_pool_stage(tmp_path, monkeypatch)
    metadata = R.qualify_pool_state_stage(
        0, backend=Backend(), fake_runtime=True
    )
    reopened, payload = R._load_material_shard(
        material / "qualification/pool-000"
    )
    assert metadata["disposition"] == "UNRESOLVED_STATE_FAILURE"
    assert reopened["stage_reached"] == "RESTORATION_PROBE"
    assert reopened["teacher"] is None
    assert reopened["behavioural_probe"]["trial_pair_comparison"]["pass"] is True
    assert all(
        trial["snapshot_behavioural_digest_v1"] is None
        for trial in reopened["behavioural_probe"]["trials"]
    )
    assert payload["probe__0__base_pose_world"].view(np.uint64)[-1, 0] == (
        trace["base_pose_world"].view(np.uint64)[-1, 0]
    )
    assert not any(
        "final_snapshot_semantic_digest_bytes" in name for name in payload
    )
    fabricated = copy.deepcopy(reopened)
    fabricated.pop("content_digest")
    fabricated["behavioural_probe"]["trials"][0][
        "snapshot_behavioural_digest_v1"
    ] = "f" * 64
    fabricated = C.attach_content_digest(fabricated)
    with pytest.raises(R.METRICS.PhysicalGraphEdgeHandoffV4MetricsError):
        R.METRICS.validate_qualification_material_shard(
            fabricated, reopened_arrays=payload, expected_pool_index=0
        )


def test_rejection_record_generation_is_byte_identical() -> None:
    assert R.canonical_bytes(_record(3, "initial_tipped")) == R.canonical_bytes(
        _record(3, "initial_tipped")
    )


def _family_teacher_trace(spec: dict) -> dict[str, np.ndarray]:
    byte_members = {
        "physics_contact",
        "source_region_member",
        "edge_region_member",
        "correct_edge_region_member",
        "wrong_edge_region_member",
        "target_region_member",
    }
    tails = {
        "base_pose_world": (7,),
        "base_twist_world": (6,),
        "joint_position": (12,),
        "joint_velocity": (12,),
        "applied_command": (3,),
        "requested_command": (3,),
        "post_slew_applied_command": (3,),
    }
    trace = {
        member: np.zeros(
            (2, *tails.get(member, ())),
            dtype=np.uint8 if member in byte_members else np.float64,
        )
        for member in V1.TEACHER_TRACE_MEMBERS
    }
    trace["timestamp_s"][:] = [0.002, 0.004]
    spawn = spec["geometry"]["spawn_se2_world"]
    trace["base_pose_world"][:, :2] = np.asarray(spawn[:2], dtype=np.float64)
    trace["base_pose_world"][:, 2] = 0.375
    trace["base_pose_world"][:, 6] = 1.0
    trace["source_region_member"][:] = 1
    return trace


def test_four_nonregistered_family_fixtures_reach_normal_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = R._nonregistered_family_fixture_specs()
    registered = C.build_candidate_specs()
    assert [row["family"] for row in specs] == list(C.V3.FAMILY_IDS)
    assert not ({row["procedural_seed"] for row in specs} & {row["procedural_seed"] for row in registered})

    class Backend:
        def qualify_fixture(self, spec: dict) -> dict:
            return {
                "mode": "TEACHER",
                "teacher_trace": _family_teacher_trace(spec),
                "graph": {
                    "goal_reachable": True,
                    "graph_edge_physically_executable": True,
                },
                "teacher_termination_flags": ZERO_FLAGS,
            }

    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path / "absent-official")
    monkeypatch.setattr(R, "MATERIAL_ROOT", tmp_path / "absent-material")
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", tmp_path / "absent-receipt")
    monkeypatch.setattr(R, "_stage_runtime", lambda *_args, **_kwargs: {})
    projection_calls: list[tuple[dict, dict]] = []
    original_projection = R._canonicalize_v4_crossing_lateral_coordinate

    def legacy_fixture_crossing(
        _poses: object,
        _target: object,
        opening: object,
        _normal: object,
        _competing: object,
    ) -> dict:
        rows = list(opening)
        return {
            "point_world": [
                (float(rows[0][axis]) + float(rows[1][axis])) / 2.0
                for axis in range(2)
            ],
            "lateral_coordinate_m": 999.0,
        }

    def record_projection(crossing: dict, edge: dict) -> dict:
        projection_calls.append((copy.deepcopy(crossing), copy.deepcopy(edge)))
        value = original_projection(crossing, edge)
        assert value is not None and value["lateral_coordinate_m"] != 999.0
        return value

    monkeypatch.setattr(V1, "canonical_port_crossing", legacy_fixture_crossing)
    monkeypatch.setattr(
        R, "_canonicalize_v4_crossing_lateral_coordinate", record_projection
    )
    result = R.family_fixtures_stage(backend=Backend())
    assert result["fixture_count"] == 4
    assert result["all_reached_normal_state_classification_path"] is True
    assert all(row["registered_identity"] is False for row in result["rows"])
    assert len(projection_calls) == 4


def test_binary64_planar_projection_is_exact_across_both_frozen_interpreters() -> None:
    script = r"""
import json
from lewm.safety import physical_graph_edge_handoff_qualification_v4_metrics as M
opening = [[-0.37, 1.11], [0.83, -0.44]]
crossing = [0.123456789012345, 0.234567890123456]
endpoint = [0.777777777777, -0.222222222222]
projected = M.canonical_v1_planar_segment_projection(crossing, opening)
endpoint_projected = M.canonical_v1_planar_segment_projection(endpoint, opening)
print(json.dumps({
    "width": projected["segment_width_m"].hex(),
    "crossing_lateral": projected["lateral_coordinate_m"].hex(),
    "crossing_fraction": (
        projected["lateral_coordinate_m"] / projected["segment_width_m"] + 0.5
    ).hex(),
    "endpoint_lateral_error": abs(
        endpoint_projected["lateral_coordinate_m"]
    ).hex(),
}, sort_keys=True, separators=(",", ":")))
"""
    environment = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    interpreters = (
        Path("/usr/bin/python3"),
        R.REPO_ROOT / ".generated/venvs/genesis_rocm_0_4_6_v1/bin/python",
    )
    outputs = [
        subprocess.run(
            [str(interpreter), "-c", script],
            cwd=R.REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
        )
        for interpreter in interpreters
    ]
    assert all(result.stderr == b"" for result in outputs)
    assert outputs[0].stdout == outputs[1].stdout
    assert json.loads(outputs[0].stdout) == {
        "crossing_fraction": "0x1.03b4e684d046bp-1",
        "crossing_lateral": "0x1.d10404345fff7p-7",
        "endpoint_lateral_error": "0x1.8d487de027eabp-1",
        "width": "0x1.f5d19b0bd7a4ap+0",
    }


def test_v4_crossing_projection_replaces_only_lateral_coordinate() -> None:
    edge = {
        "edge_id": "non-axis-edge",
        "opening_segment_world": [[-0.37, 1.11], [0.83, -0.44]],
    }
    crossing = {
        "edge_id": "non-axis-edge",
        "sample_before": 10,
        "sample_after": 11,
        "fraction": 0.25,
        "point_world": [0.123456789012345, 0.234567890123456],
        "lateral_coordinate_m": 999.0,
    }
    projected = R._canonicalize_v4_crossing_lateral_coordinate(crossing, edge)
    assert projected is not None
    assert crossing["lateral_coordinate_m"] == 999.0
    assert {
        key: value for key, value in projected.items() if key != "lateral_coordinate_m"
    } == {
        key: value for key, value in crossing.items() if key != "lateral_coordinate_m"
    }
    assert projected["lateral_coordinate_m"].hex() == "0x1.d10404345fff7p-7"
    competing = R._canonicalize_v4_competing_crossing_lateral_coordinate(
        crossing, [edge]
    )
    assert competing == projected
    with pytest.raises(R.ExperimentError, match="edge identity drift"):
        R._canonicalize_v4_competing_crossing_lateral_coordinate(
            crossing, [{**edge, "edge_id": "different-edge"}]
        )


def test_complete_population_publishes_panel_terminal_without_downstream_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = C.build_candidate_specs()

    def state(index: int) -> dict:
        return _record(index, "initial_tipped" if index < 4 else "qualified")

    states = [state(index) for index in range(256)]
    official_rows = [
        R.METRICS.build_qualification_disposition_row(
            item,
            material_metadata_binding={
                "path": f"qualification/pool-{index:03d}/metadata.json",
                "bytes": 1,
                "sha256": hashlib.sha256(f"m{index}".encode()).hexdigest(),
            },
            material_payload_binding={
                "path": f"qualification/pool-{index:03d}/payload.npz",
                "bytes": 1,
                "sha256": hashlib.sha256(f"p{index}".encode()).hexdigest(),
            },
            persisted_array_evidence_sha256=hashlib.sha256(
                f"a{index}".encode()
            ).hexdigest(),
        )
        for index, item in enumerate(states)
    ]
    official = tmp_path / "official"
    material = tmp_path / "material"
    official.mkdir()
    (material / "fanout").mkdir(parents=True)
    monkeypatch.setattr(R, "OUTPUT_ROOT", official)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_stage_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        R,
        "_require_initialized",
        lambda: (_synthetic_runtime_contract(), {"specs": specs}),
    )

    def load(directory: Path) -> tuple[dict, dict]:
        index = int(directory.name.removeprefix("pool-"))
        return {
            "pool_index": index,
            "candidate_spec": specs[index],
            "qualified": states[index]["qualified"],
            "teacher_executed": states[index]["teacher_executed"],
        }, {}

    monkeypatch.setattr(R, "_load_material_shard", load)
    monkeypatch.setattr(
        R.METRICS,
        "validate_qualification_material_shard",
        lambda *_args, **_kwargs: {"pass": True},
    )
    def runtime_gate(
        metadata_rows: list[dict], _runtime: dict, **_kwargs: object
    ) -> dict:
        assert len(metadata_rows) == 256
        assert not (official / "qualification_state_dispositions.jsonl").exists()
        assert not (official / "panel_adequacy.json").exists()
        return {"pass": True}

    monkeypatch.setattr(
        R, "_validate_qualification_runtime_before_panel", runtime_gate
    )
    monkeypatch.setattr(
        R,
        "_disposition_ledger_row",
        lambda metadata, *, terminal_directory: official_rows[
            int(terminal_directory.name.removeprefix("pool-"))
        ],
    )
    panel = R.select_teacher_pool_stage(fake_runtime=True)
    assert panel["status"] == "PHYSICAL_HANDOFF_PANEL_INADEQUATE"
    assert panel["shortfall_strata"] == [
        {"family": "STRAIGHT_PASSAGE", "stratum_index": 0}
    ]
    assert panel["downstream_scientific_execution_authorized"] is False
    assert not (official / "teacher_traces.npz").exists()
    assert not (material / "teacher_selection.json").exists()


def test_prepanel_runtime_drift_is_materialisation_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = C.build_candidate_specs()
    official = tmp_path / "official"
    material = tmp_path / "material"
    official.mkdir()
    (material / "fanout").mkdir(parents=True)
    monkeypatch.setattr(R, "OUTPUT_ROOT", official)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_stage_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        R,
        "_require_initialized",
        lambda: (_synthetic_runtime_contract(), {"specs": specs}),
    )

    def load(directory: Path) -> tuple[dict, dict]:
        index = int(directory.name.removeprefix("pool-"))
        return {
            "pool_index": index,
            "candidate_spec": specs[index],
        }, {}

    monkeypatch.setattr(R, "_load_material_shard", load)
    monkeypatch.setattr(
        R.METRICS,
        "validate_qualification_material_shard",
        lambda *_args, **_kwargs: {"pass": True},
    )
    monkeypatch.setattr(
        R,
        "_disposition_ledger_row",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        R.METRICS,
        "build_qualification_runtime_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("aggregate backend core drift")
        ),
    )
    with pytest.raises(
        R.ExperimentError,
        match="STATE_MATERIALISATION_CORRUPT: qualification runtime gate",
    ):
        R.select_teacher_pool_stage(fake_runtime=True)
    assert not (official / "qualification_state_dispositions.jsonl").exists()
    assert not (official / "panel_adequacy.json").exists()


def test_truthful_fake_panel_inadequate_flow_rejects_production_and_publishes_pure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete rejected population is a scientific terminal, not an abort."""

    from scripts import evaluate_physical_graph_edge_handoff_qualification_v4 as E

    root = tmp_path / "physical_graph_edge_handoff_qualification_v4"
    material = tmp_path / "physical_graph_edge_handoff_qualification_v4_material"
    historical_receipt = tmp_path / (
        "physical_graph_edge_handoff_qualification_v1_v2_v3_custody_receipt.json"
    )
    final_receipt = tmp_path / (
        "physical_graph_edge_handoff_qualification_v4_regeneration_receipt.json"
    )
    fixture_path = tmp_path / "v4_fixture.json"
    R._atomic_json(
        fixture_path,
        {
            "regression_results": R.METRICS.build_regression_results(
                {requirement: True for requirement in C.ALL_V4_REGRESSION_IDS}
            )
        },
    )
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", final_receipt)
    monkeypatch.setattr(R, "HISTORICAL_CUSTODY_RECEIPT", historical_receipt)
    monkeypatch.setattr(C, "HISTORICAL_CUSTODY_RECEIPT_PATH", historical_receipt)
    historical_authority = copy.deepcopy(
        C.EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY
    )
    historical_authority.pop("content_digest")
    historical_authority["receipt_path"] = str(historical_receipt)
    monkeypatch.setattr(
        C,
        "EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY",
        C.attach_content_digest(historical_authority),
    )
    monkeypatch.setattr(E, "DEFAULT_HISTORICAL_CUSTODY_RECEIPT", historical_receipt)
    monkeypatch.setattr(R, "DOC_PATHS", {**R.DOC_PATHS, "fixture": fixture_path})
    _emit_temp_historical_receipt(E, historical_receipt, monkeypatch)

    source_commit = "a" * 40
    monkeypatch.setattr(R, "require_runtime_source_freeze", lambda: source_commit)
    monkeypatch.setattr(
        V1,
        "_scene_exclusion_audit",
        lambda _specs: {
            "authority_digest": hashlib.sha256(
                C.canonical_json_bytes(C.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
            ).hexdigest(),
            "checked_before_simulator_creation": True,
            "scene_overlap_count": 0,
            "scene_hash_overlap_count": 0,
            "state_or_episode_overlap_count": 0,
            "seed_overlap_count": 0,
            "path_or_geometry_overlap_count": 0,
            "structured_path_overlap_count": 0,
            "all_zero": True,
        },
    )
    class InitialTippedBackend:
        def qualify(self, spec: dict) -> dict:
            arrays = _initial_tipped_arrays(spec)
            return {
                "mode": "INITIAL_REJECTION",
                "candidate_spec_id": spec["candidate_spec_id"],
                "disposition": "INITIAL_BOUNDARY_TIPPED",
                "reason": "fixture initial tipped boundary",
                "stage_reached": "INITIAL_BOUNDARY",
                "arrays": arrays,
                "diagnostics": {
                    "intended_pose_representation": "xyz_plus_quaternion_xyzw",
                    "termination_flag_order": list(C.TERMINATION_FLAG_ORDER),
                    "available_simulator_diagnostics": sorted(arrays),
                    "previous_applied_command_sha256": (
                        V2.persisted_array_sha256(
                            arrays["previous_applied_command"]
                        )
                    ),
                    "previous_applied_command_dtype": "<f8",
                    "previous_applied_command_shape": [3],
                },
                "runtime_evidence": _synthetic_qualification_backend_runtime(),
            }

    pool = R.initialize_stage(fake_runtime=True, evaluator_module=E)
    assert len(pool["specs"]) == 256
    backend = InitialTippedBackend()
    for pool_index in range(256):
        R.qualify_pool_state_stage(
            pool_index, backend=backend, fake_runtime=True
        )
    adequacy = R.select_teacher_pool_stage(fake_runtime=True)
    assert adequacy["adequate"] is False
    assert adequacy["status"] == "PHYSICAL_HANDOFF_PANEL_INADEQUATE"
    assert adequacy["next_decision"] == (
        "REVISE_GENERATOR_FOR_SHORTFALL_FAMILIES_KEEP_TEACHER_CONTRACT_FROZEN"
    )
    assert not (material / "teacher_selection.json").exists()
    assert not (root / "teacher_traces.npz").exists()
    assert not any((material / "fanout").iterdir())
    assert not any((material / "selected").iterdir())
    assert not any((material / "repeat").iterdir())

    source_observation = {
        "head_commit": source_commit,
        "parent_commit": C.SOURCE_PARENT_COMMIT,
        "freeze_subject": C.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "worktree_clean": True,
        "tracked_source_count": len(C.TRACKED_SOURCE_PATHS),
        "tracked_sources_sha256": "1" * 64,
        "source_closure_path": next(
            path for path in C.TRACKED_SOURCE_PATHS if "source_closure" in path
        ),
        "source_closure_bytes": 1,
        "source_closure_sha256": "2" * 64,
        "source_closure_row_count": len(C.SOURCE_CLOSURE_PATHS),
        "source_closure_live_bytes_exact": True,
        "sealed_path_accesses": 0,
        "ignore_bypasses": 0,
        "metrics_module": R.METRICS.__name__,
    }

    metrics = R.recompute_and_persist_metrics_stage(
        fake_runtime=True, evaluator_module=E
    )
    assert metrics["qualification_runtime_environment"]["fake_runtime"] is True
    with pytest.raises(
        E.RegenerationError,
        match="production qualification contains fake runtime",
    ):
        E.build_regeneration_receipt(
            root,
            metrics_module=R.METRICS,
            source_freeze_observation=source_observation,
            material_root=material,
            historical_custody_receipt=historical_receipt,
        )
    assert not final_receipt.exists()
    assert not (root / "result.json").exists()

    # Publication projection is still exercised against truthful synthetic
    # evidence, without adding a public evaluator bypass or relabeling it real.
    synthetic_receipt = {
        "schema": "physical_graph_edge_handoff_qualification_v4.pytest_only_receipt.v1",
        "truthful_fake_runtime": True,
    }
    V1.atomic_bytes(final_receipt, R.canonical_bytes(synthetic_receipt))
    result = R._write_publication(metrics, synthetic_receipt)
    assert result["primary_classification"] == "PHYSICAL_HANDOFF_PANEL_INADEQUATE"
    assert result["development_only"] is True
    assert result["final_evaluation_eligible"] is False
    assert set(path.name for path in root.iterdir()) == set(
        C.PANEL_INADEQUATE_OUTPUT_LEAVES
    )
    assert sum(1 for path in material.rglob("*") if path.is_file()) == 514
    assert result["models_trained"] == 0
    assert result["qualification_runtime_environment"]["fake_runtime"] is True
    assert result["runtime_environments"] == {
        "qualification": result["qualification_runtime_environment"],
        "physical": None,
        "encoder": None,
        "ranker": None,
        "any_fake_runtime": True,
    }
    assert len(
        (root / "qualification_state_dispositions.jsonl")
        .read_bytes()
        .splitlines()
    ) == 256


def test_compact_teacher_trace_index_resolves_original_qualification_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material"
    material.mkdir()
    selection = {
        "teacher_records": [
            {
                "trace_index": 0,
                "qualification_pool_index": 2,
                "candidate_spec_id": C.build_candidate_specs()[2][
                    "candidate_spec_id"
                ],
            },
            {
                "trace_index": 1,
                "qualification_pool_index": 3,
                "candidate_spec_id": C.build_candidate_specs()[3][
                    "candidate_spec_id"
                ],
            },
        ]
    }
    R._atomic_json(material / "teacher_selection.json", selection)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)

    original = V1._qualification_directory
    with R._teacher_trace_to_pool_namespace():
        assert V1._qualification_directory(0) == material / "qualification/pool-002"
        assert V1._qualification_directory(1) == material / "qualification/pool-003"
    assert V1._qualification_directory is original


def test_adequate_selection_emits_no_placeholder_teacher_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = C.build_candidate_specs()
    states = [
        _record(
            index,
            "initial_tipped"
            if index == 0
            else "probe_tipped"
            if index == 1
            else "unsafe"
            if index == 2
            else "qualified",
        )
        for index in range(256)
    ]
    official_rows = [
        R.METRICS.build_qualification_disposition_row(
            state,
            material_metadata_binding={
                "path": f"qualification/pool-{index:03d}/metadata.json",
                "bytes": 1,
                "sha256": hashlib.sha256(f"m{index}".encode()).hexdigest(),
            },
            material_payload_binding={
                "path": f"qualification/pool-{index:03d}/payload.npz",
                "bytes": 1,
                "sha256": hashlib.sha256(f"p{index}".encode()).hexdigest(),
            },
            persisted_array_evidence_sha256=hashlib.sha256(
                f"a{index}".encode()
            ).hexdigest(),
        )
        for index, state in enumerate(states)
    ]
    teacher = {
        "contact_free": True,
        "left_source_region": True,
        "positive_route_progress": True,
        "competing_port_entered": False,
        "teacher_valid": True,
        "crossing": {
            "normal_dot_displacement_m": 0.1,
            "lateral_coordinate_m": 0.0,
            "sustained_or_target_reached": True,
        },
    }
    trace = _family_teacher_trace(specs[2])

    def load(directory: Path) -> tuple[dict, dict]:
        index = int(directory.name.removeprefix("pool-"))
        preteacher = index in {0, 1}
        metadata = {
            "pool_index": index,
            "candidate_spec": specs[index],
            "candidate_spec_sha256": specs[index]["canonical_spec_sha256"],
            "qualified": states[index]["disposition"] == "QUALIFIED",
            "teacher_executed": not preteacher,
            "teacher": None if preteacher else teacher,
            "initial_decision_state_sha256": "1" * 64,
            "goal_reachable": True,
            "graph_edge_physically_executable": True,
            "disposition": states[index]["disposition"],
            "rejection_reason": (
                None
                if states[index]["disposition"] == "QUALIFIED"
                else states[index]["disposition"]
            ),
        }
        arrays = {
            f"teacher__{member}": value for member, value in trace.items()
        }
        return metadata, arrays

    official = tmp_path / "official"
    material = tmp_path / "material"
    official.mkdir()
    (material / "fanout").mkdir(parents=True)
    monkeypatch.setattr(R, "OUTPUT_ROOT", official)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "_stage_runtime", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        R,
        "_require_initialized",
        lambda: (_synthetic_runtime_contract(), {"specs": specs}),
    )
    monkeypatch.setattr(R, "_load_material_shard", load)
    monkeypatch.setattr(
        R.METRICS,
        "validate_qualification_material_shard",
        lambda *_args, **_kwargs: {"pass": True},
    )
    monkeypatch.setattr(
        R,
        "_validate_qualification_runtime_before_panel",
        lambda *_args, **_kwargs: {"pass": True},
    )
    monkeypatch.setattr(
        R,
        "_disposition_ledger_row",
        lambda metadata, *, terminal_directory: official_rows[
            int(terminal_directory.name.removeprefix("pool-"))
        ],
    )

    def teacher_record(
        metadata: dict,
        _arrays: dict,
        *,
        trace_index: int,
        pool_index: int,
        span: tuple[int, int],
        selected: bool,
        teacher_file: Path,
    ) -> dict:
        return {
            "candidate_spec_id": metadata["candidate_spec"]["candidate_spec_id"],
            "state_id": metadata["candidate_spec"]["state_id"],
            "teacher_trace_id": f"teacher-{trace_index}",
            "trace_index": trace_index,
            "qualification_pool_index": pool_index,
            "trace_slice": {"start": span[0], "stop": span[1]},
            "trace_array_slice_sha256s": {"base_pose_world": "2" * 64},
            "initial_decision_state_sha256": "1" * 64,
            "selected": selected,
        }

    monkeypatch.setattr(R, "_v4_teacher_record", teacher_record)
    selection = R.select_teacher_pool_stage(fake_runtime=True)
    records = selection["teacher_records"]
    assert len(records) == 254
    assert [row["trace_index"] for row in records] == list(range(254))
    assert [row["qualification_pool_index"] for row in records] == list(
        range(2, 256)
    )
    assert all(
        row["candidate_spec_id"]
        == specs[row["qualification_pool_index"]]["candidate_spec_id"]
        for row in records
    )
    assert selection["qualification_rows"][0]["teacher_trace_id"] is None
    assert selection["qualification_rows"][1]["teacher_trace_id"] is None
    assert selection["qualification_rows"][2]["teacher_trace_index"] == 0
    assert selection["qualification_rows"][2]["teacher_valid"] is True
    assert selection["qualification_rows"][2]["qualified"] is False
    assert selection["qualification_rows"][2]["disposition"] == (
        "TEACHER_TERMINATED_UNSAFELY"
    )
    qualified = [row for row in selection["qualification_rows"] if row["qualified"]]
    assert all(
        row["rejection_reason"]
        == (None if row["selected"] else "HASH_ORDER_NOT_SELECTED")
        for row in qualified
    )
    assert selection["rejection_reason_counts"] == {
        "INITIAL_BOUNDARY_TIPPED": 1,
        "RESTORATION_PROBE_TIPPED": 1,
        "TEACHER_TERMINATED_UNSAFELY": 1,
    }


def test_v4_runtime_proof_walks_all_terminal_shards_not_compact_trace_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core = R._stage_runtime("physical", fake_runtime=True)
    monkeypatch.setattr(R, "MATERIAL_ROOT", tmp_path / "material")
    observed: list[int] = []

    def load(directory: Path) -> tuple[dict, dict]:
        pool_index = int(directory.name.removeprefix("pool-"))
        observed.append(pool_index)
        return {
            # Deliberately alternate terminal shapes: runtime proof must not
            # require teacher or snapshot evidence from either kind.
            "schema": (
                "initial-tipped" if pool_index in {0, 1} else "teacher-terminal"
            ),
            "stage_runtime": copy.deepcopy(core),
            "backend_runtime": {"backend": core["backend"]},
        }, {}

    monkeypatch.setattr(R, "_load_material_shard", load)
    selected_specs = C.build_candidate_specs()[2:66]
    selected_shards = {
        spec["state_id"]: (
            {
                "stage_runtime": copy.deepcopy(core),
                "backend_runtime": {"backend": core["backend"]},
            },
            {},
        )
        for spec in selected_specs
    }
    environment = R._v4_physical_runtime_environment_from_shards(
        selected_specs, selected_shards
    )
    assert observed == list(range(256))
    assert len(environment["qualification_runtime_sha256s"]) == 256
    assert len(environment["selected_snapshot_runtime_sha256s"]) == 64


def test_runner_never_reopens_historical_snapshot_or_first_eight_gate() -> None:
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "compare_v1_v2_v3_first_eight_stage" not in source
    assert "_historical_snapshot_semantics" not in source
    assert "add_historical_probe_versions" not in source
    assert C.V3_SOURCE_FREEZE_COMMIT in C.build_contract()["scientific_invariance_authority"].values()


def test_cli_has_no_fake_runtime_or_publication_bypass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = R.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["report", "--allow-fake-runtime"])
    capsys.readouterr()
    source = Path(R.__file__).read_text(encoding="utf-8")
    assert "--fake-runtime" not in source
    assert "--allow-fake-runtime" not in source


def test_preregistration_records_exact_v3_terminal_and_sole_change() -> None:
    text = R._preregistration_text()
    prose = " ".join(text.split())
    assert C.SOLE_SCIENTIFIC_PROCEDURE_CHANGE in (R.__doc__ or "")
    assert "only collection-path change" not in (R.__doc__ or "")
    assert C.V3_TERMINAL_DIAGNOSIS in text
    assert C.SOLE_SCIENTIFIC_PROCEDURE_CHANGE in text
    assert json.dumps(
        C.DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE,
        sort_keys=True,
        separators=(",", ":"),
    ) in text
    assert json.dumps(
        C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY,
        sort_keys=True,
        separators=(",", ":"),
    ) in text
    assert "development_source_audit_disclosure" in Path(R.__file__).read_text(
        encoding="utf-8"
    )
    for statement in (
        "The semantic snapshot contract passed",
        "the behavioural snapshot contract passed",
        "V1, V2, and V3 reproduced the first eight teacher outcomes exactly",
        "Raw Torch snapshot byte equality is no longer a scientific criterion",
        "did not define how a tipped state should be recorded",
        "no panel, candidate fanout, ranker evaluation, or held-out result",
        "V3 is not a scientific handoff result",
    ):
        assert statement in prose


def test_freeze_docs_preserve_and_redigest_custody_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_authority = copy.deepcopy(C.V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY)
    receipt_binding = {
        "path": "/tmp/non-scientific-v4-custody-fixture.json",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    doc_paths = {
        name: tmp_path / path.name for name, path in R.DOC_PATHS.items()
    }
    monkeypatch.setattr(R, "DOC_PATHS", doc_paths)
    monkeypatch.setattr(
        R,
        "_git",
        lambda *arguments: R.PARENT_COMMIT
        if arguments == ("rev-parse", "HEAD")
        else (_ for _ in ()).throw(AssertionError(arguments)),
    )
    monkeypatch.setattr(
        R, "validate_historical_custody_before_creation", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        R, "_historical_custody_binding", lambda: copy.deepcopy(receipt_binding)
    )

    R.build_freeze_documents()

    assert len(doc_paths) == 7
    assert all(path.is_file() for path in doc_paths.values())
    fixture = json.loads(doc_paths["fixture"].read_bytes())
    assert fixture["pre_panel_engineering_correction"] == (
        C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
    )
    assert fixture["producer_reducer_exact_byte_regression_passed"] is True
    closure = json.loads(doc_paths["source_closure"].read_bytes())
    assert closure["pre_panel_engineering_correction"] == (
        C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY
    )
    invariance = json.loads(doc_paths["scientific_invariance"].read_bytes())
    assert invariance["pre_panel_engineering_correction"] == {
        "authority_content_digest": C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY[
            "content_digest"
        ],
        "status": "PARTIAL_QUALIFICATION_INVALIDATED_RESTART_REQUIRED",
        "existing_partial_material_reuse_authorized": False,
        "restart_from_fresh_v4_roots_required": True,
        "scientific_formula_or_decision_changed": False,
    }
    custody_raw = doc_paths["historical_custody_binding"].read_bytes()
    custody = json.loads(custody_raw)
    assert custody_raw == R.canonical_bytes(custody)
    C.validate_content_digest(custody)
    projected = copy.deepcopy(custody)
    projected.pop("content_digest")
    assert projected.pop("bound_external_receipt") == receipt_binding
    expected = copy.deepcopy(original_authority)
    expected.pop("content_digest")
    assert projected == expected
    assert C.V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY == original_authority


def test_freeze_docs_reject_tampered_custody_authority_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc_paths = {
        name: tmp_path / path.name for name, path in R.DOC_PATHS.items()
    }
    tampered = copy.deepcopy(C.V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY)
    tampered["content_digest"] = "f" * 64
    monkeypatch.setattr(R, "DOC_PATHS", doc_paths)
    monkeypatch.setattr(C, "V1_V2_V3_CUSTODY_AND_NONREUSE_AUTHORITY", tampered)
    monkeypatch.setattr(
        R,
        "_git",
        lambda *arguments: R.PARENT_COMMIT
        if arguments == ("rev-parse", "HEAD")
        else (_ for _ in ()).throw(AssertionError(arguments)),
    )
    monkeypatch.setattr(
        R, "validate_historical_custody_before_creation", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        R,
        "_historical_custody_binding",
        lambda: (_ for _ in ()).throw(AssertionError("receipt opened before validation")),
    )

    with pytest.raises(R.ExperimentError, match="custody-and-nonreuse authority digest drift"):
        R.build_freeze_documents()
    assert not any(path.exists() for path in doc_paths.values())


def test_truthful_fake_sparse_teacher_flow_rejects_production_and_publishes_pure_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the adequate branch with two genuine pre-teacher rejections."""

    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakeEncoder,
        _FakePhysicalBackend,
        _FakeRanker,
        _fake_snapshot,
    )
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v3 import (
        _snapshot_payload,
        _trace,
    )
    from scripts import evaluate_physical_graph_edge_handoff_qualification_v4 as E

    root = tmp_path / "physical_graph_edge_handoff_qualification_v4"
    material = tmp_path / "physical_graph_edge_handoff_qualification_v4_material"
    historical_receipt = tmp_path / (
        "physical_graph_edge_handoff_qualification_v1_v2_v3_custody_receipt.json"
    )
    final_receipt = tmp_path / (
        "physical_graph_edge_handoff_qualification_v4_regeneration_receipt.json"
    )
    fixture_path = tmp_path / "v4_fixture.json"
    R._atomic_json(
        fixture_path,
        {
            "regression_results": R.METRICS.build_regression_results(
                {requirement: True for requirement in C.ALL_V4_REGRESSION_IDS}
            )
        },
    )
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", final_receipt)
    monkeypatch.setattr(R, "HISTORICAL_CUSTODY_RECEIPT", historical_receipt)
    monkeypatch.setattr(C, "HISTORICAL_CUSTODY_RECEIPT_PATH", historical_receipt)
    historical_authority = copy.deepcopy(
        C.EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY
    )
    historical_authority.pop("content_digest")
    historical_authority["receipt_path"] = str(historical_receipt)
    monkeypatch.setattr(
        C,
        "EXTERNAL_HISTORICAL_CUSTODY_AUTHORITY",
        C.attach_content_digest(historical_authority),
    )
    monkeypatch.setattr(E, "DEFAULT_HISTORICAL_CUSTODY_RECEIPT", historical_receipt)
    monkeypatch.setattr(R, "DOC_PATHS", {**R.DOC_PATHS, "fixture": fixture_path})
    _emit_temp_historical_receipt(E, historical_receipt, monkeypatch)
    source_commit = "a" * 40
    monkeypatch.setattr(R, "require_runtime_source_freeze", lambda: source_commit)
    monkeypatch.setattr(
        V1,
        "_scene_exclusion_audit",
        lambda _specs: {
            "authority_digest": hashlib.sha256(
                C.canonical_json_bytes(C.PRIOR_SCENE_EXCLUSION_AUTHORITY)[:-1]
            ).hexdigest(),
            "checked_before_simulator_creation": True,
            "scene_overlap_count": 0,
            "scene_hash_overlap_count": 0,
            "state_or_episode_overlap_count": 0,
            "seed_overlap_count": 0,
            "path_or_geometry_overlap_count": 0,
            "structured_path_overlap_count": 0,
            "all_zero": True,
        },
    )

    pool_index_by_candidate = {
        spec["candidate_spec_id"]: index
        for index, spec in enumerate(C.build_candidate_specs())
    }

    class PhysicalBackend(_FakePhysicalBackend):
        runtime = _synthetic_qualification_backend_runtime()

        @staticmethod
        def _snapshot_packet(spec: dict, pool_index: int) -> tuple[dict, dict, dict]:
            teacher = _FakePhysicalBackend._teacher(spec)
            snapshot = _fake_snapshot(spec, teacher)
            snapshot["payload_bytes"] = _snapshot_payload(step_index=pool_index)
            semantics = V3._fresh_snapshot_semantics(snapshot["payload_bytes"])
            trace = _trace(
                final_digest=semantics["snapshot_semantic_digest_v1"]
            )
            comparison = V3.METRICS.compare_behavioural_probe_traces(
                trace, copy.deepcopy(trace)
            )
            trials = [
                {
                    "trial_index": trial_index,
                    "completed": True,
                    "trace": copy.deepcopy(trace),
                    "termination_flags": copy.deepcopy(ZERO_FLAGS),
                    "final_snapshot_semantic_evidence": semantics[
                        "semantic_evidence"
                    ],
                    "snapshot_behavioural_digest_v1": (
                        V3.METRICS.snapshot_behavioural_digest(trace)
                    ),
                    "trace_member_manifests": R._trace_member_manifests(trace),
                    "trial_pair_comparison": comparison,
                }
                for trial_index in range(2)
            ]
            probe_arrays, probe_metadata = R._probe_arrays_and_metadata(
                semantics=semantics, trials=trials
            )
            return snapshot, probe_arrays, probe_metadata

        def qualify(self, spec: dict) -> dict:
            pool_index = pool_index_by_candidate[spec["candidate_spec_id"]]
            if pool_index == 0:
                arrays = _initial_tipped_arrays(spec)
                return {
                    "mode": "INITIAL_REJECTION",
                    "candidate_spec_id": spec["candidate_spec_id"],
                    "disposition": "INITIAL_BOUNDARY_TIPPED",
                    "reason": "fixture initial tipped boundary",
                    "stage_reached": "INITIAL_BOUNDARY",
                    "arrays": arrays,
                    "diagnostics": {
                        "intended_pose_representation": (
                            "xyz_plus_quaternion_xyzw"
                        ),
                        "termination_flag_order": list(C.TERMINATION_FLAG_ORDER),
                        "available_simulator_diagnostics": sorted(arrays),
                        "previous_applied_command_sha256": (
                            V2.persisted_array_sha256(
                                arrays["previous_applied_command"]
                            )
                        ),
                        "previous_applied_command_dtype": "<f8",
                        "previous_applied_command_shape": [3],
                    },
                    "runtime_evidence": copy.deepcopy(self.runtime),
                }

            snapshot, probe_arrays, probe_metadata = self._snapshot_packet(
                spec, pool_index
            )
            if pool_index == 1:
                snapshot_arrays, snapshot_metadata = R._normalise_snapshot(snapshot)
                semantics = V3._fresh_snapshot_semantics(snapshot["payload_bytes"])
                trace = _partial_probe_trace(11)
                _mark_terminal_pose_tipped(trace)
                flags = {**ZERO_FLAGS, "tipped": True}
                comparison = R._probe_pair_comparison(
                    trace,
                    copy.deepcopy(trace),
                    completed=False,
                    left_flags=flags,
                    right_flags=flags,
                )
                trials = [
                    {
                        "trial_index": trial_index,
                        "completed": False,
                        "trace": copy.deepcopy(trace),
                        "termination_flags": copy.deepcopy(flags),
                        "final_snapshot_semantic_evidence": None,
                        "snapshot_behavioural_digest_v1": None,
                        "trace_member_manifests": R._trace_member_manifests(trace),
                        "trial_pair_comparison": comparison,
                    }
                    for trial_index in range(2)
                ]
                partial_arrays, partial_metadata = R._probe_arrays_and_metadata(
                    semantics=semantics, trials=trials
                )
                return {
                    "mode": "PROBE_REJECTION",
                    "candidate_spec_id": spec["candidate_spec_id"],
                    "disposition": "RESTORATION_PROBE_TIPPED",
                    "reason": "fixture restoration probe tipped",
                    "stage_reached": "RESTORATION_PROBE",
                    "arrays": {
                        "rgb": np.zeros((168, 224, 3), dtype=np.uint8),
                        **snapshot_arrays,
                        **partial_arrays,
                    },
                    "snapshot": snapshot_metadata,
                    "probe_metadata": partial_metadata,
                    "runtime_evidence": copy.deepcopy(self.runtime),
                }

            value = super().qualify(spec)
            teacher_trace = value["teacher_trace"]
            value["graph"]["teacher_positive_route_progress"] = (
                V1._teacher_route_progress_m(
                    teacher_trace["base_pose_world"],
                    spec["geometry"]["selected_directed_edge"][
                        "opening_segment_world"
                    ],
                )
                > 0.0
            )
            value["graph"]["teacher_competing_port_entered"] = (
                V1._first_competing_crossing(
                    teacher_trace["base_pose_world"],
                    spec["geometry"]["competing_directed_edges"],
                )
                is not None
            )
            snapshot_sha = hashlib.sha256(snapshot["payload_bytes"]).hexdigest()
            value.update(
                {
                    "mode": "TEACHER",
                    "initial_decision_state_sha256": snapshot_sha,
                    "snapshot": snapshot,
                    "probe_arrays": probe_arrays,
                    "probe_metadata": probe_metadata,
                    "teacher_termination_flags": copy.deepcopy(ZERO_FLAGS),
                    "teacher_completed_without_termination": True,
                }
            )
            value["runtime_evidence"]["teacher_snapshot_sha256"] = snapshot_sha
            return value

    backend = PhysicalBackend()
    pool = R.initialize_stage(fake_runtime=True, evaluator_module=E)
    assert len(pool["specs"]) == 256
    for pool_index in range(256):
        R.qualify_pool_state_stage(
            pool_index, backend=backend, fake_runtime=True
        )
    selection = R.select_teacher_pool_stage(fake_runtime=True)
    assert len(selection["teacher_records"]) == 254
    assert selection["teacher_records"][0]["trace_index"] == 0
    assert selection["teacher_records"][0]["qualification_pool_index"] == 2
    assert not any(
        row["qualification_pool_index"] in {0, 1}
        for row in selection["teacher_records"]
    )
    selected_state_ids = [row["state_id"] for row in selection["selected_specs"]]
    for state_id in selected_state_ids:
        R.capture_selected_state_stage(
            state_id, backend=backend, fake_runtime=True
        )
    panel = R.freeze_panel_stage(fake_runtime=True)
    assert len(panel["states"]) == 64
    assert {row["candidate_spec_id"] for row in panel["states"]}.isdisjoint(
        {C.build_candidate_specs()[0]["candidate_spec_id"], C.build_candidate_specs()[1]["candidate_spec_id"]}
    )
    latent = R.encode_canonical_pixels_stage(
        encoder=_FakeEncoder(), fake_runtime=True
    )
    assert len(latent["records"]) == 1
    development = [row for row in panel["states"] if row["role"] == "DEVELOPMENT"]
    heldout = [
        row for row in panel["states"] if row["role"] == "DEVELOPMENT_HELDOUT"
    ]
    assert (len(development), len(heldout)) == (48, 16)
    for state in development:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    R.development_target_selection_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    for state in heldout:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    R.heldout_ranker_scores_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    for state in heldout:
        R.repeat_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    assembled = R.assemble_row_evidence_stage(fake_runtime=True)
    assert assembled["candidate_trace_count"] == 960
    del backend, pool, selection, panel, latent, development, heldout
    gc.collect()

    external_rows = []
    for index, row in enumerate(C.EXTERNAL_ARTIFACT_BINDINGS):
        path = tmp_path / f"external-{index}.bin"
        path.write_bytes(f"{row['role']}\n".encode())
        external_rows.append(
            {
                "role": row["role"],
                "kind": row["kind"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": R.sha256_file(path),
            }
        )
    monkeypatch.setattr(
        R.METRICS,
        "external_artifact_bindings",
        lambda _contract: copy.deepcopy(external_rows),
    )
    source_observation = {
        "head_commit": source_commit,
        "parent_commit": C.SOURCE_PARENT_COMMIT,
        "freeze_subject": C.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "worktree_clean": True,
        "tracked_source_count": len(C.TRACKED_SOURCE_PATHS),
        "tracked_sources_sha256": "1" * 64,
        "source_closure_path": next(
            path for path in C.TRACKED_SOURCE_PATHS if "source_closure" in path
        ),
        "source_closure_bytes": 1,
        "source_closure_sha256": "2" * 64,
        "source_closure_row_count": len(C.SOURCE_CLOSURE_PATHS),
        "source_closure_live_bytes_exact": True,
        "sealed_path_accesses": 0,
        "ignore_bypasses": 0,
        "metrics_module": R.METRICS.__name__,
    }

    metrics = R.recompute_and_persist_metrics_stage(
        fake_runtime=True, evaluator_module=E
    )
    assert metrics["qualification_runtime_environment"]["fake_runtime"] is True
    with pytest.raises(
        E.RegenerationError,
        match="official physical runtime evidence is absent or marked fake",
    ):
        E.build_regeneration_receipt(
            root,
            metrics_module=R.METRICS,
            source_freeze_observation=source_observation,
            material_root=material,
            historical_custody_receipt=R.HISTORICAL_CUSTODY_RECEIPT,
        )
    assert not final_receipt.exists()
    assert not (root / "result.json").exists()

    synthetic_receipt = {
        "schema": "physical_graph_edge_handoff_qualification_v4.pytest_only_receipt.v1",
        "truthful_fake_runtime": True,
    }
    V1.atomic_bytes(final_receipt, R.canonical_bytes(synthetic_receipt))
    result = R._write_publication(metrics, synthetic_receipt)
    assert result["models_trained"] == 0
    assert result["development_only"] is True
    assert result["final_evaluation_eligible"] is False
    assert result["qualification_runtime_environment"]["fake_runtime"] is True
    assert (
        result["qualification_runtime_environment"][
            "qualification_stage_runtime_sha256s"
        ]
        == result["runtime_environments"]["physical"][
            "qualification_runtime_sha256s"
        ]
    )
    assert final_receipt.is_file()
    assert set(path.name for path in root.iterdir()) == set(C.SUCCESS_OUTPUT_LEAVES)
    # Preserve the two inherited, ordinary scratch receipts alongside the
    # 803 contract/selection/shard files.
    assert sum(1 for path in material.rglob("*") if path.is_file()) == 805
