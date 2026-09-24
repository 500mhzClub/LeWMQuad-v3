from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import pytest

from lewm.safety import (
    physical_handoff_stratified_generator_successor_v1_contract as C,
)
from lewm.safety import (
    physical_handoff_stratified_generator_successor_v1_metrics as M,
)


def _binding(path: str, marker: str = "a") -> dict:
    return {
        "path": path,
        "bytes": 1,
        "sha256": marker * 64,
        "nlink": 1,
        "ordinary_regular_file": True,
        "resolved_path_ancestor_symlink_count": 0,
    }


def _record(spec: dict, disposition: str) -> dict:
    flags = {name: False for name in C.TERMINATION_FLAGS_FIELDS}
    if disposition == "QUALIFIED":
        criteria = {
            name: True for name in C.TEACHER_QUALIFICATION_COMPONENT_IDS
        }
        state = M.build_state_disposition_record(
            spec,
            stage_reached="COMPLETE",
            initial_termination_flags=flags,
            probe_trial_termination_flags=[flags, flags],
            teacher_termination_flags=flags,
            probe_tip_sample_indices=[None, None],
            teacher_criteria=criteria,
            executable_snapshot_exists=True,
            teacher_executed=True,
            snapshot_identity={
                "artifact_file_sha256": "1" * 64,
                "snapshot_semantic_digest_v1": "2" * 64,
                "snapshot_behavioural_digest_v1": "3" * 64,
            },
        )
    elif disposition == "INITIAL_BOUNDARY_TIPPED":
        tipped = dict(flags)
        tipped["tipped"] = True
        state = M.build_state_disposition_record(
            spec,
            stage_reached="INITIAL_BOUNDARY",
            initial_termination_flags=tipped,
            executable_snapshot_exists=False,
            teacher_executed=False,
            diagnostics_inventory=tuple(sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)),
            payload_member_inventory=tuple(sorted(C.INITIAL_BOUNDARY_PAYLOAD_AUTHORITY)),
        )
    else:
        raise AssertionError(disposition)
    prefix = f"qualification/{spec['stream_id']}/attempt-{spec['attempt_index']:02d}"
    return M.build_generator_terminal_record(
        state,
        source_freeze_commit="4" * 40,
        runtime_contract_content_digest="5" * 64,
        material_metadata_binding=_binding(f"{prefix}/metadata.json", "6"),
        material_payload_binding=_binding(f"{prefix}/payload.npz", "7"),
        persisted_array_evidence_sha256="8" * 64,
    )


def _population(*, failure_kind: str | None = None) -> list[dict]:
    records: list[dict] = []
    for family in C.FAMILY_IDS:
        for stratum in range(C.STRATA_PER_FAMILY):
            target = family == "TURNING_JUNCTION" and stratum == 0
            if not target or failure_kind is None:
                dispositions = ["QUALIFIED"] * 4
            elif failure_kind == "zero":
                dispositions = ["INITIAL_BOUNDARY_TIPPED"] * 64
            elif failure_kind == "low":
                dispositions = ["QUALIFIED"] + ["INITIAL_BOUNDARY_TIPPED"] * 63
            else:
                raise AssertionError(failure_kind)
            records.extend(
                _record(C.build_candidate_spec(family, stratum, attempt), disposition)
                for attempt, disposition in enumerate(dispositions)
            )
    records.sort(key=lambda row: row["candidate_index"])
    return records


@pytest.fixture(scope="module")
def available_records() -> list[dict]:
    return _population()


def test_generator_available_and_both_terminal_branches_are_exact(
    available_records: list[dict],
) -> None:
    available = M.build_generator_metrics(available_records)
    assert available["status"] == C.GENERATOR_PANEL_AVAILABLE
    assert available["terminal_record_count"] == 256
    assert available["completed_stream_count"] == 64
    assert available["missing_stream_count"] == 0
    assert available["duplicated_candidate_identity_count"] == 0
    assert len(available["selected_candidate_indices"]) == 64
    resolved = available["v4_shortfall_resolution"]
    assert resolved["canonical_v4_shortfall_stream_count"] == 12
    assert resolved["resolved_stream_count"] == 12
    assert resolved["partial_stream_count"] == 0
    assert resolved["zero_yield_stream_count"] == 0
    assert resolved["conclusion"] == C.V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING
    assert resolved["supports_shallow_sampling_as_primary_cause"] is True
    assert [
        (row["family"], row["stratum_index"])
        for row in resolved["stream_rows"]
    ] == list(C.V4_CANONICAL_SHORTFALL_STREAMS)

    low = M.build_generator_metrics(_population(failure_kind="low"))
    assert low["status"] == C.GENERATOR_LOW_YIELD
    assert low["next_decision"] == C.TURNING_JUNCTION_GENERATOR_NEXT_DECISION
    failure = low["next_decision_evidence"]["failure_streams"]
    assert len(failure) == 1
    assert failure[0]["valid_initial_state_count"] == 1
    assert "common_physical_parameters" in failure[0]
    low_resolution = low["v4_shortfall_resolution"]
    assert low_resolution["resolved_stream_count"] == 11
    assert low_resolution["partial_stream_count"] == 1
    assert low_resolution["zero_yield_stream_count"] == 0
    assert low_resolution["conclusion"] == C.V4_SHORTFALL_INSUFFICIENT
    assert low_resolution["supports_shallow_sampling_as_primary_cause"] is False

    zero = M.build_generator_metrics(_population(failure_kind="zero"))
    assert zero["status"] == C.GENERATOR_FEASIBILITY_NO_GO
    assert zero["next_decision"] == C.TURNING_JUNCTION_GENERATOR_NEXT_DECISION
    assert zero["zero_yield_streams"] == [
        {"family": "TURNING_JUNCTION", "stratum_index": 0}
    ]
    zero_resolution = zero["v4_shortfall_resolution"]
    assert zero_resolution["resolved_stream_count"] == 11
    assert zero_resolution["partial_stream_count"] == 0
    assert zero_resolution["zero_yield_stream_count"] == 1
    assert zero_resolution["conclusion"] == C.GENERATOR_FEASIBILITY_NO_GO
    assert zero_resolution["intrinsic_infeasibility_established"] is False


def test_generator_metric_coherent_tamper_is_rejected(
    available_records: list[dict],
) -> None:
    metrics = M.build_generator_metrics(available_records)
    tampered = copy.deepcopy(metrics)
    tampered["qualified_count"] -= 1
    tampered["nonqualified_count"] += 1
    tampered.pop("content_digest")
    tampered = C.attach_content_digest(tampered)
    with pytest.raises(M.PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError):
        M.validate_generator_metrics(tampered, available_records)


def test_exact_canonical_binding_rejects_bytes_only_tamper() -> None:
    raw = b'{"value":1}\n'
    binding = M._canonical_file_binding("value.json", raw)
    assert M._validate_exact_canonical_file_binding(
        binding, path="value.json", raw=raw, label="fixture"
    ) == binding
    tampered = copy.deepcopy(binding)
    tampered["bytes"] += 1
    with pytest.raises(
        M.PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError,
        match="content/byte binding drift",
    ):
        M._validate_exact_canonical_file_binding(
            tampered, path="value.json", raw=raw, label="fixture"
        )

    metadata = C.attach_content_digest({"value": 1})
    metadata_raw = C.canonical_json_bytes(metadata)
    metadata_binding = M._canonical_file_binding("metadata.json", metadata_raw)
    assert M._validate_optional_metadata_binding(
        metadata_binding,
        metadata,
        expected_path="metadata.json",
        label="metadata fixture",
    ) == metadata_binding
    metadata_binding["bytes"] += 1
    with pytest.raises(
        M.PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError,
        match="byte/path binding drift",
    ):
        M._validate_optional_metadata_binding(
            metadata_binding,
            metadata,
            expected_path="metadata.json",
            label="metadata fixture",
        )


def test_frozen_v1_beyond_port_normalization_has_near_epsilon_witness() -> None:
    epsilon = float(C.NUMERICAL_TOLERANCES["se2_position_m"])
    normal = np.asarray(
        [math.cos(0.00018), math.sin(0.00018)], dtype=np.float64
    )
    assert float(np.linalg.norm(normal)) < 1.0
    direction = normal / np.linalg.norm(normal)
    distance = np.nextafter(epsilon, math.inf)
    point = direction * distance
    raw_without_normalization = float(point @ normal)
    poses = np.zeros((2, 7), dtype=np.float64)
    poses[:, :2] = point
    target = np.zeros(2, dtype=np.uint8)
    count, early = M._consecutive_beyond_samples_native(
        poses,
        0,
        [[0.0, -1.0], [0.0, 1.0]],
        normal.tolist(),
        target,
    )
    assert raw_without_normalization <= epsilon
    assert float(point @ direction) > epsilon
    assert (count, early) == (2, False)


def test_nominal_v1_stuck_activity_and_port_heading_sources_are_frozen() -> None:
    spec = C.build_candidate_spec("STRAIGHT_PASSAGE", 0, 0)
    spawn_x, spawn_y, spawn_yaw = spec["geometry"]["spawn_se2_world"]
    reset_pose = np.asarray(
        [
            spawn_x,
            spawn_y,
            0.3,
            0.0,
            0.0,
            math.sin(spawn_yaw / 2.0),
            math.cos(spawn_yaw / 2.0),
        ],
        dtype=np.float64,
    )
    poses = np.repeat(reset_pose.reshape(1, 7), 750, axis=0)
    trace = {
        "timestamp_s": np.arange(1, 751, dtype=np.float64) * 0.002,
        "base_pose_world": poses,
        "base_twist_world": np.zeros((750, 6), dtype=np.float64),
        "joint_position": np.zeros((750, 12), dtype=np.float64),
        "joint_velocity": np.zeros((750, 12), dtype=np.float64),
        # Deliberately unlike candidate 0's nominal straight command: this raw
        # tape is validated elsewhere and must not replace the frozen V1 stuck
        # formula's nominal Python candidate-bank source.
        "requested_command": np.zeros((750, 3), dtype=np.float64),
        "post_slew_applied_command": np.zeros((750, 3), dtype=np.float64),
        "physics_contact": np.zeros(750, dtype=np.uint8),
        "source_region_member": np.ones(750, dtype=np.uint8),
        "correct_edge_region_member": np.zeros(750, dtype=np.uint8),
        "wrong_edge_region_member": np.zeros(750, dtype=np.uint8),
        "target_region_member": np.zeros(750, dtype=np.uint8),
    }
    edge = spec["geometry"]["selected_directed_edge"]
    normal = edge["opening_normal_world"]
    normal_heading = math.atan2(float(normal[1]), float(normal[0]))
    opening = edge["opening_segment_world"]
    midpoint = [
        (float(opening[0][axis]) + float(opening[1][axis])) / 2.0
        for axis in range(2)
    ]
    deliberately_misaligned_teacher_yaw = normal_heading + 0.71
    outcome = M._derive_candidate_outcome_native(
        spec,
        trace,
        reset_pose,
        0,
        [*midpoint, deliberately_misaligned_teacher_yaw],
    )
    final_yaw = M._pose_roll_pitch_yaw_xyzw(poses[-1])[2]
    normal_error = abs(M._wrap_angle_v1(final_yaw - normal_heading))
    teacher_yaw_error = abs(
        M._wrap_angle_v1(final_yaw - deliberately_misaligned_teacher_yaw)
    )
    assert outcome["angular_error_rad"] == normal_error
    assert outcome["angular_error_rad"] != teacher_yaw_error
    assert outcome["stuck"] is True


def test_reset_memberships_accept_reopened_ndarray_and_reject_tamper() -> None:
    spec = C.build_candidate_spec("STRAIGHT_PASSAGE", 0, 0)
    spawn_x, spawn_y, spawn_yaw = spec["geometry"]["spawn_se2_world"]
    pose = np.asarray(
        [
            spawn_x,
            spawn_y,
            0.3,
            0.0,
            0.0,
            math.sin(spawn_yaw / 2.0),
            math.cos(spawn_yaw / 2.0),
        ],
        dtype=np.float64,
    )
    poses = np.repeat(pose.reshape(1, 7), 3, axis=0)
    expected = M._expected_reset_region_memberships(poses, spec["geometry"])
    trace = {"base_pose_world": poses, **expected}
    M._validate_reset_region_memberships(trace, spec["geometry"])

    tampered = {name: value.copy() for name, value in trace.items()}
    tampered["source_region_member"][0] ^= np.uint8(1)
    with pytest.raises(
        M.PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError,
        match="region-membership tape drift",
    ):
        M._validate_reset_region_memberships(tampered, spec["geometry"])


def _inventory(attempts: int, *, available: bool) -> dict:
    counts = C.expected_material_inventory_counts(
        attempts, panel_available=available
    )
    paths = [f"file-{index:05d}" for index in range(counts["file_count"])]
    directories = [
        f"directory-{index:05d}" for index in range(counts["directory_count"])
    ]
    return C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "material_inventory_projection.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "root": "/tmp/successor-material",
            "branch": "SUCCESS" if available else "GENERATOR_TERMINAL",
            "attempted_candidate_count": attempts,
            "file_count": len(paths),
            "directory_count": len(directories),
            "files": [
                {
                    **_binding(path, hashlib.sha256(path.encode()).hexdigest()[0]),
                    "sha256": hashlib.sha256(path.encode()).hexdigest(),
                }
                for path in paths
            ],
            "directories": directories,
            "unexpected_file_count": 0,
            "unexpected_directory_count": 0,
            "v4_physical_shard_file_count": 512,
            "v4_physical_shard_sha256_overlap_count": 0,
            "v4_physical_shard_copy_reuse_detected": False,
        }
    )


def _runtime_environment(generator: dict) -> dict:
    physical = {"fake_runtime": False}
    return C.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "generator_runtime_environment.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "source_freeze_commit": "4" * 40,
            "runtime_contract_content_digest": "5" * 64,
            "terminal_record_count": generator["terminal_record_count"],
            "candidate_indices": list(range(generator["terminal_record_count"])),
            "physical_runtime_core": physical,
            "physical_runtime_core_sha256": "1" * 64,
            "backend_runtime_core": {},
            "backend_runtime_core_sha256": "2" * 64,
            "stage_runtime_sha256s": ["1" * 64] * generator["terminal_record_count"],
            "backend_runtime_sha256s": ["2" * 64] * generator["terminal_record_count"],
            "all_stage_runtime_cores_equal": True,
            "all_backend_runtime_cores_equal": True,
            "fake_runtime": False,
            "models_trained": 0,
        }
    )


def _publication_metrics(generator: dict, *, available: bool) -> dict:
    runtime = _runtime_environment(generator)
    observation = C.build_source_freeze_observation(
        source_freeze_commit="4" * 40,
        source_freeze_tree_oid="a" * 40,
        source_closure_content_digest="b" * 64,
        source_closure_file_sha256="c" * 64,
        observed_head_commit_at_scientific_reduction="4" * 40,
    )
    if available:
        primary = C.PRIMARY_CLASSIFICATIONS[0]
        next_decision = C.NEXT_DECISION_BY_CLASSIFICATION[primary]
        panel = {"available": True}
        downstream = {"gate": {"passed": False}}
        runtime_environments = {
            "physical": runtime["physical_runtime_core"],
            "encoder": {"fake_runtime": False},
            "ranker": {"fake_runtime": False},
            "any_fake_runtime": False,
        }
    else:
        primary = generator["primary_classification"]
        next_decision = generator["next_decision"]
        panel = downstream = None
        runtime_environments = {
            "physical": runtime["physical_runtime_core"],
            "encoder": None,
            "ranker": None,
            "any_fake_runtime": False,
        }
    return C.attach_content_digest(
        {
            "schema": "physical_handoff_stratified_generator_successor_v1.metrics.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "generator_status": generator["status"],
            "primary_classification": primary,
            "secondary_classifications": [],
            "next_decision": next_decision,
            "generator_metrics": generator,
            "generator_runtime_environment": runtime,
            "source_freeze_observation": observation,
            "material_inventory_projection": _inventory(
                generator["terminal_record_count"], available=available
            ),
            "panel": panel,
            "downstream": downstream,
            "runtime_environments": runtime_environments,
            "scientific_counters": M._scientific_counters(
                generator, panel_available=available
            ),
            "models_trained": 0,
            "development_only": True,
            "final_evaluation_eligible": False,
        }
    )


def _publication_bindings(metrics: dict) -> dict:
    available = metrics["generator_status"] == C.GENERATOR_PANEL_AVAILABLE
    leaves = C.SUCCESS_OUTPUT_LEAVES if available else C.GENERATOR_TERMINAL_OUTPUT_LEAVES
    result = {
        leaf: _binding(leaf, "a")
        for leaf in leaves
        if leaf not in {"result.json", "result.md", "file_hashes.json"}
    }
    result["metrics.json"] = M._canonical_file_binding(
        "metrics.json", C.canonical_json_bytes(metrics)
    )
    return result


@pytest.mark.parametrize("failure_kind", ["zero", "low"])
def test_generator_terminal_publication_is_exact_and_deterministic(
    failure_kind: str,
) -> None:
    generator = M.build_generator_metrics(_population(failure_kind=failure_kind))
    metrics = _publication_metrics(generator, available=False)
    bindings = _publication_bindings(metrics)
    projection = M.build_result_publication_projection(metrics, bindings)
    assert projection["branch"] == "GENERATOR_TERMINAL"
    assert M.validate_result_publication_projection(
        projection["result_document"],
        recomputed_metrics=metrics,
        scientific_bindings=bindings,
    ) == projection
    report = M.build_result_report_bytes(projection)
    assert C.V4_INTERPRETATION_SENTENCES[0].encode() in report
    assert generator["next_decision"].encode() in report
    assert generator["v4_shortfall_resolution"]["conclusion"].encode() in report


def test_available_publication_surface_is_exact(
    available_records: list[dict],
) -> None:
    generator = M.build_generator_metrics(available_records)
    metrics = _publication_metrics(generator, available=True)
    bindings = _publication_bindings(metrics)
    projection = M.build_result_publication_projection(metrics, bindings)
    assert projection["branch"] == "SUCCESS"
    assert projection["result_document"]["status"] == "COMPLETE"
    report = M.build_result_report_bytes(projection)
    assert report.endswith(b"\n")
    assert C.V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING.encode() in report
    assert (
        b"Held-out comparator alias: PHYSICAL_TEACHER -> TEACHER_TRACE "
        b"(user-facing comparator 4 -> frozen internal condition_id; "
        b"no new teacher execution)."
        in report
    )
    tampered = copy.deepcopy(projection["result_document"])
    tampered["models_trained"] = 1
    tampered.pop("content_digest")
    tampered = C.attach_content_digest(tampered)
    with pytest.raises(M.PhysicalHandoffStratifiedGeneratorSuccessorV1MetricsError):
        M.validate_result_publication_projection(
            tampered,
            recomputed_metrics=metrics,
            scientific_bindings=bindings,
        )


def test_public_metrics_surface_has_every_native_stage_and_publication_api() -> None:
    required = {
        "build_frozen_panel_documents",
        "validate_frozen_panel_documents",
        "validate_selected_reset_material_shard",
        "validate_encoding_material_shard",
        "validate_fanout_material_shard",
        "build_development_target_selection",
        "validate_development_target_selection",
        "build_heldout_scores",
        "validate_heldout_scores",
        "validate_repeatability_material_shard",
        "validate_candidate_fanout_rows",
        "validate_repeatability_rows",
        "recompute_metrics",
        "build_result_publication_projection",
        "validate_result_publication_projection",
        "build_result_report_bytes",
    }
    assert required <= set(M.__all__)
    assert all(callable(getattr(M, name)) for name in required)
    assert not any(name.startswith("project_") for name in M.__all__)
