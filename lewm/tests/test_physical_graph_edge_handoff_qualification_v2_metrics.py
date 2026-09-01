from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v1_metrics as V1M
from lewm.safety import physical_graph_edge_handoff_qualification_v2_contract as C
from lewm.safety import physical_graph_edge_handoff_qualification_v2_metrics as M


def test_raw_byte_digest_binds_dtype_shape_and_save_reopen_exactly(tmp_path: Path) -> None:
    array = np.asarray([[0.0, 1.0, -2.0]], dtype=np.float64)
    path = tmp_path / "payload.npz"
    np.savez(path, previous_applied_command=array)
    with np.load(path, allow_pickle=False) as archive:
        reopened = {"previous_applied_command": archive["previous_applied_command"].copy()}
    evidence = M.build_persisted_array_evidence(
        shard_kind="TEACHER_POOL",
        shard_id="pool-000",
        payload_file={
            "path": "payload.npz",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        arrays={"previous_applied_command": array},
        reopened_arrays=reopened,
    )
    validated = M.validate_persisted_array_evidence(
        evidence, reopened_arrays=reopened
    )
    manifest = validated["arrays"][0]
    assert manifest["dtype_str"] == "<f8"
    assert manifest["shape"] == [1, 3]
    assert manifest["array_bytes_sha256"] == hashlib.sha256(
        array.tobytes(order="C")
    ).hexdigest()
    assert manifest["array_bytes_sha256"] != M.persisted_array_bytes_sha256(
        array.astype(np.float32)
    )
    wrong_dtype = copy.deepcopy(evidence)
    wrong_dtype["arrays"][0]["dtype_str"] = "<f4"
    wrong_dtype["array_inventory_sha256"] = M._inventory_sha256(
        wrong_dtype["arrays"]
    )
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="reopened"):
        M.validate_persisted_array_evidence(
            wrong_dtype, reopened_arrays=reopened
        )
    wrong_value = copy.deepcopy(reopened)
    wrong_value["previous_applied_command"][0, 0] = 1.0
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="reopened"):
        M.validate_persisted_array_evidence(evidence, reopened_arrays=wrong_value)
    bool_shape = copy.deepcopy(evidence)
    bool_shape["arrays"][0]["shape"] = [True, 3]
    bool_shape["array_inventory_sha256"] = M._inventory_sha256(
        bool_shape["arrays"]
    )
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="shape"):
        M.validate_persisted_array_evidence(bool_shape)
    noncanonical_dtype = copy.deepcopy(evidence)
    noncanonical_dtype["arrays"][0]["dtype_str"] = "float64"
    noncanonical_dtype["array_inventory_sha256"] = M._inventory_sha256(
        noncanonical_dtype["arrays"]
    )
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="dtype.str"):
        M.validate_persisted_array_evidence(noncanonical_dtype)


def test_noncontiguous_view_is_copied_without_dtype_cast() -> None:
    base = np.arange(20, dtype=np.float64).reshape(4, 5)
    view = base[:, ::2]
    assert not view.flags.c_contiguous
    row = M.persisted_array_manifest_row("view", view)
    contiguous = np.ascontiguousarray(view)
    assert row["dtype_str"] == view.dtype.str == contiguous.dtype.str
    assert row["shape"] == list(view.shape)
    assert row["array_bytes_sha256"] == hashlib.sha256(
        contiguous.tobytes(order="C")
    ).hexdigest()
    assert M.validate_npz_archive_comment(C.NPZ_ARCHIVE_COMMENT.encode()) == (
        C.NPZ_ARCHIVE_COMMENT
    )
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="provenance"):
        M.validate_npz_archive_comment(b"")


def test_identified_previous_command_field_uses_raw_bytes_and_separate_shape() -> None:
    array = np.zeros(3, dtype=np.float64)
    raw_sha = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    assert raw_sha == (
        "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0"
    )
    evidence = M.build_persisted_array_evidence(
        shard_kind="TEACHER_POOL",
        shard_id="pool-000",
        payload_file={"path": "payload.npz", "bytes": 1, "sha256": "1" * 64},
        arrays={"snapshot__previous_applied_command": array},
        reopened_arrays={"snapshot__previous_applied_command": array.copy()},
    )
    snapshot = {"previous_applied_command_sha256": raw_sha}
    binding = M.validate_snapshot_previous_applied_command_binding(
        snapshot,
        evidence,
        reopened_arrays={"snapshot__previous_applied_command": array.copy()},
    )
    assert binding == {
        "member": "snapshot__previous_applied_command",
        "dtype_str": "<f8",
        "shape": [3],
        "array_bytes_sha256": raw_sha,
    }
    canonical = C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"][
        "v1_persisted_float64_canonical_header_sha256"
    ]
    assert canonical != raw_sha
    legacy = {"previous_applied_command_sha256": canonical}
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="raw-byte"):
        M.validate_snapshot_previous_applied_command_binding(legacy, evidence)

    records = [
        {"previous_applied_command_sha256": raw_sha}
        for _ in range(C.STATE_COUNT)
    ]
    index = {
        "schema": "physical_graph_edge_handoff_qualification_v2.state_snapshot_index.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "records": records,
    }
    assert M.validate_state_snapshot_previous_applied_command_hashes(
        index, [raw_sha] * C.STATE_COUNT
    ) == [raw_sha] * C.STATE_COUNT
    bad = copy.deepcopy(records)
    bad[0]["previous_applied_command_sha256"] = canonical
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="raw-byte"):
        M.validate_state_snapshot_previous_applied_command_hashes(
            {**index, "records": bad}, [raw_sha] * C.STATE_COUNT
        )


def test_candidate_port_metric_alignment_uses_actual_port_and_only_three_fields() -> None:
    poses = np.zeros((C.PHYSICS_STEPS_PER_BRANCH, 7), dtype=np.float64)
    poses[:-1, 0] = np.linspace(0.0, 0.4, C.PHYSICS_STEPS_PER_BRANCH - 1)
    poses[-1, 0] = 0.5
    poses[-1, 1] = 0.2
    observed = M.derive_candidate_port_metrics(
        np.asarray([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        poses,
        [1.0, 0.0, 0.0],
    )
    assert set(observed) == {
        "port_progress_m",
        "lateral_error_m",
        "positive_port_progress",
    }
    assert observed["port_progress_m"] == pytest.approx(1.0 - np.hypot(0.5, 0.2))
    assert observed["lateral_error_m"] == pytest.approx(0.2)
    assert observed["positive_port_progress"] is True
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="dimensions"):
        M.derive_candidate_port_metrics([0.0, 0.0], poses[:-1], [1.0, 0.0, 0.0])


def test_recompute_bridges_only_raw_previous_command_for_frozen_v1_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_sha = C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"][
        "first_eight_expected_raw_bytes_sha256"
    ]
    canonical_sha = C.PERSISTED_ARRAY_HASH_AUTHORITY["identified_defect_field"][
        "v1_persisted_float64_canonical_header_sha256"
    ]
    state_snapshot_index = C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v2.state_snapshot_index.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "records": [
                {
                    "previous_applied_command_sha256": raw_sha,
                    "sentinel": index,
                }
                for index in range(C.STATE_COUNT)
            ],
        }
    )
    evidence = {
        "npz_inspections": [],
        "state_snapshot_index": state_snapshot_index,
    }
    original_evidence = copy.deepcopy(evidence)
    inspection = {
        "state_snapshots.npz": {
            "members": {
                "previous_applied_command": {
                    "row_or_slice_sha256s": [canonical_sha] * C.STATE_COUNT
                }
            }
        }
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(M.V1M, "validate_npz_inspections", lambda _value: inspection)

    def fake_recompute(value: dict[str, object]) -> dict[str, object]:
        C.V1.validate_content_digest(value["state_snapshot_index"])
        captured.update(value)
        return {
            "schema": "physical_graph_edge_handoff_qualification_v1.metrics.v1",
            "experiment_id": C.V1_EXPERIMENT_ID,
        }

    monkeypatch.setattr(M.V1M, "recompute_metrics", fake_recompute)
    result = M.recompute_metrics(evidence)
    assert result["experiment_id"] == C.EXPERIMENT_ID
    assert evidence == original_evidence
    assert evidence["state_snapshot_index"]["records"][0][
        "previous_applied_command_sha256"
    ] == raw_sha
    bridged = captured["state_snapshot_index"]["records"]
    assert [row["previous_applied_command_sha256"] for row in bridged] == (
        [canonical_sha] * C.STATE_COUNT
    )
    assert [row["sentinel"] for row in bridged] == list(range(C.STATE_COUNT))
    assert captured["state_snapshot_index"]["content_digest"] != (
        state_snapshot_index["content_digest"]
    )


def test_all_ten_regressions_and_scientific_invariance_receipt_recompute() -> None:
    results = M.build_regression_results(
        first_pool_production_writer_validator_passed=True
    )
    assert [row["requirement_id"] for row in results] == list(
        C.REGRESSION_REQUIREMENT_IDS
    )
    assert all(row["passed"] for row in results)
    receipt = M.build_scientific_invariance_receipt(C.build_contract(), results)
    assert "content_digest" not in receipt
    assert receipt["scientific_constants_equal"] is True
    assert receipt["official_documents_and_result_use_v2_identity_only"] is True
    assert receipt["v1_compatibility_identity_scopes_exact"] is True
    assert receipt["inherited_implementation_alignment_disposition"] == (
        C.PORT_HEADING_ALIGNMENT_DISPOSITION
    )
    assert receipt["port_heading_alignment_authority_content_digest"] == (
        C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"]
    )
    assert receipt["candidate_port_metric_alignment_disposition"] == (
        C.CANDIDATE_PORT_METRIC_ALIGNMENT_DISPOSITION
    )
    assert receipt["candidate_port_metric_alignment_authority_content_digest"] == (
        C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY["content_digest"]
    )
    assert receipt["v1_scientific_constants_sha256"] == (
        receipt["v2_scientific_constants_sha256"]
    )
    assert M.validate_scientific_invariance_receipt(receipt) == receipt
    tampered = copy.deepcopy(receipt)
    tampered["regression_results"][1]["passed"] = False
    tampered["pass"] = False
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError):
        M.validate_scientific_invariance_receipt(tampered)
    failed_writer = M.build_regression_results(
        first_pool_production_writer_validator_passed=False
    )
    failed_receipt = M.build_scientific_invariance_receipt(
        C.build_contract(), failed_writer
    )
    assert failed_receipt["pass"] is False
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError):
        M.validate_scientific_invariance_receipt(failed_receipt)


def test_external_v1_custody_receipt_is_exact_and_internal_nonreuse_is_strict() -> None:
    path = Path(C.V1_CUSTODY_RECEIPT_PATH)
    assert path.stat().st_size == C.V1_CUSTODY_RECEIPT_BINDING["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        C.V1_CUSTODY_RECEIPT_BINDING["sha256"]
    )
    external = json.loads(path.read_text())
    assert M.validate_external_v1_custody_receipt(external) == external
    projection_sha = hashlib.sha256(
        C.canonical_json_bytes(M.v1_custody_projection(external))[:-1]
    ).hexdigest()
    internal = {
        "schema": "physical_graph_edge_handoff_qualification_v2.v1_custody_and_nonreuse.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "v1_source_freeze_commit": C.V1_SOURCE_FREEZE_COMMIT,
        "v2_source_freeze_commit": "1" * 40,
        "external_custody_receipt_binding": copy.deepcopy(
            C.V1_CUSTODY_RECEIPT_BINDING
        ),
        "external_custody_projection_sha256": projection_sha,
        "v1_official_root_unchanged": True,
        "v1_material_root_unchanged": True,
        "v1_payloads_copied_into_v2": 0,
        "v1_hardlinks_into_v2": 0,
        "v1_shared_inodes_with_v2": 0,
        "v1_runtime_artifact_or_shard_reused": False,
        "allowed_read_scope": "CUSTODY_AND_FIRST_EIGHT_COMPARISON_ONLY",
        "pass": True,
    }
    assert M.validate_v1_custody_and_nonreuse(internal) == internal
    assert M.validate_v1_custody_and_nonreuse(
        internal, source_freeze_commit="1" * 40
    ) == internal
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="cross-binding"):
        M.validate_v1_custody_and_nonreuse(
            internal, source_freeze_commit="2" * 40
        )
    copied = copy.deepcopy(internal)
    copied["v1_payloads_copied_into_v2"] = 1
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="nonreuse"):
        M.validate_v1_custody_and_nonreuse(copied)


def _reproduction_row(index: int, *, passes: bool = True) -> dict[str, object]:
    spec = C.build_prospective_pool_specs()[index]
    return {
        "pool_index": index,
        "candidate_spec_id": spec["candidate_spec_id"],
        "state_id": spec["state_id"],
        "scene_id": spec["scene_id"],
        "episode_id": spec["episode_id"],
        "graph_id": spec["graph_id"],
        "identity_equal": passes,
        "snapshot_payload_sha256_equal": True,
        "teacher_trace_member_inventory_equal": True,
        "teacher_trace_dtypes_equal": True,
        "teacher_trace_shapes_equal": True,
        "teacher_trace_logical_arrays_equal": True,
        "contact_sequence_equal": True,
        "stuck_equal": True,
        "qualified_equal": True,
        "rejection_reason_equal": True,
        "all_payload_member_dtypes_equal": True,
        "all_payload_member_shapes_equal": True,
        "shared_logical_array_members_equal": True,
        "noncomparable_v1_known_bad_hash_fields": [
            "snapshot.previous_applied_command_sha256"
        ],
        "pass": passes,
    }


def _reproduction(rows: list[dict[str, object]]) -> dict[str, object]:
    passes = all(bool(row["pass"]) for row in rows)
    return {
        "schema": "physical_graph_edge_handoff_qualification_v2.first_eight_reproduction.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "status": "PASS" if passes else C.REPRODUCTION_MISMATCH_DISPOSITION,
        "v1_custody_receipt_binding": copy.deepcopy(C.V1_CUSTODY_RECEIPT_BINDING),
        "comparison_rule": C.FIRST_EIGHT_REPRODUCTION_AUTHORITY["comparison_rule"],
        "row_count": 8,
        "rows": rows,
        "pass": passes,
        "technical_disposition": None if passes else C.REPRODUCTION_MISMATCH_DISPOSITION,
        "full_collection_authorized": passes,
        "compared_before_pool_index": 8,
        "candidate_ranker_development_heldout_outcomes_opened": 0,
    }


def test_first_eight_success_authorizes_and_any_mismatch_is_terminal() -> None:
    passed = _reproduction([_reproduction_row(index) for index in range(8)])
    assert M.validate_first_eight_reproduction(passed) == passed
    assert M.authorizes_full_v2_collection(passed) is True
    rows = [_reproduction_row(index) for index in range(8)]
    rows[4] = _reproduction_row(4, passes=False)
    failed = _reproduction(rows)
    assert M.validate_first_eight_reproduction(failed)["status"] == (
        "V1_V2_PHYSICAL_REPRODUCTION_MISMATCH"
    )
    assert M.authorizes_full_v2_collection(failed) is False
    forged = copy.deepcopy(failed)
    forged["full_collection_authorized"] = True
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="disposition"):
        M.validate_first_eight_reproduction(forged)
    bool_index = copy.deepcopy(passed)
    bool_index["rows"][0]["pool_index"] = False
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="identity"):
        M.validate_first_eight_reproduction(bool_index)
    nonboolean_equality = copy.deepcopy(failed)
    nonboolean_equality["rows"][4]["identity_equal"] = 0
    with pytest.raises(M.PhysicalGraphEdgeHandoffV2MetricsError, match="not boolean"):
        M.validate_first_eight_reproduction(nonboolean_equality)


def test_reducer_authority_adds_only_v2_custody_and_hash_layers() -> None:
    authority = M.reducer_authority()
    C.validate_content_digest(authority)
    assert authority["experiment_id"] == C.EXPERIMENT_ID
    assert authority["successful_output_leaf_count"] == 26
    assert authority["reproduction_mismatch_output_leaves"] == list(
        C.REPRODUCTION_MISMATCH_LEAVES
    )
    assert set(authority["new_documents"]) == {
        "v1_custody_and_nonreuse",
        "scientific_invariance_receipt",
        "v1_v2_first_eight_reproduction",
    }
    assert authority["v1_compatibility_identity_authority"] == (
        C.V1_COMPATIBILITY_IDENTITY_AUTHORITY
    )
    assert authority["port_heading_implementation_alignment_authority"] == (
        C.PORT_HEADING_IMPLEMENTATION_ALIGNMENT_AUTHORITY
    )
    assert authority["candidate_port_metric_implementation_alignment_authority"] == (
        C.CANDIDATE_PORT_METRIC_IMPLEMENTATION_ALIGNMENT_AUTHORITY
    )
    assert all(
        row["content_digest_forbidden"]
        for row in authority["new_documents"].values()
    )
    trace = M.physical_trace_reduction_authority()
    C.validate_content_digest(trace)
    assert trace["experiment_id"] == C.EXPERIMENT_ID
    assert len(trace["specs"]) == 256
    assert trace["specs"] == V1M.physical_trace_reduction_authority()["specs"]


def test_classification_formula_and_precedence_are_literal_v1() -> None:
    evidence = {
        "teacher_correct_execution_count": 16,
        "coverage_rate": 1.0,
        "ranker_correct_edge_top1_rate": 0.8,
        "ranker_correct_edge_top3_rate": 0.95,
        "ranker_selected_correct_edge_execution_rate": 0.8,
        "ranker_normalized_port_regret": 0.2,
        "oracle_selected_correct_edge_execution_rate": 1.0,
        "oracle_covered_state_correct_execution_rate": 1.0,
        "repeatability_rate": 1.0,
        "command_tracking_pass": True,
        "minimum_family_correct_execution_count": 1,
        "selected_target_id": "TARGET_NODE_CENTRE",
        "selected_target_passes_handoff_gate": True,
        "selected_target_materially_outperforms_node_centre": False,
    }
    assert M.classify_physical_handoff_aggregates(evidence) == (
        V1M.classify_physical_handoff_aggregates(evidence)
    )
    assert M.classify_physical_handoff_aggregates(evidence)[
        "primary_classification"
    ] == "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL"
