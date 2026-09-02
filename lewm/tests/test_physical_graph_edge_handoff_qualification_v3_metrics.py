from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v3_contract as C
from lewm.safety import physical_graph_edge_handoff_qualification_v3_metrics as M


def _trace(*, final_digest: str = "a" * 64) -> dict[str, object]:
    row: dict[str, object] = {}
    for member, authority in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items():
        row[member] = np.zeros(
            [750, *authority["shape"][1:]], dtype=np.dtype(authority["descr"])
        )
    row["base_pose_world"][:, 6] = 1.0
    row["timestamp_s"][:] = np.arange(750, dtype=np.float64) * 0.002
    row["requested_command"][:] = np.asarray(
        C.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND, dtype=np.float64
    )
    row["post_slew_applied_command"][:, 0] = 0.2
    row["termination_reason"] = "H3_COMPLETE"
    row["stuck"] = True
    row["final_snapshot_semantic_digest_v1"] = final_digest
    return row


def _identity(artifact: str, semantic: str, trace: dict[str, object]) -> dict[str, str]:
    return {
        "artifact_file_sha256": artifact,
        "snapshot_semantic_digest_v1": semantic,
        "snapshot_behavioural_digest_v1": M.snapshot_behavioural_digest(trace),
    }


def _first_eight_arrays(trace: dict[str, object]) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    rows["trace_offsets"] = np.arange(0, 36_001, 750, dtype=np.int64)
    versions = []
    pools = []
    trials = []
    for pool in range(8):
        for version in (1, 2, 3):
            for trial in (0, 1):
                versions.append(version)
                pools.append(pool)
                trials.append(trial)
    rows["version_code"] = np.asarray(versions, dtype=np.int64)
    rows["pool_index"] = np.asarray(pools, dtype=np.int64)
    rows["trial_index"] = np.asarray(trials, dtype=np.int64)
    rows["stuck"] = np.ones(48, dtype=np.uint8)
    rows["termination_code"] = np.ones(48, dtype=np.int64)
    rows["final_snapshot_semantic_digest_bytes"] = np.tile(
        np.frombuffer(bytes.fromhex(trace["final_snapshot_semantic_digest_v1"]), dtype=np.uint8),
        (48, 1),
    )
    for member in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY:
        rows[member] = np.concatenate([trace[member]] * 48, axis=0)
    return rows


def test_qualification_augmentation_binds_reopened_semantics_and_two_probes() -> None:
    snapshot = {"state": [1, 2], "array": np.arange(4, dtype=np.int64)}
    canonical = M.canonical_semantic_snapshot(snapshot)
    evidence = M.snapshot_semantic_evidence(snapshot)
    payload = np.arange(17, dtype=np.uint8)
    traces = [_trace(), _trace(final_digest="b" * 64)]
    identity = _identity(
        hashlib.sha256(payload.tobytes()).hexdigest(),
        evidence["snapshot_semantic_digest_v1"],
        traces[0],
    )
    built = M.build_qualification_shard_augmentation(
        pool_index=9,
        snapshot_payload_bytes=payload,
        canonical_semantic_bytes=canonical,
        snapshot_semantic_evidence=evidence,
        version_snapshot_identities={"V3": identity},
        behavioural_probe_traces={"V3": traces},
    )
    reopened = {"snapshot_payload_bytes": payload, **built["arrays"]}
    assert M.validate_qualification_shard_augmentation(
        built["metadata"], pool_index=9, reopened_arrays=reopened
    ) == built["metadata"]
    assert built["metadata"]["snapshot_identity"] == identity
    assert built["metadata"]["behavioural_probes"]["V3"][
        "trial_1_behavioural_digest_v1"
    ] == M.snapshot_behavioural_digest(traces[1])
    tampered = dict(reopened)
    tampered["probe__V3__0__policy_output"] = reopened[
        "probe__V3__0__policy_output"
    ].copy()
    tampered["probe__V3__0__policy_output"][10, 0] = 1.0
    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.validate_qualification_shard_augmentation(
            built["metadata"], pool_index=9, reopened_arrays=tampered
        )


def test_behavioural_digest_is_single_trial_and_final_semantic_equality_is_descriptive() -> None:
    left = _trace(final_digest="a" * 64)
    right = _trace(final_digest="b" * 64)
    comparison = M.compare_behavioural_probe_traces(left, right)
    assert comparison["pass"] is True
    assert comparison["final_snapshot_semantic_digest_equal"] is False
    assert M.snapshot_behavioural_digest(left) != M.snapshot_behavioural_digest(right)
    drift = copy.deepcopy(right)
    drift["controller_observation"][10, 0] = 1.0
    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.compare_behavioural_probe_traces(left, drift)


def test_first_eight_material_probe_and_equivalence_gate_are_raw_derived() -> None:
    trace = _trace()
    arrays = _first_eight_arrays(trace)
    projection = M.behavioural_probe_npz_projection_sha256(
        arrays, first_eight_only=True
    )
    records = []
    semantic = "c" * 64
    for pool in range(8):
        versions = {
            version: _identity(
                hashlib.sha256(f"{version}-{pool}".encode()).hexdigest(),
                semantic,
                trace,
            )
            for version in C.BEHAVIOURAL_PROBE_VERSION_ORDER
        }
        records.append(
            M.build_snapshot_equivalence_record(
                pool_index=pool,
                versions=versions,
                behavioural_probe_traces={
                    version: [trace, trace]
                    for version in C.BEHAVIOURAL_PROBE_VERSION_ORDER
                },
                semantic_manifests_equal=True,
            )
        )
    receipt = M.build_first_eight_reproduction(
        records,
        historical_custody_receipt_binding=(
            C.HISTORICAL_CUSTODY_RECEIPT_BINDING
        ),
        first_eight_behavioural_probe_material_binding={
            "path": "reproduction/first_eight_behavioural_probes.npz",
            "bytes": 1,
            "sha256": "e" * 64,
        },
        first_eight_behavioural_probe_projection_sha256=projection,
    )
    assert M.validate_first_eight_reproduction_evidence(receipt, arrays)["pass"] is True
    tampered = {key: value.copy() for key, value in arrays.items()}
    tampered["physics_contact"][0] = 1
    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.validate_first_eight_reproduction_evidence(receipt, tampered)


def test_v3_only_equivalence_and_result_publication_are_exact() -> None:
    trace = _trace()
    identity = _identity("1" * 64, "2" * 64, trace)
    record = M.build_snapshot_equivalence_record(
        pool_index=8,
        versions={"V3": identity},
        behavioural_probe_traces={"V3": [trace, trace]},
        semantic_manifests_equal=True,
    )
    assert record["historical_comparison_applicable"] is False
    assert record["v1_v2_semantic_equal"] is None
    assert record["behavioural_probe_trace_indices"] == {"V3": [48, 49]}
    snapshot_projection = {
        field: None for field in C.V3_SNAPSHOT_QUALIFICATION_METRIC_FIELDS
    }
    custody_projection = {
        field: None for field in C.V3_HISTORICAL_CUSTODY_METRIC_FIELDS
    }
    metrics = C.attach_content_digest(
        {
            "schema": "physical_graph_edge_handoff_qualification_v3.metrics.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "primary_classification": "LOW_LEVEL_PREFIX_EXECUTION_NO_GO",
            "secondary_classifications": [],
            "next_experiment": "STOP",
            "development": {"selected_target_id": "TARGET_NODE_CENTRE", "target_summaries": []},
            "evidence_counts": {}, "panel": {},
            "heldout": {"condition_summaries": []}, "repeatability": {},
            "command_tracking": {}, "runtime_environments": {}, "stratified": {},
            "gate": {"passed": False}, "component_failures": {},
            "v3_snapshot_qualification": snapshot_projection,
            "v3_historical_custody": custody_projection,
        }
    )
    runtime = C.build_runtime_contract(
        "f" * 40, C.HISTORICAL_CUSTODY_RECEIPT_BINDING
    )
    result = M.build_result_document(
        metrics, runtime, metrics_sha256="4" * 64,
        independent_reducer_receipt_sha256="5" * 64,
        runtime_seconds=1.25, scientific_storage_bytes=123,
    )
    assert M.validate_result_document(
        result, metrics, runtime, metrics_sha256="4" * 64,
        independent_reducer_receipt_sha256="5" * 64,
    ) == result
    report = M.build_result_report(result, metrics)
    assert M.validate_result_report(report, result, metrics) == report
    assert [line[3:] for line in report.splitlines() if line.startswith("## ")] == list(
        C.V3_RESULT_REPORT_SECTION_ORDER
    )

    historical_binding = copy.deepcopy(C.HISTORICAL_CUSTODY_RECEIPT_BINDING)
    snapshot_projection.update(
        {
            "qualification_state_count": 256,
            "equivalence_row_count": 256,
            "historical_row_count": 8,
            "v3_only_row_count": 248,
            "first_eight_prefix_exact": True,
            "first_eight_gate_pass": True,
            "all_pass": True,
        }
    )
    custody_projection.update(
        {"external_receipt_binding": historical_binding, "pass": True}
    )
    metrics = C.attach_content_digest(
        {
            key: copy.deepcopy(
                snapshot_projection
                if key == "v3_snapshot_qualification"
                else custody_projection
                if key == "v3_historical_custody"
                else value
            )
            for key, value in metrics.items()
            if key != "content_digest"
        }
    )
    runtime = C.build_runtime_contract("f" * 40, historical_binding)
    scientific_bindings = {
        leaf: {
            "path": leaf,
            "bytes": 1,
            "sha256": hashlib.sha256(leaf.encode()).hexdigest(),
        }
        for leaf in set(C.SUCCESS_OUTPUT_LEAVES)
        - {"result.json", "result.md", "file_hashes.json"}
    }
    for leaf, document in (("contract.json", runtime), ("metrics.json", metrics)):
        payload = C.canonical_json_bytes(document)
        scientific_bindings[leaf] = {
            "path": leaf,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    equivalence_projection = {
        "terminal": False,
        "first_eight_row_count": 8,
        "complete_row_count": 256,
        "historical_row_count": 8,
        "v3_only_row_count": 248,
        "first_eight_receipt_exact_rebuild": True,
        "snapshot_equivalence_index_binding": scientific_bindings[
            "snapshot_equivalence_index.json"
        ],
        "snapshot_behavioural_probe_binding": scientific_bindings[
            "snapshot_behavioural_probes.npz"
        ],
        "snapshot_behavioural_probe_projection_sha256": "6" * 64,
        "first_eight_material_binding": {
            "path": "reproduction/first_eight_behavioural_probes.npz",
            "bytes": 1,
            "sha256": "7" * 64,
        },
        "first_eight_projection_sha256": "8" * 64,
        "all_semantic_and_behavioural_rows_pass": True,
        "scientific_result_authorized": True,
    }
    publication = M.build_result_publication_projection(
        metrics,
        scientific_bindings,
        runtime,
        independent_reducer_receipt_sha256="5" * 64,
        snapshot_equivalence=equivalence_projection,
        historical_custody_receipt_binding=historical_binding,
        runtime_seconds=1.25,
    )
    validated = M.validate_result_publication_projection(
        publication["result_document"],
        recomputed_metrics=metrics,
        scientific_bindings=scientific_bindings,
        runtime_contract=runtime,
        independent_reducer_receipt_sha256="5" * 64,
        snapshot_equivalence=equivalence_projection,
        historical_custody_receipt_binding=historical_binding,
    )
    assert validated == publication
    report_bytes = M.build_result_report_bytes(publication)
    assert report_bytes.endswith(b"\n")
    assert report_bytes.decode() == M.build_result_report(
        publication["result_document"], metrics
    )
    tampered_bindings = copy.deepcopy(scientific_bindings)
    tampered_bindings["snapshot_behavioural_probes.npz"]["bytes"] = 2
    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.validate_result_publication_projection(
            publication["result_document"],
            recomputed_metrics=metrics,
            scientific_bindings=tampered_bindings,
            runtime_contract=runtime,
            independent_reducer_receipt_sha256="5" * 64,
            snapshot_equivalence=equivalence_projection,
            historical_custody_receipt_binding=historical_binding,
        )
