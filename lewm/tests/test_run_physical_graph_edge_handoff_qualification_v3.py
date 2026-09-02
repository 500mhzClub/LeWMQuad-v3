from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v3_contract as C
from lewm.safety import physical_graph_edge_handoff_qualification_v3_metrics as M
from scripts import run_go2_oracle_branch_pilot_v1 as PILOT
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import run_physical_graph_edge_handoff_qualification_v2 as V2
from scripts import run_physical_graph_edge_handoff_qualification_v3 as R


def _snapshot_payload(*, step_index: int = 0) -> bytes:
    snapshot = PILOT.BranchSnapshot(
        solver_state={},
        step_index=step_index,
        last_actions=np.zeros((1, 12), dtype=np.float32),
        harness={},
        rng={},
        counters={},
        goal={},
        identity={},
        boundary={},
        digest="fixture",
    )
    return pickle.dumps(snapshot, protocol=4)


def _fake_production_shaped_snapshot_payload(*, step_index: int = 0) -> bytes:
    """Test-only snapshot with the exact frozen production sentinel layout."""

    import torch

    sentinel = np.zeros((18, 2), dtype=np.float32)
    sentinel[:3] = np.inf
    sentinel[3:6] = -np.inf
    # Production pool-000 contains 506 NumPy arrays in total.  BranchSnapshot's
    # last_actions and the two sentinels account for three, so the explicitly
    # independent filler arrays below provide the remaining 503 without aliases.
    solver_state = {
        "Scene._sim._coupler.rigid_solver.dofs_info.force_range": sentinel.copy(),
        "Scene._sim._coupler.rigid_solver.dofs_info.limit": sentinel.copy(),
        **{
            f"fake.production.array.{index:03d}": np.asarray(
                [index], dtype=np.int64
            )
            for index in range(503)
        },
    }
    snapshot = PILOT.BranchSnapshot(
        solver_state=solver_state,
        step_index=step_index,
        last_actions=np.zeros((1, 12), dtype=np.float32),
        harness={"empty_sets": [set(), set(), set()]},
        rng={
            "fake_torch_cpu_rng": torch.zeros(5056, dtype=torch.uint8),
            "fake_torch_device_rng": [
                torch.zeros(16, dtype=torch.uint8),
                torch.zeros(16, dtype=torch.uint8),
            ],
        },
        counters={},
        goal={},
        identity={},
        boundary={},
        digest="fake-production-shaped-fixture",
    )
    return pickle.dumps(snapshot, protocol=4)


def _trace(*, offset: float = 0.0, final_digest: str = "ab" * 32) -> dict[str, object]:
    count = C.BEHAVIOURAL_PROBE_PHYSICS_SAMPLES
    result: dict[str, object] = {}
    for member, authority in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY.items():
        shape = [count, *authority["shape"][1:]]
        result[member] = np.zeros(shape, dtype=np.dtype(authority["descr"]))
    result["timestamp_s"] = np.arange(1, count + 1, dtype=np.float64) * 0.002
    pose = result["base_pose_world"]
    assert isinstance(pose, np.ndarray)
    pose[:, 0] = np.linspace(offset, offset + 0.2, count)
    pose[:, 2] = 0.35
    pose[:, 6] = 1.0
    for member in ("requested_command", "post_slew_applied_command"):
        command = result[member]
        assert isinstance(command, np.ndarray)
        command[:] = np.asarray(
            C.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND,
            dtype=np.float64,
        )
    controller = result["controller_observation"]
    policy = result["policy_output"]
    assert isinstance(controller, np.ndarray) and isinstance(policy, np.ndarray)
    # One exact controller/action sample governs ten consecutive 2ms samples.
    for act_index in range(75):
        controller[act_index * 10 : (act_index + 1) * 10] = act_index / 100.0
        policy[act_index * 10 : (act_index + 1) * 10] = act_index / 200.0
    result["termination_reason"] = "H3_COMPLETE"
    result["stuck"] = False
    result["final_snapshot_semantic_digest_v1"] = final_digest
    return result


def _identity(payload: bytes, trace: dict[str, object]) -> dict[str, str]:
    semantics = R._fresh_snapshot_semantics(payload)
    return {
        "artifact_file_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_semantic_digest_v1": semantics["snapshot_semantic_digest_v1"],
        "snapshot_behavioural_digest_v1": M.snapshot_behavioural_digest(trace),
    }


def _edge_port_document(
    records: list[dict[str, object]],
    *,
    schema: str = "physical_graph_edge_handoff_qualification_v3.edge_port_index.v1",
    experiment_id: str = C.EXPERIMENT_ID,
) -> dict[str, object]:
    return C.attach_content_digest(
        {
            "schema": schema,
            "experiment_id": experiment_id,
            "records": copy.deepcopy(records),
        }
    )


def test_import_is_source_only_and_paths_are_v3() -> None:
    assert R.EXPERIMENT_ID == "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V3"
    assert R.OUTPUT_ROOT.name == "physical_graph_edge_handoff_qualification_v3"
    assert R.MATERIAL_ROOT.name.endswith("_v3_material")
    assert R.DOC_PATHS["historical_custody_binding"] == (
        R.REPO_ROOT
        / "docs/lewm_go2_physical_graph_edge_handoff_qualification_v3_"
        "v1_v2_custody_binding_2026-09-02.json"
    )
    assert R.DOC_PATHS["historical_custody_binding"].relative_to(
        R.REPO_ROOT
    ).as_posix() in set(C.TRACKED_SOURCE_PATHS)
    assert len(R.ALL_OUTPUT_LEAVES) == 28
    assert set(C.REPRODUCTION_MISMATCH_LEAVES) == {
        "contract.json",
        "v1_v2_custody_and_nonreuse.json",
        "scientific_invariance_receipt.json",
        "v1_v2_v3_first_eight_reproduction.json",
    }


def test_delegated_v1_documents_receive_v3_identity_only() -> None:
    logical = {
        "candidate_spec_id": "pgehq-v1-candidate-000",
        "scene_id": "pgehq-v1-scene-000",
        "state_id": "pgehq-v1-state-000",
        "snapshot_id": "pgehq-v1-snapshot-000",
        "episode_id": "pgehq-v1-episode-000",
        "graph_id": "pgehq-v1-graph-000",
        "procedural_seed": 17,
        "candidate_index": 3,
    }
    value = V1.attach_digest({
        "schema": "physical_graph_edge_handoff_qualification_v1.fixture.v1",
        "experiment_id": "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V1",
        **logical,
    })
    projected = R._v3_identity(value)
    assert projected["schema"] == (
        "physical_graph_edge_handoff_qualification_v3.fixture.v1"
    )
    assert projected["experiment_id"] == C.EXPERIMENT_ID
    assert {key: projected[key] for key in logical} == logical
    assert C.validate_content_digest(projected) == projected
    v2 = R._ORIGINAL_V2_PROJECT_V1_TO_V2(value)
    assert R._v3_identity(v2) == M.project_v2_evidence_to_v3(v2)
    assert R._v3_identity(projected) == projected


def test_v3_canonical_directed_port_identity_bridge_is_exact_and_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_id = "pgehq-v1-state-00-01"
    port = [0.25, -0.125, math.pi / 3.0]
    document = _edge_port_document(
        [{"state_id": state_id, "directed_port_world": port}]
    )
    path = tmp_path / "edge_port_index.json"
    R._atomic_json(path, document)
    before = path.read_bytes()
    original_lookup = V2._canonical_directed_port
    original_experiment = V2.EXPERIMENT_ID
    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path)

    assert R._canonical_directed_port_for_inherited_v2(state_id) == port
    assert path.read_bytes() == before
    assert C.validate_content_digest(R._ordinary_json(path)) == document
    assert V2._canonical_directed_port is original_lookup
    assert V2.EXPERIMENT_ID == original_experiment


@pytest.mark.parametrize(
    "mutation,state_id",
    [
        ("missing", "pgehq-v1-state-00-01"),
        ("duplicate", "pgehq-v1-state-00-01"),
        ("schema", "pgehq-v1-state-00-01"),
        ("experiment", "pgehq-v1-state-00-01"),
        ("digest", "pgehq-v1-state-00-01"),
        ("port", "pgehq-v1-state-00-01"),
        ("uppercase-state", "PGEHQ-V1-STATE-00-01"),
    ],
)
def test_v3_canonical_directed_port_rejects_identity_and_payload_tamper(
    mutation: str,
    state_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record: dict[str, object] = {
        "state_id": "pgehq-v1-state-00-01",
        "directed_port_world": [0.25, -0.125, 0.0],
    }
    records = [record]
    schema = "physical_graph_edge_handoff_qualification_v3.edge_port_index.v1"
    experiment = C.EXPERIMENT_ID
    if mutation == "missing":
        records = [{**record, "state_id": "pgehq-v1-state-00-02"}]
    elif mutation == "duplicate":
        records = [record, copy.deepcopy(record)]
    elif mutation == "schema":
        schema = "physical_graph_edge_handoff_qualification_v2.edge_port_index.v1"
    elif mutation == "experiment":
        experiment = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V2"
    elif mutation == "port":
        record["directed_port_world"] = [True, -0.125, 0.0]
    document = _edge_port_document(
        records, schema=schema, experiment_id=experiment
    )
    if mutation == "digest":
        document["content_digest"] = "0" * 64
    R._atomic_json(tmp_path / "edge_port_index.json", document)
    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path)
    original_lookup = V2._canonical_directed_port
    original_experiment = V2.EXPERIMENT_ID

    with pytest.raises(R.ExperimentError):
        R._canonical_directed_port_for_inherited_v2(state_id)
    assert V2._canonical_directed_port is original_lookup
    assert V2.EXPERIMENT_ID == original_experiment


def test_v3_candidate_port_bridge_changes_only_three_authorized_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _fake_physics_trace,
    )

    metadata = V1.load_json(
        R.V1_MATERIAL_ROOT / "qualification/pool-000/metadata.json"
    )
    spec = copy.deepcopy(metadata["candidate_spec"])
    point = metadata["teacher"]["crossing"]["point_world"]
    normal = spec["geometry"]["selected_directed_edge"]["opening_normal_world"]
    port = [
        float(point[0]),
        float(point[1]),
        math.atan2(float(normal[1]), float(normal[0])),
    ]
    document = _edge_port_document(
        [{"state_id": spec["state_id"], "directed_port_world": port}]
    )
    R._atomic_json(tmp_path / "edge_port_index.json", document)
    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path)
    trace = _fake_physics_trace(spec, 0, correct=False)
    reset_pose = trace["base_pose_world"][0]
    inherited = V2._ORIGINAL_V1_DERIVE_CANDIDATE_OUTCOME(
        spec, trace, reset_pose, 0
    )
    aligned = R._derive_candidate_outcome_authority_alignment(
        spec, trace, reset_pose, 0
    )
    expected = M.derive_candidate_port_metrics(
        reset_pose, trace["base_pose_world"], port
    )
    assert set(expected) == {
        "port_progress_m", "lateral_error_m", "positive_port_progress"
    }
    assert {field: aligned[field] for field in expected} == expected
    restored = copy.deepcopy(aligned)
    for field in expected:
        restored[field] = inherited[field]
    assert restored == inherited


def test_fresh_snapshot_has_three_separate_identities() -> None:
    payload = _snapshot_payload()
    semantic = R._fresh_snapshot_semantics(payload)
    trace = _trace(final_digest=semantic["snapshot_semantic_digest_v1"])
    identity = _identity(payload, trace)
    assert set(identity) == set(C.SNAPSHOT_IDENTITY_FIELDS)
    assert identity["artifact_file_sha256"] == hashlib.sha256(payload).hexdigest()
    assert identity["snapshot_semantic_digest_v1"] == hashlib.sha256(
        semantic["semantic_payload_bytes"]
    ).hexdigest()
    assert identity["snapshot_behavioural_digest_v1"] != identity[
        "snapshot_semantic_digest_v1"
    ]


def test_historical_semantics_use_public_restricted_worker_read_only() -> None:
    path = (
        Path("/home/andrewknowles/RecoveryStorage/LeWMQuad-v3")
        / "physical_graph_edge_handoff_qualification_v1_material"
        / "qualification/pool-000/payload.npz"
    )
    if not path.is_file():
        pytest.skip("immutable historical snapshot is unavailable")
    before = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    result = R._historical_snapshot_semantics(path)
    after = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    assert before == after
    assert result["semantic_payload_bytes"] == result["semantic_evidence"][
        "canonical_semantic_byte_count"
    ]
    assert result["semantic_payload_sha256"] == result[
        "snapshot_semantic_digest_v1"
    ]


def test_policy_output_is_raw_post_act_action_repeated_per_policy_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        V1._GenesisPhysicalSession,
        "_sample",
        lambda self, requested, applied, timestamp_s: {"timestamp_s": timestamp_s},
    )
    session = object.__new__(R._V3GenesisPhysicalSession)
    session._last_controller_observation = np.arange(45, dtype=np.float64)
    session.ctx = SimpleNamespace(
        policy=SimpleNamespace(_last_actions=np.arange(12, dtype=np.float32)[None])
    )
    row = session._sample([0.2, 0, 0], [0.2, 0, 0], 0.002)
    assert np.array_equal(row["controller_observation"], np.arange(45))
    assert np.array_equal(row["policy_output"], np.arange(12))
    assert row["policy_output"] is not session.ctx.policy._last_actions


def test_probe_requested_ledger_preserves_inherited_float32_tape() -> None:
    nominal = np.asarray(C.BEHAVIOURAL_PROBE_COMMAND, dtype=np.float64)
    inherited = np.asarray(
        C.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND, dtype=np.float64
    )
    assert nominal.tolist() == [0.2, 0.0, 0.0]
    assert inherited.tolist() == [float(np.float32(0.2)), 0.0, 0.0]
    assert inherited[0] != nominal[0]
    trace = _trace()
    assert np.array_equal(
        trace["requested_command"],
        np.broadcast_to(inherited, trace["requested_command"].shape),
    )


def test_two_restore_probe_uses_distinct_fresh_sessions_and_trial_zero_identity() -> None:
    payload = _snapshot_payload()
    sessions: list[Session] = []

    class Session:
        def __init__(self) -> None:
            self.restores = 0

        def restore_snapshot(self, observed: bytes) -> object:
            assert observed == payload
            self.restores += 1
            return object()

        def execute_behavioural_probe(self) -> dict[str, object]:
            return {
                key: value
                for key, value in _trace().items()
                if key in C.BEHAVIOURAL_TRACE_MEMBER_AUTHORITY
            }

        def capture_snapshot(self) -> tuple[bytes, dict[str, object], dict[str, object]]:
            return payload, {}, {}

    def factory() -> Session:
        session = Session()
        sessions.append(session)
        return session

    trials = R._execute_probe_trials(factory, payload)
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]
    assert [session.restores for session in sessions] == [1, 1]
    identity = _identity(payload, trials[0]["trace"])
    evidence = R._probe_version_evidence(identity, trials)
    assert identity["snapshot_behavioural_digest_v1"] == trials[0][
        "snapshot_behavioural_digest_v1"
    ]
    assert evidence["trial_1_behavioural_digest_v1"] == trials[1][
        "snapshot_behavioural_digest_v1"
    ]
    assert evidence["trial_stuck"] == [False, False]
    assert evidence["trial_termination_reasons"] == ["H3_COMPLETE"] * 2


def test_pure_qualification_augmentation_saves_reopens_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from scripts import evaluate_physical_graph_edge_handoff_qualification_v3 as E

    payload = _fake_production_shaped_snapshot_payload(step_index=8)
    semantic = R._fresh_snapshot_semantics(payload)
    sentinels = semantic["semantic_evidence"]["nonfinite_sentinel_inventory"]
    expected_sentinels = C.SEMANTIC_SERIALIZER_AUTHORITY[
        "nonfinite_sentinel_authority"
    ]["allowed_numpy_arrays"]
    assert [row["path"] for row in sentinels] == [
        row["path"] for row in expected_sentinels
    ]
    assert all(
        row["dtype_str"] == "<f4"
        and row["shape"] == [18, 2]
        and row["strides_bytes"] == [8, 4]
        and row["positive_infinity_count"] == 6
        and row["negative_infinity_count"] == 6
        and row["nan_count"] == 0
        for row in sentinels
    )
    production = C.SEMANTIC_SERIALIZER_AUTHORITY[
        "production_pool_000_observed"
    ]
    type_counts = {
        row["type"]: row["count"]
        for row in semantic["semantic_evidence"]["type_inventory"]
    }
    assert type_counts["numpy.ndarray"] == production["numpy_array_count"]
    assert type_counts["torch.Tensor"] == production["torch_tensor_count"]
    assert type_counts["builtins.set"] == production["set_count"]
    assert semantic["semantic_evidence"]["reference_alias_edge_count"] == 0
    assert not any(
        len(row["member_object_ids"]) > 1
        for row in semantic["semantic_evidence"]["storage_manifest"]
    )
    assert [row["semantic_device_class"] for row in semantic["semantic_evidence"]["tensor_device_manifest"]] == [
        "cpu", "cpu", "cpu"
    ]
    independently_parsed = E._validate_semantic_worker_result(
        semantic["semantic_payload_bytes"],
        semantic["semantic_evidence"],
        metrics=M,
        serializer_authority=M.semantic_snapshot_serializer_authority(),
    )
    assert independently_parsed["snapshot_semantic_digest_v1"] == semantic[
        "snapshot_semantic_digest_v1"
    ]
    trace = _trace(final_digest=semantic["snapshot_semantic_digest_v1"])
    identity = _identity(payload, trace)
    augmentation = {
        "snapshot_semantic_bytes": semantic["semantic_payload_bytes"],
        "snapshot_semantic_evidence": semantic["semantic_evidence"],
        "snapshot_identity": identity,
        "behavioural_probe_trials": {
            "V3": [
                {
                    "trial_index": index,
                    "trace": copy.deepcopy(trace),
                    "snapshot_behavioural_digest_v1": M.snapshot_behavioural_digest(trace),
                    "trace_member_manifests": R._trace_member_manifests(trace),
                }
                for index in (0, 1)
            ]
        },
        "semantic_evidence_by_version": {"V3": semantic["semantic_evidence"]},
    }
    arrays, metadata = R._augmentation_arrays_and_metadata(
        augmentation,
        pool_index=8,
        snapshot_payload_bytes=np.frombuffer(payload, dtype=np.uint8),
    )
    monkeypatch.setattr(R, "MATERIAL_ROOT", tmp_path)
    with R._v2_storage_namespace():
        directory = tmp_path / "pool-008"
        R._write_material_shard_impl(
            directory,
            {
                "schema": (
                    "physical_graph_edge_handoff_qualification_v3."
                    "teacher_pool_shard.v1"
                ),
                "experiment_id": R.EXPERIMENT_ID,
                **metadata,
            },
            {
                "snapshot_payload_bytes": np.frombuffer(payload, dtype=np.uint8),
                **arrays,
            },
            root=tmp_path,
        )
    reopened_metadata, reopened = R._load_material_shard(directory)
    validated = M.validate_qualification_shard_augmentation(
        {
            field: reopened_metadata[field]
            for field in C.QUALIFICATION_SHARD_AUGMENTATION_FIELDS
        },
        pool_index=8,
        reopened_arrays=reopened,
    )
    assert validated["snapshot_identity"] == identity


def test_fake_production_sentinel_helper_cannot_enter_production_runner() -> None:
    source = (R.REPO_ROOT / "scripts/run_physical_graph_edge_handoff_qualification_v3.py").read_text(
        encoding="utf-8"
    )
    assert "_fake_production_shaped_snapshot_payload" not in source


def test_v3_only_equivalence_row_has_null_historical_and_exact_trace_indices() -> None:
    payload = _snapshot_payload()
    semantic = R._fresh_snapshot_semantics(payload)
    trace = _trace(final_digest=semantic["snapshot_semantic_digest_v1"])
    identity = _identity(payload, trace)
    built = M.build_qualification_shard_augmentation(
        pool_index=8,
        snapshot_payload_bytes=np.frombuffer(payload, dtype=np.uint8),
        canonical_semantic_bytes=semantic["semantic_payload_bytes"],
        snapshot_semantic_evidence=semantic["semantic_evidence"],
        version_snapshot_identities={"V3": identity},
        behavioural_probe_traces={"V3": [trace, copy.deepcopy(trace)]},
    )
    metadata = dict(built["metadata"])
    arrays = {
        "snapshot_payload_bytes": np.frombuffer(payload, dtype=np.uint8),
        **built["arrays"],
    }
    row = R._equivalence_record(8, metadata, arrays, historical_custody=None)
    assert row["versions"] == {"V3": identity}
    assert row["behavioural_probe_trace_indices"] == {"V3": [48, 49]}
    assert row["historical_comparison_applicable"] is False
    assert row["v1_v3_semantic_equal"] is None
    assert row["pass"] is True


def test_first_eight_equivalence_uses_semantics_and_raw_sha_is_descriptive() -> None:
    payload = _snapshot_payload()
    semantic = R._fresh_snapshot_semantics(payload)
    trace = _trace(final_digest=semantic["snapshot_semantic_digest_v1"])
    base_identity = _identity(payload, trace)
    identities = {
        "V1": {**base_identity, "artifact_file_sha256": "11" * 32},
        "V2": {**base_identity, "artifact_file_sha256": "22" * 32},
        "V3": base_identity,
    }
    built = M.build_qualification_shard_augmentation(
        pool_index=0,
        snapshot_payload_bytes=np.frombuffer(payload, dtype=np.uint8),
        canonical_semantic_bytes=semantic["semantic_payload_bytes"],
        snapshot_semantic_evidence=semantic["semantic_evidence"],
        version_snapshot_identities=identities,
        behavioural_probe_traces={
            version: [copy.deepcopy(trace), copy.deepcopy(trace)]
            for version in identities
        },
    )
    historical = {
        "first_eight_pairs": [
            {
                "pool_index": index,
                "v1": {
                    "artifact_file_sha256": "11" * 32,
                    "snapshot_semantic_digest_v1": semantic[
                        "snapshot_semantic_digest_v1"
                    ],
                    "semantic_evidence": semantic["semantic_evidence"],
                },
                "v2": {
                    "artifact_file_sha256": "22" * 32,
                    "snapshot_semantic_digest_v1": semantic[
                        "snapshot_semantic_digest_v1"
                    ],
                    "semantic_evidence": semantic["semantic_evidence"],
                },
            }
            for index in range(8)
        ]
    }
    row = R._equivalence_record(
        0,
        built["metadata"],
        {
            "snapshot_payload_bytes": np.frombuffer(payload, dtype=np.uint8),
            **built["arrays"],
        },
        historical_custody=historical,
    )
    assert row["v1_v2_artifact_file_sha256_equal"] is False
    assert row["v1_v2_semantic_equal"] is True
    assert row["v1_v2_behavioural_equal"] is True
    assert row["behavioural_probe_trace_indices"] == {
        "V1": [0, 1], "V2": [2, 3], "V3": [4, 5]
    }
    assert row["pass"] is True


def test_pool_eight_is_fail_closed_without_first_eight_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(R, "OUTPUT_ROOT", tmp_path / "official")
    monkeypatch.setattr(R, "MATERIAL_ROOT", tmp_path / "material")
    R.OUTPUT_ROOT.mkdir()
    R.MATERIAL_ROOT.mkdir()
    with pytest.raises(R.ExperimentError, match="required file is absent"):
        R.qualify_pool_state_stage(8, backend=object(), fake_runtime=True)


def test_probe_npz_exact_trace_order_and_controller_policy_sampling() -> None:
    rows = []
    for pool_index in range(8):
        for version in C.BEHAVIOURAL_PROBE_VERSION_ORDER:
            for trial_index in (0, 1):
                rows.append((pool_index, version, trial_index, _trace()))
    arrays = R._probe_npz_arrays(rows, first_eight_only=True)
    assert arrays["trace_offsets"].tolist() == list(range(0, 36001, 750))
    assert arrays["version_code"][:6].tolist() == [1, 1, 2, 2, 3, 3]
    controller = arrays["controller_observation"][:750]
    policy = arrays["policy_output"][:750]
    for act_index in range(75):
        assert np.unique(controller[act_index * 10 : (act_index + 1) * 10], axis=0).shape[0] == 1
        assert np.unique(policy[act_index * 10 : (act_index + 1) * 10], axis=0).shape[0] == 1


def test_first_eight_material_npz_is_written_once(tmp_path: Path) -> None:
    rows = [
        (pool, version, trial, _trace())
        for pool in range(8)
        for version in C.BEHAVIOURAL_PROBE_VERSION_ORDER
        for trial in (0, 1)
    ]
    arrays = R._probe_npz_arrays(rows, first_eight_only=True)
    path = tmp_path / "reproduction/first_eight_behavioural_probes.npz"
    R._write_probe_npz_once(path, arrays)
    with pytest.raises(R.ExperimentError, match="not fresh"):
        R._write_probe_npz_once(path, arrays)


def test_probe_session_count_never_reuses_teacher_for_v3_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(R.__file__).read_text()
    qualify = source[source.index("    def qualify(self, candidate_spec"):source.index("    def add_historical_probe_versions")]
    assert "_execute_probe_trials(lambda: self._session(spec), payload)" in qualify
    assert "_execute_probe_trials(session, payload)" not in qualify
    historical = source[source.index("    def add_historical_probe_versions"):source.index("def _qualification_backend_default")]
    assert "_execute_probe_trials(lambda: self._session(spec), payload)" in historical
    assert "for version, root in" in historical


def test_preregistration_freezes_semantic_and_probe_boundary() -> None:
    text = R._preregistration_text()
    assert "snapshot_semantic_digest_v1" in text
    assert "policy._last_actions" in text
    assert "48-trace" in text and "544-trace" in text
    assert C.REPRODUCTION_MISMATCH_DISPOSITION in text
    assert "No model, threshold, scientific formula, gate" in text


def test_freeze_fixture_uses_exact_locked_seven_session_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = C.BEHAVIOURAL_PROBE_AUTHORITY["probe_session_topology"][
        "production_fixture_sessions"
    ]
    assert expected == 7
    paths = {
        name: tmp_path / path.name for name, path in R.DOC_PATHS.items()
    }
    monkeypatch.setattr(R, "DOC_PATHS", paths)
    monkeypatch.setattr(R, "_git", lambda *_args: R.PARENT_COMMIT)
    monkeypatch.setattr(
        R,
        "validate_historical_custody_before_creation",
        lambda **_kwargs: {"pass": True},
    )
    monkeypatch.setattr(
        R, "build_scientific_invariance_receipt", lambda _external: {"pass": True}
    )
    monkeypatch.setattr(
        R,
        "_historical_custody_binding",
        lambda: copy.deepcopy(C.HISTORICAL_CUSTODY_RECEIPT_BINDING),
    )
    R.build_freeze_documents(evaluator_module=object())
    fixture = R._ordinary_json(paths["fixture"])
    assert fixture["production_snapshot_fixture"] == {
        "pool_index": 0,
        "teacher_controller_executions": 0,
        "simulator_sessions": expected,
        "probe_trials": 6,
    }


def test_publication_uses_exact_pure_result_and_report_builders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "official"
    material = tmp_path / "material"
    root.mkdir(); material.mkdir()
    receipt_path = tmp_path / "receipt.json"
    receipt = {"schema": "test.receipt.v1", "pass": True}
    R._atomic_json(receipt_path, receipt)
    historical_binding = copy.deepcopy(C.HISTORICAL_CUSTODY_RECEIPT_BINDING)
    runtime = C.build_runtime_contract("f" * 40, historical_binding)
    R._atomic_json(root / "contract.json", runtime)
    snapshot_projection = {
        field: None for field in C.V3_SNAPSHOT_QUALIFICATION_METRIC_FIELDS
    }
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
    custody_projection = {
        field: None for field in C.V3_HISTORICAL_CUSTODY_METRIC_FIELDS
    }
    custody_projection.update(
        {"external_receipt_binding": historical_binding, "pass": True}
    )
    metrics = C.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v3.metrics.v1"
            ),
            "experiment_id": C.EXPERIMENT_ID,
            "primary_classification": "LOW_LEVEL_PREFIX_EXECUTION_NO_GO",
            "secondary_classifications": [],
            "next_experiment": "STOP",
            "development": {
                "selected_target_id": "TARGET_NODE_CENTRE",
                "target_summaries": [],
            },
            "evidence_counts": {},
            "panel": {},
            "heldout": {"condition_summaries": []},
            "repeatability": {},
            "command_tracking": {},
            "runtime_environments": {},
            "stratified": {},
            "gate": {"passed": False},
            "component_failures": {},
            "v3_snapshot_qualification": snapshot_projection,
            "v3_historical_custody": custody_projection,
        }
    )
    R._atomic_json(root / "metrics.json", metrics)
    R._atomic_json(
        material / "material_contract.json", {"started_at_unix_s": 1.0}
    )
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", receipt_path)
    monkeypatch.setattr(R, "SCIENTIFIC_LEAVES", ("contract.json", "metrics.json"))
    monkeypatch.setattr(
        R,
        "ALL_OUTPUT_LEAVES",
        (
            "contract.json", "metrics.json", "result.json", "result.md",
            "file_hashes.json",
        ),
    )
    scientific_bindings = {
        leaf: {"path": leaf, "bytes": 1, "sha256": "a" * 64}
        for leaf in set(C.SUCCESS_OUTPUT_LEAVES)
        - {"result.json", "result.md", "file_hashes.json"}
    }
    for leaf in ("contract.json", "metrics.json"):
        scientific_bindings[leaf] = R._file_binding(
            root / leaf, relative_to=root
        )
    snapshot_evidence = {
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
        "snapshot_behavioural_probe_projection_sha256": "b" * 64,
        "first_eight_material_binding": {
            "path": "reproduction/first_eight_behavioural_probes.npz",
            "bytes": 1,
            "sha256": "c" * 64,
        },
        "first_eight_projection_sha256": "d" * 64,
        "all_semantic_and_behavioural_rows_pass": True,
        "scientific_result_authorized": True,
    }
    monkeypatch.setattr(R, "_scientific_bindings", lambda: scientific_bindings)
    monkeypatch.setattr(
        R, "_historical_custody_binding", lambda: historical_binding
    )
    monkeypatch.setattr(
        R,
        "_snapshot_equivalence_publication_projection",
        lambda: snapshot_evidence,
    )
    result = R._write_publication(metrics, receipt)
    assert result["v3_snapshot_qualification"] == snapshot_projection
    assert M.validate_result_document(
        result,
        metrics,
        runtime,
        metrics_sha256=R.sha256_file(root / "metrics.json"),
        independent_reducer_receipt_sha256=R.sha256_file(receipt_path),
    ) == result
    report = (root / "result.md").read_text()
    assert M.validate_result_report(report, result, metrics) == report
    manifest = R._ordinary_json(root / "file_hashes.json")
    assert manifest["schema"].endswith("qualification_v3.file_hashes.v1")
    assert [row["path"] for row in manifest["files"]] == sorted(
        ("contract.json", "metrics.json", "result.json", "result.md")
    )


def test_all_fourteen_pre_simulator_semantic_regressions_pass() -> None:
    # The production gate runs under the frozen Genesis/Go2 interpreter, which
    # includes Torch.  Keep the portable system-Python suite useful without
    # weakening that runtime gate.
    pytest.importorskip("torch")
    historical = {
        "first_eight_pairs": [
            {
                "artifact_file_sha256_equal": False,
                "snapshot_semantic_digest_v1_equal": True,
                "semantic_evidence_equal": True,
                "pass": True,
            }
            for _ in range(8)
        ]
    }
    rows = R._semantic_regression_gate(historical)
    assert [row["requirement_id"] for row in rows] == list(
        C.SEMANTIC_SERIALIZER_REGRESSION_IDS
    )
    assert all(row["passed"] for row in rows)


def test_cli_exposes_gate_and_fixture_without_running_them() -> None:
    parser = R.build_parser()
    assert parser.parse_args(["production-snapshot-fixture"]).stage == "production-snapshot-fixture"
    assert parser.parse_args(["compare-v1-v2-v3-first-eight"]).stage == "compare-v1-v2-v3-first-eight"
    assert parser.parse_args(["assemble-snapshot-equivalence"]).stage == "assemble-snapshot-equivalence"


def test_production_fixture_source_forbids_teacher_and_file_emission() -> None:
    source = Path(R.__file__).read_text()
    body = source[
        source.index("def production_snapshot_fixture_stage") :
        source.index("# ---------------------------------------------------------------------------\n# Prospective documents")
    ]
    assert "execute_teacher" not in body
    assert "_write_material_shard" not in body
    assert "atomic_npz" not in body and "_atomic_json" not in body
    assert '"simulator_session_count": 7' in body


def test_production_fixture_failure_reports_frozen_component_diagnostics() -> None:
    semantics = {
        version: {"snapshot_semantic_digest_v1": "a" * 64}
        for version in ("V1", "V2", "V3")
    }
    probes = {
        version: [{"trace": _trace()}, {"trace": _trace()}]
        for version in ("V1", "V2", "V3")
    }
    # Alter a complete policy-act block so the trace remains structurally valid
    # while the frozen comparison reports the precise failing numeric member.
    probes["V2"][1]["trace"]["policy_output"][:10] += 0.25
    diagnostics = R._production_fixture_diagnostics(semantics, probes)
    assert diagnostics["semantic_equal"] is True
    assert diagnostics["within_version_probe_equal"] == {
        "V1": True,
        "V2": False,
        "V3": True,
    }
    comparison = diagnostics["within_version_comparisons"]["V2"]
    assert comparison["failing_samplewise_members"] == ["policy_output"]
    assert comparison["samplewise_max_abs_error"]["policy_output"] == 0.25
    message = R._production_fixture_failure_message(diagnostics)
    assert message.startswith("production snapshot semantic/probe fixture failed: {")
    decoded = json.loads(message.split(": ", 1)[1])
    assert decoded == diagnostics


@pytest.mark.skipif(
    not bool(__import__("os").environ.get("PGEHQ_V3_RUN_PRODUCTION_FIXTURE")),
    reason="explicit pre-freeze physical fixture opt-in required",
)
def test_production_snapshot_fixture_real_read_only() -> None:
    result = R.production_snapshot_fixture_stage()
    assert result["pass"] is True
    assert result["simulator_session_count"] == 7
    assert result["teacher_controller_executions"] == 0


def test_first_eight_probe_mismatch_is_exact_four_leaf_technical_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import evaluate_physical_graph_edge_handoff_qualification_v3 as E

    root = tmp_path / "physical_graph_edge_handoff_qualification_v3"
    material = tmp_path / "physical_graph_edge_handoff_qualification_v3_material"
    external = tmp_path / "physical_graph_edge_handoff_qualification_v3_receipt.json"
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", external)
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
    historical = E.validate_existing_historical_custody_receipt(
        R.HISTORICAL_CUSTODY_RECEIPT
    )
    R.initialize_stage(fake_runtime=True, evaluator_module=E)

    for pool_index in range(8):
        source = (
            R.V1_MATERIAL_ROOT
            / "qualification"
            / f"pool-{pool_index:03d}"
        )
        base_metadata, base_arrays = V1._load_material_shard(source)
        payload = np.ascontiguousarray(
            base_arrays["snapshot_payload_bytes"]
        ).tobytes(order="C")
        semantic = R._fresh_snapshot_semantics(payload)
        trace = _trace(
            final_digest=semantic["snapshot_semantic_digest_v1"]
        )
        v3_trials = [copy.deepcopy(trace), copy.deepcopy(trace)]
        if pool_index == 0:
            # One valid, substantive probe mismatch must terminate before the
            # scientific pipeline opens. Preserve policy-act block structure.
            v3_trials[1]["policy_output"][:10] += 0.25
        pair = historical["first_eight_pairs"][pool_index]
        identities = {
            "V3": {
                "artifact_file_sha256": semantic["artifact_file_sha256"],
                "snapshot_semantic_digest_v1": semantic[
                    "snapshot_semantic_digest_v1"
                ],
                "snapshot_behavioural_digest_v1": (
                    M.snapshot_behavioural_digest(v3_trials[0])
                ),
            }
        }
        for version, key in (("V1", "v1"), ("V2", "v2")):
            item = pair[key]
            identities[version] = {
                "artifact_file_sha256": item["artifact_file_sha256"],
                "snapshot_semantic_digest_v1": item[
                    "snapshot_semantic_digest_v1"
                ],
                "snapshot_behavioural_digest_v1": (
                    M.snapshot_behavioural_digest(trace)
                ),
            }
        augmentation = {
            "snapshot_semantic_bytes": semantic["semantic_payload_bytes"],
            "snapshot_semantic_evidence": semantic["semantic_evidence"],
            "snapshot_identity": identities["V3"],
            "snapshot_identities": {
                version: identities[version] for version in ("V1", "V2")
            },
            "behavioural_probe_trials": {
                "V1": [{"trace": copy.deepcopy(trace)} for _ in (0, 1)],
                "V2": [{"trace": copy.deepcopy(trace)} for _ in (0, 1)],
                "V3": [{"trace": value} for value in v3_trials],
            },
            "semantic_evidence_by_version": {
                "V1": pair["v1"]["semantic_evidence"],
                "V2": pair["v2"]["semantic_evidence"],
                "V3": semantic["semantic_evidence"],
            },
        }
        added_arrays, added_metadata = R._augmentation_arrays_and_metadata(
            augmentation,
            pool_index=pool_index,
            snapshot_payload_bytes=base_arrays["snapshot_payload_bytes"],
        )
        metadata = copy.deepcopy(base_metadata)
        metadata.pop("content_digest")
        metadata.pop("payload")
        metadata["snapshot"]["previous_applied_command_sha256"] = (
            R.persisted_array_sha256(
                base_arrays["snapshot__previous_applied_command"]
            )
        )
        R._write_material_shard_impl(
            material / "qualification" / f"pool-{pool_index:03d}",
            {**metadata, **added_metadata},
            {**base_arrays, **added_arrays},
            root=material,
        )

    reproduction = R.compare_v1_v2_v3_first_eight_stage(evaluator_module=E)
    assert reproduction["pass"] is False
    assert reproduction["full_collection_authorized"] is False
    assert set(path.name for path in root.iterdir()) == set(
        C.REPRODUCTION_MISMATCH_LEAVES
    )
    assert (
        material / "reproduction" / "first_eight_behavioural_probes.npz"
    ).is_file()
    assert not any(
        (material / "qualification" / f"pool-{index:03d}").exists()
        for index in range(8, 256)
    )
    delegate_calls = 0

    def forbidden_delegate(*_args, **_kwargs):
        nonlocal delegate_calls
        delegate_calls += 1
        raise AssertionError("inherited scientific stage opened")

    monkeypatch.setattr(R, "_delegate", forbidden_delegate)
    with pytest.raises(
        R.ExperimentError, match=C.REPRODUCTION_MISMATCH_DISPOSITION
    ):
        R.select_teacher_pool_stage(fake_runtime=True)
    assert delegate_calls == 0
    for leaf in (
        "metrics.json",
        "result.json",
        "result.md",
        "file_hashes.json",
    ):
        assert not (root / leaf).exists()
    assert not external.exists()
    source_observation = {
        "head_commit": source_commit,
        "parent_commit": C.SOURCE_PARENT_COMMIT,
        "freeze_subject": C.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "worktree_clean": True,
        "tracked_source_count": len(C.TRACKED_SOURCE_PATHS),
        "tracked_sources_sha256": "1" * 64,
        "source_closure_path": str(C.TRACKED_SOURCE_PATHS[4]),
        "source_closure_bytes": 1,
        "source_closure_sha256": "2" * 64,
        "source_closure_row_count": len(C.SOURCE_CLOSURE_PATHS),
        "source_closure_live_bytes_exact": True,
        "metrics_module": M.__name__,
    }
    receipt = E.build_regeneration_receipt(
        root,
        metrics_module=M,
        source_freeze_observation=source_observation,
        material_root=material,
        historical_custody_receipt=R.HISTORICAL_CUSTODY_RECEIPT,
    )
    assert receipt["scientific_result_produced"] is False
    assert receipt["technical_disposition"] == C.REPRODUCTION_MISMATCH_DISPOSITION
    assert all(
        int(value) == 0
        for value in receipt["scientific_execution_counters"].values()
    )


@pytest.mark.skipif(
    os.environ.get("RUN_FULL_PGEHQ_V3_FAKE_E2E") != "1",
    reason="full 256-state/960-trace synthetic integration is explicitly requested",
)
def test_complete_fake_v3_flow_reaches_strict_reducer_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the complete V3 wrapper without opening a real outcome.

    The synthetic backend remains explicit.  Only this test rewrites fake
    visual-runtime receipts after first proving that the production evaluator
    rejects them; no fake-to-real adapter exists in production source.
    """

    from lewm.tests.test_run_physical_graph_edge_handoff_qualification_v1 import (
        _FakeEncoder,
        _FakePhysicalBackend,
        _FakeRanker,
    )
    from scripts import evaluate_physical_graph_edge_handoff_qualification_v3 as E

    root = tmp_path / "physical_graph_edge_handoff_qualification_v3"
    material = tmp_path / "physical_graph_edge_handoff_qualification_v3_material"
    final_receipt = tmp_path / (
        "physical_graph_edge_handoff_qualification_v3_regeneration_receipt.json"
    )
    monkeypatch.setattr(R, "OUTPUT_ROOT", root)
    monkeypatch.setattr(R, "MATERIAL_ROOT", material)
    monkeypatch.setattr(R, "EXTERNAL_REGENERATION_RECEIPT", final_receipt)
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

    physical_authority = C.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    physical_core = {
        "stage_id": physical_authority["stage_id"],
        "python_executable": physical_authority["real_python_executable"],
        "python_version": physical_authority["python_version"],
        "torch_version": physical_authority["torch_version"],
        "torch_hip_version": physical_authority["torch_hip_version"],
        "genesis_version": physical_authority["genesis_version"],
        "quadrants_version": physical_authority["quadrants_version"],
        "visible_device_count": physical_authority["visible_device_count"],
        "device": physical_authority["device"],
        "backend": physical_authority["backend"],
        "deterministic_environment": dict(
            C.DIRECT_RUNTIME_POLICY[
                "required_environment_before_simulator_creation"
            ]
        ),
        "fake_runtime": False,
    }
    physical_digest = M.runtime_environment_sha256(physical_core)
    M.validate_physical_runtime_environment(
        {
            **physical_core,
            "runtime_core_sha256": physical_digest,
            "qualification_runtime_sha256s": [physical_digest] * 256,
            "selected_snapshot_runtime_sha256s": [physical_digest] * 64,
        }
    )
    original_require_runtime = V1.require_stage_runtime

    def test_runtime(
        kind: str, *, fake: bool = False, visual_role: str | None = None
    ) -> dict[str, object]:
        if kind == "physical":
            return copy.deepcopy(physical_core)
        return original_require_runtime(kind, fake=fake, visual_role=visual_role)

    monkeypatch.setattr(V1, "require_stage_runtime", test_runtime)
    pool_index_by_candidate = {
        str(spec["candidate_spec_id"]): index
        for index, spec in enumerate(C.build_prospective_pool_specs())
    }
    historical = E.validate_existing_historical_custody_receipt(
        R.HISTORICAL_CUSTODY_RECEIPT
    )

    def historical_payload(pool_index: int) -> bytes:
        path = (
            R.V1_MATERIAL_ROOT
            / "qualification"
            / f"pool-{pool_index:03d}"
            / "payload.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            return np.ascontiguousarray(
                archive["snapshot_payload_bytes"]
            ).tobytes(order="C")

    class PhysicalBackend(_FakePhysicalBackend):
        runtime = {
            **_FakePhysicalBackend.runtime,
            "backend": physical_authority["backend"],
            "policy_device": physical_authority["device"],
        }

        def __init__(self) -> None:
            self.last_v3_augmentation: dict[str, object] | None = None

        def qualify(self, spec: dict) -> dict:
            value = super().qualify(spec)
            pool_index = pool_index_by_candidate[str(spec["candidate_spec_id"])]
            payload = (
                historical_payload(pool_index)
                if pool_index < 8
                else _fake_production_shaped_snapshot_payload(
                    step_index=pool_index
                )
            )
            value["snapshot"]["payload_bytes"] = payload
            payload_sha = hashlib.sha256(payload).hexdigest()
            value["initial_decision_state_sha256"] = payload_sha
            value["runtime_evidence"]["teacher_snapshot_sha256"] = payload_sha
            semantic = R._fresh_snapshot_semantics(payload)
            trace = _trace(
                final_digest=semantic["snapshot_semantic_digest_v1"]
            )
            trials = [
                {
                    "trial_index": trial_index,
                    "trace": copy.deepcopy(trace),
                    "snapshot_behavioural_digest_v1": (
                        M.snapshot_behavioural_digest(trace)
                    ),
                    "trace_member_manifests": R._trace_member_manifests(trace),
                    "final_snapshot_semantic_evidence": semantic[
                        "semantic_evidence"
                    ],
                }
                for trial_index in (0, 1)
            ]
            v3_identity = {
                "artifact_file_sha256": semantic["artifact_file_sha256"],
                "snapshot_semantic_digest_v1": semantic[
                    "snapshot_semantic_digest_v1"
                ],
                "snapshot_behavioural_digest_v1": trials[0][
                    "snapshot_behavioural_digest_v1"
                ],
            }
            augmentation: dict[str, object] = {
                "snapshot_semantic_bytes": semantic["semantic_payload_bytes"],
                "snapshot_semantic_evidence": semantic["semantic_evidence"],
                "snapshot_identity": v3_identity,
                "behavioural_probe_trials": {"V3": trials},
                "semantic_evidence_by_version": {
                    "V3": semantic["semantic_evidence"]
                },
            }
            if pool_index < 8:
                pair = historical["first_eight_pairs"][pool_index]
                historical_identities: dict[str, dict[str, str]] = {}
                for version, key in (("V1", "v1"), ("V2", "v2")):
                    item = pair[key]
                    historical_identities[version] = {
                        "artifact_file_sha256": item["artifact_file_sha256"],
                        "snapshot_semantic_digest_v1": item[
                            "snapshot_semantic_digest_v1"
                        ],
                        "snapshot_behavioural_digest_v1": (
                            M.snapshot_behavioural_digest(trace)
                        ),
                    }
                    augmentation["behavioural_probe_trials"][version] = [
                        copy.deepcopy(trial) for trial in trials
                    ]
                    augmentation["semantic_evidence_by_version"][version] = (
                        item["semantic_evidence"]
                    )
                augmentation["snapshot_identities"] = historical_identities
            self.last_v3_augmentation = augmentation
            return value

        def reset_fixture(self, spec: dict, payload: bytes) -> dict:
            value = super().reset_fixture(spec, payload)
            pool_index = pool_index_by_candidate[str(spec["candidate_spec_id"])]
            _metadata, qualification_arrays = R._load_material_shard(
                material / "qualification" / f"pool-{pool_index:03d}"
            )
            rgb_sha256 = V1.canonical_array_sha256(
                qualification_arrays["rgb"]
            )
            for trial in value["reset_trials"]:
                trial["metadata"]["current_rgb_sha256"] = rgb_sha256
            return value

    backend = PhysicalBackend()
    pool = R.initialize_stage(fake_runtime=True, evaluator_module=E)
    assert len(pool["specs"]) == 256
    for index in range(8):
        R.qualify_pool_state_stage(index, backend=backend, fake_runtime=True)
    reproduction = R.compare_v1_v2_v3_first_eight_stage(evaluator_module=E)
    assert reproduction["pass"] is True
    for index in range(8, 256):
        R.qualify_pool_state_stage(index, backend=backend, fake_runtime=True)
    equivalence = R.assemble_snapshot_equivalence_stage(evaluator_module=E)
    assert equivalence["pass"] is True and len(equivalence["records"]) == 256

    selection = R.select_teacher_pool_stage(fake_runtime=True)
    selected = [str(row["state_id"]) for row in selection["selected_specs"]]
    assert len(selected) == 64
    for state_id in selected:
        R.capture_selected_state_stage(
            state_id, backend=backend, fake_runtime=True
        )
    panel = R.freeze_panel_stage(fake_runtime=True)
    assert len(panel["states"]) == 64
    latent = R.encode_canonical_pixels_stage(
        encoder=_FakeEncoder(), fake_runtime=True
    )
    assert latent["records"]
    development = [
        row for row in panel["states"] if row["role"] == "DEVELOPMENT"
    ]
    heldout = [
        row
        for row in panel["states"]
        if row["role"] == "DEVELOPMENT_HELDOUT"
    ]
    assert (len(development), len(heldout)) == (48, 16)
    for state in development:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    target = R.development_target_selection_stage(
        ranker=_FakeRanker(), fake_runtime=True
    )
    assert target["selection_frozen"] is True
    for state in heldout:
        R.fanout_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    assert len(
        R.heldout_ranker_scores_stage(
            ranker=_FakeRanker(), fake_runtime=True
        )
    ) == 64
    for state in heldout:
        R.repeat_state_stage(
            state["state_id"], backend=backend, fake_runtime=True
        )
    assembled = R.assemble_row_evidence_stage(fake_runtime=True)
    assert assembled["candidate_trace_count"] == 960
    del (
        pool,
        selection,
        selected,
        panel,
        latent,
        development,
        heldout,
        target,
        backend,
    )
    gc.collect()

    source_observation = {
        "head_commit": source_commit,
        "parent_commit": C.SOURCE_PARENT_COMMIT,
        "freeze_subject": C.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "worktree_clean": True,
        "tracked_source_count": len(C.TRACKED_SOURCE_PATHS),
        "tracked_sources_sha256": "1" * 64,
        "source_closure_path": str(C.TRACKED_SOURCE_PATHS[4]),
        "source_closure_bytes": 1,
        "source_closure_sha256": "2" * 64,
        "source_closure_row_count": len(C.SOURCE_CLOSURE_PATHS),
        "source_closure_live_bytes_exact": True,
        "metrics_module": M.__name__,
    }

    # Fake visual evidence must be rejected before the test-only rewrite.
    R.recompute_and_persist_metrics_stage(
        fake_runtime=True, evaluator_module=E
    )
    rejected = tmp_path / "fake-runtime-rejected.json"
    with pytest.raises(E.RegenerationError, match="marked fake"):
        E.verify_and_emit(
            root,
            rejected,
            metrics_module=M,
            source_freeze_observation=source_observation,
            material_root=material,
            historical_custody_receipt=R.HISTORICAL_CUSTODY_RECEIPT,
        )
    assert not rejected.exists()
    (root / "metrics.json").unlink()

    def visual_environment(role: str) -> dict:
        authority = C.RUNTIME_ENVIRONMENT_AUTHORITY[role]
        return M.validate_visual_runtime_environment(
            {
                field: (
                    False
                    if field == "fake_runtime"
                    else authority["real_python_executable"]
                    if field == "python_executable"
                    else authority[field]
                )
                for field in C.VISUAL_RUNTIME_ENVIRONMENT_FIELDS
            },
            runtime_role=role,
        )

    encoder_environment = visual_environment("encoder")
    ranker_environment = visual_environment("ranker")
    ranker_digest = M.runtime_environment_sha256(ranker_environment)

    def replace_document(path: Path, field: str, value: dict) -> None:
        document = R._ordinary_json(path)
        document.pop("content_digest")
        document[field] = value
        path.unlink()
        R._atomic_json(path, C.attach_content_digest(document))

    replace_document(
        root / "latent_index.json",
        "encoder_runtime_environment",
        encoder_environment,
    )
    replace_document(
        root / "development_target_selection.json",
        "ranker_runtime_environment",
        ranker_environment,
    )
    score_path = root / "heldout_ranker_scores.jsonl"
    score_rows = V1.load_jsonl(score_path)
    for row in score_rows:
        row["ranker_runtime_environment_sha256"] = ranker_digest
    score_path.unlink()
    R._v3_atomic_jsonl(score_path, score_rows)

    class EvaluatorAdapter:
        validate_existing_historical_custody_receipt = staticmethod(
            E.validate_existing_historical_custody_receipt
        )

        @staticmethod
        def verify_and_emit(output_root, output, **_kwargs):
            return E.verify_and_emit(
                output_root,
                output,
                metrics_module=M,
                source_freeze_observation=source_observation,
                material_root=material,
                historical_custody_receipt=R.HISTORICAL_CUSTODY_RECEIPT,
            )

        @staticmethod
        def validate_existing_regeneration_receipt(
            output_root, output, **_kwargs
        ):
            return E.validate_existing_regeneration_receipt(
                output_root,
                output,
                metrics_module=M,
                source_freeze_observation=source_observation,
                material_root=material,
                historical_custody_receipt=R.HISTORICAL_CUSTODY_RECEIPT,
            )

    result = R.report_stage(
        fake_runtime=True, evaluator_module=EvaluatorAdapter
    )
    assert result["models_trained"] == 0
    assert result["v3_snapshot_qualification"]["all_pass"] is True
    assert final_receipt.is_file()
    assert set(path.name for path in root.iterdir()) == set(
        C.SUCCESS_OUTPUT_LEAVES
    )
    assert sum(1 for _ in material.rglob("payload.npz")) == 400
    E.validate_existing_regeneration_receipt(
        root,
        final_receipt,
        metrics_module=M,
        source_freeze_observation=source_observation,
        material_root=material,
        historical_custody_receipt=R.HISTORICAL_CUSTODY_RECEIPT,
    )
    production_source = Path(R.__file__).read_text()
    assert "production_shaped_runtime" not in production_source
    assert "fake_to_real" not in production_source
