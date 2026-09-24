from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math

import pytest

import lewm.planning.promoted_executor_reset_evidence_adapter_v1 as adapter_module
import lewm.planning.revisioned_physical_configuration_memory as memory_module
from lewm.planning.promoted_executor_reset_evidence_adapter_v1 import (
    CANONICAL_COMMAND_CADENCE_NS,
    CANONICAL_COMMAND_COUNT,
    CANONICAL_MAX_ANGULAR_SUBSTEP_RAD,
    CANONICAL_MAX_OUTCOME_DURATION_NS,
    CANONICAL_MAX_POSE_SEQUENCE_LENGTH,
    CANONICAL_MAX_TRANSLATION_SUBSTEP_M,
    CANONICAL_POSE_SAMPLE_CADENCE_NS,
    CanonicalExecutorResetContract,
    CanonicalRunnerPoseSample,
    PromotedExecutionAdmissionUnavailableError,
    PromotedExecutorResetEvidenceAdapterV1,
    canonical_executor_reset_contract,
    validate_canonical_runner_pose_sequence,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    EvidenceAuthority,
    ExecutionBlock,
    ExecutionBlockKind,
    ExecutionEvidenceAdmission,
    ExecutionEvidenceKind,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PhysicalMemoryConfig,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
    TransactionRejectedError,
    VerifiedTraversalPolygon,
)
from lewm.planning.zero_inflation_exact_physical_adapter_v1 import (
    ZeroInflationExactPhysicalAdapterV1,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _memory(
    *,
    session_id: str = "executor-reset-blocked",
    promoted: bool = True,
) -> RevisionedPhysicalMemory:
    return RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=MapFrameIdentity(
                session_id=session_id,
                origin_xy_m=(0.0, 0.0),
            ),
            promoted_runtime=promoted,
            expected_camera_transform_sha256=_hash("camera-transform"),
        )
    )


def _pose(
    memory: RevisionedPhysicalMemory,
    *,
    xy: tuple[float, float] = (0.05, 0.05),
    yaw: float = 0.0,
    timestamp_ns: int = 1,
    source: PoseSource = PoseSource.DEPLOYMENT_ODOMETRY,
) -> PoseProvenance:
    return PoseProvenance(
        source=source,
        frame_id=memory.map_frame.frame_id,
        mean_xy_yaw=(*xy, yaw),
        covariance_xy_yaw=(
            (0.001, 0.0, 0.0),
            (0.0, 0.001, 0.0),
            (0.0, 0.0, 0.0001),
        ),
        timestamp_ns=timestamp_ns,
        synchronization_id=f"sync-{timestamp_ns}",
        camera_transform_sha256=_hash("camera-transform"),
    )


def _forged_execution_transaction(
    memory: RevisionedPhysicalMemory,
    *,
    observation_id: str,
    authority: EvidenceAuthority,
    kind: ExecutionEvidenceKind,
    traversals: tuple[VerifiedTraversalPolygon, ...] = (),
    blocks: tuple[ExecutionBlock, ...] = (),
    pose: PoseProvenance | None = None,
) -> PhysicalEvidenceTransaction:
    receipt_hash = _hash(f"receipt:{observation_id}")
    observation = ObservationIdentity(
        observation_id=observation_id,
        payload_sha256=receipt_hash,
        producer_sha256=_hash("forged-runner"),
        authority=authority,
    )
    selected_pose = pose or _pose(memory)
    evidence_hash = memory_module._execution_evidence_content_sha256(
        traversals,
        blocks,
    )
    admission = ExecutionEvidenceAdmission(
        admission_id_sha256=_hash(f"admission:{observation_id}"),
        adapter_instance_sha256=_hash("forged-adapter"),
        source_memory_instance_sha256=_hash("forged-memory"),
        receipt_content_sha256=receipt_hash,
        adapter_contract_sha256=_hash("forged-contract"),
        body_support_contract_sha256=_hash("forged-body"),
        map_frame_sha256=memory.map_frame.content_sha256,
        observation_sha256=_canonical_hash(observation.to_dict()),
        pose_sha256=selected_pose.content_sha256,
        evidence_content_sha256=evidence_hash,
        memory_revision_before=memory.revision,
        evidence_kind=kind,
    )
    return PhysicalEvidenceTransaction(
        observation=observation,
        map_frame=memory.map_frame,
        pose=selected_pose,
        verified_traversals=traversals,
        execution_blocks=blocks,
        execution_admission=admission,
    )


def _traversal(
    observation_id: str,
    vertices: tuple[tuple[float, float], ...],
) -> VerifiedTraversalPolygon:
    return VerifiedTraversalPolygon(
        traversal_id=observation_id,
        vertices_xy_m=vertices,
        outcome_sha256=_hash(f"receipt:{observation_id}"),
    )


def test_canonical_contract_reopens_exact_sources_and_keeps_promotion_false() -> None:
    contract = canonical_executor_reset_contract()
    contract.assert_integrity()

    assert contract.geometry_contract_sha256 == (
        "e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca"
    )
    assert contract.directional_policy_sha256 == (
        "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc"
    )
    assert contract.body_forward_m == pytest.approx(0.3700000000000001)
    assert contract.body_rear_m == pytest.approx(0.43210313102250314)
    assert contract.body_half_width_m == pytest.approx(0.2668059073252429)
    assert contract.reset_footprint_radius_m == pytest.approx(0.47)
    assert contract.maximum_translation_substep_m == pytest.approx(0.025)
    assert contract.maximum_angular_substep_rad == pytest.approx(0.025)
    assert contract.pose_sample_cadence_ns == 50_000_000
    assert contract.command_cadence_ns == 100_000_000
    assert contract.command_count == 5
    assert contract.maximum_pose_sequence_length == 11
    assert contract.maximum_outcome_duration_ns == 500_000_000
    assert contract.runner_producer_sha256 is None
    assert contract.reset_producer_sha256 is None
    assert contract.runner_outcome_protocol_sha256 is None
    assert contract.reset_clearance_protocol_sha256 is None
    assert contract.physical_promotion_ready is False


def test_custom_five_metre_body_and_four_metre_step_contracts_are_rejected() -> None:
    with pytest.raises(ValueError, match="body_forward_m"):
        CanonicalExecutorResetContract(body_forward_m=5.0)
    with pytest.raises(ValueError, match="maximum_translation_substep_m"):
        CanonicalExecutorResetContract(maximum_translation_substep_m=4.0)
    with pytest.raises(ValueError, match="physical_promotion_ready"):
        CanonicalExecutorResetContract(physical_promotion_ready=True)


def test_no_public_bind_or_raw_pose_reset_issuance_api_exists() -> None:
    memory = _memory()
    assert not hasattr(PromotedExecutorResetEvidenceAdapterV1, "bind")
    for method in (
        "issue_reset_clearance",
        "issue_traversal_success",
        "issue_execution_block",
        "build_transaction",
        "fuse_receipt",
    ):
        assert not hasattr(PromotedExecutorResetEvidenceAdapterV1, method)
    with pytest.raises(
        PromotedExecutionAdmissionUnavailableError,
        match="physical_promotion_ready is false",
    ):
        PromotedExecutorResetEvidenceAdapterV1(memory)
    assert memory.revision == 0
    assert not memory.known_physical_cells


def test_fake_501_pose_ten_metre_path_is_rejected_by_frozen_shape() -> None:
    samples = tuple(
        CanonicalRunnerPoseSample(
            center_xy_m=(0.02 * index, 0.0),
            yaw_rad=0.0,
            timestamp_ns=CANONICAL_POSE_SAMPLE_CADENCE_NS * index,
        )
        for index in range(501)
    )
    assert samples[-1].center_xy_m[0] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="sequence exceeds"):
        validate_canonical_runner_pose_sequence(samples)


def test_pose_sequence_rejects_four_metre_gap_huge_time_gap_and_instant_pi_yaw() -> None:
    start = CanonicalRunnerPoseSample((0.0, 0.0), 0.0, 0)
    four_metres = CanonicalRunnerPoseSample(
        (4.0, 0.0),
        0.0,
        CANONICAL_POSE_SAMPLE_CADENCE_NS,
    )
    with pytest.raises(ValueError, match="translation substep"):
        validate_canonical_runner_pose_sequence((start, four_metres))

    late = CanonicalRunnerPoseSample(
        (0.01, 0.0),
        0.0,
        CANONICAL_POSE_SAMPLE_CADENCE_NS + 1,
    )
    with pytest.raises(ValueError, match="cadence"):
        validate_canonical_runner_pose_sequence((start, late))

    huge_gap = CanonicalRunnerPoseSample(
        (0.01, 0.0),
        0.0,
        CANONICAL_MAX_OUTCOME_DURATION_NS * 10,
    )
    with pytest.raises(ValueError, match="duration"):
        validate_canonical_runner_pose_sequence((start, huge_gap))

    pi_yaw = CanonicalRunnerPoseSample(
        (0.0, 0.0),
        math.pi,
        CANONICAL_POSE_SAMPLE_CADENCE_NS,
    )
    with pytest.raises(ValueError, match="angular substep"):
        validate_canonical_runner_pose_sequence((start, pi_yaw))


def test_frozen_pose_sequence_accepts_only_bounded_cadenced_shape() -> None:
    samples = tuple(
        CanonicalRunnerPoseSample(
            center_xy_m=(0.01 * index, 0.0),
            yaw_rad=0.01 * index,
            timestamp_ns=CANONICAL_POSE_SAMPLE_CADENCE_NS * index,
        )
        for index in range(CANONICAL_MAX_POSE_SEQUENCE_LENGTH)
    )
    assert validate_canonical_runner_pose_sequence(samples) == samples
    assert samples[-1].timestamp_ns == CANONICAL_MAX_OUTCOME_DURATION_NS
    assert CANONICAL_MAX_TRANSLATION_SUBSTEP_M == pytest.approx(0.025)
    assert CANONICAL_MAX_ANGULAR_SUBSTEP_RAD == pytest.approx(0.025)
    assert CANONICAL_COMMAND_CADENCE_NS == 100_000_000
    assert CANONICAL_COMMAND_COUNT == 5


def test_reset_at_cell_1000_cannot_enter_memory_without_runner_authority() -> None:
    memory = _memory()
    cell = (1000, 1000)
    center = memory.map_frame.cell_center(cell)
    half = memory.map_frame.cell_size_m / 2.0
    traversal = _traversal(
        "forged-reset-1000",
        (
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] - half),
            (center[0] + half, center[1] + half),
            (center[0] - half, center[1] + half),
        ),
    )
    transaction = _forged_execution_transaction(
        memory,
        observation_id="forged-reset-1000",
        authority=EvidenceAuthority.RESET_CLEARANCE,
        kind=ExecutionEvidenceKind.RESET_CLEARANCE,
        traversals=(traversal,),
        pose=_pose(
            memory,
            xy=center,
            source=PoseSource.RESET_CERTIFICATE,
        ),
    )
    before = memory.serialize()
    with pytest.raises(TransactionRejectedError, match="structurally unavailable"):
        memory.apply_transaction(transaction)
    assert memory.serialize() == before
    assert memory.physical_state(cell) is PhysicalLabel.UNKNOWN


def test_forged_million_cell_square_fails_even_with_self_consistent_admission() -> None:
    memory = _memory()
    traversal = _traversal(
        "forged-million",
        ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
    )
    transaction = _forged_execution_transaction(
        memory,
        observation_id="forged-million",
        authority=EvidenceAuthority.EXECUTOR_OUTCOME,
        kind=ExecutionEvidenceKind.TRAVERSAL_SUCCESS,
        traversals=(traversal,),
    )
    before = memory.serialize()
    with pytest.raises(TransactionRejectedError, match="structurally unavailable"):
        memory.apply_transaction(transaction)
    assert memory.serialize() == before
    assert memory.revision == 0
    assert memory.physical_state((500, 500)) is PhysicalLabel.UNKNOWN


def test_imported_globals_cannot_reopen_withdrawn_admission_hooks() -> None:
    for module in (adapter_module, memory_module):
        names = set(vars(module))
        assert not any("CAPABILITY" in name for name in names)
    assert not hasattr(RevisionedPhysicalMemory, "_bind_promoted_execution_adapter")
    assert not hasattr(RevisionedPhysicalMemory, "_build_promoted_execution_transaction")
    assert not hasattr(RevisionedPhysicalMemory, "_assert_live_execution_admission")
    assert not hasattr(RevisionedPhysicalMemory, "_assert_historical_execution_admission")


def test_issuance_table_injection_and_object_new_adapter_clone_have_no_surface() -> None:
    forged_adapter = object.__new__(PromotedExecutorResetEvidenceAdapterV1)
    assert not hasattr(forged_adapter, "__dict__")
    with pytest.raises(AttributeError):
        setattr(forged_adapter, "_issued_receipts", {1: object()})
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(forged_adapter)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(forged_adapter)
    for method in ("build_transaction", "fuse_receipt", "issue_reset_clearance"):
        assert not hasattr(forged_adapter, method)


def test_memory_is_noncopyable_slotted_and_object_new_clone_cannot_receive_state() -> None:
    memory = _memory()
    assert not hasattr(memory, "__dict__")
    with pytest.raises(TypeError, match="non-copyable"):
        copy.copy(memory)
    with pytest.raises(TypeError, match="non-copyable"):
        copy.deepcopy(memory)

    forged = object.__new__(RevisionedPhysicalMemory)
    assert not hasattr(forged, "__dict__")
    with pytest.raises(AttributeError):
        setattr(forged, "_issued_execution_admissions", {})
    with pytest.raises((AttributeError, TypeError)):
        forged.apply_transaction(object())
    assert memory.revision == 0


def test_forged_transaction_transfer_is_rejected_by_every_promoted_memory() -> None:
    first = _memory(session_id="transfer-frame")
    second = _memory(session_id="transfer-frame")
    traversal = _traversal(
        "forged-transfer",
        ((0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)),
    )
    transaction = _forged_execution_transaction(
        first,
        observation_id="forged-transfer",
        authority=EvidenceAuthority.EXECUTOR_OUTCOME,
        kind=ExecutionEvidenceKind.TRAVERSAL_SUCCESS,
        traversals=(traversal,),
    )
    for memory in (first, second):
        before = memory.serialize()
        with pytest.raises(TransactionRejectedError, match="structurally unavailable"):
            memory.apply_transaction(transaction)
        assert memory.serialize() == before


def test_serialized_replay_has_no_execution_authority_bypass() -> None:
    development = _memory(session_id="replay", promoted=False)
    traversal = _traversal(
        "historical-forgery",
        ((0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)),
    )
    transaction = _forged_execution_transaction(
        development,
        observation_id="historical-forgery",
        authority=EvidenceAuthority.EXECUTOR_OUTCOME,
        kind=ExecutionEvidenceKind.TRAVERSAL_SUCCESS,
        traversals=(traversal,),
    )
    development.apply_transaction(transaction)
    payload = development.to_dict()
    config = payload["config"]
    assert isinstance(config, dict)
    config["promoted_runtime"] = True
    payload["config_sha256"] = _canonical_hash(config)
    state_core = dict(payload)
    state_core.pop("physical_content_sha256")
    payload["physical_content_sha256"] = _canonical_hash(state_core)

    with pytest.raises(TransactionRejectedError, match="structurally unavailable"):
        RevisionedPhysicalMemory.from_mapping(payload)


def test_empty_promoted_and_exact_development_serialization_still_round_trip() -> None:
    promoted = _memory()
    assert RevisionedPhysicalMemory.deserialize(promoted.serialize()).serialize() == (
        promoted.serialize()
    )

    development = _memory(session_id="exact-development", promoted=False)
    adapter = ZeroInflationExactPhysicalAdapterV1(development)
    labels = {(0, 0): PhysicalLabel.FREE}
    observation = ObservationIdentity(
        observation_id="exact-development",
        payload_sha256=memory_module._exact_physical_cells_sha256(
            (PhysicalCellEvidence((0, 0), PhysicalLabel.FREE),),
            (),
        ),
        producer_sha256=_hash("exact-producer"),
        authority=EvidenceAuthority.EXACT_PHYSICAL,
    )
    adapter.fuse_cells(
        labels,
        observation=observation,
        pose=_pose(development),
        label_inflation_radius_m=0.0,
    )
    restored = RevisionedPhysicalMemory.deserialize(development.serialize())
    assert restored.serialize() == development.serialize()
    assert restored.exact_sim_tainted is True


def test_learned_and_exact_promoted_paths_remain_locked_and_atomic() -> None:
    memory = _memory()
    before = memory.serialize()
    learned = PhysicalEvidenceTransaction(
        observation=ObservationIdentity(
            observation_id="learned-forgery",
            payload_sha256=_hash("learned-payload"),
            producer_sha256=_hash("learned-producer"),
            authority=EvidenceAuthority.LEARNED_PHYSICAL,
        ),
        map_frame=memory.map_frame,
        pose=_pose(memory),
        physical_evidence=(PhysicalCellEvidence((0, 0), PhysicalLabel.FREE),),
    )
    with pytest.raises(TransactionRejectedError, match="qualified projection adapter"):
        memory.apply_transaction(learned)
    with pytest.raises(PermissionError, match="development-only"):
        ZeroInflationExactPhysicalAdapterV1(memory)
    assert memory.serialize() == before


def test_contract_pose_and_memory_objects_reject_copy_and_mutation() -> None:
    contract = canonical_executor_reset_contract()
    sample = CanonicalRunnerPoseSample((0.0, 0.0), 0.0, 0)
    for value in (contract, sample):
        with pytest.raises(TypeError, match="non-copyable"):
            copy.copy(value)
        with pytest.raises(TypeError, match="non-copyable"):
            copy.deepcopy(value)
        assert not hasattr(value, "__dict__")

    object.__setattr__(contract, "body_forward_m", 5.0)
    with pytest.raises(ValueError, match="mutated"):
        contract.assert_integrity()
