from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
import struct
import time

import pytest

from lewm.planning.reversible_target_belief import (
    SyntheticG3TargetMemoryContextProducer,
    NegativeTargetObservation,
    PositiveTargetObservation,
    ReversibleTargetBeliefMemory,
    TargetCellValue,
    TargetClaimVerificationError,
    TargetEpisodeAuthority,
    TargetEvidenceIdentity,
    TargetEvidenceRejectedError,
    TargetMemoryConfig,
    TargetMemoryContext,
    SyntheticTargetMemoryContextIssuer,
    TargetSnapshotBindingError,
)


Cell = tuple[int, int]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hash_object(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _authority(
    *,
    context_issuer_contract_sha256: str,
    scene_id: str = "g5-unit",
    episode_id: str = "episode",
    manifest_sha256: str | None = None,
    mapping: tuple[tuple[str, str], ...] = (("red", "beacon_red"),),
) -> TargetEpisodeAuthority:
    manifest_hash = manifest_sha256 or _hash("manifest")
    object_ids = tuple(sorted(object_id for _, object_id in mapping))
    task_hash = _hash_object(
        {
            "schema": "lewm_go2_claim_task_set_v1",
            "scene_id": scene_id,
            "physical_manifest_sha256": manifest_hash,
            "task_object_ids": list(object_ids),
        }
    )
    return TargetEpisodeAuthority(
        scene_id=scene_id,
        episode_id=episode_id,
        physical_manifest_sha256=manifest_hash,
        context_issuer_contract_sha256=context_issuer_contract_sha256,
        expected_task_object_ids=object_ids,
        task_object_by_target_id=mapping,
        task_object_set_sha256=task_hash,
    )


def _advance_g3(producer: SyntheticG3TargetMemoryContextProducer) -> None:
    from lewm.planning.revisioned_physical_configuration_memory import (
        EvidenceAuthority,
        ObservationIdentity,
        PhysicalEvidenceTransaction,
        PoseProvenance,
        PoseSource,
    )

    memory = producer.context_issuer._physical_memory
    next_revision = memory.revision + 1
    observation = ObservationIdentity(
        observation_id=f"g3-step-{next_revision}",
        payload_sha256=_hash(f"g3-payload-{next_revision}"),
        producer_sha256=_hash("g3-learned-producer"),
        authority=EvidenceAuthority.LEARNED_PHYSICAL,
    )
    pose = PoseProvenance(
        source=PoseSource.DEPLOYMENT_ODOMETRY,
        frame_id=memory.map_frame.frame_id,
        mean_xy_yaw=(0.0, 0.0, 0.0),
        covariance_xy_yaw=(
            (0.01, 0.0, 0.0),
            (0.0, 0.01, 0.0),
            (0.0, 0.0, 0.001),
        ),
        timestamp_ns=next_revision,
        synchronization_id=f"g3-sync-{next_revision}",
        camera_transform_sha256=_hash("camera"),
    )
    memory.apply_transaction(
        PhysicalEvidenceTransaction(
            observation=observation,
            map_frame=memory.map_frame,
            pose=pose,
            observed_unknown_cells=((1000 + next_revision, 1000),),
        )
    )


def _issuer() -> SyntheticG3TargetMemoryContextProducer:
    from lewm.planning.revisioned_physical_configuration_memory import (
        MapFrameIdentity,
        PhysicalMemoryConfig,
        RevisionedPhysicalMemory,
    )

    physical_memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=MapFrameIdentity(
                session_id="g5-unit-reset",
                origin_xy_m=(0.0, 0.0),
            ),
            expected_camera_transform_sha256=_hash("camera"),
            promoted_runtime=False,
        )
    )
    producer = SyntheticTargetMemoryContextIssuer._bind_g3_for_tests(
        physical_memory=physical_memory,
        issuer_sha256=_hash("g3-issuer"),
        frustum_contract_sha256=_hash("frustum"),
        physical_los_contract_sha256=_hash("physical-los"),
        positive_evidence_producer_sha256=_hash("positive-head"),
        negative_visibility_producer_sha256=_hash("negative-producer"),
        camera_calibration_sha256=_hash("camera"),
        observation_model_checkpoint_sha256=_hash("model"),
    )
    _advance_g3(producer)
    return producer


def _issue_context(
    issuer: SyntheticG3TargetMemoryContextProducer,
    revision: int = 1,
    *,
    pose_timestamp_ns: int | None = None,
    pose_name: str | None = None,
    domain: frozenset[Cell] | None = None,
) -> TargetMemoryContext:
    while issuer.context_issuer._physical_memory.revision < revision:
        _advance_g3(issuer)
    if issuer.context_issuer._physical_memory.revision != revision:
        raise ValueError("test helper cannot issue a rolled-back G3 revision")
    return issuer._issue_context_for_tests(
        pose_provenance_sha256=_hash(pose_name or f"pose:{revision}"),
        pose_timestamp_ns=(
            revision * 100 if pose_timestamp_ns is None else pose_timestamp_ns
        ),
        candidate_domain=domain
        or frozenset((x, y) for x in range(-10, 11) for y in range(-10, 11)),
    )


def _memory(
    *,
    authority: TargetEpisodeAuthority | None = None,
    config: TargetMemoryConfig | None = None,
    domain: frozenset[Cell] | None = None,
) -> tuple[
    ReversibleTargetBeliefMemory,
    SyntheticG3TargetMemoryContextProducer,
    TargetMemoryContext,
    TargetEpisodeAuthority,
]:
    producer = _issuer()
    context = _issue_context(producer, domain=domain)
    episode_authority = authority or _authority(
        context_issuer_contract_sha256=producer.context_issuer.contract_sha256
    )
    memory = ReversibleTargetBeliefMemory(
        context,
        config,
        context_issuer=producer.context_issuer,
        episode_authority=episode_authority,
    )
    return memory, producer, context, episode_authority


def _identity(
    name: str,
    tick: int,
    *,
    payload: str | None = None,
    producer: str = "negative-producer",
) -> TargetEvidenceIdentity:
    return TargetEvidenceIdentity(
        observation_id=name,
        payload_sha256=_hash(payload or f"payload:{name}"),
        producer_sha256=_hash(producer),
        diversity_key=f"view-{name}",
        tick=tick,
    )


def _positive(
    memory: ReversibleTargetBeliefMemory,
    producer: SyntheticG3TargetMemoryContextProducer,
    name: str,
    tick: int,
    cells: dict[Cell, float],
    *,
    unlocalized: float = 0.0,
    confidence: float = 1.0,
    payload: str | None = None,
) -> PositiveTargetObservation:
    return producer._issue_positive_observation_for_tests(
        memory.context,
        identity=_identity(name, tick, payload=payload, producer="positive-head"),
        target_id="red",
        localized_distribution=cells,
        unlocalized_probability=unlocalized,
        confidence=confidence,
    )


def _negative(
    memory: ReversibleTargetBeliefMemory,
    issuer: SyntheticG3TargetMemoryContextProducer,
    issued_context: TargetMemoryContext,
    name: str,
    tick: int,
    visibility: dict[Cell, float],
    *,
    confidence: float = 1.0,
    payload: str | None = None,
) -> NegativeTargetObservation:
    identity = _identity(name, tick, payload=payload)
    certificate = issuer._issue_negative_visibility_certificate_for_tests(
        issued_context,
        identity=identity,
        target_id="red",
        visible_detection_probability=visibility,
        confidence=confidence,
    )
    return NegativeTargetObservation(
        identity=identity,
        context_sha256=memory.context.content_sha256,
        target_id="red",
        visible_detection_probability=tuple(
            TargetCellValue(cell=cell, value=value)
            for cell, value in visibility.items()
        ),
        confidence=confidence,
        visibility_certificate=certificate,
    )


def _mass(snapshot) -> dict[Cell, float]:
    return {row.cell: row.value for row in snapshot.cell_mass}


def test_initial_state_is_authority_bound_unlocalized_and_normalized() -> None:
    memory, _, _, authority = _memory()
    snapshot = memory.snapshot("red")
    assert snapshot.cell_mass == ()
    assert snapshot.unlocalized_mass == 1.0
    assert snapshot.episode_authority_sha256 == authority.content_sha256
    assert snapshot.physical_manifest_sha256 == authority.physical_manifest_sha256
    assert snapshot.config_sha256 == memory.config.content_sha256
    memory.assert_current_snapshot(snapshot)
    first_checkpoint = memory.serialize()
    memory.snapshot("red")
    assert memory.serialize() == first_checkpoint


def test_construction_rejects_forged_context_and_is_atomic() -> None:
    producer = _issuer()
    issuer = producer.context_issuer
    authority = _authority(
        context_issuer_contract_sha256=issuer.contract_sha256
    )
    issued = _issue_context(producer)
    forged = replace(issued, _issuance_capability=object())
    with pytest.raises(TargetSnapshotBindingError, match="issuer instance"):
        ReversibleTargetBeliefMemory(
            forged,
            context_issuer=issuer,
            episode_authority=authority,
        )

    with pytest.raises(ValueError, match="unregistered target"):
        ReversibleTargetBeliefMemory(
            issued,
            TargetMemoryConfig(target_ids=("blue",)),
            context_issuer=issuer,
            episode_authority=authority,
        )
    # The failed constructor did not consume the real capability.
    memory = ReversibleTargetBeliefMemory(
        issued,
        context_issuer=issuer,
        episode_authority=authority,
    )
    assert memory.context.content_sha256 == issued.content_sha256


def test_object_shell_issuer_clone_cannot_reuse_original_issuance_authority() -> None:
    producer = _issuer()
    issuer = producer.context_issuer
    context = _issue_context(producer)
    authority = _authority(
        context_issuer_contract_sha256=issuer.contract_sha256
    )
    clone = object.__new__(SyntheticTargetMemoryContextIssuer)
    clone.__dict__.update(issuer.__dict__)

    with pytest.raises(TargetSnapshotBindingError, match="issuer object identity"):
        ReversibleTargetBeliefMemory(
            context,
            context_issuer=clone,
            episode_authority=authority,
        )

    # Rejection happens before the context or writer lease is consumed.
    memory = ReversibleTargetBeliefMemory(
        context,
        context_issuer=issuer,
        episode_authority=authority,
    )
    assert memory.snapshot("red").unlocalized_mass == 1.0


def test_synthetic_issuer_and_episode_instance_binding_cannot_be_self_asserted() -> None:
    assert not hasattr(SyntheticTargetMemoryContextIssuer, "bind_g3")
    with pytest.raises(TypeError, match="synthetic tests"):
        SyntheticTargetMemoryContextIssuer(
            issuer_sha256=_hash("g3-issuer"),
            frustum_contract_sha256=_hash("frustum"),
            physical_los_contract_sha256=_hash("physical-los"),
            positive_evidence_producer_sha256=_hash("positive-head"),
            negative_visibility_producer_sha256=_hash("negative-producer"),
            camera_calibration_sha256=_hash("camera"),
            observation_model_checkpoint_sha256=_hash("model"),
            _physical_memory=object(),
            _synthetic_test_fixture=False,
        )
    first = _issuer()
    assert first.synthetic_only is True
    assert first.production_authority_eligible is False
    assert first.context_issuer.synthetic_only is True
    assert first.context_issuer.production_authority_eligible is False
    for synthetic_authority in (first, first.context_issuer):
        with pytest.raises(AttributeError):
            synthetic_authority.synthetic_only = False
        with pytest.raises(AttributeError):
            synthetic_authority.production_authority_eligible = True
        with pytest.raises(AttributeError):
            object.__setattr__(synthetic_authority, "synthetic_only", False)
        with pytest.raises(AttributeError):
            object.__setattr__(
                synthetic_authority,
                "production_authority_eligible",
                True,
            )
        vars(synthetic_authority)["synthetic_only"] = False
        vars(synthetic_authority)["production_authority_eligible"] = True
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(synthetic_authority)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(synthetic_authority)
        assert synthetic_authority.synthetic_only is True
        assert synthetic_authority.production_authority_eligible is False
    assert not hasattr(first, "issue_context")
    assert not hasattr(first, "issue_positive_observation")
    assert not hasattr(first, "issue_negative_visibility_certificate")
    authority = _authority(
        context_issuer_contract_sha256=first.context_issuer.contract_sha256
    )
    second = _issuer()
    assert (
        first.context_issuer.contract_sha256
        != second.context_issuer.contract_sha256
    )
    second_context = _issue_context(second)
    with pytest.raises(TargetSnapshotBindingError, match="different G3 issuer"):
        ReversibleTargetBeliefMemory(
            second_context,
            context_issuer=second.context_issuer,
            episode_authority=authority,
        )
    assert not hasattr(second.context_issuer, "issue_context")


def test_evidence_injection_copy_serialization_and_mutation_never_gain_issuance() -> None:
    from lewm.planning import reversible_target_belief as implementation

    for removed in (
        "_G3_ISSUER_BINDING_CAPABILITY",
        "_CANONICAL_EVALUATION_CAPABILITY",
        "_MEMORY_STATE_COMMITMENT_CAPABILITY",
        "_FINALIZED_RESTORE_CAPABILITY",
    ):
        assert removed not in vars(implementation)

    producer = _issuer()
    context = _issue_context(producer)
    authority = _authority(
        context_issuer_contract_sha256=producer.context_issuer.contract_sha256
    )
    for clone in (copy.copy(context), copy.deepcopy(context), replace(context)):
        with pytest.raises(TargetSnapshotBindingError, match="issuer instance"):
            ReversibleTargetBeliefMemory(
                clone,
                context_issuer=producer.context_issuer,
                episode_authority=authority,
            )

    memory = ReversibleTargetBeliefMemory(
        context,
        context_issuer=producer.context_issuer,
        episode_authority=authority,
    )
    observation = _positive(memory, producer, "copy-source", 1, {(0, 0): 1.0})
    serialized_clone = implementation._parse_positive(observation.to_dict())
    for clone in (
        copy.copy(observation),
        copy.deepcopy(observation),
        replace(observation),
        serialized_clone,
    ):
        with pytest.raises(TargetEvidenceRejectedError, match="not issued"):
            memory.apply_positive(clone)
    memory.apply_positive(observation)

    serialized = json.loads(memory.serialize())
    assert serialized["synthetic_only"] is True
    assert serialized["production_authority_eligible"] is False
    serialized["production_authority_eligible"] = True
    core = dict(serialized)
    core.pop("state_content_sha256")
    serialized["state_content_sha256"] = _hash_object(core)
    with pytest.raises(PermissionError, match="cannot become production eligible"):
        ReversibleTargetBeliefMemory.from_mapping(
            serialized,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )

    mutated = _positive(memory, producer, "mutated", 2, {(1, 0): 1.0})
    object.__setattr__(mutated, "confidence", 0.5)
    with pytest.raises(TargetEvidenceRejectedError, match="mutated"):
        memory.apply_positive(mutated)


def test_positive_transfers_only_unlocalized_mass_and_preserves_unrelated_modes() -> None:
    memory, producer, _, _ = _memory()
    first = memory.apply_positive(
        _positive(memory, producer, "p1", 1, {(0, 0): 0.5, (1, 0): 0.5})
    )
    before = _mass(first)
    second = memory.apply_positive(
        _positive(memory, producer, "p2", 2, {(8, 8): 0.5, (8, 9): 0.5})
    )
    after = _mass(second)
    assert after[(0, 0)] == before[(0, 0)]
    assert after[(1, 0)] == before[(1, 0)]
    assert after[(8, 8)] > 0.0 and after[(8, 9)] > 0.0
    hypotheses = memory.hypotheses(second)
    assert {row.cells for row in hypotheses} == {
        frozenset({(0, 0), (1, 0)}),
        frozenset({(8, 8), (8, 9)}),
    }


def test_positive_requires_registered_producer_issuance() -> None:
    memory, producer, _, _ = _memory()
    with pytest.raises(TargetEvidenceRejectedError, match="producer"):
        producer._issue_positive_observation_for_tests(
            memory.context,
            identity=_identity("wrong-producer", 1, producer="not-registered"),
            target_id="red",
            localized_distribution={(0, 0): 1.0},
            unlocalized_probability=0.0,
            confidence=1.0,
        )

    issued = _positive(memory, producer, "p1", 1, {(0, 0): 1.0})
    same_content_clone = replace(issued)
    with pytest.raises(TargetEvidenceRejectedError, match="not issued"):
        memory.apply_positive(same_content_clone)
    memory.apply_positive(issued)


def test_negative_requires_live_visibility_capability_and_is_atomic() -> None:
    memory, issuer, context, _ = _memory()
    memory.apply_positive(
        _positive(memory, issuer, "p1", 1, {(0, 0): 0.5, (1, 0): 0.5})
    )
    identity = _identity("n1", 2)
    certificate = issuer._issue_negative_visibility_certificate_for_tests(
        context,
        identity=identity,
        target_id="red",
        visible_detection_probability={(0, 0): 1.0, (1, 0): 1.0},
        confidence=1.0,
    )
    invalid = NegativeTargetObservation(
        identity=identity,
        context_sha256=memory.context.content_sha256,
        target_id="red",
        visible_detection_probability=(TargetCellValue((0, 0), 1.0),),
        confidence=1.0,
        visibility_certificate=certificate,
    )
    before = memory.content_sha256
    with pytest.raises(TargetEvidenceRejectedError, match="misbound"):
        memory.apply_negative(invalid)
    assert memory.content_sha256 == before

    valid = replace(
        invalid,
        visible_detection_probability=(
            TargetCellValue((0, 0), 1.0),
            TargetCellValue((1, 0), 1.0),
        ),
    )
    memory.apply_negative(valid)
    replayed_certificate = replace(valid, identity=_identity("n2", 3))
    with pytest.raises(TargetEvidenceRejectedError, match="not issued"):
        memory.apply_negative(replayed_certificate)


def test_same_content_forged_visibility_certificate_is_rejected() -> None:
    memory, issuer, context, _ = _memory()
    memory.apply_positive(_positive(memory, issuer, "p1", 1, {(0, 0): 1.0}))
    identity = _identity("n1", 2)
    real = issuer._issue_negative_visibility_certificate_for_tests(
        context,
        identity=identity,
        target_id="red",
        visible_detection_probability={(0, 0): 1.0},
        confidence=1.0,
    )
    clone = replace(real, _certificate_capability=object())
    forged = NegativeTargetObservation(
        identity=identity,
        context_sha256=memory.context.content_sha256,
        target_id="red",
        visible_detection_probability=(TargetCellValue((0, 0), 1.0),),
        confidence=1.0,
        visibility_certificate=clone,
    )
    with pytest.raises(TargetEvidenceRejectedError, match="not issued"):
        memory.apply_negative(forged)


def test_visibility_certificate_binds_probability_confidence_and_identity() -> None:
    memory, issuer, context, _ = _memory()
    memory.apply_positive(_positive(memory, issuer, "p1", 1, {(0, 0): 1.0}))
    identity = _identity("n1", 2)
    certificate = issuer._issue_negative_visibility_certificate_for_tests(
        context,
        identity=identity,
        target_id="red",
        visible_detection_probability={(0, 0): 0.25},
        confidence=0.5,
    )
    inflated = NegativeTargetObservation(
        identity=identity,
        context_sha256=memory.context.content_sha256,
        target_id="red",
        visible_detection_probability=(TargetCellValue((0, 0), 1.0),),
        confidence=1.0,
        visibility_certificate=certificate,
    )
    before = memory.content_sha256
    with pytest.raises(TargetEvidenceRejectedError, match="misbound"):
        memory.apply_negative(inflated)
    assert memory.content_sha256 == before
    exact = replace(
        inflated,
        visible_detection_probability=(TargetCellValue((0, 0), 0.25),),
        confidence=0.5,
    )
    memory.apply_negative(exact)


def test_payload_replay_with_changed_id_tick_and_metadata_is_rejected() -> None:
    memory, producer, _, _ = _memory()
    memory.apply_positive(
        _positive(
            memory,
            producer,
            "first",
            1,
            {(0, 0): 1.0},
            payload="same-payload",
        )
    )
    replay = _positive(
        memory,
        producer,
        "different-id",
        99,
        {(8, 8): 1.0},
        confidence=0.25,
        payload="same-payload",
    )
    before = memory.content_sha256
    with pytest.raises(TargetEvidenceRejectedError, match="payload"):
        memory.apply_positive(replay)
    assert memory.content_sha256 == before


def test_107_negative_updates_never_drop_or_underflow_a_mode() -> None:
    memory, issuer, context, _ = _memory()
    memory.apply_positive(
        _positive(memory, issuer, "p1", 1, {(0, 0): 0.5, (8, 8): 0.5})
    )
    snapshot = None
    for index in range(107):
        snapshot = memory.apply_negative(
            _negative(
                memory,
                issuer,
                context,
                f"n{index}",
                index + 2,
                {(0, 0): 1.0},
            )
        )
        if index == 1:
            indexed = {min(row.cells): row for row in memory.hypotheses(snapshot)}
            assert indexed[(0, 0)].positive_evidence_count == 1
            assert indexed[(0, 0)].negative_evidence_count == 2
    assert snapshot is not None
    mass = _mass(snapshot)
    assert (0, 0) in mass
    assert math.isfinite(mass[(0, 0)])
    assert mass[(0, 0)] >= memory.config.posterior_mass_floor
    hypotheses = {min(row.cells): row for row in memory.hypotheses(snapshot)}
    assert hypotheses[(8, 8)].positive_evidence_count == 1
    assert hypotheses[(8, 8)].negative_evidence_count == 0


def test_runtime_update_snapshot_and_hypotheses_do_not_replay_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory, producer, context, _ = _memory(domain=frozenset({(0, 0)}))
    first = memory.apply_positive(
        _positive(memory, producer, "p1", 1, {(0, 0): 1.0})
    )
    first_chain = first.evidence_chain_sha256

    class NoHistoryScanDict(dict):
        def values(self):
            raise AssertionError("runtime path scanned complete observation history")

    memory._positive = NoHistoryScanDict(memory._positive)
    memory._negative = NoHistoryScanDict(memory._negative)

    def fail_full_audit(*args, **kwargs):
        raise AssertionError("runtime path invoked exhaustive audit")

    def fail_g3_full_state(*args, **kwargs):
        raise AssertionError("runtime path materialized full G3 physical state")

    monkeypatch.setattr(memory, "to_dict", fail_full_audit)
    monkeypatch.setattr(memory, "_replay_posterior", fail_full_audit)
    monkeypatch.setattr(
        memory._context_issuer._physical_memory,
        "_physical_state_core",
        fail_g3_full_state,
    )
    snapshot = memory.apply_negative(
        _negative(memory, producer, context, "n1", 2, {(0, 0): 1.0})
    )
    assert snapshot.evidence_chain_sha256 != first_chain
    assert snapshot.positive_evidence_count == 1
    assert snapshot.negative_evidence_count == 1
    hypothesis = memory.hypotheses(snapshot)[0]
    assert hypothesis.positive_evidence_count == 1
    assert hypothesis.negative_evidence_count == 1


def test_hot_path_never_hashes_large_g3_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _issuer()
    context = _issue_context(
        producer,
        revision=300,
        domain=frozenset({(0, 0)}),
    )
    authority = _authority(
        context_issuer_contract_sha256=producer.context_issuer.contract_sha256
    )
    memory = ReversibleTargetBeliefMemory(
        context,
        context_issuer=producer.context_issuer,
        episode_authority=authority,
    )
    physical_memory = producer.context_issuer._physical_memory
    full_state_calls = 0
    original = physical_memory._physical_state_core

    def count_full_state_calls():
        nonlocal full_state_calls
        full_state_calls += 1
        return original()

    monkeypatch.setattr(
        physical_memory,
        "_physical_state_core",
        count_full_state_calls,
    )
    snapshot = None
    for tick in range(1, 101):
        snapshot = memory.apply_positive(
            _positive(
                memory,
                producer,
                f"deep-g3-{tick}",
                tick,
                {(0, 0): 1.0},
            )
        )
    assert snapshot is not None
    memory.snapshot("red")
    memory.hypotheses(snapshot)
    assert full_state_calls == 0


def test_context_rebuild_splits_cached_components_without_history_scan() -> None:
    domain = frozenset({(0, 0), (1, 0), (2, 0)})
    memory, producer, _, _ = _memory(domain=domain)
    memory.apply_positive(
        _positive(memory, producer, "ends", 1, {(0, 0): 0.5, (2, 0): 0.5})
    )
    joined = memory.apply_positive(
        _positive(memory, producer, "bridge", 2, {(1, 0): 1.0})
    )
    assert len(memory.hypotheses(joined)) == 1

    class NoHistoryScanDict(dict):
        def values(self):
            raise AssertionError("context rebuild scanned complete observation history")

    memory._positive = NoHistoryScanDict(memory._positive)
    memory._negative = NoHistoryScanDict(memory._negative)
    advanced = _issue_context(
        producer,
        revision=2,
        domain=frozenset({(0, 0), (2, 0)}),
    )
    memory.advance_context(advanced)
    split = memory.hypotheses(memory.snapshot("red"))
    assert {row.cells for row in split} == {
        frozenset({(0, 0)}),
        frozenset({(2, 0)}),
    }
    assert all(row.positive_evidence_count == 1 for row in split)
    assert all(row.evidence_diversity == 1 for row in split)


def test_online_update_work_stays_bounded_across_successive_batches() -> None:
    memory, producer, _, _ = _memory(domain=frozenset({(0, 0)}))
    batch_sizes = (100, 200, 700)
    elapsed_per_update: list[float] = []
    next_tick = 1
    snapshot = None

    for batch_size in batch_sizes:
        before = memory.performance_counters
        started = time.perf_counter()
        for _ in range(batch_size):
            snapshot = memory.apply_positive(
                _positive(
                    memory,
                    producer,
                    f"scale-{next_tick}",
                    next_tick,
                    {(0, 0): 1.0},
                )
            )
            next_tick += 1
        elapsed = time.perf_counter() - started
        after = memory.performance_counters
        delta = {key: after[key] - before[key] for key in before}

        assert delta["runtime_integrity_checks"] == 2 * batch_size
        assert delta["writer_lease_checks"] == 2 * batch_size
        assert delta["revision_checks"] == 2 * batch_size
        assert delta["runtime_revision_commits"] == batch_size
        assert delta["rolling_evidence_updates"] == batch_size
        assert delta["component_evidence_updates"] == batch_size
        assert delta["component_cell_additions"] <= 1
        assert delta["evidence_chain_replay_transactions"] == 0
        assert delta["posterior_replay_transactions"] == 0
        assert delta["exhaustive_integrity_audits"] == 0
        assert delta["full_state_materializations"] == 0
        assert delta["canonical_full_state_hashes"] == 0
        elapsed_per_update.append(elapsed / batch_size)

    assert snapshot is not None
    before_hypotheses = memory.performance_counters
    for _ in range(10):
        hypothesis = memory.hypotheses(snapshot)[0]
        assert hypothesis.positive_evidence_count == sum(batch_sizes)
    after_hypotheses = memory.performance_counters
    for counter in (
        "evidence_chain_replay_transactions",
        "posterior_replay_transactions",
        "exhaustive_integrity_audits",
        "full_state_materializations",
        "canonical_full_state_hashes",
    ):
        assert after_hypotheses[counter] == before_hypotheses[counter]
    assert (
        after_hypotheses["hypothesis_component_reads"]
        - before_hypotheses["hypothesis_component_reads"]
        == 10
    )
    assert max(elapsed_per_update) <= max(min(elapsed_per_update) * 5.0, 0.01)
    assert math.fsum(
        per_update * batch_size
        for per_update, batch_size in zip(
            elapsed_per_update,
            batch_sizes,
            strict=True,
        )
    ) < 10.0


def test_negative_is_local_and_later_positive_recovers_the_mode() -> None:
    memory, issuer, context, _ = _memory()
    before_snapshot = memory.apply_positive(
        _positive(memory, issuer, "p1", 1, {(0, 0): 0.5, (8, 8): 0.5})
    )
    before = _mass(before_snapshot)
    negative = memory.apply_negative(
        _negative(memory, issuer, context, "n1", 2, {(0, 0): 1.0})
    )
    downweighted = _mass(negative)
    assert 0.0 < downweighted[(0, 0)] < before[(0, 0)]
    assert downweighted[(8, 8)] == before[(8, 8)]
    assert negative.unlocalized_mass > before_snapshot.unlocalized_mass
    recovered = memory.apply_positive(
        _positive(memory, issuer, "p2", 3, {(0, 0): 1.0})
    )
    assert _mass(recovered)[(0, 0)] > before[(0, 0)]
    assert _mass(recovered)[(8, 8)] == before[(8, 8)]


def test_hypothesis_floor_applies_to_connected_component_total() -> None:
    config = TargetMemoryConfig(
        posterior_mass_floor=1e-3,
        component_mass_floor=0.1,
    )
    memory, producer, _, _ = _memory(config=config)
    support = {(index, 0): 0.1 for index in range(10)}
    snapshot = memory.apply_positive(
        _positive(memory, producer, "diffuse", 1, support)
    )
    assert all(row.value < config.component_mass_floor for row in snapshot.cell_mass)
    hypotheses = memory.hypotheses(snapshot)
    assert len(hypotheses) == 1
    assert hypotheses[0].cells == frozenset(support)
    assert hypotheses[0].mass == pytest.approx(0.5)


def test_snapshot_clone_and_cross_instance_content_forgery_are_rejected() -> None:
    memory, producer, _, _ = _memory()
    issued = memory.apply_positive(
        _positive(memory, producer, "p1", 1, {(0, 0): 1.0})
    )
    clone = replace(issued)
    assert clone.content_sha256 == issued.content_sha256
    with pytest.raises(TargetSnapshotBindingError, match="instance"):
        memory.assert_current_snapshot(clone)

    other, _, _, _ = _memory()
    with pytest.raises(TargetSnapshotBindingError, match="instance"):
        other.assert_current_snapshot(issued)


def test_context_pose_can_advance_without_physical_revision_and_rollback_fails() -> None:
    memory, issuer, context, _ = _memory()
    pose_advanced = _issue_context(
        issuer,
        revision=1,
        pose_timestamp_ns=101,
        pose_name="pose:advanced",
        domain=context.candidate_domain,
    )
    memory.advance_context(pose_advanced)
    assert memory.context.physical_revision == 1
    assert memory.context.pose_timestamp_ns == 101

    rollback = _issue_context(
        issuer,
        revision=1,
        pose_timestamp_ns=100,
        pose_name="pose:rollback",
        domain=context.candidate_domain,
    )
    before = memory.content_sha256
    with pytest.raises(TargetEvidenceRejectedError, match="timestamp"):
        memory.advance_context(rollback)
    assert memory.content_sha256 == before
    issuer.context_issuer.assert_issued_context(rollback)


def test_context_advance_failure_is_atomic_and_removed_mass_is_unlocalized() -> None:
    memory, issuer, _, _ = _memory()
    before_snapshot = memory.apply_positive(
        _positive(memory, issuer, "p1", 1, {(0, 0): 0.5, (1, 0): 0.5})
    )
    before_revision = memory.revision
    good = _issue_context(
        issuer,
        revision=2,
        domain=frozenset({(1, 0), (2, 0)}),
    )
    bad = replace(good, camera_calibration_sha256=_hash("other-camera"))
    with pytest.raises(TargetSnapshotBindingError, match="issued"):
        memory.advance_context(bad)
    assert memory.revision == before_revision
    issuer.context_issuer.assert_issued_context(good)

    memory.advance_context(good)
    after = memory.snapshot("red")
    assert set(_mass(after)) == {(1, 0)}
    assert after.unlocalized_mass > before_snapshot.unlocalized_mass


def test_stale_g5_context_fails_until_live_g3_context_is_installed() -> None:
    memory, producer, _, authority = _memory()
    pending_positive = _positive(memory, producer, "p1", 1, {(0, 0): 1.0})
    checkpoint = memory.serialize()
    _advance_g3(producer)
    with pytest.raises(TargetSnapshotBindingError, match="stale relative"):
        memory.apply_positive(pending_positive)
    with pytest.raises(TargetSnapshotBindingError, match="stale relative"):
        ReversibleTargetBeliefMemory.deserialize(
            checkpoint,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )
    current = _issue_context(producer, revision=2)
    memory.advance_context(current)
    assert memory.context.physical_revision == 2


def test_returned_context_config_and_attempts_are_defensive_copies() -> None:
    memory, _, original_context, _ = _memory()
    returned_context = memory.context
    returned_config = memory.config
    object.__setattr__(returned_context, "physical_revision", 999)
    object.__setattr__(returned_config, "cell_size_m", 9.0)
    object.__setattr__(original_context, "physical_revision", 777)
    assert memory.context.physical_revision == 1
    assert memory.config.cell_size_m == pytest.approx(0.1)


def test_config_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="cell_size"):
        TargetMemoryConfig(cell_size_m=0.0)
    with pytest.raises(ValueError, match="origin"):
        TargetMemoryConfig(origin_xy_m=(2_000_000.0, 0.0))
    with pytest.raises(ValueError, match="posterior/component"):
        TargetMemoryConfig(posterior_mass_floor=1e-3, component_mass_floor=1e-4)
    with pytest.raises(ValueError, match="supported.*extent"):
        TargetCellValue((10_000_001, 0), 1.0)


def test_context_rejects_cells_outside_supported_integer_extent() -> None:
    producer = _issuer()
    with pytest.raises(ValueError, match="supported.*extent"):
        _issue_context(producer, domain=frozenset({(10_000_001, 0)}))


def test_target_lattice_must_equal_bound_g3_map_frame() -> None:
    producer = _issuer()
    context = _issue_context(producer)
    authority = _authority(
        context_issuer_contract_sha256=producer.context_issuer.contract_sha256
    )
    with pytest.raises(TargetSnapshotBindingError, match="lattice"):
        ReversibleTargetBeliefMemory(
            context,
            TargetMemoryConfig(cell_size_m=1.0, origin_xy_m=(100.0, 100.0)),
            context_issuer=producer.context_issuer,
            episode_authority=authority,
        )


def _claim_fixture(
    *,
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    from lewm_worlds.manifest import (
        BoxObject,
        CameraValidityConstraints,
        SceneManifest,
        SpawnSpec,
        manifest_sha256,
    )

    red = BoxObject(
        object_id="beacon_red",
        kind="box",
        center_xyz_m=(1.0, 0.0, 0.5),
        size_xyz_m=(0.2, 0.2, 1.0),
        yaw_rad=0.0,
        material_id="landmark_red",
    )
    blue = BoxObject(
        object_id="beacon_blue",
        kind="box",
        center_xyz_m=(0.0, 2.0, 0.5),
        size_xyz_m=(0.2, 0.2, 1.0),
        yaw_rad=0.0,
        material_id="landmark_blue",
    )
    manifest = SceneManifest(
        scene_id="g5-claim-test",
        family="unit_test",
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-4.0, -4.0), (4.0, 4.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(blue, red),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=100.0,
            min_camera_clearance_m=0.1,
        ),
        split="train",
        walls=(),
        visual_randomization=None,
    )
    manifest_hash = manifest_sha256(manifest)
    producer = _issuer()
    authority = _authority(
        context_issuer_contract_sha256=(
            producer.context_issuer.contract_sha256
        ),
        scene_id=manifest.scene_id,
        manifest_sha256=manifest_hash,
        mapping=(("blue", "beacon_blue"), ("red", "beacon_red")),
    )
    reference = {"namespace": "object_id", "value": "beacon_red"}
    pose_values = list(pose)
    event = {
        "trace_id": "trace",
        "episode_id": authority.episode_id,
        "scene_id": manifest.scene_id,
        "event_id": "event-0",
        "tick": 10,
        "event_index": 0,
        "requested_target": reference,
        "claimed_target": reference,
        "robot_pose_world_xy_yaw": pose_values,
        "pose_binary64_le_sha256": hashlib.sha256(
            struct.pack("<3d", *pose_values)
        ).hexdigest(),
        "pose_hex": [value.hex() for value in pose_values],
        "pose_provenance": "runtime_full_precision",
        "physical_manifest_sha256": manifest_hash,
    }
    trace = {
        "schema": "lewm_go2_claim_trace_v1",
        "trace_id": "trace",
        "episode_id": authority.episode_id,
        "scene_id": manifest.scene_id,
        "physical_manifest_sha256": manifest_hash,
        "task_object_ids": list(authority.expected_task_object_ids),
        "task_object_set_sha256": authority.task_object_set_sha256,
        "controller_claim_attempts": [event],
        "evaluator_feedback_to_controller": [],
    }
    context = _issue_context(producer)
    memory = ReversibleTargetBeliefMemory(
        context,
        context_issuer=producer.context_issuer,
        episode_authority=authority,
    )
    return manifest, authority, event, trace, memory, producer, context


def _claim_event_variant(
    event: dict[str, object],
    *,
    event_id: str,
    event_index: int,
    tick: int,
    pose: tuple[float, float, float],
) -> dict[str, object]:
    result = json.loads(json.dumps(event))
    pose_values = list(pose)
    result.update(
        {
            "event_id": event_id,
            "event_index": event_index,
            "tick": tick,
            "robot_pose_world_xy_yaw": pose_values,
            "pose_binary64_le_sha256": hashlib.sha256(
                struct.pack("<3d", *pose_values)
            ).hexdigest(),
            "pose_hex": [value.hex() for value in pose_values],
        }
    )
    return result


def test_controller_wrong_identity_is_recorded_not_erased() -> None:
    _, _, event, _, memory, _, _ = _claim_fixture()
    wrong = json.loads(json.dumps(event))
    wrong_reference = {"namespace": "object_id", "value": "beacon_blue"}
    wrong["requested_target"] = wrong_reference
    wrong["claimed_target"] = wrong_reference
    attempt = memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=wrong,
    )
    assert attempt.identity_matches_expected is False
    assert json.loads(attempt.claimed_target_json) == wrong_reference
    returned = memory.controller_claim_attempts[0]
    object.__setattr__(returned, "identity_matches_expected", True)
    assert memory.controller_claim_attempts[0].identity_matches_expected is False


def test_evaluator_is_end_of_episode_observer_and_cannot_feed_control() -> None:
    manifest, _, event, trace, memory, producer, _ = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    with pytest.raises(TargetClaimVerificationError, match="finalized"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=trace,
            physical_manifest=manifest,
        )
    assert memory.physical_claim_evaluations == ()
    memory.finalize_episode_for_evaluation()
    with pytest.raises(TargetEvidenceRejectedError, match="finalized"):
        memory.apply_positive(
            _positive(memory, producer, "late", 11, {(0, 0): 1.0})
        )
    with pytest.raises(TargetEvidenceRejectedError, match="finalized"):
        memory.record_controller_claim_attempt(
            target_id="red",
            snapshot=memory.snapshot("red"),
            raw_event={**event, "event_id": "late", "tick": 11},
        )


def test_finalized_observer_survives_later_g3_advancement_and_restores() -> None:
    (
        manifest,
        authority,
        event,
        trace,
        memory,
        producer,
        _,
    ) = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    _advance_g3(producer)
    credit = memory.evaluate_and_record_verified_claim(
        target_id="red",
        event_id="event-0",
        raw_trace=trace,
        physical_manifest=manifest,
    )
    assert memory.controller_claim_attempts[0].event_id == "event-0"
    assert memory.physical_claim_evaluations[0].accepted is True
    payload = memory.serialize()
    restored = ReversibleTargetBeliefMemory.deserialize(
        payload,
        context_issuer=producer.context_issuer,
        expected_episode_authority=authority,
    )
    assert restored.verified_claims["red"].content_sha256 == credit.content_sha256


def test_first_canonical_accept_creates_credit_and_duplicate_only_adds_evaluation() -> None:
    manifest, _, event, trace, memory, _, _ = _claim_fixture()
    attempt = memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    credit = memory.evaluate_and_record_verified_claim(
        target_id="red",
        event_id="event-0",
        raw_trace=trace,
        physical_manifest=manifest,
    )
    assert credit.controller_attempt_sha256 == attempt.content_sha256
    assert memory.physical_claim_evaluations[0].verified_credit_created is True
    with pytest.raises(TargetClaimVerificationError, match="first verified"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=trace,
            physical_manifest=manifest,
        )
    assert len(memory.physical_claim_evaluations) == 2
    assert memory.physical_claim_evaluations[1].verified_credit_created is False
    assert len(memory.verified_claims) == 1


def test_rejected_and_unverifiable_events_are_both_immutably_logged() -> None:
    manifest, _, event, trace, memory, _, _ = _claim_fixture(
        pose=(0.0, 0.0, math.pi)
    )
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    with pytest.raises(TargetClaimVerificationError, match="rejected"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=trace,
            physical_manifest=manifest,
        )
    assert memory.physical_claim_evaluations[-1].status == "rejected"
    assert memory.verified_claims == {}

    (
        fresh_manifest,
        _,
        _,
        fresh_trace,
        fresh,
        _,
        _,
    ) = _claim_fixture(pose=(0.0, 0.0, math.pi))
    fresh.finalize_episode_for_evaluation()
    with pytest.raises(TargetClaimVerificationError, match="no controller"):
        fresh.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=fresh_trace,
            physical_manifest=fresh_manifest,
        )
    assert fresh.physical_claim_evaluations[-1].status == "unverifiable"


def test_malformed_observer_input_is_retained_as_unverifiable() -> None:
    manifest, _, event, _, memory, _, _ = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    with pytest.raises(TargetClaimVerificationError, match="canonical JSON"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace={"bad": math.nan},
            physical_manifest=manifest,
        )
    assert len(memory.physical_claim_evaluations) == 1
    assert memory.physical_claim_evaluations[0].status == "unverifiable"

    class ExplodingReprMapping(dict):
        def __repr__(self) -> str:
            raise RuntimeError("repr must not be called")

    with pytest.raises(TargetClaimVerificationError, match="canonical JSON"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=ExplodingReprMapping({"bad": object()}),
            physical_manifest=manifest,
        )
    assert len(memory.physical_claim_evaluations) == 2

    from collections.abc import Mapping as AbstractMapping

    class ExplodingIterationMapping(AbstractMapping):
        def __getitem__(self, key: object) -> object:
            raise KeyError(key)

        def __iter__(self):
            raise RuntimeError("iteration must not escape")

        def __len__(self) -> int:
            return 1

    with pytest.raises(TargetClaimVerificationError, match="canonical JSON"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=ExplodingIterationMapping(),
            physical_manifest=manifest,
        )
    assert len(memory.physical_claim_evaluations) == 3

    class ExplodingMetadata(type(AbstractMapping)):
        def __getattribute__(cls, name: str):
            if name in {"__module__", "__qualname__"}:
                raise RuntimeError("metaclass metadata must not be read")
            return super().__getattribute__(name)

    class ExplodingMetadataMapping(
        AbstractMapping,
        metaclass=ExplodingMetadata,
    ):
        def __getitem__(self, key: object) -> object:
            raise KeyError(key)

        def __iter__(self):
            raise RuntimeError("iteration must not escape")

        def __len__(self) -> int:
            return 1

    with pytest.raises(TargetClaimVerificationError, match="canonical JSON"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=ExplodingMetadataMapping(),
            physical_manifest=manifest,
        )
    assert len(memory.physical_claim_evaluations) == 4


def test_two_landmark_one_task_trace_exploit_is_rejected_from_independent_authority() -> None:
    manifest, authority, event, trace, memory, _, _ = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    one_task = json.loads(json.dumps(trace))
    one_task["task_object_ids"] = ["beacon_red"]
    one_task["task_object_set_sha256"] = _hash_object(
        {
            "schema": "lewm_go2_claim_task_set_v1",
            "scene_id": authority.scene_id,
            "physical_manifest_sha256": authority.physical_manifest_sha256,
            "task_object_ids": ["beacon_red"],
        }
    )
    with pytest.raises(TargetClaimVerificationError, match="authoritative"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=one_task,
            physical_manifest=manifest,
        )
    assert memory.physical_claim_evaluations[-1].status == "unverifiable"


def test_trace_cannot_omit_earlier_attempt_to_manufacture_first_credit() -> None:
    manifest, _, first_event, trace, memory, _, _ = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=first_event,
    )
    second_event = json.loads(json.dumps(first_event))
    second_event["event_id"] = "event-1"
    second_event["event_index"] = 1
    second_event["tick"] = 11
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=second_event,
    )
    memory.finalize_episode_for_evaluation()
    omitted = json.loads(json.dumps(trace))
    omitted["controller_claim_attempts"] = [second_event]
    with pytest.raises(TargetClaimVerificationError, match="complete ordered"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-1",
            raw_trace=omitted,
            physical_manifest=manifest,
        )
    assert memory.physical_claim_evaluations[-1].status == "unverifiable"


def test_canonical_observer_records_every_attempt_before_returning_credit() -> None:
    manifest, _, first_event, trace, memory, _, _ = _claim_fixture(
        pose=(0.0, 0.0, math.pi)
    )
    second_event = _claim_event_variant(
        first_event,
        event_id="event-1",
        event_index=1,
        tick=11,
        pose=(0.0, 0.0, 0.0),
    )
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=first_event,
    )
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=second_event,
    )
    complete_trace = json.loads(json.dumps(trace))
    complete_trace["controller_claim_attempts"] = [first_event, second_event]
    memory.finalize_episode_for_evaluation()
    credit = memory.evaluate_and_record_verified_claim(
        target_id="red",
        event_id="event-1",
        raw_trace=complete_trace,
        physical_manifest=manifest,
    )
    assert credit.event_id == "event-1"
    assert [
        (record.event_id, record.status, record.verified_credit_created)
        for record in memory.physical_claim_evaluations
    ] == [
        ("event-0", "rejected", False),
        ("event-1", "accepted", True),
    ]
    memory.serialize()


def test_alternate_manifest_is_rejected_and_logged_before_evaluation() -> None:
    manifest, _, event, trace, memory, _, _ = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    alternate = replace(manifest, topology_seed=999)
    with pytest.raises(TargetClaimVerificationError, match="differs"):
        memory.evaluate_and_record_verified_claim(
            target_id="red",
            event_id="event-0",
            raw_trace=trace,
            physical_manifest=alternate,
        )
    assert memory.physical_claim_evaluations[-1].status == "unverifiable"


def test_evaluation_and_credit_properties_return_defensive_copies() -> None:
    manifest, _, event, trace, memory, _, _ = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    memory.evaluate_and_record_verified_claim(
        target_id="red",
        event_id="event-0",
        raw_trace=trace,
        physical_manifest=manifest,
    )
    returned_record = memory.physical_claim_evaluations[0]
    returned_credit = memory.verified_claims["red"]
    object.__setattr__(returned_record, "status", "rejected")
    object.__setattr__(returned_credit, "object_id", "forged")
    assert memory.physical_claim_evaluations[0].status == "accepted"
    assert memory.verified_claims["red"].object_id == "beacon_red"


def test_full_serialization_round_trip_is_exact_and_invalidates_old_capabilities() -> None:
    manifest, authority, event, trace, memory, issuer, context = _claim_fixture()
    old_snapshot = memory.apply_positive(
        _positive(memory, issuer, "p1", 1, {(0, 0): 0.5, (8, 8): 0.5})
    )
    memory.apply_negative(
        _negative(memory, issuer, context, "n1", 2, {(0, 0): 1.0})
    )
    advanced_context = _issue_context(issuer, revision=2)
    memory.advance_context(advanced_context)
    attempt_snapshot = memory.snapshot("red")
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=attempt_snapshot,
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    memory.evaluate_and_record_verified_claim(
        target_id="red",
        event_id="event-0",
        raw_trace=trace,
        physical_manifest=manifest,
    )
    payload = memory.serialize()
    expected_content_sha256 = memory.content_sha256
    restored = ReversibleTargetBeliefMemory.deserialize(
        payload,
        context_issuer=issuer.context_issuer,
        expected_episode_authority=authority,
    )
    assert restored.serialize() == payload
    assert restored.content_sha256 == expected_content_sha256
    with pytest.raises(TargetSnapshotBindingError, match="writer lease"):
        memory.snapshot("red")
    with pytest.raises(TargetSnapshotBindingError, match="instance"):
        restored.assert_current_snapshot(old_snapshot)


def test_runtime_commitment_blocks_rollback_before_serialization() -> None:
    memory, producer, _, authority = _memory()
    old_checkpoint = memory.serialize()
    memory.apply_positive(
        _positive(memory, producer, "new-state", 1, {(0, 0): 1.0})
    )
    with pytest.raises(TargetSnapshotBindingError, match="latest runtime state"):
        ReversibleTargetBeliefMemory.deserialize(
            old_checkpoint,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )


def test_restore_transfers_single_writer_lease_to_latest_instance() -> None:
    memory, producer, _, authority = _memory()
    payload = memory.serialize()
    first_restore = ReversibleTargetBeliefMemory.deserialize(
        payload,
        context_issuer=producer.context_issuer,
        expected_episode_authority=authority,
    )
    second_restore = ReversibleTargetBeliefMemory.deserialize(
        payload,
        context_issuer=producer.context_issuer,
        expected_episode_authority=authority,
    )
    with pytest.raises(TargetSnapshotBindingError, match="writer lease"):
        memory.snapshot("red")
    with pytest.raises(TargetSnapshotBindingError, match="writer lease"):
        first_restore.snapshot("red")
    assert second_restore.snapshot("red").unlocalized_mass == 1.0


def test_memory_copies_and_object_shell_clones_never_inherit_writer_identity() -> None:
    memory, producer, _, _ = _memory()
    observation = _positive(memory, producer, "clone-attempt", 1, {(0, 0): 1.0})

    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(memory)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(memory)

    shell = object.__new__(ReversibleTargetBeliefMemory)
    shell.__dict__.update(memory.__dict__)
    with pytest.raises(TargetSnapshotBindingError, match="writer lease"):
        shell.snapshot("red")
    with pytest.raises(TargetSnapshotBindingError, match="writer lease"):
        shell.apply_positive(observation)

    # The rejected clone did not consume evidence or mutate the live posterior.
    result = memory.apply_positive(observation)
    assert _mass(result)[(0, 0)] == pytest.approx(0.5)


def test_second_fresh_writer_is_rejected_without_consuming_context() -> None:
    memory, producer, _, authority = _memory()
    second_context = _issue_context(
        producer,
        revision=1,
        pose_timestamp_ns=101,
        pose_name="second-writer",
    )
    with pytest.raises(TargetSnapshotBindingError, match="active target-memory writer"):
        ReversibleTargetBeliefMemory(
            second_context,
            context_issuer=producer.context_issuer,
            episode_authority=authority,
        )
    producer.context_issuer.assert_issued_context(second_context)
    assert memory.snapshot("red").unlocalized_mass == 1.0


def test_serialization_rejects_missing_tampered_and_alternate_authority_state() -> None:
    memory, issuer, _, authority = _memory()
    encoded = memory.serialize().decode("utf-8")
    root = json.loads(encoded)
    del root["seen_payload_sha256s"]
    with pytest.raises(ValueError, match="keys differ"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=issuer.context_issuer,
            expected_episode_authority=authority,
        )

    root = json.loads(encoded)
    root["revision"] = 999
    with pytest.raises(ValueError, match="content hash"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=issuer.context_issuer,
            expected_episode_authority=authority,
        )

    root = json.loads(encoded)
    core = dict(root)
    core.pop("state_content_sha256")
    root["state_content_sha256"] = _hash_object(core)
    alternate = _authority(
        context_issuer_contract_sha256=issuer.context_issuer.contract_sha256,
        episode_id="other-episode",
    )
    with pytest.raises(TargetSnapshotBindingError, match="independent expectation"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=issuer.context_issuer,
            expected_episode_authority=alternate,
        )

    root = json.loads(encoded)
    forged_pose = _hash("forged-serialized-pose")
    for context_row in (root["context"], root["context_history"][0]):
        context_row["pose_provenance_sha256"] = forged_pose
        context_row["issuance_id_sha256"] = _hash_object(
            {
                "schema": "lewm_g5_context_issuance_identity_v1",
                "issuer_sha256": context_row["issuer_sha256"],
                "sequence": context_row["context_sequence"],
                "physical_revision": context_row["physical_revision"],
                "pose_timestamp_ns": context_row["pose_timestamp_ns"],
                "configuration_snapshot_sha256": context_row[
                    "configuration_snapshot_sha256"
                ],
                "pose_provenance_sha256": forged_pose,
            }
        )
        context_core = dict(context_row)
        context_core.pop("content_sha256")
        context_row["content_sha256"] = _hash_object(context_core)
    core = dict(root)
    core.pop("state_content_sha256")
    root["state_content_sha256"] = _hash_object(core)
    with pytest.raises(TargetSnapshotBindingError, match="unknown to this issuer"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=issuer.context_issuer,
            expected_episode_authority=authority,
        )


def test_deserialization_replays_posterior_instead_of_trusting_mass_fields() -> None:
    memory, producer, _, authority = _memory()
    root = json.loads(memory.serialize())
    root["cell_mass"]["red"] = [{"cell": [0, 0], "value": 0.5}]
    root["unlocalized_mass"]["red"] = 0.5
    core = dict(root)
    core.pop("state_content_sha256")
    root["state_content_sha256"] = _hash_object(core)
    with pytest.raises(TargetSnapshotBindingError, match="runtime state commitment"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )

    fabricated = _positive(memory, producer, "fabricated", 1, {(0, 0): 1.0})
    root = json.loads(memory.serialize())
    root["positive_observations"] = [fabricated.to_dict()]
    root["evidence_transaction_order"] = [
        {"kind": "positive", "observation_id": "fabricated"}
    ]
    root["seen_observation_ids"] = ["fabricated"]
    root["seen_semantic_sha256s"] = [fabricated.semantic_sha256]
    root["seen_payload_sha256s"] = [fabricated.identity.payload_sha256]
    root["cell_mass"]["red"] = [{"cell": [0, 0], "value": 0.5}]
    root["unlocalized_mass"]["red"] = 0.5
    root["revision"] = 1
    root["current_tick"] = 1
    core = dict(root)
    core.pop("state_content_sha256")
    root["state_content_sha256"] = _hash_object(core)
    with pytest.raises(TargetSnapshotBindingError, match="runtime state commitment"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )


def test_deserialization_applies_runtime_pose_only_context_rules() -> None:
    memory, producer, _, authority = _memory()
    alternate = _issue_context(
        producer,
        revision=1,
        pose_timestamp_ns=101,
        pose_name="alternate-pose-domain",
        domain=frozenset({(0, 0), (1, 0)}),
    )
    root = json.loads(memory.serialize())
    root["context"] = alternate.to_dict()
    root["context_history"].append(alternate.to_dict())
    root["revision"] = 1
    core = dict(root)
    core.pop("state_content_sha256")
    root["state_content_sha256"] = _hash_object(core)
    with pytest.raises(TargetSnapshotBindingError, match="pose-only context"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )


def test_deserialization_requires_runtime_issued_evaluation_receipts() -> None:
    manifest, authority, event, trace, memory, producer, _ = _claim_fixture()
    memory.record_controller_claim_attempt(
        target_id="red",
        snapshot=memory.snapshot("red"),
        raw_event=event,
    )
    memory.finalize_episode_for_evaluation()
    memory.evaluate_and_record_verified_claim(
        target_id="red",
        event_id="event-0",
        raw_trace=trace,
        physical_manifest=manifest,
    )
    root = json.loads(memory.serialize())
    forged_contract = _hash("forged-evaluator-contract")
    evaluation = root["physical_claim_evaluations"][0]
    evaluation["evaluator_contract_sha256"] = forged_contract
    evaluation_core = dict(evaluation)
    evaluation_core.pop("content_sha256")
    evaluation["content_sha256"] = _hash_object(evaluation_core)
    credit = root["verified_claims"][0]
    credit["evaluator_contract_sha256"] = forged_contract
    credit_core = dict(credit)
    credit_core.pop("content_sha256")
    credit["content_sha256"] = _hash_object(credit_core)
    core = dict(root)
    core.pop("state_content_sha256")
    root["state_content_sha256"] = _hash_object(core)
    with pytest.raises(TargetSnapshotBindingError, match="runtime state commitment"):
        ReversibleTargetBeliefMemory.from_mapping(
            root,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )

    erased = json.loads(memory.serialize())
    erased["verified_claims"] = []
    erased_core = dict(erased)
    erased_core.pop("state_content_sha256")
    erased["state_content_sha256"] = _hash_object(erased_core)
    with pytest.raises(TargetSnapshotBindingError, match="runtime state commitment"):
        ReversibleTargetBeliefMemory.from_mapping(
            erased,
            context_issuer=producer.context_issuer,
            expected_episode_authority=authority,
        )
