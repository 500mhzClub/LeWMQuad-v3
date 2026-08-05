from __future__ import annotations

import copy
import hashlib
import pickle

import pytest
import torch

import lewm.benchmarks.qualified_shared_v5_navigation_runtime_v1 as runtime_module
from lewm.benchmarks.go2_navigation_development_trace_v1 import canonical_binary64_hex
from lewm.benchmarks.qualified_shared_v5_navigation_runtime_v1 import (
    DetachedSharedV5FeatureCacheV1,
    ExactObjectLeaseV1,
    FakeSharedV5FrameBackendV1,
    FrameTickBindingV1,
    PhysicalViewBindingPayloadV1,
    QualifiedSharedV5FrameOutcomeV1,
    QualifiedSharedV5NavigationRuntimeV1,
    QualifiedSharedV5NavigationRuntimeV1BindingError,
    QualifiedSharedV5NavigationRuntimeV1ReplayError,
    QualifiedSharedV5NavigationRuntimeV1TerminalError,
    TargetColorObservationPayloadV1,
    synthetic_frame_content_sha256_v1,
)
from lewm.models.shared_v5_target_observation_head_v1 import (
    SharedV5TargetObservationHeadConfigV1,
    SharedV5TargetObservationHeadV1,
    initialize_deterministic_mock_weights_v1 as initialize_target,
)
from lewm.models.two_resolution_frontier_value_head_v1 import (
    FrozenCandidateFeatureBatchV1,
    TwoResolutionFrontierValueHeadConfigV1,
    TwoResolutionFrontierValueHeadV1,
    initialize_deterministic_mock_weights_v1 as initialize_g4,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _frame() -> torch.Tensor:
    return torch.arange(1 * 3 * 8 * 8, dtype=torch.uint8).reshape(1, 3, 8, 8)


def _binding(frame: torch.Tensor, *, tick: int = 0, physical_revision: int = 0) -> FrameTickBindingV1:
    return FrameTickBindingV1(
        tick_index=tick,
        reset_id="reset-001",
        session_id="session-001",
        pre_physical_revision=physical_revision,
        controller_input_sha256=_h(f"controller-input-{tick}"),
        rgb_content_sha256=synthetic_frame_content_sha256_v1(frame),
        timestamp_binary64_hex=canonical_binary64_hex(float(tick) + 0.25),
        synchronization_id=f"sync-{tick:03d}",
    )


def _runtime() -> QualifiedSharedV5NavigationRuntimeV1:
    backend = FakeSharedV5FrameBackendV1(
        patch_feature_dim=8,
        bev_feature_dim=6,
        _synthetic_test_fixture=True,
    )
    return QualifiedSharedV5NavigationRuntimeV1(
        backend=backend,
        reset_id="reset-001",
        session_id="session-001",
        _synthetic_mock=True,
    )


def _target_head() -> SharedV5TargetObservationHeadV1:
    head = SharedV5TargetObservationHeadV1(
        SharedV5TargetObservationHeadConfigV1(
            patch_feature_dim=8,
            bev_feature_dim=6,
            hidden_dim=16,
            color_embedding_dim=4,
        )
    )
    initialize_target(head, seed=3)
    head.eval()
    return head


def _g4_head() -> TwoResolutionFrontierValueHeadV1:
    head = TwoResolutionFrontierValueHeadV1(
        TwoResolutionFrontierValueHeadConfigV1(
            patch_feature_dim=8,
            bev_feature_dim=6,
            candidate_feature_dim=5,
            hidden_dim=16,
        )
    )
    initialize_g4(head, seed=5)
    head.eval()
    return head


def _admit(
    runtime: QualifiedSharedV5NavigationRuntimeV1,
    outcome: QualifiedSharedV5FrameOutcomeV1,
):
    return runtime.admit_tick(
        outcome=outcome,
        post_physical_revision=1,
        physical_transaction_sha256=_h("physical-transaction"),
        physical_retraction_sha256=_h("physical-retraction"),
        post_physical_content_sha256=_h("post-physical"),
        post_configuration_revision=1,
        configuration_snapshot_sha256=_h("snapshot"),
        configuration_component_sha256=_h("component"),
        frontier_sha256=_h("frontier"),
    )


def test_one_encode_four_colour_and_at_most_one_g4_call() -> None:
    runtime = _runtime()
    frame = _frame()
    outcome = runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)
    assert type(outcome.feature_cache) is DetachedSharedV5FeatureCacheV1
    target = runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())
    assert target.colors == ("red", "yellow", "blue", "green")
    receipt = _admit(runtime, outcome)

    physical_consumer = object()
    physical_lease = receipt.mint_physical_view_lease(consumer=physical_consumer)
    physical_payload = physical_lease.consume(
        receipt=receipt, outcome=outcome, consumer=physical_consumer
    )
    assert type(physical_payload) is PhysicalViewBindingPayloadV1
    assert physical_payload.frame_outcome_sha256 == outcome.content_sha256

    consumers = (object(), object(), object(), object())
    target_leases = receipt.mint_target_evidence_leases(consumers=consumers)
    for index, lease in enumerate(target_leases):
        payload = lease.consume(
            receipt=receipt,
            outcome=outcome,
            consumer=consumers[index],
        )
        assert type(payload) is TargetColorObservationPayloadV1
        assert payload.color_index == index
        assert payload.four_color_output_sha256
        assert tuple(payload.presence_probability.shape) == (1,)
        assert not hasattr(payload, "four_color_output")

    g4 = _g4_head()
    feature_lease = receipt.mint_g4_cached_feature_lease(consumer=g4)
    candidate_batch = FrozenCandidateFeatureBatchV1(
        candidate_set_sha256=_h("candidate-set"),
        candidate_row_sha256s=(_h("candidate-0"), _h("candidate-1")),
        features=torch.linspace(0.0, 1.0, 1 * 2 * 5).reshape(1, 2, 5),
    )
    scores = runtime.run_g4_value_head_once(
        receipt=receipt,
        feature_lease=feature_lease,
        head=g4,
        candidate_batch=candidate_batch,
    )
    assert scores.candidate_batch is candidate_batch
    counts = runtime.commit_tick(receipt=receipt)
    counts.assert_one_encode_invariants()
    assert counts.observation_tick_count == 1
    assert counts.shared_v5_forward_frame_call_count == 1
    assert counts.vision_encoder_forward_tokens_call_count == 1
    assert counts.target_four_color_batch_count == 1
    assert counts.g4_value_head_call_count == 1
    assert counts.extra_rgb_decode_or_preprocess_count == 0


def test_frame_and_head_replay_are_forbidden_without_recompute() -> None:
    runtime = _runtime()
    frame = _frame()
    binding = _binding(frame)
    runtime.run_shared_frame_once(binding=binding, synthetic_frame=frame)
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1ReplayError):
        runtime.run_shared_frame_once(binding=binding, synthetic_frame=frame)
    assert runtime.call_counters.shared_v5_forward_frame_call_count == 1
    assert runtime.call_counters.vision_encoder_forward_tokens_call_count == 1

    runtime = _runtime()
    outcome = runtime.run_shared_frame_once(binding=binding, synthetic_frame=frame)
    head = _target_head()
    runtime.run_target_four_color_batch_once(outcome=outcome, head=head)
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1ReplayError):
        runtime.run_target_four_color_batch_once(outcome=outcome, head=head)
    assert runtime.call_counters.shared_v5_forward_frame_call_count == 1
    assert runtime.call_counters.vision_encoder_forward_tokens_call_count == 1


def test_exact_object_lease_rejects_foreign_consumer_and_replay() -> None:
    runtime = _runtime()
    frame = _frame()
    outcome = runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)
    runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())
    receipt = _admit(runtime, outcome)
    consumer = object()
    lease = receipt.mint_physical_view_lease(consumer=consumer)
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1BindingError):
        lease.consume(receipt=receipt, outcome=outcome, consumer=object())
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1ReplayError):
        lease.consume(receipt=receipt, outcome=outcome, consumer=consumer)

    runtime = _runtime()
    outcome = runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)
    runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())
    receipt = _admit(runtime, outcome)
    consumer = object()
    lease = receipt.mint_physical_view_lease(consumer=consumer)
    assert type(lease.consume(receipt=receipt, outcome=outcome, consumer=consumer)) is PhysicalViewBindingPayloadV1
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1ReplayError):
        lease.consume(receipt=receipt, outcome=outcome, consumer=consumer)


def test_lease_and_receipt_are_noncopyable_nonserializable_and_expire() -> None:
    runtime = _runtime()
    frame = _frame()
    outcome = runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)
    runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())
    receipt = _admit(runtime, outcome)
    consumer = object()
    lease = receipt.mint_physical_view_lease(consumer=consumer)
    for operation in (
        lambda: copy.copy(lease),
        lambda: copy.deepcopy(lease),
        lambda: pickle.dumps(lease),
        lambda: copy.copy(receipt),
        lambda: pickle.dumps(receipt),
    ):
        with pytest.raises(TypeError):
            operation()
    runtime.commit_tick(receipt=receipt)
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1ReplayError):
        lease.consume(receipt=receipt, outcome=outcome, consumer=consumer)


def test_cross_session_revision_and_frame_commitment_reject_before_backend_call() -> None:
    runtime = _runtime()
    frame = _frame()
    wrong_session = FrameTickBindingV1(
        tick_index=0,
        reset_id="reset-001",
        session_id="session-foreign",
        pre_physical_revision=0,
        controller_input_sha256=_h("input"),
        rgb_content_sha256=synthetic_frame_content_sha256_v1(frame),
        timestamp_binary64_hex=canonical_binary64_hex(0.0),
        synchronization_id="sync-000",
    )
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1BindingError):
        runtime.run_shared_frame_once(binding=wrong_session, synthetic_frame=frame)
    assert runtime.call_counters.observation_tick_count == 0

    runtime = _runtime()
    wrong_content = _binding(frame)
    changed = frame.clone()
    changed[0, 0, 0, 0] += 1
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1BindingError):
        runtime.run_shared_frame_once(binding=wrong_content, synthetic_frame=changed)
    assert runtime.call_counters.observation_tick_count == 0


def test_fault_expires_authority_and_seals_runtime() -> None:
    runtime = _runtime()
    frame = _frame()
    outcome = runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)
    runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())
    receipt = _admit(runtime, outcome)
    runtime.fault_tick(receipt=receipt)
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1TerminalError):
        runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)


def test_cached_feature_or_candidate_mutation_is_terminal_and_never_retryable() -> None:
    runtime = _runtime()
    frame = _frame()
    outcome = runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)
    outcome.feature_cache.patch_features[0, 0, 0] += 1.0
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1TerminalError):
        runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1TerminalError):
        runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())

    runtime = _runtime()
    outcome = runtime.run_shared_frame_once(binding=_binding(frame), synthetic_frame=frame)
    runtime.run_target_four_color_batch_once(outcome=outcome, head=_target_head())
    receipt = _admit(runtime, outcome)
    g4 = _g4_head()
    lease = receipt.mint_g4_cached_feature_lease(consumer=g4)
    candidates = FrozenCandidateFeatureBatchV1(
        candidate_set_sha256=_h("candidate-set-mutated"),
        candidate_row_sha256s=(_h("candidate-mutated-0"),),
        features=torch.zeros(1, 1, 5),
    )
    candidates.features[0, 0, 0] += 1.0
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1TerminalError):
        runtime.run_g4_value_head_once(
            receipt=receipt,
            feature_lease=lease,
            head=g4,
            candidate_batch=candidates,
        )
    with pytest.raises(QualifiedSharedV5NavigationRuntimeV1TerminalError):
        runtime.run_g4_value_head_once(
            receipt=receipt,
            feature_lease=lease,
            head=g4,
            candidate_batch=candidates,
        )


def test_source_only_production_identities_are_all_unresolved() -> None:
    names = [
        name
        for name in vars(runtime_module)
        if name.startswith("PRODUCTION_")
    ]
    assert names
    assert all(getattr(runtime_module, name) is None for name in names)
    with pytest.raises(PermissionError):
        FakeSharedV5FrameBackendV1(patch_feature_dim=8, bev_feature_dim=6)
    backend = FakeSharedV5FrameBackendV1(
        patch_feature_dim=8,
        bev_feature_dim=6,
        _synthetic_test_fixture=True,
    )
    with pytest.raises(PermissionError):
        QualifiedSharedV5NavigationRuntimeV1(
            backend=backend,
            reset_id="reset-001",
            session_id="session-001",
        )
