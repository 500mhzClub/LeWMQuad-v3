from __future__ import annotations

import copy
import pickle

import pytest
import torch

from lewm.benchmarks.go2_navigation_development_trace_v2 import (
    CANONICAL_COLORS_V2,
    canonical_binary64_hex_v2,
    canonical_json_sha256_v2,
)
from lewm.benchmarks.qualified_shared_v5_navigation_runtime_v2 import (
    CandidateSetAdmissionV2,
    ExactObjectLeaseV2,
    FoundationAuthorityIssuerV2,
    G4ScoringReceiptV2,
    QualifiedSharedV5NavigationRuntimeV2,
    SyntheticCandidateRowV2,
    TargetBatchReceiptV2,
)
from lewm.models.shared_v5_target_observation_head_v1 import (
    SharedV5TargetObservationHeadConfigV1,
    SharedV5TargetObservationHeadV1,
    initialize_deterministic_mock_weights_v1 as initialize_target_weights,
)
from lewm.models.two_resolution_frontier_value_head_v1 import (
    TwoResolutionFrontierValueHeadConfigV1,
    TwoResolutionFrontierValueHeadV1,
    initialize_deterministic_mock_weights_v1 as initialize_g4_weights,
)


def _sha(label: str) -> str:
    return canonical_json_sha256_v2({"label": label})


def _foundation() -> tuple[
    QualifiedSharedV5NavigationRuntimeV2,
    SharedV5TargetObservationHeadV1,
    TwoResolutionFrontierValueHeadV1,
]:
    torch.set_num_threads(1)
    issuer = FoundationAuthorityIssuerV2.create_for_source_tests(
        _source_test_capability=True
    )
    authority = issuer.mint_reset_authority()
    backend = issuer.create_fake_frame_backend(patch_feature_dim=4, bev_feature_dim=3)
    runtime = issuer.start_synthetic_runtime(
        reset_authority=authority, backend=backend
    )
    target = SharedV5TargetObservationHeadV1(
        SharedV5TargetObservationHeadConfigV1(
            patch_feature_dim=4, bev_feature_dim=3, hidden_dim=8,
            color_embedding_dim=4,
        )
    ).eval()
    initialize_target_weights(target, seed=7)
    g4 = TwoResolutionFrontierValueHeadV1(
        TwoResolutionFrontierValueHeadConfigV1(
            patch_feature_dim=4, bev_feature_dim=3,
            candidate_feature_dim=3, hidden_dim=8,
        )
    ).eval()
    initialize_g4_weights(g4, seed=11)
    return runtime, target, g4


def _admitted():
    runtime, target, g4 = _foundation()
    frame = torch.linspace(0.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)
    outcome = runtime.run_shared_frame_once(
        synthetic_frame=frame,
        controller_input_sha256=_sha("controller-input"),
        timestamp_binary64_hex=canonical_binary64_hex_v2(1.25),
        synchronization_id="sync-0",
    )
    target_receipt = runtime.run_target_four_color_batch_once(
        outcome=outcome, head=target
    )
    producer_receipt = runtime.mint_physical_projection_receipt_once(
        outcome=outcome, target_batch_receipt=target_receipt
    )
    admission = runtime.admit_tick(
        outcome=outcome,
        target_batch_receipt=target_receipt,
        producer_receipt=producer_receipt,
    )
    return runtime, target, g4, outcome, target_receipt, producer_receipt, admission


def test_issuer_tick_zero_and_one_encode_target_freeze() -> None:
    runtime, _, _, outcome, target_receipt, _, admission = _admitted()
    reset = runtime.reset_receipt
    assert outcome.tick_index == 0
    assert tuple(row.revision for row in reset.initial_owner_states.rows) == (0,) * 13
    identities = (
        reset.reset_id,
        reset.session_id,
        reset.reset_capability_id,
        reset.physical_projection_producer_id,
        reset.candidate_producer_id,
        *(row.owner_id for row in reset.initial_owner_states.rows),
    )
    assert len(identities) == len(set(identities))
    assert type(target_receipt) is TargetBatchReceiptV2
    assert set(target_receipt.diagnostic_dict()) == {
        "schema", "version", "content_sha256", "frame_outcome_sha256",
        "frozen_batch_sha256", "counter_receipt_sha256", "tick_index",
        "reset_id", "session_id",
    }

    consumers = tuple(object() for _ in CANONICAL_COLORS_V2)
    leases = admission.issue_target_evidence_leases_atomic(consumers=consumers)
    payloads = (
        admission.consume_target_red_lease(lease=leases[0], consumer=consumers[0]),
        admission.consume_target_yellow_lease(lease=leases[1], consumer=consumers[1]),
        admission.consume_target_blue_lease(lease=leases[2], consumer=consumers[2]),
        admission.consume_target_green_lease(lease=leases[3], consumer=consumers[3]),
    )
    assert tuple(payload.color for payload in payloads) == CANONICAL_COLORS_V2
    assert all(payload.tensor("presence_probability").shape == (1,) for payload in payloads)
    post = runtime.commit_tick(receipt=admission)
    for name in (
        "physical", "configuration", "view", "target_red", "target_yellow",
        "target_blue", "target_green", "integration", "tick_chain",
    ):
        assert post.row(name).revision == 1
    for name in ("router", "follower", "action_journal", "claim_journal"):
        assert post.row(name).revision == 0
    counts = runtime.call_counters
    counts.assert_complete_observation(g4_calls=0)


def test_candidate_baseline_and_g4_share_one_registered_object() -> None:
    runtime, _, g4, _, _, _, admission = _admitted()
    rows = tuple(
        SyntheticCandidateRowV2(
            selected_path_sha256=_sha(f"path-{index}"),
            terminal_yaw_binary64_hex=canonical_binary64_hex_v2(index / 10.0),
            baseline_value_binary64_hex=canonical_binary64_hex_v2(float(index)),
        )
        for index in range(3)
    )
    candidate = runtime.mint_synthetic_candidate_set_once(
        receipt=admission,
        rows=rows,
        features=torch.tensor(
            [[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.5, 0.4]]],
            dtype=torch.float32,
        ),
    )
    assert type(candidate) is CandidateSetAdmissionV2
    assert candidate.baseline_selected_row_index == 2
    baseline = runtime.run_deterministic_baseline_once(
        receipt=admission, candidate_admission=candidate
    )
    scoring = runtime.run_g4_value_head_once(
        receipt=admission,
        candidate_admission=candidate,
        baseline_receipt=baseline,
        head=g4,
    )
    assert type(scoring) is G4ScoringReceiptV2
    assert scoring.diagnostic_dict()["candidate_set_sha256"] == candidate.candidate_set_sha256
    post = runtime.commit_tick(receipt=admission)
    assert post.row("view").revision == 1
    runtime.call_counters.assert_complete_observation(g4_calls=1)


def test_live_authority_objects_are_noncopyable_and_nonserializable() -> None:
    runtime, _, _, _, _, _, admission = _admitted()
    consumer = object()
    lease = admission.issue_physical_view_lease(consumer=consumer)
    assert type(lease) is ExactObjectLeaseV2
    for operation in (
        lambda: copy.copy(lease),
        lambda: copy.deepcopy(lease),
        lambda: pickle.dumps(lease),
        lambda: copy.copy(admission),
        lambda: pickle.dumps(runtime),
    ):
        with pytest.raises(TypeError):
            operation()
