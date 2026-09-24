from __future__ import annotations

import copy
import hashlib
import pickle

import pytest
import torch

from lewm.models.two_resolution_frontier_value_head_v1 import (
    FrozenCandidateFeatureBatchV1,
    FrontierValueScoresV1,
    TwoResolutionFrontierValueHeadConfigV1,
    TwoResolutionFrontierValueHeadV1,
    TwoResolutionFrontierValueHeadV1Error,
    initialize_deterministic_mock_weights_v1,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _head() -> TwoResolutionFrontierValueHeadV1:
    head = TwoResolutionFrontierValueHeadV1(
        TwoResolutionFrontierValueHeadConfigV1(
            patch_feature_dim=8,
            bev_feature_dim=6,
            candidate_feature_dim=5,
            hidden_dim=16,
        )
    )
    initialize_deterministic_mock_weights_v1(head, seed=23)
    head.eval()
    return head


def _inputs() -> tuple[torch.Tensor, torch.Tensor, FrozenCandidateFeatureBatchV1]:
    patch = torch.linspace(-1.0, 1.0, 2 * 5 * 8).reshape(2, 5, 8)
    bev = torch.linspace(-0.5, 0.5, 2 * 6 * 3 * 4).reshape(2, 6, 3, 4)
    candidates = torch.linspace(-0.25, 0.75, 2 * 3 * 5).reshape(2, 3, 5)
    batch = FrozenCandidateFeatureBatchV1(
        candidate_set_sha256=_h("candidate-set"),
        candidate_row_sha256s=(_h("row-z"), _h("row-a"), _h("row-m")),
        features=candidates,
    )
    return patch, bev, batch


def test_frontier_head_scores_exact_candidate_object_and_order_without_mutation() -> None:
    head = _head()
    patch, bev, batch = _inputs()
    versions = (patch._version, bev._version, batch.features._version)
    with torch.no_grad():
        result = head(patch, bev, batch)
    assert type(result) is FrontierValueScoresV1
    assert result.candidate_batch is batch
    assert result.candidate_set_sha256 == batch.candidate_set_sha256
    assert result.candidate_row_sha256s == batch.candidate_row_sha256s
    assert tuple(result.scores.shape) == (2, 3)
    assert torch.isfinite(result.scores).all()
    assert versions == (patch._version, bev._version, batch.features._version)
    assert len(result.selected_row_indices()) == 2


def test_frontier_first_row_is_only_tie_break() -> None:
    _, _, batch = _inputs()
    scores = FrontierValueScoresV1(
        candidate_batch=batch,
        candidate_set_sha256=batch.candidate_set_sha256,
        candidate_row_sha256s=batch.candidate_row_sha256s,
        scores=torch.zeros(2, 3),
    )
    assert scores.selected_row_indices() == (0, 0)


def test_frozen_candidate_batch_rejects_copy_serialization_and_mutation() -> None:
    _, _, batch = _inputs()
    with pytest.raises(TypeError):
        copy.copy(batch)
    with pytest.raises(TypeError):
        copy.deepcopy(batch)
    with pytest.raises(TypeError):
        pickle.dumps(batch)
    batch.features[0, 0, 0] += 1.0
    with pytest.raises(TwoResolutionFrontierValueHeadV1Error):
        batch.assert_unchanged()


def test_frontier_head_rejects_attached_reconstructed_and_wrong_dim_features() -> None:
    head = _head()
    patch, bev, batch = _inputs()
    with pytest.raises(TwoResolutionFrontierValueHeadV1Error):
        head(patch.requires_grad_(), bev, batch)
    patch = patch.detach()
    with pytest.raises(TwoResolutionFrontierValueHeadV1Error):
        head(patch, bev, FrozenCandidateFeatureBatchV1(
            candidate_set_sha256=batch.candidate_set_sha256,
            candidate_row_sha256s=batch.candidate_row_sha256s,
            features=torch.zeros(2, 3, 4),
        ))


def test_frontier_head_owns_no_encoder_candidate_generator_or_fallback() -> None:
    head = _head()
    assert head.owned_encoder_count == 0
    assert head.owned_rgb_preprocessor_count == 0
    assert head.owns_candidate_generator is False
    assert head.has_fallback_selector is False
    assert all("encoder" not in name.lower() for name, _ in head.named_modules())
