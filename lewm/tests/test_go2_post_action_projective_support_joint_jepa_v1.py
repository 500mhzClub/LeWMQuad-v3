from __future__ import annotations

import hashlib
import math

import pytest
import torch
import torch.nn.functional as F

from lewm.benchmarks import (
    go2_post_action_projective_support_joint_jepa_v1 as api,
)
from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    latent_energy_per_row,
)


def _mask_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(bytes(value.contiguous().reshape(-1).tolist())).hexdigest()


def test_exact_full_persistence_and_projective_support_masks() -> None:
    masks = api.build_validated_corridor_masks_v1()
    assert masks.full.shape == (11, 64, 64)
    assert masks.full.dtype == torch.uint8 and masks.full.is_contiguous()
    assert tuple(int(row.sum()) for row in masks.full) == api.FULL_MASK_COUNTS
    assert int(masks.full.sum()) == api.FULL_MASK_TOTAL
    assert _mask_sha256(masks.full) == api.FULL_MASK_SHA256

    assert masks.persistence.shape == (9, 11, 64, 64)
    assert masks.persistence.dtype == torch.uint8
    for action_index, action in enumerate(api.ACTION_ORDER):
        action_mask = masks.persistence[action_index]
        assert tuple(int(row.sum()) for row in action_mask) == (
            api.PERSISTENCE_MASK_COUNTS[action]
        )
        assert _mask_sha256(action_mask) == api.PERSISTENCE_MASK_SHA256[action]
    assert int(masks.persistence.sum()) == api.PERSISTENCE_MASK_TOTAL
    assert _mask_sha256(masks.persistence) == api.PERSISTENCE_MASK_STACK_SHA256

    assert masks.projective_support.shape == (64, 64)
    assert masks.projective_support.dtype == torch.bool
    assert int(masks.projective_support.sum()) == 1_964
    assert not bool((masks.full.bool() & ~masks.projective_support).any())
    assert not bool(
        (masks.persistence.bool() & ~masks.projective_support[None, None]).any()
    )
    # Public callers cannot mutate the cached authority copy.
    masks.full.zero_()
    assert int(api.build_full_corridor_masks_v1().sum()) == api.FULL_MASK_TOTAL


def test_immediate_swept_footprints_have_exact_zero_projective_support() -> None:
    report = api.build_immediate_footprint_support_regression_v1()
    assert report.passed is True
    assert report.projective_support_cell_count == 1_964
    assert report.overlap_cell_count == 0
    assert report.action_overlap_cell_counts == (0,) * api.ACTION_COUNT
    assert len(report.action_mask_cell_counts) == api.ACTION_COUNT
    assert all(count > 0 for count in report.action_mask_cell_counts)
    assert len(report.action_sample_counts) == api.ACTION_COUNT
    assert all(count > 0 for count in report.action_sample_counts)
    assert len(report.mask_stack_sha256) == 64

    support = api.build_projective_support_mask_v1()
    masks = torch.zeros((api.ACTION_COUNT, 64, 64), dtype=torch.uint8)
    first_support_cell = support.nonzero()[0]
    masks[0, int(first_support_cell[0]), int(first_support_cell[1])] = 1
    with pytest.raises(RuntimeError, match="overlaps projective support"):
        api.validate_immediate_footprint_support_regression_v1(
            masks,
            support,
            (1,) * api.ACTION_COUNT,
        )


class _ParameterFreeModelStub:
    def predict_all_actions(self, current_latent: torch.Tensor) -> torch.Tensor:
        action = torch.arange(
            api.ACTION_COUNT,
            dtype=current_latent.dtype,
            device=current_latent.device,
        )[None, :, None, None, None]
        return current_latent[:, None].expand(-1, api.ACTION_COUNT, -1, -1, -1) + action

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :3]


def test_all_action_prediction_decode_and_full_score_are_differentiable() -> None:
    model = _ParameterFreeModelStub()
    current = torch.randn(1, 64, 64, 64, requires_grad=True)
    predicted, decoded = api.predict_and_decode_all_actions_v1(model, current)
    assert predicted.shape == (1, 9, 64, 64, 64)
    assert decoded.shape == (1, 9, 3, 64, 64)
    assert torch.equal(decoded, predicted[:, :, :3])

    scores = api.corridor_scores_from_semantic_logits_v1(
        decoded, api.build_full_corridor_masks_v1()
    )
    assert scores.free_log_odds.shape == (1, 9, 64, 64)
    assert scores.station_logits.shape == (1, 9, 11)
    assert scores.station_probabilities.shape == (1, 9, 11)
    assert scores.prefix_utility.shape == (1, 9)
    assert bool(torch.isfinite(scores.station_logits).all())
    scores.prefix_utility.sum().backward()
    assert current.grad is not None and float(current.grad.abs().sum()) > 0.0


def test_free_log_odds_normalized_smooth_min_and_persistence_broadcast() -> None:
    target_log_odds = torch.tensor(((-1.5, 0.25), (1.0, 2.0)))
    semantic = torch.zeros((1, 9, 3, 2, 2), requires_grad=True)
    semantic = semantic.clone()
    semantic[:, :, 1] = target_log_odds + math.log(2.0)
    log_odds = api.free_log_odds_v1(semantic)
    torch.testing.assert_close(
        log_odds,
        target_log_odds[None, None].expand(1, 9, -1, -1),
    )
    one_cell = torch.zeros((11, 2, 2), dtype=torch.uint8)
    one_cell[:, 1, 0] = 1
    station_logits = api.smooth_min_station_logits_v1(log_odds, one_cell)
    torch.testing.assert_close(
        station_logits,
        torch.full((1, 9, 11), target_log_odds[1, 0]),
    )

    # A uniform field remains unchanged by the log(N)-normalized smooth minimum,
    # including with the nine distinct current-frame persistence masks.
    current_semantic = torch.zeros((1, 3, 64, 64))
    persistence = api.corridor_scores_from_semantic_logits_v1(
        current_semantic, api.build_persistence_corridor_masks_v1()
    )
    torch.testing.assert_close(
        persistence.station_logits,
        torch.full((1, 9, 11), -math.log(2.0)),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        persistence.station_probabilities,
        torch.full((1, 9, 11), 1.0 / 3.0),
        rtol=1e-6,
        atol=1e-6,
    )


def _manual_ranking_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    predicted = probabilities.cumprod(-1).sum(-1) / 11
    target = labels.cumprod(-1).sum(-1) / 11
    row_means = []
    pair_count = 0
    for row in range(probabilities.shape[0]):
        pairs = []
        for better in api.NON_HOLD_ACTION_INDICES:
            for worse in api.NON_HOLD_ACTION_INDICES:
                if target[row, better] > target[row, worse]:
                    pairs.append(
                        F.softplus(
                            -8.0 * (predicted[row, better] - predicted[row, worse])
                        )
                        / math.log(2.0)
                    )
        if pairs:
            pair_count += len(pairs)
            row_means.append(torch.stack(pairs).mean())
    return torch.stack(row_means).mean(), len(row_means), pair_count


def test_q_and_r_use_equal_cells_pairs_then_eligible_rows_only() -> None:
    station_logits = torch.zeros((3, 9, 11), requires_grad=True)
    labels = torch.zeros((3, 9, 11))
    labels[1, 0] = 1.0
    labels[2, 1] = 1.0
    labels[2, 2, :5] = 1.0

    q = api.corridor_binary_loss_v1(station_logits, labels)
    assert q.item() == pytest.approx(1.0)
    probabilities = torch.sigmoid(station_logits)
    ranking = api.prefix_ranking_loss_v1(probabilities, labels)
    manual, eligible_rows, eligible_pairs = _manual_ranking_loss(
        probabilities, labels
    )
    torch.testing.assert_close(ranking.loss, manual)
    assert int(ranking.eligible_row_count) == eligible_rows == 2
    assert int(ranking.eligible_pair_count) == eligible_pairs
    ranking.loss.backward()
    assert station_logits.grad is not None
    assert torch.equal(station_logits.grad[0], torch.zeros_like(station_logits.grad[0]))
    assert float(station_logits.grad[1:].abs().sum()) > 0.0

    no_pairs = torch.full((2, 9, 11), 0.5, requires_grad=True)
    zero = api.prefix_ranking_loss_v1(no_pairs, torch.ones_like(no_pairs))
    assert zero.loss.item() == 0.0
    assert zero.loss.requires_grad
    zero.loss.backward()
    assert no_pairs.grad is not None and torch.equal(
        no_pairs.grad, torch.zeros_like(no_pairs.grad)
    )


def test_exact_microbatch_p_is_detached_from_both_ema_latents() -> None:
    generator = torch.Generator().manual_seed(20260728)
    predicted = torch.randn(
        4, 9, 64, 2, 3, generator=generator, requires_grad=True
    )
    ema_current = torch.randn(
        4, 64, 2, 3, generator=generator, requires_grad=True
    )
    ema_next = torch.randn(
        4, 64, 2, 3, generator=generator, requires_grad=True
    )
    actions = torch.tensor((0, 3, 6, 8))
    terms = api.microbatch_persistence_loss_v1(
        predicted, actions, ema_current, ema_next
    )
    rows = torch.arange(4)
    manual_executed = latent_energy_per_row(
        predicted[rows, actions], ema_next.detach()
    )
    manual_persistence = latent_energy_per_row(
        ema_current.detach(), ema_next.detach()
    )
    manual_baseline = manual_persistence.mean().detach().clamp_min(1e-6)
    torch.testing.assert_close(terms.executed_energy_per_row, manual_executed)
    torch.testing.assert_close(terms.persistence_energy_per_row, manual_persistence)
    torch.testing.assert_close(terms.baseline, manual_baseline)
    torch.testing.assert_close(terms.loss, manual_executed.mean() / manual_baseline)
    assert not terms.baseline.requires_grad
    terms.loss.backward()
    assert predicted.grad is not None and float(predicted.grad.abs().sum()) > 0.0
    assert ema_current.grad is None
    assert ema_next.grad is None
    for row, action in enumerate(actions):
        assert float(predicted.grad[row, action].abs().sum()) > 0.0
        assert torch.equal(
            predicted.grad[row, torch.arange(9) != action],
            torch.zeros_like(predicted.grad[row, torch.arange(9) != action]),
        )


def test_semantic_and_composite_loss_are_exact_s_plus_p_plus_q_plus_r() -> None:
    current_logits = torch.zeros((2, 3, 2, 2), requires_grad=True)
    next_logits = torch.zeros((2, 3, 2, 2), requires_grad=True)
    labels = torch.tensor(
        (((0, 1), (2, 0)), ((2, 1), (0, 2))), dtype=torch.long
    )
    semantic = api.semantic_loss_v1(
        current_logits, labels, next_logits, labels
    )
    assert semantic.loss.item() == pytest.approx(1.0)

    generator = torch.Generator().manual_seed(71)
    predicted = torch.randn(
        4, 9, 64, 2, 2, generator=generator, requires_grad=True
    )
    ema_current = torch.randn(4, 64, 2, 2, generator=generator)
    ema_next = torch.randn(4, 64, 2, 2, generator=generator)
    station_logits = torch.randn(
        4, 9, 11, generator=generator, requires_grad=True
    )
    station_safe = torch.zeros((4, 9, 11))
    station_safe[:, 0] = 1.0
    terms = api.joint_microbatch_loss_v1(
        semantic_loss=semantic.loss,
        predicted_latents=predicted,
        executed_action_indices=torch.tensor((0, 1, 2, 3)),
        ema_current_latent=ema_current,
        ema_next_latent=ema_next,
        station_logits=station_logits,
        station_safe=station_safe,
    )
    torch.testing.assert_close(
        terms.loss,
        terms.semantic
        + terms.persistence
        + terms.corridor_binary
        + terms.prefix_ranking,
    )
    terms.loss.backward()
    assert current_logits.grad is not None and float(current_logits.grad.abs().sum()) > 0
    assert next_logits.grad is not None and float(next_logits.grad.abs().sum()) > 0
    assert predicted.grad is not None and float(predicted.grad.abs().sum()) > 0
    assert station_logits.grad is not None and float(station_logits.grad.abs().sum()) > 0


def test_representative_invalid_shapes_fail_closed() -> None:
    with pytest.raises(ValueError, match="predicted latents"):
        api.decode_all_action_semantic_logits_v1(
            _ParameterFreeModelStub(), torch.zeros((1, 8, 64, 64, 64))
        )
    with pytest.raises(ValueError, match="nonempty"):
        api.smooth_min_station_logits_v1(
            torch.zeros((1, 9, 2, 2)), torch.zeros((11, 2, 2), dtype=torch.uint8)
        )
    with pytest.raises(ValueError, match="zero or one"):
        api.corridor_binary_loss_v1(
            torch.zeros((1, 9, 11)), torch.full((1, 9, 11), 0.5)
        )
