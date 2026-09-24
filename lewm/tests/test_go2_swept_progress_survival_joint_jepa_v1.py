from __future__ import annotations

import hashlib
import math

import pytest
import torch

from lewm.benchmarks import go2_swept_progress_survival_joint_jepa_v1 as api
from lewm.benchmarks import (
    go2_post_action_projective_support_joint_jepa_v1 as projective,
)


def test_at_risk_mask_includes_failure_and_excludes_every_later_segment() -> None:
    logits = torch.zeros((1, 9, 16), requires_grad=True)
    immediate = torch.ones((1, 9), dtype=torch.bool)
    immediate[0, 1] = False
    prefixes = torch.tensor(((2, 0, 15, 0, 1, 3, 4, 5, 6),))

    terms = api.at_risk_survival_bce_loss_v1(logits, immediate, prefixes)
    assert terms.continuation_at_risk[0, 0].tolist() == [
        True,
        True,
        True,
        *([False] * 12),
    ]
    assert terms.continuation_targets[0, 0].tolist() == [
        True,
        True,
        *([False] * 13),
    ]
    assert not bool(terms.continuation_at_risk[0, 1].any())
    expected_count = 9 + sum(
        min(prefixes[0, index].item() + 1, 15)
        for index in range(9)
        if immediate[0, index]
    )
    assert int(terms.supervised_decision_count) == expected_count
    assert api.LOSS_NORMALIZATION == pytest.approx(math.log(2.0))
    assert terms.immediate_loss.item() == pytest.approx(1.0)
    assert terms.continuation_loss.item() == pytest.approx(1.0)
    assert terms.loss.item() == pytest.approx(1.0)

    terms.loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[..., 0].abs().sum()) > 0.0
    assert float(logits.grad[0, 0, 1:4].abs().sum()) > 0.0
    assert torch.equal(logits.grad[0, 0, 4:], torch.zeros_like(logits.grad[0, 0, 4:]))
    assert torch.equal(logits.grad[0, 1, 1:], torch.zeros_like(logits.grad[0, 1, 1:]))


def test_survival_is_monotone_and_expected_progress_is_point_one_sum() -> None:
    logits = torch.tensor(
        [[[1.0, 2.0, -1.0, 0.5, *([0.25] * 12)]] * 9],
        requires_grad=True,
    )
    terms = api.survival_scores_v1(logits)
    assert terms.survival_probabilities.shape == (1, 9, 15)
    assert bool(
        (
            terms.survival_probabilities[..., 1:]
            <= terms.survival_probabilities[..., :-1]
        ).all()
    )
    torch.testing.assert_close(
        terms.expected_progress_m,
        0.1 * terms.survival_probabilities.sum(dim=-1),
    )

    certain = api.survival_scores_v1(torch.full((2, 9, 16), 30.0))
    torch.testing.assert_close(
        certain.expected_progress_m,
        torch.full((2, 9), 1.5),
        atol=1e-5,
        rtol=0.0,
    )


def test_swept_progress_masks_preserve_reviewed_geometry_and_identity() -> None:
    masks = api.build_swept_progress_masks_v1()
    assert masks.shape == (9, 16, 64, 64)
    assert masks.dtype == torch.bool
    assert masks.device.type == "cpu"
    assert masks.is_contiguous()
    assert bool(masks.flatten(start_dim=2).any(dim=-1).all())
    assert tuple(int(mask.sum()) for mask in masks[:, 0]) == (
        59,
        56,
        57,
        58,
        57,
        57,
        51,
        54,
        52,
    )
    assert torch.equal(
        masks[:, 1:], masks[0, 1:].unsqueeze(0).expand_as(masks[:, 1:])
    )
    assert tuple(int(mask.sum()) for mask in masks[0, 1:]) == (57,) * 15
    for action in api.ACTION_ORDER:
        endpoint = projective._integrated_action_endpoint(action)
        relative = api._pose_in_endpoint_frame_v1(endpoint, endpoint)
        assert relative.x_m == pytest.approx(0.0)
        assert relative.y_m == pytest.approx(0.0)
        assert relative.yaw_rad == pytest.approx(0.0)
    identity = hashlib.sha256(
        bytes(masks.to(torch.uint8).reshape(-1).tolist())
    ).hexdigest()
    assert identity == (
        "11ae5e26b182da85c8a7ca866ee4914c72b5b84b8b601dd807903097d754485c"
    )

    current_masks = api.build_current_frame_swept_progress_masks_v1()
    assert current_masks.shape == (9, 16, 64, 64)
    assert current_masks.dtype == torch.bool
    assert current_masks.device.type == "cpu"
    assert current_masks.is_contiguous()
    assert bool(current_masks.flatten(start_dim=2).any(dim=-1).all())
    assert not torch.equal(current_masks[0, 1:], current_masks[1, 1:])
    assert torch.equal(
        masks[:, 1:], masks[0, 1:].unsqueeze(0).expand_as(masks[:, 1:])
    )
    current_identity = hashlib.sha256(
        bytes(current_masks.to(torch.uint8).reshape(-1).tolist())
    ).hexdigest()
    assert current_identity == (
        "c4b8c475032433e448cd7df9decfead2c0800426219098f45306a0540154d2ff"
    )


def test_ranking_uses_all_better_non_hold_pairs_and_gradients_point_correctly() -> None:
    prefixes = torch.full((1, 9), 5, dtype=torch.long)
    prefixes[0, 0] = 15
    prefixes[0, api.HOLD_ACTION_INDEX] = 14
    correct_scores = torch.full((1, 9), 0.5)
    correct_scores[0, 0] = 1.5
    reversed_scores = correct_scores.clone()
    reversed_scores[0, 0] = 0.0

    correct = api.prefix_ranking_loss_v1(correct_scores, prefixes)
    reversed_scores.requires_grad_()
    reversed_terms = api.prefix_ranking_loss_v1(reversed_scores, prefixes)
    assert int(correct.eligible_pair_count) == 7
    assert api.RANKING_TEMPERATURE == 8.0
    assert correct.loss.item() == pytest.approx(
        math.log1p(math.exp(-8.0)) / math.log(2.0)
    )
    assert reversed_terms.loss > correct.loss

    one_bin_scores = torch.full((1, 9), 0.5)
    one_bin_scores[0, 0] += 0.1
    one_bin = api.prefix_ranking_loss_v1(one_bin_scores, prefixes)
    assert one_bin.loss.item() == pytest.approx(
        math.log1p(math.exp(-0.8)) / math.log(2.0)
    )

    reversed_terms.loss.backward()
    assert reversed_scores.grad is not None
    assert reversed_scores.grad[0, 0] < 0.0
    assert reversed_scores.grad[0, 1] > 0.0
    assert reversed_scores.grad[0, api.HOLD_ACTION_INDEX] == 0.0


def test_all_equal_prefixes_have_differentiable_zero_ranking() -> None:
    scores = torch.randn((2, 9), requires_grad=True)
    terms = api.prefix_ranking_loss_v1(
        scores, torch.full((2, 9), 7, dtype=torch.long)
    )
    assert int(terms.eligible_pair_count) == 0
    assert terms.loss.item() == 0.0
    assert terms.loss.requires_grad
    terms.loss.backward()
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores.grad))


def test_joint_compositor_preserves_every_input_gradient_without_detach() -> None:
    generator = torch.Generator().manual_seed(20260728)
    logits = torch.randn((2, 9, 16), generator=generator, requires_grad=True)
    semantic = torch.tensor(1.25, requires_grad=True)
    executed_ema = torch.tensor(0.75, requires_grad=True)
    immediate = torch.ones((2, 9), dtype=torch.bool)
    prefixes = torch.tensor(
        (
            (15, 12, 9, 6, 3, 1, 0, 4, 8),
            (0, 1, 2, 3, 4, 5, 6, 7, 8),
        )
    )
    terms = api.joint_survival_loss_v1(
        semantic_loss=semantic,
        executed_action_ema_latent_loss=executed_ema,
        survival_logits=logits,
        immediate_feasible=immediate,
        prefix_lengths=prefixes,
    )
    torch.testing.assert_close(
        terms.loss,
        terms.semantic
        + terms.executed_action_ema_latent
        + terms.survival
        + terms.progress_ranking,
    )
    terms.loss.backward()
    assert semantic.grad is not None and semantic.grad.item() == 1.0
    assert executed_ema.grad is not None and executed_ema.grad.item() == 1.0
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0


def test_representative_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="shape"):
        api.survival_scores_v1(torch.zeros((1, 9, 15)))
    with pytest.raises(ValueError, match="finite"):
        api.survival_scores_v1(torch.full((1, 9, 16), float("nan")))
    logits = torch.zeros((1, 9, 16))
    with pytest.raises(TypeError, match="bool"):
        api.at_risk_survival_bce_loss_v1(
            logits, torch.ones((1, 9)), torch.zeros((1, 9), dtype=torch.long)
        )
    with pytest.raises(ValueError, match="0 through 15"):
        api.at_risk_survival_bce_loss_v1(
            logits,
            torch.ones((1, 9), dtype=torch.bool),
            torch.full((1, 9), 16, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="infeasible immediate"):
        api.at_risk_survival_bce_loss_v1(
            logits,
            torch.zeros((1, 9), dtype=torch.bool),
            torch.ones((1, 9), dtype=torch.long),
        )
