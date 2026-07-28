"""Differentiable swept-progress survival losses for the joint JEPA probe.

This module is deliberately limited to tensor scoring and loss composition.  It
opens no data or runtime artifacts and defines no model, runner, or parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
import torch.nn.functional as F

from lewm.benchmarks import (
    go2_post_action_projective_support_joint_jepa_v1 as _projective,
)


ACTION_COUNT: Final = 9
ACTION_ORDER: Final = _projective.ACTION_ORDER
HOLD_ACTION_INDEX: Final = 6
NON_HOLD_ACTION_INDICES: Final = tuple(
    index for index in range(ACTION_COUNT) if index != HOLD_ACTION_INDEX
)
CONTINUATION_COUNT: Final = 15
SURVIVAL_LOGIT_COUNT: Final = 1 + CONTINUATION_COUNT
PROGRESS_SEGMENT_M: Final = 0.1
LOSS_NORMALIZATION: Final = math.log(2.0)
RANKING_TEMPERATURE: Final = 8.0
BEV_HEIGHT: Final = _projective.BEV_HEIGHT
BEV_WIDTH: Final = _projective.BEV_WIDTH


@dataclass(frozen=True)
class SurvivalScoreTermsV1:
    """Immediate, conditional, cumulative-survival, and progress scores."""

    immediate_probability: torch.Tensor
    conditional_probabilities: torch.Tensor
    survival_probabilities: torch.Tensor
    expected_progress_m: torch.Tensor


@dataclass(frozen=True)
class AtRiskSurvivalLossTermsV1:
    """Pooled BCE over the initial event and exactly the at-risk continuations."""

    loss: torch.Tensor
    immediate_loss: torch.Tensor
    continuation_loss: torch.Tensor
    continuation_at_risk: torch.Tensor
    continuation_targets: torch.Tensor
    supervised_decision_count: torch.Tensor


@dataclass(frozen=True)
class PrefixRankingLossTermsV1:
    """Pair-normalized ranking loss over unequal non-HOLD target prefixes."""

    loss: torch.Tensor
    eligible_pair_count: torch.Tensor


@dataclass(frozen=True)
class JointSurvivalLossTermsV1:
    """Unweighted joint loss and its differentiable component tensors."""

    loss: torch.Tensor
    semantic: torch.Tensor
    executed_action_ema_latent: torch.Tensor
    survival: torch.Tensor
    progress_ranking: torch.Tensor
    scores: SurvivalScoreTermsV1
    survival_terms: AtRiskSurvivalLossTermsV1
    ranking_terms: PrefixRankingLossTermsV1


def _validate_survival_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 3 or logits.shape[1:] != (
        ACTION_COUNT,
        SURVIVAL_LOGIT_COUNT,
    ) or logits.shape[0] < 1:
        raise ValueError("survival logits must have shape (B,9,16) with B >= 1")
    if not logits.is_floating_point():
        raise TypeError("survival logits must use a floating dtype")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("survival logits must be finite")


def _validate_targets(
    logits: torch.Tensor,
    immediate_feasible: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> None:
    expected = logits.shape[:2]
    if immediate_feasible.shape != expected:
        raise ValueError("immediate-feasible labels must have shape (B,9)")
    if immediate_feasible.dtype != torch.bool:
        raise TypeError("immediate-feasible labels must use bool dtype")
    if prefix_lengths.shape != expected:
        raise ValueError("prefix lengths must have shape (B,9)")
    if (
        prefix_lengths.dtype == torch.bool
        or prefix_lengths.is_floating_point()
        or prefix_lengths.is_complex()
    ):
        raise TypeError("prefix lengths must use an integer dtype")
    if (
        immediate_feasible.device != logits.device
        or prefix_lengths.device != logits.device
    ):
        raise ValueError("survival logits and targets must share a device")
    if bool(((prefix_lengths < 0) | (prefix_lengths > CONTINUATION_COUNT)).any()):
        raise ValueError("prefix lengths must be in the closed range 0 through 15")
    if bool(((~immediate_feasible) & (prefix_lengths != 0)).any()):
        raise ValueError(
            "an infeasible immediate primitive must have prefix length zero"
        )


def _validate_scalar_loss(name: str, value: torch.Tensor, logits: torch.Tensor) -> None:
    if value.ndim != 0 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating scalar")
    if value.dtype != logits.dtype or value.device != logits.device:
        raise ValueError(f"{name} must share the survival-logit dtype and device")
    if not bool(torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")


def survival_scores_v1(logits: torch.Tensor) -> SurvivalScoreTermsV1:
    """Convert one immediate and 15 conditional logits to monotone survival."""

    _validate_survival_logits(logits)
    probabilities = torch.sigmoid(logits)
    immediate = probabilities[..., 0]
    conditional = probabilities[..., 1:]
    survival = immediate.unsqueeze(-1) * conditional.cumprod(dim=-1)
    return SurvivalScoreTermsV1(
        immediate_probability=immediate,
        conditional_probabilities=conditional,
        survival_probabilities=survival,
        expected_progress_m=PROGRESS_SEGMENT_M * survival.sum(dim=-1),
    )


def _validate_swept_progress_masks_v1(masks: torch.Tensor) -> None:
    expected = (ACTION_COUNT, SURVIVAL_LOGIT_COUNT, BEV_HEIGHT, BEV_WIDTH)
    if masks.shape != expected:
        raise ValueError("swept-progress masks must have shape (9,16,64,64)")
    if masks.dtype != torch.bool or masks.device.type != "cpu":
        raise TypeError("swept-progress masks must be CPU bool")
    if not masks.is_contiguous():
        raise ValueError("swept-progress masks must be C-contiguous")
    if not bool(((masks == 0) | (masks == 1)).all()):
        raise ValueError("swept-progress masks must be binary")
    if not bool(masks.flatten(start_dim=2).any(dim=-1).all()):
        raise ValueError("every action/progress mask must be nonempty")


def _continuation_segment_poses_v1(
    footprint: _projective.DirectionalSupportFootprint,
    start: _projective.Pose2D,
    end: _projective.Pose2D,
) -> tuple[_projective.Pose2D, ...]:
    """Apply the reviewed sweep interpolation rule to one continuation bin."""

    delta_x = end.x_m - start.x_m
    delta_y = end.y_m - start.y_m
    delta_yaw = _projective.wrap_angle_pi(end.yaw_rad - start.yaw_rad)
    corner_motion_upper_bound_m = (
        math.hypot(delta_x, delta_y)
        + footprint.maximum_vertex_radius_m * abs(delta_yaw)
    )
    interval_count = max(
        1,
        int(math.ceil(
            corner_motion_upper_bound_m
            / _projective.IMMEDIATE_MAXIMUM_CORNER_STEP_M
        )),
        int(math.ceil(
            abs(delta_yaw) / _projective.IMMEDIATE_MAXIMUM_YAW_STEP_RAD
        )),
    )
    return tuple(
        _projective.Pose2D(
            start.x_m + fraction * delta_x,
            start.y_m + fraction * delta_y,
            _projective.wrap_angle_pi(start.yaw_rad + fraction * delta_yaw),
        )
        for index in range(interval_count + 1)
        for fraction in (index / interval_count,)
    )


def build_swept_progress_masks_v1() -> torch.Tensor:
    """Build fixed current-BEV masks ``[action, immediate+15 segments, H, W]``."""

    footprint = _projective.DirectionalSupportFootprint(
        vertices_xy_m=_projective._FOOTPRINT_VERTICES_XY_M
    )
    forward_centers, left_centers = _projective._lattice_centers()
    action_masks: list[torch.Tensor] = []
    for action in ACTION_ORDER:
        immediate_poses = _projective._interpolated_action_sweep_v1(
            footprint, action
        )
        bins = [_projective._rasterize_polygon_union(
            tuple(footprint.vertices_at(pose) for pose in immediate_poses),
            forward_centers=forward_centers,
            left_centers=left_centers,
        ).bool()]
        endpoint = _projective._integrated_action_endpoint(action)
        cos_yaw = math.cos(endpoint.yaw_rad)
        sin_yaw = math.sin(endpoint.yaw_rad)
        for segment_index in range(CONTINUATION_COUNT):
            start_distance = segment_index * PROGRESS_SEGMENT_M
            end_distance = (segment_index + 1) * PROGRESS_SEGMENT_M
            start = _projective.Pose2D(
                endpoint.x_m + start_distance * cos_yaw,
                endpoint.y_m + start_distance * sin_yaw,
                endpoint.yaw_rad,
            )
            end = _projective.Pose2D(
                endpoint.x_m + end_distance * cos_yaw,
                endpoint.y_m + end_distance * sin_yaw,
                endpoint.yaw_rad,
            )
            samples = _continuation_segment_poses_v1(footprint, start, end)
            bins.append(_projective._rasterize_polygon_union(
                tuple(footprint.vertices_at(pose) for pose in samples),
                forward_centers=forward_centers,
                left_centers=left_centers,
            ).bool())
        action_masks.append(torch.stack(bins))
    masks = torch.stack(action_masks).contiguous()
    _validate_swept_progress_masks_v1(masks)
    return masks


def at_risk_survival_bce_loss_v1(
    logits: torch.Tensor,
    immediate_feasible: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> AtRiskSurvivalLossTermsV1:
    """Supervise each conditional only when its preceding path survived.

    Conditional index ``j`` is at risk exactly when the immediate primitive is
    feasible and ``prefix_length >= j``.  Its target is
    ``prefix_length >= j + 1``, so the first failed segment is supervised while
    every segment after it is ignored.
    """

    _validate_survival_logits(logits)
    _validate_targets(logits, immediate_feasible, prefix_lengths)
    target_dtype = logits.dtype
    immediate_targets = immediate_feasible.to(dtype=target_dtype)
    immediate_elements = F.binary_cross_entropy_with_logits(
        logits[..., 0], immediate_targets, reduction="none"
    )

    segment_indices = torch.arange(
        CONTINUATION_COUNT, device=logits.device, dtype=prefix_lengths.dtype
    )
    continuation_at_risk = immediate_feasible.unsqueeze(-1) & (
        prefix_lengths.unsqueeze(-1) >= segment_indices
    )
    continuation_targets = (
        prefix_lengths.unsqueeze(-1) >= segment_indices + 1
    )
    continuation_elements = F.binary_cross_entropy_with_logits(
        logits[..., 1:], continuation_targets.to(dtype=target_dtype), reduction="none"
    )
    risk_weights = continuation_at_risk.to(dtype=target_dtype)
    risk_count = continuation_at_risk.sum()
    continuation_sum = (continuation_elements * risk_weights).sum()
    continuation_loss = (
        continuation_sum
        / risk_count.clamp_min(1).to(target_dtype)
        / LOSS_NORMALIZATION
    )
    if not bool(risk_count > 0):
        continuation_loss = 0.0 * logits[..., 1:].sum()

    supervised_count = risk_count + immediate_feasible.numel()
    loss = (
        (immediate_elements.sum() + continuation_sum)
        / supervised_count.to(dtype=target_dtype)
        / LOSS_NORMALIZATION
    )
    return AtRiskSurvivalLossTermsV1(
        loss=loss,
        immediate_loss=immediate_elements.mean() / LOSS_NORMALIZATION,
        continuation_loss=continuation_loss,
        continuation_at_risk=continuation_at_risk,
        continuation_targets=continuation_targets,
        supervised_decision_count=supervised_count.detach(),
    )


def prefix_ranking_loss_v1(
    expected_progress_m: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> PrefixRankingLossTermsV1:
    """Rank every better/worse non-HOLD pair and normalize over those pairs."""

    if expected_progress_m.ndim != 2 or expected_progress_m.shape[1:] != (
        ACTION_COUNT,
    ) or expected_progress_m.shape[0] < 1:
        raise ValueError("expected progress must have shape (B,9) with B >= 1")
    if not expected_progress_m.is_floating_point():
        raise TypeError("expected progress must use a floating dtype")
    if not bool(torch.isfinite(expected_progress_m).all()):
        raise ValueError("expected progress must be finite")
    if prefix_lengths.shape != expected_progress_m.shape:
        raise ValueError("prefix lengths must have shape (B,9)")
    if (
        prefix_lengths.dtype == torch.bool
        or prefix_lengths.is_floating_point()
        or prefix_lengths.is_complex()
    ):
        raise TypeError("prefix lengths must use an integer dtype")
    if prefix_lengths.device != expected_progress_m.device:
        raise ValueError("expected progress and prefix lengths must share a device")
    if bool(((prefix_lengths < 0) | (prefix_lengths > CONTINUATION_COUNT)).any()):
        raise ValueError("prefix lengths must be in the closed range 0 through 15")

    indices = torch.tensor(
        NON_HOLD_ACTION_INDICES,
        dtype=torch.long,
        device=expected_progress_m.device,
    )
    predicted = expected_progress_m.index_select(1, indices)
    target = prefix_lengths.index_select(1, indices)
    eligible = target[:, :, None] > target[:, None, :]
    margins = predicted[:, :, None] - predicted[:, None, :]
    pair_losses = (
        F.softplus(-RANKING_TEMPERATURE * margins) / LOSS_NORMALIZATION
    )
    pair_count = eligible.sum()
    if bool(pair_count > 0):
        loss = (
            pair_losses * eligible.to(dtype=pair_losses.dtype)
        ).sum() / pair_count.to(dtype=pair_losses.dtype)
    else:
        loss = 0.0 * expected_progress_m.sum()
    return PrefixRankingLossTermsV1(
        loss=loss,
        eligible_pair_count=pair_count.detach(),
    )


def joint_survival_loss_v1(
    *,
    semantic_loss: torch.Tensor,
    executed_action_ema_latent_loss: torch.Tensor,
    survival_logits: torch.Tensor,
    immediate_feasible: torch.Tensor,
    prefix_lengths: torch.Tensor,
) -> JointSurvivalLossTermsV1:
    """Compose ``semantic + executed-EMA + survival + ranking`` without detach."""

    _validate_survival_logits(survival_logits)
    _validate_scalar_loss("semantic loss", semantic_loss, survival_logits)
    _validate_scalar_loss(
        "executed-action EMA latent loss",
        executed_action_ema_latent_loss,
        survival_logits,
    )
    scores = survival_scores_v1(survival_logits)
    survival = at_risk_survival_bce_loss_v1(
        survival_logits, immediate_feasible, prefix_lengths
    )
    ranking = prefix_ranking_loss_v1(scores.expected_progress_m, prefix_lengths)
    total = (
        semantic_loss
        + executed_action_ema_latent_loss
        + survival.loss
        + ranking.loss
    )
    return JointSurvivalLossTermsV1(
        loss=total,
        semantic=semantic_loss,
        executed_action_ema_latent=executed_action_ema_latent_loss,
        survival=survival.loss,
        progress_ranking=ranking.loss,
        scores=scores,
        survival_terms=survival,
        ranking_terms=ranking,
    )


__all__ = [
    "ACTION_COUNT",
    "ACTION_ORDER",
    "AtRiskSurvivalLossTermsV1",
    "BEV_HEIGHT",
    "BEV_WIDTH",
    "CONTINUATION_COUNT",
    "HOLD_ACTION_INDEX",
    "JointSurvivalLossTermsV1",
    "LOSS_NORMALIZATION",
    "NON_HOLD_ACTION_INDICES",
    "PROGRESS_SEGMENT_M",
    "PrefixRankingLossTermsV1",
    "RANKING_TEMPERATURE",
    "SURVIVAL_LOGIT_COUNT",
    "SurvivalScoreTermsV1",
    "at_risk_survival_bce_loss_v1",
    "build_swept_progress_masks_v1",
    "joint_survival_loss_v1",
    "prefix_ranking_loss_v1",
    "survival_scores_v1",
]
