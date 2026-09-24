"""Pure hierarchical first-hit loss for Camera-ray N5 V9.

The loss consumes only the normalized ordered first-hit distribution and the
existing V4 ray targets. It performs no I/O, model construction, thresholding,
calibration, rasterization, or metric computation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .observable_camera_ray_evidence_v4 import (
    ordered_obstacle_first_hit_log_probabilities_v4,
)
from .observable_camera_ray_evidence_v4_training import (
    ObservableCameraRayEvidenceV4Targets,
)


@dataclass(frozen=True)
class HierarchicalFirstHitNLLBreakdownV9:
    """Balanced presence and conditional-depth negative log likelihoods."""

    total: torch.Tensor
    presence_nll: torch.Tensor
    conditional_depth_nll: torch.Tensor
    no_hit_presence_nll: torch.Tensor
    hit_presence_nll: torch.Tensor
    conditional_depth_bin_nll: tuple[torch.Tensor, ...]
    no_hit_count: int
    hit_count: int
    hit_distance_bin_counts: tuple[int, ...]
    nonempty_presence_group_count: int
    nonempty_conditional_depth_group_count: int


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} became non-finite")


def _validate_log_probability_inputs(
    *,
    hit_log_probabilities: torch.Tensor,
    no_hit_log_probability: torch.Tensor,
    pixel_in_range_hit_mask: torch.Tensor,
    pixel_no_hit_mask: torch.Tensor,
    pixel_hit_bin_index: torch.Tensor,
) -> None:
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            hit_log_probabilities,
            no_hit_log_probability,
            pixel_in_range_hit_mask,
            pixel_no_hit_mask,
            pixel_hit_bin_index,
        )
    ):
        raise TypeError("V9 hierarchical first-hit inputs must be tensors")
    hit = hit_log_probabilities
    if hit.ndim != 4 or hit.shape[1] <= 0 or not hit.is_floating_point():
        raise ValueError("hit log probabilities must have shape (B,D,H,W)")
    expected = (hit.shape[0], hit.shape[2], hit.shape[3])
    if (
        tuple(no_hit_log_probability.shape) != expected
        or not no_hit_log_probability.is_floating_point()
        or tuple(pixel_in_range_hit_mask.shape) != expected
        or tuple(pixel_no_hit_mask.shape) != expected
        or tuple(pixel_hit_bin_index.shape) != expected
    ):
        raise ValueError("V9 first-hit probabilities and targets do not align")
    if (
        pixel_in_range_hit_mask.dtype != torch.bool
        or pixel_no_hit_mask.dtype != torch.bool
        or pixel_hit_bin_index.dtype != torch.long
    ):
        raise ValueError("V9 first-hit target dtypes changed")
    device = hit.device
    if any(
        value.device != device
        for value in (
            no_hit_log_probability,
            pixel_in_range_hit_mask,
            pixel_no_hit_mask,
            pixel_hit_bin_index,
        )
    ):
        raise ValueError("V9 first-hit probabilities and targets must share a device")
    if not torch.equal(pixel_no_hit_mask, ~pixel_in_range_hit_mask):
        raise ValueError("V9 hit and no-hit targets must partition every ray")
    if bool(
        (
            pixel_in_range_hit_mask
            & (
                (pixel_hit_bin_index < 0)
                | (pixel_hit_bin_index >= hit.shape[1])
            )
        )
        .any()
        .item()
    ):
        raise ValueError("V9 target hit bin exceeds the ordered distribution")
    _require_finite(hit, name="V9 hit log probability")
    _require_finite(no_hit_log_probability, name="V9 no-hit log probability")
    if bool((hit > 1e-6).any().item()) or bool(
        (no_hit_log_probability > 1e-6).any().item()
    ):
        raise ValueError("V9 log probabilities exceed zero")
    normalizer = torch.logsumexp(
        torch.cat((hit, no_hit_log_probability[:, None]), dim=1),
        dim=1,
    )
    tolerance = max(
        1e-6,
        128.0 * float(torch.finfo(hit.dtype).eps) * float(hit.shape[1] + 1),
    )
    if not bool(
        torch.allclose(
            normalizer,
            torch.zeros_like(normalizer),
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError("V9 ordered first-hit probabilities are not normalized")


def hierarchical_first_hit_nll_from_log_probabilities_v9(
    *,
    hit_log_probabilities: torch.Tensor,
    no_hit_log_probability: torch.Tensor,
    pixel_in_range_hit_mask: torch.Tensor,
    pixel_no_hit_mask: torch.Tensor,
    pixel_hit_bin_index: torch.Tensor,
) -> HierarchicalFirstHitNLLBreakdownV9:
    """Compute the V9 objective from one normalized ordered distribution.

    Presence gives equal influence to the nonempty target no-hit and target-hit
    groups. Conditional depth gives equal influence to every represented target
    depth bin after removing the common hit-presence mass.
    """

    _validate_log_probability_inputs(
        hit_log_probabilities=hit_log_probabilities,
        no_hit_log_probability=no_hit_log_probability,
        pixel_in_range_hit_mask=pixel_in_range_hit_mask,
        pixel_no_hit_mask=pixel_no_hit_mask,
        pixel_hit_bin_index=pixel_hit_bin_index,
    )
    hit_log = hit_log_probabilities
    zero = hit_log.sum() * 0.0
    hit_mask = pixel_in_range_hit_mask
    no_hit_mask = pixel_no_hit_mask
    no_hit_count = int(no_hit_mask.sum().item())
    hit_count = int(hit_mask.sum().item())

    log_hit_mass = torch.logsumexp(hit_log, dim=1)
    _require_finite(log_hit_mass, name="V9 hit-presence log probability")

    no_hit_presence = (
        -no_hit_log_probability[no_hit_mask].mean() if no_hit_count else zero
    )
    hit_presence = -log_hit_mass[hit_mask].mean() if hit_count else zero
    presence_groups = []
    if no_hit_count:
        presence_groups.append(no_hit_presence)
    if hit_count:
        presence_groups.append(hit_presence)
    presence = (
        torch.stack(presence_groups).mean() if presence_groups else zero
    )

    conditional_log = hit_log - log_hit_mass[:, None]
    _require_finite(conditional_log, name="V9 conditional-depth log probability")
    selected_conditional = conditional_log.gather(
        1,
        pixel_hit_bin_index[:, None],
    ).squeeze(1)
    bin_counts: list[int] = []
    bin_losses: list[torch.Tensor] = []
    represented_losses: list[torch.Tensor] = []
    for depth_bin in range(hit_log.shape[1]):
        mask = hit_mask & (pixel_hit_bin_index == depth_bin)
        count = int(mask.sum().item())
        bin_counts.append(count)
        loss = -selected_conditional[mask].mean() if count else zero
        bin_losses.append(loss)
        if count:
            represented_losses.append(loss)
    conditional_depth = (
        torch.stack(represented_losses).mean() if represented_losses else zero
    )
    total = 0.5 * presence + 0.5 * conditional_depth
    for name, value in (
        ("presence NLL", presence),
        ("conditional-depth NLL", conditional_depth),
        ("hierarchical first-hit NLL", total),
    ):
        _require_finite(value, name=f"V9 {name}")
    if any(
        value.detach().item() < -max(1e-7, 32.0 * torch.finfo(value.dtype).eps)
        for value in (presence, conditional_depth, total)
    ):
        raise FloatingPointError("V9 hierarchical first-hit NLL became negative")
    return HierarchicalFirstHitNLLBreakdownV9(
        total=total,
        presence_nll=presence,
        conditional_depth_nll=conditional_depth,
        no_hit_presence_nll=no_hit_presence,
        hit_presence_nll=hit_presence,
        conditional_depth_bin_nll=tuple(bin_losses),
        no_hit_count=no_hit_count,
        hit_count=hit_count,
        hit_distance_bin_counts=tuple(bin_counts),
        nonempty_presence_group_count=len(presence_groups),
        nonempty_conditional_depth_group_count=len(represented_losses),
    )


def hierarchical_first_hit_nll_breakdown_v9(
    hazard_logits: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
) -> HierarchicalFirstHitNLLBreakdownV9:
    """Compute the V9 objective from the existing ordered hazard output."""

    if not isinstance(targets, ObservableCameraRayEvidenceV4Targets):
        raise TypeError("targets must be ObservableCameraRayEvidenceV4Targets")
    probabilities = ordered_obstacle_first_hit_log_probabilities_v4(hazard_logits)
    return hierarchical_first_hit_nll_from_log_probabilities_v9(
        hit_log_probabilities=probabilities.hit,
        no_hit_log_probability=probabilities.no_hit,
        pixel_in_range_hit_mask=targets.pixel_in_range_hit_mask,
        pixel_no_hit_mask=targets.pixel_no_hit_mask,
        pixel_hit_bin_index=targets.pixel_hit_bin_index,
    )


def hierarchical_first_hit_nll_v9(
    hazard_logits: torch.Tensor,
    targets: ObservableCameraRayEvidenceV4Targets,
) -> torch.Tensor:
    """Return only the scalar V9 hierarchical first-hit objective."""

    return hierarchical_first_hit_nll_breakdown_v9(hazard_logits, targets).total


__all__ = [
    "HierarchicalFirstHitNLLBreakdownV9",
    "hierarchical_first_hit_nll_breakdown_v9",
    "hierarchical_first_hit_nll_from_log_probabilities_v9",
    "hierarchical_first_hit_nll_v9",
]
