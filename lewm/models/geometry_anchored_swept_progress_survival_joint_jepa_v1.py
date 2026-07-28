"""Geometry-anchored swept-progress survival joint JEPA V1.

The frozen RGB encoder, BEV lift, and action predictor are unchanged.  One
shared head pools each action-predicted latent over fixed, sweep-aligned masks
and emits an immediate-primitive logit followed by fifteen progress logits.
"""
from __future__ import annotations

from typing import Mapping, NamedTuple

import torch
import torch.nn as nn

from lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1 import (
    ACTION_VOCABULARY_V1,
    FREE_CLASS_V1,
    OCCUPIED_CLASS_V1,
    UNKNOWN_CLASS_V1,
    GeometryAnchoredBevSamplingV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1 as _FrozenJointJepaV1,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    final_class_macro_nll_per_row,
    latent_energy_per_row,
)


SWEEP_PROGRESS_BIN_COUNT_V1 = 16


class SweptProgressSurvivalPredictionV1(NamedTuple):
    """All-action predicted latents and their sweep-survival logits."""

    predicted_latents: torch.Tensor
    survival_logits: torch.Tensor


def _validate_sweep_masks_v1(sweep_masks: torch.Tensor) -> None:
    if not isinstance(sweep_masks, torch.Tensor):
        raise TypeError("sweep_masks must be a tensor")
    if tuple(sweep_masks.shape) != (9, SWEEP_PROGRESS_BIN_COUNT_V1, 64, 64):
        raise ValueError("sweep_masks must have shape (9,16,64,64)")
    if sweep_masks.dtype != torch.bool:
        raise TypeError("sweep_masks must use exact bool")
    if not bool(sweep_masks.flatten(start_dim=2).any(dim=2).all()):
        raise ValueError("every action-progress sweep mask must be nonempty")


class SweptProgressSurvivalHeadV1(nn.Module):
    """Shared per-mask spatial pooling followed by one shared logit map."""

    def __init__(self, sweep_masks: torch.Tensor) -> None:
        super().__init__()
        _validate_sweep_masks_v1(sweep_masks)
        self.register_buffer(
            "sweep_masks", sweep_masks.clone(), persistent=True
        )
        self.output = nn.Linear(64, 1, bias=True)

    def forward(self, predicted_latents: torch.Tensor) -> torch.Tensor:
        if not isinstance(predicted_latents, torch.Tensor):
            raise TypeError("predicted_latents must be a tensor")
        if predicted_latents.ndim != 5 or tuple(predicted_latents.shape[1:]) != (
            9,
            64,
            64,
            64,
        ):
            raise ValueError(
                "predicted_latents must have shape (B,9,64,64,64)"
            )
        if predicted_latents.shape[0] < 1:
            raise ValueError("predicted_latents must contain at least one row")
        if predicted_latents.dtype != torch.float32:
            raise TypeError("predicted_latents must use exact float32")
        if predicted_latents.device != self.sweep_masks.device:
            raise TypeError("predicted_latents and sweep_masks must share a device")
        if not bool(torch.isfinite(predicted_latents).all()):
            raise FloatingPointError("predicted_latents is nonfinite")

        weights = self.sweep_masks.to(dtype=predicted_latents.dtype)
        counts = weights.sum(dim=(-2, -1))
        pooled = torch.einsum(
            "bachw,akhw->bakc", predicted_latents, weights
        ) / counts[None, :, :, None]
        logits = self.output(pooled).squeeze(-1)
        if tuple(logits.shape) != (
            predicted_latents.shape[0],
            9,
            SWEEP_PROGRESS_BIN_COUNT_V1,
        ):
            raise RuntimeError("swept-progress survival logit shape changed")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("swept-progress survival logits are nonfinite")
        return logits


class GeometryAnchoredSweptProgressSurvivalJointJepaV1(_FrozenJointJepaV1):
    """Frozen joint JEPA with one jointly trained swept-progress head."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        super().__init__(n320_encoder_state_dict, config)
        caller_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(self.config.initialization_seed)
            self.predictor.swept_progress_head = SweptProgressSurvivalHeadV1(
                sweep_masks
            )
        finally:
            torch.random.set_rng_state(caller_rng)

    def predict_all_actions_with_survival(
        self, current_latent: torch.Tensor
    ) -> SweptProgressSurvivalPredictionV1:
        predicted = self.predict_all_actions(current_latent)
        logits = self.predictor.swept_progress_head(predicted)
        return SweptProgressSurvivalPredictionV1(
            predicted_latents=predicted,
            survival_logits=logits,
        )


GeometryAnchoredSweptProgressSurvivalJointJepaV1Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)
# The frozen runner resolves this historical name from its selected model module.
GeometryAnchoredDeformableBevLiftJointJepaV1 = (
    GeometryAnchoredSweptProgressSurvivalJointJepaV1
)
final_class_macro_nll_per_row_v1 = final_class_macro_nll_per_row
latent_energy_per_row_v1 = latent_energy_per_row


__all__ = [
    "ACTION_VOCABULARY_V1",
    "FREE_CLASS_V1",
    "GeometryAnchoredBevSamplingV1",
    "GeometryAnchoredDeformableBevLiftJointJepaV1",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV1",
    "GeometryAnchoredSweptProgressSurvivalJointJepaV1Config",
    "OCCUPIED_CLASS_V1",
    "SWEEP_PROGRESS_BIN_COUNT_V1",
    "SweptProgressSurvivalHeadV1",
    "SweptProgressSurvivalPredictionV1",
    "UNKNOWN_CLASS_V1",
    "final_class_macro_nll_per_row",
    "final_class_macro_nll_per_row_v1",
    "latent_energy_per_row",
    "latent_energy_per_row_v1",
]
