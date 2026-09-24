"""Direct BEV V5 all-actions state-delta contrast objective adapter."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


_V3_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
_V3_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v5_state_delta_frozen_v3_model",
    _V3_SOURCE_PATH,
)
if _V3_SPEC is None or _V3_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V3 model source")
_v3 = importlib.util.module_from_spec(_V3_SPEC)
sys.modules[_V3_SPEC.name] = _v3
_V3_SPEC.loader.exec_module(_v3)


ACTION_VOCABULARY_V1 = _v3.ACTION_VOCABULARY_V1
DirectBevStateObjectiveV1 = _v3.DirectBevStateObjectiveV1
DirectEgocentricBevStateJepaV1Config = (
    _v3.DirectEgocentricBevStateJepaV1Config
)
FREE_CLASS_V1 = _v3.FREE_CLASS_V1
HOLD_ACTION_INDEX_V1 = _v3.HOLD_ACTION_INDEX_V1
HierarchicalHardLossV1 = _v3.HierarchicalHardLossV1
OCCUPIED_CLASS_V1 = _v3.OCCUPIED_CLASS_V1
UNKNOWN_CLASS_V1 = _v3.UNKNOWN_CLASS_V1
WrongRgbGroundingControlV1 = _v3.WrongRgbGroundingControlV1
direct_bev_state_objective_v1 = _v3.direct_bev_state_objective_v1
hard_hierarchical_raster_loss_v1 = _v3.hard_hierarchical_raster_loss_v1
soft_hierarchical_state_energy_v1 = _v3.soft_hierarchical_state_energy_v1
_hard_hierarchical_loss_per_row = _v3._hard_hierarchical_loss_per_row


STATE_DELTA_SCALE_FLOOR_V5 = 1e-4
STATE_DELTA_CONTRAST_WEIGHT_V5 = 1.0


def _validate_state_delta_inputs_v5(
    *,
    current_state_logits: torch.Tensor,
    all_action_prediction_logits: torch.Tensor,
    target_current_logits: torch.Tensor,
    target_next_logits: torch.Tensor,
    executed_action_indices: torch.Tensor,
) -> None:
    if (
        current_state_logits.ndim != 4
        or current_state_logits.shape[0] < 1
        or current_state_logits.shape[1] != 3
    ):
        raise ValueError("current_state_logits must have shape (B,3,H,W)")
    if not current_state_logits.is_floating_point():
        raise TypeError("state logits must use a floating dtype")
    if not bool(torch.isfinite(current_state_logits).all()):
        raise FloatingPointError("current_state_logits is nonfinite")

    batch, _channels, height, width = current_state_logits.shape
    expected_predictions = (batch, 9, 3, height, width)
    if tuple(all_action_prediction_logits.shape) != expected_predictions:
        raise ValueError(
            "all_action_prediction_logits must have shape (B,9,3,H,W)"
        )
    for name, value in (
        ("all_action_prediction_logits", all_action_prediction_logits),
        ("target_current_logits", target_current_logits),
        ("target_next_logits", target_next_logits),
    ):
        expected_shape = (
            expected_predictions
            if name == "all_action_prediction_logits"
            else tuple(current_state_logits.shape)
        )
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{name} shape differs from current state")
        if (
            value.dtype != current_state_logits.dtype
            or not value.is_floating_point()
        ):
            raise TypeError("all state logits must share one floating dtype")
        if value.device != current_state_logits.device:
            raise TypeError("all state logits must share one device")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} is nonfinite")

    if (
        executed_action_indices.shape != (batch,)
        or executed_action_indices.dtype != torch.long
    ):
        raise TypeError(
            "executed_action_indices must be long with shape (B,)"
        )
    if executed_action_indices.device != current_state_logits.device:
        raise TypeError("executed actions and state logits must share a device")
    if bool((executed_action_indices < 0).any()) or bool(
        (executed_action_indices >= 9).any()
    ):
        raise ValueError("executed action indices must lie in [0,8]")


def all_actions_state_delta_contrast_v5(
    *,
    current_state_logits: torch.Tensor,
    all_action_prediction_logits: torch.Tensor,
    target_current_logits: torch.Tensor,
    target_next_logits: torch.Tensor,
    executed_action_indices: torch.Tensor,
) -> torch.Tensor:
    """Return the preregistered normalized all-actions delta contrast ``A``."""

    _validate_state_delta_inputs_v5(
        current_state_logits=current_state_logits,
        all_action_prediction_logits=all_action_prediction_logits,
        target_current_logits=target_current_logits,
        target_next_logits=target_next_logits,
        executed_action_indices=executed_action_indices,
    )
    current_probability = torch.softmax(current_state_logits, dim=1)
    prediction_probability = torch.softmax(
        all_action_prediction_logits,
        dim=2,
    )
    predicted_delta = (
        prediction_probability - current_probability[:, None]
    )
    target_delta = (
        torch.softmax(target_next_logits.detach(), dim=1)
        - torch.softmax(target_current_logits.detach(), dim=1)
    ).detach()
    distances = (
        predicted_delta - target_delta[:, None]
    ).square().mean(dim=(2, 3, 4))
    scale = distances.mean(dim=1).detach().clamp_min(
        STATE_DELTA_SCALE_FLOOR_V5
    )
    delta_logits = -distances / scale[:, None]
    return F.cross_entropy(
        delta_logits,
        executed_action_indices,
        reduction="mean",
    ) / math.log(9.0)


class DirectEgocentricBevStateJepaV1(_v3.DirectEgocentricBevStateJepaV1):
    """Frozen V3 model with only the V5 parameter-free objective term."""

    def training_objective(
        self,
        *,
        current_rgb: torch.Tensor,
        next_rgb: torch.Tensor,
        fixed_negative_rgb: torch.Tensor,
        action_one_hot: torch.Tensor,
        non_hold_mask: torch.Tensor,
        current_labels: torch.Tensor,
        next_labels: torch.Tensor,
    ) -> DirectBevStateObjectiveV1:
        base = super().training_objective(
            current_rgb=current_rgb,
            next_rgb=next_rgb,
            fixed_negative_rgb=fixed_negative_rgb,
            action_one_hot=action_one_hot,
            non_hold_mask=non_hold_mask,
            current_labels=current_labels,
            next_labels=next_labels,
        )
        auxiliary = all_actions_state_delta_contrast_v5(
            current_state_logits=base.current_state_logits,
            all_action_prediction_logits=base.all_action_prediction_logits,
            target_current_logits=base.target_current_logits,
            target_next_logits=base.target_next_logits,
            executed_action_indices=action_one_hot.argmax(dim=1),
        )
        weighted_auxiliary = STATE_DELTA_CONTRAST_WEIGHT_V5 * auxiliary
        return base._replace(
            total=base.total + weighted_auxiliary,
            C=base.C + weighted_auxiliary,
        )


__all__ = [
    *_v3.__all__,
    "STATE_DELTA_CONTRAST_WEIGHT_V5",
    "STATE_DELTA_SCALE_FLOOR_V5",
    "all_actions_state_delta_contrast_v5",
]
