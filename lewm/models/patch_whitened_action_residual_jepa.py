"""Exact objective helpers for Patch-Whitened Action-Residual JEPA V1.

This module intentionally contains no data, schedule, runner, or custody
logic.  It only implements the frozen mathematical mechanism registered in
the V1 preregistration.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM = 192
ACTION_DIM = 9
HOLD_ACTION_INDEX = 6
ACTION_RATIO = 0.95
WHITENING_EPS = 1e-4
ACTION_GATE_INITIALIZATION_SEED = 20260712
ACTION_GATE_WEIGHT_STD = 0.01 / math.sqrt(LATENT_DIM)
ACTION_GATE_BIAS = 0.01
RESIDUAL_ALPHA = 0.1 / math.sqrt(LATENT_DIM)


class ActionLayout(NamedTuple):
    """All real actions and the eight non-true controls for each row."""

    all_actions: torch.Tensor
    requested_indices: torch.Tensor
    control_indices: torch.Tensor
    control_actions: torch.Tensor
    wrong_loss_mask: torch.Tensor
    non_hold_mask: torch.Tensor


class ResidualPredictions(NamedTuple):
    """Live requested-action prediction and detached-state controls."""

    layout: ActionLayout
    true: torch.Tensor
    controls: torch.Tensor


class WhiteningTerms(NamedTuple):
    """Registered variance and off-diagonal covariance terms."""

    variance: torch.Tensor
    covariance: torch.Tensor


class ActionResidualLosses(NamedTuple):
    """JEPA, wrong-action, and real-hold loss components."""

    jepa: torch.Tensor
    wrong: torch.Tensor
    hold: torch.Tensor
    true_energy: torch.Tensor
    control_energy: torch.Tensor
    wrong_per_row: torch.Tensor


def initialize_action_gate_rows(predictor: nn.Module) -> dict[str, object]:
    """Apply the exact isolated small-open AdaLN gate initialization.

    The predictor constructor must have left every AdaLN modulation weight and
    bias at zero.  A second application is rejected rather than silently
    changing an already-bound initialization.
    """

    blocks = getattr(predictor, "blocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) < 1:
        raise TypeError("predictor.blocks must be a non-empty ModuleList")

    linears: list[nn.Linear] = []
    expected_weight_shape = (6 * LATENT_DIM, LATENT_DIM)
    expected_bias_shape = (6 * LATENT_DIM,)
    for index, block in enumerate(blocks):
        modulation = getattr(block, "adaLN_modulation", None)
        if not isinstance(modulation, nn.Sequential) or len(modulation) < 1:
            raise TypeError(
                f"predictor block {index} has no sequential AdaLN modulation"
            )
        linear = modulation[-1]
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"predictor block {index} AdaLN output must be nn.Linear"
            )
        if tuple(linear.weight.shape) != expected_weight_shape:
            raise ValueError(
                f"predictor block {index} AdaLN weight must have shape "
                f"{expected_weight_shape}"
            )
        if linear.bias is None or tuple(linear.bias.shape) != expected_bias_shape:
            raise ValueError(
                f"predictor block {index} AdaLN bias must have shape "
                f"{expected_bias_shape}"
            )
        if linear.weight.dtype != torch.float32 or linear.bias.dtype != torch.float32:
            raise TypeError("AdaLN modulation parameters must be float32")
        if bool(torch.count_nonzero(linear.weight).item()) or bool(
            torch.count_nonzero(linear.bias).item()
        ):
            raise ValueError(
                f"predictor block {index} AdaLN modulation was not all-zero"
            )
        linears.append(linear)

    attention_rows = slice(2 * LATENT_DIM, 3 * LATENT_DIM)
    mlp_rows = slice(5 * LATENT_DIM, 6 * LATENT_DIM)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(ACTION_GATE_INITIALIZATION_SEED)

    with torch.no_grad():
        for linear in linears:
            for rows in (attention_rows, mlp_rows):
                shape = tuple(linear.weight[rows].shape)
                values = torch.randn(
                    shape,
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                )
                values.mul_(ACTION_GATE_WEIGHT_STD)
                linear.weight[rows].copy_(
                    values.to(device=linear.weight.device)
                )
                linear.bias[rows].fill_(ACTION_GATE_BIAS)

    return {
        "seed": ACTION_GATE_INITIALIZATION_SEED,
        "block_count": len(linears),
        "latent_dim": LATENT_DIM,
        "attention_gate_rows": [2 * LATENT_DIM, 3 * LATENT_DIM],
        "mlp_gate_rows": [5 * LATENT_DIM, 6 * LATENT_DIM],
        "weight_std": ACTION_GATE_WEIGHT_STD,
        "bias": ACTION_GATE_BIAS,
        "changed_weight_scalar_count":
            len(linears) * 2 * LATENT_DIM * LATENT_DIM,
        "changed_bias_scalar_count": len(linears) * 2 * LATENT_DIM,
    }


def build_action_layout(requested_actions: torch.Tensor) -> ActionLayout:
    """Return the frozen nine-action grid and exact per-row control masks."""

    if (
        requested_actions.ndim != 2
        or requested_actions.shape[1] != ACTION_DIM
    ):
        raise ValueError(
            f"requested_actions must have shape (B, {ACTION_DIM})"
        )
    if not requested_actions.is_floating_point():
        raise TypeError("requested_actions must be floating point one-hot rows")
    if requested_actions.shape[0] < 1:
        raise ValueError("requested_actions must contain at least one row")
    is_binary = (requested_actions == 0) | (requested_actions == 1)
    row_sums = requested_actions.sum(dim=1)
    if not bool(is_binary.all().item()) or not bool((row_sums == 1).all().item()):
        raise ValueError("requested_actions must be exact one-hot rows")

    batch = requested_actions.shape[0]
    eye = torch.eye(
        ACTION_DIM,
        device=requested_actions.device,
        dtype=requested_actions.dtype,
    )
    all_actions = eye.unsqueeze(0).expand(batch, -1, -1)
    requested_indices = requested_actions.argmax(dim=1)
    candidate_indices = torch.arange(
        ACTION_DIM,
        device=requested_actions.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch, -1)
    all_wrong_mask = candidate_indices != requested_indices[:, None]
    control_indices = candidate_indices[all_wrong_mask].reshape(
        batch, ACTION_DIM - 1
    )
    control_actions = eye[control_indices]
    non_hold_mask = requested_indices != HOLD_ACTION_INDEX
    wrong_loss_mask = (
        (control_indices != HOLD_ACTION_INDEX)
        | ~non_hold_mask[:, None]
    )
    expected_counts = torch.where(
        non_hold_mask,
        torch.full_like(requested_indices, ACTION_DIM - 2),
        torch.full_like(requested_indices, ACTION_DIM - 1),
    )
    if not bool(
        (wrong_loss_mask.sum(dim=1) == expected_counts).all().item()
    ):
        raise RuntimeError("action-control mask construction changed")

    return ActionLayout(
        all_actions=all_actions,
        requested_indices=requested_indices,
        control_indices=control_indices,
        control_actions=control_actions,
        wrong_loss_mask=wrong_loss_mask,
        non_hold_mask=non_hold_mask,
    )


def residual_reconstruct(
    ema_current: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct normalized future tokens from a detached EMA-current skip."""

    if (
        ema_current.ndim != 3
        or ema_current.shape[-1] != LATENT_DIM
    ):
        raise ValueError(
            f"ema_current must have shape (B, N, {LATENT_DIM})"
        )
    if residual.ndim == 3:
        if residual.shape != ema_current.shape:
            raise ValueError("3-D residual must have the EMA-current shape")
        skip = ema_current.detach()
    elif residual.ndim == 4:
        if (
            residual.shape[0] != ema_current.shape[0]
            or residual.shape[2:] != ema_current.shape[1:]
        ):
            raise ValueError(
                "4-D residual must have shape (B, K, N, D) aligned "
                "with ema_current"
            )
        skip = ema_current.detach()[:, None]
    else:
        raise ValueError("residual must have shape (B, N, D) or (B, K, N, D)")
    return F.normalize(
        skip + RESIDUAL_ALPHA * residual,
        p=2,
        dim=-1,
        eps=1e-8,
    )


def _predict_residual(
    predictor: nn.Module,
    prediction_projector: nn.Module,
    state: torch.Tensor,
    actions: torch.Tensor,
    ema_current: torch.Tensor,
) -> torch.Tensor:
    if state.ndim != 3 or state.shape[-1] != LATENT_DIM:
        raise ValueError(f"state must have shape (B, N, {LATENT_DIM})")
    if actions.ndim != 2 or actions.shape != (state.shape[0], ACTION_DIM):
        raise ValueError(
            f"actions must have shape ({state.shape[0]}, {ACTION_DIM})"
        )
    if ema_current.shape != state.shape:
        raise ValueError("ema_current must align exactly with state")
    predict_step = getattr(predictor, "predict_step", None)
    if not callable(predict_step):
        raise TypeError("predictor must expose predict_step(state, action)")
    predicted_raw = predict_step(state, actions)
    if predicted_raw.shape != state.shape:
        raise ValueError("predictor output must align exactly with state")
    residual = prediction_projector(predicted_raw)
    if residual.shape != state.shape:
        raise ValueError("prediction-projector output must align with state")
    return residual_reconstruct(ema_current, residual)


def predict_live_and_control_residuals(
    predictor: nn.Module,
    prediction_projector: nn.Module,
    online_state: torch.Tensor,
    requested_actions: torch.Tensor,
    ema_current: torch.Tensor,
) -> ResidualPredictions:
    """Predict one live true action and exactly eight detached-state controls."""

    layout = build_action_layout(requested_actions)
    if online_state.shape[0] != requested_actions.shape[0]:
        raise ValueError("online_state and requested_actions batch sizes differ")
    true_prediction = _predict_residual(
        predictor,
        prediction_projector,
        online_state,
        requested_actions,
        ema_current,
    )

    batch, tokens, dim = online_state.shape
    control_count = ACTION_DIM - 1
    control_state = online_state.detach()[:, None].expand(
        -1, control_count, -1, -1
    ).reshape(batch * control_count, tokens, dim)
    control_actions = layout.control_actions.reshape(
        batch * control_count, ACTION_DIM
    )
    control_skip = ema_current[:, None].expand(
        -1, control_count, -1, -1
    ).reshape(batch * control_count, tokens, dim)
    control_predictions = _predict_residual(
        predictor,
        prediction_projector,
        control_state,
        control_actions,
        control_skip,
    ).reshape(batch, control_count, tokens, dim)

    return ResidualPredictions(
        layout=layout,
        true=true_prediction,
        controls=control_predictions,
    )


def patch_whitening_terms(tokens: torch.Tensor) -> WhiteningTerms:
    """Compute the exact per-microbatch rank-matrix V and K terms."""

    if tokens.ndim != 3:
        raise ValueError("tokens must have shape (B, N, D)")
    if tokens.dtype != torch.float32:
        raise TypeError("patch whitening requires exact float32 tokens")
    batch, patches, dim = tokens.shape
    if batch < 2 or patches < 1 or dim < 1:
        raise ValueError("patch whitening requires B >= 2 and positive N,D")

    position_centered = tokens - tokens.mean(dim=0, keepdim=True)
    rank_matrix = position_centered.reshape(batch * patches, dim)
    rms_square = rank_matrix.square().mean().detach()
    normalized = rank_matrix / torch.sqrt(rms_square + WHITENING_EPS)
    covariance = (
        normalized.transpose(0, 1) @ normalized
    ) / float(batch * patches - 1)
    diagonal = covariance.diagonal()
    variance = F.relu(
        1.0 - torch.sqrt(diagonal + WHITENING_EPS)
    ).mean()
    off_diagonal_mask = ~torch.eye(
        dim,
        device=covariance.device,
        dtype=torch.bool,
    )
    covariance_loss = covariance.square().masked_select(
        off_diagonal_mask
    ).sum() / float(dim)
    return WhiteningTerms(
        variance=variance,
        covariance=covariance_loss,
    )


def action_residual_losses(
    predictions: ResidualPredictions,
    ema_next: torch.Tensor,
) -> ActionResidualLosses:
    """Compute live JEPA energy and exact row-balanced action hinges."""

    true_prediction = predictions.true
    controls = predictions.controls
    layout = predictions.layout
    if (
        true_prediction.ndim != 3
        or true_prediction.shape[-1] != LATENT_DIM
    ):
        raise ValueError(
            f"true prediction must have shape (B, N, {LATENT_DIM})"
        )
    expected_controls = (
        true_prediction.shape[0],
        ACTION_DIM - 1,
        true_prediction.shape[1],
        LATENT_DIM,
    )
    if tuple(controls.shape) != expected_controls:
        raise ValueError(f"controls must have shape {expected_controls}")
    if ema_next.shape != true_prediction.shape:
        raise ValueError("ema_next must align exactly with true prediction")

    target = ema_next.detach()
    true_energy = (true_prediction - target).square().mean(dim=(1, 2))
    control_energy = (
        controls - target[:, None]
    ).square().mean(dim=(2, 3))
    threshold = true_energy.detach() / ACTION_RATIO
    hinges = F.relu(threshold[:, None] - control_energy)

    wrong_mask = layout.wrong_loss_mask
    if tuple(wrong_mask.shape) != tuple(control_energy.shape):
        raise ValueError("wrong-action mask does not align with controls")
    wrong_counts = wrong_mask.sum(dim=1)
    if not bool((wrong_counts > 0).all().item()):
        raise RuntimeError("every row must have eligible wrong actions")
    wrong_per_row = (
        hinges.masked_fill(~wrong_mask, 0.0).sum(dim=1)
        / wrong_counts.to(dtype=hinges.dtype)
    )
    wrong_loss = wrong_per_row.mean()

    hold_control_mask = (
        (layout.control_indices == HOLD_ACTION_INDEX)
        & layout.non_hold_mask[:, None]
    )
    if bool(layout.non_hold_mask.any().item()):
        if not bool(
            (
                hold_control_mask.sum(dim=1)
                == layout.non_hold_mask.to(dtype=torch.long)
            ).all().item()
        ):
            raise RuntimeError("real-hold control population changed")
        hold_loss = hinges.masked_select(hold_control_mask).mean()
    else:
        hold_loss = true_energy.new_zeros(())

    return ActionResidualLosses(
        jepa=true_energy.mean(),
        wrong=wrong_loss,
        hold=hold_loss,
        true_energy=true_energy,
        control_energy=control_energy,
        wrong_per_row=wrong_per_row,
    )


__all__ = [
    "ACTION_DIM",
    "ACTION_GATE_BIAS",
    "ACTION_GATE_INITIALIZATION_SEED",
    "ACTION_GATE_WEIGHT_STD",
    "ACTION_RATIO",
    "ActionLayout",
    "ActionResidualLosses",
    "HOLD_ACTION_INDEX",
    "LATENT_DIM",
    "RESIDUAL_ALPHA",
    "ResidualPredictions",
    "WHITENING_EPS",
    "WhiteningTerms",
    "action_residual_losses",
    "build_action_layout",
    "initialize_action_gate_rows",
    "patch_whitening_terms",
    "predict_live_and_control_residuals",
    "residual_reconstruct",
]
