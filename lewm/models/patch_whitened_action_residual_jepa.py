"""Exact helpers for Patch-Whitened Action-Residual JEPA V4 Energy-NLL.

This module intentionally contains no data, schedule, runner, or custody
logic.  It only implements the frozen mathematical mechanism registered in
the V4 Action-Indexed Energy-NLL preregistration.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_DIM = 192
ACTION_DIM = 9
WHITENING_EPS = 1e-4
ACTION_GATE_INITIALIZATION_SEED = 20260712
ACTION_GATE_WEIGHT_STD = 0.01 / math.sqrt(LATENT_DIM)
ACTION_GATE_BIAS = 0.01
RESIDUAL_ALPHA = 0.1 / math.sqrt(LATENT_DIM)


class ActionIndexedPredictions(NamedTuple):
    """Uniformly ordered nine-action predictions and diagnostic gathers."""

    executed_indices: torch.Tensor
    all_predictions: torch.Tensor
    executed: torch.Tensor
    control_indices: torch.Tensor
    controls: torch.Tensor


class WhiteningTerms(NamedTuple):
    """Registered variance and off-diagonal covariance terms."""

    variance: torch.Tensor
    covariance: torch.Tensor


class ActionIndexedLosses(NamedTuple):
    """Executed JEPA energy and detached-scale all-action identification NLL."""

    jepa: torch.Tensor
    identification: torch.Tensor
    total: torch.Tensor
    energies: torch.Tensor
    row_scale: torch.Tensor
    logits: torch.Tensor
    identification_per_row: torch.Tensor


class ActionIndexedResidualOperators(nn.Module):
    """Wrap the shared projector with nine exact-zero tokenwise operators.

    ``action_weights[a]`` uses :class:`torch.nn.Linear`'s weight orientation,
    but is allocated directly so construction consumes no RNG and introduces
    no bias parameter.
    """

    def __init__(self, shared_projector: nn.Module):
        super().__init__()
        if not isinstance(shared_projector, nn.Module):
            raise TypeError("shared_projector must be an nn.Module")
        reference = next(shared_projector.parameters(), None)
        if reference is None:
            raise ValueError("shared_projector must have parameters")
        if reference.dtype != torch.float32:
            raise TypeError("shared_projector parameters must be float32")
        self.shared_projector = shared_projector
        self.action_weights = nn.Parameter(
            torch.zeros(
                ACTION_DIM,
                LATENT_DIM,
                LATENT_DIM,
                device=reference.device,
                dtype=reference.dtype,
            )
        )

    def project_shared(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the preserved shared projector exactly once."""

        projected = self.shared_projector(tokens)
        if projected.shape != tokens.shape:
            raise ValueError("shared projector output must align with tokens")
        return projected

    def project_all(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply every action operator, returning ``(B, 9, N, 192)``."""

        _validate_tokens(tokens, name="operator tokens")
        return torch.einsum("bnd,aed->bane", tokens, self.action_weights)

    def project_selected(
        self,
        tokens: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply each row's executed-action operator."""

        _validate_tokens(tokens, name="operator tokens")
        _validate_action_indices(action_indices, batch=tokens.shape[0])
        selected_weights = self.action_weights.index_select(0, action_indices)
        return torch.einsum("bnd,bed->bne", tokens, selected_weights)


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


def _validate_tokens(tokens: torch.Tensor, *, name: str) -> None:
    if tokens.ndim != 3 or tokens.shape[-1] != LATENT_DIM:
        raise ValueError(f"{name} must have shape (B, N, {LATENT_DIM})")
    if tokens.shape[0] < 1 or tokens.shape[1] < 1:
        raise ValueError(f"{name} must have positive batch and token counts")


def _validate_action_indices(indices: torch.Tensor, *, batch: int) -> None:
    if indices.ndim != 1 or tuple(indices.shape) != (batch,):
        raise ValueError(f"action indices must have shape ({batch},)")
    if indices.dtype != torch.long:
        raise TypeError("action indices must have dtype torch.long")
    if not bool(((indices >= 0) & (indices < ACTION_DIM)).all().item()):
        raise ValueError(f"action indices must lie in [0, {ACTION_DIM})")


def requested_action_indices(requested_actions: torch.Tensor) -> torch.Tensor:
    """Validate exact one-hot executed actions and return their indices."""

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
    return requested_actions.argmax(dim=1)


def action_independent_trunk(
    predictor: nn.Module,
    state: torch.Tensor,
) -> torch.Tensor:
    """Run one spatial trunk pass with exact-zero block conditioning.

    This deliberately bypasses both ``predict_step`` and ``action_embed``.
    The existing position embedding, input dropout, transformer blocks, and
    final normalization are preserved.
    """

    _validate_tokens(state, name="state")
    if getattr(predictor, "latent_dim", None) != LATENT_DIM:
        raise ValueError(f"predictor latent_dim must be {LATENT_DIM}")
    if getattr(predictor, "num_spatial_tokens", None) != state.shape[1]:
        raise ValueError("predictor spatial-token count must align with state")
    position = getattr(predictor, "spatial_pos_embed", None)
    if not isinstance(position, nn.Parameter) or tuple(position.shape) != (
        1,
        state.shape[1],
        LATENT_DIM,
    ):
        raise TypeError("predictor must expose the exact spatial_pos_embed")
    input_drop = getattr(predictor, "input_drop", None)
    blocks = getattr(predictor, "blocks", None)
    norm = getattr(predictor, "norm", None)
    if not isinstance(input_drop, nn.Module):
        raise TypeError("predictor must expose input_drop")
    if not isinstance(blocks, nn.ModuleList) or len(blocks) < 1:
        raise TypeError("predictor.blocks must be a non-empty ModuleList")
    if not isinstance(norm, nn.Module):
        raise TypeError("predictor must expose norm")

    hidden = input_drop(state + position)
    zero_condition = torch.zeros_like(hidden)
    for block in blocks:
        hidden = block(hidden, zero_condition, causal=False)
    hidden = norm(hidden)
    if hidden.shape != state.shape:
        raise ValueError("predictor trunk output must align with state")
    return hidden


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


def predict_action_indexed_residuals(
    predictor: nn.Module,
    prediction_projector: ActionIndexedResidualOperators,
    online_state: torch.Tensor,
    requested_actions: torch.Tensor,
    ema_current: torch.Tensor,
) -> ActionIndexedPredictions:
    """Predict all nine actions with the preregistered gradient isolation."""

    if not isinstance(
        prediction_projector, ActionIndexedResidualOperators
    ):
        raise TypeError(
            "prediction_projector must be ActionIndexedResidualOperators"
        )
    _validate_tokens(online_state, name="online_state")
    if online_state.shape[0] != requested_actions.shape[0]:
        raise ValueError("online_state and requested_actions batch sizes differ")
    if ema_current.shape != online_state.shape:
        raise ValueError("ema_current must align exactly with online_state")
    executed_indices = requested_action_indices(requested_actions)
    batch, tokens, dim = online_state.shape

    shared_hidden = action_independent_trunk(predictor, online_state)
    shared_residual = prediction_projector.project_shared(shared_hidden)

    detached_action_residuals = prediction_projector.project_all(
        shared_hidden.detach()
    )
    executed_action_residual = prediction_projector.project_selected(
        shared_hidden,
        executed_indices,
    )
    executed_mask = F.one_hot(
        executed_indices,
        num_classes=ACTION_DIM,
    ).to(dtype=torch.bool)[:, :, None, None]
    action_residuals = torch.where(
        executed_mask,
        executed_action_residual[:, None],
        detached_action_residuals,
    )
    shared_residuals = torch.where(
        executed_mask,
        shared_residual[:, None],
        shared_residual.detach()[:, None],
    )
    all_predictions = residual_reconstruct(
        ema_current,
        shared_residuals + action_residuals,
    )

    candidate_indices = torch.arange(
        ACTION_DIM,
        device=executed_indices.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch, -1)
    control_indices = candidate_indices[
        candidate_indices != executed_indices[:, None]
    ].reshape(batch, ACTION_DIM - 1)
    control_gather = control_indices[:, :, None, None].expand(
        -1, -1, tokens, dim
    )
    executed_gather = executed_indices[:, None, None, None].expand(
        -1, 1, tokens, dim
    )
    controls = all_predictions.gather(1, control_gather)
    executed = all_predictions.gather(1, executed_gather).squeeze(1)

    return ActionIndexedPredictions(
        executed_indices=executed_indices,
        all_predictions=all_predictions,
        executed=executed,
        control_indices=control_indices,
        controls=controls,
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


def action_indexed_energy_nll(
    predictions: ActionIndexedPredictions,
    ema_next: torch.Tensor,
) -> ActionIndexedLosses:
    """Compute executed JEPA plus detached-row-scale all-action Energy-NLL."""

    all_predictions = predictions.all_predictions
    if all_predictions.ndim != 4 or tuple(all_predictions.shape[1::2]) != (
        ACTION_DIM,
        LATENT_DIM,
    ):
        raise ValueError(
            "all_predictions must have shape "
            f"(B, {ACTION_DIM}, N, {LATENT_DIM})"
        )
    batch, _, tokens, _ = all_predictions.shape
    if tuple(ema_next.shape) != (batch, tokens, LATENT_DIM):
        raise ValueError("ema_next must align with all_predictions")
    _validate_action_indices(predictions.executed_indices, batch=batch)

    target = ema_next.detach()
    energies = (
        all_predictions - target[:, None]
    ).square().mean(dim=(2, 3))
    row_scale = energies.mean(dim=1).detach().clamp_min(1e-8)
    logits = -energies / row_scale[:, None]
    identification_per_row = row_scale * F.cross_entropy(
        logits,
        predictions.executed_indices,
        reduction="none",
    )
    identification = identification_per_row.mean()
    executed_energy = energies.gather(
        1, predictions.executed_indices[:, None]
    ).squeeze(1)
    jepa = executed_energy.mean()
    return ActionIndexedLosses(
        jepa=jepa,
        identification=identification,
        total=jepa + identification,
        energies=energies,
        row_scale=row_scale,
        logits=logits,
        identification_per_row=identification_per_row,
    )


__all__ = [
    "ACTION_DIM",
    "ACTION_GATE_BIAS",
    "ACTION_GATE_INITIALIZATION_SEED",
    "ACTION_GATE_WEIGHT_STD",
    "ActionIndexedLosses",
    "ActionIndexedPredictions",
    "ActionIndexedResidualOperators",
    "LATENT_DIM",
    "RESIDUAL_ALPHA",
    "WHITENING_EPS",
    "WhiteningTerms",
    "action_independent_trunk",
    "action_indexed_energy_nll",
    "initialize_action_gate_rows",
    "patch_whitening_terms",
    "predict_action_indexed_residuals",
    "requested_action_indices",
    "residual_reconstruct",
]
