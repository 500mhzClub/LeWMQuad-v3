"""Exact helpers for V7 Local-Correspondence Transport JEPA.

This module intentionally contains no data, schedule, runner, or custody
logic.  It implements the deterministic centered-softmax residual local
transport registered for V7.
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
TOKEN_SIDE = 16
TOKEN_COUNT = TOKEN_SIDE * TOKEN_SIDE
NEIGHBOR_COUNT = 9
NONCENTER_NEIGHBOR_COUNT = 8
CENTER_OFFSET_INDEX = 4
WHITENING_EPS = 1e-4
ACTION_GATE_INITIALIZATION_SEED = 20260712
ACTION_GATE_WEIGHT_STD = 0.01 / math.sqrt(LATENT_DIM)
ACTION_GATE_BIAS = 0.01
RESIDUAL_ALPHA = 0.1 / math.sqrt(LATENT_DIM)


class ActionIndexedPredictions(NamedTuple):
    """Uniformly ordered nine-action predictions and correspondence state."""

    executed_indices: torch.Tensor
    all_predictions: torch.Tensor
    all_transport_logits: torch.Tensor
    all_transport_probabilities: torch.Tensor
    all_expected_offsets: torch.Tensor
    all_transports: torch.Tensor
    executed: torch.Tensor
    control_indices: torch.Tensor
    controls: torch.Tensor


class CorrespondenceTargets(NamedTuple):
    """Detached EMA local-correspondence targets and viability statistic."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    mean_kl_to_uniform: torch.Tensor


class CorrespondenceTerms(NamedTuple):
    """Executed correspondence diagnostics and detached-scale loss."""

    loss: torch.Tensor
    centered_cross_entropy: torch.Tensor
    cross_entropy_per_row: torch.Tensor


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


class ActionConditionedLocalCorrespondenceTransport(nn.Module):
    """Wrap the shared projector with the exact V7 local transport map."""

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
        self.transport_weight = nn.Parameter(
            torch.zeros(
                NONCENTER_NEIGHBOR_COUNT,
                LATENT_DIM,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        rows, columns = torch.meshgrid(
            torch.arange(
                TOKEN_SIDE,
                device=reference.device,
                dtype=torch.long,
            ),
            torch.arange(
                TOKEN_SIDE,
                device=reference.device,
                dtype=torch.long,
            ),
            indexing="ij",
        )
        centers = torch.stack(
            (rows.reshape(-1), columns.reshape(-1)),
            dim=-1,
        )
        offsets = torch.tensor(
            [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 0),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
            device=reference.device,
            dtype=torch.long,
        )
        neighbors = centers[:, None] + offsets[None]
        neighbor_rows = neighbors[..., 0].clamp(0, TOKEN_SIDE - 1)
        neighbor_columns = neighbors[..., 1].clamp(0, TOKEN_SIDE - 1)
        neighbor_indices = TOKEN_SIDE * neighbor_rows + neighbor_columns
        self.register_buffer(
            "neighbor_indices",
            neighbor_indices,
            persistent=False,
        )

    def project_shared(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the preserved shared projector exactly once."""

        projected = self.shared_projector(tokens)
        if projected.shape != tokens.shape:
            raise ValueError("shared projector output must align with tokens")
        return projected

    def project_noncenter_logits(
        self,
        interactions: torch.Tensor,
    ) -> torch.Tensor:
        """Map state/action interactions to the eight learned local logits."""

        if (
            interactions.ndim != 4
            or interactions.shape[-1] != LATENT_DIM
        ):
            raise ValueError(
                "transport interactions must have shape (B,9,256,192)"
            )
        if (
            interactions.shape[0] < 1
            or tuple(interactions.shape[1:3])
            != (ACTION_DIM, TOKEN_COUNT)
        ):
            raise ValueError(
                "transport interactions must have shape (B,9,256,192)"
            )
        return F.linear(interactions, self.transport_weight, bias=None)


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


def relative_action_embeddings(
    predictor: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return all action embeddings relative to the exact hold embedding."""

    action_embed = getattr(predictor, "action_embed", None)
    if not isinstance(action_embed, nn.Module):
        raise TypeError("predictor must expose an action_embed module")
    candidates = torch.eye(
        ACTION_DIM,
        device=device,
        dtype=dtype,
    )[:, None, :]
    embeddings = action_embed(candidates)
    if tuple(embeddings.shape) != (ACTION_DIM, 1, LATENT_DIM):
        raise ValueError(
            "action_embed must map (9,1,9) to (9,1,192)"
        )
    if embeddings.device != device or embeddings.dtype != dtype:
        raise TypeError("action embeddings must align with the online state")
    if not bool(torch.isfinite(embeddings).all()):
        raise FloatingPointError("action embeddings contain a nonfinite value")
    relative = embeddings[:, 0] - embeddings[HOLD_ACTION_INDEX, 0]
    if bool(torch.count_nonzero(relative[HOLD_ACTION_INDEX]).item()):
        raise RuntimeError("hold-relative action embedding is not exact zero")
    return relative


def _local_offsets_yx(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the frozen full-offset order as ``(dy, dx)`` rows."""

    return torch.tensor(
        [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ],
        device=device,
        dtype=dtype,
    )


def _validate_local_transport_projector(
    projector: ActionConditionedLocalCorrespondenceTransport,
    *,
    device: torch.device,
) -> None:
    if not isinstance(
        projector,
        ActionConditionedLocalCorrespondenceTransport,
    ):
        raise TypeError(
            "projector must be "
            "ActionConditionedLocalCorrespondenceTransport"
        )
    if (
        tuple(projector.neighbor_indices.shape)
        != (TOKEN_COUNT, NEIGHBOR_COUNT)
        or projector.neighbor_indices.dtype != torch.long
        or projector.neighbor_indices.device != device
    ):
        raise TypeError("neighbor_indices changed or are misaligned")
    if not bool(
        (
            (projector.neighbor_indices >= 0)
            & (projector.neighbor_indices < TOKEN_COUNT)
        ).all().item()
    ):
        raise RuntimeError("neighbor_indices left the token grid")


def _validate_spatial_float32(
    tokens: torch.Tensor,
    *,
    name: str,
) -> None:
    _validate_tokens(tokens, name=name)
    if tuple(tokens.shape[1:]) != (TOKEN_COUNT, LATENT_DIM):
        raise ValueError(
            f"{name} must have shape (B, {TOKEN_COUNT}, {LATENT_DIM})"
        )
    if tokens.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError(f"{name} contains a nonfinite value")


def local_correspondence_targets(
    projector: ActionConditionedLocalCorrespondenceTransport,
    ema_current: torch.Tensor,
    ema_next: torch.Tensor,
) -> CorrespondenceTargets:
    """Construct the exact detached nine-neighbor EMA target distribution."""

    _validate_spatial_float32(ema_current, name="ema_current")
    _validate_spatial_float32(ema_next, name="ema_next")
    if (
        ema_next.shape != ema_current.shape
        or ema_next.device != ema_current.device
    ):
        raise ValueError("EMA current and next states must align exactly")
    _validate_local_transport_projector(
        projector,
        device=ema_current.device,
    )

    current = ema_current.detach()
    next_state = ema_next.detach()
    normalized_current = F.layer_norm(
        current,
        normalized_shape=(LATENT_DIM,),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    normalized_next = F.layer_norm(
        next_state,
        normalized_shape=(LATENT_DIM,),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    neighbor_current = normalized_current[:, projector.neighbor_indices]
    logits = (
        neighbor_current * normalized_next[:, :, None]
    ).sum(dim=-1) / math.sqrt(LATENT_DIM)
    probabilities = torch.softmax(logits, dim=-1)
    if (
        tuple(probabilities.shape)
        != (ema_current.shape[0], TOKEN_COUNT, NEIGHBOR_COUNT)
        or not bool(torch.isfinite(probabilities).all())
        or not bool((probabilities > 0).all().item())
        or not bool(
            torch.allclose(
                probabilities.sum(dim=-1),
                torch.ones_like(probabilities[..., 0]),
                rtol=0.0,
                atol=1e-6,
            )
        )
    ):
        raise FloatingPointError(
            "local correspondence target is invalid"
        )
    mean_kl_to_uniform = (
        probabilities
        * (probabilities.log() + math.log(NEIGHBOR_COUNT))
    ).sum(dim=-1).mean()
    if not bool(torch.isfinite(mean_kl_to_uniform)):
        raise FloatingPointError(
            "local correspondence target KL is nonfinite"
        )
    return CorrespondenceTargets(
        logits=logits.detach(),
        probabilities=probabilities.detach(),
        mean_kl_to_uniform=mean_kl_to_uniform.detach(),
    )


def centered_log_soft_cross_entropy(
    target_probs: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Evaluate exact registered ``Hc`` with full center offset index four."""

    if (
        target_probs.ndim < 2
        or student_logits.ndim < 2
        or target_probs.shape[-1] != NEIGHBOR_COUNT
        or student_logits.shape[-1] != NEIGHBOR_COUNT
    ):
        raise ValueError(
            "target_probs and student_logits must have broadcastable "
            "shape (...,9)"
        )
    try:
        torch.broadcast_shapes(
            target_probs.shape[:-1],
            student_logits.shape[:-1],
        )
    except RuntimeError as error:
        raise ValueError(
            "target_probs and student_logits must have broadcastable "
            "shape (...,9)"
        ) from error
    if (
        target_probs.dtype != torch.float32
        or student_logits.dtype != torch.float32
        or target_probs.device != student_logits.device
    ):
        raise TypeError(
            "cross-entropy inputs must be aligned float32 tensors"
        )
    target = target_probs.detach()
    if (
        not bool(torch.isfinite(target).all())
        or not bool(torch.isfinite(student_logits).all())
        or bool((target < 0).any().item())
        or not bool(
            torch.allclose(
                target.sum(dim=-1),
                torch.ones_like(target[..., 0]),
                rtol=0.0,
                atol=1e-6,
            )
        )
    ):
        raise FloatingPointError(
            "cross-entropy probabilities or logits are invalid"
        )
    log_probabilities = F.log_softmax(student_logits, dim=-1)
    center_log_probability = log_probabilities[..., CENTER_OFFSET_INDEX]
    return (
        -center_log_probability
        - (
            target
            * (
                log_probabilities
                - center_log_probability.unsqueeze(-1)
            )
        ).sum(dim=-1)
    )


def _residual_local_transport(
    projector: ActionConditionedLocalCorrespondenceTransport,
    ema_current: torch.Tensor,
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply centered-softmax transport without a five-dimensional gather."""

    batch = ema_current.shape[0]
    if tuple(probabilities.shape) != (
        batch,
        ACTION_DIM,
        TOKEN_COUNT,
        NEIGHBOR_COUNT,
    ):
        raise ValueError(
            "transport probabilities must have shape (B,9,256,9)"
        )
    source = ema_current.detach()
    neighbor_source = source[:, projector.neighbor_indices]
    neighbor_deltas = neighbor_source - source[:, :, None]
    uniform = torch.softmax(torch.zeros_like(probabilities), dim=-1)
    centered_probabilities = probabilities - uniform
    token_action_coefficients = centered_probabilities.permute(0, 2, 1, 3)
    displacement = torch.matmul(
        token_action_coefficients,
        neighbor_deltas,
    ).permute(0, 2, 1, 3)
    transports = source[:, None] + displacement
    offsets = _local_offsets_yx(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    expected_offsets = torch.matmul(centered_probabilities, offsets)
    if (
        tuple(transports.shape)
        != (batch, ACTION_DIM, TOKEN_COUNT, LATENT_DIM)
        or tuple(expected_offsets.shape)
        != (batch, ACTION_DIM, TOKEN_COUNT, 2)
        or not bool(torch.isfinite(transports).all())
        or not bool(torch.isfinite(expected_offsets).all())
        or bool((expected_offsets.abs() > 1.0).any().item())
    ):
        raise FloatingPointError("local transport output is invalid")
    return transports, expected_offsets


def predict_action_conditioned_local_transports(
    predictor: nn.Module,
    projector: ActionConditionedLocalCorrespondenceTransport,
    online_state: torch.Tensor,
    requested_actions: torch.Tensor,
    ema_current: torch.Tensor,
) -> ActionIndexedPredictions:
    """Predict all candidates through exact centered-softmax local transport."""

    _validate_spatial_float32(online_state, name="online_state")
    _validate_spatial_float32(ema_current, name="ema_current")
    if (
        ema_current.shape != online_state.shape
        or ema_current.device != online_state.device
    ):
        raise ValueError("ema_current must align exactly with online_state")
    if (
        requested_actions.shape[0] != online_state.shape[0]
        or requested_actions.device != online_state.device
        or requested_actions.dtype != torch.float32
    ):
        raise TypeError(
            "requested_actions must be aligned float32 one-hot rows"
        )
    _validate_local_transport_projector(
        projector,
        device=online_state.device,
    )
    executed_indices = requested_action_indices(requested_actions)
    batch, tokens, dim = online_state.shape

    shared_hidden = action_independent_trunk(predictor, online_state)
    shared_residual = projector.project_shared(shared_hidden)
    relative_embeddings = relative_action_embeddings(
        predictor,
        device=shared_hidden.device,
        dtype=shared_hidden.dtype,
    )
    detached_interactions = (
        shared_hidden.detach()[:, None]
        * relative_embeddings[None, :, None]
    )
    executed_interaction = (
        shared_hidden
        * relative_embeddings.index_select(
            0,
            executed_indices,
        )[:, None]
    )
    executed_mask = F.one_hot(
        executed_indices,
        num_classes=ACTION_DIM,
    ).to(dtype=torch.bool)[:, :, None, None]
    interactions = torch.where(
        executed_mask,
        executed_interaction[:, None],
        detached_interactions,
    )
    noncenter_logits = projector.project_noncenter_logits(interactions)
    center_logits = -noncenter_logits.sum(dim=-1, keepdim=True)
    all_transport_logits = torch.cat(
        (
            noncenter_logits[..., :CENTER_OFFSET_INDEX],
            center_logits,
            noncenter_logits[..., CENTER_OFFSET_INDEX:],
        ),
        dim=-1,
    )
    all_transport_probabilities = torch.softmax(
        all_transport_logits,
        dim=-1,
    )
    if (
        not bool(torch.isfinite(all_transport_logits).all())
        or not bool(torch.isfinite(all_transport_probabilities).all())
        or not bool((all_transport_probabilities > 0).all().item())
        or not bool(
            torch.allclose(
                all_transport_probabilities.sum(dim=-1),
                torch.ones_like(all_transport_probabilities[..., 0]),
                rtol=0.0,
                atol=1e-6,
            )
        )
    ):
        raise FloatingPointError(
            "student correspondence distribution is invalid"
        )
    all_transports, all_expected_offsets = _residual_local_transport(
        projector,
        ema_current,
        all_transport_probabilities,
    )
    shared_residuals = torch.where(
        executed_mask,
        shared_residual[:, None],
        shared_residual.detach()[:, None],
    )
    all_predictions = F.normalize(
        all_transports + RESIDUAL_ALPHA * shared_residuals,
        p=2,
        dim=-1,
        eps=1e-8,
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
        all_transport_logits=all_transport_logits,
        all_transport_probabilities=all_transport_probabilities,
        all_expected_offsets=all_expected_offsets,
        all_transports=all_transports,
        executed=executed,
        control_indices=control_indices,
        controls=controls,
    )


def local_correspondence_terms(
    targets: CorrespondenceTargets | torch.Tensor,
    predictions: ActionIndexedPredictions,
    row_scale: torch.Tensor,
) -> CorrespondenceTerms:
    """Compute executed ``Hc`` and its detached JEPA-row-scaled loss."""

    target_probs = (
        targets.probabilities
        if isinstance(targets, CorrespondenceTargets)
        else targets
    )
    logits = predictions.all_transport_logits
    if (
        logits.ndim != 4
        or tuple(logits.shape[1:])
        != (
            ACTION_DIM,
            TOKEN_COUNT,
            NEIGHBOR_COUNT,
        )
    ):
        raise ValueError(
            "all_transport_logits must have shape (B,9,256,9)"
        )
    batch = logits.shape[0]
    if tuple(target_probs.shape) != (
        batch,
        TOKEN_COUNT,
        NEIGHBOR_COUNT,
    ):
        raise ValueError("target probabilities must have shape (B,256,9)")
    _validate_action_indices(predictions.executed_indices, batch=batch)
    if predictions.executed_indices.device != logits.device:
        raise TypeError("executed indices must align with transport logits")
    if (
        row_scale.ndim != 1
        or tuple(row_scale.shape) != (batch,)
    ):
        raise ValueError(f"row_scale must have shape ({batch},)")
    if (
        row_scale.device != logits.device
        or row_scale.dtype != torch.float32
        or not bool(torch.isfinite(row_scale).all())
        or not bool((row_scale > 0).all().item())
    ):
        raise FloatingPointError(
            "row_scale must be aligned, finite, and positive"
        )
    executed_logits = logits.gather(
        1,
        predictions.executed_indices[:, None, None, None].expand(
            -1,
            1,
            TOKEN_COUNT,
            NEIGHBOR_COUNT,
        ),
    ).squeeze(1)
    token_cross_entropy = centered_log_soft_cross_entropy(
        target_probs,
        executed_logits,
    )
    cross_entropy_per_row = token_cross_entropy.mean(dim=1)
    centered_cross_entropy = cross_entropy_per_row.mean()
    loss = (
        row_scale.detach() * cross_entropy_per_row
    ).mean()
    return CorrespondenceTerms(
        loss=loss,
        centered_cross_entropy=centered_cross_entropy,
        cross_entropy_per_row=cross_entropy_per_row,
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
    "CENTER_OFFSET_INDEX",
    "HOLD_ACTION_INDEX",
    "NEIGHBOR_COUNT",
    "NONCENTER_NEIGHBOR_COUNT",
    "ActionIndexedLosses",
    "ActionIndexedPredictions",
    "ActionConditionedLocalCorrespondenceTransport",
    "CorrespondenceTargets",
    "CorrespondenceTerms",
    "LATENT_DIM",
    "RESIDUAL_ALPHA",
    "TOKEN_COUNT",
    "TOKEN_SIDE",
    "WHITENING_EPS",
    "WhiteningTerms",
    "action_independent_trunk",
    "action_indexed_energy_nll",
    "centered_log_soft_cross_entropy",
    "initialize_action_gate_rows",
    "local_correspondence_targets",
    "local_correspondence_terms",
    "patch_whitening_terms",
    "predict_action_conditioned_local_transports",
    "relative_action_embeddings",
    "requested_action_indices",
]
