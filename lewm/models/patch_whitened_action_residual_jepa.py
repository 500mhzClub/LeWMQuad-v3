"""V9 dense-pairwise RGB inverse head on the exact V5 latent-flow base.

This module intentionally contains no data, schedule, runner, or custody
logic.  It preserves the exact V5 State-Dependent Latent-Flow mechanism from
reviewed commit ``c93124b15387acf1fd440d281e9c4503a9e8355a`` and adds only
the V9 mechanism preregistered at source commit
``b775093897669c91d8c1b9e7d148e257881bcedf`` with preregistration file
SHA-256 beginning ``bfb0f1c2``.
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
FLOW_DIM = 2
MAXIMUM_FLOW_CELL_DISPLACEMENT = 1.0
FLOW_GRID_SCALE = 2.0 / float(TOKEN_SIDE - 1)
WHITENING_EPS = 1e-4
ACTION_GATE_INITIALIZATION_SEED = 20260712
ACTION_GATE_WEIGHT_STD = 0.01 / math.sqrt(LATENT_DIM)
ACTION_GATE_BIAS = 0.01
RESIDUAL_ALPHA = 0.1 / math.sqrt(LATENT_DIM)


class ActionIndexedPredictions(NamedTuple):
    """Uniformly ordered nine-action predictions, flows, and gathers."""

    executed_indices: torch.Tensor
    all_predictions: torch.Tensor
    all_flows_cell: torch.Tensor
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


class ActionConditionedLatentFlow(nn.Module):
    """Wrap the shared projector with one exact-zero, bias-free flow map."""

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
        self.flow_weight = nn.Parameter(
            torch.zeros(
                FLOW_DIM,
                LATENT_DIM,
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        coordinates = torch.linspace(
            -1.0,
            1.0,
            TOKEN_SIDE,
            device=reference.device,
            dtype=reference.dtype,
        )
        rows, columns = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        identity_grid_xy = torch.stack((columns, rows), dim=-1)[None]
        self.register_buffer(
            "identity_grid_xy",
            identity_grid_xy,
            persistent=False,
        )

    def project_shared(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the preserved shared projector exactly once."""

        projected = self.shared_projector(tokens)
        if projected.shape != tokens.shape:
            raise ValueError("shared projector output must align with tokens")
        return projected

    def project_flow(self, interactions: torch.Tensor) -> torch.Tensor:
        """Map state/action interactions to raw ``(x, y)`` cell offsets."""

        if (
            interactions.ndim not in {3, 4}
            or interactions.shape[-1] != LATENT_DIM
        ):
            raise ValueError(
                "flow interactions must have shape (B,N,192) "
                "or (B,9,N,192)"
            )
        if interactions.shape[0] < 1 or interactions.shape[-2] != TOKEN_COUNT:
            raise ValueError(
                f"flow interactions must contain exactly {TOKEN_COUNT} tokens"
            )
        return F.linear(interactions, self.flow_weight, bias=None)


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


def bounded_flow_cells(raw_flow: torch.Tensor) -> torch.Tensor:
    """Map raw flow to the preregistered closed one-cell range."""

    if (
        raw_flow.ndim != 4
        or tuple(raw_flow.shape[1::2]) != (ACTION_DIM, FLOW_DIM)
        or raw_flow.shape[2] != TOKEN_COUNT
    ):
        raise ValueError(
            "raw_flow must have shape (B,9,256,2)"
        )
    if not bool(torch.isfinite(raw_flow).all()):
        raise FloatingPointError("raw flow contains a nonfinite value")
    flow = MAXIMUM_FLOW_CELL_DISPLACEMENT * torch.tanh(raw_flow)
    if (
        not bool(torch.isfinite(flow).all())
        or bool((flow.abs() > MAXIMUM_FLOW_CELL_DISPLACEMENT).any())
    ):
        raise FloatingPointError("bounded flow left the closed one-cell range")
    return flow


def warp_ema_current_latents(
    prediction_projector: ActionConditionedLatentFlow,
    ema_current: torch.Tensor,
    all_flows_cell: torch.Tensor,
) -> torch.Tensor:
    """Warp detached EMA-current values on the exact row-major 16x16 grid."""

    if not isinstance(prediction_projector, ActionConditionedLatentFlow):
        raise TypeError(
            "prediction_projector must be ActionConditionedLatentFlow"
        )
    if tuple(ema_current.shape[1:]) != (TOKEN_COUNT, LATENT_DIM):
        raise ValueError(
            f"ema_current must have shape (B, {TOKEN_COUNT}, {LATENT_DIM})"
        )
    if tuple(all_flows_cell.shape) != (
        ema_current.shape[0],
        ACTION_DIM,
        TOKEN_COUNT,
        FLOW_DIM,
    ):
        raise ValueError("all_flows_cell must have shape (B,9,256,2)")
    if (
        ema_current.dtype != torch.float32
        or all_flows_cell.dtype != ema_current.dtype
        or all_flows_cell.device != ema_current.device
    ):
        raise TypeError("warp inputs must be aligned float32 tensors")
    if (
        not bool(torch.isfinite(ema_current).all())
        or not bool(torch.isfinite(all_flows_cell).all())
        or bool(
            (
                all_flows_cell.abs()
                > MAXIMUM_FLOW_CELL_DISPLACEMENT
            ).any()
        )
    ):
        raise FloatingPointError("warp inputs are nonfinite or out of bounds")

    batch = ema_current.shape[0]
    source = ema_current.detach().transpose(1, 2).reshape(
        batch,
        LATENT_DIM,
        TOKEN_SIDE,
        TOKEN_SIDE,
    )
    source = source[:, None].expand(
        -1,
        ACTION_DIM,
        -1,
        -1,
        -1,
    ).reshape(
        batch * ACTION_DIM,
        LATENT_DIM,
        TOKEN_SIDE,
        TOKEN_SIDE,
    )
    flow_grid = all_flows_cell.reshape(
        batch,
        ACTION_DIM,
        TOKEN_SIDE,
        TOKEN_SIDE,
        FLOW_DIM,
    )
    identity = prediction_projector.identity_grid_xy
    if (
        tuple(identity.shape) != (1, TOKEN_SIDE, TOKEN_SIDE, FLOW_DIM)
        or identity.device != ema_current.device
        or identity.dtype != ema_current.dtype
    ):
        raise TypeError("identity sampling grid changed or is misaligned")
    sample_grid = (
        identity[:, None]
        + FLOW_GRID_SCALE * flow_grid
    ).reshape(
        batch * ACTION_DIM,
        TOKEN_SIDE,
        TOKEN_SIDE,
        FLOW_DIM,
    )
    warped = F.grid_sample(
        source,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    warped = warped.reshape(
        batch,
        ACTION_DIM,
        LATENT_DIM,
        TOKEN_SIDE,
        TOKEN_SIDE,
    ).flatten(3).transpose(2, 3)
    if (
        tuple(warped.shape)
        != (batch, ACTION_DIM, TOKEN_COUNT, LATENT_DIM)
        or not bool(torch.isfinite(warped).all())
    ):
        raise FloatingPointError("latent warp produced invalid output")
    return warped


def flow_residual_reconstruct(
    warped_ema_current: torch.Tensor,
    shared_residuals: torch.Tensor,
) -> torch.Tensor:
    """Add the shared output-grid residual after the latent spatial warp."""

    if (
        warped_ema_current.ndim != 4
        or warped_ema_current.shape[1:] != (
            ACTION_DIM,
            TOKEN_COUNT,
            LATENT_DIM,
        )
        or shared_residuals.shape != warped_ema_current.shape
    ):
        raise ValueError(
            "warped EMA current and shared residuals must have "
            "shape (B,9,256,192)"
        )
    return F.normalize(
        warped_ema_current + RESIDUAL_ALPHA * shared_residuals,
        p=2,
        dim=-1,
        eps=1e-8,
    )


def predict_action_conditioned_flow_warps(
    predictor: nn.Module,
    prediction_projector: ActionConditionedLatentFlow,
    online_state: torch.Tensor,
    requested_actions: torch.Tensor,
    ema_current: torch.Tensor,
) -> ActionIndexedPredictions:
    """Predict all nine candidates through the shared bilinear latent flow."""

    if not isinstance(
        prediction_projector, ActionConditionedLatentFlow
    ):
        raise TypeError(
            "prediction_projector must be ActionConditionedLatentFlow"
        )
    _validate_tokens(online_state, name="online_state")
    if online_state.shape[1] != TOKEN_COUNT:
        raise ValueError(f"online_state must contain {TOKEN_COUNT} tokens")
    if online_state.shape[0] != requested_actions.shape[0]:
        raise ValueError("online_state and requested_actions batch sizes differ")
    if ema_current.shape != online_state.shape:
        raise ValueError("ema_current must align exactly with online_state")
    if (
        online_state.dtype != torch.float32
        or ema_current.dtype != torch.float32
        or requested_actions.device != online_state.device
        or ema_current.device != online_state.device
    ):
        raise TypeError("prediction inputs must be aligned float32 tensors")
    executed_indices = requested_action_indices(requested_actions)
    batch, tokens, dim = online_state.shape

    shared_hidden = action_independent_trunk(predictor, online_state)
    shared_residual = prediction_projector.project_shared(shared_hidden)
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
    all_flows_cell = bounded_flow_cells(
        prediction_projector.project_flow(interactions)
    )
    shared_residuals = torch.where(
        executed_mask,
        shared_residual[:, None],
        shared_residual.detach()[:, None],
    )
    warped_ema_current = warp_ema_current_latents(
        prediction_projector,
        ema_current,
        all_flows_cell,
    )
    all_predictions = flow_residual_reconstruct(
        warped_ema_current,
        shared_residuals,
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
        all_flows_cell=all_flows_cell,
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


DENSE_PAIRWISE_INVERSE_INITIALIZATION_SEED = 20260725
DENSE_PAIRWISE_LAYER_NORM_EPS = 1e-5
DENSE_PAIRWISE_HEAD_CHANNELS = 16
DENSE_PAIRWISE_DISPLACEMENT_BOUND = 2.0


class DensePairwiseSpatialCostVolumeInverseTerms(NamedTuple):
    """Label-blind all-pairs geometry and the executed-action inverse loss."""

    loss: torch.Tensor
    unscaled_nll: torch.Tensor
    nll_per_row: torch.Tensor
    logits: torch.Tensor
    current_next_cost_volume: torch.Tensor
    current_current_cost_volume: torch.Tensor
    current_next_probabilities: torch.Tensor
    current_current_probabilities: torch.Tensor
    probability_difference: torch.Tensor
    volume: torch.Tensor
    displacement: torch.Tensor


class DensePairwiseSpatialCostVolumeInverseHead(nn.Module):
    """Aggregate the complete label-blind 256-by-256 spatial match volume."""

    def __init__(self) -> None:
        super().__init__()
        cpu_rng_state = torch.random.get_rng_state().clone()
        accelerator_rng_states = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized()
            else None
        )
        try:
            self.channel_projection = nn.Conv2d(
                TOKEN_COUNT,
                DENSE_PAIRWISE_HEAD_CHANNELS,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            )
            self.channel_activation = nn.GELU(approximate="none")
            self.spatial_projection = nn.Conv2d(
                DENSE_PAIRWISE_HEAD_CHANNELS,
                DENSE_PAIRWISE_HEAD_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                device="cpu",
                dtype=torch.float32,
            )
            self.spatial_activation = nn.GELU(approximate="none")
            self.pool = nn.AvgPool2d(
                kernel_size=4,
                stride=4,
                padding=0,
            )
            self.classifier = nn.Linear(
                DENSE_PAIRWISE_HEAD_CHANNELS * 4 * 4,
                ACTION_DIM,
                bias=True,
                device="cpu",
                dtype=torch.float32,
            )

            axis_cpu = torch.linspace(
                -1.0,
                1.0,
                TOKEN_SIDE,
                device="cpu",
                dtype=torch.float32,
            )
            rows_y, columns_x = torch.meshgrid(
                axis_cpu,
                axis_cpu,
                indexing="ij",
            )
            coordinates_yx = torch.stack(
                [rows_y.flatten(), columns_x.flatten()],
                dim=-1,
            )
            self.register_buffer(
                "coordinates_yx",
                coordinates_yx,
                persistent=False,
            )

            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                DENSE_PAIRWISE_INVERSE_INITIALIZATION_SEED
            )
            with torch.no_grad():
                nn.init.kaiming_normal_(
                    self.channel_projection.weight,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu",
                    generator=generator,
                )
                nn.init.kaiming_normal_(
                    self.spatial_projection.weight,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu",
                    generator=generator,
                )
                nn.init.normal_(
                    self.classifier.weight,
                    mean=0.0,
                    std=1.0 / 16.0,
                    generator=generator,
                )
                self.classifier.bias.zero_()
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if accelerator_rng_states is not None:
                torch.cuda.set_rng_state_all(accelerator_rng_states)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Return nine action logits from the exact full spatial volume."""

        if volume.ndim != 4 or tuple(volume.shape[1:]) != (
            TOKEN_COUNT,
            TOKEN_SIDE,
            TOKEN_SIDE,
        ):
            raise ValueError(
                "volume must have shape (B,256,16,16)"
            )
        if volume.shape[0] < 1:
            raise ValueError("volume must contain at least one row")
        if volume.dtype != torch.float32:
            raise TypeError("volume must have dtype torch.float32")
        if volume.device != self.channel_projection.weight.device:
            raise TypeError("volume and inverse head must share a device")
        if not bool(torch.isfinite(volume).all()):
            raise FloatingPointError("volume contains a nonfinite value")

        hidden = self.channel_projection(volume)
        hidden = self.channel_activation(hidden)
        hidden = self.spatial_projection(hidden)
        hidden = self.spatial_activation(hidden)
        pooled = self.pool(hidden)
        if tuple(pooled.shape[1:]) != (
            DENSE_PAIRWISE_HEAD_CHANNELS,
            4,
            4,
        ):
            raise RuntimeError("inverse head pooling shape changed")
        logits = self.classifier(pooled.flatten(start_dim=1))
        if tuple(logits.shape) != (volume.shape[0], ACTION_DIM):
            raise RuntimeError("inverse head logits shape changed")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("inverse head logits are nonfinite")
        return logits


def _dense_pairwise_probability_difference_volume(
    probability_difference: torch.Tensor,
) -> torch.Tensor:
    """Map exact ``[source,target]`` axes to ``[target,source_y,source_x]``."""

    if probability_difference.ndim != 3 or tuple(
        probability_difference.shape[1:]
    ) != (TOKEN_COUNT, TOKEN_COUNT):
        raise ValueError(
            "probability_difference must have shape (B,256,256)"
        )
    if probability_difference.shape[0] < 1:
        raise ValueError(
            "probability_difference must contain at least one row"
        )
    if probability_difference.dtype != torch.float32:
        raise TypeError(
            "probability_difference must have dtype torch.float32"
        )
    if not bool(torch.isfinite(probability_difference).all()) or bool(
        (probability_difference < -1.0).any()
    ) or bool((probability_difference > 1.0).any()):
        raise FloatingPointError(
            "pairwise probability difference left [-1,1]"
        )
    return probability_difference.transpose(1, 2).reshape(
        probability_difference.shape[0],
        TOKEN_COUNT,
        TOKEN_SIDE,
        TOKEN_SIDE,
    ).contiguous()


def dense_pairwise_spatial_cost_volume_inverse_terms(
    head: DensePairwiseSpatialCostVolumeInverseHead,
    online_current: torch.Tensor,
    online_next: torch.Tensor,
    executed_indices: torch.Tensor,
    row_scale: torch.Tensor,
) -> DensePairwiseSpatialCostVolumeInverseTerms:
    """Build the exact V9 all-pairs volume and detached-scale inverse NLL."""

    if not isinstance(
        head,
        DensePairwiseSpatialCostVolumeInverseHead,
    ):
        raise TypeError(
            "head must be DensePairwiseSpatialCostVolumeInverseHead"
        )
    _validate_tokens(online_current, name="online_current")
    _validate_tokens(online_next, name="online_next")
    if tuple(online_current.shape[1:]) != (TOKEN_COUNT, LATENT_DIM):
        raise ValueError(
            "online_current must have shape (B,256,192)"
        )
    if online_next.shape != online_current.shape:
        raise ValueError("online_next must align exactly with online_current")
    if (
        online_current.dtype != torch.float32
        or online_next.dtype != torch.float32
    ):
        raise TypeError("online states must have dtype torch.float32")
    if online_next.device != online_current.device:
        raise TypeError("online states must share a device")
    if not bool(torch.isfinite(online_current).all()) or not bool(
        torch.isfinite(online_next).all()
    ):
        raise FloatingPointError("online states contain a nonfinite value")

    batch = online_current.shape[0]
    _validate_action_indices(executed_indices, batch=batch)
    if executed_indices.device != online_current.device:
        raise TypeError("action indices and online states must share a device")
    if tuple(row_scale.shape) != (batch,):
        raise ValueError(f"row_scale must have shape ({batch},)")
    if (
        row_scale.dtype != torch.float32
        or row_scale.device != online_current.device
    ):
        raise TypeError("row_scale must be aligned float32")
    if not bool(torch.isfinite(row_scale).all()) or not bool(
        (row_scale > 0).all()
    ):
        raise ValueError("row_scale must be finite and strictly positive")

    normalized_current = F.layer_norm(
        online_current,
        (LATENT_DIM,),
        weight=None,
        bias=None,
        eps=DENSE_PAIRWISE_LAYER_NORM_EPS,
    )
    normalized_next = F.layer_norm(
        online_next,
        (LATENT_DIM,),
        weight=None,
        bias=None,
        eps=DENSE_PAIRWISE_LAYER_NORM_EPS,
    )
    similarity_scale = math.sqrt(float(LATENT_DIM))
    current_next_cost_volume = torch.matmul(
        normalized_current,
        normalized_next.transpose(1, 2),
    ) / similarity_scale
    current_current_cost_volume = torch.matmul(
        normalized_current,
        normalized_current.transpose(1, 2),
    ) / similarity_scale
    if (
        tuple(current_next_cost_volume.shape)
        != (batch, TOKEN_COUNT, TOKEN_COUNT)
        or current_current_cost_volume.shape
        != current_next_cost_volume.shape
    ):
        raise RuntimeError("dense pairwise cost-volume shape changed")
    if not bool(torch.isfinite(current_next_cost_volume).all()) or not bool(
        torch.isfinite(current_current_cost_volume).all()
    ):
        raise FloatingPointError("dense pairwise cost volume is nonfinite")

    current_next_probabilities = F.softmax(
        current_next_cost_volume,
        dim=-1,
    )
    current_current_probabilities = F.softmax(
        current_current_cost_volume,
        dim=-1,
    )
    probability_difference = (
        current_next_probabilities - current_current_probabilities
    )
    if not bool(torch.isfinite(probability_difference).all()) or bool(
        (probability_difference < -1.0).any()
    ) or bool((probability_difference > 1.0).any()):
        raise FloatingPointError(
            "pairwise probability difference left [-1,1]"
        )

    volume = _dense_pairwise_probability_difference_volume(
        probability_difference
    )
    channel_sum = volume.sum(dim=1)
    if not torch.allclose(
        channel_sum,
        torch.zeros_like(channel_sum),
        rtol=0.0,
        atol=1e-6,
    ):
        raise FloatingPointError(
            "pairwise volume violated channel conservation"
        )

    coordinates_yx = head.coordinates_yx
    if (
        tuple(coordinates_yx.shape) != (TOKEN_COUNT, FLOW_DIM)
        or coordinates_yx.dtype != torch.float32
        or coordinates_yx.device != online_current.device
    ):
        raise TypeError(
            "inverse-head coordinates must be aligned float32 [256,2]"
        )
    displacement = torch.matmul(
        probability_difference,
        coordinates_yx,
    ).reshape(
        batch,
        TOKEN_SIDE,
        TOKEN_SIDE,
        FLOW_DIM,
    ).permute(0, 3, 1, 2).contiguous()
    if not bool(torch.isfinite(displacement).all()) or bool(
        (displacement < -DENSE_PAIRWISE_DISPLACEMENT_BOUND).any()
    ) or bool(
        (displacement > DENSE_PAIRWISE_DISPLACEMENT_BOUND).any()
    ):
        raise FloatingPointError(
            "diagnostic displacement left [-2,2]"
        )

    logits = head(volume)
    nll_per_row = F.cross_entropy(
        logits,
        executed_indices,
        reduction="none",
    )
    unscaled_nll = nll_per_row.mean()
    loss = (
        row_scale.detach() * nll_per_row
    ).mean()
    if not bool(torch.isfinite(loss)) or not bool(
        torch.isfinite(unscaled_nll)
    ):
        raise FloatingPointError("dense inverse loss is nonfinite")

    return DensePairwiseSpatialCostVolumeInverseTerms(
        loss=loss,
        unscaled_nll=unscaled_nll,
        nll_per_row=nll_per_row,
        logits=logits,
        current_next_cost_volume=current_next_cost_volume,
        current_current_cost_volume=current_current_cost_volume,
        current_next_probabilities=current_next_probabilities,
        current_current_probabilities=current_current_probabilities,
        probability_difference=probability_difference,
        volume=volume,
        displacement=displacement,
    )


__all__ = [
    "ACTION_DIM",
    "ACTION_GATE_BIAS",
    "ACTION_GATE_INITIALIZATION_SEED",
    "ACTION_GATE_WEIGHT_STD",
    "FLOW_DIM",
    "FLOW_GRID_SCALE",
    "HOLD_ACTION_INDEX",
    "ActionIndexedLosses",
    "ActionIndexedPredictions",
    "ActionConditionedLatentFlow",
    "LATENT_DIM",
    "MAXIMUM_FLOW_CELL_DISPLACEMENT",
    "RESIDUAL_ALPHA",
    "TOKEN_COUNT",
    "TOKEN_SIDE",
    "WHITENING_EPS",
    "WhiteningTerms",
    "action_independent_trunk",
    "action_indexed_energy_nll",
    "bounded_flow_cells",
    "flow_residual_reconstruct",
    "initialize_action_gate_rows",
    "patch_whitening_terms",
    "predict_action_conditioned_flow_warps",
    "relative_action_embeddings",
    "requested_action_indices",
    "warp_ema_current_latents",
    "DENSE_PAIRWISE_DISPLACEMENT_BOUND",
    "DENSE_PAIRWISE_HEAD_CHANNELS",
    "DENSE_PAIRWISE_INVERSE_INITIALIZATION_SEED",
    "DENSE_PAIRWISE_LAYER_NORM_EPS",
    "DensePairwiseSpatialCostVolumeInverseHead",
    "DensePairwiseSpatialCostVolumeInverseTerms",
    "dense_pairwise_spatial_cost_volume_inverse_terms",
]
