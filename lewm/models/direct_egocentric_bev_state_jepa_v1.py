"""Direct RGB-to-BEV-state JEPA V1.

The causal path is deliberately narrow::

    current RGB -> online encoder/decoder/head -> three state logits
    three state logits + executed one-hot action -> predicted next logits

The three UNKNOWN/FREE/OCCUPIED channels are the only state consumed by the
predictor.  Target images are encoded only by the detached EMA stack, while
the online next-image call is used only by the hard grounding term.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder


UNKNOWN_CLASS_V1 = 0
FREE_CLASS_V1 = 1
OCCUPIED_CLASS_V1 = 2
ACTION_VOCABULARY_V1 = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
HOLD_ACTION_INDEX_V1 = 6


@dataclass(frozen=True)
class DirectEgocentricBevStateJepaV1Config:
    image_size: int = 112
    patch_size: int = 7
    encoder_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    encoder_dropout: float = 0.0
    bev_dim: int = 64
    bev_size: tuple[int, int] = (64, 64)
    forward_range_m: tuple[float, float] = (-0.95, 5.35)
    left_range_m: tuple[float, float] = (-3.15, 3.15)
    bev_attention_heads: int = 4
    state_classes: int = 3
    action_dim: int = 9
    predictor_hidden_dim: int = 128
    target_ema_momentum: float = 0.996
    initialization_seed: int = 20260712

    def __post_init__(self) -> None:
        expected = {
            "image_size": 112,
            "patch_size": 7,
            "encoder_dim": 192,
            "encoder_depth": 6,
            "encoder_heads": 6,
            "encoder_mlp_ratio": 4,
            "encoder_dropout": 0.0,
            "bev_dim": 64,
            "bev_size": (64, 64),
            "forward_range_m": (-0.95, 5.35),
            "left_range_m": (-3.15, 3.15),
            "bev_attention_heads": 4,
            "state_classes": 3,
            "action_dim": 9,
            "predictor_hidden_dim": 128,
            "target_ema_momentum": 0.996,
            "initialization_seed": 20260712,
        }
        changed = [
            name for name, value in expected.items() if getattr(self, name) != value
        ]
        if changed:
            raise ValueError(
                "Direct BEV-state V1 constants cannot change: "
                + ", ".join(changed)
            )


class HierarchicalHardLossV1(NamedTuple):
    total: torch.Tensor
    occupied: torch.Tensor
    free_given_not_occupied: torch.Tensor


class DirectBevStateObjectiveV1(NamedTuple):
    total: torch.Tensor
    G: torch.Tensor
    J: torch.Tensor
    C: torch.Tensor
    G_current: torch.Tensor
    G_next: torch.Tensor
    current_state_logits: torch.Tensor
    next_online_state_logits: torch.Tensor
    executed_prediction_logits: torch.Tensor
    all_action_prediction_logits: torch.Tensor
    target_next_logits: torch.Tensor
    target_current_logits: torch.Tensor
    target_mapped_negative_logits: torch.Tensor
    action_energies: torch.Tensor
    action_logits: torch.Tensor
    action_nll_per_row: torch.Tensor
    executed_energy: torch.Tensor
    mapped_negative_energy: torch.Tensor
    current_target_energy: torch.Tensor
    candidate_energies: torch.Tensor
    candidate_logits: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_counts: torch.Tensor
    candidate_energy_scale: torch.Tensor
    conditional_nce_per_row: torch.Tensor


class WrongRgbGroundingControlV1(NamedTuple):
    correct_next_loss_per_row: torch.Tensor
    mapped_negative_loss_per_row: torch.Tensor
    correct_next_state_logits: torch.Tensor
    mapped_negative_state_logits: torch.Tensor


def _construct_n320_encoder_without_rng_draw() -> VisionEncoder:
    caller_rng = torch.random.get_rng_state()
    try:
        encoder = VisionEncoder(
            image_size=112,
            patch_size=7,
            hidden_dim=192,
            depth=6,
            n_heads=6,
            mlp_ratio=4,
            dropout=0.0,
        )
    finally:
        torch.random.set_rng_state(caller_rng)
    return encoder


def _validate_n320_encoder_state(
    encoder: VisionEncoder,
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = encoder.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(
            f"N320 encoder state keys changed; missing={missing}, extra={extra}"
        )
    for name, expected_tensor in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"N320 state {name!r} is not a tensor")
        if value.shape != expected_tensor.shape:
            raise ValueError(
                f"N320 state {name!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(expected_tensor.shape)}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"N320 state {name!r} must be exact float32")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"N320 state {name!r} is nonfinite")


class _GlobalCrossAttentionBevDecoderV1(nn.Module):
    """Learned global image-token to fixed-lattice BEV decoder."""

    def __init__(self, config: DirectEgocentricBevStateJepaV1Config) -> None:
        super().__init__()
        token_side = config.image_size // config.patch_size
        self.token_count = token_side * token_side
        self.bev_size = config.bev_size

        forward = torch.linspace(*config.forward_range_m, config.bev_size[0])
        left = torch.linspace(*config.left_range_m, config.bev_size[1])
        forward_grid, left_grid = torch.meshgrid(forward, left, indexing="ij")
        forward_grid = forward_grid / max(abs(value) for value in config.forward_range_m)
        left_grid = left_grid / max(abs(value) for value in config.left_range_m)
        coordinates = torch.stack(
            (
                forward_grid,
                left_grid,
                torch.sin(math.pi * forward_grid),
                torch.cos(math.pi * forward_grid),
                torch.sin(math.pi * left_grid),
                torch.cos(math.pi * left_grid),
            ),
            dim=-1,
        ).reshape(-1, 6)
        self.register_buffer("coordinate_features", coordinates, persistent=True)
        self.coordinate_query = nn.Sequential(
            nn.Linear(6, config.bev_dim),
            nn.GELU(),
            nn.Linear(config.bev_dim, config.bev_dim),
        )
        self.query_bias = nn.Parameter(
            torch.empty(config.bev_size[0] * config.bev_size[1], config.bev_dim)
        )
        nn.init.trunc_normal_(self.query_bias, std=0.02)
        self.token_project = nn.Linear(config.encoder_dim, config.bev_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.bev_dim,
            config.bev_attention_heads,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(config.bev_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(config.bev_dim, config.bev_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(config.bev_dim, config.bev_dim, 3, padding=1),
            nn.GroupNorm(1, config.bev_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.shape[1] != self.token_count:
            raise ValueError(
                f"patch_tokens must have shape (B,{self.token_count},D)"
            )
        tokens = self.token_project(patch_tokens)
        queries = self.coordinate_query(
            self.coordinate_features.to(dtype=patch_tokens.dtype)
        )
        queries = queries + self.query_bias.to(dtype=queries.dtype)
        queries = queries[None].expand(patch_tokens.shape[0], -1, -1)
        attended, _ = self.cross_attention(
            queries,
            tokens,
            tokens,
            need_weights=False,
        )
        features = self.query_norm(queries + attended)
        features = features.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            -1,
            self.bev_size[0],
            self.bev_size[1],
        )
        return self.refine(features)


def _validate_action_one_hot(
    action_one_hot: torch.Tensor,
    *,
    batch: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    if action_one_hot.shape != (batch, 9):
        raise ValueError("action_one_hot must have shape (B,9)")
    if action_one_hot.dtype != reference.dtype or not action_one_hot.is_floating_point():
        raise TypeError("action_one_hot and state must share a floating dtype")
    if action_one_hot.device != reference.device:
        raise TypeError("action_one_hot and state must share a device")
    if not bool(torch.isfinite(action_one_hot).all()):
        raise FloatingPointError("action_one_hot is nonfinite")
    if not bool(((action_one_hot == 0.0) | (action_one_hot == 1.0)).all()):
        raise ValueError("action_one_hot must contain exact zeros and ones")
    if not torch.equal(
        action_one_hot.sum(dim=1),
        torch.ones(batch, dtype=reference.dtype, device=reference.device),
    ):
        raise ValueError("each action row must contain exactly one active action")
    return action_one_hot.argmax(dim=1)


class _ActionOnlyResidualPredictorV1(nn.Module):
    """Residual transition whose public inputs are state and action only."""

    def __init__(self) -> None:
        super().__init__()
        self.condition = nn.Sequential(
            nn.Linear(9 + 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        self.net = nn.Sequential(
            nn.Conv2d(3 * 2, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 3, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        current_state_logits: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        if current_state_logits.ndim != 4 or current_state_logits.shape[1] != 3:
            raise ValueError("current_state_logits must have shape (B,3,H,W)")
        _validate_action_one_hot(
            action_one_hot,
            batch=current_state_logits.shape[0],
            reference=current_state_logits,
        )
        legacy_zero = current_state_logits.new_zeros(
            current_state_logits.shape[0], 3
        )
        condition = self.condition(torch.cat((action_one_hot, legacy_zero), dim=1))
        condition = condition[:, :, None, None].expand_as(current_state_logits)
        residual = self.net(torch.cat((current_state_logits, condition), dim=1))
        return current_state_logits + residual


def _balanced_probability_bce(
    probability: torch.Tensor,
    positive_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    group_losses: list[torch.Tensor] = []
    for state in (False, True):
        mask = valid_mask & (positive_target == state)
        if bool(mask.any()):
            group_losses.append(
                F.binary_cross_entropy(
                    probability[mask],
                    positive_target[mask].to(dtype=probability.dtype),
                    reduction="mean",
                )
            )
    return (
        torch.stack(group_losses).mean()
        if group_losses
        else probability.sum() * 0.0
    )


def hard_hierarchical_raster_loss_v1(
    state_logits: torch.Tensor,
    target_labels: torch.Tensor,
) -> HierarchicalHardLossV1:
    """Exact V4 OCCUPIED-first then FREE-vs-UNKNOWN reduction."""

    if state_logits.ndim != 4 or state_logits.shape[1] != 3:
        raise ValueError("state_logits must have shape (B,3,H,W)")
    expected = state_logits.shape[:1] + state_logits.shape[2:]
    if target_labels.shape != expected:
        raise ValueError("target_labels must have shape (B,H,W)")
    if target_labels.device != state_logits.device:
        raise TypeError("labels and state logits must share a device")
    if target_labels.is_floating_point() or target_labels.dtype == torch.bool:
        raise TypeError("target_labels must use an integer dtype")
    supported = (
        (target_labels == UNKNOWN_CLASS_V1)
        | (target_labels == FREE_CLASS_V1)
        | (target_labels == OCCUPIED_CLASS_V1)
    )
    if not bool(supported.all()):
        raise ValueError("target_labels contain a class outside UNKNOWN/FREE/OCCUPIED")

    probabilities = torch.softmax(state_logits, dim=1)
    epsilon = torch.finfo(probabilities.dtype).eps
    occupied_target = target_labels == OCCUPIED_CLASS_V1
    occupied_probability = probabilities[:, OCCUPIED_CLASS_V1].clamp(
        epsilon, 1.0 - epsilon
    )
    occupied = _balanced_probability_bce(
        occupied_probability,
        occupied_target,
        torch.ones_like(occupied_target),
    )
    non_occupied_probability = (
        probabilities[:, UNKNOWN_CLASS_V1] + probabilities[:, FREE_CLASS_V1]
    )
    conditional_free = (
        probabilities[:, FREE_CLASS_V1]
        / non_occupied_probability.clamp_min(epsilon)
    ).clamp(epsilon, 1.0 - epsilon)
    free = _balanced_probability_bce(
        conditional_free,
        target_labels == FREE_CLASS_V1,
        ~occupied_target,
    )
    return HierarchicalHardLossV1(0.5 * occupied + 0.5 * free, occupied, free)


def _hard_hierarchical_loss_per_row(
    state_logits: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            hard_hierarchical_raster_loss_v1(
                state_logits[index : index + 1],
                target_labels[index : index + 1],
            ).total
            for index in range(state_logits.shape[0])
        ]
    )


def soft_hierarchical_state_energy_v1(
    predicted_logits: torch.Tensor,
    detached_target_logits: torch.Tensor,
) -> torch.Tensor:
    """Return one soft hierarchical energy for each leading state row."""

    if predicted_logits.shape != detached_target_logits.shape:
        raise ValueError("predicted and target state shapes differ")
    if predicted_logits.ndim < 4 or predicted_logits.shape[-3] != 3:
        raise ValueError("state tensors must end in shape (3,H,W)")
    target_logits = detached_target_logits.detach()
    predicted = torch.softmax(predicted_logits, dim=-3)
    target = torch.softmax(target_logits, dim=-3)
    epsilon = torch.finfo(predicted.dtype).eps

    predicted_occupied = predicted[..., OCCUPIED_CLASS_V1, :, :].clamp(
        epsilon, 1.0 - epsilon
    )
    target_occupied = target[..., OCCUPIED_CLASS_V1, :, :]
    occupied = F.binary_cross_entropy(
        predicted_occupied,
        target_occupied,
        reduction="none",
    ).mean(dim=(-2, -1))

    predicted_non_occupied = (
        predicted[..., UNKNOWN_CLASS_V1, :, :]
        + predicted[..., FREE_CLASS_V1, :, :]
    )
    target_non_occupied = 1.0 - target_occupied
    predicted_conditional_free = (
        predicted[..., FREE_CLASS_V1, :, :]
        / predicted_non_occupied.clamp_min(epsilon)
    ).clamp(epsilon, 1.0 - epsilon)
    target_conditional_free = (
        target[..., FREE_CLASS_V1, :, :]
        / target_non_occupied.clamp_min(epsilon)
    )
    free_per_cell = F.binary_cross_entropy(
        predicted_conditional_free,
        target_conditional_free,
        reduction="none",
    )
    free = (free_per_cell * target_non_occupied).sum(dim=(-2, -1))
    free = free / target_non_occupied.sum(dim=(-2, -1)).clamp_min(
        torch.finfo(predicted.dtype).tiny
    )
    return 0.5 * occupied + 0.5 * free


def direct_bev_state_objective_v1(
    *,
    current_state_logits: torch.Tensor,
    next_online_state_logits: torch.Tensor,
    all_action_prediction_logits: torch.Tensor,
    target_next_logits: torch.Tensor,
    target_current_logits: torch.Tensor,
    target_mapped_negative_logits: torch.Tensor,
    current_labels: torch.Tensor,
    next_labels: torch.Tensor,
    executed_action_indices: torch.Tensor,
    non_hold_mask: torch.Tensor,
) -> DirectBevStateObjectiveV1:
    """Compute the frozen G + J + joint conditional-NCE objective."""

    if current_state_logits.ndim != 4 or current_state_logits.shape[1] != 3:
        raise ValueError("current_state_logits must have shape (B,3,H,W)")
    batch = current_state_logits.shape[0]
    state_shape = current_state_logits.shape
    if any(
        value.shape != state_shape
        for value in (
            next_online_state_logits,
            target_next_logits,
            target_current_logits,
            target_mapped_negative_logits,
        )
    ):
        raise ValueError("online and target state shapes differ")
    if all_action_prediction_logits.shape != (
        batch,
        9,
        3,
        state_shape[2],
        state_shape[3],
    ):
        raise ValueError("all_action_prediction_logits must have shape (B,9,3,H,W)")
    if executed_action_indices.shape != (batch,) or executed_action_indices.dtype != torch.long:
        raise TypeError("executed_action_indices must be long with shape (B,)")
    if executed_action_indices.device != current_state_logits.device:
        raise TypeError("executed action indices and states must share a device")
    if bool((executed_action_indices < 0).any()) or bool(
        (executed_action_indices >= 9).any()
    ):
        raise ValueError("executed action indices must lie in [0,8]")
    if non_hold_mask.shape != (batch,) or non_hold_mask.dtype != torch.bool:
        raise TypeError("non_hold_mask must be boolean with shape (B,)")
    if non_hold_mask.device != current_state_logits.device:
        raise TypeError("non_hold_mask and states must share a device")
    if not torch.equal(
        non_hold_mask,
        executed_action_indices != HOLD_ACTION_INDEX_V1,
    ):
        raise ValueError("non_hold_mask differs from the executed action identities")

    current_grounding = _hard_hierarchical_loss_per_row(
        current_state_logits, current_labels
    ).mean()
    next_grounding = _hard_hierarchical_loss_per_row(
        next_online_state_logits, next_labels
    ).mean()
    grounding = 0.5 * (current_grounding + next_grounding)

    target_next = target_next_logits.detach()
    target_current = target_current_logits.detach()
    target_mapped = target_mapped_negative_logits.detach()
    target_next_all = target_next[:, None].expand_as(all_action_prediction_logits)
    action_energies = soft_hierarchical_state_energy_v1(
        all_action_prediction_logits,
        target_next_all,
    )
    rows = torch.arange(batch, device=current_state_logits.device)
    executed_prediction = all_action_prediction_logits[
        rows, executed_action_indices
    ]
    executed_energy = action_energies[rows, executed_action_indices]
    jepa = executed_energy.mean()

    action_is_wrong = torch.ones(
        batch, 9, dtype=torch.bool, device=current_state_logits.device
    )
    action_is_wrong[rows, executed_action_indices] = False
    wrong_action_energies = action_energies[action_is_wrong].reshape(batch, 8)
    mapped_energy = soft_hierarchical_state_energy_v1(
        executed_prediction,
        target_mapped,
    )
    current_energy = soft_hierarchical_state_energy_v1(
        executed_prediction,
        target_current,
    )
    candidate_energies = torch.cat(
        (
            executed_energy[:, None],
            wrong_action_energies,
            mapped_energy[:, None],
            current_energy[:, None],
        ),
        dim=1,
    )
    candidate_mask = torch.ones(
        batch, 11, dtype=torch.bool, device=current_state_logits.device
    )
    candidate_mask[:, -1] = non_hold_mask
    candidate_counts = candidate_mask.sum(dim=1)
    candidate_energy_scale = (
        (candidate_energies * candidate_mask.to(candidate_energies.dtype)).sum(dim=1)
        / candidate_counts.to(candidate_energies.dtype)
    ).detach().clamp_min(1e-6)
    candidate_logits = -candidate_energies / candidate_energy_scale[:, None]
    candidate_logits = candidate_logits.masked_fill(~candidate_mask, -torch.inf)
    conditional_nce_per_row = (
        torch.logsumexp(candidate_logits, dim=1) - candidate_logits[:, 0]
    ) / torch.log(candidate_counts.to(candidate_logits.dtype))
    conditional_nce = conditional_nce_per_row.mean()

    action_logits = -action_energies / candidate_energy_scale[:, None]
    action_nll_per_row = F.cross_entropy(
        action_logits,
        executed_action_indices,
        reduction="none",
    )
    total = (
        grounding / math.log(2.0)
        + jepa / math.log(2.0)
        + conditional_nce
    )
    return DirectBevStateObjectiveV1(
        total=total,
        G=grounding,
        J=jepa,
        C=conditional_nce,
        G_current=current_grounding,
        G_next=next_grounding,
        current_state_logits=current_state_logits,
        next_online_state_logits=next_online_state_logits,
        executed_prediction_logits=executed_prediction,
        all_action_prediction_logits=all_action_prediction_logits,
        target_next_logits=target_next,
        target_current_logits=target_current,
        target_mapped_negative_logits=target_mapped,
        action_energies=action_energies,
        action_logits=action_logits,
        action_nll_per_row=action_nll_per_row,
        executed_energy=executed_energy,
        mapped_negative_energy=mapped_energy,
        current_target_energy=current_energy,
        candidate_energies=candidate_energies,
        candidate_logits=candidate_logits,
        candidate_mask=candidate_mask,
        candidate_counts=candidate_counts,
        candidate_energy_scale=candidate_energy_scale,
        conditional_nce_per_row=conditional_nce_per_row,
    )


class DirectEgocentricBevStateJepaV1(nn.Module):
    """Frozen direct three-state BEV JEPA mechanism."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: DirectEgocentricBevStateJepaV1Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DirectEgocentricBevStateJepaV1Config()
        self.encoder = _construct_n320_encoder_without_rng_draw()
        _validate_n320_encoder_state(self.encoder, n320_encoder_state_dict)
        self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)

        caller_rng = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(self.config.initialization_seed)
            self.bev_decoder = _GlobalCrossAttentionBevDecoderV1(self.config)
            self.state_head = nn.Conv2d(
                self.config.bev_dim,
                self.config.state_classes,
                kernel_size=1,
            )
            self.predictor = _ActionOnlyResidualPredictorV1()
        finally:
            torch.random.set_rng_state(caller_rng)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_decoder = copy.deepcopy(self.bev_decoder)
        self.target_state_head = copy.deepcopy(self.state_head)
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.hard_sync_target_from_online()

    @property
    def action_vocabulary(self) -> tuple[str, ...]:
        return ACTION_VOCABULARY_V1

    def _target_modules(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.target_encoder, self.target_bev_decoder, self.target_state_head

    def _online_modules(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.encoder, self.bev_decoder, self.state_head

    def _freeze_target(self) -> None:
        for module in self._target_modules():
            module.requires_grad_(False)
            module.eval()

    def train(self, mode: bool = True) -> DirectEgocentricBevStateJepaV1:
        super().train(mode)
        self._freeze_target()
        return self

    @torch.no_grad()
    def hard_sync_target_from_online(self) -> None:
        for target, online in zip(
            self._target_modules(), self._online_modules(), strict=True
        ):
            target.load_state_dict(online.state_dict(), strict=True)
        self.ema_update_count.zero_()
        self._freeze_target()

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        momentum = self.config.target_ema_momentum
        for target_module, online_module in zip(
            self._target_modules(), self._online_modules(), strict=True
        ):
            target_parameters = dict(target_module.named_parameters())
            online_parameters = dict(online_module.named_parameters())
            if target_parameters.keys() != online_parameters.keys():
                raise RuntimeError("online and target parameter inventories differ")
            for name, target in target_parameters.items():
                target.mul_(momentum).add_(
                    online_parameters[name], alpha=1.0 - momentum
                )
            target_buffers = dict(target_module.named_buffers())
            online_buffers = dict(online_module.named_buffers())
            if target_buffers.keys() != online_buffers.keys():
                raise RuntimeError("online and target buffer inventories differ")
            for name, target in target_buffers.items():
                target.copy_(online_buffers[name])
        self.ema_update_count.add_(1)
        self._freeze_target()

    def _validate_rgb(self, rgb: torch.Tensor, *, name: str) -> None:
        expected = (3, self.config.image_size, self.config.image_size)
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(f"{name} must have shape (B,{expected[0]},112,112)")
        if rgb.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError(f"{name} must use exact float32")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError(f"{name} is nonfinite")
        if rgb.device != next(self.parameters()).device:
            raise TypeError(f"{name} and model must share a device")

    @staticmethod
    def _encode_state(
        rgb: torch.Tensor,
        encoder: VisionEncoder,
        decoder: _GlobalCrossAttentionBevDecoderV1,
        state_head: nn.Conv2d,
    ) -> torch.Tensor:
        patch_tokens = encoder.forward_tokens(rgb)[:, 1:]
        return state_head(decoder(patch_tokens))

    def online_state(self, rgb: torch.Tensor) -> torch.Tensor:
        self._validate_rgb(rgb, name="online_rgb")
        return self._encode_state(rgb, self.encoder, self.bev_decoder, self.state_head)

    @torch.no_grad()
    def target_state(self, rgb: torch.Tensor) -> torch.Tensor:
        self._validate_rgb(rgb, name="target_rgb")
        return self._encode_state(
            rgb,
            self.target_encoder,
            self.target_bev_decoder,
            self.target_state_head,
        ).detach()

    def predict_state(
        self,
        current_state_logits: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(current_state_logits, action_one_hot)

    def predict_all_actions_from_state(
        self,
        current_state_logits: torch.Tensor,
    ) -> torch.Tensor:
        if current_state_logits.ndim != 4 or current_state_logits.shape[1] != 3:
            raise ValueError("current_state_logits must have shape (B,3,H,W)")
        batch = current_state_logits.shape[0]
        all_states = current_state_logits[:, None].expand(
            -1, 9, -1, -1, -1
        ).reshape(batch * 9, 3, *current_state_logits.shape[2:])
        actions = torch.eye(
            9,
            dtype=current_state_logits.dtype,
            device=current_state_logits.device,
        )[None].expand(batch, -1, -1).reshape(batch * 9, 9)
        return self.predict_state(all_states, actions).reshape(
            batch, 9, 3, *current_state_logits.shape[2:]
        )

    def predict_from_rgb(
        self,
        current_rgb: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_state(self.online_state(current_rgb), action_one_hot)

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
        """Run exactly the five training O/T state calls; no diagnostic O call."""

        if next_rgb.shape != current_rgb.shape or fixed_negative_rgb.shape != current_rgb.shape:
            raise ValueError("current, next, and fixed-negative RGB shapes differ")
        executed = _validate_action_one_hot(
            action_one_hot,
            batch=current_rgb.shape[0],
            reference=current_rgb,
        )
        if non_hold_mask.shape != (current_rgb.shape[0],) or non_hold_mask.dtype != torch.bool:
            raise TypeError("non_hold_mask must be boolean with shape (B,)")
        if not torch.equal(non_hold_mask, executed != HOLD_ACTION_INDEX_V1):
            raise ValueError("non_hold_mask differs from executed actions")

        current_state = self.online_state(current_rgb)
        next_online_state = self.online_state(next_rgb)
        target_next = self.target_state(next_rgb)
        target_current = self.target_state(current_rgb)
        target_mapped_negative = self.target_state(fixed_negative_rgb)
        all_action_predictions = self.predict_all_actions_from_state(current_state)
        return direct_bev_state_objective_v1(
            current_state_logits=current_state,
            next_online_state_logits=next_online_state,
            all_action_prediction_logits=all_action_predictions,
            target_next_logits=target_next,
            target_current_logits=target_current,
            target_mapped_negative_logits=target_mapped_negative,
            current_labels=current_labels,
            next_labels=next_labels,
            executed_action_indices=executed,
            non_hold_mask=non_hold_mask,
        )

    @torch.no_grad()
    def wrong_rgb_grounding_control(
        self,
        *,
        next_rgb: torch.Tensor,
        fixed_negative_rgb: torch.Tensor,
        next_labels: torch.Tensor,
    ) -> WrongRgbGroundingControlV1:
        """Observation-only online wrong-RGB grounding diagnostic."""

        if fixed_negative_rgb.shape != next_rgb.shape:
            raise ValueError("next and fixed-negative RGB shapes differ")
        correct = self.online_state(next_rgb)
        mapped_negative = self.online_state(fixed_negative_rgb)
        return WrongRgbGroundingControlV1(
            correct_next_loss_per_row=_hard_hierarchical_loss_per_row(
                correct, next_labels
            ),
            mapped_negative_loss_per_row=_hard_hierarchical_loss_per_row(
                mapped_negative, next_labels
            ),
            correct_next_state_logits=correct,
            mapped_negative_state_logits=mapped_negative,
        )


__all__ = [
    "ACTION_VOCABULARY_V1",
    "DirectBevStateObjectiveV1",
    "DirectEgocentricBevStateJepaV1",
    "DirectEgocentricBevStateJepaV1Config",
    "FREE_CLASS_V1",
    "HOLD_ACTION_INDEX_V1",
    "HierarchicalHardLossV1",
    "OCCUPIED_CLASS_V1",
    "UNKNOWN_CLASS_V1",
    "WrongRgbGroundingControlV1",
    "direct_bev_state_objective_v1",
    "hard_hierarchical_raster_loss_v1",
    "soft_hierarchical_state_energy_v1",
]
