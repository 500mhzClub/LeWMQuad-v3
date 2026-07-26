"""Masked current-to-next pair-tubelet JEPA V11 model contract.

The online branch sees only current RGB, a learned future mask, and a candidate
action.  Actual future RGB is accepted only by the detached EMA target branch.
This file intentionally contains the complete, local V11 model/loss contract.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import VisionEncoder


ACTION_VOCABULARY_V11 = (
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
HOLD_ACTION_INDEX_V11 = 6


@dataclass(frozen=True)
class MaskedPairTubeletJepaV11Config:
    image_size: int = 112
    patch_size: int = 7
    feature_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    encoder_dropout: float = 0.0
    future_token_count: int = 256
    action_count: int = 9
    target_ema_momentum: float = 0.996
    normalization_epsilon: float = 1e-8
    whitening_epsilon: float = 1e-4
    whitening_variance_weight: float = 0.50
    whitening_covariance_weight: float = 0.02

    def __post_init__(self) -> None:
        expected = {
            "image_size": 112,
            "patch_size": 7,
            "feature_dim": 192,
            "encoder_depth": 6,
            "encoder_heads": 6,
            "encoder_mlp_ratio": 4,
            "encoder_dropout": 0.0,
            "future_token_count": 256,
            "action_count": 9,
            "target_ema_momentum": 0.996,
            "normalization_epsilon": 1e-8,
            "whitening_epsilon": 1e-4,
            "whitening_variance_weight": 0.50,
            "whitening_covariance_weight": 0.02,
        }
        changed = [
            name for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if changed:
            raise ValueError(
                "V11 preregistered model constants cannot change: "
                + ", ".join(changed)
            )


class OnlineTubeletPathV11(NamedTuple):
    normalized_projected_future: torch.Tensor
    action_indices: torch.Tensor
    current_patch_tokens: torch.Tensor
    tubelet_input: torch.Tensor | None
    block_outputs: tuple[torch.Tensor, ...]


class AllActionPredictionsV11(NamedTuple):
    normalized_projected_future: torch.Tensor
    action_indices: torch.Tensor
    shared_current_patch_tokens: torch.Tensor
    tubelet_input: torch.Tensor | None
    block_outputs: tuple[torch.Tensor, ...]


class FixedCurrentTargetsV11(NamedTuple):
    correct_next: torch.Tensor
    deranged_next: torch.Tensor
    no_change_current: torch.Tensor
    shared_current_patch_tokens: torch.Tensor
    tubelet_inputs: torch.Tensor | None


class ActionRetrievalTermsV11(NamedTuple):
    loss: torch.Tensor
    energies: torch.Tensor
    logits: torch.Tensor
    nll_per_row: torch.Tensor


class TargetRetrievalTermsV11(NamedTuple):
    loss: torch.Tensor
    energies: torch.Tensor
    logits: torch.Tensor
    nll_per_row: torch.Tensor
    candidate_mask: torch.Tensor


class ProjectedFutureWhiteningTermsV11(NamedTuple):
    variance: torch.Tensor
    covariance: torch.Tensor


class MaskedPairTubeletObjectiveV11(NamedTuple):
    total: torch.Tensor
    masked_future_jepa: torch.Tensor
    action_retrieval: torch.Tensor
    target_retrieval: torch.Tensor
    whitening_variance: torch.Tensor
    whitening_covariance: torch.Tensor
    executed_prediction: torch.Tensor
    action_energies: torch.Tensor
    action_logits: torch.Tensor
    action_nll_per_row: torch.Tensor
    target_energies: torch.Tensor
    target_logits: torch.Tensor
    target_nll_per_row: torch.Tensor
    target_candidate_mask: torch.Tensor


class _ZeroInitializedEmbedding(nn.Embedding):
    """Embedding whose constructor performs no discarded random draw."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)


class _XavierInitializedLinear(nn.Linear):
    """Linear whose constructor performs exactly the registered Xavier draw."""

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def _construct_n320_encoder_without_rng_draw() -> VisionEncoder:
    # VisionEncoder's discarded defaults must not advance the V11 initialization
    # stream.  It is strict-loaded before the first registered new draw.
    cpu_rng = torch.random.get_rng_state()
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
        torch.random.set_rng_state(cpu_rng)
    return encoder


def _validate_n320_state(
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


class MaskedCurrentNextPairTubeletJepaV11(nn.Module):
    """Exact early-joint six-block V11 pair-tubelet architecture."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: MaskedPairTubeletJepaV11Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MaskedPairTubeletJepaV11Config()

        self.encoder = _construct_n320_encoder_without_rng_draw()
        _validate_n320_state(self.encoder, n320_encoder_state_dict)
        self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)
        self.encoder.cls_token.requires_grad_(False)
        self.register_buffer(
            "current_temporal_embedding",
            torch.zeros(1, 1, self.config.feature_dim),
            persistent=True,
        )

        # Registered new-parameter draw order.  The custom module reset methods
        # avoid the otherwise discarded nn.Embedding/nn.Linear constructor draws.
        self.online_future_mask_token = nn.Parameter(
            torch.empty(1, 1, self.config.feature_dim)
        )
        nn.init.trunc_normal_(self.online_future_mask_token, std=0.02)
        self.online_future_temporal_embedding = nn.Parameter(
            torch.empty(1, 1, self.config.feature_dim)
        )
        nn.init.trunc_normal_(
            self.online_future_temporal_embedding,
            std=0.02,
        )
        self.online_action_embedding = _ZeroInitializedEmbedding(
            self.config.action_count,
            self.config.feature_dim,
        )
        self.online_future_projector = _XavierInitializedLinear(
            self.config.feature_dim,
            self.config.feature_dim,
            bias=True,
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_future_temporal_embedding = nn.Parameter(
            self.online_future_temporal_embedding.detach().clone(),
            requires_grad=False,
        )
        self.target_future_projector = copy.deepcopy(
            self.online_future_projector
        )
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.hard_sync_target_from_online()

    @property
    def action_vocabulary(self) -> tuple[str, ...]:
        return ACTION_VOCABULARY_V11

    def train(self, mode: bool = True) -> MaskedCurrentNextPairTubeletJepaV11:
        super().train(mode)
        self.target_encoder.eval()
        self.target_future_projector.eval()
        return self

    def _freeze_target_inventory(self) -> None:
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.target_future_temporal_embedding.requires_grad_(False)
        self.target_future_projector.requires_grad_(False)
        self.target_future_projector.eval()

    def ema_inventory_exact(self) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        pairs.extend(
            (f"encoder.{name}", f"target_encoder.{name}")
            for name, _ in self.encoder.named_parameters()
        )
        pairs.append(
            (
                "online_future_temporal_embedding",
                "target_future_temporal_embedding",
            )
        )
        pairs.extend(
            (
                f"online_future_projector.{name}",
                f"target_future_projector.{name}",
            )
            for name, _ in self.online_future_projector.named_parameters()
        )
        return tuple(pairs)

    def _ema_parameter_pairs(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        online_encoder = dict(self.encoder.named_parameters())
        target_encoder = dict(self.target_encoder.named_parameters())
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = [
            (online_encoder[name], target_encoder[name])
            for name in online_encoder
        ]
        pairs.append(
            (
                self.online_future_temporal_embedding,
                self.target_future_temporal_embedding,
            )
        )
        online_projector = dict(
            self.online_future_projector.named_parameters()
        )
        target_projector = dict(
            self.target_future_projector.named_parameters()
        )
        pairs.extend(
            (online_projector[name], target_projector[name])
            for name in online_projector
        )
        return tuple(pairs)

    @torch.no_grad()
    def hard_sync_target_from_online(self) -> None:
        for online, target in self._ema_parameter_pairs():
            target.copy_(online)
        self.ema_update_count.zero_()
        self._freeze_target_inventory()

    @torch.no_grad()
    def update_target_ema(self) -> None:
        momentum = self.config.target_ema_momentum
        for online, target in self._ema_parameter_pairs():
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        self.ema_update_count.add_(1)
        self._freeze_target_inventory()

    def _validate_rgb(self, rgb: torch.Tensor, *, name: str) -> int:
        expected = (
            3,
            self.config.image_size,
            self.config.image_size,
        )
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(
                f"{name} must have shape (B,{expected[0]},"
                f"{expected[1]},{expected[2]})"
            )
        if rgb.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError(f"{name} must be exact float32")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError(f"{name} contains a nonfinite value")
        if rgb.device != self.online_future_mask_token.device:
            raise TypeError(f"{name} and model must share a device")
        return rgb.shape[0]

    def _validate_action_indices(
        self,
        action_indices: torch.Tensor,
        *,
        batch: int,
    ) -> None:
        if action_indices.shape != (batch,) or action_indices.dtype != torch.long:
            raise TypeError("action_indices must be long with shape (B,)")
        if action_indices.device != self.online_future_mask_token.device:
            raise TypeError("action_indices and model must share a device")
        if bool((action_indices < 0).any()) or bool(
            (action_indices >= self.config.action_count).any()
        ):
            raise ValueError("action_indices must be in the closed range [0,8]")

    def _spatial_positions(self, *, target: bool) -> torch.Tensor:
        encoder = self.target_encoder if target else self.encoder
        positions = encoder.pos_embed[:, 1:]
        if positions.shape != (
            1,
            self.config.future_token_count,
            self.config.feature_dim,
        ):
            raise RuntimeError("N320 spatial position inventory changed")
        return positions

    @staticmethod
    def _patch_tokens(encoder: VisionEncoder, rgb: torch.Tensor) -> torch.Tensor:
        return encoder.patch_embed(rgb).flatten(2).transpose(1, 2)

    def _online_current_tokens(self, current_rgb: torch.Tensor) -> torch.Tensor:
        raw = self._patch_tokens(self.encoder, current_rgb)
        return (
            raw
            + self._spatial_positions(target=False)
            + self.current_temporal_embedding
        )

    def _online_future_tokens(
        self,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch = action_indices.shape[0]
        future = self.online_future_mask_token.expand(
            batch, self.config.future_token_count, -1
        )
        future = future + self._spatial_positions(target=False)
        future = future + self.online_future_temporal_embedding
        future = future + self.online_action_embedding(action_indices)[:, None]
        return future

    def _online_joint_forward(
        self,
        tubelet_input: torch.Tensor,
        *,
        capture_intermediates: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden = tubelet_input
        captures: list[torch.Tensor] = []
        for block in self.encoder.blocks:
            hidden = block(hidden)
            if capture_intermediates:
                captures.append(hidden)
        hidden = self.encoder.norm(hidden)
        future = hidden[:, self.config.future_token_count :]
        projected = self.online_future_projector(future)
        normalized = F.normalize(
            projected,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        return normalized, tuple(captures)

    def forward_online_context(
        self,
        current_rgb: torch.Tensor,
        action_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> OnlineTubeletPathV11:
        batch = self._validate_rgb(current_rgb, name="current_rgb")
        self._validate_action_indices(action_indices, batch=batch)
        current = self._online_current_tokens(current_rgb)
        future = self._online_future_tokens(action_indices)
        tubelet = torch.cat((current, future), dim=1)
        normalized, block_outputs = self._online_joint_forward(
            tubelet,
            capture_intermediates=capture_intermediates,
        )
        return OnlineTubeletPathV11(
            normalized_projected_future=normalized,
            action_indices=action_indices,
            current_patch_tokens=current,
            tubelet_input=tubelet if capture_intermediates else None,
            block_outputs=block_outputs,
        )

    def predict_all_actions(
        self,
        current_rgb: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> AllActionPredictionsV11:
        batch = self._validate_rgb(current_rgb, name="current_rgb")
        current = self._online_current_tokens(current_rgb)
        action_indices = torch.arange(
            self.config.action_count,
            dtype=torch.long,
            device=current_rgb.device,
        )
        future = self._online_future_tokens(action_indices)
        current_all = current[:, None].expand(
            -1, self.config.action_count, -1, -1
        )
        future_all = future[None].expand(batch, -1, -1, -1)
        tubelet = torch.cat((current_all, future_all), dim=2)
        flat = tubelet.reshape(
            batch * self.config.action_count,
            self.config.future_token_count * 2,
            self.config.feature_dim,
        )
        normalized, flat_blocks = self._online_joint_forward(
            flat,
            capture_intermediates=capture_intermediates,
        )
        predictions = normalized.reshape(
            batch,
            self.config.action_count,
            self.config.future_token_count,
            self.config.feature_dim,
        )
        block_outputs = tuple(
            value.reshape(
                batch,
                self.config.action_count,
                self.config.future_token_count * 2,
                self.config.feature_dim,
            )
            for value in flat_blocks
        )
        return AllActionPredictionsV11(
            normalized_projected_future=predictions,
            action_indices=action_indices,
            shared_current_patch_tokens=current,
            tubelet_input=tubelet if capture_intermediates else None,
            block_outputs=block_outputs,
        )

    def _target_joint_forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tubelet = torch.cat((current_tokens, future_tokens), dim=1)
        hidden = tubelet
        for block in self.target_encoder.blocks:
            hidden = block(hidden)
        hidden = self.target_encoder.norm(hidden)
        future = hidden[:, self.config.future_token_count :]
        projected = self.target_future_projector(future)
        return F.normalize(
            projected,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        ), tubelet

    def _target_slots_from_raw(
        self,
        raw_patch_tokens: torch.Tensor,
        *,
        future: bool,
    ) -> torch.Tensor:
        temporal = (
            self.target_future_temporal_embedding
            if future
            else self.current_temporal_embedding
        )
        return raw_patch_tokens + self._spatial_positions(target=True) + temporal

    @torch.no_grad()
    def encode_target_future(
        self,
        current_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
    ) -> torch.Tensor:
        batch = self._validate_rgb(current_rgb, name="current_rgb")
        if self._validate_rgb(future_rgb, name="future_rgb") != batch:
            raise ValueError("current_rgb and future_rgb batch sizes differ")
        current_raw = self._patch_tokens(self.target_encoder, current_rgb)
        future_raw = self._patch_tokens(self.target_encoder, future_rgb)
        current = self._target_slots_from_raw(current_raw, future=False)
        future = self._target_slots_from_raw(future_raw, future=True)
        normalized, _ = self._target_joint_forward(current, future)
        return normalized.detach()

    @torch.no_grad()
    def build_fixed_current_targets(
        self,
        current_rgb: torch.Tensor,
        correct_next_rgb: torch.Tensor,
        deranged_next_rgb: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> FixedCurrentTargetsV11:
        batch = self._validate_rgb(current_rgb, name="current_rgb")
        if self._validate_rgb(correct_next_rgb, name="correct_next_rgb") != batch:
            raise ValueError("correct-next batch size changed")
        if self._validate_rgb(deranged_next_rgb, name="deranged_next_rgb") != batch:
            raise ValueError("deranged-next batch size changed")

        current_raw = self._patch_tokens(self.target_encoder, current_rgb)
        current = self._target_slots_from_raw(current_raw, future=False)
        correct_raw = self._patch_tokens(
            self.target_encoder, correct_next_rgb
        )
        deranged_raw = self._patch_tokens(
            self.target_encoder, deranged_next_rgb
        )
        candidate_future = torch.stack(
            (
                self._target_slots_from_raw(correct_raw, future=True),
                self._target_slots_from_raw(deranged_raw, future=True),
                self._target_slots_from_raw(current_raw, future=True),
            ),
            dim=1,
        )
        current_all = current[:, None].expand(-1, 3, -1, -1)
        flat_current = current_all.reshape(
            batch * 3,
            self.config.future_token_count,
            self.config.feature_dim,
        )
        flat_future = candidate_future.reshape(
            batch * 3,
            self.config.future_token_count,
            self.config.feature_dim,
        )
        normalized, flat_tubelet = self._target_joint_forward(
            flat_current,
            flat_future,
        )
        candidates = normalized.reshape(
            batch,
            3,
            self.config.future_token_count,
            self.config.feature_dim,
        )
        tubelet = flat_tubelet.reshape(
            batch,
            3,
            self.config.future_token_count * 2,
            self.config.feature_dim,
        )
        return FixedCurrentTargetsV11(
            correct_next=candidates[:, 0].detach(),
            deranged_next=candidates[:, 1].detach(),
            no_change_current=candidates[:, 2].detach(),
            shared_current_patch_tokens=current.detach(),
            tubelet_inputs=tubelet.detach() if capture_intermediates else None,
        )


def _require_normalized_future_shape(
    tokens: torch.Tensor,
    *,
    prefix: tuple[int, ...] | None = None,
) -> None:
    if tokens.dtype != torch.float32:
        raise TypeError("normalized projected future tokens must be float32")
    if tokens.ndim < 3 or tuple(tokens.shape[-2:]) != (256, 192):
        raise ValueError("tokens must end in shape (256,192)")
    if prefix is not None and tuple(tokens.shape[:-2]) != prefix:
        raise ValueError("normalized projected future prefix shape changed")
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError("normalized projected future is nonfinite")
    norms = tokens.square().sum(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError("projected future tokens must be per-token normalized")


def normalized_token_energy_v11(
    query: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if query.shape != target.shape:
        raise ValueError("query and target shapes differ")
    if query.device != target.device:
        raise TypeError("query and target must share a device")
    _require_normalized_future_shape(query)
    _require_normalized_future_shape(target)
    energy = (query - target).square().sum(dim=-1).mean(dim=-1)
    if not bool(torch.isfinite(energy).all()) or bool((energy < 0).any()) or bool(
        (energy > 4).any()
    ):
        raise FloatingPointError("normalized token energy left [0,4]")
    return energy


def action_retrieval_loss_v11(
    all_actions: AllActionPredictionsV11,
    correct_next_target: torch.Tensor,
    executed_action_indices: torch.Tensor,
) -> ActionRetrievalTermsV11:
    predictions = all_actions.normalized_projected_future
    if predictions.ndim != 4 or tuple(predictions.shape[1:]) != (9, 256, 192):
        raise ValueError("all-action predictions must have shape (B,9,256,192)")
    batch = predictions.shape[0]
    if executed_action_indices.shape != (batch,) or executed_action_indices.dtype != torch.long:
        raise TypeError("executed_action_indices must be long with shape (B,)")
    if executed_action_indices.device != predictions.device:
        raise TypeError("executed action indices and predictions must share a device")
    if bool((executed_action_indices < 0).any()) or bool(
        (executed_action_indices >= 9).any()
    ):
        raise ValueError("executed action indices must be in [0,8]")
    if correct_next_target.shape != (batch, 256, 192):
        raise ValueError("correct target must have shape (B,256,192)")
    targets = correct_next_target.detach()[:, None].expand_as(predictions)
    energies = normalized_token_energy_v11(predictions, targets)
    logits = -energies
    nll = F.cross_entropy(logits, executed_action_indices, reduction="none")
    return ActionRetrievalTermsV11(nll.mean(), energies, logits, nll)


def target_retrieval_loss_v11(
    executed_prediction: torch.Tensor,
    targets: FixedCurrentTargetsV11,
    executed_action_indices: torch.Tensor,
) -> TargetRetrievalTermsV11:
    batch = executed_prediction.shape[0]
    _require_normalized_future_shape(executed_prediction, prefix=(batch,))
    if executed_action_indices.shape != (batch,) or executed_action_indices.dtype != torch.long:
        raise TypeError("executed_action_indices must be long with shape (B,)")
    candidates = torch.stack(
        (
            targets.correct_next.detach(),
            targets.deranged_next.detach(),
            targets.no_change_current.detach(),
        ),
        dim=1,
    )
    query = executed_prediction[:, None].expand_as(candidates)
    energies = normalized_token_energy_v11(query, candidates)
    logits = -energies
    target_index = torch.zeros(
        batch, dtype=torch.long, device=executed_prediction.device
    )
    nll_three = F.cross_entropy(logits, target_index, reduction="none")
    nll_two = F.cross_entropy(logits[:, :2], target_index, reduction="none")
    non_hold = executed_action_indices.ne(HOLD_ACTION_INDEX_V11)
    nll = torch.where(non_hold, nll_three, nll_two)
    candidate_mask = torch.ones(
        batch, 3, dtype=torch.bool, device=executed_prediction.device
    )
    candidate_mask[:, 2] = non_hold
    return TargetRetrievalTermsV11(
        nll.mean(), energies, logits, nll, candidate_mask
    )


def projected_future_whitening_v11(
    normalized_projected_future: torch.Tensor,
) -> ProjectedFutureWhiteningTermsV11:
    _require_normalized_future_shape(normalized_projected_future)
    if normalized_projected_future.ndim != 3:
        raise ValueError("whitening population must have shape (B,256,192)")
    batch, patches, dim = normalized_projected_future.shape
    if batch < 2:
        raise ValueError("V11 whitening requires at least two rows")
    position_centered = normalized_projected_future - normalized_projected_future.mean(
        dim=0, keepdim=True
    )
    rank_matrix = position_centered.reshape(batch * patches, dim)
    rms_square = rank_matrix.square().mean().detach()
    normalized = rank_matrix / torch.sqrt(rms_square + 1e-4)
    covariance_matrix = (
        normalized.transpose(0, 1) @ normalized
    ) / float(batch * patches - 1)
    diagonal = covariance_matrix.diagonal()
    variance = F.relu(1.0 - torch.sqrt(diagonal + 1e-4)).mean()
    off_diagonal = ~torch.eye(
        dim, dtype=torch.bool, device=normalized_projected_future.device
    )
    covariance = covariance_matrix.square().masked_select(
        off_diagonal
    ).sum() / float(dim)
    return ProjectedFutureWhiteningTermsV11(variance, covariance)


def masked_pair_tubelet_objective_v11(
    all_actions: AllActionPredictionsV11,
    targets: FixedCurrentTargetsV11,
    executed_action_indices: torch.Tensor,
) -> MaskedPairTubeletObjectiveV11:
    predictions = all_actions.normalized_projected_future
    if predictions.ndim != 4 or tuple(predictions.shape[1:]) != (9, 256, 192):
        raise ValueError("all-action predictions must have shape (B,9,256,192)")
    batch = predictions.shape[0]
    if executed_action_indices.shape != (batch,) or executed_action_indices.dtype != torch.long:
        raise TypeError("executed_action_indices must be long with shape (B,)")
    executed = predictions.gather(
        1,
        executed_action_indices[:, None, None, None].expand(
            -1, 1, 256, 192
        ),
    ).squeeze(1)
    detached_correct = targets.correct_next.detach()
    masked_jepa = (executed - detached_correct).square().mean()
    action = action_retrieval_loss_v11(
        all_actions, detached_correct, executed_action_indices
    )
    target = target_retrieval_loss_v11(
        executed, targets, executed_action_indices
    )
    whitening = projected_future_whitening_v11(executed)
    total = (
        masked_jepa
        + action.loss
        + target.loss
        + 0.50 * whitening.variance
        + 0.02 * whitening.covariance
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("V11 objective is nonfinite")
    return MaskedPairTubeletObjectiveV11(
        total=total,
        masked_future_jepa=masked_jepa,
        action_retrieval=action.loss,
        target_retrieval=target.loss,
        whitening_variance=whitening.variance,
        whitening_covariance=whitening.covariance,
        executed_prediction=executed,
        action_energies=action.energies,
        action_logits=action.logits,
        action_nll_per_row=action.nll_per_row,
        target_energies=target.energies,
        target_logits=target.logits,
        target_nll_per_row=target.nll_per_row,
        target_candidate_mask=target.candidate_mask,
    )


__all__ = [
    "ACTION_VOCABULARY_V11",
    "HOLD_ACTION_INDEX_V11",
    "ActionRetrievalTermsV11",
    "AllActionPredictionsV11",
    "FixedCurrentTargetsV11",
    "MaskedCurrentNextPairTubeletJepaV11",
    "MaskedPairTubeletJepaV11Config",
    "MaskedPairTubeletObjectiveV11",
    "OnlineTubeletPathV11",
    "ProjectedFutureWhiteningTermsV11",
    "TargetRetrievalTermsV11",
    "action_retrieval_loss_v11",
    "masked_pair_tubelet_objective_v11",
    "normalized_token_energy_v11",
    "projected_future_whitening_v11",
    "target_retrieval_loss_v11",
]
