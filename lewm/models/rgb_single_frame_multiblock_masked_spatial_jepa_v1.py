"""Single-frame multi-block masked spatial JEPA V1.

The online encoder sees only the visible patch embeddings from one RGB frame.
The detached EMA encoder sees the complete copy of that same frame.  A small
spatial predictor fills the original 16x16 token lattice and predicts only the
registered target slots.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.encoders import ViTBlock, VisionEncoder


@dataclass(frozen=True)
class SingleFrameMultiblockMaskedSpatialJepaV1Config:
    """Frozen V1 architecture and optimization-independent constants."""

    image_size: int = 112
    patch_size: int = 7
    feature_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 6
    encoder_mlp_ratio: int = 4
    encoder_dropout: float = 0.0
    spatial_token_count: int = 256
    target_token_count: int = 64
    visible_token_count: int = 192
    predictor_depth: int = 2
    predictor_heads: int = 6
    predictor_mlp_ratio: int = 2
    predictor_dropout: float = 0.0
    target_ema_momentum: float = 0.996
    normalization_epsilon: float = 1e-8
    initialization_seed: int = 20260731

    def __post_init__(self) -> None:
        expected = {
            "image_size": 112,
            "patch_size": 7,
            "feature_dim": 192,
            "encoder_depth": 6,
            "encoder_heads": 6,
            "encoder_mlp_ratio": 4,
            "encoder_dropout": 0.0,
            "spatial_token_count": 256,
            "target_token_count": 64,
            "visible_token_count": 192,
            "predictor_depth": 2,
            "predictor_heads": 6,
            "predictor_mlp_ratio": 2,
            "predictor_dropout": 0.0,
            "target_ema_momentum": 0.996,
            "normalization_epsilon": 1e-8,
            "initialization_seed": 20260731,
        }
        changed = [
            name
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if changed:
            raise ValueError(
                "single-frame masked spatial JEPA V1 constants cannot change: "
                + ", ".join(changed)
            )


class OnlineMaskedSpatialPredictionV1(NamedTuple):
    raw_predicted_target_tokens: torch.Tensor
    normalized_predicted_target_tokens: torch.Tensor
    target_indices: torch.Tensor
    visible_indices: torch.Tensor
    encoded_visible_tokens: torch.Tensor
    online_input: torch.Tensor | None
    online_block_outputs: tuple[torch.Tensor, ...]
    predictor_input: torch.Tensor | None
    predictor_block_outputs: tuple[torch.Tensor, ...]


class MaskedSpatialTargetV1(NamedTuple):
    raw_target_tokens: torch.Tensor
    normalized_target_tokens: torch.Tensor
    target_indices: torch.Tensor


class MaskedSpatialJepaOutputV1(NamedTuple):
    prediction: OnlineMaskedSpatialPredictionV1
    target: MaskedSpatialTargetV1
    loss: torch.Tensor


def _construct_n320_encoder_without_rng_draw() -> VisionEncoder:
    """Construct the exact N320 encoder without consuming the V1 RNG stream."""

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


def _gather_spatial_tokens(
    tokens: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather per-row spatial tokens while retaining their registered order."""

    return tokens.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )


def _encode_selected_spatial_tokens(
    encoder: VisionEncoder,
    rgb: torch.Tensor,
    selected_indices: torch.Tensor | None,
    *,
    capture_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...]]:
    """Encode CLS plus a selected set of patches.

    ``None`` selects the complete spatial lattice.  A zero-width index tensor
    is also valid, which makes this helper easy to test at both boundaries.
    The caller owns shape and index validation.
    """

    patch_tokens = encoder.patch_embed(rgb).flatten(2).transpose(1, 2)
    spatial_positions = encoder.pos_embed[:, 1:]
    if selected_indices is not None:
        patch_tokens = _gather_spatial_tokens(patch_tokens, selected_indices)
        positions = spatial_positions.expand(rgb.shape[0], -1, -1)
        spatial_positions = _gather_spatial_tokens(positions, selected_indices)

    cls = encoder.cls_token.expand(rgb.shape[0], -1, -1)
    cls = cls + encoder.pos_embed[:, :1]
    spatial = patch_tokens + spatial_positions
    hidden = encoder.pos_drop(torch.cat((cls, spatial), dim=1))
    online_input = hidden if capture_intermediates else None
    captures: list[torch.Tensor] = []
    for block in encoder.blocks:
        hidden = block(hidden)
        if capture_intermediates:
            captures.append(hidden)
    hidden = encoder.norm(hidden)
    return hidden[:, 1:], online_input, tuple(captures)


def _require_raw_target_shape(
    tokens: torch.Tensor,
    *,
    name: str,
) -> None:
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (64, 192):
        raise ValueError(f"{name} must have shape (B,64,192)")
    if tokens.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if tokens.dtype != torch.float32:
        raise TypeError(f"{name} must use exact float32")
    if not bool(torch.isfinite(tokens).all()):
        raise FloatingPointError(f"{name} contains a nonfinite value")


def normalized_half_squared_token_energy_v1(
    raw_prediction: torch.Tensor,
    raw_target: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return the registered per-row normalized half-squared token energy."""

    _require_raw_target_shape(raw_prediction, name="raw_prediction")
    _require_raw_target_shape(raw_target, name="raw_target")
    if raw_prediction.shape != raw_target.shape:
        raise ValueError("prediction and target batch sizes differ")
    if raw_prediction.device != raw_target.device:
        raise TypeError("prediction and target must share a device")
    if epsilon != 1e-8:
        raise ValueError("V1 normalization epsilon cannot change")
    prediction = F.normalize(raw_prediction, p=2.0, dim=-1, eps=epsilon)
    target = F.normalize(
        raw_target.detach(),
        p=2.0,
        dim=-1,
        eps=epsilon,
    )
    energy = 0.5 * (prediction - target).square().sum(dim=-1).mean(dim=-1)
    if not bool(torch.isfinite(energy).all()):
        raise FloatingPointError("normalized token energy is nonfinite")
    return energy


def normalized_half_squared_jepa_loss_v1(
    raw_prediction: torch.Tensor,
    raw_target: torch.Tensor,
) -> torch.Tensor:
    """The sole V1 training objective, averaged over rows and target tokens."""

    return normalized_half_squared_token_energy_v1(
        raw_prediction,
        raw_target,
    ).mean()


class SingleFrameMultiblockMaskedSpatialJepaV1(nn.Module):
    """Joint RGB encoder and same-image masked-token predictor."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: SingleFrameMultiblockMaskedSpatialJepaV1Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SingleFrameMultiblockMaskedSpatialJepaV1Config()

        self.encoder = _construct_n320_encoder_without_rng_draw()
        _validate_n320_state(self.encoder, n320_encoder_state_dict)
        self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)

        caller_rng = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(self.config.initialization_seed)
            self.predictor_position = nn.Parameter(
                self.encoder.pos_embed[:, 1:].detach().clone().squeeze(0)
            )
            self.predictor_mask_token = nn.Parameter(
                torch.empty(1, 1, self.config.feature_dim)
            )
            nn.init.trunc_normal_(self.predictor_mask_token, std=0.02)
            self.predictor_blocks = nn.ModuleList(
                [
                    ViTBlock(
                        hidden_dim=self.config.feature_dim,
                        n_heads=self.config.predictor_heads,
                        mlp_ratio=self.config.predictor_mlp_ratio,
                        dropout=self.config.predictor_dropout,
                    )
                    for _ in range(self.config.predictor_depth)
                ]
            )
            self.predictor_norm = nn.LayerNorm(self.config.feature_dim)
            self.predictor_output = nn.Linear(
                self.config.feature_dim,
                self.config.feature_dim,
            )
        finally:
            torch.random.set_rng_state(caller_rng)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.hard_sync_target_from_online()

    def train(
        self,
        mode: bool = True,
    ) -> SingleFrameMultiblockMaskedSpatialJepaV1:
        super().train(mode)
        self._freeze_target_inventory()
        return self

    def _freeze_target_inventory(self) -> None:
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def ema_inventory_exact(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (f"encoder.{name}", f"target_encoder.{name}")
            for name, _ in self.encoder.named_parameters()
        )

    def _ema_parameter_pairs(
        self,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        online = dict(self.encoder.named_parameters())
        target = dict(self.target_encoder.named_parameters())
        if online.keys() != target.keys():
            raise RuntimeError("online and target encoder inventories differ")
        return tuple((online[name], target[name]) for name in online)

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

    def _validate_rgb(self, rgb: torch.Tensor) -> int:
        expected = (
            3,
            self.config.image_size,
            self.config.image_size,
        )
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != expected:
            raise ValueError(
                f"rgb must have shape (B,{expected[0]},"
                f"{expected[1]},{expected[2]})"
            )
        if rgb.shape[0] < 1:
            raise ValueError("rgb must contain at least one row")
        if rgb.dtype != torch.float32:
            raise TypeError("rgb must use exact float32")
        if not bool(torch.isfinite(rgb).all()):
            raise FloatingPointError("rgb contains a nonfinite value")
        if rgb.device != self.predictor_mask_token.device:
            raise TypeError("rgb and model must share a device")
        return rgb.shape[0]

    def _validate_target_indices(
        self,
        target_indices: torch.Tensor,
        *,
        batch: int,
    ) -> None:
        expected = (batch, self.config.target_token_count)
        if target_indices.shape != expected or target_indices.dtype != torch.long:
            raise TypeError(
                f"target_indices must be long with shape {expected}"
            )
        if target_indices.device != self.predictor_mask_token.device:
            raise TypeError("target_indices and model must share a device")
        if bool((target_indices < 0).any()) or bool(
            (target_indices >= self.config.spatial_token_count).any()
        ):
            raise ValueError("target_indices must be in the closed range [0,255]")
        if not bool(
            (target_indices[:, 1:] > target_indices[:, :-1]).all()
        ):
            raise ValueError(
                "target_indices must be unique and strictly increasing per row"
            )

    def _visible_indices(self, target_indices: torch.Tensor) -> torch.Tensor:
        batch = target_indices.shape[0]
        target_mask = torch.zeros(
            batch,
            self.config.spatial_token_count,
            dtype=torch.bool,
            device=target_indices.device,
        )
        target_mask.scatter_(1, target_indices, True)
        full = torch.arange(
            self.config.spatial_token_count,
            dtype=torch.long,
            device=target_indices.device,
        ).expand(batch, -1)
        visible = full.masked_select(~target_mask).reshape(
            batch,
            self.config.visible_token_count,
        )
        return visible

    def encode_online_full_frame(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return trainable full-frame online spatial tokens without a head."""

        self._validate_rgb(rgb)
        tokens, _, _ = _encode_selected_spatial_tokens(
            self.encoder,
            rgb,
            None,
        )
        return tokens

    @torch.no_grad()
    def encode_target_full_frame(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return detached full-frame EMA spatial tokens without a head."""

        self._validate_rgb(rgb)
        tokens, _, _ = _encode_selected_spatial_tokens(
            self.target_encoder,
            rgb,
            None,
        )
        return tokens.detach()

    def forward_online(
        self,
        rgb: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> OnlineMaskedSpatialPredictionV1:
        batch = self._validate_rgb(rgb)
        self._validate_target_indices(target_indices, batch=batch)
        visible_indices = self._visible_indices(target_indices)

        encoded_visible, online_input, online_blocks = (
            _encode_selected_spatial_tokens(
                self.encoder,
                rgb,
                visible_indices,
                capture_intermediates=capture_intermediates,
            )
        )
        predictor = self.predictor_mask_token.expand(
            batch,
            self.config.spatial_token_count,
            -1,
        )
        predictor = predictor.scatter(
            1,
            visible_indices.unsqueeze(-1).expand(
                -1,
                -1,
                self.config.feature_dim,
            ),
            encoded_visible,
        )
        predictor = predictor + self.predictor_position.unsqueeze(0)
        predictor_input = predictor if capture_intermediates else None
        predictor_captures: list[torch.Tensor] = []
        for block in self.predictor_blocks:
            predictor = block(predictor)
            if capture_intermediates:
                predictor_captures.append(predictor)
        predictor = self.predictor_norm(predictor)
        masked_prediction = _gather_spatial_tokens(
            predictor,
            target_indices,
        )
        raw_prediction = self.predictor_output(masked_prediction)
        normalized_prediction = F.normalize(
            raw_prediction,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        )
        return OnlineMaskedSpatialPredictionV1(
            raw_predicted_target_tokens=raw_prediction,
            normalized_predicted_target_tokens=normalized_prediction,
            target_indices=target_indices,
            visible_indices=visible_indices,
            encoded_visible_tokens=encoded_visible,
            online_input=online_input,
            online_block_outputs=online_blocks,
            predictor_input=predictor_input,
            predictor_block_outputs=tuple(predictor_captures),
        )

    @torch.no_grad()
    def encode_target(
        self,
        rgb: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> MaskedSpatialTargetV1:
        batch = self._validate_rgb(rgb)
        self._validate_target_indices(target_indices, batch=batch)
        full_tokens, _, _ = _encode_selected_spatial_tokens(
            self.target_encoder,
            rgb,
            None,
        )
        raw_target = _gather_spatial_tokens(full_tokens, target_indices).detach()
        normalized_target = F.normalize(
            raw_target,
            p=2.0,
            dim=-1,
            eps=self.config.normalization_epsilon,
        ).detach()
        return MaskedSpatialTargetV1(
            raw_target_tokens=raw_target,
            normalized_target_tokens=normalized_target,
            target_indices=target_indices,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        capture_intermediates: bool = False,
    ) -> MaskedSpatialJepaOutputV1:
        prediction = self.forward_online(
            rgb,
            target_indices,
            capture_intermediates=capture_intermediates,
        )
        target = self.encode_target(rgb, target_indices)
        loss = normalized_half_squared_jepa_loss_v1(
            prediction.raw_predicted_target_tokens,
            target.raw_target_tokens,
        )
        return MaskedSpatialJepaOutputV1(
            prediction=prediction,
            target=target,
            loss=loss,
        )
