"""Direct BEV V3 coordinate-aware FiLM U-Net predictor adapter."""
from __future__ import annotations

import copy
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models import direct_egocentric_bev_state_jepa_v1 as _v1


ACTION_VOCABULARY_V1 = _v1.ACTION_VOCABULARY_V1
DirectBevStateObjectiveV1 = _v1.DirectBevStateObjectiveV1
DirectEgocentricBevStateJepaV1Config = _v1.DirectEgocentricBevStateJepaV1Config
FREE_CLASS_V1 = _v1.FREE_CLASS_V1
HOLD_ACTION_INDEX_V1 = _v1.HOLD_ACTION_INDEX_V1
HierarchicalHardLossV1 = _v1.HierarchicalHardLossV1
OCCUPIED_CLASS_V1 = _v1.OCCUPIED_CLASS_V1
UNKNOWN_CLASS_V1 = _v1.UNKNOWN_CLASS_V1
WrongRgbGroundingControlV1 = _v1.WrongRgbGroundingControlV1
direct_bev_state_objective_v1 = _v1.direct_bev_state_objective_v1
hard_hierarchical_raster_loss_v1 = _v1.hard_hierarchical_raster_loss_v1
soft_hierarchical_state_energy_v1 = _v1.soft_hierarchical_state_energy_v1
_hard_hierarchical_loss_per_row = _v1._hard_hierarchical_loss_per_row


class _FilmUnetBlockV3(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.norm1 = nn.GroupNorm(4, output_channels, affine=True)
        self.activation1 = nn.GELU()
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.norm2 = nn.GroupNorm(4, output_channels, affine=True)
        self.activation2 = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.activation1(self.norm1(self.conv1(value)))
        return self.activation2(self.norm2(self.conv2(value)))


def _downsample_v3(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        ),
        nn.GroupNorm(4, output_channels, affine=True),
        nn.GELU(),
    )


class _CoordinateAwareFilmUnetPredictorV3(nn.Module):
    """Coordinate-aware residual state transition with shared state encoding."""

    def __init__(self) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(9, 64)
        self.enc64 = _FilmUnetBlockV3(5, 16)
        self.down32 = _downsample_v3(16, 32)
        self.enc32 = _FilmUnetBlockV3(32, 32)
        self.down16 = _downsample_v3(32, 48)
        self.enc16 = _FilmUnetBlockV3(48, 48)
        self.down8 = _downsample_v3(48, 64)
        self.bottleneck = _FilmUnetBlockV3(64, 64)
        self.film64 = nn.Linear(64, 128)
        self.dec16 = _FilmUnetBlockV3(112, 48)
        self.film48 = nn.Linear(64, 96)
        self.dec32 = _FilmUnetBlockV3(80, 32)
        self.film32 = nn.Linear(64, 64)
        self.dec64 = _FilmUnetBlockV3(48, 16)
        self.film16 = nn.Linear(64, 32)
        self.residual_head = nn.Conv2d(
            16,
            3,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    @property
    def net(self) -> tuple[nn.Conv2d]:
        """Compatibility view for the frozen runner's final-head check."""

        return (self.residual_head,)

    @staticmethod
    def _validate_state(current_state_logits: torch.Tensor) -> None:
        if (
            current_state_logits.ndim != 4
            or tuple(current_state_logits.shape[1:]) != (3, 64, 64)
        ):
            raise ValueError(
                "current_state_logits must have shape (B,3,64,64)"
            )
        if current_state_logits.shape[0] < 1:
            raise ValueError("current_state_logits must contain at least one row")
        if not current_state_logits.is_floating_point():
            raise TypeError("current_state_logits must use a floating dtype")
        if not bool(torch.isfinite(current_state_logits).all()):
            raise FloatingPointError("current_state_logits is nonfinite")

    @staticmethod
    def _coordinate_planes(current_state_logits: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, _channels, height, width = current_state_logits.shape
        row = torch.linspace(
            -1.0,
            1.0,
            height,
            dtype=current_state_logits.dtype,
            device=current_state_logits.device,
        ).view(1, 1, height, 1).expand(batch, 1, height, width)
        column = torch.linspace(
            -1.0,
            1.0,
            width,
            dtype=current_state_logits.dtype,
            device=current_state_logits.device,
        ).view(1, 1, 1, width).expand(batch, 1, height, width)
        return row, column

    def _encode_shared(
        self,
        current_state_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_state(current_state_logits)
        row, column = self._coordinate_planes(current_state_logits)
        skip64 = self.enc64(
            torch.cat((current_state_logits, row, column), dim=1)
        )
        skip32 = self.enc32(self.down32(skip64))
        skip16 = self.enc16(self.down16(skip32))
        value8 = self.bottleneck(self.down8(skip16))
        return value8, skip16, skip32, skip64

    @staticmethod
    def _film(value: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        gamma, beta = parameters.chunk(2, dim=1)
        return (
            value * (1.0 + gamma[:, :, None, None])
            + beta[:, :, None, None]
        )

    def _decode(
        self,
        encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        action_features: torch.Tensor,
        current_state_logits: torch.Tensor,
    ) -> torch.Tensor:
        value8, skip16, skip32, skip64 = encoded
        value = self._film(value8, self.film64(action_features))
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        value = self.dec16(torch.cat((value, skip16), dim=1))
        value = self._film(value, self.film48(action_features))
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        value = self.dec32(torch.cat((value, skip32), dim=1))
        value = self._film(value, self.film32(action_features))
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        value = self.dec64(torch.cat((value, skip64), dim=1))
        value = self._film(value, self.film16(action_features))
        return current_state_logits + self.residual_head(value)

    @staticmethod
    def _expand_actions(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(-1, 9, *value.shape[1:]).reshape(
            value.shape[0] * 9,
            *value.shape[1:],
        )

    def forward(
        self,
        current_state_logits: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_state(current_state_logits)
        action_indices = _v1._validate_action_one_hot(
            action_one_hot,
            batch=current_state_logits.shape[0],
            reference=current_state_logits,
        )
        encoded = self._encode_shared(current_state_logits)
        return self._decode(
            encoded,
            self.action_embedding(action_indices),
            current_state_logits,
        )

    def predict_all_actions(
        self,
        current_state_logits: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_state(current_state_logits)
        batch = current_state_logits.shape[0]
        encoded = tuple(
            self._expand_actions(value)
            for value in self._encode_shared(current_state_logits)
        )
        action_indices = torch.arange(
            9,
            dtype=torch.long,
            device=current_state_logits.device,
        )[None].expand(batch, -1).reshape(batch * 9)
        predictions = self._decode(
            encoded,
            self.action_embedding(action_indices),
            self._expand_actions(current_state_logits),
        )
        return predictions.reshape(batch, 9, 3, 64, 64)


class DirectEgocentricBevStateJepaV1(_v1.DirectEgocentricBevStateJepaV1):
    """V1 perception and objective with the frozen V3 predictor only."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: DirectEgocentricBevStateJepaV1Config | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config or DirectEgocentricBevStateJepaV1Config()
        self.encoder = _v1._construct_n320_encoder_without_rng_draw()
        _v1._validate_n320_encoder_state(
            self.encoder,
            n320_encoder_state_dict,
        )
        self.encoder.load_state_dict(n320_encoder_state_dict, strict=True)

        caller_cpu_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(
                self.config.initialization_seed
            )
            self.bev_decoder = _v1._GlobalCrossAttentionBevDecoderV1(
                self.config
            )
            self.state_head = nn.Conv2d(
                self.config.bev_dim,
                self.config.state_classes,
                kernel_size=1,
            )
            self.predictor = _CoordinateAwareFilmUnetPredictorV3()
        finally:
            torch.random.set_rng_state(caller_cpu_rng)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_decoder = copy.deepcopy(self.bev_decoder)
        self.target_state_head = copy.deepcopy(self.state_head)
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.hard_sync_target_from_online()

    def predict_all_actions_from_state(
        self,
        current_state_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor.predict_all_actions(current_state_logits)


__all__ = [
    *_v1.__all__,
    "_hard_hierarchical_loss_per_row",
]
