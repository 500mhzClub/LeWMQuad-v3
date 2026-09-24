"""Direct BEV V8 learned-query normalized-prototype perception adapter.

V8 preserves the frozen N320 encoder and V3 predictor construction inherited
through V6.  It replaces only the RGB-token-to-BEV decoder and three-channel
state head with a fully learned, factorized BEV-query decoder followed by
literal negative squared distances to learned normalized class prototypes.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


_V6_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
_V6_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v8_learned_query_frozen_v6_model",
    _V6_SOURCE_PATH,
)
if _V6_SPEC is None or _V6_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V6 model source")
_v6 = importlib.util.module_from_spec(_V6_SPEC)
sys.modules[_V6_SPEC.name] = _v6
_V6_SPEC.loader.exec_module(_v6)


ACTION_VOCABULARY_V1 = _v6.ACTION_VOCABULARY_V1
DirectBevStateObjectiveV1 = _v6.DirectBevStateObjectiveV1
DirectEgocentricBevStateJepaV1Config = (
    _v6.DirectEgocentricBevStateJepaV1Config
)
FREE_CLASS_V1 = _v6.FREE_CLASS_V1
HOLD_ACTION_INDEX_V1 = _v6.HOLD_ACTION_INDEX_V1
HierarchicalHardLossV1 = _v6.HierarchicalHardLossV1
OCCUPIED_CLASS_V1 = _v6.OCCUPIED_CLASS_V1
PHASE_ONE_LAST_CALLBACK_COUNT_V6 = _v6.PHASE_ONE_LAST_CALLBACK_COUNT_V6
PHASE_ONE_V6 = _v6.PHASE_ONE_V6
PHASE_TWO_V6 = _v6.PHASE_TWO_V6
UNKNOWN_CLASS_V1 = _v6.UNKNOWN_CLASS_V1
WrongRgbGroundingControlV1 = _v6.WrongRgbGroundingControlV1
direct_bev_state_objective_v1 = _v6.direct_bev_state_objective_v1
hard_hierarchical_raster_loss_v1 = _v6.hard_hierarchical_raster_loss_v1
soft_hierarchical_state_energy_v1 = _v6.soft_hierarchical_state_energy_v1
_hard_hierarchical_loss_per_row = _v6._hard_hierarchical_loss_per_row


BEV_QUERY_ROWS_V8 = 64
BEV_QUERY_COLUMNS_V8 = 64
BEV_FEATURE_DIMENSION_V8 = 64
RGB_TOKEN_COUNT_V8 = 256
RGB_TOKEN_DIMENSION_V8 = 192
CROSS_ATTENTION_HEADS_V8 = 4
FFN_HIDDEN_DIMENSION_V8 = 128
STATE_CLASS_COUNT_V8 = 3
L2_NORMALIZATION_EPSILON_V8 = 1e-12
MAXIMUM_PERCEPTION_UPDATES_V8 = 250
ONLINE_DECODER_PROTOTYPE_PARAMETER_COUNT_V8 = 87_808
ONLINE_DECODER_PROTOTYPE_PARAMETER_TENSOR_COUNT_V8 = 31


class LearnedBevQueryCrossAttentionFfnBlockV8(nn.Module):
    """One independent pre-norm cross-attention and FFN residual block."""

    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(BEV_FEATURE_DIMENSION_V8)
        self.cross_attention = nn.MultiheadAttention(
            BEV_FEATURE_DIMENSION_V8,
            CROSS_ATTENTION_HEADS_V8,
            dropout=0.0,
            bias=True,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(BEV_FEATURE_DIMENSION_V8)
        self.ffn_input = nn.Linear(
            BEV_FEATURE_DIMENSION_V8,
            FFN_HIDDEN_DIMENSION_V8,
            bias=True,
        )
        self.ffn_output = nn.Linear(
            FFN_HIDDEN_DIMENSION_V8,
            BEV_FEATURE_DIMENSION_V8,
            bias=True,
        )

    def forward(
        self,
        cell_features: torch.Tensor,
        projected_rgb_tokens: torch.Tensor,
    ) -> torch.Tensor:
        attended, _weights = self.cross_attention(
            self.query_norm(cell_features),
            projected_rgb_tokens,
            projected_rgb_tokens,
            need_weights=False,
        )
        cell_features = cell_features + attended
        residual = self.ffn_output(
            F.gelu(self.ffn_input(self.ffn_norm(cell_features)))
        )
        return cell_features + residual


class LearnedBevQueryDecoderV8(nn.Module):
    """Retrieve per-cell BEV features from RGB tokens using learned queries."""

    def __init__(self) -> None:
        super().__init__()
        self.token_count = RGB_TOKEN_COUNT_V8
        self.bev_size = (BEV_QUERY_ROWS_V8, BEV_QUERY_COLUMNS_V8)

        # Construction and initialization order is preregistered and frozen.
        self.row_query = nn.Parameter(
            torch.empty(BEV_QUERY_ROWS_V8, BEV_FEATURE_DIMENSION_V8)
        )
        nn.init.trunc_normal_(self.row_query, std=0.02)
        self.column_query = nn.Parameter(
            torch.empty(BEV_QUERY_COLUMNS_V8, BEV_FEATURE_DIMENSION_V8)
        )
        nn.init.trunc_normal_(self.column_query, std=0.02)
        self.token_projection = nn.Linear(
            RGB_TOKEN_DIMENSION_V8,
            BEV_FEATURE_DIMENSION_V8,
            bias=True,
        )
        self.projected_token_norm = nn.LayerNorm(
            BEV_FEATURE_DIMENSION_V8,
            elementwise_affine=True,
        )
        self.block_1 = LearnedBevQueryCrossAttentionFfnBlockV8()
        self.block_2 = LearnedBevQueryCrossAttentionFfnBlockV8()

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if (
            patch_tokens.ndim != 3
            or patch_tokens.shape[1] != RGB_TOKEN_COUNT_V8
            or patch_tokens.shape[2] != RGB_TOKEN_DIMENSION_V8
        ):
            raise ValueError(
                "patch_tokens must have shape (B,256,192)"
            )
        if not patch_tokens.is_floating_point():
            raise TypeError("patch_tokens must use a floating dtype")
        if not bool(torch.isfinite(patch_tokens).all()):
            raise FloatingPointError("patch_tokens is nonfinite")

        projected_tokens = self.projected_token_norm(
            self.token_projection(patch_tokens)
        )
        cell_features = (
            self.row_query[:, None, :] + self.column_query[None, :, :]
        ).reshape(
            BEV_QUERY_ROWS_V8 * BEV_QUERY_COLUMNS_V8,
            BEV_FEATURE_DIMENSION_V8,
        )
        cell_features = cell_features.to(dtype=patch_tokens.dtype)
        cell_features = cell_features[None].expand(
            patch_tokens.shape[0], -1, -1
        )
        cell_features = self.block_1(cell_features, projected_tokens)
        cell_features = self.block_2(cell_features, projected_tokens)
        return cell_features.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            BEV_FEATURE_DIMENSION_V8,
            BEV_QUERY_ROWS_V8,
            BEV_QUERY_COLUMNS_V8,
        )


class NormalizedPrototypeStateHeadV8(nn.Module):
    """Map BEV features to three logits by normalized prototype distance."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = BEV_FEATURE_DIMENSION_V8
        self.out_channels = STATE_CLASS_COUNT_V8
        self.prototypes = nn.Parameter(
            torch.empty(STATE_CLASS_COUNT_V8, BEV_FEATURE_DIMENSION_V8)
        )
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, cell_features: torch.Tensor) -> torch.Tensor:
        expected = (
            BEV_FEATURE_DIMENSION_V8,
            BEV_QUERY_ROWS_V8,
            BEV_QUERY_COLUMNS_V8,
        )
        if cell_features.ndim != 4 or tuple(cell_features.shape[1:]) != expected:
            raise ValueError("cell_features must have shape (B,64,64,64)")
        if not cell_features.is_floating_point():
            raise TypeError("cell_features must use a floating dtype")
        if not bool(torch.isfinite(cell_features).all()):
            raise FloatingPointError("cell_features is nonfinite")

        normalized_features = F.normalize(
            cell_features,
            p=2.0,
            dim=1,
            eps=L2_NORMALIZATION_EPSILON_V8,
        )
        normalized_prototypes = F.normalize(
            self.prototypes,
            p=2.0,
            dim=1,
            eps=L2_NORMALIZATION_EPSILON_V8,
        ).to(dtype=cell_features.dtype)
        return -(
            normalized_features[:, None, :, :, :]
            - normalized_prototypes[None, :, :, None, None]
        ).square().sum(dim=2)


class DirectEgocentricBevStateJepaV1(
    _v6.DirectEgocentricBevStateJepaV1
):
    """Frozen V6 stack with only V8's learned-query perception mechanism."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: DirectEgocentricBevStateJepaV1Config | None = None,
    ) -> None:
        # This constructs the exact frozen V3 predictor before V8 replaces the
        # decoder and head.  V6 deliberately leaves its phase policy unarmed.
        super().__init__(n320_encoder_state_dict, config=config)

        caller_cpu_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(
                self.config.initialization_seed
            )
            self.bev_decoder = LearnedBevQueryDecoderV8()
            self.state_head = NormalizedPrototypeStateHeadV8()
        finally:
            torch.random.set_rng_state(caller_cpu_rng)

        self.target_bev_decoder = copy.deepcopy(self.bev_decoder)
        self.target_state_head = copy.deepcopy(self.state_head)
        self.hard_sync_target_from_online()

        decoder_parameters = tuple(self.bev_decoder.parameters())
        head_parameters = tuple(self.state_head.parameters())
        if (
            len(decoder_parameters) + len(head_parameters)
            != ONLINE_DECODER_PROTOTYPE_PARAMETER_TENSOR_COUNT_V8
            or sum(
                value.numel()
                for value in (*decoder_parameters, *head_parameters)
            )
            != ONLINE_DECODER_PROTOTYPE_PARAMETER_COUNT_V8
        ):
            raise RuntimeError("V8 decoder/prototype inventory changed")

    def set_phase_override_for_integrity_probe_v6(
        self,
        phase: str | None,
    ) -> None:
        """Keep the only authorized V8 phase perception-only."""

        if phase == PHASE_TWO_V6:
            raise RuntimeError("V8 has no authorized predictor phase")
        super().set_phase_override_for_integrity_probe_v6(phase)

    def apply_phase_policy_v6(self) -> None:
        """Keep the predictor frozen for every registered V8 update."""

        if not self.phase_policy_armed_v6:
            return
        callback_count = int(self.ema_update_count.detach().cpu().item())
        if callback_count > MAXIMUM_PERCEPTION_UPDATES_V8:
            raise RuntimeError("V8 exceeded its 250-update perception cap")
        for module in self._online_modules():
            module.requires_grad_(True)
            module.train(bool(self.training))
        self.predictor.requires_grad_(False)
        self.predictor.eval()
        self._freeze_target()

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        """Apply V6 EMA while failing closed before a 251st update."""

        callback_count = int(self.ema_update_count.detach().cpu().item())
        if callback_count >= MAXIMUM_PERCEPTION_UPDATES_V8:
            raise RuntimeError("V8 exceeded its 250-update perception cap")
        super().update_target_ema_after_optimizer_step()

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
        """Evaluate exact grounding with persistence-only diagnostics.

        The predictor is intentionally never called.  Its former all-action
        output fields are populated by exact persistence of the current state,
        retaining the frozen objective result API for observation code.
        """

        if not self.phase_policy_armed_v6:
            raise RuntimeError("V8 objective used before phase policy was armed")
        callback_count = int(self.ema_update_count.detach().cpu().item())
        if callback_count > MAXIMUM_PERCEPTION_UPDATES_V8:
            raise RuntimeError("V8 exceeded its 250-update perception cap")
        if self.active_phase_v6 != PHASE_ONE_V6:
            raise RuntimeError("V8 has no authorized predictor objective")
        if (
            next_rgb.shape != current_rgb.shape
            or fixed_negative_rgb.shape != current_rgb.shape
        ):
            raise ValueError("current, next, and fixed-negative RGB shapes differ")
        executed = _v6._v3._v1._validate_action_one_hot(
            action_one_hot,
            batch=current_rgb.shape[0],
            reference=current_rgb,
        )
        if (
            non_hold_mask.shape != (current_rgb.shape[0],)
            or non_hold_mask.dtype != torch.bool
        ):
            raise TypeError("non_hold_mask must be boolean with shape (B,)")
        if not torch.equal(non_hold_mask, executed != HOLD_ACTION_INDEX_V1):
            raise ValueError("non_hold_mask differs from executed actions")

        current_state = self.online_state(current_rgb)
        next_online_state = self.online_state(next_rgb)
        target_next = self.target_state(next_rgb)
        target_current = self.target_state(current_rgb)
        target_mapped_negative = self.target_state(fixed_negative_rgb)
        persistence = current_state[:, None].expand(
            -1,
            len(ACTION_VOCABULARY_V1),
            -1,
            -1,
            -1,
        )
        base = direct_bev_state_objective_v1(
            current_state_logits=current_state,
            next_online_state_logits=next_online_state,
            all_action_prediction_logits=persistence,
            target_next_logits=target_next,
            target_current_logits=target_current,
            target_mapped_negative_logits=target_mapped_negative,
            current_labels=current_labels,
            next_labels=next_labels,
            executed_action_indices=executed,
            non_hold_mask=non_hold_mask,
        )
        return base._replace(total=base.G / math.log(2.0))


__all__ = [
    *_v6.__all__,
    "BEV_FEATURE_DIMENSION_V8",
    "BEV_QUERY_COLUMNS_V8",
    "BEV_QUERY_ROWS_V8",
    "CROSS_ATTENTION_HEADS_V8",
    "DirectEgocentricBevStateJepaV1",
    "FFN_HIDDEN_DIMENSION_V8",
    "L2_NORMALIZATION_EPSILON_V8",
    "LearnedBevQueryCrossAttentionFfnBlockV8",
    "LearnedBevQueryDecoderV8",
    "MAXIMUM_PERCEPTION_UPDATES_V8",
    "NormalizedPrototypeStateHeadV8",
    "ONLINE_DECODER_PROTOTYPE_PARAMETER_COUNT_V8",
    "ONLINE_DECODER_PROTOTYPE_PARAMETER_TENSOR_COUNT_V8",
    "RGB_TOKEN_COUNT_V8",
    "RGB_TOKEN_DIMENSION_V8",
    "STATE_CLASS_COUNT_V8",
    "_hard_hierarchical_loss_per_row",
]
