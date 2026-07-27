"""Direct BEV equal-weight multiprototype state-head adapter.

This source preserves the frozen V10 RGB encoder, learned-query BEV decoder,
final-class macro-grounding objective, predictor control, and target-EMA
policy.  Its only learned mechanism change replaces each single normalized
class prototype with four normalized prototypes and aggregates their logits
with an exact, equal-weight log-mean-exp.
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


_V10_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_state_jepa_v10_final_class_macro_grounding.py"
)
_V10_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_multiprototype_frozen_v10_model",
    _V10_SOURCE_PATH,
)
if _V10_SPEC is None or _V10_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V10 model source")
_v10 = importlib.util.module_from_spec(_V10_SPEC)
sys.modules[_V10_SPEC.name] = _v10
_V10_SPEC.loader.exec_module(_v10)

for _name in _v10.__all__:
    globals()[_name] = getattr(_v10, _name)


STATE_CLASS_COUNT_MULTIPROTOTYPE_V1 = 3
PROTOTYPES_PER_CLASS_MULTIPROTOTYPE_V1 = 4
PROTOTYPE_FEATURE_DIMENSION_MULTIPROTOTYPE_V1 = 64
PROTOTYPE_PARAMETER_SHAPE_MULTIPROTOTYPE_V1 = (3, 4, 64)
PROTOTYPE_PARAMETER_COUNT_MULTIPROTOTYPE_V1 = 768
ONLINE_DECODER_HEAD_PARAMETER_COUNT_MULTIPROTOTYPE_V1 = 88_384
ONLINE_DECODER_HEAD_PARAMETER_TENSOR_COUNT_MULTIPROTOTYPE_V1 = 31
MAXIMUM_PERCEPTION_UPDATES_MULTIPROTOTYPE_V1 = 250
LOG_PROTOTYPES_PER_CLASS_MULTIPROTOTYPE_V1 = math.log(4.0)


class EqualWeightMultiprototypeStateHeadV1(nn.Module):
    """Aggregate four normalized prototype distances per state class."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = PROTOTYPE_FEATURE_DIMENSION_MULTIPROTOTYPE_V1
        self.out_channels = STATE_CLASS_COUNT_MULTIPROTOTYPE_V1
        # This literal shape is part of the frozen scientific mechanism.
        self.prototypes = nn.Parameter(torch.empty(3, 4, 64))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def component_logits(self, cell_features: torch.Tensor) -> torch.Tensor:
        """Return the twelve pre-aggregation normalized-distance logits."""

        expected = (
            PROTOTYPE_FEATURE_DIMENSION_MULTIPROTOTYPE_V1,
            _v10.BEV_QUERY_ROWS_V8,
            _v10.BEV_QUERY_COLUMNS_V8,
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
            eps=_v10.L2_NORMALIZATION_EPSILON_V8,
        )
        normalized_prototypes = F.normalize(
            self.prototypes,
            p=2.0,
            dim=2,
            eps=_v10.L2_NORMALIZATION_EPSILON_V8,
        ).to(dtype=cell_features.dtype)
        return -(
            normalized_features[:, None, None, :, :, :]
            - normalized_prototypes[None, :, :, :, None, None]
        ).square().sum(dim=3)

    def forward(self, cell_features: torch.Tensor) -> torch.Tensor:
        """Return exact stable equal-weight log-mean-exp class logits."""

        return torch.logsumexp(self.component_logits(cell_features), dim=2) - (
            LOG_PROTOTYPES_PER_CLASS_MULTIPROTOTYPE_V1
        )


class DirectEgocentricBevStateJepaV1(_v10.DirectEgocentricBevStateJepaV1):
    """Frozen V10 stack with only the equal-weight multiprototype head."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: DirectEgocentricBevStateJepaV1Config | None = None,
    ) -> None:
        # The inherited construction history establishes the registered final
        # online/target equality and constructs the exact frozen decoder.
        super().__init__(n320_encoder_state_dict, config=config)

        caller_cpu_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(
                self.config.initialization_seed
            )
            # Advance through the exact frozen decoder draw sequence before
            # drawing the replacement head.  Verify that sequence still
            # reproduces the decoder already installed by the base class.
            decoder_draw_witness = _v10.LearnedBevQueryDecoderV8()
            installed_decoder_state = self.bev_decoder.state_dict()
            witness_decoder_state = decoder_draw_witness.state_dict()
            if (
                tuple(installed_decoder_state) != tuple(witness_decoder_state)
                or any(
                    not torch.equal(
                        installed_decoder_state[name],
                        witness_decoder_state[name],
                    )
                    for name in installed_decoder_state
                )
            ):
                raise RuntimeError("frozen decoder initialization changed")
            replacement_head = EqualWeightMultiprototypeStateHeadV1()
        finally:
            torch.random.set_rng_state(caller_cpu_rng)

        self.state_head = replacement_head
        self.target_state_head = copy.deepcopy(replacement_head)
        # The frozen construction history already leaves encoder and decoder
        # targets equal to their online modules; the replacement target head
        # is an exact detached copy.  Add no successor-level global sync.
        self._freeze_target()

        decoder_parameters = tuple(self.bev_decoder.parameters())
        head_parameters = tuple(self.state_head.parameters())
        if (
            tuple(self.state_head.prototypes.shape)
            != PROTOTYPE_PARAMETER_SHAPE_MULTIPROTOTYPE_V1
            or self.state_head.prototypes.numel()
            != PROTOTYPE_PARAMETER_COUNT_MULTIPROTOTYPE_V1
            or len(decoder_parameters) + len(head_parameters)
            != ONLINE_DECODER_HEAD_PARAMETER_TENSOR_COUNT_MULTIPROTOTYPE_V1
            or sum(
                value.numel()
                for value in (*decoder_parameters, *head_parameters)
            )
            != ONLINE_DECODER_HEAD_PARAMETER_COUNT_MULTIPROTOTYPE_V1
        ):
            raise RuntimeError("multiprototype decoder/head inventory changed")


__all__ = sorted({
    *_v10.__all__,
    "DirectEgocentricBevStateJepaV1",
    "EqualWeightMultiprototypeStateHeadV1",
    "LOG_PROTOTYPES_PER_CLASS_MULTIPROTOTYPE_V1",
    "MAXIMUM_PERCEPTION_UPDATES_MULTIPROTOTYPE_V1",
    "ONLINE_DECODER_HEAD_PARAMETER_COUNT_MULTIPROTOTYPE_V1",
    "ONLINE_DECODER_HEAD_PARAMETER_TENSOR_COUNT_MULTIPROTOTYPE_V1",
    "PROTOTYPES_PER_CLASS_MULTIPROTOTYPE_V1",
    "PROTOTYPE_FEATURE_DIMENSION_MULTIPROTOTYPE_V1",
    "PROTOTYPE_PARAMETER_COUNT_MULTIPROTOTYPE_V1",
    "PROTOTYPE_PARAMETER_SHAPE_MULTIPROTOTYPE_V1",
    "STATE_CLASS_COUNT_MULTIPROTOTYPE_V1",
})
