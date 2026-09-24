"""Science-identical Direct BEV V2 constructor with CPU-only RNG seeding.

This adapter reuses the complete V1 architecture, objective, and methods.  Its
only behavioral change is that fresh CPU module initialization seeds the CPU
default generator directly instead of calling ``torch.random.manual_seed``,
which also schedules accelerator-generator seeding.
"""
from __future__ import annotations

import copy
from typing import Mapping

import torch
import torch.nn as nn

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


class DirectEgocentricBevStateJepaV1(_v1.DirectEgocentricBevStateJepaV1):
    """V1 mechanism with fresh initialization confined to the CPU RNG."""

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
            self.predictor = _v1._ActionOnlyResidualPredictorV1()
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


__all__ = [
    *_v1.__all__,
    "_hard_hierarchical_loss_per_row",
]
