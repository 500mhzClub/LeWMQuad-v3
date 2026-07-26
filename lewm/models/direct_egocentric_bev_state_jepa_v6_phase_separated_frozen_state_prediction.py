"""Direct BEV V6 phase-separated frozen-state prediction adapter."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Mapping

import torch


_V3_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
_V3_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v6_phase_separated_frozen_v3_model",
    _V3_SOURCE_PATH,
)
if _V3_SPEC is None or _V3_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V3 model source")
_v3 = importlib.util.module_from_spec(_V3_SPEC)
sys.modules[_V3_SPEC.name] = _v3
_V3_SPEC.loader.exec_module(_v3)


ACTION_VOCABULARY_V1 = _v3.ACTION_VOCABULARY_V1
DirectBevStateObjectiveV1 = _v3.DirectBevStateObjectiveV1
DirectEgocentricBevStateJepaV1Config = (
    _v3.DirectEgocentricBevStateJepaV1Config
)
FREE_CLASS_V1 = _v3.FREE_CLASS_V1
HOLD_ACTION_INDEX_V1 = _v3.HOLD_ACTION_INDEX_V1
HierarchicalHardLossV1 = _v3.HierarchicalHardLossV1
OCCUPIED_CLASS_V1 = _v3.OCCUPIED_CLASS_V1
UNKNOWN_CLASS_V1 = _v3.UNKNOWN_CLASS_V1
WrongRgbGroundingControlV1 = _v3.WrongRgbGroundingControlV1
direct_bev_state_objective_v1 = _v3.direct_bev_state_objective_v1
hard_hierarchical_raster_loss_v1 = _v3.hard_hierarchical_raster_loss_v1
soft_hierarchical_state_energy_v1 = _v3.soft_hierarchical_state_energy_v1
_hard_hierarchical_loss_per_row = _v3._hard_hierarchical_loss_per_row


PHASE_ONE_LAST_CALLBACK_COUNT_V6 = 400
PHASE_ONE_V6 = "phase_one"
PHASE_TWO_V6 = "phase_two"


class DirectEgocentricBevStateJepaV1(
    _v3.DirectEgocentricBevStateJepaV1
):
    """Frozen V3 model with only V6's two-phase optimization policy."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: DirectEgocentricBevStateJepaV1Config | None = None,
    ) -> None:
        # These are deliberately plain Python attributes.  In particular,
        # none is a parameter or buffer and none enters ``state_dict``.
        object.__setattr__(self, "_v6_phase_policy_armed", False)
        object.__setattr__(self, "_v6_phase_override", None)
        object.__setattr__(self, "_v6_optimizer_for_integrity_probe", None)
        object.__setattr__(self, "_v6_target_update_callback_count", 0)
        object.__setattr__(self, "_v6_ema_arithmetic_update_count", 0)
        object.__setattr__(self, "_v6_boundary_hard_sync_count", 0)
        object.__setattr__(self, "_v6_phase_two_target_noop_count", 0)
        super().__init__(n320_encoder_state_dict, config=config)

    @property
    def phase_policy_armed_v6(self) -> bool:
        return bool(self._v6_phase_policy_armed)

    @property
    def active_phase_v6(self) -> str:
        """Return the effective phase, including a probe-only override."""

        if not self.phase_policy_armed_v6:
            raise RuntimeError("V6 phase policy is not armed")
        override = self._v6_phase_override
        if override is not None:
            return str(override)
        callback_count = int(self.ema_update_count.detach().cpu().item())
        return (
            PHASE_ONE_V6
            if callback_count < PHASE_ONE_LAST_CALLBACK_COUNT_V6
            else PHASE_TWO_V6
        )

    def phase_counters_v6(self) -> dict[str, int | bool]:
        """Return nonpersistent phase accounting for runner receipts."""

        callback_buffer = int(
            self.ema_update_count.detach().cpu().item()
        )
        return {
            "phase_policy_armed": self.phase_policy_armed_v6,
            "global_target_update_callback_count": callback_buffer,
            "target_update_callback_count": int(
                self._v6_target_update_callback_count
            ),
            "ema_arithmetic_update_count": int(
                self._v6_ema_arithmetic_update_count
            ),
            "boundary_hard_sync_count": int(
                self._v6_boundary_hard_sync_count
            ),
            "phase_two_target_noop_count": int(
                self._v6_phase_two_target_noop_count
            ),
            "perception_optimizer_update_count": min(
                callback_buffer,
                PHASE_ONE_LAST_CALLBACK_COUNT_V6,
            ),
            "predictor_optimizer_update_count": max(
                callback_buffer - PHASE_ONE_LAST_CALLBACK_COUNT_V6,
                0,
            ),
        }

    def arm_phase_schedule_v6(self) -> None:
        """Arm V6 after the inherited initialization receipt is complete."""

        if self.phase_policy_armed_v6:
            raise RuntimeError("V6 phase policy was already armed")
        if (
            int(self.ema_update_count.detach().cpu().item()) != 0
            or self._v6_target_update_callback_count != 0
            or self._v6_ema_arithmetic_update_count != 0
            or self._v6_boundary_hard_sync_count != 0
            or self._v6_phase_two_target_noop_count != 0
            or self._v6_phase_override is not None
        ):
            raise RuntimeError("V6 phase policy cannot arm after state changed")
        object.__setattr__(self, "_v6_phase_policy_armed", True)
        self.apply_phase_policy_v6()

    def set_phase_override_for_integrity_probe_v6(
        self,
        phase: str | None,
    ) -> None:
        """Temporarily select one registered objective path for a probe."""

        if not self.phase_policy_armed_v6:
            raise RuntimeError("V6 phase policy is not armed")
        if phase is not None and phase not in (PHASE_ONE_V6, PHASE_TWO_V6):
            raise ValueError(
                "V6 integrity-probe phase must be phase_one, phase_two, or None"
            )
        object.__setattr__(self, "_v6_phase_override", phase)
        self.apply_phase_policy_v6()

    def apply_phase_policy_v6(self) -> None:
        """Apply trainability and module modes for the effective phase."""

        if not self.phase_policy_armed_v6:
            return
        phase = self.active_phase_v6
        phase_one = phase == PHASE_ONE_V6
        for module in self._online_modules():
            module.requires_grad_(phase_one)
        self.predictor.requires_grad_(not phase_one)
        self._freeze_target()

        if phase_one:
            for module in self._online_modules():
                module.train(bool(self.training))
            # The inactive predictor has no stochastic role in phase one.
            self.predictor.eval()
        else:
            for module in self._online_modules():
                module.eval()
            # This remains true even while the observation wrapper has put
            # the root module in eval mode.
            self.predictor.train(True)
        for module in self._target_modules():
            module.eval()

    def train(self, mode: bool = True) -> DirectEgocentricBevStateJepaV1:
        """Preserve V6's submodule policy across train/eval round trips."""

        result = super().train(mode)
        if self.phase_policy_armed_v6:
            self.apply_phase_policy_v6()
        return result

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
        """Evaluate frozen V3 components and replace only the phase total."""

        if not self.phase_policy_armed_v6:
            raise RuntimeError("V6 objective used before phase policy was armed")
        base = super().training_objective(
            current_rgb=current_rgb,
            next_rgb=next_rgb,
            fixed_negative_rgb=fixed_negative_rgb,
            action_one_hot=action_one_hot,
            non_hold_mask=non_hold_mask,
            current_labels=current_labels,
            next_labels=next_labels,
        )
        if self.active_phase_v6 == PHASE_ONE_V6:
            total = base.G / math.log(2.0)
        else:
            total = base.J / math.log(2.0) + base.C
        return base._replace(total=total)

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        """Perform 400 EMA calls, one sync, then counted target no-ops."""

        if not self.phase_policy_armed_v6:
            raise RuntimeError("V6 target update used before policy was armed")
        if self._v6_phase_override is not None:
            raise RuntimeError("V6 target update is forbidden during a probe")
        before = int(self.ema_update_count.detach().cpu().item())
        if before != self._v6_target_update_callback_count:
            raise RuntimeError("V6 target callback accounting diverged")

        if before < PHASE_ONE_LAST_CALLBACK_COUNT_V6:
            if (
                self._v6_ema_arithmetic_update_count != before
                or self._v6_boundary_hard_sync_count != 0
                or self._v6_phase_two_target_noop_count != 0
            ):
                raise RuntimeError("V6 phase-one target accounting diverged")
            super().update_target_ema_after_optimizer_step()
            object.__setattr__(
                self,
                "_v6_ema_arithmetic_update_count",
                self._v6_ema_arithmetic_update_count + 1,
            )
            object.__setattr__(
                self,
                "_v6_target_update_callback_count",
                self._v6_target_update_callback_count + 1,
            )
            if int(self.ema_update_count.detach().cpu().item()) == (
                PHASE_ONE_LAST_CALLBACK_COUNT_V6
            ):
                preserved_count = self.ema_update_count.detach().clone()
                super().hard_sync_target_from_online()
                self.ema_update_count.copy_(preserved_count)
                object.__setattr__(
                    self,
                    "_v6_boundary_hard_sync_count",
                    self._v6_boundary_hard_sync_count + 1,
                )
        else:
            if (
                self._v6_ema_arithmetic_update_count
                != PHASE_ONE_LAST_CALLBACK_COUNT_V6
                or self._v6_boundary_hard_sync_count != 1
                or self._v6_phase_two_target_noop_count
                != before - PHASE_ONE_LAST_CALLBACK_COUNT_V6
            ):
                raise RuntimeError("V6 phase-two target accounting diverged")
            self.ema_update_count.add_(1)
            object.__setattr__(
                self,
                "_v6_phase_two_target_noop_count",
                self._v6_phase_two_target_noop_count + 1,
            )
            object.__setattr__(
                self,
                "_v6_target_update_callback_count",
                self._v6_target_update_callback_count + 1,
            )
        self.apply_phase_policy_v6()


__all__ = [
    *_v3.__all__,
    "PHASE_ONE_LAST_CALLBACK_COUNT_V6",
    "PHASE_ONE_V6",
    "PHASE_TWO_V6",
]
