"""Full-ray categorical perception with explicit hierarchical probabilities."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .categorical_radial_perception_full_ray import (
    CategoricalRadialPerceptionFullRay,
    REGISTERED_CONTEXT_DIM,
    REGISTERED_TOKEN_FEATURE_DIM,
)


EXECUTION_BINDING_SHA256 = (
    "bb691c787af0b90f813ced4e5e521f1b15b70b75c836147cd69275c50df6b5d3"
)
REGISTERED_FACTOR_NAMES = ("known", "occupied_given_known")
REGISTERED_CLASS_NAMES = ("unknown", "free", "occupied")
REGISTERED_FACTOR_COUNT = len(REGISTERED_FACTOR_NAMES)
REGISTERED_PARAMETER_COUNT = 2_887_002
REGISTERED_STATE_ENTRY_COUNT = 133
REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT = 131
REGISTERED_SHAPE_CHANGED_STATE_KEYS = (
    "polar_head.weight",
    "polar_head.bias",
)
REGISTERED_SHAPE_CHANGES = {
    "polar_head.weight": ((3, 64, 1, 1), (2, 64, 1, 1)),
    "polar_head.bias": ((3,), (2,)),
}


def hierarchical_factors_to_joint_log_probabilities(
    factors: torch.Tensor,
) -> torch.Tensor:
    """Convert KNOWN and OCCUPIED-given-KNOWN logits to joint log probabilities."""

    if not isinstance(factors, torch.Tensor):
        raise TypeError("factors must be a torch.Tensor")
    if not factors.is_floating_point():
        raise ValueError("factors must use a floating-point dtype")
    if factors.ndim < 2 or factors.shape[1] != REGISTERED_FACTOR_COUNT:
        raise ValueError("factors must have shape (B, 2, ...)")

    known_logit = factors[:, 0]
    occupied_given_known_logit = factors[:, 1]
    log_unknown = F.logsigmoid(-known_logit)
    log_known = F.logsigmoid(known_logit)
    log_free = log_known + F.logsigmoid(-occupied_given_known_logit)
    log_occupied = log_known + F.logsigmoid(occupied_given_known_logit)
    joint = torch.stack((log_unknown, log_free, log_occupied), dim=1)
    return joint - torch.logsumexp(joint, dim=1, keepdim=True)


class CategoricalRadialPerceptionFullRayHierarchical(
    CategoricalRadialPerceptionFullRay
):
    """Width-24 full-ray model with a two-factor categorical output head."""

    def __init__(
        self,
        *,
        token_feature_dim: int = REGISTERED_TOKEN_FEATURE_DIM,
        context_dim: int = REGISTERED_CONTEXT_DIM,
    ) -> None:
        super().__init__(
            token_feature_dim=token_feature_dim,
            context_dim=context_dim,
        )
        self.polar_head = nn.Conv2d(
            self.context_dim,
            REGISTERED_FACTOR_COUNT,
            kernel_size=1,
        )

    def raw_hierarchical_polar_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Return ``(B,2,64,256)`` KNOWN and OCCUPIED-given-KNOWN logits."""

        return super().polar_logits(image)

    def polar_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Return normalized ``(B,3,64,256)`` joint log probabilities."""

        factors = self.raw_hierarchical_polar_logits(image)
        return hierarchical_factors_to_joint_log_probabilities(factors)


def build_comparable_width24_and_hierarchical_models(
    cpu_rng_state: torch.Tensor,
) -> tuple[
    CategoricalRadialPerceptionFullRay,
    CategoricalRadialPerceptionFullRayHierarchical,
]:
    """Construct V2/V4 models with bit-identical shared initialization."""

    if not isinstance(cpu_rng_state, torch.Tensor):
        raise TypeError("cpu_rng_state must be a torch.Tensor")
    if (
        cpu_rng_state.device.type != "cpu"
        or cpu_rng_state.dtype != torch.uint8
        or cpu_rng_state.ndim != 1
    ):
        raise ValueError("cpu_rng_state must be a one-dimensional CPU uint8 tensor")
    replay_state = cpu_rng_state.detach().clone()

    torch.set_rng_state(replay_state)
    width24 = CategoricalRadialPerceptionFullRay()
    torch.set_rng_state(replay_state)
    hierarchical = CategoricalRadialPerceptionFullRayHierarchical()

    width24_state = width24.state_dict()
    hierarchical_state = hierarchical.state_dict()
    if tuple(width24_state) != tuple(hierarchical_state):
        raise RuntimeError("V2 and hierarchical state keys differ")
    if len(width24_state) != REGISTERED_STATE_ENTRY_COUNT:
        raise RuntimeError("registered V2/V4 state-entry count changed")

    changed_shapes = {
        name: (
            tuple(width24_state[name].shape),
            tuple(hierarchical_state[name].shape),
        )
        for name in width24_state
        if width24_state[name].shape != hierarchical_state[name].shape
    }
    if changed_shapes != REGISTERED_SHAPE_CHANGES:
        raise RuntimeError(
            "hierarchical state-shape changes do not match the frozen registration"
        )

    copied = 0
    with torch.no_grad():
        for name, width24_value in width24_state.items():
            hierarchical_value = hierarchical_state[name]
            if hierarchical_value.shape != width24_value.shape:
                continue
            if hierarchical_value.dtype != width24_value.dtype:
                raise RuntimeError(f"state dtype differs for {name}")
            hierarchical_value.copy_(width24_value)
            copied += 1
    if copied != REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT:
        raise RuntimeError("registered V2/V4 same-shape state count changed")

    return width24, hierarchical


__all__ = [
    "CategoricalRadialPerceptionFullRayHierarchical",
    "EXECUTION_BINDING_SHA256",
    "REGISTERED_CLASS_NAMES",
    "REGISTERED_CONTEXT_DIM",
    "REGISTERED_FACTOR_COUNT",
    "REGISTERED_FACTOR_NAMES",
    "REGISTERED_PARAMETER_COUNT",
    "REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT",
    "REGISTERED_SHAPE_CHANGED_STATE_KEYS",
    "REGISTERED_SHAPE_CHANGES",
    "REGISTERED_STATE_ENTRY_COUNT",
    "REGISTERED_TOKEN_FEATURE_DIM",
    "build_comparable_width24_and_hierarchical_models",
    "hierarchical_factors_to_joint_log_probabilities",
]
