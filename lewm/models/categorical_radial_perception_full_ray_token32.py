"""Registered full-ray categorical perception with 32-wide image tokens."""
from __future__ import annotations

import torch

from .categorical_radial_perception import CategoricalRadialPerception
from .categorical_radial_perception_full_ray import (
    CategoricalRadialPerceptionFullRay,
    FullRayRadialContext,
)


REGISTERED_TOKEN_FEATURE_DIM = 32
REGISTERED_CONTEXT_DIM = 64
REGISTERED_PARAMETER_COUNT = 2_891_171
REGISTERED_SHAPE_CHANGED_STATE_KEYS = (
    "token_projection.weight",
    "token_projection.bias",
    "context_stem.0.weight",
)
REGISTERED_SHAPE_CHANGES = {
    "token_projection.weight": ((24, 192, 1, 1), (32, 192, 1, 1)),
    "token_projection.bias": ((24,), (32,)),
    "context_stem.0.weight": ((64, 154, 1, 1), (64, 194, 1, 1)),
}


class CategoricalRadialPerceptionFullRayToken32(CategoricalRadialPerception):
    """Full-ray candidate whose sole architecture change is token width 32."""

    def __init__(
        self,
        *,
        token_feature_dim: int = REGISTERED_TOKEN_FEATURE_DIM,
        context_dim: int = REGISTERED_CONTEXT_DIM,
    ) -> None:
        if int(token_feature_dim) != REGISTERED_TOKEN_FEATURE_DIM:
            raise ValueError("token-32 full-ray token_feature_dim is frozen at 32")
        if int(context_dim) != REGISTERED_CONTEXT_DIM:
            raise ValueError("token-32 full-ray context_dim is frozen at 64")
        super().__init__(
            token_feature_dim=token_feature_dim,
            context_dim=context_dim,
        )
        self.radial_context = FullRayRadialContext(self.context_dim)


def build_comparable_width24_and_token32_models(
    cpu_rng_state: torch.Tensor,
) -> tuple[
    CategoricalRadialPerceptionFullRay,
    CategoricalRadialPerceptionFullRayToken32,
]:
    """Construct width-24/32 models with strict, comparable initialization.

    Both models start from the supplied CPU RNG state. Every same-shape state
    entry is then copied from width 24 into width 32; only the three registered
    shape changes retain their deterministic width-32 initialization.
    """

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
    token32 = CategoricalRadialPerceptionFullRayToken32()

    width24_state = width24.state_dict()
    token32_state = token32.state_dict()
    if tuple(width24_state) != tuple(token32_state):
        raise RuntimeError("width-24 and token-32 state keys differ")

    changed_shapes = {
        name: (tuple(width24_state[name].shape), tuple(token32_state[name].shape))
        for name in width24_state
        if width24_state[name].shape != token32_state[name].shape
    }
    if changed_shapes != REGISTERED_SHAPE_CHANGES:
        raise RuntimeError(
            "token-32 state-shape changes do not match the frozen registration"
        )

    with torch.no_grad():
        for name, width24_value in width24_state.items():
            token32_value = token32_state[name]
            if token32_value.shape != width24_value.shape:
                continue
            if token32_value.dtype != width24_value.dtype:
                raise RuntimeError(f"state dtype differs for {name}")
            token32_value.copy_(width24_value)

    return width24, token32


__all__ = [
    "CategoricalRadialPerceptionFullRayToken32",
    "REGISTERED_CONTEXT_DIM",
    "REGISTERED_PARAMETER_COUNT",
    "REGISTERED_SHAPE_CHANGED_STATE_KEYS",
    "REGISTERED_SHAPE_CHANGES",
    "REGISTERED_TOKEN_FEATURE_DIM",
    "build_comparable_width24_and_token32_models",
]
