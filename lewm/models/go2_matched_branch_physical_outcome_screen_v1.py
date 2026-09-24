"""Fixed physical-outcome MLP for the matched-branch development screen V1.

The model is intentionally limited to the preregistered ``28 -> 16 -> 4``
one-hidden-layer tanh map.  Feature construction, standardisation, residual
decoding, ensembling, scoring, and evaluation remain responsibilities of the
experiment runner.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_WIDTH = 28
HIDDEN_WIDTH = 16
OUTPUT_WIDTH = 4

PARAMETER_COUNT = 532
PARAMETER_TENSOR_COUNT = 4
STATE_IDENTITY_SCHEMA = (
    "lewm_go2_matched_branch_physical_outcome_mlp_state_identity_v1"
)

INITIALIZATION_ORDER = ("input_weight", "output_weight")

_PARAMETER_SHAPES = {
    "input_weight": (HIDDEN_WIDTH, INPUT_WIDTH),
    "input_bias": (HIDDEN_WIDTH,),
    "output_weight": (OUTPUT_WIDTH, HIDDEN_WIDTH),
    "output_bias": (OUTPUT_WIDTH,),
}


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a Python int")
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in the closed range [0, 2**63 - 1]")
    return seed


class PhysicalOutcomeMLPV1(nn.Module):
    """Exact ``28 -> 16 -> 4`` float32 tanh MLP.

    Direct construction yields an all-zero state.  Scientific execution must
    use :func:`initialize_physical_outcome_mlp_v1` so that both learned arms
    receive the preregistered dedicated-generator Xavier initialisation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_weight = nn.Parameter(
            torch.zeros((HIDDEN_WIDTH, INPUT_WIDTH), dtype=torch.float32)
        )
        self.input_bias = nn.Parameter(
            torch.zeros((HIDDEN_WIDTH,), dtype=torch.float32)
        )
        self.output_weight = nn.Parameter(
            torch.zeros((OUTPUT_WIDTH, HIDDEN_WIDTH), dtype=torch.float32)
        )
        self.output_bias = nn.Parameter(
            torch.zeros((OUTPUT_WIDTH,), dtype=torch.float32)
        )
        self._validate_inventory()

    def _validate_inventory(self) -> None:
        parameters = dict(self.named_parameters())
        if set(parameters) != set(_PARAMETER_SHAPES):
            raise RuntimeError("physical-outcome MLP parameter inventory changed")
        if len(parameters) != PARAMETER_TENSOR_COUNT:
            raise RuntimeError("physical-outcome MLP parameter tensor count changed")
        if sum(value.numel() for value in parameters.values()) != PARAMETER_COUNT:
            raise RuntimeError("physical-outcome MLP parameter count changed")

        devices = {parameter.device for parameter in parameters.values()}
        if len(devices) != 1:
            raise RuntimeError("physical-outcome MLP parameters must share one device")
        for name, expected_shape in _PARAMETER_SHAPES.items():
            parameter = parameters[name]
            if tuple(parameter.shape) != expected_shape:
                raise RuntimeError(
                    f"physical-outcome MLP parameter {name} shape changed"
                )
            if parameter.dtype != torch.float32 or not parameter.requires_grad:
                raise RuntimeError(
                    f"physical-outcome MLP parameter {name} must be trainable float32"
                )
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(
                    f"physical-outcome MLP parameter {name} is nonfinite"
                )

    def _validate_input(self, features: torch.Tensor) -> None:
        self._validate_inventory()
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if (
            features.ndim != 2
            or features.shape[0] < 1
            or features.shape[1] != INPUT_WIDTH
        ):
            raise ValueError("features must have shape (B,28) with B >= 1")
        if features.dtype != torch.float32:
            raise TypeError("features must use exact torch.float32")
        if features.device != self.input_weight.device:
            raise TypeError("features and model parameters must share a device")
        if not bool(torch.isfinite(features).all()):
            raise FloatingPointError("features contain a nonfinite value")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict four standardized action-mean residual outcomes."""

        self._validate_input(features)
        hidden = torch.tanh(F.linear(features, self.input_weight, self.input_bias))
        prediction = F.linear(hidden, self.output_weight, self.output_bias)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("physical-outcome prediction became nonfinite")
        return prediction


def initialize_physical_outcome_mlp_v1(seed: int) -> PhysicalOutcomeMLPV1:
    """Construct the exact CPU Xavier/zero state without using global RNG."""

    member_seed = _validate_seed(seed)
    model = PhysicalOutcomeMLPV1()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(member_seed)

    with torch.no_grad():
        for name in INITIALIZATION_ORDER:
            nn.init.xavier_uniform_(
                getattr(model, name),
                gain=1.0,
                generator=generator,
            )
        model.input_bias.zero_()
        model.output_bias.zero_()
    model._validate_inventory()
    return model


def _state_mapping_v1(
    model_or_state: PhysicalOutcomeMLPV1 | Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if isinstance(model_or_state, PhysicalOutcomeMLPV1):
        model_or_state._validate_inventory()
        return model_or_state.state_dict()
    if not isinstance(model_or_state, Mapping):
        raise TypeError("model_or_state must be a physical-outcome MLP or state mapping")
    return model_or_state


def physical_outcome_state_identity_v1(
    model_or_state: PhysicalOutcomeMLPV1 | Mapping[str, torch.Tensor],
) -> str:
    """Hash exact tensor names, dtypes, shapes, and contiguous CPU bytes."""

    state = _state_mapping_v1(model_or_state)
    if set(state) != set(_PARAMETER_SHAPES):
        raise ValueError("physical-outcome MLP state tensor inventory changed")

    digest = hashlib.sha256()
    digest.update(STATE_IDENTITY_SCHEMA.encode("ascii") + b"\0")
    for name in sorted(_PARAMETER_SHAPES):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"physical-outcome MLP state {name} must be a torch.Tensor"
            )
        tensor = value.detach().cpu().contiguous()
        if tuple(tensor.shape) != _PARAMETER_SHAPES[name]:
            raise ValueError(f"physical-outcome MLP state {name} shape changed")
        if tensor.dtype != torch.float32:
            raise TypeError(
                f"physical-outcome MLP state {name} must use torch.float32"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(
                f"physical-outcome MLP state {name} is nonfinite"
            )

        header = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "HIDDEN_WIDTH",
    "INITIALIZATION_ORDER",
    "INPUT_WIDTH",
    "OUTPUT_WIDTH",
    "PARAMETER_COUNT",
    "PARAMETER_TENSOR_COUNT",
    "PhysicalOutcomeMLPV1",
    "STATE_IDENTITY_SCHEMA",
    "initialize_physical_outcome_mlp_v1",
    "physical_outcome_state_identity_v1",
]
