"""Dense shared DINOv2 spatial readout for the development calibration V1.

This module implements only the preregistered 245-parameter oracle-future
readout.  PCA construction, residual fitting, ensembling, controls, and
evaluation live in the calibration runner rather than in this model module.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PATCH_GRID_SIZE = 16
PATCH_COUNT = PATCH_GRID_SIZE * PATCH_GRID_SIZE
PCA_WIDTH = 8
RELATIONAL_WIDTH = 3 * PCA_WIDTH
POSITION_WIDTH = 2
CONDITION_WIDTH = 4
HIDDEN_WIDTH = 4
VALUE_WIDTH = 4

PARAMETER_COUNT = 245
PARAMETER_TENSOR_COUNT = 9
STATE_IDENTITY_SCHEMA = (
    "lewm_go2_dinov2_dense_shared_spatial_readout_state_identity_v1"
)

INITIALIZATION_ORDER = (
    "W_r",
    "W_p",
    "W_q",
    "W_v",
    "B",
    "w_alpha",
    "w_z",
)

_PARAMETER_SHAPES = {
    "W_r": (HIDDEN_WIDTH, RELATIONAL_WIDTH),
    "W_p": (HIDDEN_WIDTH, POSITION_WIDTH),
    "W_q": (HIDDEN_WIDTH, CONDITION_WIDTH),
    "W_v": (VALUE_WIDTH, RELATIONAL_WIDTH),
    "B": (VALUE_WIDTH, CONDITION_WIDTH),
    "w_alpha": (HIDDEN_WIDTH,),
    "w_z": (VALUE_WIDTH,),
    "b_h": (HIDDEN_WIDTH,),
    "b_score": (),
}
_STATE_SHAPES = {
    **_PARAMETER_SHAPES,
    "patch_positions": (PATCH_COUNT, POSITION_WIDTH),
}


class DenseSharedSpatialReadoutOutputV1(NamedTuple):
    """The scalar branch score and its dense spatial diagnostics."""

    score: torch.Tensor
    attention: torch.Tensor
    pooled_value: torch.Tensor


def _canonical_patch_positions_v1() -> torch.Tensor:
    rows = torch.arange(PATCH_GRID_SIZE, dtype=torch.float32)
    columns = torch.arange(PATCH_GRID_SIZE, dtype=torch.float32)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    u = 2.0 * (column_grid + 0.5) / float(PATCH_GRID_SIZE) - 1.0
    v = 2.0 * (row_grid + 0.5) / float(PATCH_GRID_SIZE) - 1.0
    return torch.stack((u, v), dim=-1).reshape(PATCH_COUNT, POSITION_WIDTH)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be a Python int")
    if seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in the closed range [0, 2**63 - 1]")
    return seed


class DenseSharedSpatialReadoutV1(nn.Module):
    """Exact dense action-and-goal-conditioned scalar readout.

    Inputs are relational patch features ``[z_c, z_s, z_s-z_c]`` with shape
    ``(B,256,24)`` and condition vectors ``[goal_x/10, goal_y/10,
    requested_vx/0.30, requested_wz/0.45]`` with shape ``(B,4)``.

    Direct construction produces the deterministic all-zero parameter state.
    Scientific execution must use
    :func:`initialize_dense_shared_spatial_readout_v1`.
    """

    def __init__(self) -> None:
        super().__init__()

        # Registration and initialization order are intentionally explicit.
        self.W_r = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["W_r"]))
        self.W_p = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["W_p"]))
        self.W_q = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["W_q"]))
        self.W_v = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["W_v"]))
        self.B = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["B"]))
        self.w_alpha = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["w_alpha"]))
        self.w_z = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["w_z"]))
        self.b_h = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["b_h"]))
        self.b_score = nn.Parameter(torch.zeros(_PARAMETER_SHAPES["b_score"]))
        self.register_buffer(
            "patch_positions",
            _canonical_patch_positions_v1(),
            persistent=True,
        )
        self._validate_inventory()

    def _validate_inventory(self) -> None:
        named_parameters = dict(self.named_parameters())
        if set(named_parameters) != set(_PARAMETER_SHAPES):
            raise RuntimeError("dense readout parameter inventory changed")
        if len(named_parameters) != PARAMETER_TENSOR_COUNT:
            raise RuntimeError("dense readout parameter tensor count changed")
        if sum(value.numel() for value in named_parameters.values()) != PARAMETER_COUNT:
            raise RuntimeError("dense readout parameter count changed")

        devices = {parameter.device for parameter in named_parameters.values()}
        for name, expected_shape in _PARAMETER_SHAPES.items():
            parameter = named_parameters[name]
            if tuple(parameter.shape) != expected_shape:
                raise RuntimeError(f"dense readout parameter {name} shape changed")
            if parameter.dtype != torch.float32 or not parameter.requires_grad:
                raise RuntimeError(
                    f"dense readout parameter {name} must be trainable float32"
                )
        if len(devices) != 1:
            raise RuntimeError("dense readout parameters must share one device")

        if tuple(self.patch_positions.shape) != _STATE_SHAPES["patch_positions"]:
            raise RuntimeError("dense readout patch-position shape changed")
        if self.patch_positions.dtype != torch.float32:
            raise RuntimeError("dense readout patch positions must be float32")
        parameter_device = next(iter(devices))
        if self.patch_positions.device != parameter_device:
            raise RuntimeError("dense readout state tensors must share one device")
        expected_positions = _canonical_patch_positions_v1().to(parameter_device)
        if not torch.equal(self.patch_positions, expected_positions):
            raise RuntimeError("dense readout patch positions changed")

    def _validate_inputs(
        self,
        relational_patch_features: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        self._validate_inventory()
        if not isinstance(relational_patch_features, torch.Tensor):
            raise TypeError("relational_patch_features must be a torch.Tensor")
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor")
        if (
            relational_patch_features.ndim != 3
            or relational_patch_features.shape[0] < 1
            or tuple(relational_patch_features.shape[1:])
            != (PATCH_COUNT, RELATIONAL_WIDTH)
        ):
            raise ValueError(
                "relational_patch_features must have shape (B,256,24) with B >= 1"
            )
        if tuple(condition.shape) != (
            relational_patch_features.shape[0],
            CONDITION_WIDTH,
        ):
            raise ValueError("condition must have shape (B,4) for the same batch")

        reference = self.W_r
        if (
            relational_patch_features.dtype != torch.float32
            or condition.dtype != torch.float32
        ):
            raise TypeError("dense readout inputs must use exact torch.float32")
        if (
            relational_patch_features.device != reference.device
            or condition.device != reference.device
        ):
            raise TypeError("dense readout inputs and parameters must share a device")
        if not bool(torch.isfinite(relational_patch_features).all()):
            raise FloatingPointError("relational_patch_features are nonfinite")
        if not bool(torch.isfinite(condition).all()):
            raise FloatingPointError("condition is nonfinite")

    def forward_with_attention(
        self,
        relational_patch_features: torch.Tensor,
        condition: torch.Tensor,
    ) -> DenseSharedSpatialReadoutOutputV1:
        """Return exact scores together with attention and pooled values."""

        self._validate_inputs(relational_patch_features, condition)
        batch = relational_patch_features.shape[0]
        positions = self.patch_positions.unsqueeze(0).expand(batch, -1, -1)

        hidden = torch.tanh(
            F.linear(relational_patch_features, self.W_r)
            + F.linear(positions, self.W_p)
            + F.linear(condition, self.W_q).unsqueeze(1)
            + self.b_h
        )
        attention = torch.softmax(
            torch.einsum("bih,h->bi", hidden, self.w_alpha),
            dim=1,
        )
        values = F.linear(relational_patch_features, self.W_v)
        pooled_value = torch.sum(attention.unsqueeze(-1) * values, dim=1)
        score = (
            torch.einsum("bi,i->b", pooled_value, self.w_z)
            + torch.einsum("bi,ij,bj->b", pooled_value, self.B, condition)
            + self.b_score
        )
        return DenseSharedSpatialReadoutOutputV1(
            score=score,
            attention=attention,
            pooled_value=pooled_value,
        )

    def forward(
        self,
        relational_patch_features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Return one lower-is-better residual score per branch."""

        return self.forward_with_attention(relational_patch_features, condition).score


def initialize_dense_shared_spatial_readout_v1(
    seed: int,
) -> DenseSharedSpatialReadoutV1:
    """Construct the exact CPU Xavier state without consuming global RNG."""

    member_seed = _validate_seed(seed)
    model = DenseSharedSpatialReadoutV1()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(member_seed)

    with torch.no_grad():
        for name in INITIALIZATION_ORDER:
            parameter = getattr(model, name)
            target = parameter if parameter.ndim == 2 else parameter.view(1, -1)
            nn.init.xavier_uniform_(
                target,
                gain=1.0,
                generator=generator,
            )
        model.b_h.zero_()
        model.b_score.zero_()
    model._validate_inventory()
    if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("dense readout initialization produced nonfinite state")
    return model


def _state_mapping_v1(
    model_or_state: DenseSharedSpatialReadoutV1 | Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if isinstance(model_or_state, DenseSharedSpatialReadoutV1):
        model_or_state._validate_inventory()
        return model_or_state.state_dict()
    if not isinstance(model_or_state, Mapping):
        raise TypeError("model_or_state must be a dense readout or state mapping")
    return model_or_state


def dense_shared_state_identity_v1(
    model_or_state: DenseSharedSpatialReadoutV1 | Mapping[str, torch.Tensor],
) -> str:
    """Hash exact ordered names, dtypes, shapes, and contiguous CPU bytes."""

    state = _state_mapping_v1(model_or_state)
    if set(state) != set(_STATE_SHAPES):
        raise ValueError("dense readout state tensor inventory changed")

    digest = hashlib.sha256()
    digest.update(STATE_IDENTITY_SCHEMA.encode("ascii") + b"\0")
    for name in sorted(_STATE_SHAPES):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"dense readout state {name} must be a torch.Tensor")
        tensor = value.detach().cpu().contiguous()
        if tuple(tensor.shape) != _STATE_SHAPES[name]:
            raise ValueError(f"dense readout state {name} shape changed")
        if tensor.dtype != torch.float32:
            raise TypeError(f"dense readout state {name} must use torch.float32")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"dense readout state {name} is nonfinite")
        if name == "patch_positions" and not torch.equal(
            tensor, _canonical_patch_positions_v1()
        ):
            raise ValueError("dense readout patch-position state changed")

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
    "CONDITION_WIDTH",
    "DenseSharedSpatialReadoutOutputV1",
    "DenseSharedSpatialReadoutV1",
    "HIDDEN_WIDTH",
    "INITIALIZATION_ORDER",
    "PARAMETER_COUNT",
    "PARAMETER_TENSOR_COUNT",
    "PATCH_COUNT",
    "PATCH_GRID_SIZE",
    "PCA_WIDTH",
    "POSITION_WIDTH",
    "RELATIONAL_WIDTH",
    "STATE_IDENTITY_SCHEMA",
    "VALUE_WIDTH",
    "dense_shared_state_identity_v1",
    "initialize_dense_shared_spatial_readout_v1",
]
