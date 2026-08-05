"""Capacity-parametrized dense readout for the observability-ceiling assay V1.

This generalizes the frozen 245-parameter dense shared spatial readout of
:mod:`lewm.models.go2_dinov2_dense_shared_spatial_readout_calibration_v1` so
that the PCA width ``K`` and the hidden/value width ``H = V`` are configurable.
The rung ``K=8, H=V=4`` reproduces the frozen architecture exactly, including
its 245-parameter inventory and its initialization order, so the assay contains
its own replication anchor.

The module has no filesystem, RGB, or encoder access.
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
POSITION_WIDTH = 2
CONDITION_WIDTH = 4

STATE_IDENTITY_SCHEMA = "lewm_go2_observability_ceiling_readout_state_identity_v1"

INITIALIZATION_ORDER = ("W_r", "W_p", "W_q", "W_v", "B", "w_alpha", "w_z")


def parameter_count_v1(pca_width: int, hidden_width: int) -> int:
    """Return ``3K(H+V) + 8H + 5V + 1`` for ``V = H``."""

    relational = 3 * int(pca_width)
    hidden = int(hidden_width)
    value = int(hidden_width)
    return (
        hidden * relational
        + hidden * POSITION_WIDTH
        + hidden * CONDITION_WIDTH
        + value * relational
        + value * CONDITION_WIDTH
        + hidden
        + value
        + hidden
        + 1
    )


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


class CeilingReadoutOutputV1(NamedTuple):
    """The scalar branch score and its dense spatial diagnostics."""

    score: torch.Tensor
    attention: torch.Tensor
    pooled_value: torch.Tensor


class CeilingReadoutV1(nn.Module):
    """Action- and goal-conditioned attention-pooled scalar readout.

    Inputs are relational patch features ``[z_c, z_s, z_s - z_c]`` with shape
    ``(B, 256, 3K)`` and condition vectors ``[goal_x/10, goal_y/10,
    requested_vx/0.30, requested_wz/0.45]`` with shape ``(B, 4)``.

    Direct construction produces the deterministic all-zero parameter state.
    Use :func:`initialize_ceiling_readout_v1` for scientific execution.
    """

    def __init__(self, pca_width: int, hidden_width: int) -> None:
        super().__init__()
        if isinstance(pca_width, bool) or not isinstance(pca_width, int) or pca_width < 1:
            raise ValueError("pca_width must be a positive int")
        if (
            isinstance(hidden_width, bool)
            or not isinstance(hidden_width, int)
            or hidden_width < 1
        ):
            raise ValueError("hidden_width must be a positive int")

        self.pca_width = int(pca_width)
        self.relational_width = 3 * int(pca_width)
        self.hidden_width = int(hidden_width)
        self.value_width = int(hidden_width)

        self._parameter_shapes = {
            "W_r": (self.hidden_width, self.relational_width),
            "W_p": (self.hidden_width, POSITION_WIDTH),
            "W_q": (self.hidden_width, CONDITION_WIDTH),
            "W_v": (self.value_width, self.relational_width),
            "B": (self.value_width, CONDITION_WIDTH),
            "w_alpha": (self.hidden_width,),
            "w_z": (self.value_width,),
            "b_h": (self.hidden_width,),
            "b_score": (),
        }

        self.W_r = nn.Parameter(torch.zeros(self._parameter_shapes["W_r"]))
        self.W_p = nn.Parameter(torch.zeros(self._parameter_shapes["W_p"]))
        self.W_q = nn.Parameter(torch.zeros(self._parameter_shapes["W_q"]))
        self.W_v = nn.Parameter(torch.zeros(self._parameter_shapes["W_v"]))
        self.B = nn.Parameter(torch.zeros(self._parameter_shapes["B"]))
        self.w_alpha = nn.Parameter(torch.zeros(self._parameter_shapes["w_alpha"]))
        self.w_z = nn.Parameter(torch.zeros(self._parameter_shapes["w_z"]))
        self.b_h = nn.Parameter(torch.zeros(self._parameter_shapes["b_h"]))
        self.b_score = nn.Parameter(torch.zeros(self._parameter_shapes["b_score"]))
        self.register_buffer(
            "patch_positions", _canonical_patch_positions_v1(), persistent=True
        )
        self._validate_inventory()

    @property
    def state_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            **self._parameter_shapes,
            "patch_positions": (PATCH_COUNT, POSITION_WIDTH),
        }

    def _validate_inventory(self) -> None:
        named = dict(self.named_parameters())
        if set(named) != set(self._parameter_shapes):
            raise RuntimeError("ceiling readout parameter inventory changed")
        expected = parameter_count_v1(self.pca_width, self.hidden_width)
        if sum(value.numel() for value in named.values()) != expected:
            raise RuntimeError("ceiling readout parameter count changed")
        devices = {parameter.device for parameter in named.values()}
        for name, shape in self._parameter_shapes.items():
            parameter = named[name]
            if tuple(parameter.shape) != shape:
                raise RuntimeError(f"ceiling readout parameter {name} shape changed")
            if parameter.dtype != torch.float32 or not parameter.requires_grad:
                raise RuntimeError(
                    f"ceiling readout parameter {name} must be trainable float32"
                )
        if len(devices) != 1:
            raise RuntimeError("ceiling readout parameters must share one device")
        device = next(iter(devices))
        if self.patch_positions.device != device:
            raise RuntimeError("ceiling readout state tensors must share one device")
        if not torch.equal(
            self.patch_positions, _canonical_patch_positions_v1().to(device)
        ):
            raise RuntimeError("ceiling readout patch positions changed")

    def _validate_inputs(
        self, relational_patch_features: torch.Tensor, condition: torch.Tensor
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
            != (PATCH_COUNT, self.relational_width)
        ):
            raise ValueError(
                "relational_patch_features must have shape "
                f"(B,{PATCH_COUNT},{self.relational_width}) with B >= 1"
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
            raise TypeError("ceiling readout inputs must use exact torch.float32")
        if (
            relational_patch_features.device != reference.device
            or condition.device != reference.device
        ):
            raise TypeError("ceiling readout inputs and parameters must share a device")
        if not bool(torch.isfinite(relational_patch_features).all()):
            raise FloatingPointError("relational_patch_features are nonfinite")
        if not bool(torch.isfinite(condition).all()):
            raise FloatingPointError("condition is nonfinite")

    def forward_with_attention(
        self, relational_patch_features: torch.Tensor, condition: torch.Tensor
    ) -> CeilingReadoutOutputV1:
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
            torch.einsum("bih,h->bi", hidden, self.w_alpha), dim=1
        )
        values = F.linear(relational_patch_features, self.W_v)
        pooled_value = torch.sum(attention.unsqueeze(-1) * values, dim=1)
        score = (
            torch.einsum("bi,i->b", pooled_value, self.w_z)
            + torch.einsum("bi,ij,bj->b", pooled_value, self.B, condition)
            + self.b_score
        )
        return CeilingReadoutOutputV1(
            score=score, attention=attention, pooled_value=pooled_value
        )

    def forward(
        self, relational_patch_features: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Return one lower-is-better residual score per branch."""

        return self.forward_with_attention(relational_patch_features, condition).score


def initialize_ceiling_readout_v1(
    seed: int, *, pca_width: int, hidden_width: int
) -> CeilingReadoutV1:
    """Construct the exact CPU Xavier state without consuming global RNG."""

    member_seed = _validate_seed(seed)
    model = CeilingReadoutV1(pca_width=pca_width, hidden_width=hidden_width)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(member_seed)
    with torch.no_grad():
        for name in INITIALIZATION_ORDER:
            parameter = getattr(model, name)
            target = parameter if parameter.ndim == 2 else parameter.view(1, -1)
            nn.init.xavier_uniform_(target, gain=1.0, generator=generator)
        model.b_h.zero_()
        model.b_score.zero_()
    model._validate_inventory()
    if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise RuntimeError("ceiling readout initialization produced nonfinite state")
    return model


def ceiling_state_identity_v1(
    model_or_state: CeilingReadoutV1 | Mapping[str, torch.Tensor],
    *,
    state_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> str:
    """Hash exact ordered names, dtypes, shapes, and contiguous CPU bytes."""

    if isinstance(model_or_state, CeilingReadoutV1):
        model_or_state._validate_inventory()
        shapes = model_or_state.state_shapes
        state: Mapping[str, torch.Tensor] = model_or_state.state_dict()
    else:
        if state_shapes is None:
            raise ValueError("state_shapes is required for a raw state mapping")
        shapes = dict(state_shapes)
        state = model_or_state
    if set(state) != set(shapes):
        raise ValueError("ceiling readout state tensor inventory changed")

    digest = hashlib.sha256()
    digest.update(STATE_IDENTITY_SCHEMA.encode("ascii") + b"\0")
    for name in sorted(shapes):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"ceiling readout state {name} must be a torch.Tensor")
        tensor = value.detach().cpu().contiguous()
        if tuple(tensor.shape) != tuple(shapes[name]):
            raise ValueError(f"ceiling readout state {name} shape changed")
        if tensor.dtype != torch.float32:
            raise TypeError(f"ceiling readout state {name} must use torch.float32")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"ceiling readout state {name} is nonfinite")
        header = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class PrivilegedIdentifiabilityMLPV1(nn.Module):
    """Unconstrained MLP on the privileged feature and condition.

    Amendment 1 control 2a.  This exists to establish that the dense rank is a
    learnable function of the successor physical state at all — that is, that
    the target and labels are sound — independently of whether the dense
    attention readout can express it.  It is deliberately *not* a member of the
    readout family under test.
    """

    def __init__(self, feature_width: int, hidden_width: int = 128) -> None:
        super().__init__()
        self.feature_width = int(feature_width)
        self.hidden_width = int(hidden_width)
        self.network = nn.Sequential(
            nn.Linear(self.feature_width + CONDITION_WIDTH, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, 1),
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.feature_width:
            raise ValueError("privileged features must have shape (B,feature_width)")
        if tuple(condition.shape) != (features.shape[0], CONDITION_WIDTH):
            raise ValueError("condition must have shape (B,4) for the same batch")
        if not bool(torch.isfinite(features).all()) or not bool(
            torch.isfinite(condition).all()
        ):
            raise FloatingPointError("privileged MLP inputs are nonfinite")
        return self.network(torch.cat((features, condition), dim=-1)).squeeze(-1)


def initialize_privileged_mlp_v1(
    seed: int, *, feature_width: int, hidden_width: int = 128
) -> PrivilegedIdentifiabilityMLPV1:
    """Construct the MLP under an isolated generator, as the readout does."""

    member_seed = _validate_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(member_seed)
    model = PrivilegedIdentifiabilityMLPV1(
        feature_width=feature_width, hidden_width=hidden_width
    )
    with torch.no_grad():
        for module in model.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0, generator=generator)
                module.bias.zero_()
    if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):
        raise RuntimeError("privileged MLP initialization produced nonfinite state")
    return model


__all__ = [
    "CONDITION_WIDTH",
    "CeilingReadoutOutputV1",
    "CeilingReadoutV1",
    "PrivilegedIdentifiabilityMLPV1",
    "initialize_privileged_mlp_v1",
    "INITIALIZATION_ORDER",
    "PATCH_COUNT",
    "PATCH_GRID_SIZE",
    "POSITION_WIDTH",
    "STATE_IDENTITY_SCHEMA",
    "ceiling_state_identity_v1",
    "initialize_ceiling_readout_v1",
    "parameter_count_v1",
]
