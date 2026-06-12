"""Canonical LeWM action-block encoding.

The current flat-JEPA training corpus stores one macro action as five executed
Go2 command ticks. Keep this representation explicit so trainers, planners,
and evaluation code do not silently disagree about the 15-D layout.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ACTION_BLOCK_SIZE = 5
COMMAND_FIELDS = (
    "executed_vx_body_mps",
    "executed_vy_body_mps",
    "executed_yaw_rate_radps",
)
COMMAND_DIM_PER_TICK = len(COMMAND_FIELDS)
ACTIVE_BLOCK_DIM = ACTION_BLOCK_SIZE * COMMAND_DIM_PER_TICK

# This matches the in-flight 2026-05 flat-JEPA sweep and the v03 rendered
# corpus trainer: all vx ticks, then all vy ticks, then all yaw-rate ticks.
ACTIVE_BLOCK_ORDER = "channel_major_vx_vy_yaw"
ACTIVE_BLOCK_DESCRIPTION = (
    "concat(vx_body_mps[0:K], vy_body_mps[0:K], yaw_rate_radps[0:K])"
)


def active_block_metadata() -> dict[str, Any]:
    """Return serializable metadata to persist with checkpoints."""

    return {
        "command_representation": "active_block",
        "active_block_order": ACTIVE_BLOCK_ORDER,
        "active_block_description": ACTIVE_BLOCK_DESCRIPTION,
        "action_block_size": ACTION_BLOCK_SIZE,
        "command_dim_per_tick": COMMAND_DIM_PER_TICK,
        "cmd_dim": ACTIVE_BLOCK_DIM,
        "command_fields": list(COMMAND_FIELDS),
    }


def encode_active_block(
    vx_body_mps: Sequence[float],
    vy_body_mps: Sequence[float],
    yaw_rate_radps: Sequence[float],
) -> np.ndarray:
    """Encode one K-step command block as the canonical 15-D active block."""

    arrays = [
        np.asarray(vx_body_mps, dtype=np.float32),
        np.asarray(vy_body_mps, dtype=np.float32),
        np.asarray(yaw_rate_radps, dtype=np.float32),
    ]
    for field, arr in zip(COMMAND_FIELDS, arrays, strict=True):
        if arr.shape != (ACTION_BLOCK_SIZE,):
            raise ValueError(
                f"{field} must have shape ({ACTION_BLOCK_SIZE},), got {arr.shape}"
            )
    return np.concatenate(arrays).astype(np.float32, copy=False)


def encode_executed_command_block(payload: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Encode an ``ExecutedCommandBlock`` payload as a canonical active block."""

    missing = [field for field in COMMAND_FIELDS if field not in payload]
    if missing:
        raise KeyError(f"ExecutedCommandBlock payload missing fields: {missing}")
    return encode_active_block(
        payload["executed_vx_body_mps"],
        payload["executed_vy_body_mps"],
        payload["executed_yaw_rate_radps"],
    )


def active_block_to_matrix(active_block: Sequence[float]) -> np.ndarray:
    """Decode a canonical active block to rows of ``[vx, vy, yaw_rate]``."""

    arr = np.asarray(active_block, dtype=np.float32)
    if arr.shape != (ACTIVE_BLOCK_DIM,):
        raise ValueError(f"active_block must have shape ({ACTIVE_BLOCK_DIM},), got {arr.shape}")
    return np.stack(
        [
            arr[0:ACTION_BLOCK_SIZE],
            arr[ACTION_BLOCK_SIZE : 2 * ACTION_BLOCK_SIZE],
            arr[2 * ACTION_BLOCK_SIZE : 3 * ACTION_BLOCK_SIZE],
        ],
        axis=1,
    )


def assert_active_block_metadata_compatible(metadata: Mapping[str, Any] | None) -> None:
    """Raise if checkpoint metadata declares an incompatible action layout."""

    if metadata is None:
        return
    expected = active_block_metadata()
    for key in ("command_representation", "active_block_order", "action_block_size", "cmd_dim"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"incompatible checkpoint action metadata for {key}: "
                f"expected {expected[key]!r}, got {metadata.get(key)!r}"
            )
