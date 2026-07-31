"""Exact H6 row schedule for the V18 causal delay-line memory probe.

This module is intentionally metadata-only.  It reuses the hash-bound corrected
H6 V2 indexes and defines the one-pass 16,000-row training schedule without
opening RGB leaves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6


MEMORY_MICROBATCH_SIZE_V1 = 2
MEMORY_MICROBATCHES_PER_UPDATE_V1 = 8
MEMORY_PRESENTATIONS_PER_UPDATE_V1 = 16
MAXIMUM_UPDATES_V1 = 1_000
MAXIMUM_MEMORY_PRESENTATIONS_V1 = 16_000


class DelayLineH6ContractError(RuntimeError):
    """The frozen H6 role or one-pass schedule changed."""


@dataclass(frozen=True, slots=True)
class DelayLineH6ExampleV1:
    """The causal model-visible portion of one registered seven-frame row."""

    history_rgb: tuple[str, str, str]
    history_actions: tuple[int, int]
    future_rgb: tuple[str, str, str, str]
    future_actions: tuple[int, int, int, int]


def split_registered_row_v1(row: h6.H6V2Row) -> DelayLineH6ExampleV1:
    """Split e0:e6 and p0:p5 without changing their registered order."""

    if not isinstance(row, h6.H6V2Row):
        raise TypeError("delay-line examples require a registered H6V2Row")
    if len(row.rgb) != 7 or len(row.actions) != 6:
        raise DelayLineH6ContractError("corrected H6 row shape changed")
    return DelayLineH6ExampleV1(
        history_rgb=(row.rgb[0], row.rgb[1], row.rgb[2]),
        history_actions=(row.actions[0], row.actions[1]),
        future_rgb=(row.rgb[3], row.rgb[4], row.rgb[5], row.rgb[6]),
        future_actions=(
            row.actions[2],
            row.actions[3],
            row.actions[4],
            row.actions[5],
        ),
    )


def train_rows_for_update_v1(
    train_rows: Sequence[h6.H6V2Row], update: int
) -> tuple[tuple[h6.H6V2Row, h6.H6V2Row], ...]:
    """Return eight consecutive B2 microbatches, consuming all rows once."""

    if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATES_V1:
        raise DelayLineH6ContractError("update must be in the range 1 through 1000")
    if len(train_rows) != h6.TRAIN_INDEX_ROWS:
        raise DelayLineH6ContractError("complete frozen H6 train index is required")
    start = (update - 1) * MEMORY_PRESENTATIONS_PER_UPDATE_V1
    stop = start + MEMORY_PRESENTATIONS_PER_UPDATE_V1
    selected = tuple(train_rows[start:stop])
    if len(selected) != MEMORY_PRESENTATIONS_PER_UPDATE_V1 or any(
        row.role != "train" or row.index != start + offset
        for offset, row in enumerate(selected)
    ):
        raise DelayLineH6ContractError("H6 rows left exact consecutive order")
    result = tuple(
        (selected[offset], selected[offset + 1])
        for offset in range(0, len(selected), MEMORY_MICROBATCH_SIZE_V1)
    )
    if len(result) != MEMORY_MICROBATCHES_PER_UPDATE_V1:
        raise DelayLineH6ContractError("delay-line update lost a microbatch")
    return result


def load_exact_roles_v1(
    runtime_data_root: Path,
) -> tuple[
    tuple[h6.H6V2Row, ...],
    tuple[h6.H6V2Row, ...],
    dict[str, Any],
]:
    """Load and revalidate both exact indexes, without opening an RGB leaf."""

    root = Path(runtime_data_root)
    if not root.is_absolute():
        raise DelayLineH6ContractError("runtime data root must be absolute")
    train_rows, train_audit = h6.load_bound_index(root, role="train")
    validation_rows, validation_audit = h6.load_bound_index(root, role="val")
    train_scenes = {row.scene_id for row in train_rows}
    validation_scenes = {row.scene_id for row in validation_rows}
    train_rgb = {leaf for row in train_rows for leaf in row.rgb}
    validation_rgb = {leaf for row in validation_rows for leaf in row.rgb}
    if train_scenes & validation_scenes or train_rgb & validation_rgb:
        raise DelayLineH6ContractError("H6 train and validation roles overlap")
    return train_rows, validation_rows, {
        "schema": "lewm_go2_v18_delay_line_h6_runtime_v1_preflight",
        "status": "PASS_METADATA_ONLY_PREFLIGHT",
        "train": train_audit,
        "validation": validation_audit,
        "train_validation_scene_overlap_count": 0,
        "train_validation_rgb_path_overlap_count": 0,
        "maximum_updates": MAXIMUM_UPDATES_V1,
        "memory_presentations_per_update": MEMORY_PRESENTATIONS_PER_UPDATE_V1,
        "maximum_memory_presentations": MAXIMUM_MEMORY_PRESENTATIONS_V1,
        "rgb_open_count": 0,
        "gpu_use_count": 0,
        "generated_write_count": 0,
    }


__all__ = [
    "DelayLineH6ContractError",
    "DelayLineH6ExampleV1",
    "MAXIMUM_MEMORY_PRESENTATIONS_V1",
    "MAXIMUM_UPDATES_V1",
    "MEMORY_MICROBATCHES_PER_UPDATE_V1",
    "MEMORY_MICROBATCH_SIZE_V1",
    "MEMORY_PRESENTATIONS_PER_UPDATE_V1",
    "load_exact_roles_v1",
    "split_registered_row_v1",
    "train_rows_for_update_v1",
]
