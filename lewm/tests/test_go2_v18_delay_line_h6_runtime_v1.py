from __future__ import annotations

import pytest

from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6
from lewm.datasets import go2_v18_delay_line_h6_runtime_v1 as runtime


def _row(index: int) -> h6.H6V2Row:
    return h6.H6V2Row(
        index=index,
        role="train",
        family="corridor",
        scene_id=f"corridor_{index:012x}",
        rgb=tuple(f"corridor/frame_{index * 7 + step:06d}_env_00.png" for step in range(7)),
        actions=(0, 1, 2, 3, 4, 5),
    )


def test_split_registered_row_keeps_causal_order() -> None:
    row = _row(0)
    split = runtime.split_registered_row_v1(row)
    assert split.history_rgb == row.rgb[:3]
    assert split.history_actions == row.actions[:2]
    assert split.future_rgb == row.rgb[3:]
    assert split.future_actions == row.actions[2:]


def test_schedule_consumes_all_16000_rows_exactly_once() -> None:
    rows = tuple(_row(index) for index in range(h6.TRAIN_INDEX_ROWS))
    seen: list[int] = []
    for update in range(1, runtime.MAXIMUM_UPDATES_V1 + 1):
        microbatches = runtime.train_rows_for_update_v1(rows, update)
        assert tuple(map(len, microbatches)) == (2,) * 8
        seen.extend(row.index for batch in microbatches for row in batch)
    assert seen == list(range(h6.TRAIN_INDEX_ROWS))


def test_schedule_rejects_partial_role_and_out_of_cap_update() -> None:
    rows = tuple(_row(index) for index in range(h6.TRAIN_INDEX_ROWS))
    with pytest.raises(runtime.DelayLineH6ContractError):
        runtime.train_rows_for_update_v1(rows[:-1], 1)
    with pytest.raises(runtime.DelayLineH6ContractError):
        runtime.train_rows_for_update_v1(rows, 1_001)
