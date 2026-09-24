"""Regression: a horizon must never exceed the frames actually collected.

The first H=1-4 sequence build advanced ``max_horizon`` whenever the NEXT ACTION
existed, without requiring the corresponding FUTURE FRAME to exist.  The frame
loop breaks *before* appending a missing frame, so a row could report
``max_horizon = 4`` while holding only four frames (h = 0..3), and the evaluator
then raised ``IndexError`` reaching for ``horizon_frames[4]``.

This test reproduces exactly that shape -- action available at the next step, its
frame absent -- and asserts the invariant that fixes it.
"""
from __future__ import annotations

import pytest

MAX_H = 4


def build_horizon(frame_present, action_available, max_h: int = MAX_H):
    """The corrected collection logic from build_dev_v03_horizon_sequences_v1.

    ``frame_present(k)``    -> is the frame at step k on disk (k = 0 is always the
                              current frame and is present by construction)
    ``action_available(k)`` -> does a distinct complete command block start at k
    """
    frames, actions, horizon = [], [], 0
    for k in range(0, max_h + 1):
        if k > 0 and not frame_present(k):
            break
        frames.append(k)
        if k < max_h:
            if not action_available(k):
                break
            actions.append(k + 1)
            horizon = k + 1
    # the fix: a horizon is only usable if its frame was actually collected
    horizon = min(horizon, len(frames) - 1)
    return horizon, frames, actions


def test_action_present_but_future_frame_absent_bounds_the_horizon():
    """The exact fault: a3 exists, so horizon reached 4, but t+960 was never rendered."""
    horizon, frames, actions = build_horizon(
        frame_present=lambda k: k != 4,          # every frame but the last
        action_available=lambda k: True,         # every command block present
    )
    assert horizon == 3, "horizon must not exceed the frames collected"
    assert len(frames) == 4, "the missing frame must not be appended"
    assert horizon <= len(frames) - 1, "invariant: horizon <= len(frames) - 1"
    # the pre-fix bug: indexing frames[horizon] would have raised
    frames[horizon]  # must not raise


@pytest.mark.parametrize("missing", [1, 2, 3, 4])
def test_any_missing_future_frame_bounds_the_horizon(missing):
    horizon, frames, _ = build_horizon(
        frame_present=lambda k, m=missing: k != m,
        action_available=lambda k: True,
    )
    assert horizon == missing - 1
    assert horizon <= len(frames) - 1
    frames[horizon]


def test_full_sequence_reaches_max_horizon():
    horizon, frames, actions = build_horizon(
        frame_present=lambda k: True, action_available=lambda k: True)
    assert horizon == MAX_H
    assert len(frames) == MAX_H + 1
    assert len(actions) == MAX_H
    frames[horizon]


def test_missing_action_bounds_the_horizon_independently():
    """A missing command block also bounds the horizon, frames notwithstanding."""
    horizon, frames, actions = build_horizon(
        frame_present=lambda k: True,
        action_available=lambda k: k < 2,        # no block starts at k = 2
    )
    assert horizon == 2
    assert len(actions) == 2
    assert horizon <= len(frames) - 1
    frames[horizon]


def test_invariant_holds_over_the_frozen_manifest():
    """Every retained row in the frozen 479-row manifest satisfies the invariant."""
    import json
    from pathlib import Path

    rows_path = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/horizons/"
                     "FINAL/FINAL_horizon_rows_479.jsonl")
    if not rows_path.is_file():
        pytest.skip("frozen horizon manifest not present in this environment")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    assert rows, "manifest must not be empty"
    for row in rows:
        h = row["max_horizon"]
        assert h >= 1
        assert len(row["horizon_frames"]) == h + 1, (
            f"row {row['pair_sha256'][:12]}: max_horizon {h} but "
            f"{len(row['horizon_frames'])} frames")
        assert len(row["horizon_actions"]) == h
        row["horizon_frames"][h]  # the pre-fix IndexError site
    assert sum(1 for r in rows if r["max_horizon"] >= 4) == 479
