"""Replay scheduling helpers for deterministic Genesis rendering."""

from __future__ import annotations

from typing import Any


def replay_frame_schedule(
    command_timestamps_s: list[float],
    camera_hz: float,
    max_time_s: float | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic render-frame times aligned to command timestamps."""

    if camera_hz <= 0.0:
        raise ValueError("camera_hz must be positive")
    if not command_timestamps_s:
        return []

    start_s = min(command_timestamps_s)
    end_s = max(command_timestamps_s) if max_time_s is None else min(max(command_timestamps_s), max_time_s)
    period_s = 1.0 / camera_hz
    frames: list[dict[str, Any]] = []
    frame_index = 0
    t = start_s
    while t <= end_s + 1e-9:
        frames.append({"frame_index": frame_index, "timestamp_s": round(t, 9)})
        frame_index += 1
        t = start_s + frame_index * period_s
    return frames
