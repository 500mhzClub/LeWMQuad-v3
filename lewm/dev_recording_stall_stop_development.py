"""Recording-mission stall stop (development; Andrew, 5 October 2026, stage-2 recordings).

Andrew: "end a recording mission once it has been stalled (no translation) for 60 s".

The stall is measured on the base's simulated position (as the session's native guard measures speed). The position
is sampled every 100 ms. Once the mission has run for at least WINDOW_S, it stops if the net XY displacement between
the position WINDOW_S ago and now is below MIN_TRANSLATION_M. Net displacement (not path length) is used, so a robot
oscillating in place counts as stalled, as in smoke maze 47: an 18-m path for a 0.30-m net move.

The stop raises PhysicalStop('DEV_RECORDING_STALL_STOP'), so the run closes like a guard stop: planning, physics
trace and episode files are kept, and the decisions before the stall remain usable. It is for recording missions only
and is enabled by the v11 entry when LEWM_STALL_STOP_S is set. Evaluation missions do not use it.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np

WINDOW_S = 60.
MIN_TRANSLATION_M = .10
SAMPLE_S = .1
REASON = 'DEV_RECORDING_STALL_STOP'


class StallMonitor:
    def __init__(self, window_s=WINDOW_S, min_translation_m=MIN_TRANSLATION_M):
        self.window_s, self.min_translation_m = float(window_s), float(min_translation_m)
        self.track = deque()
        self.stopped_at = None

    def update(self, t, xy):
        """Record the position; True when the last window_s had less than min_translation_m net movement."""
        if self.track and t-self.track[-1][0] < SAMPLE_S-1e-9:
            return False
        self.track.append((float(t), np.asarray(xy, float)[:2].copy()))
        while len(self.track) > 1 and t-self.track[1][0] >= self.window_s:
            self.track.popleft()
        start_t, start_xy = self.track[0]
        if t-start_t < self.window_s-1e-9:
            return False
        if np.linalg.norm(self.track[-1][1]-start_xy) < self.min_translation_m:
            self.stopped_at = float(t)
            return True
        return False


def install(session, directory, window_s=WINDOW_S, min_translation_m=MIN_TRANSLATION_M):
    from scripts.novel_maze_round_trip_physical_session_development import PhysicalStop
    monitor = StallMonitor(window_s, min_translation_m)
    original = session._sample

    def _sample(requested, applied, timestamp_s):
        row = original(requested, applied, timestamp_s)
        if monitor.update(float(timestamp_s), row['base_pose_world'][:2]):
            Path(directory, 'dev_recording_stall_stop.json').write_text(json.dumps(dict(
                reason=REASON, stopped_at_s=monitor.stopped_at, window_s=window_s, min_translation_m=min_translation_m,
                rule='net XY displacement of the base over the trailing window below the threshold')))
            raise PhysicalStop(REASON)
        return row
    session._sample = _sample
    session.dev_stall_monitor = monitor
    return session


def stall_session(make_session, window_s=WINDOW_S, min_translation_m=MIN_TRANSLATION_M):
    def make(spec, directory, full_frames=False):
        session = make_session(spec, directory, full_frames=full_frames)
        return install(session, directory, window_s, min_translation_m)
    make.dynamics = dict(getattr(make_session, 'dynamics', {}) or {}, recording_stall_stop_s=float(window_s),
                         recording_stall_min_translation_m=float(min_translation_m))
    return make
