"""Recording-mission stall stop, v2 (development; Andrew, 5 October 2026, stage-2 recordings).

Andrew: "end a recording mission once it has been stalled (no translation) for 60 s".

v1 (lewm/dev_recording_stall_stop_development.py, unchanged) compared only the positions at the two ends of the
trailing window, so a robot that looped back to where it was 60 s earlier counted as stalled. In the first recordings
(`s2rec`), 4 of 5 were stopped that way while driving 5-15 m with 47-72% forward commands.

v2 counts a stall only when the base stays within MIN_TRANSLATION_M (0.10 m) of its position at the start of the
trailing WINDOW_S (60 s) for the whole window, sampled every 100 ms. An in-place trap (turning or holding) still stops;
a loop or oscillation that leaves the 10-cm neighbourhood does not.
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
        excursion = max(float(np.linalg.norm(xy-start_xy)) for _, xy in self.track)
        if excursion < self.min_translation_m:
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
                rule='v2: the base stayed within the threshold of its window-start position for the whole trailing window')))
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
