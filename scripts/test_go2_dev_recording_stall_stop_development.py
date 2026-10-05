"""Synthetic tests for the recording stall stop (lewm/dev_recording_stall_stop_development.py).
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_recording_stall_stop_development
"""
import json
from pathlib import Path
import tempfile

import numpy as np

from lewm.dev_recording_stall_stop_development import StallMonitor, install
from scripts.novel_maze_round_trip_physical_session_development import PhysicalStop


def drive(monitor, path):
    """path: function t -> xy; returns the first stop time or None (2-ms samples, 200 s)."""
    for t in np.arange(0., 200., .002):
        if monitor.update(t, path(t)):
            return t
    return None


def test_monitor():
    assert drive(StallMonitor(), lambda t: (.2*t, 0.)) is None, 'steady translation never stops'
    t = drive(StallMonitor(), lambda t: (.2*min(t, 30.), 0.))
    assert t is not None and 89.4 <= t <= 90.2, t  # still from 30 s on (0.10 m left after 29.5 s), so about 60 s later
    t = drive(StallMonitor(), lambda t: (.3*np.sin(t), 0.))  # oscillating +/-0.3 m: net movement over 60 s
    assert t is not None, 'oscillation in place counts as stalled when the net move over the window is small'
    assert drive(StallMonitor(), lambda t: (.2*t if t < 50 else 10., 0.)) is not None
    assert drive(StallMonitor(), lambda t: (0., 0.)) is not None and drive(StallMonitor(), lambda t: (0., 0.)) >= 59.9


class FakeSession:
    def _sample(self, requested, applied, timestamp_s):
        return dict(base_pose_world=np.array([0., 0., .3, 0, 0, 0, 1]))


def test_install():
    with tempfile.TemporaryDirectory() as d:
        s = install(FakeSession(), d)
        try:
            for t in np.arange(0., 70., .002):
                s._sample(None, None, t)
            raise AssertionError('expected a stall stop')
        except PhysicalStop as stop:
            assert 'DEV_RECORDING_STALL_STOP' in repr(stop)
        record = json.loads(Path(d, 'dev_recording_stall_stop.json').read_text())
        assert 59.9 <= record['stopped_at_s'] <= 60.2 and record['min_translation_m'] == .1


if __name__ == '__main__':
    for test in (test_monitor, test_install):
        test()
        print('ok', test.__name__)
    print('2 passed')
