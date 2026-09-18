"""Synthetic marker child owned and reaped by its diagnostic runner."""
import ctypes
import json
import os
import sys
import threading
import time

from scripts.external_body_projection_profile_replay_development import MARKER_WINDOWS, observe_window


def synthetic_observation(frame):
    # Keep an actual Python descendant below the marker during native waiting.
    time.sleep(.1)
    return frame


if __name__ == '__main__':
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    # Linux PR_SET_PTRACER permits this parent and its descendants only.
    if libc.prctl(0x59616d61, parent, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'parent-scoped profiler permission failed')
    print(json.dumps(dict(status='READY', pid=os.getpid(), parent=parent,
        native_thread_id=threading.get_native_id())), flush=True)
    if sys.stdin.readline() != 'GO\n' or os.getppid() != parent:
        raise RuntimeError('live original runner start handshake required')
    completed = [observe_window(frame, window, synthetic_observation, frame)
        for frame, window in MARKER_WINDOWS.items()]
    print(json.dumps(dict(status='EXTERNAL_OBSERVATION_MARKER_SMOKE_COMPLETE',
        pid=os.getpid(), native_thread_id=threading.get_native_id(), frames=completed,
        controller_executed=False, raw_sensor_or_checkpoint_access=False)), flush=True)
