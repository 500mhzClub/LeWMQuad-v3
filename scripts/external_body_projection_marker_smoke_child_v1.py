"""Exercise every external-stack marker without sensors, models or a controller."""
import json
import os
import threading
import time
from scripts.external_body_projection_profile_replay_development import MARKER_WINDOWS, observe_window


if __name__ == '__main__':
    completed=[]
    for frame,window in MARKER_WINDOWS.items():
        observe_window(frame,window,time.sleep,.05)
        completed.append(frame)
    print(json.dumps(dict(status='EXTERNAL_OBSERVATION_MARKER_SMOKE_COMPLETE',
        pid=os.getpid(),native_thread_id=threading.get_native_id(),frames=completed,
        controller_executed=False,raw_sensor_or_checkpoint_access=False)))
