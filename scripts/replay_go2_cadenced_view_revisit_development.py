"""Replay the queue-overflow sensors with less frequent old-view probes."""
import json
from pathlib import Path
import time

from lewm.cadenced_view_revisit_tracking_development import CadencedViewRevisitMotion
from scripts.run_go2_pipeline_age_navigation_development import study, ROOT
from scripts.live_depth_noise_session_development import NoisyPublicReplay

OUTPUT = study.BASE/'go2_cadenced_view_revisit_replay_v1_attempt_001'


def main():
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'replay.py').write_text(Path(__file__).read_text())
    source = Path('lewm/cadenced_view_revisit_tracking_development.py')
    (OUTPUT/source.name).write_text(source.read_text())
    root = study.BASE/ROOT
    reader = NoisyPublicReplay(root/'native')
    count = len(json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames'])
    study.previous.reference.initialize_pose()
    motion = CadencedViewRevisitMotion()
    rows = []
    started = time.monotonic()
    frame = -1
    try:
        for frame in range(count):
            policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
            begin = time.perf_counter_ns()
            result = motion.observe(policy, depth, fast, now_ns=now,
                auxiliary_rgb=rgb, auxiliary_depth=auxiliary)
            elapsed = time.perf_counter_ns()-begin
            current = result.get('current_pose')
            if current is None or result.get('failure') is not None:
                raise ValueError((frame, result.get('failure')))
            rows.append(dict(frame=frame, wall_ms=elapsed/1e6, pose=current,
                retained_references=len(motion.model.references),
                revisit_attempt=motion.model.last_revisit_attempt))
            if frame % 200 == 0:
                print('CADENCED_REPLAY', frame, elapsed/1e6, time.monotonic()-started, flush=True)
        status = dict(status='COMPLETE', frames=count)
    except Exception as error:
        status = dict(status='FAILED', frame=frame, error=repr(error))
        raise
    finally:
        result = status | dict(rows=rows, wall_s=time.monotonic()-started,
            public_sensor_replay_only=True, native_truth_used=False,
            native_concurrency_reproduced=False, navigation_outcome_tested=False,
            every_camera_frame_tracked=True, old_view_revisit_period_frames=4)
        (OUTPUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CADENCED_REPLAY_COMPLETE', count, time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
