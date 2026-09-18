"""Replay the saved queue-overflow mission through the unchanged tracker."""
import cProfile
import io
import json
from pathlib import Path
import pstats
import time

import numpy as np

from scripts.run_go2_pipeline_age_navigation_development import study, ROOT
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from lewm import process_mapped_runtime_development as process

OUTPUT = study.BASE/'go2_pipeline_age_tracker_replay_v1_attempt_001'


def main():
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'profile.py').write_text(Path(__file__).read_text())
    root = study.BASE/ROOT
    reader = NoisyPublicReplay(root/'native')
    recorded = {r['frame']:r['raw_pose'] for r in json.loads((root/'poses.json').read_text())}
    count = len(json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames'])
    study.previous.reference.initialize_pose()
    profiles = {name:cProfile.Profile() for name in ('early', 'late')}
    rows = []
    started = time.monotonic()
    for frame in range(count):
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        name = 'early' if 50 <= frame < 100 else 'late' if 1700 <= frame < 1750 else None
        if name: profiles[name].enable()
        begin = time.perf_counter_ns()
        result = process._motion.observe(policy, depth, fast, now_ns=now,
            auxiliary_rgb=rgb, auxiliary_depth=auxiliary)
        elapsed = time.perf_counter_ns()-begin
        if name: profiles[name].disable()
        current = result.get('current_pose')
        if current is None or result.get('failure') is not None:
            raise ValueError((frame, result.get('failure')))
        if frame in recorded:
            for key in ('position_initial_body_m', 'rotation_initial_body_from_current_body'):
                np.testing.assert_array_equal(current[key], recorded[frame][key])
            assert current['reference_frame'] == recorded[frame]['reference_frame']
            assert current['mode'] == recorded[frame]['mode']
        model = process._motion.model
        row = dict(frame=frame, wall_ms=elapsed/1e6, profiled=name,
            mode=current['mode'], reference_frame=current['reference_frame'],
            keyframes_ever_promoted=current['keyframe_count'],
            retained_references=len(model.references),
            revisit_attempt=model.last_revisit_attempt,
            recorded_pose_reproduced=frame in recorded)
        rows.append(row)
        if frame % 200 == 0:
            print('TRACKER_REPLAY', json.dumps(row | dict(elapsed_s=time.monotonic()-started)), flush=True)
    for name, profile in profiles.items():
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats('cumtime').print_stats(45)
        (OUTPUT/(name+'_profile.txt')).write_text(stream.getvalue())
        profile.dump_stats(str(OUTPUT/(name+'_profile.pstats')))
    result = dict(status='COMPLETE', frames=count, recorded_poses_reproduced=len(recorded),
        all_recorded_pose_values_exact=True, rows=rows, wall_s=time.monotonic()-started,
        public_sensor_replay_only=True, native_truth_used=False,
        native_concurrency_reproduced=False, navigation_outcome_tested=False)
    (OUTPUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('TRACKER_REPLAY_COMPLETE', count, time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
