"""Measure accepted correspondence support on both completed transfer missions.

Reproduce the unchanged tracker; use no physics and execute no new commands.
The selected pose witness is already available online at camera cadence.
"""
import argparse
import json
import time

import numpy as np

from lewm import process_mapped_runtime_development as process
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import run_go2_route_turn_memory_transfer_development as run
from scripts.replay_go2_cached_connectivity_tracking_failure_development import collection


def main(number):
    root = run.BASE / run.root_name(number)
    output = root / 'registration_support_replay_v1'
    output.mkdir(exist_ok=False)
    read = lambda name: json.loads((root / name).read_text())
    recorded = {r['frame']: r['raw_pose'] for r in read('poses.json')}
    count = len(read('native/in_memory_camera_observations.json')['frames'])
    collection.previous.baseline.initialize_pose(str(output))
    motion = process._motion
    reader = NoisyPublicReplay(root / 'native')
    rows = []
    started = time.monotonic()
    failure = None
    for frame in range(count):
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        raw = motion.observe(policy, depth, fast, auxiliary_rgb=rgb,
            auxiliary_depth=auxiliary, now_ns=now)
        pose = raw.get('current_pose')
        if pose is None:
            failure = dict(frame=frame, terminal_failure=raw.get('terminal_failure'))
            break
        expected = recorded[frame]
        for key in ('position_initial_body_m', 'rotation_initial_body_from_current_body'):
            np.testing.assert_array_equal(pose[key], expected[key])
        assert pose['reference_frame'] == expected['reference_frame']
        assert pose['mode'] == expected['mode']
        continuity = raw['continuity_evidence']
        selected = (continuity.get('selected_anchor_rotation_witness') or
            continuity.get('incremental_rotation_witness'))
        if frame:
            assert selected is not None and selected['current_frame'] == frame
            assert selected['reference_frame'] == pose['reference_frame']
            np.testing.assert_array_equal(selected['position_initial_body_m'],
                pose['position_initial_body_m'])
        counts = [raw[k]['selected_features'] for k in
            ('last_accepted_feature_witness', 'auxiliary_feature_witness')]
        keys = ('inliers', 'inlier_fraction', 'reference_grid_cells',
            'current_grid_cells', 'residual_rms_m', 'camera', 'camera_inliers')
        rows.append(dict(frame=frame, measured_ns=now, selected_features=counts,
            selected_fit=None if selected is None else {k: selected[k] for k in keys if k in selected},
            selected_camera=raw['camera_selection'].get('selected_camera'),
            selected_reference=pose['reference_frame'],
            incremental_available=continuity.get('incremental_available'),
            anchor_available=continuity.get('anchor_available')))
        if frame % 200 == 0:
            print('SUPPORT_REPLAY', number, frame, round(time.monotonic()-started, 1), flush=True)
    assert len(rows) == len(recorded)
    result = dict(schema='transfer_registration_support_replay.v1', assignment=number,
        rows=rows, failure=failure, all_recorded_raw_poses_matched=True,
        accepted_frames=len(rows), acquired_camera_frames=count,
        public_sensor_replay_only=True, native_state_used=False,
        delivered_noisy_depth_digests_verified=True, tracker_unchanged=True,
        alternative_navigation_outcome_proven=False, wall_seconds=time.monotonic()-started)
    with (output / 'result.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment', type=int, choices=(1, 2), required=True)
    main(parser.parse_args().assignment)
