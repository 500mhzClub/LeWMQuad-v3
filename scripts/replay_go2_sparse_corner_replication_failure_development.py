"""Reproduce the replication tracking failure from unchanged delivered sensors."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lewm import process_mapped_runtime_development as process
from lewm.sparse_corner_completion_development import SparseCornerCompletionPose
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import run_go2_sparse_corner_replication_development as run
from scripts.replay_go2_cadenced_maze00_failure_development import FailureViewProbePose


class CompletionFailureProbePose(FailureViewProbePose, SparseCornerCompletionPose):
    probe_stored_fits = True


def main(probe=False):
    root = run.BASE / run.root_name(6)
    retention = root / 'depth_retention.json'
    if retention.exists():
        assert json.loads(retention.read_text())['full_sensor_replay_available']
    output = root / ('tracking_failure_view_probe_v1' if probe else 'tracking_failure_replay_v1')
    output.mkdir(exist_ok=False)
    recorded = {r['frame']: r['raw_pose'] for r in
        json.loads((root / 'poses.json').read_text())}
    count = len(json.loads((root / 'native/in_memory_camera_observations.json').read_text())['frames'])
    run.previous.native.baseline.initialize_pose(str(output))
    motion = process._motion
    assert type(motion.model).__name__ == 'SparseCornerCompletionPose'
    if probe:
        motion.model = CompletionFailureProbePose()
    reader = NoisyPublicReplay(root / 'native')
    rows, matched = [], 0
    failure = None
    started = time.monotonic()
    for frame in range(count):
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        began = time.perf_counter()
        raw = motion.observe(policy, depth, fast, auxiliary_rgb=rgb,
            auxiliary_depth=auxiliary, now_ns=now)
        pose = raw.get('current_pose')
        rows.append(dict(frame=frame, measured_ns=now,
            wall_ms=1000 * (time.perf_counter() - began), pose=pose,
            reference_selection=motion.model.last_selection,
            revisit_attempt=motion.model.last_revisit_attempt,
            feature_support={camera: raw.get(key) for camera, key in (
                ('primary', 'last_accepted_feature_witness'),
                ('auxiliary', 'auxiliary_feature_witness'))}))
        if pose is None:
            failure = dict(frame=frame, terminal_failure=raw.get('terminal_failure'))
            if probe:
                failure['stored_view_probe'] = motion.model.failure_view_probe
            (output / 'terminal_raw_snapshot.json').write_text(json.dumps(raw, indent=2) + '\n')
            break
        if frame in recorded:
            for key in ('position_initial_body_m', 'rotation_initial_body_from_current_body'):
                np.testing.assert_array_equal(pose[key], recorded[frame][key])
            assert pose['mode'] == recorded[frame]['mode']
            assert pose['reference_frame'] == recorded[frame]['reference_frame']
            matched += 1
        if frame % 200 == 0:
            print('REPLICATION_TRACKING_REPLAY', frame, round(time.monotonic() - started, 1), flush=True)
    result = dict(schema='sparse_corner_replication_failure_replay.v1',
        root=str(root), rows=rows, failure=failure,
        recorded_camera_frames=count, recorded_raw_poses=len(recorded),
        matched_recorded_raw_poses=matched,
        all_recorded_raw_poses_matched=matched == len(recorded),
        accepted_frames=sum(r['pose'] is not None for r in rows),
        delivered_noisy_depth_digests_verified=True, native_state_used=False,
        navigation_reexecuted=False, tracker_and_acceptance_rules_unchanged=True,
        stored_view_probe_is_diagnostic_only=probe,
        concurrency=1, sequential_temporal_tracker=True,
        wall_seconds=time.monotonic() - started,
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
            __file__, 'lewm/sparse_corner_completion_development.py')})
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--failure-view-probe', action='store_true')
    main(parser.parse_args().failure_view_probe)
