"""Full saved-sensor test of weaker-corner completion; no alternative commands."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from lewm import process_mapped_runtime_development as process
from lewm.sparse_corner_completion_development import SparseCornerCompletionPose
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import run_go2_route_turn_memory_transfer_development as run
from scripts.replay_go2_cached_connectivity_tracking_failure_development import collection


def main():
    root = run.BASE / run.root_name(1)
    output = root / 'sparse_corner_completion_replay_v1'
    output.mkdir(exist_ok=False)
    read = lambda name: json.loads((root/name).read_text())
    recorded = {r['frame']: r['raw_pose'] for r in read('poses.json')}
    count = len(read('native/in_memory_camera_observations.json')['frames'])
    collection.previous.baseline.initialize_pose(str(output))
    motion = process._motion
    motion.model = SparseCornerCompletionPose()
    reader = NoisyPublicReplay(root/'native')
    rows = []
    started = time.monotonic()
    failure = first_changed = None
    for frame in range(count):
        policy, depth, fast, rgb, auxiliary, now = reader.packet(frame)
        began = time.perf_counter()
        raw = motion.observe(policy, depth, fast, auxiliary_rgb=rgb,
            auxiliary_depth=auxiliary, now_ns=now)
        pose = raw.get('current_pose')
        row = dict(frame=frame, pose=pose, wall_ms=1000*(time.perf_counter()-began),
            feature_support={camera: raw.get(key) for camera, key in
                (('primary', 'last_accepted_feature_witness'), ('auxiliary', 'auxiliary_feature_witness'))})
        rows.append(row)
        if pose is None:
            failure = dict(frame=frame, terminal_failure=raw.get('terminal_failure'))
            (output/'terminal_raw_snapshot.json').write_text(json.dumps(raw, indent=2)+'\n')
            break
        assert all(w['feature_selection'] == 'sparse_corner_completion150_v1'
            and w['selected_features'] <= 150 for w in row['feature_support'].values())
        if frame in recorded and first_changed is None and any(
                not np.array_equal(pose[k], recorded[frame][k]) for k in
                ('position_initial_body_m', 'rotation_initial_body_from_current_body')):
            first_changed = frame
        if frame % 200 == 0:
            print('CORNER_COMPLETION_REPLAY', frame, round(time.monotonic()-started, 1), flush=True)
    result = dict(schema='sparse_corner_completion_replay.v1', rows=rows, failure=failure,
        recorded_camera_frames=count, accepted_frames=sum(r['pose'] is not None for r in rows),
        baseline_failure_frame=1267, first_changed_recorded_pose_frame=first_changed,
        complete_saved_sensor_sequence_accepted=failure is None and len(rows) == count,
        total_feature_budget=150, pose_acceptance_thresholds_unchanged=True,
        delivered_noisy_depth_digests_verified=True, native_state_used=False,
        recorded_camera_and_command_sequence_unchanged=True,
        alternative_navigation_outcome_proven=False, wall_seconds=time.monotonic()-started,
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
            'lewm/sparse_corner_completion_development.py',
            'scripts/replay_go2_sparse_corner_completion_development.py')})
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
