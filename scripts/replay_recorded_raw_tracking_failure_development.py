"""Recover raw-tracker failure evidence from exact delivered development packets.

No navigation, registration replacement, native truth input or controller tuning.
Require matching recorded raw poses before interpreting a reproduced failure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import torch

from lewm.cached_moments_deferred_copy_tracking_development import CachedMomentsDeferredCopyMotion
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    args = parser.parse_args()
    root = path(args.root_name)
    if (root/'depth_retention.json').exists():
        raise ValueError('retained depth required')
    launch = read(root, 'launch.json')
    if launch.get('tracker') != 'CachedMomentsDeferredCopyMotion':
        raise ValueError('recorded tracker must match the replay implementation')
    recorded = {r['frame']:r['raw_pose'] for r in read(root, 'poses.json')}
    count = len(read(root, 'native/in_memory_camera_observations.json')['frames'])
    output = root/'raw_tracking_failure_replay_v1'
    output.mkdir()
    config = dict(root_name=root.name, frames=count, tracker=launch['tracker'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_delivered_depth_digests_checked=True, native_state_read=False,
        raw_tracking_only=True, closed_loop_navigation_test=False)
    (output/'launch.json').write_text(json.dumps(config, indent=2))
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = NoisyPublicReplay(root/'native')
    motion = CachedMomentsDeferredCopyMotion()
    fields = ('frame', 'measured_ns', 'reference_frame', 'position_initial_body_m',
        'rotation_initial_body_from_current_body', 'rgb_sha256', 'depth_sha256',
        'auxiliary_rgb_sha256', 'auxiliary_depth_sha256')
    matched = 0; accepted = 0; failure = None; started = time.monotonic()
    with (output/'frames.jsonl').open('x') as sink:
        for frame in range(count):
            p, d, fast, rgb, auxiliary, now = reader.packet(frame)
            raw = motion.observe(p, d, fast, auxiliary_rgb=rgb,
                auxiliary_depth=auxiliary, now_ns=now)
            pose = raw.get('current_pose')
            if pose is None or raw.get('failure') is not None:
                (output/'terminal_raw_snapshot.json').write_text(json.dumps(raw, indent=2))
                failure = dict(frame=frame, terminal_failure=raw.get('terminal_failure'),
                    failure=raw.get('failure'))
                break
            if frame in recorded:
                changed = [k for k in fields if pose[k] != recorded[frame][k]]
                if changed:
                    failure = dict(frame=frame, replay_mismatch_fields=changed)
                    break
                matched += 1
            accepted += 1
            sink.write(json.dumps(dict(frame=frame, reference_frame=pose['reference_frame'],
                position_initial_body_m=pose['position_initial_body_m'],
                camera_selection=raw.get('camera_selection'),
                local_view_revisit_attempt=raw.get('local_view_revisit_attempt')))+'\n')
            sink.flush()
            if frame % 200 == 0:
                print('RAW_TRACKING_REPLAY_FRAME', frame, flush=True)
    result = dict(config=config, accepted_frames=accepted,
        matching_recorded_raw_poses=matched, recorded_raw_poses=len(recorded),
        all_recorded_raw_poses_matched=matched == len(recorded),
        failure=failure, elapsed_s=time.monotonic()-started)
    (output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
