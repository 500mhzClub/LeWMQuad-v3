"""Fixed sequence test of gyro-conditioned initial camera consensus."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from scripts.replay_go2_depth_noise_tracking_development import (
    BASE, SEED, FRAME_COUNT, PublicReplay, perturbed_packet, configure,
    read_pose, read, rotation_xyzw)
from scripts.diagnose_go2_depth_noise_failures_development import save, serial
from lewm.local_inverse_depth_floor_tracking_development import LocalInverseDepthFloorRegistration
from lewm.gyro_initial_camera_consensus_development import GyroInitialCameraConsensusMotion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--sigma-mm', type=int, choices=(0, 2), required=True)
    args = parser.parse_args(); i = args.layout_index
    root = BASE/f'go2_routing_memory_persistent_native_layout{i:02d}_4800_v1_attempt_001'
    output = root/f'depth_noise_{args.sigma_mm}mm_gyro_initial_tracking_601_v1'
    output.mkdir()
    config = dict(layout_index=i, sigma_mm=args.sigma_mm, seed=SEED,
        frames=FRAME_COUNT, tracker='GyroInitialCameraConsensusMotion',
        single_camera_gyro_proposals_after_joint_rejection=False,
        initial_single_and_pooled_camera_fitting="gyro", maximum_concurrent_replays=4,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        consecutive_retained_camera_pair_enabled=True, bridge_allowance_changed=False,
        floor_depth_source="local_inverse_depth_5x5", image_feature_depth_changed=False,
        floor_registration_changed=True, absolute_thresholds_changed=False,
        pooled_camera_initial_fit_changed=True, native_physics_used_in_estimation=False,
        sensor_noise_calibrated=False, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in map(Path, (__file__, 'lewm/gyro_initial_camera_consensus_development.py',
                'lewm/consecutive_retained_camera_pair_development.py',
                'lewm/local_inverse_depth_floor_tracking_development.py',
                'lewm/local_inverse_depth_floor_development.py',
                'scripts/replay_go2_depth_noise_tracking_development.py',
                'lewm/stable_gyro_reference_development.py', 'lewm/partial_floor_height_development.py'))})
    save(output, 'launch.json', config)
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader = PublicReplay(root/'native'); motion = GyroInitialCameraConsensusMotion()
    registration = LocalInverseDepthFloorRegistration()
    rows = []; failure = None; raw = None; start = time.monotonic()
    with (output/'frames.jsonl').open('x') as sink:
        for frame in range(FRAME_COUNT):
            stage = 'packet'
            try:
                p,d,fast,rgb,auxiliary,now = perturbed_packet(reader.packet(frame),
                    layout=i, frame=frame, sigma_m=args.sigma_mm/1000)
                stage = 'tracking'
                raw = motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=auxiliary,now_ns=now)
                if raw['current_pose'] is None:
                    raise ValueError('tracking failed')
                stage = 'registration'
                evidence = registration.observe(p,d,auxiliary,raw,now_ns=now)
                read_pose(evidence,identity=(0,0,0),now_ns=now)
            except Exception as error:
                failure = dict(frame=frame, stage=stage, reason=repr(error))
                save(output, 'terminal_raw_snapshot.json', raw)
                save(output, 'terminal_registration_state.json', dict(
                    anchor=registration.anchor, reference=registration.reference,
                    pending_plane=None if motion.model._pending_plane is None else motion.model._pending_plane[2],
                    frame=registration.frame, failed=registration.failed))
                break
            row = dict(frame=frame, raw_pose=raw['current_pose'],
                registered_position_m=evidence['current_pose']['position_initial_body_m'],
                registered_rotation=evidence['current_pose']['rotation_initial_body_from_current_body'],
                tracker_plane=raw['measured_plane_evidence']['joint_plane'],
                registration_schema=evidence['schema'],
                consecutive_retained_pooled_fit_selected=raw['consecutive_retained_pooled_fit_selected'])
            rows.append(row);sink.write(json.dumps(row, default=serial)+'\n');sink.flush()
            if frame % 100 == 0: print('GYRO_INITIAL_FRAME', i, args.sigma_mm, frame, flush=True)
    # Score only after sensor estimation ends.
    frames = read(root,'native/in_memory_camera_observations.json')['frames']
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as archive:
        physics=archive['base_pose_world']
    origin=physics[frames[0]['physical_sample_index']]; R0=rotation_xyzw(origin[3:])
    errors=[float(np.linalg.norm((physics[frames[r['frame']]['physical_sample_index'],:3]-origin[:3])@R0
        -r['registered_position_m'])) for r in rows]
    result=dict(config=config, accepted_frames=len(rows), failure=failure,
        tracker_plane_available_frames=sum(r['tracker_plane']['available'] for r in rows),
        consecutive_retained_pooled_fit_frames=sum(r['consecutive_retained_pooled_fit_selected'] for r in rows),
        pose_error_m=None if not errors else dict(median=float(np.median(errors)), maximum=max(errors), final=errors[-1]),
        elapsed_seconds=time.monotonic()-start, closed_loop_navigation_test=False)
    save(output, 'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='config'}),flush=True)


if __name__ == '__main__':
    main()
