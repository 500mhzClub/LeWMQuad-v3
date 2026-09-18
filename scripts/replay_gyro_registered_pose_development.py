"""Evaluate the gyro-consensus observer through actual floor registration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scripts.in_memory_public_replay_development import PublicReplay
from lewm.gyro_consensus_visual_motion_development import GyroConsensusVisualMotion
from lewm.partial_floor_height_development import PartialHeightRegistration, read_pose
from lewm.physical_execution_development import rotation_xyzw
from lewm.two_cm_floor_extent_development import configure

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--output-name', default='gyro_consensus_registered_replay_v1.json')
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    for value in (args.root_name, args.output_name):
        if Path(value).name != value or value.startswith('sealed'):
            raise ValueError('ordinary development basename required')
    root = BASE/args.root_name; output = root/args.output_name
    if output.exists(): raise ValueError('preserve prior replay')
    metadata = json.loads((root/'native/in_memory_camera_observations.json').read_text())
    frames = sorted(metadata['frames'], key=lambda row: row['frame'])
    if args.max_frames is not None:
        if args.max_frames < 1: raise ValueError('positive frame budget required')
        frames = frames[:args.max_frames]
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    reader = PublicReplay(root/'native'); motion = GyroConsensusVisualMotion()
    registration = PartialHeightRegistration(); rows = []; failure = None
    start = time.monotonic()
    for item in frames:
        frame = item['frame']; stage = 'camera_pose'
        try:
            p, d, fast, rgb, auxiliary, now = reader.packet(frame)
            raw = motion.observe(p, d, fast, auxiliary_rgb=rgb, auxiliary_depth=auxiliary, now_ns=now)
            if raw['current_pose'] is None: raise ValueError(json.dumps(raw['terminal_failure']))
            stage = 'floor_registration'
            evidence = registration.observe(p, d, auxiliary, raw, now_ns=now)
            read_pose(evidence, identity=(0, 0, 0), now_ns=now)
            rows.append(dict(frame=frame, raw_pose=raw['current_pose'], pose=evidence['current_pose']))
            if frame % 100 == 0:
                print('GYRO_REGISTERED_FRAME', frame, 'elapsed_s', round(time.monotonic()-start, 2), flush=True)
        except Exception as error:
            chain = []; cause = error
            while cause is not None:
                chain.append(str(cause)); cause = cause.__cause__
            failure = dict(frame=frame, stage=stage, chain=chain); break
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as z: physics = z['base_pose_world']
    original = json.loads((root/'poses.json').read_text())
    origin = physics[frames[0]['physical_sample_index']]; R0 = rotation_xyzw(origin[3:])
    errors = dict(gyro_registered=[], original_registered=[])
    for row, item in zip(rows, frames):
        actual = (physics[item['physical_sample_index'], :3]-origin[:3])@R0
        for label, pose in (('gyro_registered', row['pose']), ('original_registered', original[row['frame']]['registered_pose'])):
            delta = actual-np.asarray(pose['position_initial_body_m'])
            errors[label].append([float(np.linalg.norm(delta[:2])), float(np.linalg.norm(delta))])
        row['evaluator_horizontal_error_m'] = errors['gyro_registered'][-1][0]
    summary = {name: dict(horizontal_median_m=float(np.median(values, axis=0)[0]),
        horizontal_maximum_m=float(np.max(values, axis=0)[0]),
        position_median_m=float(np.median(values, axis=0)[1]),
        position_maximum_m=float(np.max(values, axis=0)[1])) for name, values in errors.items() if values}
    sources = [__file__, 'lewm/gyro_consensus_visual_motion_development.py',
        'lewm/gyro_consensus_pair_pose_development.py', 'lewm/gyro_conditioned_pair_pose_development.py',
        'lewm/joint_sensor_anchored_goal_development.py', 'lewm/joint_floor_registered_evidence_development.py',
        'lewm/measured_floor_transport_development.py', 'lewm/measured_floor_transport_registration_development.py',
        'lewm/dual_camera_visual_motion_development.py']
    report = dict(root_name=args.root_name, frames=len(rows), requested_frames=len(frames), failure=failure,
        summary=summary, rows=rows, elapsed_s=time.monotonic()-start,
        actual_visual_motion_and_floor_registration_used=True, native_navigation_executed=False,
        native_state_loaded_only_after_estimation=True, real_sensor_uncertainty_calibrated=False,
        source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    with output.open('x') as f: json.dump(report, f, indent=2)
    print('GYRO_REGISTERED_RESULT', json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__': main()
