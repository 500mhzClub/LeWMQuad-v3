"""Test gyro-conditioned RGB-D estimation before native navigation integration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scripts.in_memory_public_replay_development import PublicReplay
from lewm.gyro_conditioned_pair_pose_development import GyroConditionedPairPose
from lewm.physical_execution_development import rotation_xyzw
from lewm.two_cm_floor_extent_development import configure

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--output-name', default='gyro_conditioned_pair_pose_replay_v1.json')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--gyro-consensus', action='store_true')
    args = parser.parse_args()
    for value in (args.root_name, args.output_name):
        if Path(value).name != value or value.startswith('sealed'):
            raise ValueError('ordinary development basename required')
    root = BASE/args.root_name
    output = root/args.output_name
    if output.exists(): raise ValueError('preserve prior replay result')
    metadata = json.loads((root/'native/in_memory_camera_observations.json').read_text())
    frames = sorted(metadata['frames'], key=lambda row: row['frame'])
    if args.max_frames is not None:
        if args.max_frames < 1: raise ValueError('positive frame budget required')
        frames = frames[:args.max_frames]
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    reader = PublicReplay(root/'native'); model = GyroConditionedPairPose()
    sources = ['lewm/gyro_conditioned_pair_pose_development.py', __file__]
    if args.gyro_consensus:
        from lewm.gyro_consensus_pair_pose_development import GyroConsensusPairPose
        model = GyroConsensusPairPose()
        sources.append('lewm/gyro_consensus_pair_pose_development.py')
    rows = []; failure = None; start = time.monotonic()
    for item in frames:
        frame = item['frame']
        try:
            p, d, fast, rgb, auxiliary, now = reader.packet(frame)
            state = model.observe(p, d, fast, auxiliary_rgb=rgb,
                auxiliary_depth=auxiliary, now_ns=now)
            rows.append({k: state[k] for k in ('frame', 'position_initial_body_m',
                'rotation_initial_body_from_current_body', 'gyro_role', 'mode', 'reference_frame')})
            if frame % 100 == 0:
                print('GYRO_REFIT_REPLAY_FRAME', frame, 'elapsed_s', round(time.monotonic()-start, 2), flush=True)
        except Exception as error:
            chain = []; cause = error
            while cause is not None:
                chain.append(str(cause)); cause = cause.__cause__
            failure = dict(frame=frame, chain=chain); break
    # Only now inspect evaluator truth and the original estimator's output.
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as z:
        physics = z['base_pose_world']
    original = json.loads((root/'poses.json').read_text())
    origin = physics[frames[0]['physical_sample_index']]; R0 = rotation_xyzw(origin[3:])
    values = dict(gyro_refit=[], original_raw=[])
    for row, item in zip(rows, frames):
        actual = (physics[item['physical_sample_index'], :3]-origin[:3])@R0
        for name, pose in (('gyro_refit', row), ('original_raw', original[row['frame']]['raw_pose'])):
            error = actual-np.asarray(pose['position_initial_body_m'])
            values[name].append([float(np.linalg.norm(error[:2])), float(np.linalg.norm(error))])
        row['evaluator_horizontal_error_m'] = values['gyro_refit'][-1][0]
    summary = {name: dict(horizontal_median_m=float(np.median(errors, axis=0)[0]),
        horizontal_maximum_m=float(np.max(errors, axis=0)[0]),
        position_median_m=float(np.median(errors, axis=0)[1]),
        position_maximum_m=float(np.max(errors, axis=0)[1]))
        for name, errors in values.items() if errors}
    report = dict(root_name=args.root_name, frames=len(rows), requested_frames=len(frames),
        failure=failure, elapsed_s=time.monotonic()-start, summary=summary, rows=rows,
        raw_pose_only=True, robust_gyro_consensus=args.gyro_consensus,
        downstream_floor_registration_evaluated=False,
        native_navigation_executed=False, native_state_loaded_only_after_estimation=True,
        ideal_gyro_bias_and_noise_not_validated=True,
        source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    with output.open('x') as f: json.dump(report, f, indent=2)
    print('GYRO_REFIT_REPLAY_RESULT', json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)


if __name__ == '__main__': main()
