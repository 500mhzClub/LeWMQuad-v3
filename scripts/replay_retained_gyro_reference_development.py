"""Compare reference retention on a fixed public recording; score afterward."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scripts.in_memory_public_replay_development import PublicReplay
from lewm.retained_gyro_reference_development import RetainedGyroReferenceMotion
from lewm.partial_floor_height_development import PartialHeightRegistration, read_pose
from lewm.physical_execution_development import rotation_xyzw
from lewm.two_cm_floor_extent_development import configure

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--output-name', required=True)
    parser.add_argument('--activation-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=4805)
    parser.add_argument('--stable-anchor', action='store_true')
    parser.add_argument('--jit-floor', action='store_true')
    parser.add_argument('--workspace-volume', action='store_true')
    args = parser.parse_args()
    for value in (args.root_name, args.output_name):
        if Path(value).name != value or value.startswith('sealed'):
            raise ValueError('ordinary development basename required')
    if args.activation_frame < 0 or args.max_frames < 1:
        raise ValueError('nonnegative onset and positive frame budget required')
    base = Path('/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/navigation_development_artifacts_v1') if args.workspace_volume else BASE
    root = base / args.root_name
    output = root / args.output_name
    output.mkdir()
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    reader = PublicReplay(root / 'native')
    motion = RetainedGyroReferenceMotion(activation_frame=args.activation_frame)
    if args.stable_anchor:
        from lewm.stable_gyro_reference_development import StableGyroReferenceMotion
        motion = StableGyroReferenceMotion(activation_frame=args.activation_frame)
    if args.jit_floor:
        if not args.stable_anchor:
            raise ValueError('this compiled-floor replay branch extends stable-reference selection')
        from lewm.stable_gyro_reference_development import CompiledFloorStableGyroReferenceMotion
        motion = CompiledFloorStableGyroReferenceMotion(activation_frame=args.activation_frame)
    registration = PartialHeightRegistration()
    original = json.loads((root / 'poses.json').read_text())
    rows = []; failure = None; prefix_equal = True; start = time.monotonic()
    with (output / 'frames.jsonl').open('x') as sink:
        for frame in range(args.max_frames):
            try:
                p, d, fast, rgb, auxiliary, now = reader.packet(frame)
                begin = time.monotonic()
                raw = motion.observe(p, d, fast, auxiliary_rgb=rgb,
                    auxiliary_depth=auxiliary, now_ns=now)
                if raw['current_pose'] is None:
                    raise ValueError(json.dumps(raw['terminal_failure']))
                evidence = registration.observe(p, d, auxiliary, raw, now_ns=now)
                read_pose(evidence, identity=(0, 0, 0), now_ns=now)
                pose = raw['current_pose']
                if frame < args.activation_frame:
                    prefix_equal &= pose == original[frame]['raw_pose']
                row = dict(frame=frame, tracking_and_registration_s=time.monotonic()-begin,
                    raw_position_m=pose['position_initial_body_m'],
                    registered_position_m=evidence['current_pose']['position_initial_body_m'],
                    reference_frame=pose['reference_frame'],
                    promotion_reason=pose['promotion_reason'],
                    selected_camera=raw['camera_selection'].get('selected_camera'))
                rows.append(row); sink.write(json.dumps(row)+'\n'); sink.flush()
                if frame % 100 == 0:
                    print('RETAINED_REFERENCE_FRAME', frame, 'reference', row['reference_frame'],
                        'elapsed_s', round(time.monotonic()-start, 2), flush=True)
            except Exception as error:
                failure = dict(frame=frame, reason=str(error))
                break
    # Physical truth is loaded only after estimation and never supplied to it.
    metadata = json.loads((root/'native/in_memory_camera_observations.json').read_text())
    frames = metadata['frames']
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as z:
        physics = z['base_pose_world']
    origin = physics[frames[0]['physical_sample_index']]; R0 = rotation_xyzw(origin[3:])
    errors = dict(treatment=[], original=[])
    for row in rows:
        i = row['frame']
        actual = (physics[frames[i]['physical_sample_index'], :3]-origin[:3])@R0
        for label, position in [('treatment', row['registered_position_m']),
                ('original', original[i]['registered_pose']['position_initial_body_m'])]:
            errors[label].append(float(np.linalg.norm(actual-position)))
    summary = {label: dict(median_m=float(np.median(values)), maximum_m=max(values),
        final_m=values[-1], post_activation_median_m=float(np.median(values[args.activation_frame:])))
        for label, values in errors.items() if values and len(values)>args.activation_frame}
    sources = [__file__, 'lewm/retained_gyro_reference_development.py',
        'lewm/dual_camera_anchor_pose_development.py']
    if args.stable_anchor:
        sources.append('lewm/stable_gyro_reference_development.py')
    if args.jit_floor:
        sources.extend(['lewm/jit_floor_candidates_development.py', 'lewm/jit_floor_gyro_visual_motion_development.py'])
    report = dict(root_name=args.root_name, activation_frame=args.activation_frame,
        stable_anchor_with_recent_reference_refresh=args.stable_anchor,
        compiled_floor_candidate_predicates=args.jit_floor,
        frames=len(rows), failure=failure, unchanged_prefix_raw_poses_equal=bool(prefix_equal),
        summary=summary, promotions=dict(Counter(r['promotion_reason'] for r in rows)),
        elapsed_s=time.monotonic()-start, native_state_loaded_only_after_estimation=True,
        native_navigation_executed=False,
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (output/'result.json').write_text(json.dumps(report, indent=2))
    print('RETAINED_REFERENCE_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
