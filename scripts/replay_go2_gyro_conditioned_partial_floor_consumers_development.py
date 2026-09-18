"""Sequential old/new floor consumers on one unchanged raw-tracker stream."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from lewm.batched_consensus_tracking_development import BatchedConsensusMotion
from lewm.floor_reacquisition_development import ReacquiringFloorRegistration
from lewm.auxiliary_only_turn_recovery_development import AuxiliaryTurnObstacles
from lewm.gyro_conditioned_partial_floor_consumers_development import (
    GyroConditionedReacquiringRegistration, GyroConditionedAuxiliaryObstacles)
from lewm.transport_conditioned_partial_floor_development import TransportConditionedReacquiringRegistration
from lewm.partial_floor_height_development import read_pose
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    parser.add_argument('--frames', type=int, required=True)
    parser.add_argument('--registration-normal', choices=('gyro','transport'), default='gyro')
    args = parser.parse_args(); root = path(args.root_name)
    if (root/'depth_retention.json').exists(): raise ValueError('retained depth required')
    count = len(read(root, 'native/in_memory_camera_observations.json')['frames'])
    if not 0 < args.frames <= count: raise ValueError('bounded recorded prefix required')
    output = root/f'{args.registration_normal}_conditioned_partial_floor_consumers_replay_{args.frames}_v1'
    output.mkdir()
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    sources = [__file__, 'lewm/gyro_conditioned_partial_floor_candidates_development.py',
        'lewm/gyro_conditioned_partial_floor_consumers_development.py']
    if args.registration_normal == 'transport':
        sources.append('lewm/transport_conditioned_partial_floor_development.py')
    (output/'launch.json').write_text(json.dumps(dict(root_name=root.name, frames=args.frames,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        registration_normal=args.registration_normal,
        unchanged_raw_tracker='BatchedConsensusMotion', both_observers_allow_auxiliary_only_turns=True,
        original_noise_recipe=True, native_execution=False), indent=2))
    reader = NoisyPublicReplay(root/'native'); motion = BatchedConsensusMotion()
    revised = (TransportConditionedReacquiringRegistration if args.registration_normal == 'transport'
        else GyroConditionedReacquiringRegistration)
    registrations = [ReacquiringFloorRegistration(), revised()]
    observers = [AuxiliaryTurnObstacles(), GyroConditionedAuxiliaryObstacles()]
    recorded = {r['frame']:r for r in read(root, 'poses.json')}
    rows = []; failure = None; started = time.monotonic()
    fields = ('frame','measured_ns','reference_frame','position_initial_body_m',
        'rotation_initial_body_from_current_body','rgb_sha256','depth_sha256')
    with (output/'frames.jsonl').open('x') as sink:
        for frame in range(args.frames):
            try:
                p,d,fast,rgb,a,now = reader.packet(frame)
                raw = motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=a,now_ns=now)
                if raw['current_pose'] is None: raise ValueError('unchanged raw tracking failed')
                if frame in recorded and any(raw['current_pose'][k] != recorded[frame]['raw_pose'][k] for k in fields):
                    raise ValueError('raw estimate differs from original recording')
                evidence = [r.observe(p,d,a,raw,now_ns=now) for r in registrations]
                for e in evidence:
                    if e.get('current_pose') is not None: read_pose(e, identity=(0,0,0), now_ns=now)
                old, new = [e.get('current_pose') for e in evidence]
                if (old is not None) != (frame in recorded):
                    raise ValueError('original registration availability differs')
                if old is not None and any(old[k] != recorded[frame]['registered_pose'][k] for k in fields):
                    raise ValueError('original registered pose differs')
                obstacles = [o.observe(p,d,fast,auxiliary_depth=a,measured_ns=now) for o in observers]
                row = dict(frame=frame, original_registration_available=old is not None,
                    new_registration_available=new is not None,
                    original_obstacles_available=obstacles[0] is not None,
                    new_obstacles_available=obstacles[1] is not None,
                    shared_registered_pose_exact=None if old is None or new is None else all(old[k]==new[k] for k in fields),
                    registration= [dict(status=e.get('status'), reason=e.get('reason'),
                        selection=e.get('floor_candidate_selection'),
                        position_m=None if e.get('current_pose') is None else e['current_pose']['position_initial_body_m']) for e in evidence],
                    obstacle_selections=[o.receipts[-1]['floor_candidate_selection'] for o in observers])
                rows.append(row); sink.write(json.dumps(row)+'\n'); sink.flush()
            except Exception as error:
                failure = dict(frame=frame, reason=repr(error)); break
            if frame%200 == 0: print('PARTIAL_FLOOR_CONSUMER_FRAME', frame, flush=True)
    result = dict(requested_frames=args.frames, completed_frames=len(rows), failure=failure,
        newly_available_registration_frames=[r['frame'] for r in rows if not r['original_registration_available'] and r['new_registration_available']],
        lost_registration_frames=[r['frame'] for r in rows if r['original_registration_available'] and not r['new_registration_available']],
        newly_available_obstacle_frames=[r['frame'] for r in rows if not r['original_obstacles_available'] and r['new_obstacles_available']],
        lost_obstacle_frames=[r['frame'] for r in rows if r['original_obstacles_available'] and not r['new_obstacles_available']],
        shared_registered_poses_changed=sum(r['shared_registered_pose_exact'] is False for r in rows),
        elapsed_s=time.monotonic()-started, native_pose_read=False,
        actual_navigation_recovery_tested=False, hardware_validated=False)
    (output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
