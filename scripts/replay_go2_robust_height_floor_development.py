"""Test the paired height selector on complete recorded near-wall failures."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from lewm.robust_height_floor_tracking_development import (
    RobustHeightFloorMotion, RobustHeightFloorRegistration, RobustHeightIndependentObstacles)
from lewm.gyro_seeded_height_floor_tracking_development import GyroSeededHeightFloorMotion
from lewm.two_cm_floor_extent_development import configure
from lewm.partial_floor_height_development import read_pose
from lewm.physical_execution_development import rotation_xyzw
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE
from scripts.diagnose_go2_depth_noise_failures_development import save, serial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=(0,3),required=True)
    parser.add_argument('--gyro-seeded',action='store_true')
    args=parser.parse_args();i=args.layout_index
    root=BASE/f'go2_live_local_floor_mapping_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    metadata=json.loads((root/'native/in_memory_camera_observations.json').read_text())
    count=len(metadata['frames'])
    output=root/('gyro_seeded_height_floor_replay_v1' if args.gyro_seeded else 'robust_height_floor_replay_v1')
    output.mkdir()
    save(output,'launch.json',dict(layout_index=i,frames=count,depth_sigma_mm=2,
        original_delivered_noisy_packets=True,native_physics_used_in_estimation=False,
        candidate_source='paired raw depth dominant height cluster',
        tracker_candidate_seed_uses_current_gyro=args.gyro_seeded,
        candidate_pool_minimum_fraction=.25,maximum_selection_refinements=8,
        downstream_plane_and_image_acceptance_thresholds_changed=False,
        floor_selection_changed_in_tracker_registration_and_independent_observer=True,
        navigation_executed=False,source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__,'lewm/robust_height_floor_candidates_development.py',
                'lewm/robust_height_floor_tracking_development.py',
                'lewm/gyro_seeded_height_floor_tracking_development.py')}))
    configure();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=NoisyPublicReplay(root/'native')
    motion=GyroSeededHeightFloorMotion() if args.gyro_seeded else RobustHeightFloorMotion()
    registration=RobustHeightFloorRegistration();observer=RobustHeightIndependentObstacles()
    rows=[];failure=None;raw=None;started=time.monotonic()
    with (output/'frames.jsonl').open('x') as sink:
        for frame in range(count):
            p,d,fast,rgb,auxiliary,now=reader.packet(frame);stage='tracking';t=time.perf_counter()
            try:
                raw=motion.observe(p,d,fast,auxiliary_rgb=rgb,auxiliary_depth=auxiliary,now_ns=now)
                if raw['current_pose'] is None:raise ValueError('tracking failed')
                tracked=time.perf_counter();stage='registration'
                evidence=registration.observe(p,d,auxiliary,raw,now_ns=now)
                read_pose(evidence,identity=(0,0,0),now_ns=now)
                registered=time.perf_counter();stage='independent_obstacles'
                obstacles=observer.observe(p,d,fast,auxiliary_depth=auxiliary,measured_ns=now)
                observed=time.perf_counter()
            except Exception as error:
                failure=dict(frame=frame,stage=stage,reason=repr(error))
                save(output,'terminal_raw_snapshot.json',raw)
                save(output,'terminal_registration_state.json',dict(anchor=registration.anchor,
                    reference=registration.reference,frame=registration.frame,failed=registration.failed,
                    pending_plane=None if motion.model._pending_plane is None else motion.model._pending_plane[2]))
                break
            floor=observer.receipts[-1]
            row=dict(frame=frame,registered_position_m=evidence['current_pose']['position_initial_body_m'],
                tracker_floor=raw['measured_plane_evidence']['joint_plane'],
                independent_floor=floor['joint_plane'],independent_obstacles_available=obstacles is not None,
                independent_candidate_selection=floor['floor_candidate_selection'],
                tracker_candidate_selection=motion.model.last_measured_plane.get('floor_candidate_selection')
                    if isinstance(motion.model.last_measured_plane,dict) else None,
                registration_candidate_selection=evidence['floor_candidate_selection'],
                stage_wall_ms=dict(tracking=(tracked-t)*1000,registration=(registered-tracked)*1000,
                    independent_obstacles=(observed-registered)*1000))
            rows.append(row);sink.write(json.dumps(row,default=serial)+'\n');sink.flush()
            if frame%250==0:print('ROBUST_HEIGHT_FRAME',i,frame,flush=True)
    # Physics is loaded only after all sensor estimation has stopped.
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:physics=data['base_pose_world']
    origin=physics[metadata['frames'][0]['physical_sample_index']];R0=rotation_xyzw(origin[3:])
    errors=[float(np.linalg.norm((physics[metadata['frames'][r['frame']]['physical_sample_index'],:3]-origin[:3])@R0
        -r['registered_position_m'])) for r in rows]
    result=dict(layout_index=i,accepted_frames=len(rows),recorded_frames=count,failure=failure,
        full_recording_completed=len(rows)==count,independent_floor_available=sum(r['independent_floor']['available'] for r in rows),
        independent_obstacles_available=sum(r['independent_obstacles_available'] for r in rows),
        pose_error_m=None if not errors else dict(median=float(np.median(errors)),maximum=max(errors),final=errors[-1]),
        elapsed_seconds=time.monotonic()-started,closed_loop_navigation_test=False)
    save(output,'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
