"""Reproduce a recorded floor conflict without navigation."""
import argparse
import json
import time
import cv2
import numpy as np
import torch
from lewm.retained_depth_cache_tracking_development import RetainedDepthCacheMotion
from lewm.coherent_reference_refresh_development import CoherentReferenceRefreshMotion
from lewm.robust_height_floor_tracking_development import RobustHeightFloorRegistration
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.partial_floor_height_development import height_correction, read_pose
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE
from scripts.diagnose_go2_depth_noise_failures_development import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', default='go2_current_plane_heading_reactive_noise_2mm_native_layout03_4800_v1_attempt_001')
    parser.add_argument('--tracker', choices=('baseline', 'coherent_refresh'), default='baseline')
    args = parser.parse_args()
    from scripts.compare_continuous_navigation_arms_development import path
    root = path(args.root_name)
    if (root/'depth_retention.json').exists():
        raise ValueError('full retained recording required')
    output = root/'recorded_floor_conflict_replay_v1'
    output.mkdir()
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    recorded = json.loads((root/'poses.json').read_text())
    count = len(json.loads((root/'native/in_memory_camera_observations.json').read_text())['frames'])
    reader = NoisyPublicReplay(root/'native')
    motion = {'baseline': RetainedDepthCacheMotion,
        'coherent_refresh': CoherentReferenceRefreshMotion}[args.tracker]()
    registration = RobustHeightFloorRegistration()
    matched = 0; started = time.monotonic()
    for frame in range(count):
        policy, primary, fast, rgb, auxiliary, now = reader.packet(frame)
        raw = motion.observe(policy, primary, fast, auxiliary_rgb=rgb,
            auxiliary_depth=auxiliary, now_ns=now)
        if frame < len(recorded):
            for key in ('frame','measured_ns','reference_frame','position_initial_body_m',
                    'rotation_initial_body_from_current_body','rgb_sha256','depth_sha256'):
                if raw['current_pose'][key] != recorded[frame]['raw_pose'][key]:
                    raise ValueError(f'raw replay mismatch at {frame}: {key}')
            matched += 1
        try:
            evidence = registration.observe(policy, primary, auxiliary, raw, now_ns=now)
            read_pose(evidence, identity=(0,0,0), now_ns=now)
        except ValueError as error:
            if str(error) != 'current measured candidate conflicts with transported floor reference':
                raise
            R = np.asarray(raw['current_pose']['rotation_initial_body_from_current_body'])
            up = R.T@np.asarray(registration.reference['initial_up_body'])
            selector = PairedHeightCandidates(primary, auxiliary)
            clouds = [selector(p['depth_m'],p['valid'],E,up)[0]
                for p,E in zip((primary,auxiliary),selector.mounts,strict=True)]
            plane = fit_joint_plane(*clouds,up)
            correction = height_correction(registration.anchor,raw,plane,
                identity=(0,0,0),now_ns=now)
            normal = np.asarray(correction['transported_reference_normal_body'])
            offset = correction['transported_reference_offset_body_m']
            rows = []
            for camera,cloud in zip(('primary','auxiliary'),clouds,strict=True):
                errors = cloud@normal+offset
                rows.append(dict(camera=camera,count=len(cloud),
                    maximum_residual_m=float(np.abs(errors).max()) if len(cloud) else None,
                    rms_residual_m=float(np.sqrt(np.mean(errors**2))) if len(cloud) else None,
                    over_3mm_count=int((np.abs(errors)>.003).sum())))
            points = np.concatenate(clouds)
            center = points.mean(0)
            values, vectors = np.linalg.eigh((points-center).T@(points-center)/len(points))
            fitted_normal = vectors[:,0]
            if fitted_normal@up < 0: fitted_normal = -fitted_normal
            result = dict(frame=frame,recorded_frames=count,matched_raw_pose_witnesses=matched,
                tracker=args.tracker,
                primary_valid_pixels=int(primary['valid'].sum()),
                auxiliary_valid_pixels=int(auxiliary['valid'].sum()),
                selected_cloud_eigenvalues_m2=values.tolist(),
                selected_cloud_fit_maximum_residual_m=float(np.abs((points-center)@fitted_normal).max()),
                fitted_vs_transported_normal_angle_rad=float(np.arccos(np.clip(fitted_normal@normal,-1,1))),
                reproduced_failure=str(error),registration_failure_latched=registration.failed,
                last_accepted_registration_frame=registration.frame,
                plane=plane,selection=selector.receipt,correction=correction,
                camera_residuals=rows,elapsed_s=time.monotonic()-started,
                native_pose_used=False,controller_or_physics_executed=False,
                thresholds_or_candidates_changed=False)
            save(output,'terminal_raw_snapshot.json',raw)
            save(output,'terminal_registration_state.json',dict(anchor=registration.anchor,
                reference=registration.reference,frame=registration.frame,failed=registration.failed))
            save(output,'result.json',result)
            print(json.dumps(result),flush=True)
            return
        if frame%100 == 0:
            print('MATCHED_REPLAY_FRAME',frame,flush=True)
    raise ValueError('recorded registration conflict did not reproduce')


if __name__ == '__main__':
    main()
