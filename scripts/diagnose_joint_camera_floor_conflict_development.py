"""Replay public measurements to characterize a floor-transport rejection."""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from lewm.joint_camera_anchor_tracker_development import JointCameraAnchorVisualMotion
from lewm.partial_floor_height_development import PartialHeightRegistration, height_correction
from lewm.extended_return_budget_transport_development import composition
from lewm.sampled_plane_candidates_development import measured_candidates
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from scripts.in_memory_public_replay_development import PublicReplay


def main():
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_joint_camera_extended_round_trip_native_layout00_v1_attempt_001')
    output=root/'floor_conflict_replay.json'
    if output.exists():raise ValueError('preserve diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native');tracker=JointCameraAnchorVisualMotion();registration=PartialHeightRegistration()
    count=json.loads((root/'failure.json').read_text())['acquired_frames']
    expected=json.loads((root/'poses.json').read_text())
    first_difference=None;report=None
    for frame in range(count):
        p,d,f,rgb,aux,now=reader.packet(frame)
        raw=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=aux)
        if raw['current_pose'] is None:raise ValueError(f'visual replay failed at {frame}')
        if frame<len(expected) and raw['current_pose']!=expected[frame]['raw_pose'] and first_difference is None:
            first_difference=frame
        try:
            registration.observe(p,d,aux,raw,now_ns=now)
        except ValueError as error:
            R=np.asarray(raw['current_pose']['rotation_initial_body_from_current_body'])
            up=R.T@np.asarray(registration.reference['initial_up_body'])
            clouds=[measured_candidates(depth['depth_m'],depth['valid'],E,up)[0]
                for depth,E in ((d,np.asarray(BODY_FROM_OPTICAL)),(aux,body_from_optical()))]
            plane=fit_joint_plane(*clouds,up)
            correction=composition(registration.anchor,raw,plane,identity=(0,0,0),now_ns=now)
            if plane['reason']=='insufficient_combined_two_axis_extent' and plane['candidate_count']>=100:
                correction=height_correction(registration.anchor,raw,plane,identity=(0,0,0),now_ns=now)
            normal=np.asarray(correction['transported_reference_normal_body'])
            offset=correction['transported_reference_offset_body_m']
            rows=[]
            for camera,cloud in zip(('primary','auxiliary'),clouds):
                residual=cloud@normal+offset
                rows.append(dict(camera=camera,count=len(cloud),
                    signed_min_m=float(residual.min()) if len(cloud) else None,
                    signed_max_m=float(residual.max()) if len(cloud) else None,
                    mean_m=float(residual.mean()) if len(cloud) else None,
                    over_3mm=int((np.abs(residual)>.003).sum())))
            points=np.concatenate(clouds);residual=points@normal+offset
            report=dict(frame=frame,reason=str(error),first_raw_pose_difference=first_difference,
                raw=raw,anchor=registration.anchor,plane=plane,correction=correction,camera_residuals=rows,
                candidate_points_body_m=points.tolist(),
                hypothetical_scalar_height_update_m=-float(residual.mean()) if len(points) else None,
                hypothetical_centered_maximum_residual_m=float(np.abs(residual-residual.mean()).max()) if len(points) else None,
                native_state_used=False,pose_admitted=False)
            break
        if frame%100==0:print('FLOOR_REPLAY_FRAME',frame,flush=True)
    if report is None:raise ValueError('recorded failure not reproduced')
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('FLOOR_CONFLICT',json.dumps({k:v for k,v in report.items() if k not in
        ('raw','anchor','plane','candidate_points_body_m')}),flush=True)
    print('FLOOR_REASON',report['plane']['reason'],'COUNT',report['plane']['candidate_count'],flush=True)


if __name__=='__main__':main()
