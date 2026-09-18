"""Offline feasibility only: current height with a gyro normal on a narrow floor strip."""
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from lewm.eligible_floor_registration_development import bind
from lewm.independent_depth_obstacle_development import IndependentDepthObstacles
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from scripts.in_memory_public_replay_development import PublicReplay


def fit_height(primary,auxiliary,up):
    plane=fit_joint_plane(primary,auxiliary,up)
    if plane['available'] or plane['reason']!='insufficient_combined_two_axis_extent':return plane
    points=np.concatenate((primary,auxiliary))
    if len(points)<100:return plane
    normal=np.asarray(up);offset=-float(points.mean(0)@normal)
    residuals=points@normal+offset;maximum=float(np.abs(residuals).max())
    if maximum>.003:return plane|dict(partial_height_maximum_residual_m=maximum)
    # This diagnostic adapter enables the existing point-height classifier;
    # it does not claim that a new plane normal has been measured.
    return plane|dict(available=True,reason='CURRENT_HEIGHT_WITH_GYRO_NORMAL_DIAGNOSTIC',
        normal_body=normal.tolist(),offset_body_m=offset,normal_measured_from_current_points=False,
        partial_height_maximum_residual_m=maximum,full_plane_qualification=False)


class PartialObserver(IndependentDepthObstacles):
    _observe=bind(IndependentDepthObstacles._observe,fit_joint_plane=fit_height)


def main():
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_fine_obstacle_round_trip_native_layout00_v1_attempt_001')
    output=root/'partial_height_obstacle_feasibility.json'
    if output.exists():raise ValueError('preserve diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native');observers={'original':IndependentDepthObstacles(),'partial_height':PartialObserver()}
    rows=[]
    for frame in range(193):
        p,d,f,_,a,now=reader.packet(frame);row=dict(frame=frame,arms={})
        for name,observer in observers.items():
            current=observer.observe(p,d,f,auxiliary_depth=a,measured_ns=now)
            plane=observer.receipts[-1]['joint_plane']
            row['arms'][name]=dict(obstacle_observation_available=current is not None,reason=plane['reason'],
                candidates=plane['candidate_count'],partial_height_maximum_residual_m=plane.get('partial_height_maximum_residual_m'))
        rows.append(row)
        if frame%50==0:print('PARTIAL_HEIGHT_FRAME',frame,flush=True)
    report=dict(public_sensor_only=True,native_pose_read=False,native_execution=False,rows=rows,
        available_frames={name:sum(r['arms'][name]['obstacle_observation_available'] for r in rows) for name in observers},
        current_normal_independently_measured_for_partial_frames=False,residual_threshold_m=.003)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('PARTIAL_HEIGHT_COMPLETE',report['available_frames'],flush=True)


if __name__=='__main__':main()
