"""Replay registration, current obstacles and routing on the failed turn's public sensors."""
from pathlib import Path
from threading import Lock
from collections import Counter
import json
import cv2
import numpy as np
import torch
from lewm.feature_budget_100_tracker_development import FeatureBudget100VisualMotion
from lewm.partial_floor_height_development import PartialHeightRegistration,read_pose,SCHEMA
from lewm.partial_height_round_trip_development import PartialHeightObstacles,PartialHeightMap,PartialHeightRoundTripRuntime
from lewm.physical_execution_development import rotation_xyzw
from scripts.in_memory_public_replay_development import PublicReplay


def main():
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_fine_obstacle_round_trip_native_layout00_v1_attempt_001')
    output=root/'partial_height_integration_replay.json'
    if output.exists():raise ValueError('preserve replay')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native');tracker=FeatureBudget100VisualMotion()
    registration=PartialHeightRegistration();obstacles=PartialHeightObstacles();mapper=PartialHeightMap()
    runtime=PartialHeightRoundTripRuntime.__new__(PartialHeightRoundTripRuntime)
    runtime.lock=Lock();runtime.view_recovery=None
    expected=json.loads((root/'poses.json').read_text());rows=[];failure=None
    try:
        for frame in range(193):
            p,d,f,rgb,a,now=reader.packet(frame)
            raw=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=a)
            if frame<len(expected):assert raw['current_pose']==expected[frame]['raw_pose']
            evidence=registration.observe(p,d,a,raw,now_ns=now)
            read_pose(evidence,identity=(0,0,0),now_ns=now)
            if frame<=82:assert evidence['current_pose']==expected[frame]['registered_pose']
            current=obstacles.observe(p,d,f,auxiliary_depth=a,measured_ns=now)
            row=dict(frame=frame,schema=evidence['schema'],pose=evidence['current_pose'],obstacles_available=current is not None,
                current_surface_reason=obstacles.receipts[-1]['joint_plane']['reason'])
            if evidence['schema']==SCHEMA:
                assert registration.anchor['current_pose']['frame']==82
                row['height_update_m']=evidence['floor_transport']['correction']['height_update_m']
            if frame%4==0:
                snapshot=mapper.update(p,d,evidence,auxiliary_depth=a,measured_ns=now)
                row['route']=runtime._route(snapshot,evidence,[0.,2.6],measured_ns=now)
            rows.append(row)
            if frame%40==0:print('PARTIAL_INTEGRATION_FRAME',frame,flush=True)
    except Exception as error:failure=dict(frame=frame,reason=repr(error))
    # Estimation is over before evaluator-only native positions are loaded.
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as z:poses=z['base_pose_world']
    origin=poses[749];R=rotation_xyzw(origin[3:]);errors=[]
    for row in rows:
        actual=(poses[749+50*row['frame'],:3]-origin[:3])@R
        errors.append(float(np.linalg.norm(np.asarray(row['pose']['position_initial_body_m'])-actual)))
    report=dict(frames=len(rows),failure=failure,schemas=dict(Counter(r['schema'] for r in rows)),
        obstacles_available=sum(r['obstacles_available'] for r in rows),
        native_positions_loaded_only_after_estimation=True,native_navigation_executed=False,
        maximum_position_error_m=max(errors,default=None),median_position_error_m=None if not errors else float(np.median(errors)),rows=rows)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('PARTIAL_INTEGRATION_RESULT',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
