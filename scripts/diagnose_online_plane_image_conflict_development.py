"""Preserve the public image/plane estimates behind a reproducible conflict."""
import argparse
import json
import cv2
import numpy as np
import torch
from lewm import two_cm_floor_extent_development as floor_extent
from lewm.joint_camera_anchor_tracker_development import JointCameraAnchorVisualMotion,refine_joint_if_supported
from lewm.measured_plane_dual_camera_pose_development import PlaneImageConflict
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.navigation_artifact_root_development import BASE,validate_root


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    args=parser.parse_args();root=BASE/args.root_name;validate_root(root,must_exist=True)
    output=root/'plane_image_conflict_diagnostic.json'
    if output.exists():raise ValueError('preserve diagnostic')
    floor_extent.configure();cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native');tracker=JointCameraAnchorVisualMotion()
    expected=json.loads((root/'poses.json').read_text())
    count=json.loads((root/'failure.json').read_text())['acquired_frames']
    conflicts=[];first_difference=None;failure=None
    def capture(candidate,reference_plane,current_plane,**kwargs):
        try:return refine_joint_if_supported(candidate,reference_plane,current_plane,**kwargs)
        except PlaneImageConflict as error:
            ref=candidate['reference']
            conflicts.append(dict(frame=tracker.model.frame,reason=str(error),
                candidate={k:v for k,v in candidate.items() if k!='reference'},
                reference=dict(frame=ref.frame,measured_ns=ref.measured_ns,
                    rotation=ref.rotation,position=ref.position,gyro=ref.gyro),
                reference_plane=reference_plane,current_plane=current_plane,refinement_arguments=kwargs))
            raise
    tracker.model._refine_candidate=capture
    for frame in range(count):
        p,d,f,rgb,a,now=reader.packet(frame)
        raw=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=a)
        if raw['current_pose'] is None:
            failure=dict(frame=frame,raw=raw);break
        if frame<len(expected) and raw['current_pose']!=expected[frame]['raw_pose'] and first_difference is None:
            first_difference=frame
        if frame%200==0:print('PLANE_IMAGE_DIAGNOSTIC_FRAME',frame,flush=True)
    report=dict(native_state_used=False,pose_admitted_after_conflict=False,
        first_raw_pose_difference=first_difference,failure=failure,conflicts=conflicts)
    with output.open('x') as stream:json.dump(report,stream,indent=2,default=serial);stream.write('\n')
    print('PLANE_IMAGE_DIAGNOSTIC',None if failure is None else failure['frame'],len(conflicts),flush=True)


if __name__=='__main__':main()
