"""Expose the original refinement rejection without changing estimator decisions."""
import json
from pathlib import Path
from types import MethodType,FunctionType
import cv2
import torch
from lewm.feature_budget_100_tracker_development import FeatureBudget100VisualMotion
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose,refine
from scripts.in_memory_public_replay_development import PublicReplay


def main():
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_waypoint_alignment_round_trip_native_layout00_v1_attempt_001')
    output=root/'refinement_failure_diagnostic.json'
    if output.exists():raise ValueError('preserve existing diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native');tracker=FeatureBudget100VisualMotion()
    errors=[]
    def observe_refinement(candidate,reference_plane,current_plane,**kwargs):
        try:return refine(candidate,reference_plane,current_plane,**kwargs)
        except Exception as error:
            if tracker.model.frame==415:
                errors.append(dict(frame=415,reference_frame=candidate['reference'].frame,
                    reason=repr(error),camera=kwargs['camera'],
                    registration=candidate['registration'],
                    reference_plane=reference_plane,current_plane=current_plane))
            raise
    method=MeasuredPlaneDualCameraPose._candidate
    observed=FunctionType(method.__code__,method.__globals__|dict(refine=observe_refinement),
        method.__name__,method.__defaults__,method.__closure__)
    tracker.model._candidate=MethodType(observed,tracker.model)
    for frame in range(416):
        p,d,f,rgb,a,now=reader.packet(frame)
        raw=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=a)
        if frame%100==0:print('REFINEMENT_DIAGNOSTIC_FRAME',frame,flush=True)
        if raw['current_pose'] is None:break
    report=dict(frame=frame,status=raw['status'],refinement_errors=errors,
        original_decisions_unchanged=True,public_sensor_replay_only=True,native_state_read=False)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('REFINEMENT_DIAGNOSTIC_COMPLETE',frame,[(r['camera'],r['reason']) for r in errors],flush=True)


if __name__=='__main__':main()
