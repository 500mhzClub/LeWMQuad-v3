"""Evaluate optional refinement on recorded public camera and body sensors."""
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
import torch
from lewm.optional_plane_refinement_development import OptionalPlaneVisualMotion
from lewm.partial_floor_height_development import PartialHeightRegistration
from lewm.physical_execution_development import rotation_xyzw
from scripts.in_memory_public_replay_development import PublicReplay


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',default='go2_waypoint_alignment_round_trip_native_layout00_v1_attempt_001')
    parser.add_argument('--development-support',action='store_true')
    parser.add_argument('--conditioned-support',action='store_true')
    parser.add_argument('--joint-camera',action='store_true')
    parser.add_argument('--plane-consensus',action='store_true')
    parser.add_argument('--pair-local-plane-conflicts',action='store_true')
    parser.add_argument('--output-name',default=None)
    parser.add_argument('--two-cm-floor-extent',action='store_true')
    parser.add_argument('--budget',type=int,choices=[100,150,300],default=100)
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary artifact basename required')
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')/args.root_name
    if args.development_support and args.conditioned_support:raise ValueError('one support treatment required')
    output=root/('conditioned_support_replay.json' if args.conditioned_support else
        'development_support_replay.json' if args.development_support else 'optional_plane_refinement_replay.json')
    if args.budget!=100:output=output.with_name(f'budget{args.budget}_'+output.name)
    if args.joint_camera:output=output.with_name('joint_camera_'+output.name)
    if args.output_name is not None:
        if Path(args.output_name).name != args.output_name or args.output_name.startswith('sealed') or not args.output_name.endswith('.json'):
            raise ValueError('ordinary JSON output basename required')
        output=output.with_name(args.output_name)
    if output.exists():raise ValueError('preserve replay')
    if args.two_cm_floor_extent:
        from lewm.two_cm_floor_extent_development import configure
        configure()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    reader=PublicReplay(root/'native');tracker=OptionalPlaneVisualMotion();registration=PartialHeightRegistration()
    if args.development_support:
        from lewm.development_support_tracker_development import DevelopmentSupportVisualMotion
        tracker=DevelopmentSupportVisualMotion()
    if args.conditioned_support:
        from lewm.conditioned_support_tracker_development import ConditionedSupportVisualMotion
        tracker=ConditionedSupportVisualMotion()
    if args.budget==150:
        if args.development_support:raise ValueError('five-cell 150 treatment not implemented')
        if args.conditioned_support:
            from lewm.conditioned_support_150_tracker_development import ConditionedSupport150VisualMotion
            tracker=ConditionedSupport150VisualMotion()
        else:
            from lewm.feature_budget_150_tracker_development import FeatureBudget150VisualMotion
            from lewm.optional_plane_refinement_development import refine_if_supported
            tracker=FeatureBudget150VisualMotion();tracker.model._refine_candidate=refine_if_supported
    if args.budget==300:
        if not args.conditioned_support or args.development_support:
            raise ValueError('300-feature sensitivity currently uses conditioned support')
        from lewm.conditioned_support_300_tracker_development import ConditionedSupport300VisualMotion
        tracker=ConditionedSupport300VisualMotion()
    expected=json.loads((root/'poses.json').read_text())
    if args.joint_camera:
        if not args.conditioned_support or args.budget != 150:
            raise ValueError('joint camera treatment uses conditioned 150 support')
        from lewm.joint_camera_anchor_tracker_development import JointCameraAnchorVisualMotion
        tracker=JointCameraAnchorVisualMotion()
    if args.plane_consensus:
        if not args.joint_camera or not args.two_cm_floor_extent:
            raise ValueError('plane consensus extends the joint-camera 2 cm floor treatment')
        from lewm.plane_consensus_tracker_development import PlaneConsensusVisualMotion
        tracker=PlaneConsensusVisualMotion()
    if args.pair_local_plane_conflicts:
        if not args.plane_consensus:raise ValueError('pair-local rejection extends plane consensus')
        from lewm.pair_local_plane_consensus_development import PairLocalPlaneConsensusVisualMotion
        tracker=PairLocalPlaneConsensusVisualMotion()
    count=(json.loads((root/'failure.json').read_text())['acquired_frames'] if (root/'failure.json').exists()
        else json.loads((root/'result.json').read_text())['camera_frames'])
    rows=[];failure=None;first_difference=None
    for frame in range(count):
        try:
            p,d,f,rgb,a,now=reader.packet(frame)
            raw=tracker.observe(p,d,f,now_ns=now,auxiliary_rgb=rgb,auxiliary_depth=a)
            if raw['current_pose'] is None:
                failure=dict(frame=frame,raw=raw);break
            if frame<len(expected) and raw['current_pose']!=expected[frame]['raw_pose'] and first_difference is None:
                first_difference=frame
            evidence=registration.observe(p,d,a,raw,now_ns=now)
            rows.append(dict(frame=frame,raw_pose=raw['current_pose'],pose=evidence['current_pose'],
                refinement=tracker.model.last_measured_plane_refinement))
            if args.pair_local_plane_conflicts:
                rows[-1]['rejected_plane_pairs']=raw['rejected_plane_pairs']
            if frame%100==0:print('OPTIONAL_REFINEMENT_FRAME',frame,flush=True)
        except Exception as error:
            failure=dict(frame=frame,reason=repr(error));break
    # Native state is evaluator-only and loaded after every estimation step.
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as z:native=z['base_pose_world']
    origin=native[749];R=rotation_xyzw(origin[3:]);errors=[]
    for row in rows:
        actual=(native[749+50*row['frame'],:3]-origin[:3])@R
        errors.append(float(np.linalg.norm(actual-np.asarray(row['pose']['position_initial_body_m']))))
    fallback_frames=[r['frame'] for r in rows if (r['refinement'] or {}).get('reason')=='refinement_loses_image_support_original_qualified_fit_retained']
    report=dict(frames=len(rows),failure=failure,first_raw_pose_difference=first_difference,
        development_support_rule=args.development_support,
        conditioned_support_rule=args.conditioned_support,
        joint_camera=args.joint_camera,
        robust_plane_image_consensus=args.plane_consensus,
        pair_local_plane_conflicts=args.pair_local_plane_conflicts,
        plane_consensus_frames=[r['frame'] for r in rows if
            (r['refinement'] or {}).get('robust_plane_image_consensus')],
        floor_second_extent_m=.02 if args.two_cm_floor_extent else .05,
        feature_budget=args.budget,
        fallback_frames=fallback_frames,median_position_error_m=float(np.median(errors)) if errors else None,
        maximum_position_error_m=max(errors,default=None),native_state_loaded_only_after_estimation=True,
        native_navigation_executed=False,rows=rows)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('OPTIONAL_REFINEMENT_RESULT',json.dumps({k:v for k,v in report.items() if k not in ('rows','failure')}),
        'failure',None if failure is None else failure.get('reason','visual failure'),flush=True)


if __name__=='__main__':main()
