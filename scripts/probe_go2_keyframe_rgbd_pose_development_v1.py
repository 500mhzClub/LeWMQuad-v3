"""Frozen five-member fit-only keyframe replay; native scoring after persistence."""
from dataclasses import asdict
import json
import time

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.finite_rgbd_error_members_development import ErrorMember,perturb_packets
from lewm.keyframe_rgbd_pose_development import KeyframeRGBDPose,KeyframeHypotheses,point_radius
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.rgbd_shadow_motion_development import ShadowObserver
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_startup_observation_turn_development_v1 import json_same
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.fresh_maze_session_development import priors
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_sustained_observed_floor_motion_development_v1 import OUTPUT as INPUT,PROTOCOL as INPUT_PROTOCOL
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_keyframe_rgbd_pose_development_v1_attempt_001'
PROTOCOL='docs/go2_keyframe_rgbd_pose_development_v1_2026-09-06.md'
IDENTITIES={'launch.json':'731656438060789d3d36b03cbd20f146d8b46d451d825e911254d59c70f3b575',
    'result.json':'aa554d6398fadd5b21f0a298f2f463bd926aecc31d626609f062d7fa341aa1c9',
    'raw_acquisition_audit_launch.json':'ca75f35acebe8d4c72995155f1ca4591a6f20cf4fa60af348517362c5ebe913d',
    'raw_acquisition_audit.json':'ca84564e2139f50b12609b4ee53f5bbfb73a25f3387e500cae33cd532c7102b8'}
MEMBERS=(ErrorMember('nominal'),ErrorMember('blank_rgb',blank_rgb=True),
    ErrorMember('gyro_z_positive',gyro_z_bias_rad_s=.001),ErrorMember('gyro_z_negative',gyro_z_bias_rad_s=-.001),
    ErrorMember('depth_noise_positive',independent_depth_noise_m=.0001))


def preflight():
    inputs={str((INPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(inputs)
    old=read_json(INPUT,'launch.json');verify(old)
    audit=read_json(INPUT,'raw_acquisition_audit_launch.json');verify_bindings(audit['source_sha256']|audit['input_sha256'])
    result=read_json(INPUT,'raw_acquisition_audit.json')
    inputs|={str((INPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items() if n.startswith('fit_')}
    inputs|={n:h for n,h in audit['input_sha256'].items() if n.startswith(str((INPUT/'fit').relative_to(ROOT))+'/')}
    source=discover_sources((PROTOCOL,'scripts/probe_go2_keyframe_rgbd_pose_development_v1.py',
        'lewm/tests/test_keyframe_rgbd_pose_development.py'),audit['source_sha256'])
    launch=old|dict(source_sha256=source,input_sha256=old['input_sha256']|inputs,
        diagnostic_protocol=PROTOCOL,members=[asdict(m) for m in MEMBERS],keyframe_hypotheses=asdict(KeyframeHypotheses()),
        scope='fit-only recorded diagnostic; no new physics, validation scoring, calibrated bounds or navigation')
    verify(launch);return launch


def replay(launch):
    directory=INPUT/'fit';saved=read_json(directory,'shadow_observations.json')
    prior,_=priors(launch['source_sha256'][INPUT_PROTOCOL]);old=ShadowObserver(prior)
    models={m.name:KeyframeRGBDPose() for m in MEMBERS};failures={};rows=[]
    assert len(saved)==336
    for frame,expected in enumerate(saved):
        p,d,f=(*load_rgbd_observation(directory,frame),load_fast_packet(directory,frame));now=p['sensor_state']['decision_ns']
        original=old.observe(p,d,f,now_ns=now);json_same(original,expected['shadow'])
        members={}
        for definition in MEMBERS:
            name=definition.name;start=time.perf_counter()
            if name in failures:
                item=dict(status='NOT_REINVOKED_AFTER_FAILURE',state=None,failure=failures[name])
            else:
                pp,dd,ff=perturb_packets(definition,p,d,f,anchor_ns=prior.anchor_ns)
                try:
                    state=models[name].observe(pp,dd,ff,now_ns=now)
                    item=dict(status='CONDITIONAL_KEYFRAME_POSE',state=state)
                except SensorContractError as error:
                    reasons=[];cause=error
                    while cause is not None: reasons.append(str(cause));cause=cause.__cause__
                    failures[name]=dict(frame=frame,measured_ns=now,chain=reasons)
                    item=dict(status='TERMINAL_FAILURE',state=None,failure=failures[name])
            item['wall_ms']=1000*(time.perf_counter()-start);members[name]=item
        rows.append(dict(frame=frame,measured_ns=now,members=members))
        if frame%25==0: print('REPLAY',frame,{k:v['status'] for k,v in members.items()},flush=True)
    return plain(dict(rows=rows,keyframes={n:m.nodes for n,m in models.items()},
        predecessor_exact_replay_frames=336,validation_frames_loaded=0,native_pose_loaded=False))


def score(predictions):
    directory=INPUT/'fit';cameras=read_json(directory,'camera_audit.json');old=read_json(directory,'shadow_observations.json')
    with np.load(directory/'physics_trace.npz',allow_pickle=False) as raw: poses=raw['base_pose_world'].copy()
    initial=poses[749];R0=rotation_xyzw(initial[3:]);summaries={};details={};h=KeyframeHypotheses()
    for member in MEMBERS:
        name=member.name;records=[];failures=[];common=[];old_common=[];pair_count=pair_violations=0
        for row in predictions['rows']:
            item=row['members'][name];frame=row['frame']
            if item['status']=='TERMINAL_FAILURE':failures.append(item['failure'])
            if item['state'] is None:continue
            state=item['state'];pose=poses[cameras[frame]['physical_sample_index']]
            truth=R0.T@(pose[:3]-initial[:3]);Rtrue=R0.T@rotation_xyzw(pose[3:])
            estimate=np.asarray(state['position_initial_body_m']);R=np.asarray(state['rotation_initial_body_from_current_body'])
            error=float(np.linalg.norm(estimate-truth));angle=float(np.arccos(np.clip((np.trace(R.T@Rtrue)-1)/2,-1,1)))
            entry=dict(frame=frame,position_error_m=error,orientation_error_rad=angle,
                conditional_position_radius_m=state['conditional_global_position_radius_m'],
                conditional_angle_radius_rad=state['conditional_global_angle_radius_rad'],
                position_radius_exceeded=bool(error>state['conditional_global_position_radius_m']+1e-12),
                angle_radius_exceeded=bool(angle>state['conditional_global_angle_radius_rad']+1e-7))
            registration=state['registration']
            if registration is not None:
                ref=poses[cameras[registration['reference_frame']]['physical_sample_index']];Ra=rotation_xyzw(ref[3:])
                Rt=Ra.T@rotation_xyzw(pose[3:]);tt=Ra.T@(pose[:3]-ref[:3])
                a,b,ua,ub=[np.asarray(registration[k]) for k in ('reference_inlier_points_body_m','current_inlier_points_body_m',
                    'reference_inlier_pixels','current_inlier_pixels')]
                mismatch=np.linalg.norm(a-b@Rt.T-tt,axis=1)
                allowed=point_radius(a,ua,h)+point_radius(b,ub,h)
                pair_count+=len(a);pair_violations+=int(np.sum(mismatch>allowed+1e-12))
                entry|=dict(inlier_static_point_hypothesis_checks=len(a),inlier_static_point_hypothesis_violations=int(np.sum(mismatch>allowed+1e-12)),
                    maximum_inlier_reference_consistency_m=float(mismatch.max()))
            if old[frame]['shadow']['state'] is not None:
                old_error=float(np.linalg.norm(np.asarray(old[frame]['shadow']['state']['fusion']['position_initial_body_m'])-truth))
                common.append(error);old_common.append(old_error)
            records.append(entry)
        summaries[name]=dict(admitted=len(records),failure=failures[0] if failures else None,
            keyframes=len(predictions['keyframes'][name]),
            maximum_admitted_position_error_m=max((r['position_error_m'] for r in records),default=None),
            maximum_admitted_orientation_error_rad=max((r['orientation_error_rad'] for r in records),default=None),
            maximum_conditional_position_radius_m=max((r['conditional_position_radius_m'] for r in records),default=None),
            position_radius_exceedances=sum(r['position_radius_exceeded'] for r in records),
            angle_radius_exceedances=sum(r['angle_radius_exceeded'] for r in records),
            inlier_static_point_checks=pair_count,inlier_static_point_violations=pair_violations,
            common_old_admitted_frames=len(common),common_maximum_new_error_m=max(common,default=None),
            common_maximum_original_error_m=max(old_common,default=None),conditional_bounds_calibrated=False)
        details[name]=records
    return dict(summaries=summaries,details=details,native_pose_evaluator_only=True,independent_validation=False,navigation_qualified=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('fresh exclusive keyframe diagnostic only')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        predictions=replay(launch);write_json(OUTPUT/'predictions.json',predictions)
        evaluated=score(predictions);write_json(OUTPUT/'evaluation.json',evaluated);verify(launch)
        result=dict(status='FIT_ONLY_KEYFRAME_DIAGNOSTIC_COMPLETE',summaries=evaluated['summaries'],
            artifact_sha256={n:digest(OUTPUT/n) for n in ('predictions.json','evaluation.json')},
            independent_new_physical_trials=0,validation_frames_loaded=0,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_KEYFRAME_DIAGNOSTIC_FAILURE',reason=str(error)));raise


if __name__=='__main__':main()
