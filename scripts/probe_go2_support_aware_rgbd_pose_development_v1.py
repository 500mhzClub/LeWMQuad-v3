"""Same-input support-aware comparison, separate outlier-bound evidence."""
import json
import time

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.finite_rgbd_error_members_development import perturb_packets
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.support_aware_rgbd_pose_development import SupportAwareRGBDPose,RobustReferenceEvidence,OUTLIER_FRACTION,SUPPORT_MARGIN_CELLS
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_keyframe_rgbd_pose_development_v1 import OUTPUT as PREVIOUS,INPUT,MEMBERS,score as original_score
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_support_aware_rgbd_pose_development_v1_attempt_001'
PROTOCOL='docs/go2_support_aware_rgbd_pose_development_v1_2026-09-06.md'
IDENTITIES={'launch.json':'0383825888a0cb374f38156406a483392ec948e3d74776d515b14ecfc03f8709',
    'result.json':'c6b85182b915f14fea86256926cdeacb6b2f69997023d64deadbf3ccce05f73c',
    'predictions.json':'083057a91f5622e15dc5485a1c102b41a4b794797ff9db63e8f64a48fc135791',
    'evaluation.json':'ab883d1881dd18b4836ee4bab2729999732b71d45ff75ad822f1ef8b0db7ad43',
    'accepted_pair_and_terminal_analysis.json':'ce59fd99b4439a99dc3f0d25f3cbb524e2bee560fed55e94afea4f8360fc7698'}


def preflight():
    inputs={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(inputs)
    old=read_json(PREVIOUS,'launch.json');verify(old)
    audit=read_json(PREVIOUS,'accepted_pair_and_terminal_analysis.json');verify_bindings(audit['source_sha256']|audit['input_sha256'])
    source=discover_sources((PROTOCOL,'scripts/probe_go2_support_aware_rgbd_pose_development_v1.py',
        'lewm/tests/test_support_aware_rgbd_pose_development.py'),old['source_sha256']|audit['source_sha256'])
    launch=old|dict(source_sha256=source,input_sha256=old['input_sha256']|inputs,
        diagnostic_protocol=PROTOCOL,support_margin_cells=SUPPORT_MARGIN_CELLS,outlier_fraction=OUTLIER_FRACTION,
        scope='fit-only same-input reference-selection comparison and conditional outlier-bound diagnostic')
    verify(launch);return launch


def replay():
    baseline=read_json(PREVIOUS,'predictions.json');models={m.name:SupportAwareRGBDPose() for m in MEMBERS}
    evidence={m.name:RobustReferenceEvidence() for m in MEMBERS};failures={};rows=[];prefix={m.name:0 for m in MEMBERS}
    for frame in range(336):
        p,d=load_rgbd_observation(INPUT/'fit',frame);f=load_fast_packet(INPUT/'fit',frame);now=p['sensor_state']['decision_ns'];members={}
        for member in MEMBERS:
            name=member.name;start=time.perf_counter()
            if name in failures:item=dict(status='NOT_REINVOKED_AFTER_FAILURE',state=None,failure=failures[name])
            else:
                pp,dd,ff=perturb_packets(member,p,d,f,anchor_ns=1_500_000_000)
                try:state=models[name].observe(pp,dd,ff,now_ns=now)
                except SensorContractError as error:
                    chain=[];cause=error
                    while cause is not None:chain.append(str(cause));cause=cause.__cause__
                    failures[name]=dict(frame=frame,measured_ns=now,chain=chain)
                    item=dict(status='TERMINAL_FAILURE',state=None,failure=failures[name])
                else:
                    state['robust_evidence']=evidence[name].observe(state)
                    previous=baseline['rows'][frame]['members'][name]['state']
                    if previous is not None and state['reference_frame']==previous['reference_frame']==0:
                        for key in ('position_initial_body_m','rotation_initial_body_from_current_body','conditional_global_position_radius_m'):
                            assert state[key]==previous[key]
                        prefix[name]+=1
                    item=dict(status='CONDITIONAL_KEYFRAME_POSE',state=state)
            item['wall_ms']=1000*(time.perf_counter()-start);members[name]=item
        rows.append(dict(frame=frame,measured_ns=now,members=members))
        if frame%25==0:print('SUPPORT_REPLAY',frame,{n:models[n].nodes[-1]['frame'] for n in models},flush=True)
    return plain(dict(rows=rows,keyframes={n:m.nodes for n,m in models.items()},
        exact_common_initial_reference_rows=prefix,native_pose_loaded=False,validation_frames_loaded=0))


def score(predictions):
    # All predictions already persisted. Shared predecessor native scoring is
    # retained, then same-input V1 and robust-evidence tests are added explicitly.
    evaluated=original_score(predictions);previous=read_json(PREVIOUS,'evaluation.json')
    cameras=read_json(INPUT/'fit','camera_audit.json')
    with np.load(INPUT/'fit'/'physics_trace.npz',allow_pickle=False) as raw:poses=raw['base_pose_world'].copy()
    for member in MEMBERS:
        name=member.name;records=evaluated['details'][name];old={r['frame']:r for r in previous['details'][name]};paired=[]
        for entry in records:
            frame=entry['frame'];state=predictions['rows'][frame]['members'][name]['state'];robust=state['robust_evidence']
            radius=robust['global_radius_m'];entry['robust_global_radius_m']=radius
            entry['robust_global_radius_exceeded']=None if radius is None else bool(entry['position_error_m']>radius+1e-12)
            if frame in old:paired.append((entry['position_error_m'],old[frame]['position_error_m']))
            if frame==0:continue
            reg=state['registration'];pose=poses[cameras[frame]['physical_sample_index']]
            ref=poses[cameras[reg['reference_frame']]['physical_sample_index']];Ra=rotation_xyzw(ref[3:])
            t=Ra.T@(pose[:3]-ref[:3]);Rtrue=Ra.T@rotation_xyzw(pose[3:]);R=np.asarray(reg['relative_rotation'])
            angle=float(np.arccos(np.clip((np.trace(R.T@Rtrue)-1)/2,-1,1)))
            entry|=dict(local_angle_error_rad=angle,
                local_angle_hypothesis_exceeded=bool(angle>reg['conditional_local_angle_radius_rad']+1e-7),
                outlier_allowance=robust['maximum_outliers'],
                majority_point_hypothesis_exceeded=bool(entry['inlier_static_point_hypothesis_violations']>robust['maximum_outliers']),
                robust_local_box_available=robust['status']=='CONDITIONAL_CONTAMINATED_BOX',
                native_translation_inside_robust_box=bool(np.all(t>=np.asarray(robust['lower'])-1e-12)
                    and np.all(t<=np.asarray(robust['upper'])+1e-12)))
        states=[r['members'][name]['state'] for r in predictions['rows'] if r['members'][name]['state'] is not None]
        radii=[r['robust_global_radius_m'] for r in records if r['robust_global_radius_m'] is not None]
        evaluated['summaries'][name]|=dict(support_promotions=sum(s['promotion_reason']=='accepted_support_margin' for s in states),
            motion_promotions=sum(s['promotion_reason']=='motion_threshold' for s in states),
            robust_global_available=len(radii),maximum_robust_global_radius_m=max(radii,default=None),
            robust_global_radius_exceedances=sum(r['robust_global_radius_exceeded'] is True for r in records),
            inconsistent_local_boxes=sum(not r.get('robust_local_box_available',True) for r in records),
            native_outside_available_local_box=sum(r.get('robust_local_box_available',False) and not r['native_translation_inside_robust_box'] for r in records),
            majority_point_hypothesis_exceedances=sum(r.get('majority_point_hypothesis_exceeded',False) for r in records),
            local_gyro_hypothesis_exceedances=sum(r.get('local_angle_hypothesis_exceeded',False) for r in records),
            matched_input_v1_common_frames=len(paired),matched_input_new_max_error_m=max((p[0] for p in paired),default=None),
            matched_input_v1_max_error_m=max((p[1] for p in paired),default=None))
    return evaluated


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('fresh exclusive support-aware diagnostic only')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        predictions=replay();write_json(OUTPUT/'predictions.json',predictions)
        evaluated=score(predictions);write_json(OUTPUT/'evaluation.json',evaluated);verify(launch)
        result=dict(status='FIT_ONLY_SUPPORT_AWARE_DIAGNOSTIC_COMPLETE',summaries=evaluated['summaries'],
            artifact_sha256={n:digest(OUTPUT/n) for n in ('predictions.json','evaluation.json')},
            validation_frames_loaded=0,physics_executed=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_SUPPORT_AWARE_DIAGNOSTIC_FAILURE',reason=str(error)));raise


if __name__=='__main__':main()
