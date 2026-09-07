"""Ten frozen fit-only models; all predictions precede native pose scoring."""
import json
import time

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.finite_rgbd_error_members_development import perturb_packets
from lewm.joint_rgbd_rigid_pose_development import RigidRGBDKeyframePose,RIGID_RULES,angle
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_support_aware_rgbd_pose_development_v1 import OUTPUT as PREVIOUS,INPUT,MEMBERS
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import plain
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_joint_rgbd_rigid_pose_development_v1_attempt_001'
PROTOCOL='docs/go2_joint_rgbd_rigid_pose_development_v1_2026-09-06.md'
MODES=('joint','gyro')
IDENTITIES={'launch.json':'bfaacdf54e1b813dd4767be7af3b932e3fec9ee68d6dc9e74cf4429474b1b21e',
    'result.json':'8e0103ac893398fad559976aead6a2a41b1babd4d5dc1b00b4f2700ebc729b7e',
    'predictions.json':'8567930e30879a8644e7a57b059bb315384d9c8db91f7ad60583510a0f00a9e4',
    'evaluation.json':'186592b87bfc650afaf8d14be992fc99047c478840ea07bdd82563c8b4a14844',
    'reference_chain_and_box_audit.json':'e53868496ae3df30be6b214307b6e5c11750a604417b8921293a3195bfab3725'}


def preflight():
    inputs={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(inputs)
    old=read_json(PREVIOUS,'launch.json');verify(old)
    audit=read_json(PREVIOUS,'reference_chain_and_box_audit.json');verify_bindings(audit['source_sha256']|audit['input_sha256'])
    source=discover_sources((PROTOCOL,'scripts/probe_go2_joint_rgbd_rigid_pose_development_v1.py',
        'lewm/tests/test_joint_rgbd_rigid_pose_development.py'),old['source_sha256']|audit['source_sha256'])
    launch=old|dict(source_sha256=source,input_sha256=old['input_sha256']|inputs,
        diagnostic_protocol=PROTOCOL,rigid_rules=RIGID_RULES,modes=list(MODES),
        scope='matched joint/gyro robust fitting replay; no calibrated uncertainty, validation scoring, physics or navigation')
    verify(launch);return launch


def replay():
    models={mode+'__'+m.name:RigidRGBDKeyframePose(mode) for mode in MODES for m in MEMBERS}
    failures={};rows=[]
    for frame in range(336):
        p,d=load_rgbd_observation(INPUT/'fit',frame);f=load_fast_packet(INPUT/'fit',frame);now=p['sensor_state']['decision_ns'];members={}
        for member in MEMBERS:
            pp,dd,ff=perturb_packets(member,p,d,f,anchor_ns=1_500_000_000)
            for mode in MODES:
                name=mode+'__'+member.name;start=time.perf_counter()
                if name in failures:item=dict(status='NOT_REINVOKED_AFTER_FAILURE',state=None,failure=failures[name])
                else:
                    try:
                        state=models[name].observe(pp,dd,ff,now_ns=now)
                        item=dict(status='CONDITIONAL_RIGID_POSE',state=state)
                    except SensorContractError as error:
                        chain=[];cause=error
                        while cause is not None:chain.append(str(cause));cause=cause.__cause__
                        failures[name]=dict(frame=frame,measured_ns=now,chain=chain)
                        item=dict(status='TERMINAL_FAILURE',state=None,failure=failures[name])
                item['wall_ms']=1000*(time.perf_counter()-start);members[name]=item
        rows.append(dict(frame=frame,measured_ns=now,members=members))
        if frame%25==0:print('RIGID_REPLAY',frame,'active',sum(v['state'] is not None for v in members.values()),flush=True)
    return plain(dict(rows=rows,keyframes={n:m.nodes for n,m in models.items()},native_pose_loaded=False,
        validation_frames_loaded=0,error_bounds_available=False))


def score(predictions):
    previous=read_json(PREVIOUS,'evaluation.json');cameras=read_json(INPUT/'fit','camera_audit.json')
    with np.load(INPUT/'fit'/'physics_trace.npz',allow_pickle=False) as raw:poses=raw['base_pose_world'].copy()
    start=poses[749];R0=rotation_xyzw(start[3:]);details={};summaries={}
    for name in predictions['keyframes']:
        records=[];failure=None
        for row in predictions['rows']:
            item=row['members'][name];s=item['state'];frame=row['frame']
            if item['status']=='TERMINAL_FAILURE':failure=item['failure']
            if s is None:continue
            pose=poses[cameras[frame]['physical_sample_index']];Rtrue=R0.T@rotation_xyzw(pose[3:]);truth=R0.T@(pose[:3]-start[:3])
            R=np.asarray(s['rotation_initial_body_from_current_body']);G=np.asarray(s['gyro_rotation_initial_body_from_current_body'])
            record=dict(frame=frame,position_error_m=float(np.linalg.norm(np.asarray(s['position_initial_body_m'])-truth)),
                orientation_error_rad=angle(R.T@Rtrue),gyro_orientation_error_rad=angle(G.T@Rtrue),
                global_image_gyro_disagreement_rad=s['global_image_gyro_disagreement_rad'])
            reg=s['registration']
            if reg is not None:
                ref=poses[cameras[reg['reference_frame']]['physical_sample_index']];Ra=rotation_xyzw(ref[3:])
                t=Ra.T@(pose[:3]-ref[:3]);relative=Ra.T@rotation_xyzw(pose[3:])
                record|=dict(local_translation_error_m=float(np.linalg.norm(np.asarray(reg['translation_reference_body_m'])-t)),
                    local_rotation_error_rad=angle(np.asarray(reg['relative_rotation']).T@relative),
                    reference_gyro_disagreement_rad=reg['gyro_disagreement_rad'])
            records.append(record)
        details[name]=records
        summaries[name]=dict(admitted=len(records),failure=failure,keyframes=len(predictions['keyframes'][name]),
            maximum_position_error_m=max((r['position_error_m'] for r in records),default=None),
            maximum_orientation_error_rad=max((r['orientation_error_rad'] for r in records),default=None),
            maximum_reference_gyro_disagreement_rad=max((r.get('reference_gyro_disagreement_rad',0.) for r in records),default=None),
            error_bounds_calibrated=False)
    comparisons={}
    for member in MEMBERS:
        rows={mode:{r['frame']:r for r in details[mode+'__'+member.name]} for mode in MODES}
        old={r['frame']:r for r in previous['details'][member.name]}
        shared=sorted(set(rows['joint'])&set(rows['gyro'])&set(old))
        comparisons[member.name]=dict(common_all_three_frames=len(shared),
            joint_max_error_m=max((rows['joint'][i]['position_error_m'] for i in shared),default=None),
            matched_gyro_max_error_m=max((rows['gyro'][i]['position_error_m'] for i in shared),default=None),
            previous_support_max_error_m=max((old[i]['position_error_m'] for i in shared),default=None))
    invariance={}
    for name in ('gyro_z_positive','gyro_z_negative'):
        equal=compared=0
        for row in predictions['rows']:
            a=row['members']['joint__nominal']['state'];b=row['members']['joint__'+name]['state']
            if a is None or b is None:continue
            compared+=1
            equal+=int(all(a[k]==b[k] for k in ('position_initial_body_m','rotation_initial_body_from_current_body','reference_frame','promoted_keyframe')))
        invariance[name]=dict(common_admitted_frames=compared,exact_geometry_rows=equal,
            invariance_expected_from_monitor_only_gyro=True,accuracy_not_proven_by_invariance=True)
    return dict(summaries=summaries,details=details,matched_comparisons=comparisons,joint_bias_invariance=invariance,
        native_pose_evaluator_only=True,independent_validation=False,navigation_qualified=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('fresh exclusive joint rigid pose diagnostic only')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        predictions=replay();write_json(OUTPUT/'predictions.json',predictions)
        evaluated=score(predictions);write_json(OUTPUT/'evaluation.json',evaluated);verify(launch)
        result=dict(status='FIT_ONLY_MATCHED_RIGID_POSE_COMPARISON_COMPLETE',summaries=evaluated['summaries'],
            matched_comparisons=evaluated['matched_comparisons'],joint_bias_invariance=evaluated['joint_bias_invariance'],
            artifact_sha256={n:digest(OUTPUT/n) for n in ('predictions.json','evaluation.json')},
            validation_frames_loaded=0,physics_executed=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'result.json',result);print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RIGID_POSE_DIAGNOSTIC_FAILURE',reason=str(error)));raise


if __name__=='__main__':main()
