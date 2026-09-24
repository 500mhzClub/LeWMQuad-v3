"""Compare front/downward RGB-D support on fixed previously failed pairs."""
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.return_transition_match_diagnosis_development import diagnose_pair
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import register,angle,RIGID_RULES
from lewm.auxiliary_reference_pose_adapter_development import gyro_in_reference,pose_in_body
from lewm.physical_execution_development import rotation_xyzw
from lewm.causal_sensor_state import SensorContractError
from scripts.novel_maze_auxiliary_packet_development import packet,public_acquisition
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,artifact_path,verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware,source_check
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.diagnose_go2_return_transition_matches_v1 import INPUT,CASE,FRAMES,OUTPUT as DIAGNOSIS

OUTPUT=BASE/'go2_auxiliary_return_pair_diagnosis_v1_attempt_001'
PROTOCOL='docs/go2_auxiliary_return_pair_diagnosis_v1_2026-09-09.md'
DIAGNOSIS_RESULT='29565cd5f2d87ba6c9d657780fc9ca75f908ab7cd28d0713f4da28d2b758e081'
NATIVE_RESULT='3745c0b7c45265a2fcc38caf732b6f3f4b228487f344d23502dc7395077d5755'


def verify_all(launch):
    source_check(launch['source_sha256']);verify_artifacts(INPUT,launch['input_sha256'])
    verify_artifacts(DIAGNOSIS,launch['diagnosis_artifact_sha256'])


def fit_pair(reference,current,relative_gyro,*,frame,auxiliary):
    points=matched_points(reference,current)
    try:
        R,t,mask,fit=register(*points,gyro_rotation=gyro_in_reference(relative_gyro) if auxiliary else relative_gyro,
            mode='joint',frame=frame)
        if auxiliary:R,t=pose_in_body(R,t)
        return dict(status='PAIR_RIGID_FIT_ACCEPTED',registration=fit,
            rotation_reference_body_from_current_body=R.tolist(),translation_reference_body_m=t.tolist(),
            body_reference_translation_envelope_pass=bool(np.linalg.norm(t)<=RIGID_RULES['maximum_reference_translation_m']),
            body_increment_translation_envelope_pass=bool(np.linalg.norm(t)<=RIGID_RULES['maximum_increment_translation_m']),
            body_increment_rotation_envelope_pass=bool(angle(R)<=RIGID_RULES['maximum_increment_rotation_rad']),
            auxiliary_reference_frame_adapter_used=auxiliary,online_pose_acceptance_claim=False)
    except SensorContractError as error:
        return dict(status='PAIR_RIGID_FIT_REJECTED',reason=str(error),online_pose_acceptance_claim=False)


def evaluate_pairs(pairs):
    # Native data enters only after all public-only matching and fitting.
    with np.load(artifact_path(INPUT,CASE+'/physics_trace.npz'),allow_pickle=False) as archive:
        poses=archive['base_pose_world']
    for row in pairs:
        previous,current=[poses[749+50*row[k]] for k in ('reference','current')]
        A,B=rotation_xyzw(previous[3:]),rotation_xyzw(current[3:])
        truth_R=A.T@B;truth_t=A.T@(current[:3]-previous[:3])
        row['evaluator_only_native_relative_pose']=dict(rotation=truth_R.tolist(),translation_m=truth_t.tolist())
        for view in ('primary','auxiliary'):
            fit=row[view]
            if fit['status']=='PAIR_RIGID_FIT_ACCEPTED':
                fit['evaluator_only_native_error']=dict(
                    translation_m=float(np.linalg.norm(np.asarray(fit['translation_reference_body_m'])-truth_t)),
                    rotation_rad=angle(truth_R.T@np.asarray(fit['rotation_reference_body_from_current_body'])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if not __debug__:raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive new auxiliary-pair diagnosis required')
    cv2.setNumThreads(1)
    verify_artifacts(DIAGNOSIS,{'result.json':DIAGNOSIS_RESULT})
    result=read_json(DIAGNOSIS,'result.json');diagnosis_ids={'result.json':DIAGNOSIS_RESULT,**result['artifact_sha256']}
    verify_artifacts(DIAGNOSIS,diagnosis_ids);old=read_json(DIAGNOSIS,'launch.json')
    verify_artifacts(INPUT,{'result.json':NATIVE_RESULT});native=read_json(INPUT,'result.json')
    assert native['status']=='LATER_FLOOR_RESOLUTION_MAZE_PILOT_COMPLETE'
    names=[CASE+'/auxiliary_camera_audit.json',CASE+'/physics_trace.npz']
    names += [CASE+f'/auxiliary_{kind}_{i:04d}.{suffix}' for i in FRAMES for kind,suffix in (('rgb','png'),('depth','npz'))]
    ids=old['input_sha256']|{'result.json':NATIVE_RESULT}|{n:native['artifact_sha256'][n] for n in names}
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_auxiliary_return_pairs_v1.py',
        'lewm/tests/test_auxiliary_reference_pose_adapter_development.py'),result['source_sha256'])
    resources=hardware()
    launch=dict(protocol=PROTOCOL,output_root=str(OUTPUT),source_sha256=sources,input_sha256=ids,
        diagnosis_artifact_sha256=diagnosis_ids,hardware=resources,frames=list(FRAMES),
        pair_population=[{k:r[k] for k in ('reference','current','role')} for r in result['pairs']],
        thresholds_unchanged=True,auxiliary_rgb_is_additional_uninstalled_input=True,
        primary_reference_frame_indices_are_analysis_pairs_not_auxiliary_retention_state=True,
        native_data_used_only_for_postfit_error_evaluation=True,cpu_processes=1,numerical_threads=1,
        native_scene_workers=0,native_execution=False,model_training=False,parameter_search=False,
        minimum_available_ram_bytes=2*1024**3,output_allowance_bytes=64*1024**2,os_resource_limits_enforced=False,
        concurrency_reason='bounded public pair diagnosis beside one native maze scene and one independent CPU replay')
    verify_all(launch)
    memory_ok=resources['memory_available_bytes']>=2*1024**3
    storage_ok=resources['artifact_free_bytes']>=40*1024**3+64*1024**2
    if args.preflight_only:
        print('AUXILIARY_RETURN_PAIR_PREFLIGHT',json.dumps(dict(source_count=len(sources),hardware=resources,
            memory_admission_pass=memory_ok,storage_admission_pass=storage_ok,output_created=False)),flush=True);return
    if not memory_ok or not storage_ok:raise ValueError('bounded pair diagnosis resources unavailable')
    create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('AUXILIARY_RETURN_PAIR_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True);started=time.perf_counter()
    try:
        directory=INPUT/CASE;reader=IntentReturnRGBDReplay(directory);gyro=FastRelativeOrientation()
        acquisitions=read_json(directory,'auxiliary_camera_audit.json')
        frames={'primary':{},'auxiliary':{}};rotations={};witnesses={}
        for i in FRAMES:
            p,d,f,now=reader.packet(i)
            attitude=gyro.begin(p,f,now_ns=now) if i==FRAMES[0] else gyro.step(p,f,now_ns=now)
            rotations[i]=np.asarray(attitude['rotation_initial_body_from_current_body'])
            auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
            with Image.open(artifact_path(directory,f'auxiliary_rgb_{i:04d}.png')) as image:
                rgb=np.array(image)
            rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
            if rgb_sha!=acquisitions[i]['rgb_sha256']:raise ValueError('auxiliary RGB pixels differ from acquisition receipt')
            frames['primary'][i]=CornerSupportFeatureFrame(p['image']['rgb'],d)
            frames['auxiliary'][i]=CornerSupportFeatureFrame(rgb,auxiliary)
            assert frames['primary'][i].witness()=={k:result['feature_witnesses'][str(i)][k] for k in frames['primary'][i].witness()}
            witnesses[i]=dict(primary=frames['primary'][i].witness(),auxiliary=frames['auxiliary'][i].witness(),
                measured_ns=now,auxiliary_rgb_sha256=rgb_sha,
                auxiliary_depth_sha256=hashlib.sha256(auxiliary['depth_m'].tobytes()+auxiliary['valid'].tobytes()).hexdigest())
        pairs=[]
        for saved in result['pairs']:
            a,b=saved['reference'],saved['current'];row={k:saved[k] for k in ('reference','current','role')}
            relative=rotations[a].T@rotations[b]
            for view in frames:
                counts=diagnose_pair(frames[view][a],frames[view][b])
                if view=='primary':assert counts=={k:saved[k] for k in counts}
                row[view]=counts|fit_pair(frames[view][a],frames[view][b],relative,frame=b,auxiliary=view=='auxiliary')
            pairs.append(row)
        assert len(pairs)==25
        evaluate_pairs(pairs)
        verify_all(launch)
        write_json(OUTPUT/'result.json',dict(status='AUXILIARY_RETURN_PAIR_DIAGNOSIS_COMPLETE',
            source_sha256=sources,artifact_sha256={'launch.json':digest(OUTPUT/'launch.json')},
            feature_witnesses=witnesses,pairs=pairs,wall_s=time.perf_counter()-started,hardware_after=hardware(),
            original_primary_stage_counts_exact=True,thresholds_unchanged=True,input_bytes_unchanged=True,
            native_data_used_only_for_postfit_error_evaluation=True,online_continuity_evaluated=False,
            auxiliary_rgb_packet_contract_implemented=False,candidate_installed=False,
            original_failed_navigation_outcome_unchanged=True,navigation_qualified=False,goal_achieved=False))
        print('AUXILIARY_RETURN_PAIR_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(reason=repr(error)));raise


if __name__=='__main__':main()
