"""Paired full recorded streams; descriptor budget fixed and gates unchanged."""
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from lewm.balanced_multi_reference_rgbd_development import BalancedVisualLedMotion
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.run_go2_intent_room_return_v1 import OUTPUT as INPUT,TRIALS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import artifact_path,verify_artifacts
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify

OUTPUT=ROOT/'.generated/go2_balanced_feature_replay_v1_attempt_001'
PROTOCOL='docs/go2_balanced_feature_replay_v1_2026-09-06.md'
IDENTITIES={'launch.json':'7a5c427ca521de367a301376fafd262aeda5f6e16b7ed876f2b40e87ce0b1a91',
    'result.json':'27e2f91eaece8667e48fb98d75ce3ee8cfcc3a1c9d0052a97033cad5acc321d3',
    'raw_return_audit.json':'a350c74d4f5851a7486bf01a8276be7f9eb420a83cd8198ad684159b8385923d'}


def preflight():
    verify_artifacts(INPUT,IDENTITIES);old=read_json(INPUT,'launch.json');verify(old)
    result=read_json(INPUT,'result.json')
    if result['absent_expected_artifacts']:raise ValueError('complete predecessor required')
    external=IDENTITIES|result['artifact_sha256'];verify_artifacts(INPUT,external)
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_balanced_features_v1.py',
        'lewm/tests/test_balanced_rgbd_features_development.py'),old['source_sha256'])
    verify_bindings(sources|old['input_sha256'])
    return dict(source_sha256=sources,input_sha256=old['input_sha256'],external_root=str(INPUT),
        external_input_sha256=external,physical_execution=False,model_training=False,goal_achieved=False)


def replay(condition):
    reader=IntentReturnRGBDReplay(INPUT/condition);models={'original':MultiReferenceVisualLedMotion(),'balanced':BalancedVisualLedMotion()}
    records={k:[] for k in models};original=read_json(INPUT/condition,'servo_decisions.json');verified=0
    for frame in range(len(reader.frames)):
        p,d,f,now=reader.packet(frame)
        for name,model in models.items():
            start=time.perf_counter_ns();result=model.observe(p,d,f,now_ns=now)
            elapsed=(time.perf_counter_ns()-start)/1e6
            record=dict(frame=frame,measured_ns=now,pose=result['current_pose'],status=result['status'],
                terminal_failure=result['terminal_failure'],selection=result['reference_selection'],
                feature_selection=result.get('feature_selection'),observer_wall_ms=elapsed)
            records[name].append(record)
            if name=='original' and frame<len(original):
                import json
                if json.loads(json.dumps(result['current_pose']))!=original[frame]['evidence']['current_pose']:
                    raise ValueError('original actual controller pose does not replay exactly')
                verified+=1
            if result['current_pose'] is None and (frame==0 or records[name][-2]['pose'] is not None):
                print('BALANCED_REPLAY_FIRST_FAILURE',condition,name,frame,result['terminal_failure'],flush=True)
        if frame%250==0:print('BALANCED_REPLAY',condition,frame,{k:v[-1]['status'] for k,v in records.items()},flush=True)
    for name,rows in records.items():write_json(OUTPUT/(condition+'_'+name+'_estimates.json'),rows)
    # Persist all sensor estimates BEFORE native evaluator data enters scope.
    native=read_npz(INPUT/condition,'physics_trace.npz')['base_pose_world'];R0=Rotation.from_quat(native[749,3:]).as_matrix()
    reports={}
    for name,rows in records.items():
        valid=[r for r in rows if r['pose'] is not None]
        errors=[float(np.linalg.norm(np.asarray(r['pose']['position_initial_body_m'])-
            (native[749+50*r['frame'],:3]-native[749,:3])@R0)) for r in valid]
        timing=np.array([r['observer_wall_ms'] for r in valid])
        reports[name]=dict(frames=len(rows),available_pose_frames=len(valid),
            first_failure=next((dict(frame=r['frame'],failure=r['terminal_failure'],selection=r['selection']) for r in rows if r['pose'] is None),None),
            maximum_native_position_error_m=max(errors,default=None),
            final_available_native_position_error_m=errors[-1] if errors else None,
            fallback_frames=[r['frame'] for r in valid if r['selection']['status']=='RECENT_REFERENCE_ACCEPTED'],
            median_available_observer_wall_ms=float(np.median(timing)) if len(timing) else None,
            estimates_sha256=digest(OUTPUT/(condition+'_'+name+'_estimates.json')))
    return dict(arms=reports,exact_original_decisions=verified,physical_recovery=False,navigation_qualified=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive balanced-feature replay')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);reports={}
    try:
        for c in TRIALS:
            reports[c]=replay(c);write_json(OUTPUT/(c+'_result.json'),reports[c])
            print('BALANCED_REPLAY_RESULT',c,reports[c],flush=True)
        verify_bindings(launch['source_sha256']|launch['input_sha256']);verify_artifacts(INPUT,launch['external_input_sha256'])
        write_json(OUTPUT/'result.json',dict(status='BALANCED_FEATURE_REPLAY_COMPLETE',conditions=reports,
            result_sha256={c:digest(OUTPUT/(c+'_result.json')) for c in TRIALS},goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_BALANCED_REPLAY_FAILURE',reason=repr(error),completed_conditions=reports));raise


if __name__=='__main__':main()
