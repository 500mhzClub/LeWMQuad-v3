"""Frozen source + recorded sensor replay, native evaluation only afterward."""
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from lewm.cached_rgbd_replay_development import CachedRGBDReplay
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from scripts.run_go2_coupled_room_return_v1 import OUTPUT as INPUT,TRIALS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify

OUTPUT=ROOT/'.generated/go2_multi_reference_recorded_replay_v1_attempt_001'
PROTOCOL='docs/go2_multi_reference_recorded_replay_v1_2026-09-06.md'
IDENTITIES={'launch.json':'be4267ab90f122e18ca8ef8f260cacc2150cdb6aed57f568b6170863c41e0fbf',
 'result.json':'b5ac72bf6ad35df56d99f0dc9be00ae19208d091e805aeaea4501e8d0152cf98',
 'raw_return_audit.json':'2d25a7ce77d4c7c5cf3688f50c6a5f17d872243b10d4bf613554ebca6c97c099'}


def preflight():
    ids={str((INPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(ids)
    old=read_json(INPUT,'launch.json');verify(old);result=read_json(INPUT,'result.json')
    if result['absent_expected_artifacts']:raise ValueError('complete audited predecessor required')
    sources=discover_sources((PROTOCOL,'scripts/replay_go2_multi_reference_v1.py',
        'lewm/tests/test_multi_reference_return_development.py'),old['source_sha256'])
    inputs=old['input_sha256']|ids|{str((INPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    verify_bindings(sources|inputs)
    return dict(source_sha256=sources,input_sha256=inputs,protocol=PROTOCOL,scope='recorded sensor replay only',
                physical_execution=False,model_training=False,goal_achieved=False)


def replay(condition):
    directory=INPUT/condition;reader=CachedRGBDReplay(directory);model=MultiReferenceVisualLedMotion()
    original=read_json(directory,'servo_decisions.json');estimates=[];fallbacks=[];prefix_equal=0;first_failure=None
    for index in range(len(reader.frames)):
        p,d,f,now=reader.packet(index);e=model.observe(p,d,f,now_ns=now)
        pose=e['current_pose'];selection=e['reference_selection']
        if pose is not None and selection['status']=='RECENT_REFERENCE_ACCEPTED':
            fallbacks.append(dict(frame=index,**selection))
            print('REFERENCE_FALLBACK',condition,index,selection['selected_reference'],flush=True)
        if index<len(original) and not fallbacks and original[index]['evidence']['current_pose'] is not None:
            assert json.loads(json.dumps(pose))==original[index]['evidence']['current_pose']
            prefix_equal+=1
        if pose is None and first_failure is None:first_failure=dict(frame=index,failure=e['terminal_failure'],selection=selection)
        estimates.append(dict(frame=index,measured_ns=now,pose=pose,reference_selection=selection))
        if index%500==0:print('REFERENCE_REPLAY',condition,index,e['status'],flush=True)
    # Only evaluator code sees the native trace, after all observation updates.
    raw=read_npz(directory,'physics_trace.npz');native=raw['base_pose_world'];R0=Rotation.from_quat(native[749,3:]).as_matrix()
    errors=[]
    for e in estimates:
        if e['pose'] is not None:
            actual=(native[749+50*e['frame'],:3]-native[749,:3])@R0
            errors.append(float(np.linalg.norm(np.asarray(e['pose']['position_initial_body_m'])-actual)))
    write_json(OUTPUT/(condition+'_estimates.json'),estimates)
    return dict(status='COMPLETE_RECORDED_STREAM_TRACKED' if first_failure is None else 'RECORDED_STREAM_TERMINAL_VISUAL_FAILURE',
                frames=len(estimates),available_pose_frames=len(errors),unavailable_pose_frames=len(estimates)-len(errors),
                first_failure=first_failure,exact_original_prefix_frames=prefix_equal,fallbacks=fallbacks,
                maximum_available_native_position_error_m=max(errors,default=None),
                final_available_native_position_error_m=errors[-1] if errors else None,
                estimates_sha256=digest(OUTPUT/(condition+'_estimates.json')),
                physical_recovery=False,counterfactual_return_success=False,navigation_qualified=False,goal_achieved=False)


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive recorded replay output')
    cv2.setNumThreads(1);launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch);reports={}
    try:
        for c in TRIALS:
            reports[c]=replay(c);write_json(OUTPUT/(c+'_result.json'),reports[c])
            print('REFERENCE_RESULT',c,{k:v for k,v in reports[c].items() if k!='fallbacks'},flush=True)
        verify_bindings(launch['source_sha256']|launch['input_sha256'])
        write_json(OUTPUT/'result.json',dict(status='RECORDED_REPLAY_COMPLETE',conditions=reports,
            result_sha256={c:digest(OUTPUT/(c+'_result.json')) for c in TRIALS},goal_achieved=False))
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_REPLAY_FAILURE',reason=repr(error),completed_conditions=reports));raise


if __name__=='__main__':main()
