"""Read-only gate localization, without altering a frozen estimator or inputs."""
import json
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.keyframe_rgbd_pose_development import FeatureFrame,matched_points
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.joint_rgbd_rigid_pose_development import register,cells,RULES,RIGID_RULES
from lewm.causal_sensor_state import SensorContractError
from scripts.run_go2_coupled_room_return_v1 import OUTPUT,TRIALS
from scripts.run_go2_successive_choice_maze_development_v1 import digest
from scripts.startup_raw_sensor_audit_development import read_json,read_npz


def diagnose(directory):
    result=read_json(directory,'result.json');terminal=result['controller_terminal']
    if terminal is None or terminal['evidence'] is None or terminal['evidence']['status']!='VISUAL_TERMINAL_FAILURE':
        return dict(status='NO_TERMINAL_VISUAL_FAILURE')
    rows=read_json(directory,'servo_decisions.json');failed=len(rows)-1
    reference=terminal['evidence']['last_visual']['reference_frame']
    G=np.asarray(rows[reference]['evidence']['current_pose']['gyro_rotation_initial_body_from_current_body'])
    anchor=G.copy();fast=read_npz(directory,'fast_gyro_histories.npz')
    for frame in range(reference+1,failed+1):
        assert fast['valid'][frame].all()
        assert np.array_equal(fast['values'][frame-1,-1],fast['values'][frame,0])
        assert np.array_equal(fast['measured_ns'][frame],fast['measured_ns'][frame,0]+np.arange(51)*2_000_000)
        for before,after in zip(fast['values'][frame,:-1],fast['values'][frame,1:],strict=True):
            G=G@rotation_increment((before+after)*.001)
    p0,d0=load_rgbd_observation(directory,reference);p1,d1=load_rgbd_observation(directory,failed)
    a,b,ua,ub=matched_points(FeatureFrame(p0['image']['rgb'],d0),FeatureFrame(p1['image']['rgb'],d1))
    capture={}
    def local_trace(frame,event,arg):
        if event=='exception':
            values=frame.f_locals
            if all(k in values for k in ('mask','t','ua','ub')):
                mask=values['mask'];t=values['t']
                capture.update(inlier_fraction=float(mask.mean()),inliers=int(mask.sum()),lifted_matches=len(mask),
                    reference_grid_cells=cells(values['ua'][mask]),current_grid_cells=cells(values['ub'][mask]),
                    reference_translation_m=float(np.linalg.norm(t)))
        return local_trace
    def trace(frame,event,arg):return local_trace if frame.f_code is register.__code__ else None
    prior=sys.gettrace()
    try:
        sys.settrace(trace)
        register(a,b,ua,ub,gyro_rotation=anchor.T@G,mode='gyro',frame=failed)
        reason=None
    except SensorContractError as error:reason=str(error)
    finally:sys.settrace(prior)
    if capture:
        capture['failed_gates']=dict(inlier_fraction=capture['inlier_fraction']<RULES['minimum_inlier_fraction'],
            grid_support=min(capture['reference_grid_cells'],capture['current_grid_cells'])<RULES['minimum_grid_cells'],
            reference_translation=capture['reference_translation_m']>RIGID_RULES['maximum_reference_translation_m'])
    # Fixed eight most recent previously accepted keyframes, not chosen by
    # native accuracy. This is read-only feasibility, not an online recovery.
    candidates=[i for i,row in enumerate(rows[:-1]) if row['evidence'] is not None
                and row['evidence']['current_pose'] is not None
                and (i==0 or row['evidence']['current_pose']['promoted_keyframe'])][-8:]
    alternatives=[];current=FeatureFrame(p1['image']['rgb'],d1)
    for index in candidates:
        pose=rows[index]['evidence']['current_pose'];pa,da=load_rgbd_observation(directory,index)
        try:
            a,b,ua,ub=matched_points(FeatureFrame(pa['image']['rgb'],da),current)
            Ra=np.asarray(pose['rotation_initial_body_from_current_body']);Ga=np.asarray(pose['gyro_rotation_initial_body_from_current_body'])
            _,translation,mask,quality=register(a,b,ua,ub,gyro_rotation=Ga.T@G,mode='gyro',frame=failed)
            position=np.asarray(pose['position_initial_body_m'])+Ra@translation
            alternatives.append(dict(reference_frame=index,status='UNCHANGED_PAIR_GATES_PASS',position_initial_body_m=position.tolist(),quality=quality))
        except SensorContractError as error:alternatives.append(dict(reference_frame=index,status='REJECTED',reason=str(error)))
    # Privileged pose is attached ONLY after every hypothesis/gate decision.
    raw=read_npz(directory,'physics_trace.npz');native=raw['base_pose_world']
    R0=Rotation.from_quat(native[749,3:]).as_matrix()
    actual=(native[749+50*failed,:3]-native[749,:3])@R0
    last=np.asarray(rows[failed-1]['evidence']['current_pose']['position_initial_body_m'])
    for candidate in alternatives:
        if candidate['status']=='UNCHANGED_PAIR_GATES_PASS':
            p=np.asarray(candidate['position_initial_body_m'])
            candidate['increment_from_last_visual_m']=float(np.linalg.norm(p-last))
            candidate['increment_bound_pass']=bool(np.linalg.norm(p-last)<=.15)
            candidate['evaluator_only_native_position_error_m']=float(np.linalg.norm(p-actual))
    return dict(status='POSTHOC_FROZEN_GATE_DIAGNOSTIC',failed_frame=failed,reference_frame=reference,
                captured_gate=capture,reconstructed_error=reason,recorded_error=terminal['evidence']['terminal_failure'],
                fixed_recent_keyframe_candidates=alternatives,
                per_trial_result_sha256=digest(directory/'result.json'),decisions_sha256=digest(directory/'servo_decisions.json'),
                inputs_unchanged=True,estimator_modified=False,native_pose_used_for_registration=False,
                native_pose_used_for_posthoc_evaluation_only=True,
                independently_raw_audited=False,navigation_qualified=False)


def main():
    cv2.setNumThreads(1)
    reports={}
    for c in TRIALS:
        directory=OUTPUT/c
        reports[c]=diagnose(directory) if (directory/'result.json').exists() else dict(status='COLLECTION_NOT_TERMINAL')
    print(json.dumps(reports,indent=2),flush=True)


if __name__=='__main__':main()
