"""Small sensor-only supervised response baseline on exposed servo failures."""
import json
import math
import numpy as np

from lewm.bounded_visual_servo_development import wrapped
from lewm.visual_course_response_development import VisualCourseWindow,action_features,fit_response
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

INPUT=ROOT/'.generated/go2_bounded_visual_servo_v1_attempt_001'
OUTPUT=ROOT/'.generated/go2_visual_course_response_v1_attempt_001'
PROTOCOL='docs/go2_visual_course_response_v1_2026-09-06.md'
IDENTITIES={'launch.json':'52275754c7f12902aa120a17252bc8155d23e58b8b983b383086777baf6e1b90',
    'result.json':'84de6410fa19eb489eb34d1399a80f541e0370203db03f7b04bddf3f1f3613cc',
    'raw_servo_audit_launch.json':'3821fadb3e1139f5e75edbaf0cec7f3f88fc341af0cd3f6123905aa71a376ee5',
    'raw_servo_audit.json':'291e19821a0bc88608d4dc153061a687af5bcd2012a797198d87db39af2b8f01'}


def preflight():
    ids={str((INPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(ids)
    old=read_json(INPUT,'launch.json'); verify(old); audit=read_json(INPUT,'raw_servo_audit_launch.json')
    verify_bindings(audit['source_sha256']|audit['input_sha256'])
    sources=discover_sources((PROTOCOL,'scripts/fit_go2_visual_course_response_v1.py',
        'lewm/tests/test_visual_course_response_development.py'),old['source_sha256']|audit['source_sha256'])
    launch={k:old[k] for k in ('native_sha256','native_geometry_sha256','opencv_binary_sha256','opencv_version','rules')}
    launch|=dict(source_sha256=sources,input_sha256=old['input_sha256']|audit['input_sha256']|ids,
        protocol=PROTOCOL,scope='exposed-development sensor-only regression and leave-one-condition-out diagnostic; no physical trial')
    verify(launch); return launch


def rows(condition):
    records=read_json(INPUT/condition,'servo_decisions.json'); window=VisualCourseWindow(); output=[]
    for i,row in enumerate(records):
        p=row['evidence']['current_pose']
        if p is None: break
        R=np.asarray(p['rotation_initial_body_from_current_body']); yaw=math.atan2(R[1,0],R[0,0])
        course=window.observe(measured_ns=p['measured_ns'],position_initial_xy_m=p['position_initial_body_m'][:2],yaw_rad=yaw)
        if i<5 or i==len(records)-1 or row['decision']['terminal'] is not None: continue
        after=records[i+1]['evidence']['current_pose']
        if after is None: continue
        assert after['measured_ns']-p['measured_ns']==100_000_000
        S=np.asarray(after['rotation_initial_body_from_current_body']); next_yaw=math.atan2(S[1,0],S[0,0])
        displacement=R.T@(np.asarray(after['position_initial_body_m'])-p['position_initial_body_m'])
        X=action_features(row['decision']['requested_command'],course,yaw)
        target=[float(displacement[0]/.1),float(displacement[1]/.1),wrapped(next_yaw-yaw)/.1]
        output.append(dict(frame=i,measured_ns=p['measured_ns'],label_measured_ns=after['measured_ns'],
            phase=row['decision']['phase'],features=X.tolist(),target_body_velocity_and_yaw_rate=target,course=course))
    return output


def errors(prediction,target):
    error=np.asarray(prediction)-target
    return dict(rows=len(error),planar_velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
                yaw_rate_rmse_rad_s=float(np.sqrt(np.mean(error[:,2]**2))))


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive response fitting diagnostic')
    launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        data={c:rows(c) for c in ('nominal','lower_friction')}; write_json(OUTPUT/'sensor_rows.json',data)
        arrays={c:(np.array([r['features'] for r in rs]),np.array([r['target_body_velocity_and_yaw_rate'] for r in rs])) for c,rs in data.items()}
        reports={}; models={}; predictions={}
        for train,test in [('nominal','nominal'),('lower_friction','lower_friction'),('nominal','lower_friction'),('lower_friction','nominal')]:
            X,Y=arrays[train]; model=fit_response(X,Y); A,B=arrays[test]; estimate=A@model['coefficients']
            name=train+'__to__'+test; models[name]=model; predictions[name]=estimate.tolist()
            reports[name]=dict(fitted=errors(estimate,B),past_motion_persistence=errors(A[:,4:7],B),
                action_only_command_integration=errors(np.column_stack((A[:,1],np.zeros(len(A)),A[:,2])),B),
                action_design_rank=model['action_design_rank'],full_rank=model['standardized_rank'],
                independent_validation=False,causal_action_effect_identified=False)
        verify(launch); write_json(OUTPUT/'models_and_predictions.json',dict(models=models,predictions=predictions))
        result=dict(status='VISUAL_COURSE_RESPONSE_DEVELOPMENT_COMPLETE',condition_rows={c:len(rs) for c,rs in data.items()},
            comparisons=reports,native_pose_used=False,simulator_friction_label_used_as_feature=False,
            model_used_for_controller=False,physical_error_calibrated=False,navigation_qualified=False,goal_achieved=False,
            artifact_sha256={n:digest(OUTPUT/n) for n in ('sensor_rows.json','models_and_predictions.json')})
        write_json(OUTPUT/'result.json',result); print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_VISUAL_COURSE_RESPONSE_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()
