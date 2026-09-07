"""Sensor-only stance predictions saved before native and frozen RGBD scoring."""
import json
import numpy as np

from lewm.foot_load_sensor_development import IdealFootForceSample
from lewm.physical_execution_development import rotation_xyzw
from lewm.simulated_body_observation_development import CALIBRATION
from lewm.support_kinematics_development import FootJacobians,CausalQuietUp,predict_support_motion
from scripts.analyze_go2_ground_plane_development_v1 import URDF,verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json,read_npz
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=ROOT/'.generated/go2_causal_support_kinematics_development_v1_attempt_001'
PREVIOUS=ROOT/'.generated/go2_ideal_foot_load_reconstruction_development_v1_attempt_001'
INPUT=ROOT/'.generated/go2_longer_observed_floor_motion_development_v1_attempt_001/fit'
RGBD=ROOT/'.generated/go2_longer_motion_frozen_pose_coverage_status_v1_attempt_001/predictions.json'
PROTOCOL='docs/go2_causal_support_kinematics_development_v1_2026-09-06.md'
MODES=('stationary_centre','level_sphere_rolling')
IDENTITIES={'launch.json':'8428c077fa4b4dec8c6192f65ff9e26e727c141a5b4be39473555141516d4535',
 'result.json':'c0f2de76355e43a231492743feb26e182773ea57baa90e85fb55a340f43a1717',
 'force_conservation_audit_launch.json':'fb93465a8080311f63300a38df495ac07865df725e43b52f54a3047bccd0b49e',
 'force_conservation_audit.json':'9e6056846a72aed538a1172428fde239abc4f7bd976391b3c34fee018403a87d'}


def preflight():
    ids={str((PREVIOUS/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(ids)
    old=read_json(PREVIOUS,'launch.json'); verify(old); audit=read_json(PREVIOUS,'force_conservation_audit_launch.json')
    verify_bindings(audit['source_sha256']|audit['input_sha256']); result=read_json(PREVIOUS,'result.json')
    inputs=old['input_sha256']|audit['input_sha256']|ids|{str((PREVIOUS/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    if str(RGBD.relative_to(ROOT)) not in inputs: raise ValueError('frozen RGBD comparison must be bound')
    sources=discover_sources((PROTOCOL,'scripts/probe_go2_causal_support_kinematics_development_v1.py',
        'lewm/tests/test_support_kinematics_development.py'),old['source_sha256']|audit['source_sha256'])
    launch=old|dict(source_sha256=sources,input_sha256=inputs,protocol=PROTOCOL,
        scope='fitting-only causal load/joint/IMU support hypotheses; no physics or navigation')
    verify(launch); return launch


def predictions():
    body=read_npz(INPUT,'ideal_sensor_samples.npz'); fast=read_npz(INPUT,'fast_gyro_samples.npz')
    loads=read_npz(PREVIOUS,'sensor_predictions.npz'); metadata=read_json(PREVIOUS,'sensor_metadata.json')
    identity=metadata['acquisition_identity']; up_model=CausalQuietUp(); kin=FootJacobians(URDF); rows=[]
    if metadata['axes']!='URDF_foot_link_local' or metadata['kind']!='IDEAL_SIMULATED_THREE_AXIS_NET_FOOT_CONTACT_FORCE':
        raise ValueError('declared ideal load acquisition identity required')
    np.testing.assert_array_equal(fast['measured_ns'],np.arange(1,30001)*2_000_000)
    np.testing.assert_array_equal(body['measured_ns'],np.arange(1,3001)*20_000_000)
    np.testing.assert_array_equal(loads['measured_ns'],fast['measured_ns'])
    np.testing.assert_array_equal(fast['available_ns'],fast['measured_ns'])
    for i in range(649,30000):
        stamp=int(fast['measured_ns'][i]); bindex=(i+1)//10-1
        if not fast['valid'][i].all(): raise ValueError('valid gyro required')
        has_body=stamp%20_000_000==0
        acceleration=body['specific_force_values'][bindex] if has_body else None
        if has_body:
            if not all(body[k][bindex].all() for k in ('specific_force_valid','joints_valid','gyro_valid')):
                raise ValueError('valid current body channels required')
            np.testing.assert_array_equal(body['gyro_values'][bindex],fast['values'][i])
        up=up_model.update(stamp,fast['values'][i],acceleration)
        if not has_body or up is None: continue
        packet=dict(identity=identity,calibration_id=CALIBRATION,measured_ns=stamp,available_ns=stamp,
            q=body['joints_values'][bindex,:12],dq=body['joints_values'][bindex,12:],
            gyro=body['gyro_values'][bindex],specific_force=acceleration,valid=True)
        window=[IdealFootForceSample(identity,int(loads['measured_ns'][j]),int(loads['available_ns'][j]),
            loads['force_foot_n'][j],loads['valid'][j],loads['saturated'][j]) for j in range(i-10,i+1)]
        row=predict_support_motion(kin,packet,window,identity=identity,up_body=up)
        row|=dict(sensor_sample=i,rotation_gyro_anchor_from_body=up_model.Q.tolist()); rows.append(row)
    if len(rows)!=2926: raise ValueError('complete50Hz1.5..60s prediction required')
    write_json(OUTPUT/'predictions.json',dict(rows=rows,native_pose_loaded=False,contact_labels_loaded=False,rgbd_pose_loaded=False))
    return rows


def stats(values):
    return dict(count=len(values),mean=float(np.mean(values)) if values else None,
        p95=float(np.quantile(values,.95)) if values else None,maximum=float(max(values)) if values else None)


def evaluate(rows):
    if not (OUTPUT/'predictions.json').is_file(): raise ValueError('save predictions before scoring')
    raw=read_npz(INPUT,'physics_trace.npz'); phases={}; details=[]
    for row in rows:
        i=row['sensor_sample']; R=rotation_xyzw(raw['base_pose_world'][i,3:]); true=R.T@raw['base_twist_world'][i,:3]
        up=R.T@np.array([0,0,1.]); angle=float(np.arccos(np.clip(up@row['conditional_up_body'],-1,1)))
        phase=str(int(raw['phase'][i])); phases.setdefault(phase,[])
        item=dict(measured_ns=row['measured_ns'],phase=phase,true_velocity_body_m_s=true.tolist(),up_error_rad=angle,modes={})
        selected=np.array(row['selected_loaded_feet'])
        for mode in MODES:
            pred=row['modes'][mode]; v=pred['consensus_velocity_body_m_s']
            errors=np.linalg.norm(np.array(pred['per_foot_velocity_body_m_s'])-true,axis=1)
            item['modes'][mode]=dict(consensus_error_m_s=float(np.linalg.norm(np.array(v)-true)) if v is not None else None,
                selected_per_foot_error_m_s=errors[selected].tolist(),maximum_disagreement_m_s=pred['maximum_disagreement_m_s'])
        details.append(item); phases[phase].append(item)
    summary={phase:{m:dict(error_m_s=stats([r['modes'][m]['consensus_error_m_s'] for r in items if r['modes'][m]['consensus_error_m_s'] is not None]),
        unavailable=sum(r['modes'][m]['consensus_error_m_s'] is None for r in items),
        disagreement_m_s=stats([r['modes'][m]['maximum_disagreement_m_s'] for r in items if r['modes'][m]['maximum_disagreement_m_s'] is not None]))
        for m in MODES} for phase,items in phases.items()}
    write_json(OUTPUT/'native_evaluation.json',details)
    return dict(phases=summary,up_error_rad=stats([r['up_error_rad'] for r in details]),
        total_rows=len(rows),selected_feet_histogram={str(n):sum(sum(r['selected_loaded_feet'])==n for r in rows) for n in range(5)},
        startup=details[0],selected_height_spread_m=stats([r['selected_foot_height_spread_m'] for r in rows if r['selected_foot_height_spread_m'] is not None]))


def compare_rgbd(rows):
    data=read_json(RGBD.parent,RGBD.name)['trials']['fit']['rows']; by_time={r['measured_ns']:i for i,r in enumerate(rows)}; report={}
    for model in ('joint','gyro'):
        for mode in MODES:
            differences=[]; unavailable=0
            for before,after in zip(data[:-1],data[1:]):
                a,b=before['members'][model]['state'],after['members'][model]['state']
                start,end=by_time[before['measured_ns']],by_time[after['measured_ns']]; window=rows[start:end+1]
                if a is None or b is None or any(r['modes'][mode]['consensus_velocity_body_m_s'] is None for r in window):
                    unavailable+=1; continue
                if end-start!=5: raise ValueError('matched100ms window required')
                velocities=np.array([np.array(r['rotation_gyro_anchor_from_body'])@r['modes'][mode]['consensus_velocity_body_m_s'] for r in window])
                displacement=(velocities[:-1]+velocities[1:]).sum(axis=0)*.01
                pred=np.array(window[-1]['rotation_gyro_anchor_from_body']).T@displacement
                measured=np.array(b['rotation_initial_body_from_current_body']).T@(np.array(b['position_initial_body_m'])-a['position_initial_body_m'])
                differences.append(float(np.linalg.norm(pred-measured)))
            report[model+'__'+mode]=dict(displacement_disagreement_m=stats(differences),unavailable_windows=unavailable,
                rgbd_used_for_prediction=False,comparison_is_not_independent_ground_truth=True)
    return report


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive causal support attempt')
    launch=preflight(); OUTPUT.mkdir(); write_json(OUTPUT/'launch.json',launch)
    try:
        rows=predictions(); result=evaluate(rows); result['rgbd_comparison']=compare_rgbd(rows); verify(launch)
        result|=dict(status='CAUSAL_SUPPORT_KINEMATICS_DIAGNOSTIC_COMPLETE',physical_error_calibrated=False,navigation_qualified=False,
            goal_achieved=False,artifact_sha256={n:digest(OUTPUT/n) for n in ('predictions.json','native_evaluation.json')})
        write_json(OUTPUT/'result.json',result); print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_CAUSAL_SUPPORT_DIAGNOSTIC_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()
