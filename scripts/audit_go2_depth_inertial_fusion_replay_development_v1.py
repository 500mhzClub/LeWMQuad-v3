#!/usr/bin/env python3
"""Independent error/component/composition checks, not a second experiment."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(path))
from scripts.run_go2_depth_inertial_fusion_replay_development_v1 import OUTPUT, STUDIES, COUNTS
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings


def check(condition,message):
    if not condition: raise ValueError(message)


def matrix(q):
    """Quaternion quadratic form, independent of the runtime rotation helper."""
    q=np.asarray(q,dtype=float); q=q/np.linalg.norm(q); v=q[:3]; w=q[3]
    x,y,z=v
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return (w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*cross


def audit_rows(result,observed,poses):
    rows=result['rows']; check(len(rows)==len(observed)==len(poses)==result['frames'],'complete rows')
    initial=matrix(poses[0,3:]); position=np.zeros(3); previous_scale=0.; weak_count=0
    position_errors=[]; step_errors=[]; exceedances=0
    for index,(row,record,pose) in enumerate(zip(rows,observed,poses,strict=True)):
        fusion=row['fusion']; state=record['observer']
        check(row['index']==record['observation_index']==index,'ordered rows')
        check(fusion['measured_ns']==state['measured_ns'],'current estimates')
        check(fusion['position_error_scale_m']>=previous_scale,'nonshrinking uncertainty')
        previous_scale=fusion['position_error_scale_m']
        check(not fusion['assumptions']['calibrated_covariance'] and not fusion['assumptions']['hardware_qualified'],
            'no qualified uncertainty claim')
        if index:
            motion=state['motion']; delta=np.asarray(fusion['translation_previous_body_m'])
            basis=np.asarray(motion['weak_directions_previous_body']).reshape(-1,3)
            constrained=np.eye(3)-basis.T@basis
            check(np.allclose(constrained@delta,motion['observable_projection_previous_body_m'],atol=1e-10,rtol=0),
                'fusion altered an observed component')
            check(fusion['depth_rank']==motion['rank'],'original depth rank')
            if motion['rank']==3:
                check(fusion['kind']=='DEPTH_CONSTRAINED_TRANSLATION','full depth label')
            else:
                weak_count+=1
                check(fusion['kind']=='INERTIALLY_PREDICTED_WEAK_COMPONENT'
                    and motion['translation_previous_body_m'] is None,'weak component must remain predicted')
            rotation=np.asarray(observed[index-1]['observer']['relative_orientation']['rotation_initial_body_from_current_body'])
            position=position+rotation@delta
            true_step=matrix(poses[index-1,3:]).T@(pose[:3]-poses[index-1,:3])
            error=float(np.linalg.norm(delta-true_step)); step_errors.append(error)
            check(abs(error-row['step_error_m'])<1e-10,'independent step error')
        else:
            check(fusion['translation_previous_body_m'] is None and row['step_error_m'] is None,'initial anchor')
        check(np.allclose(position,fusion['position_initial_body_m'],atol=1e-10,rtol=0),'position composition')
        true_position=initial.T@(pose[:3]-poses[0,:3])
        error=float(np.linalg.norm(np.asarray(fusion['position_initial_body_m'])-true_position))
        position_errors.append(error)
        check(abs(error-row['position_error_m'])<1e-10,'independent position error')
        exceeded=error>fusion['position_error_scale_m']+1e-12
        check(exceeded==row['error_exceeds_declared_proxy'],'proxy exceedance reduction'); exceedances+=exceeded
    check(weak_count==result['weak_intervals'] and exceedances==result['proxy_exceedances'],'weak/proxy counts')
    for key,value in (('maximum_step_error_m',max(step_errors)),('maximum_position_error_m',max(position_errors)),
            ('final_position_error_m',position_errors[-1])):
        check(abs(result[key]-value)<1e-10,'independent summary '+key)
    expected={'all_frames_processed':True,'step_error_at_most_1cm':max(step_errors)<=.01,
        'maximum_position_error_at_most_5cm':max(position_errors)<=.05,
        'final_position_error_at_most_5cm':position_errors[-1]<=.05,'no_proxy_exceedance':exceedances==0,
        'no_proxy_budget_exhaustion':all(r['fusion']['usable_under_declared_proxy_budget'] for r in rows)}
    check(result['checks']==expected and result['passes_declared_replay_checks']==all(expected.values()),'all descriptive checks')
    return {'frames':len(rows),'weak_intervals':weak_count,'independent_error_component_composition_check':'PASS'}


def main():
    check(len(sys.argv)==1 and not (OUTPUT/'independent_verification.json').exists(),'fresh fixed verification')
    launch=json.loads((OUTPUT/'launch.json').read_text()); report=json.loads((OUTPUT/'result.json').read_text())
    check(report['status']=='COMPLETE' and report['launch_sha256']==digest(OUTPUT/'launch.json'),'completed bound replay')
    check(len(launch['population'])==len(report['trials'])==10,'ten fixed trajectories')
    check(tuple(sum(p['frames'] for p in launch['population'] if p['study']==s) for s in STUDIES)==COUNTS,'fixed frame population')
    evidence={str((OUTPUT/leaf).relative_to(ROOT)):digest(OUTPUT/leaf) for leaf in
        ['launch.json','result.json',*[p['scene_id']+'.json' for p in launch['population']]]}
    source={name:digest(ROOT/name) for name in (
        'scripts/audit_go2_depth_inertial_fusion_replay_development_v1.py',
        'lewm/tests/test_depth_inertial_fusion_audit_development.py')}
    bindings=launch['source_sha256']|launch['input_sha256']|launch['artifact_sha256']|evidence|source
    verify_bindings(bindings); rows=[]
    for item,summary in zip(launch['population'],report['trials'],strict=True):
        result=json.loads((OUTPUT/(item['scene_id']+'.json')).read_text())
        check({k:v for k,v in result.items() if k!='rows'}==summary,'exact member summary')
        directory=ROOT/item['directory']; cameras=json.loads((directory/'camera_audit.json').read_text())
        observed=json.loads((directory/'relative_state_observations.json').read_text())
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            poses=archive['base_pose_world'][[c['physical_sample_index'] for c in cameras]]
        rows.append({'scene_id':item['scene_id'],**audit_rows(result,observed,poses)})
    verify_bindings(bindings)
    write_json(OUTPUT/'independent_verification.json',{'status':'PASS','trials':rows,'evidence_sha256':evidence,
        'audit_source_sha256':source,'verified_estimates':sum(r['frames'] for r in rows),
        'scope':'independent metric/component/composition verification; not new depth registration replay or navigation qualification'})
    print(json.dumps({'status':'PASS','verified_estimates':sum(r['frames'] for r in rows),'weak_intervals':sum(r['weak_intervals'] for r in rows)}),flush=True)


if __name__=='__main__': main()
