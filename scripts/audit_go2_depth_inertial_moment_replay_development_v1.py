#!/usr/bin/env python3
"""Exact sensor replay plus independent motion/error/component verification."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(path))
from scripts.run_go2_depth_inertial_moment_replay_development_v1 import (
    OUTPUT,STUDIES,COUNTS,CASES,FRAME_COUNT,estimate,estimate_stress)
from scripts.audit_go2_depth_inertial_fusion_replay_development_v1 import audit_rows,check,matrix
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings


def plain(value): return json.loads(json.dumps(value,allow_nan=False))


def assumption_diagnostics(result,poses,velocities):
    initial=matrix(poses[0,3:]); errors=[]; acceleration=[]
    for k,row in enumerate(result['rows'][1:],1):
        fusion=row['fusion']; truth=initial.T@velocities[k]
        if fusion['depth_rank']==3:
            errors.append(float(np.linalg.norm(np.asarray(fusion['velocity_initial_body_m_s'])-truth)))
        reference=initial.T@(velocities[k]-velocities[k-1])/.1
        acceleration.append(float(np.linalg.norm(np.asarray(fusion['acceleration_initial_body_m_s2'])-reference)))
    return {'full_depth_velocity_error_max_m_s':max(errors,default=None),
        'full_depth_velocity_errors_above_0_005':sum(e>.005 for e in errors),
        'interval_acceleration_error_max_m_s2':max(acceleration,default=None),
        'interval_acceleration_errors_above_0_02':sum(e>.02 for e in acceleration),
        'operating_envelope_validated':False}


def main():
    check(len(sys.argv)==1 and not (OUTPUT/'independent_verification.json').exists(),'fresh fixed audit')
    launch=json.loads((OUTPUT/'launch.json').read_text()); report=json.loads((OUTPUT/'result.json').read_text())
    check(report['status']=='COMPLETE' and report['launch_sha256']==digest(OUTPUT/'launch.json'),'completed bound study')
    check(len(report['trials'])==len(launch['population'])==10,'ten nominal trajectories')
    check(tuple(sum(p['frames'] for p in launch['population'] if p['study']==s) for s in STUDIES)==COUNTS,'fixed nominal population')
    check(tuple(p['case'] for p in launch['stress_population'])==CASES and len(report['stress_cases'])==4,'four fixed perturbations')
    evidence={str((OUTPUT/leaf).relative_to(ROOT)):digest(OUTPUT/leaf) for leaf in
        ['launch.json','result.json',*[p['scene_id']+'.json' for p in launch['population']],
         *['stress_'+case+'.json' for case in CASES]]}
    bindings=launch['source_sha256']|launch['input_sha256']|launch['artifact_sha256']|evidence
    verify_bindings(bindings); rows=[]
    for stress,items,summaries in ((False,launch['population'],report['trials']),
            (True,launch['stress_population'],report['stress_cases'])):
        for item,summary in zip(items,summaries,strict=True):
            name='stress_'+item['case'] if stress else item['scene_id']
            result=json.loads((OUTPUT/(name+'.json')).read_text()); directory=ROOT/item['directory']
            check({k:v for k,v in result.items() if k not in ('rows','observed_records')}==summary,'exact member summary')
            if stress:
                expected,records,fault=estimate_stress(directory,item['case'])
                check(plain(records)==result['observed_records'] and fault==result['sensor_fault'],'exact degraded depth/body observer and fault replay')
                check(result['expected_frames']==FRAME_COUNT
                    and result['completed_fixed_frames']==(len(expected)==FRAME_COUNT),'complete or explicit fault population')
                check(result['first_proxy_stop_index']==next((i for i,o in enumerate(expected)
                    if not o['usable_under_declared_proxy_budget']),None),'first uncertainty budget stop')
            else:
                expected=estimate(directory,item['frames'])
                records=json.loads((directory/'relative_state_observations.json').read_text())
            check(plain(expected)==[r['fusion'] for r in result['rows']],'exact causal fusion replay')
            cameras=json.loads((directory/'camera_audit.json').read_text())[:len(expected)]
            with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
                ids=[c['physical_sample_index'] for c in cameras]
                poses=archive['base_pose_world'][ids]; velocities=archive['base_twist_world'][ids,:3]
            rows.append({'name':name,**audit_rows(result,records,poses),
                'assumption_diagnostics':assumption_diagnostics(result,poses,velocities)})
            print(json.dumps({'verified':len(rows),'name':name,'frames':len(expected)}),flush=True)
    verify_bindings(bindings)
    write_json(OUTPUT/'independent_verification.json',{'status':'PASS','cases':rows,'evidence_sha256':evidence,
        'audit_source_sha256':digest(Path(__file__)),'verified_estimates':sum(r['frames'] for r in rows),
        'scope':'exact nominal and perturbed sensor/observer replay with independent motion reduction; not navigation or calibrated uncertainty'})
    print(json.dumps({'status':'PASS','cases':len(rows),'verified_estimates':sum(r['frames'] for r in rows)}),flush=True)


if __name__=='__main__': main()
