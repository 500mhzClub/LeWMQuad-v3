#!/usr/bin/env python3
"""Moment-aware fusion replay and explicitly degraded live-sensor stress cases."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(path))
from lewm.depth_inertial_moment_fusion_development import MomentWeakSubspaceIntegrator as WeakSubspaceIntegrator, MomentDepthInertialState, ASSUMPTIONS
from lewm.depth_inertial_stress_development import perturb, CASES, FRAME_COUNT
from lewm.causal_sensor_state import SensorContractError
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.fast_gyro_scan_session_development import load_fast_packet
from lewm.physical_execution_development import rotation_xyzw
from lewm.whole_task_rgb_dataset_development import load_whole_task_observation
from scripts.run_go2_depth_inertial_fusion_replay_development_v1 import preflight as prior_preflight
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings as verify

OUTPUT=ROOT/'.generated/go2_depth_inertial_moment_replay_development_v1_attempt_001'
SCHEMA='depth_inertial_moment_replay_development.v1'
STUDIES=('measured_region','measured_line_integral','release_aware','observable_hold','depth_floor_hold')
COUNTS=(511,782,864,402,1051)
NEW_SOURCES=(
    'lewm/depth_inertial_moment_fusion_development.py',
    'lewm/tests/test_depth_inertial_moment_fusion_development.py',
    'lewm/depth_inertial_stress_development.py',
    'lewm/tests/test_depth_inertial_stress_development.py',
    'scripts/run_go2_depth_inertial_moment_replay_development_v1.py',
    'scripts/audit_go2_depth_inertial_moment_replay_development_v1.py',
    'lewm/tests/test_depth_inertial_moment_replay_development.py',
    'docs/go2_depth_inertial_moment_replay_development_v1_2026-09-05.md',
    'docs/go2_depth_inertial_fusion_replay_development_v1_result_2026-09-05.md')


def closure(inherited):
    available=set(subprocess.run(['rg','--files','-g','*.py','lewm','scripts','lewm_genesis','lewm_worlds'],
        cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines())
    pending=[p for p in NEW_SOURCES if p.endswith('.py')]; visited=set()
    while pending:
        name=pending.pop()
        if name in visited or name in inherited: continue
        if name not in available or (ROOT/name).resolve()!=ROOT/name: raise ValueError('ordinary source required')
        visited.add(name); parts=Path(name).parts[:-1]
        for end in range(1,len(parts)+1):
            init=str(Path(*parts[:end])/'__init__.py')
            if init in available: pending.append(init)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            if isinstance(node,ast.Import): modules=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                module=node.module or ''
                if node.level: module='.'.join([*parts[:len(parts)-node.level+1],*([module] if module else [])])
                modules=[module,*[module+'.'+n.name for n in node.names]]
            else: continue
            for module in modules:
                prefix=module.split('.')[0]
                if prefix not in ('lewm','scripts','lewm_genesis','lewm_worlds'): continue
                stems=[module.replace('.','/')]
                if prefix in ('lewm_genesis','lewm_worlds'): stems.append(prefix+'/'+stems[0])
                for stem in stems:
                    for candidate in (stem+'.py',stem+'/__init__.py'):
                        if candidate in available: pending.append(candidate)
    result=inherited.copy()
    for name in visited|set(NEW_SOURCES):
        sha=digest(ROOT/name)
        if name in result and result[name]!=sha: raise ValueError('frozen source changed')
        result[name]=sha
    verify(result)
    return result


def preflight():
    inherited=prior_preflight()
    prior=ROOT/'.generated/go2_depth_inertial_fusion_replay_development_v1_attempt_001'
    identities={str((prior/leaf).relative_to(ROOT)):sha for leaf,sha in (
        ('launch.json','0704e85130bab1ae7be26e4e2909fdf1ae6b3f7bc0b04f50613afebd8696320e'),
        ('result.json','ad958582a364da3b04464159c29dadebb5501e4ccdafc687d45bd2d322d1974c'),
        ('independent_verification.json','f607a8ffa326181922a3b0231a353607a2aa23728f598c59576dfd0b0b070909'))}
    verify(identities)
    previous=json.loads((prior/'launch.json').read_text())
    audited=json.loads((prior/'independent_verification.json').read_text())
    if previous!=inherited or audited['status']!='PASS' or audited['verified_estimates']!=3610:
        raise ValueError('exact complete predecessor required')
    inherited['source_sha256']|=audited['audit_source_sha256']
    inherited['source_sha256']=closure(inherited['source_sha256'])
    inherited['input_sha256']|=identities|audited['evidence_sha256']
    latest=ROOT/'.generated/go2_depth_floor_hold_navigation_development_v1_attempt_001'
    north=json.loads((latest/'result.json').read_text())['trials'][0]
    if north['scene_id']!='go2-depth-floor-hold-navigation-development-v1-north_dogleg':
        raise ValueError('fixed stress trace required')
    directory=latest/north['scene_id']
    for leaf in ['depth_observations.json','fast_gyro_histories.npz',*[f'depth_{i:04d}.npz' for i in range(FRAME_COUNT)]]:
        inherited['artifact_sha256'][str((directory/leaf).relative_to(ROOT))]=north['artifact_sha256'][leaf]
    inherited.update(schema=SCHEMA,stress_population=[
        {'case':case,'directory':str(directory.relative_to(ROOT)),'frames':FRAME_COUNT} for case in CASES],
        scope='moment-aware nominal development replay and declared sensor degradation; no counterfactual physical navigation')
    verify(inherited['source_sha256']|inherited['input_sha256']|inherited['artifact_sha256'])
    return inherited


def estimate(directory,count):
    """No evaluator arrays or geometry enter this function or the kernel."""
    records=json.loads((directory/'relative_state_observations.json').read_text())
    if len(records)!=count: raise ValueError('observer population mismatch')
    model=WeakSubspaceIntegrator(); outputs=[]
    for index,record in enumerate(records):
        if record['observation_index']!=index: raise ValueError('observer sequence mismatch')
        policy=load_whole_task_observation(directory,index)
        outputs.append(model.observe(policy,record['observer']))
    return outputs


def score(outputs,poses):
    """Independent physical-reference reduction after prediction is complete."""
    poses=np.asarray(poses,dtype=float)
    if poses.shape!=(len(outputs),7) or not np.isfinite(poses).all() or not outputs:
        raise ValueError('complete finite evaluation poses required')
    initial=rotation_xyzw(poses[0,3:]); rows=[]
    for index,(out,pose) in enumerate(zip(outputs,poses,strict=True)):
        truth=initial.T@(pose[:3]-poses[0,:3])
        error=float(np.linalg.norm(np.asarray(out['position_initial_body_m'])-truth))
        step=None
        if index:
            true_delta=rotation_xyzw(poses[index-1,3:]).T@(pose[:3]-poses[index-1,:3])
            step=float(np.linalg.norm(np.asarray(out['translation_previous_body_m'])-true_delta))
        rows.append({'index':index,'fusion':out,'position_error_m':error,'step_error_m':step,
            'error_exceeds_declared_proxy':error>out['position_error_scale_m']+1e-12})
    checks={'all_frames_processed':True,
        'step_error_at_most_1cm':all(r['step_error_m']<=.01 for r in rows[1:]),
        'maximum_position_error_at_most_5cm':max(r['position_error_m'] for r in rows)<=.05,
        'final_position_error_at_most_5cm':rows[-1]['position_error_m']<=.05,
        'no_proxy_exceedance':not any(r['error_exceeds_declared_proxy'] for r in rows),
        'no_proxy_budget_exhaustion':all(r['fusion']['usable_under_declared_proxy_budget'] for r in rows)}
    return {'frames':len(rows),'weak_intervals':sum(r['fusion']['kind']=='INERTIALLY_PREDICTED_WEAK_COMPONENT' for r in rows),
        'maximum_step_error_m':max((r['step_error_m'] for r in rows[1:]),default=None),
        'maximum_position_error_m':max(r['position_error_m'] for r in rows),
        'final_position_error_m':rows[-1]['position_error_m'],
        'proxy_exceedances':sum(r['error_exceeds_declared_proxy'] for r in rows),
        'checks':checks,'passes_declared_replay_checks':all(checks.values()),'rows':rows}


def estimate_stress(directory,case):
    observer=MomentDepthInertialState(); outputs=[]; records=[]; fault=None; first_ns=None
    for index in range(FRAME_COUNT):
        policy,depth=load_rgbd_observation(directory,index)
        now=policy['sensor_state']['decision_ns']
        if first_ns is None: first_ns=now
        policy,depth=perturb(policy,depth,case,first_ns=first_ns)
        try:
            result=observer.observe(policy,depth,load_fast_packet(directory,index),now_ns=now)
        except SensorContractError as error:
            fault={'index':index,'reason':str(error),'cause':str(error.__cause__)}
            break
        outputs.append(result['fusion'])
        records.append({'observation_index':index,'observer':result['depth_state']})
    return outputs,records,fault


def main():
    if sys.argv[1:] not in ([],['--preflight']): raise ValueError('fixed study only')
    launch=preflight()
    if sys.argv[1:]:
        print(json.dumps({'status':'PASS','sources':len(launch['source_sha256']),
            'artifacts':len(launch['artifact_sha256']),'trajectories':len(launch['population']),
            'frames':sum(p['frames'] for p in launch['population']),'stress_cases':len(launch['stress_population'])}),flush=True)
        return
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT/'launch.json',launch); results=[]; stress=[]
    try:
        for item in launch['population']:
            directory=ROOT/item['directory']
            outputs=estimate(directory,item['frames'])
            # Evaluation data is opened only after the sensor-only pass.
            cameras=json.loads((directory/'camera_audit.json').read_text())
            with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
                poses=archive['base_pose_world'][[c['physical_sample_index'] for c in cameras]]
            result={**item,**score(outputs,poses)}
            write_json(OUTPUT/(item['scene_id']+'.json'),result)
            summary={k:v for k,v in result.items() if k!='rows'}; results.append(summary)
            print(json.dumps({'completed':len(results),**summary}),flush=True)
        for item in launch['stress_population']:
            directory=ROOT/item['directory']
            outputs,records,fault=estimate_stress(directory,item['case'])
            cameras=json.loads((directory/'camera_audit.json').read_text())
            with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
                poses=archive['base_pose_world'][[c['physical_sample_index'] for c in cameras[:len(outputs)]]]
            result={**item,**score(outputs,poses),'expected_frames':FRAME_COUNT,
                'completed_fixed_frames':len(outputs)==FRAME_COUNT,'sensor_fault':fault,'observed_records':records,
                'first_proxy_stop_index':next((i for i,o in enumerate(outputs) if not o['usable_under_declared_proxy_budget']),None),
                'predictions_after_proxy_stop_are_diagnostic_only':True}
            write_json(OUTPUT/('stress_'+item['case']+'.json'),result)
            summary={k:v for k,v in result.items() if k not in ('rows','observed_records')}
            stress.append(summary); print(json.dumps({'stress_completed':len(stress),**summary}),flush=True)
        verify(launch['source_sha256']|launch['input_sha256']|launch['artifact_sha256'])
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','launch_sha256':digest(OUTPUT/'launch.json'),
            'trials':results,'stress_cases':stress,'passes_declared_replay_checks':all(r['passes_declared_replay_checks'] for r in results),
            'scope':launch['scope'],'navigation_qualified':False,'hardware_qualified':False})
        print(json.dumps({'status':'COMPLETE','trials':len(results),'stress_cases':len(stress)}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'terminal_failure.json',{'status':'FAIL','error':repr(error),'completed':results,'stress_completed':stress})
        raise


if __name__=='__main__': main()
