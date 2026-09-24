#!/usr/bin/env python3
"""Source-bound, policy-only fusion of ten complete development histories."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(path))
from lewm.depth_inertial_fusion_development import WeakSubspaceIntegrator, ASSUMPTIONS
from lewm.physical_execution_development import rotation_xyzw
from lewm.whole_task_rgb_dataset_development import load_whole_task_observation
from scripts.run_go2_depth_floor_hold_navigation_development_v1 import preflight as prior_preflight
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings as verify

OUTPUT=ROOT/'.generated/go2_depth_inertial_fusion_replay_development_v1_attempt_001'
SCHEMA='depth_inertial_fusion_replay_development.v1'
STUDIES=('measured_region','measured_line_integral','release_aware','observable_hold','depth_floor_hold')
COUNTS=(511,782,864,402,1051)
NEW_SOURCES=(
    'lewm/depth_inertial_fusion_development.py',
    'lewm/tests/test_depth_inertial_fusion_development.py',
    'scripts/run_go2_depth_inertial_fusion_replay_development_v1.py',
    'lewm/tests/test_depth_inertial_fusion_replay_development.py',
    'docs/go2_depth_inertial_fusion_replay_development_v1_2026-09-05.md',
    'docs/go2_depth_floor_hold_navigation_development_v1_result_2026-09-05.md')


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
    sources,inputs,gait,native=prior_preflight()
    latest=ROOT/'.generated/go2_depth_floor_hold_navigation_development_v1_attempt_001'
    for leaf,sha in (
        ('launch.json','af7c8a2ff20e6fe68adeca5939b5f3efa1da643bd87822a55af1245fa0fe92d1'),
        ('result.json','9dff970f0f725efca2677d6c4422b124e83f11fcb12b2c88c86a65749c9adea7'),
        ('raw_artifact_audit.json','8e816c9af34aff14f47f1442a6a4c4bedfd29685bca62d8c61857c6f11722f17')):
        inputs[str((latest/leaf).relative_to(ROOT))]=sha
    verify(inputs)
    if json.loads((latest/'launch.json').read_text())['source_sha256']!=sources:
        raise ValueError('exact frozen predecessor source required')
    population=[]; artifacts={}
    for study,total in zip(STUDIES,COUNTS,strict=True):
        root=ROOT/f'.generated/go2_{study}_navigation_development_v1_attempt_001'
        for leaf in ('launch.json','result.json','raw_artifact_audit.json'):
            name=str((root/leaf).relative_to(ROOT))
            if name not in inputs or digest(ROOT/name)!=inputs[name]: raise ValueError('bound completed study required')
        report=json.loads((root/'result.json').read_text()); audit=json.loads((root/'raw_artifact_audit.json').read_text())
        if report['status']!='COMPLETE' or audit['status']!='PASS' or len(report['trials'])!=2:
            raise ValueError('complete audited predecessor required')
        if sum(t['rgb_packets'] for t in report['trials'])!=total: raise ValueError('fixed population changed')
        for row in report['trials']:
            scene=f"go2-{study.replace('_','-')}-navigation-development-v1-{row['layout_name']}"
            if row['layout_name'] not in ('north_dogleg','south_branch') or row['scene_id']!=scene:
                raise ValueError('exact development trial required')
            directory=root/scene
            leaves=['policy_observations.json','policy_histories.npz','relative_state_observations.json',
                'camera_audit.json','physics_trace.npz',*[f'rgb_{i:04d}.png' for i in range(row['rgb_packets'])]]
            for leaf in leaves:
                artifacts[str((directory/leaf).relative_to(ROOT))]=row['artifact_sha256'][leaf]
            population.append({'study':study,'directory':str(directory.relative_to(ROOT)),
                'scene_id':scene,'frames':row['rgb_packets']})
    verify(artifacts)
    return {'schema':SCHEMA,'source_sha256':closure(sources),'input_sha256':inputs,
        'artifact_sha256':artifacts,'gait_identity_only_sha256':gait,
        'native_identity_only_sha256':native,'population':population,'assumptions':ASSUMPTIONS,
        'scope':'offline previously audited causal-observation kernel replay; no counterfactual navigation'}


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


def main():
    if sys.argv[1:] not in ([],['--preflight']): raise ValueError('fixed study only')
    launch=preflight()
    if sys.argv[1:]:
        print(json.dumps({'status':'PASS','sources':len(launch['source_sha256']),
            'artifacts':len(launch['artifact_sha256']),'trajectories':len(launch['population']),
            'frames':sum(p['frames'] for p in launch['population'])}),flush=True)
        return
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT/'launch.json',launch); results=[]
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
        verify(launch['source_sha256']|launch['input_sha256']|launch['artifact_sha256'])
        write_json(OUTPUT/'result.json',{'status':'COMPLETE','launch_sha256':digest(OUTPUT/'launch.json'),
            'trials':results,'passes_declared_replay_checks':all(r['passes_declared_replay_checks'] for r in results),
            'scope':launch['scope'],'navigation_qualified':False,'hardware_qualified':False})
        print(json.dumps({'status':'COMPLETE','trials':len(results)}),flush=True)
    except Exception as error:
        write_json(OUTPUT/'terminal_failure.json',{'status':'FAIL','error':repr(error),'completed':results})
        raise


if __name__=='__main__': main()
