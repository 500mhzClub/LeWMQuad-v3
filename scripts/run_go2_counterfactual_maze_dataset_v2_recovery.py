#!/usr/bin/env python3
"""Explicit composite recovery: verify 19 retained branches, execute only 101 untouched pairs."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.counterfactual_maze_development import ACTIONS,branch_spec,corpus
from lewm.counterfactual_prefix_matching_development import compare_prefix
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import collect

ORIGINAL=ROOT/'.generated/go2_counterfactual_maze_dataset_development_v1_attempt_001'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifacts(directory,row):
    leaves={'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
        'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json',
        'prefix_decisions.json','branch_tape.json','outcome_labels.json','process.log'}
    if not 1<=row['rgb_packets']<=131: raise ValueError('branch frame budget')
    leaves|={f'rgb_{i:04d}.png' for i in range(row['rgb_packets'])}
    if set(row['artifact_sha256'])!=leaves: raise ValueError('branch artifact population')
    for name,expected in row['artifact_sha256'].items():
        if sha(directory/name)!=expected: raise ValueError(f'branch artifact drift: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if output.parent!=ROOT/'.generated' or any(p=='sealed' or p.startswith('sealed_') for p in output.parts):
        parser.error('recovery output must be a new explicit .generated child')
    if shutil.disk_usage(output.parent).free<10*1024**3: parser.error('less than 10 GiB free')
    original_launch=json.loads((ORIGINAL/'launch.json').read_text())
    original_result=json.loads((ORIGINAL/'result.json').read_text())
    specs=[branch_spec(layout,i) for layout in corpus() for i in range(len(ACTIONS))]
    if original_launch['trial_specs']!=specs: raise ValueError('original population mismatch')
    if original_result['status']!='INFRASTRUCTURE_FAILURE' or original_result['completed_trials']!=19 or len(original_result['trials'])!=19:
        raise ValueError('unexpected predecessor terminal state')
    if original_result['launch_sha256']!=sha(ORIGINAL/'launch.json'): raise ValueError('original launch binding')
    bindings=original_launch['source_sha256'] | original_launch['gait_sha256']
    for name,expected in bindings.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in path.parts):
            raise ValueError('invalid original bound path')
        if sha(ROOT/path)!=expected: raise ValueError(f'original source/gait drift: {name}')
    for spec in specs[19:]:
        if (ORIGINAL/spec['scene_id']).exists(): raise ValueError('unreported predecessor directory; no duplicate execution')
    adopted=original_result['trials']
    for spec,row in zip(specs[:19],adopted,strict=True):
        if row['scene_id']!=spec['scene_id'] or json.loads((ORIGINAL/row['scene_id']/'result.json').read_text())!=row:
            raise ValueError('original branch identity/report mismatch')
        validate_artifacts(ORIGINAL/row['scene_id'],row)
    output.mkdir(exist_ok=False)
    extra=('scripts/run_go2_counterfactual_maze_dataset_v2_recovery.py','lewm/counterfactual_prefix_matching_development.py',
        'docs/go2_counterfactual_maze_dataset_development_v2_recovery_2026-09-05.md')
    launch={'schema':'counterfactual_maze_development_v2_recovery','trial_specs':specs,
        'source_sha256':original_launch['source_sha256'] | {name:sha(ROOT/name) for name in extra},
        'gait_sha256':original_launch['gait_sha256'],'versions':original_launch['versions'],
        'roots':{'original':str(ORIGINAL.relative_to(ROOT)),'recovery':str(output.relative_to(ROOT))},
        'original_launch_sha256':sha(ORIGINAL/'launch.json'),'original_result_sha256':sha(ORIGINAL/'result.json'),
        'adopted_trials':19,'new_trials':101,'canonical_context':'first stop branch packet per layout',
        'scope':'explicit development recovery; original terminal retained; no repeated physical trials'}
    (output/'launch.json').write_text(json.dumps(launch,indent=2,allow_nan=False)+'\n')
    rows,references,status=[],{},'COMPLETE'
    def append(row,root_name):
        directory=(ORIGINAL if root_name=='original' else output)/row['scene_id']
        reference=references.setdefault(row['layout_id'],(directory,row))
        match=compare_prefix(reference[0],reference[1],directory,row)
        rows.append({'member_root':root_name,'result_sha256':sha(directory/'result.json'),'prefix_match':match,'result':row})
    try:
        for row in adopted: append(row,'original')
        print(json.dumps({'event':'retained_members_verified','completed':len(rows),'total':120}),flush=True)
        for spec in specs[19:]:
            directory=output/spec['scene_id']
            directory.mkdir(exist_ok=False)
            print(json.dumps({'event':'new_branch_started','trial':spec['scene_id'],'role':spec['data_role'],
                'completed':len(rows),'total':120}),flush=True)
            with (directory/'process.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                row=collect(spec,directory)
            leaves=['physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
                'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json',
                'prefix_decisions.json','branch_tape.json','outcome_labels.json','process.log']
            leaves.extend(f'rgb_{i:04d}.png' for i in range(row['rgb_packets']))
            row['artifact_sha256']={name:sha(directory/name) for name in leaves}
            (directory/'result.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            validate_artifacts(directory,row)
            append(row,'recovery')
            print(json.dumps({'event':'new_branch_finished','trial':spec['scene_id'],'branchable':row['branchable'],
                'stop_reason':row['stop_reason'],'completed':len(rows),'total':120}),flush=True)
    except Exception as exc:
        traceback.print_exc(); status='INFRASTRUCTURE_FAILURE'
        (output/'failure.json').write_text(json.dumps({'error':f'{type(exc).__name__}: {exc}',
            'traceback':traceback.format_exc(),'verified_members':len(rows)},indent=2)+'\n')
    result={'status':status,'planned_trials':120,'verified_trials':len(rows),'members':rows,
        'branchable_trials':sum(m['result']['branchable'] for m in rows),
        'contact_stops':sum(m['result']['stop_reason']=='DISALLOWED_CONTACT' for m in rows),
        'launch_sha256':sha(output/'launch.json')}
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='members'},indent=2))
    return 0 if status=='COMPLETE' else 1


if __name__=='__main__': raise SystemExit(main())
