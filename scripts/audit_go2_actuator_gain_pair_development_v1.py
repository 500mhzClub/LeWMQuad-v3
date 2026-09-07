#!/usr/bin/env python3
"""Raw physical, actuator, causal-decision and pairing audit for gain-pair V1."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'lewm_genesis', ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.actuator_gain_development import GAIN_ARMS
from lewm.local_execution_controller_development import continuation_geometry, evaluate_edge
from lewm.physical_execution_development import KINDS, WIDTHS, rotation_xyzw
from scripts.audit_go2_local_control_factorial_development_v1 import (
    audit_decisions, check, crossing_for, digest, read_npz, recompute_contact_flags,
)
from scripts.run_go2_actuator_gain_pair_development_v1 import gain_spec


def audit_identity(identity, terminal, arm):
    check(identity['arm']==arm and len(identity['dof_indices_rollout_order'])==12
        and len(set(identity['dof_indices_rollout_order']))==12, 'gain arm/joint identity')
    check(identity['before']=={'kp':[100.]*12,'kv':[10.]*12}, 'wrong native initial gains')
    check(identity['expected_checkpoint']=={'kp':20.,'kv':.5}, 'wrong checkpoint gains')
    expected = {'kp':[20.]*12,'kv':[.5]*12} if arm=='checkpoint' else identity['before']
    check(identity['effective']==terminal==expected, 'effective/terminal gains changed')
    initial = identity['initial_state_before_intervention']
    check(set(initial)=={'position','quaternion_wxyz','velocity','angular_velocity','joint_position','joint_velocity'}, 'initial state fields')
    check(all(np.isfinite(value).all() for value in initial.values()), 'nonfinite initial state')
    return initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output = parser.parse_args().output_dir.absolute()
    check(not any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in output.parts), 'protected path')
    target = output/'raw_artifact_audit.json'
    check(not target.exists(), 'audit exists; no overwrite')
    launch, report = [json.loads((output/name).read_text()) for name in ('launch.json','result.json')]
    specs = [gain_spec(kind,width,arm) for kind in KINDS for width in WIDTHS for arm in GAIN_ARMS]
    check(launch['trial_specs']==specs, 'fixed trial population changed')
    check(report['status']=='COMPLETE' and report['completed_trials']==report['planned_trials']==16, 'incomplete study')
    check(digest(output/'launch.json')==report['launch_sha256'], 'launch binding mismatch')
    for name,expected in (launch['source_sha256'] | launch['gait_sha256']).items():
        path=Path(name)
        check(not path.is_absolute() and '..' not in path.parts and not any(
            p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in path.parts), 'invalid source path')
        check(digest(ROOT/path)==expected, f'source binding changed: {name}')
    initials, rows = {}, []
    for spec, supplied in zip(specs,report['trials'],strict=True):
        directory=output/spec['scene_id']
        check(supplied['scene_id']==spec['scene_id'] and supplied['gain_arm']==spec['gain_arm']
            and supplied['arm']=='baseline', 'trial identity')
        check(json.loads((directory/'result.json').read_text())==supplied, 'trial/report mismatch')
        required={'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','decisions.json',
            'actuator_identity.json','terminal_actuator_gains.json','process.log','final_rgb.png'}
        allowed=required | {'initial_rgb.png','edge0_rgb.png','edge1_rgb.png'}
        check(required <= set(supplied['artifact_sha256']) <= allowed, 'artifact set')
        for leaf,expected in supplied['artifact_sha256'].items():
            check(digest(directory/leaf)==expected, f'artifact binding: {leaf}')
        identity=json.loads((directory/'actuator_identity.json').read_text())
        initial=audit_identity(identity,json.loads((directory/'terminal_actuator_gains.json').read_text()),spec['gain_arm'])
        if spec['gain_arm']=='default':
            initials[spec['case_index']]=initial
        else:
            check(initial==initials[spec['case_index']], 'pre-intervention pairing mismatch')
        arrays=read_npz(directory/'physics_trace.npz')
        times=arrays['timestamp_s']
        check(len(times)==supplied['physics_samples'] and len(times)>0, 'sample count')
        check(all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite trace')
        check(np.allclose(times,.002*np.arange(1,len(times)+1),atol=1e-10,rtol=0), 'global clock reset/gap')
        flags,first=recompute_contact_flags(read_npz(directory/'native_contacts.npz'),
            json.loads((directory/'contact_topology.json').read_text()),times)
        check(np.array_equal(flags,arrays['physics_contact'].astype(bool)), 'native reference disagrees')
        if first is not None:
            check(first['sample_index']==len(times)-1 and supplied['edges'][-1]['stop_reason']=='DISALLOWED_CONTACT', 'contact stop delay')
            check(supplied['first_disallowed_contact']['sample_index']==first['sample_index'], 'first contact index')
        else:
            check(supplied['first_disallowed_contact'] is None, 'invented contact event')
        command=arrays['applied_command']
        check(np.all(np.abs(command)<=np.array([.3,0,.5])+1e-7), 'command limits')
        check(np.all(np.abs(np.diff(command,axis=0))<=np.array([.25,0,.35])+1e-7), 'command slew')
        check(np.all(np.diff(arrays['edge_index'].astype(int))>=0), 'edge order')
        settle=arrays['phase']==0
        expected_settle=min(len(times),750)
        check(np.count_nonzero(settle)==expected_settle and np.all(settle[:expected_settle]), 'settling/reset contract')
        check(len(supplied['edges'])==(1 if supplied['edges'][0]['stop_reason'] else 2), 'skipped continuation')
        for i,edge in enumerate(supplied['edges']):
            geometry=spec['geometry'] if i==0 else continuation_geometry(spec['geometry'])
            check(edge['geometry']==geometry, 'geometry changed')
            mask=arrays['edge_index']==i
            sub={key:value[mask] for key,value in arrays.items()}
            check(set(sub['phase'])<={0,1,2} and np.all(np.diff(sub['phase'].astype(int))>=0), 'phase order')
            check(edge['terminal_global_sample_index']==int(np.flatnonzero(mask)[-1]), 'edge terminal index')
            reduced=evaluate_edge(spec | {'geometry':geometry},sub,stop_reason=edge['stop_reason'],crossing=crossing_for(sub,geometry))
            check(all(edge[key]==value for key,value in reduced.items()), 'endpoint reduction mismatch')
            if edge['stop_reason']=='BODY_STABILITY_LIMIT':
                pose=sub['base_pose_world'][-1]
                rotation=rotation_xyzw(pose[3:])
                roll=np.arctan2(rotation[2,1],rotation[2,2])
                pitch=np.arcsin(np.clip(-rotation[2,0],-1,1))
                check(pose[2]<.15 or max(abs(roll),abs(pitch))>.70, 'unsubstantiated stability stop')
        decisions=json.loads((directory/'decisions.json').read_text())
        audit_decisions(arrays,decisions,spec,supplied['edges'])
        from PIL import Image
        import hashlib
        for name,metadata in supplied['images'].items():
            check(name in ('initial','edge0','edge1','final'), 'unknown RGB role')
            pixels=np.asarray(Image.open(directory/f'{name}_rgb.png'))
            check(pixels.shape==(480,640,3) and pixels.dtype==np.uint8, 'RGB encoding')
            check(hashlib.sha256(pixels.tobytes()).hexdigest()==metadata['rgb_sha256'], 'RGB binding')
            transform=np.asarray(metadata['world_from_optical'])
            check(transform.shape==(4,4) and np.allclose(transform[3],[0,0,0,1]), 'optical frame shape')
            check(np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-10,rtol=0)
                and abs(np.linalg.det(transform[:3,:3])-1)<1e-10, 'improper optical frame')
            check(np.any(np.isclose(times,metadata['timestamp_s'],atol=1e-12,rtol=0)), 'RGB time outside trajectory')
        two=len(supplied['edges'])==2
        crossings=two and all(e['checks']['sustained_correct_crossing'] and e['checks']['no_disallowed_contact'] for e in supplied['edges'])
        success=crossings and supplied['edges'][-1]['status']=='SUCCESS'
        check(supplied['two_contact_free_crossings']==crossings and (supplied['status']=='SUCCESS')==success, 'task reduction')
        check(supplied['two_usable_arrivals']==(two and all(e['status']=='SUCCESS' for e in supplied['edges'])), 'two arrival reduction')
        rows.append({'scene_id':spec['scene_id'],'gain_arm':spec['gain_arm'],'task_success':success,
            'two_contact_free_crossings':crossings,'first_native_disallowed_contact':first,
            'sustained_final_arrival':supplied['edges'][-1]['sustained_arrival_window'],'decisions_audited':len(decisions)})
    for arm in GAIN_ARMS:
        selected=[r for r in rows if r['gain_arm']==arm]
        check(report['by_gain_arm'][arm]=={'trials':8,'task_successes':sum(r['task_success'] for r in selected),
            'two_crossings':sum(r['two_contact_free_crossings'] for r in selected)}, 'arm totals')
    audit={'status':'PASS','audited_trials':16,'exact_initial_state_pairing':True,'trials':rows,
        'study_result_sha256':digest(output/'result.json'),'audit_source_sha256':digest(Path(__file__)),
        'helper_source_sha256':{name:digest(ROOT/name) for name in (
            'scripts/audit_go2_local_control_factorial_development_v1.py',
            'scripts/audit_go2_contact_attributed_execution_development_v1.py')},
        'scope':'development raw evidence audit; no novel-maze, full plant parity or real-platform claim'}
    with target.open('x') as stream:
        json.dump(audit,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in audit.items() if key!='trials'},indent=2))


if __name__=='__main__':
    main()
