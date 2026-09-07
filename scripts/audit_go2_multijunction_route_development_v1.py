#!/usr/bin/env python3
"""Raw route, action-causality and RGB/body acquisition audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.local_execution_controller_development import evaluate_edge
from lewm.multijunction_routes_development import MOTIFS,WIDTHS,route_spec
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.simulated_body_observation_development import SCHEMAS
from scripts.audit_go2_local_control_factorial_development_v1 import (
    audit_decisions,check,crossing_for,digest,read_npz,recompute_contact_flags,
)
from scripts.audit_go2_causal_rgb_body_capture_development_v1 import reconstruct_sensors,expected_history


def audit_route(directory,spec,supplied):
    check(json.loads((directory/'result.json').read_text())==supplied,'route/report mismatch')
    required={'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
        'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json',
        'terminal_actuator_gains.json','decisions.json','process.log'}
    check(1<=supplied['rgb_packets']<=341,'route frame budget')
    check(set(supplied['artifact_sha256'])==required | {f'rgb_{i:04d}.png' for i in range(supplied['rgb_packets'])},'artifact population')
    for leaf,value in supplied['artifact_sha256'].items(): check(digest(directory/leaf)==value,f'artifact binding {leaf}')
    actuator=json.loads((directory/'actuator_identity.json').read_text())
    check(actuator['arm']=='checkpoint' and actuator['before']=={'kp':[100.]*12,'kv':[10.]*12},'initial actuator identity')
    check(actuator['effective']==json.loads((directory/'terminal_actuator_gains.json').read_text())=={'kp':[20.]*12,'kv':[.5]*12},'effective gains')
    raw=read_npz(directory/'physics_trace.npz')
    times=raw['timestamp_s']
    check(len(times)==supplied['physics_samples'] and len(times)>0,'physics population')
    check(all(np.isfinite(v).all() for v in raw.values()),'nonfinite raw state')
    check(np.allclose(times,.002*np.arange(1,len(times)+1),atol=1e-10,rtol=0),'global clock reset/gap')
    topology=json.loads((directory/'contact_topology.json').read_text())
    check(set(topology['environment_object_ids'].values())=={'ground_plane'} | {w['wall_id'] for w in spec['geometry']['wall_boxes']},'physical wall identity coverage')
    flags,first=recompute_contact_flags(read_npz(directory/'native_contacts.npz'),topology,times)
    check(np.array_equal(flags,raw['physics_contact'].astype(bool)),'native force flag mismatch')
    if first is not None:
        check(first['sample_index']==len(times)-1 and supplied['route_stop']=='DISALLOWED_CONTACT','contact stop delay')
        check(supplied['first_disallowed_contact']['sample_index']==first['sample_index'],'reported contact index')
    else: check(supplied['first_disallowed_contact'] is None,'invented contact')
    check(np.all(np.diff(raw['edge_index'].astype(int))>=0),'edge order')
    settle=raw['phase']==0
    check(np.count_nonzero(settle)==min(len(times),750) and np.all(settle[:min(len(times),750)]),'settling/reset contract')
    check(1<=len(supplied['edges'])<=len(spec['route_geometries']),'route edge population')
    for i,edge in enumerate(supplied['edges']):
        check(edge['edge_index']==i and edge['geometry']==spec['route_geometries'][i],'route geometry identity')
        mask=raw['edge_index']==i
        sub={key:value[mask] for key,value in raw.items()}
        check(set(sub['phase'])<={0,1,2} and np.all(np.diff(sub['phase'].astype(int))>=0),'phase order')
        check(edge['terminal_global_sample_index']==int(np.flatnonzero(mask)[-1]),'edge terminal index')
        reduced=evaluate_edge(spec | {'geometry':edge['geometry']},sub,stop_reason=edge['stop_reason'],crossing=crossing_for(sub,edge['geometry']))
        check(all(edge[key]==value for key,value in reduced.items()),'edge reduction mismatch')
        if i<len(supplied['edges'])-1:
            check(edge['stop_reason'] is None and edge['checks']['sustained_correct_crossing'],'continued after route stop')
    last=supplied['edges'][-1]
    expected_stop=last['stop_reason'] or ('EDGE_NOT_CROSSED' if not last['checks']['sustained_correct_crossing'] else None)
    check(supplied['route_stop']==expected_stop,'route stop classification')
    check(expected_stop is not None or len(supplied['edges'])==len(spec['route_geometries']),'unexplained early route termination')
    decisions=json.loads((directory/'decisions.json').read_text())
    audit_decisions(raw,decisions,spec,supplied['edges'])
    sensors=reconstruct_sensors(raw)
    native=read_npz(directory/'ideal_sensor_samples.npz')
    check(set(native)==set(sensors) and len(sensors['measured_ns'])==supplied['sensor_samples'],'sensor population')
    for key,value in sensors.items():
        check(native[key].shape==value.shape and np.allclose(native[key],value,atol=1e-10,rtol=0),f'sensor reconstruction {key}')
    camera=json.loads((directory/'camera_audit.json').read_text())
    check(len(camera)==supplied['rgb_packets'],'RGB population')
    # Each command's pre-state and the terminal state must be captured exactly once.
    required_times=sorted({d['timestamp_s'] for d in decisions if d['requested_command'] is not None} | {float(times[-1])}
        | {float(times[e['terminal_global_sample_index']]) for e in supplied['edges']})
    check([r['timestamp_s'] for r in camera]==required_times,'command/terminal RGB coverage')
    for i,metadata in enumerate(camera):
        packet=load_route_observation(directory,i)
        ns=packet['image']['measured_ns']
        check(metadata['rgb_file']==f'rgb_{i:04d}.png' and ns==int(round(metadata['timestamp_s']*1e9)),'RGB identity/time')
        check(hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()==metadata['rgb_sha256'],'RGB pixel binding')
        index=metadata['physical_sample_index']
        check(0<=index<len(times) and int(round(times[index]*1e9))==ns,'RGB physical boundary')
        matrix=np.asarray(metadata['world_from_optical'])
        check(matrix.shape==(4,4) and np.allclose(matrix[3],[0,0,0,1]),'camera transform shape')
        check(np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-10,rtol=0)
            and abs(np.linalg.det(matrix[:3,:3])-1)<1e-10,'improper camera transform')
        for schema in SCHEMAS:
            expected=expected_history(raw,sensors,schema,ns)
            actual=packet['sensor_state'][schema.role][schema.name]
            check(all(np.allclose(actual[key],value,atol=1e-10,rtol=0) for key,value in expected.items()),'causal history mismatch')
    for edge in supplied['edges']:
        if 'terminal_observation_index' in edge:
            check(camera[edge['terminal_observation_index']]['physical_sample_index']==edge['terminal_global_sample_index'],'terminal observation binding')
    all_crossed=len(supplied['edges'])==len(spec['route_geometries']) and all(e['checks']['sustained_correct_crossing'] and e['checks']['no_disallowed_contact'] for e in supplied['edges'])
    check(supplied['all_contact_free_crossings']==all_crossed,'route crossing reduction')
    check(supplied['all_usable_arrivals']==(all_crossed and all(e['status']=='SUCCESS' for e in supplied['edges'])),'arrival reduction')
    success=all_crossed and last['status']=='SUCCESS'
    check((supplied['status']=='SUCCESS')==success,'task reduction')
    return {'scene_id':spec['scene_id'],'task_success':success,'all_crossings':all_crossed,
        'first_native_disallowed_contact':first,'audited_rgb_packets':len(camera),'audited_sensor_samples':len(sensors['measured_ns'])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    check(not any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in output.parts),'protected path')
    target=output/'raw_artifact_audit.json'
    check(not target.exists(),'audit exists; no overwrite')
    launch,report=[json.loads((output/name).read_text()) for name in ('launch.json','result.json')]
    specs=[route_spec(m,w) for m in MOTIFS for w in WIDTHS]
    check(launch['trial_specs']==specs,'fixed population changed')
    check(report['status']=='COMPLETE' and report['completed_trials']==report['planned_trials']==8,'incomplete study')
    check(digest(output/'launch.json')==report['launch_sha256'],'launch binding')
    for name,expected in (launch['source_sha256'] | launch['gait_sha256']).items():
        path=Path(name)
        check(not path.is_absolute() and '..' not in path.parts and not any(
            p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in path.parts),'source path')
        check(digest(ROOT/path)==expected,f'source/gait binding {name}')
    rows=[]
    for spec,supplied in zip(specs,report['trials'],strict=True):
        check(supplied['scene_id']==spec['scene_id'],'trial identity')
        rows.append(audit_route(output/spec['scene_id'],spec,supplied))
    check(sum(r['task_success'] for r in rows)==report['task_successes'],'success total')
    result={'status':'PASS','audited_trials':8,'task_successes':report['task_successes'],
        'audited_rgb_packets':sum(r['audited_rgb_packets'] for r in rows),'trials':rows,
        'study_result_sha256':digest(output/'result.json'),'audit_source_sha256':digest(Path(__file__)),
        'helper_source_sha256':{name:digest(ROOT/name) for name in ('lewm/route_rgb_dataset_development.py',
            'lewm/causal_rgb_dataset_development.py','scripts/audit_go2_local_control_factorial_development_v1.py',
            'scripts/audit_go2_contact_attributed_execution_development_v1.py','scripts/audit_go2_causal_rgb_body_capture_development_v1.py')},
        'scope':'oracle-route and causal-observation development audit; no visual/JEPA navigation qualification'}
    with target.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))


if __name__=='__main__': main()
