#!/usr/bin/env python3
"""Audit composite counterfactual members, causal sensing, plans and censored labels."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.counterfactual_maze_development import ACTIONS,HORIZONS_NS,branch_spec,corpus
from lewm.counterfactual_prefix_matching_development import compare_prefix
from lewm.physical_execution_development import rotation_xyzw
from lewm.local_execution_controller_development import evaluate_edge
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.simulated_body_observation_development import SCHEMAS
from scripts.audit_go2_local_control_factorial_development_v1 import audit_decisions,check,crossing_for,digest,read_npz,recompute_contact_flags
from scripts.audit_go2_causal_rgb_body_capture_development_v1 import reconstruct_sensors,expected_history
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import array_binding
from scripts.run_go2_counterfactual_maze_dataset_v2_recovery import validate_artifacts,ORIGINAL


def check_horizons(raw,start,labels):
    times=np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    origin=raw['base_pose_world'][start]
    r0=rotation_xyzw(origin[3:])
    yaw0=math.atan2(r0[1,0],r0[0,0])
    check(len(labels)==8,'horizon population')
    for horizon,label in zip(HORIZONS_NS,labels,strict=True):
        check(set(label)=={'horizon_ns','motion_valid','delta_xy_yaw_start_body','contact_valid','contact_by_horizon'},'outcome fields')
        target=times[start]+horizon
        where=np.flatnonzero(times==target)
        contact=bool(raw['physics_contact'][(times>times[start])&(times<=target)].any())
        check(label['horizon_ns']==horizon and label['motion_valid']==bool(len(where)),'motion censoring')
        check(label['contact_by_horizon']==contact and label['contact_valid']==bool(times[-1]>=target or contact),'contact censoring')
        if len(where):
            pose=raw['base_pose_world'][where[0]]
            r=rotation_xyzw(pose[3:])
            delta=r0.T@(pose[:3]-origin[:3])
            yaw=math.atan2(r[1,0],r[0,0])-yaw0
            expected=[delta[0],delta[1],math.atan2(math.sin(yaw),math.cos(yaw))]
            check(np.allclose(label['delta_xy_yaw_start_body'],expected,rtol=0,atol=1e-12),'motion target')
        else: check(label['delta_xy_yaw_start_body'] is None,'imputed unobserved motion')


def audit_branch(directory,spec,row):
    validate_artifacts(directory,row)
    check(json.loads((directory/'result.json').read_text())==row,'branch result identity')
    check(row['scene_id']==spec['scene_id'] and row['layout_id']==spec['layout_id']
        and row['data_role']==spec['data_role'] and row['action_name']==spec['action_name'],'branch role/action identity')
    actuator=json.loads((directory/'actuator_identity.json').read_text())
    check(actuator['effective']==json.loads((directory/'terminal_actuator_gains.json').read_text())=={'kp':[20.]*12,'kv':[.5]*12},'actuator gains')
    raw=read_npz(directory/'physics_trace.npz')
    count=len(raw['timestamp_s'])
    check(count==row['physics_samples'] and count>0,'physics population')
    check(all(np.isfinite(v).all() for v in raw.values()),'nonfinite physical trace')
    check(np.allclose(raw['timestamp_s'],.002*np.arange(1,count+1),atol=1e-10,rtol=0),'global clock')
    topology=json.loads((directory/'contact_topology.json').read_text())
    check(set(topology['environment_object_ids'].values())=={'ground_plane'} | {w['wall_id'] for w in spec['geometry']['wall_boxes']},'physical scene identity')
    flags,first=recompute_contact_flags(read_npz(directory/'native_contacts.npz'),topology,raw['timestamp_s'])
    check(np.array_equal(flags,raw['physics_contact'].astype(bool)),'native contact flags')
    if first is not None: check(first['sample_index']==count-1 and row['stop_reason']=='DISALLOWED_CONTACT','contact stop timing')
    labels=json.loads((directory/'outcome_labels.json').read_text())
    start=row['prefix_terminal_sample_index']
    check(0<=start<count and labels['prefix_terminal_sample_index']==start,'prefix terminal index')
    prefix_raw={k:v[:start+1] for k,v in raw.items()}
    check(np.all(prefix_raw['edge_index']==0),'prefix edge identity')
    check(np.count_nonzero(prefix_raw['phase']==0)==min(start+1,750),'settling count')
    prefix=labels['prefix_result']
    reduced=evaluate_edge(spec,prefix_raw,stop_reason=prefix['stop_reason'],crossing=crossing_for(prefix_raw,spec['geometry']))
    check(all(prefix[k]==v for k,v in reduced.items()),'prefix endpoint')
    decisions=json.loads((directory/'prefix_decisions.json').read_text())
    audit_decisions(prefix_raw,decisions,spec,[prefix])
    branchable=prefix['stop_reason'] is None and prefix['checks']['sustained_correct_crossing'] and prefix['checks']['no_disallowed_contact']
    check(row['branchable']==labels['branchable']==branchable,'prefix availability')
    check(array_binding(prefix_raw)==row['prefix_binding']['physics_arrays'],'raw prefix binding')
    histories=read_npz(directory/'policy_histories.npz')
    packet_index=row['branch_start_observation_index']
    check(labels['branch_start_observation_index']==packet_index,'branch observation label')
    check(array_binding({k:v[packet_index] for k,v in histories.items()})==row['prefix_binding']['history_arrays'],'history prefix binding')
    tape=json.loads((directory/'branch_tape.json').read_text())
    previous_index=start
    if not branchable: check(not tape and start==count-1 and not labels['horizon_labels'],'branch after prefix failure')
    for tick,entry in enumerate(tape):
        check(tick<45 and entry['tick']==tick and entry['pre_sample_index']==previous_index,'branch command order')
        check(entry['timestamp_s']==raw['timestamp_s'][previous_index],'branch command time')
        requested=spec['branch_command'] if tick<40 else [0.,0.,0.]
        check(entry['requested_command']==requested,'candidate tape changed')
        size=min(50,count-previous_index-1)
        check(size>0,'empty branch command execution')
        interval=slice(previous_index+1,previous_index+size+1)
        check(np.all(raw['edge_index'][interval]==1) and np.all(raw['phase'][interval]==(1 if tick<40 else 2)),'branch execution phase')
        check(np.array_equal(raw['requested_command'][interval],np.tile(requested,(size,1))),'branch requested trace')
        applied=raw['applied_command'][previous_index]+np.clip(np.asarray(requested,dtype=np.float32)-raw['applied_command'][previous_index],[-.25,0,-.35],[.25,0,.35])
        check(np.allclose(raw['applied_command'][interval],applied,atol=1e-7,rtol=0),'candidate slew reconstruction')
        previous_index+=size
    check(previous_index==count-1,'unexplained post-prefix physics')
    if branchable and row['stop_reason'] is None: check(len(tape)==45 and count-start-1==2250,'incomplete nonstopped tape')
    if branchable: check_horizons(raw,start,labels['horizon_labels'])
    check(labels['horizon_labels']==row['horizon_labels'],'result/outcome labels')
    sensors=reconstruct_sensors(raw)
    recorded=read_npz(directory/'ideal_sensor_samples.npz')
    check(set(recorded)==set(sensors) and len(sensors['measured_ns'])==row['sensor_samples'],'sensor population')
    for key,value in sensors.items(): check(recorded[key].shape==value.shape and np.allclose(recorded[key],value,atol=1e-10,rtol=0),'sensor reconstruction')
    camera=json.loads((directory/'camera_audit.json').read_text())
    check(len(camera)==row['rgb_packets'],'image population')
    expected_times=sorted({d['timestamp_s'] for d in decisions if d['requested_command'] is not None}
        | {e['timestamp_s'] for e in tape} | {float(raw['timestamp_s'][start]),float(raw['timestamp_s'][-1])})
    check([e['timestamp_s'] for e in camera]==expected_times,'image command coverage')
    for i,metadata in enumerate(camera):
        packet=load_route_observation(directory,i)
        ns=packet['image']['measured_ns']
        check(ns==int(round(metadata['timestamp_s']*1e9)),'image timestamp')
        check(hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()==metadata['rgb_sha256'],'image pixel binding')
        check(raw['timestamp_s'][metadata['physical_sample_index']]==metadata['timestamp_s'],'image physical boundary')
        rotation=np.asarray(metadata['world_from_optical'])[:3,:3]
        check(np.allclose(rotation.T@rotation,np.eye(3),atol=1e-10,rtol=0) and abs(np.linalg.det(rotation)-1)<1e-10,'camera orientation')
        for schema in SCHEMAS:
            expected=expected_history(raw,sensors,schema,ns)
            actual=packet['sensor_state'][schema.role][schema.name]
            check(all(np.allclose(actual[k],v,atol=1e-10,rtol=0) for k,v in expected.items()),'history causality')
    check(camera[packet_index]['physical_sample_index']==start and camera[packet_index]['rgb_sha256']==row['prefix_binding']['rgb_pixels_sha256'],'branch start context')
    return {'scene_id':spec['scene_id'],'layout_id':spec['layout_id'],'data_role':spec['data_role'],
        'branchable':branchable,'contact_stop':row['stop_reason']=='DISALLOWED_CONTACT','audited_rgb_packets':len(camera)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    check(output.parent==ROOT/'.generated' and not any(p=='sealed' or p.startswith('sealed_') for p in output.parts),'audit root')
    target=output/'raw_artifact_audit.json'
    check(not target.exists(),'audit exists')
    launch,result=[json.loads((output/name).read_text()) for name in ('launch.json','result.json')]
    check(result['status']=='COMPLETE' and result['verified_trials']==result['planned_trials']==120,'corpus incomplete')
    specs=[branch_spec(layout,i) for layout in corpus() for i in range(len(ACTIONS))]
    check(launch['trial_specs']==specs,'corpus population')
    check(launch['roots']=={'original':str(ORIGINAL.relative_to(ROOT)),'recovery':str(output.relative_to(ROOT))},'corpus roots')
    check(digest(output/'launch.json')==result['launch_sha256'] and digest(ORIGINAL/'result.json')==launch['original_result_sha256'],'root result binding')
    for name,expected in (launch['source_sha256'] | launch['gait_sha256']).items():
        path=Path(name)
        check(not path.is_absolute() and '..' not in path.parts and not any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in path.parts),'bound source path')
        check(digest(ROOT/path)==expected,f'source binding {name}')
    rows,references=[],{}
    for i,(spec,member) in enumerate(zip(specs,result['members'],strict=True)):
        expected_root='original' if i<19 else 'recovery'
        check(member['member_root']==expected_root,'member source root')
        directory=(ORIGINAL if i<19 else output)/spec['scene_id']
        check(digest(directory/'result.json')==member['result_sha256'],'member result binding')
        row=member['result']
        rows.append(audit_branch(directory,spec,row))
        reference=references.setdefault(spec['layout_id'],(directory,row))
        check(compare_prefix(reference[0],reference[1],directory,row)==member['prefix_match'],'prefix match reduction')
        print(json.dumps({'event':'branch_audited','completed':i+1,'total':120,'scene_id':spec['scene_id']}),flush=True)
    helper_paths=(
        'scripts/audit_go2_local_control_factorial_development_v1.py',
        'scripts/audit_go2_causal_rgb_body_capture_development_v1.py',
        'lewm/route_rgb_dataset_development.py','lewm/causal_rgb_dataset_development.py',
        'lewm/simulated_body_observation_development.py','lewm/causal_sensor_state.py',
        'lewm/counterfactual_prefix_matching_development.py',
        'lewm/physical_execution_development.py','lewm/local_execution_controller_development.py',
    )
    audit={'status':'PASS','audited_trials':120,'audited_layouts':24,'audited_rgb_packets':sum(r['audited_rgb_packets'] for r in rows),
        'trials':rows,'study_result_sha256':digest(output/'result.json'),'audit_source_sha256':digest(Path(__file__)),
        'audit_helper_sha256':{name:digest(ROOT/name) for name in helper_paths},
        'scope':'development counterfactual corpus audit; not model performance or final maze evaluation'}
    with target.open('x') as stream:
        json.dump(audit,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='trials'},indent=2))


if __name__=='__main__': main()
