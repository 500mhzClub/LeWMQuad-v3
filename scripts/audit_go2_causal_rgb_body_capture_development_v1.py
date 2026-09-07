#!/usr/bin/env python3
"""Reconstruct causal simulated sensing and observations from explicit raw trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'):
    sys.path.insert(0,str(path))

from lewm.causal_rgb_dataset_development import load_policy_observation
from lewm.physical_execution_development import rotation_xyzw
from lewm.simulated_body_observation_development import SCHEMAS
from scripts.audit_go2_local_control_factorial_development_v1 import check,digest,read_npz,recompute_contact_flags
from scripts.run_go2_causal_rgb_body_capture_development_v1 import probe_spec,reduce_response


def reconstruct_sensors(raw):
    ns=np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    selected=np.flatnonzero(ns%20_000_000==0)
    rotation=np.stack([rotation_xyzw(raw['base_pose_world'][i,3:]) for i in selected])
    angular=raw['base_twist_world'][selected,3:]
    gyro=np.einsum('nji,nj->ni',rotation,angular)
    force=np.zeros_like(gyro)
    if len(selected)>1:
        acceleration=np.diff(raw['base_twist_world'][selected,:3],axis=0)/.02
        force[1:]=np.einsum('nji,nj->ni',rotation[1:],acceleration-np.array([0.,0.,-9.81]))
    valid=np.ones_like(force,dtype=bool)
    if len(valid): valid[0]=False
    return {'measured_ns':ns[selected], 'gyro_values':gyro,'gyro_valid':np.ones_like(gyro,dtype=bool),
        'specific_force_values':force,'specific_force_valid':valid,
        'joints_values':np.concatenate([raw['joint_position'][selected],raw['joint_velocity'][selected]],axis=1),
        'joints_valid':np.ones((len(selected),24),dtype=bool)}


def expected_history(raw,sensors,schema,time):
    if schema.name=='applied_command':
        times=np.rint(raw['timestamp_s']*1e9).astype(np.int64)
        mask=times%100_000_000==0
        times,values=times[mask],raw['applied_command'][mask]
        valid=np.ones_like(values,dtype=bool)
    else:
        times=sensors['measured_ns']
        values,valid=sensors[f'{schema.name}_values'],sensors[f'{schema.name}_valid']
    selected=np.flatnonzero((times<=time)&(time-times<=schema.max_age_ns))[-schema.history_length:]
    shape=(schema.history_length,len(schema.channels))
    result={'values':np.zeros(shape),'valid':np.zeros(shape,dtype=bool),
        'measured_ns':np.full(schema.history_length,-1,dtype=np.int64),'available_ns':np.full(schema.history_length,-1,dtype=np.int64)}
    if len(selected):
        result['values'][-len(selected):]=values[selected]
        result['valid'][-len(selected):]=valid[selected]
        result['measured_ns'][-len(selected):]=result['available_ns'][-len(selected):]=times[selected]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    check(not any(p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in output.parts),'protected path')
    target=output/'raw_artifact_audit.json'
    check(not target.exists(),'audit exists; no overwrite')
    launch,report=[json.loads((output/name).read_text()) for name in ('launch.json','result.json')]
    specs=[probe_spec(i) for i in range(9)]
    check(launch['trial_specs']==specs,'fixed population changed')
    check(report['status']=='COMPLETE' and report['completed_trials']==report['planned_trials']==9,'incomplete study')
    check(digest(output/'launch.json')==report['launch_sha256'],'launch binding')
    for name,expected in (launch['source_sha256'] | launch['gait_sha256']).items():
        path=Path(name)
        check(not path.is_absolute() and '..' not in path.parts and not any(
            p=='sealed' or p=='sealed_test.json' or p.startswith('sealed_') for p in path.parts),'invalid source path')
        check(digest(ROOT/path)==expected,f'source/gait binding: {name}')
    rows=[]
    for spec,supplied in zip(specs,report['trials'],strict=True):
        directory=output/spec['scene_id']
        check(supplied['scene_id']==spec['scene_id'],'trial identity')
        check(json.loads((directory/'result.json').read_text())==supplied,'trial/report mismatch')
        required={'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
            'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json','process.log'}
        images={f'rgb_{i:04d}.png' for i in range(supplied['rgb_packets'])}
        check(set(supplied['artifact_sha256'])==required | images,'artifact population')
        for leaf,expected in supplied['artifact_sha256'].items():
            check(digest(directory/leaf)==expected,f'artifact binding: {leaf}')
        identity=json.loads((directory/'actuator_identity.json').read_text())
        terminal=json.loads((directory/'terminal_actuator_gains.json').read_text())
        check(identity['before']=={'kp':[100.]*12,'kv':[10.]*12} and identity['arm']=='checkpoint','initial actuator identity')
        check(identity['effective']==terminal=={'kp':[20.]*12,'kv':[.5]*12},'effective actuator gains')
        raw=read_npz(directory/'physics_trace.npz')
        count=len(raw['timestamp_s'])
        check(count==supplied['physics_samples'] and count>0,'physics population')
        check(all(np.isfinite(v).all() for v in raw.values()),'nonfinite raw state')
        check(np.allclose(raw['timestamp_s'],.002*np.arange(1,count+1),atol=1e-10,rtol=0),'physics clock')
        flags,first=recompute_contact_flags(read_npz(directory/'native_contacts.npz'),
            json.loads((directory/'contact_topology.json').read_text()),raw['timestamp_s'])
        check(np.array_equal(flags,raw['physics_contact'].astype(bool)),'native force/flag disagreement')
        if first is not None:
            check(first['sample_index']==count-1 and supplied['response']['stop_reason']=='DISALLOWED_CONTACT','contact stop delay')
        phase=np.array([0]*750+[1]*1500+[2]*750,dtype=np.uint8)[:count]
        check(np.array_equal(raw['phase'],phase),'phase/tape alignment')
        command=np.zeros((count,3))
        command[phase==1]=spec['stimulus_command']
        check(np.array_equal(raw['requested_command'],command),'requested stimulus changed')
        check(np.allclose(raw['applied_command'],command.astype(np.float32),atol=1e-7,rtol=0),'applied stimulus changed')
        reduced=reduce_response(raw,spec['stimulus_command'],supplied['response']['stop_reason'])
        check(reduced==supplied['response'],'command response reduction')
        sensors=reconstruct_sensors(raw)
        recorded=read_npz(directory/'ideal_sensor_samples.npz')
        check(set(recorded)==set(sensors) and len(sensors['measured_ns'])==supplied['sensor_samples'],'sensor population')
        for key,expected in sensors.items():
            check(recorded[key].shape==expected.shape and np.allclose(recorded[key],expected,atol=1e-10,rtol=0),f'causal sensor reference mismatch: {key}')
        camera=json.loads((directory/'camera_audit.json').read_text())
        check(len(camera)==supplied['rgb_packets'],'camera population')
        expected_times=list(range(1_500_000_000,6_000_000_001,100_000_000)) if count==3000 else None
        if expected_times is not None:
            check([int(round(r['timestamp_s']*1e9)) for r in camera]==expected_times,'decision image timing')
        for i,metadata in enumerate(camera):
            packet=load_policy_observation(directory,i)
            time=packet['image']['measured_ns']
            check(metadata['rgb_file']==f'rgb_{i:04d}.png','camera role/path')
            check(time==int(round(metadata['timestamp_s']*1e9)),'image time binding')
            check(hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()==metadata['rgb_sha256'],'RGB pixel binding')
            sample=metadata['physical_sample_index']
            check(0<=sample<count and int(round(raw['timestamp_s'][sample]*1e9))==time,'image physical boundary')
            transform=np.asarray(metadata['world_from_optical'])
            check(transform.shape==(4,4) and np.allclose(transform[3],[0,0,0,1]),'optical transform shape')
            check(np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-10,rtol=0)
                and abs(np.linalg.det(transform[:3,:3])-1)<1e-10,'improper camera transform')
            for schema in SCHEMAS:
                expected=expected_history(raw,sensors,schema,time)
                actual=packet['sensor_state'][schema.role][schema.name]
                for field,value in expected.items():
                    check(np.allclose(actual[field],value,atol=1e-10,rtol=0),f'causal history mismatch: {i}/{schema.name}/{field}')
        rows.append({'scene_id':spec['scene_id'],'audited_rgb_packets':len(camera),'audited_sensor_samples':len(sensors['measured_ns']),
            'contact':reduced['contact'],'completed_fixed_tape':reduced['completed_fixed_tape'],
            'release_motion_window_pass':reduced['release_motion_window_pass'],
            'median_capture_wall_time_s':float(np.median([r['capture_wall_time_s'] for r in camera]))})
    result={'status':'PASS','audited_trials':9,'audited_rgb_packets':sum(r['audited_rgb_packets'] for r in rows),
        'audited_sensor_samples':sum(r['audited_sensor_samples'] for r in rows),'trials':rows,
        'study_result_sha256':digest(output/'result.json'),'audit_source_sha256':digest(Path(__file__)),
        'policy_loader_sha256':digest(ROOT/'lewm/causal_rgb_dataset_development.py'),
        'scope':'ideal simulated sensing/causal acquisition audit, not deployed sensors or navigation'}
    with target.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key!='trials'},indent=2))


if __name__=='__main__':
    main()
