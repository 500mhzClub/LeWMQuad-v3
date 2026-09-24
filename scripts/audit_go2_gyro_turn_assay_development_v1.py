#!/usr/bin/env python3
"""Replay gyro/timed decisions and audit complete raw actuator/sensor turn evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.gyro_turn_assay_development import timed_decision,reduce_turn
from lewm.relative_gyro_turn_development import RelativeGyroTurn
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.simulated_body_observation_development import SCHEMAS
from scripts.run_go2_gyro_turn_assay_development_v1 import trials,digest,write_json
from scripts.audit_go2_local_control_factorial_development_v1 import check,read_npz,recompute_contact_flags
from scripts.audit_go2_causal_rgb_body_capture_development_v1 import reconstruct_sensors,expected_history
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import array_binding


def audit_trial(directory,spec,row):
    check(row['scene_id']==spec['scene_id'] and row['method']==spec['method'] and row['case_index']==spec['case_index']
        and row['target_yaw_rad']==spec['target_yaw_rad'],'turn identity')
    required={'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
        'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json',
        'turn_decisions.json','command_tape.json','process.log'}
    check(1<=row['rgb_packets']<=126,'turn image budget')
    required|={f'rgb_{i:04d}.png' for i in range(row['rgb_packets'])}
    check(set(row['artifact_sha256'])==required,'turn artifact population')
    for name,expected in row['artifact_sha256'].items(): check(digest(directory/name)==expected,f'artifact binding {name}')
    gains=json.loads((directory/'actuator_identity.json').read_text())
    terminal_gains=json.loads((directory/'terminal_actuator_gains.json').read_text())
    check(gains['effective']==terminal_gains=={'kp':[20.]*12,'kv':[.5]*12},'effective gains')
    raw=read_npz(directory/'physics_trace.npz'); count=len(raw['timestamp_s']); start=row['prefix_terminal_sample_index']
    check(count==row['physics_samples'] and 0<count<=7000 and start==min(count,750)-1,'physics population/settling')
    check(all(np.isfinite(v).all() for v in raw.values()),'finite raw trace')
    check(np.allclose(raw['timestamp_s'],.002*np.arange(1,count+1),rtol=0,atol=1e-10),'physics clock')
    check(np.all(raw['phase'][:start+1]==0) and np.all(raw['requested_command'][:start+1]==0)
        and np.all(raw['applied_command'][:start+1]==0),'settling commands')
    check(array_binding({k:v[:start+1] for k,v in raw.items()})==row['prefix_binding']['physics_arrays'],'settling raw binding')
    histories=read_npz(directory/'policy_histories.npz'); packet_index=row['branch_start_observation_index']
    check(array_binding({k:v[packet_index] for k,v in histories.items()})==row['prefix_binding']['history_arrays'],'settling histories')
    topology=json.loads((directory/'contact_topology.json').read_text())
    check(set(topology['environment_object_ids'].values())=={'ground_plane'}|{w['wall_id'] for w in spec['geometry']['wall_boxes']},'arena identity')
    flags,first=recompute_contact_flags(read_npz(directory/'native_contacts.npz'),topology,raw['timestamp_s'])
    check(np.array_equal(flags,raw['physics_contact'].astype(bool)),'native contact flags')
    if first is not None: check(first['sample_index']==count-1 and row['response']['stop_reason']=='DISALLOWED_CONTACT','first-contact stop')
    decisions=json.loads((directory/'turn_decisions.json').read_text()); tape=json.loads((directory/'command_tape.json').read_text())
    controller=RelativeGyroTurn() if spec['method']=='gyro' else None
    active=[]; terminal=None
    for tick,decision in enumerate(decisions):
        check(tick<=120 and decision['tick']==tick and decision['pre_sample_index']==start+50*tick,'decision sequence')
        packet=load_route_observation(directory,decision['observation_index'])
        check(packet['sensor_state']['decision_ns']==decision['decision_ns']==int(round(raw['timestamp_s'][decision['pre_sample_index']]*1e9)),'decision timestamp')
        if controller is None: expected=timed_decision(tick,spec['target_yaw_rad'])
        elif tick==0: expected=controller.begin(packet,spec['target_yaw_rad'])
        else: expected=controller.step(packet)
        check(expected==decision['controller'],'actual packet controller replay')
        done=expected['status'].startswith(('COMPLETE','FAILED_'))
        check(decision['executed']==(not done),'controller execution marker')
        if done:
            check(tick==len(decisions)-1,'commands after controller terminal'); terminal=expected['status']
        else: active.append({'phase':1,'pre_sample_index':decision['pre_sample_index'],'requested_command':expected['requested_command']})
    check(tape[:len(active)]==active,'gyro/timed decision command tape')
    release=tape[len(active):]
    check(len(release)<=5 and all(e['phase']==2 and e['requested_command']==[0.,0.,0.] for e in release),'release tape')
    check(not release or terminal is not None,'release before terminal')
    check(terminal==row['response']['controller_terminal'],'controller terminal summary')
    previous=start
    for entry in tape:
        check(entry['pre_sample_index']==previous,'command order')
        size=min(50,count-previous-1); check(size>0,'empty executed command')
        sl=slice(previous+1,previous+1+size); request=entry['requested_command']
        check(np.all(raw['phase'][sl]==entry['phase']) and np.array_equal(raw['requested_command'][sl],np.tile(request,(size,1))),'requested command/phase')
        applied=raw['applied_command'][previous]+np.clip(np.array(request,dtype=np.float32)-raw['applied_command'][previous],[-.25,0,-.35],[.25,0,.35])
        check(np.allclose(raw['applied_command'][sl],applied,rtol=0,atol=1e-7),'applied slew')
        previous+=size
    check(previous==count-1,'unexplained physics after tape')
    if row['response']['stop_reason'] is None: check(terminal is not None and len(release)==5,'incomplete nonstopped trial')
    check(reduce_turn(raw,start,decisions,spec['target_yaw_rad'],terminal,row['response']['stop_reason'])==row['response'],'physical endpoint reduction')
    sensors=reconstruct_sensors(raw); recorded=read_npz(directory/'ideal_sensor_samples.npz')
    check(set(recorded)==set(sensors) and len(sensors['measured_ns'])==row['sensor_samples'],'sensor population')
    for key,value in sensors.items(): check(recorded[key].shape==value.shape and np.allclose(recorded[key],value,rtol=0,atol=1e-10),'causal sensor reconstruction')
    cameras=json.loads((directory/'camera_audit.json').read_text()); check(len(cameras)==row['rgb_packets'],'camera population')
    expected_times=sorted({float(raw['timestamp_s'][start]),float(raw['timestamp_s'][-1])}
        | {float(raw['timestamp_s'][e['pre_sample_index']]) for e in tape}
        | {float(raw['timestamp_s'][d['pre_sample_index']]) for d in decisions})
    check([c['timestamp_s'] for c in cameras]==expected_times,'command/terminal camera coverage')
    for i,camera in enumerate(cameras):
        packet=load_route_observation(directory,i); ns=packet['image']['measured_ns']
        check(ns==int(round(camera['timestamp_s']*1e9)) and raw['timestamp_s'][camera['physical_sample_index']]==camera['timestamp_s'],'image clock')
        check(hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()==camera['rgb_sha256'],'RGB binding')
        rotation=np.asarray(camera['world_from_optical'])[:3,:3]
        check(np.allclose(rotation.T@rotation,np.eye(3),atol=1e-10,rtol=0) and abs(np.linalg.det(rotation)-1)<1e-10,'proper optical frame')
        for schema in SCHEMAS:
            expected=expected_history(raw,sensors,schema,ns); actual=packet['sensor_state'][schema.role][schema.name]
            check(all(np.allclose(actual[k],v,rtol=0,atol=1e-10) for k,v in expected.items()),'causal history')
    return {'scene_id':spec['scene_id'],'method':spec['method'],'status':'PASS','rgb_packets':len(cameras),
        'physical_task_success':row['response']['physical_task_success'],'gyro_reference_agreement':row['response']['gyro_reference_agreement']}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    check(output==ROOT/'.generated/go2_gyro_turn_assay_development_v1_attempt_001','exact audit root')
    check(not (output/'raw_artifact_audit.json').exists(),'audit exists')
    launch=json.loads((output/'launch.json').read_text()); result=json.loads((output/'result.json').read_text())
    check(result['status']=='COMPLETE' and result['completed_trials']==result['planned_trials']==18,'assay incomplete')
    check(digest(output/'launch.json')==result['launch_sha256'] and launch['trial_specs']==trials(),'launch identity')
    for name,expected in (launch['source_sha256'] | launch['gait_sha256']).items():
        p=Path(name); check(not p.is_absolute() and '..' not in p.parts and not any(s in ('sealed','sealed_test.json') or s.startswith('sealed_') for s in p.parts),'source path')
        check(digest(ROOT/p)==expected,f'source/gait binding {name}')
    rows=[]; prefixes={}
    for i,(spec,row) in enumerate(zip(trials(),result['trials'],strict=True)):
        directory=output/spec['scene_id']; check(json.loads((directory/'result.json').read_text())==row,'root/trial identity')
        rows.append(audit_trial(directory,spec,row))
        reference=prefixes.setdefault(spec['case_index'],row['prefix_binding'])
        for key in ('physics_arrays','history_arrays','physics_samples','timestamp_ns'): check(reference[key]==row['prefix_binding'][key],'paired settling identity')
        print(json.dumps({'event':'turn_audited','completed':i+1,'total':18}),flush=True)
    audit={'status':'PASS','audited_trials':18,'audited_rgb_packets':sum(r['rgb_packets'] for r in rows),'trials':rows,
        'study_result_sha256':digest(output/'result.json'),'audit_source_sha256':digest(Path(__file__)),
        'scope':'raw gyro/decision/physics assay audit; ideal sensors and arena only'}
    write_json(output/'raw_artifact_audit.json',audit); print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
