#!/usr/bin/env python3
"""Independent raw checks of fixed moving-prefix development suffixes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.counterfactual_prefix_matching_development import compare_prefix
from lewm.local_execution_controller_development import evaluate_edge
from lewm.moving_prefix_counterfactual_development import trials,evidence_cells
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.simulated_body_observation_development import SCHEMAS
from scripts.audit_go2_causal_rgb_body_capture_development_v1 import reconstruct_sensors,expected_history
from scripts.audit_go2_causal_subtrajectory_development_v1 import scalar_motion
from scripts.audit_go2_local_control_factorial_development_v1 import check,read_npz,recompute_contact_flags,audit_decisions,crossing_for
from scripts.audit_go2_successive_choice_maze_development_v1 import camera_reference
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import array_binding
from scripts.run_go2_moving_prefix_counterfactual_development_v1 import OUTPUT,digest,verify,write_json

LAUNCH_SHA='f8f1c6e01017aa08363bd8ad6997145e9df1a4fe52f115d1a88536608a4f3ae7'
AUDIT_SOURCES=(str(Path(__file__).relative_to(ROOT)),
    'lewm/tests/test_moving_prefix_raw_audit.py',
    'scripts/audit_go2_causal_rgb_body_capture_development_v1.py',
    'scripts/audit_go2_causal_subtrajectory_development_v1.py',
    'scripts/audit_go2_local_control_factorial_development_v1.py',
    'scripts/audit_go2_contact_attributed_execution_development_v1.py',
    'scripts/audit_go2_successive_choice_maze_development_v1.py')


def audit_commands(raw,row,spec,tape):
    previous=row['teacher_terminal_sample_index']; stages=[]
    wanted=['moving_prefix']*10+['suffix']*30+['release']*5
    check(len(tape)<=45,'command population')
    for tick,e in enumerate(tape):
        a,b=e['pre_sample_index'],e['post_sample_index']; stage=e['stage']
        check(e['tick']==tick and a==previous and 0<b-a<=50,'contiguous positive command execution')
        check(stage==wanted[tick] and e['timestamp_s']==raw['timestamp_s'][a],'fixed stage order and pre-dispatch clock')
        command=spec['prefix_command'] if stage=='moving_prefix' else spec['future_command'] if stage=='suffix' else [0.,0.,0.]
        check(e['requested_command']==command,'fixed requested action')
        block=slice(a+1,b+1)
        check(np.all(raw['edge_index'][block]==(1 if stage=='moving_prefix' else 2)),'command edge identity')
        check(np.all(raw['phase'][block]==(2 if stage=='release' else 1)),'command phase identity')
        check(np.array_equal(raw['requested_command'][block],np.tile(command,(b-a,1))),'actual requested tape')
        prior=raw['applied_command'][a]
        clipped=np.clip(np.asarray(command,dtype=np.float32),[-.3,0,-.5],[.3,0,.5]).astype(np.float32)
        expected=prior+np.clip(clipped-prior,[-.25,0,-.35],[.25,0,.35])
        check(np.allclose(raw['applied_command'][block],expected,atol=1e-7,rtol=0),'independent command slew')
        if b-a<50: check(tick==len(tape)-1 and row['stop_reason'] in ('DISALLOWED_CONTACT','BODY_STABILITY_LIMIT'),'interruption must be terminal')
        stages.append(stage); previous=b
    check(previous==len(raw['timestamp_s'])-1,'unexplained physical suffix')
    moving=[e for e in tape if e['stage']=='moving_prefix']
    start=moving[-1]['post_sample_index'] if moving else row['teacher_terminal_sample_index']
    check(row['prefix_terminal_sample_index']==start,'actual moving endpoint')
    eligible=(row['teacher_available'] and len(moving)==10 and start-row['teacher_terminal_sample_index']==500
        and not raw['physics_contact'][:start+1].any()
        and not (start==len(raw['timestamp_s'])-1 and row['stop_reason']=='BODY_STABILITY_LIMIT'))
    check(row['branchable']==eligible,'moving conditioning availability')
    suffix=[e for e in tape if e['stage']=='suffix']
    end=suffix[-1]['post_sample_index'] if suffix else start
    check(row['suffix_terminal_sample_index']==end,'suffix excludes release')
    if row['stop_reason'] is None: check(len(tape)==45 and eligible,'nonterminal incomplete sequence')
    if not eligible: check(not suffix and 'release' not in stages,'execution after unavailable prefix')


def audit_targets(raw,row,frames):
    window=row['suffix_window']
    if not row['branchable']:
        check(window is None,'unavailable prefix has labels'); return {'motion':0,'contact':0,'positive':0}
    start,end=row['prefix_terminal_sample_index'],row['suffix_terminal_sample_index']
    times=[round(float(t)*1e9) for t in raw['timestamp_s']]; current=times[start]
    physical={times[i]:i for i in range(start,end+1)}
    first=next((times[i] for i in range(start+1,end+1) if raw['physics_contact'][i]),None)
    lookup={f['image_ns']:i for i,f in enumerate(frames)}
    check(len(lookup)==len(frames) and all(f['image_ns']==f['decision_ns'] for f in frames),'unique image clocks')
    check(window['decision_ns']==current and window['remaining_ticks']==30,'conditioning time and known horizon')
    check(window['history_observation_indices']==[lookup[current+d] for d in (-300_000_000,-200_000_000,-100_000_000,0)],'actual past history')
    check(len(window['targets'])==8,'target count'); counts={'motion':0,'contact':0,'positive':0}
    for i,t in enumerate(window['targets']):
        horizon=(i+1)*500_000_000; target=current+horizon; inside=i<6
        event=inside and first is not None and first<=target
        mv=inside and target in physical and (first is None or target<first)
        cv=inside and (target<=times[end] or event)
        check(t['horizon_ns']==horizon and t['in_plan']==inside,'known target horizon')
        check(t['motion_valid']==mv and t['contact_valid']==cv and t['contact_by_horizon']==(bool(event) if cv else None),'independent censoring and contact')
        if mv:
            expected=scalar_motion(raw['base_pose_world'][start],raw['base_pose_world'][physical[target]])
            check(np.allclose(t['delta_xy_yaw_current_body'],expected,atol=1e-10,rtol=0),'independent quaternion-vector motion')
            check(t['future_observation_index']==lookup[target],'actual future RGB time')
        else: check(t['delta_xy_yaw_current_body'] is None and t['future_observation_index'] is None,'censored motion must be absent')
        counts['motion']+=mv; counts['contact']+=cv; counts['positive']+=bool(event)
    return counts


def audit_trial(directory,spec,row,reference):
    for key in ('scene_id','layout_id','data_role','prefix_action_index','future_action_index','reference_scene_id'):
        check(row[key]==spec[key],'trial identity '+key)
    leaves={'physics_trace.npz','native_contacts.npz','contact_events.json','contact_topology.json','ideal_sensor_samples.npz',
        'policy_histories.npz','policy_observations.json','camera_audit.json','actuator_identity.json','terminal_actuator_gains.json',
        'prefix_decisions.json','command_tape.json','suffix_window.json','process.log'}
    leaves|={f'rgb_{i:04d}.png' for i in range(row['rgb_packets'])}
    check(set(row['artifact_sha256'])==leaves,'exact artifact population')
    verify({str((directory/name).relative_to(ROOT)):sha for name,sha in row['artifact_sha256'].items()})
    actuator=json.loads((directory/'actuator_identity.json').read_text()); terminal=json.loads((directory/'terminal_actuator_gains.json').read_text())
    check(actuator['effective']==terminal=={'kp':[20.]*12,'kv':[.5]*12},'actuator readback')
    raw=read_npz(directory/'physics_trace.npz'); count=len(raw['timestamp_s'])
    check(count==row['physics_samples'] and count>0 and all(np.isfinite(v).all() for v in raw.values()),'finite physical arrays')
    check(np.allclose(raw['timestamp_s'],.002*np.arange(1,count+1),atol=1e-10,rtol=0),'physical clock')
    topology=json.loads((directory/'contact_topology.json').read_text())
    check(set(topology['environment_object_ids'].values())=={'ground_plane'}|{w['wall_id'] for w in spec['geometry']['wall_boxes']},'actual maze identity')
    flags,first=recompute_contact_flags(read_npz(directory/'native_contacts.npz'),topology,raw['timestamp_s'])
    check(np.array_equal(flags,raw['physics_contact'].astype(bool)) and row['any_contact']==bool(flags.any()),'native contact reconstruction')
    if first is not None: check(first['sample_index']==count-1 and row['stop_reason']=='DISALLOWED_CONTACT','immediate native stop')
    teacher=row['teacher_terminal_sample_index']; start=row['prefix_terminal_sample_index']; packet_index=row['branch_start_observation_index']
    check(0<=teacher<=start<count,'prefix endpoint indices')
    teacher_raw={k:v[:teacher+1] for k,v in raw.items()}
    check(np.all(teacher_raw['edge_index']==0) and np.count_nonzero(teacher_raw['phase']==0)==min(teacher+1,750),'teacher phases')
    prefix=row['prefix_result']; reduced=evaluate_edge(spec,teacher_raw,stop_reason=prefix['stop_reason'],crossing=crossing_for(teacher_raw,spec['geometry']))
    check(all(prefix[k]==v for k,v in reduced.items()),'teacher eligibility recomputation')
    decisions=json.loads((directory/'prefix_decisions.json').read_text()); audit_decisions(teacher_raw,decisions,spec,[prefix])
    available=prefix['stop_reason'] is None and prefix['checks']['sustained_correct_crossing'] and prefix['checks']['no_disallowed_contact']
    check(row['teacher_available']==available,'teacher availability')
    tape=json.loads((directory/'command_tape.json').read_text()); audit_commands(raw,row,spec,tape)
    binding=row['prefix_binding']; histories=read_npz(directory/'policy_histories.npz')
    check(binding['physics_arrays']==array_binding({k:v[:start+1] for k,v in raw.items()}),'whole moving-prefix physical binding')
    check(binding['history_arrays']==array_binding({k:v[packet_index] for k,v in histories.items()}),'moving-prefix history binding')
    check(binding['physics_samples']==start+1 and binding['timestamp_ns']==round(float(raw['timestamp_s'][start])*1e9),'moving-prefix count and clock')
    sensors=reconstruct_sensors(raw); recorded=read_npz(directory/'ideal_sensor_samples.npz')
    check(set(recorded)==set(sensors) and len(sensors['measured_ns'])==row['sensor_samples'],'sensor population')
    for k,v in sensors.items(): check(recorded[k].shape==v.shape and np.allclose(recorded[k],v,atol=1e-10,rtol=0),'independent ideal sensor '+k)
    camera=json.loads((directory/'camera_audit.json').read_text()); check(len(camera)==row['rgb_packets'],'camera population')
    expected_times={float(raw['timestamp_s'][i]) for i in (teacher,start,row['suffix_terminal_sample_index'],count-1)}
    expected_times|={d['timestamp_s'] for d in decisions if d['requested_command'] is not None}|{e['timestamp_s'] for e in tape}
    check([r['timestamp_s'] for r in camera]==sorted(expected_times),'actual command-boundary images')
    for index,metadata in enumerate(camera):
        packet=load_route_observation(directory,index); ns=packet['image']['measured_ns']; physical=metadata['physical_sample_index']
        check(ns==round(metadata['timestamp_s']*1e9) and raw['timestamp_s'][physical]==metadata['timestamp_s'],'image physical clock')
        check(hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()==metadata['rgb_sha256'],'actual RGB identity')
        check(metadata['rigid_mount_no_obstacle_adjustment'] is True,'undeclared camera movement')
        camera_reference(raw['base_pose_world'][physical],np.asarray(metadata['world_from_optical']))
        for schema in SCHEMAS:
            wanted=expected_history(raw,sensors,schema,ns); actual=packet['sensor_state'][schema.role][schema.name]
            check(all(np.allclose(actual[k],v,atol=1e-10,rtol=0) for k,v in wanted.items()),'independent causal history')
    check(camera[row['teacher_terminal_observation_index']]['physical_sample_index']==teacher,'teacher image endpoint')
    check(camera[packet_index]['physical_sample_index']==start and camera[packet_index]['rgb_sha256']==binding['rgb_pixels_sha256'],'moving image endpoint')
    manifest=json.loads((directory/'policy_observations.json').read_text())
    check(json.loads((directory/'suffix_window.json').read_text())==row['suffix_window'],'target artifact identity')
    counts=audit_targets(raw,row,manifest['frames'])
    match=compare_prefix(ROOT/reference['directory'],reference['reference'],directory,row) if row['branchable'] else None
    return {'status':'PASS','scene_id':spec['scene_id'],'physics_samples':count,'sensor_samples':row['sensor_samples'],
        'rgb_packets':len(camera),'target_counts':counts,'branchable':row['branchable'],'contact':bool(flags.any()),
        'prefix_match':match,'result_sha256':digest(directory/'result.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--completed-trials',type=int,default=384)
    count=parser.parse_args().completed_trials; check(1<=count<=384,'fixed audit population')
    target=OUTPUT/('raw_artifact_audit.json' if count==384 else f'raw_artifact_audit_interim_{count:03d}.json')
    check(not target.exists(),'audit output already exists'); check(digest(OUTPUT/'launch.json')==LAUNCH_SHA,'frozen physical launch')
    launch=json.loads((OUTPUT/'launch.json').read_text()); specs=trials()
    check(launch['trial_specs']==specs and launch['planned_composite_cells']==evidence_cells(),'fixed source population')
    bindings=launch['source_sha256']|launch['input_sha256']|launch['gait_sha256']; verify(bindings)
    audit_sources={p:digest(ROOT/p) for p in AUDIT_SOURCES}; audited=[]; report=None
    if count==384:
        report=json.loads((OUTPUT/'result.json').read_text())
        check(report['status']=='COMPLETE' and report['completed_trials']==report['planned_trials']==384 and report['launch_sha256']==LAUNCH_SHA,'full collection incomplete')
    try:
        for i,spec in enumerate(specs[:count]):
            directory=OUTPUT/spec['scene_id']; row=json.loads((directory/'result.json').read_text())
            result=audit_trial(directory,spec,row,launch['references'][spec['reference_scene_id']])
            if report is not None:
                check(report['trials'][i]==row|{'result_sha256':result['result_sha256'],'prefix_match':result['prefix_match']},'root/member/match agreement')
            audited.append(result); print(json.dumps({'event':'switch_trial_audited','completed':len(audited),'planned':count,'scene_id':spec['scene_id']}),flush=True)
        verify(bindings|audit_sources)
        if report is not None:
            check(report['branchable_trials']==sum(r['branchable'] for r in audited) and report['contact_trials']==sum(r['contact'] for r in audited),'aggregate outcome counts')
        result={'status':'PASS','audited_trials':count,'launch_sha256':LAUNCH_SHA,'audit_source_sha256':audit_sources,
            'study_result_sha256':digest(OUTPUT/'result.json') if count==384 else None,'trials':audited}
        write_json(target,result); print(json.dumps({'status':'PASS','audited_trials':count}),flush=True)
    except Exception as error:
        write_json(target,{'status':'FAIL','error':str(error),'audited_trials':len(audited),'planned_trials':count,
            'launch_sha256':LAUNCH_SHA,'audit_source_sha256':audit_sources,'trials':audited}); raise


if __name__=='__main__': main()
