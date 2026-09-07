#!/usr/bin/env python3
"""Checker-only V2: account for native termination on the command boundary."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.counterfactual_prefix_matching_development import compare_prefix
from lewm.local_execution_controller_development import evaluate_edge
from lewm.online_temporal_choice_development import METHODS,OnlineTemporalChoice
from lewm.physical_execution_development import rotation_xyzw
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.simulated_body_observation_development import SCHEMAS
from lewm.successive_audit_clock_boundary_development import expected_ingested
from scripts.run_go2_successive_choice_full_audit_development_v1 import DEPENDENCIES
from lewm.successive_choice_maze_development import trials
from lewm.successive_choice_metrics_development import reduce_trial,paired_reduction
from scripts.audit_go2_causal_rgb_body_capture_development_v1 import reconstruct_sensors,expected_history
from scripts.audit_go2_local_control_factorial_development_v1 import (
    check,read_npz,recompute_contact_flags,audit_decisions,crossing_for)
from scripts.run_go2_counterfactual_maze_dataset_development_v1 import array_binding
from scripts.run_go2_successive_choice_maze_development_v1 import OUTPUT,digest,verify,write_json

LAUNCH_SHA='e9a4bd281e631f06e01a134c3d68b969613d4599bf0554299a55d34bc20f7bf5'


def body_delta(origin,target):
    # Independent quaternion-vector rotation by q inverse, rather than sharing
    # the collector's rotation-matrix reduction.
    q=np.asarray(origin[3:],dtype=float); q=q/np.linalg.norm(q)
    vector=np.asarray(target[:3])-np.asarray(origin[:3]); u=-q[:3]
    return vector+2*q[3]*np.cross(u,vector)+2*np.cross(u,np.cross(u,vector))


def close(actual,expected,message):
    if expected is None: check(actual is None,message)
    else: check(actual is not None and math.isclose(actual,expected,abs_tol=1e-10,rel_tol=0),message)


def camera_reference(pose,optical):
    # Fixed, launch-bound native camera mount; float32 simulator pose readback.
    body=rotation_xyzw(pose[3:]); expected=np.eye(4)
    expected[:3,:3]=np.column_stack((-body[:,1],-body[:,2],body[:,0]))
    expected[:3,3]=pose[:3]+body@np.array([.326,0.,.043])
    check(np.allclose(optical,expected,atol=1e-6,rtol=0),'camera rigid mount versus physical body')


def scalar_metrics(raw,row,tape,choices):
    metrics=row['metrics']; start=row['prefix_terminal_sample_index']
    times=np.rint(raw['timestamp_s']*1e9).astype(np.int64)
    control=[e for e in tape if e['stage']=='control' and e['post_sample_index']>e['pre_sample_index']]
    end=control[-1]['post_sample_index'] if control else start
    vector=body_delta(raw['base_pose_world'][start],raw['base_pose_world'][end])
    ux,uy=np.asarray(row['intent_xy_body_start_m'])/.8
    signed=float(vector[0]*ux+vector[1]*uy); lateral=float(-vector[0]*uy+vector[1]*ux)
    close(metrics['observed_control_duration_s'],(int(times[end])-int(times[start]))/1e9,'observed duration')
    close(metrics['observed_signed_control_displacement_m'],signed if row['branchable'] else None,'independent signed progress')
    close(metrics['observed_lateral_control_displacement_m'],lateral if row['branchable'] else None,'independent lateral progress')
    close(metrics['four_second_signed_displacement_m'],signed if metrics['complete_control_and_release'] else None,'fixed horizon progress')
    check(metrics['any_contact']==any(bool(x) for x in raw['physics_contact']),'independent contact count')
    for index,(choice,error) in enumerate(zip(choices,metrics['executed_decision_errors'],strict=True)):
        executed=[e for e in control if e['decision_index']==index]
        if not executed:
            check(error=={'decision_index':index,'label':None,'position_error_m':None,'yaw_error_rad':None,'contact_brier':None},'unexecuted proposal labeling')
            continue
        a=executed[0]['pre_sample_index']; b=executed[-1]['post_sample_index']; target_time=int(times[a])+500_000_000
        selected=[i for i in range(a+1,b+1) if int(times[i])<=target_time]
        contact=any(bool(raw['physics_contact'][i]) for i in selected)
        at=[i for i in selected if int(times[i])==target_time]
        motion_valid=bool(at) and not contact; contact_valid=int(times[b])>=target_time or contact
        label=error['label']
        check(label['horizon_ns']==500_000_000 and label['motion_valid']==motion_valid
            and label['contact_valid']==contact_valid and label['contact_by_horizon']==contact,'executed horizon censoring')
        displacement=None
        if motion_valid:
            pose0=raw['base_pose_world'][a]; pose=raw['base_pose_world'][at[0]]
            delta=body_delta(pose0,pose)
            def yaw(p):
                x,y,z,w=p[3:]/np.linalg.norm(p[3:])
                return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            diff=yaw(pose)-yaw(pose0); displacement=[*delta[:2],math.atan2(math.sin(diff),math.cos(diff))]
            check(np.allclose(label['delta_xy_yaw_start_body'],displacement,atol=1e-10,rtol=0),'independent endpoint motion')
        else: check(label['delta_xy_yaw_start_body'] is None,'unobserved/contact motion imputation')
        position=yaw_error=brier=None; action=choice['selected_action_index']
        if choice['mean_motion_sin_cos'] is not None:
            pred=choice['mean_motion_sin_cos'][action]
            if motion_valid:
                position=math.hypot(pred[0]-displacement[0],pred[1]-displacement[1])
                difference=math.atan2(pred[2],pred[3])-displacement[2]
                yaw_error=abs(math.atan2(math.sin(difference),math.cos(difference)))
            if contact_valid: brier=(choice['mean_contact_probability'][action]-float(contact))**2
        close(error['position_error_m'],position,'independent executed position error')
        close(error['yaw_error_rad'],yaw_error,'independent executed yaw error')
        close(error['contact_brier'],brier,'independent executed Brier')


def audit_commands(raw,row,tape,events):
    previous=row['prefix_terminal_sample_index']; next_ticks={}; released=False
    for tick,entry in enumerate(tape):
        a,b=entry['pre_sample_index'],entry['post_sample_index']; stage=entry['stage']
        check(entry['tick']==tick and a==previous and 0<=b-a<=50,'command order and executed sample count')
        check(entry['timestamp_s']==raw['timestamp_s'][a],'command pre-dispatch time')
        if stage=='control':
            check(not released,'learned command after release')
            index=entry['decision_index']; offset=next_ticks.setdefault(index,0)
            check(0<=index<len(events) and offset<5,'selection/tick population')
            choice=events[index]['selection']; requested=choice['requested_command_tape'][offset]
            expected=choice['expected_applied_command_tape'][offset]; next_ticks[index]+=1
            check(a==events[index]['pre_sample_index']+offset*50,'selected plan execution timing')
        else:
            check(stage in ('release','fault_release') and entry['decision_index'] is None,'release stage')
            released=True; requested=[0.,0.,0.]; expected=None
            check(stage!='fault_release' or row['sensor_fault'] is not None,'unreported sensor fault')
        check(entry['requested_command']==requested,'requested selected command')
        if b>a:
            interval=slice(a+1,b+1)
            check(np.all(raw['edge_index'][interval]==1) and np.all(raw['phase'][interval]==(1 if stage=='control' else 2)),'physical command phase')
            check(np.array_equal(raw['requested_command'][interval],np.tile(requested,(b-a,1))),'raw requested command')
            prior=raw['applied_command'][a]
            clipped=np.clip(np.asarray(requested,dtype=np.float32),[-.3,0,-.5],[.3,0,.5]).astype(np.float32)
            actual=prior+np.clip(clipped-prior,[-.25,0,-.35],[.25,0,.35])
            check(np.allclose(raw['applied_command'][interval],actual,atol=1e-7,rtol=0),'independent clipping/slew')
            if expected is not None: check(np.allclose(raw['applied_command'][interval],expected,atol=1e-7,rtol=0),'prospective plan versus executed slew')
        else: check(row['sensor_fault'] is not None,'empty command without sensor fault')
        if b-a not in (0,50): check(b==len(raw['timestamp_s'])-1 and row['stop_reason'] is not None,'interrupted command not terminal')
        previous=b
    check(previous==len(raw['timestamp_s'])-1,'unexplained post-prefix physics')


def audit_trial(directory,spec,row,template):
    for key in ('scene_id','layout_id','data_role','method','intent_name','intent_xy_body_start_m'):
        check(row[key]==spec[key],'trial identity '+key)
    expected={'actuator_identity.json','terminal_actuator_gains.json','physics_trace.npz','native_contacts.npz',
        'contact_topology.json','contact_events.json','process.log','ideal_sensor_samples.npz','policy_histories.npz',
        'policy_observations.json','camera_audit.json','prefix_decisions.json','command_tape.json','selection_events.json','observed_indices.json'}
    expected|={f'rgb_{i:04d}.png' for i in range(row['rgb_packets'])}|{f'choice_{i:02d}.json' for i in range(row['metrics']['decisions'])}
    check(set(row['artifact_sha256'])==expected,'complete artifact population')
    verify({str((directory/name).relative_to(ROOT)):sha for name,sha in row['artifact_sha256'].items()})
    actuator=json.loads((directory/'actuator_identity.json').read_text()); terminal=json.loads((directory/'terminal_actuator_gains.json').read_text())
    check(actuator['effective']==terminal=={'kp':[20.]*12,'kv':[.5]*12},'effective actuator identity')
    raw=read_npz(directory/'physics_trace.npz'); count=len(raw['timestamp_s'])
    check(count==row['physics_samples'] and count>0 and all(np.isfinite(v).all() for v in raw.values()),'finite physical population')
    check(np.allclose(raw['timestamp_s'],.002*np.arange(1,count+1),atol=1e-10,rtol=0),'global physical clock')
    topology=json.loads((directory/'contact_topology.json').read_text())
    check(set(topology['environment_object_ids'].values())=={'ground_plane'}|{w['wall_id'] for w in spec['geometry']['wall_boxes']},'physical scene identity')
    flags,first=recompute_contact_flags(read_npz(directory/'native_contacts.npz'),topology,raw['timestamp_s'])
    check(np.array_equal(flags,raw['physics_contact'].astype(bool)),'native force/contact agreement')
    if first is not None: check(first['sample_index']==count-1 and row['stop_reason']=='DISALLOWED_CONTACT','native immediate contact stop')
    start=row['prefix_terminal_sample_index']; packet_index=row['branch_start_observation_index']
    check(0<=start<count,'prefix index'); prefix_raw={k:v[:start+1] for k,v in raw.items()}
    check(np.all(prefix_raw['edge_index']==0) and np.count_nonzero(prefix_raw['phase']==0)==min(start+1,750),'prefix phases')
    prefix=row['prefix_result']; reduced=evaluate_edge(spec,prefix_raw,stop_reason=prefix['stop_reason'],crossing=crossing_for(prefix_raw,spec['geometry']))
    check(all(prefix[k]==v for k,v in reduced.items()),'prefix eligibility recomputation')
    decisions=json.loads((directory/'prefix_decisions.json').read_text()); audit_decisions(prefix_raw,decisions,spec,[prefix])
    eligible=prefix['stop_reason'] is None and prefix['checks']['sustained_correct_crossing'] and prefix['checks']['no_disallowed_contact']
    check(row['branchable']==eligible,'prefix availability')
    check(array_binding(prefix_raw)==row['prefix_binding']['physics_arrays'],'physical prefix binding')
    histories=read_npz(directory/'policy_histories.npz')
    check(array_binding({k:v[packet_index] for k,v in histories.items()})==row['prefix_binding']['history_arrays'],'history prefix binding')
    tape=json.loads((directory/'command_tape.json').read_text()); events=json.loads((directory/'selection_events.json').read_text())
    check(len(events)==row['metrics']['decisions'],'decision artifact population')
    for index,event in enumerate(events): check(json.loads((directory/f'choice_{index:02d}.json').read_text())==event,'per-choice artifact')
    audit_commands(raw,row,tape,events)
    sensors=reconstruct_sensors(raw); recorded=read_npz(directory/'ideal_sensor_samples.npz')
    check(set(recorded)==set(sensors) and len(sensors['measured_ns'])==row['sensor_samples'],'sensor population')
    for k,v in sensors.items(): check(recorded[k].shape==v.shape and np.allclose(recorded[k],v,atol=1e-10,rtol=0),'independent ideal sensor '+k)
    camera=json.loads((directory/'camera_audit.json').read_text()); check(len(camera)==row['rgb_packets'],'camera population')
    expected_times={float(raw['timestamp_s'][start]),float(raw['timestamp_s'][-1])}
    expected_times|={d['timestamp_s'] for d in decisions if d['requested_command'] is not None}|{e['timestamp_s'] for e in tape}
    expected_times|={float(raw['timestamp_s'][e['post_sample_index']]) for e in tape
        if e['stage']=='control' and e['post_sample_index']-e['pre_sample_index']==50}
    check([r['timestamp_s'] for r in camera]==sorted(expected_times),'RGB command and post-action coverage')
    observed=json.loads((directory/'observed_indices.json').read_text())
    check(observed==sorted(set(observed)) and all(0<=i<len(camera) for i in observed),'ingested packet indices')
    policy=OnlineTemporalChoice(spec['method'],template.models,template.bindings); policy.begin_episode((0,0,0))
    event_by_packet={e['observation_index']:e for e in events}; check(len(event_by_packet)==len(events),'duplicate decision image')
    replayed=0; maximum_heading=0.; began=False
    for index,metadata in enumerate(camera):
        packet=load_route_observation(directory,index); ns=packet['image']['measured_ns']
        check(ns==int(round(metadata['timestamp_s']*1e9)),'image timestamp')
        check(hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest()==metadata['rgb_sha256'],'actual RGB binding')
        physical_index=metadata['physical_sample_index']; check(raw['timestamp_s'][physical_index]==metadata['timestamp_s'],'camera physical sample')
        optical=np.asarray(metadata['world_from_optical']); rot=optical[:3,:3]
        check(np.allclose(rot.T@rot,np.eye(3),atol=1e-10,rtol=0) and abs(np.linalg.det(rot)-1)<1e-10,'proper optical frame')
        camera_reference(raw['base_pose_world'][physical_index],optical)
        for schema in SCHEMAS:
            wanted=expected_history(raw,sensors,schema,ns); actual=packet['sensor_state'][schema.role][schema.name]
            check(all(np.allclose(actual[k],v,atol=1e-10,rtol=0) for k,v in wanted.items()),'actual causal history')
        if index in observed:
            policy.observe(packet,now_ns=ns)
            if index==packet_index and eligible:
                policy.begin_control(spec['intent_xy_body_start_m'],now_ns=ns); began=True
            if began:
                truth=rotation_xyzw(raw['base_pose_world'][start,3:]).T@rotation_xyzw(raw['base_pose_world'][physical_index,3:])
                estimate=policy.orientation.snapshot(now_ns=ns)['rotation_initial_body_from_current_body']
                estimate=np.asarray(estimate)
                difference=math.atan2(estimate[1,0],estimate[0,0])-math.atan2(truth[1,0],truth[0,0])
                maximum_heading=max(maximum_heading,abs(math.atan2(math.sin(difference),math.cos(difference))))
            if index in event_by_packet:
                event=event_by_packet[index]; replay=policy.select(now_ns=ns); saved=event['selection']
                check(set(saved)==set(replay),'selection schema')
                for key in saved.keys()-{'adapter_ms','inference_ms'}: check(saved[key]==replay[key],'fresh actual packet policy replay '+key)
                check(math.isfinite(saved['adapter_ms']) and saved['adapter_ms']>=saved['inference_ms']>=0,'latency validity')
                check(event['pre_sample_index']==physical_index,'selection pre-action physical boundary'); replayed+=1
    check(replayed==len(events),'every selection independently replayed')
    # A sensor fault cannot be silently accepted as audited merely because its
    # zero-release trace is plausible. Demand a separate fault reproduction if
    # one actually occurs; preserve that trial and fail this audit explicitly.
    check(row['sensor_fault'] is None,'sensor-fault evidence requires explicit rejection replay; no automatic qualification')
    if eligible:
        expected=expected_ingested(camera,tape,prefix_end_time=float(raw['timestamp_s'][start]),
            terminal_index=count-1,stop_reason=row['stop_reason'])
        check(observed==expected,'continuous live history coverage')
    check(camera[packet_index]['physical_sample_index']==start and camera[packet_index]['rgb_sha256']==row['prefix_binding']['rgb_pixels_sha256'],'own initial RGB')
    choices=[e['selection'] for e in events]
    reduced=reduce_trial(raw,start_index=start,tape=tape,selections=choices,direction=spec['intent_xy_body_start_m'],
        branchable=eligible,stop_reason=row['stop_reason'],sensor_fault=row['sensor_fault'])
    check(reduced==row['metrics'],'full metric reduction'); scalar_metrics(raw,row,tape,choices)
    return {'status':'PASS','scene_id':spec['scene_id'],'layout_id':spec['layout_id'],'method':spec['method'],
        'physics_samples':count,'sensor_samples':row['sensor_samples'],'rgb_packets':len(camera),'replayed_choices':replayed,
        'contact':bool(first is not None),'maximum_reference_heading_error_rad':maximum_heading,
        'result_sha256':digest(directory/'result.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--completed-trials',type=int,default=144)
    count=parser.parse_args().completed_trials; check(1<=count<=144,'fixed prefix audit population')
    target=OUTPUT/('raw_artifact_audit_clock_boundary_v2.json' if count==144 else f'raw_artifact_audit_clock_boundary_v2_interim_{count:03d}.json')
    check(not target.exists(),'audit already exists')
    predecessor_sha='19c67f2b3fea7f614cacafcb1d480b021349eca655446b4de8f37952cb862fbd'
    check(digest(OUTPUT/'raw_artifact_audit.json')==predecessor_sha,'preserved failed predecessor audit')
    verify(DEPENDENCIES)
    check(digest(OUTPUT/'launch.json')==LAUNCH_SHA,'frozen launch binding')
    launch=json.loads((OUTPUT/'launch.json').read_text()); specs=trials(); check(launch['trial_specs']==specs,'fixed trial order')
    bindings=launch['source_sha256']|launch['gait_sha256']|launch['prerequisite_sha256']; verify(bindings)
    result=None
    if count==144:
        result=json.loads((OUTPUT/'result.json').read_text())
        check(result['status']=='COMPLETE' and result['planned_trials']==result['completed_trials']==144,'full panel incomplete')
        check(result['launch_sha256']==LAUNCH_SHA,'result launch identity')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    templates={m:OnlineTemporalChoice.from_completed_study(m) for m in METHODS}
    check({k:v.bindings for k,v in templates.items()}==launch['model_bindings'],'full fixed ensemble identities')
    audited=[]; rows=[]; references={}
    audit_sources={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
        'lewm/tests/test_successive_choice_raw_audit.py':digest(ROOT/'lewm/tests/test_successive_choice_raw_audit.py')}
    audit_sources.update(DEPENDENCIES)
    for name in ('scripts/run_go2_successive_choice_full_audit_development_v1.py',
            'lewm/successive_audit_clock_boundary_development.py',
            'lewm/tests/test_successive_audit_clock_boundary_development.py',
            'docs/go2_successive_choice_audit_clock_boundary_correction_2026-09-05.md'):
        audit_sources[name]=digest(ROOT/name)
    verify(bindings|audit_sources)
    try:
        for index,spec in enumerate(specs[:count]):
            directory=OUTPUT/spec['scene_id']; row=json.loads((directory/'result.json').read_text()); rows.append(row)
            audited.append(audit_trial(directory,spec,row,templates[spec['method']]))
            reference=references.setdefault(spec['layout_id'],(directory,row)); match=compare_prefix(reference[0],reference[1],directory,row)
            match={k:v for k,v in match.items() if not k.startswith('canonical_model_context_')}
            if result is not None:
                member=result['trials'][index]
                check(row=={k:v for k,v in member.items() if k not in ('result_sha256','prefix_match','prefix_reference_scene_id')},'root/trial equality')
                check(member['result_sha256']==digest(directory/'result.json') and member['prefix_match']==match
                    and member['prefix_reference_scene_id']==reference[1]['scene_id'],'paired prefix/result binding')
            print(json.dumps({'event':'successive_trial_audited','completed':index+1,'total':count,'scene_id':spec['scene_id']}),flush=True)
        if result is not None: check(paired_reduction(rows)==result['reduction'],'paired all-layout reduction')
        verify(bindings|audit_sources)
        audit={'status':'PASS','full_study':count==144,'audited_trials':count,'planned_trials':144,'trials':audited,
            'audited_choices':sum(r['replayed_choices'] for r in audited),'audited_rgb_packets':sum(r['rgb_packets'] for r in audited),
            'launch_sha256':LAUNCH_SHA,'audit_source_sha256':audit_sources,
            'study_result_sha256':digest(OUTPUT/'result.json') if result is not None else None,
            'scope':'raw physical/camera/history/actual policy replay and independent endpoint checks; no maze/hardware qualification'}
        audit['preserved_failed_predecessor_sha256']=predecessor_sha
        write_json(target,audit)
        if count==144:
            write_json(OUTPUT/'full_audit_source_dependency_witness_clock_boundary_v2.json',{
                'status':'PASS','source_sha256':bindings|audit_sources,'full_audit_sha256':digest(target),
                'preserved_failed_predecessor_sha256':predecessor_sha,
                'scope':'checker-only clock-boundary correction; same physical evidence and outcomes'})
        print(json.dumps({k:v for k,v in audit.items() if k!='trials'},indent=2))
    except Exception as error:
        write_json(target,{'status':'FAIL','error':str(error),'full_study':count==144,'audited_trials':len(audited),
            'requested_trials':count,'trials':audited,'launch_sha256':LAUNCH_SHA,'audit_source_sha256':audit_sources})
        raise


if __name__=='__main__': main()
