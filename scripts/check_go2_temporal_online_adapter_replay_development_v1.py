#!/usr/bin/env python3
"""Recorded-stream orientation and frozen temporal adapter replay; no commands sent."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.causal_relative_orientation_development import CausalRelativeOrientation
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT
from lewm.online_temporal_choice_development import OnlineTemporalChoice,METHODS,STUDY,AUDIT_SHA,LAUNCH_SHA,_read_bound
from lewm.route_rgb_dataset_development import load_route_observation
from lewm.physical_execution_development import rotation_xyzw

OUTPUT=ROOT/'.generated/go2_temporal_online_adapter_replay_development_v1_attempt_001'
GYRO=ROOT/'.generated/go2_gyro_turn_assay_development_v1_attempt_001'
GYRO_RESULT_SHA='a35dfbbcd50758af0d3466e333a40e5d2f71d4cf491365803fa776e55e7ee52c'
GYRO_AUDIT_SHA='30e8c2a3256efe08baa59c179954d8c874389b14a205e794481ab10991f55442'
SOURCES=('lewm/causal_relative_orientation_development.py','lewm/online_temporal_choice_development.py',
    'lewm/tests/test_causal_relative_orientation_development.py','lewm/tests/test_online_temporal_choice_development.py',
    'docs/go2_temporal_online_adapter_replay_development_v1_2026-09-05.md',
    'scripts/check_go2_temporal_online_adapter_replay_development_v1.py')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def check(value,message):
    if not value: raise ValueError(message)


def physical_reference(directory,result):
    name='physics_trace.npz'; _read_bound(directory/name,result['artifact_sha256'][name])
    with np.load(directory/name,allow_pickle=False) as archive:
        times=np.rint(archive['timestamp_s']*1e9).astype(np.int64); poses=archive['base_pose_world'].copy()
    return {int(ns):i for i,ns in enumerate(times)},poses


def orientation_error(snapshot,initial_rotation,current_pose):
    truth=initial_rotation.T@rotation_xyzw(current_pose[3:])
    estimated=np.asarray(snapshot['rotation_initial_body_from_current_body'])
    angle=math.acos(float(np.clip((np.trace(estimated.T@truth)-1)/2,-1,1)))
    check(snapshot['relative_heading_rad'] is not None,'undefined projected heading')
    delta=snapshot['relative_heading_rad']-math.atan2(truth[1,0],truth[0,0])
    return angle,abs(math.atan2(math.sin(delta),math.cos(delta)))


def main():
    check(not OUTPUT.exists(),'fresh exact adapter replay output required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    dataset=AuditedSubtrajectoryDataset(DERIVATION_ROOT,'validation')
    audit=json.loads(_read_bound(STUDY/'raw_artifact_audit.json',AUDIT_SHA))
    study=json.loads(_read_bound(STUDY/'result.json',audit['study_result_sha256']))
    order=json.loads(_read_bound(STUDY/'validation_order.json',study['artifact_sha256']['validation_order.json']))
    lookup={m['window_id']:i for i,m in enumerate(order)}
    check(set(lookup)=={r['window_id'] for r in dataset.rows},'exact validation window order/population')
    templates={name:OnlineTemporalChoice.from_completed_study(name) for name in METHODS}
    offline={}
    for row in study['models']:
        directory=STUDY/f'{row["seed"]}-{row["condition"]}'
        path=directory/'validation_predictions.npz'; _read_bound(path,row['artifact_sha256']['validation_predictions.npz'])
        with np.load(path,allow_pickle=False) as archive:
            for head in ('direct',) if row['condition']=='direct' else ('direct','rollout'):
                offline[(row['seed'],row['condition'],head)]=archive[f'intact__{head}'].copy()
    gyro_result=json.loads(_read_bound(GYRO/'result.json',GYRO_RESULT_SHA))
    gyro_audit=json.loads(_read_bound(GYRO/'raw_artifact_audit.json',GYRO_AUDIT_SHA))
    check(gyro_audit['status']=='PASS' and gyro_audit['audited_trials']==18,'gyro audit prerequisite')
    bindings={name:sha(ROOT/name) for name in SOURCES}
    OUTPUT.mkdir(); write(OUTPUT/'launch.json',{'source_sha256':bindings,'study_launch_sha256':LAUNCH_SHA,
        'study_audit_sha256':AUDIT_SHA,'windows_sha256':sha(DERIVATION_ROOT/'windows.json'),
        'gyro_result_sha256':GYRO_RESULT_SHA,'gyro_audit_sha256':GYRO_AUDIT_SHA,
        'prediction_tolerance':1e-5,'orientation_tolerance_rad':.04,'scope':'off-policy recorded-stream replay only'})
    summaries=[]; attitudes=[]; choices=0; compared=0; excluded=0; maximum=0.
    try:
        for scene in sorted({r['scene_id'] for r in dataset.rows}):
            windows={r['decision_ns']:r for r in dataset.rows if r['scene_id']==scene}
            start=min(windows); directory=dataset.corpus.paths[scene]
            member=dataset.corpus.members[scene]['result']; physical,poses=physical_reference(directory,member)
            initial_rotation=rotation_xyzw(poses[physical[start],3:])
            manifest=json.loads((directory/'policy_observations.json').read_text())
            packets=[load_route_observation(directory,i) for i,f in enumerate(manifest['frames'])
                if f['decision_ns']>=start-300_000_000 and f['decision_ns']%100_000_000==0]
            for method,template in templates.items():
                adapter=OnlineTemporalChoice(method,template.models,template.bindings); adapter.begin_episode((0,0,0))
                case_choices=0; max_angle=0.; max_heading=0.; orientation_samples=0
                with (OUTPUT/f'{scene}-{method}.jsonl').open('x') as stream:
                    for packet in packets:
                        ns=packet['image']['measured_ns']; adapter.observe(packet,now_ns=ns)
                        if ns==start: adapter.begin_control([.8,0],now_ns=ns)
                        if ns<start: continue
                        if method=='always_stop':
                            angle,heading=orientation_error(adapter.orientation.snapshot(now_ns=ns),initial_rotation,poses[physical[ns]])
                            max_angle=max(max_angle,angle); max_heading=max(max_heading,heading); orientation_samples+=1
                        if ns not in windows: continue
                        row=windows[ns]; output=adapter.select(now_ns=ns)
                        stream.write(json.dumps(output,allow_nan=False)+'\n'); choices+=1; case_choices+=1
                        if method=='always_stop': continue
                        if row['context_scene_id']!=scene:
                            check(row['offset_ns']==0,'borrowed later context'); excluded+=1; continue
                        index=lookup[row['window_id']]; action=row['action_index']
                        for member_index,binding in enumerate(adapter.bindings):
                            reference=offline[(binding['seed'],adapter.condition,adapter.head)][index,0]
                            actual=np.asarray(output['member_predictions'][member_index][action])
                            error=float(np.max(np.abs(reference-actual))); maximum=max(maximum,error)
                            check(error<=1e-5,'frozen continuation prediction mismatch'); compared+=1
                check(case_choices==len(windows),'all available successive choices')
                summaries.append({'scene_id':scene,'method':method,'proposed_choices':case_choices,
                    'artifact_sha256':sha(OUTPUT/f'{scene}-{method}.jsonl')})
                if method=='always_stop': attitudes.append({'scene_id':scene,'source':'counterfactual_validation',
                    'samples':orientation_samples,'maximum_so3_error_rad':max_angle,'maximum_heading_error_rad':max_heading})
            print(json.dumps({'event':'branch_adapter_replayed','branches':len(summaries)//6,'total_branches':40}),flush=True)
        for row in gyro_result['trials']:
            scene=row['scene_id']; directory=GYRO/scene; physical,poses=physical_reference(directory,row)
            manifest=json.loads((directory/'policy_observations.json').read_text()); tracker=CausalRelativeOrientation()
            initial=None; maximum_angle=0.; maximum_heading=0.
            for i,frame in enumerate(manifest['frames']):
                packet=load_route_observation(directory,i); ns=frame['decision_ns']
                if i==0:
                    initial=rotation_xyzw(poses[physical[ns],3:]); snapshot=tracker.begin(packet,now_ns=ns)
                else: snapshot=tracker.step(packet,now_ns=ns)
                angle,heading=orientation_error(snapshot,initial,poses[physical[ns]])
                maximum_angle=max(maximum_angle,angle); maximum_heading=max(maximum_heading,heading)
            attitudes.append({'scene_id':scene,'source':'gyro_arena','samples':len(manifest['frames']),
                'maximum_so3_error_rad':maximum_angle,'maximum_heading_error_rad':maximum_heading})
        check(choices==1824 and len(summaries)==240 and len(attitudes)==58,'fixed replay population')
        check(compared==4080 and excluded==160,'continuation comparison/exclusion population')
        check(bindings=={name:sha(ROOT/name) for name in SOURCES},'adapter sources changed')
        so3=max(r['maximum_so3_error_rad'] for r in attitudes); heading=max(r['maximum_heading_error_rad'] for r in attitudes)
        result={'status':'COMPLETE','checks':{'continuation_replay':maximum<=1e-5,'orientation_so3':so3<=.04,'orientation_heading':heading<=.04},
            'proposed_choices':choices,'method_streams':len(summaries),'compared_member_predictions':compared,
            'excluded_noncanonical_initial_learned_choices':excluded,'maximum_prediction_difference':maximum,
            'orientation_streams':58,'maximum_so3_error_rad':so3,'maximum_heading_error_rad':heading,
            'orientation_stream_results':attitudes,'method_stream_results':summaries,'launch_sha256':sha(OUTPUT/'launch.json'),
            'scope':'observations not controlled by proposed actions; no physical policy or navigation qualification'}
        write(OUTPUT/'result.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ('orientation_stream_results','method_stream_results')},indent=2))
    except Exception as exc:
        write(OUTPUT/'failure.json',{'status':'FAILED_REPLAY_INTEGRITY','error':repr(exc),'proposed_choices':choices,
            'compared_member_predictions':compared,'maximum_prediction_difference':maximum}); raise


if __name__=='__main__': main()
