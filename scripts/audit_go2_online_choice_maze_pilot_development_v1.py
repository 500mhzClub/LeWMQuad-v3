#!/usr/bin/env python3
"""Audit actual packet-driven selection, executed physics and all-trial utility."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'lewm_genesis',ROOT/'lewm_worlds'): sys.path.insert(0,str(p))
from lewm.counterfactual_maze_development import ACTIONS
from lewm.counterfactual_prefix_matching_development import compare_prefix
from lewm.online_choice_maze_pilot_development import trials
from lewm.online_local_choice_development import OnlineLocalChoice
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.audit_go2_counterfactual_maze_dataset_development import audit_branch
from scripts.run_go2_online_choice_maze_pilot_development_v1 import digest,paired_reduction,write_json


def check(value,message):
    if not value: raise ValueError(message)


def scalar_utility(row):
    if not row['branchable']: return 10.,'prefix_failure'
    label=row['horizon_labels'][-1]
    if label['contact_valid'] and label['contact_by_horizon']: return 10.,'contact_by_4s'
    if not label['contact_valid'] or not label['motion_valid']: return 10.,'noncontact_incomplete'
    if row['stop_reason'] is not None: return 10.,'post_horizon_failure'
    x,y,_=label['delta_xy_yaw_start_body']; gx,gy=row['intent_xy_body_start_m']
    return math.hypot(x-gx,y-gy),'observed'


def audit_trial(directory,spec,row,template):
    check(json.loads((directory/'result.json').read_text())==row,'trial result identity')
    check(row['method']==spec['method'] and row['intent_name']==spec['intent_name']
        and row['intent_xy_body_start_m']==spec['intent_xy_body_start_m'],'method/intent identity')
    check(digest(directory/'online_selection.json')==row['online_selection_sha256'],'selection artifact binding')
    selection=json.loads((directory/'online_selection.json').read_text())
    if row['branchable']:
        packet=load_route_observation(directory,row['branch_start_observation_index'])
        policy=OnlineLocalChoice(spec['method'],template.models,template.bindings)
        policy.begin_episode(packet['sensor_state']['identity'])
        replay=policy.select(packet,spec['intent_xy_body_start_m'],now_ns=packet['sensor_state']['decision_ns'])
        excluded={'adapter_ms','inference_ms'}
        check(set(selection)==set(replay),'selection schema')
        for key in selection.keys()-excluded: check(selection[key]==replay[key],f'actual packet selection replay: {key}')
        check(math.isfinite(selection['adapter_ms']) and selection['adapter_ms']>=selection['inference_ms']>=0,'selection timing')
        check(row['adapter_ms']==selection['adapter_ms'],'adapter timing summary')
        index=replay['selected_action_index']; name,command=ACTIONS[index]
        check(row['action_index']==index and row['action_name']==name and row['selected_stop']==(index==0),'executed selected action')
        check(selection['input_rgb_sha256']==row['prefix_binding']['rgb_pixels_sha256'],'actual own image binding')
        raw=np.load(directory/'physics_trace.npz',allow_pickle=False)
        tape=json.loads((directory/'branch_tape.json').read_text())
        for entry in tape:
            tick=entry['tick']; start=entry['pre_sample_index']+1; end=min(start+50,row['physics_samples'])
            check(entry['requested_command']==selection['requested_command_tape'][tick],'executed requested selection tape')
            check(np.allclose(raw['applied_command'][start:end],selection['expected_applied_command_tape'][tick],rtol=0,atol=1e-7),'executed prospective slew')
        raw.close()
    else:
        check(selection is None and row['action_index'] is None and row['action_name']=='unselected' and row['selected_stop'] is None,'selection after prefix failure')
        command=None; name='unselected'
    # Reuse unchanged raw physics/force/sensor/camera audit. Its declared command
    # is supplied only AFTER independently replaying the policy above.
    physical=audit_branch(directory,spec | {'action_name':name,'branch_command':None if command is None else list(command)},row)
    cost,kind=scalar_utility(row)
    check(math.isclose(row['utility']['cost'],cost,rel_tol=0,abs_tol=1e-12) and row['utility']['kind']==kind,'all-trial utility')
    if row['utility']['progress_m'] is not None:
        label=row['horizon_labels'][-1]; x,y,_=label['delta_xy_yaw_start_body']; gx,gy=spec['intent_xy_body_start_m']
        check(math.isclose(row['utility']['progress_m'],math.hypot(gx,gy)-math.hypot(x-gx,y-gy),rel_tol=0,abs_tol=1e-12),'progress reduction')
    return physical | {'method':spec['method'],'intent':spec['intent_name'],'realized_cost':cost,'status':'PASS'}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    check(output==ROOT/'.generated/go2_online_choice_maze_pilot_development_v1_attempt_001','exact pilot root required')
    check(not (output/'raw_artifact_audit.json').exists(),'audit already exists')
    launch=json.loads((output/'launch.json').read_text()); result=json.loads((output/'result.json').read_text())
    check(result['status']=='COMPLETE' and result['completed_trials']==result['planned_trials']==72,'pilot incomplete')
    check(digest(output/'launch.json')==result['launch_sha256'],'launch binding')
    specs=trials(); check(specs==launch['trial_specs'],'fixed trial population')
    for name,expected in (launch['source_sha256'] | launch['gait_sha256']).items():
        relative=Path(name)
        check(not relative.is_absolute() and '..' not in relative.parts and not any(p in ('sealed','sealed_test.json') or p.startswith('sealed_') for p in relative.parts),'bound source path')
        check(digest(ROOT/relative)==expected,f'source/gait binding {name}')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    templates={method:OnlineLocalChoice.from_completed_study(method) for method in ('always_stop','supervised_rollout','jepa')}
    references={}; audited=[]
    for i,(spec,member) in enumerate(zip(specs,result['trials'],strict=True)):
        directory=output/spec['scene_id']; row=json.loads((directory/'result.json').read_text())
        check(digest(directory/'result.json')==member['result_sha256'],'trial binding')
        check(row=={k:v for k,v in member.items() if k not in ('result_sha256','prefix_match')},'root/branch summary')
        audited.append(audit_trial(directory,spec,row,templates[spec['method']]))
        reference=references.setdefault(spec['layout_id'],(directory,row))
        match=compare_prefix(reference[0],reference[1],directory,row)
        match={k:v for k,v in match.items() if not k.startswith('canonical_model_context_')}
        match.update(reference_scene_id=reference[1]['scene_id'],selection_used_own_actual_packet=True)
        check(match==member['prefix_match'],'physical prefix matching')
        print(json.dumps({'event':'online_trial_audited','completed':i+1,'total':72,'scene_id':spec['scene_id']}),flush=True)
    reduction=paired_reduction(result['trials']); check(reduction==result['reduction'],'paired cost reduction')
    for layout in reduction['layouts']:
        for method,value in layout['methods'].items():
            costs=[r['realized_cost'] for r in audited if r['layout_id']==layout['layout_id'] and r['method']==method]
            check(len(costs)==3 and math.isclose(math.fsum(costs)/3,value['realized_cost'],rel_tol=0,abs_tol=1e-12),'independent paired means')
    audit={'status':'PASS','audited_trials':72,'audited_layouts':8,'audited_rgb_packets':sum(r['audited_rgb_packets'] for r in audited),
        'trials':audited,'study_result_sha256':digest(output/'result.json'),'audit_source_sha256':digest(Path(__file__)),
        'raw_branch_audit_source_sha256':digest(ROOT/'scripts/audit_go2_counterfactual_maze_dataset_development.py'),
        'scope':'actual packet/ensemble choice replay plus raw physical/sensor/artifact audit; conditional local pilot, not full maze or hardware qualification'}
    write_json(output/'raw_artifact_audit.json',audit); print(json.dumps({k:v for k,v in audit.items() if k!='trials'},indent=2))


if __name__=='__main__': main()
