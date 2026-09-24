#!/usr/bin/env python3
"""Test the actual bound policy adapter against saved per-seed predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lewm.counterfactual_learning_data_development import AuditedCounterfactualDataset
from lewm.counterfactual_decision_diagnostic_development import INTENTS
from lewm.online_local_choice_development import OnlineLocalChoice,SEEDS,STUDY,_read_bound
from lewm.route_rgb_dataset_development import load_route_observation


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if output!=ROOT/'.generated/go2_online_local_choice_replay_development_v1_attempt_001' or output.exists():
        raise ValueError('fresh explicit adapter replay output required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    report=json.loads(_read_bound(STUDY/'result.json','1a3c5d91910d08825487991cc999d2f0976e7bb92db620b11acc86f457b4ccb9'))
    dataset=AuditedCounterfactualDataset(ROOT/'.generated/go2_counterfactual_maze_dataset_development_v2_recovery_attempt_001','validation')
    sources=('scripts/replay_go2_online_local_choice_development_v1.py','lewm/online_local_choice_development.py',
        'docs/go2_online_local_choice_adapter_development_v1_2026-09-05.md')
    launch={'scope':'development observation replay, no new physics or fitting','source_sha256':{p:digest(ROOT/p) for p in sources},
        'saved_prediction_max_absolute_error_tolerance':1e-5,'layouts':8,'intents_per_layout':3,'methods':3}
    output.mkdir(); (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    templates={c:OnlineLocalChoice.from_completed_study(c) for c in ('supervised_rollout','jepa','always_stop')}
    witnesses={}
    for condition in ('supervised_rollout','jepa'):
        values=[]
        for seed in SEEDS:
            row=next(r for r in report['models'] if r['seed']==seed and r['condition']==condition)
            path=STUDY/f'{seed}-{condition}'/'validation_predictions.npz'
            if digest(path)!=row['artifact_sha256']['validation_predictions.npz']: raise ValueError('prediction witness changed')
            with np.load(path,allow_pickle=False) as saved: values.append(saved['intact__rollout'])
        witnesses[condition]=np.stack(values)
    rows=[]; largest=0.
    for index in range(0,40,5):
        member=dataset.rows[index]; reference=member['prefix_match']
        packet=load_route_observation(dataset.paths[reference['canonical_model_context_scene_id']],
            reference['canonical_model_context_observation_index'])
        for intent_name,goal in INTENTS:
            for condition,template in templates.items():
                policy=OnlineLocalChoice(condition,template.models,template.bindings)
                policy.begin_episode(packet['sensor_state']['identity'])
                selected=policy.select(packet,goal,now_ns=packet['sensor_state']['decision_ns'])
                error=0.
                if condition!='always_stop':
                    expected=witnesses[condition][:,index:index+5]
                    actual=np.asarray(selected['member_predictions'])
                    error=float(np.abs(actual-expected).max()); largest=max(largest,error)
                    if error>1e-5: raise ValueError('adapter predictions differ from fixed checkpoint witness')
                    predictions=expected.astype(np.float64)
                    probability=(1/(1+np.exp(-np.clip(predictions[:,:,-1,4],-60,60)))).mean(0)
                    costs=10*probability+np.linalg.norm(predictions[:,:,-1,:2].mean(0)-goal,axis=-1)
                    if not np.allclose(selected['candidate_costs'],costs,rtol=0,atol=1e-5): raise ValueError('ensemble cost mismatch')
                    if selected['selected_action_index']!=int(np.argmin(costs)): raise ValueError('selection mismatch')
                rows.append({'layout_id':member['result']['layout_id'],'intent':intent_name,
                    'prediction_max_absolute_difference':error,'selection':selected})
        print(json.dumps({'event':'layout_replayed','completed':index//5+1,'total':8}),flush=True)
    for name,expected in launch['source_sha256'].items():
        if digest(ROOT/name)!=expected: raise ValueError('adapter source changed during replay')
    result={'status':'PASS','replayed_choices':len(rows),'maximum_prediction_difference':largest,'choices':rows,
        'launch_sha256':digest(output/'launch.json'),'scope':launch['scope']}
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='choices'},indent=2))


if __name__=='__main__': main()
