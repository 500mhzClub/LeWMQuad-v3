#!/usr/bin/env python3
"""Untrained temporal model on fixed actual corpus inputs; no optimizer or checkpoint."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from lewm.temporal_rgb_body_learning_development import active_parameters,training_loss,layout_batches


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def main():
    output=ROOT/'.generated/go2_temporal_model_interface_development_v1_attempt_001'
    if output.exists(): raise ValueError('fresh exact interface-check output required')
    sources=('lewm/temporal_rgb_body_jepa_development.py','lewm/temporal_rgb_body_learning_development.py',
        'lewm/tests/test_temporal_rgb_body_jepa_development.py',
        'docs/go2_temporal_rgb_body_learning_comparison_development_v1_2026-09-05.md',
        'scripts/check_go2_temporal_model_interface_development_v1.py')
    bindings={name:sha(ROOT/name) for name in sources}
    output.mkdir(); write(output/'launch.json',{'source_sha256':bindings,
        'input_result_sha256':sha(DERIVATION_ROOT/'result.json'),
        'input_tensor_check_sha256':sha(DERIVATION_ROOT/'tensor_interface_check.json'),
        'scope':'untrained interface and finite gradient checks; no optimizer, fitting or checkpoint'})
    torch.manual_seed(2026091700); model=TemporalRGBBodyJEPA(); rows=[]
    initial={k:v.clone() for k,v in model.state_dict().items()}
    try:
        for role in ('train','validation'):
            data=AuditedSubtrajectoryDataset(DERIVATION_ROOT,role)
            indices=[]
            for layout in sorted({r['layout_id'] for r in data.rows}):
                initial_index=next(i for i,r in enumerate(data.rows) if r['layout_id']==layout and r['action_index']==0 and r['offset_ns']==0)
                late_index=max((i for i,r in enumerate(data.rows) if r['layout_id']==layout and r['action_index']==1),key=lambda i:data.rows[i]['offset_ns'])
                indices.extend([initial_index,late_index])
            if role=='train':
                schedule=[indices for epoch in range(240) for indices in layout_batches(data.rows,epoch,2026091700)]
                if len(schedule)!=1200 or any(len(set(data.rows[i]['layout_id'] for i in ids))!=16 for ids in schedule):
                    raise ValueError('full actual metadata schedule')
            for index in indices:
                item=data[index]
                history={k:v[None] for k,v in item['observation_history'].items()}
                plans=item['known_action_blocks'][None]; valid=item['known_action_valid'][None]
                with torch.no_grad(): result=model(history,plans,valid)
                if not torch.equal(result['prediction_valid'],valid.all(-1)): raise ValueError('prediction mask')
                for name in ('future_latents','direct_outcomes','rollout_outcomes'):
                    if not torch.isfinite(result[name]).all() or torch.count_nonzero(result[name][~valid.all(-1)]):
                        raise ValueError('invalid actual model output')
                rows.append({'window_id':item['metadata']['window_id'],'data_role':role,'status':'PASS',
                    'known_horizons':int(valid.all(-1).sum())})
            # Training-only finite-gradient smoke; validation labels do not enter it.
            if role=='train':
                selected=[data[i] for i in next(layout_batches(data.rows,0,2026091700))]
                targets={k:({n:torch.stack([r['targets'][k][n] for r in selected]) for n in selected[0]['targets'][k]}
                    if isinstance(selected[0]['targets'][k],dict) else torch.stack([r['targets'][k] for r in selected]))
                    for k in selected[0]['targets']}
                batch={'observation_history':{k:torch.stack([r['observation_history'][k] for r in selected]) for k in history},
                    'known_action_blocks':torch.stack([r['known_action_blocks'] for r in selected]),
                    'known_action_valid':torch.stack([r['known_action_valid'] for r in selected]),
                    'targets':targets,'metadata':[r['metadata'] for r in selected]}
                for condition in ('direct','supervised_rollout','jepa'):
                    model.zero_grad(set_to_none=True); loss,_=training_loss(model,batch,condition); loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active_parameters(model,condition)):
                        raise ValueError('actual finite active gradient')
        if any(not torch.equal(initial[k],v) for k,v in model.state_dict().items()): raise ValueError('model changed during no-fit check')
        if bindings!={name:sha(ROOT/name) for name in sources}: raise ValueError('source changed')
        result={'status':'PASS','checked_windows':len(rows),'rows':rows,'full_seed0_schedule_updates':1200,
            'training_only_gradient_conditions':['direct','supervised_rollout','jepa'],'parameters_unchanged':True,
            'active_parameters':{c:sum(p.numel() for p in active_parameters(model,c)) for c in ('direct','supervised_rollout','jepa')},
            'launch_sha256':sha(output/'launch.json'),'scope':'no-fit interface readiness, not prediction or navigation performance'}
        write(output/'result.json',result); print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    except Exception as exc:
        write(output/'failure.json',{'status':'FAILED_INTERFACE','error':repr(exc),'checked_windows':len(rows)}); raise


if __name__=='__main__': main()
