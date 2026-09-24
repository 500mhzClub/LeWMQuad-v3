#!/usr/bin/env python3
"""One fixed-budget, three-seed direct/auxiliary-rollout/JEPA development study."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from lewm.counterfactual_learning_data_development import AuditedCounterfactualDataset
from lewm.rgb_body_jepa_reference_development import RGBBodyJEPAReference
from lewm.rgb_body_learning_experiment_development import CONDITIONS,active_parameters,layout_batches,prediction_metrics,simple_predictions,training_loss

SEEDS=(2026091200,2026091201,2026091202)
EPOCHS=60
SOURCES=(
    'scripts/run_go2_rgb_body_learning_comparison_development_v1.py',
    'lewm/counterfactual_learning_data_development.py','lewm/rgb_body_learning_experiment_development.py',
    'lewm/rgb_body_jepa_reference_development.py','lewm/rgb_body_tensor_interface_development.py',
    'lewm/route_rgb_dataset_development.py','lewm/causal_rgb_dataset_development.py',
    'lewm/simulated_body_observation_development.py','lewm/causal_sensor_state.py',
    'lewm/counterfactual_maze_development.py','lewm/multijunction_routes_development.py',
    'lewm/physical_execution_development.py',
    'docs/go2_rgb_body_learning_comparison_development_v1_2026-09-05.md',
)


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def tensor_stack(rows):
    if isinstance(rows[0],dict): return {k:tensor_stack([r[k] for r in rows]) for k in rows[0]}
    return torch.stack(rows)


def take(value,indices):
    if isinstance(value,dict): return {k:take(v,indices) for k,v in value.items()}
    if isinstance(value,list): return [value[i] for i in indices]
    return value[indices]


def materialize(directory,role):
    dataset=AuditedCounterfactualDataset(directory,role)
    rows=[]
    for i in range(len(dataset)):
        rows.append(dataset[i])
        if (i+1)%10==0: print(json.dumps({'event':'learning_row_loaded','role':role,'completed':i+1,'total':len(dataset)}),flush=True)
    metadata=[row.pop('metadata') for row in rows]
    batch=tensor_stack(rows); batch['metadata']=metadata
    expected=80 if role=='train' else 40
    if len(metadata)!=expected: raise ValueError('study requires full fixed branch population; no silently selected subset')
    for layout in {r['layout_id'] for r in metadata}:
        indices=[i for i,r in enumerate(metadata) if r['layout_id']==layout]
        if len(indices)!=5 or {metadata[i]['action_index'] for i in indices}!=set(range(5)):
            raise ValueError('incomplete sibling population')
        for key,value in batch['observation'].items():
            if not all(torch.equal(value[indices[0]],value[i]) for i in indices):
                raise ValueError(f'nonidentical sibling model context: {key}')
    return batch


def permutation(metadata,kind):
    layouts=sorted({r['layout_id'] for r in metadata})
    lookup={(r['layout_id'],r['action_index']):i for i,r in enumerate(metadata)}
    if kind=='action': return [lookup[(r['layout_id'],(r['action_index']+1)%5)] for r in metadata]
    return [lookup[(layouts[(layouts.index(r['layout_id'])+1)%len(layouts)],r['action_index'])] for r in metadata]


@torch.no_grad()
def evaluate(model,batch,condition,control='intact'):
    observation={k:v for k,v in batch['observation'].items()}; plans=batch['known_action_blocks']
    if control in ('rgb_shuffle','body_shuffle'):
        indices=permutation(batch['metadata'],'scene')
        key='rgb' if control=='rgb_shuffle' else 'body'
        observation[key]=observation[key][indices]
    elif control=='action_shuffle': plans=plans[permutation(batch['metadata'],'action')]
    elif control!='intact': raise ValueError('undeclared evaluation control')
    predictions={}
    if condition=='direct': predictions['direct']=model.direct_prediction(observation,plans).numpy()
    else:
        out=model(observation,plans)
        predictions={'direct':out['direct_outcomes'].numpy(),'rollout':out['rollout_outcomes'].numpy()}
    return predictions,{k:prediction_metrics(v,batch['targets'],batch['metadata']) for k,v in predictions.items()}


@torch.no_grad()
def latent_diagnostics(model,batch,condition):
    z=model.encoder(batch['observation'])
    indices=[i for i,m in enumerate(batch['metadata']) if m['action_index']==0]
    independent=z[indices]; centered=independent-independent.mean(0)
    values=torch.linalg.svdvals(centered).square(); weights=values/values.sum().clamp_min(1e-12)
    effective_rank=float(torch.exp(-(weights*weights.clamp_min(1e-12).log()).sum())) if float(values.sum())>0 else 0.
    valid=batch['targets']['future_valid']
    future={k:v[valid] for k,v in batch['targets']['future_observations'].items()}
    target=model.target(future)
    persistence=z[:,None].expand(-1,8,-1)[valid]
    result={'independent_contexts':len(indices),'current_mean_feature_std':float(independent.std(0).mean()),
        'current_effective_rank':effective_rank,'ema_latent_persistence_mse':float((persistence-target).square().mean())}
    if condition!='direct':
        predicted=model.predict_latents(z,batch['known_action_blocks'])[valid]
        result['ema_latent_prediction_mse']=float((predicted-target).square().mean())
    return result


@torch.no_grad()
def latency(model,batch,condition):
    observation=take(batch['observation'],[0]); plans=batch['known_action_blocks'][:1]
    methods={'direct':lambda:model.direct_prediction(observation,plans)}
    if condition!='direct': methods['both_heads_and_rollout']=lambda:model(observation,plans)
    result={}
    for name,method in methods.items():
        for _ in range(3): method()
        timings=[]
        for _ in range(20):
            start=time.perf_counter(); method(); timings.append((time.perf_counter()-start)*1000)
        result[name]={'median_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95)),
            'scope':'one CPU thread; one observation and one eight-horizon plan; excludes packet acquisition/tensor conversion'}
    return result


def paired_summary(runs):
    comparisons=(('jepa','supervised_rollout'),('jepa','direct'),('supervised_rollout','direct'))
    output={}
    for a,b in comparisons:
        result={}
        for metric in ('position_error_m','yaw_error_rad','contact_brier'):
            deltas=[]
            for seed in SEEDS:
                rows_a=next(r for r in runs if r['seed']==seed and r['condition']==a)['validation']['intact']['direct']['layouts']
                rows_b=next(r for r in runs if r['seed']==seed and r['condition']==b)['validation']['intact']['direct']['layouts']
                if [r['layout_id'] for r in rows_a]!=[r['layout_id'] for r in rows_b]: raise ValueError('unpaired layouts')
                deltas.append([x[metric]-y[metric] for x,y in zip(rows_a,rows_b,strict=True)])
            per_layout=np.mean(deltas,axis=0)
            rng=np.random.default_rng(2026091299)
            samples=per_layout[rng.integers(0,len(per_layout),size=(10000,len(per_layout)))].mean(1)
            result[metric]={'mean_delta':float(per_layout.mean()),'layout_bootstrap_95_percentile':np.quantile(samples,[.025,.975]).tolist(),
                'per_seed_mean_delta':np.mean(deltas,axis=1).tolist(),'per_layout_seed_mean_delta':per_layout.tolist(),
                'interpretation':'negative favors first condition; eight development layouts, seeds averaged within layout; descriptive, no multiplicity correction'}
        output[f'{a}_minus_{b}_direct_inference']=result
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); output=args.output_dir.absolute(); dataset=args.dataset_dir.absolute()
    if output.parent!=ROOT/'.generated' or not output.name.startswith('go2_rgb_body_learning_comparison_development_v1_attempt_') or output.exists():
        raise ValueError('fresh explicit development learning root required')
    if dataset!=ROOT/'.generated/go2_counterfactual_maze_dataset_development_v2_recovery_attempt_001':
        raise ValueError('this study binds exactly the audited V2 composite corpus')
    audit=json.loads((dataset/'raw_artifact_audit.json').read_text())
    if audit['status']!='PASS' or audit['audited_trials']!=120: raise ValueError('raw audit prerequisite')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output.mkdir()
    launch={'schema':'rgb_body_learning_comparison_development.v1','dataset':str(dataset.relative_to(ROOT)),
        'dataset_result_sha256':digest(dataset/'result.json'),'dataset_audit_sha256':digest(dataset/'raw_artifact_audit.json'),
        'source_sha256':{name:digest(ROOT/name) for name in SOURCES},'conditions':CONDITIONS,'seeds':SEEDS,
        'epochs':EPOCHS,'updates_per_epoch':5,'batch_layouts':16,'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.0001,'gradient_clip_norm':5.},
        'loss_weights':{'direct':1.,'rollout_when_active':1.,'latent_when_jepa':1.,'variance':.1,'covariance':.01},
        'ema_momentum':.99,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':1,
        'checkpoint_selection':'final fixed epoch only; no validation selection','scope':'offline development, not online navigation or hardware'}
    write_json(output/'launch.json',launch)
    runs=[]; start=time.monotonic()
    try:
        train=materialize(dataset,'train'); validation=materialize(dataset,'validation')
        if {m['layout_id'] for m in train['metadata']} & {m['layout_id'] for m in validation['metadata']}:
            raise ValueError('cross-role layout leakage')
        baseline_predictions=simple_predictions(train,validation)
        baselines={k:prediction_metrics(v,validation['targets'],validation['metadata']) for k,v in baseline_predictions.items()}
        np.savez_compressed(output/'baseline_predictions.npz',**baseline_predictions)
        for seed in SEEDS:
            for condition in CONDITIONS:
                directory=output/f'{seed}-{condition}'; directory.mkdir()
                torch.manual_seed(seed); model=RGBBodyJEPAReference(); model.train()
                parameters=active_parameters(model,condition)
                optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.0001)
                history=[]; fit_start=time.monotonic()
                for epoch in range(EPOCHS):
                    for indices in layout_batches(train['metadata'],epoch,seed):
                        optimizer.zero_grad(set_to_none=True)
                        loss,parts=training_loss(model,take(train,indices),condition)
                        loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
                        optimizer.step(); model.update_target(.99)
                        history.append({'update':len(history)+1,'epoch':epoch+1,'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),**parts})
                    if (epoch+1)%10==0:
                        print(json.dumps({'event':'training_epoch','seed':seed,'condition':condition,'epoch':epoch+1,'total_epochs':EPOCHS,'last_loss':history[-1]['loss']}),flush=True)
                fit_seconds=time.monotonic()-fit_start
                model.eval(); torch.save({'model_state_dict':model.state_dict(),'seed':seed,'condition':condition,'updates':len(history),
                    'launch_sha256':digest(output/'launch.json')},directory/'final.pt')
                write_json(directory/'training_history.json',history)
                val_metrics={}; stored={}
                for control in ('intact','rgb_shuffle','body_shuffle','action_shuffle'):
                    predictions,metrics=evaluate(model,validation,condition,control)
                    val_metrics[control]=metrics
                    stored.update({f'{control}__{k}':v for k,v in predictions.items()})
                np.savez_compressed(directory/'validation_predictions.npz',**stored)
                _,train_metrics=evaluate(model,train,condition)
                row={'seed':seed,'condition':condition,'updates':len(history),'active_trainable_parameters':sum(p.numel() for p in parameters),
                    'total_stored_parameters_including_inactive_and_ema':sum(p.numel() for p in model.parameters()),'fit_seconds':fit_seconds,
                    'train':train_metrics,'validation':val_metrics,'latent_diagnostics':{'train':latent_diagnostics(model,train,condition),
                        'validation':latent_diagnostics(model,validation,condition)},'inference_latency':latency(model,validation,condition),
                    'artifact_sha256':{name:digest(directory/name) for name in ('final.pt','training_history.json','validation_predictions.npz')}}
                write_json(directory/'result.json',row); runs.append(row)
                print(json.dumps({'event':'model_finished','seed':seed,'condition':condition,'completed':len(runs),'total':9,
                    'validation_direct':val_metrics['intact']['direct']['layout_macro']}),flush=True)
        if any(digest(ROOT/name)!=expected for name,expected in launch['source_sha256'].items()): raise ValueError('bound learning source changed')
        if digest(dataset/'result.json')!=launch['dataset_result_sha256'] or digest(dataset/'raw_artifact_audit.json')!=launch['dataset_audit_sha256']:
            raise ValueError('bound dataset evidence changed')
        result={'status':'COMPLETE','models':runs,'baselines':baselines,'paired_comparisons':paired_summary(runs),
            'elapsed_seconds':time.monotonic()-start,'launch_sha256':digest(output/'launch.json'),
            'baseline_predictions_sha256':digest(output/'baseline_predictions.npz'),
            'validation_order':validation['metadata'],'scope':'offline development comparison; no executed learned controller or final maze claim'}
        write_json(output/'result.json',result)
        print(json.dumps({'status':'COMPLETE','models':len(runs),'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except Exception as error:
        write_json(output/'result.json',{'status':'INFRASTRUCTURE_FAILURE','error':str(error),'models_completed':len(runs),
            'launch_sha256':digest(output/'launch.json'),'elapsed_seconds':time.monotonic()-start})
        raise


if __name__=='__main__': main()
