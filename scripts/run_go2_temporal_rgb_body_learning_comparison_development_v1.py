#!/usr/bin/env python3
"""Fixed nine-model temporal comparison on audited development windows."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.causal_subtrajectory_learning_development import AuditedSubtrajectoryDataset,DERIVATION_ROOT
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from lewm.temporal_rgb_body_learning_development import CONDITIONS,active_parameters,layout_batches,training_loss
from lewm.temporal_prediction_metrics_development import prediction_report,shuffle_population,simple_predictions,initial_decisions

OUTPUT=ROOT/'.generated/go2_temporal_rgb_body_learning_comparison_development_v1_attempt_001'
SEEDS=(2026091700,2026091701,2026091702)
EPOCHS=240
PROTOCOL='docs/go2_temporal_rgb_body_learning_comparison_development_v1_2026-09-05.md'
TESTS=('lewm/tests/test_temporal_rgb_body_jepa_development.py','lewm/tests/test_temporal_prediction_metrics_development.py',
       'lewm/tests/test_temporal_learning_runner_development.py')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def safe_path(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or any(s in ('sealed','sealed_test.json') or s.startswith('sealed_') for s in p.parts):
        raise ValueError('protected/invalid explicit source path')
    path=ROOT/p
    if path.resolve()!=path: raise ValueError('source symlink forbidden')
    return path


def source_closure():
    # Discovery honors .ignore. Only explicit imported local files are read;
    # no source export, recursive copy or protected directory access occurs.
    found=subprocess.run(['rg','--files','-g','*.py','lewm','scripts','lewm_genesis','lewm_worlds'],
        cwd=ROOT,check=True,text=True,capture_output=True).stdout.splitlines()
    available=set(found); pending=[str(Path(__file__).relative_to(ROOT)),*TESTS]; visited=set()
    prefixes=('lewm','scripts','lewm_genesis','lewm_worlds')
    while pending:
        name=pending.pop()
        if name in visited: continue
        if name not in available: raise ValueError('source outside ignore-aware discovery: '+name)
        path=safe_path(name); tree=ast.parse(path.read_text()); visited.add(name)
        parts=Path(name).parts[:-1]
        for end in range(1,len(parts)+1):
            init=str(Path(*parts[:end])/'__init__.py')
            if init in available: pending.append(init)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): modules=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=list(parts[:len(parts)-node.level+1]); module='.'.join([*base,*((node.module or '').split('.') if node.module else [])])
                else: module=node.module or ''
                modules=[module,*[module+'.'+a.name for a in node.names]]
            else: continue
            for module in modules:
                if module.split('.')[0] not in prefixes: continue
                rel=module.replace('.','/')
                for candidate in (rel+'.py',rel+'/__init__.py'):
                    if candidate in available: pending.append(candidate)
    return {name:digest(safe_path(name)) for name in sorted(visited|{PROTOCOL})}


def input_bindings():
    inputs={}
    for directory in (DERIVATION_ROOT,ROOT/'.generated/go2_temporal_model_interface_development_v1_attempt_001'):
        launch=json.loads((directory/'launch.json').read_text())
        for name,sha in launch['source_sha256'].items():
            if digest(safe_path(name))!=sha: raise ValueError('prerequisite source changed: '+name)
        for name in ('launch.json','result.json'):
            inputs[str((directory/name).relative_to(ROOT))]=digest(directory/name)
    for name in ('windows.json','raw_artifact_audit.json','tensor_interface_check.json'):
        inputs[str((DERIVATION_ROOT/name).relative_to(ROOT))]=digest(DERIVATION_ROOT/name)
    check=json.loads((DERIVATION_ROOT/'tensor_interface_check.json').read_text())
    ready=json.loads((ROOT/'.generated/go2_temporal_model_interface_development_v1_attempt_001/result.json').read_text())
    if check['status']!='PASS' or check['windows']!={'train':610,'validation':304} or ready['status']!='PASS' or not ready['parameters_unchanged']:
        raise ValueError('completed readiness prerequisites required')
    return inputs


def stack(rows):
    if isinstance(rows[0],dict): return {k:stack([r[k] for r in rows]) for k in rows[0]}
    return torch.stack(rows)


def take(value,indices):
    if isinstance(value,dict): return {k:take(v,indices) for k,v in value.items()}
    if isinstance(value,list): return [value[int(i)] for i in indices]
    return value[indices]


def materialize(role):
    data=AuditedSubtrajectoryDataset(DERIVATION_ROOT,role); rows=[]
    for i in range(len(data)):
        rows.append(data[i])
        if (i+1)%100==0: print(json.dumps({'event':'window_loaded','role':role,'completed':i+1,'total':len(data)}),flush=True)
    metadata=[r.pop('metadata') for r in rows]; result=stack(rows); result['metadata']=metadata
    if len(metadata)!=(610 if role=='train' else 304): raise ValueError('fixed full temporal population')
    for layout in {m['layout_id'] for m in metadata}:
        indices=[i for i,m in enumerate(metadata) if m['layout_id']==layout and m['offset_ns']==0]
        if sorted(metadata[i]['action_index'] for i in indices)!=list(range(5)): raise ValueError('initial sibling population')
        for value in result['observation_history'].values():
            if any(not torch.equal(value[indices[0]],value[i]) for i in indices): raise ValueError('canonical sibling history differs')
    return result


@torch.no_grad()
def predictions(model,batch,condition,control='intact'):
    donors,eligible=shuffle_population(batch['metadata'])
    n=len(batch['metadata']); outputs={'direct':[]}; latents=[]
    if condition!='direct': outputs['rollout']=[]
    if control not in ('intact','rgb_shuffle','body_shuffle'): raise ValueError('undeclared control')
    for start in range(0,n,16):
        indices=list(range(start,min(start+16,n))); part=take(batch,indices)
        history=part['observation_history']
        if control!='intact':
            name='rgb' if control=='rgb_shuffle' else 'body'
            history[name]=batch['observation_history'][name][donors[indices]]
        plans,valid=part['known_action_blocks'],part['known_action_valid']; active=valid.all(-1)
        if condition=='direct':
            z,_=model.encode_history(history); prediction=model.direct(z,plans)
            result={'latent':z,'direct_outcomes':torch.where(active[:,:,None],prediction,torch.zeros_like(prediction))}
        else: result=model(history,plans,valid)
        latents.append(result['latent'].numpy())
        for head in outputs: outputs[head].append(result[head+'_outcomes'].numpy())
    return {k:np.concatenate(v) for k,v in outputs.items()},np.concatenate(latents),eligible


def scene_latents(z,metadata):
    _,eligible=shuffle_population(metadata); cells={}
    for action,offset in sorted({(m['action_index'],m['offset_ns']) for i,m in enumerate(metadata) if eligible[i]}):
        indices=[i for i,m in enumerate(metadata) if m['action_index']==action and m['offset_ns']==offset]
        values=z[indices].astype(float); centered=values-values.mean(0)
        energy=np.linalg.svd(centered,compute_uv=False)**2
        weights=energy/max(float(energy.sum()),1e-30)
        rank=float(np.exp(-np.sum(weights*np.log(np.maximum(weights,1e-30))))) if energy.sum()>0 else 0.
        cells[f'{action}:{offset}']={'layouts':len(indices),'mean_feature_std':float(values.std(0,ddof=1).mean()),'effective_rank':rank}
    return {'same_action_offset_cells':cells,'eligible_windows':int(eligible.sum()),'omitted_windows':int((~eligible).sum()),
        'scope':'scene dispersion, not a task-utility or noncollapse guarantee'}


@torch.no_grad()
def latency(model,batch,condition):
    part=take(batch,[0]); h=part['observation_history']; p=part['known_action_blocks']; v=part['known_action_valid']
    def direct():
        z,_=model.encode_history(h); return model.direct(z,p)
    methods={'direct':direct}
    if condition!='direct': methods['both_heads_and_rollout']=lambda:model(h,p,v)
    result={}
    for name,method in methods.items():
        for _ in range(3): method()
        times=[]
        for _ in range(20):
            start=time.perf_counter(); method(); times.append((time.perf_counter()-start)*1000)
        result[name]={'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
            'scope':'one CPU thread, one four-frame context and one full plan; excludes acquisition/conversion'}
    return result


def paired_summary(runs):
    contrasts=(('jepa','direct','supervised_rollout','direct'),('jepa','rollout','supervised_rollout','rollout'),
        ('jepa','direct','direct','direct'),('supervised_rollout','direct','direct','direct'),
        ('jepa','rollout','jepa','direct'),('supervised_rollout','rollout','supervised_rollout','direct'))
    result={}
    for a,ha,b,hb in contrasts:
        pair={}
        for endpoint in ('later_position_error_m','later_contact_brier','initial_regret','initial_contact'):
            values=[]
            for seed in SEEDS:
                arm_a=next(r for r in runs if r['seed']==seed and r['condition']==a)
                arm_b=next(r for r in runs if r['seed']==seed and r['condition']==b)
                if endpoint.startswith('later_'):
                    rows_a=arm_a['validation']['intact'][ha]['later']['layouts']; rows_b=arm_b['validation']['intact'][hb]['later']['layouts']
                    key=endpoint[len('later_'):]
                else:
                    rows_a=arm_a['initial_decisions'][ha]['layouts']; rows_b=arm_b['initial_decisions'][hb]['layouts']; key=endpoint[len('initial_'):]
                if [r['layout_id'] for r in rows_a]!=[r['layout_id'] for r in rows_b] or len(rows_a)!=8:
                    raise ValueError('paired eight-layout endpoint incomplete')
                if any(x[key] is None or y[key] is None for x,y in zip(rows_a,rows_b,strict=True)): raise ValueError('missing primary endpoint')
                values.append([x[key]-y[key] for x,y in zip(rows_a,rows_b,strict=True)])
            array=np.asarray(values); layout=array.mean(0); rng=np.random.default_rng(2026091799)
            bootstrap=layout[rng.integers(0,8,size=(10000,8))].mean(1)
            pair[endpoint]={'mean_delta':float(layout.mean()),'per_seed_mean_delta':array.mean(1).tolist(),
                'per_layout_seed_mean_delta':layout.tolist(),'layout_bootstrap_95_percentile':np.quantile(bootstrap,[.025,.975]).tolist(),
                'scope':'negative favors first; eight reused development layouts, seed-averaged, descriptive, no multiplicity correction'}
        result[f'{a}_{ha}_minus_{b}_{hb}']=pair
    return result


def state_identity(model):
    sha=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        sha.update(key.encode()); sha.update(value.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    if output!=OUTPUT or output.resolve()!=output or output.exists(): raise ValueError('fresh exact temporal training root required')
    sources=source_closure(); inputs=input_bindings()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output.mkdir(); started=time.monotonic(); runs=[]; initial_identities={}
    launch={'schema':'temporal_rgb_body_learning_comparison_development.v1','source_sha256':sources,'input_sha256':inputs,
        'seeds':SEEDS,'conditions':CONDITIONS,'epochs':EPOCHS,'updates_per_epoch':5,'updates_per_model':1200,
        'device':'cpu','threads':1,'torch':torch.__version__,'numpy':np.__version__,
        'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.0001,'gradient_clip_norm':5.},'ema_momentum':.99,
        'loss_weights':{'direct':1.,'rollout_when_active':1.,'latent_when_jepa':1.,'variance':.1,'covariance':.01},
        'checkpoint_selection':'final fixed update only','scope':'offline temporal development; no online navigation or hardware'}
    write_json(output/'launch.json',launch)
    try:
        train=materialize('train'); validation=materialize('validation')
        if {m['layout_id'] for m in train['metadata']} & {m['layout_id'] for m in validation['metadata']}: raise ValueError('role leakage')
        schedules={str(seed):[indices for epoch in range(EPOCHS) for indices in layout_batches(train['metadata'],epoch,seed)] for seed in SEEDS}
        if any(len(s)!=1200 for s in schedules.values()): raise ValueError('fixed training schedule')
        write_json(output/'schedules.json',{'train_order':train['metadata'],'indices':schedules})
        write_json(output/'validation_order.json',validation['metadata'])
        schedule_sha=digest(output/'schedules.json')
        baseline,coverage=simple_predictions(train,validation)
        baseline_result={k:{'prediction':prediction_report(v,validation),'initial_decisions':initial_decisions(v,validation)} for k,v in baseline.items()}
        write_json(output/'baseline_result.json',{'controls':baseline_result,'empirical_fallback':coverage})
        np.savez_compressed(output/'baseline_predictions.npz',**baseline)
        for seed in SEEDS:
            for condition in CONDITIONS:
                directory=output/f'{seed}-{condition}'; directory.mkdir()
                torch.manual_seed(seed); model=TemporalRGBBodyJEPA(); model.train()
                identity=state_identity(model)
                if initial_identities.setdefault(seed,identity)!=identity: raise ValueError('unpaired initial model state')
                parameters=active_parameters(model,condition); optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.0001)
                history=[]; fit_start=time.monotonic()
                with (directory/'updates.jsonl').open('x') as log:
                    for update,indices in enumerate(schedules[str(seed)],1):
                        optimizer.zero_grad(set_to_none=True); loss,parts=training_loss(model,take(train,indices),condition)
                        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
                        optimizer.step(); model.update_target(.99)
                        row={'update':update,'epoch':(update-1)//5+1,'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),**parts}
                        history.append(row); log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
                        if update%50==0: print(json.dumps({'event':'training_update','seed':seed,'condition':condition,'update':update,'total':1200,'loss':row['loss']}),flush=True)
                fit_seconds=time.monotonic()-fit_start; model.eval()
                torch.save({'model_state_dict':model.state_dict(),'seed':seed,'condition':condition,'updates':1200,
                    'launch_sha256':digest(output/'launch.json'),'schedule_sha256':schedule_sha,'initial_state_sha256':identity},directory/'final.pt')
                write_json(directory/'training_history.json',history)
                metrics={}; stored={}; choices={}; diagnostic={}
                for control in ('intact','rgb_shuffle','body_shuffle'):
                    prediction,z,eligible=predictions(model,validation,condition,control)
                    metrics[control]={head:prediction_report(p,validation,None if control=='intact' else eligible) for head,p in prediction.items()}
                    stored.update({f'{control}__{head}':p for head,p in prediction.items()})
                    if control=='intact':
                        metrics['intact_matched_shuffle']={head:prediction_report(p,validation,eligible) for head,p in prediction.items()}
                        choices={head:initial_decisions(p,validation) for head,p in prediction.items()}
                        diagnostic=scene_latents(z,validation['metadata']); stored['intact__context_latents']=z; stored['shuffle_eligible']=eligible
                np.savez_compressed(directory/'validation_predictions.npz',**stored)
                train_prediction,_,_=predictions(model,train,condition)
                train_metrics={head:prediction_report(p,train) for head,p in train_prediction.items()}
                row={'seed':seed,'condition':condition,'updates':1200,'initial_state_sha256':identity,'schedule_sha256':schedule_sha,
                    'active_trainable_parameters':sum(p.numel() for p in parameters),'fit_seconds':fit_seconds,
                    'train':train_metrics,'validation':metrics,'initial_decisions':choices,'scene_latents':diagnostic,
                    'inference_latency':latency(model,validation,condition),
                    'artifact_sha256':{name:digest(directory/name) for name in ('final.pt','updates.jsonl','training_history.json','validation_predictions.npz')}}
                write_json(directory/'result.json',row); runs.append(row)
                print(json.dumps({'event':'model_finished','seed':seed,'condition':condition,'completed':len(runs),'total':9,
                    'later_direct':metrics['intact']['direct']['later']['layout_macro']}),flush=True)
        if sources!=source_closure() or inputs!=input_bindings() or digest(output/'schedules.json')!=schedule_sha:
            raise ValueError('source/input/schedule changed')
        result={'status':'COMPLETE','models':runs,'baselines':baseline_result,'empirical_fallback':coverage,
            'paired_comparisons':paired_summary(runs),'launch_sha256':digest(output/'launch.json'),
            'schedule_sha256':schedule_sha,'elapsed_seconds':time.monotonic()-started,
            'artifact_sha256':{name:digest(output/name) for name in ('schedules.json','validation_order.json','baseline_result.json','baseline_predictions.npz')},
            'scope':'offline development; not repeated physical decisions, final maze generalization or hardware'}
        write_json(output/'result.json',result); print(json.dumps({'status':'COMPLETE','models':9,'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except Exception as exc:
        write_json(output/'result.json',{'status':'INFRASTRUCTURE_FAILURE','error':repr(exc),'models_completed':len(runs),
            'launch_sha256':digest(output/'launch.json'),'elapsed_seconds':time.monotonic()-started}); raise


if __name__=='__main__': main()
