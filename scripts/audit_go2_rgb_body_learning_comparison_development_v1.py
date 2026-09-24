#!/usr/bin/env python3
"""Reload this study's bound checkpoints and independently reduce primary metrics."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_go2_rgb_body_learning_comparison_development_v1 import (
    CONDITIONS,SEEDS,SOURCES,RGBBodyJEPAReference,active_parameters,digest,evaluate,
    latent_diagnostics,materialize,paired_summary,simple_predictions,write_json,
)


def check(value,message):
    if not value: raise ValueError(message)


def scalar_metrics(prediction,targets,metadata):
    """Separate scalar formula, not the fitting runner's NumPy reduction."""
    grouped={name:[] for name in sorted({m['layout_id'] for m in metadata})}
    for index,meta in enumerate(metadata):
        grouped[meta['layout_id']].append(index)
    rows=[]
    for layout,indices in grouped.items():
        position,yaw,brier,correct=[],[],[],[]
        for index in indices:
            for h in range(prediction.shape[1]):
                px,py,sine,cosine,logit=map(float,prediction[index,h])
                if bool(targets['motion_valid'][index,h]):
                    tx,ty,angle=map(float,targets['motion'][index,h])
                    position.append(math.hypot(px-tx,py-ty))
                    delta=math.atan2(sine,cosine)-angle
                    yaw.append(abs(math.atan2(math.sin(delta),math.cos(delta))))
                if bool(targets['contact_valid'][index,h]):
                    truth=float(targets['contact'][index,h]); probability=1/(1+math.exp(-max(-60.,min(60.,logit))))
                    brier.append((probability-truth)**2); correct.append(float((probability>=.5)==truth))
        row={'layout_id':layout,'motion_count':len(position),'contact_count':len(brier)}
        for key,values in [('position_error_m',position),('yaw_error_rad',yaw),('contact_brier',brier),('contact_accuracy_at_half',correct)]:
            row[key]=math.fsum(values)/len(values) if values else None
        rows.append(row)
    keys=('position_error_m','yaw_error_rad','contact_brier','contact_accuracy_at_half')
    macro={k:math.fsum(r[k] for r in rows if r[k] is not None)/sum(r[k] is not None for r in rows) for k in keys}
    return rows,macro


def check_metrics(prediction,targets,metadata,recorded):
    rows,macro=scalar_metrics(prediction,targets,metadata)
    check(len(rows)==len(recorded['layouts']),'layout reduction population')
    for actual,witness in zip(rows,recorded['layouts'],strict=True):
        check(actual.keys()==witness.keys(),'layout metric schema')
        for key,value in actual.items():
            if isinstance(value,float): check(math.isclose(value,witness[key],rel_tol=0,abs_tol=1e-10),f'scalar metric {key}')
            else: check(value==witness[key],f'scalar coverage {key}')
    for key,value in macro.items(): check(math.isclose(value,recorded['layout_macro'][key],rel_tol=0,abs_tol=1e-10),f'macro metric {key}')
    for h in range(prediction.shape[1]):
        _,by_horizon=scalar_metrics(prediction[:,h:h+1],{k:v[:,h:h+1] for k,v in targets.items() if k in ('motion','contact','motion_valid','contact_valid')},metadata)
        for key,value in by_horizon.items():
            check(math.isclose(value,recorded['by_horizon_seconds'][str((h+1)*.5)]['layout_macro'][key],rel_tol=0,abs_tol=1e-10),'horizon metric')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True)
    output=parser.parse_args().output_dir.absolute()
    check(output==ROOT/'.generated/go2_rgb_body_learning_comparison_development_v1_attempt_001','exact learning-study root required')
    check(not (output/'prediction_artifact_audit.json').exists(),'audit already exists')
    launch=json.loads((output/'launch.json').read_text()); result=json.loads((output/'result.json').read_text())
    check(result['status']=='COMPLETE' and len(result['models'])==9,'learning study incomplete')
    check(result['launch_sha256']==digest(output/'launch.json'),'launch binding')
    check(set(launch['source_sha256'])==set(SOURCES),'source population')
    for name,expected in launch['source_sha256'].items(): check(digest(ROOT/name)==expected,f'source binding {name}')
    dataset=ROOT/'.generated/go2_counterfactual_maze_dataset_development_v2_recovery_attempt_001'
    check(launch['dataset']==str(dataset.relative_to(ROOT)),'dataset identity')
    check(digest(dataset/'result.json')==launch['dataset_result_sha256'] and digest(dataset/'raw_artifact_audit.json')==launch['dataset_audit_sha256'],'dataset binding')
    check(launch['seeds']==list(SEEDS) and launch['conditions']==list(CONDITIONS) and launch['epochs']==60,'fixed fit population')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    train=materialize(dataset,'train'); validation=materialize(dataset,'validation')
    check(validation['metadata']==result['validation_order'],'validation ordering')
    check(digest(output/'baseline_predictions.npz')==result['baseline_predictions_sha256'],'baseline artifact binding')
    with np.load(output/'baseline_predictions.npz',allow_pickle=False) as stored:
        baseline=simple_predictions(train,validation)
        check(set(stored.files)==set(baseline),'baseline population')
        for name,values in baseline.items():
            check(np.array_equal(stored[name],values),'baseline recomputation')
            check_metrics(values,validation['targets'],validation['metadata'],result['baselines'][name])
    rows=[]
    for i,(seed,condition) in enumerate((seed,condition) for seed in SEEDS for condition in CONDITIONS):
        directory=output/f'{seed}-{condition}'; row=json.loads((directory/'result.json').read_text())
        check(row==result['models'][i] and row['seed']==seed and row['condition']==condition,'model result population')
        check(row['updates']==300,'fixed update budget')
        check(set(row['artifact_sha256'])=={'final.pt','training_history.json','validation_predictions.npz'},'model artifact population')
        for name,expected in row['artifact_sha256'].items(): check(digest(directory/name)==expected,f'model artifact binding {name}')
        history=json.loads((directory/'training_history.json').read_text())
        check(len(history)==300,'optimizer history length')
        for update,entry in enumerate(history):
            check(entry['update']==update+1 and entry['epoch']==update//5+1,'optimizer history sequence')
            expected=entry['direct_outcome']+.1*entry['variance']+.01*entry['covariance']
            if condition!='direct': expected+=entry['rollout_outcome']
            if condition=='jepa': expected+=entry['latent_prediction']
            check(math.isclose(entry['loss'],expected,abs_tol=2e-5,rel_tol=1e-6),'recorded loss decomposition')
            check(all(math.isfinite(float(v)) for v in entry.values()),'nonfinite optimizer history')
        checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
        check(checkpoint['seed']==seed and checkpoint['condition']==condition and checkpoint['updates']==300
            and checkpoint['launch_sha256']==digest(output/'launch.json'),'checkpoint identity')
        model=RGBBodyJEPAReference(); model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
        check(all(torch.isfinite(p).all() for p in model.parameters()),'nonfinite checkpoint')
        check(row['active_trainable_parameters']==sum(p.numel() for p in active_parameters(model,condition)),'active parameter count')
        with np.load(directory/'validation_predictions.npz',allow_pickle=False) as stored:
            expected_keys={f'{control}__{head}' for control in ('intact','rgb_shuffle','body_shuffle','action_shuffle')
                for head in (('direct',) if condition=='direct' else ('direct','rollout'))}
            check(set(stored.files)==expected_keys,'prediction population')
            for control in ('intact','rgb_shuffle','body_shuffle','action_shuffle'):
                predictions,metrics=evaluate(model,validation,condition,control)
                check(metrics==row['validation'][control],'prediction metric regeneration')
                for head,prediction in predictions.items():
                    check(np.array_equal(prediction,stored[f'{control}__{head}']),'checkpoint prediction regeneration')
                    check_metrics(prediction,validation['targets'],validation['metadata'],row['validation'][control][head])
        train_predictions,train_metrics=evaluate(model,train,condition)
        check(train_metrics==row['train'],'training metrics regeneration')
        for head,prediction in train_predictions.items(): check_metrics(prediction,train['targets'],train['metadata'],row['train'][head])
        check(latent_diagnostics(model,train,condition)==row['latent_diagnostics']['train'],'train latent diagnostic')
        check(latent_diagnostics(model,validation,condition)==row['latent_diagnostics']['validation'],'validation latent diagnostic')
        rows.append({'seed':seed,'condition':condition,'checkpoint_sha256':digest(directory/'final.pt'),'status':'PASS'})
        print(json.dumps({'event':'model_audited','completed':i+1,'total':9,'seed':seed,'condition':condition}),flush=True)
    check(paired_summary(result['models'])==result['paired_comparisons'],'paired layout summary')
    audit={'status':'PASS','audited_models':9,'models':rows,'study_result_sha256':digest(output/'result.json'),
        'audit_source_sha256':digest(Path(__file__)),'scope':'checkpoint/prediction replay, scalar metric recomputation, bindings and recorded optimization budget; not independent training reruns or online navigation evidence'}
    write_json(output/'prediction_artifact_audit.json',audit); print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
