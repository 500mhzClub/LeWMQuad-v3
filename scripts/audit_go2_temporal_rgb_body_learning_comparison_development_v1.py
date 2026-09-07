#!/usr/bin/env python3
"""Replay completed temporal checkpoints and independently check primary scores."""
import argparse
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from lewm.temporal_rgb_body_learning_development import active_parameters,layout_batches
from lewm.temporal_prediction_metrics_development import prediction_report,simple_predictions,initial_decisions
from scripts.run_go2_temporal_rgb_body_learning_comparison_development_v1 import (
    OUTPUT,SEEDS,CONDITIONS,EPOCHS,digest,write_json,source_closure,input_bindings,materialize,
    predictions,scene_latents,paired_summary,state_identity)


def check(value,message):
    if not value: raise ValueError(message)


def scalar_primary_scores(prediction,batch):
    """Scalar references, independent of vectorized metric reduction."""
    rows=[]; t=batch['targets']; active=np.asarray(batch['known_action_valid']).all(-1)
    for layout in sorted({m['layout_id'] for m in batch['metadata']}):
        motion=[]; contact=[]
        for i,m in enumerate(batch['metadata']):
            if m['layout_id']!=layout or m['offset_ns']==0: continue
            for h in range(8):
                if not active[i,h]: continue
                if t['motion_valid'][i,h]:
                    x=float(prediction[i,h,0])-float(t['motion'][i,h,0]); y=float(prediction[i,h,1])-float(t['motion'][i,h,1])
                    motion.append(math.hypot(x,y))
                if t['contact_valid'][i,h]:
                    logit=max(-60.,min(60.,float(prediction[i,h,4])))
                    p=1/(1+math.exp(-logit)); contact.append((p-float(t['contact'][i,h]))**2)
        rows.append({'layout_id':layout,'position_error_m':sum(motion)/len(motion) if motion else None,
            'contact_brier':sum(contact)/len(contact) if contact else None})
    return rows


def check_primary(prediction,batch,report):
    independent=scalar_primary_scores(prediction,batch)
    check([r['layout_id'] for r in independent]==[r['layout_id'] for r in report['later']['layouts']],'primary layout order')
    for key in ('position_error_m','contact_brier'):
        for a,b in zip(independent,report['later']['layouts'],strict=True):
            check(a[key] is None and b[key] is None or a[key] is not None and b[key] is not None and abs(a[key]-b[key])<1e-12,'independent primary layout metric')
        values=[r[key] for r in independent if r[key] is not None]
        expected=sum(values)/len(values) if values else None
        actual=report['later']['layout_macro'][key]
        check(expected is None and actual is None or expected is not None and actual is not None and abs(expected-actual)<1e-12,'independent primary macro metric')


def check_choices(prediction,batch,result):
    for row in result['rows']:
        indices=sorted([i for i,m in enumerate(batch['metadata']) if m['layout_id']==row['layout_id'] and m['offset_ns']==0],key=lambda i:batch['metadata'][i]['action_index'])
        check(len(indices)==5,'initial candidate population')
        goal={'forward':(.8,0.),'left':(0.,.8),'right':(0.,-.8)}[row['intent']]
        estimated=[]; realized=[]
        for i in indices:
            p=1/(1+math.exp(-max(-60.,min(60.,float(prediction[i,-1,4])))))
            estimated.append(10*p+math.hypot(float(prediction[i,-1,0])-goal[0],float(prediction[i,-1,1])-goal[1]))
            if batch['targets']['contact'][i,-1]: realized.append(10.)
            else:
                xy=batch['targets']['motion'][i,-1,:2]
                realized.append(math.hypot(float(xy[0])-goal[0],float(xy[1])-goal[1]))
        chosen=min(range(5),key=estimated.__getitem__); oracle=min(range(5),key=realized.__getitem__)
        check(chosen==row['chosen_action_index'] and oracle==row['oracle_action_index'],'independent chosen/oracle action')
        check(np.allclose(estimated,row['predicted_candidate_costs'],rtol=0,atol=1e-12)
            and np.allclose(realized,row['realized_candidate_costs'],rtol=0,atol=1e-12),'independent action costs')
        check(abs(row['regret']-(realized[chosen]-realized[oracle]))<1e-12,'independent regret')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-models',type=int,default=9)
    args=parser.parse_args(); count=args.completed_models
    check(1<=count<=9,'bounded completed-model audit count')
    audit_path=OUTPUT/('raw_artifact_audit.json' if count==9 else f'interim_audit_{count:02d}.json')
    check(not audit_path.exists(),'audit already exists')
    launch=json.loads((OUTPUT/'launch.json').read_text())
    check(launch['source_sha256']==source_closure() and launch['input_sha256']==input_bindings(),'source/input bindings')
    check(launch['seeds']==list(SEEDS) and launch['conditions']==list(CONDITIONS) and launch['updates_per_model']==1200,'fixed study identity')
    root_result=None
    if count==9:
        root_result=json.loads((OUTPUT/'result.json').read_text()); check(root_result['status']=='COMPLETE' and len(root_result['models'])==9,'full terminal study')
        check(root_result['launch_sha256']==digest(OUTPUT/'launch.json'),'terminal launch binding')
        for name,sha in root_result['artifact_sha256'].items():
            check(name in ('schedules.json','validation_order.json','baseline_result.json','baseline_predictions.npz') and digest(OUTPUT/name)==sha,'root artifact binding')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    train=materialize('train'); validation=materialize('validation')
    schedule=json.loads((OUTPUT/'schedules.json').read_text())
    expected={str(seed):[ids for epoch in range(EPOCHS) for ids in layout_batches(train['metadata'],epoch,seed)] for seed in SEEDS}
    check(schedule=={'train_order':train['metadata'],'indices':expected},'all three exact schedules')
    check(json.loads((OUTPUT/'validation_order.json').read_text())==validation['metadata'],'validation order')
    baseline,coverage=simple_predictions(train,validation)
    baseline_result=json.loads((OUTPUT/'baseline_result.json').read_text())
    check(coverage==baseline_result['empirical_fallback'],'empirical coverage')
    with np.load(OUTPUT/'baseline_predictions.npz',allow_pickle=False) as archive:
        check(set(archive.files)==set(baseline),'baseline prediction population')
        for name,prediction in baseline.items():
            check(np.array_equal(prediction,archive[name]),'training-only control replay')
            report=prediction_report(prediction,validation); choices=initial_decisions(prediction,validation)
            check(baseline_result['controls'][name]=={'prediction':report,'initial_decisions':choices},'baseline metrics/choices')
            check_primary(prediction,validation,report); check_choices(prediction,validation,choices)
    rows=[]; audits=[]
    combinations=[(seed,condition) for seed in SEEDS for condition in CONDITIONS]
    for index,(seed,condition) in enumerate(combinations[:count]):
        directory=OUTPUT/f'{seed}-{condition}'; row=json.loads((directory/'result.json').read_text())
        check(row['seed']==seed and row['condition']==condition and row['updates']==1200,'model identity/budget')
        check(set(row['artifact_sha256'])=={'final.pt','updates.jsonl','training_history.json','validation_predictions.npz'},'model artifact population')
        for name,sha in row['artifact_sha256'].items(): check(digest(directory/name)==sha,'model artifact binding')
        history=json.loads((directory/'training_history.json').read_text())
        lines=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
        check(history==lines and len(history)==1200,'complete incremental update history')
        keys={'update','epoch','loss','gradient_norm_before_clip','direct_outcome','variance','covariance'}
        if condition!='direct': keys.add('rollout_outcome')
        # A sampled minibatch can theoretically have no valid future observations;
        # latent_prediction is absent in that case, while update budget remains fixed.
        for update,entry in enumerate(history,1):
            actual=set(entry); allowed=keys|({'latent_prediction'} if condition=='jepa' else set())
            check(keys<=actual<=allowed and entry['update']==update and entry['epoch']==(update-1)//5+1,'update sequence/terms')
            check(all(math.isfinite(v) for v in entry.values()) and entry['gradient_norm_before_clip']>=0,'finite history')
            expected_loss=entry['direct_outcome']+.1*entry['variance']+.01*entry['covariance']+entry.get('rollout_outcome',0)+entry.get('latent_prediction',0)
            check(abs(expected_loss-entry['loss'])<1e-5*(1+abs(expected_loss)),'loss accounting')
        torch.manual_seed(seed); model=TemporalRGBBodyJEPA()
        check(state_identity(model)==row['initial_state_sha256'],'paired seeded initialization')
        payload=(directory/'final.pt').read_bytes(); checkpoint=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
        check(checkpoint['launch_sha256']==digest(OUTPUT/'launch.json') and checkpoint['schedule_sha256']==digest(OUTPUT/'schedules.json')==row['schedule_sha256'],'checkpoint launch/schedule')
        check(checkpoint['seed']==seed and checkpoint['condition']==condition and checkpoint['updates']==1200
            and checkpoint['initial_state_sha256']==row['initial_state_sha256'],'checkpoint identity')
        check(all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values()),'finite checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
        check(sum(p.numel() for p in active_parameters(model,condition))==row['active_trainable_parameters'],'active parameter count')
        max_difference=0.
        with np.load(directory/'validation_predictions.npz',allow_pickle=False) as archive:
            heads=('direct',) if condition=='direct' else ('direct','rollout')
            check(set(archive.files)=={f'{control}__{head}' for control in ('intact','rgb_shuffle','body_shuffle') for head in heads}
                |{'intact__context_latents','shuffle_eligible'},'prediction population')
            for control in ('intact','rgb_shuffle','body_shuffle'):
                replay,z,eligible=predictions(model,validation,condition,control)
                check(np.array_equal(eligible,archive['shuffle_eligible']),'shuffle coverage')
                for head,prediction in replay.items():
                    delta=float(np.max(np.abs(prediction-archive[f'{control}__{head}'])))
                    max_difference=max(max_difference,delta); check(delta==0.,'exact checkpoint prediction replay')
                    metrics=prediction_report(prediction,validation,None if control=='intact' else eligible)
                    check(metrics==row['validation'][control][head],'all mask-aware prediction metrics')
                    if control=='intact':
                        check_primary(prediction,validation,metrics)
                        check(prediction_report(prediction,validation,eligible)==row['validation']['intact_matched_shuffle'][head],'matched intact shuffle population')
                        choices=initial_decisions(prediction,validation)
                        check(choices==row['initial_decisions'][head],'initial choices replay'); check_choices(prediction,validation,choices)
                if control=='intact':
                    check(np.array_equal(z,archive['intact__context_latents']) and scene_latents(z,validation['metadata'])==row['scene_latents'],'scene latent diagnostics')
        train_prediction,_,_=predictions(model,train,condition)
        check({head:prediction_report(p,train) for head,p in train_prediction.items()}==row['train'],'training score replay')
        check(math.isfinite(row['fit_seconds']) and row['fit_seconds']>0,'fit time')
        for latency in row['inference_latency'].values():
            check(0<latency['median_ms']<=latency['p95_ms'] and math.isfinite(latency['p95_ms']),'latency summary bounds')
        if root_result is not None: check(root_result['models'][index]==row,'root/model result identity')
        rows.append(row); audits.append({'seed':seed,'condition':condition,'status':'PASS','maximum_prediction_difference':max_difference,
            'result_sha256':digest(directory/'result.json'),'checkpoint_sha256':digest(directory/'final.pt')})
        print(json.dumps({'event':'model_audited','completed':len(rows),'requested':count,'seed':seed,'condition':condition}),flush=True)
    if root_result is not None:
        check(root_result['paired_comparisons']==paired_summary(rows),'paired layout comparison replay')
        check(root_result['baselines']==baseline_result['controls'] and root_result['empirical_fallback']==coverage,'root baselines')
    check(launch['source_sha256']==source_closure() and launch['input_sha256']==input_bindings(),'terminal source/input identity')
    audit={'status':'PASS','audited_models':count,'models':audits,'full_study':count==9,
        'study_result_sha256':digest(OUTPUT/'result.json') if count==9 else None,'launch_sha256':digest(OUTPUT/'launch.json'),
        'schedule_sha256':digest(OUTPUT/'schedules.json'),'audit_source_sha256':digest(Path(__file__)),
        'scope':'checkpoint/prediction/schedule and scalar primary-score audit; no online or hardware claim'}
    write_json(audit_path,audit); print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
