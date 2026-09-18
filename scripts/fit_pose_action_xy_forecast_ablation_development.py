"""Measure incremental neural-forecast information on existing development windows."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.fit_closed_loop_motion_residual_development import TRAIN,VALIDATION,OUTPUT as SOURCE,metrics

OUTPUT=SOURCE.parent/'go2_pose_action_xy_forecast_ablation_study_v1_attempt_001'
VARIANTS={'all_forecast_features':np.arange(42),
    'pose_history_and_commands':np.r_[np.arange(12),np.arange(16,42)]}


def read(name):
    with np.load(SOURCE/(name+'.npz'),allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def main():
    if OUTPUT.exists():raise ValueError('preserve prior study')
    OUTPUT.mkdir()
    datasets=[read(name) for name in TRAIN]
    train={k:np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    fits={}
    for name,columns in VARIANTS.items():
        values={k:[] for k in ('mean','scale','bias','coefficient')}
        for h in range(8):
            mask=train['valid'][:,h];x=train['features'][mask,h][:,columns];y=train['target'][mask,h]
            mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.;bias=y.mean(0)
            z=(x-mean)/scale
            coefficient=np.linalg.solve(z.T@z+np.eye(len(columns)),z.T@(y-bias))
            for k,value in zip(values,(mean,scale,bias,coefficient)):values[k].append(value)
        fits[name]={k:np.asarray(v) for k,v in values.items()}|dict(columns=columns)
        np.savez_compressed(OUTPUT/(name+'.npz'),**fits[name])
    # Both fixed-penalty absolute-prediction fits are saved before validation
    # is loaded. The original hybrid has a different residual-centred prior.
    validation=read(VALIDATION);rows=json.loads((SOURCE/(VALIDATION+'.json')).read_text())
    predictions={'original_neural_xy':validation['prediction']}
    with np.load(SOURCE/'residual_fit.npz',allow_pickle=False) as d:
        predictions['original_hybrid']=validation['prediction'].copy()
        for h in range(8):predictions['original_hybrid'][:,h]+=((validation['features'][:,h]-d['mean'][h])/d['scale'][h])@d['coefficient'][h]+d['bias'][h]
    for name,fit in fits.items():
        predictions[name]=np.stack([((validation['features'][:,h,fit['columns']]-fit['mean'][h])/fit['scale'][h])@
            fit['coefficient'][h]+fit['bias'][h] for h in range(8)],axis=1)
    results={}
    for group in ('all','moving','transition','translation','turn_only'):
        selected=np.asarray([True if group=='all' else row[group] for row in rows])
        results[group]={}
        for h in (2,6,7):
            mask=selected&validation['valid'][:,h]
            results[group][str((h+1)*100)]={name:metrics((prediction-validation['target'])[mask,h])
                for name,prediction in predictions.items()}
    report=dict(status='COMPLETE',training_roots=TRAIN,validation_root=VALIDATION,
        training_windows=len(train['features']),validation_windows=len(rows),ridge_penalty=1.,
        variants={k:v.tolist() for k,v in VARIANTS.items()},results=results,
        native_state_read=False,targets='subsequent_registered_visual_pose_deltas',
        current_native_run_unchanged=True,prospective_navigation_tested=False,
        validation_is_previously_used_development_data=True,
        identical_absolute_target_objective_for_two_new_fits=True,
        original_hybrid_uses_different_residual_centred_prior=True,
        source_study_result_sha256=hashlib.sha256((SOURCE/'result.json').read_bytes()).hexdigest())
    with (OUTPUT/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'training_windows':report['training_windows'],'validation_windows':len(rows),
        'translation_700_ms':results['translation']['700'],'turn_only_700_ms':results['turn_only']['700']}))


if __name__=='__main__':main()
