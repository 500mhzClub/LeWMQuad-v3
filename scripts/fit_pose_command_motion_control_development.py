"""Separate future control: predict visual motion without neural forecast features.

Uses the original residual study's identical training/validation windows and
ridge penalty. This fit is not used in the running sixteen-trial native study.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.fit_closed_loop_motion_residual_development import TRAIN, VALIDATION, metrics
from scripts.compare_continuous_navigation_arms_development import path

SOURCE=path('go2_closed_loop_motion_residual_study_v1_attempt_001')
OUTPUT=path('go2_pose_command_motion_control_fit_v1_attempt_001')
# Original feature order: 12 past-pose, 4 neural XY/yaw, 24 known commands,
# and 2 command-integrated nominal coordinates. Remove all neural features.
COLUMNS=np.r_[np.arange(12),np.arange(16,42)]


def load(name):
    p=SOURCE/(name+'.npz')
    identity=hashlib.sha256(p.read_bytes()).hexdigest()
    with np.load(p,allow_pickle=False) as a:
        if a['features'].shape[1:]!=(8,42):raise ValueError('original feature layout required')
        d=dict(features=a['features'][:,:,COLUMNS].copy(),target=a['target'].copy(),valid=a['valid'].copy())
    return d,identity


def main():
    OUTPUT.mkdir()
    launch=dict(training_roots=list(TRAIN),validation_root=VALIDATION,
        target='subsequent_registered_visual_pose_deltas',ridge_penalty=1.,
        original_feature_columns=COLUMNS.tolist(),neural_forecast_features_used=False,
        native_state_used=False,source_study=SOURCE.name,
        running_native_cohort_uses_this_fit=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    datasets=[];identities={}
    for name in TRAIN:
        d,identity=load(name);datasets.append(d);identities[name]=identity
    train={k:np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    values={k:[] for k in ('mean','scale','bias','coefficient')}
    for h in range(8):
        valid=train['valid'][:,h];x=train['features'][valid,h];y=train['target'][valid,h]
        mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.;bias=y.mean(0)
        z=(x-mean)/scale
        coefficient=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(y-bias))
        for k,v in dict(mean=mean,scale=scale,bias=bias,coefficient=coefficient).items():values[k].append(v)
    fit={k:np.asarray(v) for k,v in values.items()}
    np.savez_compressed(OUTPUT/'motion_fit.npz',**fit)
    fit_hash=hashlib.sha256((OUTPUT/'motion_fit.npz').read_bytes()).hexdigest()
    # The fit is fixed before reading the unchanged validation targets.
    validation,identity=load(VALIDATION);identities[VALIDATION]=identity
    rows=json.loads((SOURCE/(VALIDATION+'.json')).read_text())
    prediction=np.stack([((validation['features'][:,h]-fit['mean'][h])/fit['scale'][h])@
        fit['coefficient'][h]+fit['bias'][h] for h in range(8)],axis=1)
    original=json.loads((SOURCE/'result.json').read_text())
    results={}
    for group in ('all','moving','transition','translation','turn_only'):
        group_mask=np.array([True if group=='all' else r[group] for r in rows])
        results[group]={}
        for h in (2,6,7):
            mask=group_mask&validation['valid'][:,h];key=str((h+1)*100)
            results[group][key]=dict(pose_command=metrics((prediction-validation['target'])[mask,h]),
                original_jepa=original['results'][group][key])
    launch['source_dataset_sha256']=identities
    with (OUTPUT/'launch.json').open('x') as f:json.dump(launch,f,indent=2)
    result=dict(status='COMPLETE',training_windows=len(train['features']),validation_windows=len(rows),
        motion_fit_sha256=fit_hash,results=results,
        future_visual_pose_labels_training_and_evaluator_only=True,
        neural_prediction_inputs_used=False,closed_loop_navigation_tested=False,
        validation_is_development=True,running_native_cohort_changed=False)
    with (OUTPUT/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({g:r['700'] for g,r in results.items()}),flush=True)


if __name__=='__main__':main()
