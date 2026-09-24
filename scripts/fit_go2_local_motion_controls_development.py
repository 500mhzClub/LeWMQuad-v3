"""Matched-data linear controls with and without measured local motion.

All contexts remain, including unavailable histories. Same fixed draw weights;
ridge penalty one, with no tuning or outcome-dependent sample selection.
"""
import json
import hashlib
import numpy as np
from scripts.pre_switch_training_data_development import BASE,ROOTS,load_training_rows
from scripts.prepare_go2_pre_switch_transfer_development import OUTPUT as TRANSFER
from scripts.derive_go2_causal_local_motion_inputs_development import OUTPUT as FEATURES

OUTPUT=BASE/'go2_local_motion_matched_controls_v1_attempt_001'


def nominal(commands):
    values=np.zeros((8,3));state=np.zeros(3)
    for h,(vx,vy,w) in enumerate(commands):
        angle=state[2]+.05*w
        state[:2]+=.1*np.array([vx*np.cos(angle)-vy*np.sin(angle),vx*np.sin(angle)+vy*np.cos(angle)])
        state[2]+=.1*w;values[h]=state
    return values


def dataset(rows,features):
    xs=[];ys=[];masks=[];nominals=[];available=[]
    controls={}
    for row in rows:
        feature=features[row['sample_id']];ok=feature['history_available']
        if feature['data_role']!=row['data_role'] or feature['measured_ns']!=row['observation_horizon_receipt']['departure_ns']:
            raise ValueError('same role and causal feature timestamp required')
        past=np.asarray(feature['history_features']) if ok else np.zeros(12)
        if past.shape!=(12,) or not np.isfinite(past).all():raise ValueError('finite measured or explicitly missing history')
        key=(row['source'],row['trial'])
        if key not in controls:
            with np.load(ROOTS[key[0]]/key[1]/'policy_histories.npz',allow_pickle=False) as archive:
                controls[key]={k:archive[k].copy() for k in ('decision_ns','applied_command_values',
                    'applied_command_valid','applied_command_measured_ns')}
        arrays=controls[key];past_controls=[]
        for f in feature['history_frames']:
            now=arrays['decision_ns'][f];measured=arrays['applied_command_measured_ns'][f]
            age=np.where(measured>=0,(now-measured)/1e9,1.5)[:,None]
            past_controls.append(np.concatenate((arrays['applied_command_values'][f]/[.3,1.,.5],
                arrays['applied_command_valid'][f].astype(float),age),axis=1).ravel())
        command_history=np.concatenate(past_controls)
        commands=row['known_commands'];base=nominal(commands);x=[];target=[];valid=[]
        for h in range(8):
            known=np.zeros((8,3));n=min(h+1,len(commands));known[:n]=commands[:n]
            x.append(np.r_[past,float(ok),known.ravel(),base[h],command_history])
            t=row['targets'][h];valid.append(t['motion_valid'])
            if t['motion_valid']:
                delta=np.asarray(t['motion'])-base[h]
                delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                target.append(delta)
            else:target.append([np.nan]*3)
        xs.append(x);ys.append(target);masks.append(valid);nominals.append(base);available.append(ok)
    return dict(x=np.asarray(xs),residual=np.asarray(ys),valid=np.asarray(masks),
                nominal=np.asarray(nominals),available=np.asarray(available))


def fit(train,weights,columns):
    values={k:[] for k in ('mean','scale','bias','coefficient')}
    for h in range(8):
        valid=train['valid'][:,h];x=train['x'][valid,h][:,columns];y=train['residual'][valid,h];w=weights[valid]
        mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1.
        bias=np.average(y,axis=0,weights=w);z=(x-mean)/scale
        coefficient=np.linalg.solve(z.T@(z*w[:,None])+np.eye(len(columns)),z.T@((y-bias)*w[:,None]))
        for k,v in dict(mean=mean,scale=scale,bias=bias,coefficient=coefficient).items():values[k].append(v)
    return {k:np.asarray(v) for k,v in values.items()}


def predict(data,model,columns):
    return data['nominal']+np.stack([((data['x'][:,h][:,columns]-model['mean'][h])/model['scale'][h])@
        model['coefficient'][h]+model['bias'][h] for h in range(8)],axis=1)


def scores(prediction,data,selected):
    result={}
    truth=data['nominal']+data['residual']
    for h in (2,6,7):
        mask=data['valid'][:,h]&selected
        d=prediction[mask,h]-truth[mask,h]
        yaw=np.arctan2(np.sin(d[:,2]),np.cos(d[:,2]))
        result[str((h+1)*100)]=dict(windows=int(mask.sum()),
            xy_rmse_mm=1000*float(np.sqrt(np.mean(np.sum(d[:,:2]**2,axis=1)))) if len(d) else None,
            yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw**2)))) if len(d) else None)
    return result


def main():
    if OUTPUT.exists():raise ValueError('preserve matched controls')
    terminal=json.loads((FEATURES/'result.json').read_text())
    if terminal['status']!='COMPLETE':raise ValueError('complete fixed causal feature population')
    feature_bytes=(FEATURES/'features.jsonl').read_bytes()
    if hashlib.sha256(feature_bytes).hexdigest()!=terminal['features_sha256']:raise ValueError('feature identity changed')
    feature_rows=[json.loads(s) for s in feature_bytes.splitlines()]
    features={r['sample_id']:r for r in feature_rows}
    training=load_training_rows();transfer=[r for r in json.loads((TRANSFER/'windows.json').read_text()) if r['available']]
    if len(features)!=len(feature_rows) or set(features)!={r['sample_id'] for r in training+transfer}:
        raise ValueError('every fixed context retained exactly once')
    schedule=json.loads((BASE/'go2_pre_switch_training_schedule_v1_attempt_001/schedule.json').read_text())
    weights=np.asarray([schedule['context_draw_counts'][r['sample_id']] for r in training])
    train=dataset(training,features)
    # Fits are fixed before transfer outcomes are materialized for scoring.
    columns=dict(command_only=np.arange(13,train['x'].shape[-1]),observed_motion=np.arange(train['x'].shape[-1]))
    models={name:fit(train,weights,cols) for name,cols in columns.items()}
    evaluation=dataset(transfer,features)
    predictions={name:predict(evaluation,model,columns[name]) for name,model in models.items()}
    predictions['command_integrated']=evaluation['nominal']
    masks=dict(all=np.ones(len(transfer),bool),motion_available=evaluation['available'],motion_missing=~evaluation['available'])
    for value in ('original','pre_switch'):
        masks[value]=np.asarray([r['evaluation_population']==value for r in transfer])
    for value in ('cluster_02','cluster_03'):
        masks[value]=np.asarray([r['cluster']==value for r in transfer])
    masks['future_brake']=np.asarray([any(a!=[0.,0.,0.] and b==[0.,0.,0.]
        for a,b in zip(r['known_commands'],r['known_commands'][1:])) for r in transfer])
    result=dict(status='COMPLETE',training_contexts=len(training),transfer_contexts=len(transfer),
        training_draws=int(weights.sum()),ridge_penalty=1.,
        motion_available_train=int(train['available'].sum()),motion_available_transfer=int(evaluation['available'].sum()),
        both_controls_include_full_four_observation_past_command_history=True,
        scores={name:{group:scores(pred,evaluation,mask) for group,mask in masks.items()} for name,pred in predictions.items()},
        model_columns={name:cols.tolist() for name,cols in columns.items()},
        feature_sha256=terminal['features_sha256'],missing_contexts_excluded=False,
        native_outcomes_training_targets_only=True,independent_maze_trials=0,
        same_original_trial_draw_weights=True,hyperparameter_search=False,closed_loop_tested=False)
    OUTPUT.mkdir()
    for name,model in models.items():np.savez_compressed(OUTPUT/(name+'.npz'),**model)
    np.savez_compressed(OUTPUT/'transfer_predictions.npz',**predictions)
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:result['scores'][name] for name in predictions},indent=2))


if __name__=='__main__':main()
