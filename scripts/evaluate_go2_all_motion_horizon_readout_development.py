"""Fixed translation/turn transfer panel for the all-motion horizon readout fit."""
import argparse
from collections import Counter
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts.evaluate_go2_correlation_motion_readout_development import metrics
from scripts import train_go2_horizon_dense_predictor_development as dynamics
from scripts import train_go2_all_motion_horizon_readout_development as fit

ROOT = fit.OUTPUT.parent/'go2_dense_world_model_maze_layout00_action_mixed_data_v1_attempt_001'
OUTPUT = fit.OUTPUT/'maze00_evaluation'
PLAN = fit.OUTPUT/'transfer_plan.json'


def save(path, value):
    with path.open('x') as f:
        json.dump(value,f,indent=2)
        f.write('\n')


def retention():
    return {name:json.loads((ROOT/name).read_text()) for name in
        ('depth_retention.json','native/depth_retention.json') if (ROOT/name).exists()}


def records():
    read = lambda name:json.loads((ROOT/name).read_text())
    plans = {p['frame']:p for p in read('planning.json') if 'selection' in p}
    by_time = {p['measured_ns']:p['frame'] for p in plans.values()}
    calls = {by_time[c['observed_ns']]:c for c in read('dense_model_calls.json')}
    return read,plans,calls


def prepare():
    assert not PLAN.exists() and not OUTPUT.exists()
    receipts = retention()
    read,plans,calls = records()
    assert read('result.json')['mission_terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    evaluated = read('dense_navigation_readout.json')
    frames = {r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']:r for r in read('requests.json')}
    prior_turn = read('turn_true_future_readout_v1/plan.json')['selected']
    turns = sorted(f for values in prior_turn.values() for f in values)
    assert len(set(turns)) == 32
    eligible = {'translation':[], 'turn':[]}
    skipped = Counter()
    for row in evaluated['executed_windows']['rows']:
        frame,group = row['frame'],row['group']
        if group != 'translation' and frame not in turns:
            continue
        call = calls[frame]
        idx = ACTIONS.index(plans[frame]['action'])
        if any(frame+d not in frames for d in (-10,-5,0,5,7)):
            skipped['missing_frame'] += 1
            continue
        times = [call['observed_ns']+j*100_000_000+k*20_000_000 for j in range(7) for k in range(5)]
        if any(t not in requests for t in times):
            skipped['missing_request'] += 1
            continue
        if any(not np.allclose([requests[t][field+'_command'] for t in times],
                np.repeat(np.asarray(call[field+'_commands'])[idx,:7],5,axis=0),rtol=0,atol=1e-6)
                for field in ('requested','applied')):
            skipped['changed_command_tape'] += 1
            continue
        eligible[group].append(frame)
    assert sorted(eligible['turn']) == turns
    assert len(eligible['translation']) >= 32
    selected = dict(turn=turns,translation=[sorted(eligible['translation'])[i]
        for i in np.linspace(0,len(eligible['translation'])-1,32,dtype=int)])
    with np.load(ROOT/'native/physics_trace.npz',allow_pickle=False) as a:
        poses = a['base_pose_world'].copy()
    truth = []
    for group,departures in selected.items():
        for frame in departures:
            origin = poses[frames[frame]['physical_sample_index']]
            rotation = rotation_xyzw(origin[3:])
            plan,call = plans[frame],calls[frame]
            idx = ACTIONS.index(plan['action'])
            for h in (5,7):
                future = poses[frames[frame+h]['physical_sample_index']]
                future_rotation = rotation_xyzw(future[3:])
                yaw = np.arctan2(future_rotation[1,0],future_rotation[0,0])-np.arctan2(rotation[1,0],rotation[0,0])
                actual = np.r_[((future[:3]-origin[:3])@rotation)[:2],np.arctan2(np.sin(yaw),np.cos(yaw))]
                truth.append(dict(frame=frame,group=group,action=plan['action'],horizon_ms=h*100,
                    actual=actual.tolist(),saved_starting_action=call['motion_xy_yaw'][idx][h-1],
                    command_history=plan['motion_correction']['command_history_forecast_xy_yaw'][idx][h-1]))
    needed = sorted({f+d for values in selected.values() for f in values for d in (-10,-5,0,5,7)})
    assert all((ROOT/'native'/f'rgb_{f:04d}.png').is_file() for f in needed)
    save(fit.OUTPUT/'transfer_targets.json',truth)
    save(PLAN,dict(root=str(ROOT),selected=selected,eligible={g:len(v) for g,v in eligible.items()},
        skipped=dict(skipped),horizons_ms=[500,700],primary='700-ms XY/yaw RMSE on 32 translation windows',
        secondary='500-ms translation and both horizons on the previous 32 turn diagnostic windows',
        selection='32 equally spaced chronological eligible translation windows, independent of forecast errors; reuse prior turn population',
        targets_sha256=fit.digest(fit.OUTPUT/'transfer_targets.json'),evaluator_sha256=fit.digest(__file__),
        image_frames=needed,depth_retention_receipts=receipts,heads=['starting_mixed',*fit.ARMS],
        future_inputs=['observed_future','action','no_future_action'],controls=['command_history','zero_motion'],
        exact_requested_and_applied_tapes_matched_to_700ms=True,no_training_on_this_population=True,
        no_navigation=True,automatic_promotion=False,limitations=['post hoc exposed same-family development trajectory',
            'selected overlapping executed windows, not counterfactual outcomes or independent maze trials']))
    print('ALL_MOTION_TRANSFER_PREPARED',len(truth),'rows',len(needed),'images',json.dumps({g:len(v) for g,v in eligible.items()}),flush=True)


@torch.inference_mode()
def main():
    plan = json.loads(PLAN.read_text())
    assert fit.digest(__file__) == plan['evaluator_sha256']
    assert fit.digest(fit.OUTPUT/'transfer_targets.json') == plan['targets_sha256']
    trained = json.loads((fit.OUTPUT/'result.json').read_text())
    assert trained['status'] == 'COMPLETE' and trained['steps'] == fit.STEPS
    retention()
    heads = {'starting_mixed':fit.prior.previous.load('mixed_data')}
    for arm in fit.ARMS:
        path = fit.OUTPUT/f'{arm}_final.pt'
        assert fit.digest(path) == trained['checkpoint_sha256'][arm]
        state = torch.load(path,map_location='cpu',weights_only=False)
        assert state['updates'] == fit.STEPS and state['plan_sha256'] == fit.digest(fit.OUTPUT/'plan.json')
        head = fit.prior.previous.load('mixed_data')
        head.load_state_dict(state['model_state_dict'])
        heads[arm] = head.eval().requires_grad_(False)
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    try:
        _,plans,calls = records()
        truth = json.loads((fit.OUTPUT/'transfer_targets.json').read_text())
        with np.load(ROOT/'native/policy_histories.npz',allow_pickle=False) as a:
            past = a['applied_command_values'].copy()
        stats = json.loads((dynamics.parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
        mean,std = [torch.tensor(stats[k],dtype=torch.float32) for k in ('control_mean','control_std')]
        encoder = VJepa21Arm()
        encoder.build(torch.device('cpu'),torch.float32)
        features = {}
        for j,frame in enumerate(plan['image_frames']):
            pixels = encoder.preprocess(str(ROOT/'native'/f'rgb_{frame:04d}.png'))[None]
            features[frame] = F.layer_norm(encoder.tokens(pixels).float(),(1024,))[0]
            if (j+1)%25 == 0 or j+1 == len(plan['image_frames']):
                print('ALL_MOTION_TRANSFER_FEATURES',j+1,len(plan['image_frames']),flush=True)
        del encoder
        models = {arm:dynamics.load(arm) for arm in dynamics.ARMS}
        rows = []
        for target in truth:
            frame,h = target['frame'],target['horizon_ms']//100
            call = calls[frame]
            idx = ACTIONS.index(plans[frame]['action'])
            actions = torch.tensor(np.asarray(call['applied_commands'])[idx][:,[0,2]],dtype=torch.float32)[None]
            context = torch.stack([features[frame+d] for d in (-10,-5,0)])[None]
            control = (torch.tensor(past[frame][:,[0,2]].reshape(3,5,2),dtype=torch.float32)-mean)/std
            current = pool_tokens(features[frame][None])
            futures = {'observed_future':features[frame+h][None]}
            for arm,model in models.items():
                prediction = model(context,actions,torch.tensor([h]),torch.ones(1,768,dtype=torch.bool),control=control[None])
                futures[arm] = F.layer_norm(prediction.float(),(1024,))
            predictions = dict(command_history=np.asarray(target['command_history']),zero_motion=np.zeros(3))
            for name,head in heads.items():
                for arm,tokens in futures.items():
                    predictions[name+'_'+arm] = head(current,pool_tokens(tokens))[0].numpy()
            np.testing.assert_allclose(predictions['starting_mixed_action'],target['saved_starting_action'],rtol=0,atol=2e-5)
            errors = {}
            for name,value in predictions.items():
                delta = value-np.asarray(target['actual'])
                delta[2] = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                errors[name] = delta.tolist()
            rows.append(target|dict(predictions={k:v.tolist() for k,v in predictions.items()},errors=errors))
            if len(rows)%16 == 0:
                print('ALL_MOTION_TRANSFER_ROWS',len(rows),len(truth),flush=True)
        summary = {group:{str(h):metrics([r for r in rows if r['group']==group and r['horizon_ms']==h])
            for h in (500,700)} for group in ('translation','turn')}
        save(OUTPUT/'result.json',dict(status='COMPLETE',rows=rows,by_group_horizon=summary,
            transfer_plan_sha256=fit.digest(PLAN),wall_s=time.monotonic()-started,
            starting_mixed_predictions_reproduced_atol=2e-5,navigation_tested=False,automatically_promoted=False))
        print('ALL_MOTION_TRANSFER_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as error:
        save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else main()
