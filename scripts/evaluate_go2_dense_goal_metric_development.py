"""Common fitted goal cost on observed versus predicted near-goal alternatives."""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.physical_execution_development import rotation_xyzw
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import train_go2_dense_goal_metric_development as training
from scripts import diagnose_go2_dense_visual_goal_overshoot_development as diagnostic

base = training.parent.reference
OUTPUT = training.OUTPUT/'near_goal_evaluation'
RESULT = Path('docs/go2_dense_goal_metric_near_goal_result_2026-09-17.json')


@torch.inference_mode()
def run():
    assert not OUTPUT.exists() and not RESULT.exists()
    fitted=json.loads((training.OUTPUT/'result.json').read_text())
    assert fitted['status']=='COMPLETE'
    OUTPUT.mkdir(); started=time.monotonic(); torch.set_num_threads(4)
    head=training.load().cuda()
    plan=json.loads(diagnostic.PLAN.read_text())
    previous=json.loads(diagnostic.RESULT.read_text())
    assert previous['status']=='COMPLETE' and previous['original_forward_successors_exact']
    terminal=json.loads((training.parent.OUTPUT/'result.json').read_text())
    models={}
    for arm in training.parent.ARMS:
        path=training.parent.OUTPUT/f'{arm}_latest.pt'
        assert base.digest(path)==terminal['checkpoint_sha256'][arm]
        state=torch.load(path,map_location='cpu',weights_only=False)
        model=base.ProprioActionPredictor(use_proprio=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        models[arm]=model.cuda().eval().requires_grad_(False)
    del state
    stats=json.loads((base.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean,std=(np.asarray(stats[k],np.float32) for k in ('control_mean','control_std'))
    limits=SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    encoder=base.encoders.VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
    cache={}

    def encode(path):
        key=base.digest(path)
        if key not in cache:
            value=encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float()
            cache[key]=F.layer_norm(value,(1024,))[0]
        return cache[key]

    forecasts=[];goals=[]
    for source in plan['sources']:
        directory=Path(source['source'])
        meta=json.loads((directory/'policy_observations.json').read_text())
        x=torch.stack([encode(directory/f'rgb_{i:04d}.png') for i in (25,30,35)])[None]
        with np.load(directory/'policy_histories.npz',allow_pickle=False) as archive:
            commands=archive['applied_command_values'][35].astype(np.float32)
            assert archive['applied_command_valid'][35].all()
            assert archive['applied_command_measured_ns'][35][[4,9,14]].tolist()==[meta['frames'][i]['image_ns'] for i in (25,30,35)]
            assert (archive['applied_command_available_ns'][35]<=meta['frames'][35]['image_ns']).all()
        requested=[[candidate_commands(a)[0]]*5 for a in ACTIONS]
        applied=np.asarray([apply_safety_limits_single(v,tuple(commands[-1]),limits)[0] for v in requested],np.float32)
        c=torch.from_numpy((commands[:,[0,2]].reshape(3,5,2)-mean)/std).cuda()[None]
        a=torch.from_numpy(applied[:,:,[0,2]].reshape(6,10)).cuda()
        mask=torch.ones(6,768,dtype=torch.bool,device='cuda')
        values={}
        for arm,model in models.items():
            if arm=='no_future_action':
                p=model(x,torch.zeros_like(a[:1]),mask[:1],control=c)
                values[arm]=F.layer_norm(p.float(),(1024,)).expand(6,-1,-1)
            else:
                p=model(x.expand(6,-1,-1,-1),a,mask,control=c.expand(6,-1,-1,-1))
                values[arm]=F.layer_norm(p.float(),(1024,))
        values['persistence']=x[0,-1:].expand(6,-1,-1)
        forecasts.append(values)
        goals.append(encode(diagnostic.pilot.GOAL_ROOT/source['goal_trial']/'rgb_0023.png'))
        np.testing.assert_allclose((values['action']-goals[-1]).square().mean((-1,-2)).cpu().numpy(),
            source['original_decision']['costs'],rtol=0,atol=1e-6)
    base.save(OUTPUT/'causal_forecasts_complete.json',dict(groups=2,future_branch_rgb_loaded=False,
        original_live_action_costs_reproduced=True))
    groups=[]
    for group,source in enumerate(plan['sources']):
        values=forecasts[group];goal=goals[group][None]
        values['observed_future_oracle']=torch.stack([encode(diagnostic.OUTPUT/f'branch_{i:02d}'/'rgb_0040.png')
            for i in range(group*6,(group+1)*6)])
        original=json.loads((Path(source['source'])/'result.json').read_text())
        goal_pose=np.asarray(original['goal_pose_evaluator_only'])
        goal_rotation=rotation_xyzw(goal_pose[3:]);goal_yaw=np.arctan2(goal_rotation[1,0],goal_rotation[0,0])
        physical=[]
        for i in range(group*6,(group+1)*6):
            result=json.loads((diagnostic.OUTPUT/f'branch_{i:02d}'/'result.json').read_text())
            assert result['complete_500ms'] and not result['disallowed_contact']
            pose=np.asarray(result['endpoint_pose_evaluator_only']);r=rotation_xyzw(pose[3:])
            yaw=np.arctan2(r[1,0],r[0,0])-goal_yaw;yaw=np.arctan2(np.sin(yaw),np.cos(yaw))
            physical.append(float(np.sum((pose[:2]-goal_pose[:2])**2)/training.POSITION_SCALE**2+
                                  yaw*yaw/training.HEADING_SCALE**2))
        physical=np.asarray(physical)
        costs={}
        for arm,prediction in values.items():
            count=1 if arm in ('no_future_action','persistence') else 6
            raw=(prediction[:count]-goal).square().mean((-1,-2))
            learned=head.dense_cost(prediction[:count],goal.expand(count,-1,-1))
            if count==1:raw=raw.expand(6);learned=learned.expand(6)
            costs[arm]={}
            for name,tensor in (('raw_mse',raw),('learned_metric',learned)):
                vector=tensor.cpu().numpy();assert np.isfinite(vector).all() and (vector>=0).all()
                chosen=np.flatnonzero(vector==vector.min())
                costs[arm][name]=dict(costs=vector.tolist(),chosen_actions=[ACTIONS[i] for i in chosen],
                    expected_physical_cost=float(physical[chosen].mean()),
                    physical_regret=float(physical[chosen].mean()-physical.min()))
        np.testing.assert_allclose(costs['observed_future_oracle']['raw_mse']['costs'],
            [r['actual_goal_mse'] for r in previous['groups'][group]['rows']],rtol=0,atol=1e-6)
        groups.append(dict(case=source['case'],actions=ACTIONS,physical_goal_cost=physical.tolist(),
            physical_best=ACTIONS[int(physical.argmin())],models=costs))
    result=dict(status='COMPLETE',groups=groups,wall_s=time.monotonic()-started,
        goal_metric_fit_sha256=base.digest(training.OUTPUT/'result.json'),diagnostic_sha256=base.digest(diagnostic.RESULT),
        source_sha256=base.digest(__file__),fit_used_observed_training_pairs_only=True,
        original_live_action_forecast_costs_reproduced=True,original_oracle_costs_reproduced=True,
        new_navigation=False,transfer_geometries_previously_exposed=True,
        limitations=['two post-hoc near-goal states; no independent navigation success',
            'metric trained on observed features; predicted features can have a distribution mismatch',
            'additional physical supervision; no isolated JEPA-training benefit'])
    base.save(OUTPUT/'result.json',result);base.save(RESULT,result)
    print('GOAL_METRIC_NEAR_GOAL_COMPLETE',json.dumps(groups),flush=True)


if __name__=='__main__':
    try:run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            base.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
