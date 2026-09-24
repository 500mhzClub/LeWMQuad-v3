"""Fixed turn-separation and three-state actual/forecast action ranking panel."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lewm.dense_metric_goal_control_development import MetricGoalControl
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import train_go2_cross_trajectory_goal_metric_development as fit
from scripts import diagnose_go2_fresh_goal_direction_development as right
from scripts import diagnose_go2_dense_visual_goal_overshoot_development as near

RESULT=Path('docs/go2_cross_trajectory_goal_metric_evaluation_2026-09-17.json')
OUTPUT=fit.OUTPUT/'fixed_evaluation'


@torch.inference_mode()
def main():
    assert not RESULT.exists() and not OUTPUT.exists()
    terminal=json.loads(fit.RESULT.read_text());assert terminal['status']=='COMPLETE'
    OUTPUT.mkdir();torch.set_num_threads(4);start=time.monotonic()
    right_plan=json.loads(right.PLAN.read_text());right_result=json.loads(right.RESULT.read_text())
    controller=MetricGoalControl('action',Path(right_plan['goal']))
    heads={'within_only':controller.metric,'mixed_pairs':fit.load().cuda()}
    cache={}
    def encode(path):
        key=fit.original.digest(path)
        if key not in cache:
            with Image.open(path) as im:pixels=np.asarray(im.convert('RGB'))
            cache[key]=controller.encode(pixels)
        return cache[key]
    jobs=[dict(name='fresh_right_initial',source=right.ORIGINAL,departure=10,
        goal=Path(right_plan['goal']),targets=[(right.ORIGINAL if i==4 else right.OUTPUT/f'action_{i:02d}')/'rgb_0015.png' for i in range(6)],
        physical=[r['physical_cost'] for r in right_result['rows']],
        old_predicted=[r['predicted_cost'] for r in right_result['rows']],
        old_actual=[r['actual_image_cost'] for r in right_result['rows']])]
    old_near=json.loads(Path('docs/go2_dense_goal_metric_near_goal_result_2026-09-17.json').read_text())
    for group,source in enumerate(json.loads(near.PLAN.read_text())['sources']):
        reference=old_near['groups'][group]
        jobs.append(dict(name=f"left_near_case_{source['case']}",source=Path(source['source']),departure=35,
            goal=near.pilot.GOAL_ROOT/source['goal_trial']/'rgb_0023.png',
            targets=[near.OUTPUT/f'branch_{i:02d}'/'rgb_0040.png' for i in range(6*group,6*group+6)],
            physical=reference['physical_goal_cost'],
            old_predicted=reference['models']['action']['learned_metric']['costs'],
            old_actual=reference['models']['observed_future_oracle']['learned_metric']['costs']))
    for job in jobs:
        departure=job['departure'];source=job['source']
        context=torch.stack([encode(source/f'rgb_{i:04d}.png') for i in (departure-10,departure-5,departure)])[None]
        with np.load(source/'policy_histories.npz',allow_pickle=False) as a:
            commands=a['applied_command_values'][departure].astype(np.float32)
            assert a['applied_command_valid'][departure].all()
        c=torch.from_numpy((commands[:,[0,2]].reshape(3,5,2)-controller.mean)/controller.std).cuda()[None]
        applied=np.asarray([apply_safety_limits_single([candidate_commands(a)[0]]*5,tuple(commands[-1]),controller.limits)[0] for a in ACTIONS],np.float32)
        actions=torch.from_numpy(applied[:,:,[0,2]].reshape(6,10)).cuda();mask=torch.ones(6,768,dtype=torch.bool,device='cuda')
        p=controller.model(context.expand(6,-1,-1,-1),actions,mask,control=c.expand(6,-1,-1,-1))
        job['forecast']=F.layer_norm(p.float(),(1024,));job['goal_feature']=encode(job['goal'])[None]
        old=heads['within_only'].dense_cost(job['forecast'],job['goal_feature'].expand(6,-1,-1))
        np.testing.assert_allclose(old.cpu(),job['old_predicted'],rtol=1e-6,atol=1e-6)
    fit.original.save(OUTPUT/'forecasts_complete.json',dict(states=3,future_rgb_loaded=False,old_costs_reproduced=True))
    groups=[]
    for job in jobs:
        truth=torch.stack([encode(p) for p in job['targets']]);goal=job['goal_feature'].expand(6,-1,-1)
        physical=np.asarray(job['physical']);models={}
        for name,head in heads.items():
            models[name]={}
            for kind,features in (('predicted',job['forecast']),('actual_images',truth)):
                costs=head.dense_cost(features,goal).cpu().numpy();assert np.isfinite(costs).all()
                if name=='within_only' and kind=='actual_images':np.testing.assert_allclose(costs,job['old_actual'],rtol=1e-6,atol=1e-6)
                chosen=np.flatnonzero(costs==costs.min())
                models[name][kind]=dict(costs=costs.tolist(),chosen_actions=[ACTIONS[i] for i in chosen],
                    physical_regret=float(physical[chosen].mean()-physical.min()))
        groups.append(dict(state=job['name'],actions=ACTIONS,physical_cost=physical.tolist(),
            physical_best=ACTIONS[int(physical.argmin())],models=models))
    probes=json.loads(Path('docs/go2_goal_metric_turn_separation_2026-09-17.json').read_text());separation=[]
    paths=json.loads((fit.OUTPUT/'frame_paths.json').read_text());lookup={p:i for i,p in enumerate(paths)}
    with np.load(fit.OUTPUT/'pairs.npz',allow_pickle=False) as a:pair_set={tuple(sorted(p)) for p in a['indices'].tolist()}
    for row in probes['rows']:
        features=[encode(Path(p))[None] for p in row['images']];comparisons=[]
        for position,(i,j) in enumerate(((0,1),(0,2),(1,2))):
            physical=row['comparisons'][position]['physical_cost']
            costs={name:float(head.dense_cost(features[i],features[j])) for name,head in heads.items()}
            comparisons.append(dict(pair=row['comparisons'][position]['pair'],physical_cost=physical,
                costs=costs,mixed_cost_ratio=costs['mixed_pairs']/physical,
                exact_pair_in_mixed_fit=tuple(sorted((lookup[row['images'][i]],lookup[row['images'][j]]))) in pair_set))
        angles={}
        for name in ('physical','within_only','mixed_pairs'):
            d=[r['physical_cost'] if name=='physical' else r['costs'][name] for r in comparisons]
            cosine=(d[0]+d[1]-d[2])/(2*np.sqrt(d[0]*d[1]))
            angles[name]=float(np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
        separation.append(dict(layout=row['layout'],comparisons=comparisons,
            angle_between_turn_displacements_deg=angles))
    report=dict(status='COMPLETE',groups=groups,training_turn_separation=separation,
        fit_sha256=fit.original.digest(fit.RESULT),source_sha256=fit.original.digest(__file__),
        forecasts_precede_successor_rgb=True,old_predicted_and_actual_costs_reproduced=True,
        unique_encoded_images=len(cache),wall_s=time.monotonic()-start,no_training=True,new_navigation=False,
        limitations=['three exposed post-hoc decision states','turn probes use training images and are not a transfer test',
            'goal-head pair-coverage intervention, not JEPA representation training','prospective control still required'])
    fit.original.save(RESULT,report);fit.original.save(OUTPUT/'result.json',report)
    print('CROSS_METRIC_EVALUATION_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
