"""Fixed branch fidelity and exposed failure-state ranking after continuation."""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import train_go2_dense_task_predictor_development as fit
from scripts import diagnose_go2_dense_visual_goal_overshoot_development as early
from scripts import diagnose_go2_dense_metric_late_turn_development as late

base = fit.parent.reference
OUTPUT = fit.OUTPUT/'fixed_evaluation'
RESULT = Path('docs/go2_dense_task_predictor_evaluation_2026-09-17.json')


@torch.inference_mode()
def run():
    assert not OUTPUT.exists() and not RESULT.exists()
    terminal = json.loads(fit.RESULT.read_text()); assert terminal['status'] == 'COMPLETE'
    OUTPUT.mkdir(); started = time.monotonic(); torch.set_num_threads(4)
    models = {arm:fit.load(arm).cuda() for arm in fit.ARMS}
    initial = json.loads((fit.parent.OUTPUT/'result.json').read_text())
    for arm in fit.parent.ARMS:
        path = fit.parent.OUTPUT/f'{arm}_latest.pt'; assert base.digest(path) == initial['checkpoint_sha256'][arm]
        state = torch.load(path,map_location='cpu',weights_only=False)
        model = base.ProprioActionPredictor(use_proprio=False)
        model.load_state_dict(state['model_state_dict'],strict=True)
        models['parent_'+arm] = model.cuda().eval().requires_grad_(False)
    del state
    head = fit.metric_fit.load().cuda()
    encoder = base.encoders.VJepa21Arm(); encoder.build(torch.device('cuda:0'),torch.float32)
    stats = json.loads((base.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean,std = (np.asarray(stats[k],np.float32) for k in ('control_mean','control_std'))
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    cache = {}

    def encode(path):
        key = base.digest(path)
        if key not in cache:
            z = encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float()
            cache[key] = F.layer_norm(z,(1024,))[0].cpu()
        return cache[key]

    def predict(context,actions,control):
        n = len(actions); x,a,c = context.cuda(),actions.cuda(),control.cuda()
        mask = torch.ones(n,768,dtype=torch.bool,device='cuda'); values = {}
        assert torch.equal(x,x[:1].expand_as(x)) and torch.equal(c,c[:1].expand_as(c))
        for arm,model in models.items():
            if arm.endswith('no_future_action'):
                pred = model(x[:1],torch.zeros_like(a[:1]),mask[:1],control=c[:1])
                values[arm] = F.layer_norm(pred.float(),(1024,)).cpu().expand(n,-1,-1)
            else:
                values[arm] = F.layer_norm(model(x,a,mask,control=c).float(),(1024,)).cpu()
            assert torch.isfinite(values[arm]).all()
        values['persistence'] = context[:1,-1].expand(n,-1,-1)
        return values

    def embeddings(arm,value):
        blind = arm.endswith('no_future_action') or arm == 'persistence'
        z = head.embed(pool_tokens(value[:1].cuda() if blind else value.cuda())).cpu()
        return z.expand(len(value),-1) if blind else z

    rows = base.selected_rows(); prepared = [base.inputs(r) for r in rows]
    grouped = {}
    for i,row in enumerate(rows): grouped.setdefault((row['data_role'],row['cluster'],row['prefix_action']),[]).append(i)
    pulse_jobs = []
    for key,indices in grouped.items():
        assert len(indices) == 3
        context = torch.stack([torch.stack([encode(prepared[i][0]/f'rgb_{f:04d}.png') for f in (3,8,13)]) for i in indices])
        control = torch.from_numpy(np.stack([(prepared[i][1]-mean)/std for i in indices]))
        actions = torch.from_numpy(np.stack([prepared[i][2] for i in indices]))
        values = predict(context,actions,control)
        pulse_jobs.append((key,indices,values))
    goal_jobs = []
    for kind,diagnostic in (('early',early),('late',late)):
        plan = json.loads(diagnostic.PLAN.read_text())
        for group,source in enumerate(plan['sources']):
            departure = 35 if kind == 'early' else source['departure']
            directory = Path(source['source']); frames = (departure-10,departure-5,departure)
            meta = json.loads((directory/'policy_observations.json').read_text())
            with np.load(directory/'policy_histories.npz',allow_pickle=False) as a:
                commands = a['applied_command_values'][departure].astype(np.float32)
                assert a['applied_command_valid'][departure].all()
                assert a['applied_command_measured_ns'][departure][[4,9,14]].tolist() == [meta['frames'][i]['image_ns'] for i in frames]
                assert (a['applied_command_available_ns'][departure] <= meta['frames'][departure]['image_ns']).all()
            x = torch.stack([encode(directory/f'rgb_{i:04d}.png') for i in frames])[None].expand(6,-1,-1,-1)
            c = torch.from_numpy((commands[:,[0,2]].reshape(3,5,2)-mean)/std)[None].expand(6,-1,-1,-1)
            applied = np.asarray([apply_safety_limits_single([candidate_commands(action)[0]]*5,tuple(commands[-1]),limits)[0]
                                  for action in ACTIONS],np.float32)
            actions = torch.from_numpy(applied[:,:,[0,2]].reshape(6,10))
            goal = encode(early.pilot.GOAL_ROOT/source['goal_trial']/'rgb_0023.png')
            goal_jobs.append((kind,group,source,departure,goal,predict(x,actions,c)))
    base.save(OUTPUT/'causal_forecasts_complete.json',dict(pulse_groups=len(pulse_jobs),goal_groups=len(goal_jobs),
        future_target_images_loaded=False,models=list(models),fit_sha256=base.digest(fit.RESULT),source_sha256=base.digest(__file__)))
    pulse_details = []
    for key,indices,values in pulse_jobs:
        targets = []
        for i in indices:
            directory,_,expected = prepared[i]
            with np.load(directory/'policy_histories.npz',allow_pickle=False) as a:
                np.testing.assert_allclose(a['applied_command_values'][18][-5:,[0,2]].reshape(10),expected,rtol=0,atol=1e-6)
            targets.append(encode(directory/'rgb_0018.png'))
        truth = torch.stack(targets); truth_embedding = head.embed(pool_tokens(truth.cuda())).cpu()
        record = dict(role=key[0],cluster=key[1],prefix_action=key[2],models={})
        for arm,p in values.items():
            z = embeddings(arm,p); metrics = {}
            for name,pred,target in (('dense',p,truth),('goal_embedding',z,truth_embedding)):
                axes = tuple(range(2,pred.ndim+1))
                distances = (pred[:,None]-target[None]).square().mean(axes)
                effect = ((pred-pred.mean(0))-(target-target.mean(0))).square().mean()
                zero = (target-target.mean(0)).square().mean()
                metrics[name] = dict(factual_mse=distances.diag().tolist(),mse_matrix=distances.tolist(),
                    correct_action_wins=[bool(distances[j,j] < torch.cat((distances[:j,j],distances[j+1:,j])).min()) for j in range(3)],
                    centered_effect_mse=float(effect),centered_zero_mse=float(zero))
            record['models'][arm] = metrics
        pulse_details.append(record)
    pulse_summary = {}
    for role in ('train','geometry_transfer'):
        selected = [r for r in pulse_details if r['role'] == role]; pulse_summary[role] = {}
        for arm in (*models,'persistence'):
            pulse_summary[role][arm] = {}
            for name in ('dense','goal_embedding'):
                entries = [r['models'][arm][name] for r in selected]
                pulse_summary[role][arm][name] = dict(mse=float(np.mean([e['factual_mse'] for e in entries])),
                    correct_action_wins=sum(sum(e['correct_action_wins']) for e in entries),
                    centered_effect_ratio=float(np.mean([e['centered_effect_mse'] for e in entries])/np.mean([e['centered_zero_mse'] for e in entries])))
    goal_details = []
    early_result = json.loads(fit.metric_fit.RESULT.with_name('go2_dense_goal_metric_near_goal_result_2026-09-17.json').read_text())
    late_result = json.loads(late.RESULT.read_text())
    for kind,group,source,departure,goal,values in goal_jobs:
        if kind == 'early':
            selected = list(range(6)); physical = np.array(early_result['groups'][group]['physical_goal_cost'])
            actual = torch.stack([encode(early.OUTPUT/f'branch_{i:02d}'/'rgb_0040.png') for i in range(group*6,(group+1)*6)])
        else:
            selected = [0,5]; physical = np.array([r['physical_cost'] for r in late_result['groups'][group]['rows']])
            actual = torch.stack([encode(late.OUTPUT/f'branch_{group:02d}'/f'rgb_{departure+5:04d}.png'),
                                  encode(Path(source['source'])/f'rgb_{departure+5:04d}.png')])
        goal_embedding = head.embed(pool_tokens(goal[None].cuda())).cpu()
        results = {}
        for arm,pred in values.items():
            z = embeddings(arm,pred); costs = (z-goal_embedding).square().mean(-1).numpy()
            pair = costs[selected]; ties = np.flatnonzero(pair == pair.min())
            results[arm] = dict(all_six_costs=costs.tolist(),all_six_choices=[ACTIONS[i] for i in np.flatnonzero(costs==costs.min())],
                evaluated_choices=[ACTIONS[selected[i]] for i in ties],
                physical_regret_among_evaluated=float(physical[ties].mean()-physical.min()),
                factual_dense_mse=float((pred[selected]-actual).square().mean()),
                factual_goal_embedding_mse=float((z[selected]-head.embed(pool_tokens(actual.cuda())).cpu()).square().mean()))
        oracle = (head.embed(pool_tokens(actual.cuda())).cpu()-goal_embedding).square().mean(-1).numpy()
        goal_details.append(dict(kind=kind,case=source['case'],departure=departure,
            evaluated_actions=[ACTIONS[i] for i in selected],physical_cost=physical.tolist(),
            actual_image_cost=oracle.tolist(),actual_image_choice=ACTIONS[selected[int(oracle.argmin())]],models=results))
    previous = json.loads(Path('docs/go2_frozen_vjepa_native_adaptation_branch_result_2026-09-17.json').read_text())
    for role in pulse_summary:
        for arm in fit.parent.ARMS:
            np.testing.assert_allclose(pulse_summary[role]['parent_'+arm]['dense']['mse'],previous['summaries'][role][arm]['mse'],rtol=0,atol=1e-6)
    report = dict(status='COMPLETE',pulse_summaries=pulse_summary,pulse_groups=pulse_details,goal_groups=goal_details,
        fit_sha256=base.digest(fit.RESULT),source_sha256=base.digest(__file__),wall_s=time.monotonic()-started,
        no_dense_cache_retained=True,navigation_tested=False,exposed_development_only=True,
        limitations=['training-role pulse scores are in sample','four post-hoc failure states',
            'late-state physical regrets compare only hold and right turn','extra physical supervision; no JEPA objective superiority claim'])
    base.save(OUTPUT/'result.json',report); base.save(RESULT,report)
    print('TASK_PREDICTOR_EVALUATION_COMPLETE',json.dumps(dict(transfer=pulse_summary['geometry_transfer'],
        goal_choices=[dict(kind=g['kind'],case=g['case'],choices={a:r['evaluated_choices'] for a,r in g['models'].items()}) for g in goal_details])),flush=True)


if __name__ == '__main__':
    try:run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            base.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
