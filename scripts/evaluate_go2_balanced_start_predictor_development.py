"""Fixed retained branch panels for all four coverage-continuation arms."""
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_metric_goal_control_development import MetricGoalControl
from lewm.eligible_floor_registration_development import bind
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import train_go2_balanced_start_predictor_development as fit
from scripts import train_go2_cross_trajectory_goal_metric_development as metric
from scripts import evaluate_go2_dense_task_predictor_development as retained
from scripts import diagnose_go2_fresh_goal_direction_development as right

OUTPUT=fit.OUTPUT/'fixed_evaluation'
RESULT=Path('docs/go2_balanced_start_predictor_evaluation_2026-09-17.json')


@torch.inference_mode()
def main():
    assert not OUTPUT.exists() and not RESULT.exists()
    terminal=json.loads(fit.RESULT.read_text());assert terminal['status']=='COMPLETE'
    OUTPUT.mkdir();started=time.monotonic();torch.set_num_threads(4)
    # Reuse the established pulse/near-goal/late-turn panel, substituting only
    # final model loader and the fixed cross-trajectory metric shared by arms.
    adapted=SimpleNamespace(**vars(fit));adapted.metric_fit=metric
    bind(retained.run.__wrapped__,fit=adapted,OUTPUT=OUTPUT/'retained_panel',
         RESULT=OUTPUT/'retained_panel_result.json',__file__=__file__)()
    gc.collect();torch.cuda.empty_cache()
    panel=json.loads((OUTPUT/'retained_panel_result.json').read_text())
    plan=json.loads(right.PLAN.read_text());original=json.loads(right.RESULT.read_text())
    controller=MetricGoalControl('action',Path(plan['goal']))
    head=metric.load().cuda()
    def encode(path):
        pixel=controller.encoder.preprocess(str(path))[None].cuda()
        return F.layer_norm(controller.encoder.tokens(pixel).float(),(1024,))[0]
    context=torch.stack([encode(right.ORIGINAL/f'rgb_{i:04d}.png') for i in (0,5,10)])[None]
    with np.load(right.ORIGINAL/'policy_histories.npz',allow_pickle=False) as a:
        commands=a['applied_command_values'][10].astype(np.float32)
        assert a['applied_command_valid'][10].all()
    control=torch.from_numpy((commands[:,[0,2]].reshape(3,5,2)-controller.mean)/controller.std).cuda()[None]
    applied=np.asarray([apply_safety_limits_single([candidate_commands(name)[0]]*5,tuple(commands[-1]),controller.limits)[0]
                        for name in ACTIONS],np.float32)
    actions=torch.from_numpy(applied[:,:,[0,2]].reshape(6,10)).cuda()
    mask=torch.ones(6,768,dtype=torch.bool,device='cuda')
    models={name:fit.load(name).cuda() for name in fit.ARMS}
    models['parent_action']=controller.model
    path=fit.parent.OUTPUT/'no_future_action_latest.pt'
    assert fit.digest(path)==json.loads(fit.PLAN.read_text())['initial_checkpoint_sha256']['no_future_action']
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=fit.parent.reference.ProprioActionPredictor(use_proprio=False).cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model_state_dict'],strict=True);models['parent_no_future_action']=model;del state
    forecasts={}
    for name,model in models.items():
        blind=name.endswith('no_future_action')
        n=1 if blind else 6
        value=model(context.expand(n,-1,-1,-1),torch.zeros_like(actions[:n]) if blind else actions,
                    mask[:n],control=control.expand(n,-1,-1,-1))
        forecasts[name]=F.layer_norm(value.float(),(1024,)).expand(6,-1,-1)
    forecasts['persistence']=context[:,-1].expand(6,-1,-1)
    fit.save(OUTPUT/'right_start_forecasts_complete.json',dict(future_rgb_loaded=False,fit_sha256=fit.digest(fit.RESULT)))
    targets=[]
    for i in range(6):
        root=right.ORIGINAL if i==4 else right.OUTPUT/f'action_{i:02d}'
        with np.load(root/'policy_histories.npz',allow_pickle=False) as a:
            np.testing.assert_allclose(a['applied_command_values'][15][-5:],applied[i],rtol=0,atol=1e-6)
        targets.append(encode(root/'rgb_0015.png'))
    truth=torch.stack(targets);goal=controller.goal[None].expand(6,-1,-1)
    physical=np.asarray([r['physical_cost'] for r in original['rows']]);rows={}
    for name,pred in forecasts.items():
        distances=(pred[:,None]-truth[None]).square().mean((-1,-2))
        # A single blind/persistence forecast must yield a single cost, as in
        # online control. Batched GEMM roundoff must not create action choices.
        blind=name.endswith('no_future_action') or name=='persistence'
        value=head.dense_cost(pred[:1],goal[:1]).expand(6) if blind else head.dense_cost(pred,goal)
        costs=value.cpu().numpy();ties=np.flatnonzero(costs==costs.min())
        rows[name]=dict(dense_mse=float(distances.diag().mean()),factual_mse=distances.diag().tolist(),
                        action_retrieval=sum(bool(distances[i,i]<torch.cat((distances[:i,i],distances[i+1:,i])).min()) for i in range(6)),
                        predicted_goal_costs=costs.tolist(),chosen_actions=[ACTIONS[i] for i in ties],
                        physical_regret=float(physical[ties].mean()-physical.min()))
    expected=json.loads(Path('docs/go2_cross_trajectory_goal_metric_evaluation_2026-09-17.json').read_text())['groups'][0]['models']['mixed_pairs']['predicted']['costs']
    np.testing.assert_allclose(rows['parent_action']['predicted_goal_costs'],expected,rtol=1e-6,atol=1e-5)
    np.testing.assert_allclose(rows['parent_action']['dense_mse'],original['mean_future_dense_mse'],rtol=0,atol=1e-6)
    report=dict(status='COMPLETE',pulse_summaries=panel['pulse_summaries'],goal_groups=panel['goal_groups'],
                right_start=dict(models=rows,physical_costs=physical.tolist(),actual_image_goal_costs=head.dense_cost(truth,goal).cpu().tolist()),
                fit_sha256=fit.digest(fit.RESULT),goal_metric_sha256=fit.digest(metric.OUTPUT/'metric.pt'),
                retained_panel_sha256=fit.digest(OUTPUT/'retained_panel_result.json'),source_sha256=fit.digest(__file__),
                wall_s=time.monotonic()-started,new_navigation=False,
                limitations=['exposed development panels','training-role pulse scores are in sample',
                             'same frozen physically supervised goal metric for every arm',
                             'data coverage and future-action effects, not JEPA encoder objective isolation',
                             'prospective closed-loop control still required'])
    fit.save(RESULT,report);fit.save(OUTPUT/'result.json',report)
    print('BALANCED_PREDICTOR_EVALUATION_COMPLETE',json.dumps(rows),flush=True)


if __name__=='__main__':main()
