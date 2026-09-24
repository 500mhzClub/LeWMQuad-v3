"""One fixed alternative: shared signed readout on predicted/actual goal pairs."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lewm.dense_metric_goal_control_development import MetricGoalControl
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import diagnose_go2_fresh_goal_direction_development as diagnostic
from scripts import train_go2_direct_visual_goal_readout_development as fitted

RESULT = Path('docs/go2_signed_goal_cost_diagnostic_2026-09-17.json')


@torch.inference_mode()
def main():
    assert not RESULT.exists()
    old=json.loads(diagnostic.RESULT.read_text());assert old['status']=='COMPLETE'
    plan=diagnostic.validate();torch.set_num_threads(4);start=time.monotonic()
    control=MetricGoalControl('action',Path(plan['goal']));head=fitted.load().cuda()
    def encode(path):
        with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
        return control.encode(rgb)
    context=torch.stack([encode(diagnostic.ORIGINAL/f'rgb_{i:04d}.png') for i in (0,5,10)])[None]
    with np.load(diagnostic.ORIGINAL/'policy_histories.npz',allow_pickle=False) as a:
        commands=a['applied_command_values'][10].astype(np.float32)
    history=torch.from_numpy((commands[:,[0,2]].reshape(3,5,2)-control.mean)/control.std).cuda()[None]
    applied=np.asarray([apply_safety_limits_single([candidate_commands(a)[0]]*5,tuple(commands[-1]),control.limits)[0] for a in ACTIONS],np.float32)
    actions=torch.from_numpy(applied[:,:,[0,2]].reshape(6,10)).cuda();mask=torch.ones(6,768,dtype=torch.bool,device='cuda')
    pred=F.layer_norm(control.model(context.expand(6,-1,-1,-1),actions,mask,control=history.expand(6,-1,-1,-1)).float(),(1024,))
    old_cost=(control.metric.embed(pool_tokens(pred))-control.goal_embedding).square().mean(-1)
    np.testing.assert_allclose(old_cost.cpu(),plan['original_decision']['costs'],atol=1e-6,rtol=0)
    goal=pool_tokens(control.goal[None]);estimate=head(pool_tokens(pred),goal.expand(6,-1,-1))
    predicted_cost=(estimate/head.scale).square().sum(-1)
    assert torch.isfinite(predicted_cost).all()
    # All six alternative forecasts/scoring above precede loading future RGB.
    rows=[]
    for i,name in enumerate(ACTIONS):
        root=diagnostic.ORIGINAL if i==4 else diagnostic.OUTPUT/f'action_{i:02d}'
        target=encode(root/'rgb_0015.png');actual=head(pool_tokens(target[None]),goal)[0]
        rows.append(dict(action=name,predicted_pose=estimate[i].cpu().tolist(),
            actual_image_pose=actual.cpu().tolist(),predicted_signed_cost=float(predicted_cost[i]),
            actual_image_signed_cost=float((actual/head.scale).square().sum()),
            physical_cost=old['rows'][i]['physical_cost']))
    report=dict(status='COMPLETE',rows=rows,
        predicted_choice=min(rows,key=lambda r:r['predicted_signed_cost'])['action'],
        actual_image_choice=min(rows,key=lambda r:r['actual_image_signed_cost'])['action'],
        physical_choice=old['physical_choice'],
        cost_definition='sum of squared signed readout displacement/heading in fixed 3-cm/5-degree units',
        readout_fit_sha256=diagnostic.base.digest(fitted.RESULT),
        readout_checkpoint_sha256=json.loads(fitted.RESULT.read_text())['checkpoint_sha256'],
        source_sha256=diagnostic.base.digest(__file__),diagnosis_sha256=diagnostic.base.digest(diagnostic.RESULT),
        old_costs_reproduced=True,forecasts_precede_future_rgb=True,wall_s=time.monotonic()-start,
        no_training=True,new_navigation=False,post_hoc_diagnostic=True,
        limitations=['one previously exposed diagnostic state',
            'readout trained on observed features; predicted-feature transfer assessed only here',
            'same faulty arrival detector remains; no new reliability claim'])
    diagnostic.base.save(RESULT,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
