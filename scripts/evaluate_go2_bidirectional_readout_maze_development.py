"""Evaluate an existing bidirectional visual-goal head as a frozen motion probe."""
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts import evaluate_go2_all_motion_horizon_readout_development as panel
from scripts import train_go2_direct_visual_goal_readout_development as direct

OUTPUT = panel.fit.OUTPUT / 'existing_bidirectional_readout_maze_v1'


@torch.inference_mode()
def main():
    fixed_plan = json.loads(panel.PLAN.read_text())
    assert panel.fit.digest(panel.__file__) == fixed_plan['evaluator_sha256']
    assert panel.fit.digest(panel.fit.OUTPUT / 'transfer_targets.json') == fixed_plan['targets_sha256']
    previous_path = panel.OUTPUT / 'result.json'
    previous = json.loads(previous_path.read_text())
    assert previous['status'] == 'COMPLETE'
    assert previous['transfer_plan_sha256'] == panel.fit.digest(panel.PLAN)
    head = direct.load()
    fit_result = json.loads(direct.RESULT.read_text())
    reference_head = panel.fit.prior.previous.load('mixed_data')
    _, plans, calls = panel.records()
    frames = {r['frame']: r for r in json.loads(
        (panel.ROOT / 'native/in_memory_camera_observations.json').read_text())['frames']}
    with np.load(panel.ROOT / 'native/physics_trace.npz', allow_pickle=False) as archive:
        poses = archive['base_pose_world'].copy()
    targets = []
    for old in previous['rows']:
        frame, h = old['frame'], old['horizon_ms'] // 100
        a = poses[frames[frame]['physical_sample_index']]
        b = poses[frames[frame+h]['physical_sample_index']]
        rotation = panel.rotation_xyzw(a[3:])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        delta = b[:2]-a[:2]
        planar = [float(c*delta[0]+s*delta[1]), float(-s*delta[0]+c*delta[1]), old['actual'][2]]
        targets.append(dict(frame=frame, group=old['group'], action=old['action'],
            horizon_ms=old['horizon_ms'], actual=planar, body_target=old['actual'],
            reference_predictions=old['predictions']))
    discrepancy = [float(np.linalg.norm(np.asarray(r['actual'])[:2]-np.asarray(r['body_target'])[:2]))
                   for r in targets]
    OUTPUT.mkdir(exist_ok=False)
    panel.save(OUTPUT / 'plan.json', dict(source_sha256=panel.fit.digest(__file__),
        fixed_transfer_plan_sha256=panel.fit.digest(panel.PLAN), previous_result_sha256=panel.fit.digest(previous_path),
        checkpoint_sha256=fit_result['checkpoint_sha256'], checkpoint=str(direct.OUTPUT/'readout.pt'),
        source_training_plan_sha256=direct.digest(direct.PLAN), selected=fixed_plan['selected'],
        image_frames=fixed_plan['image_frames'], future_inputs=['observed_future','action','no_future_action'],
        primary='700-ms translation XY/yaw on the unchanged 32-window panel',
        secondary='500-ms translation and 500/700-ms turns; both planar and body-XY target conventions',
        common_primary_target='planar body-heading XY and wrapped world-yaw change',
        body_vs_planar_xy_difference_max_mm=max(discrepancy)*1000,
        body_vs_planar_xy_difference_rmse_mm=float(np.sqrt(np.mean(np.square(discrepancy))))*1000,
        depth_retention_receipts=panel.retention(), cpu_cores=[4,5,6,7],
        no_training=True, no_navigation=True, automatic_promotion=False,
        limitations=['Post-hoc frozen-checkpoint reuse on exposed overlapping windows.',
            'Head architecture, training pairs, directions, horizons, normalization and update budget differ.',
            'Existing head was trained on actual images, not predicted features.',
            'Baseline motion heads were trained on full-body XY; score both conventions without correcting predictions.',
            'No isolated symmetry, coverage or JEPA-objective claim.']))
    started = time.monotonic()
    torch.set_num_threads(4)
    try:
        with np.load(panel.ROOT/'native/policy_histories.npz',allow_pickle=False) as archive:
            past = archive['applied_command_values'].copy()
        stats = json.loads((panel.dynamics.parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
        mean, std = [torch.tensor(stats[k],dtype=torch.float32) for k in ('control_mean','control_std')]
        encoder = panel.VJepa21Arm()
        encoder.build(torch.device('cpu'),torch.float32)
        features = {}
        for j, frame in enumerate(fixed_plan['image_frames']):
            pixels = encoder.preprocess(str(panel.ROOT/'native'/f'rgb_{frame:04d}.png'))[None]
            features[frame] = F.layer_norm(encoder.tokens(pixels).float(),(1024,))[0]
            if (j+1)%25 == 0 or j+1 == len(fixed_plan['image_frames']):
                print('BIDIRECTIONAL_MAZE_FEATURES',j+1,len(fixed_plan['image_frames']),flush=True)
        del encoder
        models = {arm:panel.dynamics.load(arm) for arm in panel.dynamics.ARMS}
        rows = []
        for target in targets:
            frame, h = target['frame'],target['horizon_ms']//100
            call = calls[frame]
            idx = panel.ACTIONS.index(plans[frame]['action'])
            actions = torch.tensor(np.asarray(call['applied_commands'])[idx][:,[0,2]],dtype=torch.float32)[None]
            context = torch.stack([features[frame+d] for d in (-10,-5,0)])[None]
            control = (torch.tensor(past[frame][:,[0,2]].reshape(3,5,2),dtype=torch.float32)-mean)/std
            current = pool_tokens(features[frame][None])
            futures = {'observed_future':features[frame+h][None]}
            for arm, model in models.items():
                predicted = model(context,actions,torch.tensor([h]),torch.ones(1,768,dtype=torch.bool),control=control[None])
                futures[arm] = F.layer_norm(predicted.float(),(1024,))
            predictions = {k:np.asarray(v) for k,v in target['reference_predictions'].items()}
            for arm, tokens in futures.items():
                pooled = pool_tokens(tokens)
                predictions['bidirectional_'+arm] = head(current,pooled)[0].numpy()
                np.testing.assert_allclose(reference_head(current,pooled)[0].numpy(),
                    predictions['starting_mixed_'+arm],rtol=0,atol=2e-5)
            assert torch.equal(head(current,current),torch.zeros(1,3))
            errors, body_errors = {}, {}
            for name, values in predictions.items():
                for dest, truth in ((errors,target['actual']),(body_errors,target['body_target'])):
                    delta = values-np.asarray(truth)
                    delta[2] = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                    dest[name] = delta.tolist()
            rows.append(target | dict(predictions={k:v.tolist() for k,v in predictions.items()},
                                      errors=errors,body_target_errors=body_errors))
            if len(rows)%16 == 0:
                print('BIDIRECTIONAL_MAZE_ROWS',len(rows),len(targets),flush=True)
        summaries = {}
        for convention in ('planar','body'):
            scored = rows if convention == 'planar' else [r | dict(errors=r['body_target_errors']) for r in rows]
            summaries[convention] = {group:{str(h):panel.metrics([r for r in scored if r['group']==group and r['horizon_ms']==h])
                for h in (500,700)} for group in ('translation','turn')}
        panel.save(OUTPUT/'result.json',dict(status='COMPLETE',rows=rows,by_target_group_horizon=summaries,
            wall_s=time.monotonic()-started,plan_sha256=panel.fit.digest(OUTPUT/'plan.json'),
            reference_predictions_reproduced_atol=2e-5,identity_pairs_exactly_zero=True,
            automatically_promoted=False,navigation_tested=False))
        brief = {g:{h:{k:v for k,v in m.items() if k.startswith('bidirectional_') or k in ('command_history','starting_mixed_action')}
                    for h,m in horizons.items()} for g,horizons in summaries['planar'].items()}
        print('BIDIRECTIONAL_MAZE_COMPLETE',json.dumps(brief),flush=True)
    except BaseException as error:
        panel.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
