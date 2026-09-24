"""Frozen latent prediction and oracle motion decoding at the exposed stall."""
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.physical_execution_development import rotation_xyzw
from scripts import collect_go2_dense_stall_turn_branches_development as collection
from scripts import train_go2_horizon_dense_predictor_development as fit
from scripts import train_go2_dense_visual_motion_readout_development as motion_fit


def wrapped(angle):
    return np.arctan2(np.sin(angle),np.cos(angle))


@torch.inference_mode()
def main():
    root = collection.OUTPUT
    plan = collection.read(root/'plan.json')
    frame = plan['frame']; actions = plan['actions']; horizons = [h//100 for h in plan['horizons_ms']]
    for action in actions:
        result = collection.read(root/action/'result.json')
        assert result['status']=='COMPLETE' and not (root/action/'failure.json').exists()
        assert len(result['context_checks'])==3 and all(r['primary_rgb_exact'] for r in result['context_checks'])
    output = root/'evaluation'
    output.mkdir(exist_ok=False)
    save = lambda name,value:collection.save(output/name,value)
    save('plan.json',dict(source_sha256=collection.digest(__file__),collection_plan_sha256=collection.digest(root/'plan.json'),
        frame=frame,actions=actions,horizons_ms=plan['horizons_ms'],device='cpu',precision='float32',
        predictor_sha256=collection.read(fit.RESULT)['checkpoint_sha256'],
        motion_readout_sha256=collection.digest(motion_fit.OUTPUT/'readout.pt'),
        no_training=True,future_images_target_only=True,closed_loop_navigation=False,
        purpose='Separate branch prediction from frozen physical-readout transfer at one exposed stalled context.'))
    started = time.monotonic()
    try:
        torch.set_num_threads(4)
        reference = fit.parent.reference
        encoder = reference.encoders.VJepa21Arm()
        encoder.build(torch.device('cpu'),torch.float32)
        cache = {}

        def encode(path):
            key = collection.digest(path)
            if key not in cache:
                pixels = encoder.preprocess(str(path))[None]
                cache[key] = F.layer_norm(encoder.tokens(pixels).float(),(1024,))[0]
            return cache[key]

        # All forecasts are computed from original causal images/history before
        # any newly collected future image is loaded.
        context = torch.stack([encode(collection.REFERENCE/'native'/f'rgb_{f:04d}.png')
            for f in (frame-10,frame-5,frame)])[None]
        with np.load(collection.REFERENCE/'native/policy_histories.npz',allow_pickle=False) as archive:
            past = archive['applied_command_values'][frame].astype(np.float32)
        assert past.shape==(15,3)
        stats = collection.read(reference.CACHE/'proprio_v1/proprio_norm_stats.json')
        control = (torch.from_numpy(past[:,[0,2]].reshape(3,5,2))-torch.tensor(stats['control_mean']))/torch.tensor(stats['control_std'])
        control = control.float()[None]
        tape = torch.tensor([plan['branches'][a]['applied'] for a in actions],dtype=torch.float32)[:,:,[0,2]]
        mask = torch.ones(3,768,dtype=torch.bool)
        head = motion_fit.load(); current = pool_tokens(context[:,-1])
        forecasts = {}; decoded = {}
        for arm in ('action','no_future_action'):
            predictor = fit.load(arm)
            forecasts[arm] = {}; decoded[arm] = {}
            for h in horizons:
                n = 1 if arm=='no_future_action' or h<=3 else 3
                z = predictor(context.expand(n,-1,-1,-1),tape[:n],torch.full((n,),h,dtype=torch.long),
                    mask[:n],control=control.expand(n,-1,-1,-1))
                z = F.layer_norm(z.float(),(1024,)).expand(3,-1,-1)
                forecasts[arm][h] = z
                decoded[arm][h] = head(current.expand(3,-1,-1),pool_tokens(z)).numpy()
                if arm=='action':
                    saved = np.asarray([plan['branches'][a]['saved_predicted_motion'][h-1] for a in actions])
                    np.testing.assert_allclose(decoded[arm][h],saved,rtol=0,atol=2e-5)
            del predictor

        truths = {}; observed = {}; oracle = {}; records = []
        for i,action in enumerate(actions):
            directory = root/action
            metadata = {r['frame']:r for r in collection.read(directory/'in_memory_camera_observations.json')['frames']}
            with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
                poses = archive['base_pose_world'].copy()
            origin = poses[metadata[frame]['physical_sample_index']]
            rotation = rotation_xyzw(origin[3:])
            for h in horizons:
                target = poses[metadata[frame+h]['physical_sample_index']]
                future_rotation = rotation_xyzw(target[3:]); relative = rotation.T@future_rotation
                actual = np.r_[((target[:3]-origin[:3])@rotation)[:2],np.arctan2(relative[1,0],relative[0,0])]
                z = encode(directory/f'rgb_{frame+h:04d}.png')
                decoded_oracle = head(current,pool_tokens(z[None]))[0].numpy()
                truths[i,h] = actual; observed[i,h] = z; oracle[i,h] = decoded_oracle
                predictions = {a:decoded[a][h][i] for a in decoded} | dict(observed_future=decoded_oracle)
                errors = {}
                for name,value in predictions.items():
                    error = value-actual; error[2] = wrapped(error[2]); errors[name] = error.tolist()
                records.append(dict(action=action,horizon_ms=h*100,actual_xy_yaw=actual.tolist(),
                    actual_world_heading_change_rad=float(wrapped(np.arctan2(future_rotation[1,0],future_rotation[0,0])-np.arctan2(rotation[1,0],rotation[0,0]))),
                    predictions={k:v.tolist() for k,v in predictions.items()},errors=errors))
        latent = []
        for h in horizons:
            target = torch.stack([observed[i,h] for i in range(3)])
            models = forecasts.keys()
            for arm in models:
                matrix = ((forecasts[arm][h][:,None]-target[None])**2).mean((-1,-2)).numpy()
                # Columns hold a fixed actual future; compare correct versus
                # wrong command forecasts. Identical blind forecasts are ties.
                wins = [bool(matrix[i,i] < min(matrix[j,i] for j in range(3) if j!=i)) for i in range(3)]
                latent.append(dict(arm=arm,horizon_ms=h*100,rows_predicted_action_columns_observed_action=matrix.tolist(),
                    correct_action_beats_both_wrong_forecasts=wins,strict_wins=sum(wins),
                    prebranch=h<=3,prebranch_retrieval_applicable=h>3))
            latent.append(dict(arm='current_image_persistence',horizon_ms=h*100,
                error_by_observed_action=((context[0,-1][None]-target)**2).mean((-1,-2)).tolist()))
        increments = []
        for i,action in enumerate(actions):
            actual = truths[i,7]-truths[i,3]; actual[2] = wrapped(actual[2])
            predictions = {a:decoded[a][7][i]-decoded[a][3][i] for a in decoded}
            predictions['observed_future'] = oracle[i,7]-oracle[i,3]
            for value in predictions.values(): value[2] = wrapped(value[2])
            increments.append(dict(action=action,actual_300_to_700_xy_yaw=actual.tolist(),
                predictions={k:v.tolist() for k,v in predictions.items()},
                actual_turn_toward_requested_view=bool(actual[2]*plan['scan_heading_error_rad']>0)))
        result = dict(status='COMPLETE',rows=records,latent=latent,commit_interval=increments,
            action_order=actions,frame=frame,wall_s=time.monotonic()-started,
            saved_action_forecasts_reproduced_atol=2e-5,no_training=True,
            native_motion_evaluator_only=True,actual_future_unavailable_online=True,
            limitations=['one post hoc exposed context','fixed counterfactual command tapes, not navigation',
                'readout trained at 500 ms; other horizons are temporal transfer',
                'branch discrimination is not JEPA-training contribution or obstacle-dependent dynamics evidence'])
        save('result.json',result)
        print('STALL_BRANCH_EVALUATION',json.dumps(increments),flush=True)
    except BaseException as error:
        save('failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
