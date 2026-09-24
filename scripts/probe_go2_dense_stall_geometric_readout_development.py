"""One fixed feature/PnP diagnostic; no fitting or navigation intervention."""
import json
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_feature_pnp_readout_development import decode, SETTINGS
from scripts import collect_go2_dense_stall_turn_branches_development as branch
from scripts import train_go2_horizon_dense_predictor_development as fit
from scripts.live_depth_noise_session_development import NoisyPublicReplay

OUTPUT = branch.OUTPUT/'geometric_readout_v1'


@torch.inference_mode()
def main():
    plan = branch.read(branch.OUTPUT/'plan.json')
    reference = branch.read(branch.OUTPUT/'evaluation/result.json')
    assert reference['status']=='COMPLETE'
    OUTPUT.mkdir(exist_ok=False)
    save = lambda name,value:branch.save(OUTPUT/name,value)
    save('plan.json',dict(settings=SETTINGS,frame=plan['frame'],horizons_ms=plan['horizons_ms'],
        actions=plan['actions'],device='cuda float32 features; CPU OpenCV geometry',
        sources={p:branch.digest(p) for p in (__file__,'lewm/dense_feature_pnp_readout_development.py')},
        native_current_depth_reconstructed_with_delivered_noise=True,
        future_depth_unavailable=True,no_fit=True,no_navigation=True,
        comparison='same fixed geometric decoder on actual, action-predicted, blind-predicted, and persistent features',
        limitations=['one exposed diagnostic context','coarse semantic token correspondences may not track physical points',
            'static scene assumption','no threshold search or automatic promotion']))
    started = time.monotonic()
    try:
        torch.set_num_threads(4); cv2.setNumThreads(1)
        frame = plan['frame']; actions = plan['actions']; horizons = [h//100 for h in plan['horizons_ms']]
        replay = NoisyPublicReplay(branch.REFERENCE/'native')
        _,depth,_,_,_,now = replay.packet(frame)
        assert now==plan['observed_ns']
        encoder = fit.parent.reference.encoders.VJepa21Arm()
        encoder.build(torch.device('cuda:0'),torch.float32)
        cache = {}

        def encode(path):
            key = branch.digest(path)
            if key not in cache:
                cache[key] = F.layer_norm(encoder.tokens(encoder.preprocess(str(path))[None].cuda()).float(),(1024,))[0]
            return cache[key]

        context = torch.stack([encode(branch.REFERENCE/'native'/f'rgb_{f:04d}.png') for f in (frame-10,frame-5,frame)])[None]
        stats = branch.read(fit.parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json')
        with np.load(branch.REFERENCE/'native/policy_histories.npz',allow_pickle=False) as archive:
            past = archive['applied_command_values'][frame].astype(np.float32)
        control = ((torch.from_numpy(past[:,[0,2]].reshape(3,5,2))-torch.tensor(stats['control_mean']))/torch.tensor(stats['control_std'])).float()[None].cuda()
        tape = torch.tensor([plan['branches'][a]['applied'] for a in actions],dtype=torch.float32,device='cuda')[:,:,[0,2]]
        mask = torch.ones(3,768,dtype=torch.bool,device='cuda')
        forecasts = {}
        for arm in ('action','no_future_action'):
            model = fit.load(arm).cuda()
            forecasts[arm] = {}
            for h in horizons:
                n = 1 if h<=3 or arm=='no_future_action' else 3
                predicted = model(context.expand(n,-1,-1,-1),tape[:n],torch.full((n,),h,dtype=torch.long,device='cuda'),
                    mask[:n],control=control.expand(n,-1,-1,-1))
                forecasts[arm][h] = F.layer_norm(predicted.float(),(1024,)).expand(3,-1,-1)
            del model
        rows = []
        for i,action in enumerate(actions):
            for h in horizons:
                truth = next(r for r in reference['rows'] if r['action']==action and r['horizon_ms']==h*100)['actual_xy_yaw']
                features = {arm:values[h][i] for arm,values in forecasts.items()}
                features.update(observed_future=encode(branch.OUTPUT/action/f'rgb_{frame+h:04d}.png'),persistence=context[0,-1])
                results = {}
                for name,value in features.items():
                    result = decode(context[0,-1],value,depth)
                    if result['valid']:
                        error = np.asarray(result['motion_xy_yaw'])-truth
                        error[2] = np.arctan2(np.sin(error[2]),np.cos(error[2]))
                        result['error_xy_yaw'] = error.tolist()
                    results[name] = result
                rows.append(dict(action=action,horizon_ms=h*100,actual_xy_yaw=truth,readouts=results))
        metrics = {}
        for name in ('action','no_future_action','observed_future','persistence'):
            valid = [r['readouts'][name] for r in rows if r['readouts'][name]['valid']]
            errors = np.asarray([r['error_xy_yaw'] for r in valid])
            metrics[name] = dict(valid=len(valid),total=len(rows),
                xy_rmse_mm=float(1000*np.sqrt(np.mean(np.sum(errors[:,:2]**2,axis=1)))) if len(valid) else None,
                yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(errors[:,2]**2)))) if len(valid) else None)
        result = dict(status='COMPLETE',rows=rows,metrics=metrics,wall_s=time.monotonic()-started,
            no_fit=True,no_navigation=True,invalid_estimates_not_replaced_by_command_motion=True,
            native_truth_used_only_for_evaluation=True)
        save('result.json',result)
        print('GEOMETRIC_READOUT',json.dumps(metrics),flush=True)
    except BaseException as error:
        save('failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
