"""Prospective actual-future motion decoding on separately fixed maze views."""
import argparse
from collections import Counter
import json
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts import collect_go2_maze_view_transfer_development as collection
from scripts import train_go2_maze_view_readout_development as training

ROOT, FIT = collection.OUTPUT, collection.FIT_ROOT
OUTPUT = ROOT/'evaluation'
save, digest = collection.save, collection.digest


def prepare():
    assert not (FIT/'result.json').exists(), 'fix evaluation targets before fit results'
    plan = json.loads((ROOT/'plan.json').read_text())
    targets, cases, exclusions = [], [], []
    for case in range(collection.layouts.CASE_COUNT):
        directory = ROOT/f'case_{case:02d}'
        result = json.loads((directory/'result.json').read_text())
        spec = json.loads((directory/'specification.json').read_text())
        assert result['data_role'] == spec['data_role'] == 'development_transfer'
        frames = json.loads((directory/'in_memory_camera_observations.json').read_text())['frames']
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as data:
            poses, contacts = data['base_pose_world'].copy(), data['physics_contact'].copy()
        count = 0
        for frame in plan['evaluation']['departures']:
            for horizon in plan['evaluation']['horizons_ms']:
                h = horizon//100
                if frame+h >= len(frames):
                    exclusions.append(dict(case=case, frame=frame, horizon_ms=horizon, reason='missing_future'))
                    continue
                start, end = [frames[f]['physical_sample_index'] for f in (frame,frame+h)]
                if contacts[:end+1].any():
                    exclusions.append(dict(case=case, frame=frame, horizon_ms=horizon, reason='contact_prefix'))
                    continue
                assert frames[frame+h]['measured_ns']-frames[frame]['measured_ns'] == horizon*1_000_000
                rotation = collection.previous.rotation_xyzw(poses[start,3:])
                relative = rotation.T@collection.previous.rotation_xyzw(poses[end,3:])
                delta = (poses[end,:3]-poses[start,:3])@rotation
                action = plan['phases'][frame].split('_',2)[2]
                group = 'hold' if action=='hold' else 'turn' if action.endswith('_turn') else 'translation'
                targets.append(dict(case=case, maze=spec['evaluation_maze_index'], frame=frame,
                    horizon_ms=horizon, action_at_departure=action, group=group,
                    actual=[float(delta[0]),float(delta[1]),float(np.arctan2(relative[1,0],relative[0,0]))],
                    current_rgb=str(directory/f'rgb_{frame:04d}.png'),
                    future_rgb=str(directory/f'rgb_{frame+h:04d}.png')))
                count += 1
        cases.append(result | dict(evaluation_windows=count))
    assert targets
    save(ROOT/'transfer_targets.json', targets)
    save(ROOT/'collection_result.json', dict(status='COMPLETE', cases=cases,
        evaluation_windows=len(targets), exclusions=exclusions, training_used=False,
        complete_recordings=sum(c['status']=='COMPLETE' for c in cases),
        physical_stops=sum(c['status']=='PHYSICAL_STOP' for c in cases)))
    save(ROOT/'evaluation_plan.json', dict(collection_plan_sha256=digest(ROOT/'plan.json'),
        targets_sha256=digest(ROOT/'transfer_targets.json'), source_sha256=digest(__file__),
        fit_plan_sha256=digest(FIT/'plan.json'),
        target_convention='Initial full-body-frame translation and atan2 of relative body rotation, matching training.',
        grouping='Command at departure; windows can cross command transitions.',
        heads=['initial_mixed','old_data_final','maze_data_final'], device='cpu',
        fitting=False, checkpoint_selection=False, navigation=False,
        feature_storage='pooled FP16 RAM only',
        limitations=['Two independent maze geometries, four contexts each; rows within a maze are dependent.',
            'Actual-future decoder transfer; no action-conditioned forecast evaluation.',
            'Same fixed motion tape and view-selection rule as the training collection.']))
    print('MAZE_TRANSFER_TARGETS', len(targets), 'exclusions', len(exclusions), flush=True)


def metrics(rows):
    if not rows:
        return dict(windows=0)
    values = {}
    for name in rows[0]['errors']:
        error = np.asarray([r['errors'][name] for r in rows])
        values[name] = dict(xy_rmse_mm=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))*1000),
            yaw_rmse_deg=float(np.sqrt(np.mean(error[:,2]**2))*180/np.pi))
    return dict(windows=len(rows), metrics=values)


@torch.inference_mode()
def evaluate():
    plan = json.loads((ROOT/'evaluation_plan.json').read_text())
    assert digest(__file__) == plan['source_sha256']
    assert digest(ROOT/'transfer_targets.json') == plan['targets_sha256']
    assert digest(FIT/'plan.json') == plan['fit_plan_sha256']
    result = json.loads((FIT/'result.json').read_text())
    assert result['status']=='COMPLETE' and result['steps']==training.STEPS
    heads = {'initial_mixed':training.prior.previous.load('mixed_data')}
    identities = {'initial_mixed':json.loads((FIT/'plan.json').read_text())['initial_checkpoint_sha256']}
    for arm in training.ARMS:
        path = FIT/f'{arm}_final.pt'
        assert digest(path) == result['checkpoint_sha256'][arm]
        state = torch.load(path,map_location='cpu',weights_only=False)
        assert state['updates']==training.STEPS and state['plan_sha256']==plan['fit_plan_sha256']
        model = training.prior.previous.load('mixed_data')
        model.load_state_dict(state['model_state_dict'])
        heads[arm+'_final'] = model.eval().requires_grad_(False)
        identities[arm+'_final'] = digest(path)
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    try:
        targets = json.loads((ROOT/'transfer_targets.json').read_text())
        paths = sorted({r[k] for r in targets for k in ('current_rgb','future_rgb')})
        encoder = VJepa21Arm()
        encoder.build(torch.device('cpu'),torch.float32)
        features = {}
        for i,path in enumerate(paths):
            tokens = F.layer_norm(encoder.tokens(encoder.preprocess(path)[None]).float(),(1024,))
            assert torch.isfinite(tokens).all()
            features[path] = pool_tokens(tokens).half()
            if (i+1)%32==0 or i+1==len(paths):
                print('MAZE_TRANSFER_FEATURES',i+1,len(paths),flush=True)
        del encoder
        rows = []
        for target in targets:
            current, future = [features[target[k]].float() for k in ('current_rgb','future_rgb')]
            predictions = {name:model(current,future)[0].numpy() for name,model in heads.items()}
            errors = {}
            for name,value in predictions.items():
                error = value-np.asarray(target['actual'])
                error[2] = np.arctan2(np.sin(error[2]),np.cos(error[2]))
                errors[name] = error.tolist()
            rows.append(target | dict(predictions={k:v.tolist() for k,v in predictions.items()},errors=errors))
        summary = {str(maze):{str(h):{group:metrics([r for r in rows if r['maze']==maze
            and r['horizon_ms']==h and (group=='all' or r['group']==group)])
            for group in ('all','translation','turn','hold')} for h in (500,700)} for maze in (0,1)}
        save(OUTPUT/'result.json', dict(status='COMPLETE',rows=rows,by_maze_horizon_group=summary,
            checkpoint_sha256=identities, images=len(paths), wall_s=time.monotonic()-started,
            evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'),
            trained_on_evaluation=False,navigation_tested=False,automatically_promoted=False))
        print('MAZE_TRANSFER_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as error:
        save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else evaluate()
