"""Fixed 384-window dense visual prediction diagnostic on four saved missions.

No new navigation or fitting. Both failed returns are included. Selection is
evenly spaced within each recording and fixed before predictor evaluation.
"""
import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from scripts import train_go2_frozen_vjepa_native_adaptation_development as training

reference = training.reference
PLAN = Path('docs/go2_dense_native_recordings_plan_2026-09-17.json')
RESULT = Path('docs/go2_dense_native_recordings_result_2026-09-17.json')
OUTPUT = training.OUTPUT/'recorded_mission_evaluation'
SOURCE_PLAN = Path('docs/go2_anchored_visual_navigation_plan_2026-09-17.json')
ARMS = ('action', 'no_future_action', 'unchanged_rollout', 'persistence')


def prepare():
    roots = []
    for record in json.loads(SOURCE_PLAN.read_text())['roots']:
        root = Path(record['root'])
        path = root/'saved_executed_motion_forecast_evaluation_v1.json'
        assert reference.digest(path) == record['window_sha256']
        windows = json.loads(path.read_text())['rows']
        eligible = sorted((w for w in windows if w['frame'] >= 10), key=lambda w:w['frame'])
        assert len(eligible) >= 96
        selected = [eligible[i] for i in np.linspace(0, len(eligible)-1, 96, dtype=int)]
        assert len({w['frame'] for w in selected}) == 96
        roots.append(dict(root=str(root), eligible=len(eligible),
            excluded_short_history=len(windows)-len(eligible), windows=selected,
            action_counts=dict(Counter(w['action'] for w in selected)),
            window_sha256=reference.digest(path),
            planning_sha256=reference.digest(root/'planning.json')))
    plan = dict(roots=roots, windows=384, arms=ARMS, horizon_ms=500,
        context_offsets_ms=[-1000,-500,0], samples_per_recording=96,
        selection='96 evenly spaced eligible rows per recording, integer linspace including endpoints',
        source_sha256=reference.digest(__file__),
        fit_plan_sha256=reference.digest(training.PLAN),
        all_four_recordings_including_failed_returns=True,
        selection_uses_prediction_errors=False, no_fitting=True, new_navigation=False,
        pooling='equal sample count per recording; not natural-frequency all-window estimate',
        overlap_is_not_independent_replication=True, sensor_reads='RGB and applied-command histories only')
    reference.save(PLAN, plan)
    print('PREPARED', sum(r['eligible'] for r in roots), 'eligible;', plan['windows'], 'selected', flush=True)
    print(json.dumps([r['action_counts'] for r in roots]), flush=True)


def planned_commands(p, action):
    pulse = p['motion_correction']['terminal_translation_pulse']
    count = 1 if pulse and action in ('forward','left_arc','right_arc') else 4
    return (p['committed_prefix'] + [candidate_commands(action)[0]]*count + [[0.,0.,0.]]*4)[:5]


def summarize(rows):
    return dict(windows=len(rows), models={arm:dict(
        mse=float(np.mean([r['mse'][arm] for r in rows])),
        mse_over_persistence=float(sum(r['mse'][arm] for r in rows)/sum(r['mse']['persistence'] for r in rows)))
        for arm in ARMS})


@torch.inference_mode()
def run():
    plan = json.loads(PLAN.read_text())
    assert reference.digest(__file__) == plan['source_sha256']
    assert reference.digest(training.PLAN) == plan['fit_plan_sha256']
    terminal = json.loads((training.OUTPUT/'result.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['epochs'] == training.EPOCHS
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic(); torch.set_num_threads(4)
    models = {}
    for arm in ARMS[:3]:
        path = training.INITIAL if arm == 'unchanged_rollout' else training.OUTPUT/f'{arm}_latest.pt'
        expected = json.loads(training.PLAN.read_text())['initialization_sha256'] if arm == 'unchanged_rollout' else terminal['checkpoint_sha256'][arm]
        assert reference.digest(path) == expected
        state = torch.load(path, map_location='cpu', weights_only=False)
        if arm != 'unchanged_rollout': assert state['epoch'] == training.EPOCHS-1
        model = reference.ProprioActionPredictor(use_proprio=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        models[arm] = model.cuda().eval().requires_grad_(False)
        del state
    encoder = reference.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'), torch.float32)
    stats = json.loads((reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    mean, std = (np.asarray(stats[k], np.float32) for k in ('control_mean','control_std'))
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    cache = OrderedDict()
    encoded = 0

    def encode(path):
        nonlocal encoded
        key = reference.digest(path)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        pixel = encoder.preprocess(str(path))[None].cuda()
        value = F.layer_norm(encoder.tokens(pixel).float(), (1024,))[0].cpu()
        cache[key] = value; encoded += 1
        if len(cache) > 64: cache.popitem(last=False)
        return value

    all_rows = []; summaries = []
    for number, record in enumerate(plan['roots'], 1):
        root = Path(record['root']); native = root/'native'
        assert reference.digest(root/'planning.json') == record['planning_sha256']
        assert reference.digest(root/'saved_executed_motion_forecast_evaluation_v1.json') == record['window_sha256']
        plans = {p['frame']:p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p}
        with np.load(native/'policy_histories.npz', allow_pickle=False) as archive:
            body = {k:archive[k].copy() for k in ('image_ns','applied_command_values',
                'applied_command_valid','applied_command_measured_ns','applied_command_available_ns')}
        rows = []
        for ordinal, window in enumerate(record['windows'], 1):
            f = window['frame']; p = plans[f]; now = p['measured_ns']
            frames = [f-10,f-5,f]
            assert body['image_ns'][frames].tolist() == [now-1_000_000_000,now-500_000_000,now]
            commands = body['applied_command_values'][f].astype(np.float32)
            assert body['applied_command_valid'][f].all()
            assert (body['applied_command_available_ns'][f] <= now).all()
            assert body['applied_command_measured_ns'][f][[4,9,14]].tolist() == body['image_ns'][frames].tolist()
            applied, _ = apply_safety_limits_single(planned_commands(p,window['action']),tuple(commands[-1]),limits)
            applied = np.asarray(applied,np.float32)
            assert not commands[:,1].any() and not applied[:,1].any()
            x = torch.stack([encode(native/f'rgb_{i:04d}.png') for i in frames])[None].cuda()
            action = torch.from_numpy(applied[:,[0,2]].reshape(1,10)).cuda()
            control = torch.from_numpy(((commands[:,[0,2]].reshape(3,5,2)-mean)/std)[None]).cuda()
            mask = torch.ones(1,768,dtype=torch.bool,device='cuda')
            predicted = {arm:F.layer_norm(model(x,torch.zeros_like(action) if arm == 'no_future_action' else action,
                mask,control=control).float(),(1024,))[0].cpu() for arm,model in models.items()}
            predicted['persistence'] = x[0,-1].cpu()
            # Future commands and RGB are target-side checks, after forecasts.
            assert body['image_ns'][f+5] == now+500_000_000
            np.testing.assert_allclose(applied,body['applied_command_values'][f+5,-5:],rtol=0,atol=1e-6)
            target = encode(native/f'rgb_{f+5:04d}.png')
            rows.append(dict(run=number,frame=f,action=window['action'],
                terminal_translation_pulse=bool(p['motion_correction']['terminal_translation_pulse']),
                mse={arm:float((value-target).square().mean()) for arm,value in predicted.items()}))
            if ordinal%24 == 0: print('DENSE_RECORDING',number,ordinal,'encoded',encoded,'seconds',round(time.monotonic()-started,1),flush=True)
        summary = dict(run=number,root=str(root),total=summarize(rows),
            by_action={a:summarize([r for r in rows if r['action']==a]) for a in ACTIONS if any(r['action']==a for r in rows)})
        reference.save(OUTPUT/f'run_{number:02d}.json',dict(summary=summary,rows=rows))
        summaries.append(summary); all_rows.extend(rows)
    result = dict(status='COMPLETE',runs=summaries,total=summarize(all_rows),
        by_action={a:summarize([r for r in all_rows if r['action']==a]) for a in ACTIONS if any(r['action']==a for r in all_rows)},
        plan_sha256=reference.digest(PLAN),fit_sha256=reference.digest(training.OUTPUT/'result.json'),
        encoded_frames=encoded,wall_s=time.monotonic()-started,new_navigation=False,
        no_dense_cache_retained=True,physical_outcomes_not_scored=True,
        limitations=['384 systematically sampled dependent windows in four exposed recordings',
            'one training seed; both failed returns included',
            'executed-action feature prediction, not counterfactual ranking or navigation improvement'])
    reference.save(OUTPUT/'result.json',result);reference.save(RESULT,result)
    print('DENSE_RECORDINGS_COMPLETE',json.dumps(result['total']),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    else:
        try: run()
        except Exception as error:
            if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
                reference.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
            raise
