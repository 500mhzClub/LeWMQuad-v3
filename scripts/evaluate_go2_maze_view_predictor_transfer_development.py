"""Prediction/readout decomposition on the fixed, now-exposed maze-view panel."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import evaluate_go2_maze_view_transfer_development as decoder
from scripts import train_go2_horizon_dense_predictor_development as dynamics

ROOT, FIT = decoder.ROOT, decoder.FIT
OUTPUT = ROOT/'predictor_diagnostic'
save, digest = decoder.save, decoder.digest
KEYS = ('decision_ns', 'applied_command_values', 'applied_command_valid',
        'applied_command_measured_ns', 'applied_command_available_ns')


def prepare():
    assert not OUTPUT.exists()
    targets = json.loads((ROOT/'transfer_targets.json').read_text())
    prior = json.loads((ROOT/'evaluation/result.json').read_text())
    assert prior['status'] == 'COMPLETE' and len(prior['rows']) == len(targets)
    plan = json.loads((ROOT/'plan.json').read_text())
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    stats_path = dynamics.parent.reference.CACHE/'proprio_v1/proprio_norm_stats.json'
    stats = json.loads(stats_path.read_text())
    mean, std = [np.asarray(stats[k], np.float32) for k in ('control_mean', 'control_std')]
    samples, paths, identities = [], set(), {}
    maximum = 0.
    for case in sorted({r['case'] for r in targets}):
        directory = ROOT/f'case_{case:02d}'
        archive_path = directory/'policy_histories.npz'
        identities[str(archive_path)] = digest(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            data = {k:archive[k].copy() for k in KEYS}
        camera = json.loads((directory/'in_memory_camera_observations.json').read_text())['frames']
        for target in [r for r in targets if r['case']==case]:
            frame, h = target['frame'], target['horizon_ms']//100
            now = int(data['decision_ns'][frame])
            times = [camera[frame+d]['measured_ns'] for d in (-10,-5,0)]
            past = data['applied_command_values'][frame].astype(np.float32)
            measured = data['applied_command_measured_ns'][frame]
            assert times == [now-1_000_000_000, now-500_000_000, now]
            assert measured[[4,9,14]].tolist() == times
            assert np.all(np.diff(measured)==100_000_000)
            assert data['applied_command_valid'][frame].all()
            assert (data['applied_command_available_ns'][frame]<=now).all()
            assert (measured<=now).all() and np.isfinite(past).all()
            applied = np.asarray(apply_safety_limits_single(plan['tape'][frame:frame+8],
                tuple(past[-1]), limits)[0], np.float32)
            assert (past[:,1]==0).all() and (applied[:,1]==0).all()
            # Future measured commands verify alignment only; predictor inputs
            # above are calculated from the causal history and fixed plan.
            error = float(np.max(np.abs(applied[:h]-data['applied_command_values'][frame+h,-h:])))
            maximum = max(maximum, error)
            assert error < 1e-6
            context_paths = [str(directory/f'rgb_{frame+d:04d}.png') for d in (-10,-5,0)]
            paths.update(context_paths+[target['future_rgb']])
            samples.append(dict(case=case,frame=frame,horizon_ms=target['horizon_ms'],
                context_paths=context_paths,action=applied[:,[0,2]].tolist(),
                control=((past[:,[0,2]].reshape(3,5,2)-mean)/std).tolist()))
    assert len(samples)==len(targets)==240 and all(Path(p).is_file() for p in paths)
    OUTPUT.mkdir()
    save(OUTPUT/'samples.json',samples)
    save(OUTPUT/'plan.json',dict(source_sha256=digest(__file__),
        samples_sha256=digest(OUTPUT/'samples.json'),targets_sha256=digest(ROOT/'transfer_targets.json'),
        prior_decoder_sha256=digest(ROOT/'evaluation/result.json'),
        command_reference_sha256=digest(ROOT/'command_references/result.json'),
        readout_fit_sha256=digest(FIT/'result.json'),
        predictor_result_sha256=digest(dynamics.RESULT),
        normalization_sha256=digest(stats_path),public_history_sha256=identities,
        limiter_recording_maximum_difference=maximum,images=sorted(paths),
        device='cpu',cpu_cores=[0,1,2,3],feature_storage='FP32 dense tokens in RAM only',
        heads=['initial_mixed','old_data_final','maze_data_final'],
        feature_sources=['observed_future','action','no_future_action','persistence'],
        targets_changed=False,training=False,navigation=False,automatic_promotion=False,
        limitations=['Diagnostic selected after decoder results; same previously fixed 240 windows.',
            'Two same-family maze geometries and fixed motion tape; no new independent navigation test.',
            'Observed future is an evaluator comparison, never a predictor input.',
            'Command history receives the fixed requested tape; dense predictor receives its limiter projection.']))
    print('MAZE_PREDICTOR_PREPARED',len(samples),'rows',len(paths),'images',flush=True)


@torch.inference_mode()
def evaluate():
    plan = json.loads((OUTPUT/'plan.json').read_text())
    assert digest(__file__) == plan['source_sha256']
    for path,key in ((OUTPUT/'samples.json','samples_sha256'),(ROOT/'transfer_targets.json','targets_sha256'),
            (ROOT/'evaluation/result.json','prior_decoder_sha256'),
            (ROOT/'command_references/result.json','command_reference_sha256'),
            (FIT/'result.json','readout_fit_sha256'),(dynamics.RESULT,'predictor_result_sha256')):
        assert digest(path) == plan[key]
    assert not (OUTPUT/'result.json').exists() and not (OUTPUT/'failure.json').exists()
    started = time.monotonic()
    torch.set_num_threads(4)
    try:
        trained = json.loads((FIT/'result.json').read_text())
        assert trained['status']=='COMPLETE' and trained['steps']==decoder.training.STEPS
        heads = {'initial_mixed':decoder.training.prior.previous.load('mixed_data')}
        for arm in decoder.training.ARMS:
            path = FIT/f'{arm}_final.pt'
            assert digest(path)==trained['checkpoint_sha256'][arm]
            state = torch.load(path,map_location='cpu',weights_only=False)
            assert state['updates']==decoder.training.STEPS and state['plan_sha256']==digest(FIT/'plan.json')
            model = decoder.training.prior.previous.load('mixed_data')
            model.load_state_dict(state['model_state_dict'])
            heads[arm+'_final'] = model.eval().requires_grad_(False)
        targets = json.loads((ROOT/'transfer_targets.json').read_text())
        samples = json.loads((OUTPUT/'samples.json').read_text())
        key = lambda row:(row['case'],row['frame'],row['horizon_ms'])
        samples = {key(r):r for r in samples}
        prior = {key(r):r for r in json.loads((ROOT/'evaluation/result.json').read_text())['rows']}
        refs = {key(r):r for r in json.loads((ROOT/'command_references/result.json').read_text())['rows']}
        encoder = decoder.VJepa21Arm()
        encoder.build(torch.device('cpu'),torch.float32)
        features = {}
        for i,path in enumerate(plan['images']):
            features[path] = F.layer_norm(encoder.tokens(encoder.preprocess(path)[None]).float(),(1024,))[0]
            assert torch.isfinite(features[path]).all()
            if (i+1)%32==0 or i+1==len(plan['images']):
                print('MAZE_PREDICTOR_FEATURES',i+1,len(plan['images']),flush=True)
        del encoder
        models = {arm:dynamics.load(arm) for arm in dynamics.ARMS}
        rows, reproduction_max = [], 0.
        for target in targets:
            identity = key(target)
            sample = samples[identity]
            context = torch.stack([features[p] for p in sample['context_paths']])[None]
            actions = torch.tensor(sample['action'],dtype=torch.float32)[None]
            control = torch.tensor(sample['control'],dtype=torch.float32)[None]
            future = features[target['future_rgb']][None]
            futures = dict(observed_future=future,persistence=context[:,-1])
            for arm,model in models.items():
                tokens = model(context,actions,torch.tensor([target['horizon_ms']//100]),
                    torch.ones(1,768,dtype=torch.bool),control=control)
                futures[arm] = F.layer_norm(tokens.float(),(1024,))
            # Reproduce the completed decoder's pooled-FP16 comparison exactly.
            current = pool_tokens(context[:,-1])
            predictions = {k:np.asarray(v) for k,v in refs[identity]['predictions'].items()}
            latent = {}
            for arm,tokens in futures.items():
                latent[arm] = dict(mse=float(F.mse_loss(tokens,future)),l1=float(F.l1_loss(tokens,future)))
                pooled = pool_tokens(tokens)
                for name,head in heads.items():
                    c,f = (current.half().float(),pooled.half().float()) if arm=='observed_future' else (current,pooled)
                    prediction = head(c,f)[0].numpy()
                    predictions[name+'_'+arm] = prediction
                    if arm=='observed_future':
                        difference = float(np.max(np.abs(prediction-prior[identity]['predictions'][name])))
                        reproduction_max = max(reproduction_max,difference)
                        assert difference < 2e-5
            errors = {}
            for name,value in predictions.items():
                error = value-np.asarray(target['actual'])
                error[2] = np.arctan2(np.sin(error[2]),np.cos(error[2]))
                errors[name] = error.tolist()
            rows.append(target | dict(predictions={k:v.tolist() for k,v in predictions.items()},
                errors=errors,latent_errors=latent))
            if len(rows)%16==0:
                print('MAZE_PREDICTOR_ROWS',len(rows),len(targets),flush=True)
        def metrics(selected):
            return decoder.metrics(selected) | dict(latent={arm:{metric:float(np.mean([
                r['latent_errors'][arm][metric] for r in selected])) for metric in ('mse','l1')}
                for arm in futures})
        grouped = {str(m):{str(h):{g:metrics([r for r in rows if r['maze']==m
            and r['horizon_ms']==h and (g=='all' or r['group']==g)])
            for g in ('all','translation','turn','hold')} for h in (500,700)} for m in (0,1)}
        save(OUTPUT/'result.json',dict(status='COMPLETE',rows=rows,by_maze_horizon_group=grouped,
            plan_sha256=digest(OUTPUT/'plan.json'),wall_s=time.monotonic()-started,
            decoder_reproduction_maximum_difference=reproduction_max,
            navigation_tested=False,automatically_promoted=False))
        print('MAZE_PREDICTOR_COMPLETE',len(rows),flush=True)
    except BaseException as error:
        save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else evaluate()
