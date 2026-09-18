"""Read-only transfer of existing dense V-JEPA predictors to native RGB branches.

No training or simulation. Keep all 36 existing pulse branches, both roles;
forecast at 500 ms from three observed frames 500 ms apart. All predictions
are made before future RGB is loaded. Store small metric/distance records,
not regenerable dense feature caches. Historical checkpoints stay untouched.
"""
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single

from scripts import dev_frozen_dense_representation_encoders_v1 as encoders
from scripts.dev_proprio_predictor_v1 import ProprioActionPredictor
from scripts.probe_go2_jepa_latent_branch_science_development import selected_rows, data

CACHE = Path('/home/andrewknowles/.cache/lewm_go2_temporal_v03')
OUTPUT = Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_vjepa_native_branches_v1_attempt_003')
RESULT = Path('docs/go2_frozen_vjepa_native_branches_result_2026-09-17.json')
ARMS = ('rgb_one_step', 'rgb_rollout')
SEED = 2026080901  # first registered seed, selected before new outcomes


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def inputs(row):
    directory = data.PULSE / row['trial']
    spec = json.loads((directory/'branch_specification.json').read_text())
    observations = json.loads((directory/'policy_observations.json').read_text())
    assert spec['branch_tick'] == 13 and len(observations['frames']) == 26
    times = [observations['frames'][i]['image_ns'] for i in (3, 8, 13, 18)]
    assert np.diff(times).tolist() == [500_000_000]*3
    with np.load(directory/'policy_histories.npz', allow_pickle=False) as archive:
        # Read only the causal departure slice; no future body state is used.
        commands = archive['applied_command_values'][13].astype(np.float32)
        assert archive['applied_command_valid'][13].all()
        command_times = archive['applied_command_measured_ns'][13]
        assert command_times[[4, 9, 14]].tolist() == times[:3]
        assert (archive['applied_command_available_ns'][13] <= times[2]).all()
    future = np.asarray(spec['prospective_commands'][13:18], dtype=np.float32)
    np.testing.assert_allclose(future, np.asarray(row['known_commands'][:5]), atol=1e-7, rtol=0)
    assert np.all(future[:, 1] == 0) and np.all(commands[:, 1] == 0)
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    # The historical predictor consumes applied commands, whereas the recent
    # pulse windows store requested commands. Reconstruct using the live
    # platform limiter and the observed previous command, never future motion.
    future, _ = apply_safety_limits_single(future.tolist(), tuple(commands[-1]), limits)
    future = np.asarray(future, dtype=np.float32)
    return directory, commands[:, [0, 2]].reshape(3, 5, 2), future[:, [0, 2]].reshape(10)


@torch.inference_mode()
def run():
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(2026091703)
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    rows = selected_rows()
    stats_path = CACHE/'proprio_v1/proprio_norm_stats.json'
    stats = json.loads(stats_path.read_text())
    mean, std = (np.asarray(stats[k], dtype=np.float32) for k in ('control_mean', 'control_std'))
    paths = {arm: CACHE/f'factorial_v1/seed_{SEED}/seed_{SEED}_{arm}_epoch21.pt' for arm in ARMS}
    prepared = [inputs(row) for row in rows]
    plan = dict(schema='frozen_vjepa_native_branches.v1', seed=SEED, checkpoint_epoch=21,
        arms=ARMS, contexts=len(rows), context_frames=[3,8,13], target_frame=18,
        horizon_ms=500, source_sha256=digest(__file__),
        checkpoint_sha256={a:digest(p) for a,p in paths.items()},
        normalization_sha256=digest(stats_path), windows_sha256=digest(data.PULSE/'windows.json'),
        command_semantics='platform post-limiter trajectory reconstructed from requested tape and causal last applied command',
        platform_sha256=digest('config/go2_platform_manifest.yaml'),
        encoder_checkpoint=str(encoders.VJEPA_CHECKPOINT),
        encoder_source_commit=encoders.VJEPA_REPOSITORY_COMMIT,
        precision='float32 encoder and predictor, no autocast',
        input='full native 4:3 RGB resized directly to 512x384, ImageNet normalization; no square-image crop',
        hardware=dict(gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                      total_vram=torch.cuda.get_device_properties(device).total_memory),
        concurrency='one GPU process; encoder batch 1 with identical-image reuse; predictor batch 3; four CPU threads',
        no_training=True, no_simulation=True, no_depth_reads=True,
        target_mask='all tokens; do not apply historical scene-dependent changed-token thresholds',
        limits=['one preselected seed', 'two already exposed geometries per role',
                'native sensor/domain and pulse-command distribution differ from historical training',
                'component transfer diagnostic, not geometry probe or planning utility'])
    save(OUTPUT/'plan.json', plan)
    models = {}
    for arm, path in paths.items():
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = ProprioActionPredictor(use_proprio=False)
        model.load_state_dict(state['model_state_dict'], strict=True)
        models[arm] = model.to(device).eval().requires_grad_(False)
        del state
    encoder = encoders.VJepa21Arm()
    encoder.build(device, torch.float32)
    print('FROZEN_MODELS_LOADED', flush=True)
    token_cache = {}

    def encode(paths_to_encode):
        values = []
        for path in paths_to_encode:
            key = digest(path)
            if key not in token_cache:
                with Image.open(path) as im:
                    if im.size[0]*3 != im.size[1]*4:
                        raise ValueError(f'native 4:3 image expected: {path}')
                pixels = encoder.preprocess(str(path))[None].to(device)
                raw = encoder.tokens(pixels)
                assert tuple(raw.shape) == (1,768,1024)
                token_cache[key] = F.layer_norm(raw.float(), (1024,))[0].cpu()
            values.append(token_cache[key])
        return torch.stack(values)

    contexts = []
    for i, (directory, _, _) in enumerate(prepared):
        contexts.append(encode([directory/f'rgb_{f:04d}.png' for f in (3,8,13)]))
        if i % 6 == 5:
            print('CAUSAL_CONTEXTS', i+1, 'unique_frames', len(token_cache), flush=True)
    contexts = torch.stack(contexts)
    control = torch.from_numpy(np.stack([(c-mean)/std for _,c,_ in prepared]))
    actions = torch.from_numpy(np.stack([a for _,_,a in prepared]))
    predictions = {a:[] for a in ARMS}
    for start in range(0, len(rows), 3):
        context, action, c = (v[start:start+3].to(device) for v in (contexts, actions, control))
        mask = torch.ones(len(context), 768, dtype=torch.bool, device=device)
        for arm, model in models.items():
            pred = F.layer_norm(model(context, action, mask, control=c).float(), (1024,))
            assert torch.isfinite(pred).all()
            predictions[arm].append(pred.cpu())
        print('CAUSAL_FORECASTS', start+len(context), flush=True)
    predictions = {a:torch.cat(v) for a,v in predictions.items()}
    save(OUTPUT/'causal_predictions_complete.json', dict(
        future_rgb_loaded=False, contexts=len(rows),
        prediction_sha256={a:hashlib.sha256(v.numpy().tobytes()).hexdigest() for a,v in predictions.items()},
        seconds=time.monotonic()-started))
    # Future-side scoring begins only after every predictor has finished.
    targets = []
    for i, (directory, _, expected) in enumerate(prepared):
        with np.load(directory/'policy_histories.npz', allow_pickle=False) as archive:
            executed = archive['applied_command_values'][18][-5:, [0,2]].reshape(10)
        np.testing.assert_allclose(executed, expected, atol=1e-6, rtol=0)
        targets.append(encode([directory/'rgb_0018.png'])[0])
        if i % 6 == 5:
            print('FUTURE_TARGETS', i+1, flush=True)
    target = torch.stack(targets)
    predictions['persistence'] = contexts[:, -1]
    groups = {}
    for i,row in enumerate(rows):
        key = (row['data_role'],row['cluster'],row['prefix_action'])
        groups.setdefault(key,[]).append(i)
    details = []
    for key, indices in groups.items():
        assert len(indices) == 3
        # Same-history branches must genuinely share the encoded observations.
        for i in indices[1:]:
            assert torch.equal(contexts[indices[0]], contexts[i])
            assert torch.equal(control[indices[0]], control[i])
        truth = target[indices]
        record = dict(role=key[0],cluster=key[1],prefix_action=key[2],
                      trials=[rows[i]['trial'] for i in indices],
                      actions=[rows[i]['pulse_action'] for i in indices], models={})
        for arm, values in predictions.items():
            p = values[indices]
            distances = (p[:,None] - truth[None]).square().mean((-1,-2))
            centered_error = ((p-p.mean(0))-(truth-truth.mean(0))).square().mean()
            centered_zero = (truth-truth.mean(0)).square().mean()
            wins = [bool(distances[j,j] < torch.cat((distances[:j,j],distances[j+1:,j])).min())
                    for j in range(3)]
            record['models'][arm] = dict(mse_matrix=distances.tolist(),
                factual_mse=distances.diag().tolist(),
                factual_cosine=(p*truth).mean((-1,-2)).tolist(),
                correct_action_beats_both_wrong=wins,
                centered_effect_mse=float(centered_error),
                centered_zero_mse=float(centered_zero))
        details.append(record)
    summaries = {}
    for role in ('train','geometry_transfer'):
        chosen=[d for d in details if d['role']==role]
        summaries[role]={}
        for arm in predictions:
            entries=[d['models'][arm] for d in chosen]
            mse=float(np.mean([m['factual_mse'] for m in entries]))
            baseline=float(np.mean([d['models']['persistence']['factual_mse'] for d in chosen]))
            summaries[role][arm]=dict(contexts=3*len(chosen),mse=mse,
                mse_over_persistence=mse/baseline,
                cosine=float(np.mean([m['factual_cosine'] for m in entries])),
                correct_action_wins=sum(sum(m['correct_action_beats_both_wrong']) for m in entries),
                centered_effect_ratio=float(np.mean([m['centered_effect_mse'] for m in entries])/
                                            np.mean([m['centered_zero_mse'] for m in entries])))
    result=dict(status='COMPLETE', plan=plan, summaries=summaries, groups=details,
                wall_s=time.monotonic()-started,unique_encoded_frames=len(token_cache),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                dense_cache_retained=False, navigation_tested=False)
    save(OUTPUT/'result.json',result)
    save(RESULT,result)
    print('FROZEN_VJEPA_NATIVE_COMPLETE', json.dumps(summaries), flush=True)


if __name__=='__main__':
    try:
        run()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            save(OUTPUT/'failure.json',dict(status='FAILED',reason=repr(error),traceback=traceback.format_exc()))
        raise
