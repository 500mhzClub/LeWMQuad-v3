"""Match existing no-RGB neural controls to the full-input correction procedure."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.eligible_floor_registration_development import bind
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import fit_closed_loop_motion_residual_development as original
from scripts.navigation_artifact_root_development import validate_root
from scripts.public_policy_replay_development import PublicPolicyReplay

SEEDS = (2026091001, 2026091401, 2026091402)
METHODS = ('jepa', 'direct', 'supervised_rollout')
ROOT = 'go2_no_rgb_matched_motion_residual_seed_{seed}_{condition}_v1_attempt_001'


def fit_one(seed, condition):
    assignment = f'seed_{seed}_no_rgb_{condition}'
    head = 'direct_outcomes' if condition == 'direct' else 'rollout_outcomes'
    output = validate_root(original.BASE / ROOT.format(seed=seed, condition=condition), must_exist=False)
    output.mkdir()
    began = time.time()
    admission_path = original.BASE / 'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001/launch.json'
    admission = json.loads(admission_path.read_text())['input_admission']['correction_admission']
    model, loaded_condition, variant = original.load_assigned(admission, assignment)
    if (loaded_condition, variant) != (condition, 'no_rgb'):
        raise ValueError('assigned no-RGB trained model required')
    identity = state_digest(model.state_dict())
    sources = (__file__, original.__file__, 'scripts/public_policy_replay_development.py',
        'lewm/observation_horizon_input_ablation_development.py')
    launch = dict(base_model=assignment, model_training_seed=seed, input_variant=variant,
        prediction_head=head, model_state_sha256=identity, training_roots=original.TRAIN,
        validation_root=original.VALIDATION, labels='subsequent_registered_visual_pose_deltas',
        native_state_read=False, ridge_penalty=1., stationary_training_stride_frames=40,
        original_jepa_study_root=original.OUTPUT.name, input_admission=admission,
        validation_is_development_not_sealed=True, fresh_maze_inputs_used=False,
        model_rgb_zero_checked_every_forward=True, camera_based_pose_history_retained=True,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    with (output/'launch.json').open('x') as f:
        json.dump(launch, f, indent=2)

    def selected_head(**inputs):
        if torch.count_nonzero(inputs['observation_history']['rgb']).item():
            raise ValueError('no-RGB model received nonzero image inputs')
        result = model(**inputs)
        return dict(prediction_valid=result['prediction_valid'], rollout_outcomes=result[head])

    collect = bind(original.collect.__wrapped__, PublicReplay=PublicPolicyReplay, OUTPUT=output)

    def matched_collect(name, training):
        with torch.inference_mode():
            data, rows = collect(name, selected_head, variant, training)
        with np.load(original.OUTPUT/(name+'.npz'), allow_pickle=False) as reference:
            for field in ('target', 'valid'):
                if not np.array_equal(data[field], reference[field]):
                    raise ValueError('original residual targets/masks changed: '+name)
        if rows != json.loads((original.OUTPUT/(name+'.json')).read_text()):
            raise ValueError('original residual windows/groups changed: '+name)
        return data, rows

    datasets = [matched_collect(root, True)[0] for root in original.TRAIN]
    train = {k:np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    fit = {k:[] for k in ('mean', 'scale', 'bias', 'coefficient')}
    for h in range(8):
        valid = train['valid'][:,h]
        x = train['features'][valid,h]; y = (train['target']-train['prediction'])[valid,h]
        mean = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1.; bias = y.mean(0)
        z = (x-mean)/scale
        coefficient = np.linalg.solve(z.T@z+np.eye(z.shape[1]), z.T@(y-bias))
        for key, value in zip(fit, (mean, scale, bias, coefficient)):
            fit[key].append(value)
    fit = {k:np.asarray(v) for k,v in fit.items()}
    np.savez_compressed(output/'residual_fit.npz', **fit)
    # Validation is first collected after coefficients have been frozen.
    validation, rows = matched_collect(original.VALIDATION, False)
    corrected = validation['prediction'].copy()
    for h in range(8):
        corrected[:,h] += ((validation['features'][:,h]-fit['mean'][h])/fit['scale'][h])@fit['coefficient'][h]+fit['bias'][h]
    results = {}
    for group in ('all', 'moving', 'transition', 'translation', 'turn_only'):
        group_mask = np.array([True if group == 'all' else row[group] for row in rows])
        results[group] = {}
        for h in (2,6,7):
            mask = group_mask & validation['valid'][:,h]
            results[group][str((h+1)*100)] = dict(
                base=original.metrics((validation['prediction']-validation['target'])[mask,h]),
                corrected=original.metrics((corrected-validation['target'])[mask,h]))
    if state_digest(model.state_dict()) != identity:
        raise ValueError('frozen neural model changed')
    report = dict(status='COMPLETE', base_model=assignment, model_training_seed=seed,
        input_variant=variant, prediction_head=head, model_state_sha256=identity,
        results=results, training_windows=len(train['features']), validation_windows=len(rows),
        original_jepa_windows_targets_and_groups_exact=True,
        fit_sha256=hashlib.sha256((output/'residual_fit.npz').read_bytes()).hexdigest(),
        elapsed_s=time.time()-began, native_state_read=False, neural_weights_changed=False,
        model_rgb_zero_checked_every_forward=True, camera_based_pose_history_retained=True,
        prospective_navigation_tested=False)
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2)
    print('NO_RGB_MATCHED_MOTION_RESULT', json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for seed in SEEDS if args.seed is None else (args.seed,):
        fit_one(seed, args.condition)


if __name__ == '__main__':
    main()
