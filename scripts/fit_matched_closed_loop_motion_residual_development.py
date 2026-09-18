"""Fit fixed training conditions/seeds on the original residual population."""
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
from scripts.public_policy_replay_development import PublicPolicyReplay
from scripts.navigation_artifact_root_development import validate_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=('jepa', 'direct', 'supervised_rollout'), required=True)
    parser.add_argument('--seed', type=int, choices=(2026091001,2026091401,2026091402), default=2026091001)
    args = parser.parse_args()
    assignment = f'seed_{args.seed}_full_{args.condition}'
    head = 'direct_outcomes' if args.condition == 'direct' else 'rollout_outcomes'
    seed_prefix = '' if args.seed == 2026091001 else f'seed_{args.seed}_'
    output = original.BASE/f'go2_matched_motion_residual_{seed_prefix}{args.condition}_v1_attempt_001'
    validate_root(output, must_exist=False)
    output.mkdir()
    began = time.time()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    admission_path = original.BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001/launch.json'
    admission = json.loads(admission_path.read_text())['input_admission']['correction_admission']
    model, condition, variant = original.load_assigned(admission, assignment)
    if (condition, variant) != (args.condition, 'full'):
        raise ValueError('fixed matched training condition required')
    model_identity = state_digest(model.state_dict())
    launch = dict(training_roots=original.TRAIN, validation_root=original.VALIDATION,
        labels='subsequent_registered_visual_pose_deltas', native_state_read=False,
        base_model=assignment, model_training_seed=args.seed,
        prediction_head=head, model_state_sha256=model_identity,
        ridge_penalty=1., stationary_training_stride_frames=40,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (__file__, 'scripts/public_policy_replay_development.py', original.__file__)},
        input_admission=admission, original_jepa_study_root=original.OUTPUT.name,
        validation_is_development_not_sealed=True, fresh_maze_inputs_used=False)
    with (output/'launch.json').open('x') as f: json.dump(launch, f, indent=2)

    # The original collector names its selected head rollout_outcomes internally.
    # Select the actual trained head explicitly; saved arrays use prediction.
    def selected_head(**inputs):
        result = model(**inputs)
        return dict(prediction_valid=result['prediction_valid'], rollout_outcomes=result[head])

    collect = bind(original.collect.__wrapped__, PublicReplay=PublicPolicyReplay, OUTPUT=output)

    def matched_collect(name, training):
        with torch.inference_mode():
            data, rows = collect(name, selected_head, variant, training)
        with np.load(original.OUTPUT/(name+'.npz'), allow_pickle=False) as reference:
            for field in ('target', 'valid'):
                if not np.array_equal(data[field], reference[field]):
                    raise ValueError('original residual windows/targets changed: '+name)
        reference_rows = json.loads((original.OUTPUT/(name+'.json')).read_text())
        if rows != reference_rows:
            raise ValueError('original residual population changed: '+name)
        return data, rows

    datasets = [matched_collect(root, True)[0] for root in original.TRAIN]
    train = {k:np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    fit = {k:[] for k in ('mean', 'scale', 'bias', 'coefficient')}
    for h in range(8):
        valid = train['valid'][:, h]
        x = train['features'][valid, h]; y = (train['target']-train['prediction'])[valid, h]
        mean = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1.; bias = y.mean(0)
        z = (x-mean)/scale
        coefficient = np.linalg.solve(z.T@z+np.eye(z.shape[1]), z.T@(y-bias))
        for key, value in zip(fit, (mean, scale, bias, coefficient)):
            fit[key].append(value)
    fit = {k:np.asarray(v) for k, v in fit.items()}
    np.savez_compressed(output/'residual_fit.npz', **fit)
    # Freeze coefficients before touching validation windows, as in the JEPA fit.
    validation, rows = matched_collect(original.VALIDATION, False)
    corrected = validation['prediction'].copy()
    for h in range(8):
        corrected[:, h] += ((validation['features'][:, h]-fit['mean'][h])/fit['scale'][h])@fit['coefficient'][h]+fit['bias'][h]
    results = {}
    for group in ('all', 'moving', 'transition', 'translation', 'turn_only'):
        group_mask = np.array([True if group == 'all' else row[group] for row in rows])
        results[group] = {}
        for h in (2, 6, 7):
            mask = group_mask & validation['valid'][:, h]
            results[group][str((h+1)*100)] = dict(
                base=original.metrics((validation['prediction']-validation['target'])[mask, h]),
                corrected=original.metrics((corrected-validation['target'])[mask, h]))
    if state_digest(model.state_dict()) != model_identity:
        raise ValueError('frozen neural model changed during correction fit')
    report = dict(status='COMPLETE', base_model=assignment, model_training_seed=args.seed, prediction_head=head,
        model_state_sha256=model_identity, results=results,
        training_windows=len(train['features']), validation_windows=len(rows),
        original_jepa_windows_targets_and_groups_exact=True,
        fit_sha256=hashlib.sha256((output/'residual_fit.npz').read_bytes()).hexdigest(),
        elapsed_s=time.time()-began, native_state_read=False, neural_weights_changed=False,
        inference_uses_causal_pose_history_and_known_commands=True,
        prospective_navigation_tested=False)
    with (output/'result.json').open('x') as f: json.dump(report, f, indent=2)
    print('MATCHED_MOTION_RESIDUAL_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
