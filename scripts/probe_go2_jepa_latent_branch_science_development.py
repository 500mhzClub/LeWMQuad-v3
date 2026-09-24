"""Fixed latent persistence, action-branch and matched-scene tests.

Existing train/development-transfer roles are preserved. No fitting, native
execution, depth reads, threshold selection or model promotion occurs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.pulse_timed_dataset_development import stack_samples
from lewm.rgb_body_tensor_interface_development import observation_tensors
from scripts import fit_go2_frozen_motion_readout_development as fits
from scripts import prepare_go2_short_pulse_training_development as data
from scripts.pre_switch_training_data_development import PacketReader, inputs
from lewm.route_rgb_dataset_development import load_route_observation

OUTPUT = fits.BASE / 'go2_jepa_latent_branch_science_v1_attempt_001'
PLAN = Path('docs/go2_jepa_latent_branch_science_plan_2026-09-17.json')
RESULT = Path('docs/go2_jepa_latent_branch_science_result_2026-09-17.json')
ARMS = ('jepa', 'supervised_rollout', 'untrained')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_rows():
    rows = json.loads((data.PULSE/'windows.json').read_text())
    selected = [r for r in rows if r['observation_frame'] == 13]
    assert len(selected) == 36 and all(r['available'] for r in selected)
    assert all(r['data_role'] in ('train', 'geometry_transfer') for r in selected)
    return sorted(selected, key=lambda r: (r['data_role'], r['cluster'], r['prefix_action'], r['pulse_action']))


def prepare():
    rows = selected_rows()
    assert not OUTPUT.exists()
    plan = dict(schema='jepa_latent_branch_science_plan.v1',
        sample_ids=[r['sample_id'] for r in rows], arms=ARMS,
        windows_sha256=digest(data.PULSE/'windows.json'),
        source_sha256=digest(__file__), fit_plan_sha256=digest(fits.PLAN),
        fit_result_sha256=digest(fits.OUTPUT/'result.json'),
        primary_role='geometry_transfer', training_role_is_diagnostic=True,
        contexts_per_role=18, branch_groups_per_role=6, branches_per_group=3,
        primary_horizon_ms=800, all_horizons_ms=list(range(100, 801, 100)),
        metrics=['within-model prediction MSE versus current-target persistence',
            'correct-action forecast MSE versus both wrong-action forecasts',
            'three-way future action-branch retrieval on identical causal inputs',
            'two-way scene retrieval against a target differing only in RGB',
            'causal-history RGB swap with body, control and candidate actions preserved',
            'target effective rank and variance', 'common physical motion readout errors'],
        no_fitting=True, future_packets_are_target_only=True,
        latent_losses_across_models_are_not_common_representation_errors=True,
        limits=['two previously exposed transfer geometries', 'one training seed',
            'short pulses and 0.8-second horizon', 'scene retrieval may use static appearance',
            'no new navigation evidence or JEPA superiority assumed'])
    save(PLAN, plan)
    print('PREPARED', len(rows), 'fixed action-branch contexts', flush=True)


def summary(rows, predictions, targets, persistence, scene_targets, swapped, motion, reference):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['cluster'], row['prefix_action']].append(i)
    curve = []
    for h in range(8):
        error = ((predictions[:, h]-targets[:, h])**2).mean(-1)
        persistent_error = ((persistence-targets[:, h])**2).mean(-1)
        scene_error = ((predictions[:, h]-scene_targets[:, h])**2).mean(-1)
        persistence_scene_error = ((persistence-scene_targets[:, h])**2).mean(-1)
        swapped_error = ((swapped[:, h]-targets[:, h])**2).mean(-1)
        retrieval = []; wrong_errors = []; matched_separation = []
        for indices in groups.values():
            assert len(indices) == 3
            p, t = predictions[indices, h], targets[indices, h]
            matrix = ((p[:, None]-t[None])**2).mean(-1)
            for j in range(3):
                others = [k for k in range(3) if k != j]
                # Correct prediction must beat both alternatives strictly; ties count separately.
                retrieval.append(dict(win=bool(matrix[j, j] < matrix[j, others].min()),
                    tie=bool(matrix[j, j] == matrix[j, others].min())))
                wrong_errors.extend(matrix[others, j].tolist())
                matched_separation.extend(((t[j]-t[others])**2).mean(-1).tolist())
        variance = float(((targets[:, h]-targets[:, h].mean(0))**2).mean())
        truth = np.asarray([r['targets'][h]['motion'] for r in rows])
        delta = motion[:, h, :2]-truth[:, :2]
        yaw = np.arctan2(motion[:, h, 2], motion[:, h, 3])-truth[:, 2]
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        ref_delta = reference[:, h, :2]-truth[:, :2]
        ref_yaw = reference[:, h, 2]-truth[:, 2]
        ref_yaw = np.arctan2(np.sin(ref_yaw), np.cos(ref_yaw))
        curve.append(dict(horizon_ms=(h+1)*100, contexts=len(rows),
            prediction_mse=float(error.mean()), persistence_mse=float(persistent_error.mean()),
            prediction_to_persistence_ratio=float(error.mean()/persistent_error.mean()),
            target_variance=variance, prediction_mse_over_target_variance=float(error.mean()/variance),
            wrong_action_prediction_mse=float(np.mean(wrong_errors)),
            correct_action_beats_wrong_average=bool(error.mean() < np.mean(wrong_errors)),
            action_branch_retrieval_wins=sum(r['win'] for r in retrieval),
            action_branch_retrieval_ties=sum(r['tie'] for r in retrieval),
            mean_action_target_separation_mse=float(np.mean(matched_separation)),
            scene_retrieval_wins=int((error < scene_error).sum()),
            scene_retrieval_ties=int((error == scene_error).sum()),
            persistence_scene_retrieval_wins=int((persistent_error < persistence_scene_error).sum()),
            scene_target_separation_mse=float(((targets[:, h]-scene_targets[:, h])**2).mean()),
            rgb_swap_prediction_change_rms=float(np.sqrt(((swapped[:, h]-predictions[:, h])**2).mean())),
            rgb_swap_prediction_mse=float(swapped_error.mean()),
            motion_xy_rmse_mm=float(1000*np.sqrt((delta**2).sum(-1).mean())),
            motion_yaw_rmse_deg=float(np.degrees(np.sqrt((yaw**2).mean()))),
            reference_xy_rmse_mm=float(1000*np.sqrt((ref_delta**2).sum(-1).mean())),
            reference_yaw_rmse_deg=float(np.degrees(np.sqrt((ref_yaw**2).mean())))))
    flat = targets.reshape(-1, targets.shape[-1])
    singular = np.linalg.svd(flat-flat.mean(0), compute_uv=False)
    probabilities = singular**2 / (singular**2).sum()
    positive = probabilities[probabilities > 0]
    return dict(curve=curve, target_effective_rank=float(np.exp(-(positive*np.log(positive)).sum())),
        target_mean_coordinate_std=float(flat.std(0).mean()),
        primary_800ms=curve[-1])


@torch.inference_mode()
def run():
    plan = json.loads(PLAN.read_text())
    assert digest(__file__) == plan['source_sha256']
    assert digest(data.PULSE/'windows.json') == plan['windows_sha256']
    assert digest(fits.PLAN) == plan['fit_plan_sha256']
    assert digest(fits.OUTPUT/'result.json') == plan['fit_result_sha256']
    torch.set_num_threads(1); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    rows = selected_rows(); assert [r['sample_id'] for r in rows] == plan['sample_ids']
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic(); samples = []; packets = {}; input_hashes = {}
    for row in rows:
        directory = data.PULSE/row['trial']
        past = row['history_observation_indices']
        current = {i:load_route_observation(directory, i) for i in past}
        packets[row['trial']] = current
        samples.append(inputs(row, PacketReader(current, past)))
        for filename in ['policy_observations.json', 'policy_histories.npz']+[f'rgb_{i:04d}.png' for i in past]:
            input_hashes[str(directory/filename)] = digest(directory/filename)
    batch = stack_samples(samples)
    group_indices = defaultdict(list); scenes = defaultdict(list)
    for i, row in enumerate(rows):
        group_indices[row['cluster'], row['prefix_action']].append(i)
        scenes[row['data_role'], row['prefix_action'], row['pulse_action']].append(i)
    for indices in group_indices.values():
        assert len(indices) == 3
        for key, tensor in batch['observation_history'].items():
            assert all(torch.equal(tensor[indices[0]], tensor[i]) for i in indices[1:]), key
    swapped_indices = list(range(len(rows)))
    for indices in scenes.values():
        assert len(indices) == 2
        a, b = indices; swapped_indices[a], swapped_indices[b] = b, a
    swapped_history = batch['observation_history'] | dict(rgb=batch['observation_history']['rgb'][swapped_indices])
    models = {arm:fits.load_readout(arm) for arm in ARMS}
    forecasts = {}; arrays = {}
    for arm, model in models.items():
        original = model(**batch)
        swapped = model(**(batch | dict(observation_history=swapped_history)))
        current_target = model.target({k:v[:, -1] for k, v in batch['observation_history'].items()})
        reference = model.reference_motion(batch['observation_history'], batch['known_action_blocks'], batch['known_action_valid'])
        forecasts[arm] = dict(predictions=original['future_latents'].numpy(),
            swapped=swapped['future_latents'].numpy(), persistence=current_target.numpy(),
            motion=original['rollout_outcomes'].numpy(), reference=reference.numpy())
        arrays.update({arm+'_'+k:v for k,v in forecasts[arm].items()})
    np.savez_compressed(OUTPUT/'causal_predictions.npz', **arrays)
    # Future observations are loaded only after all original/intervened predictions are fixed.
    future = []
    for row in rows:
        values = []
        for h, target in enumerate(row['targets']):
            assert target['future_image_valid'] and target['motion_valid']
            index = target['future_observation_index']
            assert index == 14+h
            packet = load_route_observation(data.PULSE/row['trial'], index)
            assert packet['image']['measured_ns'] == row['decision_ns']+(h+1)*100_000_000
            values.append(observation_tensors(packet))
            path = data.PULSE/row['trial']/f'rgb_{index:04d}.png'; input_hashes[str(path)] = digest(path)
        future.append(stack_samples(values))
    future = stack_samples(future)
    false_scene = future | dict(rgb=future['rgb'][swapped_indices])
    scores = {}
    target_arrays = {}
    for arm, model in models.items():
        def encode(values):
            return model.target({k:v.flatten(0, 1) for k,v in values.items()}).reshape(len(rows), 8, -1).numpy()
        targets, scene_targets = encode(future), encode(false_scene)
        target_arrays.update({arm+'_targets':targets, arm+'_scene_targets':scene_targets})
        scores[arm] = {}
        for role in ('train', 'geometry_transfer'):
            idx = [i for i,r in enumerate(rows) if r['data_role'] == role]
            scores[arm][role] = summary([rows[i] for i in idx], targets=targets[idx],
                scene_targets=scene_targets[idx], **{k:v[idx] for k,v in forecasts[arm].items()})
    np.savez_compressed(OUTPUT/'encoded_targets.npz', **target_arrays)
    result = dict(status='complete', plan_sha256=digest(PLAN), contexts=36,
        matched_causal_action_groups=12, all_branch_group_history_tensors_exactly_equal=True,
        future_observations_loaded_after_predictions_saved=True, scores=scores,
        input_sha256=input_hashes, wall_s=time.monotonic()-started,
        model_fit_records=json.loads((fits.OUTPUT/'result.json').read_text())['records'],
        neural_weights_changed=False, navigation_executed=False,
        prediction_sha256=digest(OUTPUT/'causal_predictions.npz'),
        limits=plan['limits'])
    save(OUTPUT/'result.json', result)
    save(RESULT, result)
    for arm in ARMS:
        print(arm, json.dumps(scores[arm]['geometry_transfer']['primary_800ms']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
