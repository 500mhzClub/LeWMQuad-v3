"""Factorial current/future modality substitutions in frozen mixed JEPA targets.

These hybrid target observations are sensitivity diagnostics, not physical
counterfactuals or modality-retrained models. No predictor sees future inputs.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.pulse_timed_dataset_development import stack_samples
from lewm.rgb_body_tensor_interface_development import observation_tensors
from lewm.route_rgb_dataset_development import load_route_observation
from scripts import probe_go2_jepa_latent_branch_science_development as original
from scripts import fit_go2_refitted_dynamics_readout_development as refitted

OUTPUT = original.fits.BASE/'go2_jepa_target_modalities_v1_attempt_001'
PLAN = Path('docs/go2_jepa_target_modalities_plan_2026-09-17.json')
RESULT = Path('docs/go2_jepa_target_modalities_result_2026-09-17.json')
MODALITIES = ('rgb', 'body', 'control')
MASKS = tuple(''.join(map(str, bits)) for bits in itertools.product((0, 1), repeat=3))


def prepare():
    assert not OUTPUT.exists()
    original.save(PLAN, dict(schema='jepa_target_modalities.v1',
        source_sha256=original.digest(__file__), sample_ids=[r['sample_id'] for r in original.selected_rows()],
        original_result_sha256=original.digest(original.RESULT),
        refitted_result_sha256=original.digest(Path('docs/go2_refitted_dynamics_readout_result_2026-09-17.json')),
        modalities=MODALITIES, masks=MASKS, mask_one='actual future', mask_zero='current observation repeated',
        primary_role='geometry_transfer', primary_horizon_ms=800, all_horizons_ms=list(range(100,801,100)),
        arms=original.ARMS, no_fitting=True, hybrid_targets_not_physical_counterfactuals=True,
        metrics=['temporal target change MSE', 'within-identical-history action target variance',
                 'target and centered action effect change versus all-future target',
                 'frozen original/refitted forecasts versus each diagnostic target'],
        limits=['two exposed transfer geometries', 'nonlinear modality effects are not additive',
                'sensitivity is not causal attribution of learning',
                'future command history is part of the existing mixed target',
                'no navigation or JEPA advantage established'],
        resources=dict(cpu_core=8, numerical_threads=1, ram_available_gib=72,
                       output_free_gib=4.4, both_gpus_idle=True,
                       reason='small shared-input frozen-model assay; sequential arms avoid repeated input loading')))
    print('PREPARED 36 contexts, three representations, eight target substitutions', flush=True)


def centered(values, rows):
    output = np.empty_like(values)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['cluster'], row['prefix_action']].append(i)
    for indices in groups.values():
        assert len(indices) == 3
        output[indices] = values[indices]-values[indices].mean(0)
    return output


def mse(values):
    return float(np.mean(np.square(values.astype(np.float64))))


def ratio(a, b):
    return a/b if b > 0 else None


def retrieval(prediction, target, rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['cluster'], row['prefix_action']].append(i)
    wins = ties = 0
    for indices in groups.values():
        matrix = ((prediction[indices, None]-target[None, indices])**2).mean(-1)
        for j in range(3):
            other = min(matrix[j, k] for k in range(3) if k != j)
            wins += int(matrix[j,j] < other)
            ties += int(matrix[j,j] == other)
    return dict(wins=wins, ties=ties, contexts=len(rows))


@torch.inference_mode()
def run():
    plan = json.loads(PLAN.read_text())
    assert original.digest(__file__) == plan['source_sha256']
    assert original.digest(original.RESULT) == plan['original_result_sha256']
    assert original.digest(Path('docs/go2_refitted_dynamics_readout_result_2026-09-17.json')) == plan['refitted_result_sha256']
    torch.set_num_threads(1); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    rows = original.selected_rows()
    assert [r['sample_id'] for r in rows] == plan['sample_ids']
    OUTPUT.mkdir(exist_ok=False); started = time.monotonic()
    # Predictions are immutable outputs of the earlier causal evaluation.
    predictions = {}
    for name, root in (('original', original.OUTPUT), ('refitted', refitted.OUTPUT/'branch_evaluation')):
        with np.load(root/'causal_predictions.npz', allow_pickle=False) as archive:
            predictions[name] = {arm:archive[arm+'_predictions'].copy() for arm in original.ARMS}
    with np.load(original.OUTPUT/'encoded_targets.npz', allow_pickle=False) as archive:
        saved_targets = {arm:archive[arm+'_targets'].copy() for arm in original.ARMS}
    current = []; future = []
    for row in rows:
        root = original.data.PULSE/row['trial']
        current.append(observation_tensors(load_route_observation(root, 13)))
        future.append(stack_samples([observation_tensors(load_route_observation(root, 14+h)) for h in range(8)]))
    current = stack_samples(current); future = stack_samples(future)
    frozen = {k:v[:,None].expand_as(future[k]) for k,v in current.items()}
    raw = {}
    for role in ('train', 'geometry_transfer'):
        idx = [i for i,r in enumerate(rows) if r['data_role']==role]
        raw[role] = [dict(horizon_ms=100*(h+1),
            rgb_temporal_pixel_mse=mse((future['rgb']-frozen['rgb'])[idx,h].numpy()),
            rgb_action_pixel_variance=mse(centered(future['rgb'][idx,h].numpy(), [rows[i] for i in idx])))
            for h in range(8)]
    scores = {}; arrays = {}
    for arm in original.ARMS:
        model = original.fits.load_readout(arm)
        targets = {}
        for mask in MASKS:
            values = {k:(future if bit=='1' else frozen)[k].flatten(0,1) for k,bit in zip(MODALITIES,mask)}
            targets[mask] = model.target(values).reshape(len(rows),8,-1).numpy()
            arrays[arm+'_'+mask] = targets[mask]
        np.testing.assert_array_equal(targets['111'], saved_targets[arm])
        scores[arm] = {}
        for role in ('train','geometry_transfer'):
            idx = [i for i,r in enumerate(rows) if r['data_role']==role]
            selected = [rows[i] for i in idx]; curve = []
            for h in range(8):
                full = targets['111'][idx,h]; present = targets['000'][idx,h]
                effects = centered(full, selected)
                temporal_energy = mse(full-present); action_energy = mse(effects)
                variants = {}
                for mask in MASKS:
                    value = targets[mask][idx,h]
                    value_effects = centered(value,selected)
                    variants[mask] = dict(temporal_change_mse=mse(value-present),
                        action_variance=mse(value_effects),
                        difference_from_full_mse=mse(value-full),
                        difference_over_full_temporal_change=ratio(mse(value-full),temporal_energy),
                        action_effect_difference_over_full=ratio(mse(value_effects-effects),action_energy),
                        forecasts={name:dict(mse=mse(p[arm][idx,h]-value),
                            action_retrieval=retrieval(p[arm][idx,h],value,selected))
                            for name,p in predictions.items()})
                curve.append(dict(horizon_ms=100*(h+1), full_temporal_change_mse=temporal_energy,
                                  full_action_variance=action_energy, variants=variants))
            scores[arm][role] = dict(curve=curve, primary_800ms=curve[-1])
        print('TARGET_MODALITIES_COMPLETE', arm, flush=True)
    np.savez_compressed(OUTPUT/'targets.npz', **arrays)
    result = dict(status='complete', plan_sha256=original.digest(PLAN), scores=scores,
        raw_rgb=raw, all_future_targets_exactly_reproduced=True,
        future_data_never_passed_to_predictors=True, weights_changed=False,
        navigation_executed=False, wall_s=time.monotonic()-started, limits=plan['limits'])
    original.save(OUTPUT/'result.json', result); original.save(RESULT, result)
    for arm in original.ARMS:
        p = scores[arm]['geometry_transfer']['primary_800ms']
        print(arm, json.dumps({m:{k:v for k,v in p['variants'][m].items() if k!='forecasts'}
                               for m in ('011','100','001','010')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
