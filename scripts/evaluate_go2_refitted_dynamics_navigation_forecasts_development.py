"""All matched executed windows from the complete latest four-run memory study."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.eligible_floor_registration_development import bind
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts import fit_go2_refitted_dynamics_readout_development as study
from scripts import run_go2_return_routing_memory_development as memory
from scripts import evaluate_go2_longer_residual_fit_development as scoring
from scripts.read_go2_action_forecast_bias_development import components, PARTS

OUTPUT = study.OUTPUT/'navigation_forecasts'
PLAN = Path('docs/go2_refitted_dynamics_navigation_forecast_plan_2026-09-17.json')
VARIANTS = tuple(prefix+arm for prefix in ('original_', 'refitted_') for arm in study.original.ARMS)+('command_history', 'pose_command')
metrics = bind(scoring.metrics, VARIANTS=VARIANTS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    assert not OUTPUT.exists()
    roots = [memory.BASE/memory.root_name(n) for n in range(1,5)]
    records = []
    for root in roots:
        path = root/'saved_executed_motion_forecast_evaluation_v1.json'
        record = json.loads(path.read_text())
        assert record['matched_requested_sequence_through_ns'] == 700_000_000
        records.append(dict(root=str(root), windows=len(record['rows']), window_sha256=digest(path)))
    study.original.write(PLAN, dict(schema='refitted_dynamics_navigation_forecast_plan.v1',
        roots=records, variants=VARIANTS, source_sha256=digest(__file__),
        readout_plan_sha256=digest(study.PLAN),
        every_matched_executed_window=True, retain_both_failed_returns=True,
        no_new_fitting=True, exposed_development_trajectories=True,
        primary='700ms XY and yaw errors on identical executed windows',
        secondary='prefix and action-increment errors, per-run and action breakdowns',
        original_JEPA_forecasts_must_reproduce=True, native_state_target_only=True,
        nonexecuted_candidate_accuracy_or_navigation_benefit_established=False))
    print('PREPARED', sum(r['windows'] for r in records), 'windows in all four recordings', flush=True)


@torch.inference_mode()
def evaluate():
    plan = json.loads(PLAN.read_text())
    assert digest(__file__) == plan['source_sha256'] and digest(study.PLAN) == plan['readout_plan_sha256']
    torch.set_num_threads(1); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    models = {prefix+arm:loader(arm) for prefix,loader in
        (('original_', study.original.load_readout), ('refitted_', study.load_readout)) for arm in study.original.ARMS}
    OUTPUT.mkdir(exist_ok=False); started = time.monotonic()
    all_rows = []; runs = []; largest_difference = 0.
    for number, record in enumerate(plan['roots'], 1):
        root = Path(record['root'])
        path = root/'saved_executed_motion_forecast_evaluation_v1.json'
        assert digest(path) == record['window_sha256']
        windows = json.loads(path.read_text())['rows']
        plans = {r['frame']:r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r}
        reader = NoisyPublicReplay(root/'native')
        packet = lru_cache(maxsize=8)(reader.policy_packet)
        cameras = reader.noise_rows
        # Physics is used only below for scoring, never to construct model inputs.
        with np.load(root/'native/physics_trace.npz', allow_pickle=False) as archive:
            physics = archive['base_pose_world']
        rows = []
        for ordinal, window in enumerate(windows, 1):
            frame = window['frame']; p = plans[frame]; c = p['motion_correction']
            history = causal_history_tensors([packet(i) for i in range(frame-3, frame+1)], p['measured_ns'])
            inputs = delayed_candidate_inputs(history, p['committed_prefix'], delay_ticks=3, commit_ticks=4)
            if c['terminal_translation_pulse']:
                inputs['known_action_blocks'] = torch.as_tensor(command_sequences(p['committed_prefix'], pulse=True)[:,:,None], dtype=torch.float32)/torch.tensor([.3,1.,.5])
            predictions = {name:model(**inputs)['rollout_outcomes'].numpy() for name,model in models.items()}
            saved = np.asarray(c['upstream_prediction_for_yaw_ablation'])[:,:,:4]
            np.testing.assert_allclose(predictions['original_jepa'][:,:,:4], saved, rtol=0, atol=2e-6)
            largest_difference = max(largest_difference, float(np.max(np.abs(predictions['original_jepa'][:,:,:4]-saved))))
            command = np.asarray(c['command_history_forecast_xy_yaw'])
            predictions['command_history'] = np.concatenate((command[:,:,:2], np.sin(command[:,:,2:3]), np.cos(command[:,:,2:3])), -1)
            # Existing pose-command prediction uses command yaw; use that saved common yaw baseline.
            nominal = scoring.bias.nominal(p['committed_prefix'], pulse=bool(c['terminal_translation_pulse']))
            predictions['pose_command'] = np.concatenate((np.asarray(c['pose_command_forecast_xy_m']), nominal[:,:,2:4]), -1)
            origin = physics[cameras[frame]['physical_sample_index']]
            actual_poses = physics[[cameras[frame+h]['physical_sample_index'] for h in range(1,8)]]
            actual_xy = ((actual_poses[:,:3]-origin[:3])@rotation_xyzw(origin[3:]))[:,:2]
            np.testing.assert_allclose(actual_xy[-1], window['actual_endpoint_xy_m'], rtol=0, atol=1e-12)
            actual = components(actual_xy)
            rotations = [rotation_xyzw(v[3:]) for v in (origin, actual_poses[-1])]
            actual_yaw = math.atan2(rotations[1][1,0], rotations[1][0,0])-math.atan2(rotations[0][1,0], rotations[0][0,0])
            action = ACTIONS.index(window['action']); errors = {}; yaw_errors = {}
            for name,prediction in predictions.items():
                parts = components(prediction[action,:,:2])
                errors[name] = {part:(1000*(np.asarray(parts[part])-actual[part])).tolist() for part in PARTS}
                delta = math.atan2(prediction[action,6,2], prediction[action,6,3])-actual_yaw
                yaw_errors[name] = math.atan2(math.sin(delta), math.cos(delta))
            rows.append(dict(run=number, frame=frame, action=window['action'], errors_mm=errors, yaw_error_rad=yaw_errors))
            if ordinal % 300 == 0: print('REFITTED_NAVIGATION_FORECAST', number, ordinal, flush=True)
        result = dict(run=number, root=str(root), total=metrics(rows),
            by_action={a:metrics([r for r in rows if r['action']==a]) for a in ACTIONS})
        old_scores = json.loads((root/'saved_short_pulse_same_window_xy_v1.json').read_text())['rmse_mm']
        for name, old_name in (('original_jepa','neural'), ('command_history','command_history'), ('pose_command','pose_command')):
            np.testing.assert_allclose(result['total']['models'][name]['parts']['whole_700ms']['rmse_mm'], old_scores[old_name], rtol=0, atol=1e-8)
        study.original.write(OUTPUT/f'run_{number:02d}.json', dict(summary=result, rows=rows))
        runs.append(result); all_rows.extend(rows)
        print('REFITTED_NAVIGATION_FORECAST_RUN_COMPLETE', number, len(rows), flush=True)
    result = dict(status='complete', plan_sha256=digest(PLAN), runs=runs, total=metrics(all_rows),
        by_action={a:metrics([r for r in all_rows if r['action']==a]) for a in ACTIONS},
        original_predictions_and_saved_RMSE_reproduced=True,
        maximum_original_prediction_difference=largest_difference,
        failures_included=True, depth_not_loaded=True, weights_changed_during_evaluation=False,
        overlapping_windows_not_independent=True, new_navigation_executed=False,
        alternative_action_outcomes_established=False, wall_s=time.monotonic()-started)
    study.original.write(OUTPUT/'result.json', result)
    study.original.write(Path('docs/go2_refitted_dynamics_navigation_forecast_result_2026-09-17.json'), result)
    print(json.dumps(result['total'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else evaluate()
