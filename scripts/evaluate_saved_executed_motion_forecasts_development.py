"""Measure saved XY forecasts only where the requested 700-ms sequence occurred."""
import argparse
from collections import defaultdict
import json

import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.physical_execution_development import rotation_xyzw
from lewm.terminal_translation_pulse_development import command_sequences
from scripts.navigation_artifact_root_development import BASE, validate_root


def metrics(rows):
    if not rows:
        return dict(windows=0)
    raw = np.asarray([r['raw_endpoint_error_m'] for r in rows])
    corrected = np.asarray([r['corrected_endpoint_error_m'] for r in rows])
    return dict(windows=len(rows), raw_endpoint_xy_rmse_m=float(np.sqrt(np.mean(raw**2))),
        corrected_endpoint_xy_rmse_m=float(np.sqrt(np.mean(corrected**2))),
        corrected_endpoint_error_median_m=float(np.median(corrected)),
        corrected_endpoint_error_p95_m=float(np.percentile(corrected, 95)),
        corrected_endpoint_error_maximum_m=float(corrected.max()),
        corrected_path_error_maximum_m=max(r['corrected_path_error_maximum_m'] for r in rows),
        windows_with_any_corrected_xy_error_above_30mm=sum(r['corrected_path_error_maximum_m']>.03 for r in rows))


def evaluate(root):
    validate_root(root)
    read = lambda name: json.loads((root/name).read_text())
    frames = {r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']:r['requested_command'] for r in read('requests.json')}
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as arrays:
        physics = arrays['base_pose_world'].copy()
    plans = [p for p in read('planning.json') if 'motion_correction' in p and 'selection' in p]
    rows = []; groups = defaultdict(list)
    for p in plans:
        frame = p['frame']; now = p['measured_ns']; correction = p['motion_correction']
        if frame+7 not in frames:
            continue
        index = ACTIONS.index(p['action'])
        pulse = bool(correction.get('terminal_translation_pulse', False))
        commands = command_sequences(p['committed_prefix'], pulse=pulse)[index, :7]
        times = [now+j*100_000_000+k*20_000_000 for j in range(7) for k in range(5)]
        if any(t not in requests for t in times):
            continue
        if not np.allclose(np.asarray([requests[t] for t in times]),
                           np.repeat(commands, 5, axis=0), rtol=0, atol=1e-8):
            continue
        origin = physics[frames[frame]['physical_sample_index']]
        R = rotation_xyzw(origin[3:])
        future = physics[[frames[frame+h]['physical_sample_index'] for h in range(1, 8)], :3]
        actual = ((future-origin[:3])@R)[:, :2]
        raw = np.asarray(correction['raw_forecast_xy_m'])[index, :7]
        corrected = np.asarray(correction['corrected_forecast_xy_m'])[index, :7]
        raw_error = np.linalg.norm(raw-actual, axis=1)
        corrected_error = np.linalg.norm(corrected-actual, axis=1)
        group = ('hold' if p['action']=='hold' else 'turn' if p['action'] in
                 ('left_turn', 'right_turn') else 'translation_pulse' if pulse else 'translation')
        row = dict(frame=frame, action=p['action'], group=group, plan_on_time=p['on_time'],
            actual_endpoint_xy_m=actual[-1].tolist(), raw_endpoint_error_m=float(raw_error[-1]),
            corrected_endpoint_error_m=float(corrected_error[-1]),
            corrected_path_error_maximum_m=float(corrected_error.max()))
        rows.append(row); groups[group].append(row)
    return dict(root_name=root.name, selected_plans=len(plans), **metrics(rows),
        by_action_group={k:metrics(v) for k,v in groups.items()},
        matched_requested_sequence_through_ns=700_000_000,
        native_state_evaluator_only=True, corrected_forecast_recomputed=False,
        unexecuted_candidates_evaluated=False, horizon_800ms_evaluated=False,
        overlapping_windows_not_independent=True, reserve_coverage_certified=False, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    args = parser.parse_args(); root = BASE/args.root_name
    result = evaluate(root)
    with (root/'saved_executed_motion_forecast_evaluation_v1.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))
