"""Post-hoc prefix errors on the completed readout study's executed windows."""
import argparse
import json
from pathlib import Path

import numpy as np

from lewm.physical_execution_development import rotation_xyzw


BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
HORIZONS = (1, 2, 3, 7)


def evaluate(layout, arm):
    root = BASE / f'go2_dense_world_model_maze_layout{layout:02d}_action_maze_view_{arm}_v1_attempt_001'
    read = lambda name: json.loads((root / name).read_text())
    report = read('dense_navigation_readout.json')
    assert report['physical']['clean_result_available']
    plans = {p['frame']: p for p in read('planning.json') if 'selection' in p}
    frames = {r['frame']: r for r in read('native/in_memory_camera_observations.json')['frames']}
    requests = {r['simulator_ns']: r['requested_command'] for r in read('requests.json')}
    with np.load(root / 'native/physics_trace.npz', allow_pickle=False) as arrays:
        physics = arrays['base_pose_world'].copy()
    rows = []
    for window in report['executed_windows']['rows']:
        frame = window['frame']
        plan = plans[frame]
        selection = plan['selection']
        correction = plan['motion_correction']
        index = ACTIONS.index(plan['action'])
        raw = np.asarray(correction['raw_forecast_xy_m'])
        np.testing.assert_allclose(raw[:, :3], np.broadcast_to(raw[0:1, :3], raw[:, :3].shape), rtol=0, atol=1e-8)
        command = np.asarray(correction['command_history_forecast_xy_yaw'])
        origin = physics[frames[frame]['physical_sample_index']]
        rotation = rotation_xyzw(origin[3:])
        utilities = selection.get('scan_utilities', selection['candidates'])
        higher_moving = max(utilities, key=lambda c: c['utility_m'])['action'] != 'hold'
        only_hold = [c['action'] for c in selection['memory_forecast_candidates'] if c['full_reserve_path_clear']] == ['hold']
        reserve_hold = plan['action'] == 'hold' and higher_moving and only_hold
        zero_prefix = all(np.allclose(requests[plan['measured_ns'] + j * 20_000_000], 0, rtol=0, atol=1e-8) for j in range(15))
        for h in HORIZONS:
            future = physics[frames[frame + h]['physical_sample_index'], :3]
            actual = ((future - origin[:3]) @ rotation)[:2]
            predictions = dict(neural=raw[index, h-1], command_history=command[index, h-1, :2], zero=np.zeros(2))
            rows.append(dict(frame=frame, horizon_ms=h*100, group=window['group'], reserve_hold=reserve_hold,
                zero_requested_prefix=zero_prefix, actual_xy_m=actual.tolist(),
                predicted_xy_m={k:v.tolist() for k,v in predictions.items()},
                squared_xy_error_m2={k:float(np.sum((v-actual)**2)) for k,v in predictions.items()}))

    populations = dict(all=lambda r: True, hold=lambda r:r['group']=='hold',
        translation=lambda r:r['group']=='translation', turn=lambda r:r['group']=='turn',
        reserve_hold=lambda r:r['reserve_hold'],
        zero_prefix_reserve_hold=lambda r:r['reserve_hold'] and r['zero_requested_prefix'])
    summary = []
    for name, predicate in populations.items():
        for h in HORIZONS:
            selected = [r for r in rows if r['horizon_ms']==100*h and predicate(r)]
            if not selected:
                continue
            summary.append(dict(population=name, horizon_ms=h*100, windows=len(selected),
                xy_rmse_mm={k:float(1000*np.sqrt(np.mean([r['squared_xy_error_m2'][k] for r in selected])))
                    for k in ('neural', 'command_history', 'zero')}))
    endpoint = next(r for r in summary if r['population']=='all' and r['horizon_ms']==700)
    for name in ('neural', 'command_history'):
        np.testing.assert_allclose(endpoint['xy_rmse_mm'][name], report['same_window_xy']['rmse_mm'][name], rtol=0, atol=1e-8)
    result = dict(status='COMPLETE', version=2, root=str(root), source=__file__, summary=summary, rows=rows,
        supersedes='prefix_motion_diagnostic_v1.json',
        correction='Use scan_utilities when present, matching the physical decision reader; V1 used base candidate utilities for every decision. Original outputs retained.',
        selection='Existing reader windows with the complete executed requested sequence matched through 700 ms; subgroup definitions are post hoc.',
        native_state_evaluator_only=True, shared_prefix_xy_equal_across_candidates=True,
        original_700ms_metrics_reproduced=True, overlapping_windows_not_independent=True,
        sensor_replay=False, new_training=False, counterfactual_execution=False,
        reserve_safety_certified=False, controller_changed=False)
    with (root / 'prefix_motion_diagnostic_v2.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=int, choices=(0, 2), required=True)
    parser.add_argument('--arm', choices=('old_data', 'maze_data'), required=True)
    args = parser.parse_args()
    evaluate(args.layout, args.arm)
