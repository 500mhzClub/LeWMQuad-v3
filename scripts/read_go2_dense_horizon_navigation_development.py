"""Physical readout of a completed dense-model navigation recording."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from lewm.eligible_floor_registration_development import bind
from lewm.navigation_decision_diagnostic_development import summarize_decisions
from scripts.evaluate_continuous_native_arrivals_development import evaluate as arrivals
from scripts.evaluate_saved_executed_motion_forecasts_development import evaluate as xy
from scripts.evaluate_go2_short_pulse_navigation_development import same_window_xy, yaw_metrics
from scripts.navigation_artifact_root_development import BASE as RECOVERY_BASE, validate_root
from scripts import train_go2_horizon_dense_predictor_development as fit


def main(root):
    if root.parent not in (fit.OUTPUT.parent, RECOVERY_BASE):
        raise ValueError('explicit existing development artifact base required')
    validate = bind(validate_root, BASE=root.parent)
    validate(root)
    read = lambda name:json.loads((root/name).read_text())
    if not (root/'result.json').exists() and not (root/'failure.json').exists():
        raise ValueError('terminal native attempt required')
    output = root/'dense_navigation_readout.json'
    if output.exists():
        raise ValueError('preserve completed evaluation')
    plans = [p for p in read('planning.json') if 'selection' in p]
    calls = read('dense_model_calls.json')
    launch = read('launch.json')
    arm = launch['model_assignment']
    neural_used = arm in ('action','no_future_action')
    assert len(plans)==len(calls)>0
    for plan, call in zip(plans, calls, strict=True):
        assert plan['measured_ns']==call['observed_ns']
        assert call['context_times_ns']==[plan['measured_ns']-d for d in (1_000_000_000, 500_000_000, 0)]
        c = plan['motion_correction']
        assert not c['external_neural_correction_applied']
        np.testing.assert_array_equal(c['raw_forecast_xy_m'], np.asarray(call['motion_xy_yaw'])[:, :, :2])
        if neural_used:
            assert c['neural_outcomes_used_for_scoring']
            np.testing.assert_array_equal(c['upstream_prediction_for_yaw_ablation'], c['applied_prediction_after_yaw_ablation'])
        elif arm=='command_history':
            assert c['prediction_source']=='command_history' and not c['neural_outcomes_used_for_scoring']
        elif arm=='reactive_feedback':
            assert c['neural_outcomes_used_for_action_selection'] is False
            assert plan['selection']['forecast_values_used_for_selection'] is False
        else:
            raise ValueError('unknown comparison arm')
    physical = arrivals(root)
    windows = bind(xy, validate_root=validate)(root)
    comparisons = same_window_xy(root, windows)
    yaw = yaw_metrics(root, windows)
    model_ms = np.asarray([c['wall_ns']/1e6 for c in calls])
    result = dict(root=str(root), physical=physical,
        selected_actions=dict(Counter(p['action'] for p in plans)),
        dispatch_reasons=dict(Counter(r['reason'] for r in read('requests.json'))),
        arm=arm, neural_calls=len(calls), model_forecasts_used_without_external_motion_correction=neural_used,
        motion_readout=read('launch.json').get('motion_readout', {'arm': 'original'}),
        new_independent_development_layout=launch.get('new_independent_development_layout', False),
        layout_index=launch['layout_index'],
        model_wall_ms=dict(zip(('median', 'p95', 'max'), np.percentile(model_ms, [50, 95, 100]).tolist())),
        executed_windows=windows, same_window_xy=comparisons, same_window_yaw=yaw,
        decisions=summarize_decisions(root),
        pipeline_faults=read('pipeline_faults.json'),
        limitations=['prospective same-family development maze; no sealed evaluation' if launch.get('new_independent_development_layout') else 'exposed development maze', 'untimed synchronous simulation',
            'matched executed windows are overlapping and selection-biased',
            'command-history forecasts on these windows do not establish alternative navigation outcomes',
            'readout training horizons: '+str(launch.get('motion_readout', {}).get('training_horizons_ms', [500])),
            'no learned collision prediction',
            'ideal body gyro and synthetic depth noise; no hardware validation'])
    fit.save(output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('executed_windows', 'same_window_xy', 'same_window_yaw','decisions')}, indent=2))
    print('DECISIONS',json.dumps({k:v for k,v in result['decisions'].items() if k!='rows'}),flush=True)
    print('MATCHED_XY', json.dumps({k:v for k,v in comparisons.items() if k!='rows'}), flush=True)
    print('MATCHED_YAW', json.dumps({k:v for k,v in yaw.items() if k!='rows'}), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-name', required=True)
    args = parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary development artifact basename required')
    base = RECOVERY_BASE if '_maze_view_' in args.root_name else fit.OUTPUT.parent
    main(base/args.root_name)
