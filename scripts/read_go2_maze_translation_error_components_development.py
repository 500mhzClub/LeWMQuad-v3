"""Describe saved 700-ms translation errors; no fitting or sensor replay."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


BASE = Path('.generated/navigation_development_artifacts_v1')
ROOT = BASE / 'go2_dense_world_model_maze_layout00_action_mixed_data_v1_attempt_001'
OUTPUT = ROOT / 'translation_error_components_v1.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve completed diagnosis')
    paths = {
        'readout': ROOT / 'dense_navigation_readout.json',
        'plans': ROOT / 'planning.json',
        'coverage': BASE / 'go2_multihorizon_motion_readout_v1_attempt_001'
        / 'motion_magnitude_coverage_diagnostic.json',
    }
    data = {k: json.loads(p.read_text()) for k, p in paths.items()}
    plans = {p['frame']: p for p in data['plans'] if 'selection' in p}
    checks = {r['frame']: r for r in data['readout']['same_window_xy']['rows']}
    maximum = data['coverage']['training']['old_500ms']['xy_mm']['maximum'] / 1000
    actions = ['hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn']
    rows = []
    for w in data['readout']['executed_windows']['rows']:
        if w['group'] != 'translation':
            continue
        p = plans[w['frame']]
        assert p['action'] == w['action']
        i = actions.index(w['action'])
        a = w['actual_endpoint_xy_m']
        length = math.hypot(*a)
        assert length > 0
        u = [v / length for v in a]
        models = {}
        for name, field in [('neural', 'raw_forecast_xy_m'),
                            ('command_history', 'command_history_forecast_xy_yaw')]:
            predicted = p['motion_correction'][field][i][6][:2]
            error = [predicted[j] - a[j] for j in range(2)]
            norm = math.hypot(*error)
            assert abs(norm - checks[w['frame']]['errors_m'][name]) < 1e-10
            parallel = sum(error[j] * u[j] for j in range(2))
            transverse = -error[0] * u[1] + error[1] * u[0]
            assert abs(norm ** 2 - parallel ** 2 - transverse ** 2) < 1e-12
            models[name] = dict(predicted_xy_m=predicted, error_m=norm,
                                parallel_error_m=parallel, transverse_error_m=transverse,
                                predicted_progress_fraction=(length + parallel) / length)
        rows.append(dict(frame=w['frame'], action=w['action'], actual_xy_m=a,
                         actual_length_m=length, exceeds_old_training_max=length > maximum,
                         models=models))
    assert len(rows) == 116
    groups = {}
    for group, population in [('all', rows),
                              ('within_old_training_max', [r for r in rows if not r['exceeds_old_training_max']]),
                              ('above_old_training_max', [r for r in rows if r['exceeds_old_training_max']])]:
        summary = {}
        for model in ('neural', 'command_history'):
            m = [r['models'][model] for r in population]
            energy = sum(v['error_m'] ** 2 for v in m)
            summary[model] = dict(
                xy_rmse_mm=1000 * math.sqrt(mean(v['error_m'] ** 2 for v in m)),
                parallel_bias_mm=1000 * mean(v['parallel_error_m'] for v in m),
                parallel_rmse_mm=1000 * math.sqrt(mean(v['parallel_error_m'] ** 2 for v in m)),
                transverse_rmse_mm=1000 * math.sqrt(mean(v['transverse_error_m'] ** 2 for v in m)),
                parallel_fraction_squared_error=sum(v['parallel_error_m'] ** 2 for v in m) / energy,
                underpredicts_progress=sum(v['parallel_error_m'] < 0 for v in m),
                predicts_opposite_progress=sum(v['predicted_progress_fraction'] < 0 for v in m),
                median_predicted_progress_fraction=median(v['predicted_progress_fraction'] for v in m))
        groups[group] = dict(windows=len(population), models=summary)
    result = dict(status='COMPLETE', horizon_ms=700, old_training_maximum_m=maximum,
                  groups=groups, rows=rows,
                  inputs={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for k, p in paths.items()},
                  method='Project prediction-minus-truth onto the unit actual displacement and its perpendicular; negative parallel error means underestimated progress.',
                  limitations=['Post hoc descriptive decomposition on one completed exposed trajectory.',
                               'Overlapping windows are not independent trials.',
                               'Native endpoint truth is evaluator-only; no fit or controller change.',
                               'Errors do not isolate encoder, predictor, readout, or training-coverage causes.',
                               'No magnitude calibration or alternative navigation result is inferred.'])
    with OUTPUT.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
