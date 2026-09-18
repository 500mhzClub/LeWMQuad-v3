"""Compare saved XY alternatives on identical executed windows in all 36 runs."""
import json
import numpy as np

from scripts.compare_continuous_navigation_arms_development import path, read

SOURCES = dict(raw_neural='raw_forecast_xy_m',
               corrected_neural='learned_corrected_forecast_xy_m',
               fitted_pose_command='pose_command_forecast_xy_m')


def summarize(rows):
    return dict(windows=len(rows), **{name:dict(
        endpoint_rmse_mm=1000*float(np.sqrt(np.mean([r[name]**2 for r in rows]))),
        endpoint_mean_error_mm=1000*float(np.mean([r[name] for r in rows])))
        for name in SOURCES})


def main():
    cohort = read(path('go2_neural_rgb_transfer_complete_comparison_v1_attempt_001'), 'result.json')
    output = path('go2_neural_rgb_motion_controls_readout_v1_attempt_001')
    if output.exists():
        raise ValueError('preserve completed readout')
    reports = []; pooled = []; identities = set()
    for assignment in cohort['rows']:
        root = path(assignment['root_name'])
        plans = {p['frame']:p for p in read(root, 'planning.json') if 'selection' in p}
        executed = read(root, 'saved_executed_motion_forecast_evaluation_v1.json')
        if (executed['matched_requested_sequence_through_ns'] != 700_000_000
                or executed['unexecuted_candidates_evaluated']):
            raise ValueError('matched executed 700-ms windows required')
        rows = []
        for window in executed['rows']:
            plan = plans[window['frame']]; selected = plan['selection']; receipt = plan['motion_correction']
            if window['action'] != selected['action'] or not receipt['both_xy_alternatives_computed_in_both_arms']:
                raise ValueError('saved alternatives do not match the selected action')
            identities.add(receipt['pose_command_fit_sha256'])
            actual = np.asarray(window['actual_endpoint_xy_m'])
            row = dict(frame=window['frame'], action=window['action'], group=window['group'])
            for name, key in SOURCES.items():
                prediction = np.asarray(receipt[key])[selected['action_index'], 6]
                row[name] = float(np.linalg.norm(prediction-actual))
            if not np.isclose(row['corrected_neural'], window['corrected_endpoint_error_m'], rtol=0., atol=1e-12):
                raise ValueError('corrected endpoint error differs from completed evaluation')
            rows.append(row)
        groups = {group:summarize([r for r in rows if r['group']==group])
                  for group in sorted({r['group'] for r in rows})}
        reports.append({k:assignment[k] for k in ('root_name', 'seed', 'method', 'layout_index', 'condition', 'round_trip')}
                       | dict(summary=summarize(rows), by_action_group=groups))
        pooled.extend(rows)
    report = dict(assignments=len(reports), distinct_mazes=2, runs=reports,
        descriptive_pooled_windows=summarize(pooled),
        by_action_group={g:summarize([r for r in pooled if r['group']==g]) for g in sorted({r['group'] for r in pooled})},
        corrected_neural_lower_rmse_runs=sum(r['summary']['corrected_neural']['endpoint_rmse_mm']
            < r['summary']['fitted_pose_command']['endpoint_rmse_mm'] for r in reports),
        fitted_pose_command_fit_sha256=sorted(identities),
        scope=dict(same_actual_command_windows_for_all_forecasts=True,
            command_prefix_and_terminal_translation_pulse_preserved=True,
            failed_mission_windows_retained=True, overlapping_windows_not_independent=True,
            pooled_errors_are_descriptive_not_independent_trials=True,
            unexecuted_candidate_accuracy_established=False,
            alternative_navigation_outcome_established=False, hardware_validated=False))
    output.mkdir()
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='runs'}), flush=True)


if __name__ == '__main__':
    main()
