"""Descriptive results for the fixed RGB pilot; failures remain explicit."""
import csv
import json
from statistics import mean

from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.run_go2_neural_rgb_transfer_development import SEEDS, METHODS


def collect(seeds=SEEDS):
    rows, pairs = [], []
    for seed in seeds:
        for method in METHODS:
            for layout in (0, 1):
                root = path(f'go2_neural_rgb_transfer_comparison_seed_{seed}_{method}'
                            f'_layout{layout:02d}_v1_attempt_001')
                comparison = read(root, 'result.json')
                terminal = read(root, 'terminal_approach_diagnostic_v1.json')['conditions']
                pair_rows = {}
                for source in comparison['rows']:
                    variant = source['condition']
                    recording = path(source['root_name'])
                    xy = read(recording, 'saved_executed_motion_forecast_evaluation_v1.json')
                    yaw = read(recording, 'saved_neural_command_yaw_evaluation_v1.json')
                    result = comparison['conditions'][variant]['result']
                    completed = source['round_trip']
                    terminal_s = sum(p.get('measured_approach_seconds', 0.)
                                     for p in terminal[variant].values())
                    row = source | dict(seed=seed, method=method, layout_index=layout,
                        completion_time_s=source['simulation_s'] if completed else None,
                        terminal_window_s=terminal_s,
                        outside_terminal_window_s=(source['simulation_s']-terminal_s)
                            if completed else None,
                        peak_simulator_lag_ms=None if result is None else result['max_simulator_lag_ms'],
                        executed_forecast_windows=xy['windows'],
                        raw_xy_rmse_mm=1000*xy['raw_endpoint_xy_rmse_m'],
                        corrected_xy_rmse_mm=1000*xy['corrected_endpoint_xy_rmse_m'],
                        neural_yaw_rmse_deg=yaw['neural_endpoint_rmse_deg'],
                        command_yaw_rmse_deg=yaw['command_endpoint_rmse_deg'])
                    rows.append(row)
                    pair_rows[variant] = row
                full, no_rgb = pair_rows['full'], pair_rows['no_rgb']
                both = full['round_trip'] and no_rgb['round_trip']
                pairs.append(dict(seed=seed, method=method, layout_index=layout,
                    full_round_trip=full['round_trip'], no_rgb_round_trip=no_rgb['round_trip'],
                    both_round_trip=both,
                    no_rgb_minus_full_completion_s=(no_rgb['completion_time_s']-full['completion_time_s'])
                        if both else None,
                    no_rgb_minus_full_terminal_s=(no_rgb['terminal_window_s']-full['terminal_window_s'])
                        if both else None,
                    no_rgb_minus_full_outside_terminal_s=(no_rgb['outside_terminal_window_s']
                        -full['outside_terminal_window_s']) if both else None))
    groups = []
    for layout in (0, 1):
        for method in METHODS:
            for variant in ('full', 'no_rgb'):
                selected = [r for r in rows if (r['layout_index'], r['method'], r['condition'])
                            == (layout, method, variant)]
                successes = [r for r in selected if r['round_trip']]
                groups.append(dict(layout_index=layout, method=method, condition=variant,
                    assignments=len(selected), round_trips=len(successes),
                    contacts=sum(r['contacts'] for r in selected),
                    successful_completion_mean_s=mean(r['completion_time_s'] for r in successes)
                        if successes else None,
                    successful_completion_min_s=min((r['completion_time_s'] for r in successes), default=None),
                    successful_completion_max_s=max((r['completion_time_s'] for r in successes), default=None),
                    failure_roots=[r['root_name'] for r in selected if not r['round_trip']]))
    return dict(completed_assignments=len(rows), round_trips=sum(r['round_trip'] for r in rows),
        contacts=sum(r['contacts'] for r in rows), distinct_development_mazes=2,
        training_seeds=list(seeds), rows=rows, pairs=pairs, by_layout_method_input=groups,
        full_faster_successful_pairs=sum(p['both_round_trip'] and p['no_rgb_minus_full_completion_s']>0
                                         for p in pairs),
        both_successful_pairs=sum(p['both_round_trip'] for p in pairs),
        command_yaw_lower_error_runs=sum(r['command_yaw_rmse_deg']<r['neural_yaw_rmse_deg'] for r in rows),
        scope=dict(descriptive_development_pilot=True, independent_environmental_units=2,
            seed_repetitions_are_not_independent_mazes=True,
            failed_missions_excluded_only_from_successful_completion_times=True,
            terminal_windows_are_post_hoc=True, forecast_windows_overlap=True,
            forecast_errors_use_each_controllers_own_executed_trajectory=True,
            input_ablation_includes_training_and_correction=True,
            rgbd_tracking_and_mapping_retained_in_both_inputs=True,
            statistical_superiority_established=False, real_time_qualified=False,
            hardware_validated=False, final_evaluation=False))


def main():
    result = collect()
    output = path('go2_neural_rgb_transfer_complete_comparison_v1_attempt_001')
    output.mkdir()
    with (output/'result.json').open('x') as f:
        json.dump(result, f, indent=2)
    for name in ('rows', 'pairs', 'by_layout_method_input'):
        with (output/f'{name}.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(result[name][0]))
            writer.writeheader()
            writer.writerows(result[name])
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ('rows', 'pairs', 'by_layout_method_input')}), flush=True)


if __name__ == '__main__':
    main()
