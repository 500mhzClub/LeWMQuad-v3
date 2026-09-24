"""Summarize every fixed mission, preserving unsuccessful navigation outcomes."""
import json

from scripts import run_go2_short_pulse_navigation_development as study
from scripts.navigation_artifact_root_development import create_output

OUTPUT = study.BASE/'go2_short_pulse_navigation_complete_v1_attempt_001'
MAZE1 = study.BASE/'go2_short_pulse_navigation_maze01_complete_v1_attempt_001'


def main():
    rows = []
    conditions = {}
    for number, (index, arm) in enumerate(study.ASSIGNMENTS, 1):
        root = study.BASE/study.ROOT.format(index=index, arm=arm)
        read = lambda name: json.loads((root/name).read_text())
        evaluated = read('short_pulse_navigation_evaluation_v1.json')
        summary = read('live_navigation_summary_v1.json')
        treatment = read('actual_controller_treatment_v1.json')
        if evaluated['assignment'] != number or not treatment['actual_treatment_verified']:
            raise ValueError('complete fixed assignment and actual treatment required')
        physical = summary['independent_arrival_evaluation']
        outbound = any(a['phase'] == 'OUTBOUND' and a['arrival_checks_passed']
            for a in evaluated['arrivals'])
        row = dict(assignment=number, layout_index=index, arm=arm, root_name=root.name,
            round_trip=evaluated['round_trip'], outbound_goal=outbound,
            contact_samples=evaluated['contacts'], terminal=evaluated['terminal'],
            failure=evaluated['failure'], simulation_s=evaluated['simulation_s'],
            plans=evaluated['plans'], plans_on_time=evaluated['plans_on_time'],
            path_length_m=summary['native_10hz_horizontal_path_length_m'],
            minimum_goal_distance_m=summary['minimum_outbound_native_goal_distance_m'],
            final_goal_distance_m=summary['final_native_goal_distance_m'],
            final_home_distance_m=summary['final_native_home_distance_m'],
            maximum_position_error_m=physical['maximum_position_error_m'])
        if arm != 'reactive':
            xy = read('saved_short_pulse_same_window_xy_v1.json')
            yaw = read('saved_short_pulse_yaw_evaluation_v1.json')
            row.update(executed_forecast_windows=xy['windows'],
                same_window_xy_rmse_mm=xy['rmse_mm'], same_window_yaw_rmse_deg=yaw['rmse_deg'])
        rows.append(row)
        if index == 1:
            conditions[arm] = summary
    if OUTPUT.exists() or MAZE1.exists():
        raise ValueError('preserve existing complete summaries')

    def totals(selected):
        return dict(missions=len(selected), round_trips=sum(r['round_trip'] for r in selected),
            outbound_goals=sum(r['outbound_goal'] for r in selected),
            contact_failures=sum(r['contact_samples'] > 0 for r in selected))

    result = dict(status='COMPLETE', rows=rows, **totals(rows),
        by_method={a:totals([r for r in rows if r['arm'] == a]) for a in study.ARMS},
        by_layout={str(i):totals([r for r in rows if r['layout_index'] == i]) for i in (0, 1)},
        neural_methods=totals([r for r in rows if r['arm'] in study.ARMS[:3]]),
        independent_layouts=2, executions_per_method_per_layout=1, training_seed=2026091001,
        all_negative_results_retained=True, final_evaluation=False, hardware_validated=False,
        sensor_noise_sd_mm=2, gyro_noise_model='ideal', host_real_time_qualified=False,
        external_neural_motion_correction=False,
        reactive_changes_forecast_guards_terminal_rule_and_compute=True,
        instantaneous_retains_predictive_guards=True,
        unused_forecasts_are_not_alternative_navigation_outcomes=True,
        overlapping_forecast_windows_are_not_independent_trials=True)
    create_output(OUTPUT)
    (OUTPUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    maze = dict(layout_index=1, figure_title='Complete second-maze comparison: seven fixed controllers',
        conditions={a:conditions[a] for a in study.ARMS},
        rows=[r for r in rows if r['layout_index'] == 1],
        **totals([r for r in rows if r['layout_index'] == 1]),
        independent_layouts=1, final_evaluation=False,
        reactive_is_broader_controller_comparison=True,
        instantaneous_retains_predictive_guards=True)
    create_output(MAZE1)
    (MAZE1/'result.json').write_text(json.dumps(maze, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
