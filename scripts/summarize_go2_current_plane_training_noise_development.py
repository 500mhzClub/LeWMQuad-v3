"""Collect the complete fixed four-layout, three-training-method experiment."""
from itertools import combinations
import json

from scripts.compare_continuous_navigation_arms_development import path, read

CONDITIONS = ('jepa', 'direct', 'supervised_rollout')


def main():
    output = path('go2_current_plane_matched_training_noise_four_layout_summary_v1_attempt_001')
    if output.exists():
        raise ValueError('preserve existing complete experiment summary')
    layouts = []
    for index in range(4):
        comparison = read(path(f'go2_current_plane_matched_training_comparison_layout{index:02d}_v1_attempt_001'), 'result.json')
        if (comparison['layout_index'] != index or
                set(comparison['conditions']) != set(CONDITIONS) or
                comparison['common_settings_and_sources_equal'] is not True):
            raise ValueError('complete matched three-condition layout result required')
        rows = {}
        for condition in CONDITIONS:
            summary = comparison['conditions'][condition]
            evaluation = summary['independent_arrival_evaluation']
            forecast = read(path(summary['root_name']), 'saved_executed_motion_forecast_evaluation_v1.json')
            rows[condition] = dict(
                root_name=summary['root_name'],
                goal=any(a['phase'] == 'OUTBOUND' and a['arrival_checks_passed'] for a in evaluation['arrivals']),
                round_trip=evaluation['round_trip_arrival_checks_passed'],
                disallowed_contacts=evaluation['disallowed_contact_samples'],
                navigation_summary=summary,
                executed_forecast_metrics={k:v for k,v in forecast.items() if k != 'rows'})
        layouts.append(dict(layout_index=index, conditions=rows))
    totals = {condition:dict(
        assignments=4,
        goals=sum(row['conditions'][condition]['goal'] for row in layouts),
        round_trips=sum(row['conditions'][condition]['round_trip'] for row in layouts),
        disallowed_contacts=sum(row['conditions'][condition]['disallowed_contacts'] for row in layouts))
        for condition in CONDITIONS}
    pairs = []
    for a, b in combinations(CONDITIONS, 2):
        outcomes = [(row['conditions'][a]['round_trip'], row['conditions'][b]['round_trip']) for row in layouts]
        pairs.append(dict(first=a, second=b, first_only=sum(x and not y for x,y in outcomes),
            second_only=sum(y and not x for x,y in outcomes), ties=sum(x == y for x,y in outcomes)))
    report = dict(layouts=layouts, totals=totals, paired_round_trip_outcomes=pairs,
        fixed_assignments=12, model_training_seed=2026091001,
        comparison='training_methods_with_condition_matched_frozen_visual_motion_corrections',
        all_per_layout_common_settings_sources_and_runtime_corrections_checked=True,
        development_exposed_layouts=True, independent_training_seeds=False,
        executed_forecast_errors_are_controller_trajectory_conditional=True,
        forecast_windows_overlap=True, statistical_training_advantage_established=False,
        host_real_time_qualified=False, real_sensor_uncertainty_calibrated=False,
        final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as sink:
        json.dump(report, sink, indent=2)
    print(json.dumps(dict(totals=totals, paired_round_trip_outcomes=pairs)), flush=True)


if __name__ == '__main__':
    main()
