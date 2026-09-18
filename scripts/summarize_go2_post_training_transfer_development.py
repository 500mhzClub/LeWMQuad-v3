"""Collect all sixteen fixed outcomes, including failed missions."""
from itertools import combinations
import json

from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_post_training_transfer_development import CONDITIONS


def main():
    output = path('go2_post_training_transfer_four_layout_summary_v1_attempt_001')
    if output.exists(): raise ValueError('preserve completed experiment summary')
    layouts = []; common_sources = None
    for index in range(4):
        comparison_root = path(f'go2_post_training_transfer_comparison_layout{index:02d}_v1_attempt_001')
        comparison = read(comparison_root, 'result.json')
        if (comparison['layout_index'] != index or
                set(comparison['conditions']) != set(CONDITIONS) or
                comparison['common_source_hashes_equal'] is not True):
            raise ValueError('complete matched four-condition layout required')
        if common_sources is None: common_sources = comparison['common_sources']
        if comparison['common_sources'] != common_sources:
            raise ValueError('shared implementation changed across layouts')
        rows = {}
        for condition in CONDITIONS:
            summary = comparison['conditions'][condition]
            evaluation = summary['independent_arrival_evaluation']
            forecast = None if condition == 'reactive' else read(
                path(summary['root_name']), 'saved_executed_motion_forecast_evaluation_v1.json')
            rows[condition] = dict(root_name=summary['root_name'],
                goal=any(a['phase'] == 'OUTBOUND' and a['arrival_checks_passed']
                    for a in evaluation['arrivals']),
                round_trip=evaluation['round_trip_arrival_checks_passed'],
                disallowed_contacts=evaluation['disallowed_contact_samples'],
                navigation_summary=summary,
                executed_forecast_metrics=None if forecast is None else {
                    k:v for k,v in forecast.items() if k != 'rows'})
        layouts.append(dict(layout_index=index, comparison_root_name=comparison_root.name,
            conditions=rows))
    totals = {condition:dict(assignments=4,
        goals=sum(row['conditions'][condition]['goal'] for row in layouts),
        round_trips=sum(row['conditions'][condition]['round_trip'] for row in layouts),
        disallowed_contacts=sum(row['conditions'][condition]['disallowed_contacts'] for row in layouts))
        for condition in CONDITIONS}
    pairs = []
    for a,b in combinations(CONDITIONS, 2):
        outcomes = [(row['conditions'][a]['round_trip'],row['conditions'][b]['round_trip'])
            for row in layouts]
        pairs.append(dict(first=a, second=b,
            first_only=sum(x and not y for x,y in outcomes),
            second_only=sum(y and not x for x,y in outcomes), ties=sum(x == y for x,y in outcomes)))
    report = dict(layouts=layouts, totals=totals, paired_round_trip_outcomes=pairs,
        fixed_assignments=16, model_training_seed=2026091001,
        comparison='three_frozen_training_methods_and_heading_first_reactive',
        all_assigned_failures_retained=True, common_sources=common_sources,
        all_per_layout_shared_settings_sources_and_runtime_corrections_checked=True,
        prospective_new_development_layouts=True,
        novelty_scope='distinct_from_explicit_64_layout_source_registry_same_generator_family',
        reactive_recovery_rules_differ=True, condition_specific_motion_corrections=True,
        isolated_jepa_training_effect_established=False,
        isolated_predictive_scoring_effect_established=False,
        executed_forecast_errors_are_controller_trajectory_conditional=True,
        forecast_windows_overlap=True, statistical_superiority_established=False,
        independent_training_seeds=False, host_real_time_qualified=False,
        real_sensor_uncertainty_calibrated=False, hardware_validated=False, final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as sink: json.dump(report,sink,indent=2)
    print(json.dumps(dict(totals=totals,paired_round_trip_outcomes=pairs)),flush=True)


if __name__ == '__main__': main()
