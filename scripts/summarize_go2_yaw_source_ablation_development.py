"""Summarize all four fixed yaw assignments, including early failures."""
import json

from scripts.compare_continuous_navigation_arms_development import path, read


def main():
    pairs = [read(path(f'go2_yaw_source_ablation_comparison_layout{i:02d}_v1_attempt_001'),
                  'result.json') for i in (0, 1)]
    if pairs[0]['common_sources'] != pairs[1]['common_sources']:
        raise ValueError('shared runtime changed between layouts')
    rows = []; totals = {}
    for source in ('learned', 'command'):
        for pair in pairs:
            outcome = pair['conditions'][source]
            evaluation = outcome['independent_arrival_evaluation']
            arrivals = evaluation['arrivals']
            goal = any(a['phase'] == 'OUTBOUND' and a['arrival_checks_passed'] for a in arrivals)
            rows.append(dict(yaw_source=source, layout_index=pair['layout_index'],
                root_name=outcome['root_name'], goal=goal,
                round_trip=evaluation['round_trip_arrival_checks_passed'],
                disallowed_contacts=evaluation['disallowed_contact_samples'],
                camera_pairs=evaluation['camera_pairs'], failure=outcome['failure'],
                mission_terminal=evaluation['mission_terminal'],
                executed_yaw=pair['executed_yaw_metrics'][source],
                executed_xy=pair['executed_xy_metrics'][source]))
        selected = [r for r in rows if r['yaw_source'] == source]
        totals[source] = dict(assignments=len(selected), goals=sum(r['goal'] for r in selected),
            round_trips=sum(r['round_trip'] for r in selected),
            disallowed_contacts=sum(r['disallowed_contacts'] for r in selected))
    report = dict(totals=totals, rows=rows, fixed_assignments_completed=4,
        common_runtime_sources_equal=True, exposed_development_layouts=[0, 1],
        single_training_seed=True, fitted_pose_command_xy_retained=True,
        contact_score_disabled=True, predictive_planning_in_both_arms=True,
        repaired_frontier_policy_in_both_arms=True, neural_network_evaluated_in_both_arms=True,
        statistical_advantage_established=False, general_reliability_established=False,
        host_real_time_qualified=False, hardware_validated=False)
    output = path('go2_yaw_source_ablation_two_layout_summary_v1_attempt_001')
    output.mkdir()
    with (output/'result.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
