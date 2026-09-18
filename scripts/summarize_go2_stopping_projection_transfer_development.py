"""Summarize all twelve fixed transfer assignments without dropping failures."""
from collections import Counter
import json
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.run_go2_stopping_projection_transfer_development import CONDITIONS


def main():
    output=path('go2_stopping_projection_transfer_four_layout_summary_v1_attempt_001')
    if output.exists():raise ValueError('preserve completed aggregate')
    rows=[]
    for index in range(4):
        comparison=read(path(f'go2_stopping_projection_transfer_comparison_layout{index:02d}_v1_attempt_001'),'result.json')
        if set(comparison['conditions'])!=set(CONDITIONS):raise ValueError('all three assigned controllers required')
        for condition in CONDITIONS:
            summary=comparison['conditions'][condition];physical=summary['independent_arrival_evaluation']
            rows.append(dict(layout_index=index,condition=condition,root_name=summary['root_name'],
                verified_goal=any(a['phase']=='OUTBOUND' and a['arrival_checks_passed'] for a in physical['arrivals']),
                verified_round_trip=physical['round_trip_arrival_checks_passed'],
                disallowed_contact_samples=physical['disallowed_contact_samples'],
                captured_frames=physical['camera_pairs'],accepted_poses=physical['registered_poses'],
                maximum_pose_error_m=physical['maximum_position_error_m'],
                mission_terminal=physical['mission_terminal'],failure=summary['failure'],
                simulation_s=(summary['result'] or {}).get('simulation_s'),
                physical_arrivals=physical['arrivals'],
                path_length_m=summary['native_10hz_horizontal_path_length_m'],
                behavior=comparison['behavior_metrics'][condition],
                views=comparison['view_metrics'][condition],
                actual_treatment=comparison['actual_treatments'][condition]))
    totals={}
    for condition in CONDITIONS:
        population=[r for r in rows if r['condition']==condition]
        totals[condition]=dict(assignments=len(population),verified_goals=sum(r['verified_goal'] for r in population),
            verified_round_trips=sum(r['verified_round_trip'] for r in population),
            disallowed_contact_samples=sum(r['disallowed_contact_samples'] for r in population),
            terminal_counts=dict(Counter(str(r['mission_terminal']) for r in population)))
    report=dict(study='fixed_three_controller_stopping_projection_transfer',native_assignments=12,
        fresh_development_layouts=4,explicit_prior_registry_layouts=72,rows=rows,totals=totals,
        all_failures_retained=True,reactive_predictive_clearance_and_recovery_differ=True,
        isolated_predictive_ranking_effect_established=False,jepa_training_effect_established=False,
        multiple_training_seeds_evaluated=False,statistical_advantage_established=False,
        out_of_family_generalization_established=False,real_sensor_uncertainty_calibrated=False,
        host_real_time_qualified=False,hardware_validated=False)
    output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
