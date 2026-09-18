"""Compare completed learned/fitted wall-clock missions, including failures."""
import json
from scripts.compare_continuous_navigation_arms_development import (
    path,read,MATCHED_FIELDS,OPTIONAL_MATCHED_FIELDS)
from scripts import run_go2_async_wall_fitted_control_development as fitted


def main():
    roots={'learned':path(fitted.learned.ROOT),'pose_command':path(fitted.ROOT)}
    launches={k:read(r,'launch.json') for k,r in roots.items()}
    a,b=launches.values()
    fields=(*MATCHED_FIELDS,*OPTIONAL_MATCHED_FIELDS,'synthetic_depth_noise',
        'tracker','asynchronous_camera_acquisition','sensor_packet_assembly_in_renderer_worker',
        'owner_cyclic_gc_deferred_during_bounded_mission','planned_stopping_projection')
    differences={k:[a.get(k),b.get(k)] for k in fields if a.get(k)!=b.get(k)}
    if differences:raise ValueError(f'controller-independent settings differ: {differences}')
    sources_a=a['source_sha256']|a['extra_sources'];sources_b=b['source_sha256']|b['extra_sources']
    common=sources_a.keys()&sources_b.keys()
    changed={k:[sources_a[k],sources_b[k]] for k in common if sources_a[k]!=sources_b[k]}
    if changed:raise ValueError(f'common runtime source changed: {changed}')
    for condition,root in roots.items():
        treatment=read(root,'actual_controller_treatment_v1.json')
        if not treatment['actual_treatment_verified'] or treatment['condition']!=condition:
            raise ValueError('actual forecast-source treatment must be verified')
    conditions={k:read(r,'live_navigation_summary_v1.json') for k,r in roots.items()}
    result=dict(comparison='learned_corrected_xy_and_yaw_vs_fitted_pose_command_xy_and_integrated_yaw',
        figure_title='Matched wall-clock navigation: learned and fitted motion',layout_index=1,
        conditions=conditions,matched_settings={k:a.get(k) for k in fields},
        common_source_hashes_equal=True,common_sources=len(common),
        additional_control_sources=sorted(sources_b.keys()-sources_a.keys()),
        actual_forecast_treatments_verified=True,predictive_planning_in_both_arms=True,
        neural_inference_computed_in_both_arms=True,both_missions_run_alone=True,
        same_physics_owner_and_nonstepping_renderer_allocation=True,
        one_execution_per_arm=True,execution_order_randomized=False,
        exposed_development_layout=True,jepa_training_advantage_established=False,
        isolated_predictive_planning_advantage_established=False,
        statistical_superiority_established=False,hardware_validated=False,
        real_time_qualified=False)
    output=path('go2_wall_clock_motion_sources_comparison_layout01_v1_attempt_001');output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:dict(round_trip=v['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        contacts=v['independent_arrival_evaluation']['disallowed_contact_samples'],
        simulation_s=v['result']['simulation_s'] if v['result'] else None,failure=v['failure'])
        for k,v in conditions.items()}))


if __name__=='__main__':main()
