"""Compare the three original full wall-clock controller assignments."""
import json
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_post_training_transfer_development import SHARED_FIELDS

ROOT='go2_wall_three_controls_comparison_layout01_v1_attempt_001'
CONDITIONS=('learned','pose_command','reactive')
FIELDS=SHARED_FIELDS+('asynchronous_camera_acquisition',
    'sensor_packet_assembly_in_renderer_worker',
    'owner_cyclic_gc_deferred_during_bounded_mission',
    'maximum_motion_dispatch_simulator_lag_ns',
    'committed_camera_view_turn','local_view_reference_bank',
    'maximum_extra_local_view_references','maximum_recent_reference_age_ns')


def main():
    roots={c:path(f'go2_async_camera_wall_mission_{c}_layout01_4800_v1_attempt_001')
        for c in CONDITIONS}
    launches={c:read(r,'launch.json') for c,r in roots.items()}
    reference=launches['learned']
    differences={c:[k for k in FIELDS if a.get(k)!=reference.get(k)] for c,a in launches.items()}
    if any(differences.values()):raise ValueError(f'shared settings differ: {differences}')
    sources={c:a['source_sha256']|a['extra_sources'] for c,a in launches.items()}
    common=set.intersection(*(set(s) for s in sources.values()))
    if any(len({s[k] for s in sources.values()})!=1 for k in common):
        raise ValueError('shared runtime source changed')
    treatments={c:read(r,'actual_controller_treatment_v1.json') for c,r in roots.items()}
    for c,t in treatments.items():
        if t['condition']!=c or not t['actual_treatment_verified']:
            raise ValueError('actual treatment differs')
    summaries={c:read(r,'live_navigation_summary_v1.json') for c,r in roots.items()}
    result=dict(layout_index=1,conditions=summaries,actual_treatments=treatments,
        figure_title='Full wall-clock navigation: learned, fitted and reactive',
        matched_settings={k:reference.get(k) for k in FIELDS},
        common_source_hashes_equal=True,common_source_count=len(common),
        host_timing={c:read(r,'async_host_deadline_diagnostic_v1.json') for c,r in roots.items()},
        original_failures_retained=True,one_execution_per_condition=True,
        same_frozen_supervised_model_in_predictive_arms=True,
        fitted_arm_uses_predictive_planning=True,reactive_has_no_neural_inference=True,
        reactive_clearance_and_recovery_rules_differ=True,
        isolated_prediction_ranking_effect_established=False,
        jepa_training_advantage_established=False,statistical_superiority_established=False,
        exposed_development_maze=True,execution_order_randomized=False,
        hardware_validated=False,hard_real_time_qualified=False)
    output=path(ROOT);output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({c:dict(round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        contacts=s['independent_arrival_evaluation']['disallowed_contact_samples'],failure=s['failure'])
        for c,s in summaries.items()}))


if __name__=='__main__':main()
