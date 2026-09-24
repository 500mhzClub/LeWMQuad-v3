"""Report all four assigned controllers on one completed new development maze."""
import argparse
import json
from lewm.matched_motion_residual_runtime_development import FITS
from scripts.compare_continuous_navigation_arms_development import (
    path, read, summarize, MATCHED_FIELDS, OPTIONAL_MATCHED_FIELDS)

CONDITIONS = ('jepa', 'direct', 'supervised_rollout', 'reactive')
SHARED_FIELDS = MATCHED_FIELDS + OPTIONAL_MATCHED_FIELDS + (
    'tracker', 'registration', 'independent_obstacle_observer', 'mapper',
    'synthetic_depth_noise', 'fresh_layout_inventory', 'frozen_layout_inventory_sha256',
    'routing_memory_scope', 'gyro_coherent_paired_floor_constraint',
    'floor_and_correspondence_depth_source', 'mapping_floor_depth_source',
    'height_cluster_minimum_pool_fraction', 'height_cluster_maximum_refinements',
    'mapping_floor_height_tolerance_m', 'retained_local_depth_cache_maximum_entries')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=range(4),required=True)
    i=parser.parse_args().layout_index
    output=path(f'go2_post_training_transfer_comparison_layout{i:02d}_v1_attempt_001')
    if output.exists():raise ValueError('preserve completed comparison')
    summaries={}; launches={}; sources={}; corrections={}
    for condition in CONDITIONS:
        root=path(f'go2_post_training_transfer_{condition}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001')
        launch=read(root,'launch.json')
        assignment='reactive' if condition=='reactive' else f'seed_2026091001_full_{condition}'
        if (launch['model_assignment']!=assignment or launch['layout_index']!=i or
                launch['comparison_condition']!=condition or
                launch['experiment']!='post_training_four_controller_transfer_noise_development_v1' or
                launch['tracker']!='GyroCoherentFloorMotion'):
            raise ValueError('recorded fixed assignment or shared estimator differs')
        plans=[p for p in read(root,'planning.json') if 'selection' in p]
        if condition!='reactive':
            fit_root,fit_hash=FITS[condition]
            if (launch['motion_residual_correction_root']!=fit_root or
                    launch['closed_loop_motion_residual_fit_sha256']!=fit_hash):
                raise ValueError('frozen correction assignment differs')
            for p in plans:
                c=p['motion_correction']
                if (c['fit_sha256']!=fit_hash or c['correction_root']!=fit_root or
                        c['correction_base_model']!=assignment):
                    raise ValueError('selected plan used a different model correction')
            corrections[condition]=dict(selected_plans=len(plans),runtime_correction_matches_assignment=True)
        summaries[condition]=summarize(root)
        launches[condition]=launch
        sources[condition]=launch['source_sha256']|launch['extra_sources']
    reference=launches['jepa']
    differences={condition:[k for k in SHARED_FIELDS if launch.get(k)!=reference.get(k)]
        for condition,launch in launches.items()}
    if any(differences.values()):raise ValueError(f'shared mission/perception/settings differ: {differences}')
    common=set.intersection(*(set(s) for s in sources.values()))
    changed=[name for name in sorted(common) if len({s[name] for s in sources.values()})!=1]
    if changed:raise ValueError(f'common source changed within fixed cohort: {changed}')
    report=dict(layout_index=i,conditions=summaries,
        matched_settings={k:reference.get(k) for k in SHARED_FIELDS},
        common_source_hashes_equal=True,common_source_count=len(common),
        common_sources={k:sources['jepa'][k] for k in sorted(common)},
        correction_bindings=corrections,
        frozen_model_training_seed=2026091001,
        comparison='three_frozen_training_methods_and_heading_first_reactive',
        reactive_recovery_rules_differ=True,condition_specific_motion_corrections=True,
        isolated_jepa_training_effect_established=False,
        isolated_predictive_scoring_effect_established=False,
        statistical_superiority_established=False,host_real_time_qualified=False,
        calibrated_hardware_sensing=False,final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as sink:json.dump(report,sink,indent=2)
    print(json.dumps({c:dict(goals=sum(a['phase']=='OUTBOUND' and a['arrival_checks_passed']
        for a in s['independent_arrival_evaluation']['arrivals']),
        round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'])
        for c,s in summaries.items()}))


if __name__=='__main__':main()
