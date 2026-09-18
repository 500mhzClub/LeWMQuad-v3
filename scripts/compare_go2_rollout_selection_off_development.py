"""Compare full forecast selection, instantaneous ranking, and rollout off."""
import argparse
import hashlib
import json
from scripts.compare_continuous_navigation_arms_development import path,read
from scripts.compare_go2_shared_recovery_transfer_development import SHARED
from scripts.compare_go2_combined_perception_motion_development import behavior
from scripts import run_go2_rollout_selection_off_development as experiment

TREATMENT_FIELDS={'planned_native_assignments','turn_recovery_releases_for_full_reserve_progress',
    'turn_prediction_error_reserve_m','stepwise_turn_reserve_recovery',
    'predictive_terminal_arrival_hold','waypoint_predicted_alignment_progress',
    'translation_prediction_error_reserve_m','reserve_recovery_requires_no_further_encroachment',
    'terminal_priority_requires_predicted_arrival','hold_relative_clearance_recovery',
    'hold_relative_heading_recovery','downstream_planner_and_recovery_unchanged',
    'predictive_planning_in_both_arms','planned_stopping_projection',
    'perception_and_recovery_shared_across_conditions','perception_and_measured_view_recovery_shared'}


def compare(index,arm,roots,output,figure_title):
    if output.exists():raise ValueError('preserve completed comparison')
    summaries={};launches={};sources={};rows=[];metrics={};treatments={}
    for label,root in roots.items():
        launch=read(root,'launch.json');summary=read(root,'live_navigation_summary_v1.json')
        if (root/'launch_annotation_correction.json').exists():
            correction=read(root,'launch_annotation_correction.json')
            if hashlib.sha256((root/'launch.json').read_bytes()).hexdigest()!=correction['original_launch_sha256']:
                raise ValueError('annotation correction does not bind original launch')
            launch=launch|correction['corrected_fields']
        treatment=read(root,'actual_controller_treatment_v1.json')
        plans=[p for p in read(root,'planning.json') if 'selection' in p]
        if (launch['study_arm']!=arm or launch['layout_index']!=index
                or not treatment['actual_treatment_verified']
                or not treatment['model_and_correction_binding_verified']):
            raise ValueError('computed model/correction assignment differs')
        if launch.get('full_learned_rollout_selection_disabled',False) and (treatment['predictive_outcomes_used']
                or not treatment['actual_rollout_selection_off_verified']):
            raise ValueError('actual rollout-off selection not verified')
        summaries[label]=summary;launches[label]=launch;treatments[label]=treatment
        sources[label]=launch['source_sha256']|launch['extra_sources'];metrics[label]=behavior(root,plans)
        e=summary['independent_arrival_evaluation'];result=summary['result']
        rows.append(dict(condition=label,root_name=root.name,
            round_trip=e['round_trip_arrival_checks_passed'],contacts=e['disallowed_contact_samples'],
            failure=summary['failure'],maximum_pose_error_m=e['maximum_position_error_m'],
            simulation_s=None if result is None else result['simulation_s'],
            selected_plans=len(plans),on_time_plans=sum(p['on_time'] for p in plans),
            native_path_length_m=summary['native_10hz_horizontal_path_length_m']))
    fields=tuple(k for k in SHARED if k not in TREATMENT_FIELDS)+(
        'model_assignment','frozen_model_state_sha256','closed_loop_motion_residual_fit_sha256',
        'frozen_layout_inventory_sha256')
    differences={label:[k for k in fields if launch.get(k)!=launches['forecast'].get(k)]
        for label,launch in launches.items()}
    if any(differences.values()):raise ValueError(f'non-treatment settings differ: {differences}')
    common=set.intersection(*(set(v) for v in sources.values()))
    changed=[p for p in common if len({v[p] for v in sources.values()})!=1]
    if changed:raise ValueError(f'common source changed: {changed}')
    report=dict(layout_index=index,conditions=summaries,rows=rows,behavior_metrics=metrics,
        actual_treatments=treatments,figure_title=figure_title,
        matched_settings={k:launches['forecast'].get(k) for k in fields},
        selection_treatment_settings={label:{k:launch.get(k) for k in sorted(TREATMENT_FIELDS)}
            for label,launch in launches.items()},
        common_sources={p:sources['forecast'][p] for p in sorted(common)},
        learned_rollout_values_unused_for_off_selection=True,
        network_still_computed_and_validated=True,total_computation_identical=False,
        geometric_view_and_actual_dispatch_projections_retained=True,
        repeated_exposed_development_layout=True,training_seeds=[2026091001],
        host_real_time_qualified=False,hardware_validated=False,
        statistical_superiority_established=False,final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(rows=rows,unchanged_common_source_count=len(common))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout-index',type=int,choices=(0,1),required=True)
    index=parser.parse_args().layout_index;arm=experiment.ARMS[0]
    roots=dict(forecast=path(experiment.reference.ROOT.format(index=index,arm=arm)),
        instantaneous_with_predictive_checks=path(f'go2_instantaneous_waypoint_score_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'),
        rollout_selection_off=path(experiment.ROOT.format(index=index,arm=arm)))
    compare(index,arm,roots,path(f'go2_rollout_selection_off_comparison_layout{index:02d}_v1_attempt_001'),
        f'Learned rollout selection: exposed maze {index}')


if __name__=='__main__':main()
