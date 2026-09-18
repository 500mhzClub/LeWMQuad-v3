"""Compare native outcomes with forecast versus instantaneous main utilities."""
import argparse
import json
from scripts.compare_continuous_navigation_arms_development import path,read
from scripts.compare_go2_shared_recovery_transfer_development import SHARED
from scripts.compare_go2_combined_perception_motion_development import behavior
from scripts import run_go2_instantaneous_waypoint_score_development as experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=(0,1),required=True)
    index=parser.parse_args().layout_index;arm=experiment.ARMS[0]
    output=path(f'go2_instantaneous_waypoint_score_comparison_layout{index:02d}_v1_attempt_001')
    if output.exists():raise ValueError('preserve comparison')
    roots=dict(forecast=path(experiment.reference.ROOT.format(index=index,arm=arm)),
        instantaneous=path(experiment.ROOT.format(index=index,arm=arm)))
    summaries={};launches={};sources={};rows=[];metrics={}
    for label,root in roots.items():
        launch=read(root,'launch.json');summary=read(root,'live_navigation_summary_v1.json')
        treatment=read(root,'actual_controller_treatment_v1.json')
        plans=[p for p in read(root,'planning.json') if 'selection' in p]
        if (launch['study_arm']!=arm or launch['layout_index']!=index
                or not treatment['actual_treatment_verified']
                or not treatment['model_and_correction_binding_verified']):
            raise ValueError('controller assignment differs')
        if label=='instantaneous' and not read(root,'instantaneous_waypoint_score_treatment_v1.json')['actual_instantaneous_main_utilities_verified']:
            raise ValueError('instantaneous ranking not verified')
        summaries[label]=summary;launches[label]=launch
        sources[label]=launch['source_sha256']|launch['extra_sources'];metrics[label]=behavior(root,plans)
        e=summary['independent_arrival_evaluation'];result=summary['result']
        rows.append(dict(condition=label,root_name=root.name,
            round_trip=e['round_trip_arrival_checks_passed'],contacts=e['disallowed_contact_samples'],
            failure=summary['failure'],maximum_pose_error_m=e['maximum_position_error_m'],
            simulation_s=None if result is None else result['simulation_s'],
            selected_plans=len(plans),on_time_plans=sum(p['on_time'] for p in plans),
            native_path_length_m=summary['native_10hz_horizontal_path_length_m']))
    fields=tuple(k for k in SHARED if k!='planned_native_assignments')+(
        'model_assignment','frozen_model_state_sha256','closed_loop_motion_residual_fit_sha256',
        'frozen_layout_inventory_sha256')
    differences=[k for k in fields if launches['forecast'].get(k)!=launches['instantaneous'].get(k)]
    if differences:raise ValueError(f'shared settings differ: {differences}')
    common=sources['forecast'].keys()&sources['instantaneous'].keys()
    changed=[p for p in common if sources['forecast'][p]!=sources['instantaneous'][p]]
    if changed:raise ValueError(f'common source changed: {changed}')
    report=dict(layout_index=index,conditions=summaries,rows=rows,behavior_metrics=metrics,
        figure_title=f'Forecast versus instantaneous main scores: exposed maze {index}',
        instantaneous_treatment=read(roots['instantaneous'],'instantaneous_waypoint_score_treatment_v1.json'),
        matched_settings={k:launches['forecast'].get(k) for k in fields},
        common_sources={p:sources['forecast'][p] for p in sorted(common)},
        predictive_clearance_recovery_arrival_and_stopping_retained=True,
        repeated_exposed_development_layout=True,full_online_rollout_ablation=False,
        training_seeds=[2026091001],host_real_time_qualified=False,hardware_validated=False,
        statistical_superiority_established=False,final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(rows=rows,unchanged_common_source_count=len(common))),flush=True)


if __name__=='__main__':main()
