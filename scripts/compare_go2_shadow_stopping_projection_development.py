"""Compare native stopping-enforcement off with its retained on reference."""
import argparse
import json
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_shared_recovery_transfer_development import SHARED
from scripts.compare_go2_combined_perception_motion_development import behavior
from scripts import run_go2_shadow_stopping_projection_development as experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=(0,1),required=True)
    index = parser.parse_args().layout_index
    output = path(f'go2_shadow_stopping_projection_comparison_layout{index:02d}_v1_attempt_001')
    if output.exists(): raise ValueError('preserve completed comparison')
    arm = experiment.ARMS[0]
    roots = dict(enforced=path(experiment.reference.ROOT.format(index=index,arm=arm)),
        shadow_only=path(experiment.ROOT.format(index=index,arm=arm)))
    summaries = {}; launches = {}; sources = {}; rows = []; metrics = {}
    for label,root in roots.items():
        launch = read(root,'launch.json')
        summary = read(root,'live_navigation_summary_v1.json')
        treatment = read(root,'actual_controller_treatment_v1.json')
        plans = [p for p in read(root,'planning.json') if 'selection' in p]
        if (launch['study_arm']!=arm or launch['layout_index']!=index
                or not treatment['actual_treatment_verified']
                or not treatment['model_and_correction_binding_verified']
                or len(plans)!=treatment['selected_plans']):
            raise ValueError('recorded controller assignment differs')
        if label=='shadow_only':
            evidence = read(root,'shadow_stopping_intervention_evaluation_v1.json')
            if not evidence['all_stopping_interventions_unapplied']:
                raise ValueError('shadow treatment not verified')
        checks = [p['selection']['planned_stopping_projection'] for p in plans]
        summaries[label] = summary; launches[label] = launch
        sources[label] = launch['source_sha256'] | launch['extra_sources']
        metrics[label] = behavior(root,plans)
        evaluation = summary['independent_arrival_evaluation']; result = summary['result']
        rows.append(dict(condition=label,root_name=root.name,
            round_trip=evaluation['round_trip_arrival_checks_passed'],
            contacts=evaluation['disallowed_contact_samples'],failure=summary['failure'],
            maximum_pose_error_m=evaluation['maximum_position_error_m'],
            simulation_s=None if result is None else result['simulation_s'],
            selected_plans=len(plans),on_time_plans=sum(p['on_time'] for p in plans),
            applied_stopping_interventions=sum(c['changed'] for c in checks),
            proposed_stopping_interventions=sum(c.get('would_change_action',c['changed']) for c in checks),
            native_path_length_m=summary['native_10hz_horizontal_path_length_m']))
    fields = tuple(k for k in SHARED if k!='planned_native_assignments') + ('model_assignment','frozen_model_state_sha256',
        'closed_loop_motion_residual_fit_sha256','frozen_layout_inventory_sha256')
    differing = [k for k in fields if launches['enforced'].get(k)!=launches['shadow_only'].get(k)]
    if differing: raise ValueError(f'matched settings differ: {differing}')
    common = sources['enforced'].keys() & sources['shadow_only'].keys()
    changed = [p for p in common if sources['enforced'][p]!=sources['shadow_only'][p]]
    if changed: raise ValueError(f'common sources changed: {changed}')
    report = dict(layout_index=index,conditions=summaries,rows=rows,behavior_metrics=metrics,
        figure_title=f'Planned stopping enforcement on/off: exposed maze {index}',
        matched_settings={k:launches['enforced'].get(k) for k in fields},
        common_sources={p:sources['enforced'][p] for p in sorted(common)},
        repeated_exposed_development_layout=True,full_online_rollout_ablation=False,
        asynchronous_execution_can_differ=True,training_seeds=[2026091001],
        host_real_time_qualified=False,hardware_validated=False,
        statistical_superiority_established=False,final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(dict(rows=rows,unchanged_common_source_count=len(common))),flush=True)


if __name__=='__main__': main()
