"""Describe the complete fixed 22-mission cohort, preserving all failures."""
import json
from scripts import run_go2_multiseed_navigation_development as study
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_stopping_projection_transfer_development import FIELDS
from scripts.compare_go2_combined_perception_motion_development import behavior


def main():
    output = path('go2_multiseed_navigation_complete_comparison_v1_attempt_001')
    if output.exists(): raise ValueError('preserve completed comparison')
    pending = [(i,a) for i in range(2) for a in study.ARMS
        if not (path(study.ROOT.format(index=i,arm=a))/'live_navigation_summary_v1.json').exists()]
    if pending: raise ValueError(f'complete and evaluate every fixed assignment first: {pending}')
    layouts = {}; rows = []
    for i in range(2):
        summaries = {}; sources = {}; launches = {}; metrics = {}
        for arm in study.ARMS:
            root = path(study.ROOT.format(index=i,arm=arm))
            launch = read(root,'launch.json'); treatment = read(root,'actual_controller_treatment_v1.json')
            if (launch['study_arm'] != arm or launch['layout_index'] != i
                    or launch['model_assignment'] != study.assignment_for(arm)
                    or launch['frozen_model_registry_sha256'] != study.REGISTRY_SHA256
                    or launch['frozen_layout_inventory_sha256'] != study.INVENTORY_SHA256
                    or treatment['study_arm'] != arm):
                raise ValueError('recorded assignment differs from the fixed cohort')
            summary = read(root,'live_navigation_summary_v1.json')
            plans = [p for p in read(root,'planning.json') if 'selection' in p]
            if len(plans) != treatment['selected_plans']: raise ValueError('evaluated plan population differs')
            if plans and (not treatment['actual_treatment_verified'] or
                    arm != 'reactive' and not treatment['model_and_correction_binding_verified']):
                raise ValueError('actual controller treatment was not verified')
            mission = read(root,'mission.json')
            metrics[arm] = behavior(root,plans) | dict(
                rejected_floor_observations=sum('floor_rejection' in m for m in mission),
                floor_reacquisition_hold_frames=sum(m.get('floor_reacquisition_hold',False) for m in mission))
            summaries[arm] = summary; launches[arm] = launch
            sources[arm] = launch['source_sha256'] | launch['extra_sources']
            evaluation = summary['independent_arrival_evaluation']
            result = summary['result']
            rows.append(dict(layout_index=i,arm=arm,training_seed=launch['training_seed'],
                method=launch['training_condition'] if arm not in ('pose_command','reactive') else arm,
                goal=any(a['phase']=='OUTBOUND' and a['arrival_checks_passed'] for a in evaluation['arrivals']),
                round_trip=evaluation['round_trip_arrival_checks_passed'],
                contacts=evaluation['disallowed_contact_samples'],failure=summary['failure'],
                mission_terminal=evaluation['mission_terminal'],
                simulation_s=None if result is None else result['simulation_s'],
                native_10hz_horizontal_path_length_m=summary['native_10hz_horizontal_path_length_m']))
        reference = launches[study.ARMS[0]]
        differences = {a:[k for k in FIELDS if v.get(k)!=reference.get(k)] for a,v in launches.items()}
        if any(differences.values()): raise ValueError(f'shared settings differ: {differences}')
        common = set.intersection(*(set(v) for v in sources.values()))
        changed = [p for p in sorted(common) if len({v[p] for v in sources.values()})!=1]
        if changed: raise ValueError(f'common runtime source changed: {changed}')
        layouts[str(i)] = dict(conditions=summaries,behavior_metrics=metrics,
            matched_settings={k:reference.get(k) for k in FIELDS},
            common_sources={p:sources[study.ARMS[0]][p] for p in sorted(common)})
    totals = {}
    for method in (*study.METHODS,'pose_command','reactive'):
        selected = [r for r in rows if r['method']==method]
        totals[method] = dict(assignments=len(selected),goals=sum(r['goal'] for r in selected),
            round_trips=sum(r['round_trip'] for r in selected),contacts=sum(r['contacts'] for r in selected))
    report = dict(layouts=layouts,rows=rows,totals=totals,completed_assignments=22,
        independent_development_layouts=2,training_seeds=list(study.SEEDS),
        seed_repetitions_are_not_additional_independent_mazes=True,
        condition_specific_fixed_motion_corrections=True,
        reactive_predictive_clearance_and_recovery_rules_differ=True,
        isolated_prediction_ranking_effect_established=False,
        statistical_superiority_established=False,host_real_time_qualified=False,
        calibrated_hardware_sensing=False,hardware_validated=False,final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(totals),flush=True)


if __name__ == '__main__': main()
