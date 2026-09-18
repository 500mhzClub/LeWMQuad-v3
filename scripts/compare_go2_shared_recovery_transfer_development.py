"""Compare all five fixed controller outcomes on one new development maze."""
import argparse
import json
from statistics import median
from scripts import run_go2_shared_recovery_transfer_development as study
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_stopping_projection_transfer_development import FIELDS
from scripts.compare_go2_combined_perception_motion_development import behavior

SHARED = FIELDS + ('tracker', 'registration', 'independent_obstacle_observer',
    'shared_translation_veto_view_direction', 'view_recovery_angle_degrees',
    'view_recovery_completion_tolerance_rad')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    index = parser.parse_args().layout_index
    output = path(f'go2_shared_recovery_transfer_comparison_layout{index:02d}_v1_attempt_001')
    if output.exists():
        raise ValueError('preserve completed comparison')
    summaries = {}; launches = {}; sources = {}; treatments = {}; metrics = {}; rows = []
    timing = {}; forecasts = {}
    for arm in study.ARMS:
        root = path(study.ROOT.format(index=index, arm=arm))
        launch = read(root, 'launch.json')
        summary = read(root, 'live_navigation_summary_v1.json')
        treatment = read(root, 'actual_controller_treatment_v1.json')
        if (launch['study_arm'] != arm or launch['layout_index'] != index
                or launch['model_assignment'] != study.study.assignment_for(arm)
                or launch['frozen_layout_inventory_sha256'] != study.INVENTORY_SHA256
                or launch['frozen_model_registry_sha256'] != study.study.REGISTRY_SHA256
                or treatment['study_arm'] != arm):
            raise ValueError('recorded assignment differs from fixed comparison')
        plans = [p for p in read(root, 'planning.json') if 'selection' in p]
        if len(plans) != treatment['selected_plans']:
            raise ValueError('evaluated plan population differs')
        if plans and (not treatment['actual_treatment_verified'] or
                arm != 'reactive' and not treatment['model_and_correction_binding_verified']):
            raise ValueError('actual controller treatment not verified')
        label = arm.split('_full_')[-1]
        summaries[label] = summary; launches[label] = launch; treatments[label] = treatment
        sources[label] = launch['source_sha256'] | launch['extra_sources']
        metrics[label] = behavior(root, plans)
        events = read(root, 'stage_events.json')
        timing[label] = {}
        for stage in sorted({e['stage'] for e in events}):
            selected = [e for e in events if e['stage']==stage]
            timing[label][stage] = dict(count=len(selected),
                median_recorded_service_ms=median((e['completed_ns']-e['started_ns'])/1e6 for e in selected),
                maximum_recorded_service_ms=max((e['completed_ns']-e['started_ns'])/1e6 for e in selected),
                maximum_recorded_completion_age_ms=max((e['completed_ns']-e['measured_ns'])/1e6 for e in selected))
        if arm != 'reactive':
            forecasts[label] = {kind:{k:v for k,v in read(root, filename).items() if k!='rows'}
                for kind,filename in (
                    ('xy','saved_executed_motion_forecast_evaluation_v1.json'),
                    ('yaw','saved_neural_command_yaw_evaluation_v1.json'))}
        evaluation = summary['independent_arrival_evaluation']; result = summary['result']
        rows.append(dict(condition=label, arm=arm, root_name=root.name,
            goal=any(a['phase']=='OUTBOUND' and a['arrival_checks_passed'] for a in evaluation['arrivals']),
            round_trip=evaluation['round_trip_arrival_checks_passed'],
            contacts=evaluation['disallowed_contact_samples'], failure=summary['failure'],
            maximum_pose_error_m=evaluation['maximum_position_error_m'],
            simulation_s=None if result is None else result['simulation_s'],
            selected_plans=len(plans), on_time_plans=sum(p['on_time'] for p in plans),
            native_path_length_m=summary['native_10hz_horizontal_path_length_m']))
    reference = launches['reactive']
    differences = {a:[k for k in SHARED if v.get(k)!=reference.get(k)] for a,v in launches.items()}
    if any(differences.values()):
        raise ValueError(f'shared settings differ: {differences}')
    common = set.intersection(*(set(v) for v in sources.values()))
    changed = [p for p in sorted(common) if len({v[p] for v in sources.values()}) != 1]
    if changed:
        raise ValueError(f'common runtime source changed: {changed}')
    report = dict(layout_index=index, conditions=summaries, rows=rows,
        figure_title=f'Shared recovery: five controllers — new development maze {index}',
        actual_treatments=treatments, behavior_metrics=metrics,
        recorded_stage_timing=timing, executed_forecast_metrics=forecasts,
        timing_clock_scope='recorded measured-simulation timestamps, not hard wall-clock qualification',
        forecast_scope='matched executed windows; overlapping and not independently sampled; no counterfactual navigation claim',
        matched_settings={k:reference.get(k) for k in SHARED},
        common_sources={p:sources['reactive'][p] for p in sorted(common)},
        completed_assignments=len(rows), training_seeds=[2026091001],
        fresh_development_layout=True, prior_registry_layout_count=78, same_maze_family=True,
        reactive_predictive_clearance_and_other_recovery_rules_differ=True,
        shared_actual_translation_veto_recovery=True,
        isolated_prediction_ranking_effect_established=False,
        statistical_superiority_established=False, host_real_time_qualified=False,
        calibrated_hardware_sensing=False, hardware_validated=False, final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(rows=rows, unchanged_common_source_count=len(common))), flush=True)


if __name__ == '__main__': main()
