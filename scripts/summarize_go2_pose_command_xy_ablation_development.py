"""Collect all eight fixed XY-source outcomes and same-window forecast errors."""
import json
import numpy as np

from lewm.geometry_progress_pilot_development import ACTIONS
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_pose_command_xy_ablation_development import SOURCES


def recorded_errors(root):
    plans = {p['frame']:p for p in read(root, 'planning.json') if 'selection' in p}
    rows = []
    for row in read(root, 'saved_executed_motion_forecast_evaluation_v1.json')['rows']:
        plan = plans[row['frame']]
        correction = plan['motion_correction']; index = ACTIONS.index(plan['action'])
        actual = np.asarray(row['actual_endpoint_xy_m'])
        errors = {source:float(np.linalg.norm(np.asarray(correction[key])[index,6]-actual))
            for source,key in (('learned','learned_corrected_forecast_xy_m'),
                ('pose_command','pose_command_forecast_xy_m'))}
        if abs(errors[correction['forecast_xy_source']]-row['corrected_endpoint_error_m']) > 1e-7:
            raise ValueError('recorded alternative differs from evaluated applied XY')
        rows.append(dict(frame=row['frame'], group=row['group'], errors_m=errors))
    return rows


def metrics(rows):
    return dict(windows=len(rows), endpoint_xy_rmse_m={source:
        float(np.sqrt(np.mean([r['errors_m'][source]**2 for r in rows]))) if rows else None
        for source in SOURCES})


def main():
    output = path('go2_pose_command_xy_ablation_supervised_rollout_four_layout_summary_v1_attempt_001')
    if output.exists(): raise ValueError('preserve completed eight-assignment summary')
    layouts = []; common_sources = None; pooled = {source:[] for source in SOURCES}
    for index in range(4):
        comparison_root = path(f'go2_pose_command_xy_ablation_supervised_rollout_comparison_layout{index:02d}_v1_attempt_001')
        comparison = read(comparison_root, 'result.json')
        if (comparison['layout_index'] != index or set(comparison['conditions']) != set(SOURCES)
                or comparison['reference_training_condition'] != 'supervised_rollout'):
            raise ValueError('complete fixed paired development outcome required')
        if common_sources is None: common_sources = comparison['common_sources']
        if comparison['common_sources'] != common_sources:
            raise ValueError('common runtime implementation changed across layouts')
        conditions = {}
        for source in SOURCES:
            summary = comparison['conditions'][source]
            evaluation = summary['independent_arrival_evaluation']
            rows = recorded_errors(path(summary['root_name'])); pooled[source].extend(rows)
            conditions[source] = dict(navigation_summary=summary,
                goal=any(a['phase']=='OUTBOUND' and a['arrival_checks_passed'] for a in evaluation['arrivals']),
                round_trip=evaluation['round_trip_arrival_checks_passed'],
                disallowed_contacts=evaluation['disallowed_contact_samples'],
                same_window_alternatives=metrics(rows),
                actual_treatment_binding=comparison['actual_treatment_bindings'][source])
        layouts.append(dict(layout_index=index, conditions=conditions))
    totals = {source:dict(assignments=4,
        goals=sum(r['conditions'][source]['goal'] for r in layouts),
        round_trips=sum(r['conditions'][source]['round_trip'] for r in layouts),
        disallowed_contacts=sum(r['conditions'][source]['disallowed_contacts'] for r in layouts))
        for source in SOURCES}
    paired = {}
    for outcome in ('goal','round_trip'):
        outcomes = [(r['conditions']['learned'][outcome],r['conditions']['pose_command'][outcome]) for r in layouts]
        paired[outcome] = dict(learned_only=sum(a and not b for a,b in outcomes),
            pose_command_only=sum(b and not a for a,b in outcomes),
            both=sum(a and b for a,b in outcomes), neither=sum(not a and not b for a,b in outcomes))
    report = dict(layouts=layouts, totals=totals, paired_outcomes=paired,
        same_window_alternatives_by_executed_controller={s:metrics(pooled[s]) for s in SOURCES},
        fixed_assignments=8, common_sources=common_sources, reference_training_condition='supervised_rollout',
        comparison='learned_vs_pose_command_XY_with_learned_yaw_contact_retained',
        all_assigned_outcomes_retained=True, development_layout_revisits=True,
        reference_selected_using_previous_development_outcomes=True,
        both_forecast_alternatives_computed=True, fully_model_free_comparison=False,
        forecast_errors_conditional_on_executed_controller=True, forecast_windows_overlap=True,
        unexecuted_action_outcomes_compared=False, statistical_advantage_established=False,
        independent_training_seeds=False, host_real_time_qualified=False,
        real_sensor_uncertainty_calibrated=False, hardware_validated=False, final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(dict(totals=totals, paired_outcomes=paired,
        same_window_alternatives_by_executed_controller=report['same_window_alternatives_by_executed_controller'])))


if __name__ == '__main__': main()
