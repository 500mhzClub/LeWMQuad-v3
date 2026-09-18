"""Summarize all fifteen fixed missions, including failures and matched forecasts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
OUT = BASE/'go2_publication_repaired_replication_readout_v1_attempt_001'
ARMS = ('jepa', 'supervised_rollout', 'pose_command', 'instantaneous', 'reactive_feedback')
LABELS = ('JEPA', 'Supervised rollout', 'Pose-command', 'Instantaneous ranking', 'Reactive feedback')


def read(path):
    return json.loads(path.read_text())


def main():
    aggregate = read(OUT/'result.json')
    assert aggregate['status'] == 'COMPLETE' and aggregate['evaluated_assignments'] == 15
    rows = []
    for row in aggregate['rows']:
        root = BASE/row['root']
        xy = read(root/'saved_short_pulse_same_window_xy_v1.json')
        yaw = read(root/'saved_short_pulse_yaw_evaluation_v1.json')
        actual = read(root/'actual_controller_treatment_v1.json')
        timing = read(root/'planning_latency_stress_diagnosis_v1.json')
        physical = read(root/'continuous_native_arrival_evaluation.json')
        assert actual['actual_treatment_verified']
        assert xy['windows'] == yaw['windows']
        assert xy['matched_horizon_ms'] == yaw['matched_horizon_ms'] == 700
        backtracking = (read(root/'physical_return_corridor_readout_v1.json')
                        if (root/'physical_return_corridor_readout_v1.json').exists() else None)
        if row['round_trip']:
            assert physical['round_trip_arrival_checks_passed']
            assert backtracking['physical_backtracking_observed']
            assert backtracking['invalid_graph_transitions'] == 0
        rows.append(row | dict(
            actual_controller_treatment_verified=True,
            xy_windows=xy['windows'], xy_endpoint_rmse_mm=xy['rmse_mm'],
            xy_by_action_group=xy['by_action_group'],
            yaw_endpoint_rmse_deg=yaw['rmse_deg'],
            pose_error_m={k: physical[k] for k in ('median_position_error_m', 'maximum_position_error_m')},
            delay_crossed_deadlines=timing['deadlines_crossed_by_added_wait'],
            actual_extra_min_ms=timing['actual_extra_min_ms'],
            actual_extra_max_ms=timing['actual_extra_max_ms'],
            recovery_publications=len(read(root/'visual_dispatch_events.json')),
            backtracking=None if backtracking is None else {
                k: backtracking[k] for k in ('physical_backtracking_observed',
                    'outbound_unique_directed_edges', 'return_unique_directed_edges',
                    'return_edges_reversing_observed_outbound_edges', 'invalid_graph_transitions')},
            depth_retired=(root/'DEPTH_RETIRED').exists()))
    total_plans = sum(r['plans'] for r in rows)
    on_time = sum(r['plans_on_time'] for r in rows)
    result = dict(schema='publication_repaired_navigation_complete_scientific_readout.v1',
        status='COMPLETE', assignments=15, layouts=3, training_seeds=1,
        executions_per_arm_layout=1, round_trips=sum(r['round_trip'] for r in rows),
        goal_arrivals=sum(r['arrivals'] > 0 for r in rows),
        contacts=sum(r['contacts'] for r in rows),
        tracking_failures=sum(r['failure'] is not None for r in rows),
        budget_failures=sum(not r['round_trip'] and r['failure'] is None for r in rows),
        plans=total_plans, on_time_plans=on_time, on_time_fraction=on_time/total_plans,
        deadlines_crossed_by_extra_wait=sum(r['delay_crossed_deadlines'] for r in rows),
        fitted_xy_better_than_neural_recordings=sum(
            r['xy_endpoint_rmse_mm']['pose_command'] < r['xy_endpoint_rmse_mm']['neural'] for r in rows),
        history_yaw_better_than_neural_recordings=sum(
            r['yaw_endpoint_rmse_deg']['command_history'] < r['yaw_endpoint_rmse_deg']['neural'] for r in rows),
        per_arm=aggregate['per_arm'], rows=rows,
        matched_forecast_horizon_ms=700, forecast_windows_overlap=True,
        forecast_comparisons_conditioned_on_executed_trajectory=True,
        unexecuted_candidate_outcomes_evaluated=False,
        reactive_neural_reference='supervised model, computed but unused for action selection',
        pose_command_neural_reference='supervised model, computed but unused for action selection',
        instantaneous_predictive_guards_retained=True,
        all_arms_have_persistent_memory=True, memory_causal_advantage_isolated=False,
        jepa_superiority_established=False, reliable_transfer_established=False,
        ideal_gyro=True, synthetic_depth_noise_mm=2, timing='measured simulation',
        hardware_validated=False, final_evaluation=False, predecessor_cohorts_pooled=False)
    (OUT/'complete_scientific_readout_v1.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.set_xlim(-.5, 2.5); ax.set_ylim(4.5, -.5)
    for i, arm in enumerate(ARMS):
        for j, layout in enumerate((1, 2, 3)):
            r = next(r for r in rows if r['arm'] == arm and r['layout_index'] == layout)
            if r['round_trip']:
                color, title = '#d8eadf', f"Round trip · {r['simulation_s']:.2f} s"
            elif r['failure']:
                color, title = '#f5dddd', 'Tracking failure'
            else:
                color = '#f5e7cb'
                title = 'Goal only · budget' if r['arrivals'] else 'No goal · budget'
            ax.add_patch(plt.Rectangle((j-.48, i-.45), .96, .90, color=color))
            ax.text(j, i-.10, title, ha='center', va='center', fontsize=10)
            ax.text(j, i+.17, f"{r['plans_on_time']}/{r['plans']} plans on time", ha='center', fontsize=9)
    ax.set_xticks(range(3), ['Maze 1', 'Maze 2', 'Maze 3'])
    ax.set_yticks(range(5), LABELS); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Fixed corrected-runtime comparison: 10/15 round trips, zero contacts', pad=18)
    fig.text(.5, .025, 'Three new same-family mazes · one training seed · one execution per cell\n'
             'Times are simulated seconds. Instantaneous ranking retains predictive guards.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for extension in ('png', 'svg'):
        fig.savefig(OUT/f'complete_outcomes_v1.{extension}', dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'per_arm')}, indent=2))
    print('FORECASTS: assignment arm layout neuralXY fittedXY neuralYaw historyYaw')
    for r in rows:
        print(r['assignment'], r['arm'], r['layout_index'],
              *(round(r['xy_endpoint_rmse_mm'][k], 3) for k in ('neural', 'pose_command')),
              *(round(r['yaw_endpoint_rmse_deg'][k], 3) for k in ('neural', 'command_history')))


if __name__ == '__main__':
    main()
