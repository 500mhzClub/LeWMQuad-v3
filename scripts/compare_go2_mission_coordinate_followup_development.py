"""Compare each original/consistent pair, retaining failures and terminal costs."""
import argparse
import json

from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_shared_recovery_transfer_development import SHARED
from scripts.diagnose_navigation_terminal_approach_development import phases
from scripts import run_go2_mission_coordinate_followup_development as study


def compare(index):
    arm = next(arm for layout, arm, mode in study.ASSIGNMENTS if layout == index)
    output = path(f'go2_mission_coordinate_comparison_layout{index:02d}_v1_attempt_001')
    if output.exists():
        raise ValueError('preserve completed comparison')
    rows, summaries, launches, sources, treatments, terminal, decisions = [], {}, {}, {}, {}, {}, {}
    for mode in ('original', 'consistent'):
        root = path(study.ROOT.format(index=index, arm=arm, mode=mode))
        launch = read(root, 'launch.json')
        summary = read(root, 'live_navigation_summary_v1.json')
        treatment = read(root, 'actual_mission_coordinate_treatment_v1.json')
        neural = read(root, 'actual_neural_rgb_treatment_v1.json')
        controller = read(root, 'actual_controller_treatment_v1.json')
        plans = [p for p in read(root, 'planning.json') if 'selection' in p]
        if (launch['coordinate_mode'] != mode or launch['study_arm'] != arm
                or launch['layout_index'] != index or treatment['mode'] != mode
                or treatment['selected_plans'] != len(plans)
                or not controller['model_and_correction_binding_verified']
                or not neural['model_rgb_treatment_verified']):
            raise ValueError('evaluated coordinate/model treatment differs')
        summaries[mode] = summary; launches[mode] = launch
        sources[mode] = launch['source_sha256'] | launch['extra_sources']
        treatments[mode] = dict(coordinate=treatment, neural=neural, controller=controller)
        terminal[mode] = phases(root)
        decisions[mode] = dict(
            predicted_hold_eligible=sum((p['selection'].get('predictive_arrival_hold') or {}).get('eligible', False) for p in plans),
            predicted_hold_changed=sum((p['selection'].get('predictive_arrival_hold') or {}).get('changed', False) for p in plans),
            arrival_entry_restored_heading=sum((p['selection'].get('arrival_entry_terminal_priority') or {}).get('changed', False) for p in plans))
        evaluation = summary['independent_arrival_evaluation']; result = summary['result']
        rows.append(dict(condition=mode, root_name=root.name,
            round_trip=evaluation['round_trip_arrival_checks_passed'],
            contacts=evaluation['disallowed_contact_samples'], failure=summary['failure'],
            maximum_pose_error_m=evaluation['maximum_position_error_m'],
            simulation_s=None if result is None else result['simulation_s'],
            peak_simulator_lag_ms=None if result is None else result['max_simulator_lag_ms'],
            selected_plans=len(plans), on_time_plans=sum(p['on_time'] for p in plans),
            native_path_length_m=summary['native_10hz_horizontal_path_length_m']))
    fields = SHARED + ('model_assignment', 'model_input_variant', 'frozen_model_state_sha256',
        'frozen_model_registry_sha256', 'closed_loop_motion_residual_fit_sha256',
        'frozen_layout_inventory_sha256')
    changed = [k for k in fields if launches['original'].get(k) != launches['consistent'].get(k)]
    if changed or sources['original'] != sources['consistent']:
        raise ValueError(f'non-treatment settings or sources differ: {changed}')
    report = dict(layout_index=index, model_assignment=arm, rows=rows, conditions=summaries,
        actual_treatments=treatments, terminal_decisions=decisions,
        matched_settings={k:launches['original'].get(k) for k in fields},
        common_sources=sources['original'], targeted_exposed_development_cases=True,
        statistical_superiority_established=False, hardware_validated=False,
        figure_title=f'Terminal coordinate metric: exposed maze {index}')
    output.mkdir()
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2)
    with (output/'terminal_approach_diagnostic_v1.json').open('x') as f:
        json.dump(dict(conditions=terminal, recorded_trajectory_diagnostic_only=True,
            definition='First observed distance <= 0.10 m through observed arrival or phase end'), f, indent=2)
    print(json.dumps(dict(rows=rows, terminal=terminal, terminal_decisions=decisions,
        unchanged_common_source_count=len(sources['original']))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0, 1), required=True)
    compare(parser.parse_args().layout_index)
