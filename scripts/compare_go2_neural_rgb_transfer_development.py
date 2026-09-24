"""Compare a completed full/no-RGB pair, retaining failures and actual treatments."""
import argparse
import json
from lewm.seeded_motion_correction_development import registry, registry_identity
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.compare_go2_shared_recovery_transfer_development import SHARED
from scripts.compare_go2_combined_perception_motion_development import behavior
from scripts import run_go2_neural_rgb_transfer_development as study


def compare(index, seed, method):
    output = path(f'go2_neural_rgb_transfer_comparison_seed_{seed}_{method}_layout{index:02d}_v1_attempt_001')
    if output.exists():
        raise ValueError('preserve completed comparison')
    summaries = {}; launches = {}; sources = {}; treatments = {}; metrics = {}; rows = []
    for variant in ('full', 'no_rgb'):
        arm = f'seed_{seed}_{variant}_{method}'
        root = path(study.ROOT.format(index=index, arm=arm))
        entry = registry(variant)[arm]
        launch = read(root, 'launch.json')
        treatment = read(root, 'actual_controller_treatment_v1.json')
        rgb = read(root, 'actual_neural_rgb_treatment_v1.json')
        summary = read(root, 'live_navigation_summary_v1.json')
        plans = [p for p in read(root, 'planning.json') if 'selection' in p]
        if (launch['study_arm'] != arm or launch['model_assignment'] != arm
                or launch['layout_index'] != index or launch['model_input_variant'] != variant
                or launch['frozen_layout_inventory_sha256'] != study.INVENTORY_SHA256
                or launch['frozen_model_registry_sha256'] != registry_identity(variant)[1]
                or launch['frozen_model_state_sha256'] != entry['model_state_sha256']
                or launch['closed_loop_motion_residual_fit_sha256'] != entry['fit_sha256']
                or launch['planned_native_assignments'] != len(study.ASSIGNMENTS)
                or rgb['input_variant'] != variant
                or rgb['selected_plans'] != len(plans) or treatment['selected_plans'] != len(plans)):
            raise ValueError('fixed assignment or evaluated population differs')
        if plans and (not treatment['actual_treatment_verified']
                or not treatment['model_and_correction_binding_verified']
                or not rgb['model_rgb_treatment_verified']):
            raise ValueError('actual model/input treatment not verified')
        if variant == 'no_rgb' and rgb['plans_with_nonzero_model_rgb']:
            raise ValueError('no-RGB received images')
        summaries[variant] = summary; launches[variant] = launch
        sources[variant] = launch['source_sha256'] | launch['extra_sources']
        treatments[variant] = dict(controller=treatment, neural_rgb=rgb)
        metrics[variant] = behavior(root, plans)
        evaluation = summary['independent_arrival_evaluation']; result = summary['result']
        rows.append(dict(condition=variant, root_name=root.name,
            round_trip=evaluation['round_trip_arrival_checks_passed'],
            contacts=evaluation['disallowed_contact_samples'], failure=summary['failure'],
            mission_terminal=evaluation['mission_terminal'],
            maximum_pose_error_m=evaluation['maximum_position_error_m'],
            simulation_s=None if result is None else result['simulation_s'],
            native_path_length_m=summary['native_10hz_horizontal_path_length_m'],
            selected_plans=len(plans), on_time_plans=sum(p['on_time'] for p in plans)))
    fields = SHARED + ('training_seed', 'training_condition', 'prediction_head',
        'neural_rgb_input_checked_at_every_forward', 'camera_based_tracking_and_mapping_retained',
        'frozen_layout_inventory_sha256')
    changed = [k for k in fields if launches['full'].get(k) != launches['no_rgb'].get(k)]
    if changed:
        raise ValueError(f'non-treatment settings differ: {changed}')
    common = set(sources['full']) & set(sources['no_rgb'])
    changed = [p for p in common if sources['full'][p] != sources['no_rgb'][p]]
    if changed:
        raise ValueError(f'common runtime sources differ: {changed}')
    report = dict(layout_index=index, training_seed=seed, training_method=method,
        conditions=summaries, rows=rows, actual_treatments=treatments, behavior_metrics=metrics,
        figure_title=f'Neural RGB: {method}, seed {seed}, fresh maze {index}',
        matched_settings={k:launches['full'].get(k) for k in fields},
        common_sources={p:sources['full'][p] for p in sorted(common)},
        camera_based_tracking_and_mapping_retained=True, inference_only_ablation=False,
        statistical_superiority_established=False, hardware_validated=False, final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(dict(rows=rows, unchanged_common_source_count=len(common))), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=(0,1), required=True)
    parser.add_argument('--seed', type=int, choices=study.SEEDS, required=True)
    parser.add_argument('--method', choices=study.METHODS, required=True)
    args = parser.parse_args()
    compare(args.layout_index, args.seed, args.method)
