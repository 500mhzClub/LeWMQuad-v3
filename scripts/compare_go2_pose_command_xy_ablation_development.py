"""Compare the two fixed XY-source interventions on one development revisit."""
import argparse
import json
import numpy as np
from lewm.matched_motion_residual_runtime_development import FITS
from lewm.pose_command_xy_control_development import FIT_SHA256
from scripts.compare_continuous_navigation_arms_development import path, read, summarize
from scripts.compare_go2_post_training_transfer_development import SHARED_FIELDS

SOURCES = ('learned', 'pose_command')
ASSIGNMENT = 'seed_2026091001_full_supervised_rollout'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    index = parser.parse_args().layout_index
    output = path(f'go2_pose_command_xy_ablation_supervised_rollout_comparison_layout{index:02d}_v1_attempt_001')
    if output.exists(): raise ValueError('preserve completed paired outcome')
    launches = {}; summaries = {}; bindings = {}; forecasts = {}; hashes = {}
    for source in SOURCES:
        root = path(f'go2_pose_command_xy_ablation_supervised_rollout_{source}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001')
        launch = read(root, 'launch.json')
        if (launch['experiment'] != 'pose_command_xy_source_ablation_development_v1' or
                launch['model_assignment'] != ASSIGNMENT or launch['forecast_xy_source'] != source or
                launch['layout_index'] != index or launch['pose_command_fit_sha256'] != FIT_SHA256 or
                launch['both_xy_alternatives_computed_in_both_arms'] is not True or
                launch['new_independent_development_layout'] is not False):
            raise ValueError('fixed reference model, XY source and revisit assignment required')
        plans = [p for p in read(root, 'planning.json') if 'selection' in p]
        for p in plans:
            c = p['motion_correction']; original = c['learned_motion_correction']
            if ((original['correction_root'], original['fit_sha256']) != FITS['supervised_rollout'] or
                    original['correction_base_model'] != ASSIGNMENT or
                    c['forecast_xy_source'] != source or c['pose_command_fit_sha256'] != FIT_SHA256 or
                    c['both_xy_alternatives_computed_in_both_arms'] is not True or
                    c['learned_yaw_and_contact_retained'] is not True):
                raise ValueError('actual runtime model or treatment binding differs')
            expected_hash = FITS['supervised_rollout'][1] if source == 'learned' else FIT_SHA256
            expected_base = ASSIGNMENT if source == 'learned' else 'pose_command_only'
            if c['fit_sha256'] != expected_hash or c['correction_base_model'] != expected_base:
                raise ValueError('applied fit does not match assigned XY source')
            expected = c['learned_corrected_forecast_xy_m' if source == 'learned' else 'pose_command_forecast_xy_m']
            actual = np.asarray(c['corrected_forecast_xy_m'])
            if actual.shape != (6,8,2) or not np.allclose(actual, expected, rtol=0, atol=1e-7):
                raise ValueError('planner XY input does not match the assigned alternative')
        bindings[source] = dict(selected_plans=len(plans), checked_actual_xy_source=True,
            recorded_alternative_match_absolute_tolerance_m=1e-7)
        summaries[source] = summarize(root)
        launches[source] = launch
        hashes[source] = launch['source_sha256'] | launch['extra_sources']
        forecasts[source] = {k:v for k,v in read(root, 'saved_executed_motion_forecast_evaluation_v1.json').items() if k != 'rows'}
    differences = [k for k in SHARED_FIELDS if launches['learned'].get(k) != launches['pose_command'].get(k)]
    if differences: raise ValueError(f'paired settings differ: {differences}')
    common = set(hashes['learned']) & set(hashes['pose_command'])
    if any(hashes['learned'][k] != hashes['pose_command'][k] for k in common):
        raise ValueError('common implementation changed between arms')
    report = dict(layout_index=index, conditions=summaries, actual_treatment_bindings=bindings,
        executed_forecast_metrics=forecasts,
        matched_settings={k:launches['learned'].get(k) for k in SHARED_FIELDS},
        common_sources={k:hashes['learned'][k] for k in sorted(common)},
        reference_training_condition='supervised_rollout',
        comparison='learned_vs_pose_command_XY_with_learned_yaw_contact_retained',
        development_layout_revisits=True, both_forecast_alternatives_computed=True,
        fully_model_free_comparison=False, statistical_advantage_established=False,
        hardware_validated=False, host_real_time_qualified=False, final_evaluation=False)
    output.mkdir()
    with (output/'result.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps({source:dict(round_trip=s['independent_arrival_evaluation']['round_trip_arrival_checks_passed'],
        disallowed_contacts=s['independent_arrival_evaluation']['disallowed_contact_samples'])
        for source,s in summaries.items()}))


if __name__ == '__main__': main()
