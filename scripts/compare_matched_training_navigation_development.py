"""Summarize all three frozen training conditions on one completed native layout."""
import argparse
import json

from lewm.matched_motion_residual_runtime_development import FITS
from scripts.compare_continuous_navigation_arms_development import path, read, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--current-plane-noise', action='store_true')
    args = parser.parse_args(); i = args.layout_index
    if args.current_plane_noise:
        output_name = f'go2_current_plane_matched_training_comparison_layout{i:02d}_v1_attempt_001'
        root_template = 'go2_current_plane_matched_training_{condition}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
    else:
        output_name = f'go2_matched_training_comparison_layout{i:02d}_v1_attempt_001'
        root_template = 'go2_matched_training_{condition}_heading_release_native_layout{index:02d}_4800_v1_attempt_001'
    output = path(output_name)
    if output.exists(): raise ValueError('preserve previous training comparison')
    treatment_fields = {'owner', 'model_assignment', 'training_condition',
        'closed_loop_motion_residual_fit_sha256', 'motion_residual_correction_root'}
    if args.current_plane_noise:
        treatment_fields.add('comparison_condition')
    reference = None; summaries = {}; diagnostics = {}
    for condition, (correction_root, correction_hash) in FITS.items():
        root = path(root_template.format(condition=condition, index=i))
        launch = read(root, 'launch.json')
        assignment = f'seed_2026091001_full_{condition}'
        if (launch['model_assignment'] != assignment
                or launch['training_condition'] != condition
                or launch['closed_loop_motion_residual_fit_sha256'] != correction_hash
                or launch['motion_residual_correction_root'] != correction_root):
            raise ValueError('assigned training condition/correction differs')
        if args.current_plane_noise and (
                launch.get('comparison_condition') != condition or
                launch.get('experiment') != 'current_plane_matched_training_noise_development_v1'):
            raise ValueError('current-plane noisy comparison assignment required')
        common = {k:v for k,v in launch.items() if k not in treatment_fields}
        if reference is not None:
            changed = [k for k in set(reference)|set(common)
                if reference.get(k) != common.get(k)]
            if changed: raise ValueError(f'non-treatment launch fields or sources differ: {changed}')
        reference = common
        plans = [p for p in read(root, 'planning.json') if 'selection' in p]
        for p in plans:
            correction = p['motion_correction']
            if (correction['fit_sha256'] != correction_hash
                    or correction['correction_root'] != correction_root
                    or correction['correction_base_model'] != assignment):
                raise ValueError('runtime correction binding differs from assignment')
        summaries[condition] = summarize(root)
        diagnostics[condition] = dict(plans=len(plans),
            release_count=sum(p['selection'].get('full_reserve_heading_release', {}).get('applied', False)
                for p in plans),
            pure_turn_plans=sum(p['action'] in ('left_turn', 'right_turn') for p in plans),
            runtime_correction_matches_assignment=True)
    report = dict(layout_index=i, conditions=summaries, mechanism=diagnostics,
        common_settings_and_sources_equal=True, frozen_model_seed=2026091001,
        comparison='training_methods_with_condition_matched_frozen_visual_motion_corrections',
        native_state_evaluator_only=True, repeatability_established=False,
        statistical_training_method_advantage_established=False,
        memory_causal_contribution_established=False, host_real_time_qualified=False)
    output.mkdir()
    with (output/'result.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps({condition:s['independent_arrival_evaluation']
        for condition,s in summaries.items()}))


if __name__ == '__main__': main()
