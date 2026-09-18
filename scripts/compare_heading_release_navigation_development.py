"""Compare the fixed recovery-release treatment with its original native run."""
import argparse
from collections import Counter
import json

from scripts.compare_continuous_navigation_arms_development import path, read, summarize


def mechanism(root):
    plans = [p for p in read(root, 'planning.json') if 'selection' in p]
    releases = [p for p in plans
        if p['selection'].get('full_reserve_heading_release', {}).get('applied')]
    counts = Counter(p['action'] for p in plans)
    return dict(selected_actions=dict(counts),
        pure_turn_fraction=(counts['left_turn']+counts['right_turn'])/len(plans),
        release_count=len(releases), release_frames=[p['frame'] for p in releases],
        release_plans_on_time=sum(p['on_time'] for p in releases),
        selected_actions_are_not_full_executed_physics_intervals=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    args = parser.parse_args(); i = args.layout_index
    baseline = path(f'go2_fresh_stable_reference_learned_round_trip_native_layout{i:02d}_4800_v1_attempt_001')
    revised = path(f'go2_heading_release_fresh_learned_round_trip_native_layout{i:02d}_4800_v1_attempt_001')
    output = path(f'go2_heading_release_matched_comparison_layout{i:02d}_v1_attempt_001')
    if output.exists(): raise ValueError('preserve existing comparison')
    a, b = read(baseline, 'launch.json'), read(revised, 'launch.json')
    treatment_fields = {'owner', 'source_sha256', 'extra_sources',
        'fresh_fixed_controller_transfer', 'full_reserve_heading_release',
        'baseline_root_name', 'terminal_approach_excluded_from_heading_release'}
    differences = [k for k in set(a)|set(b)
        if k not in treatment_fields and a.get(k) != b.get(k)]
    if differences: raise ValueError(f'non-treatment launch settings differ: {differences}')
    if not (b['full_reserve_heading_release']
            and b['terminal_approach_excluded_from_heading_release']
            and b['baseline_root_name'] == baseline.name):
        raise ValueError('expected one-change recovery treatment required')
    sources_a = a['source_sha256']|a['extra_sources']
    sources_b = b['source_sha256']|b['extra_sources']
    changed = [k for k in sources_a if sources_a[k] != sources_b.get(k)]
    if changed: raise ValueError(f'original controller sources changed: {changed}')
    report = dict(layout_index=i, matched_non_treatment_settings=True,
        original_sources_unchanged=True, baseline=summarize(baseline),
        revised=summarize(revised), baseline_mechanism=mechanism(baseline),
        revised_mechanism=mechanism(revised), repeatability_established=False,
        jepa_specific_advantage_established=False,
        native_state_used_by_evaluator_only=True)
    output.mkdir()
    with (output/'result.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps({arm:dict(
        evaluation=report[arm]['independent_arrival_evaluation'],
        mechanism=report[arm+'_mechanism']) for arm in ('baseline', 'revised')}))


if __name__ == '__main__': main()
