"""Compare live reference refresh against the recorded original-tracker control."""
import argparse
from collections import Counter
import json

from scripts.compare_continuous_navigation_arms_development import BASE, read, summarize


def tracking_evidence(root):
    poses = read(root, 'poses.json')
    rows = [p['raw_pose'] for p in poses]
    refreshes = [r['frame'] for r in rows
        if r.get('promotion_reason') == 'accepted_anchor_recent_reference_age']
    return dict(accepted_registered_poses=len(poses),
        pose_modes=dict(Counter(r['mode'] for r in rows)),
        promotion_reasons=dict(Counter(r.get('promotion_reason') for r in rows
            if r['promoted_keyframe'])),
        accepted_anchor_refresh_frames=refreshes,
        global_history_resets=sum(bool(r['global_history_reset']) for r in rows),
        pipeline_faults=read(root, 'pipeline_faults.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    i = parser.parse_args().layout_index
    roots = dict(original_tracker=BASE/f'go2_heading_first_terminal_reactive_native_layout{i:02d}_4800_v1_attempt_001',
        recent_reference_refresh=BASE/f'go2_recent_reference_refresh_reactive_native_layout{i:02d}_4800_v1_attempt_001')
    a, b = [read(r, 'launch.json') for r in roots.values()]
    allowed = {'owner', 'experiment', 'comparison_condition', 'extra_sources',
        'tracker_control_root_name', 'recent_reference_refresh_from_accepted_anchor',
        'maximum_recent_reference_age_ns', 'bridge_measurements_promoted',
        'bridge_budget_and_measurement_acceptance_rules_unchanged'}
    changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if set(changed)-allowed or any(b['extra_sources'].get(k) != v for k,v in a['extra_sources'].items()):
        raise ValueError('original navigation settings or source changed')
    if (b['model_assignment'] != 'reactive' or not b['heading_first_terminal_control']
            or not b['recent_reference_refresh_from_accepted_anchor']
            or b['maximum_recent_reference_age_ns'] != 400_000_000
            or b['bridge_measurements_promoted']):
        raise ValueError('fixed tracking-refresh comparison required')
    report = dict(layout_index=i, comparison='accepted_anchor_reference_refresh_with_same_stronger_reactive_control',
        changed_launch_fields=changed, all_original_source_hashes_equal=True,
        conditions={label:summarize(r) for label,r in roots.items()},
        tracking_evidence={label:tracking_evidence(r) for label,r in roots.items()},
        original_control_reused=True, development_layout_revisit=True,
        timing_and_future_sensor_streams_may_diverge=True,
        hardware_validated=False, jepa_specific_advantage_established=False)
    output = BASE/f'go2_recent_reference_refresh_comparison_layout{i:02d}_v1_attempt_001'
    output.mkdir()
    with (output/'result.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(dict(output=str(output),
        refreshes=len(report['tracking_evidence']['recent_reference_refresh']['accepted_anchor_refresh_frames']))))


if __name__ == '__main__':
    main()
