"""Bound saved numerical comparison, not a calibration or permission gate."""
import argparse
import json

import numpy as np

from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.audit_go2_fresh_fused_maze_development_v1 import exact
from scripts.joint_rgbd_pose_plane_comparison_development import compare
from scripts.probe_go2_joint_rgbd_pose_plane_development_v1 import OUTPUT, INPUT, SOURCES, STEPS, preflight
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json

OWN = ('scripts/analyze_go2_joint_rgbd_pose_plane_development_v1.py',
       'scripts/joint_rgbd_pose_plane_comparison_development.py',
       'lewm/tests/test_joint_rgbd_pose_plane_comparison_development.py',
       'lewm/tests/test_paired_rgbd_physical_plane_geometry_development.py')


def analyze(identities):
    if set(identities) != {'launch.json', 'result.json'}:
        raise ValueError('explicit completed diagnostic identities required')
    bound = {str((OUTPUT/n).relative_to(ROOT)): h for n, h in identities.items()}
    own = {n: digest(ROOT/n) for n in OWN}; verify_bindings(bound | own)
    launch, result = [read_json(OUTPUT, n) for n in ('launch.json', 'result.json')]
    verify(launch); exact(preflight(), launch)
    assert result['status'] == 'TWO_STEP_JOINT_RGBD_POSE_PLANE_DIAGNOSTIC_COMPLETE'
    assert set(result['artifact_sha256']) == {'step_00.json', 'step_01.json'}
    bound |= {str((OUTPUT/n).relative_to(ROOT)): h for n, h in result['artifact_sha256'].items()}
    verify_bindings(bound)
    original = read_json(INPUT/'mission', 'task_decisions.json')
    steps = [read_json(OUTPUT, f'step_{i:02d}.json') for i in range(2)]
    for value, step in zip(steps, STEPS, strict=True):
        assert value['step'] == step and value['exact_nominal_frames'] == len(original) == 219
        assert value['member_models'] == 1+2*len(SOURCES)
        assert len(value['observations']) == len(original)
        for i, (row, expected) in enumerate(zip(value['observations'], original, strict=True)):
            exact(row['nominal_fusion'], expected['controller']['sensor_fusion'])
            assert row['source_ids'] == list(SOURCES) and row['difference_step'] == step
            assert row['nominal_fusion']['measured_ns'] == 1_500_000_000+i*100_000_000
            factor = np.asarray(row['pose_error_factor']); covariance = np.asarray(row['conditional_pose_covariance'])
            assert factor.shape == (6, len(SOURCES)) and np.isfinite(factor).all()
            np.testing.assert_allclose(covariance, factor@factor.T, rtol=1e-12, atol=1e-20)
            assert np.linalg.eigvalsh(covariance).min() >= -1e-15
            assert not any(row[k] for k in ('source_error_model_calibrated', 'linearization_validated', 'motion_permission'))
        for relation in value['physical_gap_relations'].values():
            assert relation['source_ids'] == list(SOURCES) and relation['difference_step'] == step
            assert not any(relation[k] for k in ('navigation_action_permitted', 'foot_contact_permitted',
                'future_gait_qualified', 'linearization_validated', 'uncertainty_model_calibrated'))
    a, b = [np.asarray(s['observations'][-1]['pose_error_factor']) for s in steps]
    pose = []
    for j, name in enumerate(SOURCES):
        pose.append(dict(source_id=name, terminal_position_factor_left_m=a[:3, j].tolist(),
            terminal_position_factor_right_m=b[:3, j].tolist(),
            position_factor_difference_norm_m=float(np.linalg.norm(a[:3, j]-b[:3, j])),
            rotation_factor_difference_norm_rad=float(np.linalg.norm(a[3:, j]-b[3:, j]))))
    relations = {label: compare(steps[0]['physical_gap_relations'][label], steps[1]['physical_gap_relations'][label])
                 for label in ('initial', 'terminal')}
    report = dict(status='BOUND_NUMERICAL_ACCOUNTING_AND_TWO_STEP_COMPARISON_COMPLETE',
        identity_sha256=identities, analysis_source_sha256=own, exact_saved_nominal_frames_per_step=219,
        pose_factor_covariance_accounting_pass=True, terminal_pose_source_comparison=pose,
        physical_gap_comparison=relations,
        categorical_changed_frames=[len(s['categorical_changes']) for s in steps],
        plane_seed_changed_members={label: [s['physical_gap_relations'][label]['changed_plane_seed_members'] for s in steps]
                                    for label in ('initial', 'terminal')},
        perturbation_steps=list(STEPS), step_selected_for_clearance=None, calibration_validated=False,
        continuous_error_coverage_established=False, independent_experimental_trials=0,
        physical_action_executed=False, navigation_qualified=False)
    verify(launch); exact(preflight(), launch); verify_bindings(bound | own)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch-sha256', required=True); parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args(); target = OUTPUT/'numerical_comparison.json'
    if target.exists() or target.is_symlink(): raise ValueError('fresh numerical comparison only')
    result = analyze({'launch.json': args.launch_sha256, 'result.json': args.result_sha256})
    write_json(target, result); print(result['status'], flush=True)
