from copy import deepcopy

import numpy as np
import pytest

from scripts.joint_rgbd_pose_plane_comparison_development import gap_metrics, compare


def relation():
    return dict(shape_ids=['physical:0'], source_ids=['shared'], joint_gap_factor_m=[[0.]],
        pose_only_gap_factor_m=[[.001]], floor_only_gap_factor_m=[[-.001]],
        joint_gap_variance_m2=[0.], incorrect_independent_gap_variance_m2=[.000002],
        all_perturbed_footprints_observed=[True], stored_measured_ns=1, current_measured_ns=2)


def test_shared_cancellation_does_not_turn_independent_sum_into_uncertainty():
    result = gap_metrics(relation())[0]
    assert result['joint_one_source_unit_scale_m'] == 0.
    assert result['incorrect_independent_one_source_unit_scale_m'] == pytest.approx(np.sqrt(2)*.001)
    assert result['joint_minus_split_factor_norm_m'] == 0.
    assert not result['motion_permission'] and not result['calibrated_coverage']


def test_step_change_reported_without_selecting_favourable_result():
    left = relation(); right = deepcopy(left)
    right['joint_gap_factor_m'] = [[.0002]]; right['joint_gap_variance_m2'] = [4e-8]
    row = compare(left, right)[0]
    assert row['joint_factor_step_difference_norm_m'] == .0002
    assert row['relative_factor_step_difference'] == 1.
    assert not row['step_stability_accepted'] and not row['calibrated_bound_selected']


def test_unobserved_null_remains_missing_not_a_zero_error_bound():
    r = relation()
    for key in ('joint_gap_factor_m', 'pose_only_gap_factor_m', 'floor_only_gap_factor_m'): r[key] = [[None]]
    for key in ('joint_gap_variance_m2', 'incorrect_independent_gap_variance_m2'): r[key] = [None]
    r['all_perturbed_footprints_observed'] = [False]
    row = compare(r, r)[0]
    assert row['joint_factor_step_difference_norm_m'] is None
    assert row['step_left']['joint_one_source_unit_scale_m'] is None


@pytest.mark.parametrize('fault', ['variance', 'source', 'shape', 'coverage'])
def test_malformed_or_mismatched_evidence_rejected(fault):
    a, b = relation(), relation()
    if fault == 'variance': b['joint_gap_variance_m2'] = [1.]
    if fault == 'source': b['source_ids'] = ['other']
    if fault == 'shape': b['joint_gap_factor_m'] = [[0., 0.]]
    if fault == 'coverage': b['all_perturbed_footprints_observed'] = [1]
    with pytest.raises((ValueError, AssertionError)): compare(a, b)
