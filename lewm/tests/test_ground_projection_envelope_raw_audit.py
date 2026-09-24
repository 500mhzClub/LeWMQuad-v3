import copy

import numpy as np
import pytest

from lewm.ground_projection_envelope_development import project_ground_rays
from scripts.analyze_go2_ground_projection_envelope_development_v1 import range_accounting, aggregate


def example():
    rays = np.array([[[1., 0, -.5], [1., 0, -.3], [1., 0, -.2]]])
    envelope = project_ground_rays([0, 0, 1], .3, rays, height_radius=.01, angle_radius=.025)
    envelope.update(rays_body=rays, bottom_connected_floor_pixels=np.ones((1, 3), bool))
    truth = {'valid': np.ones((1, 3), bool), 'visible_floor': np.array([[True, True, False]]),
             'ground_optical_depth_m': .343 / -rays[..., 2]}
    state = {'up_current_body': [0, 0, 1], 'body_origin_height_m': .3}
    return envelope, truth, state


def test_false_surface_not_deleted_or_mislabeled_as_a_valid_floor_range():
    envelope, truth, state = example()
    row = range_accounting(envelope, truth, .3, np.array([0., 0, 1]), state)
    assert row['pixels'] == 3 and row['visible_floor'] == row['accepted_visible_floor'] == 2
    assert row['positive_false_surface'] == row['accepted_false_surface'] == 1
    assert row['covered_visible_floor'] == 2 and row['outside_interval_visible_floor'] == 0
    assert row['true_plane_inside_hypothesis'] and row['nominal_error_max_m'] < 1e-12


def test_uncertainty_assumption_failure_reported_not_false_mathematical_pass():
    envelope, truth, state = example()
    truth['ground_optical_depth_m'] *= 2
    row = range_accounting(envelope, truth, .643, np.array([0., 0, 1]), state)
    assert not row['true_plane_inside_hypothesis'] and row['outside_interval_visible_floor'] == 2


def test_conditional_containment_violation_is_an_integrity_failure():
    envelope, truth, state = example()
    envelope['upper_optical_depth_m'] *= .1
    with pytest.raises(ValueError, match='enclosure violated'):
        range_accounting(envelope, truth, .3, np.array([0., 0, 1]), state)


def test_abstention_and_ambiguous_reference_have_explicit_denominators():
    envelope, truth, state = example()
    envelope['interval_valid'][:] = False
    truth['valid'][0, 2] = False
    row = range_accounting(envelope, truth, .3, np.array([0., 0, 1]), state)
    assert row['reference_ambiguous'] == 1 and row['accepted_visible_floor'] == 0
    assert row['abstained_positive_visible_floor'] == 2
    assert row['accepted_width_max_m'] is None


def test_aggregate_preserves_pixel_counts_and_never_averages_frame_means():
    envelope, truth, state = example()
    one = range_accounting(envelope, truth, .3, np.array([0., 0, 1]), state)
    two = copy.deepcopy(one)
    two['nominal_visible_points'] = 1
    two['nominal_error_sum_m'] = 3.
    two['nominal_error_max_m'] = 3.
    result = aggregate([one, two])
    assert result['frames'] == 2 and result['pixels'] == 6
    assert result['nominal_error_mean_m'] == pytest.approx(1.)
    assert result['true_plane_inside_hypothesis_frames'] == 2
    assert aggregate([])['nominal_error_mean_m'] is None
