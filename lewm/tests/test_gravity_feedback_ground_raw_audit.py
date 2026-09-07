import numpy as np
import pytest

from scripts.analyze_go2_gravity_feedback_ground_development_v1 import frame_metrics, summarize


def example():
    state = {'up_current_body': [0, 0, 1], 'body_origin_height_m': .3,
             'feedback_applied': False, 'feedback': {'reason': 'test'}}
    rays = np.array([[[1., 0, -.5], [1., 0, -.3]]])
    truth = {'valid': np.ones((1, 2), bool), 'visible_floor': np.ones((1, 2), bool),
             'ground_optical_depth_m': .343 / -rays[..., 2]}
    pose = np.array([0., 0, .3, 0, 0, 0, 1.])
    return state, rays, truth, pose


def test_ground_and_range_scalar_references_on_an_exact_plane():
    row = frame_metrics(*example())
    assert row['normal_error_rad'] == 0 and row['signed_height_error_m'] == 0
    assert row['error_max_m'] < 1e-12 and row['quarter_m_failure_points'] == 0
    assert row['visible_floor'] == row['nominal_points'] == 2


def test_unavailable_prediction_is_a_failure_not_an_invented_perfect_point():
    _, rays, truth, pose = example()
    row = frame_metrics(None, rays, truth, pose)
    assert not row['sensor_available'] and row['quarter_m_failure_points'] == 2
    report = summarize([row])
    assert report['quarter_m_failure_fraction'] == 1 and report['point_error_mean_m'] is None
    assert not report['normal_metrics_complete']


def test_point_error_and_absence_share_fixed_visible_pixel_denominator():
    state, rays, truth, pose = example()
    state['body_origin_height_m'] = .6
    row = frame_metrics(state, rays, truth, pose)
    assert row['quarter_m_failure_points'] == 2
    report = summarize([row, frame_metrics(None, rays, truth, pose)])
    assert report['visible_floor'] == 4 and report['unavailable_floor_points'] == 2
    assert report['quarter_m_failure_fraction'] == 1
    assert report['sensor_available_frames'] == 1 and not report['normal_metrics_complete']


def test_empty_source_population_rejected():
    with pytest.raises(ValueError):
        summarize([])
