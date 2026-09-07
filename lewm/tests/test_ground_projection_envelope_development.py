import math

import numpy as np
import pytest

from lewm.ground_projection_envelope_development import cap_dot_bounds, project_ground_rays, observe_ground_envelope, ORIGIN_BODY
from lewm.causal_ground_plane_development import CausalGroundPlane
from lewm.tests.test_causal_ground_plane_development import example


def test_cap_extrema_include_interior_aligned_normal_and_antipode():
    low, high = cap_dot_bounds([0, 0, 1], [[0, 0, 2], [0, 0, -2], [0, 0, 0]], .1)
    np.testing.assert_allclose(low, [2 * math.cos(.1), -2, 0], atol=1e-15)
    np.testing.assert_allclose(high, [2, -2 * math.cos(.1), 0], atol=1e-15)


def test_zero_uncertainty_recovers_exact_ground_intersection():
    ray = np.array([[1., 0, -.5], [1, -.4, -.3]])
    result = project_ground_rays([0, 0, 1], .3, ray, height_radius=0., angle_radius=0.)
    truth = .343 / -ray[:, 2]
    assert result['interval_valid'].all()
    np.testing.assert_allclose(result['lower_optical_depth_m'], truth, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result['upper_optical_depth_m'], truth, rtol=0, atol=1e-14)
    assert not result['uncertainty_calibrated'] and not result['metric_clearance_qualified']


def test_interval_contains_sampled_correlated_plane_ratios_without_fitting():
    rng = np.random.default_rng(19)
    ray = np.stack((np.ones(100), rng.uniform(-.8, .8, 100), rng.uniform(-.6, -.02, 100)), -1)
    result = project_ground_rays([0, 0, 1], .3, ray, height_radius=.03, angle_radius=.1)
    valid = result['interval_valid']
    assert valid.any() and (~valid).any()
    for _ in range(300):
        theta, phi = rng.uniform(0, .1), rng.uniform(-math.pi, math.pi)
        normal = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
        height = .3 + rng.uniform(-.03, .03)
        depth = -(height + normal @ ORIGIN_BODY) / (ray[valid] @ normal)
        assert np.all(depth >= result['lower_optical_depth_m'][valid] - 1e-12)
        assert np.all(depth <= result['upper_optical_depth_m'][valid] + 1e-12)


def test_possible_horizon_or_nonpositive_camera_height_abstains_not_far_clip():
    rays = [[1., 0, -.01], [1., 0, 0], [1., 0, .2]]
    result = project_ground_rays([0, 0, 1], .3, rays, height_radius=.03, angle_radius=.1)
    assert not result['interval_valid'].any() and result['possibly_nonforward'].all()
    assert np.isnan(result['upper_optical_depth_m']).all()
    result = project_ground_rays([0, 0, 1], .3, [[1., 0, -.5]], height_radius=.4, angle_radius=.1)
    assert not result['interval_valid'].any()


def test_clip_ambiguity_is_reported_not_truncated_into_a_false_valid_interval():
    result = project_ground_rays([0, 0, 1], .3, [[1., 0, -.001]], height_radius=0., angle_radius=0.)
    assert result['clip_uncertain'].all() and not result['interval_valid'].any()


@pytest.mark.parametrize('normal,angle', [([0, 0, 2], .1), ([0, 0, 1], -.1), ([0, 0, 1], True), ([0, 0, 1], float('nan'))])
def test_invalid_normal_or_radius_rejected(normal, angle):
    with pytest.raises(ValueError):
        project_ground_rays(normal, .3, [[1., 0, -.5]], height_radius=.01, angle_radius=angle)


def test_camera_lever_arm_is_included_in_uncertain_height():
    result = project_ground_rays([0, 0, 1], .3, [[1., 0, -.5]], height_radius=0., angle_radius=.1)
    lower, upper = result['camera_height_interval_m']
    assert lower < .343 < upper and upper - lower > .06


def test_fresh_pixel_evidence_intersects_ground_but_never_certifies_clearance_or_identity():
    _, packet = example()
    now = packet['image']['measured_ns']
    state = CausalGroundPlane().begin(packet, now_ns=now)
    packet['image']['rgb'][300:] = [115, 120, 108]
    result = observe_ground_envelope(packet, state, now_ns=now, height_radius=.03, angle_radius=.1)
    assert result['observed_interval_valid'].any()
    assert not result['observed_interval_valid'][:37].any()
    assert not result['metric_clearance_qualified'] and result['place_or_exit_identity'] is None
    packet['image']['rgb'][:] = 100
    unknown = observe_ground_envelope(packet, state, now_ns=now, height_radius=.03, angle_radius=.1)
    assert not unknown['observed_interval_valid'].any()


@pytest.mark.parametrize('fault', ['clock', 'calibration', 'privilege'])
def test_stale_or_privileged_envelope_input_rejected(fault):
    _, packet = example()
    now = packet['image']['measured_ns']
    state = CausalGroundPlane().begin(packet, now_ns=now)
    if fault == 'clock':
        state['decision_ns'] += 1
    if fault == 'calibration':
        state['robot_geometry_sha256'] = 'different'
    if fault == 'privilege':
        packet['world_pose'] = [0] * 7
    with pytest.raises(ValueError):
        observe_ground_envelope(packet, state, now_ns=now, height_radius=.03, angle_radius=.1)
