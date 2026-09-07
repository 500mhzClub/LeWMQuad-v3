from copy import deepcopy

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.paired_rgbd_physical_plane_development import PairedRGBDPhysicalPlane
from lewm.raw_complementary_rgbd_sensitivity_development import RawComplementaryRGBDSensitivity, SHAPES
from lewm.rgbd_inertial_fusion_development import RGBDInertialState
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
from lewm.tests.test_rgbd_inertial_fusion_development import prior, hypotheses
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def zeros(count=1):
    return {k: np.zeros((*shape, count)) for k, shape in SHAPES.items()}


def model(names=('zero',), step=.001):
    return RawComplementaryRGBDSensitivity(names, prior=prior(), hypotheses=hypotheses(), difference_step=step)


def room(count=3):
    for _, p, f, _ in stream(count): yield p, room_depth(p), f, p['sensor_state']['decision_ns']


def query(planes, name, now, position=(1.3, 0., 0.)):
    return planes.query(name, ArticulatedCollisionGeometry(URDF), np.repeat([0., .8, -1.5], 4),
        position, np.eye(3), now_ns=now, normal_error=.002, up_error=.001, plane_offset_error=.001)


def test_zero_sources_preserve_actual_rgbd_and_inputs_exactly():
    m = model(); reference = RGBDInertialState(prior=prior(), hypotheses=hypotheses())
    for p, d, f, now in room():
        before = deepcopy((p, d, f))
        result = m.observe(p, d, f, zeros())
        expected = reference.observe(p, d, f, now_ns=now)
        assert result['nominal_fusion'] == expected['fusion']
        assert result['nominal_raw_depth_state'] == expected['depth_state']
        assert result['nominal_point_state'] == expected['point_state']
        np.testing.assert_array_equal(result['pose_error_factor'], np.zeros((6, 1)))
        assert not result['categorical_branch_changes'] and not result['motion_permission']
        for current, old in zip((p, d, f), before):
            def equal(a, b):
                if isinstance(a, dict):
                    assert a.keys() == b.keys()
                    for k in a: equal(a[k], b[k])
                elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
                else: assert a == b
            equal(current, old)


def test_initial_velocity_source_propagates_in_weak_depth_and_is_not_reset():
    m = model(('prior_y',))
    for tick, (p, d, f, now) in enumerate(packets([np.zeros((480, 640, 3), np.uint8)]*4)):
        factors = zeros(); factors['initial_velocity'][1, 0] = .01
        result = m.observe(p, d, f, factors)
        assert result['pose_error_factor'][1, 0] == pytest.approx(tick*.001, abs=1e-12)
        if tick: assert result['nominal_fusion']['depth_rank'] == 1
    assert result['nominal_fusion']['kind'] == 'INERTIALLY_PREDICTED_WEAK_COMPONENT'


def test_rgb_points_fill_weak_subspace_and_are_recomputed_for_every_pair():
    import cv2
    image = texture(); moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 0]]), (640, 480))
    m = model(('depth_scale',), step=.01)
    for p, d, f, now in packets([image, moved]):
        loading = zeros(); loading['depth_m'][..., 0] = .001*d['depth_m']
        result = m.observe(p, d, f, loading)
    assert result['nominal_fusion']['kind'] == 'POINT_COMPLEMENTED_PLANE_TRANSLATION'
    assert all(o['fusion']['kind'] == 'POINT_COMPLEMENTED_PLANE_TRANSLATION' for o in m.current_observations)
    assert abs(result['pose_error_factor'][1, 0]) > 1e-6
    assert result['raw_plane_and_point_registration_recomputed'] and result['sensor_quantization_preserved']


def test_shared_gyro_error_uses_one_history_at_both_sensor_rates():
    m = model(('yaw_bias',))
    for tick, (p, d, f, now) in enumerate(room(4)):
        loading = zeros(); loading['gyro'][:, 2, 0] = .001; loading['fast_gyro'][:, 2, 0] = .001
        result = m.observe(p, d, f, loading)
    assert result['pose_error_factor'][5, 0] == pytest.approx(.0003, rel=1e-8)
    m.retain('current')
    relative = m.relative_moments('current', [[1., .2, -.3]])
    np.testing.assert_allclose(relative['point_covariance_m2'], 0., atol=1e-20)


def test_zero_joint_plane_factors_and_no_permission():
    m = model(); plane = PairedRGBDPhysicalPlane(m)
    for i, (p, d, f, now) in enumerate(room()):
        m.observe(p, d, f, zeros())
        if i == 0: plane.retain('initial')
    result = query(plane, 'initial', now)
    assert len(result['shape_ids']) == 27
    assert result['all_perturbed_footprints_observed'].any()
    for key in ('joint_gap_factor_m', 'pose_only_gap_factor_m', 'floor_only_gap_factor_m'):
        np.testing.assert_array_equal(result[key], np.zeros((27, 1)))
    assert not result['changed_plane_seed_members']
    assert not any(result[k] for k in ('navigation_action_permitted', 'foot_contact_permitted',
        'future_gait_qualified', 'uncertainty_model_calibrated', 'linearization_validated',
        'finite_pairs_are_continuous_error_bounds'))


def test_same_view_depth_error_changes_plane_but_pose_cancels():
    m = model(('depth_scale',), step=.01); plane = PairedRGBDPhysicalPlane(m)
    p, d, f, now = next(room())
    loading = zeros(); loading['depth_m'][..., 0] = .001*d['depth_m']
    m.observe(p, d, f, loading); plane.retain('same')
    result = query(plane, 'same', now)
    np.testing.assert_array_equal(result['pose_only_gap_factor_m'], np.zeros((27, 1)))
    np.testing.assert_array_equal(result['joint_gap_factor_m'], result['floor_only_gap_factor_m'])
    assert np.max(np.abs(result['joint_gap_factor_m'])) > .0001


@pytest.mark.parametrize('fault', ['history', 'gyro_rates', 'prior_history', 'missing_depth_loading', 'range_crossing', 'nonfinite'])
def test_errors_and_discontinuities_latch_without_stale_use(fault):
    m = model(); data = list(room(2)); p, d, f, now = data[0]
    m.observe(p, d, f, zeros()); plane = PairedRGBDPhysicalPlane(m); plane.retain('old')
    p, d, f, now = data[1]; loading = zeros()
    if fault == 'history': loading['specific_force'][0, 1, 0] = .01
    elif fault == 'gyro_rates': loading['fast_gyro'][:, 2, 0] = .001
    elif fault == 'prior_history': loading['initial_velocity'][1, 0] = .01
    elif fault == 'missing_depth_loading':
        d['valid'][300, 320] = False; d['depth_m'][300, 320] = 0.; loading['depth_m'][300, 320, 0] = .01
    elif fault == 'range_crossing': loading['depth_m'][..., 0] = 10000.
    elif fault == 'nonfinite': loading['depth_m'][0, 0, 0] = np.nan
    with pytest.raises(SensorContractError): m.observe(p, d, f, loading)
    assert m.failed and m.current is None
    with pytest.raises(SensorContractError): query(plane, 'old', now)
    with pytest.raises(SensorContractError): m.observe(p, d, f, zeros())


def test_absent_plane_is_unknown_not_a_fabricated_zero_gap():
    p, d, f, now = next(packets([texture()]))
    m = model(); m.observe(p, d, f, zeros())
    plane = PairedRGBDPhysicalPlane(m); plane.retain('wall')
    result = query(plane, 'wall', now)
    assert not result['all_perturbed_footprints_observed'].any()
    assert np.isnan(result['nominal_minimum_gap_m']).all()
    assert not result['navigation_action_permitted']
