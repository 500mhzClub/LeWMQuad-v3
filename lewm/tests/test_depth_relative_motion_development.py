import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.depth_relative_motion_development import register_translation, DepthRelativeState
from lewm.tests.test_depth_local_surfaces_development import packet
from lewm.tests.test_observed_traversal_controller_development import Stream


def planes(axes):
    rows = []; normals = []
    for axis in axes:
        u, v = np.meshgrid(np.linspace(-.6, .6, 25), np.linspace(-.6, .6, 25))
        p = np.zeros((u.size, 3)); p[:, axis] = 2.
        p[:, (axis+1)%3] = u.ravel(); p[:, (axis+2)%3] = v.ravel()
        n = np.zeros_like(p); n[:, axis] = 1.
        rows.append(p); normals.append(n)
    return np.concatenate(rows), np.concatenate(normals)


def test_three_plane_translation_and_known_rotation_recover_measured_motion():
    previous = planes([0, 1, 2]); delta = np.array([.031, -.019, .007])
    angle = .025
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    current = ((previous[0]-delta)@rotation, previous[1]@rotation)
    result = register_translation(previous, current, rotation)
    assert result['status'] == 'OBSERVED_TRANSLATION' and result['rank'] == 3
    np.testing.assert_allclose(result['translation_previous_body_m'], delta, atol=1e-8)


def test_corridor_missing_forward_component_is_not_filled_by_zero_motion():
    previous = planes([1, 2]); delta = np.array([.06, .02, -.003])
    current = (previous[0]-delta, previous[1])
    result = register_translation(previous, current, np.eye(3))
    assert result['status'] == 'PARTIALLY_OBSERVED_TRANSLATION' and result['rank'] == 2
    assert result['translation_previous_body_m'] is None
    np.testing.assert_allclose(result['observable_projection_previous_body_m'], [0., .02, -.003], atol=1e-8)
    np.testing.assert_allclose(np.abs(result['weak_directions_previous_body'][0]), [1., 0., 0.])


def test_insufficient_or_distant_clouds_are_not_accepted_as_stationary():
    previous = planes([0, 1, 2])
    for current in ((previous[0][:10], previous[1][:10]), (previous[0]+10., previous[1])):
        result = register_translation(previous, current, np.eye(3))
        assert result['translation_previous_body_m'] is None
        assert result['status'] == 'INSUFFICIENT_SURFACE_SUPPORT'


def test_invalid_rotation_rejected():
    cloud = planes([0, 1, 2])
    with pytest.raises(SensorContractError): register_translation(cloud, cloud, np.zeros((3, 3)))


def test_live_observer_front_wall_has_missing_components_and_causal_fault_latches():
    stream = Stream(); state = DepthRelativeState()
    for tick in range(3):
        p, fast, now = stream.frame(tick)
        _, depth, _ = packet('front', tick=tick)
        result = state.observe(p, depth, fast, now_ns=now)
    assert result['motion']['rank'] == 1 and result['position_initial_body_m'] is None
    assert not result['arrival_verified'] and not result['turn_clearance_qualified']
    with pytest.raises(SensorContractError): state.observe(p, depth, fast, now_ns=now)
    assert state.failed
