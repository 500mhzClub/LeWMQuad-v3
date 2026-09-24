"""Consumer integration checks; synthetic sensors are not mission evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import FOCAL, from_native_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.rgbd_fused_navigation_development import (
    RGBDFusedNavigation, RGBDFusedTraversal, SharedAttitudeView, PreparedRayView, fused_forward_speed)
from lewm.rgbd_inertial_ray_memory_development import RGBDInertialRayMemory
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_rgbd_correspondence_motion_development import texture
from lewm.tests.test_rgbd_inertial_fusion_development import prior, hypotheses
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def frames(count, *, textured=True):
    stream = Stream()
    noise = (texture()[:, :, 0].astype(float)*.7).astype(np.uint8)
    image = np.stack((noise+30, noise+40, noise+20), axis=-1)
    if not textured:
        image[:] = [115, 120, 108]
    v = (np.arange(480)+.5-240)/FOCAL
    floor = np.divide(.36, v, out=np.full(480, 200.), where=v>0)
    native = np.minimum(2., floor)[:, None]*np.ones((1, 640))
    for tick in range(count):
        p, f, now = stream.frame(tick)
        p['image']['rgb'] = image.copy()
        d = from_native_depth(native.astype(np.float32), p, measured_ns=now, available_ns=now, now_ns=now)
        yield p, f, d, now


def navigator():
    return RGBDFusedNavigation(ArticulatedCollisionGeometry(URDF), memory_arm='local_only',
                              prior=prior(), hypotheses=hypotheses())


def test_actual_sensor_kernel_flows_to_regions_global_orientation_and_traversal():
    c = navigator()
    reference = RGBDInertialRayMemory(prior=prior(), hypotheses=hypotheses())
    for p, f, d, now in frames(24):
        expected = reference.observe(p, d, f, now_ns=now)
        row = c.observe_rgbd(p, f, d, now_ns=now)
        assert row['sensor_fusion'] == expected['fusion']
        assert row['raw_depth_motion'] == expected['depth_state']['motion']
        assert row['global_orientation'] == expected['depth_state']['relative_orientation']
        assert c.regions.memory.pending is None and c._context is None
        np.testing.assert_array_equal(c.regions.memory.position, reference.rays.position)
        assert not row['terminal'] and not row['learned_navigation_policy']
    assert isinstance(c.child, RGBDFusedTraversal)
    assert row['child']['status'] == 'TRAVERSING'
    assert row['requested_command'][0] > 0
    assert row['raw_depth_motion']['rank'] == 2
    assert row['raw_depth_motion']['translation_previous_body_m'] is None
    assert c.sensor_memory.state.depth.orientation.samples_integrated == 23*50
    assert len(c.sensor_memory.rays.frames) == 1
    assert np.linalg.norm(c.regions.memory.position) < 1e-7


def test_neutral_budget_failure_latches_controller_memory_and_queries():
    c = navigator()
    for p, f, d, now in frames(30, textured=False):
        try:
            c.observe_rgbd(p, f, d, now_ns=now)
        except SensorContractError:
            break
    else:
        pytest.fail('blank weak-depth stream should exhaust unchanged budget')
    assert c.status == 'FAILED_SENSOR' and c.sensor_memory.failed
    assert c._context is None and c.regions.memory.pending is None
    with pytest.raises(SensorContractError):
        c.regions.memory.query([[1., 0., 0.]], np.array([False]), now_ns=now)
    with pytest.raises(SensorContractError):
        c.observe_rgbd(p, f, d, now_ns=now)
    with pytest.raises(SensorContractError):
        c.observe_stopping_tail(p, f, d, now_ns=now)


@pytest.mark.parametrize('fault', ['depth', 'rgb', 'gyro', 'clock', 'privileged'])
def test_sensor_faults_cannot_fall_back_to_old_memory(fault):
    c = navigator()
    data = list(frames(2))
    p, f, d, now = data[0]
    c.observe_rgbd(p, f, d, now_ns=now)
    p, f, d, now = data[1]
    if fault == 'depth': d['depth_m'][0, 0] = 6.  # Outside declared calibrated range.
    elif fault == 'rgb': p['image']['rgb'][0, 0] = 0
    elif fault == 'gyro': f['values'][0, 0] += .1
    elif fault == 'clock': now += 100_000_000
    else: p['native_pose'] = [0.]*7
    with pytest.raises(SensorContractError): c.observe_rgbd(p, f, d, now_ns=now)
    assert c.status == 'FAILED_SENSOR' and c.sensor_memory.failed


def test_fused_speed_uses_current_axes_and_not_raw_depth_or_commands():
    rotation = rotation_increment([0., 0., np.pi/2])
    memory = SimpleNamespace(last_ns=100, rotation=rotation,
        fusion=dict(measured_ns=100, usable_under_declared_proxy_budget=True,
                    velocity_initial_body_m_s=[0., .2, 0.]))
    assert fused_forward_speed(memory, now_ns=100) == pytest.approx(.2)
    memory.fusion['velocity_initial_body_m_s'] = [0., -.2, 0.]
    assert fused_forward_speed(memory, now_ns=100) == 0.
    with pytest.raises(SensorContractError): fused_forward_speed(memory, now_ns=200)


def test_region_facade_rejects_duplicate_or_different_prepared_observation():
    p, f, d, now = next(frames(1))
    owner = RGBDInertialRayMemory(prior=prior(), hypotheses=hypotheses())
    row = owner.observe(p, d, f, now_ns=now)
    view = PreparedRayView(owner)
    view.stage(p, d, row)
    with pytest.raises(SensorContractError): view.stage(p, d, row)
    with pytest.raises(SensorContractError): view.observe(p, d, deepcopy(row['depth_state']), now_ns=now)
    with pytest.raises(SensorContractError): view.observe(p, d, row['depth_state'], now_ns=now)


def test_scan_attitude_is_local_view_of_global_owner_not_new_integration():
    nav = SimpleNamespace(_context=None, sensor_memory=SimpleNamespace(failed=False))
    view = SharedAttitudeView(nav, local=True)
    anchor = rotation_increment([.1, -.2, .7])
    p, f = {}, {}
    for tick in range(3):
        now = 3_000_000_000+tick*100_000_000
        relative = rotation_increment([0., 0., .03*tick])
        attitude = dict(rotation_initial_body_from_current_body=(anchor@relative).tolist(),
                        decision_ns=now, start_ns=1_600_000_000, samples_integrated=700+50*tick,
                        gyro_rate_hz=500, integration='causal_midpoint', hardware_calibrated=False)
        nav._context = (p, f, dict(measured_ns=now, depth_state=dict(relative_orientation=attitude)))
        result = (view.begin if tick == 0 else view.step)(p, f, now_ns=now)
        np.testing.assert_allclose(result['rotation_initial_body_from_current_body'], relative, atol=1e-12)
        assert result['start_ns'] == 3_000_000_000 and result['samples_integrated'] == 50*tick
        assert attitude['start_ns'] == 1_600_000_000 and attitude['samples_integrated'] == 700+50*tick
    nav._context = None
    with pytest.raises(SensorContractError): view.step(p, f, now_ns=now)


def test_terminal_tail_updates_sensor_owner_without_restarting_mission():
    c = navigator()
    data = list(frames(3))
    p, f, d, now = data[0]
    c.observe_rgbd(p, f, d, now_ns=now)
    with pytest.raises(SensorContractError): c.observe_stopping_tail(p, f, d, now_ns=now)
    c.finish_physical_stop(now_ns=now)
    mission_tick = c.tick
    for p, f, d, now in data[1:]:
        row = c.observe_stopping_tail(p, f, d, now_ns=now)
        assert row['requested_command'] == [0., 0., 0.] and not row['controller_restarted']
        assert c.tick == mission_tick and c.status == 'PHYSICAL_STOP'
    assert c.sensor_memory.state.depth.orientation.samples_integrated == 100
    assert c.sensor_memory.rays.frames[0]['measured_ns'] == data[0][-1]


def test_inherited_scan_is_given_shared_local_attitude_before_first_observation():
    c = navigator()
    data = list(frames(2))
    p, f, d, now = data[0]
    c.observe_rgbd(p, f, d, now_ns=now)
    # State-machine fixture only: no claim a real traversal reached this stage.
    c.stage = 'HOLD_SCAN'
    c.hold_since = now-1_500_000_000
    p, f, d, now = data[1]
    c.observe_rgbd(p, f, d, now_ns=now)
    assert c.stage == 'SCAN' and c.scan.status == 'NEW'
    assert isinstance(c.scan.orientation, SharedAttitudeView) and c.scan.orientation.local
    assert not isinstance(c.orientation, FastRelativeOrientation)
