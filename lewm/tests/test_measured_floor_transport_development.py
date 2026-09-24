"""Synthetic witnesses/public packets: these tests do not establish raw-fit accuracy."""
import ast
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_joint_floor_registered_controller_development import packets
from lewm.tests.test_joint_measured_floor_plane_development import grid
from lewm.tests.test_floor_pose_registration_development import render_plane
from lewm.joint_floor_registered_evidence_development import JointFloorRegistration, current_joint_floor_registered_pose, depth_hash
from lewm.joint_measured_floor_plane_development import fit_joint_plane
from lewm.measured_floor_transport_development import (
    current_measured_floor_pose, transport_evidence, SCHEMA)
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, from_native_depth
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_packet
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))


def dual(raw, policy, primary, auxiliary, image):
    raw['current_pose'].update(rgb_sha256=primary['rgb_sha256'], depth_sha256=depth_hash(primary),
        auxiliary_rgb_sha256=image['rgb_sha256'], auxiliary_depth_sha256=depth_hash(auxiliary))
    raw.update(observer_variant='front_first_dual_camera_anchor_v1', camera_selection_current=True)
    raw.setdefault('calibration_ids', {}).update(auxiliary_rgb=image['calibration_id'], auxiliary_depth=auxiliary['calibration_id'])
    raw['camera_selection'] = (dict(selected_camera=None, initial_paired_reference=True)
        if raw['current_pose']['frame'] == 0 else dict(selected_camera='primary', auxiliary_attempted=False))
    current_dual_camera_pose(raw, policy, image, auxiliary, identity=(0, 0, 0), now_ns=raw['decision_ns'])
    return raw


def item(frame=0, previous=None, narrow=False, height=.32):
    p, _, _, raw, now = packets(frame, previous)
    dd, _ = render_plane(np.asarray(BODY_FROM_OPTICAL), np.array([0., 0., 1.]), height)
    aa, _ = render_plane(body_from_optical(), np.array([0., 0., 1.]), height)
    if narrow:
        dd[:] = 0.
        aa[:, :300] = 0.; aa[:, 320:] = 0.
    d = from_native_depth(dd, p, measured_ns=now, available_ns=now, now_ns=now)
    a = auxiliary_packet(aa, p, measured_ns=now, available_ns=now, now_ns=now)
    image = from_captured_rgb(p['image']['rgb'], a, p, measured_ns=now, available_ns=now, now_ns=now)
    return p, d, a, dual(raw, p, d, a, image), now, image


def transported():
    state = JointFloorRegistration(); previous = None
    for frame in range(2):
        p, d, a, raw, now, image = item(frame, previous, height=.32+.02*frame)
        anchor = state.observe(p, d, a, raw, now_ns=now); previous = raw
    _, _, a, raw, now, _ = item(2, previous)
    clouds = (np.empty((0, 3)), grid(height=.01)+[0., 0., -.02])
    fit = fit_joint_plane(*clouds, [0., 0., 1.])
    result = transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
        auxiliary_depth_sha256=depth_hash(a))
    return result, anchor, raw, fit, clouds, now


def test_transport_uses_visual_motion_and_preserves_separate_unavailable_plane():
    e, anchor, raw, fit, _, now = transported()
    p, R, pose = current_measured_floor_pose(e, identity=(0, 0, 0), now_ns=now)
    np.testing.assert_allclose(p, [.02, 0., .02], atol=1e-6, rtol=0)
    assert e['schema'] == SCHEMA and e['original_visual_evidence'] == raw
    assert e['floor_transport']['anchor'] == anchor
    assert e['floor_transport']['unavailable_current_plane'] == fit and not fit['available']
    assert not pose['current_floor_registration_used']
    assert e['floor_transport']['correction']['position_error_bound'] is None
    with pytest.raises(ValueError): current_joint_floor_registered_pose(e, identity=(0, 0, 0), now_ns=now)
    old_anchor = deepcopy(anchor)
    e['floor_transport']['anchor']['current_pose']['position_initial_body_m'][0] += 1
    assert anchor == old_anchor


@pytest.mark.parametrize('fault', ['pose', 'anchor_pose', 'raw', 'plane', 'residual', 'count',
    'auxiliary_hash', 'clock', 'identity', 'claim', 'correction', 'future_anchor'])
def test_forged_or_stale_transport_witness_is_rejected(fault):
    e, _, _, _, _, now = transported(); w = e['floor_transport']
    if fault == 'pose': e['current_pose']['position_initial_body_m'][0] += .001
    elif fault == 'anchor_pose': w['anchor']['current_pose']['position_initial_body_m'][0] += .001
    elif fault == 'raw': e['original_visual_evidence']['current_pose']['position_initial_body_m'][0] += .001
    elif fault == 'plane': w['unavailable_current_plane']['available'] = True
    elif fault == 'residual': w['camera_residuals'][1]['maximum_residual_m'] = .00301
    elif fault == 'count': w['camera_residuals'][1]['count'] -= 1
    elif fault == 'auxiliary_hash': w['auxiliary_depth_sha256'] = 'a'*64
    elif fault == 'clock': e['decision_ns'] += 1
    elif fault == 'identity': e['identity'] = (0, False, 0)
    elif fault == 'claim': e['navigation_qualified'] = True
    elif fault == 'correction': w['correction']['correction_magnitude_m'] = 0.
    elif fault == 'future_anchor': w['anchor']['decision_ns'] = now+100_000_000
    with pytest.raises((ValueError, KeyError)):
        current_measured_floor_pose(e, identity=(0, 0, 0), now_ns=now)


def test_current_geometric_conflict_cannot_hide_behind_insufficient_extent():
    _, anchor, raw, _, clouds, now = transported()
    clouds = (clouds[0], clouds[1]+[0., 0., .01])
    fit = fit_joint_plane(*clouds, [0., 0., 1.])
    assert fit['reason'] == 'insufficient_combined_two_axis_extent'
    with pytest.raises(ValueError, match='conflicts'):
        transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
            auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])


def test_absence_has_no_geometric_agreement_and_cannot_initialize_floor():
    _, anchor, raw, _, _, now = transported(); clouds = (np.empty((0, 3)), np.empty((0, 3)))
    fit = fit_joint_plane(*clouds, [0., 0., 1.])
    e = transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
        auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])
    assert all(r['maximum_residual_m'] is None for r in e['floor_transport']['camera_residuals'])
    p, d, a, raw, now, _ = item(narrow=True)
    state = MeasuredFloorTransportRegistration()
    with pytest.raises(ValueError, match='initial floor'):
        state.observe(p, d, a, raw, now_ns=now)
    assert state.failed and state.anchor is None and state.reference is None


def test_full_plane_conflict_and_available_plane_do_not_use_missingness_route():
    _, anchor, raw, _, _, now = transported()
    for clouds in ((grid(), grid()), (grid(), grid()+[0., 0., .02])):
        fit = fit_joint_plane(*clouds, [0., 0., 1.])
        with pytest.raises(ValueError, match='only exactly reconstructed'):
            transport_evidence(anchor, raw, fit, clouds, identity=(0, 0, 0), now_ns=now,
                auxiliary_depth_sha256=raw['current_pose']['auxiliary_depth_sha256'])


def test_registration_preserves_full_plane_prefix_transport_anchor_and_reacquisition():
    old, new = JointFloorRegistration(), MeasuredFloorTransportRegistration(); previous = None
    for frame, narrow in enumerate((False, True, True, False)):
        p, d, a, raw, now, _ = item(frame, previous, narrow=narrow)
        before = deepcopy(raw); e = new.observe(p, d, a, raw, now_ns=now)
        if frame == 0:
            assert e == old.observe(p, d, a, raw, now_ns=now)
            anchor = deepcopy(e)
        elif narrow:
            assert e['floor_transport']['anchor'] == anchor == new.anchor
            assert e['floor_transport']['correction']['anchor_age_frames'] == frame
            assert not e['floor_transport']['unavailable_current_plane']['available']
        else:
            # Original algorithm with its original initial reference gives exactly the same reacquisition.
            old.frame = frame-1
            assert e == old.observe(p, d, a, raw, now_ns=now) == new.anchor
        assert raw == before
        previous = raw
    count = new.frame
    with pytest.raises(ValueError): new.observe(p, d, a, raw, now_ns=now)
    assert new.failed and new.frame == count


def test_all_online_consumers_use_transport_without_map_rewrite():
    c = MeasuredFloorTransportController(None, None, public_mission=dict(
        goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    previous = None
    for frame in range(3):
        p, d, a, raw, now, image = item(frame, previous, narrow=frame>0)
        old_route = deepcopy(c.memory.route)
        c.motion = SimpleNamespace(observe=lambda *args, **kwargs: raw)
        result = c.observe(p, d, None, auxiliary_depth=a, auxiliary_rgb=image, now_ns=now)
        assert result['terminal'] is None, result['failure']
        position, rotation, _ = current_measured_floor_pose(result['evidence'], identity=(0, 0, 0), now_ns=now)
        np.testing.assert_array_equal(c.memory.position, position)
        np.testing.assert_array_equal(c.memory.rotation, rotation)
        np.testing.assert_array_equal(c.residual.pose['position'], position)
        np.testing.assert_array_equal(c.residual.pose['rotation'], rotation)
        assert result['observed_goal_distance_m'] == np.linalg.norm(position[:2]-[1., 0.])
        assert c.memory.route[:-1] == old_route and c.selector.residual is c.residual
        assert result['requested_command'] == [0., 0., 0.]
        previous = raw
    count = c.memory.partition.total_returns
    result = c.observe(p, d, None, auxiliary_depth=a, auxiliary_rgb=None, now_ns=now+100_000_000)
    assert result['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and result['requested_command'] == [0., 0., 0.]
    assert c.memory.partition.total_returns == count


def test_consumer_methods_only_change_the_explicit_pose_accessor():
    def method(path, cls, name):
        tree = ast.parse(Path(path).read_text())
        c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'current_measured_floor_pose': node.id = 'current_joint_floor_registered_pose'
            return node
    for path, old, new, name in (
        ('joint_floor_registered_controller', 'JointFloorRegisteredSurfaceMemory', 'MeasuredFloorTransportMemory', 'observe'),
        ('joint_floor_registered_controller', 'JointFloorRegisteredResidual', 'MeasuredFloorTransportResidual', 'observe'),
        ('measured_settling_round_trip_controller', 'MeasuredSettlingRoundTripController', 'MeasuredFloorTransportController', 'advance')):
        expected = method('lewm/'+path+'_development.py', old, name)
        actual = Normalize().visit(method('lewm/measured_floor_transport_controller_development.py', new, name))
        assert ast.dump(expected) == ast.dump(actual)


def test_noncommuting_rotation_and_translation_match_homogeneous_composition():
    p, d, a, raw0, now, _ = item()
    state = JointFloorRegistration(); state.observe(p, d, a, raw0, now_ns=now)
    p, _, _, _, now, _ = item(1, raw0)
    normal = np.array([0., np.sin(.04), np.cos(.04)])
    dd, _ = render_plane(np.asarray(BODY_FROM_OPTICAL), normal, .34)
    aa, _ = render_plane(body_from_optical(), normal, .34)
    d = from_native_depth(dd, p, measured_ns=now, available_ns=now, now_ns=now)
    a = auxiliary_packet(aa, p, measured_ns=now, available_ns=now, now_ns=now)
    image = from_captured_rgb(p['image']['rgb'], a, p, measured_ns=now, available_ns=now, now_ns=now)
    raw1, _ = fixture.joint_visual(1, (.01, 0., .2), previous=raw0)
    raw1 = dual(raw1, p, d, a, image)
    anchor = state.observe(p, d, a, raw1, now_ns=now)
    p, d, a, _, now, image = item(2, raw1)
    raw2, _ = fixture.joint_visual(2, (.02, .03, .3), previous=raw1)
    raw2 = dual(raw2, p, d, a, image)
    def transform(pose):
        T = np.eye(4); T[:3, :3] = pose['rotation_initial_body_from_current_body']
        T[:3, 3] = pose['position_initial_body_m']; return T
    expected = transform(anchor['current_pose'])@np.linalg.inv(transform(raw1['current_pose']))@transform(raw2['current_pose'])
    world = grid(height=.01)
    clouds = (np.empty((0, 3)), (world-expected[:3, 3])@expected[:3, :3])
    fit = fit_joint_plane(*clouds, np.asarray(raw2['current_pose']['rotation_initial_body_from_current_body']).T@[0., 0., 1.])
    e = transport_evidence(anchor, raw2, fit, clouds, identity=(0, 0, 0), now_ns=now,
        auxiliary_depth_sha256=depth_hash(a))
    np.testing.assert_allclose(transform(e['current_pose']), expected, atol=1e-12, rtol=0)


def test_registration_conflict_latches_without_promoting_failed_floor_anchor():
    state = MeasuredFloorTransportRegistration()
    p, d, a, raw, now, _ = item()
    first = state.observe(p, d, a, raw, now_ns=now)
    p, d, a, raw, now, _ = item(1, raw, narrow=True, height=.34)
    with pytest.raises(ValueError, match='conflicts'): state.observe(p, d, a, raw, now_ns=now)
    assert state.failed and state.frame == 0 and state.anchor == first
    with pytest.raises(ValueError, match='latched'): state.observe(p, d, a, raw, now_ns=now)
