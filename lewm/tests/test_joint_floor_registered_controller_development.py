"""Synthetic public packets and pose witnesses; no physical navigation claim."""
import ast
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from lewm.simulated_body_observation_development import BodyObservationBuffer
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, from_native_depth
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_packet
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.joint_floor_registered_evidence_development import JointFloorRegistration, current_joint_floor_registered_pose, depth_hash
from lewm.joint_floor_registered_controller_development import JointFloorRegisteredRoundTripController
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_floor_pose_registration_development import render_plane


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))


def packets(frame=0, previous=None):
    now = 1_500_000_000+frame*100_000_000
    buffer = BodyObservationBuffer((0, 0, 0))
    for t in range(20_000_000, now+1, 20_000_000):
        buffer.append_sensors(dict(gyro=(np.zeros(3), np.ones(3, bool)),
            specific_force=(np.array([0., 0., 9.81]), np.ones(3, bool)),
            joints=(np.zeros(24), np.ones(24, bool))), t)
        if t % 100_000_000 == 0: buffer.append_applied_command([0., 0., 0.], t)
    p = buffer.packet(np.full((480, 640, 3), 127, np.uint8), now)
    # A raised body sees the floor farther below it while raw visual Z stays 0.
    d, _ = render_plane(np.asarray(BODY_FROM_OPTICAL), np.array([0., 0., 1.]), .32+frame*.02)
    a, _ = render_plane(body_from_optical(), np.array([0., 0., 1.]), .32+frame*.02)
    primary = from_native_depth(d, p, measured_ns=now, available_ns=now, now_ns=now)
    auxiliary = auxiliary_packet(a, p, measured_ns=now, available_ns=now, now_ns=now)
    raw, _ = fixture.joint_visual(frame, (frame*.01, 0., 0.), previous=previous)
    raw['current_pose'].update(rgb_sha256=primary['rgb_sha256'], depth_sha256=depth_hash(primary))
    return p, primary, auxiliary, raw, now


def test_all_online_consumers_share_registered_pose_and_raw_witness_survives():
    c = JointFloorRegisteredRoundTripController(None, None, public_mission=dict(
        goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    previous = None
    for frame in range(2):
        p, d, a, raw, now = packets(frame, previous)
        before = deepcopy(raw)
        c.motion = SimpleNamespace(observe=lambda *args, **kwargs: raw)
        result = c.observe(p, d, None, now_ns=now, auxiliary_depth=a)
        assert result['terminal'] is None, result['failure']
        e = result['evidence']; position, R, _ = current_joint_floor_registered_pose(e, identity=(0, 0, 0), now_ns=now)
        np.testing.assert_allclose(position, [frame*.01, 0., frame*.02], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(c.memory.position, position)
        np.testing.assert_array_equal(c.memory.rotation, R)
        np.testing.assert_array_equal(c.residual.pose['position'], position)
        np.testing.assert_array_equal(c.residual.pose['rotation'], R)
        assert result['observed_goal_distance_m'] == np.linalg.norm(position[:2]-[1., 0.])
        assert c.selector.residual is c.residual and len(c.memory.route) == frame+1
        assert c.memory.partition.total_returns == sum(c.memory.index.sample_counts.values())
        assert c.memory.auxiliary_partition.total_returns == sum(c.memory.auxiliary_index.sample_counts.values())
        assert e['original_visual_evidence'] == raw == before
        with pytest.raises(ValueError): current_joint_pose(e, identity=(0, 0, 0), now_ns=now)
        current_joint_pose(raw, identity=(0, 0, 0), now_ns=now)
        previous = raw
    # Duplicated observation cannot mutate the pose/map or keep issuing motion.
    count = c.memory.partition.total_returns
    result = c.observe(p, d, None, now_ns=now, auxiliary_depth=a)
    assert result['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and result['requested_command'] == [0., 0., 0.]
    assert c.registration.failed and c.memory.partition.total_returns == count


@pytest.mark.parametrize('fault', ['pose', 'correction', 'raw_witness', 'camera_hash', 'pair', 'reference', 'clock', 'identity'])
def test_composition_and_causal_witness_tampering_rejected(fault):
    p, d, a, raw, now = packets()
    e = JointFloorRegistration().observe(p, d, a, raw, now_ns=now)
    if fault == 'pose': e['current_pose']['position_initial_body_m'][2] = .01
    elif fault == 'correction': e['floor_registration']['correction']['normal_translation_correction_m'] = .01
    elif fault == 'raw_witness': e['original_visual_evidence']['current_pose']['position_initial_body_m'][2] = .01
    elif fault == 'camera_hash': e['floor_registration']['primary_depth_sha256'] = 'a'*64
    elif fault == 'pair': e['floor_registration']['joint_plane']['offset_body_m'] += .01
    elif fault == 'reference': e['floor_registration']['reference']['frame'] = 1
    elif fault == 'clock': e['decision_ns'] += 1
    else: e['identity'] = (0, False, 0)
    with pytest.raises(ValueError): current_joint_floor_registered_pose(e, identity=(0, 0, 0), now_ns=now)


def test_missing_auxiliary_or_visual_rgb_mismatch_latches_without_admitting_reference():
    p, d, a, raw, now = packets()
    for faulty_a, faulty_raw in ((None, raw), (a, deepcopy(raw))):
        state = JointFloorRegistration()
        if faulty_a is not None: faulty_raw['current_pose']['rgb_sha256'] = 'a'*64
        with pytest.raises((ValueError, TypeError)): state.observe(p, d, faulty_a, faulty_raw, now_ns=now)
        assert state.failed and state.reference is None and state.frame == -1


def test_three_pose_consumer_derivatives_only_change_admission_accessor():
    def method(path, cls, name):
        module = ast.parse(Path(path).read_text())
        klass = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'current_joint_floor_registered_pose': node.id = 'current_joint_pose'
            return node
    for path, old, new, name in (
        ('joint_visual_surface_memory', 'JointVisualSurfaceMemory', 'JointFloorRegisteredSurfaceMemory', 'observe'),
        ('online_executed_residual', 'OnlineExecutedResidual', 'JointFloorRegisteredResidual', 'observe'),
        ('observed_round_trip_controller', 'ObservedRoundTripController', 'JointFloorRegisteredRoundTripController', 'advance')):
        a = method('lewm/'+path+'_development.py', old, name)
        b = Normalize().visit(method('lewm/joint_floor_registered_controller_development.py', new, name))
        assert ast.dump(a) == ast.dump(b)
