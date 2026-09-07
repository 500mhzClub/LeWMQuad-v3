"""Narrow geometry-free logger correction, including the actual wrapper chain."""
import ast
from copy import deepcopy
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.fast_gyro_development import FastGyroBuffer
from lewm.physical_execution_development import build_case
from lewm.simulated_body_observation_development import IdealBodySensor, BodyObservationBuffer, JOINT_NAMES
from lewm.simulated_fast_gyro_development import IdealFastGyro
from scripts.whole_task_physics_session_development import GeometryFreePhysicalSample, WholeTaskPhysicsSession, BASE
from scripts.fast_gyro_scan_session_development import FastGyroSession
from scripts.run_go2_contact_attributed_execution_development_v1 import AttributedSession, PhysicalStop
from scripts.run_go2_causal_rgb_body_capture_development_v1 import ObservationSession
from scripts.run_go2_local_control_factorial_development_v1 import FactorialSession
from scripts.run_go2_multijunction_route_development_v1 import RouteSession
from scripts import run_go2_whole_task_navigation_development_v1 as original
from scripts import run_go2_whole_task_navigation_sampling_correction_development_v1 as corrected
from scripts import audit_go2_whole_task_navigation_development_v1 as original_audit
from scripts import audit_go2_whole_task_navigation_sampling_correction_development_v1 as corrected_audit

REGIONS = {'source_region_member', 'correct_edge_region_member', 'edge_region_member',
           'wrong_edge_region_member', 'target_region_member'}


def tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def test_only_legacy_annotation_removed_from_native_sampling():
    old = tree(BASE._GenesisPhysicalSession._sample)
    function = old.body[0]
    function.body = [n for n in function.body if not (isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {'xy', 'source', 'target', 'correct', 'wrong'})]
    result = function.body[-1].value
    pairs = [(k, v) for k, v in zip(result.keys, result.values) if k.value not in REGIONS]
    result.keys, result.values = map(list, zip(*pairs))
    assert ast.dump(old) == ast.dump(tree(GeometryFreePhysicalSample._sample))
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'geometry' for n in ast.walk(tree(GeometryFreePhysicalSample._sample)))


def test_settling_gait_timing_and_reset_path_unchanged_except_returned_field_inventory():
    old = tree(BASE._GenesisPhysicalSession.execute_requested_ticks)
    changes = 0
    for node in ast.walk(old):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'members':
            assert isinstance(node.value, ast.Name) and node.value.id == 'CANDIDATE_TRACE_MEMBERS'
            node.value = ast.parse('tuple(rows[0])', mode='eval').body
            changes += 1
    assert changes == 1 and ast.dump(old) == ast.dump(tree(GeometryFreePhysicalSample.execute_requested_ticks))
    assert WholeTaskPhysicsSession.settle_recorded is AttributedSession.settle_recorded
    assert WholeTaskPhysicsSession.command_tick is RouteSession.command_tick


def test_c3_insertion_preserves_all_live_sensor_and_safety_wrappers():
    assert WholeTaskPhysicsSession.__mro__ == (WholeTaskPhysicsSession, FastGyroSession, RouteSession,
        ObservationSession, FactorialSession, AttributedSession, GeometryFreePhysicalSample,
        BASE._GenesisPhysicalSession, object)


def native_session(*, contact=False, height=.32):
    session = object.__new__(WholeTaskPhysicsSession)
    class ForbiddenGeometry:
        def __getitem__(self, key): raise AssertionError('native trace cannot read route geometry')
    session.geometry = ForbiddenGeometry()
    robot = SimpleNamespace(get_pos=lambda: np.array([.1, -.2, height]),
        get_quat=lambda: np.array([1., 0., 0., 0.]), get_vel=lambda: np.zeros(3),
        get_ang=lambda: np.array([0., 0., .1]),
        get_dofs_position=lambda _: np.linspace(-.1, .1, 12), get_dofs_velocity=lambda _: np.zeros(12))
    session.ctx = SimpleNamespace(runner=SimpleNamespace(_as_np=np.asarray, _leg_dof_idx=np.arange(12)),
                                  build=SimpleNamespace(robot=robot))
    session.phase = session.edge_index = 0
    session.samples = []; session.sensor_rows = []; session.fast_rows = []
    session.sensor = IdealBodySensor(); session.fast_sensor = IdealFastGyro()
    session.observations = BodyObservationBuffer((0, 0, 0)); session.fast_buffer = FastGyroBuffer((0, 0, 0))
    session.joint_names = JOINT_NAMES
    session.contact_calls = []
    def contacts():
        session.contact_calls.append(session.sample_time)
        return contact
    session._disallowed_contact = contacts
    return session


def test_native_state_and_contact_readback_match_original_on_all_common_fields():
    old = native_session(); old.geometry = build_case('straight', 1.)['geometry']; old.sample_time = .02
    expected = BASE._GenesisPhysicalSession._sample(old, [.2, 0., .1], [.15, 0., .1], .02)
    new = native_session(); new.sample_time = .02
    actual = GeometryFreePhysicalSample._sample(new, [.2, 0., .1], [.15, 0., .1], .02)
    assert set(actual) == set(expected)-REGIONS
    for key in actual: np.testing.assert_array_equal(actual[key], expected[key])
    assert new.contact_calls == old.contact_calls == [.02]


@pytest.mark.parametrize('contact,height,reason', [(False, .32, None), (True, .32, 'DISALLOWED_CONTACT'),
                                                  (False, .10, 'BODY_STABILITY_LIMIT')])
def test_full_sample_chain_keeps_terminal_rows_and_both_sensor_streams(contact, height, reason):
    session = native_session(contact=contact, height=height)
    if reason is None:
        session._sample([0., 0., 0.], [0., 0., 0.], .02)
    else:
        with pytest.raises(PhysicalStop, match=reason): session._sample([0., 0., 0.], [0., 0., 0.], .02)
    assert session.contact_calls == [.02]
    assert len(session.samples) == len(session.sensor_rows) == len(session.fast_rows) == 1
    assert session.fast_rows[0]['measured_ns'] == session.sensor_rows[0]['measured_ns'] == 20_000_000
    assert session.samples[0]['phase'] == session.samples[0]['edge_index'] == 0
    assert not (set(session.samples[0]) & REGIONS)


def test_all_native_steps_retained_with_unchanged_ordinary_and_command_cadence():
    session = native_session()
    for step in range(1, 51): session._sample([0., 0., 0.], [0., 0., 0.], step*.002)
    assert len(session.samples) == len(session.fast_rows) == len(session.contact_calls) == 50
    assert [int(r['measured_ns']) for r in session.sensor_rows] == list(range(20_000_000, 100_000_001, 20_000_000))
    assert [int(r['measured_ns']) for r in session.fast_rows] == list(range(2_000_000, 100_000_001, 2_000_000))


def test_collector_and_audit_science_bodies_are_identical_with_separate_bound_root():
    assert ast.dump(tree(corrected.collect)) == ast.dump(tree(original.collect))
    assert ast.dump(tree(corrected_audit.audit_trial)) == ast.dump(tree(original_audit.audit_trial))
    assert ast.dump(tree(corrected_audit.check_static_objects)) == ast.dump(tree(original_audit.check_static_objects))
    assert corrected.FastGyroSession is WholeTaskPhysicsSession
    assert original.FastGyroSession is FastGyroSession
    assert corrected.OUTPUT != original.OUTPUT and corrected_audit.OUTPUT == corrected.OUTPUT
    assert corrected.trial_specs() == original.trial_specs()
    assert corrected.WholeTaskNavigation is original.WholeTaskNavigation
    assert corrected.reduce_whole_task is original.reduce_whole_task
