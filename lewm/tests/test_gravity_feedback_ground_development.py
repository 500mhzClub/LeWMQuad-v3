import copy

import numpy as np
import pytest

from lewm.causal_ground_plane_development import CausalGroundPlane
from lewm.causal_sensor_state import SensorContractError
from lewm.gravity_feedback_ground_development import (
    force_in_current_frame, force_observation, CausalGravityFeedbackGround, MODES)
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_relative_gyro_turn_development import initialized, packet


def advance(buffer, step, gyro=(0., 0., 0.), force=(0., 0., 9.81), command=(0., 0., 0.)):
    now = step * 20_000_000
    buffer.append_sensors({'gyro': (np.asarray(gyro), np.ones(3, bool)),
                          'specific_force': (np.asarray(force), np.ones(3, bool)),
                          'joints': (np.zeros(24), np.ones(24, bool))}, now)
    if step % 5 == 0:
        buffer.append_applied_command(command, now)


def test_transport_averages_vectors_in_current_frame_not_in_twenty_different_frames():
    rate = np.array([.4, -.3, .2])
    rotations = [rotation_increment(rate * .02 * i) for i in range(20)]
    force = np.stack([r.T @ [0., 0., 9.81] for r in rotations])
    aligned = force_in_current_frame(force, np.broadcast_to(rate, (20, 3)))
    truth = rotations[-1].T @ [0., 0., 9.81]
    np.testing.assert_allclose(aligned, np.broadcast_to(truth, (20, 3)), rtol=0, atol=1e-12)
    assert np.linalg.norm(force.mean(0) - truth) > .5


@pytest.mark.parametrize('mode', MODES)
def test_stationary_initialization_shared_and_stable_without_false_qualification(mode):
    buffer = initialized()
    p = packet(buffer, 80)
    model = CausalGravityFeedbackGround(mode)
    state = model.begin(p, now_ns=1_600_000_000)
    assert state['body_origin_height_m'] == pytest.approx(.448)
    for step in range(81, 86):
        advance(buffer, step)
    state = model.step(packet(buffer, 85), now_ns=1_700_000_000)
    np.testing.assert_allclose(state['up_current_body'], [0, 0, 1], atol=1e-12)
    assert not state['ground_plane_qualified']
    assert state['feedback_applied'] == (mode != 'gyro_only')
    assert not state['feedback']['acceleration_separation_qualified']
    state['up_current_body'][0] = 8
    assert model.snapshot(now_ns=1_700_000_000)['up_current_body'][0] == 0


def test_gyro_only_mode_matches_frozen_baseline_exactly():
    buffer = initialized((.1, .2, .3))
    a, b = CausalGroundPlane(), CausalGravityFeedbackGround('gyro_only')
    a.begin(packet(buffer, 80), now_ns=1_600_000_000)
    b.begin(packet(buffer, 80), now_ns=1_600_000_000)
    for tick in range(1, 6):
        for step in range(80 + (tick - 1) * 5 + 1, 81 + tick * 5):
            advance(buffer, step, gyro=(.1, .2, .3))
        p = packet(buffer, 80 + tick * 5)
        x = a.step(p, now_ns=p['image']['measured_ns'])
        y = b.step(p, now_ns=p['image']['measured_ns'])
        for field in ('up_current_body', 'body_origin_height_m', 'per_foot_support_height_m'):
            assert x[field] == y[field]


@pytest.mark.parametrize('mode', ['body_mean_feedback', 'transported_feedback'])
def test_quiet_gravity_feedback_reduces_synthetic_gyro_bias_drift(mode):
    buffer = initialized((0., .02, 0.))
    model = CausalGravityFeedbackGround(mode)
    model.begin(packet(buffer, 80), now_ns=1_600_000_000)
    for tick in range(1, 101):
        for step in range(80 + (tick - 1) * 5 + 1, 81 + tick * 5):
            advance(buffer, step, gyro=(0., .02, 0.))
        state = model.step(packet(buffer, 80 + tick * 5), now_ns=(80 + tick * 5) * 20_000_000)
    angle = np.arccos(np.clip(state['up_current_body'][2], -1, 1))
    assert angle < .06
    assert np.arccos(model.baseline.snapshot(now_ns=model.last_ns)['up_current_body'][2]) == pytest.approx(.2)


@pytest.mark.parametrize('fault,reason', [('invalid', 'force_unavailable'), ('magnitude', 'mean_force_not_near_gravity'),
                                        ('variance', 'force_history_not_quiet'), ('command', 'recent_command_change')])
def test_unreliable_force_is_rejected_without_being_mislabeled_as_zero_gravity(fault, reason):
    p = packet(initialized(), 80)
    force = p['sensor_state']['sensed']['specific_force']
    if fault == 'invalid':
        force['valid'][-1] = False
        force['values'][-1] = 0
    if fault == 'magnitude':
        force['values'][:] = [0, 0, 2]
    if fault == 'variance':
        force['values'][::2, 0] = 4
        force['values'][1::2, 0] = -4
    if fault == 'command':
        p['sensor_state']['control']['applied_command']['values'][-1, 0] = .3
    result = force_observation(p, 'transported_feedback')
    assert not result['accepted'] and result['reason'] == reason


def test_constant_unknown_acceleration_is_explicitly_an_unresolved_observability_limit():
    p = packet(initialized(), 80)
    p['sensor_state']['sensed']['specific_force']['values'][:] = [2, 0, 9.81]
    result = force_observation(p, 'transported_feedback')
    assert result['accepted']  # quiet, near-g magnitude is not proof of zero acceleration
    assert result['mean_force_current_body'][0] == 2
    assert not result['acceleration_separation_qualified']


@pytest.mark.parametrize('fault', ['force', 'command', 'gyro', 'privilege', 'clock'])
def test_history_rewrite_or_bad_policy_input_latches_failure(fault):
    buffer = initialized()
    model = CausalGravityFeedbackGround('transported_feedback')
    model.begin(packet(buffer, 80), now_ns=1_600_000_000)
    for step in range(81, 86):
        advance(buffer, step)
    p = packet(buffer, 85)
    if fault == 'force':
        p['sensor_state']['sensed']['specific_force']['values'][-6, 0] = .1
    if fault == 'command':
        p['sensor_state']['control']['applied_command']['values'][-2, 0] = .1
    if fault == 'gyro':
        p['sensor_state']['sensed']['gyro']['values'][-6, 0] = .1
    if fault == 'privilege':
        p['world_pose'] = [0] * 7
    with pytest.raises(SensorContractError):
        model.step(p, now_ns=1_700_000_001 if fault == 'clock' else 1_700_000_000)
    assert model.status == 'FAILED_SENSOR'
    with pytest.raises(SensorContractError):
        model.snapshot(now_ns=1_600_000_000)


def test_bad_mode_and_stale_query_rejected():
    with pytest.raises(ValueError):
        CausalGravityFeedbackGround('tuned')
    model = CausalGravityFeedbackGround('gyro_only')
    model.begin(packet(initialized(), 80), now_ns=1_600_000_000)
    with pytest.raises(SensorContractError):
        model.snapshot(now_ns=1_700_000_000)
