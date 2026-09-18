"""Synthetic controller causality and arrival tests; no navigation claim."""
from copy import deepcopy
import numpy as np
import pytest

from lewm import learned_goal_probe_development as mission
from lewm.tests.test_joint_pulse_execution_development import joint_visual
from scripts.learned_goal_probe_audit_development import native_goal


def shift_clock(value):
    if isinstance(value, dict):
        return {k: v-100_000_000 if k.endswith('_ns') and (type(v) is int or isinstance(v, np.ndarray)) else shift_clock(v)
            for k, v in value.items()}
    if isinstance(value, list):
        return [shift_clock(v) for v in value]
    return value


class Fixture:
    def __init__(self, monkeypatch):
        self.calls = []
        self.previous = None
        self.frame = 0
        self.choice = 'left_arc'
        self.controller = mission.LearnedGoalProbe(object())
        def history(packets, now):
            assert [p['frame'] for p in packets] == list(range(self.frame-3, self.frame+1))
            return dict(frames=[p['frame'] for p in packets])
        def select(model, observed, **kwargs):
            self.calls.append(dict(history=deepcopy(observed), **kwargs))
            return dict(action=self.choice, selection_wall_ms=0.)
        monkeypatch.setattr(mission, 'causal_history_tensors', history)
        monkeypatch.setattr(mission, 'select', select)

    def step(self, state=(0., 0., 0.), fault=None):
        evidence, now = joint_visual(self.frame, state, self.previous)
        self.previous = deepcopy(evidence)
        evidence = shift_clock(evidence)
        now -= 100_000_000
        if fault is not None:
            fault(evidence)
        result = self.controller.advance(dict(frame=self.frame), evidence, now_ns=now)
        self.frame += 1
        return result


def test_replanning_uses_moving_past_packets_and_changed_predictions(monkeypatch):
    f = Fixture(monkeypatch)
    for i in range(8):
        row = f.step((.01*i, 0., 0.)) if i else f.step()
        assert row['terminal'] is None
        assert row['requested_command'] == ([0., 0., 0.] if i < 3 else [.16, 0., .45])
    assert len(f.calls) == 1 and f.calls[0]['history']['frames'] == [0, 1, 2, 3]
    f.choice = 'right_arc'
    row = f.step((.08, 0., np.pi/2))
    assert row['requested_command'] == [.16, 0., -.45]
    assert f.calls[1]['history']['frames'] == [5, 6, 7, 8]
    np.testing.assert_allclose(f.calls[1]['goal_body_xy_m'], [0., -1.12], atol=1e-12)


@pytest.mark.parametrize('fault', ['missing', 'frame', 'clock', 'identity', 'witness'])
def test_committed_command_is_cancelled_by_bad_current_pose_and_failure_latches(monkeypatch, fault):
    f = Fixture(monkeypatch)
    for _ in range(4):
        f.step()
    def corrupt(e):
        if fault == 'missing': e['current_pose'] = None
        if fault == 'frame': e['current_pose']['frame'] += 1
        if fault == 'clock': e['current_pose']['measured_ns'] -= 1
        if fault == 'identity': e['identity'] = (1, 0, 0)
        if fault == 'witness': e['continuity_evidence']['rotation_measurement_witnesses'] = []
    row = f.step(fault=corrupt)
    assert row['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and row['requested_command'] == [0., 0., 0.]
    assert f.step()['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and len(f.calls) == 1


def test_small_progress_is_not_arrival_and_budget_stops_hold(monkeypatch):
    f = Fixture(monkeypatch)
    f.choice = 'hold'
    f.step()
    for _ in range(mission.WARMUP_TICKS + mission.NAVIGATION_TICKS):
        row = f.step((.35, 0., 0.))
    assert row['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    assert row['observed_goal_distance_m'] == pytest.approx(.85)
    assert len(f.calls) == 48


def test_arrival_requires_goal_and_zero_command_dwell(monkeypatch):
    f = Fixture(monkeypatch)
    for _ in range(4):
        f.step()
    row = f.step((1.2, 0., 0.))
    assert row['terminal'] is None and row['quiet_intervals'] == 0
    for _ in range(9):
        assert f.step((1.2, 0., 0.))['terminal'] is None
    assert f.step((1.2, 0., 0.))['terminal'] == 'OBSERVED_GOAL_CANDIDATE'
    assert len(f.calls) == 1


def test_native_readout_requires_real_goal_quiet_and_no_contact():
    poses = np.zeros((1251, 7)); poses[:, 6] = 1.
    poses[750:, 0] = 1.2
    raw = dict(base_pose_world=poses, base_twist_world=np.zeros((1251, 6)),
        requested_command=np.zeros((1251, 3)), physics_contact=np.zeros(1251, bool))
    result = dict(schedule_terminal='OBSERVED_GOAL_CANDIDATE', terminal_zero_ticks=10,
        physical_stop=None, acquisition_stop=None)
    assert native_goal(raw, result)['verified_goal_reached']
    poses[750:, 0] = .35
    assert not native_goal(raw, result)['verified_goal_reached']
    poses[750:, 0] = 1.2
    raw['physics_contact'][800] = True
    assert not native_goal(raw, result)['verified_goal_reached']
    raw['physics_contact'][800] = False
    raw['base_twist_world'][-1, 0] = .051
    assert not native_goal(raw, result)['verified_goal_reached']


@pytest.mark.parametrize('corrupt', [None, 'requested_command', 'applied_command', 'post_slew_applied_command'])
def test_online_command_audit_checks_exact_recording_representations(corrupt):
    from scripts.learned_goal_probe_audit_development import audit_commands
    from lewm.tests.test_geometry_progress_command_representation_development import recorded_trace
    raw, tape, rows, result = recorded_trace()
    for row in rows:
        row['decision']['terminal'] = None
    for i, item in enumerate(tape):
        phase, role = (1, 'causal_history_warmup') if i < 3 else (2, 'online_learned_goal_command')
        item.update(phase=phase, role=role)
        raw['phase'][item['pre_sample_index']+1:item['post_sample_index']+1] = phase
    result.update(schedule_terminal=None, terminal_zero_ticks=0, acquisition_stop='STORAGE_RESERVE_STOP')
    audit_commands(raw, tape, rows, result)
    if corrupt is not None:
        raw[corrupt][900, 2] = np.nextafter(raw[corrupt][900, 2], np.inf)
        with pytest.raises(AssertionError):
            audit_commands(raw, tape, rows, result)
    else:
        audit_commands(raw, tape, rows, result)


def test_actual_rgbd_observer_and_model_join_without_synthetic_pose_injection():
    import torch
    from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
    from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026090960)
        model = CumulativePulseRGBBodyJEPA(32).eval()
    controller = mission.LearnedGoalProbe(model)
    for p, d, f, now in packets([texture()]*4):
        row = controller.observe(shift_clock(p), shift_clock(d), shift_clock(f), now_ns=now-100_000_000)
        assert row['terminal'] is None, row['failure']
    assert row['new_selection'] is not None
    assert np.asarray(row['new_selection']['prediction']).shape == (6, 8, 5)
    assert row['evidence']['current_pose']['mode'] == 'joint'
    assert all(p.grad is None for p in model.parameters())
