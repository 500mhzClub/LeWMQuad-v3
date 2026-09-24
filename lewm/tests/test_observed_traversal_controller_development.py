"""Synthetic integration checks, not physical navigation evidence."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

import lewm.observed_traversal_controller_development as runtime
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastGyroBuffer
from lewm.provisional_traversal_ledger_development import ProvisionalTraversalLedger
from lewm.rgb_exit_candidates_development import ExitCandidate
from lewm.observed_traversal_scene_development import trials, METHODS
from lewm.tests.test_relative_gyro_turn_development import initialized, packet
from scripts.analyze_go2_ground_plane_development_v1 import URDF


class Stream:
    def __init__(self):
        self.slow = initialized()
        self.fast = FastGyroBuffer((0, 0, 0))
        for step in range(750, 801): self.fast.append([0., 0., 0.], np.ones(3, bool),
            measured_ns=step*2_000_000, available_ns=step*2_000_000)

    def frame(self, tick, applied=(0., 0., 0.), *, changed=False, quiet=True, floor=True):
        if tick:
            for step in range(800+(tick-1)*50+1, 801+tick*50):
                ns = step*2_000_000
                self.fast.append([0., 0., 0.], np.ones(3, bool), measured_ns=ns, available_ns=ns)
                if step % 10 == 0:
                    joints = np.zeros(24)
                    if not quiet: joints[12:] = 2.
                    self.slow.append_sensors({'gyro': (np.zeros(3), np.ones(3, bool)),
                        'specific_force': (np.array([0., 0., 9.81]), np.ones(3, bool)),
                        'joints': (joints, np.ones(24, bool))}, ns)
                if step % 50 == 0: self.slow.append_applied_command(applied, ns)
        p = packet(self.slow, 80+tick*5)
        p['image']['rgb'][:] = [115, 120, 108] if floor else [100, 100, 100]
        if changed: p['image']['rgb'][:, :320] = 100
        now = p['image']['measured_ns']
        return p, self.fast.packet(now_ns=now), now


def controller(method='directional_gait', template=None):
    return runtime.ObservedTraversalController(method, ArticulatedCollisionGeometry(URDF), template)


def run_synthetic(method='directional_gait', *, changed=True, quiet=True, applied_override=None):
    model, stream = controller(method), Stream()
    rows, applied = [], [0., 0., 0.]
    for tick in range(145):
        p, fast, now = stream.frame(tick, applied, changed=changed and tick >= 4, quiet=quiet)
        row = model.observe(p, fast, now_ns=now)
        rows.append(row)
        if row['terminal']: return model, rows
        applied = row['requested_command'] if applied_override is None or tick < 3 else applied_override
    raise AssertionError('controller exceeded bounded traversal/braking budget')


def test_synthetic_applied_command_progress_can_only_produce_unqualified_arrival():
    model, rows = run_synthetic()
    assert [r['status'] for r in rows[:4]] == ['WARMUP']*3+['TRAVERSING']
    assert rows[-1]['status'] == 'ARRIVAL_CANDIDATE'
    assert all(r['requested_command'] == [0., 0., 0.] for r in rows if r['status'] != 'TRAVERSING')
    record = model.ledger.snapshot()
    assert record['trusted_graph_edges'] == 0 and record['qualified_traversal'] is False
    assert record['arrival']['place_identity'] is None and record['arrival']['qualified_arrival'] is False
    # No translation sensor exists in this stream: applied commands alone are
    # enough to fool the progress proxy. This test documents that limitation.
    assert rows[-1]['command_progress_proxy_m'] >= rows[-1]['required_progress_m']
    record['arrival']['place_identity'] = 'invented'
    assert model.ledger.snapshot()['arrival']['place_identity'] is None


@pytest.mark.parametrize('method,changed,quiet,override,status', [
    ('always_stop', True, True, None, 'FAILED_TIMEOUT'),
    ('directional_gait', False, True, None, 'FAILED_NO_VISUAL_CHANGE'),
    ('directional_gait', True, False, None, 'FAILED_SETTLING'),
    ('directional_gait', True, True, [0., 0., 0.], 'FAILED_TIMEOUT')])
def test_explicit_bounded_failures(method, changed, quiet, override, status):
    model, rows = run_synthetic(method, changed=changed, quiet=quiet, applied_override=override)
    assert rows[-1]['status'] == model.ledger.snapshot()['status'] == status
    assert rows[-1]['requested_command'] == [0., 0., 0.]
    assert model.ledger.snapshot()['arrival'] is None


def test_no_floor_evidence_fails_before_creating_an_attempt():
    model, stream = controller(), Stream()
    for tick in range(4):
        p, fast, now = stream.frame(tick, floor=False)
        result = model.observe(p, fast, now_ns=now)
    assert result['status'] == 'FAILED_NO_EXIT' and result['ledger'] is None
    with pytest.raises(SensorContractError): model.observe(p, fast, now_ns=now)


def test_unavailable_joint_speed_is_not_evidence_of_quiet_body():
    model, stream = controller(), Stream()
    for tick in range(4):
        p, fast, now = stream.frame(tick)
        if tick == 3:
            p['sensor_state']['sensed']['joints']['valid'][-5:, 12:] = False
        result = model.observe(p, fast, now_ns=now)
    assert result['status'] == 'TRAVERSING'
    assert result['quiet_body_proxy'] is False


@pytest.mark.parametrize('fault', ['world_pose', 'command_rewrite', 'fast_invalid', 'clock'])
def test_sensor_fault_latches_and_fails_pending_attempt(fault):
    model, stream = controller(), Stream()
    for tick in range(4):
        p, fast, now = stream.frame(tick)
        model.observe(p, fast, now_ns=now)
    p, fast, now = stream.frame(4)
    if fault == 'world_pose': p['world_pose'] = [0.]*7
    elif fault == 'command_rewrite': p['sensor_state']['control']['applied_command']['values'][-2, 0] = .1
    elif fault == 'fast_invalid': fast['valid'][-3] = False
    elif fault == 'clock': now += 1
    with pytest.raises(SensorContractError): model.observe(p, fast, now_ns=now)
    assert model.status == model.ledger.snapshot()['status'] == 'FAILED_SENSOR'
    with pytest.raises(SensorContractError): model.observe(p, fast, now_ns=now)


def test_learned_adapter_is_observed_every_tick_but_selects_only_every_half_second(monkeypatch):
    class Adapter:
        def __init__(self, *args): self.observations, self.selections, self.start = [], [], None
        def begin_episode(self, identity): assert identity == (0, 0, 0)
        def observe(self, p, *, now_ns): self.observations.append(now_ns)
        def begin_control(self, cue, *, now_ns):
            assert np.linalg.norm(cue) == pytest.approx(.8)
            self.start = now_ns
        def select(self, *, now_ns):
            self.selections.append(now_ns)
            return {'requested_command_tape': [[.3, 0., 0.]]*5}
    monkeypatch.setattr(runtime, 'OnlineTemporalChoice', Adapter)
    model = controller('jepa_rollout', SimpleNamespace(method='jepa_rollout', models=(), bindings=()))
    stream, command, rows = Stream(), [0., 0., 0.], []
    for tick in range(145):
        p, fast, now = stream.frame(tick, command, changed=tick >= 4)
        result = model.observe(p, fast, now_ns=now)
        rows.append(result)
        if result['terminal']: break
        command = result['requested_command']
    assert result['status'] == 'ARRIVAL_CANDIDATE'
    assert len(model.adapter.observations) == len(rows)
    assert np.diff(model.adapter.selections).tolist() == [500_000_000]*(len(model.adapter.selections)-1)
    assert model.adapter.start == rows[3]['decision_ns']
    braking = next(r for r in rows if r['status'] == 'BRAKING')
    assert (braking['decision_ns']-model.adapter.start) % 500_000_000 == 0
    assert model.adapter.selections[-1] < braking['decision_ns']


def proposal():
    return asdict(ExitCandidate('frame:proposal-0', 10, 0., -.1, .1, 4, 12, 3))


@pytest.mark.parametrize('fault', ['stale', 'qualified', 'wrong_image', 'missing_field'])
def test_ledger_rejects_misattributed_or_qualified_proposals(fault):
    candidate = proposal()
    if fault == 'stale': candidate['timestamp_ns'] -= 1
    if fault == 'qualified': candidate['qualified_exit'] = True
    if fault == 'wrong_image': candidate['observation_id'] = 'other:proposal-0'
    if fault == 'missing_field': candidate.pop('support_points')
    ledger = ProvisionalTraversalLedger()
    with pytest.raises((ValueError, TypeError)): ledger.begin(observation_id='frame', decision_ns=10, candidate=candidate)
    assert ledger.snapshot() is None


def test_ledger_rejects_missing_arrival_and_cannot_finish_twice():
    ledger = ProvisionalTraversalLedger()
    ledger.begin(observation_id='frame', decision_ns=10, candidate=proposal())
    with pytest.raises(ValueError): ledger.finish(status='ARRIVAL_CANDIDATE')
    with pytest.raises(ValueError): ledger.finish(status='PHYSICAL_STOP', arrival={'qualified_arrival': True})
    assert ledger.snapshot()['status'] == 'PENDING'
    ledger.finish(status='PHYSICAL_STOP')
    with pytest.raises(ValueError): ledger.finish(status='FAILED_TIMEOUT')


def test_four_fixtures_pair_all_five_methods_without_runtime_graph():
    rows = trials()
    assert len(rows) == 20 and len({r['scene_id'] for r in rows}) == 20
    for case in range(4):
        group = [r for r in rows if r['case_index'] == case]
        assert tuple(r['method'] for r in group) == METHODS
        assert all(r['geometry'] == group[0]['geometry'] for r in group)
        assert len({r['procedural_seed'] for r in group}) == 1
        assert all(not {'route_geometries', 'route_cells', 'graph_connections'} & r.keys() for r in group)
    original = copy.deepcopy(rows[1]['geometry'])
    rows[0]['geometry']['spawn_se2_world'][0] = 99
    assert rows[1]['geometry'] == original
