"""Synthetic state/sensor/branch checks, not physical continuation evidence."""
import copy
from dataclasses import asdict
import math

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.continuation_branch_development import observed_branch, choose_side_branch
from lewm.observed_continuation_development import ObservedContinuation
from lewm.observed_continuation_scene_development import trials
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.rgb_exit_candidates_development import ExitCandidate
from lewm.tests.test_observed_traversal_controller_development import Stream
from lewm.tests.test_relative_gyro_turn_development import packet
from scripts.analyze_go2_ground_plane_development_v1 import URDF


class MovingStream(Stream):
    def frame(self, tick, applied=(0., 0., 0.), *, changed=False, quiet=True, floor=True):
        if not tick: return super().frame(tick, applied, changed=changed, quiet=quiet, floor=floor)
        rate = np.array([0., 0., applied[2]])
        for step in range(800+(tick-1)*50+1, 801+tick*50):
            ns = step*2_000_000
            self.fast.append(rate, np.ones(3, bool), measured_ns=ns, available_ns=ns)
            if step % 10 == 0:
                self.slow.append_sensors({'gyro': (rate, np.ones(3, bool)),
                    'specific_force': (np.array([0., 0., 9.81]), np.ones(3, bool)),
                    'joints': (np.zeros(24), np.ones(24, bool))}, ns)
            if step % 50 == 0: self.slow.append_applied_command(applied, ns)
        p = packet(self.slow, 80+tick*5)
        p['image']['rgb'][:] = [115, 120, 108] if floor else [100, 100, 100]
        if changed: p['image']['rgb'][:, :320] = 100
        now = p['image']['measured_ns']
        return p, self.fast.packet(now_ns=now), now


def branch(angle, timestamp=10):
    candidate = asdict(ExitCandidate('image:proposal-0', timestamp, 0., -.1, .1, 4, 12, 3))
    return observed_branch(candidate, rotation_increment([0., 0., angle]), decision_ns=timestamp)


def test_side_branch_uses_observed_bearings_and_excludes_forward_reverse():
    observations = [branch(0.), branch(math.pi), branch(-math.pi/2), branch(math.pi/2)]
    result = choose_side_branch(observations, [1., 0., 0.], now_ns=100)
    assert result['relative_to_incoming_rad'] == pytest.approx(math.pi/2)
    assert result['requires_fresh_forward_reobservation'] and result['place_identity'] is None
    result['candidate']['support_points'] = 999
    assert observations[-1]['candidate']['support_points'] == 12
    assert choose_side_branch(observations[:2], [1., 0., 0.], now_ns=100) is None
    result = choose_side_branch(observations[:3], [1., 0., 0.], now_ns=100)
    assert result['relative_to_incoming_rad'] == pytest.approx(-math.pi/2)


@pytest.mark.parametrize('fault', ['future', 'too_old', 'qualified', 'invalid_rotation'])
def test_invalid_branch_evidence_is_not_a_navigation_target(fault):
    item = branch(math.pi/2)
    if fault == 'future': item['observed_ns'] = 101
    elif fault == 'too_old': item['observed_ns'] = -30_000_000_000
    elif fault == 'qualified': item['qualified_exit'] = True
    else:
        with pytest.raises(SensorContractError): observed_branch(item['candidate'], np.zeros((3, 3)), decision_ns=10)
        return
    with pytest.raises(SensorContractError): choose_side_branch([item], [1., 0., 0.], now_ns=100)


def synthetic_continuation(*, scan_floor=True, second_floor=True):
    controller = ObservedContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream, command, rows = MovingStream(), [0., 0., 0.], []
    for tick in range(801):
        child = controller.first if controller.stage == 'FIRST' else controller.second if controller.stage == 'SECOND' else None
        changed = child is not None and child.tick >= 3
        floor = scan_floor if controller.stage == 'SCAN' else second_floor if controller.stage == 'SECOND' else True
        p, fast, now = stream.frame(tick, command, changed=changed, floor=floor)
        row = controller.observe(p, fast, now_ns=now)
        rows.append(row)
        if row['terminal']: return controller, rows
        command = row['requested_command']
    raise AssertionError('unbounded continuation')


def test_two_traversals_use_continuous_sensing_real_warmups_and_no_trusted_edges():
    controller, rows = synthetic_continuation()
    assert rows[-1]['status'] == 'COMPLETE_PROVISIONAL'
    stages = [row['stage'] for row in rows]
    assert list(dict.fromkeys(stages)) == ['FIRST', 'HOLD_SCAN', 'SCAN', 'HOLD_ALIGN', 'ALIGN', 'HOLD_SECOND', 'SECOND']
    for i, row in enumerate(rows):
        assert row['global_orientation']['samples_integrated'] == i*50
        assert row['trusted_graph_edges'] == 0
        if row['stage'].startswith('HOLD'): assert row['requested_command'] == [0., 0., 0.]
    for stage in ('FIRST', 'SECOND'):
        children = [row['child'] for row in rows if row['stage'] == stage]
        assert [c['status'] for c in children[:4]] == ['WARMUP']*3+['TRAVERSING']
        assert children[-1]['status'] == 'ARRIVAL_CANDIDATE'
    selected = next(row['selected_side_branch'] for row in rows if row['selected_side_branch'])
    assert selected['relative_to_incoming_rad'] > math.pi/4
    records = controller.ledgers()
    assert len(records) == 2 and all(r['record']['trusted_graph_edges'] == 0 for r in records)
    records[0]['record']['arrival']['place_identity'] = 'made-up'
    assert controller.ledgers()[0]['record']['arrival']['place_identity'] is None


@pytest.mark.parametrize('scan_floor,second_floor,status', [
    (False, True, 'FAILED_NO_SIDE_BRANCH'), (True, False, 'FAILED_SECOND_NO_EXIT')])
def test_missing_scan_or_fresh_exit_evidence_stops_continuation(scan_floor, second_floor, status):
    controller, rows = synthetic_continuation(scan_floor=scan_floor, second_floor=second_floor)
    assert rows[-1]['status'] == status and rows[-1]['requested_command'] == [0., 0., 0.]
    assert controller.first.ledger.snapshot()['status'] == 'ARRIVAL_CANDIDATE'
    assert controller.second is None or controller.second.ledger.snapshot() is None


def test_history_integrity_and_physical_stop_preserve_first_provisional_record():
    controller = ObservedContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream = MovingStream()
    for tick in range(4):
        p, fast, now = stream.frame(tick)
        controller.observe(p, fast, now_ns=now)
    p, fast, now = stream.frame(4)
    p['sensor_state']['control']['applied_command']['values'][-2, 0] = .1
    with pytest.raises(SensorContractError): controller.observe(p, fast, now_ns=now)
    assert controller.status == controller.first.ledger.snapshot()['status'] == 'FAILED_SENSOR'
    with pytest.raises(SensorContractError): controller.observe(p, fast, now_ns=now)
    controller, rows = synthetic_continuation()
    before = copy.deepcopy(controller.ledgers())
    controller.finish_physical_stop()
    assert controller.status == 'PHYSICAL_STOP' and controller.ledgers() == before


def test_fixture_methods_share_geometry_but_never_runtime_destination_labels():
    specs = trials()
    assert len(specs) == 16 and len({s['scene_id'] for s in specs}) == 16
    for case in range(4):
        group = [s for s in specs if s['case_index'] == case]
        assert [s['method'] for s in group] == ['fixed_forward', 'direct_direct', 'supervised_rollout', 'jepa_rollout']
        assert all(s['geometry'] == group[0]['geometry'] for s in group)
        assert all(s['evaluation_leg_geometries'] == group[0]['evaluation_leg_geometries'] for s in group)
