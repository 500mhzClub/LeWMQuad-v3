"""Fault, disturbance and matched-source tests; not physical qualification."""
import ast
import inspect
import json
import math
import textwrap

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.initially_aligned_continuation_development import FineInitialBearingAlignment
from lewm.initially_aligned_scene_development import trials as previous_trials
from lewm.online_temporal_choice_development import OnlineTemporalChoice
from lewm.persistent_alignment_continuation_development import (
    PersistentBearingAlignment, PersistentAlignedContinuation)
from lewm.persistent_alignment_scene_development import trials
from lewm.tests.test_observed_continuation_development import MovingStream
from lewm.tests.test_relative_gyro_turn_development import packet
from scripts.analyze_go2_ground_plane_development_v1 import URDF


class DisturbedStream(MovingStream):
    """Synthetic constant yaw disturbance; recorded commands exclude disturbance."""
    def __init__(self, drift):
        super().__init__()
        self.drift = drift

    def frame(self, tick, applied=(0., 0., 0.), *, changed=False, quiet=True, floor=True):
        if not tick: return super().frame(tick, applied, changed=changed, quiet=quiet, floor=floor)
        rate = np.array([0., 0., applied[2]+self.drift])
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


def alignment_run(target, drift, *, unresponsive=False):
    stream, tracker = DisturbedStream(drift), FastRelativeOrientation()
    align = PersistentBearingAlignment([math.cos(target), math.sin(target), 0.])
    command, rows = [0., 0., 0.], []
    for tick in range(121):
        p, fast, now = stream.frame(tick, [0., 0., 0.] if unresponsive else command)
        attitude = tracker.begin(p, fast, now_ns=now) if tick == 0 else tracker.step(p, fast, now_ns=now)
        row = align.observe(p, attitude, now_ns=now)
        rows.append(row); command = row['requested_command']
        assert command[:2] == [0., 0.] and abs(command[2]) <= .35
        if row['status'] != 'ALIGNING': break
    return rows


@pytest.mark.parametrize('target,drift', [(s*t, -s*d) for s in (-1, 1) for t in (.087266, .3) for d in (.009, .02)])
def test_both_signs_align_under_persistent_disturbance_without_relaxed_dwell(target, drift):
    rows = alignment_run(target, drift)
    assert rows[-1]['status'] == 'COMPLETE'
    assert rows[-1]['requested_command'] == [0., 0., 0.]
    assert len(rows) >= 4
    for row in rows[-4:]:
        assert abs(row['heading_error_rad']) <= .02
        assert abs(row['projected_heading_rate_rad_s']) <= .1
    assert rows[-1]['decision_ns']-rows[-4]['decision_ns'] == 300_000_000
    assert any(0 < abs(r['heading_error_rad']) <= .02 and r['requested_command'][2] != 0 for r in rows[:-1])


@pytest.mark.parametrize('unresponsive,drift', [(True, 0.), (False, -.05)])
def test_persistent_feedback_does_not_hide_unreachable_alignment(unresponsive, drift):
    rows = alignment_run(.3, drift, unresponsive=unresponsive)
    assert rows[-1]['status'] == 'FAILED_TIMEOUT' and len(rows) == 121
    assert rows[-1]['requested_command'] == [0., 0., 0.]


def test_acceptance_fields_match_predecessor_on_identical_observation_stream():
    stream, tracker = DisturbedStream(-.009), FastRelativeOrientation()
    target = [math.cos(.015), math.sin(.015), 0.]
    old, new = FineInitialBearingAlignment(target), PersistentBearingAlignment(target)
    for tick in range(4):
        p, fast, now = stream.frame(tick)
        attitude = tracker.begin(p, fast, now_ns=now) if tick == 0 else tracker.step(p, fast, now_ns=now)
        a, b = old.observe(p, attitude, now_ns=now), new.observe(p, attitude, now_ns=now)
        ac, bc = a.pop('requested_command'), b.pop('requested_command')
        assert a == b
        if tick < 3: assert ac == [0., 0., 0.] and bc[2] > 0
        else: assert ac == bc == [0., 0., 0.]


def test_controller_replacement_occurs_once_before_use_and_keeps_fresh_warmup():
    controller = PersistentAlignedContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream, command, first_rows = MovingStream(), [0., 0., 0.], []
    operator = None
    for tick in range(801):
        child = controller.first if controller.stage == 'FIRST' else controller.second if controller.stage == 'SECOND' else None
        p, fast, now = stream.frame(tick, command, changed=child is not None and child.tick >= 3)
        row = controller.observe(p, fast, now_ns=now)
        assert row['global_orientation']['samples_integrated'] == tick*50
        assert json.loads(json.dumps(row, allow_nan=False)) == row
        if row.get('selected_initial_bearing'):
            operator = controller.initial_turn
            assert type(operator) is PersistentBearingAlignment and operator.start_ns is None
            assert row['requested_command'] == [0., 0., 0.]
        if operator is not None: assert controller.initial_turn is operator
        if row['stage'] == 'FIRST': first_rows.append(row['child'])
        if row['terminal']: break
        command = row['requested_command']
    assert row['status'] == 'COMPLETE_PROVISIONAL'
    assert [r['status'] for r in first_rows[:4]] == ['WARMUP']*3+['TRAVERSING']
    assert row['trusted_graph_edges'] == 0


@pytest.mark.parametrize('method', ['direct_direct', 'supervised_rollout', 'jepa_rollout'])
def test_frozen_models_receive_four_new_first_traversal_frames(method):
    template = OnlineTemporalChoice.from_completed_study(method)
    controller = PersistentAlignedContinuation(method, ArticulatedCollisionGeometry(URDF), template)
    stream, command, times = MovingStream(), [0., 0., 0.], []
    for tick in range(100):
        p, fast, now = stream.frame(tick, command)
        row = controller.observe(p, fast, now_ns=now)
        assert not row['terminal']
        command = row['requested_command']
        if row['stage'] == 'FIRST': times.append(now)
        if row['child'] and row['child']['selection']:
            selected = row['child']['selection']
            assert len(times) == 4
            assert [r['measured_ns'] for r in selected['input_images']] == times
            assert selected['model_bindings'] == template.bindings
            break
    else: raise AssertionError('fitted controller never received fresh first context')


def test_faults_latch_and_missing_initial_evidence_never_starts_traversal():
    controller = PersistentAlignedContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream = MovingStream()
    for tick in range(4):
        p, fast, now = stream.frame(tick, floor=False)
        row = controller.observe(p, fast, now_ns=now)
    assert row['status'] == 'FAILED_INITIAL_NO_EXIT' and controller.initial_turn is None
    controller = PersistentAlignedContinuation('fixed_forward', ArticulatedCollisionGeometry(URDF))
    stream = MovingStream()
    for tick in range(4):
        p, fast, now = stream.frame(tick)
        controller.observe(p, fast, now_ns=now)
    p, fast, now = stream.frame(4)
    fast['values'][-1, 2] = float('nan')
    with pytest.raises(SensorContractError): controller.observe(p, fast, now_ns=now)
    assert controller.status == 'FAILED_SENSOR' and controller.first.tick == -1
    with pytest.raises(SensorContractError): controller.observe(p, fast, now_ns=now)


def test_same_fixtures_and_exact_inherited_collector_auditor_bodies():
    import scripts.run_go2_initially_aligned_continuation_development_v1 as old_runner
    import scripts.audit_go2_initially_aligned_continuation_development_v1 as old_auditor
    import scripts.run_go2_persistent_alignment_continuation_development_v1 as runner
    import scripts.audit_go2_persistent_alignment_continuation_development_v1 as auditor
    assert runner.ObservedContinuation is auditor.ObservedContinuation is PersistentAlignedContinuation
    assert runner.reduce_continuation is old_runner.reduce_continuation is auditor.reduce_continuation
    assert auditor.load_route_observation is old_auditor.load_route_observation
    for a, b in [(runner.collect, old_runner.collect), (auditor.audit_trial, old_auditor.audit_trial)]:
        assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(a)))) == ast.dump(ast.parse(textwrap.dedent(inspect.getsource(b))))
    for a, b in zip(trials(), previous_trials(), strict=True):
        assert a['geometry'] == b['geometry'] and a['evaluation_leg_geometries'] == b['evaluation_leg_geometries']
        assert a['method'] == b['method'] and a['scene_id'] != b['scene_id']
    assert runner.OUTPUT == auditor.OUTPUT
