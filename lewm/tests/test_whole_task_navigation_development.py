"""Synthetic whole-task integration; no physical task success is asserted here."""
from copy import deepcopy
import math

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_observed_continuation_development import MovingStream, branch
from lewm.whole_task_navigation_development import WholeTaskNavigation, branch_order
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def marker_pixels(packet):
    packet['image']['rgb'][150:250, 240:290] = [150, 20, 10]
    packet['image']['rgb'][150:250, 295:345] = [15, 30, 150]


def controller(arm='episodic'):
    return WholeTaskNavigation('fixed_forward', ArticulatedCollisionGeometry(URDF), memory_arm=arm)


def synthetic(arm='episodic', *, detect_during='SCAN', max_ticks=2200):
    model = controller(arm); stream = MovingStream(); command = [0., 0., 0.]
    rows = []; activated = False
    for tick in range(max_ticks):
        child = model.child if model.stage == 'TRAVERSE' else None
        p, fast, now = stream.frame(tick, command, changed=child is not None and child.tick >= 3)
        # Distinct actually supplied initial reference prevents a featureless
        # synthetic image from accidentally terminating this route test.
        if tick == 0: p['image']['rgb'][:120] = [180, 180, 180]
        if model.completed_legs >= 2 and (model.stage == detect_during or (
                detect_during == 'ACTIVE_TRAVERSE' and model.stage == 'TRAVERSE' and child.tick >= 10)):
            activated = True
        if activated: marker_pixels(p)
        row = model.observe(p, fast, now_ns=now); rows.append(row)
        if row['terminal']: return model, rows
        command = row['requested_command']
    return model, rows


def test_observed_wall_follow_order_includes_reverse_for_dead_ends():
    incoming = [1., 0., 0.]
    bearings = [math.pi, -math.pi/2, 0., math.pi/2]
    assert [branch_order(branch(a), incoming)[0] for a in bearings] == [3, 2, 1, 0]


def test_continuous_two_outward_two_return_sequence_with_actual_pixel_marker():
    model, rows = synthetic()
    assert rows[-1]['status'] == 'HOME_CANDIDATE_ROUTE_HYPOTHESIS'
    assert model.completed_legs == 4
    memory = model.memory_snapshot()
    assert [a['mode'] for a in memory['attempts']] == ['OUTWARD', 'OUTWARD', 'RETURN', 'RETURN']
    assert len(memory['visits']) == 5 and memory['hypothesized_route_depth'] == 0
    assert memory['trusted_graph_edges'] == 0 and not memory['mission_complete']
    assert not rows[-1]['home_verified']
    assert sum(bool(r['marker']['newly_discovered']) for r in rows) == 1
    assert sum(r['mission_changed'] for r in rows) == 1
    assert all(r['global_orientation']['samples_integrated'] == r['tick']*50 for r in rows)
    assert all(r['requested_command'] == [0., 0., 0.] for r in rows if r['terminal'])
    for leg in range(4):
        children = [r['child'] for r in rows if r['stage'] == 'TRAVERSE' and r['leg_index'] == leg]
        assert [c['status'] for c in children[:4]] == ['WARMUP']*3+['TRAVERSING']


def test_discovery_during_unstarted_outward_leg_cancels_it_without_relabeling():
    model, rows = synthetic(detect_during='TRAVERSE')
    assert rows[-1]['status'] == 'HOME_CANDIDATE_ROUTE_HYPOTHESIS'
    # Activation is initially during warmup, so the unstarted third leg must
    # be canceled rather than being relabeled as an already executed return.
    assert any(r['canceled_unstarted_leg'] for r in rows)
    assert [a['mode'] for a in model.memory_snapshot()['attempts']] == ['OUTWARD', 'OUTWARD', 'RETURN', 'RETURN']


def test_discovery_during_committed_outward_leg_finishes_it_before_return():
    model, rows = synthetic(detect_during='ACTIVE_TRAVERSE')
    assert rows[-1]['status'] == 'HOME_CANDIDATE_ROUTE_HYPOTHESIS'
    assert model.completed_legs == 6 and not any(r['canceled_unstarted_leg'] for r in rows)
    assert [a['mode'] for a in model.memory_snapshot()['attempts']] == ['OUTWARD']*3+['RETURN']*3


def test_local_only_has_no_route_memory_or_automatic_empty_stack_home_claim():
    model, rows = synthetic('local_only', max_ticks=1000)
    assert model.memory is None and model.memory_snapshot() is None
    assert all(r['memory_digest'] is None for r in rows)
    assert any(r['mission'] == 'RETURN' for r in rows)
    assert all(r['status'] != 'HOME_CANDIDATE_ROUTE_HYPOTHESIS' for r in rows)
    assert model.completed_legs > 2


def test_initially_visible_marker_is_explicit_unverified_early_home_claim():
    model = controller(); stream = MovingStream()
    for tick in range(3):
        p, fast, now = stream.frame(tick); marker_pixels(p)
        row = model.observe(p, fast, now_ns=now)
    assert row['status'] == 'HOME_CANDIDATE_INITIAL_MARKER' and row['completed_legs'] == 0
    assert row['requested_command'] == [0., 0., 0.] and not row['home_verified']


@pytest.mark.parametrize('fault', ['privilege', 'fast_gap', 'episode', 'future'])
def test_sensor_fault_latches_whole_task_and_memory(fault):
    model = controller(); stream = MovingStream()
    p, fast, now = stream.frame(0); model.observe(p, fast, now_ns=now)
    p, fast, now = stream.frame(1)
    if fault == 'privilege': p['maze_cell'] = [0, 0]
    if fault == 'fast_gap': fast['measured_ns'][-1] -= 1
    if fault == 'episode': p['sensor_state']['identity'] = (0, 0, 1)
    if fault == 'future': p['image']['available_ns'] += 1
    with pytest.raises(SensorContractError): model.observe(p, fast, now_ns=now)
    assert model.status == 'FAILED_SENSOR'
    assert model.memory_snapshot()['phase'] == 'UNCERTAIN_AFTER_FAILURE'
    with pytest.raises(SensorContractError): model.observe(p, fast, now_ns=now)


def test_between_leg_native_stop_suspends_memory_without_inventing_arrival():
    model = controller(); stream = MovingStream()
    p, fast, now = stream.frame(0); model.observe(p, fast, now_ns=now)
    model.finish_physical_stop(now_ns=now+2_000_000)
    assert model.status == 'PHYSICAL_STOP'
    s = model.memory_snapshot()
    assert s['phase'] == 'UNCERTAIN_AFTER_FAILURE' and len(s['visits']) == 1 and s['attempts'] == []


def test_home_quiet_dwell_cannot_bridge_an_unobserved_gap():
    model = controller(); stream = MovingStream()
    p, fast, now = stream.frame(0); model.observe(p, fast, now_ns=now)
    attitude = model.orientation._result()
    assert not model._home_appearance(p, attitude, now)['provisional_match']
    for tick in range(1, 6):
        p, fast, later = stream.frame(tick)
        attitude = model.orientation.step(p, fast, now_ns=later)
    assert not model._home_appearance(p, attitude, later)['provisional_match']
