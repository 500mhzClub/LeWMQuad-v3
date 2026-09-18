"""Synthetic ledger sequences and padded boundary fixtures, not a native run."""
import ast
from copy import deepcopy
from functools import partial
import inspect
import textwrap
from types import SimpleNamespace as NS

import numpy as np
import pytest

from lewm import extended_return_budget_memory_development as new
from lewm.tests import test_measured_floor_transport_development as transport_fixture
from lewm.tests.test_extended_return_budget_transport_development import late_inputs, produce
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_current_primary_floor_plane_development import depth_plane
from lewm.tests.test_frame_floor_cache_development import equal
from lewm.tests.test_later_floor_evidence_development import observations, record, BOX


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(transport_fixture.fixture, 'visual', partial(visual, origin=1_500_000_000))


def retime(value, delta, key=None):
    if isinstance(value, dict): return {k:retime(v, delta, k) for k,v in value.items()}
    if isinstance(value, list): return [retime(v, delta) for v in value]
    if key is not None and key.endswith('_ns') and (type(value) is int or isinstance(value, np.ndarray)):
        return value+delta
    return deepcopy(value)


def memory_inputs(frame):
    anchor, _, _, _, _ = late_inputs(frame+1)
    policy, depth, _, _, before_ns, _ = transport_fixture.item(
        1, anchor['original_visual_evidence'], height=.34)
    now = anchor['decision_ns']; delta = now-before_ns
    policy, depth = retime(policy, delta), retime(depth, delta)
    assert transport_fixture.depth_hash(depth) == anchor['floor_registration']['primary_depth_sha256']
    return policy, depth, anchor, now


def padded_memory(cls, frame, now):
    memory = cls(identity=(0, 0, 0))
    # Metadata placeholders exercise only the next append/boundary behavior.
    memory.route = [dict(boundary_fixture=True)]*frame
    memory.last_ns = now-100_000_000
    return memory


@pytest.mark.parametrize('frame', [2, 4095, 4096, 8013])
def test_surface_append_preserves_existing_route_and_measured_pose(frame):
    policy, depth, evidence, now = memory_inputs(frame)
    memory = padded_memory(new.ExtendedReturnBudgetMemory, frame, now)
    prefix_ids = [id(row) for row in memory.route]
    result = memory.observe(policy, depth, evidence, now_ns=now)
    assert len(memory.route) == frame+1 and [id(row) for row in memory.route[:-1]] == prefix_ids
    assert result['visited_poses'] == frame+1 and result['frame'] == frame
    assert not memory.failed and not result['free_space_established']
    position, rotation, _ = new.current_measured_floor_pose(evidence, identity=(0, 0, 0), now_ns=now)
    np.testing.assert_array_equal(memory.position, position)
    np.testing.assert_array_equal(memory.rotation, rotation)
    old = padded_memory(new.MeasuredFloorTransportMemory, frame, now)
    if frame < 4096:
        expected = old.observe(policy, depth, evidence, now_ns=now)
        equal(result, expected); equal(memory.route, old.route)
        equal(vars(memory.index), vars(old.index))
    else:
        with pytest.raises(ValueError): old.observe(policy, depth, evidence, now_ns=now)
        assert old.failed and len(old.route) == frame


@pytest.mark.parametrize('fault', ['capacity', 'clock', 'frame', 'depth_hash'])
def test_surface_failure_latches_without_appending_or_evicting(fault):
    frame = 8014 if fault == 'capacity' else 4096
    policy, depth, evidence, now = memory_inputs(frame)
    memory = padded_memory(new.ExtendedReturnBudgetMemory, frame, now)
    if fault == 'clock': memory.last_ns -= 1
    if fault == 'frame': memory.route.append({})
    if fault == 'depth_hash': evidence['current_pose']['depth_sha256'] = '0'*64
    count = len(memory.route); before = deepcopy(vars(memory.index))
    with pytest.raises(ValueError): memory.observe(policy, depth, evidence, now_ns=now)
    assert memory.failed and len(memory.route) == count
    equal(before, vars(memory.index))
    with pytest.raises(ValueError, match='latched'): memory.observe(policy, depth, evidence, now_ns=now)


def test_complete_later_floor_ledger_retains_order_and_strictly_later_resolution():
    ledger = new.ExtendedReturnBudgetLaterFloorEvidence()
    first = None
    for frame in range(8014):
        record(ledger, frame, auxiliary=[(0, 0)] if frame == 8013 else [])
        if frame == 0: first = deepcopy(ledger._records)
    assert ledger.frame == 8013 and len(ledger._records) == 16028
    assert ledger._records[:2] == first
    assert [(r['frame'], r['camera']) for r in ledger._records] == [
        (i, camera) for i in range(8014) for camera in ('primary', 'auxiliary')]
    resolved = ledger.resolve(BOX, 8012, now_ns=ledger.now_ns)
    assert resolved['resolved'] and resolved['later_single_view']['frame'] == 8013
    assert resolved['later_single_view']['camera'] == 'auxiliary'
    assert not resolved['ground_support_approved']
    assert not ledger.resolve(BOX, 8013, now_ns=ledger.now_ns)['resolved']
    before = deepcopy(vars(ledger))
    with pytest.raises(ValueError): record(ledger, 8014)
    equal(before, vars(ledger))


@pytest.mark.parametrize('fault', ['clock', 'calibration', 'transform', 'rgb_pair'])
def test_late_floor_pair_rejection_does_not_publish_partial_evidence(fault):
    ledger = new.ExtendedReturnBudgetLaterFloorEvidence()
    ledger.frame = 4095
    views = observations(4096, [(0, 0)])
    if fault == 'clock': views[1]['witness']['measured_ns'] += 1
    if fault == 'calibration': views[1]['witness']['calibration_id'] = 'wrong'
    if fault == 'transform': views[1]['witness']['position_map_m'][0] += .01
    if fault == 'rgb_pair': views[1]['witness']['rgb_sha256'] = 'c'*64
    before = deepcopy(vars(ledger))
    with pytest.raises(ValueError):
        ledger.record_pair(4096, 1_500_000_000+4096*100_000_000, np.eye(3), -.3, views)
    equal(before, vars(ledger))


@pytest.mark.parametrize('frame', [0, 4095, 4096, 8013])
def test_retained_patch_prefix_bytes_and_prior_entries_survive_extended_append(frame):
    depth, valid = depth_plane(); now = 1_500_000_000+frame*100_000_000
    sentinel = dict(floor_height=-.32, witness=dict(measured_ns=now-100_000_000))
    history = NS(frames=[sentinel]*frame)
    geometry = new.ExtendedReturnBudgetFloorGeometry()
    original = new.BodyProjectedFloorGeometry(); reference = NS(frames=[])
    try:
        original.append_patch(reference, depth, valid, np.eye(3), np.zeros(3), -.32,
            dict(frame=0, measured_ns=1_500_000_000))
        geometry.append_patch(history, depth, valid, np.eye(3), np.zeros(3), -.32,
            dict(frame=frame, measured_ns=now))
        assert len(history.frames) == frame+1
        assert all(row is sentinel for row in history.frames[:-1])
        equal(history.frames[-1]['prefix'], reference.frames[0]['prefix'])
        assert not history.frames[-1]['prefix'].flags.writeable
        history.frames[-1]['witness']['frame'] = -1
        assert reference.frames[0]['witness']['frame'] == 0
    finally:
        geometry.close(); original.close()


@pytest.mark.parametrize('fault', ['capacity', 'gap', 'floor', 'depth', 'rotation'])
def test_late_patch_rejection_keeps_history_intact(fault):
    frame = 8014 if fault == 'capacity' else 4096
    now = 1_500_000_000+frame*100_000_000
    old = dict(floor_height=-.32, witness=dict(measured_ns=now-100_000_000))
    history = NS(frames=[old]*frame); depth, valid = depth_plane(); rotation = np.eye(3)
    if fault == 'gap': old['witness']['measured_ns'] -= 1
    if fault == 'floor': old['floor_height'] = -.3
    if fault == 'depth': depth[0, 0] = np.nan
    if fault == 'rotation': rotation[0, 0] = 2.
    geometry = new.ExtendedReturnBudgetFloorGeometry()
    try:
        with pytest.raises(ValueError):
            geometry.append_patch(history, depth, valid, rotation, np.zeros(3), -.32,
                dict(frame=frame, measured_ns=now))
        assert len(history.frames) == frame and all(row is old for row in history.frames)
    finally: geometry.close()


def test_residual_uses_same_extended_transport_pose():
    evidence, _, _, _, _, now = produce(4096)
    residual = new.ExtendedReturnBudgetResidual(); residual.frame = 4095
    residual.observe(evidence, now_ns=now)
    position, rotation, _ = new.current_measured_floor_pose(evidence, identity=(0, 0, 0), now_ns=now)
    assert residual.frame == 4096 and residual.now_ns == now
    np.testing.assert_array_equal(residual.pose['position'], position)
    np.testing.assert_array_equal(residual.pose['rotation'], rotation)
    assert not residual.history and residual.pending is None


def test_recording_geometry_dispatch_and_only_declared_patch_bound_change():
    assert new.ExtendedReturnBudgetRecordingFloorGeometry.append_patch is new.ExtendedReturnBudgetFloorGeometry.append_patch
    assert new.ExtendedReturnBudgetRecordingFloorGeometry.index is new.BodyProjectedRecordingFloorGeometry.index
    assert new.ExtendedReturnBudgetRecordingFloorGeometry.floor_coverage is new.BodyProjectedRecordingFloorGeometry.floor_coverage
    original = ast.parse(textwrap.dedent(inspect.getsource(new.BodyProjectedFloorGeometry.append_patch)))
    actual = ast.parse(textwrap.dedent(inspect.getsource(new.ExtendedReturnBudgetFloorGeometry.append_patch)))
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.Constant(value=4096) if node.id == 'MAX_OBSERVATIONS' else node
    assert ast.dump(Normalize().visit(actual)) == ast.dump(original)
