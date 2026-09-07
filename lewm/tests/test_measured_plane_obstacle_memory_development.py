from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import FOCAL
from lewm.measured_plane_obstacle_memory_development import (
    MeasuredPlaneHypothesis, MeasuredPlaneObstacleMemory, relation_gated_non_floor_evidence)
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.primitive_obstacle_memory_development import non_floor_box_evidence, combine_primitive_views
from lewm.tests.test_primitive_obstacle_memory_development import observed_stream, PLANE
from lewm.tests.test_floor_footprint_bounds_development import scene
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.probe_go2_primitive_beam_kernel_development import exact


ERRORS = dict(normal_error=.002, plane_offset_error=.001, range_error_m=.001)


def test_hypothesis_is_immutable_measured_and_query_independent():
    d, v = scene(); frame = PreparedFloorFrame(d, v, [0, 0, 1])
    h = MeasuredPlaneHypothesis.from_frame(frame)
    cells = np.argwhere(frame._index['ground_cells'])
    assert h.cell_for(frame) == tuple(cells[len(cells) // 2])
    d[:] = 0.; v[:] = False
    assert MeasuredPlaneHypothesis.from_frame(frame) == h
    with pytest.raises(FrozenInstanceError): h.cell_rc = (0, 0)
    for other in (replace(h, depth_sha256='bad'), replace(h, up=(0., 1., 0.)), replace(h, policy='other')):
        with pytest.raises(SensorContractError): other.cell_for(frame)


@pytest.mark.parametrize('depth', [0., 1.])
def test_empty_or_vertical_scene_has_no_plane_hypothesis(depth):
    d = np.full((480, 640), depth, np.float32)
    frame = PreparedFloorFrame(d, d > 0, [0, 0, 1])
    assert MeasuredPlaneHypothesis.from_frame(frame).cell_rc is None


def test_missing_selected_cell_is_not_filled_in_or_reused_for_new_frame():
    d, v = scene(); a = PreparedFloorFrame(d, v, [0, 0, 1])
    h = MeasuredPlaneHypothesis.from_frame(a); y, x = h.cell_rc
    d[y:y+2, x:x+2] = 0.; v[y:y+2, x:x+2] = False
    b = PreparedFloorFrame(d, v, [0, 0, 1]); other = MeasuredPlaneHypothesis.from_frame(b)
    assert other.cell_rc != h.cell_rc and other.depth_sha256 != h.depth_sha256
    with pytest.raises(SensorContractError): h.cell_for(b)


@pytest.mark.parametrize('backend', ['reference', 'compiled'])
def test_partial_view_plane_label_cannot_suppress_nonfoot_collision(backend):
    # Isolate an infinite floor, without the room fixture's side-wall veto.
    y = (np.arange(480) - 239.5) / FOCAL
    raw = np.broadcast_to((.343 / np.maximum(y, 1e-12))[:, None], (480, 640))
    v = (raw >= .2) & (raw <= 5.)
    d = np.where(v, raw, 0.).astype(np.float32)
    frame = PreparedFloorFrame(d, v, [0, 0, 1])
    # The observed plane cuts this non-foot shape, but its full footprint is
    # outside the image. A global plane mask previously erased the near returns.
    low, high = [[1.4, -2., -.32]], [[1.6, .1, -.24]]
    old = non_floor_box_evidence(frame, low, high, plane=PLANE, backend=backend, **ERRORS)
    assert not old['projection_within_observed_camera'].any()
    assert not old['non_floor_conflict'].any() and old['floor_family_pixels'][0] > 0
    new = relation_gated_non_floor_evidence(frame, low, high, plane=PLANE, shape_ids=('base:0',),
                                           gap_lower=[-.021], gap_upper=[-.019], backend=backend, **ERRORS)
    assert new['non_floor_conflict'].all() and not new['plane_exemption'].any()
    assert new['floor_family_pixels'][0] == 0 and not new['non_floor_clearance'].any()
    positive = dict(shape_ids=('base:0',), clear=np.array([True]), contact_candidate=np.array([False]),
                    non_floor_conflict=np.array([False]), floor_penetration=np.array([False]))
    negative = positive | dict(clear=np.array([False]), non_floor_conflict=new['non_floor_conflict'])
    assert not combine_primitive_views(('base:0',), [positive, negative])['conditional_clearance'].any()


def test_per_shape_partition_scatter_and_backend_parity():
    d, v = scene(); frame = PreparedFloorFrame(d, v, [0, 0, 1])
    ids = ('base:0', 'FL_foot:0', 'FR_foot:0', 'other:0', 'RL_foot:0')
    gl = np.array([.01, -.001, -.03, -.001, .02]); gh = gl + .002
    low = np.tile([1.4, -2., -.32], (5, 1)); high = np.tile([1.6, .1, -.24], (5, 1))
    rows = []
    for backend in ('reference', 'compiled'):
        r = relation_gated_non_floor_evidence(frame, low, high, plane=PLANE, shape_ids=ids,
                                              gap_lower=gl, gap_upper=gh, backend=backend, **ERRORS)
        np.testing.assert_array_equal(r['plane_exemption'], [True, True, False, False, True])
        for i, exempt in enumerate(r['plane_exemption']):
            direct = non_floor_box_evidence(frame, low[i:i+1], high[i:i+1], plane=PLANE if exempt else None,
                                             backend=backend, **ERRORS)
            for key, value in direct.items():
                if isinstance(value, np.ndarray): np.testing.assert_array_equal(r[key][i:i+1], value)
                else: assert r[key] == value
        rows.append(r)
    exact(*rows)


def test_absent_plane_never_exempts_even_a_separated_shape():
    d, v = scene(); frame = PreparedFloorFrame(d, v, [0, 0, 1])
    r = relation_gated_non_floor_evidence(frame, [[1.4, -.1, -.32]], [[1.6, .1, -.24]],
                                          plane=None, shape_ids=('base:0',), gap_lower=[.1], gap_upper=[.2], **ERRORS)
    assert not r['plane_exemption'].any() and r['non_floor_conflict'].all()


@pytest.mark.parametrize('fault', ['nan', 'reversed', 'shape', 'duplicate'])
def test_bad_gap_alignment_rejected(fault):
    d, v = scene(); frame = PreparedFloorFrame(d, v, [0, 0, 1])
    ids, gl, gh = ('base:0',), [0.], [.1]
    if fault == 'nan': gl = [np.nan]
    if fault == 'reversed': gh = [-1.]
    if fault == 'shape': gl = [0., 0.]
    if fault == 'duplicate': ids = ('base:0', 'base:0')
    with pytest.raises(SensorContractError):
        relation_gated_non_floor_evidence(frame, [[1.4, -.1, -.32]], [[1.6, .1, -.24]],
                                          plane=PLANE, shape_ids=ids, gap_lower=gl, gap_upper=gh, **ERRORS)


def memory():
    return MeasuredPlaneObstacleMemory(ArticulatedCollisionGeometry(URDF), up_error=.001, **ERRORS)


def test_live_consumer_retains_hypotheses_and_queries_do_not_reselect(monkeypatch):
    model = memory()
    for p, d, r, now in observed_stream(3):
        model.observe(p, d, r, now_ns=now)
        assert set(model._hypotheses) == set(model._prepared)
    before = model._hypotheses.copy()
    def forbidden(*args): raise AssertionError('selection must only occur at observation')
    monkeypatch.setattr(MeasuredPlaneHypothesis, 'from_frame', forbidden)
    a = model.query_current_primitives(now_ns=now, beam_backend='reference')
    b = model.query_current_primitives(now_ns=now, beam_backend='compiled')
    exact(a, b)
    assert model._hypotheses == before
    assert all(v['seed_observed'] for v in a['views'])
    assert not a['navigation_qualified'] and not a['contact_permitted']
    with pytest.raises(SensorContractError): model.query_current_primitives(now_ns=now + 1)


def test_fault_discards_hypotheses_and_latches():
    model = memory(); packets = list(observed_stream(2))
    p, d, r, now = packets[0]; model.observe(p, d, r, now_ns=now)
    p, d, r, now = packets[1]; d['depth_m'][300, 320] += .01
    with pytest.raises(SensorContractError): model.observe(p, d, r, now_ns=now)
    assert model.failed and not model._hypotheses and not model._prepared
    with pytest.raises(SensorContractError): model.query_current_primitives(now_ns=now)
