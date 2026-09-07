from copy import deepcopy

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.depth_relative_motion_development import DepthRelativeState
from lewm.floor_footprint_bounds_development import projected_footprint_rectangles
from lewm.primitive_floor_observation_development import PreparedFloorFrame, observe_primitive_floor
from lewm.primitive_obstacle_memory_development import (
    non_floor_box_evidence, plane_family_cells, pixel_cell_box_depth_intervals,
    combine_primitive_views, PrimitiveObstacleMemory)
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth
from lewm.tests.test_floor_footprint_bounds_development import scene
from scripts.analyze_go2_ground_plane_development_v1 import URDF


LOW = np.array([[1.45, -.04, -.28]])
HIGH = np.array([[1.55, .04, -.26]])
PLANE = (np.array([1.5, 0., -.3]), np.array([0., 0., 1.]))


def box_query(frame, low=LOW, high=HIGH, plane=PLANE, error=.001):
    return non_floor_box_evidence(frame, low, high, plane=plane, normal_error=.002,
                                  plane_offset_error=.001, range_error_m=error)


def rectangle():
    row = projected_footprint_rectangles(LOW, HIGH, [0.], [0.], [0, 0, 1])
    x0, y0 = row['lower_cells_xy'][0]; x1, y1 = row['upper_cells_xy'][0]
    return int(x0), int(y0), int(x1), int(y1)


def test_floor_returns_only_remove_floor_contribution_not_grant_navigation():
    d, valid = scene(); frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    result = box_query(frame)
    assert result['non_floor_clearance'].all() and result['floor_family_pixels'][0] > 0
    assert not result['navigation_qualified'] and not result['floor_classification_calibrated']


def test_pixel_beam_intervals_contain_every_sampled_optical_box_point():
    rng = np.random.default_rng(3598)
    low, high = np.array([-.2, .1, .7]), np.array([.3, .25, 2.])
    q = rng.uniform(low, high, (10000, 3))
    uv = FOCAL * q[:, :2] / q[:, 2, None] + [319.5, 239.5]
    cells = np.floor(uv[:, ::-1]).astype(int)
    near, far, intersects = pixel_cell_box_depth_intervals(low, high, cells)
    assert intersects.all() and (q[:, 2] >= near).all() and (q[:, 2] <= far).all()


def test_global_rectangle_depth_overlap_can_be_outside_actual_pixel_beam():
    # The pixel beam intersects this box only near its far end. A 0.8-m
    # return is in the box's global z interval but outside its x interval.
    low, high = [.2, -.1, .5], [.3, .1, 2.]
    near, far, intersects = pixel_cell_box_depth_intervals(low, high, np.array([[239, 380]]))
    assert intersects[0] and near[0] > .8 and far[0] <= 2. + 1e-9


@pytest.mark.parametrize('obstacle', ['wall', 'horizontal_overhang'])
def test_measured_wall_or_floor_like_overhang_is_not_erased_by_floor_family(obstacle):
    d, valid = scene(); x0, y0, x1, y1 = rectangle()
    sl = np.s_[y0-2:y1+3, x0-2:x1+3]
    if obstacle == 'wall': d[sl] = 1.17
    else:
        # Horizontal obstacle 3 cm above the selected floor, inside the box.
        # Its normal passes the ground-cell orientation test; its plane does not.
        rows = np.arange(y0-2, y1+3)
        d[sl] = ((.043 + .27) * FOCAL / (rows - 239.5))[:, None]
    frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    if obstacle == 'horizontal_overhang':
        cell = np.array([[(y0+y1)//2, (x0+x1)//2]])
        assert frame._index['ground_cells'][tuple(cell[0])]
        assert not plane_family_cells(frame, cell, *PLANE, normal_error=.002, plane_offset_error=.001).any()
    result = box_query(frame)
    assert result['non_floor_conflict'].all() and not result['non_floor_clearance'].any()


def test_missing_interior_ray_prevents_nonfloor_clearance():
    d, valid = scene(); x0, y0, x1, y1 = rectangle(); y, x = (y0+y1)//2, (x0+x1)//2
    d[y, x] = 0.; valid[y, x] = False
    result = box_query(PreparedFloorFrame(d, valid, [0, 0, 1]))
    assert result['unknown_or_blocked'].all() and not result['non_floor_clearance'].any()


def test_occluding_surface_is_unknown_even_if_not_at_box_depth():
    d = np.full((480, 640), .7, np.float32); valid = np.ones_like(d, bool)
    result = box_query(PreparedFloorFrame(d, valid, [0, 0, 1]), plane=None)
    assert not result['non_floor_conflict'].any() and result['unknown_or_blocked'].all()


def test_partial_view_preserves_near_obstacle_veto():
    d = np.full((480, 640), 1.17, np.float32); valid = np.ones_like(d, bool)
    result = box_query(PreparedFloorFrame(d, valid, [0, 0, 1]), [[1.4, -2., -.3]], [[1.6, .1, -.24]], plane=None)
    assert not result['projection_within_observed_camera'].any()
    assert result['non_floor_conflict'].all() and not result['non_floor_clearance'].any()


@pytest.mark.parametrize('edge', ['last_column', 'last_row'])
def test_incomplete_view_retains_image_border_return_as_nonfloor_obstacle(edge):
    d = np.full((480, 640), 4., np.float32); valid = np.ones_like(d, bool)
    if edge == 'last_column':
        d[350, 639] = 1.17; low, high = [[1.4, -2., -.3]], [[1.6, .1, -.24]]
    else:
        d[479, 320] = 1.17; low, high = [[1.4, -.1, -2.]], [[1.6, .1, -.24]]
    result = box_query(PreparedFloorFrame(d, valid, [0, 0, 1]), low, high)
    assert not result['projection_within_observed_camera'].any()
    assert result['non_floor_conflict'].all()


def test_observed_border_floor_is_not_invented_obstacle_but_view_stays_incomplete():
    d, valid = scene()
    result = box_query(PreparedFloorFrame(d, valid, [0, 0, 1]), [[.7, -.1, -.5]], [[1.2, .1, -.27]])
    assert not result['projection_within_observed_camera'].any()
    assert not result['non_floor_conflict'].any()
    assert not result['non_floor_clearance'].any() and result['unknown_or_blocked'].all()


def test_incomplete_view_global_depth_bound_skips_scan_without_approving(monkeypatch):
    import lewm.primitive_obstacle_memory_development as module
    d = np.full((480, 640), 4., np.float32); valid = np.ones_like(d, bool)
    frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    def should_not_scan(*args, **kwargs): raise AssertionError('all returns provably beyond this incomplete box')
    monkeypatch.setattr(module, 'plane_family_cells', should_not_scan)
    result = box_query(frame, [[1.4, -2., -.3]], [[1.6, .1, -.24]])
    assert result['scanned_pixels'][0] == 0 and result['unknown_or_blocked'].all()
    assert not result['non_floor_conflict'].any() and not result['non_floor_clearance'].any()


def test_range_error_expansion_cannot_erase_a_near_obstacle():
    d = np.full((480, 640), 1.23, np.float32); valid = np.ones_like(d, bool)
    frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    assert box_query(frame, plane=None, error=0.)['non_floor_clearance'].all()
    for error in (.01, .03, .1):
        result = box_query(frame, plane=None, error=error)
        assert result['non_floor_conflict'].all() and not result['non_floor_clearance'].any()


def view(*, clear=False, candidate=False, conflict=False, penetration=False):
    return {'shape_ids': ('FL_foot:0',), 'clear': np.array([clear]), 'contact_candidate': np.array([candidate]),
            'non_floor_conflict': np.array([conflict]), 'floor_penetration': np.array([penetration])}


@pytest.mark.parametrize('negative', ['conflict', 'penetration'])
@pytest.mark.parametrize('reverse', [False, True])
def test_any_view_negative_vetoes_older_clearance_and_contact_candidates(negative, reverse):
    rows = [view(clear=True, candidate=True), view(**{negative: True})]
    if reverse: rows.reverse()
    result = combine_primitive_views(('FL_foot:0',), rows)
    assert not result['conditional_clearance'].any() and not result['foot_contact_candidate'].any()
    assert not result['all_primitives_conditionally_clear'] and not result['contact_permitted']


def test_empty_view_population_is_unknown_not_clearance():
    result = combine_primitive_views(('base:0',), [])
    assert not result['conditional_clearance'].any() and not result['all_primitives_conditionally_clear']


@pytest.mark.parametrize('fault', ['identity', 'shape', 'dtype', 'fake_foot'])
def test_misaligned_or_invalid_view_evidence_rejected(fault):
    row = view()
    if fault == 'identity': row['shape_ids'] = ('base:0',)
    if fault == 'shape': row['clear'] = np.zeros(2, bool)
    if fault == 'dtype': row['clear'] = np.array([1])
    if fault == 'fake_foot':
        row['shape_ids'] = ('base:0',); row['contact_candidate'][:] = True
    with pytest.raises(SensorContractError):
        combine_primitive_views(('base:0',) if fault == 'fake_foot' else ('FL_foot:0',), [row])


def test_prepared_floor_queries_reuse_index_and_isolate_input_mutations(monkeypatch):
    import lewm.primitive_floor_observation_development as module
    calls = []; original = module.observed_floor_cell_index
    def counted(*args): calls.append(1); return original(*args)
    monkeypatch.setattr(module, 'observed_floor_cell_index', counted)
    model = ArticulatedCollisionGeometry(URDF); q = np.repeat([0., .8, -1.5], 4)
    errors = {r['shape_id']: .001 for r in model.supports(q, np.eye(3))['shapes']}
    d, valid = scene(); frame = PreparedFloorFrame(d, valid, [0, 0, 1])
    kwargs = dict(rotation_observation_from_body=np.eye(3), translation_observation_from_body=[1.5, 0, 0],
                  normal_error=.002, up_error=.001, plane_offset_error=.001, point_error_by_shape=errors)
    a = frame.query(model, q, [354, 319], **kwargs)
    b = observe_primitive_floor(model, q, d, valid, [0, 0, 1], [354, 319], **kwargs)
    assert a['floor_coverage'] == b['floor_coverage']
    d[:] = 0.; valid[:] = False
    c = frame.query(model, q, [354, 319], **kwargs)
    assert a['gap_bounds'] == c['gap_bounds'] and a['floor_coverage'] == c['floor_coverage']
    assert len(calls) == 2  # One prepared frame, one standalone reference, no per-query rebuild.
    with pytest.raises(ValueError): frame._depth[0, 0] = 1.


def observed_stream(count=2):
    original = DepthRelativeState()
    for _, p, f, _ in stream(count):
        d = room_depth(p); now = d['measured_ns']
        yield p, d, original.observe(p, d, f, now_ns=now), now


def memory():
    return PrimitiveObstacleMemory(ArticulatedCollisionGeometry(URDF), normal_error=.002,
                                    up_error=.001, plane_offset_error=.001, range_error_m=.001)


def test_consumer_binds_current_joint_posture_and_depth_and_reuses_cached_indexes(monkeypatch):
    import lewm.primitive_floor_observation_development as module
    calls = []; original = module.observed_floor_cell_index
    def counted(*args): calls.append(1); return original(*args)
    monkeypatch.setattr(module, 'observed_floor_cell_index', counted)
    model = memory()
    for p, d, r, now in observed_stream(): model.observe(p, d, r, now_ns=now)
    a = model.query_current_primitives(now_ns=now)
    assert len(a['shape_ids']) == 27 and a['current_measured_posture_only']
    assert a['measured_ns'] == now and not a['navigation_qualified'] and not a['contact_permitted']
    assert a['views'][-1]['depth_sha256'] == r['local_surfaces']['depth_sha256']
    p['sensor_state']['sensed']['joints']['values'][:] = 1.; d['depth_m'][:] = 0.
    a['conditional_clearance'][:] = True
    b = model.query_current_primitives(now_ns=now)
    assert not b['all_primitives_conditionally_clear']
    assert len(calls) == 2
    with pytest.raises(SensorContractError): model.query_current_primitives(now_ns=now + 1)


@pytest.mark.parametrize('fault', ['depth_bytes', 'depth_time', 'joint_validity', 'duplicate'])
def test_observation_failures_latch_and_discard_queryable_history(fault):
    model = memory(); packets = list(observed_stream())
    p, d, r, now = packets[0]; model.observe(p, d, r, now_ns=now)
    p, d, r, now = deepcopy(packets[0] if fault == 'duplicate' else packets[1])
    if fault == 'depth_bytes': d['depth_m'][300, 320] += .01
    if fault == 'depth_time': d['measured_ns'] -= 100_000_000
    if fault == 'joint_validity':
        p['sensor_state']['sensed']['joints']['valid'][-1, 0] = False
        p['sensor_state']['sensed']['joints']['values'][-1, 0] = 0.
    with pytest.raises(SensorContractError): model.observe(p, d, r, now_ns=now)
    assert model.failed and not model._prepared
    with pytest.raises(SensorContractError): model.query_current_primitives(now_ns=now)
    with pytest.raises(SensorContractError): model.observe(*packets[1][:3], now_ns=packets[1][3])
