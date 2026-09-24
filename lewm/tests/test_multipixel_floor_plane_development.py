"""Synthetic tests; no experimental sensor-error calibration."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.multipixel_floor_plane_development import (
    PatchRules, fit_multipixel_plane, query_multipixel_floor)
from lewm.primitive_floor_observation_development import PreparedFloorFrame
from lewm.primitive_floor_relation_development import assess_primitive_floor_relation
from lewm.tests.test_floor_footprint_bounds_development import scene
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def frame():
    d, valid = scene()
    return PreparedFloorFrame(d.astype(np.float32), valid, [0., 0., 1.])


@pytest.fixture(scope='module')
def geometry():
    return ArticulatedCollisionGeometry(URDF)


def query(f, plane, geometry, *, t=(1.5, 0., 0.), backend='reference', R=None):
    q = np.repeat([0., .8, -1.5], 4)
    errors = {s['shape_id']: .001 for s in geometry.supports(q, np.eye(3))['shapes']}
    return query_multipixel_floor(f, plane, geometry, q,
        rotation_observation_from_body=np.eye(3) if R is None else R,
        translation_observation_from_body=t, normal_error=.002, up_error=.001,
        plane_offset_error=.001, point_error_by_shape=errors, floor_backend=backend)


def test_all_pixels_fit_a_plane_without_permission(geometry):
    f = frame(); p = fit_multipixel_plane(f)
    assert p.status == 'MEASURED_PATCH_CONSISTENT'
    np.testing.assert_allclose(p.normal, [0, 0, 1], atol=2e-6, rtol=0)
    assert p.maximum_pixel_residual_m < 1e-6
    assert min(p.tangent_rms_m) >= .005
    result = query(f, p, geometry)
    assert len(result['floor_coverage']) == 27 and all(result['floor_coverage'].values())
    assert result['covered_cells'].min() > 1
    for name in ('contact_permitted', 'non_floor_clearance_established', 'navigation_qualified',
                 'supplied_error_bounds_validated', 'future_gait_qualified'):
        assert result[name] is False
    np.testing.assert_array_equal(result['plane_anchor_observation_m'], p.anchor)


def test_cached_and_reference_whole_footprints_agree(geometry):
    f = frame(); p = fit_multipixel_plane(f)
    a = query(f, p, geometry); b = query(f, p, geometry, backend='cached')
    assert a['floor_coverage'] == b['floor_coverage'] and a['gap_bounds'] == b['gap_bounds']
    for key in ('lower_cells_xy', 'upper_cells_xy', 'invalid_cells', 'covered_cells',
                'measured_planes_within_supplied_family'):
        np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize('fault', ['missing', 'step', 'submillimetre_spike'])
def test_patch_is_not_silently_trimmed_or_inpainted(fault):
    f = frame(); p = fit_multipixel_plane(f)
    r, c = p.seed_cell_rc; d, v = f._depth.copy(), f._valid.copy()
    # Preserve the median selection approximately; corruption is inside fixed patch.
    if fault == 'missing': d[r, c] = 0.; v[r, c] = False
    elif fault == 'step': d[r, c] += .03
    else: d[r, c] += .0001
    changed = PreparedFloorFrame(d, v, f._up); rejected = fit_multipixel_plane(changed)
    assert rejected.status != 'MEASURED_PATCH_CONSISTENT'
    assert rejected.anchor is None and rejected.normal is None
    with pytest.raises(SensorContractError): rejected.for_frame(changed)


def test_insufficient_tangent_spread_retains_unknown():
    f = frame(); p = fit_multipixel_plane(f, rules=PatchRules(minimum_tangent_rms_m=1.))
    assert p.status == 'FIXED_PATCH_ILL_CONDITIONED' and p.anchor is None


def test_missing_entire_frame_and_border_do_not_search_for_another_patch():
    f = PreparedFloorFrame(np.zeros((480, 640), np.float32), np.zeros((480, 640), bool), [0, 0, 1])
    assert fit_multipixel_plane(f).status == 'NO_ELIGIBLE_CELL'
    original = frame(); d = np.zeros_like(original._depth); v = np.zeros_like(original._valid)
    d[478:480, 320:322] = original._depth[478:480, 320:322]; v[478:480, 320:322] = True
    edge = PreparedFloorFrame(d, v, original._up)
    p = fit_multipixel_plane(edge)
    assert p.seed_cell_rc == (478, 320) and p.status == 'FIXED_PATCH_OUTSIDE_IMAGE'


def test_identity_and_immutability():
    f = frame(); p = fit_multipixel_plane(f)
    with pytest.raises(FrozenInstanceError): p.anchor = (0, 0, 0)
    with pytest.raises(ValueError): f._depth[0, 0] = 0
    d = f._depth.copy(); d[0, 0] = 0; v = f._valid.copy(); v[0, 0] = False
    # Use a guaranteed change outside the selected patch.
    d[401, 200] *= np.float32(1.0001)
    other = PreparedFloorFrame(d, v, f._up)
    with pytest.raises(SensorContractError): p.for_frame(other)
    changed_up = PreparedFloorFrame(f._depth, f._valid, [0., .001, np.sqrt(1-.001**2)])
    with pytest.raises(SensorContractError): p.for_frame(changed_up)
    with pytest.raises(SensorContractError): replace(p, policy='search-until-clear').for_frame(f)


@pytest.mark.parametrize('fault', ['missing', 'step'])
def test_fit_cannot_bridge_an_obstacle_or_hole_outside_patch(geometry, fault):
    f = frame(); p = fit_multipixel_plane(f); before = query(f, p, geometry)
    i = list(before['floor_coverage']).index('FL_foot:0')
    x, y = before['lower_cells_xy'][i]+[1, 1]
    r, c = p.seed_cell_rc
    assert abs(y-r) > p.rules.radius_cells+1 or abs(x-c) > p.rules.radius_cells+1
    d, v = f._depth.copy(), f._valid.copy()
    if fault == 'missing': d[y, x] = 0.; v[y, x] = False
    else: d[y, x] += .03
    changed = PreparedFloorFrame(d, v, f._up); cp = fit_multipixel_plane(changed)
    assert cp.status == 'MEASURED_PATCH_CONSISTENT'
    for backend in ('reference', 'cached'):
        after = query(changed, cp, geometry, backend=backend)
        assert not after['floor_coverage']['FL_foot:0'] and after['invalid_cells'][i] > 0


def test_no_unseen_floor_or_penetration_permission(geometry):
    f = frame(); p = fit_multipixel_plane(f)
    behind = query(f, p, geometry, t=(-1., 0., 0.))
    assert not any(behind['floor_coverage'].values())
    penetrated = query(f, p, geometry, t=(1.5, 0., -.1))
    result = assess_primitive_floor_relation(penetrated['gap_bounds'],
        floor_coverage=penetrated['floor_coverage'],
        non_floor_clearance=dict.fromkeys(penetrated['floor_coverage'], True))
    feet = [s for s in result['primitives'] if s['shape_id'].endswith('_foot:0')]
    assert len(feet) == 4 and all(s['penetration_under_every_supplied_model'] for s in feet)
    assert all(not s['conditional_clearance'] and not s['contact_permitted'] for s in feet)


@pytest.mark.parametrize('bad', [dict(radius_cells=True), dict(radius_cells=0),
    dict(radius_cells=33), dict(maximum_pixel_residual_m=np.nan),
    dict(maximum_triangle_normal_error=1.), dict(minimum_tangent_ratio=2.),
    dict(minimum_tangent_rms_m=-1.)])
def test_invalid_rules_rejected(bad):
    with pytest.raises(SensorContractError): PatchRules(**bad)


def test_bad_transform_rejected(geometry):
    f = frame(); p = fit_multipixel_plane(f)
    with pytest.raises(SensorContractError): query(f, p, geometry, R=np.diag([-1., 1., 1.]))


@pytest.mark.parametrize('gain', [-.001, 0., .001])
def test_finite_gain_through_float32_matches_independent_plane_geometry(gain):
    f = frame(); original = fit_multipixel_plane(f)
    d = (f._depth.astype(float)*(1+gain)).astype(np.float32)
    changed = PreparedFloorFrame(d, f._valid, f._up); p = fit_multipixel_plane(changed)
    assert p.status == 'MEASURED_PATCH_CONSISTENT'
    # Scaling optical range scales space around the camera centre, not body origin.
    centre = np.asarray(BODY_FROM_OPTICAL)[:3, 3]
    expected_anchor = centre+(1+gain)*(np.asarray(original.anchor)-centre)
    np.testing.assert_allclose(p.normal, original.normal, atol=2e-6, rtol=0)
    assert abs(np.asarray(p.normal) @ (np.asarray(p.anchor)-expected_anchor)) < 1e-6


def test_tilted_elevated_measured_plane_not_replaced_with_world_floor():
    n = np.array([.03, -.02, 1.]); n /= np.linalg.norm(n)
    a = np.array([0., 0., -.22]); T = np.asarray(BODY_FROM_OPTICAL)
    yy, xx = np.mgrid[:480, :640]
    rays = np.stack(((xx+.5-320)/FOCAL, (yy+.5-240)/FOCAL, np.ones_like(xx)), axis=-1)
    directions = rays@T[:3, :3].T
    denominator = directions@n
    z = np.divide(n@(a-T[:3, 3]), denominator, out=np.zeros_like(denominator), where=abs(denominator)>1e-12)
    valid = (z >= .2)&(z <= 5.)
    f = PreparedFloorFrame(np.where(valid, z, 0).astype(np.float32), valid, [0, 0, 1])
    p = fit_multipixel_plane(f)
    assert p.status == 'MEASURED_PATCH_CONSISTENT'
    np.testing.assert_allclose(p.normal, n, atol=2e-6, rtol=0)
    assert abs(n@(np.asarray(p.anchor)-a)) < 1e-6


def test_recorded_member_population_preserves_actual_representation_and_missing_data():
    from scripts.probe_go2_multipixel_floor_plane_development_v1 import members
    f = frame(); d, valid = f._depth.copy(), f._valid.copy()
    d[390, 310] = 0.; valid[390, 310] = False
    population = list(members(dict(depth_m=d, valid=valid), 0))
    assert len(population) == 17 and len({n for n, _ in population}) == 17
    for name, z in population:
        assert z.dtype == np.float32 and not np.any(z[~valid])
        assert not np.shares_memory(z, d)
    np.testing.assert_array_equal(population[0][1], d)
    for (_, a), (_, b) in zip(population, members(dict(depth_m=d, valid=valid), 0), strict=True):
        np.testing.assert_array_equal(a, b)


def test_comparison_preserves_rejected_pairs_as_unknown():
    from scripts.probe_go2_multipixel_floor_plane_development_v1 import compare, members
    f = frame()
    rows = [dict(member=name, three_point_gap_m=[1., 2.]) for name, _ in
            members(dict(depth_m=f._depth, valid=f._valid), 0)]
    rows[0]['multipixel_gap_m'] = [1., 2.]
    result = compare(rows)
    for source in ('gain', 'offset'):
        assert result['multipixel'][source]['factor_step_difference_m'] is None
        assert all(v is None for v in result['multipixel'][source]['central_factors_m'].values())
        np.testing.assert_array_equal(result['three_point'][source]['factor_step_difference_m'], [0., 0.])
