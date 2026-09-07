"""Analytic/adversarial conditional surface checks, not sensor calibration."""
import numpy as np
import pytest

from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.bounded_depth_surface_development import BoundedDepthSurface, triangle_orientation_bounds
from lewm.causal_sensor_state import SensorContractError
from lewm.primitive_floor_relation_development import assess_primitive_floor_relation
from lewm.tests.test_floor_footprint_bounds_development import scene
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def surface(d=None, v=None, *, noise=.0001, tube=.001):
    if d is None: d, v = scene()
    return BoundedDepthSurface(d.astype(np.float32), v, [0., 0., 1.],
        range_error_m=noise, surface_tube_m=tube, up_error=.001)


@pytest.fixture(scope='module')
def model():
    return ArticulatedCollisionGeometry(URDF)


@pytest.fixture(scope='module')
def nominal():
    return surface()


def query(s, model, *, t=(1.5, 0., 0.), backend='prefix'):
    q = np.repeat([0., .8, -1.5], 4)
    errors = {r['shape_id']: .001 for r in model.supports(q, np.eye(3))['shapes']}
    return s.query(model, q, rotation_observation_from_body=np.eye(3),
        translation_observation_from_body=t, point_error_by_shape=errors, backend=backend)


def test_declared_interval_surface_preserves_physical_clearance_semantics(nominal, model):
    assert nominal.status == 'BOUNDED_MEASURED_SURFACE_AVAILABLE'
    result = query(nominal, model)
    assert len(result['floor_coverage']) == 27 and all(result['floor_coverage'].values())
    assert result['gap_bounds']['normal_error'] == 0
    assert result['gap_bounds']['plane_offset_error_m'] == .001
    assert result['reference_plane_not_assumed_true_surface']
    for key in ('supplied_error_bounds_validated', 'non_floor_clearance_established',
                'contact_permitted', 'navigation_qualified', 'future_gait_qualified',
                'true_surface_identity_established'):
        assert not result[key]


def test_prefix_reference_parity_and_immutable_arrays(nominal, model):
    a, b = query(nominal, model), query(nominal, model, backend='reference')
    assert a['floor_coverage'] == b['floor_coverage'] and a['gap_bounds'] == b['gap_bounds']
    for key in ('lower_cells_xy', 'upper_cells_xy', 'invalid_cells', 'covered_cells'):
        np.testing.assert_array_equal(a[key], b[key])
    for array in (nominal._mask, nominal._prefix, nominal.anchor, nominal.normal):
        assert not array.flags.writeable


@pytest.mark.parametrize('sign', [-1., 1.])
def test_independent_range_noise_not_mistaken_for_exact_triangle_normal(sign, model):
    d, v = scene(); d = d.astype(np.float32)
    delta = np.random.default_rng(9931).uniform(-.0001, .0001, d.shape)*v
    noisy = (d.astype(float)+sign*delta).astype(np.float32)
    s = surface(noisy, v)
    assert s.status == 'BOUNDED_MEASURED_SURFACE_AVAILABLE'
    assert s.diagnostics['maximum_patch_vertex_residual_bound_m'] < .001
    result = query(s, model)
    assert all(result['floor_coverage'].values())
    # Underlying analytic floor is z=-.3, not an uncertain fitted plane.
    assert abs(np.asarray(s.normal)@np.array([1.5, 0., -.3])-s.normal@s.anchor) < .001


@pytest.mark.parametrize('kind', ['missing', 'obstacle'])
def test_no_fit_through_corrupted_patch(nominal, kind):
    d, v = scene(); r, c = nominal.seed
    if kind == 'missing': d[r, c] = 0; v[r, c] = False
    else: d[r, c] += .03
    s = surface(d, v)
    assert s.status != 'BOUNDED_MEASURED_SURFACE_AVAILABLE'
    assert s.anchor is None and s.normal is None


@pytest.mark.parametrize('kind', ['missing', 'obstacle'])
def test_whole_footprint_does_not_bridge_holes_or_steps(nominal, model, kind):
    before = query(nominal, model); i = list(before['floor_coverage']).index('FL_foot:0')
    x, y = before['lower_cells_xy'][i]+[1, 1]; r, c = nominal.seed
    assert abs(y-r)>9 or abs(x-c)>9
    d, v = scene()
    if kind == 'missing': d[y, x] = 0; v[y, x] = False
    else: d[y, x] += .03
    s = surface(d, v); assert s.status == 'BOUNDED_MEASURED_SURFACE_AVAILABLE'
    for backend in ('prefix', 'reference'):
        after = query(s, model, backend=backend)
        assert not after['floor_coverage']['FL_foot:0'] and after['invalid_cells'][i] > 0


def test_no_unseen_floor_or_penetration_permission(nominal, model):
    assert not any(query(nominal, model, t=(-1., 0., 0.))['floor_coverage'].values())
    row = query(nominal, model, t=(1.5, 0., -.1))
    assessed = assess_primitive_floor_relation(row['gap_bounds'], floor_coverage=row['floor_coverage'],
        non_floor_clearance=dict.fromkeys(row['floor_coverage'], True))
    feet = [x for x in assessed['primitives'] if x['shape_id'].endswith('_foot:0')]
    assert len(feet) == 4 and all(x['penetration_under_every_supplied_model'] for x in feet)
    assert not any(x['contact_permitted'] for x in feet)


def test_error_cannot_be_inferred_from_fit_residual_or_exceed_tube():
    s = surface(noise=.01)
    assert s.status == 'FIXED_PATCH_OUTSIDE_SURFACE_TUBE'
    assert s.diagnostics['maximum_patch_vertex_residual_bound_m'] > .001


def test_range_intervals_include_float32_quantizer_even_at_zero_declared_noise():
    s = surface(noise=0., tube=1e-10)
    assert s.status == 'FIXED_PATCH_OUTSIDE_SURFACE_TUBE'


def test_all_missing_is_unknown(model):
    s = surface(np.zeros((480, 640), np.float32), np.zeros((480, 640), bool))
    assert s.status == 'NO_ELIGIBLE_SEED'
    with pytest.raises(SensorContractError): query(s, model)


@pytest.mark.parametrize('bad', [dict(range_error_m=-1), dict(surface_tube_m=0),
                               dict(up_error=np.nan), dict(up_error=1)])
def test_invalid_interval_contract(bad):
    d, v = scene(); args = dict(range_error_m=.0001, surface_tube_m=.001, up_error=.001)|bad
    with pytest.raises(SensorContractError): BoundedDepthSurface(d.astype(np.float32), v, [0, 0, 1], **args)


def test_projected_triangle_interval_contains_dependent_interior_perturbations():
    rng = np.random.default_rng(6071)
    rays = rng.normal(size=(40, 3, 3)); rays[..., 2] += 2.
    lo = rng.uniform(.5, 1., (40, 3)); hi = lo+rng.uniform(.001, .05, (40, 3))
    up = np.array([.03, -.02, 1.]); up /= np.linalg.norm(up); eu = .02
    bounds = triangle_orientation_bounds(rays, lo, hi, up, eu)
    for _ in range(100):
        z = rng.uniform(lo, hi); p = z[..., None]*rays
        n = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        du = rng.normal(size=(40, 3)); du *= eu/np.linalg.norm(du, axis=1)[:, None]
        value = np.sum(n*(up+du), axis=1)
        assert (value >= bounds['projected_normal_lower']).all()
        assert (value <= bounds['projected_normal_upper']).all()


def test_degenerate_or_orientation_reversing_triangle_is_not_certified():
    rays = np.array([[[0., 0., 1.], [.001, 0., 1.], [0., .001, 1.]]])
    lo = np.ones((1, 3)); hi = lo+.5
    # Viewed edge-on, the nominal projected orientation has no positive margin.
    b = triangle_orientation_bounds(rays, lo, hi, [1., 0., 0.], .001)
    assert b['projected_normal_lower'][0] < 0 < b['projected_normal_upper'][0]


def test_every_interval_vertex_and_interpolated_point_remains_in_tube(nominal):
    from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
    s = nominal; rng = np.random.default_rng(7181)
    cells = np.argwhere(s._mask)
    chosen = cells[rng.integers(0, len(cells), 400)]
    corners = chosen[:, None]+np.array([[0, 0], [0, 1], [1, 1]])[None]
    yy, xx = corners[..., 0], corners[..., 1]
    d = s.frame._depth[yy, xx]; z = d.astype(float)
    half = .5*np.maximum(np.nextafter(d, np.float32(np.inf)).astype(float)-z,
                         z-np.nextafter(d, np.float32(-np.inf)).astype(float))
    eps = .0001+half
    T = np.asarray(BODY_FROM_OPTICAL)
    rays = np.stack(((xx+.5-320)/FOCAL, (yy+.5-240)/FOCAL, np.ones_like(xx)), axis=-1)@T[:3, :3].T
    for sign in (-1, 1):
        points = (z+sign*eps)[..., None]*rays+T[:3, 3]
        assert (np.abs((points-s.anchor)@s.normal) <= .001).all()
        weights = rng.dirichlet([1., 1., 1.], len(points))
        interior = (points*weights[..., None]).sum(axis=1)
        assert (np.abs((interior-s.anchor)@s.normal) <= .001).all()


def test_saved_point_errors_require_exact_view_and_shape_identity():
    from scripts.probe_go2_bounded_depth_surface_development_v1 import original_errors
    row = dict(primitives=[dict(shape_id='base:0', ground_witnesses=[dict(measured_ns=123,
        depth_sha256='original', gap=dict(shape_id='base:0', point_error_m=.028))])])
    assert original_errors(row, 123, 'original') == {'base:0': .028}
    for ns, sha in ((124, 'original'), (123, 'different')):
        with pytest.raises(ValueError): original_errors(row, ns, sha)
    row['primitives'][0]['ground_witnesses'][0]['gap']['shape_id'] = 'FL_foot:0'
    with pytest.raises(ValueError): original_errors(row, 123, 'original')


def test_summary_does_not_report_unseen_numerical_gap_as_clearance():
    from scripts.probe_go2_bounded_depth_surface_development_v1 import summarize
    result = summarize(dict(floor_coverage={'base:0': False}, gap_bounds=dict(primitives=[dict(
        shape_id='base:0', minimum_gap_lower_m=1., minimum_gap_upper_m=2.)])))
    assert result == dict(covered=0, covered_separated=0, covered_ambiguous=[], covered_penetrated=[])
