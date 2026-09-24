from itertools import product

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.coupled_floor_enclosure_development import coupled_footprint_enclosure, query_observed_coupled_floor
from lewm.floor_footprint_bounds_development import projected_footprint_rectangles
from lewm.tests.test_floor_footprint_bounds_development import scene


def bound(low=None, high=None, **kwargs):
    return coupled_footprint_enclosure(
        [[1.45, -.04, -.28]] if low is None else low,
        [[1.55, .04, -.26]] if high is None else high,
        [[1.5, 0., -.3]], [[0., 0., 1.]], [0., 0., 1.],
        normal_error=kwargs.get('normal_error', 0.), up_error=kwargs.get('up_error', 0.),
        plane_offset_error=kwargs.get('plane_offset_error', 0.))


def test_coupling_removes_impossible_vertical_footprint_extremes():
    result = bound()
    np.testing.assert_allclose(result['footprint_lower_m'], [[1.45, -.04, -.3]], atol=2e-12)
    np.testing.assert_allclose(result['footprint_upper_m'], [[1.55, .04, -.3]], atol=2e-12)
    np.testing.assert_allclose(result['height_lower_m'], [.02], atol=2e-12)
    np.testing.assert_allclose(result['height_upper_m'], [.04], atol=2e-12)
    assert result['height_within_existing_band'].all()
    old = projected_footprint_rectangles([[1.45, -.04, -.28]], [[1.55, .04, -.26]], [-.06], [.06], [0, 0, 1])
    new = projected_footprint_rectangles(result['footprint_lower_m'], result['footprint_upper_m'], [0.], [0.], [0, 0, 1])
    assert (new['lower_cells_xy'] >= old['lower_cells_xy']).all()
    assert (new['upper_cells_xy'] <= old['upper_cells_xy']).all()
    assert new['lower_cells_xy'][0, 1] > old['lower_cells_xy'][0, 1]
    assert not result['ground_support_approved'] and not result['supplied_error_bounds_validated']


@pytest.mark.parametrize('seed', [271, 918, 3187])
def test_exact_perturbed_plane_intersections_inside_finite_enclosure(seed):
    rng = np.random.default_rng(seed)
    u = np.array([.04, -.06, 1.]); u /= np.linalg.norm(u)
    n = np.array([-.02, .04, 1.]); n /= np.linalg.norm(n)
    a = np.array([1.5, .05, -.3]); low = a + [-.2, -.1, -.04]; high = a + [.15, .2, .03]
    en, eu, eb = .035, .017, .006
    result = coupled_footprint_enclosure(low[None], high[None], a[None], n[None], u,
                                        normal_error=en, up_error=eu, plane_offset_error=eb)
    q = rng.uniform(low, high, (20000, 3))
    # Include every box vertex and extreme error magnitudes, not just centres.
    q[:8] = np.array(list(product(*zip(low, high))))
    dn, du = rng.normal(size=(2, len(q), 3))
    dn *= en / np.linalg.norm(dn, axis=1)[:, None]
    du *= eu / np.linalg.norm(du, axis=1)[:, None]
    b = rng.choice([-eb, eb], len(q)); nn, uu = n + dn, u + du
    heights = (np.sum(nn * (q - a), axis=1) - b) / np.sum(nn * uu, axis=1)
    feet = q - heights[:, None] * uu
    assert (feet >= result['footprint_lower_m']).all() and (feet <= result['footprint_upper_m']).all()
    assert (heights >= result['height_lower_m']).all() and (heights <= result['height_upper_m']).all()


def test_error_allowances_never_shrink_enclosure():
    previous = bound()
    for error in (.001, .01, .1):
        current = bound(normal_error=error, up_error=error, plane_offset_error=error)
        assert (current['footprint_lower_m'] <= previous['footprint_lower_m']).all()
        assert (current['footprint_upper_m'] >= previous['footprint_upper_m']).all()
        assert (current['height_lower_m'] <= previous['height_lower_m']).all()
        assert (current['height_upper_m'] >= previous['height_upper_m']).all()
        previous = current
    assert not previous['height_within_existing_band'].any()


@pytest.mark.parametrize('fault', ['reverse', 'nan', 'normal', 'up', 'negative', 'parallel', 'overflow'])
def test_bad_enclosures_fail_closed(fault):
    low = np.array([[1.45, -.04, -.28]]); high = np.array([[1.55, .04, -.26]])
    n = np.array([[0., 0., 1.]]); u = np.array([0., 0., 1.]); en = 0.
    if fault == 'reverse': high[0, 0] = 0.
    if fault == 'nan': low[0, 0] = np.nan
    if fault == 'normal': n *= 2
    if fault == 'up': u *= 2
    if fault == 'negative': en = -.01
    if fault == 'parallel': n[:] = [1., 0., 0.]
    if fault == 'overflow': low[0, 0] = -1e308; high[0, 0] = 1e308
    with pytest.raises(SensorContractError):
        coupled_footprint_enclosure(low, high, [[1.5, 0., -.3]], n, u,
                                    normal_error=en, up_error=0., plane_offset_error=0.)


def observed(d, valid, low=None, high=None, **kwargs):
    return query_observed_coupled_floor(d, valid,
        [[1.45, -.04, -.28]] if low is None else low,
        [[1.55, .04, -.26]] if high is None else high, [0, 0, 1], np.array([True]),
        normal_error=kwargs.get('normal_error', .002), up_error=kwargs.get('up_error', .001),
        plane_offset_error=kwargs.get('plane_offset_error', .001))


def test_observed_region_checks_plane_family_without_approval():
    d, valid = scene(); result = observed(d, valid)
    assert result['seed_observed'].all() and result['conditional_mesh_coverage'].all()
    assert result['covered_cells'][0] > 10
    assert not result['ground_support_approved'] and not result['navigation_qualified']
    assert not result['supplied_error_bounds_validated'] and not result['continuous_surface_qualified']


def test_missing_interior_cell_rejects_despite_observed_seed():
    d, valid = scene(); reference = observed(d, valid)
    x, y = reference['lower_cells_xy'][0] + [1, 1]
    d[y, x] = 0.; valid[y, x] = False
    result = observed(d, valid)
    assert result['seed_observed'].all()
    assert result['invalid_cells'][0] > 0
    assert not result['conditional_mesh_coverage'].any()


def test_non_ground_discontinuity_rejects_without_cropping_region():
    d, valid = scene(); reference = observed(d, valid)
    x, y = reference['lower_cells_xy'][0] + [1, 1]
    d[y, x] += .03  # A valid return, but not a smooth floor triangle.
    result = observed(d, valid)
    assert result['seed_observed'].all() and result['invalid_cells'][0] > 0
    assert not result['conditional_mesh_coverage'].any()


def test_quantized_nominal_mesh_can_violate_zero_error_plane_family():
    d, valid = scene(); result = observed(d, valid, normal_error=0., up_error=0., plane_offset_error=0.)
    assert result['all_projected_cells_observed_ground'].all()
    assert not result['measured_planes_within_supplied_family'].any()
    assert not result['conditional_mesh_coverage'].any()


def test_height_failure_does_not_clip_supplied_box_to_make_it_pass():
    d, valid = scene(); result = observed(d, valid, [[1.45, -.04, -.30]], [[1.55, .04, -.235]])
    assert result['seed_observed'].all() and result['all_projected_cells_observed_ground'].all()
    assert result['height_upper_m'][0] > .06
    assert not result['height_within_existing_band'].any() and not result['conditional_mesh_coverage'].any()


def test_no_seed_and_non_ground_role_cannot_approve():
    d, valid = scene()
    for point, role in (([1.5, 0., 0.], True), ([1.5, 0., -.27], False)):
        result = query_observed_coupled_floor(d, valid, [point], [point], [0, 0, 1], np.array([role]),
                                             normal_error=.002, up_error=.001, plane_offset_error=.001)
        assert not result['conditional_mesh_coverage'].any()


def test_empty_bounds_remain_empty():
    result = coupled_footprint_enclosure(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)),
                                        np.empty((0, 3)), [0, 0, 1], normal_error=0., up_error=0., plane_offset_error=0.)
    assert result['footprint_lower_m'].shape == (0, 3)
