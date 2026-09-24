import numpy as np
import pytest

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.correlated_floor_evidence_development import patch_relation, _sample
from lewm.floor_footprint_bounds_development import (
    observed_floor_cell_index, projected_footprint_rectangles, query_floor_bounds)
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth


def scene():
    _, p, _, _ = next(stream(1)); d = room_depth(p)
    return d['depth_m'], d['valid']


def test_cell_index_matches_direct_triangle_eligibility_and_is_immutable():
    d, valid = scene(); d[300:304, 310:314] = 0.; valid[300:304, 310:314] = False
    index = observed_floor_cell_index(d, valid, [0, 0, 1])
    rng = np.random.default_rng(2951)
    cells = np.stack((rng.integers(0, 479, 500), rng.integers(0, 639, 500)), axis=1)
    patches, masks = _sample(d, valid, cells)
    q = patches.mean(axis=1)
    direct = patch_relation(q, patches, masks, [0, 0, 1])
    np.testing.assert_array_equal(index['ground_cells'][cells[:, 0], cells[:, 1]], direct['observed_footprint'])
    before = index['ground_cells'].copy(); d[:] = 0.; valid[:] = False
    np.testing.assert_array_equal(index['ground_cells'], before)
    with pytest.raises(ValueError): index['ground_cells'][0, 0] = True
    with pytest.raises(TypeError): index['up'] = np.array([0, 1, 0])


def test_positive_floor_box_coverage_is_not_navigation_approval():
    d, valid = scene(); index = observed_floor_cell_index(d, valid, [0, 0, 1])
    row = query_floor_bounds(index, [[1.45, -.04, -.28]], [[1.55, .04, -.26]], [.015], [.045], np.array([True]))
    assert row['all_projected_cells_observed_ground'].all()
    assert row['covered_cells'][0] > 20 and row['invalid_cells'][0] == 0
    assert not row['supplied_bounds_validated'] and not row['uncertain_surface_geometry_covered']
    assert not row['ground_support_approved'] and not row['navigation_qualified']


def test_missing_cell_inside_region_is_not_missed_by_endpoint_samples():
    d, valid = scene()
    low, high = np.array([[1.5, -.1, -.27]]), np.array([[1.5, .1, -.27]])
    rect = projected_footprint_rectangles(low, high, [.03], [.03], [0, 0, 1])
    a, b = rect['lower_cells_xy'][0], rect['upper_cells_xy'][0]
    column = int((a[0] + b[0]) // 2); row = int(a[1])
    d[row, column] = 0.; valid[row, column] = False
    index = observed_floor_cell_index(d, valid, [0, 0, 1])
    ends = query_floor_bounds(index, np.vstack((low, high)), np.vstack((low, high)),
                              [.03, .03], [.03, .03], np.ones(2, bool))
    whole = query_floor_bounds(index, low, high, [.03], [.03], np.array([True]))
    assert ends['all_projected_cells_observed_ground'].all()
    assert not whole['all_projected_cells_observed_ground'].any()
    assert whole['invalid_cells'][0] > 0


def test_projection_rectangle_contains_random_prism_points_with_tilted_up():
    rng = np.random.default_rng(8642); up = np.array([.05, -.03, 1.]); up /= np.linalg.norm(up)
    low, high = np.array([[1.1, -.1, -.34]]), np.array([[1.8, .12, -.26]])
    rect = projected_footprint_rectangles(low, high, [-.05], [.06], up)
    q = rng.uniform(low[0], high[0], (10000, 3)); h = rng.uniform(-.05, .06, (10000, 1))
    feet = q - h * up; transform = np.asarray(BODY_FROM_OPTICAL)
    optical = (feet - transform[:3, 3]) @ transform[:3, :3]
    uv = FOCAL * optical[:, :2] / optical[:, 2, None] + [319.5, 239.5]
    cells = np.floor(uv).astype(int)
    assert (cells >= rect['lower_cells_xy'][0]).all() and (cells <= rect['upper_cells_xy'][0]).all()


def test_growing_box_cannot_gain_floor_coverage_or_lose_an_internal_missing_cell():
    d, valid = scene(); d[353, 320] = 0.; valid[353, 320] = False
    index = observed_floor_cell_index(d, valid, [0, 0, 1]); centre = np.array([[1.5, .013, -.27]])
    previous = True; missing = 0
    for radius in (0., .001, .01, .03, .1):
        row = query_floor_bounds(index, centre - radius, centre + radius, [.03], [.03], np.array([True]))
        assert previous or not row['all_projected_cells_observed_ground'][0]
        if row['projection_within_observed_camera'][0]: assert row['invalid_cells'][0] >= missing
        previous = bool(row['all_projected_cells_observed_ground'][0]); missing = int(row['invalid_cells'][0])


@pytest.mark.parametrize('point', [[-.5, 0, -.3], [.4, 0, -.3], [6., 0, -.3], [1.5, 2., -.3]])
def test_out_of_camera_region_never_approves_coverage(point):
    d, valid = scene(); index = observed_floor_cell_index(d, valid, [0, 0, 1])
    result = query_floor_bounds(index, [point], [point], [0.], [0.], np.array([True]))
    assert not result['all_projected_cells_observed_ground'].any()


@pytest.mark.parametrize('fault', ['reversed_box', 'reversed_height', 'height_expansion', 'nan', 'up', 'role'])
def test_invalid_bounds_rejected(fault):
    d, valid = scene(); index = observed_floor_cell_index(d, valid, [0, 0, 1])
    low = np.array([[1.5, 0, -.27]]); high = low.copy(); hl = [.03]; hh = [.03]; roles = np.array([True])
    if fault == 'reversed_box': high[0, 0] -= .1
    elif fault == 'reversed_height': hl = [.04]
    elif fault == 'height_expansion': hh = [.061]
    elif fault == 'nan': low[0, 0] = np.nan
    elif fault == 'up':
        with pytest.raises(SensorContractError): projected_footprint_rectangles(low, high, hl, hh, [0, 0, 2])
        return
    elif fault == 'role': roles = np.array([1])
    with pytest.raises(SensorContractError): query_floor_bounds(index, low, high, hl, hh, roles)


def test_empty_and_non_ground_queries_do_not_approve_coverage():
    d, valid = scene(); index = observed_floor_cell_index(d, valid, [0, 0, 1])
    for points, h, roles in ((np.empty((0, 3)), [], np.zeros(0, bool)),
                             (np.array([[1.5, 0, -.27]]), [.03], np.array([False]))):
        result = query_floor_bounds(index, points, points, h, h, roles)
        assert not result['all_projected_cells_observed_ground'].any()
