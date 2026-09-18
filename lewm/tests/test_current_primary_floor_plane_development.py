import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.observed_geometry_refinement_development import sampled_floor_patch
from lewm.current_primary_floor_plane_development import (
    ROWS, COLUMNS, primary_floor_plane, confirm_auxiliary_floor)


def depth_plane(height=-.303, slope=-.007):
    yy, xx = np.indices((480, 640)); T = np.asarray(BODY_FROM_OPTICAL)
    optical = np.stack(((xx+.5-320)/FOCAL, (yy+.5-240)/FOCAL, np.ones_like(xx)), axis=-1)
    rays = optical@T[:3,:3].T; normal = np.array([-slope, 0., 1.])
    denominator = rays@normal
    z = np.divide(height-normal@T[:3,3], denominator, out=np.zeros_like(denominator), where=np.abs(denominator)>1e-12)
    valid = (z >= .2) & (z <= 5.)
    return np.where(valid, z, 0.).astype(np.float32), valid


def fit(d, v):
    return primary_floor_plane(d, v, np.eye(3), np.zeros(3), -.32)


def confirm(d, v, plane):
    return confirm_auxiliary_floor(d, v, np.eye(3), np.zeros(3), -.32, ROWS, COLUMNS, plane)


def test_measured_plane_extends_only_observed_local_floor_without_changing_originals():
    d, v = depth_plane(); plane = fit(d, v)
    assert plane['available'] and plane['seed_count'] >= 100
    expected = np.array([.007, 0., 1.]); expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(plane['normal_map'], expected, atol=1e-6, rtol=0)
    old = sampled_floor_patch(d, v, np.eye(3), np.zeros(3), -.32, ROWS, COLUMNS)['measured_floor_patch']
    new, receipt = confirm(d, v, plane)
    assert np.all(new[old]) and receipt['added_floor_count'] > 0
    assert not receipt['ground_support_approved'] and not receipt['unobserved_space_certified']
    assert receipt['measured_plane_band_m'] == .01
    # A missing neighbor of a newly confirmed sample cannot be inferred free.
    i, j = np.argwhere(new & ~old)[0]; r, c = ROWS[i], COLUMNS[j]
    d[r-1, c-1] = 0.; v[r-1, c-1] = False
    masked, _ = confirm(d, v, plane)
    assert not masked[i, j]


def test_raised_patch_wall_and_missing_depth_remain_unconfirmed():
    d, v = depth_plane(); plane = fit(d, v)
    raised, rv = depth_plane(height=-.283)
    mask, receipt = confirm(raised, rv, plane)
    original_raised = sampled_floor_patch(raised, rv, np.eye(3), np.zeros(3), -.32, ROWS, COLUMNS)['measured_floor_patch']
    np.testing.assert_array_equal(mask, original_raised)
    assert receipt['added_floor_count'] == 0
    wall = np.ones((480, 640), np.float32); valid = np.ones_like(wall, bool)
    mask, _ = confirm(wall, valid, plane)
    assert not mask.any()
    mask, receipt = confirm(np.zeros_like(d), np.zeros_like(v), plane)
    assert not mask.any()
    assert receipt['added_floor_count'] == 0


def test_missing_primary_floor_keeps_original_classification_exact():
    wall = np.ones((480, 640), np.float32); valid = np.ones_like(wall, bool)
    unavailable = fit(wall, valid)
    assert not unavailable['available']
    d, v = depth_plane(); mask, receipt = confirm(d, v, unavailable)
    old = sampled_floor_patch(d, v, np.eye(3), np.zeros(3), -.32, ROWS, COLUMNS)['measured_floor_patch']
    np.testing.assert_array_equal(mask, old)
    assert receipt['added_floor_count'] == 0
