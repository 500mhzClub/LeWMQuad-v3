from fractions import Fraction as F
from itertools import product
import numpy as np
import pytest
from lewm.causal_depth_observation_development import FOCAL, BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.pixel_support_point_bounds_development import point_bounds


def exact(value): return F.from_float(float(value))


@pytest.mark.parametrize('mount', [np.asarray(BODY_FROM_OPTICAL), body_from_optical()])
def test_encloses_exact_rational_backprojection_for_both_public_mounts(mount):
    depths = np.array([.2, 1.3, 5.], dtype=np.float64)
    rr = np.array([0, 239, 479]); cc = np.array([0, 319, 639])
    before = depths.copy(); radius = .5; error = .001
    r = point_bounds(depths, rr, cc, mount, pixel_radius=radius, depth_error_m=error)
    for i in range(len(depths)):
        for su, sv, sz in product((-1, 1), repeat=3):
            z = exact(depths[i])+sz*exact(error)
            u = exact(cc[i])+F(1, 2)+su*exact(radius)
            v = exact(rr[i])+F(1, 2)+sv*exact(radius)
            optical = [(u-320)/exact(FOCAL)*z, (v-240)/exact(FOCAL)*z, z]
            for axis in range(3):
                value = exact(mount[axis, 3])+sum(exact(mount[axis, j])*optical[j] for j in range(3))
                assert exact(r['lower_body_m'][i, axis]) <= value <= exact(r['upper_body_m'][i, axis])
    np.testing.assert_array_equal(depths, before)
    assert r['optical_depth_lower_m'][0] < .2 and r['optical_depth_upper_m'][-1] > 5.
    assert not r['public_valid_range_used_to_clip_uncertainty']
    assert not r['supplied_sensor_error_bound_validated'] and not r['pose_error_included']


def test_larger_bounds_never_shrink_point_enclosure():
    args = (np.array([.3, 4.9]), np.array([12, 466]), np.array([16, 626]), body_from_optical())
    a = point_bounds(*args, pixel_radius=1/256, depth_error_m=0.)
    b = point_bounds(*args, pixel_radius=.5, depth_error_m=.001)
    assert np.all(b['lower_body_m'] <= a['lower_body_m'])
    assert np.all(b['upper_body_m'] >= a['upper_body_m'])


@pytest.mark.parametrize('bad', ['depth', 'index', 'pixel_radius', 'depth_error', 'mount'])
def test_missing_or_invalid_measurement_assumptions_rejected(bad):
    d = np.array([1.]); rows = np.array([240]); cols = np.array([320]); T = np.eye(4)
    radius, error = .5, .001
    if bad == 'depth': d[0] = 0.
    elif bad == 'index': cols[0] = 640
    elif bad == 'pixel_radius': radius = 1.
    elif bad == 'depth_error': error = float('nan')
    else: T[2, 2] = 0.
    with pytest.raises(ValueError): point_bounds(d, rows, cols, T, pixel_radius=radius, depth_error_m=error)
