"""Analytic recovery, constrained optimum, coordinate invariance and rejection."""
from copy import deepcopy
import numpy as np
import pytest

from lewm.measured_plane_rigid_fit_development import fit
from lewm.joint_measured_floor_plane_development import fit_joint_plane


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def plane(n, d):
    n = np.asarray(n, float)
    e = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = e-n*(n@e)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    xx, yy = np.meshgrid(np.linspace(-.5, .5, 12), np.linspace(-.4, .4, 12))
    points = xx.ravel()[:, None]*u+yy.ravel()[:, None]*v-d*n
    return fit_joint_plane(points, points.copy(), n)


def example(noise=0., seed=45):
    rng = np.random.default_rng(seed)
    R = rotation([1., 2., 3.], .43)
    t = np.array([.04, -.03, .012])
    na = np.array([0., 0., 1.])
    nb = R.T@na
    b = rng.normal(size=(80, 3))*.3
    a = b@R.T+t+rng.normal(size=b.shape)*noise
    return a, b, plane(na, .32), plane(nb, .32+na@t), R, t


@pytest.mark.parametrize('seed', [3, 45, 876])
def test_recovers_known_rigid_motion_and_plane_offset_sign(seed):
    a, b, pa, pb, expected_R, expected_t = example(seed=seed)
    R, t, report = fit(a, b, pa, pb)
    np.testing.assert_allclose(R, expected_R, atol=1e-12, rtol=0)
    np.testing.assert_allclose(t, expected_t, atol=1e-12, rtol=0)
    assert report['residual_rms_m'] < 1e-12
    assert report['plane_normal_residual'] < 1e-12
    assert abs(report['plane_offset_residual_m']) < 1e-12
    assert not report['pose_admitted']


def test_noisy_fit_is_global_minimum_along_feasible_yaw_family():
    a, b, pa, pb, _, _ = example(noise=.003)
    R, t, _ = fit(a, b, pa, pb)
    normal = np.asarray(pa['normal_body'])
    objective = np.sum((a-b@R.T-t)**2)
    # Every feasible rotation is a twist about the reference normal from R.
    # Translation's two unconstrained coordinates have their analytic optimum.
    for theta in np.linspace(-np.pi, np.pi, 1001):
        candidate_R = rotation(normal, theta)@R
        candidate_t = a.mean(0)-candidate_R@b.mean(0)
        candidate_t += normal*(pb['offset_body_m']-pa['offset_body_m']-normal@candidate_t)
        value = np.sum((a-b@candidate_R.T-candidate_t)**2)
        assert value >= objective-1e-12


def test_same_coordinate_rotation_leaves_physical_solution_unchanged():
    a, b, pa, pb, _, _ = example(noise=.002)
    R, t, _ = fit(a, b, pa, pb)
    Q = rotation([3., -1., 2.], 1.2)
    qa = plane(Q@pa['normal_body'], pa['offset_body_m'])
    qb = plane(Q@pb['normal_body'], pb['offset_body_m'])
    changed_R, changed_t, _ = fit(a@Q.T, b@Q.T, qa, qb)
    np.testing.assert_allclose(changed_R, Q@R@Q.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(changed_t, Q@t, atol=1e-12, rtol=0)


def test_inputs_are_not_mutated_and_repeated_fit_is_exact():
    a, b, pa, pb, _, _ = example(noise=.003)
    before = a.copy(), b.copy(), deepcopy(pa), deepcopy(pb)
    first = fit(a, b, pa, pb)
    second = fit(a, b, pa, pb)
    assert np.array_equal(a, before[0]) and np.array_equal(b, before[1])
    assert pa == before[2] and pb == before[3]
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert first[2] == second[2]


@pytest.mark.parametrize('invalid', ['nan_points', 'shape', 'line', 'plane_unavailable', 'plane_residual'])
def test_rejects_invalid_or_unqualified_inputs(invalid):
    a, b, pa, pb, _, _ = example()
    if invalid == 'nan_points': a[0, 0] = np.nan
    elif invalid == 'shape': b = b[:-1]
    elif invalid == 'line': a[:, 1:] = 0
    elif invalid == 'plane_unavailable': pa['available'] = False
    elif invalid == 'plane_residual': pa['camera_residuals'][0]['maximum_residual_m'] = .004
    with pytest.raises((ValueError, TypeError)):
        fit(a, b, pa, pb)


def test_zero_tangent_cross_covariance_cannot_choose_an_arbitrary_yaw():
    # Both clouds are noncollinear; their paired tangent cross covariance is zero.
    a = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    b = np.array([[0., 1., 0.], [0., -1., 0.], [1., 0., 0.], [-1., 0., 0.]])
    p = plane([0., 0., 1.], .32)
    with pytest.raises(ValueError, match='yaw must be identifiable'):
        fit(a, b, p, p)
