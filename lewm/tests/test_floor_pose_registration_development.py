import numpy as np
import pytest

from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.floor_pose_registration_development import (
    ROWS, COLUMNS, measured_candidates, fit_measured_plane, paired_plane, register_pose)


def rotation(axis, angle):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*K+(1.-np.cos(angle))*(K@K)


def plane(n=(0., 0., 1.), d=.32):
    return dict(normal_body=list(n), offset_body_m=d)


def render_plane(E, n, d):
    rr, cc = np.indices((480, 640))
    rays = np.stack(((cc+.5-320)/FOCAL, (rr+.5-240)/FOCAL, np.ones_like(rr)), axis=-1)
    denominator = (rays@E[:3, :3].T)@n
    z = np.divide(-d-E[:3, 3]@n, denominator, out=np.zeros_like(denominator), where=np.abs(denominator)>1e-12)
    valid = (z >= .2) & (z <= 5.)
    return np.where(valid, z, 0.).astype(np.float32), valid


@pytest.mark.parametrize('E', [np.asarray(BODY_FROM_OPTICAL), body_from_optical()])
def test_measured_plane_admission_has_no_old_absolute_height_dependency(E):
    n = rotation([1., 1., 0.], .012)@np.array([0., 0., 1.])
    d, v = render_plane(E, n, .295)  # 25 mm from a hypothetical initial floor.
    original_d, original_v = d.copy(), v.copy()
    points, mask = measured_candidates(d, v, E, np.array([0., 0., 1.]))
    fit = fit_measured_plane(points, [0., 0., 1.])
    assert fit['available'] and fit['candidate_count'] > 100
    np.testing.assert_allclose(fit['normal_body'], n, atol=1e-6, rtol=0)
    assert abs(fit['offset_body_m']-.295) < 1e-6
    np.testing.assert_array_equal(d, original_d); np.testing.assert_array_equal(v, original_v)
    i, j = np.argwhere(mask)[len(points)//2]
    d[ROWS[i]-1, COLUMNS[j]-1] = 0.; v[ROWS[i]-1, COLUMNS[j]-1] = False
    _, reduced = measured_candidates(d, v, E, [0., 0., 1.])
    assert not reduced[i, j] and np.all(~reduced | mask)


def test_missing_floor_and_two_distinct_surfaces_are_not_admitted():
    wall = np.ones((480, 640), np.float32)
    points, _ = measured_candidates(wall, np.ones_like(wall, bool), np.asarray(BODY_FROM_OPTICAL), [0., 0., 1.])
    assert not fit_measured_plane(points, [0., 0., 1.])['available']
    x, y = np.meshgrid(np.linspace(-1., 1., 20), np.linspace(-1., 1., 20))
    p = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -.32)))
    fit = fit_measured_plane(np.concatenate((p, p+[0., 0., .03])), [0., 0., 1.])
    assert not fit['available'] and fit['reason'] == 'multiple_or_incoherent_planes'
    assert not fit['candidates_trimmed']
    assert not fit_measured_plane(np.column_stack((x.ravel(), x.ravel(), np.full(x.size, -.32))), [0., 0., 1.])['available']


def test_pair_disagreement_or_unavailable_camera_stops_registration():
    a = plane() | dict(available=True)
    assert paired_plane(a, a)['offset_body_m'] == .32
    for b in (plane(d=.324) | dict(available=True),
              plane(rotation([1, 0, 0], .02)@[0., 0., 1.]) | dict(available=True),
              dict(available=False)):
        with pytest.raises(ValueError): paired_plane(a, b)


def test_registration_preserves_initial_pose_exactly_with_same_reference():
    n = rotation([1, 2, 0], .015)@[0., 0., 1.]
    ref = plane(n)
    result = register_pose(np.zeros(3), np.eye(3), ref, ref)
    np.testing.assert_allclose(result['position_initial_body_m'], 0., atol=1e-15, rtol=0)
    np.testing.assert_allclose(result['rotation_initial_body_from_current_body'], np.eye(3), atol=1e-15, rtol=0)


def test_registration_recovers_known_height_tilt_without_overwriting_visual_xy_heading():
    ref_n = rotation([1., -2., 0.], .007)@[0., 0., 1.]
    ref = plane(ref_n)
    raw_R = rotation([.3, .1, 1.], 1.2)
    raw_p = np.array([2., -.7, .03])
    current_n = raw_R.T@rotation([1., 0., 0.], .016)@ref_n
    current = plane(current_n, .32+ref_n@raw_p-.019)
    result = register_pose(raw_p, raw_R, ref, current)
    p = np.asarray(result['position_initial_body_m']); R = np.asarray(result['rotation_initial_body_from_current_body'])
    np.testing.assert_allclose(R@current_n, ref_n, atol=1e-12, rtol=0)
    assert abs(current['offset_body_m']-ref_n@p-ref['offset_body_m']) < 1e-12
    projector = np.eye(3)-np.outer(ref_n, ref_n)
    np.testing.assert_allclose(projector@p, projector@raw_p, atol=1e-12, rtol=0)
    a, b = projector@R[:, 0], projector@raw_R[:, 0]
    np.testing.assert_allclose(a/np.linalg.norm(a), b/np.linalg.norm(b), atol=1e-12, rtol=0)
    # Independent points on the current plane must map to the reference plane.
    tangent = np.cross(current_n, [1., 0., 0.]); tangent /= np.linalg.norm(tangent)
    other = np.cross(current_n, tangent)
    samples = -current['offset_body_m']*current_n+np.arange(-4, 5)[:, None]*tangent+.7*other
    np.testing.assert_allclose((samples@R.T+p)@ref_n+.32, 0., atol=1e-12, rtol=0)
    assert not result['original_visual_witness_replaced']
    assert result['position_error_bound'] is None and not result['uncertainty_model_validated']


@pytest.mark.parametrize('p,R,current', [
    ([0., 0., .051], np.eye(3), plane()),
    ([0., 0., 0.], rotation([1., 0., 0.], .11), plane()),
    ([0., 0., 0.], rotation([0., 1., 0.], np.pi/2), plane([1., 0., 0.])),
    ([float('nan'), 0., 0.], np.eye(3), plane()),
    ([0., 0., 0.], np.eye(3), plane(d=float('nan'))),
])
def test_invalid_or_excessive_corrections_rejected(p, R, current):
    with pytest.raises(ValueError): register_pose(p, R, plane(), current)
