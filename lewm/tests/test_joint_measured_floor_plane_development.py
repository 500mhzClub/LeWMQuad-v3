from copy import deepcopy
import numpy as np
import pytest
from lewm.floor_pose_registration_development import fit_measured_plane
from lewm.joint_measured_floor_plane_development import fit_joint_plane, validate_joint_plane


def grid(width=1., height=1., count=20):
    x, y = np.meshgrid(np.linspace(-width, width, count), np.linspace(-height, height, count))
    return np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -.32)))


def test_combined_observability_accepts_narrow_view_without_reducing_rank_gate():
    primary, auxiliary = grid(height=.025), grid()
    assert not fit_measured_plane(primary, [0., 0., 1.])['available']
    fit = fit_joint_plane(primary, auxiliary, [0., 0., 1.])
    validate_joint_plane(fit, [0., 0., 1.])
    assert fit['candidate_count'] == 800 and fit['minimum_second_eigenvalue_m2'] == .05**2
    assert not fit['independent_camera_plane_rank_required'] and not fit['candidates_trimmed']
    np.testing.assert_allclose(fit['normal_body'], [0., 0., 1.], atol=1e-12, rtol=0)
    assert fit['offset_body_m'] == pytest.approx(.32)


def test_each_measured_point_keeps_coherence_gate_even_in_sparse_camera():
    primary = grid(count=11)+[0., 0., .02]; auxiliary = grid(count=100)
    fit = fit_joint_plane(primary, auxiliary, [0., 0., 1.])
    assert not fit['available'] and fit['reason'] == 'combined_points_not_one_coherent_plane'
    assert fit['camera_residuals'][0]['maximum_residual_m'] > .003
    with pytest.raises(ValueError): validate_joint_plane(fit, [0., 0., 1.])


def test_combined_rank_deficiency_and_absence_still_rejected():
    for a, b in ((grid(height=0.), grid(height=0.)), (np.empty((0, 3)), np.empty((0, 3)))):
        fit = fit_joint_plane(a, b, [0., 0., 1.])
        assert not fit['available']
        with pytest.raises(ValueError): validate_joint_plane(fit, [0., 0., 1.])


def test_two_cm_development_extent_retains_rank_and_residual_checks(monkeypatch):
    from lewm import joint_measured_floor_plane_development as plane
    narrow = grid(width=.3, height=.06)
    empty = np.empty((0, 3))
    assert not fit_joint_plane(empty, narrow, [0., 0., 1.])['available']
    monkeypatch.setattr(plane, 'MINIMUM_SECOND_EXTENT_M', .02)
    fitted = fit_joint_plane(empty, narrow, [0., 0., 1.])
    validate_joint_plane(fitted, [0., 0., 1.])
    np.testing.assert_allclose(fitted['normal_body'], [0., 0., 1.], atol=1e-12)
    for bad in (grid(width=.3, height=0.), narrow+np.column_stack((
            np.zeros(len(narrow)), np.zeros(len(narrow)), np.where(np.arange(len(narrow))==0, .02, 0.)))):
        assert not fit_joint_plane(empty, bad, [0., 0., 1.])['available']


def test_one_empty_view_adds_no_points_or_residuals_and_does_not_infer_free_space():
    fit = fit_joint_plane(np.empty((0, 3)), grid(), [0., 0., 1.])
    validate_joint_plane(fit, [0., 0., 1.])
    assert fit['camera_statistics'][0] == dict(camera='primary', count=0, mean_body_m=None, covariance_body_m2=None)
    assert fit['camera_residuals'][0] == dict(camera='primary', count=0, maximum_residual_m=None, rms_residual_m=None)
    assert not fit['floor_identity_certified']


def test_camera_moment_composition_matches_direct_all_point_fit():
    rng = np.random.default_rng(7)
    a, b = grid(.3, .08, 11), grid(.7, .6, 32)+[.5, .2, 0.]
    a[:, 2] += rng.uniform(-.0001, .0001, len(a)); b[:, 2] += rng.uniform(-.0001, .0001, len(b))
    fit = fit_joint_plane(a, b, [0., 0., 1.]); before = deepcopy(fit)
    direct = fit_measured_plane(np.concatenate((a, b)), [0., 0., 1.])
    validate_joint_plane(fit, [0., 0., 1.]); assert fit == before
    np.testing.assert_allclose(fit['normal_body'], direct['normal_body'], atol=1e-12, rtol=0)
    assert fit['offset_body_m'] == pytest.approx(direct['offset_body_m'], abs=1e-12)


@pytest.mark.parametrize('fault', ['normal', 'moments', 'count', 'up', 'residual', 'availability'])
def test_tampered_composition_or_residual_gate_rejected(fault):
    fit = fit_joint_plane(grid(), grid(), [0., 0., 1.])
    if fault == 'normal': fit['normal_body'][0] = .1
    elif fault == 'moments': fit['camera_statistics'][0]['covariance_body_m2'][0][1] = 99.
    elif fault == 'count': fit['camera_residuals'][0]['count'] -= 1
    elif fault == 'up': fit['up_body'][2] = -1.
    elif fault == 'residual': fit['camera_residuals'][0]['maximum_residual_m'] = .00301
    else: fit['available'] = False
    with pytest.raises(ValueError): validate_joint_plane(fit, [0., 0., 1.])
