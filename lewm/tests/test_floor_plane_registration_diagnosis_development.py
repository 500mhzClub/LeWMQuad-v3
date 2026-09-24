import numpy as np
import pytest
from scripts.view_reentry_floor_plane_registration_diagnosis_development import fit_plane


def test_recovers_tilted_measured_plane_and_reports_nonplanar_residual():
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-.5, .5, 15))
    z = -.32+.002*x-.003*y
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    expected = np.array([-.002, .003, 1.]); expected /= np.linalg.norm(expected)
    fit = fit_plane(points)
    np.testing.assert_allclose(fit['normal_map'], expected, atol=1e-12, rtol=0)
    assert fit['rms_residual_m'] < 1e-12
    assert not fit['physical_floor_certified'] and not fit['policy_admission_performed']
    points[0, 2] += .02
    noisy = fit_plane(points)
    assert noisy['max_absolute_residual_m'] > .01
    assert not noisy['physical_floor_certified']


def test_rejects_missing_two_dimensional_support_and_nonfinite_points():
    points = np.zeros((100, 3)); points[:, 0] = np.linspace(-1, 1, 100)
    with pytest.raises(ValueError): fit_plane(points)
    points[0, 0] = np.nan
    with pytest.raises(ValueError): fit_plane(points)
