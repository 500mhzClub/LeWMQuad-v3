import math

import numpy as np
import pytest

from lewm.gyro_integration_diagnostic_development import integrate_rates, orientation_errors, METHODS
from lewm.relative_gyro_turn_development import rotation_increment


@pytest.mark.parametrize('method', METHODS)
def test_constant_body_rate_has_exact_solution(method):
    times = np.arange(51, dtype=np.int64) * 20_000_000
    rates = np.tile([.1, -.2, .3], (51, 1))
    actual = integrate_rates(times, rates, method)
    assert np.allclose(actual[-1], rotation_increment(rates[0]), atol=1e-13, rtol=0)
    error = orientation_errors(actual, actual)
    assert error['heading_error_max_rad'] == 0 and error['rotation_error_max_rad'] < 1e-12


def test_coning_sign_against_independent_finely_composed_linear_rate_path():
    # No fitted coefficient: a known smooth noncommuting rate transition.
    samples = 10000
    reference = np.eye(3)
    for i in range(samples):
        t = (i + .5) / samples
        reference = reference @ rotation_increment(np.array([1 - t, t, 0.]) * (.1 / samples))
    times = np.array([0, 100_000_000], dtype=np.int64)
    rates = np.array([[1., 0, 0], [0., 1, 0]])
    mid = integrate_rates(times, rates, 'midpoint')[-1]
    coning = integrate_rates(times, rates, 'coning')[-1]
    assert np.linalg.norm(coning - reference) < .02 * np.linalg.norm(mid - reference)


@pytest.mark.parametrize('times,rates,method', [([0, 0], [[0, 0, 0]] * 2, 'midpoint'),
    ([0., 1.], [[0, 0, 0]] * 2, 'midpoint'), ([0, 1], [[0, math.nan, 0]] * 2, 'midpoint'),
    ([0, 1], [[0, 0, 0]] * 2, 'unknown')])
def test_invalid_samples_or_undeclared_method_rejected(times, rates, method):
    with pytest.raises(ValueError):
        integrate_rates(times, rates, method)
