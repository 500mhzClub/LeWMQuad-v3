import numpy as np
import pytest

from lewm.representation_aware_tracking_coverage_development import coverage_with_sensor_rotation_convention
from lewm.independent_tracking_coverage_development import measured_coverage


def inputs():
    n = 22850
    t = np.arange(1, n + 1) * .002
    p = np.zeros((n, 7)); p[:, 6] = 1.
    v = np.zeros((n, 6))
    completion = dict(completed_ticks=442, schedule_complete=True, physical_stop=None, acquisition_stop=None)
    return t, p, v, completion


def test_no_motion_cannot_pass_with_representation_correction():
    t, p, v, c = inputs()
    p[:, 6] += 2e-7
    before = p.copy()
    with pytest.raises(ValueError, match='unit native quaternions'):
        measured_coverage('left', t, p, v, **c)
    report = coverage_with_sensor_rotation_convention('left', t, p, v, **c)
    assert not report['original_frozen_norm_gate_passes']
    assert report['numerical_coverage_cross_check_passed']
    assert not report['coverage']['intended_motion_covered']
    assert not report['original_attempt_passed']
    np.testing.assert_array_equal(p, before)


@pytest.mark.parametrize('scale', [0., 1.001, float('nan'), float('inf')])
def test_invalid_quaternion_not_repaired(scale):
    t, p, v, c = inputs(); p[0, 6] = scale
    with pytest.raises(ValueError): coverage_with_sensor_rotation_convention('left', t, p, v, **c)


def test_original_stop_rules_retained():
    t, p, v, c = inputs(); v[-1, 0] = .021
    result = coverage_with_sensor_rotation_convention('right', t, p, v, **c)
    assert not result['coverage']['final_second_stop_covered']
    assert result['original_frozen_norm_gate_passes']


def test_missing_samples_still_rejected():
    t, p, v, c = inputs()
    with pytest.raises(ValueError):
        coverage_with_sensor_rotation_convention('left', t[:-1], p[:-1], v[:-1], **c)
