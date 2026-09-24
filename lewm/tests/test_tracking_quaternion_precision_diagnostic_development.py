import math

import numpy as np
import pytest

from lewm.tracking_quaternion_precision_diagnostic_development import summarize_quaternions


def report(q):
    return summarize_quaternions(q, np.arange(1, len(q) + 1) * .002)


def test_identity_pass_and_no_mutation():
    q = np.tile([0., 0., 0., 1.], (800, 1))
    before = q.copy()
    r = report(q)
    assert r['frozen_norm_gate_passes']
    assert r['absolute_norm_error_max'] == 0
    assert r['first_rejected_time_s'] is None
    assert not r['tracking_accuracy_verified'] and not r['goal_achieved']
    np.testing.assert_array_equal(q, before)


def test_scaled_rotation_reproduces_rejection_without_repair():
    q = np.array([[0., 0., math.sin(.7), math.cos(.7)]]) * (1 + 2e-7)
    before = q.copy()
    r = report(q)
    assert not r['frozen_norm_gate_passes']
    assert r['frozen_gate_rejected_sample_count'] == 1
    assert r['first_rejected_sample_indices'] == [0]
    assert 1e-7 < r['absolute_norm_error_max'] < 3e-7
    assert r['diagnostic_normalized_copy_yaw_max_difference_rad'] > 0
    assert not r['normalized_copies_used_for_scoring']
    np.testing.assert_array_equal(q, before)


def test_float32_widening_and_permutation_do_not_fix_norm():
    q = np.array([[.3, .4, .1, .85]], dtype=np.float32).astype(np.float64)
    r = report(q)
    assert r['components_exactly_representable_as_binary32'] == 4
    assert report(q[:, [3, 0, 1, 2]])['absolute_norm_error_max'] == r['absolute_norm_error_max']
    assert not r['simulator_precision_cause_proved']


def test_capture_and_settling_denominators():
    q = np.tile([0., 0., 0., 1.], (850, 1))
    q[[0, 749, 750, 799, 849], 3] += 1e-6
    r = report(q)
    assert r['frozen_gate_rejected_sample_count'] == 5
    assert r['rejected_settle_samples'] == 2
    assert r['rejected_capture_samples'] == 3


@pytest.mark.parametrize('q', [np.zeros((1, 4)), np.ones((1, 3)), np.ones((0, 4)),
    np.ones((22851, 4)), np.array([[np.nan, 0., 0., 1.]]),
    np.array([[np.inf, 0., 0., 1.]]), np.ones((1, 4), dtype=int)])
def test_reject_malformed(q):
    with pytest.raises(ValueError): report(q)


def test_clock_failure():
    with pytest.raises(ValueError, match='500Hz'):
        summarize_quaternions(np.array([[0., 0., 0., 1.]]), np.array([0.]))


def test_float32_normalization_can_exceed_frozen_tolerance():
    # A synthetic numerical mechanism, not a claim about the actual simulator.
    from lewm.physical_execution_development import rotation_xyzw
    rng = np.random.default_rng(20260907)
    q = rng.normal(size=(10000, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1)[:, None]
    widened = q.astype(np.float64)
    r = report(widened)
    assert r['frozen_gate_rejected_sample_count'] > 0
    assert r['absolute_norm_error_max'] < 1e-5
    worst = r['worst_sample']['xyzw']
    rotation = rotation_xyzw(worst)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14, rtol=0)
    assert not r['simulator_precision_cause_proved']
