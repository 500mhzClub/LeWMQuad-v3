"""Analytic sensitivity and unchanged-observer trace checks; no physical recovery."""
from copy import deepcopy
import hashlib
import sys

import numpy as np
import pytest

from lewm.rgbd_registration_trace_development import conditioning, registration_summary, TracePairs
from lewm.joint_rgbd_rigid_pose_development import register
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from lewm.balanced_multi_reference_rgbd_development import BalancedVisualLedMotion
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_rgbd_correspondence_motion_development import point_fixture, texture, packets


def test_fixed_rotation_translation_rank_does_not_depend_on_spatial_spread():
    for points in (np.ones((12, 3)), np.column_stack([np.arange(12), np.zeros((12, 2))])):
        row = conditioning(points, points, np.eye(3), np.zeros(3))
        assert row['fixed_rotation_translation_normal_eigenvalues'] == [12.] * 3
        assert row['fixed_rotation_translation_condition_number'] == 1
        assert row['fixed_rotation_translation_rank'] == 3
        assert row['centered_rotation_normal_eigenvalues_m2'][0] == pytest.approx(0)
        assert not row['covariance_calibrated'] and not row['changes_acceptance']


def test_common_mode_point_bias_can_have_zero_residual_and_wrong_translation():
    a, b, ua, ub, R, t = point_fixture(); bias = np.array([.03, .02, -.01])
    before = conditioning(a, b, R, t)
    after = conditioning(a + bias, b, R, t + bias)
    assert before['residual_rms_m'] < 1e-12 and after['residual_rms_m'] < 1e-12
    assert np.linalg.norm(bias) > .03 and not after['common_mode_bias_bounded']


def test_gyro_lever_arm_sensitivity_agrees_with_finite_difference():
    a, b, ua, ub, R, t = point_fixture(); mean = b.mean(0)
    axis = np.cross(mean, [1., 0., 0.]); axis /= np.linalg.norm(axis)
    eps = 1e-6
    changed = a.mean(0) - R @ rotation_increment(axis * eps) @ mean
    original = a.mean(0) - R @ mean
    row = conditioning(a, b, R, t)
    assert np.linalg.norm(changed - original) / eps == pytest.approx(
        row['gyro_common_angle_translation_sensitivity_m_per_rad'], rel=1e-5)


@pytest.mark.parametrize('fault', ['few', 'shape', 'nan', 'rotation'])
def test_invalid_conditioning_inputs_rejected(fault):
    a, b, ua, ub, R, t = point_fixture()
    if fault == 'few': a, b = a[:2], b[:2]
    elif fault == 'shape': b = b[:-1]
    elif fault == 'nan': a[0, 0] = np.nan
    else: R[0, 0] = 2.
    with pytest.raises(ValueError): conditioning(a, b, R, t)


@pytest.mark.parametrize('error', ['insufficient rigid-pose matches', 'no conditioned rigid-pose proposal',
    'insufficient rigid consensus after pruning', 'noncollinear well-spread point support required'])
def test_failed_intermediate_fits_never_become_converged_candidates(error):
    values = dict(a=np.zeros((12, 3)), R=np.eye(3), t=np.ones(3), mask=np.ones(12, bool))
    row = registration_summary(values, None, error)
    assert not row['converged_candidate_available'] and row['candidate'] is None


def test_final_support_failure_retains_candidate_without_accepting_it():
    a, b, ua, ub, R, t = point_fixture()
    # This pure summary does not change or call the actual acceptance function.
    values = dict(a=a, b=b, ua=np.zeros_like(ua), ub=np.zeros_like(ub), R=R, t=t,
        mask=np.ones(len(a), bool), valid_candidates=10, initial_count=len(a), rounds=1)
    row = registration_summary(values, None, 'rigid consensus fraction, grid support or displacement rejected')
    assert row['converged_candidate_available'] and not row['original_registration_accepted']
    assert row['original_gate_failures']['reference_grid'] and row['original_gate_failures']['current_grid']
    assert not row['candidate']['passes_registration_only']


@pytest.mark.parametrize('observer', [MultiReferenceVisualLedMotion, BalancedVisualLedMotion])
def test_scoped_trace_preserves_complete_observer_outputs_and_counts(observer):
    original, traced = observer(), observer()
    for i, (p, d, f, now) in enumerate(packets([texture()] * 3)):
        expected = original.observe(p, d, f, now_ns=now)
        with TracePairs() as trace:
            result = traced.observe(p, d, f, now_ns=now)
        assert result == expected and sys.gettrace() is None
        rows = trace.record()
        if i == 0: assert rows == []
        else:
            assert len(rows) == 1 and rows[0]['original_candidate_accepted']
            m, r = rows[0]['matching'], rows[0]['registration']
            assert m['unique_mutual_pairs'] <= m['mutual_ratio_before_dedup']
            assert m['paired_depth_survivors'] <= m['bidirectional_flow_survivors'] <= m['unique_mutual_pairs']
            assert m['paired_depth_survivors'] == r['lifted_matches'] >= r['inliers']
            assert r['original_registration_accepted'] and not any(r['original_gate_failures'].values())
            assert rows[0]['diagnostic_position_initial_body_m'] == result['current_pose']['position_initial_body_m']
            rows[0]['reference_frame'] = 999
            assert trace.record()[0]['reference_frame'] != 999


def test_trace_keeps_terminal_failure_and_postfailure_frames_without_recovery():
    samples = list(packets([texture()] * 3)); p, d, f, now = samples[1]
    p['image']['rgb'][:] = 128; d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    original, traced = MultiReferenceVisualLedMotion(), MultiReferenceVisualLedMotion()
    for i, (p, d, f, now) in enumerate(samples):
        expected = original.observe(p, d, f, now_ns=now)
        with TracePairs() as trace: result = traced.observe(p, d, f, now_ns=now)
        assert result == expected
        if i == 1:
            assert result['current_pose'] is None and result['terminal_failure']
            assert not trace.pairs[0]['original_candidate_accepted']
            assert trace.pairs[0]['matching']['paired_depth_survivors'] == 0
            assert trace.pairs[0]['registration']['candidate'] is None
        elif i == 2: assert result['current_pose'] is None and trace.pairs == []


def test_existing_tracer_is_not_replaced():
    def previous(frame, event, arg): return None
    sys.settrace(previous)
    try:
        with pytest.raises(ValueError, match='another tracer'):
            with TracePairs(): pass
        assert sys.gettrace() is previous
    finally: sys.settrace(None)


def test_trace_fault_is_not_silently_reported_as_observer_success(monkeypatch):
    model = MultiReferenceVisualLedMotion(); samples = list(packets([texture()] * 2))
    p, d, f, now = samples[0]; model.observe(p, d, f, now_ns=now)
    def broken(*args): raise ValueError('synthetic witness fault')
    monkeypatch.setattr('lewm.rgbd_registration_trace_development.match_summary', broken)
    p, d, f, now = samples[1]
    with pytest.raises(RuntimeError, match='trace failed'):
        with TracePairs(): model.observe(p, d, f, now_ns=now)
    assert sys.gettrace() is None
