"""Actual synthetic RGB-D and existing joint-fit checks; no native replay."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorVisualLedMotion
from lewm.temporal_anchor_continuity_development import TemporalAnchorVisualLedMotion, CONTINUITY_RULES
from lewm.independent_tracking_numerical_verification_development import _angle
from lewm.tests.test_temporal_anchor_continuity_development import force_missing, observe
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
from lewm.tests.test_measured_pose_seeded_rgbd_development import sequence


def check_witness(evidence):
    witnesses = evidence['rotation_measurement_witnesses']
    assert 1 <= len(witnesses) <= 9
    for w in witnesses:
        a = np.asarray(w['reference_rotation_initial_body_from_reference_body'])
        b = np.asarray(w['fitted_rotation_reference_body_from_current_body'])
        c = np.asarray(w['composed_rotation_initial_body_from_current_body'])
        g = np.asarray(w['gyro_rotation_reference_body_from_current_body'])
        np.testing.assert_allclose(a @ b, c, atol=1e-12, rtol=0)
        assert _angle(g.T @ b) == pytest.approx(w['gyro_disagreement_rad'], abs=1e-12)
        assert w['fitting_mode'] == 'joint' and w['candidate_envelope_passed']
    a = evidence['selected_anchor_rotation_witness']
    b = evidence['incremental_rotation_witness']
    if a is not None and b is not None:
        aR = np.asarray(a['composed_rotation_initial_body_from_current_body'])
        bR = np.asarray(b['composed_rotation_initial_body_from_current_body'])
        assert _angle(aR.T @ bR) == pytest.approx(evidence['disagreement_rad'], abs=1e-12)


@pytest.mark.parametrize('bias', [-.02, .02])
def test_both_joint_branches_measure_visual_rotation_under_signed_common_bias(bias):
    joint = JointTemporalAnchorVisualLedMotion()
    unbiased_joint = JointTemporalAnchorVisualLedMotion()
    gyro = TemporalAnchorVisualLedMotion()
    for frame, item in enumerate(packets([texture()] * 4)):
        unbiased = observe(unbiased_joint, deepcopy(item))
        p, d, f, now = deepcopy(item)
        for channel in (p['sensor_state']['sensed']['gyro'], f):
            channel['values'][:, 2] += bias
        original_packet = deepcopy((p, d, f, now))
        result = observe(joint, (p, d, f, now))
        baseline = observe(gyro, original_packet)
        assert result['current_pose'] is not None
        assert result['current_pose']['mode'] == 'joint'
        assert result['current_pose']['gyro_role'] == 'consistency_monitor_only'
        if frame:
            evidence = result['continuity_evidence']
            check_witness(evidence)
            assert evidence['incremental_rotation_witness_saved']
            assert evidence['selected_anchor_rotation_witness'] is not None
            assert evidence['incremental_rotation_witness'] is not None
            # SIFT/LK interpolation need not produce exact identity on identical
            # images. The stronger mechanism check is exact equality to the
            # same visual fit without the injected gyro bias.
            np.testing.assert_array_equal(result['current_pose']['rotation_initial_body_from_current_body'],
                unbiased['current_pose']['rotation_initial_body_from_current_body'])
            R = np.asarray(baseline['current_pose']['rotation_initial_body_from_current_body'])
            assert _angle(R) > abs(bias) * frame * .09
        assert not result['candidate_adopted'] and not result['gyro_bias_estimated']


def test_real_rendered_motion_bridges_then_rejoins_with_both_rotation_witnesses(monkeypatch):
    force_missing(monkeypatch, {3, 4})
    model = JointTemporalAnchorVisualLedMotion()
    for frame, item in enumerate(sequence(2., 1, rate=(.01, -.02, .03), steps=7)):
        result = observe(model, item)
        pose = result['current_pose']
        assert pose is not None, result['terminal_failure']
        np.testing.assert_allclose(pose['position_initial_body_m'], item[4], atol=.006, rtol=0)
        if frame:
            check_witness(result['continuity_evidence'])
        if frame in (3, 4):
            e = result['continuity_evidence']
            assert e['status'] == 'MEASURED_INCREMENT_BRIDGE'
            assert e['incremental_rotation_witness_saved']
            assert e['selected_anchor_rotation_witness'] is None
            assert not pose['promoted_keyframe']
        if frame == 5:
            assert result['continuity_evidence']['preceding_bridge_frames'] == 2
            assert result['continuity_evidence']['selected_anchor_rotation_witness'] is not None
        assert not pose['global_history_reset']


def test_joint_bridge_budget_stays_terminal_at_eleventh_missing_anchor(monkeypatch):
    force_missing(monkeypatch, set(range(2, 15)))
    model = JointTemporalAnchorVisualLedMotion()
    for frame, item in enumerate(packets([texture()] * 14)):
        result = observe(model, item)
        assert (result['current_pose'] is not None) == (frame <= 11)
    assert model.model.bridge_frames == CONTINUITY_RULES['maximum_bridge_frames'] == 10
    assert result['continuity_evidence']['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    assert result['continuity_evidence']['incremental_rotation_witness_saved']
    assert not result['continuity_evidence_current']


@pytest.mark.parametrize('fault', ['rgb', 'depth', 'gyro', 'clock', 'identity'])
def test_missing_or_invalid_current_sensor_still_latches_terminal(fault):
    model = JointTemporalAnchorVisualLedMotion()
    items = list(packets([texture()] * 3))
    observe(model, items[0])
    p, d, f, now = deepcopy(items[1])
    if fault == 'rgb':
        p['image']['rgb'][:] = 128
        d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    elif fault == 'depth': d['valid'][:] = False; d['depth_m'][:] = 0.
    elif fault == 'gyro': f['valid'][:] = False; f['values'][:] = 0.
    elif fault == 'clock': now += 1
    else: p['sensor_state']['identity'] = (0, 0, 99)
    rejected = observe(model, (p, d, f, now))
    assert rejected['current_pose'] is None
    later = observe(model, items[2])
    assert later['current_pose'] is None and later['terminal_failure'] == rejected['terminal_failure']


def test_missing_increment_does_not_claim_saved_incremental_rotation(monkeypatch):
    model = JointTemporalAnchorVisualLedMotion()
    candidate = model.model._candidate
    def missing(ref, current, G):
        if ref is model.model.previous and model.model.frame >= 2:
            raise SensorContractError('synthetic missing current increment')
        return candidate(ref, current, G)
    monkeypatch.setattr(model.model, '_candidate', missing)
    for item in packets([texture()] * 3):
        result = observe(model, item)
    assert result['current_pose'] is not None
    e = result['continuity_evidence']
    assert e['selected_anchor_rotation_witness'] is not None
    assert e['incremental_rotation_witness'] is None
    assert not e['incremental_rotation_witness_saved']


def test_returned_rotation_witnesses_do_not_alias_estimator_state():
    model = JointTemporalAnchorVisualLedMotion()
    for item in packets([texture()] * 3): result = observe(model, item)
    result['continuity_evidence']['incremental_rotation_witness'][
        'composed_rotation_initial_body_from_current_body'][0][0] = 123.
    fresh = model.snapshot(now_ns=item[3])
    check_witness(fresh['continuity_evidence'])
