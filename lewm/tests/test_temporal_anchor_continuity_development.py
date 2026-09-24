"""Real measured geometry plus explicitly injected continuity-state faults.

Forced anchor rejection tests the state machine, not natural tracking recovery.
No fixture truth is supplied to either estimator.
"""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.multi_reference_rgbd_pose_development import MultiReferenceRGBDPose, MultiReferenceVisualLedMotion
from lewm.temporal_anchor_continuity_development import (
    TemporalAnchorVisualLedMotion, CONTINUITY_RULES)
from lewm.tests.test_measured_pose_seeded_rgbd_development import sequence
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture


def observe(m, item):
    p, d, f, now = item[:4]
    return m.observe(p, d, f, now_ns=now)


def force_missing(monkeypatch, frames):
    original = MultiReferenceRGBDPose._choose
    def choose(self, current, G):
        if self.frame in frames:
            self.last_selection = dict(status='NO_QUALIFIED_REFERENCE', attempts=[])
            raise SensorContractError('synthetically unavailable retained anchors')
        return original(self, current, G)
    monkeypatch.setattr(MultiReferenceRGBDPose, '_choose', choose)


@pytest.mark.parametrize('depth,axis,sign,foreground,rate', [
    (.7, 1, 1, False, (0., 0., 0.)),
    (2., 0, -1, False, (0., 0., 0.)),
    (3.5, 1, -1, False, (0., 0., 0.)),
    (2., 1, 1, True, (.01, -.02, .03)),
])
def test_actual_geometry_keeps_original_anchor_poses_when_measurements_agree(depth, axis, sign, foreground, rate):
    m = TemporalAnchorVisualLedMotion(); original = MultiReferenceVisualLedMotion()
    for item in sequence(depth, axis, sign, foreground, rate):
        row = observe(m, item); old = observe(original, item)
        assert row['current_pose'] == old['current_pose']
        assert row['reference_selection'] == old['reference_selection']
        assert row['current_pose'] is not None
        np.testing.assert_allclose(row['current_pose']['position_initial_body_m'], item[4], atol=.003, rtol=0)
        assert row['continuity_evidence_current'] and not row['bridge_is_calibrated_uncertainty']
    assert m.model.total_bridge_frames == 0


def test_measured_bridge_preserves_anchors_and_rejoins_without_pose_reset(monkeypatch):
    force_missing(monkeypatch, set(range(4, 9)))
    m = TemporalAnchorVisualLedMotion(); frozen = None
    for frame, item in enumerate(sequence(2., 1, steps=13)):
        row = observe(m, item); pose = row['current_pose']; assert pose is not None
        evidence = row['continuity_evidence']
        np.testing.assert_allclose(pose['position_initial_body_m'], item[4], atol=.003, rtol=0)
        if frame == 3:frozen = deepcopy(m.model.nodes), tuple(ref.frame for ref in m.model.references)
        if 4 <= frame <= 8:
            assert evidence['status'] == 'MEASURED_INCREMENT_BRIDGE'
            assert evidence['previous_frame'] == frame - 1
            assert evidence['previous_measured_ns'] == item[3] - 100_000_000
            assert evidence['bridge_frames'] == frame - 3
            assert not pose['promoted_keyframe'] and pose['promotion_reason'] is None
            assert (m.model.nodes, tuple(ref.frame for ref in m.model.references)) == frozen
            assert pose['reference_frame'] == frame - 1
            assert row['reference_selection']['status'] == 'MEASURED_INCREMENT_BRIDGE'
            assert not row['reference_selection']['selected_reference_retained_anchor']
            assert evidence['anchor_available'] is False and evidence['incremental_available']
        if frame == 9:
            assert evidence['status'] == 'ANCHOR_MEASUREMENT' and evidence['preceding_bridge_frames'] == 5
            assert evidence['disagreement_m'] < .003
            assert evidence['preceding_bridge_path_m'] > .19 and evidence['bridge_path_m'] == 0
        assert pose['global_history_reset'] is False and pose['position_error_bound'] is None
    assert m.model.total_bridge_frames == 5 and m.model.bridge_frames == 0


def test_bridge_budget_latches_even_when_increment_remains_measurable(monkeypatch):
    force_missing(monkeypatch, set(range(2, 15)))
    m = TemporalAnchorVisualLedMotion()
    for frame, item in enumerate(packets([texture()] * 14)):
        row = observe(m, item)
        if frame <= 11:assert row['current_pose'] is not None
        else:assert row['current_pose'] is None and row['terminal_failure'] is not None
        if frame == 12:assert row['continuity_evidence']['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    assert m.model.bridge_frames == CONTINUITY_RULES['maximum_bridge_frames'] == 10
    assert m.model.previous.frame == 11 and len(m.model.references) == 1


@pytest.mark.parametrize('conflict', ['translation', 'rotation'])
def test_qualified_anchor_increment_conflict_is_terminal(monkeypatch, conflict):
    m = TemporalAnchorVisualLedMotion(); candidate = m.model._candidate
    def injected(ref, current, G):
        c = candidate(ref, current, G)
        if ref is m.model.previous and m.model.frame >= 2:
            if conflict == 'translation':c['p'] = c['p'] + [0., .03, 0.]
            else:
                from scipy.spatial.transform import Rotation
                c['R'] = c['R'] @ Rotation.from_rotvec([0., 0., .11]).as_matrix()
        return c
    monkeypatch.setattr(m.model, '_candidate', injected)
    for frame, item in enumerate(packets([texture()] * 4)):
        row = observe(m, item)
        assert (row['current_pose'] is not None) == (frame < 2)
    assert row['continuity_evidence']['status'] == 'ANCHOR_INCREMENT_CONFLICT'


def test_conflicting_anchors_cannot_be_overridden_by_increment(monkeypatch):
    m = TemporalAnchorVisualLedMotion(); items = list(packets([texture()] * 2)); observe(m, items[0])
    def conflict(self, current, G):
        self.last_selection = dict(status='CONFLICTING_ALTERNATIVES')
        raise SensorContractError('synthetic qualified anchors disagree')
    monkeypatch.setattr(MultiReferenceRGBDPose, '_choose', conflict)
    def forbidden(*args):raise AssertionError('must not try incremental override')
    monkeypatch.setattr(m.model, '_candidate', forbidden)
    row = observe(m, items[1])
    assert row['current_pose'] is None and row['continuity_evidence']['status'] == 'ANCHOR_CONFLICT_OR_INVALID'


def test_missing_increment_does_not_veto_a_qualified_anchor(monkeypatch):
    m = TemporalAnchorVisualLedMotion(); candidate = m.model._candidate
    def injected(ref, current, G):
        if ref is m.model.previous and m.model.frame >= 2:raise SensorContractError('injected increment missing')
        return candidate(ref, current, G)
    monkeypatch.setattr(m.model, '_candidate', injected)
    for item in packets([texture()] * 3):row = observe(m, item)
    assert row['current_pose'] is not None
    assert row['continuity_evidence']['status'] == 'ANCHOR_MEASUREMENT'
    assert not row['continuity_evidence']['incremental_available']


def test_missing_both_measurements_does_not_extrapolate(monkeypatch):
    m = TemporalAnchorVisualLedMotion(); items = list(packets([texture()] * 3));observe(m, items[0])
    force_missing(monkeypatch, {1, 2})
    def missing(*args):raise SensorContractError('injected increment missing')
    monkeypatch.setattr(m.model, '_candidate', missing)
    row = observe(m, items[1]);assert row['current_pose'] is None
    assert row['continuity_evidence']['status'] == 'NO_CURRENT_MEASURED_TRANSLATION'
    assert observe(m, items[2])['current_pose'] is None


@pytest.mark.parametrize('fault', ['stale', 'future', 'depth_time', 'image_binding', 'blank', 'privileged', 'calibration', 'identity'])
def test_invalid_current_measurement_cannot_be_bridged(monkeypatch, fault):
    m = TemporalAnchorVisualLedMotion(); items = list(packets([texture()] * 3)); observe(m, items[0])
    force_missing(monkeypatch, {1, 2})
    p, d, f, now = deepcopy(items[1])
    if fault == 'stale':now += 100_000_000
    if fault == 'future':m.model.previous = type(m.model.previous)(1, now+100_000_000,
        m.model.previous.features, m.model.previous.rotation, m.model.previous.gyro, m.model.previous.position)
    if fault == 'depth_time':d['available_ns'] = now+1
    if fault == 'image_binding':p['image']['rgb'][0, 0] = 0
    if fault == 'blank':
        p['image']['rgb'][:] = 128;d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    if fault == 'privileged':p['native_pose'] = [0., 0., 0.]
    if fault == 'calibration':d['calibration_id'] = 'unknown'
    if fault == 'identity':p['sensor_state']['identity'] = (0, 0, 99)
    assert observe(m, (p, d, f, now))['current_pose'] is None
    # Use a forward snapshot even when the injected time was advanced.
    row = m.snapshot(now_ns=max(now, items[2][3]))
    assert row['current_pose'] is None and not row['continuity_evidence_current']


def test_old_pose_is_not_current_between_measurements():
    m = TemporalAnchorVisualLedMotion(); item = next(packets([texture()]));observe(m, item)
    row = m.snapshot(now_ns=item[3]+1)
    assert row['current_pose'] is None and not row['continuity_evidence_current']
    assert row['physical_pose_error_bound'] is None and not row['support_established']


def test_reference_storage_remains_bounded_and_returned_evidence_does_not_alias(monkeypatch):
    monkeypatch.setattr('lewm.temporal_anchor_continuity_development.support_near_limit', lambda _: True)
    m = TemporalAnchorVisualLedMotion()
    for item in packets([texture()] * 12):row = observe(m, item)
    assert len(m.model.references) == 8 and m.model.previous.frame == 11
    assert [ref.frame for ref in m.model.references] == list(range(4, 12))
    row['continuity_evidence']['status'] = 'CORRUPTED_COPY'
    row['current_pose']['position_initial_body_m'][0] = 1e6
    fresh = m.snapshot(now_ns=item[3])
    assert fresh['continuity_evidence']['status'] == 'ANCHOR_MEASUREMENT'
    assert abs(fresh['current_pose']['position_initial_body_m'][0]) < .001


def test_command_values_do_not_supply_bridge_translation(monkeypatch):
    force_missing(monkeypatch, {2, 3})
    m = TemporalAnchorVisualLedMotion()
    for frame, item in enumerate(sequence(2., 1, steps=4)):
        # The fixture controls remain zero while the actually rendered camera moves.
        assert np.all(item[0]['sensor_state']['control']['applied_command']['values'] == 0)
        row = observe(m, item)
    assert row['continuity_evidence']['status'] == 'MEASURED_INCREMENT_BRIDGE'
    assert np.linalg.norm(row['current_pose']['position_initial_body_m']) > .1
    assert not row['command_integration_used'] and not row['bridge_is_command_or_inertial_extrapolation']


def test_small_common_depth_bias_can_pass_agreement_without_a_true_motion_bound():
    # Static RGB and true geometry; deliberately biased reported optical depth.
    # This tests an unknown sensor error, not a legitimate calibrated depth change.
    m = TemporalAnchorVisualLedMotion()
    for frame, item in enumerate(packets([texture()] * 3)):
        p, d, f, now = item
        d['depth_m'] += np.float32(.002 * frame)
        row = observe(m, (p, d, f, now))
        assert row['current_pose'] is not None
    assert np.linalg.norm(row['current_pose']['position_initial_body_m']) > .003
    assert row['continuity_evidence']['disagreement_m'] < .001
    assert row['physical_pose_error_bound'] is None
    assert row['continuity_evidence']['uncertainty_calibrated'] is False
