from copy import deepcopy
import hashlib
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.overlap_retention_joint_observer_development import OverlapRetentionJointRGBDPose
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorRGBDPose
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
from lewm.tests.test_temporal_anchor_continuity_development import force_missing


def paired(item, rgb=None):
    p, d, f, now = deepcopy(item[:4])
    depth = from_native_depth(np.full((480, 640), 2., np.float32), p,
        measured_ns=now, available_ns=now, now_ns=now)
    image = from_captured_rgb(texture() if rgb is None else rgb, depth, p,
        measured_ns=now, available_ns=now, now_ns=now)
    return p, d, f, now, image, depth


def observe(model, item):
    p, d, f, now, image, depth = paired(item)
    return model.observe(p, d, f, auxiliary_rgb=image, auxiliary_depth=depth, now_ns=now)


def test_primary_path_matches_original_complete_pose_and_continuity():
    original = OverlapRetentionJointRGBDPose(); dual = DualCameraAnchorPose()
    for item in packets([texture()] * 4):
        p, d, f, now = deepcopy(item)
        expected = original.observe(p, d, f, now_ns=now)
        actual = observe(dual, item)
        assert {k: actual[k] for k in expected} == expected
        assert dual.last_continuity == original.last_continuity
        assert dual.last_selection == original.last_selection
        assert dual.last_overlap_retention == original.last_overlap_retention
        assert not actual['camera_selection']['auxiliary_attempted']


def test_current_primary_feature_loss_uses_auxiliary_without_reset_and_front_can_return():
    model = DualCameraAnchorPose()
    images = [texture(), texture(), np.full((480, 640, 3), 128, np.uint8), texture()]
    rows = [observe(model, item) for item in packets(images)]
    assert rows[2]['camera_selection']['selected_camera'] == 'auxiliary'
    assert rows[2]['camera_selection']['primary_continuity']['status'] == 'NO_CURRENT_MEASURED_TRANSLATION'
    assert rows[3]['camera_selection']['selected_camera'] == 'primary'
    for i, row in enumerate(rows):
        assert row['frame'] == i and not row['global_history_reset']
        assert np.linalg.norm(row['position_initial_body_m']) < .005
    assert rows[2]['registration']['fixed_reference_frame_adapter_used']


@pytest.mark.parametrize('status', ['ANCHOR_CONFLICT_OR_INVALID', 'ANCHOR_INCREMENT_CONFLICT'])
def test_qualified_primary_conflict_cannot_trigger_auxiliary_fallback(monkeypatch, status):
    model = DualCameraAnchorPose(); calls = []
    def conflict(self, current, G, now):
        calls.append(self.camera)
        self.last_continuity = dict(status=status)
        raise SensorContractError('synthetic qualified conflict')
    monkeypatch.setattr(JointTemporalAnchorRGBDPose, '_measure', conflict)
    with pytest.raises(SensorContractError): model._measure({}, np.eye(3), 100)
    assert calls == ['primary'] and not model.last_camera_selection['auxiliary_attempted']


def test_cross_camera_qualified_increment_disagreement_is_rejected(monkeypatch):
    model = DualCameraAnchorPose()
    def measure(self, current, G, now):
        if self.camera == 'primary':
            self.last_continuity = dict(status='MEASURED_BRIDGE_BUDGET_EXHAUSTED',
                incremental_available=True, incremental_position_initial_body_m=[0., 0., 0.],
                incremental_rotation_witness=dict(composed_rotation_initial_body_from_current_body=np.eye(3).tolist()))
            raise SensorContractError('synthetic exhausted bridge')
        self.last_continuity = dict(status='ANCHOR_MEASUREMENT')
        return dict(p=np.array([.03, 0., 0.]), R=np.eye(3)), False, False
    monkeypatch.setattr(JointTemporalAnchorRGBDPose, '_measure', measure)
    with pytest.raises(SensorContractError, match='conflict'): model._measure({}, np.eye(3), 100)
    assert model.last_continuity['status'] == 'CROSS_CAMERA_MEASUREMENT_CONFLICT'
    assert model.last_camera_selection['selected_camera'] is None


def test_ten_frame_bridge_limit_shared_between_cameras(monkeypatch):
    force_missing(monkeypatch, set(range(2, 15)))
    model = DualCameraAnchorPose()
    for i, item in enumerate(packets([texture()] * 14)):
        if i <= 11:
            row = observe(model, item)
            if i >= 2: assert not row['promoted_keyframe']
        else:
            with pytest.raises(SensorContractError): observe(model, item)
    assert model.failed and model.bridge_frames == 10
    assert model.last_continuity['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'


@pytest.mark.parametrize('fault', ['aux_pixels', 'aux_time', 'gap', 'gyro'])
def test_sensor_faults_latch_before_any_camera_fallback(fault):
    model = DualCameraAnchorPose(); items = list(packets([texture()] * 3))
    observe(model, items[0])
    p, d, f, now, image, depth = paired(items[1])
    if fault == 'aux_pixels': image['rgb'][0, 0, 0] ^= 1
    if fault == 'aux_time': image['available_ns'] = now+1
    if fault == 'gap': now += 100_000_000
    if fault == 'gyro': f['valid'][:] = False
    with pytest.raises(SensorContractError):
        model.observe(p, d, f, auxiliary_rgb=image, auxiliary_depth=depth, now_ns=now)
    assert model.failed and model.frame == 0
    with pytest.raises(SensorContractError): observe(model, items[2])
