from copy import deepcopy
from functools import partial
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.dual_camera_visual_motion_development import DualCameraVisualMotion, current_dual_camera_pose
from lewm.dual_camera_settled_controller_development import DualCameraSettledController
from lewm.overlap_retention_joint_observer_development import OverlapRetentionVisualLedMotion
from lewm.joint_floor_registered_evidence_development import current_joint_floor_registered_pose
from lewm.tests.test_dual_camera_anchor_pose_development import paired
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture


def observe(model, item):
    p, d, f, now, image, depth = paired(item)
    row = model.observe(p, d, f, auxiliary_rgb=image, auxiliary_depth=depth, now_ns=now)
    current_dual_camera_pose(row, p, image, depth, identity=(0, 0, 0), now_ns=now)
    return row


def test_primary_motion_preserves_original_fields_and_stale_pose_is_not_current():
    old, new = OverlapRetentionVisualLedMotion(), DualCameraVisualMotion()
    for item in packets([texture()] * 3):
        p, d, f, now = deepcopy(item)
        expected = old.observe(p, d, f, now_ns=now)
        actual = observe(new, item)
        normalized = {k: deepcopy(actual[k]) for k in expected}
        normalized['observer_variant'] = expected['observer_variant']
        for key in ('current_pose', 'last_visual'):
            for field in ('auxiliary_rgb_sha256', 'auxiliary_depth_sha256'):
                normalized[key].pop(field)
        for key in ('auxiliary_rgb', 'auxiliary_depth'): normalized['calibration_ids'].pop(key)
        assert normalized == expected
    stale = new.snapshot(now_ns=now+1)
    assert stale['current_pose'] is None and not stale['camera_selection_current']
    assert stale['last_visual']['measured_ns'] == now


def test_actual_auxiliary_fallback_has_valid_body_rotation_witnesses_and_camera_identity():
    model = DualCameraVisualMotion()
    images = [texture(), texture(), np.full((480, 640, 3), 128, np.uint8), texture()]
    rows = [observe(model, item) for item in packets(images)]
    assert rows[2]['camera_selection']['selected_camera'] == 'auxiliary'
    assert all(w['camera'] == 'auxiliary' for w in rows[2]['continuity_evidence']['rotation_measurement_witnesses'])
    assert rows[3]['camera_selection']['selected_camera'] == 'primary'
    rows[3]['camera_selection']['selected_camera'] = 'auxiliary'
    assert model.snapshot(now_ns=rows[3]['decision_ns'])['camera_selection']['selected_camera'] == 'primary'


@pytest.mark.parametrize('fault', ['auxiliary_hash', 'camera', 'calibration', 'current_flag'])
def test_additional_modality_witness_tampering_rejected(fault):
    model = DualCameraVisualMotion()
    for item in packets([texture()] * 2): row = observe(model, item)
    p, d, f, now, image, depth = paired(item)
    if fault == 'auxiliary_hash': row['current_pose']['auxiliary_rgb_sha256'] = '0'*64
    if fault == 'camera': row['camera_selection']['selected_camera'] = 'auxiliary'
    if fault == 'calibration': row['calibration_ids']['auxiliary_rgb'] = 'primary'
    if fault == 'current_flag': row['camera_selection_current'] = False
    with pytest.raises(SensorContractError):
        current_dual_camera_pose(row, p, image, depth, identity=(0, 0, 0), now_ns=now)


def test_controller_initial_public_pipeline_and_terminal_missing_rgb_stop(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets as floor_packets
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    controller = DualCameraSettledController(None, None, public_mission=dict(
        goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
        navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    p, d, a, _, now = floor_packets()
    image = from_captured_rgb(p['image']['rgb'], a, p, measured_ns=now, available_ns=now, now_ns=now)
    # Use a real fast-gyro packet on the same public body history.
    from lewm.fast_gyro_development import FastGyroBuffer
    gyro = FastGyroBuffer((0, 0, 0))
    for t in range(now-100_000_000, now+1, 2_000_000):
        gyro.append(np.zeros(3), np.ones(3, bool), measured_ns=t, available_ns=t)
    f = gyro.packet(now_ns=now)
    row = controller.observe(p, d, f, auxiliary_rgb=image, auxiliary_depth=a, now_ns=now)
    assert row['terminal'] is None, row['failure']
    position, rotation, _ = current_joint_floor_registered_pose(row['evidence'], identity=(0, 0, 0), now_ns=now)
    np.testing.assert_array_equal(controller.memory.position, position)
    np.testing.assert_array_equal(controller.residual.pose['position'], position)
    np.testing.assert_array_equal(controller.memory.rotation, rotation)
    assert row['requested_command'] == [0., 0., 0.]
    count = controller.memory.partition.total_returns
    row = controller.observe(p, d, f, auxiliary_rgb=None, auxiliary_depth=a, now_ns=now+100_000_000)
    assert row['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and row['requested_command'] == [0., 0., 0.]
    assert controller.memory.partition.total_returns == count
