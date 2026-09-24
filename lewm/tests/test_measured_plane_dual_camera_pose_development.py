"""Actual synthetic RGB-D sequences and inherited failure/ownership semantics."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, from_native_depth
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_packet
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorRGBDPose
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture
from lewm.tests.test_floor_pose_registration_development import render_plane


def sequence(count=4, blank_primary_at=(), blank_both_at=()):
    image = texture()
    raw_d, _ = render_plane(np.asarray(BODY_FROM_OPTICAL), np.array([0., 0., 1.]), .32)
    raw_a, _ = render_plane(body_from_optical(), np.array([0., 0., 1.]), .32)
    images = [np.full_like(image, 128) if i in (*blank_primary_at, *blank_both_at) else image for i in range(count)]
    for i, (policy, _, fast, now) in enumerate(packets(images)):
        primary = from_native_depth(raw_d, policy, measured_ns=now, available_ns=now, now_ns=now)
        auxiliary = auxiliary_packet(raw_a, policy, measured_ns=now, available_ns=now, now_ns=now)
        rgb = np.full_like(image, 128) if i in blank_both_at else image
        auxiliary_rgb = from_captured_rgb(rgb, auxiliary, policy,
            measured_ns=now, available_ns=now, now_ns=now)
        yield policy, primary, fast, dict(auxiliary_rgb=auxiliary_rgb, auxiliary_depth=auxiliary, now_ns=now)


def run(model, item):
    return model.observe(*item[:3], **item[3])


def test_actual_static_packets_produce_causal_pose_and_bound_floor_history():
    model = MeasuredPlaneDualCameraPose()
    previous_receipts = []
    for i, item in enumerate(sequence()):
        row = run(model, item)
        assert row['frame'] == i and row['measured_plane_constrained_estimator']
        np.testing.assert_allclose(row['position_initial_body_m'], 0, atol=1e-12)
        np.testing.assert_allclose(row['rotation_initial_body_from_current_body'], np.eye(3), atol=1e-12)
        retained = {r.frame for r in model.references} | {model.previous.frame}
        assert set(model._planes) == retained and len(retained) <= 9
        for f, (features, receipt) in model._planes.items():
            assert receipt['frame'] == f and receipt['measured_ns'] <= item[3]['now_ns']
            assert features is next(r.features for r in [*model.references, model.previous] if r.frame == f)
        if i:
            refinement = row['registration']['measured_plane_refinement']
            assert refinement['original_inliers_preserved']
            assert refinement['reference_floor']['frame'] == row['reference_frame']
            assert refinement['current_floor']['frame'] == i
            assert all(w['measured_plane_constrained'] for w in model.rotation_measurements)
        previous_receipts.append((deepcopy(row), row))
    assert all(before == after for before, after in previous_receipts)


def test_public_motion_witness_composes_new_fits_without_relabeling_gyro_role():
    model = MeasuredPlaneVisualMotion()
    for item in sequence():
        row = run(model, item)
        assert row['terminal_failure'] is None, row['terminal_failure']
        p, R, pose = current_joint_pose(row, identity=(0, 0, 0), now_ns=item[3]['now_ns'])
        assert row['measured_plane_constrained_estimator']
        assert pose['gyro_role'] == 'consistency_monitor_only'
        np.testing.assert_allclose(p, 0., atol=1e-12)


def test_missing_primary_uses_real_auxiliary_projection_and_returns_to_primary():
    model = MeasuredPlaneDualCameraPose()
    rows = [run(model, item) for item in sequence(blank_primary_at=(2,))]
    assert rows[2]['camera_selection']['selected_camera'] == 'auxiliary'
    assert rows[3]['camera_selection']['selected_camera'] == 'primary'
    assert rows[2]['registration']['fixed_reference_frame_adapter_used']
    assert rows[2]['registration']['measured_plane_refinement']['original_inliers_preserved']
    np.testing.assert_allclose(rows[2]['position_initial_body_m'], 0., atol=1e-12)


@pytest.mark.parametrize('fault', ['duplicate', 'missing_image', 'bad_initial_force', 'commanded_initial', 'auxiliary_clock'])
def test_invalid_or_missing_measurement_latches_and_cannot_restart(fault):
    model = MeasuredPlaneDualCameraPose()
    items = list(sequence(blank_both_at=(1,) if fault == 'missing_image' else ()))
    item = items[0]
    if fault in ('duplicate', 'missing_image'):
        run(model, item)
        item = items[0 if fault == 'duplicate' else 1]
    elif fault == 'bad_initial_force': item[0]['sensor_state']['sensed']['specific_force']['values'][:] = 0
    elif fault == 'commanded_initial': item[0]['sensor_state']['control']['applied_command']['values'][:] = .01
    elif fault == 'auxiliary_clock': item[3]['auxiliary_depth']['measured_ns'] += 1
    with pytest.raises((ValueError, TypeError)):
        run(model, item)
    assert model.failed
    state = model.frame, tuple(model._planes), len(model.references)
    with pytest.raises(SensorContractError, match='terminal'):
        run(model, items[-1])
    assert state == (model.frame, tuple(model._planes), len(model.references))


def test_reference_floor_cannot_be_attached_to_different_feature_owner():
    model = MeasuredPlaneDualCameraPose()
    run(model, next(sequence(1)))
    ref = model.references[-1]
    substituted = replace(ref, features=dict(ref.features))
    with pytest.raises(SensorContractError, match='ownership'):
        model._candidate(substituted, ref.features, np.eye(3))


@pytest.mark.parametrize('status', ['ANCHOR_CONFLICT_OR_INVALID', 'ANCHOR_INCREMENT_CONFLICT'])
def test_existing_qualified_conflicts_remain_terminal_without_auxiliary_fallback(monkeypatch, status):
    model = MeasuredPlaneDualCameraPose()
    monkeypatch.setattr(model, '_prepare_plane', lambda *args: None)
    calls = []
    def conflict(self, current, G, now):
        calls.append(self.camera)
        self.last_continuity = dict(status=status)
        raise SensorContractError('original qualified conflict')
    monkeypatch.setattr(JointTemporalAnchorRGBDPose, '_measure', conflict)
    with pytest.raises(SensorContractError, match='qualified conflict'):
        model._measure({}, np.eye(3), 0)
    assert calls == ['primary']


def test_returned_plane_receipts_do_not_alias_private_reference_history():
    model = MeasuredPlaneDualCameraPose()
    row = run(model, next(sequence(1)))
    row['measured_plane_evidence']['joint_plane']['normal_body'][2] = 0.
    assert model._planes[0][1]['joint_plane']['normal_body'][2] > .99
    assert model.last_measured_plane['joint_plane']['normal_body'][2] > .99


def test_ten_measured_bridges_keep_original_references_and_eleventh_fails(monkeypatch):
    from lewm.tests.test_temporal_anchor_continuity_development import force_missing
    force_missing(monkeypatch, set(range(1, 12)))
    model = MeasuredPlaneDualCameraPose()
    items = list(sequence(12))
    initial = run(model, items[0])
    reference = model.references[0]
    for frame in range(1, 11):
        row = run(model, items[frame])
        assert not row['promoted_keyframe']
        assert model.last_continuity['status'] == 'MEASURED_INCREMENT_BRIDGE'
        assert model.bridge_frames == model.total_bridge_frames == frame
        assert model.references == [reference] and model.references[0] is reference
        assert set(model._planes) == {0, frame}
    with pytest.raises(SensorContractError):
        run(model, items[11])
    assert model.failed and model.bridge_frames == 10 and model.total_bridge_frames == 10
    assert model.last_continuity['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
    assert model.references[0] is reference


def test_plane_image_conflict_cannot_be_swallowed_as_missingness(monkeypatch):
    from lewm import measured_plane_dual_camera_pose_development as module
    model = MeasuredPlaneDualCameraPose()
    items = list(sequence(2))
    run(model, items[0])
    before = model.references[0]
    calls = []
    def conflict(*args, **kwargs):
        calls.append(kwargs['camera'])
        raise module.PlaneImageConflict('synthetic qualified plane/image disagreement')
    monkeypatch.setattr(module, 'refine', conflict)
    with pytest.raises(SensorContractError) as error:
        run(model, items[1])
    assert isinstance(error.value.__cause__, module.PlaneImageConflict)
    assert calls == ['primary'] and model.failed
    assert model.last_continuity['status'] == 'MEASURED_PLANE_IMAGE_CONFLICT'
    assert model.references == [before] and model.previous.frame == 0
    assert not model.rotation_measurements


def test_refinement_rejection_never_publishes_preliminary_unconstrained_witness(monkeypatch):
    from lewm import measured_plane_dual_camera_pose_development as module
    model = MeasuredPlaneDualCameraPose()
    items = list(sequence(2))
    run(model, items[0])
    def rejection(*args, **kwargs):
        raise SensorContractError('synthetic original inlier lost')
    monkeypatch.setattr(module, 'refine', rejection)
    with pytest.raises(SensorContractError):
        run(model, items[1])
    assert model.failed and model.previous.frame == 0
    assert not model.rotation_measurements
    assert model.last_continuity['rotation_measurement_witnesses'] == []


def test_floor_candidate_up_uses_previous_visual_attitude_not_current_gyro():
    model = MeasuredPlaneDualCameraPose()
    items = list(sequence(2))
    run(model, items[0])
    model.frame = 1
    from lewm.corner_support_features_development import CornerSupportFeatureFrame
    item = items[1]
    current = dict(primary=CornerSupportFeatureFrame(item[0]['image']['rgb'], item[1]),
        auxiliary=CornerSupportFeatureFrame(item[3]['auxiliary_rgb']['rgb'], item[3]['auxiliary_depth']))
    model._pending_plane = None
    from lewm.tests.test_floor_pose_registration_development import rotation
    receipt = model._prepare_plane(current, rotation([1., 0., 0.], .7), item[3]['now_ns'])
    assert receipt['current_up_uses_public_gyro'] is False
    assert receipt['up_reference_visual_frame'] == 0
    np.testing.assert_allclose(receipt['joint_plane']['up_body'], model._initial_up, atol=0, rtol=0)


def test_current_floor_count_missingness_retains_qualified_visual_fit(monkeypatch):
    from lewm import measured_plane_dual_camera_pose_development as module
    model = MeasuredPlaneDualCameraPose()
    items = list(sequence(3))
    run(model, items[0])
    original = module.fit_joint_plane
    def missing(a, b, up):
        return original(np.empty((0, 3)), np.empty((0, 3)), up)
    monkeypatch.setattr(module, 'fit_joint_plane', missing)
    row = run(model, items[1])
    receipt = row['registration']['measured_plane_refinement']
    assert not receipt['applied'] and receipt['original_image_fit_retained']
    assert receipt['missing_plane_not_admitted']
    assert not receipt['current_floor']['joint_plane']['available']
    assert model.previous.frame == 1 and not model.failed
    assert all(not w['measured_plane_constrained'] for w in model.rotation_measurements)
    monkeypatch.setattr(module, 'fit_joint_plane', original)
    row = run(model, items[2])
    assert row['registration']['measured_plane_refinement']['applied']
    assert model.previous.frame == 2 and not model.failed


def test_coherent_moments_with_conflicting_points_do_not_use_missingness_fallback(monkeypatch):
    from lewm import measured_plane_dual_camera_pose_development as module
    model = MeasuredPlaneDualCameraPose()
    items = list(sequence(2))
    run(model, items[0])
    original = module.fit_joint_plane
    def incoherent(a, b, up):
        receipt = original(a, b, up)
        receipt.update(available=False, reason='combined_points_not_one_coherent_plane')
        return receipt
    monkeypatch.setattr(module, 'fit_joint_plane', incoherent)
    with pytest.raises(SensorContractError):
        run(model, items[1])
    assert model.failed and model.previous.frame == 0
