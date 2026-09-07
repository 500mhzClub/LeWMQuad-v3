"""Known-image geometry and explicit counterexamples, not navigation evidence."""
from copy import deepcopy
import hashlib

import cv2
import numpy as np
import pytest

import lewm.gyro_seeded_rgbd_correspondence_development as flow
from lewm.causal_depth_observation_development import FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.keyframe_rgbd_pose_development import FeatureFrame
from lewm.rgbd_correspondence_motion_development import T
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_rgbd_correspondence_motion_development import texture, packets, point_fixture


def frames(left, right):
    return tuple(FeatureFrame(p['image']['rgb'], d) for p, d, _, _ in packets([left, right]))


def plane_pair(R, t):
    """Render a known static fronto-parallel plane into the moved camera.

    Fixture truth is used only to synthesize images/depth, never in the matcher.
    Includes body-camera translation and pixel-centre conventions.
    """
    image = texture()
    Q = T[:3, :3].T @ R @ T[:3, :3]
    c = T[:3, :3].T @ (R @ T[:3, 3] + t - T[:3, 3])
    yy, xx = np.indices((480, 640), dtype=float)
    rays = np.stack(((xx + .5 - 320) / FOCAL, (yy + .5 - 240) / FOCAL, np.ones_like(xx)), -1)
    reference_rays = rays @ Q.T
    depth = (2 - c[2]) / reference_rays[:, :, 2]
    ref = reference_rays * depth[:, :, None] + c
    uv = ref[:, :, :2] / ref[:, :, 2:] * FOCAL + [319.5, 239.5]
    moved = cv2.remap(image, uv[:, :, 0].astype(np.float32), uv[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    left, right = frames(image, moved)
    right.depth['depth_m'] = depth.astype(np.float32)
    return left, right


@pytest.mark.parametrize('rotation,translation', [
    ([0., 0., 0.], [0., .02, 0.]),
    ([.01, -.015, .03], [.005, .01, -.003]),
    ([0., 0., -.03], [0., -.01, .005]),
])
def test_known_plane_motion_with_camera_lever_arm(rotation, translation):
    R = rotation_increment(rotation); t = np.asarray(translation)
    a, b = plane_pair(R, t)
    left, right, ua, ub, counts = flow.matched_points(a, b, R)
    _, actual, _, quality = flow.register(left, right, ua, ub, gyro_rotation=R, mode='gyro', frame=1)
    np.testing.assert_allclose(actual, t, atol=.001, rtol=0)
    assert quality['matched_consensus_rules'] and counts['paired_depth_survivors'] >= 12
    assert not counts['command_translation_prior_used'] and not counts['descriptor_association_used']


def test_current_descriptor_detection_is_not_required_and_inputs_are_unchanged():
    image = texture(); moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 0]]), (640, 480))
    left, right = frames(image, moved)
    expected = flow.matched_points(left, right, np.eye(3))
    before = (left.gray.copy(), right.gray.copy(), left.depth['depth_m'].copy(), right.depth['depth_m'].copy())
    right.keypoints = []; right.descriptors = None; left.descriptors = None
    actual = flow.matched_points(left, right, np.eye(3))
    for x, y in zip(expected[:4], actual[:4], strict=True): np.testing.assert_array_equal(x, y)
    for x, y in zip(before, (left.gray, right.gray, left.depth['depth_m'], right.depth['depth_m']), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('kind', ['blank', 'occluded', 'unrelated'])
def test_no_pose_for_uninformative_or_inconsistent_images(kind):
    image = texture(); other = image.copy()
    if kind == 'blank': image[:] = 128; other[:] = 128
    if kind == 'occluded': other[:] = 0
    if kind == 'unrelated': other = np.flip(other, axis=0).copy()
    a, b = frames(image, other)
    left, right, ua, ub, _ = flow.matched_points(a, b, np.eye(3))
    with pytest.raises(SensorContractError):
        flow.register(left, right, ua, ub, gyro_rotation=np.eye(3), mode='gyro', frame=1)


def test_periodic_texture_exposes_unobservable_motion_not_a_false_safety_claim():
    rng = np.random.default_rng(917)
    tile = cv2.GaussianBlur(rng.integers(0, 256, (40, 40), np.uint8), (5, 5), 1.)
    image = np.repeat(np.tile(tile, (12, 16))[:, :, None], 3, axis=2)
    # A 40-pixel tangent shift of an infinite periodic plane produces exactly
    # the same two images and depths as zero motion. No image-only method can
    # discriminate these two fixture worlds from these observations.
    np.testing.assert_array_equal(image, np.roll(image, 40, axis=1))
    a, b = frames(image, image)
    left, right, ua, ub, counts = flow.matched_points(a, b, np.eye(3))
    _, t, _, q = flow.register(left, right, ua, ub, gyro_rotation=np.eye(3), mode='gyro', frame=1)
    assert np.linalg.norm(t) < 1e-6 and q['matched_consensus_rules']
    alternative_true_translation = np.array([0., 80 / FOCAL, 0.])
    assert np.linalg.norm(t - alternative_true_translation) > .1
    assert not counts['uncertainty_calibrated']


@pytest.mark.parametrize('bias,rejected', [(.002, False), (.01, True)])
def test_common_depth_bias_is_not_always_detected_by_rigid_gates(bias, rejected):
    a, b = plane_pair(np.eye(3), np.zeros(3))
    b.depth['depth_m'] += np.float32(bias)
    left, right, ua, ub, _ = flow.matched_points(a, b, np.eye(3))
    if rejected:
        with pytest.raises(SensorContractError):
            flow.register(left, right, ua, ub, gyro_rotation=np.eye(3), mode='gyro', frame=1)
        return
    _, t, _, q = flow.register(left, right, ua, ub, gyro_rotation=np.eye(3), mode='gyro', frame=1)
    assert q['matched_consensus_rules'] and np.linalg.norm(t) > .001


@pytest.mark.parametrize('fault', ['nan_rotation', 'reflection', 'image_shape', 'depth_nan', 'mask', 'unknown_value'])
def test_invalid_matching_inputs_fail(fault):
    a, b = frames(texture(), texture()); R = np.eye(3)
    if fault == 'nan_rotation': R[0, 0] = np.nan
    if fault == 'reflection': R[0, 0] = -1
    if fault == 'image_shape': b.gray = b.gray[:-1]
    if fault == 'depth_nan': b.depth['depth_m'][0, 0] = np.nan
    if fault == 'mask': b.depth['valid'] = b.depth['valid'].astype(np.uint8)
    if fault == 'unknown_value': b.depth['valid'][0, 0] = False
    with pytest.raises(SensorContractError): flow.matched_points(a, b, R)


def test_unknown_reference_depth_and_duplicate_orientations_do_not_create_points():
    a, b = frames(texture(), texture())
    actual = flow.matched_points(a, b, np.eye(3))
    a.keypoints = [p for p in a.keypoints for _ in range(3)]
    repeated = flow.matched_points(a, b, np.eye(3))
    for x, y in zip(actual[:4], repeated[:4], strict=True): np.testing.assert_array_equal(x, y)
    a.depth['depth_m'][:] = 0; a.depth['valid'][:] = False
    *points, counts = flow.matched_points(a, b, np.eye(3))
    assert all(len(x) == 0 for x in points) and counts['reference_depth_survivors'] == 0


@pytest.mark.parametrize('fault', ['depth_edge', 'behind_camera', 'forward_nan', 'converging_tracks'])
def test_bad_seed_depth_or_flow_never_creates_multiple_constraints(monkeypatch, fault):
    a, b = frames(texture(), texture()); R = np.eye(3)
    if fault == 'depth_edge':
        a.keypoints = [cv2.KeyPoint(100.5, 100.5, 5.)]
        a.depth['depth_m'][100, 100] = 3.
    if fault == 'behind_camera': R = rotation_increment([0., 0., np.pi])
    calls = []
    def optical(left, right, p, initial, **kwargs):
        calls.append(p.copy())
        assert np.isfinite(p).all() and np.isfinite(initial).all()
        q = initial.copy()
        if len(calls) == 1: q[:] = np.nan if fault == 'forward_nan' else [100., 100.]
        return q, np.ones((len(p), 1), np.uint8), np.zeros((len(p), 1), np.float32)
    monkeypatch.setattr(flow.cv2, 'calcOpticalFlowPyrLK', optical)
    *points, counts = flow.matched_points(a, b, R)
    if fault in ('depth_edge', 'behind_camera'): assert not calls and len(points[0]) == 0
    if fault == 'forward_nan': assert len(calls) == 1 and len(points[0]) == 0
    if fault == 'converging_tracks':
        assert len(calls) == 2 and len(points[0]) == 1 and counts['unique_current_points'] == 1


@pytest.mark.parametrize('fault', ['clock', 'depth_clock', 'episode', 'image_binding', 'privileged', 'blank'])
def test_runtime_failure_latches_and_does_not_repeat_stale_candidate_evidence(fault):
    items = list(packets([texture()] * 3)); m = flow.GyroSeededVisualLedMotion()
    p, d, f, now = items[0]; m.observe(p, d, f, now_ns=now)
    p, d, f, now = deepcopy(items[1])
    if fault == 'clock': now += 1
    if fault == 'depth_clock': d['available_ns'] = now + 1
    if fault == 'episode': d['identity'] = (9, 0, 0)
    if fault == 'image_binding': p['image']['rgb'][0, 0] = 0
    if fault == 'privileged': p['native_pose'] = [0, 0, 0]
    if fault == 'blank':
        p['image']['rgb'][:] = 128; d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    r = m.observe(p, d, f, now_ns=now)
    assert r['current_pose'] is None and r['status'] == 'VISUAL_TERMINAL_FAILURE'
    p, d, f, now = items[2]; r = m.observe(p, d, f, now_ns=now)
    assert r['current_pose'] is None and r['correspondence_attempts'] == []
    assert len(m.model.references) == 1


def test_runtime_uses_new_matcher_and_preserves_registration_gates():
    image = texture(); moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 0]]), (640, 480))
    m = flow.GyroSeededVisualLedMotion()
    for p, d, f, now in packets([image, moved]): r = m.observe(p, d, f, now_ns=now)
    assert r['current_pose'] is not None and len(r['correspondence_attempts']) == 1
    np.testing.assert_allclose(r['current_pose']['position_initial_body_m'], [0., 6 / FOCAL, 0.], atol=.001)
    assert not r['rigid_registration_gates_changed'] and r['correspondence_checks_changed']
    assert not r['navigation_qualified']


@pytest.mark.parametrize('fault', ['coverage', 'wrong_associations', 'gyro_error'])
def test_original_registration_still_rejects_inconsistent_point_constraints(fault):
    a, b, ua, ub, R, t = point_fixture()
    if fault == 'coverage': ua[:] = [80, 80]; ub[:] = [80, 80]
    if fault == 'wrong_associations': b = b[::-1].copy()
    if fault == 'gyro_error': R = R @ rotation_increment([0., 0., .15])
    with pytest.raises(SensorContractError): flow.register(a, b, ua, ub, gyro_rotation=R, mode='gyro', frame=1)
