"""Multi-frame measured-seed geometry, bad priors and retained limitations."""
from copy import deepcopy
import hashlib

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import lewm.measured_pose_seeded_rgbd_development as warm
from lewm.gyro_seeded_rgbd_correspondence_development import matched_points as zero_seed
from lewm.causal_depth_observation_development import FOCAL, from_native_depth
from lewm.causal_sensor_state import SensorContractError
from lewm.keyframe_rgbd_pose_development import FeatureFrame
from lewm.rgbd_correspondence_motion_development import T
from lewm.tests.test_correlated_moment_sensitivity_development import stream
from lewm.tests.test_rgbd_correspondence_motion_development import texture, packets
from lewm.tests.test_gyro_seeded_rgbd_correspondence_development import plane_pair, frames


def scene_frame(depth, position, rotation, *, foreground=False):
    """Synthetic static optical-reference planes with a bounded nearer panel.

    Independent fixture geometry produces only RGB and optical depth for the
    estimator. Panel visibility uses nearest positive intersections, including
    disocclusion; each plane has distinct texture. No geometry enters tracking.
    """
    R = np.asarray(rotation); t = np.asarray(position)
    Q = T[:3, :3].T @ R @ T[:3, :3]
    c = T[:3, :3].T @ (R @ T[:3, 3] + t - T[:3, 3])
    yy, xx = np.indices((480, 640), dtype=float)
    rays = np.stack(((xx + .5 - 320) / FOCAL, (yy + .5 - 240) / FOCAL, np.ones_like(xx)), -1)
    directions = rays @ Q.T
    result = np.zeros((480, 640, 3), np.uint8); zbuffer = np.full((480, 640), np.inf)
    for plane_index, z in enumerate((depth, depth * .65) if foreground else (depth,)):
        distance = (z - c[2]) / directions[:, :, 2]
        point = directions * distance[:, :, None] + c
        use = (distance >= .2) & (distance <= 5.) & (distance < zbuffer)
        if plane_index: use &= (np.abs(point[:, :, 0]) < .28) & (np.abs(point[:, :, 1]) < .23)
        uv = point[:, :, :2] / z * FOCAL + [319.5, 239.5]
        image = texture() if not plane_index else np.flip(texture(), axis=0).copy()
        rendered = cv2.remap(image, uv[:, :, 0].astype(np.float32), uv[:, :, 1].astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        result[use] = rendered[use]; zbuffer[use] = distance[use]
    return result, np.where(np.isfinite(zbuffer), zbuffer, 0.).astype(np.float32)


def sequence(depth, axis, sign=1, foreground=False, rate=(0., 0., 0.), steps=13):
    for tick, policy, fast, _ in stream(steps, rate):
        position = np.zeros(3); position[axis] = sign * tick * .04
        R = Rotation.from_rotvec(np.asarray(rate) * tick * .1).as_matrix()
        image, optical_depth = scene_frame(depth, position, R, foreground=foreground)
        policy['image']['rgb'] = image; now = policy['sensor_state']['decision_ns']
        packet = from_native_depth(optical_depth, policy, measured_ns=now, available_ns=now, now_ns=now)
        yield policy, packet, fast, now, position, R


@pytest.mark.parametrize('depth,axis,sign,foreground,rate', [
    (.7, 1, 1, False, (0., 0., 0.)),
    (2., 1, 1, False, (0., 0., 0.)),
    (3.5, 1, -1, False, (0., 0., 0.)),
    (2., 0, 1, False, (0., 0., 0.)),
    (2., 0, -1, False, (0., 0., 0.)),
    (2., 1, 1, True, (0., 0., 0.)),
    (2., 1, -1, True, (.01, -.02, .03)),
])
def test_multiframe_seed_is_previous_estimate_and_current_pose_is_measured(depth, axis, sign, foreground, rate):
    m = warm.MeasuredPoseSeededVisualLedMotion(); previous = None; promoted = 0
    for p, d, f, now, truth, R in sequence(depth, axis, sign, foreground, rate):
        refs = {ref.frame: ref for ref in m.model.references}
        row = m.observe(p, d, f, now_ns=now)
        pose = row['current_pose']; assert pose is not None, (m.model.frame, row['correspondence_attempts'])
        np.testing.assert_allclose(pose['position_initial_body_m'], truth, atol=.003, rtol=0)
        np.testing.assert_allclose(pose['rotation_initial_body_from_current_body'], R, atol=1e-10, rtol=0)
        for attempt in row['correspondence_attempts']:
            ref = refs[attempt['reference_frame']]
            expected = ref.rotation.T @ (np.asarray(previous['position_initial_body_m']) - ref.position)
            np.testing.assert_array_equal(attempt['counts']['translation_seed_reference_m'], expected)
            assert attempt['seed_measured_ns'] == previous['measured_ns'] == now - 100_000_000
            assert not attempt['seed_is_current_pose']
        promoted += pose['promoted_keyframe']; previous = pose
    assert promoted > 0 and np.linalg.norm(previous['position_initial_body_m']) > .45
    assert not row['navigation_qualified'] and not row['rigid_registration_gates_changed']


def test_zero_seed_reproduces_frozen_correspondence_arrays_exactly():
    a, b = plane_pair(np.eye(3), np.array([0., .1, 0.]))
    original = zero_seed(a, b, np.eye(3)); actual = warm.matched_points(a, b, np.eye(3), [0., 0., 0.])
    for x, y in zip(original[:4], actual[:4], strict=True): np.testing.assert_array_equal(x, y)


def test_seed_is_not_added_as_a_translation_constraint_or_returned_as_pose():
    a, b = plane_pair(np.eye(3), np.array([0., .02, 0.]))
    seed = np.array([0., .06, 0.]); frozen = seed.copy()
    left, right, ua, ub, counts = warm.matched_points(a, b, np.eye(3), seed)
    _, estimated, _, _ = warm.register(left, right, ua, ub, gyro_rotation=np.eye(3), mode='gyro', frame=1)
    np.testing.assert_allclose(estimated, [0., .02, 0.], atol=.001, rtol=0)
    assert np.linalg.norm(estimated - seed) > .03 and not counts['seed_is_current_pose']
    np.testing.assert_array_equal(seed, frozen)


@pytest.mark.parametrize('seed', [[np.nan, 0., 0.], [4., 0., 0.], [0., 0.], [[0., 0., 0.]]])
def test_invalid_or_unbounded_prior_is_rejected(seed):
    a, b = frames(texture(), texture())
    with pytest.raises(SensorContractError): warm.matched_points(a, b, np.eye(3), seed)


def test_large_wrong_prior_does_not_create_a_pose_on_unrelated_images():
    a, b = frames(texture(), np.flip(texture(), axis=0).copy())
    left, right, ua, ub, _ = warm.matched_points(a, b, np.eye(3), [0., .3, 0.])
    with pytest.raises(SensorContractError): warm.register(left, right, ua, ub, gyro_rotation=np.eye(3), mode='gyro', frame=1)


@pytest.mark.parametrize('fault', ['stale', 'future', 'wrong_clock', 'depth_time', 'blank', 'privileged'])
def test_bad_seed_clock_or_sensor_failure_latches_without_extrapolation(fault):
    m = warm.MeasuredPoseSeededVisualLedMotion(); items = list(packets([texture()] * 3))
    p, d, f, now = items[0]; m.observe(p, d, f, now_ns=now)
    p, d, f, now = deepcopy(items[1])
    if fault == 'stale': m.model.last_pose_ns -= 100_000_000
    if fault == 'future': m.model.last_pose_ns += 200_000_000
    if fault == 'wrong_clock': now = True
    if fault == 'depth_time': d['available_ns'] = now + 1
    if fault == 'blank':
        p['image']['rgb'][:] = 128; d['rgb_sha256'] = hashlib.sha256(p['image']['rgb'].tobytes()).hexdigest()
    if fault == 'privileged': p['native_pose'] = [0., 0., 0.]
    if fault == 'wrong_clock':
        with pytest.raises(SensorContractError): m.model.observe(p, d, f, now_ns=now)
        assert m.model.failed
    else:
        row = m.observe(p, d, f, now_ns=now); assert row['current_pose'] is None
        p, d, f, now = items[2]; row = m.observe(p, d, f, now_ns=now)
        assert row['current_pose'] is None and row['correspondence_attempts'] == []


def test_seed_cannot_resolve_periodic_image_ambiguity():
    rng = np.random.default_rng(917)
    tile = cv2.GaussianBlur(rng.integers(0, 256, (40, 40), np.uint8), (5, 5), 1.)
    image = np.repeat(np.tile(tile, (12, 16))[:, :, None], 3, 2); a, b = frames(image, image)
    left, right, ua, ub, _ = warm.matched_points(a, b, np.eye(3), [0., 0., 0.])
    _, t, _, _ = warm.register(left, right, ua, ub, gyro_rotation=np.eye(3), mode='gyro', frame=1)
    assert np.linalg.norm(t) < 1e-6 and np.linalg.norm(t - [0., 80 / FOCAL, 0.]) > .1
