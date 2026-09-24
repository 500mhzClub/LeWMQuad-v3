"""Corner support, frozen gate identity and real RGB-D interface tests."""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.corner_support_joint_observer_development import CornerSupportJointRGBDPose, CornerSupportVisualLedMotion
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorRGBDPose
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.tests.test_rgbd_correspondence_motion_development import texture, packets
from lewm.tests.test_measured_pose_seeded_rgbd_development import sequence


def test_only_feature_constructor_and_explicit_joint_labels_change_in_observe():
    def method(path, cls):
        tree = ast.parse(Path(path).read_text())
        c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        return next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'observe')
    old = method('lewm/temporal_anchor_continuity_development.py', 'TemporalAnchorRGBDPose')
    new = method('lewm/corner_support_joint_observer_development.py', 'CornerSupportJointRGBDPose')
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'CornerSupportFeatureFrame': node.id = 'FeatureFrame'
            return node
        def visit_Constant(self, node):
            if node.value == 'joint': node.value = 'gyro'
            if node.value == 'consistency_monitor_only': node.value = 'rotation_estimator'
            return node
    assert ast.dump(old) == ast.dump(Normalize().visit(new))
    for name in ('_candidate', '_measure', '_choose', '_remember'):
        assert getattr(CornerSupportJointRGBDPose, name) is getattr(JointTemporalAnchorRGBDPose, name)


def test_spatial_selection_is_bounded_deterministic_and_has_no_input_alias():
    p, d, _, _ = next(packets([texture()]))
    before = p['image']['rgb'].copy()
    a = CornerSupportFeatureFrame(p['image']['rgb'], d)
    b = CornerSupportFeatureFrame(p['image']['rgb'], d)
    assert 0 < len(a.keypoints) <= 600 and max(a.cell_counts) <= 50
    assert min(a.cell_counts) > 0
    assert [k.pt for k in a.keypoints] == [k.pt for k in b.keypoints]
    np.testing.assert_array_equal(a.descriptors, b.descriptors)
    a.rgb[:] = 0; a.depth['depth_m'][:] = 0
    np.testing.assert_array_equal(p['image']['rgb'], before)
    assert d['depth_m'].max() == 2.


def test_unknown_depth_cannot_contribute_to_spatial_coverage():
    p, d, _, _ = next(packets([texture()]))
    d['valid'][:, :320] = False; d['depth_m'][:, :320] = 0.
    a = CornerSupportFeatureFrame(p['image']['rgb'], d)
    assert all(k.pt[0] >= 320 for k in a.keypoints)
    assert all(a.cell_counts[i] == 0 for i in (0, 1, 4, 5, 8, 9))
    d['valid'][:] = False; d['depth_m'][:] = 0.
    b = CornerSupportFeatureFrame(p['image']['rgb'], d)
    assert not b.keypoints and b.descriptors is None


def test_real_rendered_rgbd_preserves_joint_measurement_and_goal_admission():
    runtime = CornerSupportVisualLedMotion()
    for item in sequence(2., 1, rate=(.01, -.02, .03), steps=5):
        p, d, f, now = item[:4]
        result = runtime.observe(p, d, f, now_ns=now)
        assert result['terminal_failure'] is None, result['terminal_failure']
        measured, _, _ = current_joint_pose(result, identity=(0, 0, 0), now_ns=now)
        np.testing.assert_allclose(measured, item[4], atol=.006, rtol=0)
        assert result['last_accepted_feature_witness']['selected_features'] <= 600


@pytest.mark.parametrize('fault', ['blank', 'clock'])
def test_failure_latches_without_pose_or_reinitialization(fault):
    runtime = CornerSupportVisualLedMotion()
    items = list(packets([texture()]*3))
    p, d, f, now = items[0]
    assert runtime.observe(p, d, f, now_ns=now)['current_pose'] is not None
    p, d, f, now = deepcopy(items[1])
    if fault == 'blank': p['image']['rgb'][:] = 128
    else: now += 1
    assert runtime.observe(p, d, f, now_ns=now)['current_pose'] is None
    p, d, f, now = items[2]
    result = runtime.observe(p, d, f, now_ns=now)
    assert result['current_pose'] is None and result['terminal_failure'] is not None


def test_coarse_block_corners_support_unchanged_registration():
    import cv2
    from lewm.keyframe_rgbd_pose_development import matched_points
    from lewm.joint_rgbd_rigid_pose_development import register
    from lewm.causal_depth_observation_development import FOCAL
    rng = np.random.default_rng(2026090970)
    values = rng.integers(10, 245, (6, 8), dtype=np.uint8)
    gray = np.repeat(np.repeat(values, 80, axis=0), 80, axis=1)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 0]]), (640, 480))
    frames = [CornerSupportFeatureFrame(p["image"]["rgb"], d) for p, d, _, _ in packets([image, moved])]
    a, b, ua, ub = matched_points(*frames)
    R, t, _, witness = register(a, b, ua, ub, gyro_rotation=np.eye(3), mode="joint", frame=1)
    assert min(witness["reference_grid_cells"], witness["current_grid_cells"]) >= 6
    np.testing.assert_allclose(R, np.eye(3), atol=.001, rtol=0)
    np.testing.assert_allclose(t, [0., 6/FOCAL, 0.], atol=.002, rtol=0)


def test_repeated_checker_does_not_gain_descriptor_distinctiveness():
    from lewm.keyframe_rgbd_pose_development import matched_points
    gray = ((np.indices((480, 640))//20).sum(0)%2*255).astype(np.uint8)
    image = np.repeat(gray[:, :, None], 3, 2)
    frames = [CornerSupportFeatureFrame(p["image"]["rgb"], d) for p, d, _, _ in packets([image, np.roll(image, 3, axis=1)])]
    assert len(matched_points(*frames)[0]) < 12

