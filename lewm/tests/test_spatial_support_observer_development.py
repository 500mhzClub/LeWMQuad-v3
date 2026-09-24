"""Spatial support, frozen gate identity and real RGB-D interface tests."""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from lewm.spatial_support_features_development import SpatialSupportFeatureFrame
from lewm.spatial_support_joint_observer_development import SpatialSupportJointRGBDPose, SpatialSupportVisualLedMotion
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
    new = method('lewm/spatial_support_joint_observer_development.py', 'SpatialSupportJointRGBDPose')
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == 'SpatialSupportFeatureFrame': node.id = 'FeatureFrame'
            return node
        def visit_Constant(self, node):
            if node.value == 'joint': node.value = 'gyro'
            if node.value == 'consistency_monitor_only': node.value = 'rotation_estimator'
            return node
    assert ast.dump(old) == ast.dump(Normalize().visit(new))
    for name in ('_candidate', '_measure', '_choose', '_remember'):
        assert getattr(SpatialSupportJointRGBDPose, name) is getattr(JointTemporalAnchorRGBDPose, name)


def test_spatial_selection_is_bounded_deterministic_and_has_no_input_alias():
    p, d, _, _ = next(packets([texture()]))
    before = p['image']['rgb'].copy()
    a = SpatialSupportFeatureFrame(p['image']['rgb'], d)
    b = SpatialSupportFeatureFrame(p['image']['rgb'], d)
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
    a = SpatialSupportFeatureFrame(p['image']['rgb'], d)
    assert all(k.pt[0] >= 320 for k in a.keypoints)
    assert all(a.cell_counts[i] == 0 for i in (0, 1, 4, 5, 8, 9))
    d['valid'][:] = False; d['depth_m'][:] = 0.
    b = SpatialSupportFeatureFrame(p['image']['rgb'], d)
    assert not b.keypoints and b.descriptors is None


def test_real_rendered_rgbd_preserves_joint_measurement_and_goal_admission():
    runtime = SpatialSupportVisualLedMotion()
    for item in sequence(2., 1, rate=(.01, -.02, .03), steps=5):
        p, d, f, now = item[:4]
        result = runtime.observe(p, d, f, now_ns=now)
        assert result['terminal_failure'] is None, result['terminal_failure']
        measured, _, _ = current_joint_pose(result, identity=(0, 0, 0), now_ns=now)
        np.testing.assert_allclose(measured, item[4], atol=.006, rtol=0)
        assert result['last_accepted_feature_witness']['selected_features'] <= 600


@pytest.mark.parametrize('fault', ['blank', 'clock'])
def test_failure_latches_without_pose_or_reinitialization(fault):
    runtime = SpatialSupportVisualLedMotion()
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
