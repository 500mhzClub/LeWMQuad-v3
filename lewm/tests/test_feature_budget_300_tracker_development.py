import numpy as np

from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.feature_budget_300_tracker_development import FeatureFrame300, FeatureBudget300Pose
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run


def test_smaller_spatial_population_keeps_original_sample_descriptors():
    rng=np.random.default_rng(2026091303)
    rgb=rng.integers(0,256,(480,640,3),dtype=np.uint8)
    depth=dict(depth_m=np.ones((480,640)),valid=np.ones((480,640),bool))
    full=CornerSupportFeatureFrame(rgb,depth);small=FeatureFrame300(rgb,depth)
    assert len(small.keypoints)==300 and max(small.cell_counts)<=25
    descriptors={k.pt:d for k,d in zip(full.keypoints,full.descriptors,strict=True)}
    for keypoint,descriptor in zip(small.keypoints,small.descriptors,strict=True):
        np.testing.assert_array_equal(descriptor,descriptors[keypoint.pt])
    assert small.witness()['maximum_features']==300


def test_actual_image_motion_keeps_plane_wrapper_and_admitted_pose():
    model=FeatureBudget300Pose()
    for i,item in enumerate(sequence(5)):
        result=run(model,item)
        assert result['frame']==i and result['global_history_reset'] is False
        assert len(model.previous.features['primary'].keypoints)<=300
        assert len(model.previous.features['auxiliary'].keypoints)<=300
        if i:assert result['registration']['measured_plane_refinement'] is not None
