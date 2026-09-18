import numpy as np
from lewm.feature_budget_150_tracker_development import FeatureFrame150
from lewm.feature_budget_300_tracker_development import FeatureFrame300


def test_total_cap_and_descriptors_match_retained_original_features():
    rgb=np.random.default_rng(2026091305).integers(0,256,(480,640,3),dtype=np.uint8)
    depth=dict(depth_m=np.ones((480,640)),valid=np.ones((480,640),bool))
    small=FeatureFrame150(rgb,depth);large=FeatureFrame300(rgb,depth)
    assert len(small.keypoints)==sum(small.cell_counts)==150 and max(small.cell_counts)<=13
    lookup={k.pt:d for k,d in zip(large.keypoints,large.descriptors,strict=True)}
    for keypoint,descriptor in zip(small.keypoints,small.descriptors,strict=True):
        np.testing.assert_array_equal(descriptor,lookup[keypoint.pt])
