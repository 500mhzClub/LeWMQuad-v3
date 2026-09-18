import numpy as np

from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.lazy_auxiliary_features_development import LazyCornerSupportFeatureFrame, LazyAuxiliaryPose
from lewm.full_consensus_tracker_development import FullConsensusPose
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run


def test_deferred_features_use_owned_original_pixels_and_identical_descriptors():
    p, d, _, _ = next(iter(sequence()))
    expected = CornerSupportFeatureFrame(p['image']['rgb'], d)
    actual = LazyCornerSupportFeatureFrame(p['image']['rgb'], d)
    p['image']['rgb'].fill(0); d['depth_m'].fill(0); d['valid'].fill(False)
    assert actual._computed is None
    np.testing.assert_array_equal(actual.gray, expected.gray)
    np.testing.assert_array_equal(actual.depth['depth_m'], expected.depth['depth_m'])
    np.testing.assert_array_equal(actual.descriptors, expected.descriptors)
    assert [k.pt for k in actual.keypoints] == [k.pt for k in expected.keypoints]
    assert actual.witness() == expected.witness()
    computed = actual._computed
    assert actual._features() is computed


def test_primary_success_defers_auxiliary_but_later_fallback_matches_original():
    baseline = FullConsensusPose(); candidate = LazyAuxiliaryPose()
    for frame, item in enumerate(sequence(4, blank_primary_at=(2,))):
        expected = run(baseline, item); actual = run(candidate, item)
        assert actual == expected
        aux = candidate.previous.features['auxiliary']
        assert isinstance(aux, LazyCornerSupportFeatureFrame)
        if frame < 2: assert aux._computed is None
        if frame == 2:
            assert actual['camera_selection']['selected_camera'] == 'auxiliary'
            assert aux._computed is not None
