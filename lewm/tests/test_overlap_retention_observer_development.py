import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_joint_observer_development import CornerSupportJointRGBDPose
from lewm.overlap_retention_joint_observer_development import (
    overlap_retention, OverlapRetentionJointRGBDPose)


def test_overlap_boundary_and_no_bridge_promotion():
    def check(n, **kw):
        return overlap_retention(inliers=n, reference_features=400,
            **(dict(anchored=True, already_promoted=False) | kw))['retain']
    assert not check(201)
    assert check(200) and check(199)
    assert not check(100, anchored=False)
    assert not check(100, already_promoted=True)


@pytest.mark.parametrize('inliers,total', [(0, 400), (401, 400), (200, 601), (True, 400), (200., 400)])
def test_invalid_population_fails(inliers, total):
    with pytest.raises(SensorContractError):
        overlap_retention(inliers=inliers, reference_features=total,
            anchored=True, already_promoted=False)


def test_original_acceptance_and_bridge_implementations_inherited():
    for name in ('_measure', '_choose', '_candidate', '_remember'):
        assert getattr(OverlapRetentionJointRGBDPose, name) is getattr(CornerSupportJointRGBDPose, name)
