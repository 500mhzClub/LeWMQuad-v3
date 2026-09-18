"""Exercise real optical flow and depth fitting when descriptors are missing."""
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.gapped_camera_plane_tracker_development import GappedCameraPlaneTracker
from lewm.gapped_direct_flow_tracker_development import GappedDirectFlowTracker
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.tests.test_gapped_camera_plane_tracker_development import ingest
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run


@pytest.mark.parametrize('gap', [1, 3, 5])
def test_real_flow_recovers_missing_descriptor_pairs_without_retiming(monkeypatch, gap):
    old = GappedCameraPlaneTracker(); new = GappedDirectFlowTracker()
    items = list(sequence(gap+1))
    for model in (old, new):
        ingest(model, items[0]); run(model, items[0])
        for item in items[1:]: ingest(model, item)
    def missing(self, ref, current, G):
        raise SensorContractError('synthetic missing descriptor matches; real images retained')
    monkeypatch.setattr(DualCameraAnchorPose, '_candidate', missing)
    if gap == 1:
        expected = run(old, items[-1])
    else:
        with pytest.raises(SensorContractError): run(old, items[-1])
    actual = run(new, items[-1])
    assert actual['visual_interval_ns'] == gap*100_000_000
    assert actual['measured_ns'] == items[-1][3]['now_ns']
    assert actual['global_history_reset'] is False
    fallback = new.last_direct_flow_fallback
    assert fallback['accepted'] and any(p['qualified'] for p in fallback['pair_attempts'])
    if gap == 1:
        assert {k:actual[k] for k in expected} == expected
