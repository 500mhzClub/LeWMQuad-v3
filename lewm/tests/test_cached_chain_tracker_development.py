import json

from lewm.batched_patch_tracker_development import BatchedPatchPose
from lewm.cached_chain_tracker_development import CachedChainPose
from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence, run


def test_measured_plane_refinement_still_wraps_cached_chain_candidate():
    original = BatchedPatchPose(); candidate = CachedChainPose()
    for item in sequence(3):
        assert json.loads(json.dumps(run(candidate, item))) == json.loads(json.dumps(run(original, item)))
