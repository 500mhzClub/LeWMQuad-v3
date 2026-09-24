"""Per-observer image-link cache; all existing tracker admission is retained."""
from lewm.batched_patch_tracker_development import _BatchedDirect, BatchedPatchVisualMotion, use_batch
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose
from lewm.cached_chain_association_development import CachedChainAssociation
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneChainedPose


class _CachedChained(ChainedAnchorDualCameraPose, _BatchedDirect):
    def _candidate(self, ref, current, G):
        return self._cached_candidate(ref, current, G)


class CachedChainPose(SampledPlaneChainedPose, _CachedChained):
    def __init__(self):
        super().__init__()
        self.chain_association = CachedChainAssociation()
        self._cached_candidate = use_batch(ChainedAnchorDualCameraPose._candidate,
            chained_points=self.chain_association).__get__(self, type(self))

class CachedChainVisualMotion(BatchedPatchVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = CachedChainPose()
