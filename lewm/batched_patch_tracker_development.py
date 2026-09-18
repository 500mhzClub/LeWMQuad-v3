"""Use batched photometry in direct and retained-chain association only."""
from types import FunctionType

from lewm.eligible_floor_registration_development import bind
from lewm.batched_patch_agreement_development import tracked_points
from lewm.chained_flow_memo_development import tracked_points as reusable_chain_link
from lewm.chained_corner_flow_association_development import chained_points as original_chain
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.full_consensus_early_exit_development import register
from lewm.full_consensus_tracker_development import _Dual, FullConsensusVisualMotion
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneChainedPose

chained_points = bind(original_chain, tracked_points=reusable_chain_link)


def use_batch(method, **replacements):
    return FunctionType(method.__code__, method.__globals__ | dict(register=register) | replacements,
        method.__name__, method.__defaults__, method.__closure__)


class _BatchedDirect(DirectFlowDualCameraAnchorPose, _Dual):
    _candidate = use_batch(DirectFlowDualCameraAnchorPose._candidate, tracked_points=tracked_points)


class _BatchedChained(ChainedAnchorDualCameraPose, _BatchedDirect):
    _candidate = use_batch(ChainedAnchorDualCameraPose._candidate, chained_points=chained_points)


class BatchedPatchPose(SampledPlaneChainedPose, _BatchedChained):
    pass


class BatchedPatchVisualMotion(FullConsensusVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = BatchedPatchPose()
