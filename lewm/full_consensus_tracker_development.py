"""Isolated original-cadence tracker with full-consensus proposal early exit."""
from types import FunctionType

from lewm.full_consensus_early_exit_development import register
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorRGBDPose
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose
from lewm.sampled_plane_stop_conditioned_controller_development import (
    SampledPlaneChainedPose, SampledPlaneChainedVisualMotion)


def use_register(method):
    return FunctionType(method.__code__, method.__globals__ | dict(register=register),
        method.__name__, method.__defaults__, method.__closure__)


class _Joint(JointTemporalAnchorRGBDPose):
    _candidate = use_register(JointTemporalAnchorRGBDPose._candidate)


class _Dual(DualCameraAnchorPose, _Joint):
    _candidate = use_register(DualCameraAnchorPose._candidate)


class _Direct(DirectFlowDualCameraAnchorPose, _Dual):
    _candidate = use_register(DirectFlowDualCameraAnchorPose._candidate)


class _Chained(ChainedAnchorDualCameraPose, _Direct):
    _candidate = use_register(ChainedAnchorDualCameraPose._candidate)


class FullConsensusPose(SampledPlaneChainedPose, _Chained):
    pass


class FullConsensusVisualMotion(SampledPlaneChainedVisualMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = FullConsensusPose()
