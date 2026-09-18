"""Explicit support sensitivity experiment: five image cells and strict majority."""
from types import FunctionType
from lewm.eligible_floor_registration_development import bind
from lewm.full_consensus_early_exit_development import register as original_register,RULES
from lewm.support_aware_rgbd_pose_development import support_near_limit
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorRGBDPose
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.direct_flow_dual_camera_pose_development import DirectFlowDualCameraAnchorPose
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose
from lewm.sampled_plane_stop_conditioned_controller_development import SampledPlaneChainedPose
from lewm.feature_budget_100_tracker_development import FeatureFrame100,FeatureBudget100VisualMotion
from lewm.batched_patch_tracker_development import tracked_points,chained_points
from lewm.optional_plane_refinement_development import refine_if_supported
from lewm.causal_sensor_state import SensorContractError

DEVELOPMENT_RULES=RULES|dict(minimum_grid_cells=5,minimum_inlier_fraction=.5)
_register=bind(original_register,RULES=DEVELOPMENT_RULES)
_support=bind(support_near_limit,RULES=DEVELOPMENT_RULES,SUPPORT_MARGIN_CELLS=2)


def register(*args,**kwargs):
    R,t,mask,receipt=_register(*args,**kwargs)
    if 2*int(mask.sum())<=len(mask):raise SensorContractError('strict inlier majority required')
    return R,t,mask,receipt|dict(development_support_rule=True,minimum_image_grid_cells=5,
        strict_inlier_majority_required=True,original_six_cell_sixty_percent_rule_used=False)


def use(method,**replacements):
    copied=FunctionType(method.__code__,method.__globals__|dict(register=register)|replacements,
        method.__name__,method.__defaults__,method.__closure__)
    copied.__kwdefaults__=method.__kwdefaults__
    return copied


class _Joint(JointTemporalAnchorRGBDPose):
    _candidate=use(JointTemporalAnchorRGBDPose._candidate)


class _Dual(DualCameraAnchorPose,_Joint):
    _candidate=use(DualCameraAnchorPose._candidate)
    observe=use(DualCameraAnchorPose.observe,CornerSupportFeatureFrame=FeatureFrame100,support_near_limit=_support)


class _Direct(DirectFlowDualCameraAnchorPose,_Dual):
    _candidate=use(DirectFlowDualCameraAnchorPose._candidate,tracked_points=tracked_points)


class _Chained(ChainedAnchorDualCameraPose,_Direct):
    _candidate=use(ChainedAnchorDualCameraPose._candidate,chained_points=chained_points)


class DevelopmentSupportPose(SampledPlaneChainedPose,_Chained):
    _refine_candidate=staticmethod(refine_if_supported)


class DevelopmentSupportVisualMotion(FeatureBudget100VisualMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=DevelopmentSupportPose()

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(
            development_support_rule=True,minimum_image_grid_cells=5,
            strict_inlier_majority_required=True,spatial_and_inlier_fraction_thresholds_changed=True,
            geometric_and_temporal_gate_values_unchanged=False,
            remaining_reprojection_conditioning_motion_and_temporal_thresholds_unchanged=True,
            accepted_support_promotion_threshold_cells=7)


def initialize_development_support_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    process._motion=DevelopmentSupportVisualMotion()
