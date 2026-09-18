"""Ablate image-bin counting while retaining measured 3-D pose conditioning."""
from lewm.development_support_tracker_development import (
    use,original_register,RULES,support_near_limit,JointTemporalAnchorRGBDPose,
    DualCameraAnchorPose,DirectFlowDualCameraAnchorPose,ChainedAnchorDualCameraPose,
    SampledPlaneChainedPose,FeatureFrame100,FeatureBudget100VisualMotion,
    tracked_points,chained_points,refine_if_supported,SensorContractError,bind)

CONDITIONED_RULES=RULES|dict(minimum_grid_cells=0,minimum_inlier_fraction=.5)
_register=bind(original_register,RULES=CONDITIONED_RULES)
_support=bind(support_near_limit,RULES=CONDITIONED_RULES,SUPPORT_MARGIN_CELLS=7)


def register(*args,**kwargs):
    R,t,mask,receipt=_register(*args,**kwargs)
    if 2*int(mask.sum())<=len(mask):raise SensorContractError('strict inlier majority required')
    return R,t,mask,receipt|dict(image_bin_count_gate_used=False,strict_inlier_majority_required=True,
        original_measured_3d_conditioning_required=True)


class _Joint(JointTemporalAnchorRGBDPose):
    _candidate=use(JointTemporalAnchorRGBDPose._candidate,register=register)


class _Dual(DualCameraAnchorPose,_Joint):
    _candidate=use(DualCameraAnchorPose._candidate,register=register)
    observe=use(DualCameraAnchorPose.observe,CornerSupportFeatureFrame=FeatureFrame100,support_near_limit=_support)


class _Direct(DirectFlowDualCameraAnchorPose,_Dual):
    _candidate=use(DirectFlowDualCameraAnchorPose._candidate,register=register,tracked_points=tracked_points)


class _Chained(ChainedAnchorDualCameraPose,_Direct):
    _candidate=use(ChainedAnchorDualCameraPose._candidate,register=register,chained_points=chained_points)


class ConditionedSupportPose(SampledPlaneChainedPose,_Chained):
    _refine_candidate=staticmethod(refine_if_supported)


class ConditionedSupportVisualMotion(FeatureBudget100VisualMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=ConditionedSupportPose()

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(image_bin_count_gate_used=False,
            strict_inlier_majority_required=True,original_measured_3d_conditioning_required=True,
            spatial_and_inlier_fraction_thresholds_changed=True,
            geometric_and_temporal_gate_values_unchanged=False,
            remaining_reprojection_conditioning_motion_and_temporal_thresholds_unchanged=True,
            accepted_support_promotion_threshold_cells=7)


def initialize_conditioned_support_pose():
    import cv2
    import torch
    from lewm import process_mapped_runtime_development as process
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    process._motion=ConditionedSupportVisualMotion()
