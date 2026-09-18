"""Resolve a height-conflicting image fit through measured plane/image consensus."""
from copy import deepcopy
import numpy as np
from lewm.joint_camera_anchor_tracker_development import (
    JointCameraAnchorPose,JointCameraAnchorVisualMotion,refine_joint_if_supported)
from lewm import measured_plane_dual_camera_pose_development as plane
from lewm.joint_camera_registration_development import project_views
from lewm.joint_rgbd_rigid_pose_development import inliers,SensorContractError
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference
from lewm.conditioned_support_tracker_development import CONDITIONED_RULES,bind
from lewm.rgbd_correspondence_motion_development import cells


def plane_consensus(candidate,reference_plane,current_plane,**kwargs):
    original=candidate['registration']
    names=('reference_inlier_points_body_m','current_inlier_points_body_m',
        'reference_inlier_pixels','current_inlier_pixels')
    a,b,ua,ub=[np.asarray(original[k],float) for k in names]
    joint=original.get('calibrated_body_frame_fit',False)
    split=original['camera_inliers'][0] if joint else None
    if joint:
        check=bind(inliers,project=lambda points:project_views(points,split))
    elif kwargs['camera']=='auxiliary':
        A,offset=body_from_reference()
        def check(a,b,ua,ub,R,t):
            return inliers((a-offset)@A,(b-offset)@A,ua,ub,A.T@R@A,A.T@(t-offset+R@offset))
    else:check=inliers
    mask=np.ones(len(a),dtype=bool);rounds=0
    while True:
        count=int(mask.sum())
        if count<CONDITIONED_RULES['minimum_matches'] or 2*count<=original['lifted_matches']:
            raise SensorContractError('plane/image consensus requires original match count and strict majority')
        if joint and min(int(mask[:split].sum()),int(mask[split:].sum()))<3:
            raise SensorContractError('plane/image consensus needs support from both cameras')
        R,t,_=plane.fit(a[mask],b[mask],reference_plane,current_plane)
        good,_=check(a,b,ua,ub,R,t);use=mask&good;rounds+=1
        if np.array_equal(use,mask):break
        mask=use
    reg=deepcopy(original)
    for name,values in zip(names,(a,b,ua,ub),strict=True):reg[name]=values[mask].tolist()
    reg.update(inliers=int(mask.sum()),inlier_fraction=float(mask.sum()/original['lifted_matches']),
        reference_grid_cells=cells(ua[mask]),current_grid_cells=cells(ub[mask]))
    if joint:
        reg['camera_inliers']=[int(mask[:split].sum()),int(mask[split:].sum())]
        refined=bind(plane.refine,inliers=bind(inliers,
            project=lambda points:project_views(points,reg['camera_inliers'][0])))
        arguments=kwargs|dict(camera='primary')
    else:refined=plane.refine;arguments=kwargs
    # Reuse the existing constrained fit's displacement, gyro, continuity,
    # scatter and per-point image checks on the converged population.
    result=refined(candidate|dict(registration=reg),reference_plane,current_plane,**arguments)
    result['registration']['measured_plane_refinement'].update(
        robust_plane_image_consensus=True,all_original_image_inliers_required=False,
        original_inliers_preserved=bool(mask.all()),retained_consensus_inliers_preserved=True,
        initial_image_inliers=len(a),original_lifted_matches=original['lifted_matches'],
        retained_original_inlier_indices=np.flatnonzero(mask).tolist(),
        rejected_original_inlier_indices=np.flatnonzero(~mask).tolist(),
        plane_consensus_pruning_rounds=rounds,strict_majority_of_original_matches=True,
        absolute_image_residual_and_motion_thresholds_unchanged=True,
        original_height_conflict_resolved_by_new_fit=True)
    return result


def refine_with_plane_consensus(candidate,reference_plane,current_plane,**kwargs):
    try:return refine_joint_if_supported(candidate,reference_plane,current_plane,**kwargs)
    except plane.PlaneImageConflict as error:
        if str(error)!='retained image fit conflicts with measured floor height':raise
        # A failed alternative preserves the original contradiction and stop.
        try:return plane_consensus(candidate,reference_plane,current_plane,**kwargs)
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError,np.linalg.LinAlgError):raise error


class PlaneConsensusPose(JointCameraAnchorPose):
    _refine_candidate=staticmethod(refine_with_plane_consensus)


class PlaneConsensusVisualMotion(JointCameraAnchorVisualMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=PlaneConsensusPose()

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(robust_plane_image_consensus=True,
            all_original_image_inliers_required=False,
            original_height_conflicts_require_new_consistent_fit=True)


def initialize_pose():
    from lewm import two_cm_floor_extent_development as floor_extent
    from lewm import process_mapped_runtime_development as process
    floor_extent.configure()
    import cv2
    import torch
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    process._motion=PlaneConsensusVisualMotion()
