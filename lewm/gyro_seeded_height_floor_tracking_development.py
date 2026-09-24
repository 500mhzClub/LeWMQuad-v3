"""Initialize raw height-cluster selection with this frame's public gyro attitude."""
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.measured_plane_dual_camera_pose_development import MeasuredPlaneDualCameraPose
from lewm.robust_height_floor_candidates_development import PairedHeightCandidates
from lewm.robust_height_floor_tracking_development import (
    RobustHeightFloorPose, RobustHeightFloorMotion, DESCRIPTION)


class GyroSeededCandidates(PairedHeightCandidates):
    def __init__(self, primary, auxiliary, current_up):
        super().__init__(primary,auxiliary)
        self.current_up=np.asarray(current_up,float)

    def __call__(self, depth, valid, mount, up_body):
        # The enclosing original plane fitter still receives its original
        # alignment reference. Only candidate selection uses the current gyro.
        result=super().__call__(depth,valid,mount,self.current_up)
        self.receipt.update(seed_up_source='current_public_gyro',
            seed_up_body=self.current_up.tolist(),
            downstream_alignment_up_body=np.asarray(up_body).tolist())
        return result


class GyroSeededHeightFloorPose(RobustHeightFloorPose):
    def _prepare_plane(self, current, G, now):
        selector=GyroSeededCandidates(current['primary'].depth,current['auxiliary'].depth,
            np.asarray(G).T@self._initial_up)
        receipt=bind(MeasuredPlaneDualCameraPose._prepare_plane,
            measured_candidates=selector)(self,current,G,now)
        receipt.update(DESCRIPTION,image_feature_depth_changed=True,
            candidate_selection_uses_current_public_gyro=True,
            downstream_plane_alignment_reference_unchanged=True)
        if selector.receipt is not None:receipt['floor_candidate_selection']=selector.receipt
        return receipt


class GyroSeededHeightFloorMotion(RobustHeightFloorMotion):
    def __init__(self, *, identity=(0,0,0), activation_frame=0):
        super().__init__(identity=identity,activation_frame=activation_frame)
        self.model=GyroSeededHeightFloorPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(candidate_selection_uses_current_public_gyro=True,
            downstream_plane_alignment_reference_unchanged=True)
