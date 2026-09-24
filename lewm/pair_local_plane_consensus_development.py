"""Reject a conflicting image pair while still testing other measured references."""
from copy import deepcopy
from lewm.plane_consensus_tracker_development import (
    PlaneConsensusPose,PlaneConsensusVisualMotion,refine_with_plane_consensus)
from lewm.measured_plane_dual_camera_pose_development import PlaneImageConflict
from lewm.causal_sensor_state import SensorContractError


class PairLocalPlaneConsensusPose(PlaneConsensusPose):
    def __init__(self):
        super().__init__();self.rejected_plane_pairs=[]

    def observe(self,*args,**kwargs):
        self.rejected_plane_pairs=[]
        return super().observe(*args,**kwargs)

    def _refine_candidate(self,candidate,reference_plane,current_plane,**kwargs):
        try:return refine_with_plane_consensus(candidate,reference_plane,current_plane,**kwargs)
        except PlaneImageConflict as error:
            if str(error)!='retained image fit conflicts with measured floor height':raise
            # The inconsistent pose is never admitted. A different retained
            # reference must independently pass image, plane, gyro and motion
            # checks. If none does, ordinary tracking failure still stops motion.
            self.rejected_plane_pairs.append(dict(frame=self.frame,
                reference_frame=candidate['reference'].frame,camera=kwargs['camera'],
                reason=str(error),inliers=candidate['registration']['inliers'],
                lifted_matches=candidate['registration']['lifted_matches'],
                conflicting_pose_admitted=False))
            raise SensorContractError('image/floor pair rejected; other measured references remain eligible') from error


class PairLocalPlaneConsensusVisualMotion(PlaneConsensusVisualMotion):
    def __init__(self,*,identity=(0,0,0)):
        super().__init__(identity=identity);self.model=PairLocalPlaneConsensusPose()

    def snapshot(self,*,now_ns):
        return super().snapshot(now_ns=now_ns)|dict(
            floor_height_conflict_rejection_scope='image_reference_pair',
            rejected_plane_pairs=deepcopy(self.model.rejected_plane_pairs),
            conflicting_image_pose_admitted=False,original_pair_acceptance_checks_unchanged=True)


def initialize_pose():
    from lewm.plane_consensus_tracker_development import initialize_pose as original
    from lewm import process_mapped_runtime_development as process
    original();process._motion=PairLocalPlaneConsensusVisualMotion()
