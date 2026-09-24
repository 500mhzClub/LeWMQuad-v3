"""Combine the two separately replay-tested tracking cost reductions."""
from lewm.batched_consensus_tracking_development import BatchedConsensusMotion
from lewm.cached_floor_moments_development import CachedFloorMomentsPose
from lewm.deferred_registration_copy_development import DeferredRegistrationCopyPose


class CachedMomentsDeferredCopyPose(CachedFloorMomentsPose, DeferredRegistrationCopyPose):
    pass


class CachedMomentsDeferredCopyMotion(BatchedConsensusMotion):
    def __init__(self, *, identity=(0,0,0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = CachedMomentsDeferredCopyPose(activation_frame=activation_frame)
