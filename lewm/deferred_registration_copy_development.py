"""Avoid copying registration evidence twice during one gyro consensus refit."""
from lewm.eligible_floor_registration_development import bind
from lewm.development_support_tracker_development import use
from lewm.gyro_conditioned_pair_pose_development import GyroConditionedPairPose
from lewm.gyro_consensus_pair_pose_development import GyroConsensusPairPose, consensus_refit
from lewm.batched_consensus_tracking_development import BatchedConsensusPose, BatchedConsensusMotion

# consensus_refit replaces top-level entries only. Its final refit still makes
# the complete deep copy before returning or adding nested output evidence.
# Thus the temporary dictionary can share untouched values with its input.
deferred_consensus_refit = bind(consensus_refit, deepcopy=dict.copy)
_candidate = use(GyroConditionedPairPose._candidate, refit=deferred_consensus_refit)


class _DeferredConsensus(GyroConsensusPairPose):
    _candidate = use(GyroConsensusPairPose._candidate, _candidate_with_consensus=_candidate)


class DeferredRegistrationCopyPose(BatchedConsensusPose, _DeferredConsensus):
    pass


class DeferredRegistrationCopyMotion(BatchedConsensusMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = DeferredRegistrationCopyPose(activation_frame=activation_frame)
