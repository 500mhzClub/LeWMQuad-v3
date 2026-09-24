"""Isolated registration/obstacle consumers of gyro-conditioned candidates."""
from lewm.eligible_floor_registration_development import bind
from lewm.robust_height_floor_tracking_development import (
    RobustHeightFloorRegistration, RobustHeightIndependentObstacles, DESCRIPTION)
from lewm.floor_reacquisition_development import ReacquiringFloorRegistration
from lewm.auxiliary_only_turn_recovery_development import AuxiliaryTurnObstacles
from lewm.gyro_conditioned_partial_floor_candidates_development import GyroConditionedPartialFloorCandidates

DETAILS = DESCRIPTION | dict(weak_extent_candidate_pruning_uses_existing_gyro_prior=True,
    raw_tracker_changed=False, candidate_support_and_residual_limits_unchanged=True)


class _ConditionedRegistration(RobustHeightFloorRegistration):
    observe = bind(RobustHeightFloorRegistration.observe,
        PairedHeightCandidates=GyroConditionedPartialFloorCandidates, DESCRIPTION=DETAILS)


class GyroConditionedReacquiringRegistration(ReacquiringFloorRegistration, _ConditionedRegistration):
    pass


class _ConditionedObstacles(RobustHeightIndependentObstacles):
    _observe = bind(RobustHeightIndependentObstacles._observe,
        PairedHeightCandidates=GyroConditionedPartialFloorCandidates, DESCRIPTION=DETAILS)


class GyroConditionedAuxiliaryObstacles(AuxiliaryTurnObstacles, _ConditionedObstacles):
    pass
