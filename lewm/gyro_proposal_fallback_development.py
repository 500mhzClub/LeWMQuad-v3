"""Try gyro-conditioned single-camera proposals after a joint-fit rejection.

Existing successful fits are returned unchanged. The alternate fit preserves
the match population, strict majority, conditioning, residual and image gates.
Subsequent plane, gyro-refit and temporal checks remain in the original chain.
"""
from lewm.causal_sensor_state import SensorContractError
from lewm.conditioned_support_tracker_development import (
    register as original_register, use, _Joint, _Dual, _Direct, _Chained)
from lewm.stable_gyro_reference_development import (
    CompiledFloorStableGyroReferencePose, CompiledFloorStableGyroReferenceMotion)


def register(*args, **kwargs):
    try:
        return original_register(*args, **kwargs)
    except SensorContractError as error:
        if kwargs.get('mode') != 'joint':
            raise
        original_failure = str(error)
    R, t, mask, receipt = original_register(*args, **(kwargs | dict(mode='gyro')))
    return R, t, mask, receipt | dict(
        gyro_conditioned_proposal_fallback=True,
        initial_joint_fit_failure=original_failure,
        initial_joint_fit_required=False, geometric_thresholds_changed=False)


class _ProposalJoint(_Joint):
    _candidate = use(_Joint._candidate, register=register)


class _ProposalDual(_Dual, _ProposalJoint):
    _candidate = use(_Dual._candidate, register=register)


class _ProposalDirect(_Direct, _ProposalDual):
    _candidate = use(_Direct._candidate, register=register)


class _ProposalChained(_Chained, _ProposalDirect):
    _candidate = use(_Chained._candidate, register=register)


class GyroProposalFallbackPose(CompiledFloorStableGyroReferencePose, _ProposalChained):
    pass


class GyroProposalFallbackMotion(CompiledFloorStableGyroReferenceMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = GyroProposalFallbackPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            gyro_conditioned_single_camera_proposal_fallback_enabled=True,
            successful_initial_joint_fits_preserved=True,
            pooled_camera_initial_fit_unchanged=True,
            floor_registration_changed=False, absolute_geometric_thresholds_changed=False)
