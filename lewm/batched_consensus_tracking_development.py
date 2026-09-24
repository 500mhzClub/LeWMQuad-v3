"""Local-reference tracker with batched gyro proposal fitting only."""
from lewm import gyro_initial_camera_consensus_development as initial
from lewm import local_feature_depth_consensus_development as local
from lewm.batched_gyro_consensus_development import register as full_register
from lewm.eligible_floor_registration_development import bind
from lewm.development_support_tracker_development import use
from lewm.local_view_revisit_tracking_development import LocalViewRevisitPose, LocalViewRevisitMotion


gyro_core = bind(initial.gyro_core, full_register=full_register)
register = bind(initial.register, gyro_core=gyro_core)
register_views = bind(initial.original_views, original_register=gyro_core)


class _Joint(local._LiftJoint):
    _candidate = use(local._LiftJoint._candidate, register=register)


class _Dual(local._LiftDual, _Joint):
    _candidate = use(local._LiftDual._candidate, register=register)


class _Direct(local._LiftDirect, _Dual):
    _candidate = use(local._LiftDirect._candidate, register=register)


class _Chained(local._LiftChained, _Direct):
    _candidate = use(local._LiftChained._candidate, register=register)


class _Views(local._LiftViews, _Chained):
    _candidate = use(local._LiftViews._candidate, register=register, register_views=register_views)


class BatchedConsensusPose(LocalViewRevisitPose, _Views):
    pass


class BatchedConsensusMotion(LocalViewRevisitMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = BatchedConsensusPose(activation_frame=activation_frame)
