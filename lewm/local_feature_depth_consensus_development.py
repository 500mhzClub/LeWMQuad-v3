"""Locally estimate correspondence endpoint depths without rewriting packets.

Detection and intermediate chained-flow checks retain original sensor depth.
Descriptor, direct-flow and chained endpoints use the fixed 5x5 estimator.
Floor estimation still reads original packets and therefore filters only once.
"""
from contextvars import ContextVar
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.development_support_tracker_development import use
from lewm.local_inverse_depth_floor_development import local_depth
from lewm.local_inverse_depth_floor_tracking_development import LocalInverseDepthFloorRegistration
from lewm.rgbd_correspondence_motion_development import lift as original_lift
from lewm.keyframe_rgbd_pose_development import matched_points as original_matched
from lewm.batched_patch_agreement_development import tracked_points as original_direct
from lewm.batched_patch_tracker_development import chained_points as original_chain
from lewm.gyro_initial_camera_consensus_development import (
    register, register_views, _InitialJoint, _InitialDual, _InitialDirect,
    _InitialChained, _InitialViews, GyroInitialCameraConsensusPose,
    GyroInitialCameraConsensusMotion)

_observation_cache = ContextVar('local_feature_depth_observation_cache', default=None)


def lift(depth, pixels):
    cache = _observation_cache.get()
    arrays = (depth['depth_m'], depth['valid'])
    key = tuple(id(a) for a in arrays)
    entry = None if cache is None else cache.get(key)
    if entry is None:
        values, valid = local_depth(*arrays)
        estimated = dict(depth_m=values.astype(np.float32), valid=valid)
        # Retain the input objects until the synchronous observation ends, so
        # IDs cannot be reused. No derived depth is cached across observations.
        entry = (arrays, estimated)
        if cache is not None:
            cache[key] = entry
    return original_lift(entry[1], pixels)


matched_points = bind(original_matched, lift=lift)
tracked_points = bind(original_direct, lift=lift)
# Preserve original intermediate-link validity/photometry and its exact memo;
# only the two final endpoint lifts receive locally estimated depths.
chained_points = bind(original_chain, lift=lift)


class _LiftJoint(_InitialJoint):
    _candidate = use(_InitialJoint._candidate, register=register, matched_points=matched_points)


class _LiftDual(_InitialDual, _LiftJoint):
    _candidate = use(_InitialDual._candidate, register=register, matched_points=matched_points)


class _LiftDirect(_InitialDirect, _LiftDual):
    _candidate = use(_InitialDirect._candidate, register=register, tracked_points=tracked_points)


class _LiftChained(_InitialChained, _LiftDirect):
    _candidate = use(_InitialChained._candidate, register=register, chained_points=chained_points)


class _LiftViews(_InitialViews, _LiftChained):
    _candidate = use(_InitialViews._candidate, register=register, register_views=register_views,
        matched_points=matched_points, chained_points=chained_points)


class LocalFeatureDepthConsensusPose(GyroInitialCameraConsensusPose, _LiftViews):
    def _prepare_plane(self, *args, **kwargs):
        receipt = super()._prepare_plane(*args, **kwargs)
        receipt['image_feature_depth_changed'] = True
        return receipt

    def observe(self, *args, **kwargs):
        token = _observation_cache.set({})
        try:
            return super().observe(*args, **kwargs)
        finally:
            _observation_cache.reset(token)


class LocalFeatureDepthRegistration(LocalInverseDepthFloorRegistration):
    def observe(self, *args, **kwargs):
        return super().observe(*args, **kwargs) | dict(image_feature_depth_changed=True)


class LocalFeatureDepthConsensusMotion(GyroInitialCameraConsensusMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = LocalFeatureDepthConsensusPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            image_feature_depth_changed=True,
            image_correspondence_depth_source='local_inverse_depth_5x5',
            image_correspondence_points_are_raw_pixel_depth=False,
            original_camera_packets_preserved=True,
            feature_detection_depth_changed=False,
            intermediate_chained_flow_depth_changed=False,
            floor_depth_filtered_twice=False)
