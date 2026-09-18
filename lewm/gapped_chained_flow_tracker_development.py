"""Chain original corner identities through actual, bounded camera timestamps.

Only acquired images participate. Pairwise pixel/depth checks and final rigid
pose admission are unchanged. The original 32-link and 3.2-second limits both
apply, so sparse observations cannot silently lengthen the reference lifetime.
"""
from types import FunctionType, SimpleNamespace

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.chained_corner_flow_association_development import CHAIN_RULES, location
from lewm.direct_corner_flow_association_development import FLOW_RULES, tracked_points
from lewm.rgbd_correspondence_motion_development import lift
from lewm.chained_anchor_dual_camera_pose_development import ChainedAnchorDualCameraPose
from lewm.gapped_camera_plane_tracker_development import MAX_CAMERA_INTERVAL_NS
from lewm.gapped_direct_flow_tracker_development import GappedDirectFlowTracker

MAX_CHAIN_NS = CHAIN_RULES['maximum_intervals']*CHAIN_RULES['sample_interval_ns']


def chained_points(frames):
    if not isinstance(frames, (tuple, list)) or not 2 <= len(frames) <= CHAIN_RULES['maximum_intervals']+1:
        raise SensorContractError('bounded nonempty measured image chain required')
    for i, row in enumerate(frames):
        if (not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int
                or type(row[1]) is not int or min(row[:2]) < 0):
            raise SensorContractError('explicit nonnegative frame and measured clock required')
        if i:
            elapsed = row[1]-frames[i-1][1]
            if (row[0] != frames[i-1][0]+1 or elapsed % 100_000_000
                    or not 100_000_000 <= elapsed <= MAX_CAMERA_INTERVAL_NS):
                raise SensorContractError('ordered processed images with actual 100–500 ms gaps required')
        gray = row[2].gray
        if not isinstance(gray, np.ndarray) or gray.shape != (480, 640) or gray.dtype != np.uint8:
            raise SensorContractError('exact finite uint8 chain grayscale required')
    if frames[-1][1]-frames[0][1] > MAX_CHAIN_NS:
        raise SensorContractError('original 3.2-second measured image-chain lifetime exceeded')
    reference = frames[0][2]
    if len(reference.keypoints) > FLOW_RULES['maximum_reference_corners']:
        raise SensorContractError('bounded original reference-corner population required')
    origins = {}
    for keypoint in reference.keypoints:
        point = np.asarray(keypoint.pt, np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise SensorContractError('finite original reference corner required')
        origins.setdefault(location(point), point.copy())
    view = reference; steps = []
    origin_pixels = np.empty((0, 2), np.float32)
    current_pixels = np.empty((0, 2), np.float32)
    for previous, current in zip(frames, frames[1:]):
        values, receipt = tracked_points(view, current[2])
        _, _, before, after = values
        try:
            origin_pixels = np.asarray([origins[location(p)] for p in before], np.float32).reshape(-1, 2)
        except KeyError as error:
            raise SensorContractError('each surviving track requires its original reference pixel') from error
        current_pixels = np.asarray(after, np.float32).reshape(-1, 2)
        origins = {location(p): origin.copy() for p, origin in zip(current_pixels, origin_pixels, strict=True)}
        if len(origins) != len(current_pixels):
            raise SensorContractError('distinct surviving endpoint identities required')
        steps.append(dict(reference_frame=previous[0], current_frame=current[0],
            reference_measured_ns=previous[1], current_measured_ns=current[1], association=receipt))
        view = SimpleNamespace(gray=current[2].gray, depth=current[2].depth,
            keypoints=[cv2.KeyPoint(float(p[0]), float(p[1]), 8.) for p in current_pixels])
    a, va = lift(reference.depth, origin_pixels)
    b, vb = lift(frames[-1][2].depth, current_pixels)
    valid = va & vb & np.isfinite(a).all(1) & np.isfinite(b).all(1)
    result = (a[valid], b[valid], origin_pixels[valid], current_pixels[valid])
    receipt = dict(association='gapped_chained_original_corner_lk_photometric_v1',
        reference_frame=frames[0][0], current_frame=frames[-1][0],
        maximum_intervals=CHAIN_RULES['maximum_intervals'], maximum_elapsed_ns=MAX_CHAIN_NS,
        maximum_camera_interval_ns=MAX_CAMERA_INTERVAL_NS,
        elapsed_ns=frames[-1][1]-frames[0][1], steps=steps, intervals=len(steps),
        endpoint_depth_pairs=int(valid.sum()), original_reference_pixels_retained=True,
        every_intermediate_depth_pair_required=True, lost_tracks_reintroduced=False,
        endpoint_depth_lifted_directly=True, pose_increments_composed=False,
        original_pixel_and_depth_gates_unchanged=True, image_timestamps_changed=False,
        intermediate_images_synthesized=False, rigid_geometry_evaluated=False,
        pose_admitted=False, native_state_used=False)
    return result, receipt


_original_candidate = ChainedAnchorDualCameraPose._candidate
_candidate = FunctionType(_original_candidate.__code__,
    _original_candidate.__globals__ | dict(chained_points=chained_points),
    _original_candidate.__name__, _original_candidate.__defaults__,
    _original_candidate.__closure__)


class _ActualIntervalChainedFlow(ChainedAnchorDualCameraPose):
    def _candidate(self, ref, current, G):
        return _candidate(self, ref, current, G)


class GappedChainedFlowTracker(GappedDirectFlowTracker, _ActualIntervalChainedFlow):
    def observe(self, *args, **kwargs):
        result = super().observe(*args, **kwargs)
        return result | dict(chained_flow_uses_actual_visual_intervals=True,
            chained_flow_maximum_elapsed_ns=MAX_CHAIN_NS,
            chained_flow_pixel_depth_and_geometry_limits_unchanged=True)
