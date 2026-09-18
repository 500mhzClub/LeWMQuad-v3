"""Track retained-reference pixels through bounded measured image history.

Every link uses the existing short-interval association gates. The output lifts
the original reference and final pixels directly; it never sums translations,
rotations, or point clouds. Endpoint rigid fitting and continuity are separate.
"""
from types import SimpleNamespace

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.direct_corner_flow_association_development import FLOW_RULES, tracked_points
from lewm.rgbd_correspondence_motion_development import lift

CHAIN_RULES = dict(maximum_intervals=32, sample_interval_ns=100_000_000)


def location(point):
    return tuple(np.rint(np.asarray(point)*2).astype(int))


def chained_points(frames):
    """Accept [(frame_index, measured_ns, feature_frame), ...] in clock order.

    Only the first frame's detected keypoints seed tracks. Lost tracks never
    reappear and no later-frame detector supplies replacement anchor points.
    Arrays in the supplied feature frames are read without modification.
    """
    if not isinstance(frames, (tuple, list)) or not 2 <= len(frames) <= CHAIN_RULES['maximum_intervals']+1:
        raise SensorContractError('bounded nonempty measured image chain required')
    for i, row in enumerate(frames):
        if (not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int
                or type(row[1]) is not int or min(row[:2]) < 0):
            raise SensorContractError('explicit nonnegative frame and measured clock required')
        if i and (row[0] != frames[i-1][0]+1 or
                  row[1]-frames[i-1][1] != CHAIN_RULES['sample_interval_ns']):
            raise SensorContractError('consecutive measured 100ms images required')
        gray = row[2].gray
        if not isinstance(gray, np.ndarray) or gray.shape != (480, 640) or gray.dtype != np.uint8:
            raise SensorContractError('exact finite uint8 chain grayscale required')
    reference = frames[0][2]
    if len(reference.keypoints) > FLOW_RULES['maximum_reference_corners']:
        raise SensorContractError('bounded original reference-corner population required')
    # Each half-pixel-distinct current position retains its original pixel ID.
    origins = {}
    for keypoint in reference.keypoints:
        point = np.asarray(keypoint.pt, np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise SensorContractError('finite original reference corner required')
        origins.setdefault(location(point), point.copy())
    view = reference
    steps = []
    origin_pixels = np.empty((0, 2), np.float32)
    current_pixels = np.empty((0, 2), np.float32)
    for previous, current in zip(frames, frames[1:]):
        values, receipt = tracked_points(view, current[2])
        _, _, before, after = values
        # The matcher preserves source coordinates and deduplicates both ends.
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
    receipt = dict(association='chained_original_corner_lk_photometric_v1',
        reference_frame=frames[0][0], current_frame=frames[-1][0],
        chain_rules=dict(CHAIN_RULES), steps=steps, intervals=len(steps),
        endpoint_depth_pairs=int(valid.sum()), original_reference_pixels_retained=True,
        every_intermediate_depth_pair_required=True, lost_tracks_reintroduced=False,
        endpoint_depth_lifted_directly=True, pose_increments_composed=False,
        original_short_interval_gates_unchanged=True, rigid_geometry_evaluated=False,
        pose_admitted=False, native_state_used=False)
    return result, receipt
