"""Experimental floor candidates from local inverse-depth averages.

Only fully valid 5x5 neighborhoods within the existing depth-lifting surface
jump limit contribute. This does not fill missing depth, alter obstacle sensing
or certify support. The output is locally estimated geometry, not raw pixels.
"""
import cv2
import numpy as np
from lewm.rgbd_correspondence_motion_development import RULES
from lewm.jit_floor_candidates_development import measured_candidates as original_candidates

WINDOW = 5


def local_depth(depth, valid):
    d, v = np.asarray(depth), np.asarray(valid)
    if (d.shape != (480, 640) or v.shape != d.shape or v.dtype != bool
            or not np.isfinite(d).all() or np.any(d[~v] != 0)
            or np.any((d[v] < .2) | (d[v] > 5))):
        raise ValueError('finite original depth and validity required')
    kernel = np.ones((WINDOW, WINDOW), np.uint8)
    supported = cv2.erode(v.astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    lo = cv2.erode(d.astype(np.float64), kernel)
    hi = cv2.dilate(d.astype(np.float64), kernel)
    mean = cv2.boxFilter(d.astype(np.float64), -1, (WINDOW, WINDOW), normalize=True)
    supported &= hi-lo <= RULES['depth_spread_base_m'] + RULES['depth_spread_fraction']*mean
    inverse = np.divide(1., d, out=np.zeros(d.shape, np.float64), where=v)
    average = cv2.boxFilter(inverse, -1, (WINDOW, WINDOW), normalize=True)
    result = np.divide(1., average, out=np.zeros_like(average), where=supported)
    # No invalid input pixel becomes a candidate measurement.
    return result, supported


def measured_candidates(depth, valid, body_from_optical, up_body):
    estimated, supported = local_depth(depth, valid)
    return original_candidates(estimated, supported, body_from_optical, up_body)
