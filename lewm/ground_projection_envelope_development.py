"""Conditional ray/flat-ground intervals, never a robot-clearance certificate.

The caller declares a spherical cap on the plane normal and an interval on body
height. These are hypotheses, not learned or calibrated confidence regions.
Intervals enclose intersections only if that plane/calibration model is true.
"""
import math

import numpy as np

from lewm.causal_ground_plane_development import URDF_SHA256
from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.rgb_floor_evidence_development import observe_floor

ORIGIN_BODY = np.array([.326, 0., .043])


def cap_dot_bounds(normal, vectors, angle):
    """Exact extrema of n·v over unit n within angle of the nominal normal."""
    normal = np.asarray(normal, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or abs(np.linalg.norm(normal) - 1) > 1e-8
            or vectors.shape[-1:] != (3,) or not np.isfinite(vectors).all()
            or isinstance(angle, bool) or not isinstance(angle, (int, float))
            or not math.isfinite(angle) or not 0 <= angle < math.pi / 2):
        raise ValueError('finite vectors, unit normal and explicit cap angle required')
    length = np.linalg.norm(vectors, axis=-1)
    cosine = np.divide(vectors @ normal, length, out=np.zeros_like(length), where=length > 0)
    theta = np.arccos(np.clip(cosine, -1, 1))
    lower = length * np.cos(np.minimum(math.pi, theta + angle))
    upper = length * np.cos(np.maximum(0, theta - angle))
    return lower, upper


def project_ground_rays(normal, body_height, rays_body, *, height_radius, angle_radius):
    """Conservative independent-ratio bound; correlations can make it loose.

    Rays have body x=1, so the ray parameter is calibrated optical depth. A valid
    interval must be wholly in front of the camera and inside its depth clip.
    A possibly horizontal/upward ray is unknown, never clipped to a safe range.
    """
    rays = np.asarray(rays_body, dtype=float)
    if (rays.shape[-1:] != (3,) or not np.isfinite(rays).all()
            or not np.all(rays[..., 0] == 1)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in (body_height, height_radius))
            or body_height <= 0 or height_radius < 0):
        raise ValueError('finite unit-optical-depth rays and positive height hypothesis required')
    low_dot, high_dot = cap_dot_bounds(normal, rays, angle_radius)
    low_origin, high_origin = cap_dot_bounds(normal, ORIGIN_BODY, angle_radius)
    height_low = body_height - height_radius + float(low_origin)
    height_high = body_height + height_radius + float(high_origin)
    denominator = rays @ np.asarray(normal)
    nominal = np.divide(-(body_height + np.asarray(normal) @ ORIGIN_BODY), denominator,
                        out=np.full(denominator.shape, np.nan), where=denominator < -1e-12)
    nominal_valid = np.isfinite(nominal) & (nominal >= .05) & (nominal <= 200)
    # Divide only where every normal in the cap intersects ground forward.
    forward = (high_dot < -1e-12) & (height_low > 0)
    lower = np.divide(height_low, -low_dot, out=np.full(low_dot.shape, np.nan), where=forward)
    upper = np.divide(height_high, -high_dot, out=np.full(low_dot.shape, np.nan), where=forward)
    valid = forward & (lower >= .05) & (upper <= 200)
    return {'nominal_optical_depth_m': np.where(nominal_valid, nominal, np.nan),
            'nominal_valid': nominal_valid, 'interval_valid': valid,
            'lower_optical_depth_m': np.where(valid, lower, np.nan),
            'upper_optical_depth_m': np.where(valid, upper, np.nan),
            'possibly_nonforward': ~forward, 'clip_uncertain': forward & ~valid,
            'camera_height_interval_m': (height_low, height_high),
            'height_radius_m': height_radius, 'angle_radius_rad': angle_radius,
            'uncertainty_calibrated': False, 'metric_clearance_qualified': False}


def observe_ground_envelope(packet, state, *, now_ns, height_radius, angle_radius, stride=8):
    """Fuse fresh positive RGB floor evidence with a same-packet body hypothesis."""
    now_ns = _ns(now_ns, 'ground envelope clock')
    if (state['decision_ns'] != now_ns or state['robot_geometry_sha256'] != URDF_SHA256
            or state['ground_plane_qualified'] is not False):
        raise SensorContractError('fresh explicit unqualified ground hypothesis required')
    if type(stride) is not int or stride < 1 or 480 % stride or 640 % stride:
        raise ValueError('native image-grid divisor required')
    floor = observe_floor(packet, now_ns=now_ns)
    rows, columns = np.arange(stride // 2, 480, stride), np.arange(stride // 2, 640, stride)
    u, v = np.meshgrid(columns + .5, rows + .5)
    focal = 320 / math.tan(math.radians(78.323) / 2)
    rays = np.stack((np.ones_like(u), -(u - 320) / focal, -(v - 240) / focal), axis=-1)
    positive = floor['floor_evidence_mask'][np.ix_(rows, columns)]
    bottom_connected = positive & floor['valid_columns'][columns][None, :] & (
        rows[:, None] >= floor['first_floor_row'][columns][None, :])
    result = project_ground_rays(state['up_current_body'], state['body_origin_height_m'], rays,
                                 height_radius=height_radius, angle_radius=angle_radius)
    return {**result, 'decision_ns': now_ns, 'rows': rows, 'columns': columns, 'rays_body': rays,
            'positive_floor_pixels': positive, 'bottom_connected_floor_pixels': bottom_connected,
            'observed_interval_valid': result['interval_valid'] & bottom_connected,
            'place_or_exit_identity': None,
            'scope': 'conditional visible floor-ray intervals; unknown surrounding volume, no footprint or traversal guarantee'}
