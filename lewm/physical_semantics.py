"""Pure measurements for the independent physical-interface development assay."""
from __future__ import annotations

import math
import numpy as np

from lewm.interface_semantics import InterfaceError, validate_rigid_transform


def world_from_optical(position, forward, up) -> np.ndarray:
    """Proper RDF optical frame: x right, y down, z forward; metres."""
    position, forward, up = (np.asarray(v, dtype=np.float64) for v in (position, forward, up))
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (position, forward, up)):
        raise InterfaceError("camera vectors must be finite triples")
    if np.linalg.norm(forward) < 1e-12:
        raise InterfaceError("camera forward is degenerate")
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-12:
        raise InterfaceError("camera up and forward are collinear")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, :3] = np.stack([right, -up, forward], axis=1)
    transform[:3, 3] = position
    validate_rigid_transform(transform)
    return transform


def project_world(point, transform, *, width=640, height=480, horizontal_fov_deg=78.323):
    """Pinhole prediction in edge-origin pixel coordinates; no renderer calls."""
    validate_rigid_transform(transform)
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise InterfaceError("world point must be a finite triple")
    if width <= 0 or height <= 0 or not 0 < horizontal_fov_deg < 180:
        raise InterfaceError("invalid camera intrinsics")
    local = transform[:3, :3].T @ (point - transform[:3, 3])
    if local[2] <= 0:
        raise InterfaceError("point is not in front of the camera")
    focal = width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
    return np.array([width / 2 + focal * local[0] / local[2],
                     height / 2 + focal * local[1] / local[2]])


def blue_marker_centroid(rgb):
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise InterfaceError("expected HWC RGB uint8")
    values = array.astype(np.float64)
    mask = ((values[..., 2] > 50) & (values[..., 2] > 1.5 * values[..., 0]) &
            (values[..., 2] > 1.5 * values[..., 1]))
    y, x = np.nonzero(mask)
    if len(x) < 4:
        raise InterfaceError("blue calibration marker absent or too small")
    return np.array([x.mean() + 0.5, y.mean() + 0.5])


def evaluate_cases(cases):
    """Fixed development tolerances; never substitute missing evidence with pass."""
    clear, red, green = (cases[name] for name in ("clear", "wall_red", "wall_green"))
    checks = {}
    for name, case in cases.items():
        positions = np.asarray(case["positions_m"], dtype=np.float64)
        checks[f"{name}_complete_finite_trace"] = bool(
            positions.shape == (401, 3) and np.isfinite(positions).all())
        if not checks[f"{name}_complete_finite_trace"]:
            raise InterfaceError("incomplete or nonfinite physical trace")
    checks["clear_free_motion"] = bool(np.linalg.norm(np.array(clear["positions_m"])[-1] - [1.3, 0, 0.5]) < 0.02)
    checks["clear_no_contact"] = clear["first_contact_step"] is None
    checks["visible_wall_changes_pixels"] = bool(np.mean(np.abs(
        np.asarray(clear["rgb"], dtype=float) - np.asarray(red["rgb"], dtype=float))) > 5)
    checks["material_changes_pixels"] = bool(np.mean(np.abs(
        np.asarray(red["rgb"], dtype=float) - np.asarray(green["rgb"], dtype=float))) > 5)
    for name, case in (("wall_red", red), ("wall_green", green)):
        step = case["first_contact_step"]
        checks[f"{name}_physical_contact"] = bool(step is not None and 225 <= step <= 275)
        # Endpoint bound only: collision rebound is allowed. This is not a
        # zero-velocity or commanded-stop qualification.
        checks[f"{name}_bounded_wall_endpoint"] = bool(0.95 < case["positions_m"][-1][0] < 1.03)
    checks["material_invariant_dynamics"] = bool(np.allclose(
        red["positions_m"], green["positions_m"], atol=1e-5, rtol=0)
        and red["first_contact_step"] == green["first_contact_step"])
    checks["camera_projection_zero_yaw"] = clear["projection_error_px"][0] <= 3
    checks["camera_projection_rotated"] = clear["projection_error_px"][1] <= 3
    checks["reset_stops_probe"] = all(case["reset_position_error_m"] < 1e-5 and
                                       case["reset_speed_mps"] < 1e-5 for case in cases.values())
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
