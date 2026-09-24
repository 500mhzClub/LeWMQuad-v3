"""Fixed two-sided camera perturbations and measurement-independent controls.

These are evaluator diagnostics, not a policy mask or boundary certification.
"""
import numpy as np
from lewm.causal_depth_observation_development import FOCAL
from lewm.raster_footprint_visibility_development import projected_boundary_mask, evaluate_footprint

OFFSETS_PIXELS = (-1., -.25, -1. / 256, -1. / 4096, 0., 1. / 4096, 1. / 256, .25, 1.)


def poses(base):
    """Rotate about optical y; offset denotes principal-axis pixel displacement."""
    T = np.asarray(base, float)
    if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0., 0., 0., 1.])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-10, rtol=0)):
        raise ValueError('finite proper optical pose required')
    result = []
    for offset in OFFSETS_PIXELS:
        angle = np.arctan(offset / FOCAL)
        c, s = np.cos(angle), np.sin(angle)
        out = T.copy()
        out[:3, :3] = T[:3, :3] @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        result.append(dict(principal_axis_offset_pixels=offset, world_from_optical=out.tolist()))
    return result


def score_frame(native, boxes, transform):
    """Score every sampled ray, plus a fixed non-boundary +1cm corruption control.

    Select the control from expected geometry alone, not measured residuals.
    Never write the corrupted array back to recorded depth or policy packets.
    """
    ref, boundary = projected_boundary_mask(boxes, transform)
    expected = ref['expected_depth_m']
    stable = ref['surface_interior'] & np.isfinite(expected) & (expected > .2) & (expected < 4.9) & ~boundary
    candidates = np.argwhere(stable)
    if not len(candidates):
        raise ValueError('interior negative-control ray required')
    y, x = candidates[0]
    row, col = int(ref['rows'][y]), int(ref['columns'][x])
    original = evaluate_footprint(native, boxes, transform, render_near_m=.005)
    control = np.array(native, copy=True)
    control[row, col] = expected[y, x] + .01
    changed = evaluate_footprint(control, boxes, transform, render_near_m=.005)
    return dict(footprint=original, interior_negative_control=dict(row=row, column=col,
        prescribed_error_m=.01, rejected=not changed['stable_interior_metric_pass'],
        stable_bad_rays=changed['stable_interior_bad_rays']),
        boundary_pixels_certified=False, policy_filter=False, navigation_qualified=False)
