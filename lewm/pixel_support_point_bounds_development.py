"""Public-depth point boxes conditional on supplied pixel/depth error bounds.

Directed float64 interval arithmetic encloses backprojection under the supplied
fixed intrinsics and mount. This does not calibrate those inputs, establish pose
uncertainty or certify space between measurements as free.
"""
import numpy as np
from lewm.causal_depth_observation_development import FOCAL


def down(value): return np.nextafter(value, -np.inf)
def up(value): return np.nextafter(value, np.inf)


def product(a, b):
    terms = np.stack((a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    return down(terms.min(axis=0)), up(terms.max(axis=0))


def point_bounds(depth_m, rows, columns, body_from_optical, *, pixel_radius, depth_error_m):
    depth, rr, cc, T = np.asarray(depth_m), np.asarray(rows), np.asarray(columns), np.asarray(body_from_optical, float)
    if (depth.ndim != 1 or not 1 <= len(depth) <= 307200 or depth.dtype.kind != 'f'
            or not np.isfinite(depth).all() or np.any(depth < .2) or np.any(depth > 5.)
            or rr.shape != depth.shape or cc.shape != depth.shape or rr.dtype.kind not in 'iu' or cc.dtype.kind not in 'iu'
            or np.any(rr < 0) or np.any(rr >= 480) or np.any(cc < 0) or np.any(cc >= 640)
            or not np.isfinite(pixel_radius) or not 0 <= pixel_radius <= .5
            or not np.isfinite(depth_error_m) or not 0 <= depth_error_m <= .001
            or T.shape != (4, 4) or not np.isfinite(T).all() or not np.array_equal(T[3], [0., 0., 0., 1.])
            or not np.allclose(T[:3, :3].T@T[:3, :3], np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-10, rtol=0)):
        raise ValueError('valid public depth samples, fixed proper mount and explicit bounded uncertainty required')
    d = depth.astype(float)
    z = down(d-depth_error_m), up(d+depth_error_m)
    optical = []
    for coordinate, principal in ((cc, 320.), (rr, 240.)):
        center = coordinate.astype(float)+.5  # These bounded half-integers are exact.
        lo = down(down(center-pixel_radius)-principal)
        hi = up(up(center+pixel_radius)-principal)
        ray = down(lo/FOCAL), up(hi/FOCAL)
        optical.append(product(ray, z))
    optical.append(z)
    optical_low = np.stack([x[0] for x in optical], axis=1)
    optical_high = np.stack([x[1] for x in optical], axis=1)
    lower = np.empty((len(depth), 3)); upper = np.empty_like(lower)
    for i in range(3):
        lo = np.full(len(depth), T[i, 3]); hi = lo.copy()
        for j in range(3):
            term = product((optical_low[:, j], optical_high[:, j]), (T[i, j], T[i, j]))
            lo, hi = down(lo+term[0]), up(hi+term[1])
        lower[:, i], upper[:, i] = lo, hi
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('finite representable point enclosure required')
    return dict(lower_body_m=lower, upper_body_m=upper,
        optical_depth_lower_m=z[0], optical_depth_upper_m=z[1],
        supplied_pixel_radius=float(pixel_radius), supplied_depth_error_m=float(depth_error_m),
        interval_arithmetic='float64 outward rounding after every elementary interval operation',
        conditional_on_fixed_supplied_intrinsics_and_mount=True,
        supplied_sensor_error_bound_validated=False, pose_error_included=False,
        public_valid_range_used_to_clip_uncertainty=False,
        native_state_used=False, sensor_pixels_changed=False,
        unobserved_space_certified=False, navigation_qualified=False)
