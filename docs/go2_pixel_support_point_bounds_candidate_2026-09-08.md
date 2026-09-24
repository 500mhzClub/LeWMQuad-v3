# Uninstalled public-depth point-bound candidate

lewm/pixel_support_point_bounds_development.py represents a measured RGB-D point
as a body-coordinate box under explicitly supplied pixel radius and optical
depth error. It uses only sample depth/index and fixed public intrinsics/mount,
with outward-rounded float64 interval operations. It changes no pixel and
does not infer free space between or beyond returns. Uncertain extents are not
clipped back to the public valid-depth range.

Eight tests pass in 0.16 s. Exact rational corner calculations verify enclosure
for primary and downward-45-degree mounts, image extremes and depths at both
public range boundaries. Tests also check monotonic expansion and invalid
sample/calibration/error inputs. These verify conditional arithmetic, not an
empirically calibrated sensor model. Sensor error bounds, mount/intrinsic error
and observer pose uncertainty are not established by this helper.

The helper is not installed in a controller or index. Future use must carry
the full boxes through persistent memory and collision queries, retain unknown
returns, and account for pose uncertainty separately. Applying only an evaluator
boundary exception while continuing to treat policy points as exact would leave
the measurement contract inconsistent. No current native source was modified.
