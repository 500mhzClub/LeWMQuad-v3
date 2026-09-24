"""Explicit native OpenGL (right/up/back) to optical (right/down/forward) readback."""
import numpy as np


def check_optical_pose(native_transform, world_from_optical):
    native = np.asarray(native_transform, float)
    optical = np.asarray(world_from_optical, float)
    if native.shape != (4, 4) or optical.shape != (4, 4):
        raise ValueError('two finite 4x4 camera poses required')
    if not np.isfinite(native).all() or not np.isfinite(optical).all():
        raise ValueError('two finite 4x4 camera poses required')
    # Same convention used by Genesis Camera.render_pointcloud.
    converted = native @ np.diag([1., -1., -1., 1.])
    np.testing.assert_allclose(converted, optical, atol=1e-6, rtol=0)
    return converted
