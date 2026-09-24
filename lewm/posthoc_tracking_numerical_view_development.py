"""Private evaluator view under the existing sensor representation convention.

Independent arithmetic on this view is not independent native-data acquisition.
The original raw-quaternion gate still rejects the original failed recordings.
"""
import numpy as np

from lewm.physical_execution_development import rotation_xyzw


def sensor_convention_view(raw):
    poses = np.asarray(raw['base_pose_world'], dtype=float)
    if (poses.ndim != 2 or poses.shape[1] != 7 or not 1 <= len(poses) <= 22850
            or not np.isfinite(poses).all()):
        raise ValueError('bounded finite nonempty pose population required')
    for quaternion in poses[:, 3:]:
        rotation_xyzw(quaternion)
    norms = np.linalg.norm(poses[:, 3:], axis=1)
    private = poses.copy()
    private[:, 3:] /= norms[:, None]
    private.setflags(write=False)
    return dict(raw, base_pose_world=private), dict(
        original_frozen_norm_gate_passes=bool(np.allclose(norms, 1., atol=1e-7, rtol=0)),
        original_max_absolute_norm_deviation=float(np.max(np.abs(norms - 1.))),
        acceptance_convention='lewm.physical_execution_development.rotation_xyzw',
        private_normalized_pose_view=True, original_arrays_modified=False,
        independent_arithmetic_on_shared_representation=True,
        independent_native_data=False, original_attempt_passed=False)
