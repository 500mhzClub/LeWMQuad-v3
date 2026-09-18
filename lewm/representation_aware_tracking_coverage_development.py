"""Distinct post-hoc coverage analysis under the existing sensor convention.

The frozen V1 gate and its failed attempt remain unchanged. The physical sensor
rotation helper already bounds norm deviations and normalizes accepted inputs.
This adapter applies that same convention to a private numerical view, reports
the original gate outcome, and checks coverage with two existing algorithms.
It grants no raw-artifact access, sensor admission, rescore or robot authority.
"""
import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from lewm.independent_tracking_coverage_development import measured_coverage
from lewm.independent_tracking_numerical_verification_development import reconstruct_coverage, same


def coverage_with_sensor_rotation_convention(direction, timestamps_s, base_pose_world, base_twist_world, **completion):
    poses = np.asarray(base_pose_world, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or not 1 <= len(poses) <= 22850:
        raise ValueError('bounded nonempty complete pose population required')
    q = poses[:, 3:]
    # Reuse, do not independently relax, the existing sensor acceptance rule.
    for row in q:
        rotation_xyzw(row)
    norms = np.linalg.norm(q, axis=1)
    numerical_view = poses.copy()
    numerical_view[:, 3:] = q / norms[:, None]
    coverage = measured_coverage(direction, timestamps_s, numerical_view, base_twist_world, **completion)
    reconstructed = reconstruct_coverage(dict(timestamp_s=timestamps_s,
        base_pose_world=numerical_view, base_twist_world=base_twist_world), direction=direction, **completion)
    same(coverage, reconstructed, tolerance=1e-6)
    return dict(status='POSTHOC_SENSOR_CONVENTION_COVERAGE_ONLY', coverage=coverage,
        original_frozen_norm_gate_passes=bool(np.allclose(norms, 1., atol=1e-7, rtol=0)),
        original_max_absolute_norm_deviation=float(np.max(np.abs(norms - 1.))),
        acceptance_convention='lewm.physical_execution_development.rotation_xyzw',
        bounded_normalization_applied_to_private_numerical_view=True,
        original_arrays_modified=False, numerical_coverage_cross_check_passed=True,
        original_attempt_passed=False, raw_physics_sensor_audit_completed=False,
        observer_accuracy_evaluated=False, navigation_qualified=False, goal_achieved=False)
