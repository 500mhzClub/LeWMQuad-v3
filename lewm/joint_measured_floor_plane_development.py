"""A common measured plane: observability belongs to the combined camera data.

All contributing points retain the unchanged 3 mm coherence gate. A narrow
camera patch contributes its measurements without needing to identify a plane
on its own. This is a static-floor hypothesis, not a support certificate.
"""
import numpy as np
from lewm.floor_pose_registration_development import unit

CAMERAS = ('primary', 'auxiliary')
MINIMUM_SECOND_EXTENT_M = .05


def compose_plane(statistics, up_body):
    up = unit(up_body)
    if type(statistics) is not list or [s['camera'] for s in statistics] != list(CAMERAS):
        raise ValueError('two ordered current camera statistics required')
    populated = []
    for s in statistics:
        count = s['count']
        if type(count) is not int or not 0 <= count <= 19200:
            raise ValueError('bounded integer camera candidate count required')
        if count == 0:
            if s['mean_body_m'] is not None or s['covariance_body_m2'] is not None:
                raise ValueError('empty camera cannot invent geometric moments')
            continue
        mean = np.asarray(s['mean_body_m'], float); covariance = np.asarray(s['covariance_body_m2'], float)
        if (mean.shape != (3,) or covariance.shape != (3, 3)
                or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)
                or np.linalg.eigvalsh(covariance)[0] < -1e-10):
            raise ValueError('finite symmetric nonnegative measured moments required')
        populated.append((count, mean, covariance))
    total = sum(s['count'] for s in statistics)
    receipt = dict(available=False, candidate_count=total, minimum_candidates=100,
        maximum_allowed_residual_m=.003, minimum_second_eigenvalue_m2=MINIMUM_SECOND_EXTENT_M**2,
        minimum_up_alignment=.97, absolute_initial_height_band_used=False,
        candidates_trimmed=False, floor_identity_certified=False,
        independent_camera_plane_rank_required=False, per_point_equal_weight=True)
    if total < 100: return receipt | dict(reason='insufficient_combined_measured_candidates')
    mean = sum(count*center for count, center, _ in populated)/total
    covariance = sum(count*(cov+np.outer(center-mean, center-mean))
        for count, center, cov in populated)/total
    values, vectors = np.linalg.eigh(covariance); normal = vectors[:, 0]
    if normal@up < 0: normal = -normal
    receipt.update(normal_body=normal.tolist(), offset_body_m=-float(normal@mean),
        covariance_eigenvalues_m2=values.tolist())
    if values[1] < MINIMUM_SECOND_EXTENT_M**2: return receipt | dict(reason='insufficient_combined_two_axis_extent')
    if normal@up < .97: return receipt | dict(reason='combined_plane_up_disagreement')
    return receipt | dict(available=True, reason='combined_measured_plane_requires_all_point_coherence')


def fit_joint_plane(primary_points, auxiliary_points, up_body):
    statistics = []; clouds = []
    for camera, points in zip(CAMERAS, (primary_points, auxiliary_points), strict=True):
        points = np.asarray(points, float)
        if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) > 19200
                or not np.isfinite(points).all()):
            raise ValueError('bounded finite current measured camera points required')
        mean = points.mean(0) if len(points) else None
        covariance = (points-mean).T@(points-mean)/len(points) if len(points) else None
        statistics.append(dict(camera=camera, count=len(points), mean_body_m=None if mean is None else mean.tolist(),
            covariance_body_m2=None if covariance is None else covariance.tolist()))
        clouds.append(points)
    receipt = compose_plane(statistics, up_body)
    receipt.update(camera_statistics=statistics, up_body=unit(up_body).tolist(), camera_residuals=None)
    if not receipt['available']: return receipt
    normal = np.asarray(receipt['normal_body']); offset = receipt['offset_body_m']; residuals = []
    for camera, points in zip(CAMERAS, clouds, strict=True):
        errors = points@normal+offset
        residuals.append(dict(camera=camera, count=len(points),
            maximum_residual_m=float(np.abs(errors).max()) if len(points) else None,
            rms_residual_m=float(np.sqrt(np.mean(errors**2))) if len(points) else None))
    receipt['camera_residuals'] = residuals
    if any(r['count'] and r['maximum_residual_m'] > .003 for r in residuals):
        return receipt | dict(available=False, reason='combined_points_not_one_coherent_plane')
    return receipt


def validate_joint_plane(receipt, up_body):
    """Composition/admission check; raw replay owns reconstruction of the points."""
    up = unit(up_body)
    if not np.array_equal(unit(receipt['up_body']), up):
        raise ValueError('same raw-visual initial-gravity direction required')
    expected = compose_plane(receipt['camera_statistics'], up)
    if not expected['available'] or any(receipt[k] != v for k, v in expected.items()):
        raise ValueError('admitted combined measured moments must reconstruct exactly')
    rows = receipt['camera_residuals']
    if type(rows) is not list or [r['camera'] for r in rows] != list(CAMERAS):
        raise ValueError('all contributing cameras require residual accounting')
    for r, s in zip(rows, receipt['camera_statistics'], strict=True):
        if r['count'] != s['count']: raise ValueError('every measured candidate requires a residual')
        if not s['count']:
            if r['maximum_residual_m'] is not None or r['rms_residual_m'] is not None:
                raise ValueError('unobserved camera has no residual')
            continue
        maximum, rms = r['maximum_residual_m'], r['rms_residual_m']
        if not np.isfinite([maximum, rms]).all() or not 0 <= rms <= maximum <= .003:
            raise ValueError('all measured points must retain the original 3 mm coherence gate')
    return receipt
