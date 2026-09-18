"""Same stride-four floor candidates, projecting only required depth pixels."""
import numpy as np
from lewm.sampled_plane_candidates_development import CELL_ROWS, CELL_COLUMNS
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.causal_sensor_state import SensorContractError
from lewm.floor_pose_registration_development import (
    ROWS, COLUMNS, unit, proper, measured_candidates as dense_candidates)

PIXEL_ROWS = np.unique(np.concatenate([ROWS+d for d in (-1, 0, 1)]))
PIXEL_COLUMNS = np.unique(np.concatenate([COLUMNS+d for d in (-1, 0, 1)]))
CORNER_OFFSETS = ((0, 0), (0, 1), (1, 1), (1, 0))
CORNERS = [np.ix_(CELL_ROWS+dr, CELL_COLUMNS+dc) for dr, dc in CORNER_OFFSETS]
SPARSE_CORNERS = [np.ix_(np.searchsorted(PIXEL_ROWS, CELL_ROWS+dr),
    np.searchsorted(PIXEL_COLUMNS, CELL_COLUMNS+dc)) for dr, dc in CORNER_OFFSETS]
U = ((np.arange(640)+.5-320)/FOCAL)[PIXEL_COLUMNS]
W = ((np.arange(480)+.5-240)/FOCAL)[PIXEL_ROWS]


def measured_candidates(depth, valid, body_from_optical, up_body):
    E = np.asarray(body_from_optical, float)
    if (E.shape != (4, 4) or not np.isfinite(E).all()
            or not np.array_equal(E[3], [0., 0., 0., 1.])):
        raise ValueError('finite rigid camera mount required')
    proper(E[:3, :3]); up = unit(up_body)
    T = np.asarray(BODY_FROM_OPTICAL)
    Q = E[:3, :3]@T[:3, :3].T
    reference_up = Q.T@up
    d, v = np.asarray(depth), np.asarray(valid)
    if (d.shape != (480, 640) or v.shape != d.shape or v.dtype != bool
            or not np.isfinite(d).all() or np.any(d[~v] != 0.)
            or np.any((d[v] < .2) | (d[v] > 5.))
            or not np.isfinite(reference_up).all()
            or abs(np.linalg.norm(reference_up)-1) > 1e-6):
        raise SensorContractError('measured depth grid and unit up required')
    z = d[np.ix_(PIXEL_ROWS, PIXEL_COLUMNS)]
    optical = np.stack((z*U[None], z*W[:, None], z), axis=2)
    points = optical@T[:3, :3].T+T[:3, 3]
    a, b, c, e = [points[index] for index in SPARSE_CORNERS]
    good = np.ones(a.shape[:2], bool)
    eps = 64*np.finfo(float).eps
    for index, p in zip(CORNERS, (a, b, c, e), strict=True):
        good &= v[index]
        height = p@reference_up
        if np.any(good & (np.abs(height+.15) <= eps)):
            return dense_candidates(depth, valid, E, up)
        good &= height < -.15
    # Rejected quads cannot become candidates in later tests. Preserve their
    # original mask positions while computing normals only for eligible quads.
    active = good.copy()
    a, b, c, e = [p[active] for p in (a, b, c, e)]
    keep = np.ones(len(a), bool)
    for pair, (left, right) in enumerate(((b, e), (b, c), (c, e))):
        normal = np.cross(left-a, right-a)
        length = np.linalg.norm(normal, axis=1)
        alignment = np.abs(normal@reference_up)
        near = ((np.abs(length-1e-10) <= eps*1e-10)
            | (np.abs(alignment-.97*length) <= eps*length))
        if np.any(keep & near):
            return dense_candidates(depth, valid, E, up)
        keep &= (length > 1e-10) & (alignment >= .97*length)
        if pair == 0:
            error = np.abs(np.sum((c-a)*normal, axis=1))
            if np.any(keep & (np.abs(error-.003*length) <= eps*length)):
                return dense_candidates(depth, valid, E, up)
            keep &= error <= .003*length
    good[active] = keep
    accepted = good.reshape(len(ROWS), 2, len(COLUMNS), 2).all(axis=(1, 3))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            accepted &= v[np.ix_(ROWS+dr, COLUMNS+dc)]
    yy, xx = np.meshgrid(ROWS+.5, COLUMNS+.5, indexing='ij')
    z = d[np.ix_(ROWS, COLUMNS)]
    optical = np.stack((z*(xx-320)/FOCAL, z*(yy-240)/FOCAL, z), axis=-1)
    candidates = optical@E[:3, :3].T+E[:3, 3]
    accepted &= candidates@up < -.15
    return candidates[accepted], accepted
