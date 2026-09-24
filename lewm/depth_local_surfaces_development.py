"""Observed body-height wall cross-sections; no map, pose or motion certificate.

This deliberately does not infer a traversable portal from a depth jump. It is
the local measurement front end for replacing image-change arrival heuristics.
The horizontal-slice model is only appropriate for locally vertical obstacles;
it neither covers the articulated body nor certifies turning clearance.
"""
from copy import deepcopy
import hashlib

import numpy as np

from lewm.causal_depth_observation_development import body_points, CausalDepthHistory

STRIDE = 2
HALF_HEIGHT_M = .06
MIN_VERTICAL_RETURNS = 3
MAX_COLUMN_DEPTH_SPREAD_M = .04
MAX_NEIGHBOUR_SEPARATION_M = .12
MIN_SEGMENT_COLUMNS = 8
MIN_SEGMENT_LENGTH_M = .05
MAX_LINE_RESIDUAL_M = .015


def _line(points):
    centre = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points-centre, full_matrices=False)
    tangent, normal = vh[0].copy(), vh[1].copy()
    if np.dot(points[-1]-points[0], tangent) < 0: tangent *= -1
    if np.dot(centre, normal) < 0: normal *= -1
    residual = np.abs((points-centre)@normal)
    along = (points-centre)@tangent
    return centre, tangent, normal, residual, along


def _segments(points, columns):
    """Split ordered observations at corners; never join through missing rays."""
    pending = [(0, len(points))]
    result = []
    while pending:
        start, end = pending.pop()
        if end-start < MIN_SEGMENT_COLUMNS: continue
        p = points[start:end]
        centre, tangent, normal, residual, along = _line(p)
        if residual.max() > MAX_LINE_RESIDUAL_M:
            split = int(np.argmax(residual))
            # A farthest endpoint is common when fitting two unequal corner
            # arms. Bisect in that case, rather than accepting a bad fit.
            if split < MIN_SEGMENT_COLUMNS or len(p)-split < MIN_SEGMENT_COLUMNS:
                split = len(p)//2
            pending.extend(((start, start+split), (start+split, end)))
            continue
        if np.ptp(along) < MIN_SEGMENT_LENGTH_M: continue
        endpoints = centre+np.array([along.min(), along.max()])[:, None]*tangent
        result.append({'first_column': int(columns[start]), 'last_column': int(columns[end-1]),
            'support_columns': end-start, 'normal_body_xy': normal.tolist(),
            'offset_body_m': float(centre@normal), 'endpoints_body_xy_m': endpoints.tolist(),
            'maximum_fit_residual_m': float(residual.max()),
            'extent_kind': 'observed_support_only_not_physical_wall_endpoints'})
    return sorted(result, key=lambda row: row['first_column'])


def translation_constraint_geometry(segments):
    """Line-normal span, conditional on fixed attitude and correct association.

    This is not an odometry estimate or calibrated information/covariance.
    Parallel wall normals leave tangent translation unobservable, irrespective
    of the number of points. Unknown associations can only weaken this result.
    """
    normals = np.asarray([s['normal_body_xy'] for s in segments], dtype=float).reshape(-1, 2)
    matrix = normals.T@normals/max(1, len(normals))
    values, vectors = np.linalg.eigh(matrix)
    threshold = .01
    return {'normal_gram_eigenvalues': values.tolist(),
        'conditional_rank_xy': int(np.count_nonzero(values > threshold)),
        'weak_directions_body_xy': vectors[:, values <= threshold].T.tolist(),
        'translation_estimated': False,
        'assumption': 'fixed_attitude_correct_static_surface_correspondences',
        'calibrated_uncertainty': False}


def _grazing_continuity(points, valid, index):
    """Large range spacing alone is not a boundary on a grazing wall.

    Require available two-point tangents to agree with each other and with
    the joining displacement. At a missing-data/FOV boundary only one tangent
    may be available; no continuation into the missing data is inferred.
    This avoids bridging a step
    between two parallel front surfaces just because its four points nearly
    lie along the long joining line.
    """
    tangents = []
    if index >= 2 and valid[index-2:index].all():
        tangents.append(points[index-1]-points[index-2])
    if index+1 < len(points) and valid[index:index+2].all():
        tangents.append(points[index+1]-points[index])
    if not tangents or any(np.linalg.norm(t) < 1e-6 for t in tangents): return False
    tangents = [t/np.linalg.norm(t) for t in tangents]
    if len(tangents) == 2 and abs(tangents[0]@tangents[1]) < .995: return False
    jump = points[index]-points[index-1]
    return all(abs(np.array([-t[1], t[0]])@jump) <= MAX_LINE_RESIDUAL_M for t in tangents)


def observe_local_surfaces(depth, policy, *, now_ns):
    cloud = body_points(depth, policy, now_ns=now_ns, stride=STRIDE)
    p = cloud['points_body_m']
    selected = cloud['valid'] & (np.abs(p[..., 2]) <= HALF_HEIGHT_M)
    points = np.full((p.shape[1], 2), np.nan)
    counts = selected.sum(axis=0)
    valid = np.zeros(p.shape[1], dtype=bool)
    for column in np.flatnonzero(counts >= MIN_VERTICAL_RETURNS):
        values = p[selected[:, column], column, :2]
        # Reject mixed foreground/background within the slice instead of
        # averaging two surfaces into an invented intermediate obstacle.
        if np.ptp(values[:, 0]) > MAX_COLUMN_DEPTH_SPREAD_M: continue
        points[column] = np.median(values, axis=0)
        valid[column] = True
    columns = cloud['columns']
    segments, breaks, run = [], [], []
    def flush():
        if run:
            indices = np.asarray(run)
            segments.extend(_segments(points[indices], columns[indices]))
            run.clear()
    for i in range(len(columns)):
        if not valid[i]:
            flush()
            continue
        if (run and np.linalg.norm(points[i]-points[i-1]) > MAX_NEIGHBOUR_SEPARATION_M
                and not _grazing_continuity(points, valid, i)):
            before, after = points[i-1], points[i]
            breaks.append({'columns': [int(columns[i-1]), int(columns[i])],
                'observed_points_body_xy_m': [before.tolist(), after.tolist()],
                'nearer_column': int(columns[i-1] if before[0] <= after[0] else columns[i]),
                'kind': 'depth_discontinuity_occlusion_or_surface_end',
                'portal_width_m': None, 'traversable': None})
            flush()
        run.append(i)
    flush()
    unknown_runs = []
    start = None
    for i, good in enumerate([*valid.tolist(), True]):
        if not good and start is None: start = i
        if good and start is not None:
            unknown_runs.append([int(columns[start]), int(columns[i-1])]); start = None
    modelled = np.zeros(len(columns), dtype=bool)
    for segment in segments:
        modelled |= (columns >= segment['first_column']) & (columns <= segment['last_column'])
    return {'schema': 'depth_local_surfaces_development.v1', 'measured_ns': depth['measured_ns'],
        'identity': tuple(depth['identity']), 'rgb_sha256': depth['rgb_sha256'],
        'depth_sha256': hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
        'slice_half_height_body_m': HALF_HEIGHT_M, 'sampled_columns': columns.tolist(),
        'valid_columns': valid.tolist(), 'vertical_support_counts': counts.tolist(),
        'sampled_points_body_xy_m': [point.tolist() if good else None for point, good in zip(points, valid, strict=True)],
        'unmodelled_valid_columns': columns[valid & ~modelled].tolist(),
        'unknown_column_runs': unknown_runs, 'surface_segments': segments,
        'depth_discontinuities': breaks,
        'translation_constraint_geometry': translation_constraint_geometry(segments),
        'body_clearance_qualified': False, 'turn_clearance_qualified': False,
        'arrival_verified': False, 'free_volume_inferred': False,
        'scope': 'current observed body-height surface slice; unseen side/rear/height volume unknown'}


class LocalSurfaceHistory:
    """Bounded causal runtime observer, independent of evaluator/scene metadata."""
    def __init__(self):
        self.depth_history = CausalDepthHistory()
        self.frames = []

    def observe(self, depth, policy, *, now_ns):
        self.depth_history.push(depth, policy, now_ns=now_ns)
        result = observe_local_surfaces(depth, policy, now_ns=now_ns)
        self.frames = [*self.frames[-3:], deepcopy(result)]
        return result

    def snapshot(self):
        return deepcopy(self.frames)
