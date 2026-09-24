"""Batch repeated gyro proposal scatter calculations without changing the search."""

import numpy as np

from lewm.joint_rgbd_rigid_pose_development import (
    SensorContractError, proper, fit, inliers, cells, angle, RULES, RIGID_RULES)


def proposal_fits(a, b, G, fixed, frame):
    """Same ordered proposals; batch only the gyro-conditioned triple fits."""
    try:
        R, t, _ = fit(a, b, gyro_rotation=fixed)
    except SensorContractError:
        pass
    else:
        yield R, t
    rng = np.random.default_rng(np.random.SeedSequence([RIGID_RULES['seed'], frame]))
    indices = np.array([rng.choice(len(a), 3, replace=False)
        for _ in range(RIGID_RULES['proposals'])])
    if fixed is None:
        for idx in indices:
            try:
                R, t, _ = fit(a[idx], b[idx], gyro_rotation=None)
            except SensorContractError:
                continue
            yield R, t
        return
    # G was validated once on entry. It is the identical rotation used by
    # every gyro proposal; scatter thresholds still apply to both triples.
    aa, bb = a[indices], b[indices]
    ma, mb = aa.mean(axis=1), bb.mean(axis=1)
    sa = np.linalg.svd(aa-ma[:, None, :], compute_uv=False)/np.sqrt(3)
    sb = np.linalg.svd(bb-mb[:, None, :], compute_uv=False)/np.sqrt(3)
    valid = np.ones(len(indices), dtype=bool)
    for scatter in (sa, sb):
        valid &= scatter[:, 1] >= RIGID_RULES['minimum_second_scatter_rms_m']
        valid &= scatter[:, 1] >= RIGID_RULES['minimum_tangent_scatter_ratio']*scatter[:, 0]
    # Keep scalar matrix-vector evaluation and inlier/ranking operations in
    # their original order. Final fitting and monotonic pruning are unchanged.
    for i in np.flatnonzero(valid):
        yield G, ma[i]-G@mb[i]


def register(a, b, ua, ub, *, gyro_rotation, mode, frame):
    a, b, ua, ub = [np.asarray(x, float) for x in (a, b, ua, ub)]; G = proper(gyro_rotation)
    if (mode not in ('joint', 'gyro') or type(frame) is not int or frame < 0
            or a.ndim != 2 or a.shape[1:] != (3,) or b.shape != a.shape
            or ua.shape != (len(a), 2) or ub.shape != ua.shape
            or not all(np.isfinite(x).all() for x in (a, b, ua, ub))):
        raise SensorContractError('paired points, pixels and declared fitting mode required')
    if len(a) < RULES['minimum_matches']: raise SensorContractError('insufficient rigid-pose matches')
    fixed = None if mode == 'joint' else G
    best = None; rank = None; valid_candidates = 0
    for R, t in proposal_fits(a, b, G, fixed, frame):
        valid_candidates += 1; mask, residual = inliers(a, b, ua, ub, R, t)
        count = int(mask.sum()); candidate = (count, -float(np.sum(residual[mask]**2)))
        if best is None or candidate > rank: best = mask; rank = candidate
        if count == len(a): break
    if best is None: raise SensorContractError('no conditioned rigid-pose proposal')
    mask = best.copy(); initial_count = int(mask.sum()); rounds = 0
    while True:
        if mask.sum() < RULES['minimum_matches']:
            raise SensorContractError('insufficient rigid consensus after pruning')
        R, t, conditioning = fit(a[mask], b[mask], gyro_rotation=fixed)
        good, residual = inliers(a, b, ua, ub, R, t); use = mask & good; rounds += 1
        if np.array_equal(use, mask): break
        mask = use
    if (mask.mean() < RULES['minimum_inlier_fraction']
            or min(cells(ua[mask]), cells(ub[mask])) < RULES['minimum_grid_cells']
            or np.linalg.norm(t) > RIGID_RULES['maximum_reference_translation_m']):
        raise SensorContractError('rigid consensus fraction, grid support or displacement rejected')
    disagreement = angle(G.T@R)
    if disagreement > RIGID_RULES['maximum_gyro_disagreement_rad']:
        raise SensorContractError('image and gyro reference rotations disagree beyond diagnostic envelope')
    return R, t, mask, conditioning | dict(mode=mode, lifted_matches=len(a), inliers=int(mask.sum()),
        inlier_fraction=float(mask.mean()), reference_grid_cells=cells(ua[mask]), current_grid_cells=cells(ub[mask]),
        residual_rms_m=float(np.sqrt(np.mean(residual[mask]**2))), valid_proposals=valid_candidates,
        initial_consensus_points=initial_count, pruning_rounds=rounds, gyro_disagreement_rad=disagreement,
        matched_consensus_rules=True, pose_error_bound=None, conditioning_is_not_covariance=True)
