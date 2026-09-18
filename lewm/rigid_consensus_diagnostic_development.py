"""Explain one rejected rigid consensus without changing or granting a pose."""
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_rgbd_rigid_pose_development import register, fit, inliers, RIGID_RULES
from lewm.rgbd_correspondence_motion_development import RULES, cells


def diagnose(a, b, ua, ub, *, gyro_rotation, frame):
    try:
        _, _, _, row = register(a, b, ua, ub, gyro_rotation=gyro_rotation, mode='joint', frame=frame)
        return dict(original_accepted=True, original_registration=row, diagnostic_pose_permission=False)
    except SensorContractError as error:
        reason = str(error)
    row = dict(original_accepted=False, original_rejection=reason, diagnostic_pose_permission=False,
        lifted_matches=len(a), decomposed_consensus_gate=False)
    if reason != 'rigid consensus fraction, grid support or displacement rejected':
        return row
    # Reconstruct the unchanged deterministic proposal/ranking/pruning path.
    # Core fit and inlier calculations call the frozen functions directly.
    rng = np.random.default_rng(np.random.SeedSequence([RIGID_RULES['seed'], frame]))
    subsets = [np.arange(len(a))] + [rng.choice(len(a), 3, replace=False) for _ in range(RIGID_RULES['proposals'])]
    best = rank = None
    for indices in subsets:
        try:
            R, t, _ = fit(a[indices], b[indices], gyro_rotation=None)
        except SensorContractError:
            continue
        mask, residual = inliers(a, b, ua, ub, R, t)
        candidate = (int(mask.sum()), -float(np.sum(residual[mask]**2)))
        if best is None or candidate > rank:
            best, rank = mask, candidate
    mask = best.copy()
    while True:
        assert mask.sum() >= RULES['minimum_matches']
        R, t, _ = fit(a[mask], b[mask], gyro_rotation=None)
        good, residual = inliers(a, b, ua, ub, R, t)
        use = mask & good
        if np.array_equal(use, mask):
            break
        mask = use
    gates = dict(inlier_fraction=bool(mask.mean() >= RULES['minimum_inlier_fraction']),
        grid_support=bool(min(cells(ua[mask]), cells(ub[mask])) >= RULES['minimum_grid_cells']),
        displacement=bool(np.linalg.norm(t) <= RIGID_RULES['maximum_reference_translation_m']))
    assert not all(gates.values())
    return row | dict(decomposed_consensus_gate=True, inliers=int(mask.sum()),
        inlier_fraction=float(mask.mean()), minimum_inlier_fraction=RULES['minimum_inlier_fraction'],
        reference_grid_cells=cells(ua[mask]), current_grid_cells=cells(ub[mask]),
        minimum_grid_cells=RULES['minimum_grid_cells'], translation_norm_m=float(np.linalg.norm(t)),
        maximum_reference_translation_m=RIGID_RULES['maximum_reference_translation_m'], gate_passes=gates,
        residual_rms_m=float(np.sqrt(np.mean(residual[mask]**2))))
