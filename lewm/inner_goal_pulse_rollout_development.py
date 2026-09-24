"""Inner planning region; unchanged external acceptance, not an error bound."""
import math
import numpy as np
from lewm.coupled_pulse_rollout_development import PulseTable,pose3,goal_error

INTERNAL_POSITION_TOLERANCE_M=.04


def plan(table, start, target, *, yaw_mode='orientation', horizon=24, beam_width=256,
         maximum_excursion_m=1.):
    """Return an offline candidate, never a requested platform command.

    Each edge includes a 2/5-tick pulse and at least 20 zero-command ticks.
    Stable beam search jointly scores translation and yaw, with pose-bin
    diversity. Search exhaustion is not an infeasibility certificate. Bounds
    apply only to predicted endpoints, NOT intervening body sweep.
    """
    if not isinstance(table, PulseTable):
        raise ValueError('typed pulse table required')
    if (type(horizon) is not int or not 1 <= horizon <= 35
            or type(beam_width) is not int or not 1 <= beam_width <= 2048
            or isinstance(maximum_excursion_m, bool) or not math.isfinite(maximum_excursion_m)
            or not 0 < maximum_excursion_m <= 1.):
        raise ValueError('bounded search and excursion required')
    start, target = pose3(start), pose3(target)
    goal_error(start, target, yaw_mode)
    states = start[None, :]; paths = np.empty((1, 0), dtype=int)
    ticks = np.zeros(1, dtype=int); expanded = 0
    effects = np.array([e.delta_xy_yaw for e in table.effects])
    durations = np.array([e.ticks+20 for e in table.effects])
    best = None
    for depth in range(horizon+1):
        pe, ye = goal_error(states, target, yaw_mode)
        reached = (pe <= INTERNAL_POSITION_TOLERANCE_M) & (ye <= .05)
        cost = (pe/INTERNAL_POSITION_TOLERANCE_M)**2 + (ye/.15)**2 + .001*ticks
        choice = int(np.argmin(np.where(reached, cost, np.inf))) if reached.any() else int(np.argmin(cost))
        candidate = (float(cost[choice]), states[choice].copy(), paths[choice].copy(), int(ticks[choice]))
        if best is None or candidate[0] < best[0] or reached.any():
            best = candidate
        if reached.any() or depth == horizon:
            break
        c, s = np.cos(states[:, 2, None]), np.sin(states[:, 2, None])
        children = np.repeat(states[:, None, :], 6, axis=1)
        children[:, :, 0] += c*effects[:, 0]-s*effects[:, 1]
        children[:, :, 1] += s*effects[:, 0]+c*effects[:, 1]
        children[:, :, 2] += effects[:, 2]
        child_ticks = (ticks[:, None]+durations).ravel()
        child_paths = np.concatenate((np.repeat(paths, 6, axis=0), np.tile(np.arange(6), len(states))[:, None]), axis=1)
        children = children.reshape(-1, 3); expanded += len(children)
        valid = np.linalg.norm(children[:, :2]-start[:2], axis=1) <= maximum_excursion_m
        children, child_ticks, child_paths = children[valid], child_ticks[valid], child_paths[valid]
        if not len(children):
            break
        pe, ye = goal_error(children, target, yaw_mode)
        cost = (pe/INTERNAL_POSITION_TOLERANCE_M)**2 + (ye/.15)**2 + .001*child_ticks
        # Preserve goal-reaching candidates before diversity reduction.
        order = np.lexsort((np.arange(len(cost)), cost, ~((pe <= INTERNAL_POSITION_TOLERANCE_M) & (ye <= .05))))
        keys = children.copy()
        if yaw_mode == 'orientation':
            keys[:, 2] = np.arctan2(np.sin(keys[:, 2]), np.cos(keys[:, 2]))
        bins = np.rint(keys / [.01, .01, .025]).astype(np.int64)
        seen = set(); keep = []
        for i in order:
            key = tuple(bins[i])
            if key in seen:
                continue
            seen.add(key); keep.append(int(i))
            if len(keep) == beam_width:
                break
        states, ticks, paths = children[keep], child_ticks[keep], child_paths[keep]
    _, endpoint, indices, minimum_ticks = best
    pe, ye = goal_error(endpoint, target, yaw_mode)
    return dict(status='PREDICTED_GOAL_CANDIDATE' if pe <= INTERNAL_POSITION_TOLERANCE_M and ye <= .05 else 'SEARCH_EXHAUSTED',
                action_indices=indices.tolist(), predicted_endpoint=endpoint.tolist(),
                predicted_position_error_m=float(pe), predicted_yaw_error_rad=float(ye),
                minimum_command_ticks=minimum_ticks, expanded_nodes=expanded, yaw_mode=yaw_mode,
                internal_position_tolerance_m=INTERNAL_POSITION_TOLERANCE_M,
                fitting_audit_sha256=table.fitting_audit_sha256, model='empirical_action_duration_mean',
                search_complete=False, motion_permission=False, body_sweep_checked=False,
                uncertainty_calibrated=False, learned_jepa_used=False, navigation_qualified=False)
