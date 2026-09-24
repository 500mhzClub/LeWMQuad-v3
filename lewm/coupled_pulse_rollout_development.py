"""Bounded, offline SE(2) pulse+brake planning baseline, not a motion permit.

The empirical model is action/duration-only: no friction inference, body-state
conditioning, uncertainty calibration, swept-volume model, or JEPA. A successor
controller must re-observe after actual braking; a predicted endpoint is not an
executed arrival. Orientation targets and signed winding targets are distinct.
"""
from dataclasses import dataclass
import math
import numpy as np


COMMANDS = ((.2, 0., 0.), (0., 0., .45), (0., 0., -.45))


@dataclass(frozen=True)
class PulseEffect:
    command: tuple
    ticks: int
    delta_xy_yaw: tuple
    samples: int

    def __post_init__(self):
        if (type(self.command) is not tuple or self.command not in COMMANDS
                or type(self.ticks) is not int or self.ticks not in (2, 5)
                or type(self.samples) is not int or self.samples < 2
                or type(self.delta_xy_yaw) is not tuple):
            raise ValueError('fixed pulse vocabulary and at least two fitting samples required')
        d = np.asarray(self.delta_xy_yaw, float)
        if d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d[:2]) > .2 or abs(d[2]) > .6:
            raise ValueError('finite bounded measured endpoint effect required')


@dataclass(frozen=True)
class PulseTable:
    effects: tuple
    fitting_audit_sha256: str

    def __post_init__(self):
        if (type(self.effects) is not tuple or len(self.effects) != 6
                or any(not isinstance(v, PulseEffect) for v in self.effects)
                or {(v.command, v.ticks) for v in self.effects} != {(c, t) for c in COMMANDS for t in (2, 5)}):
            raise ValueError('exactly six distinct supported action/duration cells required')
        h = self.fitting_audit_sha256
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('explicit fitting audit identity required; not itself verification')


def pose3(value):
    p = np.asarray(value, float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('finite x,y,yaw required')
    return p.copy()


def compose(pose, delta):
    """Translate in the START body frame, then add measured unwrapped yaw."""
    p, d = pose3(pose), pose3(delta)
    c, s = math.cos(p[2]), math.sin(p[2])
    return p + [c*d[0]-s*d[1], s*d[0]+c*d[1], d[2]]


def goal_error(poses, target, yaw_mode):
    if yaw_mode not in ('orientation', 'winding'):
        raise ValueError('explicit orientation or winding semantics required')
    delta = poses-target
    if yaw_mode == 'orientation':
        delta[..., 2] = np.arctan2(np.sin(delta[..., 2]), np.cos(delta[..., 2]))
    return np.linalg.norm(delta[..., :2], axis=-1), abs(delta[..., 2])


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
        reached = (pe <= .06) & (ye <= .05)
        cost = (pe/.06)**2 + (ye/.15)**2 + .001*ticks
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
        cost = (pe/.06)**2 + (ye/.15)**2 + .001*child_ticks
        # Preserve goal-reaching candidates before diversity reduction.
        order = np.lexsort((np.arange(len(cost)), cost, ~((pe <= .06) & (ye <= .05))))
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
    return dict(status='PREDICTED_GOAL_CANDIDATE' if pe <= .06 and ye <= .05 else 'SEARCH_EXHAUSTED',
                action_indices=indices.tolist(), predicted_endpoint=endpoint.tolist(),
                predicted_position_error_m=float(pe), predicted_yaw_error_rad=float(ye),
                minimum_command_ticks=minimum_ticks, expanded_nodes=expanded, yaw_mode=yaw_mode,
                fitting_audit_sha256=table.fitting_audit_sha256, model='empirical_action_duration_mean',
                search_complete=False, motion_permission=False, body_sweep_checked=False,
                uncertainty_calibrated=False, learned_jepa_used=False, navigation_qualified=False)
