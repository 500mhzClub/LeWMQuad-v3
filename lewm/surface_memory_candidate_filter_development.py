"""Matched experimental surface-conflict filter for learned candidate rankings."""
from copy import deepcopy
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands


def filter_selection(selection, memory, geometry, *, now_ns, persistent):
    result = deepcopy(selection)
    prediction = np.asarray(result['prediction'], float)
    if (prediction.shape != (6, 8, 5) or not np.isfinite(prediction).all()
            or [r['action'] for r in result['candidates']] != list(ACTIONS)
            or type(persistent) is not bool):
        raise ValueError('complete ordered six-plan predictions and explicit memory variant required')
    checks = []
    for row in prediction:
        dx, dy, sy, cy, _ = row[0]
        if np.hypot(sy, cy) <= 1e-8:
            raise ValueError('defined first-horizon predicted yaw required')
        checks.append(memory.footprint(geometry, [dx, dy], float(np.arctan2(sy, cy)),
            now_ns=now_ns, persistent=persistent))
    feasible = [i for i, row in enumerate(checks) if not row['possible_intersection']]
    utilities = np.asarray([r['utility_m'] for r in result['candidates']], float)
    if utilities.shape != (6,) or not np.isfinite(utilities).all():
        raise ValueError('finite original learned utilities required')
    chosen = max(feasible, key=lambda i: utilities[i]) if feasible else None
    result.update(unfiltered_action=result['action'], surface_checks=checks,
        action=None if chosen is None else ACTIONS[chosen], action_index=chosen,
        requested_command=[0., 0., 0.] if chosen is None else candidate_commands(ACTIONS[chosen])[0],
        surface_filter_horizon_ns=500_000_000, surface_memory_variant='persistent' if persistent else 'current_frame',
        no_surface_conflict_candidates=len(feasible), free_space_established=False,
        selection_rule='maximum_original_utility_among_no_sampled_surface_intersection',
        experimental_motion_without_clearance_certificate=True)
    return result
