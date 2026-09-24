"""Account for the existing target setter's reset at a delayed arrival only."""
from copy import deepcopy
from lewm.settled_boundary_prefix_comparison_development import compare_current as compare_mission


def compare_current(original, candidate, *, previous_candidate=None):
    normalized = candidate
    reset = original['planner_mode'] != candidate['planner_mode']
    if reset:
        if previous_candidate is None:
            raise ValueError('target reset requires the preceding candidate decision')
        a, b, p = (d['mission_receipt'] for d in (original, candidate, previous_candidate))
        # set_goal resets mode only when the target changes. During a held
        # observation choose() is not called, so an unchanged target retains
        # precisely the preceding mode. Stop the prefix at this transition.
        if not (
                a['frame'] == b['frame'] == p['frame'] + 1
                and a['measured_ns'] == b['measured_ns'] == p['measured_ns'] + 100_000_000
                and a['phase_transition'] == 'OUTBOUND_TO_RETURN'
                and a['phase'] == 'RETURN' and b['phase'] == p['phase'] == 'OUTBOUND'
                and b['phase_transition'] is None
                and a['arrival_confirmed_this_frame'] is True
                and b['arrival_confirmed_this_frame'] is False
                and not b['arrivals'] and not p['arrivals']
                and len(a['arrivals']) == 1
                and a['active_goal_initial_body_xy_m'] == [0., 0.]
                and b['active_goal_initial_body_xy_m'] == p['active_goal_initial_body_xy_m']
                and a['active_goal_initial_body_xy_m'] != b['active_goal_initial_body_xy_m']
                and original['planner_mode'] == 'NEW'
                and candidate['planner_mode'] == previous_candidate['planner_mode']
                and candidate['planner_mode'] in ('WAYPOINT', 'VIEW_ACQUISITION')
                and previous_candidate['terminal'] is None
                and previous_candidate['failure'] is None
                and all(d['mission_receipt']['hold_required'] is True
                        and d['new_selection'] is None
                        and d['requested_command'] == [0., 0., 0.]
                        and d['goal_initial_body_xy_m'] == d['mission_receipt']['active_goal_initial_body_xy_m']
                        for d in (original, candidate, previous_candidate))):
            raise ValueError('planner mode difference is not a held delayed-target reset')
        normalized = deepcopy(candidate)
        normalized['planner_mode'] = original['planner_mode']
    result = compare_mission(original, normalized)
    return result | dict(delayed_target_reset_mode_difference=reset)
