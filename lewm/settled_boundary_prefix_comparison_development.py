"""Exact full decision comparison with only declared mission-state differences."""
from copy import deepcopy

BEHAVIOR_FIELDS = ('phase', 'active_goal_initial_body_xy_m', 'hold_required', 'terminal')
MISSION_FIELDS = ('quiet_intervals', 'phase', 'phase_transition', 'active_goal_initial_body_xy_m',
    'arrival_confirmed_this_frame', 'arrivals')


def compare_current(original, candidate):
    if original['terminal'] is not None or candidate['terminal'] is not None or candidate['failure'] is not None:
        raise ValueError('nonterminal admitted controller prefix required')
    a, b = original['mission_receipt'], candidate['mission_receipt']
    behavior = [k for k in BEHAVIOR_FIELDS if a[k] != b[k]]
    if len(b['arrivals']) > len(a['arrivals']):
        raise ValueError('settling cannot create an earlier arrival')
    normalized = deepcopy(candidate)
    if normalized['controller'] != 'settled_boundary_round_trip_controller_v1':
        raise ValueError('explicit boundary-settling controller required')
    normalized['controller'] = original['controller']
    for key in ('measured_settling_required_for_arrival', 'measured_quiet_boundary_required_before_dwell'):
        if normalized.pop(key) is not True:
            raise ValueError('measured boundary settling must remain enabled')
    mission = normalized['mission_receipt']
    if mission.pop('measured_settling_required') is not True:
        raise ValueError('measured settling mission required')
    receipt = mission.pop('observed_settling')
    if (receipt['current_frame'] != b['frame'] or receipt['measured_ns'] != b['measured_ns']
            or receipt['continuous_speed_bound'] is not False or receipt['native_state_used'] is not False
            or receipt['first_quiet_observation_starts_dwell'] is not True):
        raise ValueError('current observed settling with honest limitations required')
    # The complete new mission receipt is saved. Strip only this declared
    # intervention when comparing the untouched observer/map/model/planner path.
    for key in MISSION_FIELDS:
        mission[key] = deepcopy(a[key])
    normalized['quiet_intervals'] = original['quiet_intervals']
    normalized['goal_initial_body_xy_m'] = original['goal_initial_body_xy_m']
    if normalized != original:
        raise ValueError('complete decision changed outside declared mission settling')
    return dict(mission_behavior_differences=behavior,
        quiet_counter_changed=a['quiet_intervals'] != b['quiet_intervals'],
        requested_command_changed=original['requested_command'] != candidate['requested_command'],
        all_other_complete_decision_fields_exact=True)
