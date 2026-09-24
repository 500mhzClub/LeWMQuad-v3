from copy import deepcopy
import numpy as np
import pytest
from lewm.exact_terminal_waypoint_development import exact_terminal_target
from lewm.geometry_progress_pilot_development import ACTIONS


def selection():
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.; p[:, :, 4] = -10.
    p[1, :, :2] = [.124, .026]; p[2, :, :2] = [.149, .049]
    return dict(mode='WAYPOINT', proposal=dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL', route_cells=[[2, 0]]),
        waypoint_map_xy_m=[.125, .025], goal_body_xy_m=[.125, .025],
        action='forward', action_index=1, prediction=p.tolist(),
        candidates=[dict(action=a, utility_m=0.) for a in ACTIONS],
        surface_checks=[dict(possible_intersection=False) for a in ACTIONS],
        nominal_action_checks=[dict(action=a, nominal_disk_connector_clear=True) for a in ACTIONS],
        phase_allowed_actions=list(ACTIONS), nominal_constraint_horizon_ns=500_000_000)


def apply(s):
    return exact_terminal_target(s, np.zeros(3), np.eye(3), [.149, .049])


def test_actual_point_changes_ranking_without_changing_predictions_or_constraints():
    s=selection(); before=deepcopy(s); r=apply(s)
    assert r['action']=='left_arc' and r['intermediate_target_is_mission_goal']
    assert r['waypoint_map_xy_m']==[.149,.049] and s==before
    for k in ('prediction','surface_checks','nominal_action_checks','proposal'):
        assert r[k]==s[k]


@pytest.mark.parametrize('veto', ['surface', 'nominal'])
def test_exact_target_cannot_remove_a_veto(veto):
    s=selection()
    if veto=='surface':s['surface_checks'][2]['possible_intersection']=True
    else:s['nominal_action_checks'][2]['nominal_disk_connector_clear']=False
    assert apply(s)['action']=='forward'


def test_frontier_and_intermediate_targets_are_unchanged():
    s=selection(); s['proposal']['status']='OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    assert apply(s) is s
    s=selection(); s['waypoint_map_xy_m']=[.025,.025]
    assert apply(s) is s


def test_outside_cell_and_empty_feasible_population():
    s=selection()
    with pytest.raises(ValueError):exact_terminal_target(s,np.zeros(3),np.eye(3),[.151,.049])
    for c in s['nominal_action_checks']:c['nominal_disk_connector_clear']=False
    r=apply(s)
    assert r['action'] is None and r['requested_command']==[0.,0.,0.]
