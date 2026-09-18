import numpy as np
import pytest
from lewm.exact_mission_target_development import terminal_target
from lewm.observed_floor_waypoint_development import centre,segment_cells,CELL_M


def setup():
    p=np.array([1.22,-.10]);g=np.array([1.2,0.]);cell=tuple(map(int,np.floor(g/CELL_M)))
    proposal=dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL',route_cells=[[24,-2],list(cell)])
    return proposal,centre(cell),p,g,segment_cells(p,g),set()


def test_exact_instruction_replaces_goal_cell_centre_without_tolerance_change():
    args=setup();before=set(args[4]);target,e=terminal_target(*args)
    np.testing.assert_array_equal(target,args[3]);assert not np.array_equal(target,args[1])
    assert e['selected'] and not e['goal_tolerance_changed'] and not e['unobserved_floor_inferred']
    assert args[4]==before and not e['ground_support_approved']


def test_intermediate_and_frontier_waypoints_are_preserved():
    args=list(setup());args[1]=centre(args[0]['route_cells'][0])
    target,e=terminal_target(*args);np.testing.assert_array_equal(target,args[1]);assert not e['selected']
    args=list(setup());args[0]['status']='OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    target,e=terminal_target(*args);np.testing.assert_array_equal(target,args[1]);assert not e['selected']


def test_every_closed_connector_cell_including_goal_corner_must_be_observed():
    args=list(setup());args[4].remove(tuple(map(int,np.floor(args[3]/CELL_M))))
    target,e=terminal_target(*args);np.testing.assert_array_equal(target,args[1])
    assert not e['selected'] and e['unknown_connector_cells']


def test_known_obstacle_clearance_remains_required_for_final_connector():
    args=list(setup());args[5]={(23,0)}
    target,e=terminal_target(*args);np.testing.assert_array_equal(target,args[1])
    assert not e['selected'] and not e['nominal_connector']['nominal_disk_connector_clear']


def test_inconsistent_goal_cell_and_nonfinite_inputs_are_rejected():
    args=list(setup());args[0]['route_cells'][-1]=[0,0]
    with pytest.raises(ValueError,match='endpoint'):terminal_target(*args)
    args=list(setup());args[3]=[np.nan,0.]
    with pytest.raises(ValueError,match='bounded'):terminal_target(*args)
