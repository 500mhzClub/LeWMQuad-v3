from copy import deepcopy
from types import SimpleNamespace
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.terminal_position_priority_development import select_terminal_progress,TerminalPositionPriorityRuntime


def selection():
    positions=dict(hold=-.0039,forward=-.0025,left_arc=-.001,right_arc=.0053,left_turn=-.0041,right_turn=-.0042)
    return dict(action='right_turn',candidates=[dict(action=a,position_contact_utility_m=positions[a],
        predicted_progress_during_commit_m=.012 if a=='right_arc' else -.001) for a in ACTIONS],
        memory_forecast_candidates=[dict(action=a,nominal_predicted_path_clear=True,
            reserve_recovery_path_clear=False) for a in ACTIONS])


def test_terminal_translation_can_improve_position_despite_heading_penalty():
    original=selection();before=deepcopy(original);result=select_terminal_progress(original)
    assert result['action']=='right_arc' and result['terminal_position_priority']['changed']
    assert original==before


def test_no_improving_clear_translation_keeps_heading_guidance_and_recovery():
    original=selection()
    for row in original['memory_forecast_candidates']:
        if row['action']=='right_arc':row['nominal_predicted_path_clear']=False
    assert select_terminal_progress(original)['action']=='right_turn'
    original=selection();original['clearance_turn']={'active':True}
    assert select_terminal_progress(original)==original


def test_only_the_near_exact_mission_endpoint_enables_position_priority():
    runtime=object.__new__(TerminalPositionPriorityRuntime)
    snapshot=SimpleNamespace(fine_occupied=frozenset())
    route=dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL',route_cells=[[0,0]],goal_map_xy_m=[.01,.02])
    runtime._route_target(route,snapshot,[0.,0.]);assert runtime.terminal_position_approach
    route['status']='OBSERVED_FLOOR_ROUTE_TO_FRONTIER'
    runtime._route_target(route,snapshot,[0.,0.]);assert not runtime.terminal_position_approach
    route['status']='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    runtime._route_target(route,snapshot,[.5,0.]);assert not runtime.terminal_position_approach
