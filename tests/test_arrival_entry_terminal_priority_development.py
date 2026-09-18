import numpy as np
from lewm.arrival_entry_terminal_priority_development import require_arrival_entry


def selection():
    return dict(action='forward', waypoint_body_xy_m=[0., .03],
        memory_forecast_candidates=[dict(reserve_recovery_path_clear=False) for _ in range(6)],
        terminal_position_priority=dict(changed=True, original_action='left_turn',
            selected_action='forward', original_selection_objective='distance_and_heading_progress_minus_contact'))


def test_sideways_goal_restores_turn_but_predicted_arrival_allows_translation():
    saved = selection(); prediction = np.zeros((6,8,3))
    prediction[1,6,:2] = [.005,0.]
    result = require_arrival_entry(saved,prediction,arrival_radius_m=.02)
    assert result['action']=='left_turn'
    assert saved['action']=='forward'
    assert result['arrival_entry_terminal_priority']['changed']
    prediction[1,6,:2] = [0.,.025]
    result = require_arrival_entry(saved,prediction,arrival_radius_m=.02)
    assert result['action']=='forward'
    assert not result['arrival_entry_terminal_priority']['changed']


def test_subsequent_hold_and_normal_aligned_translation_are_preserved():
    prediction = np.zeros((6,8,3))
    saved = selection(); saved['action']='hold'
    assert require_arrival_entry(saved,prediction,arrival_radius_m=.02)['action']=='hold'
    saved = selection(); saved['terminal_position_priority']['changed']=False
    assert require_arrival_entry(saved,prediction,arrival_radius_m=.02)['action']=='forward'
