import numpy as np
from lewm.planned_stopping_projection_development import stopping_projection_checks, avoid_blocked_translation
from lewm.geometry_progress_pilot_development import ACTIONS


def predictions():
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.
    return p


def test_stopping_extension_rejects_translation_but_short_pulse_fits():
    p = predictions(); wall = {(65, y) for y in range(-100, 101)}
    full = stopping_projection_checks(p, wall, np.zeros(3), np.eye(3))
    short = stopping_projection_checks(p, wall, np.zeros(3), np.eye(3), pulse=True)
    assert not full[1]['projection_clear']
    assert short[1]['projection_clear']
    assert full[1]['samples'][0]['requested_speed_horizon_s'] == 1.1
    assert full[4]['projection_clear'] and full[4]['samples'] == []


def test_predicted_heading_changes_stopping_projection_direction():
    p = predictions(); p[:, :, 2] = 1.; p[:, :, 3] = 0.
    xwall = {(65, y) for y in range(-100, 101)}
    ywall = {(x, 65) for x in range(-100, 101)}
    assert stopping_projection_checks(p, xwall, np.zeros(3), np.eye(3))[1]['projection_clear']
    assert not stopping_projection_checks(p, ywall, np.zeros(3), np.eye(3))[1]['projection_clear']


def test_observed_map_translation_is_applied_to_predicted_positions():
    p = predictions(); wall = {(165, y) for y in range(-100, 101)}
    assert stopping_projection_checks(p, wall, np.zeros(3), np.eye(3))[1]['projection_clear']
    assert not stopping_projection_checks(p, wall, np.array([1., 0., 0.]), np.eye(3))[1]['projection_clear']


def test_blocked_translation_selects_clear_turn_without_reenabling_blocked_turn():
    checks = stopping_projection_checks(predictions(), {(65,y) for y in range(-100,101)}, np.zeros(3), np.eye(3))
    selection = dict(action='forward', action_index=1, requested_command=[.2,0.,0.],
        candidates=[dict(action=a, utility_m=i) for i,a in enumerate(ACTIONS)],
        memory_forecast_candidates=[dict(action=a, nominal_predicted_path_clear=a!='right_turn',
            reserve_recovery_path_clear=False) for a in ACTIONS])
    revised = avoid_blocked_translation(selection, checks)
    assert revised['action']=='left_turn' and selection['action']=='forward'
    assert revised['planned_stopping_projection']['changed']
    selection.update(action='hold', action_index=0, requested_command=[0.,0.,0.])
    assert avoid_blocked_translation(selection, checks)['action']=='hold'
