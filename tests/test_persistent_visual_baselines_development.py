from types import SimpleNamespace
import numpy as np
from lewm.persistent_visual_baselines_development import RUNTIMES, select_reactive_terminal


def test_off_runtimes_ignore_adversarial_forecasts_and_scores():
    snapshot = SimpleNamespace(fine_occupied={(100,y) for y in range(-100,101)})
    for arm in ('reserved_off', 'reactive_feedback'):
        runtime = RUNTIMES[arm].__new__(RUNTIMES[arm])
        runtime.planning_translation_pulse = False
        runtime.mission = SimpleNamespace(arrival_radius_m=.02)
        outputs = []
        for action, forecast in [('left_turn', object()), ('hold', np.full((6,8,5),np.nan))]:
            selection = dict(waypoint_body_xy_m=[1.,0.], action=action, candidates=object())
            outputs.append(runtime._select_clear_prediction(selection, forecast,
                snapshot, np.zeros(3), np.eye(3)))
        assert outputs[0] == outputs[1]
        assert outputs[0]['action'] == 'forward'


def test_reactive_retains_heading_then_pulse_and_measured_arrival_hold():
    settings = dict(scan_error=None, clearance_m=.6, pulse=True, arrival_radius_m=.02)
    turn = select_reactive_terminal([.03,.02], **settings)
    assert turn['action'] == 'left_turn' and turn['command_duration_ns'] == 400_000_000
    move = select_reactive_terminal([.04,0.], **settings)
    assert move['action'] == 'forward' and move['command_duration_ns'] == 100_000_000
    assert select_reactive_terminal([.01,0.], **settings)['action'] == 'hold'
    assert select_reactive_terminal([.04,0.], **(settings | dict(clearance_m=.449)))['action'] == 'hold'


def test_reactive_view_requests_never_translate():
    selection = select_reactive_terminal([.1,0.], scan_error=-.6,
        clearance_m=.6, pulse=False, arrival_radius_m=.02)
    assert selection['action'] == 'right_turn'
    assert not any(selection['requested_command'][:2])
