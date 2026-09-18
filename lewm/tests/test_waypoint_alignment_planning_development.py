import numpy as np
from lewm.waypoint_alignment_planning_development import score_waypoint_alignment


def test_turn_gets_credit_without_inventing_positional_progress():
    p = np.zeros((6, 8, 5), dtype=float)
    p[:, :, 3] = 1.
    p[:, :, 4] = -8.
    p[4, 3:, 2] = np.sin(.18)
    p[4, 3:, 3] = np.cos(.18)
    p[5, 3:, 2] = np.sin(-.18)
    p[5, 3:, 3] = np.cos(-.18)
    result = score_waypoint_alignment(p, [0., 1.], delay_ticks=3, commit_ticks=4)
    assert result['action'] == 'left_turn'
    assert result['candidates'][4]['predicted_progress_during_commit_m'] == 0.
    assert np.isclose(result['candidates'][4]['predicted_alignment_progress_m'], .35*.18)
    # A turn already completed in the fixed prefix earns no candidate credit.
    p[4, :3, 2] = np.sin(.18)
    p[4, :3, 3] = np.cos(.18)
    result = score_waypoint_alignment(p, [0., 1.], delay_ticks=3, commit_ticks=4)
    assert result['action'] == 'hold'


def test_aligned_forward_and_wrapped_bearing():
    p = np.zeros((6, 8, 5), dtype=float)
    p[:, :, 3] = 1.
    p[:, :, 4] = -8.
    p[1, 3:, 0] = .08
    assert score_waypoint_alignment(p, [1., 0.], delay_ticks=3, commit_ticks=4)['action'] == 'forward'
    p[:, :, :2] = 0.
    p[:, :, 2] = np.sin(3.10)
    p[:, :, 3] = np.cos(3.10)
    p[4, 3:, 2] = np.sin(-3.10)
    p[4, 3:, 3] = np.cos(-3.10)
    goal = [np.cos(-3.10), np.sin(-3.10)]
    result = score_waypoint_alignment(p, goal, delay_ticks=3, commit_ticks=4)
    assert result['action'] == 'left_turn'
    assert np.isclose(result['candidates'][4]['predicted_alignment_progress_m'], .35*(2*np.pi-6.2))
