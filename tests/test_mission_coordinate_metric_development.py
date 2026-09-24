import numpy as np
import pytest

from lewm.mission_coordinate_metric_development import planar_body_to_initial_xy
from lewm.delayed_action_planning_development import score_delayed_predictions
from lewm.waypoint_alignment_planning_development import score_waypoint_alignment
from lewm.predictive_arrival_hold_development import select_predicted_arrival_hold
from lewm.arrival_entry_terminal_priority_development import require_arrival_entry
from lewm.mission_coordinate_runtime_development import MissionCoordinateMixin
from types import SimpleNamespace


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_level_robot_preserves_horizontal_distances_at_any_heading():
    xy = np.array([[.01, .02], [-.03, .04], [0., 0.]])
    for heading in (0., .7, -2.4):
        Q = rotation_z(heading)
        A = planar_body_to_initial_xy(np.eye(3), Q)
        np.testing.assert_allclose(xy @ A.T, (np.c_[xy, np.zeros(3)] @ Q.T)[:, :2])
        np.testing.assert_allclose(np.linalg.norm(xy @ A.T, axis=1), np.linalg.norm(xy, axis=1))


def test_tilted_projection_matches_independent_three_dimensional_plane_intersection():
    B = rotation_y(.04)
    for pitch, heading in ((-.09, .6), (.12, -1.3), (.01, 2.8)):
        Q = rotation_y(pitch) @ rotation_z(heading)
        R = B.T @ Q
        A = planar_body_to_initial_xy(B, Q)
        # Solve directly for a 3D initial-frame displacement constrained by
        # its two body components and zero gravity-aligned height change.
        constraints = np.stack((R.T[0], R.T[1], B[2]))
        for xy in ([.02, -.01], [-.04, .03]):
            displacement = np.linalg.solve(constraints, [*xy, 0.])
            np.testing.assert_allclose(A @ xy, displacement[:2], atol=1e-15)
            np.testing.assert_allclose((B @ displacement)[2], 0., atol=1e-15)


def test_flat_map_goal_embedding_can_falsely_appear_inside_arrival_radius():
    B = rotation_y(.04)
    Q = rotation_y(.01) @ rotation_z(.6)
    position = np.array([0., 0., .08])
    goal = np.array([.022, 0.])
    old_map_delta = (B @ np.r_[goal, 0.])[:2] - (B @ position)[:2]
    old_body_target = (Q.T @ np.r_[old_map_delta, 0.])[:2]
    assert np.linalg.norm(old_body_target) < .02
    assert np.linalg.norm(goal-position[:2]) > .02
    A = planar_body_to_initial_xy(B, Q)
    corrected_body_target = np.linalg.solve(A, goal-position[:2])
    np.testing.assert_allclose(position[:2] + A @ corrected_body_target, goal, atol=1e-15)
    assert np.linalg.norm(position[:2] + A @ old_body_target - goal) > .003


def test_unparameterizable_orientation_is_rejected():
    with pytest.raises(ValueError, match='cannot parameterize'):
        planar_body_to_initial_xy(np.eye(3), rotation_y(np.pi/2))


def test_progress_uses_metric_without_modifying_forecasts():
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.
    p[:, 6, :2] = [[0, 0], [.01, 0], [0, .01], [.01, .01], [-.01, 0], [0, -.01]]
    original = p.copy(); goal = np.array([.03, .02]); A = np.array([[1.2, .1], [0., 1.]])
    r = score_delayed_predictions(p, goal, delay_ticks=3, commit_ticks=4, position_metric_matrix=A)
    expected = np.linalg.norm(A @ goal)-np.linalg.norm((goal-p[:, 6, :2]) @ A.T, axis=1)
    np.testing.assert_allclose([c['predicted_progress_during_commit_m'] for c in r['candidates']], expected)
    np.testing.assert_array_equal(p, original)
    old = score_delayed_predictions(p, goal, delay_ticks=3, commit_ticks=4)
    expected_old = np.linalg.norm(goal)-np.linalg.norm(goal-p[:, 6, :2], axis=1)
    np.testing.assert_array_equal([c['predicted_progress_during_commit_m'] for c in old['candidates']], expected_old)
    assert 'position_metric_matrix' not in old


def test_hold_and_arrival_entry_share_the_position_metric():
    p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.
    selection = dict(action='forward', waypoint_body_xy_m=[.018, 0.],
        memory_forecast_candidates=[dict(nominal_predicted_path_clear=True,
            reserve_recovery_path_clear=False) for _ in range(6)],
        terminal_position_priority=dict(changed=True, original_action='left_turn',
            selected_action='forward', original_selection_objective='heading'))
    assert select_predicted_arrival_hold(selection, p, arrival_radius_m=.02)['action'] == 'hold'
    assert require_arrival_entry(selection, p, arrival_radius_m=.02)['action'] == 'forward'
    corrected = selection | dict(position_metric_matrix=[[1.2, 0.], [0., 1.]])
    hold = select_predicted_arrival_hold(corrected, p, arrival_radius_m=.02)
    entry = require_arrival_entry(corrected, p, arrival_radius_m=.02)
    assert hold['action'] == 'forward' and not hold['predictive_arrival_hold']['eligible']
    assert entry['action'] == 'left_turn'
    np.testing.assert_allclose(hold['predictive_arrival_hold']['predicted_terminal_distances_m'], [.0216, .0216])
    assert selection['action'] == 'forward'


class ScoringBase:
    def __init__(self):
        self.terminal_position_approach = True

    def _route(self, snapshot, evidence, goal, **kwargs):
        return goal

    _score = staticmethod(score_waypoint_alignment)

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        p = np.zeros((6, 8, 5)); p[:, :, 3] = 1.
        return self._score(p, goal_body, delay_ticks=3, commit_ticks=4), {}


class MetricScoringRuntime(MissionCoordinateMixin, ScoringBase):
    pass


@pytest.mark.parametrize('mode', ['original', 'consistent'])
def test_runtime_captures_planning_goal_and_clears_metric_after_selection(mode):
    runtime = MetricScoringRuntime(coordinate_mode=mode)
    B = rotation_y(.04); Q = rotation_y(.01) @ rotation_z(.6)
    snapshot = SimpleNamespace(map_from_initial=B)
    position = np.array([0., 0., .08]); goal = np.array([.022, 0.])
    runtime._route(snapshot, None, goal)
    goal[:] = [9., 9.]  # Simulate later mutation of the caller's mission goal.
    old_target = np.array([.018, .002])
    selected, _ = runtime._select_action(None, None, None, old_target, None, snapshot, B@position, Q)
    assert runtime._terminal_position_metric is None
    if mode == 'consistent':
        np.testing.assert_allclose(np.asarray(selected['position_metric_matrix']) @ selected['waypoint_body_xy_m'], [.022, 0.], atol=1e-15)
    else:
        np.testing.assert_array_equal(selected['waypoint_body_xy_m'], old_target)
        assert 'position_metric_matrix' not in selected
    runtime.terminal_position_approach = False
    inactive, _ = runtime._select_action(None, None, None, old_target, None, snapshot, B@position, Q)
    assert 'position_metric_matrix' not in inactive
    np.testing.assert_array_equal(inactive['waypoint_body_xy_m'], old_target)
