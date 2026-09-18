import numpy as np
import pytest
from scripts.diagnose_go2_recent_qualified_maze03_frontier_stagnation_v1 import motion_summary, selection_summary, ACTIONS


def test_yaw_unwrap_separates_rotation_from_translation():
    angles = np.linspace(0, 6*np.pi, 301); poses = np.zeros((301, 7))
    poses[:, 5] = np.sin(angles/2); poses[:, 6] = np.cos(angles/2)
    report = motion_summary(poses)
    assert report['signed_yaw_revolutions'] == pytest.approx(3)
    assert report['absolute_sampled_yaw_revolutions'] == pytest.approx(3)
    assert report['net_displacement_m'] == report['sampled_xy_path_length_m'] == 0
    poses[:, 0] = np.sin(angles)
    report = motion_summary(poses)
    assert report['maximum_displacement_from_segment_start_m'] == pytest.approx(1)
    assert report['sampled_xy_path_length_m'] > 11.9 and report['net_displacement_m'] < 1e-12


def test_one_pose_has_no_invented_rotation_or_distance():
    report = motion_summary([[1, 2, 0, 0, 0, 0, 1]])
    assert report['samples'] == 1 and report['signed_yaw_revolutions'] == report['sampled_xy_path_length_m'] == 0
    with pytest.raises(ValueError): motion_summary([[1, 2, 0, 0, 0, 0, 0]])


def test_saved_candidate_gates_do_not_treat_score_preference_as_infeasibility():
    selection = dict(action='left_turn', mode='WAYPOINT', prediction=[], phase_allowed_actions=list(ACTIONS),
        candidates=[dict(action=a, utility_m=float(i)) for i,a in enumerate(ACTIONS)],
        nominal_path_checks=[dict(action=a, all_predicted_segments_nominally_clear=True) for a in ACTIONS],
        surface_checks=[dict(possible_intersection=False) for a in ACTIONS],
        proposal=dict(status='OBSERVED_FLOOR_ROUTE_TO_FRONTIER',route_cells=[[1, 0]],observed_floor_cells=5,occupied_cells=1),
        waypoint_map_xy_m=[.075, .025], goal_body_xy_m=[.01, 0.], view_budget_exhausted=False)
    report = selection_summary(selection)
    assert report['route_cell_count'] == 1 and report['waypoint_distance_m'] == .01
    assert report['candidates'][1]['eligible_by_saved_gates'] and report['action'] == 'left_turn'
    selection['surface_checks'][1]['possible_intersection'] = True
    assert not selection_summary(selection)['candidates'][1]['eligible_by_saved_gates']
    selection['candidates'].reverse()
    with pytest.raises(ValueError): selection_summary(selection)
