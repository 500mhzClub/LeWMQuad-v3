import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lewm import mission_target_waypoint_selection_development as current
from lewm import exact_mission_target_waypoint_selection_development as prior
from lewm import mission_target_eight_step_selection_development as eight
from lewm import exact_mission_target_goal_probe_development as prior_eight
from lewm.observed_floor_waypoint_development import segment_cells


def choose_body(source, cls):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
    return ast.dump(next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'choose'))


def test_only_explicit_mission_coordinate_sources_change_route_and_eight_step_logic():
    source = Path(current.__file__).read_text().replace('self.goal.copy()', '[1.2,0.]')
    source = source.replace('np.r_[self.goal,0.]', 'np.array([1.2,0.,0.])')
    assert choose_body(source, 'MissionTargetWaypointSelector') == choose_body(
        Path(prior.__file__).read_text(), 'ExactMissionTargetWaypointSelector')
    assert choose_body(Path(eight.__file__).read_text(), 'MissionTargetEightStepSelector') == choose_body(
        Path(prior_eight.__file__).read_text(), 'ExactMissionTargetEightStepSelector')


def test_instruction_reaches_measured_mapper_and_local_model_target_without_scene_input(monkeypatch):
    goal = [.2, .1]; calls = {}
    selector = current.MissionTargetWaypointSelector(condition='jepa', variant='full', goal_initial_body_xy_m=goal)
    def waypoint(value, **kwargs):
        calls['mapper_goal'] = value.copy()
        return dict(status='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL', route_cells=[[4, 2]], unknown_connector_cells=[])
    mapper = SimpleNamespace(map_from_initial=np.eye(3),
        surface=SimpleNamespace(rotation=np.eye(3), position=np.zeros(3)), waypoint=waypoint,
        floor=segment_cells([0., 0.], goal), occupied=set())
    def select(model, history, **kwargs):
        calls['model_target'] = kwargs['goal_body_xy_m'].copy()
        calls['model_keys'] = set(kwargs)
        return dict(action='hold', selection_wall_ms=0.)
    monkeypatch.setattr(current, 'select', select)
    monkeypatch.setattr(current, 'filter_selection', lambda s, *a, **k: s)
    monkeypatch.setattr(current, 'restrict', lambda s, *a: s)
    r = selector.choose(object(), {}, mapper, object(), now_ns=1)
    np.testing.assert_array_equal(calls['mapper_goal'], goal)
    np.testing.assert_array_equal(calls['model_target'], goal)
    assert calls['model_keys'] == {'head', 'input_variant', 'goal_body_xy_m', 'contact_penalty_m'}
    assert r['waypoint_map_xy_m'] == goal and r['intermediate_target_is_mission_goal']


def test_target_change_resets_only_view_search_and_input_arrays_are_owned():
    goal = [.2, .1]
    s = current.MissionTargetWaypointSelector(condition='direct', variant='full', goal_initial_body_xy_m=goal)
    goal[0] = 4.
    assert s.goal.tolist() == [.2, .1]
    s.scan_sign = 1; s.scan_index = 3; s.scan_target = 1.; s.mode = 'VIEW_ACQUISITION'
    s.set_goal([.2, .1]); assert s.scan_index == 3
    s.set_goal([0., 0.]); assert s.scan_index == 0 and s.scan_target is None and s.scan_sign is None
    assert s.mode == 'NEW'
    for bad in ([5., 0.], [float('nan'), 0.], [1., 0., 0.]):
        with pytest.raises(ValueError): s.set_goal(bad)
    assert s.goal.tolist() == [0., 0.]
