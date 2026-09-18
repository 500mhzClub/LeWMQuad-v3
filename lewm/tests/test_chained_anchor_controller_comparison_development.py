"""Controller integration and causal comparison scope at a live bridge boundary."""
from copy import deepcopy

import pytest

from lewm.chained_anchor_residual_controller_development import ChainedAnchorResidualController, CONTROLLER, FLAG
from lewm.chained_anchor_visual_motion_development import ChainedAnchorVisualMotion
from lewm.direct_flow_residual_anchored_controller_development import DirectFlowResidualAnchoredController
from lewm.novel_maze_round_trip_scene_development import public_mission
from scripts.chained_anchor_controller_comparison_development import compare, ORIGINAL, ORIGINAL_FLAG


def rows(frame=3):
    original = dict(controller=ORIGINAL, tick=frame, terminal=None, failure=None,
        requested_command=[0., 0., 0.], new_selection={'prediction': {'x': [1]}, 'action': None},
        original_visual_evidence={'status': 'CURRENT_VISUAL_POSE'}, evidence={'pose': 'registered'},
        **{ORIGINAL_FLAG: True})
    candidate = deepcopy(original)
    candidate.update(controller=CONTROLLER, **{FLAG: True})
    return original, candidate


def boundary():
    old, new = rows(853)
    old['original_visual_evidence'].update(continuity_evidence=dict(status='MEASURED_INCREMENT_BRIDGE', bridge_frames=1),
                                         camera_selection={'selected_camera': 'auxiliary'})
    raw = deepcopy(old['original_visual_evidence'])
    raw.update(chained_anchor_fallback=dict(accepted=True, original_bridge_available=True,
        original_continuity=deepcopy(raw['continuity_evidence']), original_camera_selection=deepcopy(raw['camera_selection']),
        original_direct_flow_fallback=None), continuity_evidence={'status': 'ANCHOR_MEASUREMENT'})
    new['original_visual_evidence'] = raw
    return old, new


def check(old, new, frame=853):
    return compare(old, new, old['requested_command'], new['original_visual_evidence'], frame=frame, boundary=853)


def test_integration_changes_only_observer_and_declared_result_identity():
    model, geometry = object(), object()
    options = dict(public_mission=public_mission(2), navigation_ticks=3000, condition='jepa', variant='no_rgb', persistent=True)
    old = DirectFlowResidualAnchoredController(model, geometry, **options)
    new = ChainedAnchorResidualController(model, geometry, **options)
    assert type(new.motion) is ChainedAnchorVisualMotion
    assert new.model is old.model is model
    assert new.observe.__func__ is old.observe.__func__ and new.advance.__func__ is old.advance.__func__
    for name in ('selector', 'registration', 'mapper', 'memory', 'mission', 'residual'):
        assert type(getattr(new, name)) is type(getattr(old, name))
        assert getattr(new, name) is not getattr(old, name)
    result = new.observe({}, {}, {}, now_ns=1)
    assert result['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and result['requested_command'] == [0., 0., 0.]
    assert result[FLAG] and result[ORIGINAL_FLAG] and result['controller'] == CONTROLLER
    assert new.advance({}, {}, now_ns=2)['terminal'] == result['terminal']


def test_early_whole_decision_and_forecast_must_match():
    old, new = rows()
    result = check(old, new, 3)
    assert result['original_forecast_compared'] and not result['stop']
    new['new_selection']['prediction']['x'][0] = 2
    with pytest.raises(ValueError, match='complete original decision'): check(old, new, 3)


def test_anchor_admission_is_not_labelled_navigation_or_controller_failure_recovery():
    old, new = boundary()
    result = check(old, new)
    assert result['stop'] and result['controller_admitted_reacquired_pose']
    assert not result['original_controller_failed'] and not result['navigation_recovered']
    assert not result['requested_command_changed']


def test_changed_action_request_is_explicit_and_stops():
    old, new = boundary()
    new['new_selection']['action'] = 'right_turn'
    new['requested_command'] = [0., 0., -.45]
    result = check(old, new)
    assert result['stop'] and result['requested_command_changed']


@pytest.mark.parametrize('change', [{'evidence': None}, {'new_selection': None}, {'tick': 852}])
def test_partial_controller_admission_rejected(change):
    old, new = boundary()
    new.update(change)
    with pytest.raises(ValueError, match='registered current pose'): check(old, new)


@pytest.mark.parametrize('key', ['original_continuity', 'original_camera_selection', 'original_direct_flow_fallback'])
def test_original_bridge_evidence_cannot_be_rewritten(key):
    old, new = boundary()
    new['original_visual_evidence']['chained_anchor_fallback'][key] = {'changed': True}
    with pytest.raises(ValueError, match='original bridge'): check(old, new)


def test_unaccepted_anchor_cannot_be_reported_as_controller_admission():
    old, new = boundary()
    new['original_visual_evidence']['chained_anchor_fallback']['accepted'] = False
    with pytest.raises(ValueError, match='original bridge'): check(old, new)


def test_floor_or_controller_failure_is_a_preserved_negative_result():
    old, new = boundary()
    new.update(terminal='SENSOR_OR_MODEL_FAILURE', failure='floor failed', evidence=None)
    result = check(old, new)
    assert result['stop'] and not result['controller_admitted_reacquired_pose']
    new['requested_command'] = [.2, 0., 0.]
    with pytest.raises(ValueError, match='terminal zero'): check(old, new)


def test_full_observer_evidence_must_equal_completed_replay():
    old, new = boundary()
    with pytest.raises(ValueError, match='authenticated observer'):
        compare(old, new, old['requested_command'], {}, frame=853, boundary=853)


def test_original_live_controller_cannot_be_changed_to_failure():
    old, new = boundary()
    old['terminal'] = 'SENSOR_OR_MODEL_FAILURE'
    with pytest.raises(ValueError, match='original live'): check(old, new)


def test_command_must_match_forecast_action():
    old, new = boundary()
    new['requested_command'] = [.2, 0., 0.]
    with pytest.raises(ValueError, match='selected action'): check(old, new)


@pytest.mark.parametrize('frame,boundary_frame', [(True, 853), (854, 853), (-1, 853), (3, True), (3, 864)])
def test_unbounded_or_ambiguous_frames_rejected(frame, boundary_frame):
    old, new = rows()
    with pytest.raises(ValueError, match='bounded intervention'):
        compare(old, new, old['requested_command'], new['original_visual_evidence'], frame=frame, boundary=boundary_frame)
