"""Reject changed early behavior and incomplete claims of boundary recovery."""
from copy import deepcopy
import pytest
from scripts.replay_go2_no_rgb_jepa_direct_flow_controller_prefix_v1 import compare, CONTROLLER, FLAG
from lewm.direct_flow_residual_anchored_controller_development import DirectFlowResidualAnchoredController
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationController
from lewm.direct_flow_floor_transport_controller_development import DirectFlowDualCameraVisualMotion
from lewm.novel_maze_round_trip_scene_development import public_mission


def early():
    original = dict(controller='residual_anchored_continuation_controller_v1',tick=3,
        terminal=None,failure=None,requested_command=[0.,0.,0.],new_selection={'prediction':{'x':[1]}},
        original_visual_evidence={'status':'CURRENT_VISUAL_POSE'},evidence={'pose':'registered'})
    candidate = deepcopy(original);candidate.update(controller=CONTROLLER,**{FLAG:True})
    return original,candidate


def boundary():
    original,candidate = early();original.update(tick=858,terminal='SENSOR_OR_MODEL_FAILURE',failure='missing',new_selection=None,evidence=None)
    original['original_visual_evidence'] = dict(status='VISUAL_TERMINAL_FAILURE',camera_selection={'a':1},
        continuity_evidence={'b':2},reference_selection={'c':3})
    candidate.update(tick=859,new_selection={'prediction':{'x':[1]},'action':None})
    candidate['original_visual_evidence'] = dict(status='CURRENT_VISUAL_POSE',direct_corner_flow_fallback=dict(
        accepted=True,original_camera_selection={'a':1},original_auxiliary_continuity={'b':2},original_reference_selection={'c':3}))
    return original,candidate


def test_complete_early_decision_and_prediction_are_compared():
    a,b=early();r=compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=3)
    assert r['original_forecast_compared'] and not r['stop']
    b['new_selection']['prediction']['x'][0]=2
    with pytest.raises(ValueError,match='complete original'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=3)


def test_raw_observer_mismatch_is_never_normalized_away():
    a,b=boundary()
    with pytest.raises(ValueError,match='observer evidence'):compare(a,b,a['requested_command'],{},frame=859)


def test_recovered_boundary_always_stops_even_with_same_zero_request():
    a,b=boundary();r=compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=859)
    assert r['stop'] and r['full_controller_recovered'] and not r['requested_command_changed']


@pytest.mark.parametrize('change',[{'evidence':None},{'new_selection':None},{'tick':858}])
def test_partial_recovery_cannot_claim_full_controller_recovery(change):
    a,b=boundary();b.update(change)
    with pytest.raises(ValueError,match='registered pose'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=859)


def test_wrong_executed_original_command_is_rejected():
    a,b=early()
    with pytest.raises(ValueError,match='executed command'):compare(a,b,[.2,0.,0.],b['original_visual_evidence'],frame=3)


def test_wrong_new_action_command_is_rejected():
    a,b=boundary();b['requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError,match='selected action'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=859)


def test_floor_failure_is_a_preserved_negative_result():
    a,b=boundary();b.update(terminal='SENSOR_OR_MODEL_FAILURE',failure='floor registration failed',evidence=None)
    r=compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=859)
    assert r['stop'] and not r['full_controller_recovered']
    b['requested_command']=[.2,0.,0.]
    with pytest.raises(ValueError,match='zero command'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=859)


@pytest.mark.parametrize('key',['original_camera_selection','original_auxiliary_continuity','original_reference_selection'])
def test_original_failure_evidence_cannot_be_dropped(key):
    a,b=boundary();b['original_visual_evidence']['direct_corner_flow_fallback'][key]={}
    with pytest.raises(ValueError,match='original failure'):compare(a,b,a['requested_command'],b['original_visual_evidence'],frame=859)


def test_integration_preserves_original_planning_and_registration_classes():
    options=dict(public_mission=public_mission(2),navigation_ticks=3000,condition='jepa',variant='no_rgb',persistent=True)
    model=object();geometry=object()
    original=ResidualAnchoredContinuationController(model,geometry,**options)
    candidate=DirectFlowResidualAnchoredController(model,geometry,**options)
    assert type(candidate.motion) is DirectFlowDualCameraVisualMotion
    for name in ('selector','registration','mapper','memory','mission','residual'):
        assert type(getattr(original,name)) is type(getattr(candidate,name))
        assert getattr(original,name) is not getattr(candidate,name)
    assert candidate.model is original.model is model
    assert candidate.observe.__func__ is original.observe.__func__
    assert candidate.advance.__func__ is original.advance.__func__
