"""Explicit dependency isolation and preservation of selector policy/state."""
from copy import deepcopy
import ast
import inspect
import textwrap
import pytest
from lewm import receipt_copied_selector_development as copied
from lewm.tests.test_executed_waypoint_score_development import fixture


def test_only_declared_bindings_change_and_original_code_is_retained():
    assert len(copied.FORKS)==15
    for original,new,replacements in copied.FORKS:
        assert new.__code__ is original.__code__ and new.__closure__ is original.__closure__
        assert new.__defaults__ is original.__defaults__ and new.__kwdefaults__ is original.__kwdefaults__
        assert new.__globals__ is not original.__globals__
        assert set(new.__globals__)==set(original.__globals__)
        assert all(new.__globals__[k] is replacements.get(k,v) for k,v in original.__globals__.items())
        assert all(original.__globals__[k] is not v for k,v in replacements.items())
        if 'deepcopy' in replacements:
            tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='deepcopy']
            assert calls and all(len(n.args)==1 and not n.keywords for n in calls)


def test_super_calls_visit_each_fork_in_original_stage_order():
    stages=[copied.ViewReentrySelector,copied.RoundTripMissionSelector,
        copied.MissionTargetEightStepSelector,copied.MissionTargetWaypointSelector]
    new=[copied.ReceiptCopiedViewReentrySelector,copied.ReceiptCopiedRoundTripSelector,
        copied.ReceiptCopiedEightStepSelector,copied.ReceiptCopiedMissionTargetSelector]
    mro=copied.ReceiptCopiedViewReentrySelector.__mro__
    assert list(mro)==[c for pair in zip(new,stages,strict=True) for c in pair]+[object]
    selector=new[0](residual=object(),condition='jepa',variant='full',goal_initial_body_xy_m=[1.,2.])
    baseline=stages[0](residual=selector.residual,condition='jepa',variant='full',goal_initial_body_xy_m=[1.,2.])
    assert selector.__dict__.keys()==baseline.__dict__.keys()
    assert selector.head==baseline.head and selector.goal.tolist()==baseline.goal.tolist()
    selector.scan_index=baseline.scan_index=2
    for goal in ([1.,2.],[2.,1.]):
        selector.set_goal(goal); baseline.set_goal(goal)
        assert selector.scan_index==baseline.scan_index and selector.mode==baseline.mode


@pytest.mark.parametrize('bad',[False,True])
def test_waypoint_result_and_rejections_match_without_mutating_input(bad):
    selection,receipt=fixture()
    shared={'data':[1.,2.]}; selection['alias_a']=shared; selection['alias_b']=shared
    if bad: receipt['native_outcomes_used']=True
    before=deepcopy((selection,receipt))
    if bad:
        for fn in (copied.score_waypoint_execution,copied.copied_score_waypoint_execution):
            with pytest.raises(ValueError): fn(selection,receipt)
    else:
        original=copied.score_waypoint_execution(selection,receipt)
        new=copied.copied_score_waypoint_execution(selection,receipt)
        assert new==original
        assert new['alias_a'] is new['alias_b'] and new['alias_a'] is not shared
        new['alias_a']['data'][0]=9.
    assert (selection,receipt)==before


def test_candidate_controller_preserves_original_state_owners_and_methods():
    controller=copied.ReceiptCopiedMeasuredFloorController(object(),object(),public_mission=dict(
        goal_initial_body_xy_m=[1.,0.],return_initial_body_xy_m=[0.,0.],require_return_after_goal=True),
        navigation_ticks=40,condition='jepa',variant='full',persistent=True)
    assert controller.selector.residual is controller.residual and controller.memory is controller.mapper.surface
    for method in ('observe','advance','_result'):
        assert getattr(copied.ReceiptCopiedMeasuredFloorController,method) is getattr(copied.MeasuredFloorTransportController,method)
