"""Inner-region consistency and matched-plant reachability, not physics proof."""
import math
import numpy as np
import pytest
from lewm.inner_goal_pulse_rollout_development import plan,INTERNAL_POSITION_TOLERANCE_M
from lewm.inner_goal_pulse_feedback_development import InnerGoalPulseServo
from lewm.inner_goal_room_return_development import InnerGoalPulseExecution,InnerGoalRoomReturn
from lewm.coupled_pulse_rollout_development import plan as old_plan,compose
from lewm.coupled_pulse_feedback_development import CoupledPulseServo
from lewm.sensor_anchored_goal_development import AnchoredGoal
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_coupled_pulse_rollout_development import table
from scripts.fixed_nominal_pulse_table_development import load_fixed_table


def test_planner_does_not_stop_in_old_outer_region():
    assert INTERNAL_POSITION_TOLERANCE_M==.04
    t=table();old=old_plan(t,[0,0,0],[.05,0,0]);new=plan(t,[0,0,0],[.05,0,0])
    assert old['action_indices']==[]
    assert new['action_indices'] and new['predicted_position_error_m']<=.04
    assert not new['uncertainty_calibrated'] and not new['motion_permission']


@pytest.mark.parametrize('distance',[.0400001,.05,.06])
def test_visual_settling_cannot_complete_in_outer_region(distance):
    e,t=visual(0);g=AnchoredGoal.from_observation(e,[distance,0],0.,identity=(0,0,0),now_ns=t)
    model=InnerGoalPulseServo(g,table());commands=[]
    for tick in range(1,33):
        e,t=visual(tick);r=model.step(e,now_ns=t);commands.append(r['requested_command'])
        assert r['terminal']!='VISUAL_TARGET_SEQUENCE_COMPLETE'
    assert any(any(c) for c in commands)


def test_actual_inner_goal_still_requires_ten_new_quiet_intervals():
    e,t=visual(0);g=AnchoredGoal.from_observation(e,[0,0],0.,identity=(0,0,0),now_ns=t)
    model=InnerGoalPulseServo(g,table())
    for tick in range(1,30):
        e,t=visual(tick);assert model.step(e,now_ns=t)['terminal'] is None
    e,t=visual(30);assert model.step(e,now_ns=t)['terminal']=='VISUAL_TARGET_SEQUENCE_COMPLETE'


def test_outer_controller_never_steps_after_transactional_admission(monkeypatch):
    monkeypatch.setattr(CoupledPulseServo,'step',lambda *a,**k:pytest.fail('old servo stepped'))
    model=InnerGoalPulseExecution(table());e,t=visual(0);model.observe(e,now_ns=t);model.begin([.05,0],0.,now_ns=t)
    assert isinstance(model.active,InnerGoalPulseServo)
    e,t=visual(1);assert model.observe(e,now_ns=t)['status']=='EXECUTING'


@pytest.mark.parametrize('sign',[-1,1])
@pytest.mark.parametrize('empirical',[False,True])
def test_complete_matched_plant_keeps_inner_region_for_every_clipped_leg(sign,empirical):
    tab=load_fixed_table() if empirical else table();model=InnerGoalRoomReturn(sign,tab)
    state=np.zeros(3);increment=np.zeros(3)
    for tick in range(3601):
        e,t=visual(tick,*state);ex=model.runtime.executor.observe(e,now_ns=t)
        r=model.advance(dict(evidence=e,execution=ex,requested_command=ex['requested_command']),now_ns=t)
        if r['terminal']:break
        local=ex['local_decision']
        if local and 'new_pulse' in local['diagnostic']:
            effect=tab.effects[local['diagnostic']['new_pulse']['action_index']]
            increment=(compose(state,effect.delta_xy_yaw)-state)/effect.ticks
        if any(r['requested_command']):state+=increment
    assert r['terminal']=='ROOM_RETURN_CANDIDATE',r
    assert len(model.completed)==7 and np.linalg.norm(state[:2])<=.04
    for leg in model.runtime.executor.legs:
        d=leg['final_decision']
        assert d['diagnostic']['position_error_m']<=.04 and d['internal_position_tolerance_m']==.04
    assert model.runtime.executor.start_ns==1_600_000_000
    assert not model.snapshot()['home_verified']


@pytest.mark.parametrize('sign',[-1,1])
def test_inner_planner_preserves_signed_turn_winding(sign):
    r=plan(load_fixed_table(),[0,0,0],[0,0,sign*math.pi],yaw_mode='winding')
    assert r['status']=='PREDICTED_GOAL_CANDIDATE'
    assert r['predicted_position_error_m']<=.04 and abs(r['predicted_endpoint'][2]-sign*math.pi)<=.05


def test_current_pose_fault_and_outer_budget_still_latch_zero():
    m=InnerGoalPulseExecution(table(),maximum_pulses=1);e,t=visual(0);m.observe(e,now_ns=t);m.begin([.4,0],0.,now_ns=t)
    for tick in range(1,100):
        e,t=visual(tick);r=m.observe(e,now_ns=t)
        if r['status']=='FAILED':break
    assert r['status']=='FAILED' and not any(r['requested_command']) and m.pulses==1
    e,t=visual(100);e['current_pose']=None
    assert m.observe(e,now_ns=t)['status']=='FAILED'
