from threading import Lock
from queue import Queue
from types import SimpleNamespace
import pytest
from lewm.floor_reacquisition_development import (
    FloorReacquisitionRuntimeMixin,ReacquiringFloorMission,ReacquiringFloorRegistration,
    RobustHeightFloorRegistration,CONFLICT,UNAVAILABLE,WARMUP_TICKS)


def mission(budget=100):
    return ReacquiringFloorMission(dict(goal_initial_body_xy_m=[1.,0.],
        return_initial_body_xy_m=[0.,0.],require_return_after_goal=True),
        navigation_ticks=budget,arrival_radius_m=.02)


def test_missing_pose_resets_dwell_without_inventing_motion_or_extending_budget():
    m=mission(30)
    for f in range(10):
        m.advance([1.,0.,0.],frame=f,now_ns=1_500_000_000+f*100_000_000,
            previous_requested_command=[0.,0.,0.])
    assert m.quiet>0 and not m.arrivals
    r=m.unavailable(frame=10,now_ns=2_500_000_000,previous_requested_command=[0.,0.,0.])
    assert r['observed_goal_distance_m'] is None and r['hold_required']
    assert m.quiet==0 and m.previous_visual_position is None and not m.arrivals
    r=m.advance([1.,0.,0.],frame=11,now_ns=2_600_000_000,previous_requested_command=[0.,0.,0.])
    assert r['observed_settling']['observed_interval_speed_m_s'] is None
    assert not r['arrival_confirmed_this_frame'] and m.quiet==0
    for f in range(12,WARMUP_TICKS+30+1):
        r=m.unavailable(frame=f,now_ns=1_500_000_000+f*100_000_000,
            previous_requested_command=[0.,0.,0.])
    assert r['terminal']=='MISSION_TICK_BUDGET_EXHAUSTED' and not m.arrivals


def test_only_expected_floor_conflict_is_recoverable(monkeypatch):
    r=ReacquiringFloorRegistration()
    anchor={'current_pose':{'frame':4}};reference={'fixed':True}
    r.anchor=anchor;r.reference=reference;r.frame=4
    def conflict(self,*args,**kwargs):
        self.failed=True
        raise ValueError(CONFLICT)
    monkeypatch.setattr(RobustHeightFloorRegistration,'observe',conflict)
    value=r.observe(None,None,None,{'current_pose':{'frame':5}},now_ns=2_000_000_000)
    assert value['status']==UNAVAILABLE and value['current_pose'] is None
    assert r.anchor is anchor and r.reference is reference and r.frame==5 and not r.failed
    def invalid(self,*args,**kwargs):
        self.failed=True
        raise ValueError('wrong camera timestamp')
    monkeypatch.setattr(RobustHeightFloorRegistration,'observe',invalid)
    with pytest.raises(ValueError,match='wrong camera timestamp'):
        r.observe(None,None,None,{'current_pose':{'frame':6}},now_ns=2_100_000_000)
    assert r.failed


class Base:
    def _command_gate(self,result,now):return result
    def _store_plan(self,*args):self.stored=True
    def _plan(self,item):self.planned=True


class Runtime(FloorReacquisitionRuntimeMixin,Base):
    pass


def test_hold_cancels_motion_and_rejects_pre_gap_inflight_plans():
    r=Runtime.__new__(Runtime);r.lock=Lock();r.floor_waiting=True
    r.floor_plan_minimum_ns=2_400_000_000;r.planning=[{}]
    r.stored=False;r.planned=False
    assert r._command_gate({'requested_command':[.2,0.,0.]},2_000_000_000)['requested_command']==[0.,0.,0.]
    r._store_plan(SimpleNamespace(observed_ns=2_000_000_000),0,[])
    assert not r.stored and r.planning[-1]['committed'] is False
    r.floor_waiting=False;r.planning.append({})
    r._store_plan(SimpleNamespace(observed_ns=2_000_000_000),0,[])
    assert not r.stored
    r._plan((SimpleNamespace(frame=8,measured_ns=2_300_000_000),None))
    assert not r.planned
    r._store_plan(SimpleNamespace(observed_ns=2_400_000_000),0,[])
    assert r.stored


def test_rejected_frame_never_reaches_mapping_or_pose_history_and_four_valid_frames_resume():
    r=Runtime.__new__(Runtime);r.lock=Lock();r.mission=mission()
    r.floor_waiting=False;r.floor_valid_streak=0;r.floor_plan_minimum_ns=-1
    r.frame_request_history={};r.mission_rows=[];r.mission_generation=0
    r.plans=[];r.rejected_windows={};r.clock_ns=lambda:3_000_000_000
    r.queues={'mapping':Queue(),'planning':Queue()};published=[]
    r.evidence_sink=lambda f,raw,evidence:published.append(f)
    for frame in range(9):
        now=1_500_000_000+frame*100_000_000
        packet=SimpleNamespace(frame=frame,measured_ns=now,policy={},depth={},auxiliary_depth={})
        r.frame_request_history[frame]=([0.,0.,0.],[[0.,0.,0.]]*5)
        e=({'status':UNAVAILABLE,'current_pose':None} if frame==4 else
            {'status':'CURRENT_FLOOR_REGISTERED_POSE','current_pose':
                {'frame':frame,'position_initial_body_m':[0.,0.,0.]}})
        r.registration=SimpleNamespace(observe=lambda *args,**kwargs:e)
        r._register((packet,{}))
        if 4<=frame<8:assert r.floor_waiting
    assert not r.floor_waiting
    assert published==[0,1,2,3,5,6,7,8]
    assert [r.queues['mapping'].get()[0].frame for _ in range(r.queues['mapping'].qsize())]==[0,8]
    assert r.mission_rows[4]['consumed_pose_frame'] is None
    assert r.mission_rows[5]['observed_settling']['previous_position_initial_body_m'] is None
