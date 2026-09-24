import copy
import math

import pytest
import torch

from lewm.causal_sensor_state import SensorContractError
from lewm.exploration_local_bridge_development import ExplorationLocalBridge
from lewm.memory.observed_exploration_development import ObservedExploration,PlaceFix,ExitObservation
from lewm.online_temporal_choice_development import OnlineTemporalChoice
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from lewm.tests.test_online_rgb_history_development import stream


def example():
    rows=stream(); memory=ObservedExploration(); bridge=ExplorationLocalBridge(memory,OnlineTemporalChoice('always_stop',(),()),(0,0,0))
    for p in rows[:4]: bridge.observe(p,now_ns=p['image']['measured_ns'])
    now=rows[3]['image']['measured_ns']
    memory.observe_place(PlaceFix('fix',now,'A',True))
    memory.observe_exit(ExitObservation('exit',now,'A','observed-left',math.pi/2))
    return bridge,rows,now


def test_observed_exit_drives_real_local_adapter_without_creating_route():
    bridge,rows,now=example(); result=bridge.start_next('attempt',now_ns=now)
    choice=result['local_selection']
    assert choice['initial_direction_xy']==pytest.approx([0,.8])
    assert [r['measured_ns'] for r in choice['input_images']]==[p['image']['measured_ns'] for p in rows[:4]]
    assert choice['requested_command_tape']==[[0.,0.,0.]]*5
    assert bridge.memory.graph.places=={'A'} and bridge.memory.pending['action_id']=='attempt'
    assert bridge.memory.attempts==[]


def test_actual_arrival_evidence_finishes_attempt_and_fresh_fix_starts_next_frame():
    b,rows,now=example(); b.start_next('out',now_ns=now)
    p=rows[4]; t=p['image']['measured_ns']; b.observe(p,now_ns=t)
    b.finish(PlaceFix('arrive',t,'B',True),reached=True,viable_arrival=True)
    assert b.memory.graph.route('A','B')==['A','B'] and b.memory.graph.route('B','A') is None
    b.memory.observe_exit(ExitObservation('back',t,'B','new-exit',-math.pi/2))
    result=b.start_next('back',now_ns=t)
    assert result['local_selection']['initial_direction_xy']==pytest.approx([0,-.8])
    assert result['local_selection']['decision_index']==0
    assert result['local_selection']['orientation']['start_ns']==t


def test_missing_association_returns_observation_action_without_inference():
    b,_,now=example(); b.memory.observe_place(PlaceFix('ambiguous',now,None,True))
    assert b.start_next('not-started',now_ns=now)=={'memory_decision':{'kind':'LOCALIZE'},'local_selection':None}
    assert b.active is None and b.memory.pending is None


def test_old_bearing_cannot_drive_new_packet_without_orientation_transport():
    b,rows,now=example(); p=rows[4]; b.observe(p,now_ns=p['image']['measured_ns'])
    with pytest.raises(SensorContractError): b.start_next('stale-bearing',now_ns=p['image']['measured_ns'])
    assert b.fault and b.memory.pending is None


def test_fault_latches_requires_explicit_failed_terminal_and_never_qualifies_edge():
    b,rows,now=example(); b.start_next('out',now_ns=now)
    p=copy.deepcopy(rows[4]); t=p['image']['measured_ns']; p['world_pose']=[0]*7
    with pytest.raises(SensorContractError): b.observe(p,now_ns=t)
    with pytest.raises(SensorContractError): b.select_active(now_ns=t)
    with pytest.raises(ValueError): b.finish(PlaceFix('false-arrival',t,'B',True),reached=True,viable_arrival=True)
    b.finish(PlaceFix('failed',t,None,False),reached=False,viable_arrival=False)
    assert b.memory.graph.places=={'A'} and not b.memory.attempts[0]['qualified'] and b.fault


def test_wrong_selection_cadence_fails_without_silently_reusing_command():
    b,_,now=example(); b.start_next('out',now_ns=now)
    with pytest.raises(SensorContractError): b.select_active(now_ns=now)
    assert b.fault


def test_active_execution_cannot_be_replaced_by_another_frontier():
    b,_,now=example(); b.start_next('out',now_ns=now)
    with pytest.raises(ValueError): b.start_next('replacement',now_ns=now)
    assert b.memory.pending['action_id']=='out'


@pytest.mark.parametrize('method',['direct_direct','supervised_rollout','jepa_rollout'])
def test_learned_head_path_uses_all_three_members_and_causal_five_candidate_plan(method):
    b,_,now=example(); torch.manual_seed(23)
    b.template=OnlineTemporalChoice(method,[TemporalRGBBodyJEPA() for _ in range(3)],[{'synthetic_seed':i} for i in range(3)])
    result=b.start_next('learned-path',now_ns=now)['local_selection']
    assert len(result['member_predictions'])==3 and len(result['candidate_costs'])==5
    assert all(math.isfinite(v) for v in result['candidate_costs'])
    assert 0<=result['selected_action_index']<5 and result['branch_ticks']==5
    assert result['decision_ns']==now and b.memory.attempts==[]
