"""Causal boundary and unchanged-state checks for the actual-packet memory replay."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from lewm.current_observation_planning_prefix_development import METADATA,SHARED,compare_step
from scripts import replay_go2_current_observation_planning_prefix_v1 as runner


def decisions(frame=3):
    old={k:None for k in SHARED}
    old.update(tick=frame,controller='dual_camera_settled_round_trip_controller_v1',
        requested_command=[0.,0.,0.],terminal=None,failure=None,new_selection=None,
        causal_residual_receipt=dict(frame=frame,pending_forecast_tick=frame,
            residuals=[],correction_xy_m=[0.,0.],native_outcomes_used=False))
    candidate=deepcopy(old)|METADATA|dict(controller='current_observation_planning_round_trip_controller_v1',
        floor_transport_during_missingness_enabled=True)
    return old,candidate


@pytest.mark.parametrize('fault',[None,'metadata','map','mission','residual','old_command','forecast','view','pending'])
def test_exact_shared_state_and_forecasts_required_but_planning_choice_can_change(fault):
    old,new=decisions()
    old['new_selection']={'prediction':[[1.]]}
    new['new_selection']=dict(prediction=[[1.]],planning_map_receipt=dict(frame=3,measured_ns=1_800_000_000,
        planning_map_variant='current_paired_observation',accumulated_planning_cells_queried=False,
        persistent_contact_history_retained=True))
    new['requested_command']=[.2,0.,0.]
    if fault=='metadata':new['persistent_contact_history_retained']=False
    elif fault=='map':new['memory_receipt']={'changed':True}
    elif fault=='mission':new['mission_receipt']={'changed':True}
    elif fault=='residual':new['causal_residual_receipt']['correction_xy_m']=[.1,0.]
    elif fault=='old_command':old['requested_command']=[.1,0.,0.]
    elif fault=='forecast':new['new_selection']['prediction']=[[2.]]
    elif fault=='view':new['new_selection']['planning_map_receipt']['frame']=2
    elif fault=='pending':new['causal_residual_receipt']['pending_forecast_tick']=4
    before=deepcopy((old,new))
    if fault is None:
        check=compare_step(old,new,[0.,0.,0.],frame=3)
        assert check['requested_command_changed'] and check['raw_model_forecasts_exact']
        assert check['unchanged_observed_and_executed_residual_state_exact']
    else:
        with pytest.raises(ValueError):compare_step(old,new,[0.,0.,0.],frame=3)
    assert (old,new)==before


def test_only_validated_mission_wording_and_current_terminal_pending_change_allowed():
    old,new=decisions()
    old['mission_receipt']={'observed_settling':{'motion_source':'consecutive_admitted_floor_registered_visual_positions'}}
    new['mission_receipt']={'observed_settling':{'motion_source':'consecutive_admitted_visual_positions_in_floor_reference'}}
    new['terminal']='VIEW_BUDGET_EXHAUSTED';new['causal_residual_receipt']['pending_forecast_tick']=None
    assert compare_step(old,new,[0.,0.,0.],frame=3)['terminal_changed']
    new['mission_receipt']['observed_settling']['motion_source']='command_integrated_pose'
    with pytest.raises(ValueError):compare_step(old,new,[0.,0.,0.],frame=3)


@pytest.mark.parametrize('boundary',['command','terminal','limit','mutated_input','model_change'])
def test_replay_never_consumes_observation_after_changed_request(monkeypatch,tmp_path,boundary):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda *a:SimpleNamespace(free=100*1024**3))
    weight={'value':0};model=SimpleNamespace(state_dict=lambda:dict(weight),parameters=lambda:[])
    monkeypatch.setattr(runner,'load_assigned',lambda *a:(model,runner.CASE[3],runner.CASE[2]))
    monkeypatch.setattr(runner,'state_digest',lambda d:str(d['value']))
    packet_reads=[];decision_reads=[]
    def rows(path):
        for i in range(64):
            decision_reads.append(i)
            if boundary in ('command','terminal') and i>3:pytest.fail('consumed a post-intervention decision')
            old,_=decisions(i)
            yield dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old)
        pytest.fail('consumed beyond fixed64frame limit')
    monkeypatch.setattr(runner,'read_rows',rows)
    class Reader:
        frames=list(range(100))
        def packet(self,i):
            packet_reads.append(i)
            if boundary in ('command','terminal') and i>3:pytest.fail('consumed post-intervention packet')
            return {'tick':i},{},{},1_500_000_000+i*100_000_000
    monkeypatch.setattr(runner,'IntentReturnRGBDReplay',lambda *a:Reader())
    tape=[dict(requested_command=[0.,0.,0.],completed=True) for _ in range(99)]
    monkeypatch.setattr(runner,'read_json',lambda p,n:tape if n=='command_tape.json' else [{}]*100)
    monkeypatch.setattr(runner,'public_acquisition',lambda r:r)
    monkeypatch.setattr(runner,'packet',lambda *a,**k:({},{}))
    class Controller:
        def observe(self,policy,*a,**k):
            _,result=decisions(policy['tick'])
            if policy['tick']==3:
                if boundary=='command':result['requested_command']=[.2,0.,0.]
                elif boundary=='terminal':result['terminal']='VIEW_BUDGET_EXHAUSTED'
                elif boundary=='model_change':weight['value']=1
                elif boundary=='mutated_input':policy['changed']=True
            return result
    monkeypatch.setattr(runner,'CurrentObservationPlanningController',lambda *a,**k:Controller())
    launch=dict(correction_admission={},model_state_sha256='0')
    if boundary in ('mutated_input','model_change'):
        with pytest.raises(ValueError,match='mutated public' if boundary=='mutated_input' else 'unchanged weights'):
            runner.replay(launch)
    else:
        report=runner.replay(launch)
        count=64 if boundary=='limit' else 4
        assert packet_reads==decision_reads==list(range(count))
        assert report['frames']==count and not report['following_recorded_observations_consumed']
        assert not report['unexecuted_outcomes_inferred'] and report['model_state_unchanged']
        assert report['stopped_at_first_command_or_terminal_difference']==(boundary!='limit')
        assert report['first_requested_command_difference']==(3 if boundary=='command' else None)
        assert report['first_terminal_policy_difference']==(3 if boundary=='terminal' else None)
