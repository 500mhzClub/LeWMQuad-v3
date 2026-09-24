"""Preserve complete causal state and exclude observations after intervention."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.tests.test_residual_anchored_continuation_development import fixture, apply
from lewm.residual_anchored_continuation_prefix_development import compare_step, MAX_FRAMES
from scripts import replay_go2_residual_anchored_continuation_prefix_v1 as runner


def decisions(frame=10, changed=False):
    selection, receipt, mapper = fixture()
    selection['view_budget_exhausted'] = False
    old = dict(tick=frame, controller='residual_first_interval_feasibility_controller_v1',
        residual_first_interval_feasibility_fallback_enabled=True,
        requested_command=[0.,0.,0.], terminal=None, failure=None, selected_action='hold', plan_offset=1,
        infeasible_wait_active=False, consecutive_infeasible_observations=0, feasible_action_recoveries=1,
        evidence={'pose':[1.,2.]}, mission_receipt={'phase':'OUTBOUND'}, memory_receipt={'cells':17},
        causal_residual_receipt=deepcopy(receipt), new_selection=selection)
    new = deepcopy(old)
    new.update(controller='residual_anchored_continuation_controller_v1', residual_hold_feasibility_enabled=True,
        residual_anchored_continuation_enabled=True)
    if changed:
        assert frame == 10
        new['new_selection'] = apply(selection, receipt, mapper)
        new['requested_command'] = new['new_selection']['requested_command']
        new['selected_action'] = new['new_selection']['action']
    return old, new


@pytest.mark.parametrize('fault',[None,'pose','mission','memory','residual','forecast','utility',
    'surface','path','fallback','wait','plan','terminal','metadata','command','score','eligibility',
    'time','first_point','clearance_claim','later_point','calibration_claim','unchanged_claim','horizon',
    'missing_segment','reduced_radius','hidden_veto','disconnected_path'])
def test_first_change_preserves_every_undeclared_field(fault):
    old,new = decisions(changed=True); selection=new['new_selection']; r=selection['residual_anchored_continuation']
    if fault=='pose': new['evidence']['pose'][0]=9.
    elif fault=='mission': new['mission_receipt']['phase']='RETURN'
    elif fault=='memory': new['memory_receipt']['cells']+=1
    elif fault=='residual': new['causal_residual_receipt']['correction_xy_m'][0]+=.1
    elif fault=='forecast': selection['prediction'][0][0][0]+=.1
    elif fault=='utility': selection['candidates'][0]['utility_m']+=.1
    elif fault=='surface': selection['surface_checks'][0]['possible_intersection']=True
    elif fault=='path': selection['nominal_path_checks'][0]['all_predicted_segments_nominally_clear']=False
    elif fault=='fallback': selection['residual_first_interval_feasibility']={'changed':True}
    elif fault=='wait': new['consecutive_infeasible_observations']=1
    elif fault=='plan': new['plan_offset']=2
    elif fault=='terminal': new['terminal']='VIEW_BUDGET_EXHAUSTED'
    elif fault=='metadata': new['residual_first_interval_feasibility_fallback_enabled']=False
    elif fault=='command': new['requested_command']=[.3,0.,0.]
    elif fault=='score': r['selected_utility_m']=r['original_hold_utility_m']
    elif fault=='eligibility': r['eligible_actions']=[]
    elif fault=='time': r['frame']=11
    elif fault=='first_point': r['corrected_first_body_xy_m'][1][0]+=.1
    elif fault=='clearance_claim': r['physical_clearance_certified']=True
    elif fault=='later_point': r['corrected_body_xy_m'][1][4][0]+=.001
    elif fault=='calibration_claim': r['later_prediction_correction_calibrated']=True
    elif fault=='unchanged_claim': r['later_predicted_points_unchanged']=True
    elif fault=='horizon': r['correction_horizon_ns']=100_000_000
    elif fault=='missing_segment': r['corrected_nominal_path_checks'][1]['segments'].pop()
    elif fault=='reduced_radius': r['corrected_nominal_path_checks'][1]['segments'][4]['radius_m']=.4
    elif fault=='hidden_veto': r['corrected_nominal_path_checks'][1]['segments'][4]['nominal_disk_connector_clear']=False
    elif fault=='disconnected_path': r['corrected_nominal_path_checks'][1]['segments'][4]['predicted_start_map_xy_m'][0]+=.1
    before=deepcopy((old,new))
    if fault is None:
        check=compare_step(old,new,[0.,0.,0.],frame=10)
        assert check['requested_command_changed'] and check['raw_model_forecasts_compared']
    else:
        with pytest.raises(ValueError): compare_step(old,new,[0.,0.,0.],frame=10)
    assert (old,new)==before


def test_unchanged_existing_fallback_is_compared_exactly():
    old,new=decisions()
    for d in (old,new): d['new_selection']['residual_first_interval_feasibility']={'prior':'same'}
    assert not compare_step(old,new,[0.,0.,0.],frame=10)['requested_command_changed']
    new['new_selection']['residual_first_interval_feasibility']['prior']='changed'
    with pytest.raises(ValueError): compare_step(old,new,[0.,0.,0.],frame=10)


@pytest.mark.parametrize('boundary',['command','old_terminal','limit','mutated_input','model_change','truncated'])
def test_replay_never_consumes_next_packet_or_decision(monkeypatch,tmp_path,boundary):
    assert MAX_FRAMES==3004
    limit=14; monkeypatch.setattr(runner,'MAX_FRAMES',limit)
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda *a:SimpleNamespace(free=100*1024**3))
    weight={'value':runner.MODEL_STATE}; model=SimpleNamespace(state_dict=lambda:dict(weight),parameters=lambda:[])
    monkeypatch.setattr(runner,'load_assigned',lambda *a:(model,'jepa','full'))
    monkeypatch.setattr(runner,'state_digest',lambda d:d['value'])
    packets=[]; reads=[]; stop=boundary in ('command','old_terminal')
    def rows(path):
        for i in range(limit):
            if boundary=='truncated' and i==9: return
            reads.append(i)
            if stop and i>10: pytest.fail('read decision after intervention')
            old,_=decisions(i)
            if boundary=='old_terminal' and i==10: old['terminal']='MISSION_TICK_BUDGET_EXHAUSTED'
            yield dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old)
        pytest.fail('read beyond fixed replay limit')
    monkeypatch.setattr(runner,'read_rows',rows)
    class Reader:
        frames=list(range(limit+2))
        def packet(self,i):
            packets.append(i)
            if stop and i>10: pytest.fail('read packet after intervention')
            return {'tick':i},{},{},1_500_000_000+i*100_000_000
    monkeypatch.setattr(runner,'IntentReturnRGBDReplay',lambda *a:Reader())
    tape=[dict(requested_command=[0.,0.,0.],completed=True) for _ in range(limit+1)]
    monkeypatch.setattr(runner,'read_json',lambda p,n:tape if n=='command_tape.json' else [{}]*(limit+2))
    monkeypatch.setattr(runner,'public_acquisition',lambda r:r)
    monkeypatch.setattr(runner,'packet',lambda *a,**k:({},{}))
    class Controller:
        def observe(self,policy,*a,**k):
            i=policy['tick']; _,result=decisions(i,changed=boundary=='command' and i==10)
            if i==10:
                if boundary=='old_terminal': result['terminal']='MISSION_TICK_BUDGET_EXHAUSTED'
                elif boundary=='model_change': weight['value']='changed'
                elif boundary=='mutated_input': policy['changed']=True
            return result
    monkeypatch.setattr(runner,'ResidualAnchoredContinuationController',lambda *a,**k:Controller())
    if boundary in ('mutated_input','model_change','truncated'):
        message={'mutated_input':'mutated public','model_change':'unchanged weights','truncated':'truncated decisions'}[boundary]
        with pytest.raises(ValueError,match=message):
            runner.replay(dict(correction_admission={}))
    else:
        report=runner.replay(dict(correction_admission={}))
        assert packets==reads==list(range(11 if stop else limit))
        assert report['first_requested_command_difference']==(10 if boundary=='command' else None)
        assert not report['following_recorded_observations_consumed'] and not report['native_execution']



def test_existing_first_point_recovery_is_still_compared():
    from lewm.tests.test_residual_hold_prefix_development import decisions as original_decisions
    old,new=original_decisions(changed=True)
    new.update(controller='residual_anchored_continuation_controller_v1',residual_anchored_continuation_enabled=True)
    report=compare_step(old,new,[0.,0.,0.],frame=10)
    assert report['requested_command_changed'] and not report['anchored_continuation_changed_action']


@pytest.mark.parametrize('fault',[None,'identity','frames','changed','terminal','model','diagnosis','source_conflict'])
def test_completed_predecessors_are_required(monkeypatch,fault):
    report=dict(frames=3004,first_requested_command_difference=None,hold_reconsideration_interventions=0,
        final_terminal='MISSION_TICK_BUDGET_EXHAUSTED',raw_model_forecast_comparisons=3000,
        complete_original_selection_preserved=True,unchanged_observed_mission_and_residual_state_exact=True,
        model_state_sha256=runner.MODEL_STATE,model_state_unchanged=True)
    prior=dict(status='RESIDUAL_HOLD_PREFIX_V2_COMPLETE',native_result_sha256=runner.NATIVE_SHA,
        native_execution=False,source_sha256={'synthetic':'a'*64},artifact_sha256={},report=report)
    diagnosis=deepcopy(prior)
    diagnosis.update(status='RESIDUAL_HOLD_VETO_READOUT_V1_COMPLETE',report=dict(observations=3014,
        selected_holds=2643,hold_reason_counts={'no_strictly_better_allowed_nonhold':24,
        'every_better_nonhold_has_a_first_point_invariant_veto':2619}))
    if fault=='identity': prior['native_result_sha256']='b'*64
    elif fault=='frames': report['frames']=1185
    elif fault=='changed': report['first_requested_command_difference']=180
    elif fault=='terminal': report['final_terminal']=None
    elif fault=='model': report['model_state_unchanged']=False
    elif fault=='diagnosis': diagnosis['report']['selected_holds']=0
    elif fault=='source_conflict': diagnosis['source_sha256']['synthetic']='b'*64
    monkeypatch.setattr(runner,'read_json',lambda root,name:prior if root==runner.PREDECESSOR else diagnosis)
    calls=[]
    monkeypatch.setattr(runner,'verify_artifacts',lambda root,bindings:calls.append((root,bindings)))
    monkeypatch.setattr(runner,'verify_sources',lambda sources:calls.append(sources))
    if fault:
        with pytest.raises(ValueError): runner.verify_predecessor()
    else:
        assert runner.verify_predecessor()=={'synthetic':'a'*64}
        assert calls[0]==(runner.PREDECESSOR,{'result.json':runner.PREDECESSOR_SHA})
        assert calls[3]==(runner.DIAGNOSIS,{'result.json':runner.DIAGNOSIS_SHA})


def test_original_native_verification_remains_inside_fresh_scope(monkeypatch):
    launch=dict(verification_benchmark_result_sha256=runner.BENCHMARK_SHA,
        native_result_sha256=runner.NATIVE_SHA,predecessor_result_sha256=runner.PREDECESSOR_SHA,
        diagnosis_result_sha256=runner.DIAGNOSIS_SHA,replay_input_bindings={'result.json':runner.NATIVE_SHA})
    events=[]
    monkeypatch.setattr(runner,'admit_benchmark',lambda:events.append('benchmark'))
    monkeypatch.setattr(runner,'verify',lambda item:events.append('source_and_environment'))
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:events.append('native_artifacts'))
    monkeypatch.setattr(runner,'verify_native',lambda *a:events.append('original_native_verifier'))
    monkeypatch.setattr(runner,'read_json',lambda *a:{'original':'launch'})
    monkeypatch.setattr(runner,'verify_predecessor',lambda:events.append('completed_predecessors'))
    def scoped(fn,digest,*args):
        events.append('fresh_scope'); return fn(*args),{}
    monkeypatch.setattr(runner,'verify_with_scoped_digests',scoped)
    runner.verify_inputs(launch)
    assert events==['benchmark','fresh_scope','source_and_environment','native_artifacts',
        'original_native_verifier','completed_predecessors']
    launch['native_result_sha256']='c'*64
    with pytest.raises(ValueError): runner.verify_inputs(launch)
