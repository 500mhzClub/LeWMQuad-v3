"""Preserve complete causal state and exclude observations after intervention."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm.tests.test_residual_hold_feasibility_development import fixture, apply
from lewm.residual_hold_prefix_development import compare_step, MAX_FRAMES
from scripts import replay_go2_residual_hold_prefix_v1 as runner


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
    new.update(controller='residual_hold_feasibility_controller_v1', residual_hold_feasibility_enabled=True)
    if changed:
        assert frame == 10
        new['new_selection'] = apply(selection, receipt, mapper)
        new['requested_command'] = new['new_selection']['requested_command']
        new['selected_action'] = new['new_selection']['action']
    return old, new


@pytest.mark.parametrize('fault',[None,'pose','mission','memory','residual','forecast','utility',
    'surface','path','fallback','wait','plan','terminal','metadata','command','score','eligibility',
    'time','first_point','clearance_claim'])
def test_first_change_preserves_every_undeclared_field(fault):
    old,new = decisions(changed=True); selection=new['new_selection']; r=selection['residual_hold_feasibility']
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
    monkeypatch.setattr(runner,'ResidualHoldFeasibilityController',lambda *a,**k:Controller())
    if boundary in ('mutated_input','model_change','truncated'):
        message={'mutated_input':'mutated public','model_change':'unchanged weights','truncated':'truncated decisions'}[boundary]
        with pytest.raises(ValueError,match=message):
            runner.replay(dict(correction_admission={}))
    else:
        report=runner.replay(dict(correction_admission={}))
        assert packets==reads==list(range(11 if stop else limit))
        assert report['first_requested_command_difference']==(10 if boundary=='command' else None)
        assert not report['following_recorded_observations_consumed'] and not report['native_execution']


def test_main_requires_completed_audit_and_retains_all_verifier_inputs(monkeypatch,tmp_path):
    from lewm.tests.test_residual_first_interval_readout_admission_development import fixture as admission_fixture
    current,launch,learned,old,audit,old_audit=admission_fixture()
    current.update(artifact_sha256={},source_sha256={'synthetic':'a'*64})
    monkeypatch.setattr(runner,'OUTPUT',tmp_path/'output')
    monkeypatch.setattr('sys.argv',['replay','--native-result-sha256','b'*64,'--preflight-only'])
    monkeypatch.setattr(runner,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(runner,'verify_native',lambda *a:None)
    def read(root,name):
        if root==runner.INPUT: return {'result.json':current,'launch.json':launch,runner.CASE[0]+'_audit.json':audit}[name]
        return {'result.json':learned,'launch.json':old,runner.LEARNED_CASE[0]+'_audit.json':old_audit}[name]
    monkeypatch.setattr(runner,'read_json',read)
    monkeypatch.setattr(runner,'discover_sources',lambda *a:current['source_sha256'])
    monkeypatch.setattr(runner,'hardware',lambda:dict(memory_available_bytes=2**40,artifact_free_bytes=2**40))
    checked=[]
    def verify(definition):
        for key in ('input_sha256','native_sha256','native_geometry_sha256','opencv_binary_sha256',
                'opencv_version','rules','native_scene_sha256'):
            assert definition[key]==launch[key]
        assert not definition['native_execution'] and definition['native_scene_workers']==0
        assert definition['replay_input_bindings']['result.json']=='b'*64
        checked.append(True)
    monkeypatch.setattr(runner,'verify',verify)
    runner.main(); assert checked==[True] and not runner.OUTPUT.exists()
    audit['raw_model_command_replay_pass']=False
    with pytest.raises(ValueError,match='raw audit'): runner.main()
    assert checked==[True] and not runner.OUTPUT.exists()
