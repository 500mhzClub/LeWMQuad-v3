"""Full replay admission and physics bounded before the changed action."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.recent_qualified_anchor_prefix_development import PrefixComparison,MAX_FRAMES
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.tests.test_recent_qualified_anchor_prefix_development import row
from scripts import recent_qualified_anchor_native_prefix_development as prefix
from scripts.replay_go2_recent_qualified_anchor_prefix_v1 import CASE,INPUT_SHA


def evidence():
    original=[];saved=[];comparison=PrefixComparison();exact=banks=attempts=qualified=0
    for i in range(3):
        old,new=row(i,previous=None if i==0 else i-1,attempt=i==2)
        old['failure']=new['failure']=None
        if i==2:new['requested_command']=[.16,0.,.45]
        check=comparison.compare(old,new,old['requested_command'],frame=i)
        original.append(dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old))
        saved.append(dict(tick=i,decision=new,comparison=check,original_requested_command=old['requested_command'],public_input_arrays_unchanged=True))
        exact+=int(check['complete_original_decision_exact']);banks+=int(check['raw_model_forecasts_compared'])
        attempts+=check['extra_reference_attempts'];qualified+=check['extra_qualified_references']
    report=dict(case=CASE[0],layout_index=1,frames=3,maximum_frames=MAX_FRAMES,model_state_sha256=MODEL_STATE,
        model_state_unchanged=True,original_actual_commands_before_intervention_exact=True,
        stopped_at_first_changed_command_or_either_terminal=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,unexecuted_outcomes_inferred=False,native_execution=False,navigation_verified=False,
        first_requested_command_difference=2,final_requested_command=[.16,0.,.45],prior_requested_command=[0.,0.,0.],
        final_terminal=None,final_failure=None,prior_terminal=None,boundary_comparison=check,
        raw_model_forecast_comparisons=banks,exact_original_decisions=exact,extra_reference_attempts=attempts,
        extra_qualified_references=qualified,first_reference_attempt=2,first_qualified_reference=2,first_decision_difference=2)
    result=dict(status='RECENT_QUALIFIED_ANCHOR_PREFIX_V1_COMPLETE',native_result_sha256=INPUT_SHA,
        model_loaded=True,model_training=False,native_execution=False,shadow_replay_only=True,report=report)
    return result,original,saved


@pytest.mark.parametrize('fault',[None,'short','extra','model','result_input','future','terminal','zero',
    'boundary','forecast_count','old_state','saved_state','extra_qualified'])
def test_admission_reconstructs_every_comparison_and_summary(monkeypatch,fault):
    result,old,rows=evidence();report=result['report']
    if fault=='short':rows.pop()
    elif fault=='extra':rows.append(deepcopy(rows[-1]))
    elif fault=='model':report['model_state_sha256']='0'*64
    elif fault=='result_input':result['native_result_sha256']='0'*64
    elif fault=='future':report['following_recorded_observations_consumed']=True
    elif fault=='terminal':report['final_terminal']='SENSOR_OR_MODEL_FAILURE'
    elif fault=='zero':report['final_requested_command']=[0.,0.,0.]
    elif fault=='boundary':report['first_requested_command_difference']=1
    elif fault=='forecast_count':report['raw_model_forecast_comparisons']+=1
    elif fault=='old_state':old[1]['decision']['mission_receipt']['changed']=True
    elif fault=='saved_state':rows[1]['decision']['mission_receipt']['changed']=True
    elif fault=='extra_qualified':report['extra_qualified_references']+=1
    monkeypatch.setattr(prefix,'read_rows',lambda p:iter(rows if p is None else old))
    if fault is None:assert prefix.admit_prefix(None,result)==report
    else:
        with pytest.raises(ValueError):prefix.admit_prefix(None,result)


@pytest.mark.parametrize('fault',[None,'physical','future_physics','public','decision','prior_command',
    'boundary_command','partial_boundary','short_physics','short_decisions'])
def test_native_comparison_consumes_only_shared_physics_and_recorded_decisions(monkeypatch,tmp_path,fault):
    result,old,saved=evidence();report=result['report']
    paths=[tmp_path/n for n in ('old','new','saved')]
    for p in paths:p.mkdir()
    oldpath,newpath,savedpath=paths
    rows={oldpath:old,newpath:[dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=deepcopy(r['decision'])) for i,r in enumerate(saved)],savedpath:saved}
    data=dict(timestamp_s=np.arange(900)*.002,base_pose_world=np.zeros((900,7)))
    for p in (oldpath,newpath):np.savez(p/'physics_trace.npz',**data)
    tapes={p:[dict(requested_command=r['decision']['requested_command'],completed=True) for r in rows[p]] for p in (oldpath,newpath)}
    if fault=='physical':data['base_pose_world'][849,0]=.1;np.savez(newpath/'physics_trace.npz',**data)
    elif fault=='future_physics':data['base_pose_world'][850:,0]=100.;np.savez(newpath/'physics_trace.npz',**data)
    elif fault=='decision':rows[newpath][2]['decision']['mission_receipt']['changed']=True
    elif fault=='prior_command':tapes[newpath][1]['requested_command']=[.2,0.,0.]
    elif fault=='boundary_command':tapes[newpath][2]['requested_command']=[0.,0.,0.]
    elif fault=='partial_boundary':tapes[newpath][2]['completed']=False
    elif fault=='short_physics':np.savez(newpath/'physics_trace.npz',**{k:v[:849] for k,v in data.items()})
    elif fault=='short_decisions':rows[newpath].pop()
    reads=[];packets=[]
    def read_rows(p):
        for i,r in enumerate(rows[p]):reads.append((p,i));yield r
        if fault!='short_decisions':pytest.fail('consumed following decision')
    def reader(p):
        def packet(i):
            packets.append((p,i))
            return dict(frame=i,value=int(fault=='public' and p==newpath and i==2)),{},{},1_500_000_000+100_000_000*i
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix,'read_rows',read_rows);monkeypatch.setattr(prefix,'IntentReturnRGBDReplay',reader)
    monkeypatch.setattr(prefix,'read_json',lambda p,n:tapes[p] if n=='command_tape.json' else [{}]*3)
    monkeypatch.setattr(prefix,'packet',lambda *a,**k:({},{}));monkeypatch.setattr(prefix,'public_acquisition',lambda v:v)
    if fault in (None,'future_physics','partial_boundary'):
        check=prefix.compare(oldpath,newpath,savedpath,report)
        assert check['physical_prefix_samples']==850 and check['common_prefix_frames']==3
        assert check['raw_model_forecast_comparisons']==3 and check['complete_original_decisions_exact']==2
        assert len(reads)==9 and len(packets)==6
        assert check['candidate_intervention_command_completed']==(fault!='partial_boundary')
        assert not check['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError):prefix.compare(oldpath,newpath,savedpath,report)
