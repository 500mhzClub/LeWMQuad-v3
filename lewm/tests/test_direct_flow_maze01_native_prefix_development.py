"""Complete tracking prefix, exact physical past, and exclusion of new outcomes."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.direct_flow_prefix_development import compare_step
from lewm.tests.test_direct_flow_prefix_development import rows as decisions, boundary
from scripts import direct_flow_maze01_native_prefix_development as prefix


def evidence():
    saved=[]; original=[]
    for i in range(prefix.FRAMES):
        old,new=boundary() if i==prefix.CHANGED else decisions(i)
        if i<3: old['new_selection']=new['new_selection']=None
        if i==prefix.CHANGED:
            new['new_selection']['action']='left_turn'; new['requested_command']=[0.,0.,.45]
        check=compare_step(old,new,old['requested_command'],frame=i)
        saved.append(dict(tick=i,decision=new,comparison=check,
            original_requested_command=old['requested_command'],public_input_arrays_unchanged=True))
        original.append(dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old))
    report=dict(case='full_jepa_novel_maze_01',frames=215,maximum_frames=215,boundary_frame=214,
        exact_original_decisions=214,raw_model_forecast_comparisons=211,
        original_actual_commands_before_intervention_exact=True,prior_commands_compared=214,
        final_requested_command=[0.,0.,.45],prior_requested_command=[0.,0.,0.],final_terminal=None,final_failure=None,
        final_visual_status='CURRENT_VISUAL_POSE',full_controller_recovered_at_boundary=True,
        stopped_at_original_failed_observation=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,model_state_sha256=MODEL_STATE,model_state_unchanged=True,
        unexecuted_outcomes_inferred=False,native_execution=False,navigation_verified=False,
        boundary_comparison=deepcopy(saved[-1]['comparison']),
        fallback_receipt=deepcopy(saved[-1]['decision']['original_visual_evidence']['direct_corner_flow_fallback']))
    return dict(status='DIRECT_FLOW_MAZE01_PREFIX_V2_COMPLETE',model_loaded=True,
        model_training=False,native_execution=False,report=report),saved,original


@pytest.mark.parametrize('fault',[None,'short','extra','model','following','recovery','prior_request',
    'state','forecast_count','fallback','early_recovery','failed_v1'])
def test_complete_corrected_replay_required_before_native_use(monkeypatch,fault):
    result,rows,_=evidence()
    if fault=='short': rows.pop()
    elif fault=='extra': rows.append(deepcopy(rows[-1]))
    elif fault=='model': result['report']['model_state_sha256']='0'*64
    elif fault=='following': result['report']['following_recorded_observations_consumed']=True
    elif fault=='recovery': result['report']['full_controller_recovered_at_boundary']=False
    elif fault=='prior_request': rows[5]['original_requested_command']=[1.,0.,0.]
    elif fault=='state': rows[3]['comparison']['complete_original_decision_exact']=False
    elif fault=='forecast_count': rows[3]['comparison']['raw_model_forecasts_compared']=False
    elif fault=='fallback': result['report']['fallback_receipt']={}
    elif fault=='early_recovery': rows[213]['comparison']['controller_recovered']=True
    elif fault=='failed_v1': result['status']='TERMINAL_DIRECT_FLOW_MAZE01_PREFIX_FAILURE'
    monkeypatch.setattr(prefix,'read_rows',lambda *a:iter(rows))
    if fault is None: assert prefix.admit_prefix(None,result)==result['report']
    else:
        with pytest.raises(ValueError): prefix.admit_prefix(None,result)


def native_fixture(monkeypatch,tmp_path):
    result,bound,original=evidence(); old=tmp_path/'old'; new=tmp_path/'new'; saved=tmp_path/'saved'
    data=dict(timestamp_s=np.arange(11500)*.002,base_pose_world=np.zeros((11500,7)))
    for p in (old,new,saved): p.mkdir()
    for p in (old,new): np.savez(p/'physics_trace.npz',**data)
    newrows=[deepcopy(r)|dict(observation_index=i,pre_sample_index=749+50*i) for i,r in enumerate(bound)]
    rows={old:original,new:newrows,saved:bound}
    tapes={p:[dict(requested_command=r['decision']['requested_command'],completed=True) for r in rows[p]] for p in (old,new)}
    reads=[]; packets=[]; changes={}
    def read_rows(p):
        for i,r in enumerate(rows[p]): reads.append((p,i)); yield r
        pytest.fail('read beyond the215-observation intervention')
    monkeypatch.setattr(prefix,'read_rows',read_rows)
    monkeypatch.setattr(prefix,'read_json',lambda p,n:tapes[p] if n=='command_tape.json' else [{}]*215)
    def reader(p):
        def packet(i):
            packets.append((p,i))
            return {'frame':i,'value':changes.get((p,i),0)},{},{},1_500_000_000+i*100_000_000
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix,'IntentReturnRGBDReplay',reader)
    monkeypatch.setattr(prefix,'public_acquisition',lambda r:r)
    monkeypatch.setattr(prefix,'packet',lambda *a,**k:({},{}))
    return result['report'],old,new,saved,rows,tapes,reads,packets,changes,data


@pytest.mark.parametrize('fault',[None,'physical','public','original_failure','decision','prior_command',
    'new_command','short','new_future','partial_new_command'])
def test_exact_native_past_with_no_borrowed_changed_command_outcomes(monkeypatch,tmp_path,fault):
    report,old,new,saved,rows,tapes,reads,packets,changes,data=native_fixture(monkeypatch,tmp_path)
    if fault=='physical': data['base_pose_world'][11449,0]=.1; np.savez(new/'physics_trace.npz',**data)
    elif fault=='new_future': data['base_pose_world'][11450:,0]=100.; np.savez(new/'physics_trace.npz',**data)
    elif fault=='public': changes[(new,214)]=1
    elif fault=='original_failure': rows[old][214]['decision']['original_visual_evidence']['reference_selection']={}
    elif fault=='decision': rows[new][214]['decision']['new_selection']['prediction']=[[2.]]
    elif fault=='prior_command': tapes[new][213]['requested_command']=[.2,0.,0.]
    elif fault=='new_command': tapes[new][214]['requested_command']=[0.,0.,0.]
    elif fault=='short': np.savez(new/'physics_trace.npz',**{k:v[:11449] for k,v in data.items()})
    elif fault=='partial_new_command': tapes[new][214]['completed']=False
    if fault in (None,'new_future','partial_new_command'):
        r=prefix.compare(old,new,saved,report)
        assert r['physical_prefix_samples']==11450 and r['common_prefix_frames']==215
        assert r['raw_model_forecast_comparisons']==211 and r['complete_candidate_decisions_match_prospective_prefix']
        assert r['candidate_intervention_command_completed']==(fault!='partial_new_command')
        assert len(reads)==645 and len(packets)==430 and not r['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError): prefix.compare(old,new,saved,report)
