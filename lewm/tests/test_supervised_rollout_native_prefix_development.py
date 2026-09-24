"""Require the paired physical past without borrowing changed-action outcomes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.tests.test_matched_objective_prefix_development import decision, CORRECTIONS
from scripts import supervised_rollout_native_prefix_development as prefix


def evidence():
    admission={'coefficients':{n:{'heads':{'rollout_outcomes':{'applied_bias_xy_m':CORRECTIONS[c]}}}
        for n,c in zip(prefix.NAMES,prefix.CONDITIONS,strict=True)}}
    reports=[]; streams={}; originals={}
    for index in prefix.LAYOUTS:
        name=f'full_jepa_novel_maze_{index:02d}'; saved=[]; original=[]
        for i in range(4):
            old=decision('jepa',i); new=decision('supervised_rollout',i,change=i==3)
            if i==3:
                old['new_selection']['action']='left_arc'; old['requested_command']=[.16,0.,.45]
            check=prefix.compare_step(old,old,new,old['requested_command'],frame=i,layout=index,corrections=CORRECTIONS)
            saved.append(dict(tick=i,jepa_decision=old,decision=new,comparison=check,
                original_requested_command=old['requested_command'],public_input_arrays_unchanged=True))
            original.append(dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=deepcopy(old)))
        reports.append(dict(case=name,layout_index=index,frames=4,maximum_frames=prefix.MAX_FRAMES[index],
            first_prediction_difference=3,paired_forecast_banks=1,first_requested_command_difference=3,
            first_terminal_difference=None,jepa_final_requested_command=[.16,0.,.45],
            supervised_final_requested_command=[0.,0.,-.45],jepa_terminal=None,supervised_terminal=None,
            complete_original_jepa_decisions_exact=True,shared_observed_state_exact=True,
            prior_actual_commands_exact=True,prior_commands_compared=3,stopped_before_following_a_changed_command=True,
            following_recorded_observations_consumed=False,public_input_arrays_unchanged=True,
            model_states_unchanged=True,model_training=False,native_execution=False,
            unexecuted_outcomes_inferred=False,jepa_advantage_established=False,
            models=[dict(name=n,condition=c,base_state_sha256=b,corrected_state_sha256=s,
                head='rollout_outcomes',model_training=False) for n,c,b,s in zip(prefix.NAMES,prefix.CONDITIONS,
                    prefix.BASE_STATE,(prefix.MODEL_STATE,prefix.SUPERVISED_STATE),strict=True)]))
        streams[name]=saved; originals[name]=original
    result=dict(status='MATCHED_OBJECTIVE_PREFIXES_V1_COMPLETE',model_loaded=True,model_training=False,
        native_execution=False,all_fixed_cases_executed=True,conditions=reports)
    return result,admission,streams,originals


@pytest.mark.parametrize('fault',[None,'missing_layout','order','short','extra','model','head','following',
    'report_command','prior_command','state','comparison','public','early_change'])
def test_all_actual_prefixes_required(monkeypatch,fault):
    result,admission,streams,_=evidence(); report=result['conditions'][0]; rows=streams[report['case']]
    if fault=='missing_layout': result['conditions'].pop()
    elif fault=='order': result['conditions'].reverse()
    elif fault=='short': rows.pop()
    elif fault=='extra': rows.append(deepcopy(rows[-1]))
    elif fault=='model': report['models'][1]['corrected_state_sha256']='0'*64
    elif fault=='head': report['models'][1]['head']='direct_outcomes'
    elif fault=='following': report['following_recorded_observations_consumed']=True
    elif fault=='report_command': report['supervised_final_requested_command']=[0.,0.,0.]
    elif fault=='prior_command': rows[2]['original_requested_command']=[.2,0.,0.]
    elif fault=='state': rows[3]['decision']['quiet_intervals']={'altered':True}
    elif fault=='comparison': rows[3]['comparison']['raw_prediction_changed']=False
    elif fault=='public': rows[0]['public_input_arrays_unchanged']=False
    elif fault=='early_change': rows[2]['decision']['requested_command']=[0.,0.,-.45]
    monkeypatch.setattr(prefix,'read_rows',lambda p:iter(streams[p.name]))
    if fault is None: assert prefix.admit_prefixes(Path('/synthetic'),result,admission)==result['conditions']
    else:
        with pytest.raises(ValueError): prefix.admit_prefixes(Path('/synthetic'),result,admission)


@pytest.mark.parametrize('fault',[None,'physical','public','original','decision','prior_command',
    'new_command','short','new_future','partial_new_command','prior_incomplete'])
def test_native_prefix_exact_and_no_following_observation(monkeypatch,tmp_path,fault):
    result,admission,streams,originals=evidence(); report=result['conditions'][0]
    old=tmp_path/'old'; new=tmp_path/'new'; saved=tmp_path/'saved'; bound=saved/report['case']
    for p in (old,new,bound): p.mkdir(parents=True)
    data=dict(timestamp_s=np.arange(950)*.002,base_pose_world=np.zeros((950,7)))
    for p in (old,new): np.savez(p/'physics_trace.npz',**data)
    rows={old:originals[report['case']],bound:streams[report['case']]}
    rows[new]=[deepcopy(r)|dict(observation_index=i,pre_sample_index=749+50*i) for i,r in enumerate(rows[bound])]
    tapes={p:[dict(requested_command=deepcopy(r['decision']['requested_command']),completed=True)
        for r in rows[p]] for p in (old,new)}
    reads=[]; packets=[]; changes={}
    def read_rows(p):
        for i,r in enumerate(rows[p]): reads.append((p,i)); yield r
        pytest.fail('read observation after first changed command')
    monkeypatch.setattr(prefix,'read_rows',read_rows)
    monkeypatch.setattr(prefix,'read_json',lambda p,n:tapes[p] if n=='command_tape.json' else [{}]*4)
    def reader(p):
        def packet(i):
            packets.append((p,i))
            return {'frame':i,'value':changes.get((p,i),0)},{},{},1_500_000_000+i*100_000_000
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix,'IntentReturnRGBDReplay',reader)
    monkeypatch.setattr(prefix,'public_acquisition',lambda r:r)
    monkeypatch.setattr(prefix,'packet',lambda *a,**k:({},{}))
    if fault=='physical': data['base_pose_world'][899,0]=.1; np.savez(new/'physics_trace.npz',**data)
    elif fault=='new_future': data['base_pose_world'][900:,0]=100.; np.savez(new/'physics_trace.npz',**data)
    elif fault=='public': changes[(new,3)]=1
    elif fault=='original': rows[old][3]['decision']['quiet_intervals']={'changed':True}
    elif fault=='decision': rows[new][3]['decision']['new_selection']['prediction']=[[2.]]
    elif fault=='prior_command': tapes[new][2]['requested_command']=[.2,0.,0.]
    elif fault=='new_command': tapes[new][3]['requested_command']=[0.,0.,0.]
    elif fault=='short': np.savez(new/'physics_trace.npz',**{k:v[:899] for k,v in data.items()})
    elif fault=='partial_new_command': tapes[new][3]['completed']=False
    elif fault=='prior_incomplete': tapes[new][2]['completed']=False
    if fault in (None,'new_future','partial_new_command'):
        r=prefix.compare(old,new,saved,report,admission)
        assert r['physical_prefix_samples']==900 and r['common_prefix_frames']==4
        assert r['paired_forecast_banks']==1 and r['complete_candidate_decisions_match_prospective_prefix']
        assert r['candidate_intervention_command_completed']==(fault!='partial_new_command')
        assert len(reads)==12 and len(packets)==8 and not r['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError): prefix.compare(old,new,saved,report,admission)
