"""Exact physical prefix, preserved native calculations and failure artifacts."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from lewm.current_observation_planning_prefix_development import compare_step
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.tests.test_current_observation_planning_prefix_development import decisions
from scripts import current_observation_planning_native_prefix_development as prefix
from scripts import run_go2_current_observation_planning_maze_pilot_v1 as runner


def evidence():
    saved=[];original=[]
    for i in range(11):
        old,new=decisions(i)
        if i>=3:
            old['new_selection']={'prediction':[[1.]]}
            new['new_selection']=dict(prediction=[[1.]],planning_map_receipt=dict(frame=i,
                measured_ns=1_500_000_000+100_000_000*i,planning_map_variant='current_paired_observation',
                accumulated_planning_cells_queried=False,persistent_contact_history_retained=True))
        if i==10:old['requested_command']=[0.,0.,.45];new['requested_command']=[.16,0.,.45]
        check=compare_step(old,new,old['requested_command'],frame=i)
        saved.append(dict(tick=i,decision=new,comparison=check,
            original_requested_command=old['requested_command'],public_input_arrays_unchanged=True))
        old.update(controller='measured_floor_transport_round_trip_controller_v1',floor_transport_during_missingness_enabled=True)
        original.append(dict(tick=i,observation_index=i,pre_sample_index=749+50*i,decision=old))
    report=dict(case='full_jepa_novel_maze_00',frames=11,first_requested_command_difference=10,
        first_terminal_policy_difference=None,final_requested_command=[.16,0.,.45],prior_requested_command=[0.,0.,.45],
        final_terminal=None,unchanged_observed_and_executed_residual_state_exact=True,raw_model_forecast_comparisons=8,
        all_compared_raw_model_forecasts_exact=True,original_actual_commands_before_intervention_exact=True,
        stopped_at_first_command_or_terminal_difference=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,model_state_sha256=MODEL_STATE,model_state_unchanged=True,
        accumulated_planning_cells_queried=False,persistent_contact_history_retained=True,selector_scan_state_retained=True,
        memoryless_controller=False,unexecuted_outcomes_inferred=False,changed_selection=deepcopy(saved[-1]['decision']['new_selection']))
    return dict(status='CURRENT_OBSERVATION_PLANNING_PREFIX_V1_COMPLETE',model_loaded=True,model_training=False,
        native_execution=False,report=report),saved,original


@pytest.mark.parametrize('fault',[None,'short','extra','model','frames','following','prior_request','state','forecast_count','selection'])
def test_prefix_admission_requires_all_saved_decisions_and_exact_completed_report(monkeypatch,fault):
    result,rows,_=evidence()
    if fault=='short':rows.pop()
    elif fault=='extra':rows.append(deepcopy(rows[-1]))
    elif fault=='model':result['report']['model_state_sha256']='0'*64
    elif fault=='frames':result['report']['frames']=10
    elif fault=='following':result['report']['following_recorded_observations_consumed']=True
    elif fault=='prior_request':rows[5]['original_requested_command']=[1.,0.,0.]
    elif fault=='state':rows[3]['comparison']['unchanged_observed_and_executed_residual_state_exact']=False
    elif fault=='forecast_count':rows[3]['comparison']['raw_model_forecasts_compared']=False
    elif fault=='selection':result['report']['changed_selection']={}
    monkeypatch.setattr(prefix,'read_rows',lambda *a:iter(rows))
    if fault is None:assert prefix.admit_prefix(None,result)==result['report']
    else:
        with pytest.raises(ValueError):prefix.admit_prefix(None,result)


def native_fixture(monkeypatch,tmp_path):
    result,bound,original=evidence();old=tmp_path/'old';new=tmp_path/'new';saved=tmp_path/'saved'
    data=dict(timestamp_s=np.arange(1300)*.002,base_pose_world=np.zeros((1300,7)))
    for p in (old,new,saved):p.mkdir()
    for p in (old,new):np.savez(p/'physics_trace.npz',**data)
    newrows=[deepcopy(r)|dict(observation_index=i,pre_sample_index=749+50*i) for i,r in enumerate(bound)]
    rows={old:original,new:newrows,saved:bound};tapes={p:[dict(requested_command=r['decision']['requested_command'],completed=True) for r in rows[p]] for p in (old,new)}
    reads=[];packets=[];changes={}
    def read_rows(p):
        for i,r in enumerate(rows[p]):reads.append((p,i));yield r
        pytest.fail('read beyond the eleven-observation prefix')
    monkeypatch.setattr(prefix,'read_rows',read_rows)
    monkeypatch.setattr(prefix,'read_json',lambda p,n:tapes[p] if n=='command_tape.json' else [{}]*11)
    def reader(p):
        def packet(i):
            packets.append((p,i));return {'frame':i,'value':changes.get((p,i),0)},{},{},1_500_000_000+i*100_000_000
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix,'IntentReturnRGBDReplay',reader)
    monkeypatch.setattr(prefix,'public_acquisition',lambda r:r)
    monkeypatch.setattr(prefix,'packet',lambda *a,**k:({},{}))
    return result['report'],old,new,saved,rows,tapes,reads,packets,changes,data


@pytest.mark.parametrize('fault',[None,'physical','public','state','decision','prior_command','new_command','short','new_future','partial_new_command'])
def test_native_prefix_requires_exact_past_and_never_compares_new_physical_outcomes(monkeypatch,tmp_path,fault):
    report,old,new,saved,rows,tapes,reads,packets,changes,data=native_fixture(monkeypatch,tmp_path)
    if fault=='physical':data['base_pose_world'][1249,0]=.1;np.savez(new/'physics_trace.npz',**data)
    elif fault=='new_future':data['base_pose_world'][1250:,0]=100.;np.savez(new/'physics_trace.npz',**data)
    elif fault=='public':changes[(new,10)]=1
    elif fault=='state':rows[old][10]['decision']['memory_receipt']={'changed':True}
    elif fault=='decision':rows[new][10]['decision']['new_selection']['prediction']=[[2.]]
    elif fault=='prior_command':tapes[new][9]['requested_command']=[.2,0.,0.]
    elif fault=='new_command':tapes[new][10]['requested_command']=[0.,0.,.45]
    elif fault=='short':np.savez(new/'physics_trace.npz',**{k:v[:1249] for k,v in data.items()})
    elif fault=='partial_new_command':tapes[new][10]['completed']=False
    if fault in (None,'new_future','partial_new_command'):
        r=prefix.compare(old,new,saved,report)
        assert r['physical_prefix_samples']==1250 and r['common_prefix_frames']==11
        assert r['raw_model_forecast_comparisons']==8 and r['complete_candidate_decisions_match_prospective_prefix']
        assert r['candidate_intervention_command_completed']==(fault!='partial_new_command')
        assert len(reads)==33 and len(packets)==22 and not r['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError):prefix.compare(old,new,saved,report)


def function(path,name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)


def test_collector_and_audit_preserve_all_original_physical_and_evaluation_calculations():
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id=='CurrentObservationPlanningController':n.id='MeasuredFloorTransportController'
            return n
        def visit_Constant(self,n):
            if isinstance(n.value,str):n.value=n.value.replace('CURRENT_OBSERVATION_PLANNING_MAZE','MEASURED_FLOOR_TRANSPORT_MAZE')
            return n
        def visit_Call(self,n):
            n.keywords=[k for k in n.keywords if k.arg not in ('planning_map_variant','persistent_contact_history_retained','selector_scan_state_retained')]
            return self.generic_visit(n)
        def visit_Assert(self,n):
            if ast.unparse(n.test) in ("result['planning_map_variant'] == 'current_paired_observation'",
                    "result['persistent_contact_history_retained'] is True and result['selector_scan_state_retained'] is True"):
                return None
            return self.generic_visit(n)
    for kind,names in [('episode',('collect','artifacts')),('audit',('audit',))]:
        for name in names:
            old=function('scripts/measured_floor_transport_maze_'+kind+'_development.py',name)
            new=function('scripts/current_observation_planning_maze_'+kind+'_development.py',name)
            assert ast.dump(old)==ast.dump(Normalize().visit(new)),(kind,name)


@pytest.mark.parametrize('failed_stage',[None,'audit','prefix','verification'])
def test_worker_uses_fresh_identical_models_and_retains_prior_evidence_on_later_failure(monkeypatch,tmp_path,failed_stage):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    launch=dict(robot_urdf_sha256='a'*64,source_sha256={runner.PROTOCOL:'b'*64},correction_admission={},
        prefix_report={'model_state_sha256':MODEL_STATE})
    monkeypatch.setattr(runner,'read_json',lambda *a:launch)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    calls=[];models=[]
    def verify(*a):
        calls.append('verify')
        if failed_stage=='verification' and calls.count('verify')==2:raise ValueError('synthetic verification failure')
    monkeypatch.setattr(runner,'verify_inputs',verify)
    digest=runner.digest;monkeypatch.setattr(runner,'digest',lambda p:'a'*64 if p==runner.URDF else digest(p))
    monkeypatch.setattr(runner,'state_digest',lambda d:MODEL_STATE)
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    def load(*a):
        models.append(SimpleNamespace(state_dict=lambda:{}));return models[-1],runner.CASE[3],runner.CASE[2]
    monkeypatch.setattr(runner,'load_assigned',load)
    def collect(index,definition,**kwargs):
        assert index==0 and kwargs['model'] is models[0] and kwargs['episode_name']==runner.CASE[0]
        path=tmp_path/runner.CASE[0];path.mkdir();(path/'retained_raw.json').write_text('{}');calls.append('collect')
        return {'fixture':True}
    def audit(index,result,definition,**kwargs):
        assert kwargs['model'] is models[1] and models[1] is not models[0];calls.append('audit')
        if failed_stage=='audit':raise ValueError('synthetic audit failure')
        return dict(verified_round_trip=False,native_evaluation={},strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[],renderer_capture_audit={})
    def compare(*a):
        calls.append('prefix')
        if failed_stage=='prefix':raise ValueError('synthetic prefix failure')
        return {'physical_and_public_prefix_exact':True}
    monkeypatch.setattr(runner,'collect',collect);monkeypatch.setattr(runner,'audit',audit)
    monkeypatch.setattr(runner,'compare',compare);monkeypatch.setattr(runner,'artifacts',lambda *a:['retained_raw.json'])
    record=runner.worker('c'*64)
    assert len(models)==2 and runner.CASE[0]+'/retained_raw.json' in record['artifact_sha256']
    assert (tmp_path/(runner.CASE[0]+'_worker_terminal.json')).is_file()
    if failed_stage is None:
        assert record['status']=='CURRENT_OBSERVATION_PLANNING_COLLECTED_AND_RAW_AUDITED'
        assert not record['verified_round_trip'] and record['model_state_unchanged']
    else:
        assert record['status']=='CURRENT_OBSERVATION_PLANNING_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in record['failure']
    if failed_stage!='audit':assert runner.CASE[0]+'_audit.json' in record['artifact_sha256']
