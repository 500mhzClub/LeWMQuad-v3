"""Matched cohort admission and failure accounting without native execution."""
from copy import deepcopy
import json
import sys
import pytest
from lewm.independent_reactive_floor_transport_study_development import (
    MATCHED_KEYS, OUTCOME_KEYS, MODEL_STATE, admit_inputs, merge_sources, paired_outcomes)
from scripts import run_go2_independent_reactive_floor_transport_mazes_v1 as runner


def evidence():
    audits = []; records = []
    for i in (1,2,3):
        audit = dict(layout_index=i, raw_sensor_reconstruction_pass=True,
            raw_model_command_replay_pass=True, raw_command_audit_pass=True, model_state_unchanged=True,
            verified_round_trip=False, native_evaluation=dict(native_round_trip_candidate_pass=False,
                arrival_windows=[{'pass':False}], return_traversal=None),
            strict_physical_visibility_pass=False, hard_measurement_failed_frames=[9],
            renderer_capture_audit={'witnesses_match':True})
        audits.append(audit)
        records.append(deepcopy(audit) | dict(case=f'full_jepa_novel_maze_{i:02d}',
            status='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'))
    learned = dict(status='INDEPENDENT_FLOOR_TRANSPORT_MAZES_V1_COMPLETE', all_fixed_cases_executed=True,
        original_case_order=[1,2,3], conditions=records, predecessor_result_sha256='a'*64, artifact_sha256={})
    old = {key:{} for key in MATCHED_KEYS}
    old.update(case_order=[1,2,3], model_state_sha256=MODEL_STATE,
        implementation_class='MeasuredFloorTransportController', controller_or_model_changes_between_cases=False,
        fresh_controller_and_memory_per_case=True, source_sha256={'frozen.py':'a'*64},
        planned_cases=[[r['case'],r['layout_index'],'full','jepa','fixed'] for r in records],
        scene_specifications=[runner.specification(i) for i in (1,2,3)],
        public_missions=[runner.public_mission(i) for i in (1,2,3)])
    reactive_audit = deepcopy(audits[0]) | dict(layout_index=0,
        raw_controller_command_replay_pass=True, high_level_world_model_used=False)
    rr = deepcopy(reactive_audit) | dict(case=runner.PILOT_CASE,
        status='REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED', prefix_comparison=dict(
            physical_and_public_prefix_exact=True, shared_observed_state_exact=True,
            all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
            common_prefix_frames=4, physical_prefix_samples=900))
    reactive = dict(status='REACTIVE_FLOOR_TRANSPORT_MAZE_PILOT_V1_COMPLETE', conditions=[rr],
        learned_result_sha256='a'*64, artifact_sha256={})
    rl = {key:deepcopy(old[key]) for key in MATCHED_KEYS}
    rl.update(implementation_class='ReactiveFloorTransportController', high_level_world_model_loaded=False,
        candidate_future_outcomes_evaluated=False, learned_residual_used=False,
        learned_result_sha256='a'*64, source_sha256={'frozen.py':'a'*64})
    return learned, old, audits, reactive, rl, reactive_audit


@pytest.mark.parametrize('fault', [None,'missing_case','order','audit_order','raw','changed_model',
    'changed_controller','reset','reactive_model','pilot_incomplete','prefix','outcome','predecessor','runtime'])
def test_admission_preserves_negatives_and_rejects_identity_or_audit_faults(fault):
    learned, old, audits, reactive, rl, ra = evidence()
    if fault == 'missing_case': learned['conditions'].pop()
    elif fault == 'order': learned['conditions'].reverse()
    elif fault == 'audit_order': audits.reverse()
    elif fault == 'raw': audits[1]['raw_model_command_replay_pass'] = False
    elif fault == 'changed_model': old['model_state_sha256'] = 'b'*64
    elif fault == 'changed_controller': old['controller_or_model_changes_between_cases'] = True
    elif fault == 'reset': old['fresh_controller_and_memory_per_case'] = False
    elif fault == 'reactive_model': ra['high_level_world_model_used'] = True
    elif fault == 'pilot_incomplete': reactive['status'] = 'RUNNING'
    elif fault == 'prefix': reactive['conditions'][0]['prefix_comparison']['physical_prefix_samples'] = 899
    elif fault == 'outcome': learned['conditions'][1]['verified_round_trip'] = True
    elif fault == 'predecessor': reactive['learned_result_sha256'] = 'b'*64
    elif fault == 'runtime': rl['opencv_binary_sha256'] = 'different'
    if fault is None:
        admission = admit_inputs(learned,old,audits,reactive,rl,ra)
        assert admission['learned_successes'] == 0 and not admission['reactive_pilot_verified_round_trip']
        assert admission['learned_layouts'] == admission['reactive_layouts'] == [1,2,3]
        assert not admission['predecessor_scientific_success_required']
    else:
        with pytest.raises(ValueError): admit_inputs(learned,old,audits,reactive,rl,ra)


def test_source_merge_rejects_overwrite_and_preserves_disjoint_frozen_paths():
    a={'a.py':'a'*64};b={'a.py':'a'*64,'b.py':'b'*64}
    assert merge_sources(a,b) == b and a == {'a.py':'a'*64}
    with pytest.raises(ValueError): merge_sources(a,{'a.py':'c'*64})


def reactive_records():
    return [dict(case=name,layout_index=i,status='INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED',
        artifact_sha256={},**{k:deepcopy(evidence()[2][i-1][k]) for k in OUTCOME_KEYS})
        for name,i in runner.planned_cases()]


def test_pairing_keeps_absent_return_and_hard_failures_without_mutating_inputs():
    learned=evidence()[0]['conditions'];reactive=reactive_records();before=deepcopy((learned,reactive))
    pairs=paired_outcomes(learned,reactive)
    assert len(pairs)==3 and all(p['reactive']['native_evaluation']['return_traversal'] is None for p in pairs)
    assert all(p['learned']['hard_measurement_failed_frames']==[9] for p in pairs)
    pairs[0]['learned']['native_evaluation']['arrival_windows'].clear()
    assert (learned,reactive)==before
    with pytest.raises(ValueError): paired_outcomes(learned,reactive[:2])
    with pytest.raises(ValueError): paired_outcomes(learned,list(reversed(reactive)))


def prepare_main(monkeypatch,tmp_path,*,failed_index=None,resource_failure=False):
    learned,old,audits,reactive,rl,ra=evidence();submitted=[];pools=[];admissions=[]
    monkeypatch.setattr(runner,'OUTPUT',tmp_path/'output')
    monkeypatch.setattr(runner,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a,**k:None)
    monkeypatch.setattr(runner,'verify_inputs',lambda *a,**k:None)
    monkeypatch.setattr(runner,'discover_sources',lambda *a,**k:{'fixture':'a'*64})
    monkeypatch.setattr(runner,'create_output',lambda p:p.mkdir())
    def read(root,name):
        if root==runner.LEARNED:
            if name=='result.json':return learned
            if name=='launch.json':return old
            return audits[int(name.split('_maze_')[1][:2])-1]
        if name=='result.json':return reactive
        if name=='launch.json':return rl
        return ra
    monkeypatch.setattr(runner,'read_json',read)
    monkeypatch.setattr(runner,'hardware',lambda:dict(memory_available_bytes=80*1024**3,artifact_free_bytes=100*1024**3))
    original=runner.resources_for
    def resources(hardware,remaining):
        admissions.append(remaining)
        if resource_failure and remaining==2:raise ValueError('synthetic resource failure')
        return original(hardware,remaining)
    monkeypatch.setattr(runner,'resources_for',resources)
    class Future:
        def __init__(self,record):self.record=record
        def result(self):return self.record
    class Pool:
        def __init__(self,**kwargs):pools.append(kwargs)
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def submit(self,function,case,launch_sha):
            assert function is runner.worker;submitted.append(case)
            record=reactive_records()[case[1]-1]
            if case[1]==failed_index:record['status']='INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_WORKER_FAILED'
            (runner.OUTPUT/(case[0]+'_worker.log')).write_text('synthetic\n')
            runner.write_json(runner.OUTPUT/(case[0]+'_worker_terminal.json'),record)
            return Future(record)
    monkeypatch.setattr(runner,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(runner,'wait',lambda futures,**k:(set(futures),set()))
    monkeypatch.setattr(sys,'argv',['runner','--learned-cohort-result-sha256','b'*64,'--reactive-pilot-result-sha256','c'*64])
    return submitted,pools,admissions


def test_all_negative_cases_execute_and_persist_in_order(monkeypatch,tmp_path):
    submitted,pools,admissions=prepare_main(monkeypatch,tmp_path);runner.main()
    result=json.loads((runner.OUTPUT/'result.json').read_text())
    assert [c[1] for c in submitted]==[1,2,3] and admissions==[3,3,3,2,1]
    assert len(pools)==1 and pools[0]['max_workers']==pools[0]['max_tasks_per_child']==1
    assert pools[0]['mp_context'].get_start_method()=='spawn'
    assert result['measured_round_trip_successes']==0 and len(result['paired_native_outcomes'])==3
    assert result['new_independent_layout_executions']==3 and not result['goal_achieved']
    for count in (1,2,3):
        name=f'cohort_progress_after_{count:02d}.json'
        report=json.loads((runner.OUTPUT/name).read_text())
        assert len(report['completed_conditions'])==count and report['remaining_layouts']==list(range(count+1,4))
        assert result['artifact_sha256'][name]==runner.digest(runner.OUTPUT/name)


@pytest.mark.parametrize('resource_failure',[False,True])
def test_infrastructure_failure_retains_partial_cohort_without_retry(monkeypatch,tmp_path,resource_failure):
    submitted,_,_=prepare_main(monkeypatch,tmp_path,failed_index=None if resource_failure else 2,resource_failure=resource_failure)
    with pytest.raises(ValueError):runner.main()
    assert [c[1] for c in submitted]==([1] if resource_failure else [1,2])
    failure=json.loads((runner.OUTPUT/'failure.json').read_text())
    assert len(failure['completed_conditions'])==len(submitted) and failure['automatic_retry'] is False
    assert not (runner.OUTPUT/'result.json').exists()


def test_preflight_never_spawns_or_creates_output(monkeypatch,tmp_path):
    submitted,pools,_=prepare_main(monkeypatch,tmp_path)
    monkeypatch.setattr(sys,'argv',sys.argv+['--preflight-only']);runner.main()
    assert not submitted and not pools and not runner.OUTPUT.exists()


@pytest.mark.parametrize('failed_stage',[None,'audit','verification'])
def test_worker_keeps_raw_and_completed_audit_when_later_verification_fails(monkeypatch,tmp_path,failed_stage):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path);case=runner.planned_cases()[0]
    launch=dict(planned_cases=[list(c) for c in runner.planned_cases()],robot_urdf_sha256='a'*64,
        source_sha256={runner.PROTOCOL:'b'*64})
    monkeypatch.setattr(runner,'read_json',lambda *a:launch)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    calls=[]
    def verify(*a):
        calls.append('verify')
        if failed_stage=='verification' and calls.count('verify')==2:raise ValueError('synthetic verification failure')
    monkeypatch.setattr(runner,'verify_inputs',verify)
    digest=runner.digest;monkeypatch.setattr(runner,'digest',lambda p:'a'*64 if p==runner.URDF else digest(p))
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    def collect(i,definition,**kwargs):
        assert i==1 and kwargs['episode_name']==case[0] and 'model' not in kwargs
        directory=tmp_path/case[0];directory.mkdir();(directory/'raw.json').write_text('{}')
        calls.append('collect');return {'fixture':True}
    def audit(i,result,definition,**kwargs):
        assert i==1 and result=={'fixture':True} and 'model' not in kwargs
        calls.append('audit')
        if failed_stage=='audit':raise ValueError('synthetic audit failure')
        return evidence()[2][0]
    monkeypatch.setattr(runner,'collect',collect);monkeypatch.setattr(runner,'audit',audit)
    monkeypatch.setattr(runner,'artifacts',lambda *a:['raw.json'])
    record=runner.worker(case,'c'*64)
    assert case[0]+'/raw.json' in record['artifact_sha256']
    assert (tmp_path/(case[0]+'_worker_terminal.json')).is_file()
    if failed_stage is None:
        assert record['status']=='INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
        assert record['independent_layout_development_execution'] and not record['verified_round_trip']
    else:
        assert record['status']=='INDEPENDENT_REACTIVE_FLOOR_TRANSPORT_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in record['failure']
    if failed_stage!='audit':assert case[0]+'_audit.json' in record['artifact_sha256']


@pytest.mark.parametrize('fault',[None,'scene','mission','runtime'])
def test_verify_inputs_enforces_exact_independent_scene_mission_and_runtime(monkeypatch,fault):
    _,old,_,_,pilot,_=evidence()
    launch={k:deepcopy(pilot[k]) for k in MATCHED_KEYS}
    launch.update(planned_cases=[list(c) for c in runner.planned_cases()],learned_artifact_sha256={},
        reactive_pilot_artifact_sha256={},scene_specifications=deepcopy(old['scene_specifications']),
        public_missions=deepcopy(old['public_missions']))
    if fault=='scene':launch['scene_specifications'].reverse()
    elif fault=='mission':launch['public_missions'][0]={'other':True}
    elif fault=='runtime':launch['opencv_binary_sha256']='other'
    monkeypatch.setattr(runner,'verify',lambda *a:None)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(runner,'verify_learned',lambda *a:None)
    monkeypatch.setattr(runner,'verify_pilot',lambda *a:None)
    monkeypatch.setattr(runner,'read_json',lambda root,name:old if root==runner.LEARNED else pilot)
    if fault is None:runner.verify_inputs(launch)
    else:
        with pytest.raises(ValueError):runner.verify_inputs(launch)
