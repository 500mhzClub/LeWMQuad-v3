"""Fixed paired outcomes, original execution, and retained partial evidence."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from lewm import supervised_rollout_maze_study_development as study
from scripts import run_go2_supervised_rollout_mazes_v1 as runner
from scripts import measured_floor_transport_maze_episode_development as episode
from scripts import measured_floor_transport_maze_audit_development as original_audit


def test_same_original_execution_and_evaluation():
    assert runner.collect is episode.collect and runner.artifacts is episode.artifacts
    assert runner.audit is original_audit.audit
    assert [c[1] for c in study.planned_cases()]==[1,2,3]
    assert all(c[2:] == ('full','supervised_rollout',study.NAMES[1]) for c in study.planned_cases())


@pytest.mark.parametrize('fault',[None,'missing','order','state','prefix','status'])
def test_complete_ordered_pair_retains_negative_outcomes(fault):
    old=[]; new=[]
    for case in study.planned_cases():
        name,index,*_=case
        outcomes=dict(verified_round_trip=False,native_evaluation={'arrivals':[]},strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[18],renderer_capture_audit={'pass':True})
        old.append(dict(case=f'full_jepa_novel_maze_{index:02d}',layout_index=index,
            status='INDEPENDENT_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED',**deepcopy(outcomes)))
        new.append(dict(case=name,layout_index=index,status='SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED',
            model_state_sha256=study.SUPERVISED_STATE,model_state_unchanged=True,
            prefix_comparison={'complete_candidate_decisions_match_prospective_prefix':True},**deepcopy(outcomes)))
    if fault=='missing': new.pop()
    elif fault=='order': new.reverse()
    elif fault=='state': new[0]['model_state_sha256']='0'*64
    elif fault=='prefix': new[0]['prefix_comparison']['complete_candidate_decisions_match_prospective_prefix']=False
    elif fault=='status': new[0]['status']='SUPERVISED_ROLLOUT_WORKER_FAILED'
    if fault is None:
        pairs=study.paired_outcomes(old,new)
        assert [p['layout_index'] for p in pairs]==[1,2,3]
        assert all(not p['supervised_rollout']['verified_round_trip'] for p in pairs)
        new[0]['hard_measurement_failed_frames'].append(99)
        assert pairs[0]['supervised_rollout']['hard_measurement_failed_frames']==[18]
    else:
        with pytest.raises(ValueError): study.paired_outcomes(old,new)


@pytest.mark.parametrize('failed_stage',[None,'audit','prefix','verification'])
def test_worker_fresh_models_and_retained_evidence(monkeypatch,tmp_path,failed_stage):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path); case=study.planned_cases()[0]; name=case[0]
    launch=dict(robot_urdf_sha256='a'*64,source_sha256={runner.PROTOCOL:'b'*64},correction_admission={},
        planned_cases=[list(c) for c in study.planned_cases()],prefix_reports=[{'case':'full_jepa_novel_maze_01'}])
    monkeypatch.setattr(runner,'read_json',lambda *a:launch)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    calls=[]; models=[]
    def verify(*a):
        calls.append('verify')
        if failed_stage=='verification' and calls.count('verify')==2: raise ValueError('synthetic verification failure')
    monkeypatch.setattr(runner,'verify_inputs',verify)
    digest=runner.digest; monkeypatch.setattr(runner,'digest',lambda p:'a'*64 if p==runner.URDF else digest(p))
    monkeypatch.setattr(runner,'state_digest',lambda d:study.SUPERVISED_STATE)
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    def load(*a):
        models.append(SimpleNamespace(state_dict=lambda:{})); return models[-1],case[3],case[2]
    monkeypatch.setattr(runner,'load_assigned',load)
    def collect(index,definition,**kwargs):
        assert index==1 and kwargs['model'] is models[0] and kwargs['episode_name']==name
        assert (kwargs['condition'],kwargs['variant'])==('supervised_rollout','full')
        path=tmp_path/name; path.mkdir(); (path/'retained_raw.json').write_text('{}')
        return {'fixture':True}
    def audit(index,result,definition,**kwargs):
        assert kwargs['model'] is models[1] and models[1] is not models[0]
        if failed_stage=='audit': raise ValueError('synthetic audit failure')
        return dict(verified_round_trip=False,native_evaluation={},strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[],renderer_capture_audit={})
    def compare(*a):
        if failed_stage=='prefix': raise ValueError('synthetic prefix failure')
        return {'complete_candidate_decisions_match_prospective_prefix':True}
    monkeypatch.setattr(runner,'collect',collect); monkeypatch.setattr(runner,'audit',audit)
    monkeypatch.setattr(runner,'compare',compare); monkeypatch.setattr(runner,'artifacts',lambda *a:['retained_raw.json'])
    record=runner.worker(case,'c'*64)
    assert len(models)==2 and name+'/retained_raw.json' in record['artifact_sha256']
    assert (tmp_path/(name+'_worker_terminal.json')).is_file()
    if failed_stage is None:
        assert record['status']=='SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED'
        assert not record['verified_round_trip'] and record['model_state_unchanged']
        assert record['reused_development_layout'] and not record['independent_layout_development_execution']
    else:
        assert record['status']=='SUPERVISED_ROLLOUT_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in record['failure']
    if failed_stage!='audit': assert name+'_audit.json' in record['artifact_sha256']
    if failed_stage in (None,'verification'):
        assert name+'_prefix_comparison.json' in record['artifact_sha256']
        assert record['prefix_comparison']['complete_candidate_decisions_match_prospective_prefix']


@pytest.mark.parametrize('fault',[None,'artifact_map','correction','mission','setting'])
def test_upstream_verification_reuse_requires_exact_input_map(monkeypatch,fault):
    old={k:'same' for k in study.MATCHED_KEYS}
    old.update(scene_specifications=[{'layout':i} for i in (1,2,3)],public_missions=[{'goal':i} for i in (1,2,3)])
    prior=dict(replay_input_bindings={'raw':'a'*64},correction_admission={'fit':'fixed'})
    launch=deepcopy(old)|dict(planned_cases=[list(c) for c in study.planned_cases()],
        model_state_sha256=study.SUPERVISED_STATE,implementation_class='MeasuredFloorTransportController',
        learned_artifact_sha256=deepcopy(prior['replay_input_bindings']),
        correction_admission=deepcopy(prior['correction_admission']),prefix_artifact_sha256={})
    calls=[]
    monkeypatch.setattr(runner,'verify',lambda *a:None)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(runner,'read_json',lambda p,n:prior if p==runner.PREFIX else old)
    monkeypatch.setattr(runner,'verify_prefix',lambda p:calls.append(p))
    monkeypatch.setattr(runner,'specification',lambda i:{'layout':i})
    monkeypatch.setattr(runner,'public_mission',lambda i:{'goal':i})
    if fault=='artifact_map': launch['learned_artifact_sha256']['raw']='b'*64
    elif fault=='correction': launch['correction_admission']['fit']='changed'
    elif fault=='mission': launch['public_missions'][0]['goal']=9
    elif fault=='setting': launch['navigation_ticks']='changed'
    if fault is None:
        runner.verify_inputs(launch); assert calls==[prior]
    else:
        with pytest.raises(ValueError): runner.verify_inputs(launch)
        if fault in ('artifact_map','correction'): assert not calls
