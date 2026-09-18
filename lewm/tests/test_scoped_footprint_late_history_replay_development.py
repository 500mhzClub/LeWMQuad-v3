"""Full-length synthetic replay checks; no native scene or trained model."""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import replay_go2_scoped_footprint_late_history_v1 as replay


def synthetic(monkeypatch, tmp_path, fault=None):
    profile = replay.profile
    decisions = [dict(controller='residual_anchored_continuation_controller_v1', terminal=None,
        requested_command=[0.,0.,0.], new_selection=None if i<3 else
        dict(action='hold', prediction=[[float(i)]], receipt={'frame': i})) for i in range(replay.FRAMES)]
    tape = [dict(tick=i, completed=True, pre_sample_index=749+50*i, post_sample_index=799+50*i,
                 requested_command=[0.,0.,0.]) for i in range(replay.FRAMES)]
    models = []; calls = []
    def model_factory(*args):
        parameter = torch.nn.Parameter(torch.zeros(1))
        if fault == 'shared_storage' and models:
            parameter = next(iter(models[0].parameters()))
        model = SimpleNamespace(state_dict=lambda:{'weight':parameter}, parameters=lambda:[parameter])
        models.append(model)
        return models[0] if fault == 'shared_model' else model
    class Reader:
        def __init__(self, directory): pass
        def packet(self, i):
            assert i < replay.FRAMES
            return np.array([i]), np.array([i]), np.array([i]), i
    def public(i):
        return (np.array([i]),np.array([i]),np.array([i]),np.array([2]),np.array([1]),i)
    references = []
    for i,decision in enumerate(decisions):
        baseline = decision | {'controller': profile.previous.completed_replay.CONTROLLER,
                              profile.previous.completed_replay.FLAG: True}
        references.append(dict(frame=i, public_input_sha256=profile.fingerprint(public(i)),
            original_decision_sha256=profile.reference.saved.identity(decision),
            candidate_decision_sha256=profile.reference.saved.identity(baseline)))
    class Baseline:
        marker = profile.previous.completed_replay.CONTROLLER
        flag = profile.previous.completed_replay.FLAG
        index = 0
        def __init__(self, model, geometry, **kwargs):
            self.model = model
            self.history = []
            self.memory = SimpleNamespace(index={'synthetic':True})
            self.mapper = SimpleNamespace(floor={},occupied={})
            self.residual = SimpleNamespace(pending=None)
        def observe(self,p,d,f,*,now_ns,auxiliary_depth,auxiliary_rgb):
            i = int(p[0]); assert len(self.history) == i
            self.history.append(i); calls.append((i,self.index))
            result = deepcopy(decisions[i]); result.update(controller=self.marker, **{self.flag:True})
            if self.index == 1 and i == 1001:
                if fault == 'receipt': result['new_selection']['receipt']['frame'] = -1
                if fault == 'input': auxiliary_depth[0] = -1
                if fault == 'metadata': result[self.flag] = 1
                if fault == 'terminal': result['terminal'] = 'unexpected'
                if fault == 'state': self.memory.index['synthetic'] = False
                if fault == 'gradient': next(iter(self.model.parameters())).grad = torch.ones(1)
                if fault == 'weight':
                    with torch.no_grad(): next(iter(self.model.parameters())).add_(1)
            return result
    class Candidate(Baseline):
        marker = replay.CONTROLLER
        flag = replay.FLAG
        index = 1
    monkeypatch.setattr(replay,'OUTPUT',tmp_path)
    monkeypatch.setattr(replay,'FrozenFootprintAnchoredController',Baseline)
    monkeypatch.setattr(replay,'ScopedFootprintAnchoredController',Candidate)
    monkeypatch.setattr(profile.reference.original,'assigned_model',model_factory)
    monkeypatch.setattr(profile,'state_digest',lambda state:
        profile.reference.MODEL_SHA if state['weight'].detach().item() == 0 else 'changed')
    monkeypatch.setattr(profile,'ArticulatedCollisionGeometry',lambda *args:object())
    monkeypatch.setattr(profile,'IntentReturnRGBDReplay',Reader)
    monkeypatch.setattr(replay,'read_json',lambda root,name:
        tape if name=='command_tape.json' else ([{}]*replay.FRAMES if name=='auxiliary_camera_audit.json' else {}))
    monkeypatch.setattr(profile,'packet',lambda *args,**kwargs:(np.array([1]),np.array([2])))
    monkeypatch.setattr(profile,'public_acquisition',lambda x:x)
    def rows(directory):
        for i in range(replay.FRAMES+1):
            assert i < replay.FRAMES, 'following observation must not be consumed'
            yield {'tick':i,'decision':decisions[i]}
    monkeypatch.setattr(profile,'read_rows',rows)
    return references,tape,calls,models


def test_complete_paired_history_and_state_checks_with_no_following_observation(monkeypatch,tmp_path):
    references,_,calls,models = synthetic(monkeypatch,tmp_path)
    result = replay.replay(references)
    assert calls == [(i,index) for i in range(1428) for index in replay.execution_order(i)]
    assert models[0] is not models[1]
    assert result['frames'] == 1428 and result['raw_model_forecast_comparisons'] == 1425
    assert [r['frame'] for r in result['observed_state_checks']] == [3,12,395,404,1173,1418,1427]
    assert result['normalized_state_type_paths'] == [] and result['no_observation_1428_consumed']
    assert result['incremental_reuse_comparison'] and not result['profiling_enabled']
    assert not result['native_execution'] and not result['real_time_qualified']
    rows = [json.loads(line) for line in (tmp_path/'comparison.jsonl').read_text().splitlines()]
    assert len(rows) == 1428 and result['timing_windows'] == replay.timing_summary(rows)
    for i,row in enumerate(rows):
        assert row['baseline_decision_sha256'] == references[i]['candidate_decision_sha256']
        assert row['original_decision_sha256'] == references[i]['original_decision_sha256']


@pytest.mark.parametrize('fault', ['receipt','input','metadata','terminal','state','gradient','weight','shared_model','shared_storage'])
def test_corruption_or_shared_storage_cannot_complete(monkeypatch,tmp_path,fault):
    references,_,calls,_ = synthetic(monkeypatch,tmp_path,fault)
    with pytest.raises(ValueError): replay.replay(references)
    if fault in ('shared_model','shared_storage'): assert calls == []
    elif fault == 'state': assert calls[-1][0] == 1173
    elif fault in ('weight','gradient'): assert calls[-1][0] == 1427
    else: assert calls[-1][0] == 1001


@pytest.mark.parametrize('fault', ['endpoint','input_hash','original_hash','baseline_hash'])
def test_exact_recorded_inputs_commands_and_decision_identities_required(monkeypatch,tmp_path,fault):
    references,tape,calls,_ = synthetic(monkeypatch,tmp_path)
    if fault=='endpoint': tape[1001]['post_sample_index'] += 1
    else:
        key = {'input_hash':'public_input_sha256','original_hash':'original_decision_sha256',
               'baseline_hash':'candidate_decision_sha256'}[fault]
        references[1001][key] = 'wrong'
    with pytest.raises(ValueError): replay.replay(references)
    assert calls[-1][0] == (1000 if fault in ('endpoint','input_hash') else 1001)


def test_normalization_changes_only_declared_top_level_metadata():
    d = dict(controller=replay.CONTROLLER, **{replay.FLAG:True}, requested_command=[.2,0.,0.],
             new_selection={'receipt':{replay.FLAG:True,'controller':replay.CONTROLLER}})
    before = deepcopy(d); expected = deepcopy(d); expected.pop(replay.FLAG)
    expected['controller'] = 'residual_anchored_continuation_controller_v1'
    assert replay.normalize_candidate(d) == expected and d == before
    for bad in (d|{replay.FLAG:1},d|{'controller':'other'}):
        with pytest.raises(ValueError): replay.normalize_candidate(bad)


@pytest.mark.parametrize('fault', ['missing','order','negative','nan','bool'])
def test_timing_population_cannot_drop_or_hide_bad_measurements(fault):
    rows = [dict(frame=i, execution_order=list(replay.execution_order(i)),
        baseline_controller_s=.2, candidate_controller_s=.1) for i in range(replay.FRAMES)]
    if fault=='missing': rows.pop()
    elif fault=='order': rows[1001]['execution_order'] = [0,1]
    else: rows[1001]['candidate_controller_s'] = {'negative':-1.,'nan':float('nan'),'bool':True}[fault]
    with pytest.raises(ValueError): replay.timing_summary(rows)


@pytest.mark.parametrize('mode', ['live','reused','gone'])
def test_original_profile_owner_must_have_ended(monkeypatch,mode):
    monkeypatch.setattr(replay.Path,'read_text',lambda *args,**kwargs: replay.BOOT)
    def process(pid):
        assert pid == replay.PROFILE_PID
        if mode=='gone': raise replay.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda: replay.PROFILE_CREATED+(mode=='reused'),
                               cmdline=lambda: replay.PROFILE_ARGV)
    monkeypatch.setattr(replay.psutil,'Process',process)
    if mode=='gone': replay.profile_owner_ended()
    else:
        with pytest.raises(ValueError): replay.profile_owner_ended()


def test_source_preflight_has_no_profile_admission_model_or_output_side_effect(monkeypatch,tmp_path,capsys):
    def forbidden(*args,**kwargs): raise AssertionError('runtime during source preflight')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): monkeypatch.setenv(key,'1')
    monkeypatch.setenv('PYTHONHASHSEED','0')
    monkeypatch.setattr(replay,'OUTPUT',tmp_path/'absent')
    monkeypatch.setattr(replay,'validate_root',lambda *args,**kwargs:None)
    monkeypatch.setattr(replay,'prepared_sources',lambda:{})
    monkeypatch.setattr(replay.profile.reference,'hardware',lambda:{})
    monkeypatch.setattr(replay,'resources_for',lambda *args:None)
    for name in ('completed_profile','replay','create_output'): monkeypatch.setattr(replay,name,forbidden)
    monkeypatch.setattr(replay.profile.reference,'admit_worker',forbidden)
    monkeypatch.setattr(sys,'argv',[replay.SOURCE,'--source-preflight-only'])
    replay.main()
    assert 'SCOPED_FOOTPRINT_PAIRED_SOURCE_PREFLIGHT_PASS 0' in capsys.readouterr().out
    assert not replay.OUTPUT.exists()


@pytest.mark.parametrize('fault', [None,'terminal_failure','status','missing_artifact','source',
    'frames','forecasts','model','visibility','missing_row','row_order','row_failed'])
def test_only_complete_exact_profile_evidence_is_admitted(monkeypatch,tmp_path,fault):
    sources = {'reviewed_source.py':'a'*64}
    scope = dict(original_strict_physical_visibility_pass=False,
                 original_hard_measurement_failed_frames=[1173], original_verified_round_trip=False)
    artifacts = {n:'b'*64 for n in {'launch.json','comparison.jsonl'} |
                 {n+s for n in replay.profile.WINDOWS for s in ('.prof','.json')}}
    artifacts['launch.json'] = replay.PROFILE_LAUNCH_SHA
    result = dict(status='FROZEN_FOOTPRINT_LATE_HISTORY_PROFILE_V1_COMPLETE', source_sha256=sources.copy(),
        artifact_sha256=artifacts, sensing_scope=deepcopy(scope), report=dict(frames=1428,
        raw_model_forecast_comparisons=1425, model_state_sha256=replay.profile.reference.MODEL_SHA,
        no_observation_1428_consumed=True, complete_normalized_candidate_decisions_exact=True, model_state_unchanged=True))
    rows = [dict(frame=i,candidate_normalized_decision_exact=True,
        complete_original_decision_reconstructed=True,public_input_arrays_unchanged=True) for i in range(1428)]
    if fault=='terminal_failure': (tmp_path/'failure.json').write_text('{}')
    if fault=='status': result['status'] = 'RUNNING'
    if fault=='missing_artifact': artifacts.pop('late_navigation.prof')
    if fault=='source': result['source_sha256']['reviewed_source.py'] = 'c'*64
    if fault=='frames': result['report']['frames'] -= 1
    if fault=='forecasts': result['report']['raw_model_forecast_comparisons'] -= 1
    if fault=='model': result['report']['model_state_sha256'] = 'other'
    if fault=='visibility': result['sensing_scope']['original_strict_physical_visibility_pass'] = True
    if fault=='missing_row': rows.pop()
    if fault=='row_order': rows[1001]['frame'] = 1000
    if fault=='row_failed': rows[1001]['public_input_arrays_unchanged'] = False
    (tmp_path/'comparison.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    monkeypatch.setattr(replay.profile,'OUTPUT',tmp_path)
    monkeypatch.setattr(replay,'profile_owner_ended',lambda:None)
    monkeypatch.setattr(replay,'verify_artifacts',lambda *args:None)
    monkeypatch.setattr(replay,'verify',lambda *args:None)
    monkeypatch.setattr(replay.profile,'sensing_scope',lambda:scope)
    monkeypatch.setattr(replay,'read_json',lambda root,name:
        result if name=='result.json' else {'source_sha256':sources})
    if fault:
        with pytest.raises(ValueError): replay.completed_profile('d'*64,sources)
    else:
        assert replay.completed_profile('d'*64,sources) == rows


@pytest.mark.parametrize('resource', ['memory_available_bytes','artifact_free_bytes','physical_cpus'])
def test_resource_admission_requires_the_full_pair_envelope(resource):
    resources = dict(memory_available_bytes=64*1024**3,artifact_free_bytes=41*1024**3,physical_cpus=4)
    replay.resources_for(resources)
    resources[resource] -= 1
    with pytest.raises(ValueError): replay.resources_for(resources)
