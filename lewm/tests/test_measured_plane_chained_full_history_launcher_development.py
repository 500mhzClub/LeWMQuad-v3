"""Exclusive dispatch, complete output binding and terminal failure preservation."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import replay_go2_measured_plane_chained_single_pass_full_history_v1 as job


def setup(tmp_path, monkeypatch, fault=None):
    output = tmp_path/'attempt'; calls = []; admissions = []
    source = {'synthetic_source':'frozen'}
    admission = dict(frames=5, learned_result_sha256='a'*64, native_case='synthetic_chained')
    report = dict(frames=5, timing={'all_observations':{'observations':5}}, navigation_qualified=False)
    for key,value in job.run.ENV.items(): monkeypatch.setenv(key,value)
    monkeypatch.setattr(job.run.cv2.ocl, 'useOpenCL', lambda: False)
    monkeypatch.setattr(job, 'OUTPUT', output)
    monkeypatch.setattr(job.run, 'validate_root', lambda *args, **kwargs: None)
    monkeypatch.setattr(job.run, 'hardware', lambda: {'memory_available_bytes':1})
    def sources(seeds):
        assert job.SOURCE in seeds and job.PROTOCOL in seeds
        calls.append('sources'); return source.copy()
    monkeypatch.setattr(job.inputs, 'prepared_sources', sources)
    def resources():
        calls.append('resources')
        if fault == 'resources': raise ValueError('resource admission rejected')
        return {'memory_available_bytes':64*1024**3}
    monkeypatch.setattr(job, 'resources', resources)
    def idle():
        calls.append('idle')
        if fault == 'busy': raise ValueError('existing CPU replay')
    monkeypatch.setattr(job, 'cpu_idle', idle)
    def admit(sha, bindings):
        calls.append('admit'); admissions.append(sha)
        assert bindings == source
        if fault == 'admission': raise ValueError('native audit incomplete')
        return admission | ({'changed':True} if fault == 'reauthentication' and len(admissions)==2 else {})
    monkeypatch.setattr(job.inputs, 'admit', admit)
    monkeypatch.setattr(job.run, 'create_output', lambda path: path.mkdir())
    def write_json(path, value):
        with path.open('x') as f: json.dump(value,f)
    monkeypatch.setattr(job.run, 'write_json', write_json)
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(job.run, 'digest', digest)
    def verify_artifacts(root, bindings):
        calls.append('verify_outputs')
        for path,sha in bindings.items(): assert digest(root/path)==sha
    monkeypatch.setattr(job.run, 'verify_artifacts', verify_artifacts)
    monkeypatch.setattr(job.run, 'verify', lambda bindings: calls.append('verify_sources'))
    def replay(admitted, *, output):
        calls.append('replay'); assert admitted == admission
        if fault == 'replay': raise ValueError('decision mismatch')
        for name in ('comparison.jsonl','state_checks.json','resource_monitor.jsonl'):
            (output/name).write_text('synthetic complete output\n')
        return deepcopy(report)
    monkeypatch.setattr(job.pair, 'replay', replay)
    def check(actual, admitted, *, output):
        calls.append('check_output'); assert actual == report and admitted == admission
        if fault == 'checker': raise ValueError('saved packet mismatch')
    monkeypatch.setattr(job.pair, 'check_output', check)
    return output,calls,admissions,report


def test_source_only_preflight_does_not_admit_resources_inputs_or_create_output(tmp_path, monkeypatch):
    output,calls,admissions,_ = setup(tmp_path,monkeypatch)
    job.main(source_only=True)
    assert calls == ['sources'] and admissions == []
    assert not output.exists()


def test_complete_pair_binds_all_outputs_and_reauthenticates_native_input(tmp_path, monkeypatch):
    output,calls,admissions,report = setup(tmp_path,monkeypatch)
    job.main('a'*64)
    assert calls == ['sources','resources','idle','admit','resources','idle',
        'replay','check_output','admit','verify_outputs','verify_sources']
    assert admissions == ['a'*64,'a'*64]
    result = json.loads((output/'result.json').read_text())
    launch = json.loads((output/'launch.json').read_text())
    assert result['status'] == 'MEASURED_PLANE_CHAINED_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE'
    assert result['report'] == report
    assert launch['state_frames'] == [0,3,4]
    assert not launch['automatic_retry'] and not result['native_execution']
    assert not result['navigation_outcomes_inferred'] and not result['goal_achieved']
    assert set(result['artifact_sha256']) == {
        'launch.json','comparison.jsonl','state_checks.json','resource_monitor.jsonl','report.json'}
    for path,sha in result['artifact_sha256'].items():
        assert hashlib.sha256((output/path).read_bytes()).hexdigest()==sha
    previous = list(calls)
    with pytest.raises(ValueError,match='exclusive'): job.main('a'*64)
    assert calls == previous


@pytest.mark.parametrize('fault',['resources','busy','admission'])
def test_dispatch_rejection_does_not_create_an_attempt(tmp_path,monkeypatch,fault):
    output,calls,_,_ = setup(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError): job.main('a'*64)
    assert not output.exists() and 'replay' not in calls


def test_missing_result_hash_does_not_dispatch(tmp_path,monkeypatch):
    output,calls,_,_ = setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='actual completed'): job.main()
    assert calls == ['sources'] and not output.exists()


@pytest.mark.parametrize('fault',['replay','checker','reauthentication'])
def test_terminal_replay_failure_is_preserved_without_retry(tmp_path,monkeypatch,fault):
    output,calls,_,_ = setup(tmp_path,monkeypatch,fault)
    with pytest.raises(ValueError): job.main('a'*64)
    failure = json.loads((output/'failure.json').read_text())
    assert failure['status']=='TERMINAL_CHAINED_FULL_HISTORY_TIMING_FAILURE'
    assert failure['automatic_retry'] is False
    assert not (output/'result.json').exists()
    assert (output/'launch.json').exists()
    before = list(calls)
    with pytest.raises(ValueError,match='exclusive'): job.main('a'*64)
    assert calls == before


@pytest.mark.parametrize('script',[
    'scripts/replay_go2_synthetic_v1.py', '/tmp/probe_go2_synthetic_v1.py',
    'scripts/run_go2_single_pass_synthetic.py','scripts/run_go2_deferred_memo_synthetic.py'])
def test_named_existing_cpu_worker_blocks_dispatch(monkeypatch,script):
    monkeypatch.setattr(job.inputs.native.original,'require_native_idle',lambda:None)
    own=job.psutil.Process().pid
    process=SimpleNamespace(info={'pid':own+1,'cmdline':['python',script]})
    monkeypatch.setattr(job.psutil,'process_iter',lambda attrs:[process])
    with pytest.raises(ValueError,match='existing CPU replay'):job.cpu_idle()


def test_own_process_and_unrelated_process_are_not_misclassified(monkeypatch):
    monkeypatch.setattr(job.inputs.native.original,'require_native_idle',lambda:None)
    own=job.psutil.Process().pid
    processes=[SimpleNamespace(info={'pid':own,'cmdline':['python',job.SOURCE]}),
        SimpleNamespace(info={'pid':own+1,'cmdline':['python','scripts/unrelated.py']})]
    monkeypatch.setattr(job.psutil,'process_iter',lambda attrs:processes)
    job.cpu_idle()
