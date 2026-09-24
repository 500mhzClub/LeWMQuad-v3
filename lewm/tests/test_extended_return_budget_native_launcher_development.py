"""Exclusive native dispatch, post-worker verification and terminal failures."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts import run_go2_extended_return_budget_maze02_v1 as job


def setup(tmp_path, monkeypatch, fault=None, paired=True):
    output = tmp_path/'attempt'; events = []; ended = []; source = {job.PROTOCOL: 'a'*64}
    admission = {'synthetic_admission': True}; baseline = {key: 'same' for key in job.MATCHED_KEYS}
    monkeypatch.setattr(job, 'OUTPUT', output)
    monkeypatch.setattr(job.run, 'validate_root', lambda *args, **kwargs: None)
    monkeypatch.setattr(job.run, 'create_output', lambda path: path.mkdir())
    monkeypatch.setattr(job.run, 'digest', lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(job.run, 'read_json', lambda root, name: json.loads((root/name).read_text()))
    hardware = dict(memory_available_bytes=70*1024**3, artifact_free_bytes=100*1024**3)
    def hardware_now():
        if ended and fault == 'final_memory': return hardware | {'memory_available_bytes': 15*1024**3}
        if ended and fault == 'final_disk': return hardware | {'artifact_free_bytes': 40*1024**3}
        return dict(hardware)
    monkeypatch.setattr(job.run, 'hardware', hardware_now)
    def prepared(seeds):
        assert job.SOURCE in seeds and job.PROTOCOL in seeds and all(path in seeds for path in job.TESTS)
        events.append('sources'); return source
    monkeypatch.setattr(job.inputs, 'prepared_sources', prepared)
    def resources():
        events.append('resources')
        if fault == 'resources': raise ValueError('resources unavailable')
        return dict(hardware)
    monkeypatch.setattr(job, 'resources', resources)
    def idle():
        events.append('idle')
        if fault == 'busy': raise ValueError('CPU busy')
    monkeypatch.setattr(job, 'cpu_idle', idle)
    def admit(sha, sources):
        events.append('admit'); assert sha == 'b'*64 and sources == source
        if fault == 'admission': raise ValueError('completed prefix missing')
        return admission
    monkeypatch.setattr(job.inputs, 'admit', admit)
    monkeypatch.setattr(job.inputs.old_inputs, 'native_launch', lambda: baseline)
    monkeypatch.setattr(job, 'definition', lambda: {'navigation_ticks': 8000})
    def verify(launch):
        events.append('inputs')
        assert launch['input_admission'] == admission and launch['source_sha256'] == source
        if fault == 'reauthentication' and ended: raise ValueError('predecessor changed after worker')
    monkeypatch.setattr(job, 'verify_inputs', verify)
    monkeypatch.setattr(job, 'assigned_model', lambda: events.append('model'))
    name = job.CASE[0]
    def returned_record():
        if fault == 'worker_exception': raise RuntimeError('late resource breach in worker')
        leaves = ['synthetic_closed_artifact', name+'_worker.log', *job.worker_resources.names(name)]
        for leaf in leaves: (output/leaf).write_text('synthetic closed evidence')
        record = dict(status=job.WORKER_FAILURE if fault == 'worker_status' else job.WORKER_STATUS,
            verified_round_trip=False, prefix_comparison={'actual_paired_execution_compared': paired},
            artifact_sha256={'synthetic_closed_artifact': job.run.digest(output/'synthetic_closed_artifact')})
        job.bounded_json(output/(name+'_audit.json'), {'synthetic_raw_audit': True})
        job.bounded_json(output/(name+'_worker_terminal.json'), record if fault != 'terminal_mismatch' else {})
        return record
    class Future:
        def result(self): return returned_record()
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == 1 and kwargs['max_tasks_per_child'] == 1
            assert kwargs['mp_context'].get_start_method() == 'spawn'
        def __enter__(self): return self
        def submit(self, function, sha):
            assert function is job.worker and sha == job.run.digest(output/'launch.json')
            events.append('dispatch'); return Future()
        def __exit__(self, *args): ended.append(True); events.append('worker_ended')
    monkeypatch.setattr(job, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(job, 'wait', lambda futures, **kwargs: (set(futures), set()))
    def whole_worker(root, episode):
        assert ended and root == output and episode == name
        events.append('worker_resources')
        if fault == 'worker_resources': raise ValueError('whole worker failed')
        return {'worker_completed': True}
    monkeypatch.setattr(job.worker_resources, 'check', whole_worker)
    def require(record, audit, supplied, **kwargs):
        assert ended and 'prefix_receipt' not in kwargs and supplied == admission
        assert audit == {'synthetic_raw_audit': True}
        events.append('actual_prefix_reconstruction')
        if fault == 'actual_prefix': raise ValueError('actual physical prefix differs')
    monkeypatch.setattr(job.results, 'require_worker', require)
    def artifacts(root, ids):
        events.append('closed_artifacts')
        assert ended and all(job.run.digest(root/leaf) == sha for leaf, sha in ids.items())
        if fault == 'artifacts': raise ValueError('closed output changed')
    monkeypatch.setattr(job.run, 'verify_artifacts', artifacts)
    original_write = job.bounded_json
    def write(path, value):
        if path.name == 'result.json' and fault == 'result_write': raise OSError('result write failed')
        original_write(path, value)
    monkeypatch.setattr(job, 'bounded_json', write)
    return output, events


def test_source_only_does_not_admit_or_dispatch(tmp_path, monkeypatch):
    output, events = setup(tmp_path, monkeypatch)
    job.main(source_only=True)
    assert events == ['sources'] and not output.exists()


def test_preflight_authenticates_without_creating_attempt(tmp_path, monkeypatch):
    output, events = setup(tmp_path, monkeypatch)
    job.main('b'*64, preflight=True)
    assert events == ['sources', 'resources', 'idle', 'admit', 'inputs', 'model', 'resources', 'idle']
    assert not output.exists()


@pytest.mark.parametrize('paired', [True, False])
def test_completed_native_evidence_is_verified_after_worker_ends(tmp_path, monkeypatch, paired):
    output, events = setup(tmp_path, monkeypatch, paired=paired)
    job.main('b'*64)
    result = json.loads((output/'result.json').read_text())
    assert result['status'] == 'EXTENDED_RETURN_BUDGET_MAZE02_V1_COMPLETE'
    assert result['actual_physical_prefix_reconstructed_after_worker_end'] is paired
    assert result['measured_round_trip_successes'] == 0 and result['new_independent_layout_executions'] == 0
    assert not result['goal_achieved'] and not result['navigation_qualified'] and not result['automatic_retry']
    assert events[-5:] == ['worker_ended', 'worker_resources', 'actual_prefix_reconstruction', 'inputs', 'closed_artifacts']
    assert len(result['artifact_sha256']) == 7
    assert all(job.run.digest(output/path) == sha for path, sha in result['artifact_sha256'].items())
    before = list(events)
    with pytest.raises(ValueError, match='exclusive'): job.main('b'*64)
    assert events == before


@pytest.mark.parametrize('fault', ['resources', 'busy', 'admission', 'missing_sha'])
def test_missing_prerequisites_create_no_attempt(tmp_path, monkeypatch, fault):
    output, events = setup(tmp_path, monkeypatch, fault)
    with pytest.raises(ValueError): job.main(None if fault == 'missing_sha' else 'b'*64)
    assert not output.exists() and 'dispatch' not in events


@pytest.mark.parametrize('fault', ['worker_exception', 'worker_status', 'terminal_mismatch',
    'worker_resources', 'actual_prefix', 'reauthentication', 'artifacts', 'final_memory', 'final_disk', 'result_write'])
def test_postlaunch_failure_retains_exclusive_attempt_without_success(tmp_path, monkeypatch, fault):
    output, events = setup(tmp_path, monkeypatch, fault)
    with pytest.raises((ValueError, RuntimeError, OSError)): job.main('b'*64)
    failure = json.loads((output/'failure.json').read_text())
    assert failure['status'] == 'TERMINAL_EXTENDED_RETURN_NATIVE_FAILURE'
    assert failure['original_evidence_preserved'] and not failure['automatic_retry']
    assert (output/'launch.json').is_file() and not (output/'result.json').exists()
    assert events.count('dispatch') == 1
    with pytest.raises(ValueError, match='exclusive'): job.main('b'*64)


@pytest.mark.parametrize('changed', [None, 'navigation_ticks', 'physics_paused_during_compute', 'input_sha256'])
def test_real_input_checker_allows_only_declared_budget_change(monkeypatch, changed):
    baseline = {key: 'original' for key in job.MATCHED_KEYS}; baseline['navigation_ticks'] = 4000
    expected = dict(navigation_ticks=8000, native_execution=True)
    launch = baseline | expected | {'source_sha256': {'source': 'sha'}, 'input_admission': {'proof': True}}
    calls = []
    monkeypatch.setattr(job, 'definition', lambda: deepcopy(expected))
    monkeypatch.setattr(job.original, 'require_environment', lambda value: calls.append('environment'))
    monkeypatch.setattr(job.original.old, 'verify_ordered_launch', lambda value: calls.append('native_implementation'))
    monkeypatch.setattr(job.inputs.old_inputs, 'native_launch', lambda: baseline)
    monkeypatch.setattr(job.inputs, 'verify_bound', lambda admission, sources: calls.append('bound_inputs'))
    if changed is not None: launch[changed] = 'changed'
    if changed is None:
        job.verify_inputs(launch); assert calls == ['environment', 'native_implementation', 'bound_inputs']
    else:
        with pytest.raises(ValueError): job.verify_inputs(launch)
        assert 'bound_inputs' not in calls


def test_bounded_metadata_rejects_nonfinite_oversize_and_existing_files(tmp_path, monkeypatch):
    path = tmp_path/'record.json'; monkeypatch.setattr(job, 'MAX_METADATA_BYTES', 40)
    for payload in ({'value': float('nan')}, {'value': 'x'*40}):
        with pytest.raises(ValueError): job.bounded_json(path, payload)
        assert not path.exists()
    job.bounded_json(path, {'complete': False})
    with pytest.raises(FileExistsError): job.bounded_json(path, {'complete': True})
    assert json.loads(path.read_text()) == {'complete': False}
