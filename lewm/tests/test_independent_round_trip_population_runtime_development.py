"""Synthetic lifecycle integration, including one actual spawned-process crash.

No native scene, model load, new-layout packet or original raw audit is run.
"""
from concurrent.futures import Future, ProcessPoolExecutor as RealPool
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
import json
import os

import numpy as np
import pytest

from scripts import independent_round_trip_population_runtime_development as runtime
from scripts import navigation_artifact_root_development as guard
from scripts import independent_round_trip_population_case_evidence_development as evidence
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from lewm.tests.test_independent_round_trip_population_readout_development import fixture as original_fixture


def synthetic_verifier(launch, *, full=False):
    if launch.get('synthetic_reject'):
        raise ValueError('synthetic final admission rejected')


def mutating_verifier(launch, *, full=False):
    launch['changed'] = True


def false_verifier(launch, *, full=False):
    return False


def crash_worker():
    # Actual child-process death, without reading an artifact or creating a scene.
    os._exit(23)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'BASE', tmp_path)
    for name, value in {'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1', 'PYTHONHASHSEED':'0'}.items():
        monkeypatch.setenv(name, value)
    output = tmp_path/'go2_synthetic_population_runtime_attempt_001'; output.mkdir()
    source_names = (runtime.SOURCE, runtime.TEST, runtime.PROTOCOL, evidence.SOURCE, evidence.TEST, evidence.PROTOCOL)
    launch = dict(source_sha256={name:digest(ROOT/name) for name in source_names},
        protocol=runtime.PROTOCOL, output_root=str(output), ordered_cases=runtime.manifest()['ordered_cases'],
        runtime=deepcopy(runtime.FIXED_RUNTIME), final_policy_review_completed=True,
        complete_input_admission_performed=True, native_queue_completion_verified=True,
        robot_urdf_sha256=digest(runtime.URDF),
        runtime_verifier=dict(source=runtime.TEST, function='synthetic_verifier'),
        input_admission={'factory_correction_admission': {'synthetic_fixture': True}})
    write_json(output/'launch.json', launch)
    records, reports, _ = original_fixture()
    for record, report in zip(records, reports, strict=True):
        record['collection'].update(physics_samples=750, decisions=0, completed_ticks=0, command_ticks=0,
            terminal_zero_ticks=0, acquisition_stop='SYNTHETIC_EARLY_PACKET_STOP',
            schedule_terminal=None, mission_receipt=None, setup_checked=False, rgbd_frames=0, auxiliary_frames=0)
        report['observation_and_control_wall_ms'] = []; report['iteration_with_receipt_wall_ms'] = []
    return dict(output=output, launch=launch, records=records, reports=reports)


def rewrite_launch(env, **updates):
    env['launch'].update(updates)
    (env['output']/'launch.json').write_text(json.dumps(env['launch'], indent=2)+'\n')
    return digest(env['output']/'launch.json')


def make_collection(env, case):
    collection = deepcopy(env['records'][runtime.CASES.index(case)]['collection'])
    for name in evidence.collection_names(case, collection):
        path = env['output']/name; path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('/result.json'): write_json(path, collection)
        elif name.endswith('/physics_trace.npz'): np.savez(path, physics_contact=np.zeros(750, np.uint8))
        else: path.write_bytes(b'SYNTHETIC PLACEHOLDER; NOT RAW EVIDENCE\n')
    return collection


@pytest.mark.parametrize('field,value', [
    ('final_policy_review_completed', False), ('complete_input_admission_performed', False),
    ('native_queue_completion_verified', False), ('robot_urdf_sha256', '0'*64),
    ('ordered_cases', []), ('runtime', {}), ('protocol', 'unbound.md'),
    ('runtime_verifier', {'source':runtime.TEST, 'function':'wrong'})])
def test_unadmitted_or_changed_launch_rejected(environment, field, value):
    env = environment; sha = rewrite_launch(env, **{field:value})
    with pytest.raises(ValueError): runtime.checked_launch(env['output'], sha, synthetic_verifier)


@pytest.mark.parametrize('verifier', [mutating_verifier, false_verifier])
def test_verifier_cannot_return_false_or_mutate_launch(environment, verifier):
    sha = rewrite_launch(environment, runtime_verifier=dict(source=runtime.TEST, function=verifier.__name__))
    with pytest.raises(ValueError): runtime.checked_launch(environment['output'], sha, verifier)


def test_bound_verifier_checks_actual_evidence_even_when_flags_true(environment):
    sha = rewrite_launch(environment, synthetic_reject=True)
    with pytest.raises(ValueError, match='synthetic final admission rejected'):
        runtime.checked_launch(environment['output'], sha, synthetic_verifier, full=True)


def test_unbound_closure_verifier_rejected(environment):
    sha = rewrite_launch(environment)
    def unbound(launch, *, full=False): pass
    with pytest.raises(ValueError): runtime.checked_launch(environment['output'], sha, unbound)


@pytest.mark.parametrize('name', ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONHASHSEED'])
def test_changed_execution_environment_rejects_before_collection(environment, monkeypatch, name):
    monkeypatch.setenv(name, '8'); sha = rewrite_launch(environment)
    with pytest.raises(ValueError, match='fixed BLAS thread counts'):
        runtime.checked_launch(environment['output'], sha, synthetic_verifier)


def test_boolean_cannot_substitute_for_integer_worker_count(environment):
    changed = deepcopy(runtime.FIXED_RUNTIME); changed['native_scene_workers'] = True
    sha = rewrite_launch(environment, runtime=changed)
    with pytest.raises(ValueError): runtime.checked_launch(environment['output'], sha, synthetic_verifier)


def install_worker_stubs(env, monkeypatch, failure=None):
    events = []; geometries = []
    def geometry(path):
        assert path == runtime.URDF
        result = object(); geometries.append(result); return result
    def collect(case, definition, *, output, geometry, correction_admission):
        events.append('collect'); print('synthetic collect', flush=True)
        assert output == env['output'] and definition == env['launch']['source_sha256'][runtime.PROTOCOL]
        assert correction_admission == {'synthetic_fixture': True}
        if failure == 'collection': raise RuntimeError('synthetic collection failure')
        return make_collection(env, case)
    def audit(case, collection, definition, *, input_root, robot_geometry, correction_admission):
        events.append('audit'); print('synthetic audit', flush=True)
        assert (input_root/(case.name+'/result.json')).is_file()
        assert len(geometries) == 2 and geometries[0] is not geometries[1]
        if failure == 'audit': raise RuntimeError('synthetic raw audit failure')
        if failure == 'changed_raw': (input_root/(case.name+'/command_tape.json')).write_text('changed during audit')
        return deepcopy(env['reports'][runtime.CASES.index(case)])
    monkeypatch.setattr(runtime, 'ArticulatedCollisionGeometry', geometry)
    monkeypatch.setattr(runtime, 'collect', collect); monkeypatch.setattr(runtime, 'audit', audit)
    return events


def test_worker_calls_collection_then_audit_with_fresh_geometry_and_closed_log(environment, monkeypatch):
    env = environment; events = install_worker_stubs(env, monkeypatch)
    sha = rewrite_launch(env); case = runtime.CASES[0]
    returned = runtime.case_worker(env['output'], case, sha, None, synthetic_verifier)
    assert returned['status'] == runtime.WORKER_STATUS and events == ['collect', 'audit']
    execution = evidence.read_json(env['output'], case.name+'_worker_execution.json')
    assert execution['worker_pid'] == os.getpid()
    assert execution['collection_returned'] and execution['original_raw_audit_returned']
    assert execution['worker_log_sha256'] == digest(env['output']/(case.name+'_worker.log'))
    log = (env['output']/(case.name+'_worker.log')).read_text()
    assert log.index('synthetic collect') < log.index('synthetic audit')
    assert execution['worker_terminal_sha256'] == returned['worker_terminal_sha256']
    # The direct unit invocation was not a fresh process and cannot be accepted
    # by a production parent as though it were one.
    with pytest.raises(ValueError, match='separate fresh worker'):
        runtime.accept_worker(env['output'], case, returned, sha, None)


@pytest.mark.parametrize('failure', ['collection', 'audit', 'changed_raw'])
def test_worker_preserves_original_failure_and_stops_before_success(environment, monkeypatch, failure):
    env = environment; events = install_worker_stubs(env, monkeypatch, failure)
    sha = rewrite_launch(env); case = runtime.CASES[0]
    returned = runtime.case_worker(env['output'], case, sha, None, synthetic_verifier)
    assert returned['status'] == runtime.FAILED
    record = evidence.read_json(env['output'], case.name+'_worker_failure.json')
    assert not record['automatic_retry'] and record['evidence_preserved']
    assert not (env['output']/(case.name+'_worker_terminal.json')).exists()
    assert (env['output']/(case.name+'_worker.log')).exists()
    assert events == (['collect'] if failure == 'collection' else ['collect', 'audit'])
    if failure != 'collection':
        assert (env['output']/(case.name+'/result.json')).exists() and record['collection'] is not None
    if failure == 'changed_raw': assert record['raw_audit_report'] is not None
    with pytest.raises(ValueError): runtime.accept_worker(env['output'], case, returned, sha, None)


def test_worker_rechecks_launch_after_raw_audit(environment, monkeypatch):
    env = environment; install_worker_stubs(env, monkeypatch)
    original = runtime.audit
    def audit(*args, **kwargs):
        result = original(*args, **kwargs); rewrite_launch(env, synthetic_reject=True); return result
    monkeypatch.setattr(runtime, 'audit', audit)
    returned = runtime.case_worker(env['output'], runtime.CASES[0], rewrite_launch(env), None, synthetic_verifier)
    assert returned['status'] == runtime.FAILED
    failure = evidence.read_json(env['output'], runtime.CASES[0].name+'_worker_failure.json')
    assert failure['stage'] == 'post_audit_launch_admission' and failure['raw_audit_report'] is not None


def test_missing_fixed_reference_blocks_collection_itself(environment, monkeypatch):
    env = environment; events = install_worker_stubs(env, monkeypatch)
    returned = runtime.case_worker(env['output'], runtime.CASES[1], rewrite_launch(env), None, synthetic_verifier)
    assert returned['status'] == runtime.FAILED and events == []
    failure = evidence.read_json(env['output'], runtime.CASES[1].name+'_worker_failure.json')
    assert failure['stage'] == 'completed_reference_admission'
    assert not (env['output']/runtime.CASES[1].name).exists()


def install_parent_stubs(env, monkeypatch, *, fail_at=None, reuse_pid=False, resource_stop=None):
    events = []; submissions = []; active = [False]; checks = []
    def checked(output, sha, verifier, *, full=False):
        checks.append(full); return env['launch']
    def measured():
        free = (39 if resource_stop == len(submissions) else 1000)*1024**3
        return dict(memory_available_bytes=64*1024**3, artifact_free_bytes=free)
    def idle():
        assert not active[0]; events.append('idle')
    def worker(output, case, sha, reference_sha, verifier):
        index = runtime.CASES.index(case)
        if index == fail_at: return dict(case=case.name, status=runtime.FAILED)
        collection = make_collection(env, case)
        log = output/(case.name+'_worker.log'); log.write_text('synthetic worker, no process/native execution\n')
        ids = evidence.bind_collection(output, case, collection)
        evidence.persist_audited_case(output, case, collection, env['reports'][index], launch_sha256=sha,
            collection_artifact_sha256=ids, worker_log_sha256=digest(log), reference_worker_sha256=reference_sha)
        terminal = digest(output/(case.name+'_worker_terminal.json'))
        execution = dict(status=runtime.EXECUTION_STATUS, case=case.name, launch_sha256=sha,
            worker_terminal_sha256=terminal, worker_pid=4000000+(0 if reuse_pid else index),
            worker_create_time=10.0, collection_returned=True, original_raw_audit_returned=True,
            same_process_collection_and_raw_audit=True, collection_artifact_sha256=ids,
            worker_log_sha256=digest(log), wall_s=1., maximum_rss_bytes=1024)
        name = case.name+'_worker_execution.json'; write_json(output/name, execution)
        return dict(case=case.name, status=runtime.WORKER_STATUS, worker_terminal_sha256=terminal,
            worker_execution_sha256=digest(output/name))
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs['max_workers'] == kwargs['max_tasks_per_child'] == 1
            assert kwargs['mp_context'].get_start_method() == 'spawn'
        def __enter__(self):
            assert not active[0]; active[0] = True; events.append('enter'); return self
        def submit(self, function, *args):
            assert function is worker; submissions.append(args[1].name)
            future = Future(); future.set_result(function(*args)); return future
        def __exit__(self, *args): active[0] = False; events.append('exit')
    monkeypatch.setattr(runtime, 'checked_launch', checked)
    monkeypatch.setattr(runtime, 'hardware', measured); monkeypatch.setattr(runtime, 'require_native_idle', idle)
    monkeypatch.setattr(runtime, 'case_worker', worker); monkeypatch.setattr(runtime, 'ProcessPoolExecutor', Pool)
    return submissions, events, checks


def test_complete_serial_parent_keeps_failures_and_binds_all_parent_receipts(environment, monkeypatch):
    env = environment; submitted, events, checks = install_parent_stubs(env, monkeypatch)
    result = runtime.run_population(env['output'], rewrite_launch(env), synthetic_verifier)
    assert submitted == [case.name for case in runtime.CASES]
    assert events == ['idle', 'enter', 'exit']*32
    assert checks.count(True) == 2
    assert result['completed_episodes'] == result['fresh_worker_processes'] == 32
    assert result['population_readout']['measured_round_trip_successes'] == 0
    assert result['population_readout']['scientific_failures_retained']
    assert not result['navigation_qualified'] and not result['goal_achieved']
    assert all(case.name+'_parent_completion.json' in result['artifact_sha256'] for case in runtime.CASES)
    assert not (env['output']/'failure.json').exists()
    monitor = [json.loads(line) for line in (env['output']/'resource_monitor.jsonl').read_text().splitlines()]
    assert [row['resource_admission']['remaining_cases'] for row in monitor] == list(range(32, 0, -1))
    assert monitor[0]['resource_admission']['required_free_bytes'] == 392*1024**3


@pytest.mark.parametrize('fault', ['worker_failure', 'resource_stop', 'process_reused'])
def test_parent_stops_at_exact_incomplete_prefix(environment, monkeypatch, fault):
    options = dict(fail_at=2) if fault == 'worker_failure' else (
        dict(resource_stop=2) if fault == 'resource_stop' else dict(reuse_pid=True))
    env = environment; submitted, _, _ = install_parent_stubs(env, monkeypatch, **options)
    with pytest.raises(ValueError): runtime.run_population(env['output'], rewrite_launch(env), synthetic_verifier)
    record = evidence.read_json(env['output'], 'failure.json')
    count = 1 if fault == 'process_reused' else 2
    assert record['completed_cases'] == [case.name for case in runtime.CASES[:count]]
    assert len(submitted) == (count if fault == 'resource_stop' else count+1)
    assert not record['automatic_retry'] and not (env['output']/'result.json').exists()
    assert not (env['output']/(runtime.CASES[count].name+'_parent_completion.json')).exists()


def test_full_admission_rejection_launches_no_process(environment, monkeypatch):
    env = environment; sha = rewrite_launch(env, synthetic_reject=True)
    def forbidden(*args, **kwargs): raise AssertionError('must not launch after admission failed')
    monkeypatch.setattr(runtime, 'ProcessPoolExecutor', forbidden)
    with pytest.raises(ValueError, match='synthetic final admission rejected'):
        runtime.run_population(env['output'], sha, synthetic_verifier)
    assert evidence.read_json(env['output'], 'failure.json')['completed_cases'] == []


def test_existing_partial_population_cannot_be_resumed(environment, monkeypatch):
    env = environment; (env['output']/runtime.CASES[0].name).mkdir()
    def forbidden(*args, **kwargs): raise AssertionError('existing evidence must reject before admission')
    monkeypatch.setattr(runtime, 'checked_launch', forbidden)
    with pytest.raises(ValueError, match='no retry or resume'):
        runtime.run_population(env['output'], rewrite_launch(env), synthetic_verifier)
    assert not (env['output']/'failure.json').exists()


def test_orphan_artifact_rejects_without_opening_its_target(environment):
    env = environment
    (env['output']/'orphan_audit.json').symlink_to(env['output']/'nonexistent')
    with pytest.raises(ValueError, match='no retry or resume'):
        runtime.run_population(env['output'], rewrite_launch(env), synthetic_verifier)
    assert not (env['output']/'failure.json').exists()


def test_actual_spawned_worker_death_records_failure_without_retry(environment, monkeypatch):
    env = environment; submitted, _, _ = install_parent_stubs(env, monkeypatch)
    real_submissions = []
    class CrashPool(RealPool):
        def submit(self, function, *args, **kwargs):
            real_submissions.append(args[1].name)
            return super().submit(crash_worker)
    monkeypatch.setattr(runtime, 'ProcessPoolExecutor', CrashPool)
    with pytest.raises(BrokenProcessPool):
        runtime.run_population(env['output'], rewrite_launch(env), synthetic_verifier)
    record = evidence.read_json(env['output'], 'failure.json')
    assert record['stage'] == 'fresh_worker_collection_and_raw_audit'
    assert record['completed_cases'] == [] and not record['automatic_retry']
    assert real_submissions == [runtime.CASES[0].name] and submitted == []
    assert not (env['output']/'result.json').exists()
