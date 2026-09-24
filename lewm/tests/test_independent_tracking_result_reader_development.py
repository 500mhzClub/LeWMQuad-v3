"""Complete reader on synthetic96-stream evidence; no real processes or physics.

Source-definition/learning authentication and raw acquisition are explicit mocks.
Worker/keeper records, real score verification and predecessor comparison run.
"""
from copy import deepcopy
import hashlib
import os

import numpy as np
import pytest

from scripts import read_go2_independent_tracking_challenge_v1 as mod
from scripts import navigation_artifact_root_development as custody
from lewm.tests.test_independent_tracking_scored_population_verification_development import scored
from lewm.tests.test_independent_tracking_stress_cohort_development import prepared, template
from lewm.tests.test_independent_tracking_predecessor_comparison_development import data, wall


def save(root, name, value):
    payload = mod.base.encode(value); (root / name).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def complete(scored, data, tmp_path, monkeypatch):
    store, result, calls = scored; c = mod.challenge; m = mod.memory
    monkeypatch.setattr(c, 'OUTPUT', store.output)
    outside = tmp_path / 'go2_outside_fixture_attempt_001'; outside.mkdir()
    monkeypatch.setattr(m, 'OUTPUT', outside); monkeypatch.setattr(m, 'INNER_OUTPUT', store.output)
    definition = dict(source_sha256={c.PROTOCOL: 'a' * 64}, synthetic_source_definition=True)
    definition_sha = c.learning.identity(definition)
    completed = dict(study_result_sha256='b' * 64, synthetic_completed_learning=True)
    monkeypatch.setattr(c, 'definition', lambda: deepcopy(definition))
    monkeypatch.setattr(c, 'completed_learning', lambda sha: deepcopy(completed) if sha == 'b' * 64 else None)
    monkeypatch.setattr(c, 'verify_ordered_launch', lambda d: None)
    supervisor = dict(pid=123, cgroup='/synthetic/outside')
    request = dict(schema='independent_tracking_outside_supervision_request.v1',
        definition_sha256=definition_sha, study_result_sha256='b' * 64,
        scope_contract=m.contract(), output_root=str(store.output), supervisor=supervisor)
    request_sha = save(outside, 'request.json', request)
    scope = dict(unit=m.UNIT, pid=124,
        cgroup=f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{m.UNIT}',
        controls={'memory.max': str(m.MEMORY_BYTES), 'memory.swap.max': '0',
            'memory.oom.group': '1', 'pids.max': str(m.TASKS)})
    launch = dict(definition=definition, definition_sha256=definition_sha, completed_learning=completed,
        memory_supervision=dict(request_sha256=request_sha, scope=scope, supervisor=supervisor))
    store.hashes['launch.json'] = save(store.output, 'launch.json', launch)
    for index, trial in enumerate(c.TRIALS):
        worker = dict(trial=trial, launch_sha256=store.hashes['launch.json'], specification=c.specification(trial),
            protocol_sha256='a' * 64,
            previous_receipt_sha256=None if index == 0 else store.hashes[c.TRIALS[index - 1] + '_receipt.json'])
        name = trial + '_worker_request.json'; store.hashes[name] = save(store.output, name, worker)
        episode = mod.base.read(store.output, trial + '_receipt.json')
        receipt = dict(trial=trial, launch_sha256=store.hashes['launch.json'],
            request_sha256=store.hashes[name], **episode)
        for suffix, value in (('_worker_receipt.json', receipt), ('_worker_exit.json', dict(trial=trial, returncode=0))):
            store.hashes[trial + suffix] = save(store.output, trial + suffix, value)
    result['output_sha256'] = {n: store.hashes[n] for n in mod.scores.expected_names()}
    store.hashes['result.json'] = save(store.output, 'result.json', result)
    # Authenticated synthetic predecessor roots, not historical runtime access.
    cohorts = {}
    for key in ('inner', 'intent'):
        root = tmp_path / f'go2_prior_{key}_fixture_attempt_001'; root.mkdir()
        sha = save(root, 'launch.json', dict(synthetic=True))
        cohorts[key] = dict(root=str(root), receipt_sha256={'launch.json': sha}, selected_artifact_sha256={},
            witnesses={t: deepcopy(data[3]) for t in mod.predecessor.PRIOR_TRIALS})
    monkeypatch.setattr(mod.predecessor, 'load_predecessors', lambda: dict(cohorts=deepcopy(cohorts)))
    monkeypatch.setattr(mod.predecessor, '_read_pose', lambda root, name: dict(
        timestamp_s=np.arange(1, 852) * .002, base_pose_world=np.tile([0., 0., 0., 0., 0., 0., 1.], (851, 1))))
    monkeypatch.setattr(mod.predecessor, 'IntentReturnRGBDReplay', lambda root: data[2](3))
    read = mod.base.read
    def synthetic_read(root, name):
        if root == store.output and name.endswith('/static_objects.json'): return [wall()]
        if root == store.output and name.endswith('/camera_audit.json'): return data[1][:3]
        return read(root, name)
    monkeypatch.setattr(mod.base, 'read', synthetic_read)
    comparison = mod._compare_predecessors(store.output, store.hashes['result.json'], store.episodes)
    store.hashes['predecessor_comparison.json'] = save(store.output, 'predecessor_comparison.json', comparison)
    terminal = dict(status='NATIVE_TRACKING_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE',
        definition_sha256=definition_sha, result_sha256=store.hashes['result.json'],
        comparison_sha256=store.hashes['predecessor_comparison.json'], output_sha256=dict(store.hashes),
        resource_contract=c.resource_contract(), independent_result_verification_complete=False,
        full_challenge_pass=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False)
    challenge_sha = save(store.output, 'challenge_result.json', terminal)
    log_sha = save(outside, 'unit.log', dict(synthetic_diagnostic=True)); size = (outside / 'unit.log').stat().st_size
    keeper = dict(status='SCOPED_CHALLENGE_COLLECTION_AND_EVALUATIONS_COMPLETE',
        request_sha256=request_sha, definition_sha256=definition_sha, unit=m.UNIT,
        child_handle_terminal=True, retry_performed=False, navigation_qualified=False, goal_achieved=False,
        systemd_run_returncode=0, workload_termination_inferred_from_returncode=False,
        log_total_bytes=size, log_retained_bytes=size, log_omitted_bytes=0, log_sha256=log_sha,
        log_complete=True, failure_cause_inferred_from_exit_code=False,
        challenge_result_sha256=challenge_sha, workload_completion_verified=True)
    keeper_sha = save(outside, 'terminal.json', keeper)
    calls.clear()
    return store, terminal, keeper, (challenge_sha, keeper_sha, definition_sha), calls


def test_actual_scoring_and48_unavailable_comparisons_do_not_become_a_scientific_pass(complete):
    store, terminal, keeper, args, calls = complete
    result = mod.read_result(*args)
    assert calls == list(mod.challenge.TRIALS)
    assert result['source_launch_worker_and_supervisor_records_verified']
    assert result['scored_population']['pose_stream_count'] == 96
    comparison = result['predecessor_comparison']
    assert sum(len(v) for v in comparison['comparisons'].values()) == 48
    assert all(p['status'] == 'UNAVAILABLE_COMPARISON' for row in comparison['comparisons'].values() for p in row.values())
    assert not comparison['all_six_predecessor_nonidentity_checks_pass']
    assert not result['historical_kernel_controls_independently_observed']
    assert not result['full_challenge_pass'] and not result['goal_achieved']
    assert result['recorded_artifact_bytes'] > 0


@pytest.mark.parametrize('fault', ['keeper_exit', 'keeper_live', 'keeper_error', 'log_omitted', 'log_size',
    'wrong_unit', 'wrong_definition', 'missing_artifact', 'worker_order', 'worker_exit', 'worker_receipt',
    'scope_limit', 'same_pid', 'failure'])
def test_authority_corruption_blocks_before_scoring_or_predecessor_access(complete, monkeypatch, fault):
    store, terminal, keeper, args, calls = complete
    challenge_sha, keeper_sha, definition_sha = args
    if fault == 'keeper_exit': keeper['systemd_run_returncode'] = 1
    elif fault == 'keeper_live': keeper['child_handle_terminal'] = False
    elif fault == 'keeper_error': keeper['error'] = 'retained interruption'
    elif fault == 'log_omitted': keeper['log_omitted_bytes'] = 1
    elif fault == 'log_size': keeper['log_total_bytes'] += 1
    elif fault == 'wrong_unit': keeper['unit'] = 'unrelated.service'
    elif fault == 'wrong_definition': definition_sha = 'f' * 64
    elif fault == 'missing_artifact': del terminal['output_sha256']['predecessor_comparison.json']
    elif fault == 'failure': save(store.output, 'failure.json', dict(retained_failure=True))
    else:
        trial = mod.challenge.TRIALS[0]
        if fault.startswith('worker_'):
            suffix = {'worker_order': '_worker_request.json', 'worker_exit': '_worker_exit.json',
                'worker_receipt': '_worker_receipt.json'}[fault]
            name = trial + suffix; row = mod.base.read(store.output, name)
            if fault == 'worker_order': row['previous_receipt_sha256'] = 'f' * 64
            elif fault == 'worker_exit': row['returncode'] = 1
            else: row['request_sha256'] = 'f' * 64
        else:
            name = 'launch.json'; row = mod.base.read(store.output, name)
            if fault == 'scope_limit': row['memory_supervision']['scope']['controls']['memory.max'] = 'max'
            else: row['memory_supervision']['scope']['pid'] = row['memory_supervision']['supervisor']['pid']
        sha = save(store.output, name, row); terminal['output_sha256'][name] = sha
        result = mod.base.read(store.output, 'result.json'); result['output_sha256'][name] = sha
        sha = save(store.output, 'result.json', result)
        terminal['result_sha256'] = terminal['output_sha256']['result.json'] = sha
    challenge_sha = save(store.output, 'challenge_result.json', terminal)
    keeper['challenge_result_sha256'] = challenge_sha
    keeper_sha = save(mod.memory.OUTPUT, 'terminal.json', keeper)
    def forbidden(*args, **kwargs): raise AssertionError('authority failure must precede scoring/native access')
    monkeypatch.setattr(mod.scores, 'verify_scored_population', forbidden)
    monkeypatch.setattr(mod, '_compare_predecessors', forbidden)
    with pytest.raises(ValueError): mod.read_result(challenge_sha, keeper_sha, definition_sha)
    assert calls == []


def test_rebound_predecessor_pass_claim_rejected_after_real_scoring(complete):
    store, terminal, keeper, args, calls = complete
    comparison = mod.base.read(store.output, 'predecessor_comparison.json')
    comparison['all_six_predecessor_nonidentity_checks_pass'] = True
    sha = save(store.output, 'predecessor_comparison.json', comparison)
    terminal['comparison_sha256'] = terminal['output_sha256']['predecessor_comparison.json'] = sha
    challenge_sha = save(store.output, 'challenge_result.json', terminal)
    keeper['challenge_result_sha256'] = challenge_sha
    keeper_sha = save(mod.memory.OUTPUT, 'terminal.json', keeper)
    with pytest.raises(ValueError, match='predecessor comparisons differ'):
        mod.read_result(challenge_sha, keeper_sha, args[2])
    assert calls == list(mod.challenge.TRIALS)
