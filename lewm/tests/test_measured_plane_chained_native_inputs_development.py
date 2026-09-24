"""Closed-owner admission, artifact integrity and actual replay-boundary gates."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import measured_plane_chained_native_inputs_development as inputs
from lewm.tests.test_measured_plane_chained_native_prefix_development import fixture_report


def setup(tmp_path, monkeypatch, *, terminal_only=False):
    roots = {name: tmp_path/name for name in ('chained', 'reactive', 'replay')}
    for root in roots.values(): root.mkdir()
    monkeypatch.setattr(inputs.chained_wait, 'OUTPUT', roots['chained'])
    monkeypatch.setattr(inputs.reactive_wait, 'OUTPUT', roots['reactive'])
    monkeypatch.setattr(inputs.chained_wait.job, 'OUTPUT', roots['replay'])
    sources = {'frozen.py': 'frozen-source'}
    live = set(); calls = []

    def write(root, name, value):
        (root/name).write_text(json.dumps(value))

    def read(root, name):
        return json.loads((root/name).read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def verify_artifacts(root, ids):
        for name, expected in ids.items():
            if digest(root/name) != expected: raise ValueError('artifact digest changed')

    monkeypatch.setattr(inputs.run, 'read_json', read)
    monkeypatch.setattr(inputs.run, 'digest', digest)
    monkeypatch.setattr(inputs.run, 'verify_artifacts', verify_artifacts)
    monkeypatch.setattr(inputs.run, 'verify', lambda bindings: None)
    monkeypatch.setattr(inputs.run, 'owner_live', lambda owner: owner['pid'] in live)
    report = fixture_report(7, terminal_only)
    write(roots['replay'], 'result.json', {'report': report})
    expected_chained = dict(native_result_sha256=inputs.LEARNED_RESULT_SHA,
        timing_waiter_result_sha256='timing-result', replay_result_sha256=digest(roots['replay']/'result.json'))
    expected_reactive = dict(learned_result_sha256=inputs.LEARNED_RESULT_SHA,
        reactive_result_sha256=inputs.REACTIVE_NATIVE_RESULT_SHA,
        nominal_wait_result_sha256=inputs.NOMINAL_WAIT_RESULT_SHA,
        actual_physical_prefix_reconstructed=True, reactive_measured_round_trip_successes=0)
    boot = inputs.run.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for name, owner, receipt in [('chained', inputs.CHAINED_OWNER, expected_chained),
            ('reactive', inputs.REACTIVE_OWNER, expected_reactive)]:
        launch = dict(owner=owner, boot_id=boot, automatic_retry=False, source_sha256=sources)
        if name == 'chained':
            launch.update(predecessor_owner=inputs.chained_wait.job.CPU_OWNER,
                predecessor_launch_sha256=inputs.chained_wait.job.CPU_LAUNCH_SHA)
        else:
            launch.update(original_owner=inputs.reactive_wait.native.inputs.NOMINAL_WAIT_OWNER,
                original_launch_sha256=inputs.reactive_wait.native.inputs.NOMINAL_WAIT_LAUNCH_SHA)
        write(roots[name], 'launch.json', launch)
        write(roots[name], 'events.jsonl', [])
        completion = 'completion.json' if name == 'chained' else 'native_completion.json'
        write(roots[name], completion, receipt)
        write(roots[name], 'replay_stdout.log' if name == 'chained' else 'native_stdout.log', '')
        status = ('MEASURED_PLANE_CHAINED_CONTROLLER_WAIT_V1_COMPLETE' if name == 'chained'
            else 'REACTIVE_MEASURED_PLANE_NATIVE_WAIT_V1_COMPLETE')
        write(roots[name], 'result.json', dict(status=status, source_sha256=sources,
            report=receipt, artifact_sha256={}, automatic_retry=False, native_execution=False,
            navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))

    def seal(name):
        root = roots[name]; result = read(root, 'result.json')
        names = ['launch.json', 'events.jsonl'] + (['completion.json', 'replay_stdout.log']
            if name == 'chained' else ['native_completion.json', 'native_stdout.log'])
        result['artifact_sha256'] = {n: digest(root/n) for n in names}
        write(root, 'result.json', result)
        monkeypatch.setattr(inputs, name.upper()+'_LAUNCH_SHA', digest(root/'launch.json'))
        sha = digest(root/'result.json')
        if name == 'reactive': monkeypatch.setattr(inputs, 'REACTIVE_WAIT_RESULT_SHA', sha)
        return sha

    def chained_check(actual_sources, native_sha, timing_sha):
        assert actual_sources == sources and native_sha == inputs.LEARNED_RESULT_SHA and timing_sha == 'timing-result'
        calls.append('chained')
        return deepcopy(expected_chained)

    def reactive_check(actual_sources, nominal_sha):
        assert actual_sources == sources and nominal_sha == inputs.NOMINAL_WAIT_RESULT_SHA
        calls.append('reactive')
        return deepcopy(expected_reactive)

    monkeypatch.setattr(inputs.chained_wait, 'completed_child', chained_check)
    monkeypatch.setattr(inputs.reactive_wait, 'completed_child', reactive_check)
    sha = seal('chained'); seal('reactive')
    return SimpleNamespace(roots=roots, sources=sources, live=live, calls=calls, sha=sha,
        write=write, read=read, digest=digest, seal=seal, report=report,
        expected_chained=expected_chained, expected_reactive=expected_reactive)


@pytest.mark.parametrize('terminal_only', [False, True])
def test_complete_receipts_and_actual_boundary_are_required_without_success_selection(tmp_path, monkeypatch, terminal_only):
    h = setup(tmp_path, monkeypatch, terminal_only=terminal_only)
    result = inputs.admit(h.sha, h.sources)
    assert h.calls == ['chained', 'reactive']
    assert result['boundary']['intervention'] == 7
    assert result['boundary']['command_changed'] is not terminal_only
    assert result['boundary']['terminal_changed'] is terminal_only
    assert result['prefix_report'] == h.report
    assert result['predecessor_scientific_success_required'] is False
    assert result['complete_predecessor_artifact_rosters_reauthenticated'] is True
    for key in ('controller_replay_reexecuted', 'future_native_execution_started',
            'future_native_physical_prefix_reconstructed', 'hardware_resources_admitted',
            'native_execution_serialization_checked', 'navigation_recovered', 'goal_achieved'):
        assert result[key] is False


@pytest.mark.parametrize('name', ['chained', 'reactive'])
def test_live_owner_prevents_any_completed_child_reconstruction(tmp_path, monkeypatch, name):
    h = setup(tmp_path, monkeypatch)
    h.live.add(getattr(inputs, name.upper()+'_OWNER')['pid'])
    with pytest.raises(ValueError, match='finish and end'): inputs.admit(h.sha, h.sources)
    assert h.calls == []


@pytest.mark.parametrize('name', ['chained', 'reactive'])
@pytest.mark.parametrize('symlink', [False, True])
def test_failure_marker_including_dangling_symlink_prevents_bypass(tmp_path, monkeypatch, name, symlink):
    h = setup(tmp_path, monkeypatch); failure = h.roots[name]/'failure.json'
    if symlink: failure.symlink_to(h.roots[name]/'absent')
    else: failure.write_text('{}')
    with pytest.raises(ValueError, match='failed predecessor'): inputs.admit(h.sha, h.sources)
    assert h.calls == []


@pytest.mark.parametrize('fault', ['owner', 'boot', 'predecessor', 'retry'])
def test_rebound_launch_still_requires_exact_queue_identity(tmp_path, monkeypatch, fault):
    h = setup(tmp_path, monkeypatch); root = h.roots['chained']; launch = h.read(root, 'launch.json')
    if fault == 'owner': launch['owner']['created'] += 1
    elif fault == 'boot': launch['boot_id'] = 'different-boot'
    elif fault == 'predecessor': launch['predecessor_launch_sha256'] = 'different'
    else: launch['automatic_retry'] = True
    h.write(root, 'launch.json', launch); sha = h.seal('chained')
    with pytest.raises(ValueError): inputs.admit(sha, h.sources)


@pytest.mark.parametrize('fault', ['status', 'source', 'claim', 'missing_artifact', 'receipt', 'native_identity', 'raw_tamper'])
def test_completed_waiter_evidence_cannot_be_replaced_or_weakened(tmp_path, monkeypatch, fault):
    h = setup(tmp_path, monkeypatch); root = h.roots['chained']; d = h.read(root, 'result.json')
    if fault == 'status': d['status'] = 'RUNNING'
    elif fault == 'source': d['source_sha256']['frozen.py'] = 'changed'
    elif fault == 'claim': d['native_execution'] = True
    elif fault == 'missing_artifact': del d['artifact_sha256']['events.jsonl']
    elif fault == 'receipt': d['report']['extra'] = 'unbound'
    elif fault == 'native_identity':
        d['report']['native_result_sha256'] = 'different'; h.write(root, 'completion.json', d['report'])
        d['artifact_sha256']['completion.json'] = h.digest(root/'completion.json')
    else: (root/'events.jsonl').write_text('changed')
    h.write(root, 'result.json', d)
    with pytest.raises(ValueError): inputs.admit(h.digest(root/'result.json'), h.sources)
    assert h.calls == []


@pytest.mark.parametrize('name', ['chained', 'reactive'])
def test_existing_full_child_checker_rejection_is_not_bypassed(tmp_path, monkeypatch, name):
    h = setup(tmp_path, monkeypatch)
    def reject(*args): raise ValueError('full child artifact or replay reconstruction failed')
    monkeypatch.setattr(getattr(inputs, name+'_wait'), 'completed_child', reject)
    with pytest.raises(ValueError, match='full child'): inputs.admit(h.sha, h.sources)


@pytest.mark.parametrize('name', ['chained', 'reactive'])
def test_changed_reconstructed_receipt_is_rejected(tmp_path, monkeypatch, name):
    h = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(getattr(inputs, name+'_wait'), 'completed_child', lambda *args: {'changed': True})
    with pytest.raises(ValueError, match='reconstruct'): inputs.admit(h.sha, h.sources)


@pytest.mark.parametrize('fault', ['candidate_failure', 'following_outcome', 'changed_replay_hash'])
def test_no_native_admission_from_failed_or_contaminated_candidate(tmp_path, monkeypatch, fault):
    h = setup(tmp_path, monkeypatch); report = deepcopy(h.report)
    if fault == 'candidate_failure': report['boundary_candidate']['failure'] = 'tracking failed'
    elif fault == 'following_outcome': report['following_changed_command_outcome_consumed'] = True
    else: report['changed'] = True
    h.write(h.roots['replay'], 'result.json', {'report': report})
    if fault != 'changed_replay_hash':
        h.expected_chained['replay_result_sha256'] = h.digest(h.roots['replay']/'result.json')
        d = h.read(h.roots['chained'], 'result.json'); d['report'] = h.expected_chained
        h.write(h.roots['chained'], 'result.json', d)
        h.write(h.roots['chained'], 'completion.json', h.expected_chained)
        h.sha = h.seal('chained')
    with pytest.raises(ValueError): inputs.admit(h.sha, h.sources)
    assert 'reactive' not in h.calls
