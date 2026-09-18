"""Original evidence reproduction, causal stopping and exclusive CPU ordering."""
from copy import deepcopy
import json

import pytest

from scripts import replay_go2_chained_anchor_observer_prefix_v1 as run


def evidence(frame=0):
    return dict(decision_ns=1_500_000_000+frame*100_000_000,
        status='CURRENT_VISUAL_POSE', current_pose={'frame': frame, 'p': [0., 0., 0.]})


def test_exact_original_evidence_continues():
    old = evidence()
    check = run.compare(old, deepcopy(old), deepcopy(old), frame=0)
    assert check['candidate_original_fields_exact'] and not check['stop']


def test_ineffective_fallback_metadata_does_not_claim_pose_change():
    old = evidence()
    live = old | {'chained_anchor_fallback': {'accepted': False}}
    check = run.compare(old, deepcopy(old), live, frame=0)
    assert check['candidate_original_fields_exact'] and check['fallback_attempted'] and not check['stop']


def test_first_changed_pose_stops_even_when_both_observers_admit_pose():
    old = evidence(853)
    live = deepcopy(old)
    live['current_pose']['p'][0] = .0001
    live['chained_anchor_fallback'] = {'accepted': True}
    check = run.compare(old, deepcopy(old), live, frame=853)
    assert check['stop'] and check['stop_reason'] == 'FIRST_CHANGED_OBSERVER_EVIDENCE'


@pytest.mark.parametrize('field', ['current_pose', 'status', 'extra'])
def test_original_reproduction_cannot_be_normalized_away(field):
    old = evidence()
    replayed = deepcopy(old)
    replayed[field] = None
    with pytest.raises(ValueError, match='did not reproduce'):
        run.compare(old, replayed, deepcopy(old), frame=0)


def test_unexplained_candidate_change_rejected():
    old = evidence()
    live = old | {'current_pose': None}
    with pytest.raises(ValueError, match='unexplained'):
        run.compare(old, deepcopy(old), live, frame=0)


@pytest.mark.parametrize('frame', [True, -1, 864, 1.0])
def test_invalid_frame_rejected(frame):
    with pytest.raises(ValueError, match='bounded'):
        run.compare(evidence(), evidence(), evidence(), frame=frame)


def test_invalid_clock_rejected():
    with pytest.raises(ValueError, match='clock'):
        run.compare(evidence(), evidence(), evidence(1), frame=0)


def test_original_terminal_and_fixed_limit_are_preserved():
    old = evidence() | {'status': 'VISUAL_TERMINAL_FAILURE', 'current_pose': None}
    check = run.compare(old, deepcopy(old), deepcopy(old), frame=0)
    assert check['stop_reason'] == 'OBSERVER_TERMINAL'
    old = evidence(863)
    assert run.compare(old, deepcopy(old), deepcopy(old), frame=863)['stop_reason'] == 'FIXED_PREFIX_LIMIT'


@pytest.mark.parametrize('terminal', ['result.json', 'failure.json'])
def test_wait_requires_live_owner_to_end_and_preserves_terminal(monkeypatch, tmp_path, terminal):
    monkeypatch.setattr(run, 'CPU_ROOT', tmp_path)
    payload = dict(status='COMPLETE' if terminal == 'result.json' else 'FAILED')
    if terminal == 'result.json': payload['artifact_sha256'] = {'launch.json': run.CPU_LAUNCH_SHA}
    (tmp_path/terminal).write_text(json.dumps(payload))
    states = iter([True, True, False])
    checked, sleeps, events = [], [], []
    def owner(value):
        assert value == run.CPU_OWNER
        return next(states)
    monkeypatch.setattr(run, 'owner_live', owner)
    monkeypatch.setattr(run, 'verify_artifacts', lambda root, bindings: checked.append((root, bindings.copy())))
    monkeypatch.setattr(run, 'verify', lambda sources: None)
    monkeypatch.setattr(run, 'read_json', lambda root, name: json.loads((root/name).read_text()))
    result = run.wait_for_cpu({}, lambda *a, **kw: events.append(a), sleep=sleeps.append, clock=lambda: 0)
    assert sleeps == [30, 30] and len(events) == 2
    assert result['original_cpu_owner_ended'] and result['terminal_name'] == terminal
    assert result['terminal_status'] == payload['status'] and result['no_replay_restarted']
    assert checked[0][1] == {'launch.json': run.CPU_LAUNCH_SHA}


def test_wait_timeout_does_not_admit_or_restart(monkeypatch):
    monkeypatch.setattr(run, 'owner_live', lambda owner: True)
    times = iter([0, run.WAIT_SECONDS])
    with pytest.raises(ValueError, match='no replacement'):
        run.wait_for_cpu({}, lambda *a, **kw: None, clock=lambda: next(times),
                         sleep=lambda seconds: pytest.fail('must not sleep after deadline'))


@pytest.mark.parametrize('terminals', [[], ['result.json', 'failure.json']])
def test_ended_owner_without_unique_terminal_is_not_admitted(monkeypatch, tmp_path, terminals):
    monkeypatch.setattr(run, 'CPU_ROOT', tmp_path)
    monkeypatch.setattr(run, 'owner_live', lambda owner: False)
    monkeypatch.setattr(run, 'verify_artifacts', lambda *a: None)
    for name in terminals: (tmp_path/name).write_text('{}')
    with pytest.raises(ValueError, match='one preserved terminal'):
        run.wait_for_cpu({}, lambda *a, **kw: None)
