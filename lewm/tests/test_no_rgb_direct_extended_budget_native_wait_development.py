"""Exact original owners and one deferred child; no real waiting or simulation."""
from copy import deepcopy

import pytest

from scripts import await_go2_no_rgb_direct_extended_budget_maze02_native_v1 as waiter


def identities():
    return {key:format(i+1, '064x') for i, (key, *_) in enumerate(waiter.PREREQUISITES)}


def test_fixed_roster_contains_batch_and_original_four_native_waiters():
    assert [s[0] for s in waiter.PREREQUISITES] == ['batch', 'frontier', 'hold', 'contact', 'tracking']
    assert [s[1]['pid'] for s in waiter.PREREQUISITES] == [2659758, 2663938, 2671835, 2703476, 2753911]
    assert waiter.PREREQUISITES[-1][3] == 'eb6c5b8b5e2f26c1ff70c03c17b3aa6761fdca5693c907b034df938447fbb68e'
    original = waiter.original.wait_for_inputs; actual = waiter.wait_for_inputs
    assert actual.__code__ is original.__code__ and actual.__closure__ is original.__closure__
    for k, v in original.__globals__.items():
        if k not in ('PREREQUISITES', 'WAIT_SECONDS', 'completion_identity'): assert actual.__globals__[k] is v


def test_command_forwards_every_exact_completion_once():
    ids = identities(); cmd = waiter.command_for(ids)
    assert cmd[:3] == [waiter.sys.executable, '-B', waiter.native.SOURCE]
    assert dict(zip(cmd[3::2], cmd[4::2], strict=True)) == dict(
        zip(['--adapter-batch-result-sha256', '--frontier-wait-result-sha256', '--hold-wait-result-sha256',
            '--contact-wait-result-sha256', '--tracking-wait-result-sha256'], ids.values(), strict=True))


@pytest.mark.parametrize('fault', ['missing', 'extra', 'invalid', 'non_string'])
def test_incomplete_or_invented_command_roster_rejected(fault):
    ids = identities()
    if fault == 'missing': ids.pop('tracking')
    elif fault == 'extra': ids['replacement'] = 'a'*64
    elif fault == 'invalid': ids['tracking'] = 'not-a-hash'
    else: ids['tracking'] = True
    with pytest.raises(ValueError): waiter.command_for(ids)


def loop_fixture(monkeypatch):
    g = waiter.wait_for_inputs.__globals__; state = dict(tick=0); events = []; checked = []
    ids = identities()
    def owner(owner):
        return state['tick'] == 0 or (state['tick'] == 1 and owner['pid'] == 2753911)
    def complete(spec, sources): checked.append((state['tick'], spec[0])); return ids[spec[0]]
    monkeypatch.setitem(g, 'owner_live', owner)
    monkeypatch.setitem(g, 'completion_identity', complete)
    monkeypatch.setitem(g, 'verify', lambda sources:None)
    def sleep(seconds): assert seconds == 30; state['tick'] += 1
    return g, state, events, checked, ids, sleep


def test_no_identity_is_inferred_while_its_original_owner_is_live(monkeypatch):
    _, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    result = waiter.wait_for_inputs({}, lambda status, **kw:events.append(kw), sleep=sleep, clock=lambda:state['tick']*30)
    assert result == ids and len(events) == 2
    assert not any(tick == 0 for tick, key in checked)
    assert (1, 'tracking') not in checked and (2, 'tracking') in checked
    assert events[0]['completed_result_sha256'] == {}


def test_changed_completed_identity_is_not_accepted_on_later_poll(monkeypatch):
    g, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    def changed(spec, sources): return 'f'*64 if state['tick'] == 2 and spec[0] == 'batch' else ids[spec[0]]
    monkeypatch.setitem(g, 'completion_identity', changed)
    with pytest.raises(ValueError, match='identity changed'):
        waiter.wait_for_inputs({}, lambda *a, **kw:None, sleep=sleep, clock=lambda:state['tick']*30)


def test_owner_ending_without_valid_completion_is_terminal(monkeypatch):
    g, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    def failed(*a): raise ValueError('original prerequisite failed')
    monkeypatch.setitem(g, 'completion_identity', failed)
    with pytest.raises(ValueError, match='prerequisite failed'):
        waiter.wait_for_inputs({}, lambda *a, **kw:None, sleep=sleep, clock=lambda:state['tick']*30)
    assert state['tick'] == 1


def test_live_owner_timeout_does_not_restart_any_job(monkeypatch):
    g, state, events, checked, ids, sleep = loop_fixture(monkeypatch)
    monkeypatch.setitem(g, 'owner_live', lambda owner:True)
    monkeypatch.setitem(g, 'WAIT_SECONDS', 30)
    with pytest.raises(ValueError, match='expired'):
        waiter.wait_for_inputs({}, lambda *a, **kw:None, sleep=sleep, clock=lambda:state['tick']*30)
    assert checked == [] and state['tick'] == 1


def test_existing_waiter_or_child_prevents_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(waiter, 'OUTPUT', tmp_path)
    monkeypatch.setattr(waiter, 'validate_root', lambda *a, **kw:None)
    monkeypatch.setattr(waiter, 'prepared_sources', lambda:pytest.fail('no source work for an existing attempt'))
    with pytest.raises(ValueError, match='exclusive'): waiter.main()


def test_completion_failure_file_is_not_bypassed(tmp_path, monkeypatch):
    monkeypatch.setattr(waiter.native, 'OUTPUT', tmp_path)
    (tmp_path/'failure.json').write_text('{}')
    monkeypatch.setattr(waiter, 'digest', lambda *a:pytest.fail('must retain failure before reading result'))
    with pytest.raises(ValueError, match='failure must be preserved'): waiter.authenticate_completed({}, identities())
