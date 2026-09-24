"""An exact live owner is waited for; missing or interrupted evidence is not success."""
from types import SimpleNamespace

import pytest

from scripts import await_go2_nominal_measured_plane_native_v1 as waiter


def test_original_owner_is_waited_for_before_reading_its_actual_result(monkeypatch, tmp_path):
    live = iter([True, True, False]); sleeps = []; events = []; checks = []; digests = []
    monkeypatch.setattr(waiter.native.inputs.learned, 'OUTPUT', tmp_path)
    monkeypatch.setattr(waiter.run, 'owner_live', lambda owner: next(live))
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda root, ids: checks.append((root, ids)))
    monkeypatch.setattr(waiter.time, 'sleep', sleeps.append)
    monkeypatch.setattr(waiter.run, 'digest', lambda path: digests.append(path) or 'actual-result-sha')
    assert waiter.wait_for_learned(events.append) == 'actual-result-sha'
    assert sleeps == [30, 30] and len(checks) == 3
    assert events == ['EXACT_LEARNED_NATIVE_OWNER_LIVE']*2
    assert digests == [tmp_path/'result.json']


def test_observation_error_repolls_the_same_owner_without_consuming_completion(monkeypatch, tmp_path):
    seen, events, sleeps = [], [], []
    def observe(owner):
        seen.append(owner)
        if len(seen) == 1: raise PermissionError('transient process inspection failure')
        return len(seen) == 2
    monkeypatch.setattr(waiter.native.inputs.learned, 'OUTPUT', tmp_path)
    monkeypatch.setattr(waiter.run, 'owner_live', observe)
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(waiter.time, 'sleep', sleeps.append)
    def digest(path):
        assert len(seen) == 3
        return 'actual-ended-result'
    monkeypatch.setattr(waiter.run, 'digest', digest)
    assert waiter.wait_for_learned(lambda status, **details: events.append(status)) == 'actual-ended-result'
    assert seen == [waiter.native.inputs.LEARNED_OWNER]*3 and sleeps == [30, 30]
    assert events == ['LEARNED_OWNER_OBSERVATION_RETRY', 'EXACT_LEARNED_NATIVE_OWNER_LIVE']


def test_ended_failed_parent_cannot_dispatch_nominal_child(monkeypatch, tmp_path):
    (tmp_path/'failure.json').write_text('{}')
    monkeypatch.setattr(waiter.native.inputs.learned, 'OUTPUT', tmp_path)
    monkeypatch.setattr(waiter.run, 'owner_live', lambda owner: False)
    monkeypatch.setattr(waiter.run, 'verify_artifacts', lambda *args: None)
    monkeypatch.setattr(waiter.run, 'digest', lambda path: pytest.fail('failed parent treated as complete'))
    with pytest.raises(ValueError): waiter.wait_for_learned(lambda status: None)
