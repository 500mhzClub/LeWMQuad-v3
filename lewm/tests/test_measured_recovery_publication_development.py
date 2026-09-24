from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest

from lewm.measured_latency_simulation_development import MeasuredLatencyClock
from lewm.measured_recovery_publication_development import MeasuredRecoveryPublicationMixin
from lewm.visual_recovery_dispatch_hold_development import VisualRecoveryDispatchHoldRuntime, REASON


class ObservedClock(MeasuredLatencyClock):
    def __init__(self):
        super().__init__(100)
        self.entered = Event()
        self.returned = Event()

    def __call__(self):
        self.entered.set()
        value = super().__call__()
        self.returned.set()
        return value


def runtime(method, clock):
    vetoes = []
    state = SimpleNamespace(lock=RLock(), clock_ns=clock, visual_plan_minimum_ns=-1,
        plans=[SimpleNamespace(observed_ns=50, expires_ns=10**12)], rejected_windows={},
        commitment_ledger=SimpleNamespace(veto=lambda p, ns: vetoes.append((p.observed_ns, ns))),
        visual_dispatch_events=[])
    receipt = dict(frame=1, measured_ns=100, camera_cadence_recovery=True,
        selected_features=10, recovery_state_at_observation=dict(trigger_ns=100))
    errors = []

    def publish():
        clock.begin('registration')
        try:
            method(state, receipt)
        except BaseException as error:
            errors.append(error)
        finally:
            clock.end()

    return state, receipt, vetoes, errors, publish


@pytest.mark.parametrize('method,request_lock_available', [
    (VisualRecoveryDispatchHoldRuntime._publish_visual_recovery, False),
    (MeasuredRecoveryPublicationMixin._publish_visual_recovery, True),
])
def test_clock_wait_does_not_prevent_simulator_request(method, request_lock_available):
    clock = ObservedClock()
    state, _, vetoes, errors, publish = runtime(method, clock)
    worker = Thread(target=publish, daemon=True)
    worker.start()
    try:
        assert clock.entered.wait(2)
        acquired = state.lock.acquire(timeout=.1)
        if acquired:
            state.lock.release()
        assert acquired is request_lock_available
        assert not clock.returned.is_set()
        # An external advance releases even the old deadlock for test cleanup.
        clock.advance(10**9)
        worker.join(2)
        assert not worker.is_alive() and not errors
        assert vetoes == [(50, 10**9)]
    finally:
        clock.close()
        worker.join(2)


def test_publication_timestamp_includes_wait_for_controller_lock_and_duplicate_is_ignored():
    clock = ObservedClock()
    method = MeasuredRecoveryPublicationMixin._publish_visual_recovery
    state, receipt, vetoes, errors, publish = runtime(method, clock)
    worker = Thread(target=publish, daemon=True)
    worker.start()
    try:
        assert clock.entered.wait(2)
        with state.lock:
            clock.advance(10**9)
            assert clock.returned.wait(2)
            clock.advance(2*10**9)
        worker.join(2)
        assert not worker.is_alive() and not errors
        assert vetoes == [(50, 2*10**9)]
        assert state.rejected_windows == {50: REASON}
        assert state.visual_dispatch_events[0]['published_ns'] == 2*10**9
        method(state, receipt)
        assert len(vetoes) == len(state.visual_dispatch_events) == 1
    finally:
        clock.close()
        worker.join(2)
