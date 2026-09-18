from threading import Lock
from unittest.mock import patch

from lewm.continuous_commitment_ledger_development import ContinuousCommitmentLedger
from lewm.delayed_action_planning_development import ScheduledCommand
from lewm.persistent_local_visual_recovery_development import PersistentLocalVisualRuntime
from lewm.visual_recovery_dispatch_hold_development import VisualRecoveryDispatchHoldRuntime, REASON


def fixture():
    runtime = VisualRecoveryDispatchHoldRuntime.__new__(VisualRecoveryDispatchHoldRuntime)
    runtime.lock = Lock()
    runtime.clock_ns = lambda: 1_280_000_000
    runtime.visual_plan_minimum_ns = -1
    runtime.visual_dispatch_events = []
    runtime.rejected_windows = {}
    runtime.commitment_ledger = ContinuousCommitmentLedger()
    plan = ScheduledCommand.prepare('left_turn', observed_ns=1_000_000_000,
        completed_ns=1_200_000_000, delay_ticks=3, commit_ticks=4)
    runtime.plans = [plan]
    runtime.commitment_ledger.commit(plan, 1_200_000_000, ((0., 0., 0.),)*3)
    receipt = dict(frame=1, measured_ns=1_100_000_000, selected_features=[22, 34],
        camera_cadence_recovery=True,
        recovery_state_at_observation=dict(trigger_ns=1_100_000_000))
    return runtime, plan, receipt


def test_publication_cancels_future_window_without_backdating_prefix():
    runtime, plan, receipt = fixture()
    runtime._publish_visual_recovery(receipt)
    assert runtime.rejected_windows[plan.observed_ns] == REASON
    ledger = runtime.commitment_ledger
    assert ledger.vetoes[plan.observed_ns] == 1_280_000_000
    assert ledger.prefix_at(1_200_000_000)[1] == plan.command
    assert ledger.prefix_at(1_300_000_000) == ((0., 0., 0.),)*3
    runtime._publish_visual_recovery(receipt)
    assert len(runtime.visual_dispatch_events) == 1


def test_inflight_pretrigger_plan_is_discarded():
    runtime, plan, receipt = fixture()
    runtime._publish_visual_recovery(receipt)
    runtime.planning = [{}]
    with patch.object(PersistentLocalVisualRuntime, '_store_plan') as store:
        runtime._store_plan(plan, 1_290_000_000, ((0., 0., 0.),)*3)
        store.assert_not_called()
    assert runtime.planning[-1] == dict(committed=False, discard_reason=REASON)


def test_gate_and_request_ledger_preserve_new_plan_and_existing_vetoes():
    runtime, plan, receipt = fixture()
    runtime._publish_visual_recovery(receipt)
    passed = dict(reason='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED',
        requested_command=[0., 0., .45], command_observation_ns=plan.observed_ns)
    with patch.object(PersistentLocalVisualRuntime, '_command_gate', side_effect=lambda r, n: r):
        held = runtime._command_gate(passed, 1_300_000_000)
        assert held['reason'] == REASON and held['requested_command'] == [0., 0., 0.]
        newer = passed | dict(command_observation_ns=1_400_000_000)
        assert runtime._command_gate(newer, 1_700_000_000)['requested_command'] == newer['requested_command']
        veto = newer | dict(reason='CURRENT_STOPPING_MARGIN_VETO', requested_command=[0., 0., 0.])
        assert runtime._command_gate(veto, 1_700_000_000)['reason'] == veto['reason']
    # A later forecast based on a prefix interrupted by this hold must fail
    # the existing ledger check, rather than silently use its old forecast.
    ledger = runtime.commitment_ledger
    ledger.record_request(1_000_000_000, (0., 0., .45))
    for ns in range(1_020_000_000, 1_300_000_000, 20_000_000):
        ledger.record_request(ns, (0., 0., 0.))
    assert not ledger.prefix_was_requested(plan)
