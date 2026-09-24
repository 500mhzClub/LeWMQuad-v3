from lewm.continuous_commitment_ledger_development import ContinuousCommitmentLedger
from lewm.delayed_action_planning_development import ScheduledCommand


def test_next_forecast_knows_existing_motion_and_checks_actual_prefix():
    ledger=ContinuousCommitmentLedger()
    first=ScheduledCommand(0,300_000_000,700_000_000,(.2,0.,0.))
    ledger.commit(first,250_000_000,[(0.,0.,0.)]*3)
    assert ledger.prefix_at(400_000_000)==((.2,0.,0.),)*3
    second=ScheduledCommand(400_000_000,700_000_000,1_100_000_000,(.2,0.,0.))
    ledger.commit(second,600_000_000,ledger.prefix_at(400_000_000))
    for ns in range(0,700_000_000,20_000_000):
        ledger.record_request(ns,[0.,0.,0.] if ns<300_000_000 else [.2,0.,0.])
    assert ledger.prefix_was_requested(second)


def test_midprefix_veto_invalidates_next_forecast_without_rewriting_its_input():
    ledger=ContinuousCommitmentLedger()
    first=ScheduledCommand(0,300_000_000,700_000_000,(.2,0.,0.))
    ledger.commit(first,250_000_000,[(0.,0.,0.)]*3)
    second=ScheduledCommand(400_000_000,700_000_000,1_100_000_000,(.2,0.,0.))
    ledger.veto(first,500_000_000)
    prefix=ledger.prefix_at(400_000_000)
    assert prefix==((.2,0.,0.),)*3
    ledger.commit(second,600_000_000,prefix)
    for ns in range(0,700_000_000,20_000_000):
        ledger.record_request(ns,[.2,0.,0.] if 300_000_000<=ns<500_000_000 else [0.,0.,0.])
    assert not ledger.prefix_was_requested(second)
