from types import SimpleNamespace
import pytest
from lewm.continuous_round_trip_runtime_development import preceding_requests
from lewm.continuous_round_trip_runtime_development import ContinuousRoundTripRuntime
from lewm.continuous_commitment_runtime_development import ContinuousCommitmentRuntime
from lewm.continuous_commitment_ledger_development import ContinuousCommitmentLedger
from threading import Lock


def test_arrival_dwell_cannot_hide_earlier_motion_with_a_final_zero():
    ns=1_600_000_000
    rows={t:(0.,0.,0.) for t in range(ns-100_000_000,ns,20_000_000)}
    rows[ns-60_000_000]=(.2,0.,0.)
    command,history=preceding_requests(SimpleNamespace(requests=rows),1,ns)
    assert command==(.2,0.,0.) and history[-1]==(0.,0.,0.) and len(history)==5
    rows[ns-60_000_000]=(0.,0.,0.)
    assert preceding_requests(SimpleNamespace(requests=rows),1,ns)[0]==(0.,0.,0.)
    del rows[ns-80_000_000]
    with pytest.raises(ValueError,match='all five'):preceding_requests(SimpleNamespace(requests=rows),1,ns)


def test_delayed_registration_keeps_actual_acquisition_interval(monkeypatch):
    runtime=ContinuousRoundTripRuntime.__new__(ContinuousRoundTripRuntime)
    runtime.lock=Lock();runtime.frame_request_history={}
    runtime.commitment_ledger=ledger=ContinuousCommitmentLedger()
    monkeypatch.setattr(ContinuousCommitmentRuntime,'submit',lambda self,packet:None)
    start=1_500_000_000
    for i in range(5):ledger.record_request(start+i*20_000_000,(.2,0.,0.) if i==2 else (0.,0.,0.))
    runtime.submit(SimpleNamespace(frame=1,measured_ns=start+100_000_000))
    for i in range(5,160):ledger.record_request(start+i*20_000_000,(0.,0.,0.))
    assert start not in ledger.requests
    command,history=runtime.frame_request_history.pop(1)
    assert command==(.2,0.,0.) and history[2]==(.2,0.,0.) and len(history)==5
