import numpy as np
import pytest
import torch
from lewm import terminal_translation_pulse_development as pulse
from lewm.delayed_action_planning_development import ScheduledCommand, delayed_candidate_inputs
from lewm.predictive_arrival_hold_development import PredictiveArrivalHoldRuntime


def test_forecast_and_correction_use_same_pulse_and_unchanged_turns(monkeypatch):
    history = {k:torch.zeros(s) for k,s in dict(rgb=(4,3,96,128),body=(4,20,63),control=(4,15,7)).items()}
    prefix = [[.2, 0., 0.], [0., 0., .45], [0., 0., 0.]]
    original = delayed_candidate_inputs(history, prefix, delay_ticks=3, commit_ticks=4)
    unchanged = pulse.pulse_inputs(history, prefix, delay_ticks=3, commit_ticks=4)
    assert torch.equal(original['known_action_blocks'], unchanged['known_action_blocks'])
    captured = []
    monkeypatch.setattr(pulse, 'pose_features', lambda *_: (None, None, None))
    def features(prediction, past, commands):
        captured.append(commands.copy())
        return np.zeros((8, 4))
    monkeypatch.setattr(pulse, 'features', features)
    correction = object.__new__(pulse.PulseMotionResidual)
    correction.fit = dict(mean=np.zeros((8,4)), scale=np.ones((8,4)), coefficient=np.zeros((8,4,2)), bias=np.zeros((8,2)))
    token = pulse._pulse.set(True)
    try:
        inputs = pulse.pulse_inputs(history, prefix, delay_ticks=3, commit_ticks=4)
        correction.correct(np.zeros((6,8,5)), {}, 4, prefix)
    finally:
        pulse._pulse.reset(token)
    decoded = (inputs['known_action_blocks']*torch.tensor([.3,1.,.5]))[:,:,0].numpy()
    np.testing.assert_allclose(decoded, np.stack(captured), atol=1e-7, rtol=0)
    np.testing.assert_allclose(decoded[:,:3], np.broadcast_to(prefix,(6,3,3)), atol=1e-7, rtol=0)
    assert np.count_nonzero(decoded[1:4,3]) and not np.count_nonzero(decoded[1:4,4:])
    assert np.count_nonzero(decoded[4:6,3:7]) and not np.count_nonzero(decoded[:,7:])


def test_pulse_expires_and_next_committed_prefix_contains_actual_zero_tail():
    ledger = pulse.PulseCommitmentLedger()
    first = ScheduledCommand(0,300_000_000,400_000_000,(.2,0.,0.))
    ledger.commit(first,250_000_000,[(0.,0.,0.)]*3)
    prefix = ledger.prefix_at(400_000_000)
    assert prefix == ((0.,0.,0.),)*3
    second = ScheduledCommand(400_000_000,700_000_000,1_100_000_000,(0.,0.,.45))
    ledger.commit(second,650_000_000,prefix)
    for now in range(0,700_000_000,20_000_000):
        ledger.record_request(now,first.request(now_ns=now,fresh_observation_allows_motion=True))
    assert ledger.prefix_was_requested(second)
    assert first.request(now_ns=399_999_999,fresh_observation_allows_motion=True)==[.2,0.,0.]
    assert first.request(now_ns=400_000_000,fresh_observation_allows_motion=True)==[0.,0.,0.]
    ledger.requests[500_000_000] = (.2,0.,0.)
    assert not ledger.prefix_was_requested(second)


@pytest.mark.parametrize('fault',['late','wrong_prefix','overlap','short_turn'])
def test_pulse_keeps_deadline_prefix_and_overlap_checks(fault):
    ledger = pulse.PulseCommitmentLedger()
    plan = ScheduledCommand(0,300_000_000,400_000_000,(.2,0.,0.))
    prefix = [(0.,0.,0.)]*3; completed = 200_000_000
    if fault == 'late': completed = 300_000_001
    elif fault == 'wrong_prefix': prefix[0] = (.2,0.,0.)
    elif fault == 'overlap': ledger.commit(plan,completed,prefix)
    else: plan = ScheduledCommand(0,300_000_000,400_000_000,(0.,0.,.45))
    with pytest.raises(ValueError): ledger.commit(plan,completed,prefix)


def test_only_terminal_translations_change_the_stored_command_window(monkeypatch):
    stored = []
    monkeypatch.setattr(PredictiveArrivalHoldRuntime, '_store_plan', lambda self, plan, *_: stored.append(plan))
    runtime = object.__new__(pulse.TerminalTranslationPulseRuntime)
    for terminal, command, expected in ((False,(.2,0.,0.),700_000_000),
            (True,(.2,0.,0.),400_000_000),(True,(0.,0.,.45),700_000_000)):
        runtime.planning_translation_pulse = terminal
        original = ScheduledCommand(0,300_000_000,700_000_000,command)
        runtime._store_plan(original,200_000_000,[(0.,0.,0.)]*3)
        assert stored[-1].expires_ns == expected
        assert original.expires_ns == 700_000_000
