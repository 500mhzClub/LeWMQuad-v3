"""Forecast and execute 100-ms translations near the exact mission endpoint.

Planning still occurs every 400 ms. Translation candidates become one command
interval followed by zero; their costs include the original settling horizon.
Turns, measured arrival, clearance, and actual-prefix checks are unchanged.
"""
from contextvars import ContextVar
from dataclasses import replace
import numpy as np
import torch
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.continuous_commitment_ledger_development import ContinuousCommitmentLedger, Commitment, PERIOD
from lewm.closed_loop_motion_residual_development import FrozenMotionResidual, features, pose_features
from lewm.paced_multirate_controller_development import PacedMultirateController
from lewm.eligible_floor_registration_development import bind
from lewm.predictive_arrival_hold_development import PredictiveArrivalHoldRuntime

TRANSLATIONS = ('forward', 'left_arc', 'right_arc')
_pulse = ContextVar('terminal_translation_pulse', default=False)


def command_sequences(prefix, *, pulse):
    commands = np.zeros((6, 8, 3), dtype=float)
    commands[:, :3] = np.asarray(prefix, float)
    for i, action in enumerate(ACTIONS):
        count = 1 if pulse and action in TRANSLATIONS else 4
        commands[i, 3:3+count] = candidate_commands(action)[0]
    return commands


def pulse_inputs(history, prefix, *, delay_ticks, commit_ticks):
    result = delayed_candidate_inputs(history, prefix, delay_ticks=delay_ticks, commit_ticks=commit_ticks)
    if _pulse.get():
        if (delay_ticks, commit_ticks) != (3, 4):
            raise ValueError('terminal pulse retains three prefix and four scoring intervals')
        result['known_action_blocks'] = torch.as_tensor(
            command_sequences(prefix, pulse=True)[:, :, None], dtype=torch.float32)/torch.tensor([.3, 1., .5])
    return result


class PulseMotionResidual(FrozenMotionResidual):
    def correct(self, prediction, poses, frame, prefix):
        if not _pulse.get():
            return super().correct(prediction, poses, frame, prefix)
        past, _, _ = pose_features(poses, frame)
        corrected = prediction.copy()
        for i, commands in enumerate(command_sequences(prefix, pulse=True)):
            x = features(prediction[i], past, commands)
            for h in range(8):
                correction = ((x[h]-self.fit['mean'][h])/self.fit['scale'][h])@self.fit['coefficient'][h]+self.fit['bias'][h]
                corrected[i, h, :2] += correction
        if not np.isfinite(corrected).all():
            raise ValueError('finite motion correction required')
        return corrected


class PulseCommitmentLedger(ContinuousCommitmentLedger):
    def commit(self, plan, completed_ns, prefix):
        duration = plan.expires_ns-plan.dispatch_ns
        if duration == 4*PERIOD:
            return super().commit(plan, completed_ns, prefix)
        prefix = tuple(tuple(c) for c in prefix)
        if (duration != PERIOD or not any(plan.command[:2])
                or plan.dispatch_ns != plan.observed_ns+3*PERIOD
                or completed_ns > plan.dispatch_ns or prefix != self.prefix_at(plan.observed_ns)):
            raise ValueError('on-time translation pulse must preserve the known observation-time prefix')
        if any(max(c.plan.dispatch_ns, plan.dispatch_ns) < min(c.plan.expires_ns, plan.expires_ns)
                for c in self.commitments):
            raise ValueError('committed action windows overlap')
        self.commitments.append(Commitment(plan, completed_ns, prefix))


_select_pulse_action = bind(PacedMultirateController._select_action, delayed_candidate_inputs=pulse_inputs)


class TerminalTranslationPulseRuntime(PredictiveArrivalHoldRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commitment_ledger = PulseCommitmentLedger()
        correction = object.__new__(PulseMotionResidual)
        correction.fit = self.motion_residual.fit
        self.motion_residual = correction
        self.planning_translation_pulse = False

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        self.planning_translation_pulse = bool(self.terminal_position_approach and scan_error is None)
        token = _pulse.set(self.planning_translation_pulse)
        try:
            selected, correction = _select_pulse_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q)
        finally:
            _pulse.reset(token)
        short = self.planning_translation_pulse and selected['action'] in TRANSLATIONS
        selected['command_duration_ns'] = PERIOD if short else 4*PERIOD
        selected['terminal_translation_pulse'] = dict(enabled=self.planning_translation_pulse,
            selected_translation_pulse=short, translation_command_duration_ns=PERIOD,
            planning_cadence_ns=4*PERIOD, scoring_endpoint_offset_ns=7*PERIOD,
            translation_zero_tail_before_next_dispatch_ns=3*PERIOD,
            progress_score_includes_settling_tail=True,
            forecast_and_residual_command_sequences_identical=True,
            frozen_residual_accuracy_on_short_pulses_established=False)
        correction['terminal_translation_pulse'] = self.planning_translation_pulse
        return selected, correction

    def _store_plan(self, plan, completed, prefix):
        if self.planning_translation_pulse and any(plan.command[:2]):
            plan = replace(plan, expires_ns=plan.dispatch_ns+PERIOD)
        super()._store_plan(plan, completed, prefix)
