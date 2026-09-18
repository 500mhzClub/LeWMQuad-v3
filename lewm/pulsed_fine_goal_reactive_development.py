"""Model-free waypoint feedback with matched observed routing and pulse windows."""
from dataclasses import replace
from lewm.clearance_preferred_reactive_runtime_development import ClearancePreferredReactiveRuntime
from lewm.terminal_position_priority_development import TerminalPositionPriorityMixin
from lewm.cached_fine_goal_route_development import cached_fine_goal_route
from lewm.terminal_translation_pulse_development import PulseCommitmentLedger, PERIOD


class PulsedFineGoalReactiveRuntime(TerminalPositionPriorityMixin, ClearancePreferredReactiveRuntime):
    def __init__(self, *args, **kwargs):
        self.planning_translation_pulse = False
        self.terminal_position_approach = False
        super().__init__(*args, **kwargs)
        self.commitment_ledger = PulseCommitmentLedger()

    def _routing_proposer(self, snapshot):
        original = super()._routing_proposer(snapshot)
        def propose(floor, occupied, position, goal, **kwargs):
            route = original(floor, occupied, position, goal, **kwargs)
            return cached_fine_goal_route(snapshot, position, goal, route)
        return propose

    def _select_action(self, packet, evidence, prefix, goal_body, scan_error, snapshot, q, Q):
        selected, correction = super()._select_action(packet, evidence, prefix,
            goal_body, scan_error, snapshot, q, Q)
        self.planning_translation_pulse = bool(self.terminal_position_approach and scan_error is None)
        short = self.planning_translation_pulse and any(selected['requested_command'][:2])
        selected['command_duration_ns'] = PERIOD if short else 4*PERIOD
        selected['terminal_translation_pulse'] = dict(enabled=self.planning_translation_pulse,
            selected_translation_pulse=bool(short), translation_command_duration_ns=PERIOD,
            planning_cadence_ns=4*PERIOD, candidate_future_outcomes_evaluated=False)
        return selected, correction

    def _store_plan(self, plan, completed, prefix):
        if self.planning_translation_pulse and any(plan.command[:2]):
            plan = replace(plan, expires_ns=plan.dispatch_ns+PERIOD)
        super()._store_plan(plan, completed, prefix)
