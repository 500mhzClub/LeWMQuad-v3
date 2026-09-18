"""Ablate early preferred-heading reversal of a latched recovery turn."""
from lewm.full_reserve_heading_release_development import (
    FullReserveHeadingReleaseRuntime, release_heading_turn,
)


def retain_recovery_heading(selection, state):
    proposed, _ = release_heading_turn(selection, state)
    if proposed is selection:
        return selection, state
    return selection | dict(suppressed_early_heading_release=proposed['full_reserve_heading_release']), state


class NoEarlyHeadingReleaseMixin:
    """Insert immediately before FullReserveHeadingReleaseRuntime in the MRO.

    Only that layer's preferred-turn release is ablated. Existing measured
    completion, translation-progress release, clearance checks, recovery
    direction switches and outer selection/dispatch layers still execute.
    """
    def _select_clear_prediction(self, *args, **kwargs):
        result = super(FullReserveHeadingReleaseRuntime, self)._select_clear_prediction(*args, **kwargs)
        if self.terminal_position_approach:
            return result
        result, self.clearance_turn = retain_recovery_heading(result, self.clearance_turn)
        return result
