"""Common decision computation for enabled/disabled early heading release."""
from lewm.full_reserve_heading_release_development import (
    FullReserveHeadingReleaseRuntime, release_heading_turn,
)


def compare_heading_release(selection, state, *, suppress):
    proposed, updated = release_heading_turn(selection, state)
    if proposed is selection:
        return selection, state
    receipt = dict(eligible=True, suppressed=suppress,
                   proposed=proposed['full_reserve_heading_release'])
    return ((selection | dict(heading_release_comparison=receipt), state) if suppress
            else (proposed | dict(heading_release_comparison=receipt), updated))


class MatchedHeadingReleaseMixin:
    def _select_clear_prediction(self, *args, **kwargs):
        result = super(FullReserveHeadingReleaseRuntime, self)._select_clear_prediction(*args, **kwargs)
        if self.terminal_position_approach:
            return result
        result, self.clearance_turn = compare_heading_release(
            result, self.clearance_turn, suppress=self.suppress_early_heading_release)
        return result
