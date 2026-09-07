"""Keep bounded heading feedback active through initial acceptance dwell.

This successor changes command generation, not the frozen acceptance predicate,
deadline or later task. It does not establish clearance or post-release accuracy.
"""
import numpy as np

from lewm.initially_aligned_continuation_development import (
    FineInitialBearingAlignment, InitiallyAlignedContinuation)


class PersistentBearingAlignment(FineInitialBearingAlignment):
    def observe(self, packet, attitude, *, now_ns):
        result = super().observe(packet, attitude, now_ns=now_ns)
        if not self.terminal:
            result['requested_command'] = [0., 0., float(np.clip(
                1.5*result['heading_error_rad'], -.35, .35))]
        return result


class PersistentAlignedContinuation(InitiallyAlignedContinuation):
    def _observe(self, packet, fast_packet, *, now_ns):
        result = super()._observe(packet, fast_packet, now_ns=now_ns)
        if result.get('selected_initial_bearing') is not None:
            # The predecessor creates its unused controller on this zero-command
            # proposal frame. Select the new operator before its first observation;
            # never replace an active controller or its accumulated dwell/history.
            if self.initial_turn.start_ns is not None:
                raise RuntimeError('cannot replace an active alignment operator')
            self.initial_turn = PersistentBearingAlignment(
                self.initial_bearing['direction_initial_body'])
        return result
