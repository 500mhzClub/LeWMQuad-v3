"""Pessimistic unknown cells with the scripted look-around exempt (development, 2 October 2026; Andrew's option A).

The precondition-seeded rule (lewm/dev_pessimistic_unknown_seeded_development.py) still stalled
the scripted initial look-around (route status INITIAL_PANORAMA_REQUIRES_VIEW) on half the smoke
mazes. The robot's body drifts 4.9-6.8 cm (median 5.8 cm) while turning on the spot for the
look-around, the same for every controller, and that exceeds the room between the blocking
radius (0.425 m + the e_f bound) and the 0.5-m precondition.

Exemption: during the scripted look-around only, never-observed cells do not block a move.
Justification: under the operating precondition (robot placed in a cleared 0.5-m area), the body
cannot leave the cleared disc while its centre stays within 0.5 - 0.425 = 0.075 m of the start
(0.425 m is the body's largest reach). The observed maximum drift is 6.8 cm; the exemption ends
for good if the centre drifts more than 0.07 m from the start, and it ends when the look-around
completes. From then on the rule applies in full, with the start disc and traversed track seeded
as before. The exemption is the same for every controller. Remembered walls keep their own
requirements throughout.
"""
import math

from lewm import dev_harness_fixes_development as fixes
from lewm.dev_pessimistic_unknown_seeded_development import PRECONDITION_CLEAR_M, seeded_pessimistic_unknown_mixin

LOOKAROUND = 'INITIAL_PANORAMA_REQUIRES_VIEW'
DRIFT_LIMIT_M = .07


def lookaround_exempt_pessimistic_unknown_mixin(bound_m, level, start_clear_m=PRECONDITION_CLEAR_M, drift_limit_m=DRIFT_LIMIT_M):
    """The precondition-seeded rule with the scripted look-around exempt while centre drift stays within the limit."""
    Seeded = seeded_pessimistic_unknown_mixin(bound_m, level, start_clear_m)

    class LookaroundExemptPessimisticUnknownMixin(Seeded):
        pessimistic_unknown = dict(Seeded.pessimistic_unknown, lookaround_exempt=True, lookaround_drift_limit_m=drift_limit_m,
                                   exemption='scripted look-around only; ends on completion or centre drift beyond the limit')

        def _route(self, *args, **kwargs):
            route = super()._route(*args, **kwargs)
            self._dev_route_status = route.get('status') if isinstance(route, dict) else None
            return route

        def _lookaround(self, position):
            state = self.__dict__.setdefault('_dev_lookaround', dict(ended=None, max_drift_m=0., seen=False))
            start = self._dev_unknown_state['start']
            drift = math.dist(tuple(float(v) for v in position[:2]), start)
            state['max_drift_m'] = max(state['max_drift_m'], drift)
            inside = getattr(self, '_dev_route_status', None) == LOOKAROUND
            if state['ended'] is None:
                if inside:
                    state['seen'] = True
                    if drift > drift_limit_m:
                        state['ended'] = 'centre drift beyond limit'
                elif state['seen']:
                    state['ended'] = 'look-around complete'
            exempt = inside and state['ended'] is None
            return exempt, dict(lookaround_exempt=exempt, lookaround_drift_m=round(drift, 4), lookaround_max_drift_m=round(state['max_drift_m'], 4),
                                lookaround_exemption_ended=state['ended'])

        def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
            context, record = self._unknown_context(snapshot, position)
            exempt, info = self._lookaround(position)
            token = fixes._UNKNOWN_CELLS.set(None if exempt else context)
            try:
                result = super(Seeded, self)._select_clear_prediction(selected, prediction, snapshot, position, rotation)
            finally:
                fixes._UNKNOWN_CELLS.reset(token)
            selection = result[0] if isinstance(result, tuple) else result
            selection['dev_pessimistic_unknown'] = dict(record, **info)
            return result

        def _route_target(self, route, snapshot, position):
            context, _ = self._unknown_context(snapshot, position)
            exempt, _ = self._lookaround(position)
            token = fixes._UNKNOWN_CELLS.set(None if exempt else context)
            try:
                return super(Seeded, self)._route_target(route, snapshot, position)
            finally:
                fixes._UNKNOWN_CELLS.reset(token)

    LookaroundExemptPessimisticUnknownMixin.__name__ = f'LookaroundExemptPessimisticUnknown_{level}'
    return LookaroundExemptPessimisticUnknownMixin
