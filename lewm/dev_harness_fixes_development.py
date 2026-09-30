"""Development fixes for the shared-harness traps (development mode, 30 Sep 2026).

Each fix is a mixin placed ahead of the frozen V4 `CompletedSupportRuntimeMixin`. No frozen file
is edited; the development owner swaps the composed mixin in through `bind`.

- `terminal`: near the goal, score positional progress only. The frozen scorer adds a
  heading-alignment term scaled by min(0.35 m, distance). Within a few centimetres it
  dominates, and in-place turns win even though they never bring an in-footprint target round
  (the terminal limit cycle; for example C1 validation 10/0 chose 99 right turns in a row, 4.5 cm
  from home). The arrival definition (2 cm observed, 4 cm physical, 1-s dwell) is unchanged.
"""
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.mission_coordinate_metric_development import position_distance

TERMINAL_RADIUS_M = .10


class TerminalPositionScoringMixin:
    """Within TERMINAL_RADIUS_M of the scored waypoint, select by position/contact utility alone."""

    def _score(self, prediction, goal_body, **kwargs):
        result = super()._score(prediction, goal_body, **kwargs)
        distance = float(position_distance(goal_body, kwargs.get('position_metric_matrix')))
        if distance >= TERMINAL_RADIUS_M or not result.get('candidates') or 'position_contact_utility_m' not in result['candidates'][0]:
            return result
        for row in result['candidates']:
            row['alignment_dropped_terminal'] = True
            row['utility_m'] = row['position_contact_utility_m']
        selected = max(range(len(result['candidates'])), key=lambda i: result['candidates'][i]['utility_m'])
        action = result['candidates'][selected]['action']
        result.update(action=action, action_index=selected, requested_command=candidate_commands(action)[0],
                      selection_objective='terminal_position_progress_minus_contact', terminal_alignment_dropped=True,
                      terminal_radius_m=TERMINAL_RADIUS_M)
        return result


FIXES = {'terminal': TerminalPositionScoringMixin}


def compose(fixes, base):
    """The frozen startup mixin with the requested fixes ahead of it."""
    unknown = set(fixes)-set(FIXES)
    if unknown:
        raise ValueError(f'unknown fixes: {sorted(unknown)}')
    bases = tuple(FIXES[f] for f in sorted(fixes))+(base,)
    return type('DevStartupRecoveryRuntimeMixin', bases, {})
