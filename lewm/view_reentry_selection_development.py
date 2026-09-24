"""Explicit translating recovery while the original selector is acquiring a view.

Reuse the frozen all-cell recovery checks; retain the original phase receipt.
Only a violated nominal radius can activate this experimental exception.
"""
from copy import deepcopy
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.nominal_clearance_reentry_development import reenter


def reenter_with_translation(selection, position_map, rotation_map_from_body, occupied):
    if (selection.get('mode') != 'VIEW_ACQUISITION' or 'prediction' not in selection
            or selection.get('action') is not None or selection.get('view_budget_exhausted', False)):
        return reenter(selection, position_map, rotation_map_from_body, occupied)
    allowed = selection['phase_allowed_actions']
    if allowed != ['hold', 'left_turn', 'right_turn']:
        raise ValueError('original ordered view-acquisition phase receipt required')
    expanded = deepcopy(selection)
    expanded['phase_allowed_actions'] = list(ACTIONS)
    result = reenter(expanded, position_map, rotation_map_from_body, occupied)
    if result is expanded:
        return selection
    result['phase_allowed_actions'] = deepcopy(allowed)
    for row in result['reentry_candidates']:
        row['reentry_phase_allowed'] = row['phase_allowed']
        row['phase_allowed'] = row['action'] in allowed
        row['requires_view_phase_exception'] = not row['phase_allowed']
    result.update(view_reentry_translation_enabled=True,
        original_phase_allowance_preserved=True,
        selected_action_requires_view_phase_exception=result['action'] not in allowed,
        reentry_phase_allowed_actions=list(ACTIONS),
        reentry_admissible_candidates=result['phase_admissible_candidates'],
        original_phase_admissible_candidates=selection['phase_admissible_candidates'],
        phase_admissible_candidates=sum(r['eligible'] and r['phase_allowed'] for r in result['reentry_candidates']))
    return result
