"""Verify a view-recovery change against the exact frozen predecessor transform."""
from copy import deepcopy
from lewm.nominal_clearance_reentry_development import reenter
from lewm.view_reentry_selection_development import reenter_with_translation

REENTRY_ADDITIONS = (
    'original_score_contract', 'nominal_clearance_reentry', 'reentry_current_clearance',
    'reentry_candidates', 'original_nominal_radius_m', 'original_nominal_path_veto_preserved',
    'original_candidate_utilities_preserved', 'recovery_is_nominal_policy_exception',
    'reentry_guaranteed', 'articulated_motion_certified')


def compare_selection(old, new, position_map, rotation_map_from_body, occupied):
    if old is None or old.get('mode') != 'VIEW_ACQUISITION' or 'prediction' not in old:
        if new != old: raise ValueError('selection outside view recovery changed')
        return
    base = deepcopy(old)
    if old.get('nominal_clearance_reentry', False):
        original_contract = base['original_score_contract']
        for key in REENTRY_ADDITIONS: del base[key]
        base.update(action=None, action_index=None, requested_command=[0., 0., 0.],
            phase_admissible_candidates=0, score_contract=original_contract)
    # Inversion is never trusted alone: reconstruct the complete recorded old
    # selection, including all geometry checks, using the frozen old function.
    if reenter(base, position_map, rotation_map_from_body, occupied) != old:
        raise ValueError('complete original view recovery does not reconstruct')
    if reenter_with_translation(base, position_map, rotation_map_from_body, occupied) != new:
        raise ValueError('complete translating view recovery does not reconstruct')
