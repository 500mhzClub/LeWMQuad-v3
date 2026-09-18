"""Actual selected intervals cleared by the additional measured floor partition."""
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.nominal_reentry_execution_readout_development import executed_motion


def confirmed_floor_execution(poses, tape, rows):
    entries = []
    for row in rows:
        d = row['decision']; s = d['new_selection'] or {}
        if s.get('action') is None or d['terminal'] is not None: continue
        action = s['action']; i = ACTIONS.index(action)
        check = s['surface_checks'][i]; old = check['original_auxiliary_floor_contact_check']
        if (check['shapes'] != old['shapes']
                or check['primary_possible_intersection'] != old['primary_possible_intersection']
                or check['non_foot_contacts_exempted'] or check['non_floor_or_unknown_contacts_exempted']):
            raise ValueError('only declared measured auxiliary foot classification may change')
        if check['possible_intersection']:
            raise ValueError('selected command must satisfy revised surface check')
        if not old['possible_intersection']: continue
        if (old['primary_possible_intersection'] or not old['auxiliary_possible_intersection']
                or d['selected_action'] != action or d['requested_command'] != candidate_commands(action)[0]
                or d['current_primary_floor_confirmation_enabled'] is not True):
            raise ValueError('actual command newly cleared by auxiliary floor confirmation required')
        entries.append(dict(tick=row['tick'], action=action, requested_command=d['requested_command'],
            predicted_body_xy_m=s['prediction'][i][0][:2],
            original_surface_possible_intersection=True, revised_surface_possible_intersection=False,
            original_auxiliary_shapes=old['auxiliary_shapes'],
            revised_auxiliary_shapes=check['auxiliary_shapes'],
            current_primary_floor_confirmation=check['current_primary_floor_confirmation']))
    records = executed_motion(poses, tape, entries)
    return dict(records=records, completed_intervals=sum(r['complete_100ms_execution'] for r in records),
        censored_intervals=sum(not r['complete_100ms_execution'] for r in records),
        scope='actually selected commands whose original auxiliary contact check blocked',
        all_policy_changes_attributed_to_these_intervals=False,
        original_controller_counterfactual_trajectory_inferred=False,
        native_outcomes_used_for_policy=False)
