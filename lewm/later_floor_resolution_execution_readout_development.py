"""Read actually executed intervals cleared by later measured floor evidence."""
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.measured_floor_partition_development import FOOT_IDS
from lewm.nominal_reentry_execution_readout_development import executed_motion


def later_floor_resolution_execution(poses, tape, rows):
    entries = []
    for row in rows:
        d = row['decision']; s = d['new_selection'] or {}
        if s.get('action') is None or d['terminal'] is not None: continue
        action = s['action']; i = ACTIONS.index(action); check = s['surface_checks'][i]
        original = check['original_contact_check_before_later_floor_resolution']
        if (check['possible_intersection'] or check['non_foot_contacts_exempted']
                or check['unresolved_contacts_exempted']):
            raise ValueError('selected command must preserve unresolved and non-foot contact gates')
        if not original['possible_intersection']: continue
        if (d['selected_action'] != action or d['requested_command'] != candidate_commands(action)[0]
                or d['later_measured_floor_contact_resolution_enabled'] is not True):
            raise ValueError('actual matching later-floor-resolution command required')
        proofs = check['later_floor_contact_resolution']
        keys = [(q['source_camera'], q['shape_id']) for q in proofs]
        expected = {(camera, foot) for camera in ('primary', 'auxiliary') for foot in FOOT_IDS}
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError('all eight original foot queries required')
        total = 0
        for camera, field in (('primary', 'shapes'), ('auxiliary', 'auxiliary_shapes')):
            before, after = original[field], check[field]
            if len(before) != len(after): raise ValueError('all original shapes required')
            for a, b in zip(before, after, strict=True):
                if a['shape_id'] != b['shape_id']: raise ValueError('original ordered shapes required')
                if a['shape_id'] not in FOOT_IDS:
                    if a != b: raise ValueError('non-foot query changed')
                    continue
                q = proofs[keys.index((camera, a['shape_id']))]
                if (q['original_intersections'] != a['intersecting_voxels']
                        or q['remaining_intersections'] != b['intersecting_voxels']
                        or q['remaining_intersections'] != 0
                        or q['original_intersections'] != q['resolved_intersections']+q['remaining_intersections']
                        or len(q['enclosures']) != q['original_intersections']
                        or not q['all_original_returns_and_partitions_retained']):
                    raise ValueError('complete original/resolved/remaining contact accounting required')
                for enclosure in q['enclosures']:
                    witness = enclosure['later_single_view']
                    if (not enclosure['resolved'] or not witness
                            or not enclosure['latest_ambiguous_sample_frame'] < witness['frame'] <= row['tick']
                            or witness['measured_ns'] != 1_500_000_000+100_000_000*witness['frame']):
                        raise ValueError('strictly later already observed floor witness required')
                total += q['resolved_intersections']
        if total <= 0: raise ValueError('actual original contact conflict must be resolved')
        entries.append(dict(tick=row['tick'], action=action, requested_command=d['requested_command'],
            predicted_body_xy_m=s['prediction'][i][0][:2],
            original_surface_possible_intersection=True, revised_surface_possible_intersection=False,
            resolved_intersections=total, later_floor_contact_resolution=proofs))
    records = executed_motion(poses, tape, entries)
    return dict(records=records, completed_intervals=sum(r['complete_100ms_execution'] for r in records),
        censored_intervals=sum(not r['complete_100ms_execution'] for r in records),
        scope='actually selected commands whose original contact check blocked before later-floor resolution',
        all_policy_changes_attributed_to_these_intervals=False,
        original_controller_counterfactual_trajectory_inferred=False,
        native_outcomes_used_for_policy=False)
