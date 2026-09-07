"""Navigation-task outcome separate from the unchanged full-scan assay."""
import copy

from lewm.initially_aligned_metrics_development import reduce_aligned_continuation


def task_scan_acquisition(decisions, policy):
    events = [r for r in decisions if r['controller'].get('scan_stop_evidence') is not None]
    if len(events) > 1: raise ValueError('one partial scan stopping event maximum')
    if not events: return None
    row = events[0]; c = row['controller']; event = c['scan_stop_evidence']
    scan = c['scan']; selected = c['selected_side_branch']; now = row['decision_ns']
    if (policy not in ('scan_only', 'both') or c['acquisition_policy'] != policy
            or c['stage'] != 'SCAN' or c['next_stage'] != 'HOLD_ALIGN'
            or c['status'] != 'RUNNING' or c['terminal']
            or c['requested_command'] != [0., 0., 0.]
            or scan is None or scan['status'] != 'SCANNING'
            or scan['new_completed_view'] is None or not 1 <= scan['completed_target_views'] <= 4
            or selected is None or selected['observed_ns'] != now or selected['selected_ns'] != now
            or selected['candidate']['timestamp_ns'] != now
            or selected['candidate'] not in (c['selected_view_proposals'] or [])
            or selected['requires_fresh_forward_reobservation'] is not True
            or selected['qualified_exit'] is not False or selected['qualified_traversal'] is not False
            or selected['place_identity'] is not None
            or event != {'reason': 'FRESH_SIDE_BRANCH_AT_ACQUIRED_VIEW',
                         'decision_ns': now, 'observed_ns': now,
                         'completed_target_views': scan['completed_target_views'],
                         'full_circle_complete': False}):
        raise ValueError('partial scan requires fresh unqualified branch evidence and actual stop decision')
    return copy.deepcopy(event)


def task_outcome_checks(original_checks, acquisition):
    result = {k: v for k, v in original_checks.items() if k != 'completed_scan'}
    result['scan_evidence_acquired'] = bool(original_checks['completed_scan'] or acquisition is not None)
    return result


def reduce_task_acquisition(spec, raw, start, decisions, terminal, stop_reason, sensor_fault, model):
    original = reduce_aligned_continuation(spec, raw, start, decisions, terminal, stop_reason, sensor_fault, model)
    acquisition = task_scan_acquisition(decisions, spec['acquisition_policy'])
    checks = task_outcome_checks(original['checks'], acquisition)
    return {**original, 'task_checks': checks, 'task_two_leg_integration_success': all(checks.values()),
            'scan_interrupted_after_evidence': acquisition is not None,
            'scan_stopping_evidence': acquisition,
            'task_scope': 'observed two-leg navigation, retaining full-scan metric separately; no place, memory or hardware qualification'}
