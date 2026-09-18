"""Complete paired development outcomes; every fixed layout stays in the denominator."""
from copy import deepcopy
import numpy as np

from lewm.independent_round_trip_comparison_study_development import ARMS, CASES, require_case
from lewm.independent_round_trip_multiarm_contract_development import (
    COLLECTION_STATUS, require_collection, replay_receipt)
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit
from lewm.all_phase_residual_maze02_readout_development import traversal_counts, timing
from scripts.independent_round_trip_paired_startup_development import (
    MATCHED, require_startup, reference_case)

WORKER_STATUS = 'INDEPENDENT_ROUND_TRIP_MULTIARM_COLLECTED_AND_RAW_AUDITED'
COMPARISONS = (
    ('training_objective', 'persistent_jepa', 'persistent_supervised'),
    ('predictive_method', 'persistent_jepa', 'reactive'),
    ('planning_grid_persistence', 'persistent_jepa', 'current_pair_jepa'),
)


def observed_timing(values, expected):
    if len(values) != expected:
        raise ValueError('complete per-decision timing population required')
    if not expected:
        return dict(samples=0, median_ms=None, p95_ms=None, maximum_ms=None,
            samples_above_command_interval_100ms=0)
    return timing(values)


def case_readout(report, collection, physics_contact):
    contact = np.asarray(physics_contact)
    if (contact.ndim != 1 or len(contact) != collection['physics_samples'] or len(contact) < 750
            or not np.isin(contact, [0, 1]).all()):
        raise ValueError('complete recorded binary contact population required')
    evaluation = report['native_evaluation']; mission = collection['mission_receipt']
    decisions = collection['decisions']
    if type(decisions) is not int or decisions < 0:
        raise ValueError('nonnegative actual decision count required')
    return dict(schedule_terminal=collection['schedule_terminal'], physical_stop=collection['physical_stop'],
        acquisition_stop=collection['acquisition_stop'], terminal_zero_ticks=collection['terminal_zero_ticks'],
        observed_arrivals=[] if mission is None else deepcopy(mission['arrivals']),
        native_arrival_windows=deepcopy(evaluation['arrival_windows']),
        outbound=traversal_counts(evaluation['outbound_traversal']),
        returning=traversal_counts(evaluation['return_traversal']),
        physically_retraced_outbound_route=evaluation['physically_retraced_outbound_route'],
        native_physics_samples=len(contact), native_contact_samples=int(np.count_nonzero(contact)),
        native_contact_free=bool(not contact.any()),
        observation_and_control=observed_timing(report['observation_and_control_wall_ms'], decisions),
        iteration_with_receipt=observed_timing(report['iteration_with_receipt_wall_ms'], decisions),
        strict_physical_visibility_pass=report['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=deepcopy(report['hard_measurement_failed_frames']),
        verified_round_trip=report['verified_round_trip'], evaluator_only=True, real_time_qualified=False)


def require_completed_case(case, record, report, physics_contact, reference_collection):
    arm = require_case(case)
    for receipt in (record, report):
        require_collection(case, receipt | {'status': COLLECTION_STATUS})
    if record['status'] != WORKER_STATUS or 'failure' in record:
        raise ValueError('complete original worker required, including negative scientific results')
    collection = record['collection']; require_collection(case, collection)
    if record['model_state_sha256'] != arm.model_state_sha256:
        raise ValueError('exact assigned model identity or reactive absence required')
    require_raw_audit(record, report, learned=arm.model_name is not None)
    for key, expected in replay_receipt(case).items():
        if type(report.get(key)) is not type(expected) or report[key] != expected:
            raise ValueError('exact arm-specific replay receipt required: '+key)
    native = report['native_evaluation']
    if (type(native['native_round_trip_candidate_pass']) is not bool
            or type(report['strict_physical_visibility_pass']) is not bool):
        raise ValueError('explicit boolean native and visibility outcomes required')
    expected_success = bool(native['native_round_trip_candidate_pass']
        and report['strict_physical_visibility_pass'] and not report['hard_measurement_failed_frames'])
    if report['verified_round_trip'] is not expected_success:
        raise ValueError('joint native and strict sensing evidence required for success')
    readout = case_readout(report, collection, physics_contact)
    if record['readout'] != readout:
        raise ValueError('compact readout must reconstruct from complete original evidence')
    if expected_success and (not readout['native_contact_free'] or collection['physical_stop'] is not None
            or collection['acquisition_stop'] is not None):
        raise ValueError('physical or acquisition failure cannot become a verified round trip')
    if expected_success:
        windows = native['arrival_windows']; mission = collection['mission_receipt']
        if (len(windows) != 2 or [window['phase'] for window in windows] != ['OUTBOUND', 'RETURN']
                or not all(window['native_one_second_arrival_and_quiet_pass'] is True for window in windows)
                or native['physically_retraced_outbound_route'] is not True
                or native['terminal_native_quiet_pass'] is not True or collection['terminal_zero_ticks'] != 10
                or collection['schedule_terminal'] != 'OBSERVED_ROUND_TRIP_CANDIDATE'
                or mission is None or mission['terminal'] != 'OBSERVED_ROUND_TRIP_CANDIDATE'
                or len(mission['arrivals']) != 2):
            raise ValueError('two verified arrivals, return traversal and terminal quiet evidence required')
    require_startup(case, reference_collection, collection, record['startup_comparison'])
    return readout


def complete_population(records, audits, physics_contacts):
    if any(len(items) != len(CASES) for items in (records, audits, physics_contacts)):
        raise ValueError('all 32 ordered cases and original evidence required; no failure exclusion')
    rows = []
    for case, record, report, contact in zip(CASES, records, audits, physics_contacts, strict=True):
        first = reference_case(case); reference_index = CASES.index(first)
        readout = require_completed_case(case, record, report, contact, records[reference_index]['collection'])
        startup = record['startup_comparison']
        if startup['status'] == MATCHED:
            first_startup = records[reference_index]['startup_comparison']
            for key in ('raw_physics_prefix_sha256', 'public_startup_sha256',
                    'reference_first_command', 'reference_first_command_terminal'):
                if startup[key] != first_startup[key]:
                    raise ValueError('all arms must bind the same fixed within-layout startup reference')
        rows.append(dict(case=case.name, layout_index=case.layout_index, arm=case.arm_name,
            readout=readout, startup_comparison=deepcopy(startup)))
    lookup = {(row['layout_index'], row['arm']): row for row in rows}
    arms = []
    for arm in ARMS:
        chosen = [lookup[(index, arm.name)] for index in range(8)]
        successes = sum(int(row['readout']['verified_round_trip']) for row in chosen)
        arms.append(dict(arm=arm.name, planned_layouts=8, completed_layouts=8,
            verified_round_trips=successes, verified_round_trip_rate=successes/8,
            cases=[row['case'] for row in chosen],
            native_outbound_arrivals=sum(any(w['phase']=='OUTBOUND' and w['native_one_second_arrival_and_quiet_pass']
                for w in row['readout']['native_arrival_windows']) for row in chosen),
            contact_free_cases=sum(row['readout']['native_contact_free'] for row in chosen),
            strict_visibility_pass_cases=sum(row['readout']['strict_physical_visibility_pass'] for row in chosen),
            scientific_failures=8-successes))
    comparisons = []
    for name, left, right in COMPARISONS:
        pairs = []
        for index in range(8):
            a, b = lookup[index, left], lookup[index, right]
            pairs.append(dict(layout_index=index, left_case=a['case'], right_case=b['case'],
                left_success=a['readout']['verified_round_trip'], right_success=b['readout']['verified_round_trip'],
                startup_matched=a['startup_comparison']['status']==b['startup_comparison']['status']==MATCHED,
                left_readout=deepcopy(a['readout']), right_readout=deepcopy(b['readout'])))
        wins = sum(p['left_success'] and not p['right_success'] for p in pairs)
        losses = sum(p['right_success'] and not p['left_success'] for p in pairs)
        comparisons.append(dict(comparison=name, left_arm=left, right_arm=right, layout_pairs=8,
            left_only_successes=wins, right_only_successes=losses, tied_outcomes=8-wins-losses,
            paired_success_rate_difference=(wins-losses)/8, pairs=pairs,
            all_pairs_startup_matched=all(p['startup_matched'] for p in pairs),
            descriptive_only=True, advantage_established=False))
    return dict(status='INDEPENDENT_ROUND_TRIP_MULTIARM_COMPLETE', all_fixed_cases_executed=True,
        planned_episodes=32, completed_episodes=32, independent_layout_units=8,
        measured_round_trip_successes=sum(arm['verified_round_trips'] for arm in arms),
        arms=arms, comparisons=comparisons, ordered_case_readouts=rows,
        all_startups_matched=all(row['startup_comparison']['status']==MATCHED for row in rows),
        scientific_failures_retained=True, outcome_based_layout_replacement=False,
        treatment_repetitions_counted_as_independent_layouts=False,
        online_planning_advantage_established=False, persistent_memory_advantage_established=False,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
