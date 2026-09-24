"""Fixed six-model development comparison; scientific failures are retained."""
from copy import deepcopy
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES, PERSISTENCE_HEADROOM_BYTES
from lewm.executed_waypoint_resource_envelope_development import COLLECTION_ALLOWANCE_BYTES
from lewm.independent_reactive_floor_transport_study_development import OUTCOME_KEYS, require_raw_audit

CASES = tuple((f'all_phase_{variant}_{condition}_residual_maze_02', 2, variant, condition,
    f'seed_2026091001_{variant}_{condition}') for variant in ('full', 'no_rgb')
    for condition in ('jepa', 'supervised_rollout', 'direct'))
WORKER_STATUS = 'ALL_PHASE_RESIDUAL_MAZE02_COLLECTED_AND_RAW_AUDITED'
IMPLEMENTATION = 'ResidualAnchoredContinuationController'
FROZEN = {
    'lewm/residual_anchored_continuation_controller_development.py':
        'f35ff18c78b4db81c0c3c766eed1015823bf972d484600907954941ef7fb946a',
    'scripts/residual_anchored_continuation_maze_episode_development.py':
        '4e98a79ea5ad16c1d2021deae52abd2aec0a38d4ace1df80d2ca3994d990e949',
    'scripts/residual_anchored_continuation_maze_audit_development.py':
        '6ae658b91a0859b8d4b7ae99ba9da22dc2541183c4a52409301bd857a24675f3',
    'docs/go2_all_phase_residual_maze02_matched_native_v1_2026-09-10.md':
        'aa239cd4933e30f6e6af64f564250a44f99233d3639ee99946481fb506aac84e',
}


def resources_for(resources, remaining):
    if type(remaining) is not int or not 1<=remaining<=len(CASES):
        raise ValueError('remaining fixed six-case cohort size required')
    required = RESERVE_BYTES+remaining*(COLLECTION_ALLOWANCE_BYTES+PERSISTENCE_HEADROOM_BYTES)
    if resources['memory_available_bytes']<32*1024**3 or resources['artifact_free_bytes']<required:
        raise ValueError('remaining complete native cohort resource allowance unavailable')
    return dict(remaining_cases=remaining, required_free_bytes=required,
        memory_admission_bytes=32*1024**3, native_scene_workers=1, os_resource_limits_enforced=False)


def require_case(case, record, report):
    if (tuple(case) not in CASES or record['case']!=case[0] or record['layout_index']!=case[1]
            or record['variant']!=case[2] or record['condition']!=case[3] or record['model_name']!=case[4]
            or record['status']!=WORKER_STATUS or 'failure' in record
            or report['layout_index']!=2 or record['model_state_unchanged'] is not True):
        raise ValueError('complete exact assigned native case and raw audit required')
    require_raw_audit(record, report, learned=True)
    expected_success=bool(report['native_evaluation']['native_round_trip_candidate_pass']
        and report['strict_physical_visibility_pass'] and not report['hard_measurement_failed_frames'])
    if report['verified_round_trip'] is not expected_success:
        raise ValueError('joint native, observed and visibility success criteria required')
    startup = record['startup_comparison']
    if (startup['common_prefix_frames']!=4 or startup['physical_prefix_samples']!=900
            or startup['physical_and_public_startup_exact'] is not True
            or startup['completed_zero_warmup_commands']!=3
            or startup['later_physical_outcomes_compared'] is not False):
        raise ValueError('complete matched physical/public startup required')


def complete_cohort(records, audits):
    if len(records)!=6 or len(audits)!=6:
        raise ValueError('all six ordered cases required, including negative outcomes')
    for case, record, report in zip(CASES, records, audits, strict=True): require_case(case, record, report)
    return dict(ordered_models=[c[4] for c in CASES], all_fixed_cases_executed=True,
        measured_round_trip_successes=sum(int(r['verified_round_trip']) for r in records),
        outcomes=[dict(model_name=r['model_name'], **deepcopy({k:r[k] for k in OUTCOME_KEYS})) for r in records],
        reused_layout_executions=6, new_independent_layout_executions=0,
        checkpoint_selection_performed=False, online_planning_advantage_established=False,
        persistent_memory_advantage_established=False, navigation_qualified=False,
        real_time_qualified=False, hardware_qualified=False, goal_achieved=False)
