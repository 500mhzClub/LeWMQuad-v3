"""Full saved maze2 score decomposition using the frozen maze1 analysis."""
import argparse
import json
import time

from lewm.independent_reactive_floor_transport_study_development import require_raw_audit
from lewm.supervised_rollout_maze_study_development import SUPERVISED_STATE
from scripts.read_go2_supervised_maze01_turn_scores_v1 import summarize
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.maze_decision_stream_development import read_rows
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

INPUT = BASE/'go2_supervised_rollout_mazes_v1_attempt_001'
OUTPUT = BASE/'go2_supervised_maze02_turn_scores_v1_attempt_001'
CASE = 'full_supervised_rollout_novel_maze_02'
SOURCE = 'scripts/read_go2_supervised_maze02_turn_scores_v1.py'
PROTOCOL = 'docs/go2_supervised_maze02_turn_scores_v1_2026-09-10.md'
REUSED_SOURCE = 'scripts/read_go2_supervised_maze01_turn_scores_v1.py'
REUSED_SHA = '8c7119b3711bd27541275d58d746f48c8763073d187e04dc2c7e91b1f3941dc0'
FIRST_CASE = 'full_supervised_rollout_novel_maze_01'
FIXED = {
    'launch.json': '49a182da3d795f1b31910fb2b6123732db66349dfc011d768bb72fc9732a09e3',
    CASE+'_worker_terminal.json': '7a836a87a8c03a668ce5b7528bda536e905c0ea67d27d2a75523afa9f36fad38',
    'cohort_progress_after_02.json': 'a110ec00ee205b070cda97c415f71724c00bf5a704eb26ec5f408ade2b6933d1',
    FIRST_CASE+'_worker_terminal.json': '730b2d8a20d680066854427e6a660c58346d296f6073ee12ec3ae984574380e5',
    CASE+'_audit.json': 'c001cde994dd2d1ac8011fcdc9485a5a03d6cac69a2499d9e7e13cf7434e725c',
    CASE+'_prefix_comparison.json': 'b585690d9af0400fab25bc6c5109ab1c0f2adce4634dfa6b1208c2d6402ef0bc',
}
ALLOWANCE = 16*1024**2


def admit():
    verify({REUSED_SOURCE: REUSED_SHA})
    verify_artifacts(INPUT, FIXED)
    launch = read_json(INPUT, 'launch.json')
    record = read_json(INPUT, CASE+'_worker_terminal.json')
    progress = read_json(INPUT, 'cohort_progress_after_02.json')
    first = read_json(INPUT, FIRST_CASE+'_worker_terminal.json')
    if (record['status'] != 'SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED' or 'failure' in record
            or record['case'] != CASE or record['layout_index'] != 2
            or record['model_state_sha256'] != SUPERVISED_STATE or record['model_state_unchanged'] is not True
            or record['condition'] != 'supervised_rollout' or record['variant'] != 'full'
            or record['head'] != 'rollout_outcomes' or progress['completed_conditions'] != [first, record]
            or progress['remaining_layouts'] != [3] or progress['original_case_order'] != [1, 2, 3]):
        raise ValueError('exact completed second original supervised case required')
    bindings = dict(record['artifact_sha256'])
    for name, sha in {**FIXED, CASE+'_worker.log': record['worker_log_sha256']}.items():
        if name in bindings and bindings[name] != sha:
            raise ValueError('conflicting original artifact identity: '+name)
        bindings[name] = sha
    verify(launch['source_sha256']); verify_artifacts(INPUT, bindings)
    audit = read_json(INPUT, CASE+'_audit.json')
    require_raw_audit(record, audit, learned=True)
    prefix = read_json(INPUT, CASE+'_prefix_comparison.json')
    if record['prefix_comparison'] != prefix:
        raise ValueError('original terminal and physical prefix must agree')
    for key in ('physical_and_public_prefix_exact', 'shared_observed_state_exact',
            'all_preintervention_requested_commands_exact', 'complete_candidate_decisions_match_prospective_prefix',
            'complete_original_jepa_decisions_exact', 'candidate_intervention_command_completed'):
        if prefix[key] is not True:
            raise ValueError('original physical prefix required: '+key)
    if (prefix['common_prefix_frames'], prefix['physical_prefix_samples'], prefix['paired_forecast_banks'],
            prefix['first_intervention_frame']) != (4, 900, 1, 3):
        raise ValueError('exact original supervised intervention required')
    return launch, record, audit, bindings


def bounded_json(name, value):
    size = len((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
    used = sum((OUTPUT/n).stat().st_size for n in ('launch.json', 'result.json', 'failure.json')
        if (OUTPUT/n).exists())
    if used+size > ALLOWANCE:
        raise ValueError('16MiB output allowance exceeded')
    write_json(OUTPUT/name, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive new maze2 score readout required')
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+ALLOWANCE:
        raise ValueError('8GiB RAM and original40GiB reserve plus16MiB output required')
    native_launch, record, audit, bindings = admit()
    sources = discover_sources((SOURCE, PROTOCOL), native_launch['source_sha256'])
    verify(sources)
    if sources[REUSED_SOURCE] != REUSED_SHA:
        raise ValueError('reused analysis must be bound in recursive sources')
    if args.preflight_only:
        print('SUPERVISED_MAZE02_TURN_SCORES_PREFLIGHT', json.dumps(dict(
            source_count=len(sources), input_bindings=len(bindings), hardware=resources,
            output_created=False, native_execution=False, model_loaded=False)), flush=True)
        return
    create_output(OUTPUT)
    bounded_json('launch.json', dict(source_sha256=sources, input_bindings=bindings,
        hardware=resources, memory_admission_bytes=8*1024**3, output_allowance_bytes=ALLOWANCE,
        maximum_workers=1, os_resource_limits_enforced=False, original_worker_input_verification_completed=True,
        independent_transitive_input_verifier_reexecuted=False, reused_analysis_sha256=REUSED_SHA,
        native_execution=False, model_loaded=False, scientific_success_required=False))
    print('SUPERVISED_MAZE02_TURN_SCORES_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        report = summarize(read_rows(INPUT/CASE), read_json(INPUT/CASE, 'command_tape.json'))
        if (report['observations'] != 3014 or report['selections'] != 3000
                or report['selected_actions'] != audit['selected_actions']
                or report['first_terminal'] != dict(frame=3003, terminal='MISSION_TICK_BUDGET_EXHAUSTED')):
            raise ValueError('entire authenticated second-case population required')
        verify(sources); verify_artifacts(INPUT, bindings)
        bounded_json('result.json', dict(status='SUPERVISED_MAZE02_TURN_SCORES_V1_COMPLETE',
            source_sha256=sources, artifact_sha256={'launch.json':digest(OUTPUT/'launch.json')},
            input_worker_terminal_sha256=FIXED[CASE+'_worker_terminal.json'], report=report,
            original_strict_physical_visibility_pass=audit['strict_physical_visibility_pass'],
            hardware_after=hardware(), wall_s=time.perf_counter()-started,
            navigation_qualified=False, goal_achieved=False))
        print('SUPERVISED_MAZE02_TURN_SCORES_COMPLETE', digest(OUTPUT/'result.json'), json.dumps(report), flush=True)
    except Exception as error:
        bounded_json('failure.json', dict(status='TERMINAL_SUPERVISED_MAZE02_TURN_SCORES_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
