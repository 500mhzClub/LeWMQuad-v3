"""All saved supervised maze1 selections: exact scores and translation vetoes."""
from collections import Counter
from copy import deepcopy
import json
import time

from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.independent_reactive_floor_transport_study_development import require_raw_audit
from lewm.supervised_rollout_maze_study_development import SUPERVISED_STATE
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.maze_decision_stream_development import read_rows
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

INPUT = BASE/'go2_supervised_rollout_mazes_v1_attempt_001'
OUTPUT = BASE/'go2_supervised_maze01_turn_scores_v1_attempt_001'
CASE = 'full_supervised_rollout_novel_maze_01'
SOURCE = 'scripts/read_go2_supervised_maze01_turn_scores_v1.py'
PROTOCOL = 'docs/go2_supervised_maze01_turn_scores_v1_2026-09-09.md'
FIXED = {
    'launch.json': '49a182da3d795f1b31910fb2b6123732db66349dfc011d768bb72fc9732a09e3',
    CASE+'_worker_terminal.json': '730b2d8a20d680066854427e6a660c58346d296f6073ee12ec3ae984574380e5',
    'cohort_progress_after_01.json': 'd55cafb5cab562a258a19e12840385653e3cbee3d298f0ddc1e0f5758aa1a30e',
}
TRANSLATION = ('forward', 'left_arc', 'right_arc')


def admit():
    verify_artifacts(INPUT, FIXED)
    launch = read_json(INPUT, 'launch.json')
    record = read_json(INPUT, CASE+'_worker_terminal.json')
    progress = read_json(INPUT, 'cohort_progress_after_01.json')
    if (record['status'] != 'SUPERVISED_ROLLOUT_COLLECTED_AND_RAW_AUDITED' or 'failure' in record
            or record['case'] != CASE or record['layout_index'] != 1
            or record['model_state_sha256'] != SUPERVISED_STATE or record['model_state_unchanged'] is not True
            or record['condition'] != 'supervised_rollout' or record['variant'] != 'full'
            or record['head'] != 'rollout_outcomes' or progress['completed_conditions'] != [record]
            or progress['remaining_layouts'] != [2, 3] or progress['original_case_order'] != [1, 2, 3]):
        raise ValueError('exact completed first case of the unchanged original cohort required')
    bindings = {**record['artifact_sha256'], **FIXED,
        CASE+'_worker.log': record['worker_log_sha256']}
    verify(launch['source_sha256']); verify_artifacts(INPUT, bindings)
    audit = read_json(INPUT, CASE+'_audit.json')
    require_raw_audit(record, audit, learned=True)
    prefix = read_json(INPUT, CASE+'_prefix_comparison.json')
    if record['prefix_comparison'] != prefix:
        raise ValueError('original terminal and physical-prefix report must agree')
    for key in ('physical_and_public_prefix_exact', 'shared_observed_state_exact',
            'all_preintervention_requested_commands_exact', 'complete_candidate_decisions_match_prospective_prefix',
            'complete_original_jepa_decisions_exact', 'candidate_intervention_command_completed'):
        if prefix[key] is not True: raise ValueError('original physical prefix required: '+key)
    if (prefix['common_prefix_frames'], prefix['physical_prefix_samples'], prefix['paired_forecast_banks'],
            prefix['first_intervention_frame']) != (4, 900, 1, 3):
        raise ValueError('exact original supervised intervention required')
    return launch, record, audit, bindings


def reconstruct(selection):
    if (selection.get('score_contract') != 'causal_executed_waypoint_potential_minus_full_plan_contact'
            or selection.get('mode') != 'WAYPOINT' or selection.get('nominal_clearance_reentry', False)):
        raise ValueError('readout requires the original executed-waypoint scoring contract for every selection')
    prior = deepcopy(selection)
    prior.update(scored_horizon_ns=800_000_000,
        candidates=deepcopy(selection['original_waypoint_candidates']),
        action=selection['original_waypoint_action'], score_contract=selection['original_waypoint_score_contract'])
    rebuilt = score_waypoint_execution(prior, selection['causal_score_residual_receipt'])
    if rebuilt != selection:
        raise ValueError('complete original score output must reconstruct exactly')


def summarize(rows, tape):
    observations = selections = turns = switches = previous_frame = 0
    previous_action = first_terminal = None
    actions = Counter(); reasons = Counter(); vetoes = Counter(); first = {}
    per_action = {a:dict(count=0, feasible=0, utility_sum_m=0., distance_sum_m=0.,
        alignment_sum_m=0., contact_sum=0., utility_margin_sum_m=0.,
        more_geometric_progress_but_lower_utility=0) for a in ACTIONS}
    for row in rows:
        frame = observations; observations += 1; d = row['decision']; s = d['new_selection']
        if row['tick'] != frame or row['observation_index'] != frame or row['pre_sample_index'] != 749+50*frame:
            raise ValueError('complete ordered original observations required')
        if frame < len(tape) and (tape[frame]['completed'] is not True
                or tape[frame]['requested_command'] != d['requested_command']):
            raise ValueError('exact completed original requests required')
        if first_terminal is None and d['terminal'] is not None:
            first_terminal = dict(frame=frame, terminal=d['terminal'])
        if s is None: continue
        if d['terminal'] is not None or s['requested_command'] != d['requested_command']:
            raise ValueError('only actual nonterminal selections, without population filtering')
        reconstruct(s); selections += 1; actions[s['action']] += 1
        chosen = s['candidates'][s['action_index']]
        chosen_geometry = chosen['executed_waypoint_distance_progress_m']+chosen['executed_waypoint_alignment_progress_m']
        alternatives = []; feasible_translation = []; reversals = []
        for i, candidate in enumerate(s['candidates']):
            action = candidate['action']; path = s['nominal_path_checks'][i]
            segments = path['segments']
            if len(segments) != 8 or path['action'] != action:
                raise ValueError('eight original ordered segments for every action required')
            blocked = [h for h, segment in enumerate(segments) if not segment['nominal_disk_connector_clear']]
            if path['all_predicted_segments_nominally_clear'] is not (not blocked):
                raise ValueError('saved path summary must equal its original segment checks')
            surface = s['surface_checks'][i]['possible_intersection']
            phase = action in s['phase_allowed_actions']
            feasible = phase and not surface and not blocked
            geometry = candidate['executed_waypoint_distance_progress_m']+candidate['executed_waypoint_alignment_progress_m']
            reversal = bool(feasible and geometry > chosen_geometry and candidate['utility_m'] < chosen['utility_m'])
            stats = per_action[action]; stats['count'] += 1; stats['feasible'] += int(feasible)
            stats['utility_sum_m'] += candidate['utility_m']
            stats['distance_sum_m'] += candidate['executed_waypoint_distance_progress_m']
            stats['alignment_sum_m'] += candidate['executed_waypoint_alignment_progress_m']
            stats['contact_sum'] += candidate['full_plan_contact_score']
            stats['utility_margin_sum_m'] += candidate['utility_m']-chosen['utility_m']
            stats['more_geometric_progress_but_lower_utility'] += int(reversal)
            if action in TRANSLATION:
                if not phase: vetoes[action+':phase'] += 1
                if surface: vetoes[action+':surface'] += 1
                for h in blocked: vetoes[action+':segment_'+str(h)] += 1
                if feasible: feasible_translation.append(action)
                if reversal: reversals.append(action)
                alternatives.append(dict(action=action, feasible=bool(feasible), surface_veto=surface,
                    phase_allowed=phase, blocked_segment_indices=blocked, utility_m=candidate['utility_m'],
                    utility_margin_to_selected_m=candidate['utility_m']-chosen['utility_m'],
                    geometric_potential_progress_m=geometry, full_plan_contact_score=candidate['full_plan_contact_score']))
        if s['action'] in ('left_turn', 'right_turn'):
            turns += 1
            if (frame == previous_frame+1 and previous_action in ('left_turn', 'right_turn')
                    and previous_action != s['action']): switches += 1
            reason = ('no_feasible_translation' if not feasible_translation else
                'feasible_translation_with_more_geometric_progress_loses_after_contact_penalty' if reversals else
                'turn_wins_with_no_feasible_translation_having_more_geometric_progress')
            reasons[reason] += 1
            first.setdefault(reason, dict(frame=frame, selected=s['action'], selected_utility_m=chosen['utility_m'],
                selected_geometric_potential_progress_m=chosen_geometry,
                selected_full_plan_contact_score=chosen['full_plan_contact_score'],
                goal_body_xy_m=s['goal_body_xy_m'], alternatives=alternatives))
        previous_frame, previous_action = frame, s['action']
    if observations != len(tape)+1:
        raise ValueError('complete final observation and all original actual requests required')
    for stats in per_action.values():
        count = stats.pop('count')
        if count != selections or not count: raise ValueError('every candidate included at every selection')
        for key in list(stats):
            if '_sum' in key: stats[key.replace('_sum', '_mean')] = stats.pop(key)/count
    return dict(observations=observations, selections=selections, selected_actions=dict(actions),
        selected_turns=turns, consecutive_turn_direction_changes=switches, first_terminal=first_terminal,
        turn_reason_counts=dict(reasons), translation_veto_counts=dict(vetoes), per_action=per_action,
        first_reason_examples=first, complete_original_score_reconstructions=selections,
        scores_and_action_reconstructed_with_original_function=True,
        raw_surface_and_path_checks_reexecuted=False, original_controller_reexecuted=False,
        native_pose_used=False, model_loaded=False, native_execution=False,
        alternative_physical_outcomes_inferred=False, contact_scores_calibrated=False,
        score_horizons_changed=False, controller_or_model_changed=False)


def main():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive new score readout required')
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+16*1024**2:
        raise ValueError('8GiB readout RAM and original reserve plus16MiB output allowance required')
    native_launch, record, audit, bindings = admit()
    sources = discover_sources((SOURCE, PROTOCOL), native_launch['source_sha256']); verify(sources)
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_bindings=bindings,
        hardware=resources, memory_admission_bytes=8*1024**3, output_allowance_bytes=16*1024**2,
        original_worker_input_verification_completed=True, independent_transitive_input_verifier_reexecuted=False,
        native_execution=False, model_loaded=False, scientific_success_required=False))
    print('SUPERVISED_TURN_SCORES_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    started = time.perf_counter()
    try:
        report = summarize(read_rows(INPUT/CASE), read_json(INPUT/CASE, 'command_tape.json'))
        if (report['observations'] != 3014 or report['selections'] != 3000
                or report['selected_actions'] != audit['selected_actions']
                or report['first_terminal'] != dict(frame=3003, terminal='MISSION_TICK_BUDGET_EXHAUSTED')):
            raise ValueError('entire authenticated first-case population required')
        verify(sources); verify_artifacts(INPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SUPERVISED_MAZE01_TURN_SCORES_V1_COMPLETE',
            source_sha256=sources, artifact_sha256={'launch.json':digest(OUTPUT/'launch.json')},
            input_worker_terminal_sha256=FIXED[CASE+'_worker_terminal.json'], report=report,
            wall_s=time.perf_counter()-started, navigation_qualified=False, goal_achieved=False))
        print('SUPERVISED_TURN_SCORES_COMPLETE', digest(OUTPUT/'result.json'), json.dumps(report), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SUPERVISED_TURN_SCORES_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
