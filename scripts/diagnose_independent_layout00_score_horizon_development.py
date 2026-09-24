"""Recorded score-term sensitivity, not a counterfactual navigation rollout."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
CASE = ROOT/'go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_jepa_v1_attempt_001'/'independent_00_frozen_reference_seed_2026091001_full_jepa'
OUTPUT = Path('docs/go2_independent_layout00_score_horizon_sensitivity_2026-09-13.json')
START, END = 200, 900
TURNS = {'left_turn', 'right_turn'}
TRANSLATIONS = {'forward', 'left_arc', 'right_arc'}


def main():
    counts = Counter(); actions = Counter(); alternatives = Counter(); transitions = Counter()
    frontier = Counter(); skipped = Counter(); examples = []; positions = []; endpoints = []
    digest = hashlib.sha256(); last_target = None; target_changes = 0; last_frame = -1
    with gzip.open(CASE/'context_decisions.jsonl.gz', 'rb') as stream:
        for frame, line in enumerate(stream):
            digest.update(line); last_frame = frame
            if frame < START:
                continue
            row = json.loads(line)
            if row['tick'] != frame:
                raise ValueError('ordered completed decision prefix required')
            decision = row['decision']; mission = decision['mission_receipt']
            positions.append(mission['observed_settling']['current_position_initial_body_m'][:2])
            counts['observations'] += 1
            if frame in (START, END):
                endpoints.append(dict(frame=frame, phase=mission['phase'],
                    observed_goal_distance_m=mission['observed_goal_distance_m'],
                    observed_position_xy_m=positions[-1], terminal=decision['terminal']))
            selection = decision['new_selection']
            actions[str(decision['selected_action'])] += 1
            if not selection or selection.get('executed_waypoint_scoring') is not True:
                skipped['outside_intermediate_executed_waypoint_score'] += 1
            elif (selection['scored_pose_horizon_ns'] != 100_000_000
                    or selection['scored_contact_horizon_ns'] != 800_000_000):
                skipped['different_score_horizons'] += 1
            else:
                counts['executed_waypoint_selections'] += 1
                proposal = selection.get('proposal') or {}
                frontier[str(proposal.get('status'))] += 1
                target = proposal.get('target_map_xy_m')
                if last_target is not None and target != last_target:
                    target_changes += 1
                last_target = target
                candidates = selection['candidates']; penalty = selection['contact_penalty_m']
                # Reconstruct the existing score and feasible bank from the
                # recorded fields; separate later policy overrides explicitly.
                feasible = [i for i, candidate in enumerate(candidates)
                    if candidate['action'] in selection['phase_allowed_actions']
                    and not selection['surface_checks'][i]['possible_intersection']
                    and selection['nominal_path_checks'][i]['all_predicted_segments_nominally_clear']]
                terms = [c['executed_waypoint_distance_progress_m']+
                    c['executed_waypoint_alignment_progress_m'] for c in candidates]
                original = [v-penalty*c['full_plan_contact_score'] for v,c in zip(terms,candidates)]
                if any(abs(v-c['utility_m']) > 1e-12 for v,c in zip(original,candidates)):
                    skipped['recorded_score_algebra_differs'] += 1
                elif not feasible:
                    skipped['no_candidate_in_recorded_feasible_bank'] += 1
                else:
                    chosen = max(feasible,key=lambda i:original[i])
                    action = candidates[chosen]['action']
                    if action != selection['action']:
                        skipped['final_selection_differs_from_recorded_score_maximum'] += 1
                    else:
                        counts['reconstructed_final_score_decisions'] += 1
                        short = [v-penalty*c['commitment_contact_score'] for v,c in zip(terms,candidates)]
                        other = max(feasible,key=lambda i:short[i]); alternative = candidates[other]['action']
                        alternatives[alternative] += 1; transitions[action+' -> '+alternative] += 1
                        counts['ranking_changed'] += action != alternative
                        counts['recorded_turns_in_reconstructed_population'] += action in TURNS
                        if action in TURNS and alternative in TRANSLATIONS:
                            counts['turn_to_translation_ranking_changes'] += 1
                            if len(examples) < 10:
                                examples.append(dict(frame=frame,recorded_action=action,
                                    shorter_contact_horizon_top_action=alternative,
                                    original_utilities={candidates[i]['action']:original[i] for i in feasible},
                                    shorter_contact_utilities={candidates[i]['action']:short[i] for i in feasible}))
            if frame == END:
                break
    if last_frame != END or counts['observations'] != END-START+1:
        raise ValueError('entire declared completed prefix required')
    report = dict(frames=[START,END],counts=dict(counts),recorded_selected_actions=dict(actions),
        shorter_contact_horizon_top_actions=dict(alternatives),ranking_transitions=dict(transitions),
        skipped=dict(skipped),proposal_statuses=dict(frontier),frontier_target_changes=target_changes,
        observed_xy_ranges=[[min(p[i] for p in positions),max(p[i] for p in positions)] for i in (0,1)],
        endpoints=endpoints,examples=examples,decoded_prefix_through_end_sha256=digest.hexdigest(),
        recorded_feasibility_filters_retained=True,recorded_distance_alignment_and_residuals_retained=True,
        model_inference_executed=False,controller_replay_executed=False,native_execution=False,
        live_or_queued_policy_changed=False,counterfactual_trajectory_inferred=False,
        navigation_improvement_established=False,terminal_independent_result=False)
    with OUTPUT.open('x') as output:
        json.dump(report,output,indent=2,allow_nan=False);output.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='examples'}))


if __name__ == '__main__':
    main()
