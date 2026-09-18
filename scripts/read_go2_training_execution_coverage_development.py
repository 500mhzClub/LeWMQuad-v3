"""Describe training/action support against the completed, exposed RGB pilot.

Command equality is rounded to six decimals to remove float32 serialization
noise. Exact tape support is a diagnostic, not a requirement for generalization.
Only saved training metadata, plans and completed evaluation rows are read.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
OUTPUT = BASE/'go2_training_execution_coverage_v1_attempt_001'
COMMANDS = dict(hold=(0., 0., 0.), forward=(.2, 0., 0.),
                left_arc=(.16, 0., .45), right_arc=(.16, 0., -.45),
                left_turn=(0., 0., .45), right_turn=(0., 0., -.45))
ZERO = COMMANDS['hold']


def tape(commands):
    return tuple(tuple(round(float(v), 6) for v in c) for c in commands)


def transitions(commands):
    return [(a, b) for a, b in zip(commands, commands[1:]) if a != b]


def short_internal_pulse(commands):
    # Both onset and offset must be visible; a long action's final tick is not
    # evidence that training included a one-tick movement from rest.
    return any(commands[i-1] == ZERO and commands[i] != ZERO and commands[i+1] == ZERO
               for i in range(1, len(commands)-1))


def main():
    if OUTPUT.exists():
        raise ValueError('preserve existing coverage result')
    identities = {}

    def read(path):
        if any(p == 'sealed' or p.startswith('sealed_') for p in path.parts):
            raise ValueError('protected path')
        data = path.read_bytes()
        identities[str(path.relative_to(BASE))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    rows = read(BASE/'go2_all_phase_training_targets_v1_attempt_001/windows.json')
    available = [r for r in rows if r['available']]
    if len(rows) != 4800 or len(available) != 4010 or any(r['data_role'] != 'train' for r in rows):
        raise ValueError('expected unchanged training population')
    known_support = {}; motion_support = {}
    for h in (7, 8):
        known_support[h] = {tape(r['known_commands'][:h]) for r in available if len(r['known_commands']) >= h}
        motion_support[h] = {tape(r['known_commands'][:h]) for r in available
                             if r['targets'][h-1]['motion_valid']}
    training_tapes = [tape(r['known_commands']) for r in available]
    transition_support = {pair for t in training_tapes for pair in transitions(t)}
    training = dict(slots=len(rows), available=len(available),
        independent_recordings=len({(r['source'], r['trial']) for r in rows}),
        source_available=dict(Counter(r['source'] for r in available)),
        contexts_by_future_transition_count=dict(Counter(len(transitions(t)) for t in training_tapes)),
        visible_zero_nonzero_zero_single_tick_pulses=sum(map(short_internal_pulse, training_tapes)),
        future_transition_pairs=[list(pair) for pair in sorted(transition_support)],
        horizons={str(h*100): dict(known_contexts=sum(len(r['known_commands']) >= h for r in available),
            motion_valid_contexts=sum(r['targets'][h-1]['motion_valid'] for r in available),
            distinct_known_tapes=len(known_support[h]), distinct_motion_labeled_tapes=len(motion_support[h]))
            for h in (7, 8)})

    summaries = defaultdict(Counter); by_run = []; errors = defaultdict(list)

    def count(category, commands):
        c = summaries[category]; c['windows'] += 1
        pairs = transitions(commands)
        c['two_or_more_future_transitions'] += len(pairs) >= 2
        c['unseen_future_transition_pair'] += any(p not in transition_support for p in pairs)
        c['visible_single_tick_pulse'] += short_internal_pulse(commands)
        for h in (7, 8):
            c[f'outside_known_tape_support_{h*100}ms'] += commands[:h] not in known_support[h]
            c[f'outside_motion_labeled_tape_support_{h*100}ms'] += commands[:h] not in motion_support[h]

    cohort = read(BASE/'go2_neural_rgb_transfer_complete_comparison_v1_attempt_001/result.json')
    for assignment in cohort['rows']:
        root = BASE/assignment['root_name']
        plans = [r for r in read(root/'planning.json') if 'selection' in r]
        executed = read(root/'saved_executed_motion_forecast_evaluation_v1.json')
        if executed['matched_requested_sequence_through_ns'] != 700_000_000:
            raise ValueError('expected matched 700-ms execution')
        by_frame = {r['frame']: r for r in executed['rows']}
        local = Counter()
        for plan in plans:
            pulse = plan['motion_correction']['terminal_translation_pulse']
            prefix = tape(plan['committed_prefix'])
            if len(prefix) != 3:
                raise ValueError('three committed prefix ticks required')
            for action, command in COMMANDS.items():
                ticks = 1 if pulse and action in ('forward', 'left_arc', 'right_arc') else 4
                commands = prefix + (command,)*ticks + (ZERO,)*(5-ticks)
                count('all_scored_candidates', commands)
                if action != plan['selection']['action']:
                    continue
                count('selected_plans', commands)
                local['selected_plans'] += 1
                local['selected_outside_motion_support_700ms'] += commands[:7] not in motion_support[7]
                if plan['frame'] not in by_frame:
                    continue
                window = by_frame[plan['frame']]
                if window['action'] != action:
                    raise ValueError('executed action mismatch')
                count('matched_executed_windows', commands)
                count('executed_'+window['group'], commands)
                supported = commands[:7] in motion_support[7]
                local['matched_executed_windows'] += 1
                local['matched_outside_motion_support_700ms'] += not supported
                correction = plan['motion_correction']; index = plan['selection']['action_index']
                actual = window['actual_endpoint_xy_m']
                error = {}
                for name, key in dict(raw='raw_forecast_xy_m', corrected='learned_corrected_forecast_xy_m',
                                      fitted='pose_command_forecast_xy_m').items():
                    pred = correction[key][index][6]
                    error[name] = sum((a-b)**2 for a, b in zip(pred, actual))
                errors['supported' if supported else 'unsupported'].append(error)
        by_run.append({k:assignment[k] for k in ('root_name', 'method', 'seed', 'condition', 'layout_index')}
                      | dict(local))
    result = dict(training=training, online={k:dict(v) for k,v in summaries.items()},
        executed_700ms_errors_by_exact_motion_tape_support={k:dict(windows=len(v),
            **{name+'_rmse_mm':1000*(sum(x[name] for x in v)/len(v))**.5
               for name in ('raw', 'corrected', 'fitted')}) for k,v in errors.items()},
        runs=by_run, input_sha256=identities,
        scope=dict(command_rounding_decimals=6, explicit_registered_pose_in_neural_inputs=False,
            neural_inputs='four RGB/body/control observations and eight known command ticks',
            fitted_control_inputs='four registered visual poses and same command sequence',
            exact_support_is_not_a_generalization_requirement=True,
            support_groups_have_different_state_and_action_distributions=True,
            error_groups_are_not_causal_effect_estimates=True,
            scored_candidates_are_not_executed_counterfactual_outcomes=True,
            independent_online_maze_units=2, overlapping_windows=True,
            native_simulation_launched=False, model_trained=False))
    OUTPUT.mkdir()
    (OUTPUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
