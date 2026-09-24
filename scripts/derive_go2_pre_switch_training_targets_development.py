"""Recover prospective switches from existing training recordings only.

Frames 6..12 precede the recorded switch at frame 13 by 700..100 ms.
No new physics, sensor replay, model fitting or geometry-transfer data access.
Native poses are read exclusively to create separately stored target labels.
"""
from collections import Counter
import hashlib
import json
import numpy as np

from lewm.moving_action_switch_family_development import assignments, schedule
from lewm.observation_horizon_targets_development import derive
from scripts.read_go2_training_execution_coverage_development import (
    BASE, COMMANDS, ZERO, tape, transitions)

OUTPUT = BASE/'go2_pre_switch_training_targets_v1_attempt_001'
SOURCE = BASE/'go2_moving_action_switch_family_v1_attempt_001'


def main():
    if OUTPUT.exists():
        raise ValueError('preserve previous target derivation')
    identities = {}

    def read(path):
        data = path.read_bytes()
        identities[str(path.relative_to(BASE))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    original = read(BASE/'go2_all_phase_training_targets_v1_attempt_001/windows.json')
    groups = {}
    for row in original:
        if row['source'] == 'switch' and row['offset_ticks'] == 0:
            if row['data_role'] != 'train':
                raise ValueError('training role only')
            groups[row['trial']] = row
    cells = {k:v for k,v in assignments().items() if v['data_role'] == 'train'}
    if len(groups) != 72 or set(groups) != set(cells):
        raise ValueError('same 72 training recordings required')
    rows = []
    for trial, old in sorted(groups.items()):
        cell = cells[trial]
        if old['action'] != cell['suffix_action'] or old['cluster'] != cell['cluster']:
            raise ValueError('unchanged training assignment')
        raw_path = SOURCE/trial/'physics_trace.npz'
        identities[str(raw_path.relative_to(BASE))] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        with np.load(raw_path, allow_pickle=False) as archive:
            raw = {k:archive[k] for k in ('timestamp_s', 'base_pose_world', 'physics_contact', 'requested_command')}
        cameras = read(SOURCE/trial/'camera_audit.json')
        policy = read(SOURCE/trial/'policy_observations.json')
        commands = [r['requested_command'] for r in schedule(trial)]
        check = derive(raw, cameras, frame=13, commands=commands[13:21])
        if check['targets'] != old['targets'] or check['available'] != old['available']:
            raise ValueError('shared original branch targets changed')
        for frame in range(6, 13):
            known = commands[frame:frame+8]
            labels = derive(raw, cameras, frame=frame, commands=known)
            if labels['available']:
                indices = labels['history_observation_indices'] + [t['future_observation_index']
                    for t in labels['targets'] if t['future_image_valid']]
                for index in indices:
                    item = policy['frames'][index]
                    now = 1_500_000_000+100_000_000*index
                    if item['decision_ns'] != now or item['image_ns'] != now:
                        raise ValueError('actual past/future packet boundary mismatch')
            rows.append(dict(sample_id=f'pre_switch_train/{trial}/frame_{frame:02d}',
                source='switch', trial=trial, data_role='train', cluster=cell['cluster'],
                prefix_action=cell['prefix_action'], action=cell['suffix_action'],
                observation_frame=frame, ticks_until_switch=13-frame,
                decision_ns=labels['departure_ns'],
                history_observation_indices=labels['history_observation_indices'],
                known_commands=known, available=labels['available'], reason=labels['reason'],
                targets=labels['targets'],
                observation_horizon_receipt={k:v for k,v in labels.items() if k != 'targets'},
                native_labels_are_target_only=True))
    old_available = [r for r in original if r['available']]
    new_available = [r for r in rows if r['available']]
    support = lambda data, h: {tape(r['known_commands'][:h]) for r in data
                              if r['targets'][h-1]['motion_valid']}
    old_support = {h:support(old_available, h) for h in (7, 8)}
    union_support = {h:old_support[h] | support(new_available, h) for h in (7, 8)}
    coverage = {k:Counter() for k in ('all_candidates', 'selected', 'executed')}
    cohort = read(BASE/'go2_neural_rgb_transfer_complete_comparison_v1_attempt_001/result.json')
    for run in cohort['rows']:
        root = BASE/run['root_name']
        executed = read(root/'saved_executed_motion_forecast_evaluation_v1.json')
        frames = {r['frame'] for r in executed['rows']}
        for plan in read(root/'planning.json'):
            if 'selection' not in plan:
                continue
            pulse = plan['motion_correction']['terminal_translation_pulse']
            for action, command in COMMANDS.items():
                ticks = 1 if pulse and action in ('forward', 'left_arc', 'right_arc') else 4
                commands = tape(plan['committed_prefix']) + (command,)*ticks + (ZERO,)*(5-ticks)
                categories = ['all_candidates']
                if action == plan['selection']['action']:
                    categories.append('selected')
                    if plan['frame'] in frames:
                        categories.append('executed')
                for category in categories:
                    c = coverage[category]; c['windows'] += 1
                    for h in (7, 8):
                        c[f'old_supported_{h*100}ms'] += commands[:h] in old_support[h]
                        c[f'augmented_supported_{h*100}ms'] += commands[:h] in union_support[h]
    result = dict(status='COMPLETE', contexts=len(rows), available=len(new_available),
        original_training_recordings=len(groups), additional_independent_recordings=0,
        first_motion_valid=sum(r['targets'][0]['motion_valid'] for r in new_available),
        eight_motion_valid=sum(r['targets'][7]['motion_valid'] for r in new_available),
        future_transition_count=dict(Counter(len(transitions(tape(r['known_commands']))) for r in new_available)),
        distinct_motion_labeled_tapes={str(h*100):dict(original=len(old_support[h]),
            augmented=len(union_support[h])) for h in (7, 8)},
        exposed_pilot_command_coverage={k:dict(v) for k,v in coverage.items()},
        input_sha256=identities, new_simulation=False, model_training=False,
        original_targets_unchanged=True, geometry_transfer_read=False,
        native_values_are_targets_only=True, new_short_pulse_execution_collected=False,
        prediction_improvement_established=False, navigation_improvement_established=False)
    OUTPUT.mkdir()
    (OUTPUT/'windows.json').write_text(json.dumps(rows, indent=2)+'\n')
    result['windows_sha256'] = hashlib.sha256((OUTPUT/'windows.json').read_bytes()).hexdigest()
    (OUTPUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
