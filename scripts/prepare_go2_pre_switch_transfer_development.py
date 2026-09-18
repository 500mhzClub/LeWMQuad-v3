"""Fixed development evaluation: original transfer plus pre-switch departures."""
import hashlib
import json
import numpy as np
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.moving_action_switch_family_development import assignments, schedule
from lewm.observation_horizon_targets_development import derive
from scripts.pre_switch_training_data_development import BASE, ROOTS

OUTPUT = BASE/'go2_pre_switch_transfer_targets_v1_attempt_001'


def main():
    if OUTPUT.exists():
        raise ValueError('preserve transfer population')
    path = BASE/'go2_observation_horizon_family_targets_v1_attempt_001/windows.json'
    data = path.read_bytes(); identities = {str(path.relative_to(BASE)):hashlib.sha256(data).hexdigest()}
    old = [r for r in json.loads(data) if r['data_role'] == 'geometry_transfer']
    if len(old) != 456:
        raise ValueError('unchanged transfer slots')
    rows = []
    for row in old:
        offset = row['offset_ticks'] if row['source'] == 'family' else 0
        rows.append(row | dict(known_commands=candidate_commands(row['action'])[offset:offset+8],
                               evaluation_population='original'))
    original_switch = {r['trial']:r for r in old if r['source'] == 'switch'}
    cells = {k:v for k,v in assignments().items() if v['data_role'] == 'geometry_transfer'}
    if set(cells) != set(original_switch) or len(cells) != 72:
        raise ValueError('unchanged 72 transfer switch recordings')
    for trial, cell in sorted(cells.items()):
        root = ROOTS['switch']/trial
        p = root/'physics_trace.npz'
        identities[str(p.relative_to(BASE))] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p, allow_pickle=False) as archive:
            raw = {k:archive[k] for k in ('timestamp_s', 'base_pose_world', 'physics_contact', 'requested_command')}
        p = root/'camera_audit.json'; raw_json = p.read_bytes()
        identities[str(p.relative_to(BASE))] = hashlib.sha256(raw_json).hexdigest()
        cameras = json.loads(raw_json)
        commands = [r['requested_command'] for r in schedule(trial)]
        shared = derive(raw, cameras, frame=13, commands=commands[13:21])
        if shared['targets'] != original_switch[trial]['targets']:
            raise ValueError('shared transfer targets changed')
        for frame in range(6, 13):
            known = commands[frame:frame+8]
            labels = derive(raw, cameras, frame=frame, commands=known)
            rows.append(dict(sample_id=f'pre_switch_transfer/{trial}/frame_{frame:02d}',
                source='switch', trial=trial, data_role='geometry_transfer', cluster=cell['cluster'],
                prefix_action=cell['prefix_action'], action=cell['suffix_action'],
                observation_frame=frame, ticks_until_switch=13-frame,
                decision_ns=labels['departure_ns'], history_observation_indices=labels['history_observation_indices'],
                known_commands=known, available=labels['available'], reason=labels['reason'],
                targets=labels['targets'],
                observation_horizon_receipt={k:v for k,v in labels.items() if k != 'targets'},
                native_labels_are_target_only=True, evaluation_population='pre_switch'))
    train_clusters={'cluster_00','cluster_01'}
    if any(r['cluster'] in train_clusters for r in rows):
        raise ValueError('training geometry in transfer population')
    OUTPUT.mkdir()
    (OUTPUT/'windows.json').write_text(json.dumps(rows, indent=2)+'\n')
    report = dict(status='COMPLETE', slots=len(rows), available=sum(r['available'] for r in rows),
        original_available=sum(r['available'] and r['evaluation_population']=='original' for r in rows),
        pre_switch_available=sum(r['available'] and r['evaluation_population']=='pre_switch' for r in rows),
        parameter_clusters=sorted({r['cluster'] for r in rows}), independent_maze_trials=0,
        development_only=True, inputs_are_causal=True, future_rgb_materialized=False,
        native_state_is_target_only=True, input_sha256=identities,
        windows_sha256=hashlib.sha256((OUTPUT/'windows.json').read_bytes()).hexdigest())
    (OUTPUT/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}))


if __name__ == '__main__':
    main()
