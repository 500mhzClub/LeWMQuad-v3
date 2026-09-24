"""Expand existing training recordings; native values remain target-only."""
from copy import deepcopy
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.observation_horizon_targets_development import derive


def expand_trial(original, raw, cameras):
    if not isinstance(original, list) or not original:
        raise ValueError('nonempty original training trial required')
    first = original[0]
    source = first['source']
    metadata = ('source', 'trial', 'action', 'data_role', 'geometry', 'cluster',
        'opening', 'appearance_seed')
    identity = {k:first[k] for k in metadata if k in first}
    if (source not in ('family', 'switch') or first['data_role'] != 'train'
            or any({k:r[k] for k in metadata if k in r} != identity for r in original)):
        raise ValueError('one homogeneous original training role/trial/action required')
    old = {}
    for row in original:
        offset = row['offset_ticks'] if source == 'family' else 0
        if type(offset) is not int or offset in old:
            raise ValueError('distinct original context offsets required')
        old[offset] = row
    expected = set(range(0, 40, 5)) if source == 'family' else {0}
    if set(old) != expected:
        raise ValueError('complete original trial context population required')
    commands = candidate_commands(first['action'])
    start = 3 if source == 'family' else 13
    rows = []
    for offset in range(40):
        frame = start+offset
        known = commands[offset:offset+8]
        labels = derive(raw, cameras, frame=frame, commands=known)
        original_id = None
        if offset in old:
            previous = old[offset]
            if (labels['available'] != previous['available']
                    or labels['targets'] != previous['targets']):
                raise ValueError('every original context and native target must remain exact')
            original_id = previous['sample_id']
        rows.append(deepcopy(identity) | dict(
            sample_id=f"all_phase_train/{source}/{first['trial']}/offset_{offset:02d}",
            original_sample_id=original_id, offset_ticks=offset, remaining_ticks=40-offset,
            decision_ns=labels['departure_ns'],
            history_observation_indices=labels['history_observation_indices'],
            control_phase_modulo_five=frame % 5, known_commands=deepcopy(known),
            available=labels['available'], reason=labels['reason'], targets=labels['targets'],
            observation_horizon_receipt={k:v for k,v in labels.items() if k != 'targets'},
            native_labels_are_target_only=True, future_rgb_materialized=False,
            source_role_changed=False))
    return rows
