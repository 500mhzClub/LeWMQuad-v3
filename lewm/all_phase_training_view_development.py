"""Expanded training contexts with the unchanged original transfer population.

This is a study index, not a training schedule or a new target derivation.
The caller authenticates the completed input/target artifacts before use.
"""
from collections import defaultdict
from copy import deepcopy
from lewm.observation_horizon_view_development import ObservationHorizonView


class AllPhaseTrainingView:
    training_slots = 4800

    def __init__(self, original, training_rows):
        if (not isinstance(original, ObservationHorizonView)
                or not isinstance(training_rows, list) or len(training_rows) != 4800):
            raise ValueError('complete original view and 4800 expanded training slots required')
        old_trials = defaultdict(dict)
        for row in original.rows:
            if row['data_role'] == 'train':
                offset = row['offset_ticks'] if row['source'] == 'family' else 0
                old_trials[row['source'], row['trial']][offset] = row
        if ({s: sum(key[0] == s for key in old_trials) for s in ('family', 'switch')}
                != dict(family=48, switch=72)):
            raise ValueError('unchanged 48 family and 72 switch training trials required')
        seen = set(); shared = set(); rows = []
        for row in training_rows:
            source, trial, offset = row['source'], row['trial'], row['offset_ticks']
            key = (source, trial)
            if (key not in old_trials or row['data_role'] != 'train'
                    or type(offset) is not int or not 0 <= offset < 40
                    or (key, offset) in seen or type(row['available']) is not bool):
                raise ValueError('distinct original training trial and every bounded offset required')
            seen.add((key, offset)); old = old_trials[key]; first = old[0]
            identity = ('action', 'geometry', 'cluster', 'opening', 'appearance_seed')
            if any((k in row) != (k in first) or row.get(k) != first.get(k) for k in identity):
                raise ValueError('expanded training context changed original trial identity')
            frame = (3 if source == 'family' else 13) + offset
            now = 1_500_000_000 + 100_000_000 * frame
            history = list(range(frame - 3, frame + 1))
            if (row['sample_id'] != f'all_phase_train/{source}/{trial}/offset_{offset:02d}'
                    or row['remaining_ticks'] != 40-offset or row['decision_ns'] != now
                    or row['history_observation_indices'] != history
                    or row['control_phase_modulo_five'] != frame % 5
                    or row['native_labels_are_target_only'] is not True
                    or row['source_role_changed'] is not False):
                raise ValueError('exact expanded identity, clock and target-only role required')
            receipt = dict(target_only=True, departure_tick=frame, departure_ns=now,
                history_observation_indices=history, target_cadence_ns=100_000_000,
                maximum_horizon_ns=800_000_000, available=row['available'], reason=row['reason'])
            if row['observation_horizon_receipt'] != receipt:
                raise ValueError('unchanged derivation receipt required')
            if offset in old:
                previous = old[offset]
                if (row['original_sample_id'] != previous['sample_id']
                        or row['available'] != previous['available']
                        or row['targets'] != previous['targets']):
                    raise ValueError('every shared original target and availability must remain exact')
                shared.add(previous['sample_id'])
            elif row['original_sample_id'] is not None:
                raise ValueError('new offset cannot claim an original input witness')
            if row['available']:
                if row['reason'] is not None or len(row['targets']) != 8:
                    raise ValueError('eight available target slots and no exclusion reason required')
                for horizon, target in enumerate(row['targets'], 1):
                    active = horizon <= min(8, 40-offset)
                    if (target['in_plan'] is not active
                            or target['offset_ns'] != (horizon*100_000_000 if active else 0)):
                        raise ValueError('exact expanded target clocks required')
            elif row['targets'] is not None:
                raise ValueError('unavailable context cannot acquire targets')
            stratum = ('initial' if offset == 0 else 'moving') if source == 'family' else first['stratum']
            metadata = dict(stratum=stratum)
            if source == 'switch': metadata['prefix_action'] = first['prefix_action']
            if any(k in row and row[k] != v for k, v in metadata.items()):
                raise ValueError('scoring metadata must come from original trial assignment')
            rows.append(deepcopy(row) | metadata)
        if len(seen) != 4800 or len(shared) != 456:
            raise ValueError('all original training witnesses and all expanded slots required')
        self.transfer_original_indices = tuple(i for i, row in enumerate(original.rows)
            if row['data_role'] == 'geometry_transfer')
        if len(self.transfer_original_indices) != 456:
            raise ValueError('all 456 original transfer slots required')
        rows.extend(deepcopy(original.rows[i]) for i in self.transfer_original_indices)
        if len({r['sample_id'] for r in rows}) != 5256:
            raise ValueError('globally distinct training and transfer identities required')
        self.rows = rows

    def indices(self, role, *, source=None):
        if role not in ('train', 'geometry_transfer') or source not in (None, 'family', 'switch'):
            raise ValueError('explicit study source and role required')
        sources = (source,) if source is not None else ('family', 'switch')
        return [i for s in sources for i, row in enumerate(self.rows)
            if row['source'] == s and row['data_role'] == role and row['available']]

    def transfer_index(self, index):
        if type(index) is not int or not 4800 <= index < 5256:
            raise ValueError('explicit combined transfer index required')
        return self.transfer_original_indices[index-4800]
