"""Small in-memory adapter for existing and recovered training-only windows."""
from collections import defaultdict
import hashlib
import json
import torch
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.observation_horizon_sample_development import materialize_training
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.read_go2_training_execution_coverage_development import BASE

ROOTS = dict(family=BASE/'go2_geometry_progress_family_v1_attempt_001',
             switch=BASE/'go2_moving_action_switch_family_v1_attempt_001')


class PacketReader:
    def __init__(self, packets, allowed):
        self.packets = packets; self.allowed = frozenset(allowed); self.requests = []

    def packet(self, index):
        if index not in self.allowed:
            raise ValueError('packet outside causal-input or private-target scope')
        self.requests.append(index)
        return (self.packets[index],)


def inputs(row, reader):
    past = row['history_observation_indices']; now = row['decision_ns']
    frame = row['observation_horizon_receipt']['departure_tick']
    if past != list(range(frame-3, frame+1)):
        raise ValueError('exact four-frame causal history required')
    history = causal_history_tensors([reader.packet(i)[0] for i in past], now)
    commands = row['known_commands']; n = len(commands)
    blocks = torch.zeros((8, 1, 3), dtype=torch.float32)
    valid = torch.zeros((8, 1), dtype=torch.bool)
    blocks[:n, 0] = torch.tensor(commands, dtype=torch.float32)/torch.tensor([.3, 1., .5])
    valid[:n] = True
    return dict(observation_history=history, known_action_blocks=blocks, known_action_valid=valid)


def prepare(rows):
    """Load each recorded packet once per trial; preserve separate reader scopes."""
    groups = defaultdict(list)
    for row in rows:
        if not row['available'] or row['data_role'] != 'train':
            raise ValueError('available training rows only')
        groups[row['source'], row['trial']].append(row)
    cache = {}; identities = {}
    for (source, trial), selected in sorted(groups.items()):
        directory = ROOTS[source]/trial
        past = {i for r in selected for i in r['history_observation_indices']}
        future = {t['future_observation_index'] for r in selected for t in r['targets'] if t['future_image_valid']}
        for name in ['policy_observations.json', 'policy_histories.npz'] + [f'rgb_{i:04d}.png' for i in sorted(past | future)]:
            path = directory/name
            identities[str(path.relative_to(BASE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        packets = {i:load_route_observation(directory, i) for i in sorted(past | future)}
        for row in selected:
            causal = PacketReader(packets, row['history_observation_indices'])
            inp = inputs(row, causal)
            future_indices = [t['future_observation_index'] for t in row['targets'] if t['future_image_valid']]
            target_reader = PacketReader(packets, future_indices)
            sample = materialize_training(target_reader, row, inp)
            if causal.requests != row['history_observation_indices'] or target_reader.requests != future_indices:
                raise ValueError('exact separate input and target requests required')
            cache[row['sample_id']] = sample
        print('TRAINING_CONTEXTS_READY', len(cache), flush=True)
    return cache, identities


def load_training_rows():
    old = json.loads((BASE/'go2_all_phase_training_targets_v1_attempt_001/windows.json').read_text())
    new = json.loads((BASE/'go2_pre_switch_training_targets_v1_attempt_001/windows.json').read_text())
    return [r for r in old+new if r['available']]


def batch(cache, identifiers):
    return stack_samples([cache[i] for i in identifiers])
