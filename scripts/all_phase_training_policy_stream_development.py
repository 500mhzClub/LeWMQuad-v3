"""Bounded training-only stream with separate causal inference and future readers.

The caller authenticates the complete target population and collection bindings
before construction and again after use. Native labels remain outside inputs.
This stream does not change the original geometry-transfer data or fit a model.
"""
from collections import OrderedDict
from copy import deepcopy
import re
import torch
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.observation_horizon_plan_development import validate_plan
from lewm.observation_horizon_sample_development import materialize_training
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.derive_go2_all_phase_training_targets_v1 import ROOTS
from scripts.navigation_artifact_root_development import verify_artifacts

MAX_CACHE_BYTES = 8*1024**3


def context(row):
    source = row['source']; offset = row['offset_ticks']
    if (source not in ROOTS or row['data_role'] != 'train' or row['available'] is not True
            or type(offset) is not int or not 0 <= offset < 40
            or row['remaining_ticks'] != 40-offset or row['source_role_changed'] is not False
            or not isinstance(row['trial'], str)
            or re.fullmatch(r'[A-Za-z0-9_]+', row['trial']) is None):
        raise ValueError('available unchanged training role and bounded trial/offset required')
    frame = (3 if source == 'family' else 13)+offset
    now = 1_500_000_000+100_000_000*frame
    past = list(range(frame-3, frame+1))
    if (row['decision_ns'] != now or row['history_observation_indices'] != past
            or row['control_phase_modulo_five'] != frame % 5
            or row['sample_id'] != f"all_phase_train/{source}/{row['trial']}/offset_{offset:02d}"):
        raise ValueError('exact causal departure, identity and history required')
    receipt = dict(target_only=True, departure_tick=frame, departure_ns=now,
        history_observation_indices=past, target_cadence_ns=100_000_000,
        maximum_horizon_ns=800_000_000, available=True, reason=None)
    if (row['observation_horizon_receipt'] != receipt
            or row['native_labels_are_target_only'] is not True):
        raise ValueError('exact target-only derivation clock and availability receipt required')
    commands = candidate_commands(row['action'])[offset:offset+8]
    if row['known_commands'] != commands:
        raise ValueError('unchanged original known command suffix required')
    return frame, now, past, commands


def plan(row):
    _,_,_,commands = context(row)
    blocks = torch.zeros((8,1,3), dtype=torch.float32)
    valid = torch.zeros((8,1), dtype=torch.bool)
    blocks[:len(commands),0] = torch.tensor(commands,dtype=torch.float32)/torch.tensor([.3,1.,.5])
    valid[:len(commands),0] = True
    validate_plan(blocks.unsqueeze(0), valid.unsqueeze(0), 1)
    return blocks, valid


def policy_leaves(row, *, include_future):
    if type(include_future) is not bool: raise ValueError('explicit reader scope required')
    frame,_,past,_ = context(row)
    future = []
    if include_future:
        for horizon,target in enumerate(row['targets'],1):
            if target['future_image_valid']:
                index = target['future_observation_index']
                if (type(index) is not int or not 1 <= horizon <= 8 or index != frame+horizon
                        or target['motion_valid'] is not True or target['in_plan'] is not True
                        or target['offset_ns'] != horizon*100_000_000):
                    raise ValueError('explicit current-trial training future boundary required')
                future.append(index)
    names = ['policy_observations.json','policy_histories.npz']+[
        f'rgb_{i:04d}.png' for i in sorted(set(past+future))]
    return names, past, future


class ScopedReader:
    def __init__(self, directory, indices):
        self.directory = directory; self.indices = frozenset(indices); self.requests = []

    def packet(self, index):
        if type(index) is not int or index not in self.indices:
            raise ValueError('packet outside explicit causal or private-training scope')
        self.requests.append(index)
        return (load_route_observation(self.directory,index),)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor): return value.numel()*value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(v) for v in value.values())
    raise ValueError('only private tensor dictionaries may enter the cache')


class AllPhaseTrainingStream:
    def __init__(self, rows, bindings, *, maximum_cache_bytes=MAX_CACHE_BYTES):
        if (type(maximum_cache_bytes) is not int or not 0 <= maximum_cache_bytes <= MAX_CACHE_BYTES
                or not isinstance(rows,list) or not rows or len(rows) > 4800
                or any(r['data_role'] != 'train' for r in rows)
                or len({r['sample_id'] for r in rows}) != len(rows)
                or set(bindings) != set(ROOTS)):
            raise ValueError('distinct training-only rows, exact source roots and bounded cache required')
        self.rows = deepcopy(rows); self.bindings = deepcopy(bindings)
        self.maximum_cache_bytes = maximum_cache_bytes
        self.cache_bytes = 0; self.maximum_observed_cache_bytes = 0
        self._cache = OrderedDict(); self.failed = False; self.last_access = None

    def indices(self): return [i for i,r in enumerate(self.rows) if r['available']]

    def _selected(self, row, *, training):
        names,past,future = policy_leaves(row, include_future=training)
        paths = [row['trial']+'/'+n for n in names]
        bindings = self.bindings[row['source']]
        if not set(paths) <= set(bindings): raise ValueError('every consumed policy leaf must be bound')
        return {n:bindings[n] for n in paths}, past, future

    def materialize(self, index, *, training):
        if self.failed: raise ValueError('stream failure latched')
        try:
            if (type(training) is not bool or type(index) is not int
                    or not 0 <= index < len(self.rows)):
                raise ValueError('explicit materialization role and bounded index required')
            row = self.rows[index]; _,now,_,_ = context(row)
            selected,past,future = self._selected(row, training=training)
            root = ROOTS[row['source']]; verify_artifacts(root,selected)
            reader = ScopedReader(root/row['trial'],past)
            history = causal_history_tensors([reader.packet(i)[0] for i in past], now)
            blocks,valid = plan(row)
            inputs = dict(observation_history=history,known_action_blocks=blocks,known_action_valid=valid)
            future_reader = ScopedReader(root/row['trial'],future)
            sample = materialize_training(future_reader,row,inputs) if training else inputs
            if reader.requests != past or future_reader.requests != future:
                raise ValueError('exact separate past and future packet populations required')
            verify_artifacts(root,selected)
            self.last_access = dict(sample_id=row['sample_id'],training=training,
                past_packet_indices=list(reader.requests),future_packet_indices=list(future_reader.requests),
                consumed_policy_leaves=sorted(selected),native_artifacts_opened=False)
            return sample
        except Exception:
            self.failed = True; raise

    def _batch(self, indices, *, training):
        if self.failed: raise ValueError('stream failure latched')
        try:
            if (not isinstance(indices,list) or not 1 <= len(indices) <= 16
                    or any(type(i) is not int or not 0 <= i < len(self.rows)
                        or self.rows[i]['available'] is not True for i in indices)):
                raise ValueError('bounded available training indices required')
            samples = []
            for index in indices:
                if training and index in self._cache:
                    selected,_,_ = self._selected(self.rows[index],training=True)
                    verify_artifacts(ROOTS[self.rows[index]['source']],selected)
                    sample,_ = self._cache[index]; self._cache.move_to_end(index)
                else:
                    sample = self.materialize(index,training=training)
                    size = tensor_bytes(sample)
                    if training and size <= self.maximum_cache_bytes:
                        while self._cache and self.cache_bytes+size > self.maximum_cache_bytes:
                            _,(_,removed) = self._cache.popitem(last=False); self.cache_bytes -= removed
                        self._cache[index] = (sample,size); self.cache_bytes += size
                        self.maximum_observed_cache_bytes = max(self.maximum_observed_cache_bytes,self.cache_bytes)
                samples.append(sample)
            # Fresh stacks keep callers from mutating private cached samples.
            return stack_samples(samples)
        except Exception:
            self.failed = True; raise

    def training_batch(self, indices): return self._batch(indices,training=True)

    def inference_batch(self, indices, *, role):
        if role != 'train':
            self.failed = True
            raise ValueError('expanded stream has training rows only; use original transfer stream')
        return self._batch(indices,training=False)
