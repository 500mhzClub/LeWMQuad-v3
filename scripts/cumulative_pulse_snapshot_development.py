"""Exclusive bounded snapshots and receipt-bound evaluation-only reloads.

No checkpoint discovery, selection, resume or scientific execution authority.
The prospective caller binds its experiment, dataset, schedule, input variant
and exact expected trainer configuration; a model hash alone is insufficient.
"""
from copy import deepcopy
import hashlib
import io
import math
import os
import re
import shutil

import torch

from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer
from lewm.pulse_position_scale_learning_development import POSITION_SCALE_M
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.navigation_artifact_root_development import validate_root, artifact_path, verify_artifacts

SCHEMA = 'receipt_bound_cumulative_pulse_snapshot.v1'
MAX_BYTES = 64 * 1024**2
RESERVE = 40 * 1024**3
CONFIG_KEYS = ('condition', 'seed', 'latent_dim', 'learning_rate', 'ema_momentum', 'updates')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_tree(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.device == right.device and torch.equal(left, right)
    return left == right


def validate_binding(binding):
    require(isinstance(binding, dict) and set(binding) ==
        {'experiment_sha256', 'dataset_sha256', 'schedule_sha256', 'input_variant'}, 'exact experiment/data/schedule/variant binding required')
    require(all(type(binding[k]) is str and re.fullmatch('[0-9a-f]{64}', binding[k]) for k in
        ('experiment_sha256', 'dataset_sha256', 'schedule_sha256')), 'SHA-256 experiment identities required')
    require(type(binding['input_variant']) is str and re.fullmatch('[a-z][a-z0-9_]{0,63}', binding['input_variant']),
        'explicit named input variant required')


def config(trainer):
    return {k: getattr(trainer, k) for k in CONFIG_KEYS}


def snapshot_path(directory, name, *, existing):
    root = validate_root(directory)
    require(type(name) is str and re.fullmatch('[a-z][a-z0-9_]*[.]pt', name)
        and not name.startswith('sealed_'), 'ordinary explicit snapshot filename required')
    p = root / name
    require(not p.is_symlink(), 'nonsymlink snapshot required')
    return artifact_path(root, name) if existing else p


class EvaluationOnlyTrainer(CumulativePulseTrainer):
    """Compatible with streamed scoring, but not a training-resume interface."""
    def step(self, batch):
        raise ValueError('reloaded checkpoint is evaluation-only; no resume')


def validate_payload(payload, expected_binding, expected_config):
    validate_binding(expected_binding)
    require(isinstance(expected_config, dict) and set(expected_config) == set(CONFIG_KEYS), 'exact expected trainer configuration required')
    require(isinstance(payload, dict) and set(payload) == {'schema', 'binding', 'trainer'}
        and payload['schema'] == SCHEMA and equal_tree(payload['binding'], expected_binding),
        'snapshot experiment identity mismatch')
    saved = payload['trainer']
    fields = {'schema', 'contact_semantics', 'position_scale_m', *CONFIG_KEYS, 'failed', 'initial_sha256',
        'model_sha256', 'model_state', 'optimizer_state', 'navigation_qualified'}
    require(isinstance(saved, dict) and set(saved) == fields, 'exact cumulative checkpoint fields required')
    require(saved['schema'] == 'cumulative_pulse_training_development.v1'
        and saved['contact_semantics'] == 'integrated_softplus_hazard_per_second'
        and saved['position_scale_m'] == POSITION_SCALE_M
        and saved['failed'] is False and saved['navigation_qualified'] is False,
        'nonfailed cumulative-event, fixed-position-scale checkpoint required')
    require(equal_tree({k: saved[k] for k in CONFIG_KEYS}, expected_config), 'checkpoint configuration or update budget mismatch')
    require(type(saved['updates']) is int and saved['updates'] >= 0, 'nonnegative exact optimizer update count required')
    require(all(type(saved[k]) in (int, float) and math.isfinite(saved[k]) for k in ('learning_rate', 'ema_momentum')),
        'finite numeric optimizer/EMA configuration required')
    clone = EvaluationOnlyTrainer(**{k: saved[k] for k in CONFIG_KEYS if k != 'updates'})
    require(saved['initial_sha256'] == clone.initial_sha256, 'seeded initial model identity mismatch')
    reference = clone.model.state_dict(); state = saved['model_state']
    require(isinstance(state, dict) and set(state) == set(reference), 'exact model state roster required')
    for name, value in state.items():
        r = reference[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == r.dtype and value.shape == r.shape and torch.isfinite(value).all().item(),
            'finite exact CPU model tensor required: ' + name)
    require(state_digest(state) == saved['model_sha256'], 'model tensor hash mismatch')
    require(saved['updates'] != 0 or saved['model_sha256'] == saved['initial_sha256'],
        'zero-update checkpoint must retain its seeded initial model')
    optim = saved['optimizer_state']; fresh = clone.optimizer.state_dict()
    require(isinstance(optim, dict) and set(optim) == {'state', 'param_groups'}
        and equal_tree(optim['param_groups'], fresh['param_groups']), 'unchanged optimizer groups/hyperparameters required')
    ids = fresh['param_groups'][0]['params']
    require(isinstance(optim['state'], dict) and set(optim['state']) == (set(ids) if saved['updates'] else set()),
        'optimizer state must cover every active parameter exactly after an update')
    for i, parameter in zip(ids, clone.parameters, strict=True):
        if not saved['updates']:
            continue
        row = optim['state'][i]
        require(isinstance(row, dict) and set(row) == {'step', 'exp_avg', 'exp_avg_sq'}, 'exact AdamW moment fields required')
        step = row['step']
        require(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and step.dtype == torch.float32
            and step.shape == () and step.item() == saved['updates'], 'every optimizer step must equal the recorded update count')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = row[name]
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == parameter.dtype and value.shape == parameter.shape and torch.isfinite(value).all().item(),
                'finite exact optimizer moment tensor required')
        require((row['exp_avg_sq'] >= 0).all().item(), 'second moments cannot be negative')
    clone.model.load_state_dict(state, strict=True); clone.optimizer.load_state_dict(deepcopy(optim)); clone.updates = saved['updates']
    require(state_digest(clone.model.state_dict()) == saved['model_sha256']
        and equal_tree(clone.optimizer.state_dict(), optim), 'model/optimizer reload identity mismatch')
    clone.model.eval()
    require(all(p.grad is None for p in clone.model.parameters()), 'reload must not retain training gradients')
    return clone


def load_snapshot(directory, name, *, sha256, expected_binding, expected_config):
    """Verify bounded bytes before restricted deserialization and exact reload."""
    p = snapshot_path(directory, name, existing=True)
    require(p.stat().st_size <= MAX_BYTES, 'snapshot exceeds bounded size')
    verify_artifacts(directory, {name: sha256})
    with p.open('rb') as source:
        raw = source.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES and hashlib.sha256(raw).hexdigest() == sha256, 'snapshot bytes changed before deserialization')
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    clone = validate_payload(payload, expected_binding, expected_config)
    verify_artifacts(directory, {name: sha256})
    return clone


def save_snapshot(directory, name, trainer, binding):
    """Exclusive write, fsync and full evaluation-only reload; retain any failure."""
    require(isinstance(trainer, CumulativePulseTrainer) and not isinstance(trainer, EvaluationOnlyTrainer)
        and not trainer.failed, 'live nonfailed trainer required, not a reloaded checkpoint')
    p = snapshot_path(directory, name, existing=False)
    require(not p.exists(), 'exclusive snapshot; never overwrite or replace')
    expected_binding = deepcopy(binding); expected_config = config(trainer)
    payload = dict(schema=SCHEMA, binding=expected_binding, trainer=trainer.checkpoint())
    validate_payload(payload, expected_binding, expected_config)
    buffer = io.BytesIO(); torch.save(payload, buffer); raw = buffer.getvalue()
    require(len(raw) <= MAX_BYTES and shutil.disk_usage(directory).free >= RESERVE + len(raw), 'snapshot size/storage reserve')
    with p.open('xb') as destination:
        destination.write(raw); destination.flush(); os.fsync(destination.fileno())
    sha = hashlib.sha256(raw).hexdigest()
    clone = load_snapshot(directory, name, sha256=sha, expected_binding=expected_binding, expected_config=expected_config)
    require(equal_tree(clone.checkpoint(), trainer.checkpoint()), 'saved/reloaded complete trainer snapshot mismatch')
    return dict(filename=name, sha256=sha, bytes=len(raw), binding=expected_binding, configuration=expected_config,
        model_sha256=payload['trainer']['model_sha256'], evaluation_only_reload_verified=True,
        training_resume_authorized=False, checkpoint_selection_performed=False, navigation_qualified=False)
