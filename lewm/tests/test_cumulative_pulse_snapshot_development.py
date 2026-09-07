"""Synthetic checkpoint persistence/identity tests; no recorded-data fitting."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lewm.cumulative_pulse_learning_development import CumulativePulseTrainer
from lewm.tests.test_pulse_timed_learning_development import batch
import scripts.cumulative_pulse_snapshot_development as snapshot


def binding():
    return dict(experiment_sha256='a' * 64, dataset_sha256='b' * 64,
        schedule_sha256='c' * 64, input_variant='full')


@pytest.fixture(scope='module')
def trainers():
    data = batch(); result = {}
    for condition in ('direct', 'supervised_rollout', 'jepa'):
        trainer = CumulativePulseTrainer(condition, seed=47, latent_dim=8); trainer.step(data)
        result[condition] = trainer
    result['initial'] = CumulativePulseTrainer('jepa', seed=47, latent_dim=8)
    return result, data


@pytest.fixture
def owned_io(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, 'validate_root', lambda p: Path(p))
    monkeypatch.setattr(snapshot, 'artifact_path', lambda p, n: Path(p) / n)
    def verify(p, hashes):
        for n, h in hashes.items():
            if hashlib.sha256((Path(p) / n).read_bytes()).hexdigest() != h:
                raise ValueError('synthetic artifact identity mismatch')
    monkeypatch.setattr(snapshot, 'verify_artifacts', verify)
    monkeypatch.setattr(snapshot.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 1024**3))
    return tmp_path


@pytest.mark.parametrize('condition', ['direct', 'supervised_rollout', 'jepa', 'initial'])
def test_exclusive_round_trip_preserves_model_optimizer_and_forward(condition, trainers, owned_io):
    mapping, data = trainers; trainer = mapping[condition]
    result = snapshot.save_snapshot(owned_io, 'snapshot.pt', trainer, binding())
    clone = snapshot.load_snapshot(owned_io, 'snapshot.pt', sha256=result['sha256'],
        expected_binding=binding(), expected_config=snapshot.config(trainer))
    assert snapshot.equal_tree(clone.checkpoint(), trainer.checkpoint())
    with torch.no_grad():
        before = trainer.model(**data['inputs']); after = clone.model(**data['inputs'])
    for k in before: torch.testing.assert_close(before[k], after[k], rtol=0, atol=0)
    assert result['evaluation_only_reload_verified'] and not result['training_resume_authorized']
    assert not clone.model.training and clone.updates == trainer.updates
    with pytest.raises(ValueError, match='evaluation-only'): clone.step(data)
    with pytest.raises(ValueError, match='reloaded'): snapshot.save_snapshot(owned_io, 'copy.pt', clone, binding())
    old = (owned_io / 'snapshot.pt').read_bytes()
    with pytest.raises(ValueError, match='overwrite'): snapshot.save_snapshot(owned_io, 'snapshot.pt', trainer, binding())
    assert (owned_io / 'snapshot.pt').read_bytes() == old


@pytest.mark.parametrize('fault', ['binding', 'variant', 'config', 'update', 'legacy', 'contact', 'scale',
    'failed', 'promoted', 'initial_hash', 'model_hash', 'model_nan', 'model_dtype', 'model_roster',
    'optimizer_group', 'optimizer_roster', 'optimizer_step', 'optimizer_nan', 'negative_second_moment', 'extra'])
def test_payload_corruption_or_wrong_expectation_is_rejected(trainers, fault):
    trainer = trainers[0]['jepa']; b = binding(); cfg = snapshot.config(trainer)
    payload = dict(schema=snapshot.SCHEMA, binding=deepcopy(b), trainer=trainer.checkpoint()); s = payload['trainer']
    param = next(iter(s['model_state'])); optim = s['optimizer_state']; first = next(iter(optim['state']))
    if fault == 'binding': b['dataset_sha256'] = 'd' * 64
    elif fault == 'variant': b['input_variant'] = 'no_rgb'
    elif fault == 'config': cfg['seed'] += 1
    elif fault == 'update': cfg['updates'] += 1
    elif fault == 'legacy': s['schema'] = 'old_checkpoint'
    elif fault == 'contact': s['contact_semantics'] = 'independent_logits'
    elif fault == 'scale': s['position_scale_m'] = 1.
    elif fault == 'failed': s['failed'] = True
    elif fault == 'promoted': s['navigation_qualified'] = True
    elif fault == 'initial_hash': s['initial_sha256'] = '0' * 64
    elif fault == 'model_hash': s['model_sha256'] = '0' * 64
    elif fault == 'model_nan': s['model_state'][param].flatten()[0] = float('nan')
    elif fault == 'model_dtype': s['model_state'][param] = s['model_state'][param].double()
    elif fault == 'model_roster': s['model_state'].pop(param)
    elif fault == 'optimizer_group': optim['param_groups'][0]['lr'] *= 2
    elif fault == 'optimizer_roster': optim['state'].pop(first)
    elif fault == 'optimizer_step': optim['state'][first]['step'] += 1
    elif fault == 'optimizer_nan': optim['state'][first]['exp_avg'].flatten()[0] = float('nan')
    elif fault == 'negative_second_moment': optim['state'][first]['exp_avg_sq'].flatten()[0] = -1.
    else: s['undeclared'] = 1
    with pytest.raises(ValueError): snapshot.validate_payload(payload, b, cfg)


def test_zero_update_snapshot_cannot_hide_modified_weights(trainers):
    trainer = trainers[0]['initial']; payload = dict(schema=snapshot.SCHEMA, binding=binding(), trainer=trainer.checkpoint())
    state = payload['trainer']['model_state']; next(iter(state.values())).add_(.1)
    payload['trainer']['model_sha256'] = snapshot.state_digest(state)
    with pytest.raises(ValueError, match='zero-update'): snapshot.validate_payload(payload, binding(), snapshot.config(trainer))


@pytest.mark.parametrize('name', ['../escape.pt', '/tmp/escape.pt', 'nested/snapshot.pt', 'sealed_snapshot.pt', 'snapshot.pkl'])
def test_bad_filename_is_rejected_without_writing(owned_io, trainers, name):
    with pytest.raises(ValueError): snapshot.save_snapshot(owned_io, name, trainers[0]['jepa'], binding())
    assert not list(owned_io.iterdir())


@pytest.mark.parametrize('fault', ['hash', 'changed_bytes', 'size', 'symlink', 'storage'])
def test_persistence_and_load_fail_closed(monkeypatch, owned_io, trainers, fault):
    trainer = trainers[0]['jepa']
    if fault == 'storage':
        monkeypatch.setattr(snapshot.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0))
        with pytest.raises(ValueError, match='reserve'): snapshot.save_snapshot(owned_io, 'snapshot.pt', trainer, binding())
        assert not (owned_io / 'snapshot.pt').exists(); return
    result = snapshot.save_snapshot(owned_io, 'snapshot.pt', trainer, binding()); name = 'snapshot.pt'; sha = result['sha256']
    if fault == 'hash': sha = '0' * 64
    elif fault == 'changed_bytes':
        with (owned_io / name).open('ab') as f: f.write(b'changed')
    elif fault == 'size': monkeypatch.setattr(snapshot, 'MAX_BYTES', 100)
    else: (owned_io / 'link.pt').symlink_to(owned_io / name); name = 'link.pt'
    monkeypatch.setattr(snapshot.torch, 'load', lambda *a, **k: pytest.fail('must reject before deserialization'))
    with pytest.raises(ValueError): snapshot.load_snapshot(owned_io, name, sha256=sha,
        expected_binding=binding(), expected_config=snapshot.config(trainer))


def test_load_uses_restricted_cpu_deserialization(monkeypatch, owned_io, trainers):
    trainer = trainers[0]['direct']; result = snapshot.save_snapshot(owned_io, 'snapshot.pt', trainer, binding())
    real_load = torch.load; calls = []
    def load(*a, **kwargs): calls.append(kwargs); return real_load(*a, **kwargs)
    monkeypatch.setattr(snapshot.torch, 'load', load)
    clone = snapshot.load_snapshot(owned_io, 'snapshot.pt', sha256=result['sha256'],
        expected_binding=binding(), expected_config=snapshot.config(trainer))
    assert calls == [dict(map_location='cpu', weights_only=True)]
    clone.optimizer.state[next(iter(clone.optimizer.state))]['exp_avg'].fill_(999.)
    assert not snapshot.equal_tree(clone.optimizer.state_dict(), trainer.optimizer.state_dict())


def test_payload_mutation_cannot_change_reloaded_optimizer_or_model(trainers):
    trainer = trainers[0]['jepa']; payload = dict(schema=snapshot.SCHEMA, binding=binding(), trainer=trainer.checkpoint())
    clone = snapshot.validate_payload(payload, binding(), snapshot.config(trainer)); before = clone.checkpoint()
    next(iter(payload['trainer']['model_state'].values())).add_(99.)
    next(iter(payload['trainer']['optimizer_state']['state'].values()))['exp_avg'].fill_(99.)
    assert snapshot.equal_tree(clone.checkpoint(), before)


def test_change_after_file_hash_is_rejected_before_deserialization(monkeypatch, owned_io, trainers):
    trainer = trainers[0]['direct']; result = snapshot.save_snapshot(owned_io, 'snapshot.pt', trainer, binding())
    def verify(p, hashes):
        with (Path(p) / 'snapshot.pt').open('ab') as f: f.write(b'changed-after-verification')
    monkeypatch.setattr(snapshot, 'verify_artifacts', verify)
    monkeypatch.setattr(snapshot.torch, 'load', lambda *a, **k: pytest.fail('must check exact read bytes before deserialization'))
    with pytest.raises(ValueError, match='bytes changed'): snapshot.load_snapshot(owned_io, 'snapshot.pt',
        sha256=result['sha256'], expected_binding=binding(), expected_config=snapshot.config(trainer))


def test_fsync_failure_keeps_written_evidence_and_does_not_retry(monkeypatch, owned_io, trainers):
    calls = []
    def fail(fd): calls.append(fd); raise OSError('synthetic persistence failure')
    monkeypatch.setattr(snapshot.os, 'fsync', fail)
    with pytest.raises(OSError): snapshot.save_snapshot(owned_io, 'snapshot.pt', trainers[0]['direct'], binding())
    assert len(calls) == 1 and (owned_io / 'snapshot.pt').is_file() and (owned_io / 'snapshot.pt').stat().st_size > 0
    with pytest.raises(ValueError, match='overwrite'): snapshot.save_snapshot(owned_io, 'snapshot.pt', trainers[0]['direct'], binding())
