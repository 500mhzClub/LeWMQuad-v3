"""Synthetic stage processes and receipts; no native collection is launched."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_go2_independent_rgb_body_remaining_stages_v1 as supervisor


def row(batch):
    return dict(batch=batch, receipt={'launch.json': 'a' * 64, supervisor.AUDIT: 'b' * 64},
        eligible_departures=119, contact_positive_targets=5, model_training=False)


@pytest.mark.parametrize('value', ['l00', 'l12', '../l01', True])
def test_exact_remaining_commands_reject_restart_or_other_targets(value):
    with pytest.raises(ValueError): supervisor.command(value)
    assert supervisor.command('l01') == [str(supervisor.PYTHON), supervisor.STAGE, '--batch', 'l01']


def test_child_uses_fresh_exact_interpreter_environment_and_no_timeout(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs)); return SimpleNamespace(returncode=23)
    monkeypatch.setattr(supervisor.subprocess, 'run', run)
    assert supervisor.run_child('l03') == 23
    command, args = calls[0]
    assert command == supervisor.command('l03')
    assert set(args) == {'cwd', 'env', 'check'} and args['cwd'] == supervisor.ROOT and not args['check']
    assert all(args['env'][k] == v for k, v in supervisor.CHILD_ENV.items())


@pytest.fixture
def synthetic_sequence(monkeypatch, tmp_path):
    output = tmp_path / 'supervisor'; roots = {b: tmp_path / b for b in supervisor.BATCHES}; roots['l00'].mkdir()
    sources = {'synthetic_collection.py': 'c' * 64}
    definition = dict(original_collection_source_sha256=sources)
    monkeypatch.setattr(supervisor, 'OUTPUT', output)
    monkeypatch.setattr(supervisor, 'output_root', roots.__getitem__)
    monkeypatch.setattr(supervisor, 'preflight', lambda sha: (object(), definition, row('l00')))
    monkeypatch.setattr(supervisor, 'create_output', lambda p: p.mkdir())
    monkeypatch.setattr(supervisor, 'verify_definition', lambda d: None)
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 1024**3))
    def verify(root, hashes):
        for n, h in hashes.items():
            assert supervisor.digest(root / n) == h
    monkeypatch.setattr(supervisor, 'verify_artifacts', verify)
    calls = []
    def child(batch):
        calls.append(('child', batch)); roots[batch].mkdir()
        (roots[batch] / 'launch.json').write_text(json.dumps(dict(source_sha256=sources)))
        (roots[batch] / supervisor.AUDIT).write_text('{}')
        return 0
    def load(batch, receipt, inv):
        calls.append(('load', batch)); return row(batch)
    monkeypatch.setattr(supervisor, 'run_child', child)
    monkeypatch.setattr(supervisor, 'load_batch', load)
    monkeypatch.setattr(supervisor, 'summarize', lambda b, joined: deepcopy(joined))
    return output, roots, calls, child, load


def test_success_executes_only_fixed_remaining_stages_and_rechecks_all12(synthetic_sequence):
    output, _, calls, _, _ = synthetic_sequence
    supervisor.run_sequence('b' * 64)
    result = json.loads((output / 'result.json').read_text())
    assert [b for kind, b in calls if kind == 'child'] == list(supervisor.BATCHES[1:])
    assert calls[:22] == [(kind, b) for b in supervisor.BATCHES[1:] for kind in ('child', 'load')]
    assert calls[22:] == [('load', b) for b in supervisor.BATCHES]
    assert result['completed_batches'] == list(supervisor.BATCHES) and len(result['receipts']) == 12
    assert not result['model_training'] and not result['final_evaluation'] and not result['goal_achieved']
    assert not (output / 'failure.json').exists()
    for batch in supervisor.BATCHES[1:]:
        candidate = json.loads((output / (batch + '_candidate_receipt.json')).read_text())
        assert candidate['verified'] is False
        assert json.loads((output / (batch + '_stage_exit.json')).read_text())['returncode'] == 0


@pytest.mark.parametrize('fault', ['child_nonzero', 'child_exception', 'loader', 'changed_source',
    'preexisting', 'metadata', 'final_recheck'])
def test_any_stage_failure_preserves_evidence_and_never_retries(monkeypatch, synthetic_sequence, fault):
    output, roots, calls, child, load = synthetic_sequence
    if fault in ('child_nonzero', 'child_exception', 'changed_source', 'metadata'):
        def changed(batch):
            result = child(batch)
            if batch == 'l02':
                if fault == 'child_exception': raise OSError('synthetic launch failure')
                if fault == 'child_nonzero': return 4
                if fault == 'changed_source': (roots[batch] / 'launch.json').write_text('{"source_sha256": {}}')
                if fault == 'metadata': monkeypatch.setattr(supervisor, 'METADATA_BUDGET', 1)
            return result
        monkeypatch.setattr(supervisor, 'run_child', changed)
    elif fault in ('loader', 'final_recheck'):
        def changed_load(batch, receipt, inv):
            if (fault == 'loader' and batch == 'l02') or (fault == 'final_recheck' and batch == 'l00'):
                raise ValueError('synthetic receipt failure')
            return load(batch, receipt, inv)
        monkeypatch.setattr(supervisor, 'load_batch', changed_load)
    else: roots['l02'].mkdir()
    with pytest.raises((ValueError, OSError)): supervisor.run_sequence('b' * 64)
    failure = json.loads((output / 'failure.json').read_text())
    children = [b for kind, b in calls if kind == 'child']
    expected = list(supervisor.BATCHES[1:]) if fault == 'final_recheck' else ['l01'] if fault == 'preexisting' else ['l01', 'l02']
    assert children == expected and len(set(children)) == len(children)
    assert failure['completed_batches'] == (list(supervisor.BATCHES) if fault == 'final_recheck' else ['l00', 'l01'])
    assert not (output / 'result.json').exists() and not failure['retry_performed']
    if fault == 'child_nonzero': assert failure['child_returncode'] == 4 and failure['active_batch'] == 'l02'


@pytest.mark.parametrize('fault', ['none', 'output_exists', 'remaining_exists', 'receipt', 'loader', 'reserve', 'launch_size'])
def test_preflight_requires_completed_first_receipt_and_all_remaining_roots_absent(monkeypatch, tmp_path, fault):
    output = tmp_path / 'supervisor'; roots = {b: tmp_path / b for b in supervisor.BATCHES}; roots['l00'].mkdir()
    monkeypatch.setattr(supervisor, 'OUTPUT', output)
    monkeypatch.setattr(supervisor, 'output_root', roots.__getitem__)
    monkeypatch.setattr(supervisor, 'validate_root', lambda *a, **k: None)
    if fault == 'output_exists': output.mkdir()
    if fault == 'remaining_exists': roots['l11'].mkdir()
    calls = []
    def verify(root, hashes):
        calls.append('receipt')
        assert hashes['launch.json'] == supervisor.FIRST_LAUNCH and hashes[supervisor.AUDIT] == 'd' * 64
        if fault == 'receipt': raise ValueError('missing/incomplete audit or wrong hash')
    monkeypatch.setattr(supervisor, 'verify_artifacts', verify)
    monkeypatch.setattr(supervisor, 'load_inventory', lambda: object())
    def load(*a):
        calls.append('load')
        if fault == 'loader': raise ValueError('failed completed-data verification')
        return row('l00')
    monkeypatch.setattr(supervisor, 'load_batch', load)
    old = {k: {} for k in ('source_sha256', 'input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_binary_sha256', 'rules')}; old['opencv_version'] = 'synthetic'
    monkeypatch.setattr(supervisor, 'read_json', lambda *a: old)
    monkeypatch.setattr(supervisor, 'discover_sources', lambda seeds, inherited: inherited | {'synthetic.py': '0' * 64})
    monkeypatch.setattr(supervisor, 'verify_definition', lambda d: None)
    monkeypatch.setattr(supervisor, 'summarize', lambda b, joined: joined)
    monkeypatch.setattr(supervisor.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0 if fault == 'reserve' else 100 * 1024**3))
    if fault == 'launch_size': monkeypatch.setattr(supervisor, 'METADATA_BUDGET', 1)
    if fault == 'none':
        _, definition, first = supervisor.preflight('d' * 64)
        assert definition['remaining_batches'] == list(supervisor.BATCHES[1:])
        assert len(definition['commands']) == 11 and first['batch'] == 'l00'
    else:
        with pytest.raises(ValueError): supervisor.preflight('d' * 64)
        if fault in ('output_exists', 'remaining_exists'): assert not calls
    assert not (output / 'launch.json').exists()
