"""Synthetic experiment orchestration, accounting and failure preservation."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.tests.test_matched_action_hazard_evaluation_development import fixture
from lewm.tests.test_independent_pulse_study_runner_development import SyntheticStream
from scripts.independent_rgb_body_study_data_development import StudyData
import scripts.navigation_artifact_root_development as authority
import scripts.run_go2_independent_pulse_matched_study_v1 as runner


@lru_cache(maxsize=1)
def population():
    inv, windows, targets, roles, prefixes = fixture()
    # All planned prefixes are known in this synthetic collection; most windows
    # remain excluded. Keep those planned denominators in the report.
    for e in inv.episodes.values():
        group = tuple(e[k] for k in ('layout_id', 'context_kind', 'history_kind', 'support'))
        prefixes[e['episode_id']] = dict(status='COMPLETE_PREFIX', native_samples=1150, frames=9,
            sha256={'synthetic_group': hashlib.sha256(repr(group).encode()).hexdigest()})
    return inv, windows, targets, roles, prefixes


def make_study(*, omitted=()):
    inv, windows, targets, roles, prefixes = deepcopy(population())
    names = {windows[i]['condition'] for i in omitted}
    windows = [w for w in windows if w['condition'] not in names]
    targets = [t for t in targets if t['condition'] not in names]
    roles = {k: v for k, v in roles.items() if k not in names}
    view = IndependentPulseEvaluation(inv, PulseTimedDataset(windows, targets, roles))
    batches = {b: dict(role=inv.episodes[inv.episode_ids(b)[0]]['role'], population={},
        excluded_trials=[c for c in inv.episode_ids(b) if c not in roles], strict_visibility_failed_trials=[])
        for b in runner.BATCHES}
    receipts = {b: {'launch.json': 'a' * 64, 'rgb_body_layout_audit.json': 'b' * 64} for b in runner.BATCHES}
    return StudyData(view, prefixes, batches, receipts)


def test_fixed_scientific_budget_and_comparison_roster():
    assert len(runner.SEEDS) == len(set(runner.SEEDS)) == 3
    assert runner.CONDITIONS == ('direct', 'supervised_rollout', 'jepa')
    assert runner.VARIANTS == ('full', 'no_rgb', 'latest_packet_only', 'no_candidate_command')
    assert runner.UPDATES == 1200 and runner.BATCH_SIZE == 6 and runner.LATENT_DIM == 32
    assert runner.BUDGET == 8 * 1024**3 and runner.RESERVE == 40 * 1024**3


def test_coverage_retains_planned_denominators_and_separate_supervision():
    study = make_study(); d = study.evaluation.dataset
    # Preserve a positive contact but censor its future image.
    d.windows[0]['targets'][0]['future_valid'] = False
    report = runner.coverage(study)
    assert report['eligible_to_fit'] and not report['blocking_reasons']
    train = report['roles']['train']
    assert train['population']['planned_episodes'] == 720
    assert not train['population']['all_planned_episodes_eligible']
    assert train['totals']['positive_without_future_image'] == 1
    assert train['totals']['positive_with_future_image'] + 1 == train['totals']['contact_positive']
    assert train['totals']['motion_with_future_image'] == train['totals']['motion']
    assert train['observed_contact_contrast_available']
    assert report['roles']['development_eval']['population']['planned_episodes'] == 360
    assert not report['goal_achieved'] and not report['navigation_qualified']


@pytest.mark.parametrize('fault', ['action', 'layout', 'missing_prefix', 'unequal_prefix'])
def test_coverage_blocks_missing_cells_and_invalid_matching_without_dropping_them(fault):
    study = make_study(omitted=[0] if fault == 'action' else range(6) if fault == 'layout' else ())
    first = study.evaluation.inventory.episode_ids('l00')[0]
    if fault == 'missing_prefix': study.prefixes.pop(first)
    elif fault == 'unequal_prefix': study.prefixes[first]['sha256']['synthetic_group'] = 'f' * 64
    result = runner.coverage(study)
    assert not result['eligible_to_fit'] and result['blocking_reasons']
    assert result['roles']['train']['population']['planned_episodes'] == 720


def test_no_positive_contacts_is_explicitly_not_hazard_discrimination_evidence():
    study = make_study()
    for t in study.evaluation.dataset._targets: t['contact'][t['contact_valid']] = 0.
    result = runner.coverage(study)
    assert result['eligible_to_fit']
    assert all(not r['observed_contact_contrast_available'] and r['totals']['contact_positive'] == 0
        for r in result['roles'].values())
    assert result['no_contact_contrast_is_not_positive_hazard_evidence']


def test_zero_motion_baseline_reuses_empirical_contact_and_exposes_missing_cells():
    study = make_study(); view = study.evaluation
    train = view.arrays('train'); ids = train['indices'].tolist()
    empirical, exposure = view.fit_action_time(ids + ids[:3])
    heads, missing = runner.baseline_heads(view, empirical, 'development_eval')
    a, z = heads['action_time']['prediction'], heads['zero_motion_empirical_contact']['prediction']
    np.testing.assert_array_equal(z[..., 4], a[..., 4])
    np.testing.assert_array_equal(z[..., :4], np.broadcast_to([0., 0., 0., 1.], z[..., :4].shape))
    assert missing and exposure['draws'] == len(ids) + 3  # No observed motion for forward actions in this fixture.


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(authority, 'BASE', tmp_path)
    monkeypatch.setattr(runner, 'BASE', tmp_path)
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path / 'go2_synthetic_matched_attempt_001')
    monkeypatch.setattr(runner, 'SEQUENCE', tmp_path / 'go2_synthetic_sequence_attempt_001')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda p: SimpleNamespace(free=200 * 1024**3))
    return tmp_path


@pytest.fixture
def small_experiment(monkeypatch, isolated):
    study = make_study()
    monkeypatch.setattr(runner, 'SEEDS', (37,))
    monkeypatch.setattr(runner, 'UPDATES', 1)
    monkeypatch.setattr(runner, 'BATCH_SIZE', 2)
    monkeypatch.setattr(runner, 'INFERENCE_BATCH', 16)
    monkeypatch.setattr(runner, 'LATENT_DIM', 8)
    class Stream(SyntheticStream):
        def __init__(self, supplied):
            self.evaluation = supplied.evaluation; self.calls = []
    stream = Stream(study)
    monkeypatch.setattr(runner, 'AuditedStudyStream', lambda supplied: stream)
    checks = []
    monkeypatch.setattr(runner, 'verify_study_definition', lambda d, supplied: checks.append(True))
    monkeypatch.setattr(runner, 'load_study', lambda receipts: study)
    return study, runner.settings(), stream, checks


def read(name):
    return json.loads((runner.OUTPUT / name).read_text())


def test_complete_synthetic_factorial_has_bound_reload_predictions_paired_scores_and_ledgers(small_experiment):
    study, definition, stream, checks = small_experiment
    runner.execute(study, definition, {'result.json': 'c' * 64})
    result = read('result.json')
    assert result['fits'] == result['optimizer_updates'] == 12
    assert len(result['completed_fits']) == 12 and not result['goal_achieved']
    assert len(checks) == 25 and not (runner.OUTPUT / 'failure.json').exists()
    schedule = read('seed_37_schedule.json'); first_hashes = set()
    for variant in runner.VARIANTS:
        for condition in runner.CONDITIONS:
            name = f'seed_37_{variant}_{condition}'
            request = read(name + '_request.json'); first_hashes.add(request['initial_sha256'])
            fit = read(name + '_fit.json')
            ledger = [json.loads(s) for s in (runner.OUTPUT / (name + '_updates.jsonl')).read_text().splitlines()]
            assert len(ledger) == 1 and ledger[0]['update'] == 1
            assert ledger[0]['sample_indices'] == schedule['batches'][0]
            assert ledger[0]['input_variant'] == fit['fit']['input_variant'] == request['binding']['input_variant'] == variant
            assert fit['snapshot']['binding'] == request['binding']
            assert fit['snapshot']['evaluation_only_reload_verified']
            for role in runner.ROLES:
                prediction = read(name + '_' + role + '_prediction.json')
                assert prediction['model_sha256'] == fit['fit']['model_sha256'] and prediction['updates'] == 1
                assert prediction['input_variant'] == variant
                with np.load(runner.OUTPUT / (name + '_' + role + '.npz'), allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['indices'], study.evaluation.arrays(role)['indices'])
    assert len(first_hashes) == 1
    for role in runner.ROLES:
        score = read('seed_37_' + role + '_scores.json')
        assert len(score['primary_heads']) == 12 and score['no_best_seed_or_checkpoint_selection']
        assert score['prediction']['paired_comparisons'] and score['matched_contact']['paired_comparisons']
        assert score['baseline_missing_cells']
    assert [ids for kind, ids, role in stream.calls if kind == 'training'] == [schedule['batches'][0]] * 12
    authority.verify_artifacts(runner.OUTPUT, result['output_sha256'])
    with pytest.raises(ValueError, match='exclusive'): runner.execute(study, definition, {})


@pytest.mark.parametrize('fault', ['coverage', 'materialization', 'accounting', 'snapshot', 'inference', 'final_data'])
def test_failures_stop_without_retry_and_keep_actual_update_accounting(monkeypatch, small_experiment, fault):
    study, definition, stream, checks = small_experiment
    if fault == 'coverage':
        monkeypatch.setattr(runner, 'coverage', lambda s: {'eligible_to_fit': False, 'blocking_reasons': ['synthetic']})
    elif fault == 'materialization':
        monkeypatch.setattr(stream, 'training_batch', lambda ids: (_ for _ in ()).throw(ValueError('synthetic materialization')))
    elif fault == 'accounting':
        original = runner.os.fsync
        request_seen = 0
        def fail(fd):
            nonlocal request_seen
            # First per-fit ledger is the first fsync after the request is written.
            if (runner.OUTPUT / 'seed_37_full_direct_request.json').exists():
                request_seen += 1
                if request_seen == 2: raise OSError('synthetic ledger durability')
            return original(fd)
        monkeypatch.setattr(runner.os, 'fsync', fail)
    elif fault == 'snapshot':
        monkeypatch.setattr(runner, 'save_snapshot', lambda *a: (_ for _ in ()).throw(ValueError('synthetic snapshot')))
    elif fault == 'inference':
        monkeypatch.setattr(runner, 'predict_heads', lambda *a, **k: (_ for _ in ()).throw(ValueError('synthetic inference')))
    else:
        monkeypatch.setattr(runner, 'load_study', lambda receipts: (_ for _ in ()).throw(ValueError('synthetic changed final data')))
    with pytest.raises((ValueError, OSError)): runner.execute(study, definition, {})
    failure = read('failure.json')
    assert not (runner.OUTPUT / 'result.json').exists() and not failure['retry_performed']
    if fault == 'coverage':
        assert failure['actual_optimizer_updates_current_fit'] is None and not stream.calls
    elif fault == 'materialization': assert failure['actual_optimizer_updates_current_fit'] == 0
    elif fault == 'accounting':
        assert failure['actual_optimizer_updates_current_fit'] == 1
        assert (runner.OUTPUT / failure['partial_ledger']).stat().st_size > 0
    elif fault in ('snapshot', 'inference'): assert failure['actual_optimizer_updates_current_fit'] == 1
    else: assert len(failure['completed_fits']) == 12
    with pytest.raises(ValueError, match='exclusive'): runner.execute(study, definition, {})


def test_mismatched_settings_rejected_before_output_creation(small_experiment):
    study, definition, stream, checks = small_experiment
    definition['updates_per_fit'] += 1
    with pytest.raises(ValueError, match='settings'): runner.execute(study, definition, {})
    assert not runner.OUTPUT.exists() and not stream.calls


@pytest.mark.parametrize('fault', ['snapshot_variant', 'prediction_variant', 'prediction_weights', 'post_fit_source'])
def test_mismatched_provenance_cannot_publish_scores(monkeypatch, small_experiment, fault):
    study, definition, stream, checks = small_experiment
    if fault == 'snapshot_variant':
        original = runner.Artifacts.snapshot
        def broken(self, name, trainer, binding):
            row, clone = original(self, name, trainer, binding)
            row['binding']['input_variant'] = 'no_rgb'
            return row, clone
        monkeypatch.setattr(runner.Artifacts, 'snapshot', broken)
    elif fault.startswith('prediction_'):
        original = runner.predict_heads
        def broken(*args, **kwargs):
            row = original(*args, **kwargs)
            row['input_variant' if fault == 'prediction_variant' else 'model_sha256'] = 'wrong'
            return row
        monkeypatch.setattr(runner, 'predict_heads', broken)
    else:
        def verify(d, s):
            checks.append(True)
            if len(checks) == 2: raise ValueError('synthetic changed source')
        monkeypatch.setattr(runner, 'verify_study_definition', verify)
    with pytest.raises(ValueError): runner.execute(study, definition, {})
    failure = read('failure.json')
    assert failure['actual_optimizer_updates_current_fit'] == 1
    assert not failure['completed_fits'] and not (runner.OUTPUT / 'result.json').exists()
    assert not (runner.OUTPUT / 'seed_37_development_eval_scores.json').exists()


@pytest.mark.parametrize('name', ['sealed_test.json', '../escape.json', 'nested/file.json', 'result.txt'])
def test_artifact_names_rejected_without_writes(isolated, name):
    runner.create_output(runner.OUTPUT); files = runner.Artifacts(runner.OUTPUT)
    with pytest.raises(ValueError): files.save(name, {})
    assert files.used == 0 and not files.hashes


def test_artifact_bounds_exclusive_identity_and_reserve(monkeypatch, isolated):
    runner.create_output(runner.OUTPUT); files = runner.Artifacts(runner.OUTPUT)
    files.save('first.json', {'value': 1}); original = (runner.OUTPUT / 'first.json').read_bytes()
    with pytest.raises(ValueError, match='exclusive'): files.save('first.json', {'value': 2})
    assert (runner.OUTPUT / 'first.json').read_bytes() == original
    monkeypatch.setattr(runner, 'BUDGET', files.used)
    with pytest.raises(ValueError, match='allowance'): files.save('second.json', {})
    assert not (runner.OUTPUT / 'second.json').exists()
    monkeypatch.setattr(runner, 'BUDGET', 8 * 1024**3)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda p: SimpleNamespace(free=runner.RESERVE))
    with pytest.raises(ValueError, match='reserve'): files.save('second.json', {})


@pytest.fixture
def preflight_io(monkeypatch, isolated):
    d = runner.settings(); study = make_study(); reads = []
    terminal = dict(status='ALL12_FIXED_RGB_BODY_BATCH_RECEIPTS_VERIFIED',
        completed_batches=list(runner.BATCHES), receipts=study.receipts,
        planned_layouts=12, planned_episodes=1440, model_training=False, final_evaluation=False,
        navigation_qualified=False, goal_achieved=False, output_sha256={})
    for key, value in runner.ENVIRONMENT.items(): monkeypatch.setenv(key, value)
    before = torch.are_deterministic_algorithms_enabled(); torch.use_deterministic_algorithms(True)
    monkeypatch.setattr(runner, 'definition', lambda: deepcopy(d))
    monkeypatch.setattr(runner, 'verify_artifacts', lambda root, bindings: reads.append(('verify', deepcopy(bindings))))
    monkeypatch.setattr(runner, 'read_json', lambda root, name: reads.append(('read', name)) or deepcopy(terminal))
    monkeypatch.setattr(runner, 'load_study', lambda receipts: reads.append(('load', receipts)) or study)
    yield d, study, terminal, reads
    torch.use_deterministic_algorithms(before)


def test_preflight_verifies_terminal_receipts_before_loading_and_creates_nothing(preflight_io):
    d, study, terminal, reads = preflight_io
    loaded, actual, receipt = runner.preflight('c' * 64, runner.identity(d))
    assert loaded is study and actual == d and receipt['result.json'] == 'c' * 64
    assert [r[0] for r in reads] == ['verify', 'read', 'verify', 'load', 'verify']
    assert not runner.OUTPUT.exists()


@pytest.mark.parametrize('fault', ['definition', 'incomplete', 'role_roster', 'promoted', 'environment', 'missing_terminal'])
def test_preflight_rejects_incomplete_or_unbound_experiment_before_loading(monkeypatch, preflight_io, fault):
    d, study, terminal, reads = preflight_io; expected = runner.identity(d)
    if fault == 'definition': expected = '0' * 64
    elif fault == 'incomplete': terminal['completed_batches'].pop()
    elif fault == 'role_roster': terminal['receipts'].pop('l11')
    elif fault == 'promoted': terminal['navigation_qualified'] = True
    elif fault == 'environment': monkeypatch.setenv('OMP_NUM_THREADS', '2')
    else: monkeypatch.setattr(runner, 'verify_artifacts', lambda *a: (_ for _ in ()).throw(ValueError('missing terminal')))
    with pytest.raises(ValueError): runner.preflight('c' * 64, expected)
    assert not any(r[0] == 'load' for r in reads) and not runner.OUTPUT.exists()
