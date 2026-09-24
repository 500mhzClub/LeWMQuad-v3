"""Synthetic terminal receipts and joins; no recorded training or new physics."""
from copy import deepcopy
import hashlib
import json
import pytest

import scripts.independent_rgb_body_study_data_development as study
from lewm.tests.test_independent_pulse_evaluation_development import fixture
from lewm.pulse_timed_dataset_development import PulseTimedDataset


@pytest.fixture(scope='module')
def population():
    return fixture()


def documents(population, batch='l00', *, empty=False):
    inv, ww, tt, _ = population
    ids = list(inv.episode_ids(batch)); labels = {t['condition']: t for t in tt}
    chosen = {} if empty else {w['condition']: w for w in ww if w['condition'] in ids}
    docs = {}; reports = {}; prefixes = {}; covs = {}; windows = []; targets = []
    for c in ids:
        e = inv.episodes[c]; w = deepcopy(chosen.get(c)); t = deepcopy(labels[c]) if w else None
        # First eligible case retains a strict boundary failure, with no hard
        # measurement failure. Remaining missing windows represent failed setup.
        boundary = bool(w) and c == ids[0]
        coverage = dict(candidate_acquisition_complete=bool(w), paired_frames=1,
            classification='SCHEDULE_RECORDED' if w else 'SETUP_FAILED')
        frames = [dict(frame=0, score=dict(stable_interior_metric_pass=True, near_occlusion_failure=False,
            original_strict_score=dict(passes_sampled_physical_visibility=not boundary)))]
        elig = study.eligibility(coverage, w, frames)
        report = dict(trial=c, layout_id=e['layout_id'], data_role=e['role'],
            **{k: e[k] for k in ('context_kind', 'history_kind', 'support')},
            recorded_sensor_reconstruction_pass=True, setup_admitted=bool(w), schedule_complete=bool(w),
            target_contact_positive=0, physical_visibility_pass=not boundary)
        prefix = dict(status='COMPLETE_PREFIX' if w else 'MISSING_DEPARTURE_PREFIX',
            native_samples=1150 if w else 750, frames=9 if w else 1,
            sha256={'synthetic': '0' * 64} if w else {})
        row = dict(report=report, prefix=prefix, window=w, targets=t, coverage=coverage,
            eligibility=elig, footprint_diagnostics=frames)
        docs[c + '_rgb_body_evaluation.json'] = row
        reports[c] = report | dict(status='RAW_RGB_BODY_EPISODE_AUDITED', eligibility=elig)
        prefixes[c] = prefix; covs[c] = coverage
        if w: windows.append(w); targets.append(t)
    roles = {w['condition']: {k: inv.episodes[w['condition']][k] for k in ('layout_id', 'role')} for w in windows}
    for name, value in zip(study.PRODUCTS, (windows, targets, windows, targets, roles, prefixes), strict=True):
        docs[name] = deepcopy(value)
    dataset = PulseTimedDataset(windows, targets, roles) if windows else None
    integrated = []
    if dataset:
        for w, t in zip(dataset.windows, dataset._targets, strict=True):
            integrated.append(dict(condition=w['condition'], input_fields=study.INPUT_FIELDS,
                motion_valid=int(t['motion_valid'].sum()), contact_positive=0))
    pairs = study.prefix_comparisons(inv, batch, prefixes)
    audit = dict(status='RGB_BODY_LAYOUT_AVAILABLE_EVIDENCE_AUDITED', batch=batch,
        role=inv.episodes[ids[0]]['role'], collection_complete=True, expected_trials=120,
        committed_trials=120, audited_trials=120, conditions=reports, independent_layouts=1,
        model_trained=False, final_evaluation=False, navigation_qualified=False, goal_achieved=False,
        output_sha256={k: '0' * 64 for k in list(docs) + [study.AUDIT_LAUNCH]},
        prefix_comparisons=pairs, exact_nonreference_prefix_matches=sum(p['matched'] for p in pairs.values() if not p['is_reference']),
        population=study.population_coverage(ids, covs), departures=len(windows), eligible_departures=len(windows),
        action_coverage=dataset.coverage(inv.episodes[ids[0]]['role']) if dataset else {},
        setup_admitted=len(windows), schedule_completions=len(windows), contact_positive_targets=0,
        hard_measurement_failed_trials=[], strict_visibility_failed_trials=[ids[0]] if windows else [],
        materialized_samples=integrated)
    return audit, docs


def test_full_population_join_retains_exclusions_and_boundary_failures(population):
    inv = population[0]; audit, docs = documents(population)
    result = study._join_audit(inv, 'l00', audit, docs.__getitem__)
    assert len(result['planned_trials']) == 120 and len(result['windows']) == 6
    assert len(result['excluded_trials']) == 114 and len(result['prefixes']) == 120
    assert result['population']['counts']['SETUP_FAILED'] == 114
    assert result['strict_visibility_failed_trials'] == [result['planned_trials'][0]]
    assert 'source_and_artifact_bindings_verified' not in result


def test_zero_eligible_batch_is_retained_not_dropped(population):
    audit, docs = documents(population, empty=True)
    result = study._join_audit(population[0], 'l00', audit, docs.__getitem__)
    assert result['windows'] == [] and len(result['excluded_trials']) == 120


@pytest.mark.parametrize('fault', ['partial', 'count', 'role', 'promoted', 'roster', 'reorder',
    'episode_identity', 'eligibility', 'hard_measurement', 'summary', 'product', 'labels',
    'prefix', 'population', 'eligible_count', 'action_coverage', 'contact_count', 'strict_failure', 'materialized'])
def test_corrupted_or_selected_metadata_cannot_enter_study(population, fault):
    inv = population[0]; audit, docs = documents(population); c = inv.episode_ids('l00')[0]
    row = docs[c + '_rgb_body_evaluation.json']
    if fault == 'partial': audit['collection_complete'] = False
    elif fault == 'count': audit['audited_trials'] -= 1
    elif fault == 'role': audit['role'] = 'development_eval'
    elif fault == 'promoted': audit['navigation_qualified'] = True
    elif fault == 'roster': audit['output_sha256']['unrelated.json'] = '0' * 64
    elif fault == 'reorder': audit['conditions'] = dict(reversed(list(audit['conditions'].items())))
    elif fault == 'episode_identity': row['report']['trial'] = inv.episode_ids('l01')[0]
    elif fault == 'eligibility': row['eligibility']['rgb_body_prediction_eligible'] = False
    elif fault == 'hard_measurement':
        row['footprint_diagnostics'][0]['score']['stable_interior_metric_pass'] = False
        row['eligibility'] = study.eligibility(row['coverage'], row['window'], row['footprint_diagnostics'])
    elif fault == 'summary': audit['conditions'][c]['setup_admitted'] = False
    elif fault == 'product': docs['rgb_body_eligible_windows.json'].pop()
    elif fault == 'labels':
        row['targets']['targets'][0]['contact'] = .5
        for n in ('all_departure_targets.json', 'rgb_body_eligible_targets.json'):
            docs[n][0]['targets'][0]['contact'] = .5
    elif fault == 'prefix': audit['exact_nonreference_prefix_matches'] += 1
    elif fault == 'population': audit['population']['counts']['SETUP_FAILED'] -= 1
    elif fault == 'eligible_count': audit['eligible_departures'] -= 1
    elif fault == 'action_coverage': audit['action_coverage'] = {}
    elif fault == 'contact_count': audit['contact_positive_targets'] += 1
    elif fault == 'strict_failure': audit['strict_visibility_failed_trials'] = []
    else: audit['materialized_samples'][0]['motion_valid'] -= 1
    with pytest.raises(ValueError): study._join_audit(inv, 'l00', audit, docs.__getitem__)


@pytest.mark.parametrize('fault', ['missing', 'extra', 'not_mapping'])
def test_no_partial_layout_study_and_no_discovery(monkeypatch, fault):
    receipts = {b: {} for b in study.BATCHES}
    if fault == 'missing': receipts.pop('l11')
    elif fault == 'extra': receipts['l12'] = {}
    else: receipts = list(receipts)
    monkeypatch.setattr(study, 'load_inventory', lambda: pytest.fail('must reject before any dataset read'))
    with pytest.raises(ValueError, match='all twelve'): study.load_study(receipts)


def test_all12_join_preserves_roles_and_empty_layout_without_fitting(monkeypatch, population):
    inv = population[0]; receipts = {b: {'launch.json': 'a' * 64, study.AUDIT: 'b' * 64} for b in study.BATCHES}; calls = []
    monkeypatch.setattr(study, 'load_inventory', lambda: inv)
    def load(batch, receipt, inventory):
        calls.append(batch); audit, docs = documents(population, batch, empty=batch == 'l11')
        return study._join_audit(inv, batch, audit, docs.__getitem__) | dict(artifact_sha256={})
    monkeypatch.setattr(study, 'load_batch', load)
    monkeypatch.setattr(study, 'verify_artifacts', lambda *a: None)
    result = study.load_study(receipts); report = result.report()
    assert calls == list(study.BATCHES) and len(result.prefixes) == 1440
    assert report['planned_episodes'] == 1440 and len(report['batches']['l11']['excluded_trials']) == 120
    assert len(report['roles']['development_eval']['layouts']) == 3
    assert not report['roles']['development_eval']['all_planned_episodes_eligible']
    assert not report['model_training'] and not report['hardware_qualified']
    receipts['l00']['launch.json'] = 'c' * 64
    assert result.receipts['l00']['launch.json'] == 'a' * 64


@pytest.mark.parametrize('fault', ['none', 'receipt', 'changed_bytes', 'failure', 'missing_audit',
    'raw_bindings', 'source_witness', 'precheck', 'missing_artifact', 'roster'])
def test_receipt_loader_authenticates_before_join_and_rejects_invalid_evidence(monkeypatch, tmp_path, population, fault):
    inv = population[0]; audit, docs = documents(population); ids = inv.episode_ids('l00')
    launch = dict(source_sha256={'synthetic_source.py': 'a' * 64})
    inputs = {'synthetic_raw.bin': hashlib.sha256(b'raw').hexdigest()}
    docs[study.AUDIT_LAUNCH] = dict(batch='l00', source_sha256=launch['source_sha256'],
        artifact_sha256=inputs, complete_collection=True, model_training=False)
    if fault == 'raw_bindings': docs[study.AUDIT_LAUNCH]['artifact_sha256'] = {}
    elif fault == 'source_witness': docs[study.AUDIT_LAUNCH]['source_sha256'] = {}
    (tmp_path / 'synthetic_raw.bin').write_bytes(b'raw')
    for n, v in docs.items(): (tmp_path / n).write_text(json.dumps(v))
    def sha(n): return hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
    audit['output_sha256'] = {n: sha(n) for n in docs}
    if fault == 'roster': audit['output_sha256']['unexpected.json'] = '0' * 64
    (tmp_path / study.AUDIT).write_text(json.dumps(audit)); (tmp_path / 'launch.json').write_text(json.dumps(launch))
    receipt = {'launch.json': sha('launch.json'), study.AUDIT: sha(study.AUDIT)}
    if fault == 'receipt': receipt[study.AUDIT] = '0' * 64
    elif fault == 'changed_bytes': (tmp_path / 'rgb_body_eligible_windows.json').write_text('[]')
    elif fault == 'failure': (tmp_path / 'rgb_body_layout_audit_failure.json').write_text('{}')
    elif fault == 'missing_audit': (tmp_path / study.AUDIT).unlink()
    committed = {c: dict(absent_expected_artifacts=[], raw_precheck=docs[c + '_rgb_body_evaluation.json']) for c in ids}
    if fault == 'precheck': committed[ids[0]]['raw_precheck'] = {}
    elif fault == 'missing_artifact': committed[ids[0]]['absent_expected_artifacts'] = ['missing']
    reads = []; checks = []
    monkeypatch.setattr(study, 'output_root', lambda b: tmp_path)
    def verify(root, hashes):
        checks.append(set(hashes))
        for n, h in hashes.items():
            if not (root / n).is_file() or sha(n) != h: raise ValueError('synthetic identity mismatch')
    monkeypatch.setattr(study, 'verify_artifacts', verify)
    monkeypatch.setattr(study, 'read_json', lambda root, n: reads.append(n) or json.loads((root / n).read_text()))
    monkeypatch.setattr(study, 'verify_ordered_launch', lambda l: None)
    monkeypatch.setattr(study, 'load_terminal_batch', lambda *a: (launch,
        dict(status='RGB_BODY_LAYOUT_COLLECTION_COMPLETE'), committed, inputs))
    if fault == 'none':
        result = study.load_batch('l00', receipt, inv)
        assert result['source_and_artifact_bindings_verified'] and not result['raw_audit_reexecuted']
        assert checks[0] == {'launch.json', study.AUDIT}
        assert set(result['artifact_sha256']) == checks[-1]
    else:
        with pytest.raises(ValueError): study.load_batch('l00', receipt, inv)
        if fault in ('receipt', 'failure', 'missing_audit'): assert not reads
