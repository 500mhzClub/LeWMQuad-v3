"""Synthetic numerical checks; no original experiment/data/service execution."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
import pytest

from lewm import independent_pulse_scientific_readout_development as science
from lewm.matched_action_hazard_evaluation_development import evaluate_matched_hazards
from lewm.tests.test_independent_pulse_matched_study_development import make_study


def cells(values, *, higher=False):
    seeds = (1, 2, 3); layouts = ('a', 'b', 'c')
    left = {s: dict(zip(layouts, row, strict=True)) for s, row in zip(seeds, values, strict=True)}
    right = {s: dict.fromkeys(layouts, 0.) for s in seeds}
    return science.paired_cells(left, right, seeds=seeds, layouts=layouts,
        metric='synthetic', higher_is_better=higher)


def test_crossed_units_and_macro_not_nine_independent_layouts():
    r = cells([[-1., -2., -3.], [-4., -5., -6.], [-7., -8., -9.]])
    assert r['complete_population_mean'] == -5.
    assert r['independent_layout_units'] == r['repeated_optimization_seeds'] == 3
    assert r['by_layout']['a']['complete_population_mean'] == -4.
    assert r['by_seed']['1']['complete_population_mean'] == -2.
    assert r['descriptive_direction'] == 'ALL_CELLS_LOWER'
    assert r['confidence_interval'] is r['p_value'] is None
    assert not r['practical_benefit_established']


def test_missing_cell_never_becomes_complete_or_favorable_subset_claim():
    r = cells([[-1., None, -3.], [-4., -5., -6.], [-7., -8., -9.]])
    assert not r['complete'] and r['complete_population_mean'] is None
    assert r['observed_cells'] == 8 and r['planned_cells'] == 9
    assert r['observed_subset_mean'] < 0
    assert r['by_layout']['b']['complete_population_mean'] is None
    assert r['by_seed']['1']['complete_population_mean'] is None
    assert r['descriptive_direction'] == 'INCOMPLETE_NO_FULL_POPULATION_DIRECTION'


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, True, '1'])
def test_invalid_metric_rejected(value):
    with pytest.raises(ValueError, match='finite scalar'):
        cells([[value, 0., 0.], [0., 0., 0.], [0., 0., 0.]])


def test_exact_ties_mixed_results_and_higher_is_better():
    assert cells([[0.] * 3] * 3)['descriptive_direction'] == 'ALL_CELLS_EXACTLY_TIED'
    assert cells([[1., -1., 0.]] * 3)['descriptive_direction'] == 'MIXED_OR_PARTLY_TIED'
    result = cells([[1.] * 3] * 3, higher=True)
    assert result['cells'][0]['left_minus_right'] == 1.
    assert result['complete_population_mean'] == -1.
    assert result['descriptive_direction'] == 'ALL_CELLS_LOWER'


def test_contrasts_isolate_latent_objective_and_do_not_choose_auxiliary_heads():
    cs = science.contrasts()
    assert len(cs) == 27
    assert len({(c['left'], c['right']) for c in cs}) == 27
    latent = [c for c in cs if c['question'] == 'added_latent_prediction_objective']
    assert len(latent) == 4 and all(c['left'].endswith('jepa_rollout_outcomes')
        and c['right'].endswith('supervised_rollout_rollout_outcomes') for c in latent)
    assert len([c for c in cs if c['question'] == 'RGB_information']) == 3
    assert len(science._expected_heads()) == 22


@lru_cache(maxsize=1)
def real_schema_scores():
    study = make_study(); view = study.evaluation; scores = {}
    for role in science.ROLES:
        data = view.arrays(role)
        prediction = np.zeros((*data['active'].shape, 5)); prediction[..., 3] = 1.
        heads = {n: dict(indices=data['indices'].copy(), prediction=prediction.copy())
            for n in sorted(science._expected_heads())}
        scores[role] = dict(prediction=view.compare(heads, role=role),
            matched_contact=evaluate_matched_hazards(view, study.prefixes, heads, role=role),
            baseline_missing_cells={}, primary_heads={v + '_' + c: science.primary(v, c)
                for v in science.VARIANTS for c in science.CONDITIONS}, no_best_seed_or_checkpoint_selection=True)
    return {s: {r: deepcopy(scores[r]) | dict(seed=s) for r in science.ROLES} for s in science.SEEDS}


def test_real_scorer_schema_all_scopes_and_no_scientific_promotion():
    source = deepcopy(real_schema_scores()); before = deepcopy(source)
    result = science.summarize(source)
    assert source == before
    assert result['score_aggregation_only'] and not result['raw_prediction_reconstruction_performed']
    assert not result['predictive_JEPA_benefit_established'] and not result['goal_achieved']
    for role, row in result['roles'].items():
        assert row['resubstitution'] == (role == 'train')
        expected = (len(science._scopes(source[science.SEEDS[0]][role]['prediction'])) + 1) * 27
        assert len(row['analyses']) == expected
        assert {r['section'] for r in row['analyses']} == {'prediction', 'matched_contact'}
        for analysis in row['analyses']:
            for value in analysis['metrics'].values():
                assert value['complete_population_mean'] in (0., None)
                assert value['independent_layout_units'] == (6 if role == 'train' else 3)


@pytest.mark.parametrize('fault,match', [
    ('seed', 'all three'), ('role', 'all three'), ('primary', 'primary-head'),
    ('promoted', 'development-only'), ('rows', 'same exact population'),
    ('head', 'head roster'), ('layout', 'ordered planned'), ('scope', 'horizon and stratum'),
    ('counts', 'target counts'), ('hazard_counts', 'contact group denominators')])
def test_reject_changed_population_or_incomplete_factorial(fault, match):
    scores = deepcopy(real_schema_scores()); seed = science.SEEDS[0]
    r = scores[seed]['development_eval']; p = r['prediction']; h = r['matched_contact']
    head = sorted(p['metrics'])[0]
    if fault == 'seed': scores.pop(seed)
    elif fault == 'role': scores[seed].pop('selection')
    elif fault == 'primary': r['primary_heads']['full_jepa'] = 'full_jepa_direct_outcomes'
    elif fault == 'promoted': p['goal_achieved'] = True
    elif fault == 'rows': scores[science.SEEDS[1]]['development_eval']['prediction']['scored_row_indices'][-1] += 1
    elif fault == 'head': p['metrics'].pop(head)
    elif fault == 'layout': p['metrics'][head]['all']['layouts'].pop()
    elif fault == 'scope': p['metrics'][head]['by_actual_offset_ns'].pop(next(iter(p['metrics'][head]['by_actual_offset_ns'])))
    elif fault == 'counts': p['metrics'][head]['all']['layouts'][0]['motion_count'] += 1
    elif fault == 'hazard_counts': h['metrics'][head]['layouts'][0]['scored_groups'] += 1
    with pytest.raises(ValueError, match=match): science.summarize(scores)


def test_missing_head_retained_without_selecting_auxiliary_or_seed():
    scores = deepcopy(real_schema_scores()); part = scores[science.SEEDS[0]]['development_eval']['prediction']
    name = science.primary('full', 'jepa')
    part['metrics'].pop(name); part['unavailable_heads'][name] = dict(reason='synthetic missing prediction')
    result = science.summarize(scores)
    row = next(r for r in result['roles']['development_eval']['analyses'] if r['left'] == name
        and r['section'] == 'prediction' and r['scope'] == ['all'])
    assert not row['metrics']['position_error_m']['complete']
    assert row['metrics']['position_error_m']['complete_population_mean'] is None
    assert len(row['metrics']['position_error_m']['cells']) == 9


def test_saved_pair_and_macro_not_trusted():
    scores = deepcopy(real_schema_scores())
    before = science.summarize(scores)
    for seed in science.SEEDS:
        p = scores[seed]['development_eval']['prediction']
        p['paired_comparisons'] = {'invented': 'not used'}
        for row in p['metrics'].values(): row['all']['layout_macro']['position_error_m'] = -1000.
    assert science.summarize(scores) == before


def test_absent_contrast_never_becomes_perfect_discrimination():
    r = cells([[None] * 3] * 3, higher=True)
    assert r['observed_cells'] == 0 and r['observed_subset_mean'] is None
    assert r['complete_population_mean'] is None and not r['complete']


@pytest.fixture
def reader_fixture(monkeypatch, tmp_path):
    import scripts.navigation_artifact_root_development as authority
    import scripts.read_go2_independent_pulse_science_v1 as reader
    from scripts.run_go2_independent_pulse_matched_study_v1 import identity
    monkeypatch.setattr(authority, 'BASE', tmp_path)
    root = tmp_path / 'go2_synthetic_science_readout_attempt_001'; root.mkdir()
    monkeypatch.setattr(reader, 'OUTPUT', root)
    definition = dict(synthetic=True)
    monkeypatch.setattr(reader, 'definition', lambda: deepcopy(definition))
    monkeypatch.setattr(reader, 'DEFINITION_SHA256', identity(definition))
    def save(name, value):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        (root / name).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
    hashes = {'launch.json': save('launch.json', dict(definition=definition, definition_sha256=identity(definition)))}
    for s, roles in real_schema_scores().items():
        for r, value in roles.items():
            name = f'seed_{s}_{r}_scores.json'; hashes[name] = save(name, value)
    terminal = dict(status='MATCHED_DEVELOPMENT_COMPARISON_COMPLETE', fits=36, optimizer_updates=43200,
        completed_fits=[f'seed_{s}_{v}_{c}' for s in science.SEEDS for v in science.VARIANTS for c in science.CONDITIONS],
        seeds=list(science.SEEDS), objective_conditions=list(science.CONDITIONS), input_variants=list(science.VARIANTS),
        output_sha256=hashes, final_evaluation=False, checkpoint_selection_performed=False,
        navigation_qualified=False, hardware_qualified=False, goal_achieved=False)
    digest = save('result.json', terminal)
    return reader, root, terminal, digest, save


def test_reader_authenticates_complete_result_and_does_not_write_or_claim_raw_audit(reader_fixture):
    reader, root, _, digest, _ = reader_fixture
    before = {p.name: p.read_bytes() for p in root.iterdir()}  # Synthetic tmp root only.
    r = reader.read_result(digest)
    assert r['study_output_hashes_verified'] and len(r['authenticated_score_sha256']) == 9
    assert not r['independent_training_or_raw_scoring_audit_performed']
    assert not r['goal_achieved'] and not r['source_artifacts_verified_by_interface']
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize('fault', ['missing_result', 'wrong_digest', 'partial', 'failure', 'changed_scores',
    'changed_definition', 'promoted', 'missing_score_binding'])
def test_reader_cannot_analyze_live_partial_or_modified_study(reader_fixture, monkeypatch, fault):
    reader, root, terminal, digest, save = reader_fixture
    if fault == 'missing_result': (root / 'result.json').unlink()
    elif fault == 'wrong_digest': digest = 'a' * 64
    elif fault == 'partial': terminal['fits'] = 35; digest = save('result.json', terminal)
    elif fault == 'failure': save('failure.json', {'synthetic': True})
    elif fault == 'changed_scores': save(f'seed_{science.SEEDS[0]}_train_scores.json', {'synthetic_corruption': True})
    elif fault == 'changed_definition': monkeypatch.setattr(reader, 'definition', lambda: {'synthetic': False})
    elif fault == 'promoted': terminal['goal_achieved'] = True; digest = save('result.json', terminal)
    elif fault == 'missing_score_binding':
        terminal['output_sha256'].pop(f'seed_{science.SEEDS[0]}_train_scores.json'); digest = save('result.json', terminal)
    with pytest.raises(ValueError): reader.read_result(digest)


def test_reader_rechecks_source_after_aggregation(reader_fixture, monkeypatch):
    reader, _, _, digest, _ = reader_fixture
    calls = []
    def changing():
        calls.append(True)
        return {'synthetic': len(calls) == 1}
    monkeypatch.setattr(reader, 'definition', changing)
    with pytest.raises(ValueError, match='changed during'): reader.read_result(digest)
