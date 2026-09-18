from copy import deepcopy
from collections import Counter
import pytest
from lewm.all_phase_training_view_development import AllPhaseTrainingView
from lewm.all_phase_training_schedule_development import schedule, SEEDS
from lewm.tests.test_all_phase_study_stream_development import population


@pytest.mark.parametrize('seed', SEEDS)
def test_every_context_covered_with_original_per_trial_weight_and_equal_source_batches(seed):
    old, rows = population(); view = AllPhaseTrainingView(old, rows)
    actual = schedule(view, seed=seed)
    assert actual == schedule(view, seed=seed)
    assert len(actual['batches']) == 1200 and all(len(b) == 6 for b in actual['batches'])
    assert set(i for b in actual['batches'] for i in b) == set(view.indices('train'))
    counts = Counter((view.rows[i]['source'], view.rows[i]['trial']) for b in actual['batches'] for i in b)
    assert all(n == (75 if source == 'family' else 50) for (source, _), n in counts.items())
    for i in range(0, 1200, 2):
        assert [{view.rows[j]['source'] for j in b} for b in actual['batches'][i:i+2]] in (
            [{'family'}, {'switch'}], [{'switch'}, {'family'}])
    assert actual['geometry_transfer_draws'] == 0 and len(counts) == 120


def test_target_values_do_not_influence_order_or_digest():
    old, rows = population(); view = AllPhaseTrainingView(old, rows)
    expected = schedule(view, seed=SEEDS[0])
    for row in view.rows:
        row['targets'] = None
    assert schedule(view, seed=SEEDS[0]) == expected


def test_seeds_change_order_without_changing_trial_weight_or_coverage():
    old, rows = population(); view = AllPhaseTrainingView(old, rows)
    a, b = [schedule(view, seed=s) for s in SEEDS[:2]]
    assert a['batches'] != b['batches'] and a['schedule_sha256'] != b['schedule_sha256']
    assert a['trial_draw_counts'] == b['trial_draw_counts']
    assert set(a['context_draw_counts']) == set(b['context_draw_counts'])


@pytest.mark.parametrize('kwargs', [dict(seed=True), dict(seed=-1), dict(seed=2026),
    dict(seed=SEEDS[0], updates=1201), dict(seed=SEEDS[0], batch_size=8)])
def test_changed_roster_or_budget_is_rejected(kwargs):
    old, rows = population()
    with pytest.raises(ValueError): schedule(AllPhaseTrainingView(old, rows), **kwargs)
