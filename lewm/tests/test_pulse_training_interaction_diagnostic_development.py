"""Synthetic label arithmetic only; no corpus access or scientific claim."""
from copy import deepcopy
from itertools import combinations
import random
import pytest
from lewm.pulse_training_interaction_diagnostic_development import summarize


def group(name, contact, **kw):
    return dict(group_id=name, role='train', layout_id=name, context_kind='corner',
                history_kind='quiet', support='nominal', matched_prefix=True,
                contact=contact) | kw


def run(rows):
    return summarize(rows, planned_group_ids=[g['group_id'] for g in rows])


def test_constant_turn_shortcut_is_not_scene_dependent_success():
    r = run([group('a', [0]*6), group('b', [1, 1, 0, 0, 0, 0])])
    assert r['complete_population_common_zero_contact_actions'] == [2, 3, 4, 5]
    assert r['strict_pair_reversal_strata'] == 0
    assert not r['goal_progress_evaluated'] and not r['rgb_contribution_established']


def test_strict_reversal_is_found_but_is_not_proof_of_vision():
    rows = [group('a', [1, 0, 1, 1, 1, 1]), group('b', [0, 1, 1, 1, 1, 1])]
    r = run(rows)
    assert r['complete_population_common_zero_contact_actions'] == []
    assert r['strict_pair_reversal_strata'] == 1
    p = r['stratum_comparisons'][0]['pairs'][0]
    assert p['action_pair'] == [0, 1] and p['higher_groups'] == ['a'] and p['lower_groups'] == ['b']
    assert run(list(reversed(rows))) == r
    assert not r['rgb_contribution_established']


def test_support_changes_do_not_count_as_within_support_reversals():
    r = run([group('a', [1, 0, 1, 1, 1, 1]),
             group('b', [0, 1, 1, 1, 1, 1], support='slippery')])
    assert r['strict_pair_reversal_strata'] == 0


def test_reversal_counts_match_brute_force_group_pair_definition():
    rng = random.Random(17)
    for _ in range(20):
        rows = [group(str(i), [rng.randrange(2) for _ in range(6)],
                      history_kind=str(i % 2), support=str((i // 2) % 2)) for i in range(16)]
        expected = 0
        for history in ('0', '1'):
            for support in ('0', '1'):
                selected = [g for g in rows if (g['history_kind'], g['support']) == (history, support)]
                for a, b in combinations(range(6), 2):
                    expected += any((x['contact'][a]-x['contact'][b]) *
                                    (y['contact'][a]-y['contact'][b]) < 0
                                    for x, y in combinations(selected, 2))
        assert run(rows)['strict_pair_reversal_strata'] == expected


@pytest.mark.parametrize('fault', ['censored', 'unmatched'])
def test_incomplete_groups_suppress_full_population_claims(fault):
    rows = [group('a', [0]*6), group('b', [1]*6)]
    if fault == 'censored': rows[1]['contact'][0] = None
    else: rows[1]['matched_prefix'] = False
    r = run(rows)
    assert r['planned_groups'] == 2 and r['observed_groups'] == 1
    assert not r['complete_population']
    assert r['complete_population_common_zero_contact_actions'] is None
    assert r['observed_subset_common_zero_contact_actions'] == list(range(6))


def test_all_contact_is_not_zero_contact_and_all_missing_is_not_empty_safe_set():
    r = run([group('a', [1]*6)])
    assert r['complete_population_common_zero_contact_actions'] == []
    assert r['complete_population_common_minimum_contact_actions'] == list(range(6))
    r = run([group('a', [None]*6)])
    assert r['observed_subset_common_zero_contact_actions'] is None


@pytest.mark.parametrize('fault', ['eval', 'bool', 'nan', 'fraction', 'short', 'extra', 'duplicate', 'missing'])
def test_invalid_or_selected_population_rejected(fault):
    rows = [group('a', [0]*6)]
    planned = ['a']
    if fault == 'eval': rows[0]['role'] = 'development_eval'
    elif fault == 'bool': rows[0]['contact'][0] = True
    elif fault == 'nan': rows[0]['contact'][0] = float('nan')
    elif fault == 'fraction': rows[0]['contact'][0] = .5
    elif fault == 'short': rows[0]['contact'].pop()
    elif fault == 'extra': rows[0]['goal'] = 'undeclared'
    elif fault == 'duplicate': rows.append(deepcopy(rows[0]))
    else: planned.append('b')
    with pytest.raises(ValueError): summarize(rows, planned_group_ids=planned)
