"""Complete timing populations, predecessor identity and unchanged report witnesses."""
from copy import deepcopy

import pytest

from scripts import verify_go2_density_routed_floor_controller_completion_v1 as check


def histories():
    prior = []
    rows = []
    for i in range(1428):
        old = dict(public_input_sha256='a'*64, original_decision_sha256='b'*64, candidate_decision_sha256='c'*64)
        prior.append(old)
        rows.append(dict(frame=i, public_input_sha256=old['public_input_sha256'],
            original_decision_sha256=old['original_decision_sha256'], baseline_decision_sha256=old['candidate_decision_sha256'],
            candidate_decision_sha256='d'*64, complete_original_decision_reconstructed=True,
            candidate_normalized_decision_exact=True, public_input_arrays_unchanged=True,
            execution_order=[0, 1] if i % 2 == 0 else [1, 0], baseline_controller_s=.2, candidate_controller_s=.18))
    return rows, prior


def test_complete_population_rebuilds_deadline_counts_and_all_windows():
    rows, prior = histories()
    timing = check.check_rows(rows, prior)
    all_rows = timing['all_navigation']
    assert all_rows['observations'] == all_rows['baseline_over_100ms'] == all_rows['candidate_over_100ms'] == 1425
    assert all_rows['candidate_total_s'] == pytest.approx(256.5)
    assert set(timing) == {'all_navigation', 'early_navigation', 'repeated_hold', 'late_navigation'}


@pytest.mark.parametrize('key,value', [('frame', True), ('frame', 1), ('public_input_sha256', 'e'*64),
    ('original_decision_sha256', 'e'*64), ('baseline_decision_sha256', 'e'*64), ('candidate_decision_sha256', 'bad'),
    ('complete_original_decision_reconstructed', False), ('candidate_normalized_decision_exact', False),
    ('public_input_arrays_unchanged', 1)])
def test_rewritten_row_or_nonboolean_claim_rejected(key, value):
    rows, prior = histories()
    rows[0][key] = value
    with pytest.raises(ValueError, match='every original'): check.check_rows(rows, prior)


@pytest.mark.parametrize('key,value', [('baseline_controller_s', float('nan')), ('candidate_controller_s', -1.),
                                     ('baseline_controller_s', True), ('execution_order', [1, 0])])
def test_invalid_timing_or_execution_order_rejected(key, value):
    rows, prior = histories()
    rows[0][key] = value
    with pytest.raises(ValueError): check.check_rows(rows, prior)


@pytest.mark.parametrize('side', ['current', 'prior'])
def test_missing_population_rejected(side):
    rows, prior = histories()
    (rows if side == 'current' else prior).pop()
    with pytest.raises(ValueError, match='complete paired'): check.check_rows(rows, prior)


def test_report_retains_original_state_and_negative_evidence_without_mutation():
    prior = dict(incremental_empty_patch_visibility_skip_comparison=True,
        both_controllers_use_receipt_copied_packed_fused_queries=True, observed_state_checks=[{'frame': 1173, 'sha': 'old'}],
        model_state_unchanged=True, navigation_qualified=False, real_time_qualified=False,
        timing_windows={'old': 'timing'})
    old = deepcopy(prior)
    expected = check.expected_report(prior, {'all_navigation': 'new'})
    assert prior == old
    assert expected['observed_state_checks'] == old['observed_state_checks']
    assert not expected['navigation_qualified'] and not expected['real_time_qualified']
    assert expected['timing_windows'] == {'all_navigation': 'new'}
    assert expected['candidate'] == 'DensityRoutedFloorController'
    assert 'incremental_empty_patch_visibility_skip_comparison' not in expected


def test_wrong_predecessor_scope_rejected():
    with pytest.raises(ValueError, match='visibility-batched'):
        check.expected_report(dict(incremental_empty_patch_visibility_skip_comparison=False), {})
