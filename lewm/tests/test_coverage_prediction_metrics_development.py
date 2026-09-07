import copy
import json

import numpy as np
import pytest

from lewm.coverage_prediction_metrics_development import (
    coverage_report, coverage_shuffle, moving_decisions, paired_layout_delta)
from lewm.tests.test_temporal_prediction_metrics_development import fixture


def moving_fixture():
    p, b = fixture(10)
    for i, m in enumerate(b['metadata']):
        m.update(layout_id='a' if i < 5 else 'b', offset_ns=1_000_000_000,
                 prefix_action_index=1, action_index=i % 5,
                 coverage_source='old' if i % 5 == 1 else 'switch')
    return p, b


def test_strata_do_not_mix_old_continuation_with_new_switches():
    p, b = moving_fixture()
    p[[0, 2, 3, 4, 5, 7, 8, 9], 0, 0] = 2
    result = coverage_report(p, b)
    assert result['moving_switch']['first_half_second']['windows'] == 8
    assert result['moving_continuation']['first_half_second']['windows'] == 2
    assert result['old_later']['first_half_second']['layout_macro']['position_error_m'] == 0
    assert result['moving_switch']['first_half_second']['layout_macro']['position_error_m'] == 2
    assert result['old_initial']['all_known']['layout_macro']['position_error_m'] is None
    json.dumps(result, allow_nan=False)


def test_shuffle_preserves_past_future_actions_and_offset_not_just_future_action():
    _, b = moving_fixture()
    rows = b['metadata'] + [dict(r, prefix_action_index=2) for r in b['metadata']]
    donors, eligible = coverage_shuffle(rows)
    assert eligible.all()
    for i, j in enumerate(donors):
        assert rows[i]['layout_id'] != rows[j]['layout_id']
        for key in ('prefix_action_index', 'action_index', 'offset_ns'):
            assert rows[i][key] == rows[j][key]
    with pytest.raises(ValueError, match='duplicate'):
        coverage_shuffle(rows + [rows[0]])


def test_missing_shuffle_cell_stays_ineligible_without_cross_action_donor():
    _, b = moving_fixture()
    donors, eligible = coverage_shuffle(b['metadata'][:-1])
    assert eligible.sum() == 8 and not eligible[4] and donors[4] == 4


def test_contact_is_retained_with_censored_motion_and_stop_is_a_real_alternative():
    p, b = moving_fixture()
    b['targets']['contact'][1, 0] = 1
    b['targets']['motion_valid'][1, 0] = False
    b['targets']['motion'][1, 0] = np.nan
    p[:, 0, 4] = -30
    p[1, 0, :2] = [.8, 0]
    result = moving_decisions(p, b, 0)
    row = result['rows'][0]
    assert row['chosen_action'] == 1 and row['contact']
    assert row['realized_cost'] == 10 and row['regret'] == pytest.approx(9.2)
    assert row['always_stop_cost'] == .8 and not row['always_stop_contact']


def test_unknown_three_second_horizon_is_reported_not_imputed():
    p, b = moving_fixture()
    result = moving_decisions(p, b, 5)
    assert len(result['unavailable_contexts']) == 2 and not result['rows']
    assert all(r['regret'] is None for r in result['layouts'])


def test_missing_action_or_nonfinite_prediction_is_rejected():
    p, b = moving_fixture()
    bad = copy.deepcopy(b)
    bad['metadata'][0]['action_index'] = 1
    with pytest.raises(ValueError, match='five observed'):
        moving_decisions(p, bad, 0)
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        moving_decisions(p, b, 0)


def test_paired_seed_average_then_layout_bootstrap_and_missing_not_deleted():
    a = [[{'layout_id': 'a', 'cost': i}, {'layout_id': 'b', 'cost': i + 2}] for i in (1, 3, 5)]
    b = [[{'layout_id': 'a', 'cost': 0}, {'layout_id': 'b', 'cost': 0}] for _ in range(3)]
    result = paired_layout_delta(a, b, 'cost')
    assert result['per_seed_mean_delta'] == [2, 4, 6]
    assert result['per_layout_seed_mean_delta'] == [3, 5] and result['mean_delta'] == 4
    a[0][0]['cost'] = None
    assert not paired_layout_delta(a, b, 'cost')['available']
    b[0].reverse()
    # Missing endpoints are deliberately reported before a later-seed audit;
    # use complete rows to test the structural pairing check.
    a[0][0]['cost'] = 1
    with pytest.raises(ValueError, match='pairing'):
        paired_layout_delta(a, b, 'cost')
