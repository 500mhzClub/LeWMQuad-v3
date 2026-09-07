import copy

import pytest

from lewm.tests.test_coverage_prediction_metrics_development import moving_fixture
from lewm.coverage_prediction_metrics_development import coverage_report, moving_decisions
from scripts.audit_go2_context_matched_coverage_learning_development_v1 import (
    check_primary, check_choices, check_updates, UPDATES)


def test_independent_scalar_primary_accepts_original_and_rejects_count_or_score_changes():
    p, batch = moving_fixture()
    report = coverage_report(p, batch)
    check_primary(p, batch, report)
    for key in ('contact_brier', 'motion_count'):
        bad = copy.deepcopy(report)
        bad['moving_switch']['first_half_second']['layouts'][0][key] += 1
        with pytest.raises(ValueError):
            check_primary(p, batch, bad)


def test_independent_offline_choice_accepts_and_rejects_altered_actions_costs():
    p, batch = moving_fixture()
    result = moving_decisions(p, batch, 0)
    check_choices(p, batch, result, 0)
    for key in ('chosen_action', 'realized_cost', 'regret', 'always_stop_cost'):
        bad = copy.deepcopy(result)
        bad['rows'][0][key] += 1
        with pytest.raises(ValueError):
            check_choices(p, batch, bad, 0)


@pytest.mark.parametrize('condition', ['direct', 'supervised_rollout', 'jepa'])
def test_update_budget_weighted_objectives_and_nonfinite_rejected(condition):
    entries = []
    for update in range(1, UPDATES + 1):
        row = {'update': update, 'loss': 1.12, 'gradient_norm_before_clip': 2.,
               'direct_outcome': 1., 'variance': 1., 'covariance': 2.}
        if condition != 'direct':
            row.update(rollout_outcome=3., loss=4.12)
        if condition == 'jepa':
            row.update(latent_prediction=4., loss=8.12)
        entries.append(row)
    check_updates(entries, condition)
    with pytest.raises(ValueError, match='count'):
        check_updates(entries[:-1], condition)
    entries[-1]['loss'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        check_updates(entries, condition)
