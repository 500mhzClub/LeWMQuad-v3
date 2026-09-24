"""Synthetic-only matched action scoring; no recorded data or fitted heads."""
from copy import deepcopy
import hashlib
import numpy as np
import pytest
from lewm.independent_layout_inventory_development import build_inventory
from lewm.independent_layout_collection_development import CollectionInventory
from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan, validate_timed_plan
from lewm.matched_action_hazard_evaluation_development import HORIZON_NS, score_actions, evaluate_matched_hazards


def fixture():
    inv = CollectionInventory(build_inventory()); windows, targets, roles, prefixes = [], [], {}, {}
    for layout in inv.data['layouts']:
        groups = 1 + layout['layout_index'] % 3
        episodes = [e for e in inv.episodes.values() if e['layout_id'] == layout['layout_id']][:6 * groups]
        for e in episodes:
            blocks, mask = pulse_brake_plan(tuple(e['command']), e['pulse_ticks'])
            active, offsets = validate_timed_plan(blocks[None], mask[None], 1)
            truth = float(e['action_index'] < 2)
            window = dict(condition=e['episode_id'], departure_tick=8, decision_ns=2_300_000_000,
                action_index=e['action_index'], command=e['command'], pulse_ticks=e['pulse_ticks'],
                history_ready=True, history_observation_indices=[5, 6, 7, 8],
                targets=[dict(offset_ns=int(o), future_valid=bool(v)) for o, v in zip(offsets[0], active[0])])
            label = {k: window[k] for k in ('condition', 'departure_tick', 'decision_ns', 'action_index')}
            label.update(target_only=True, targets=[dict(offset_ns=t['offset_ns'], image_target_valid=t['future_valid'],
                motion_valid=t['future_valid'] and truth == 0, contact_valid=t['future_valid'],
                motion=[.01, 0., 0.] if t['future_valid'] and truth == 0 else None,
                contact=truth if t['future_valid'] else None) for t in window['targets']])
            windows.append(window); targets.append(label)
            roles[e['episode_id']] = dict(layout_id=e['layout_id'], role=e['role'])
            group = tuple(e[k] for k in ('layout_id', 'context_kind', 'history_kind', 'support'))
            prefixes[e['episode_id']] = dict(status='COMPLETE_PREFIX', native_samples=1150, frames=9,
                sha256={'synthetic_group': hashlib.sha256(repr(group).encode()).hexdigest()})
    return inv, windows, targets, roles, prefixes


def view(data):
    inv, windows, targets, roles, _ = data
    return IndependentPulseEvaluation(inv, PulseTimedDataset(windows, targets, roles))


def head(v, role='development_eval', mode='correct'):
    data = v.arrays(role); pred = np.zeros((*data['active'].shape, 5))
    pred[..., 4] = np.where(data['targets']['contact'] == 1, 8., -8.)
    if mode == 'reverse': pred[..., 4] *= -1
    elif mode == 'tie': pred[..., 4] = 0.
    return dict(indices=data['indices'], prediction=pred)


def test_ties_and_uniform_outcomes_cannot_fake_hazard_discrimination():
    truth = [1., 1., 0., 0., 0., 0.]
    tied = score_actions(np.zeros(6), truth)
    assert tied['hazard_concordance'] == .5 and tied['minimum_score_contact_fraction'] == pytest.approx(1 / 3)
    assert tied['avoidable_contact_regret'] == pytest.approx(1 / 3)
    for value in (0., 1.):
        row = score_actions(np.zeros(6), [value] * 6)
        assert row['hazard_concordance'] is None and row['discordant_outcome_pairs'] == 0
        assert row['minimum_score_contact_fraction'] == value and row['avoidable_contact_regret'] == 0


def test_finite_extreme_logits_do_not_overflow_or_create_sigmoid_ranking_ties():
    with np.errstate(all='raise', under='ignore'):
        row = score_actions([1e308, 1e308, -1e308, -1e308, -1e308, -1e308], [1, 1, 0, 0, 0, 0])
    assert row['hazard_concordance'] == 1 and row['contact_brier'] == 0
    row = score_actions([1001., 1001., 1000., 1000., 1000., 1000.], [1, 1, 0, 0, 0, 0])
    assert row['hazard_concordance'] == 1


@pytest.mark.parametrize('fault', ['shape', 'nan', 'truth'])
def test_invalid_action_scores_are_rejected(fault):
    logits, truth = np.zeros(6), np.zeros(6)
    if fault == 'shape': logits = logits[:5]
    elif fault == 'nan': logits[0] = np.nan
    else: truth[0] = .5
    with pytest.raises(ValueError): score_actions(logits, truth)


def test_exact_six_action_groups_common_horizon_and_paired_layout_differences():
    data = fixture(); v = view(data)
    result = evaluate_matched_hazards(v, data[-1], dict(correct=head(v), reversed=head(v, mode='reverse')), role='development_eval')
    assert result['horizon_ns'] == HORIZON_NS == 2_000_000_000
    assert result['planned_groups'] == 60 and result['scored_groups'] == result['contrastive_groups'] == 6
    assert result['status_counts']['MISSING_ACTIONS'] == 54 and result['planned_layouts'] == 3
    assert result['metrics']['correct']['layout_macro']['hazard_concordance'] == 1
    assert result['metrics']['reversed']['layout_macro']['hazard_concordance'] == 0
    assert result['metrics']['correct']['layout_macro']['minimum_score_contact_fraction'] == 0
    assert result['metrics']['reversed']['layout_macro']['avoidable_contact_regret'] == 1
    pair = result['paired_comparisons']['correct minus reversed']
    assert pair['macro_difference']['hazard_concordance'] == 1
    assert pair['contributing_layouts']['hazard_concordance'] == 3 and pair['confidence_interval'] is None
    assert not result['training_performed'] and not result['navigation_qualified']


def test_macro_weights_layouts_not_number_of_action_sets():
    data = fixture(); v = view(data); entry = head(v); arrays = v.arrays('development_eval')
    layout = arrays['metadata'][0]['layout_id']
    for i, meta in enumerate(arrays['metadata']):
        if meta['layout_id'] == layout: entry['prediction'][i, :, 4] *= -1
    result = evaluate_matched_hazards(v, data[-1], dict(mixed=entry), role='development_eval')
    metrics = result['metrics']['mixed']
    assert metrics['layout_macro']['hazard_concordance'] == pytest.approx(2 / 3)
    pooled = np.mean([r['hazard_concordance'] for r in metrics['groups']])
    assert not np.isclose(pooled, metrics['layout_macro']['hazard_concordance'])


@pytest.mark.parametrize('fault', ['missing_action', 'missing_prefix', 'short_prefix', 'unequal_prefix', 'censored_contact'])
def test_missing_or_unmatched_branches_remain_in_full_denominator(fault):
    data = fixture(); inv, windows, targets, roles, prefixes = data
    index = next(i for i, w in enumerate(windows) if roles[w['condition']]['role'] == 'development_eval')
    case = windows[index]['condition']
    if fault == 'missing_action':
        windows.pop(index); targets.pop(index); roles.pop(case)
    elif fault == 'missing_prefix': prefixes.pop(case)
    elif fault == 'short_prefix': prefixes[case]['native_samples'] = 1149
    elif fault == 'unequal_prefix': prefixes[case]['sha256']['synthetic_group'] = 'changed'
    else:
        target = next(t for t in targets[index]['targets'] if t['offset_ns'] == HORIZON_NS)
        target.update(contact_valid=False, contact=None, motion_valid=False, motion=None)
    v = view(data); result = evaluate_matched_hazards(v, prefixes, dict(model=head(v)), role='development_eval')
    assert result['planned_groups'] == len(result['groups']) == 60 and result['scored_groups'] == 5
    issue = {'missing_action': 'MISSING_ACTIONS', 'missing_prefix': 'UNAVAILABLE_PREFIX',
             'short_prefix': 'UNAVAILABLE_PREFIX', 'unequal_prefix': 'UNEQUAL_PREFIX', 'censored_contact': 'CENSORED_CONTACT'}[fault]
    assert any(issue in r['issues'] and case in r['conditions'] for r in result['groups'])


def test_one_missing_forecast_makes_entire_head_unavailable_not_favorable_subset():
    data = fixture(); v = view(data); entry = head(v); entry['prediction'][0, 3, 4] = np.nan
    result = evaluate_matched_hazards(v, data[-1], dict(missing=entry, complete=head(v)), role='development_eval')
    assert 'missing' in result['unavailable_heads'] and 'missing' not in result['metrics']
    assert not result['all_requested_heads_comparable'] and not result['paired_comparisons']


@pytest.mark.parametrize('fault', ['rows', 'dtype', 'shape', 'name', 'role'])
def test_prediction_row_identity_and_role_are_not_inferred(fault):
    data = fixture(); v = view(data); entry = head(v); name, role = 'model', 'development_eval'
    if fault == 'rows': entry['indices'] = entry['indices'][::-1]
    elif fault == 'dtype': entry['indices'] = entry['indices'].astype(float)
    elif fault == 'shape': entry['prediction'] = entry['prediction'][:1]
    elif fault == 'name': name = 'ambiguous minus name'
    else: role = 'held_out'
    with pytest.raises(ValueError): evaluate_matched_hazards(v, data[-1], {name: entry}, role=role)


def test_role_with_no_eligible_data_stays_entirely_unavailable():
    data = fixture(); inv, windows, targets, roles, prefixes = data
    for window in windows:
        if roles[window['condition']]['role'] == 'development_eval':
            window['history_ready'] = False; window['history_observation_indices'][0] = None
    result = evaluate_matched_hazards(view(data), prefixes, {}, role='development_eval')
    assert result['planned_groups'] == 60 and result['scored_groups'] == 0
    assert result['status_counts'] == {'MISSING_ACTIONS': 60} and not result['all_requested_heads_comparable']


def test_hazard_diagnostic_does_not_require_unobserved_motion_predictions():
    data = fixture(); v = view(data); entry = head(v); entry['prediction'][..., :4] = np.nan
    result = evaluate_matched_hazards(v, data[-1], dict(hazard_only=entry), role='development_eval')
    assert result['metrics']['hazard_only']['layout_macro']['hazard_concordance'] == 1
    assert result['all_requested_heads_comparable']  # For this declared hazard-only diagnostic, not motion scoring.
