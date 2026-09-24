"""Six-action, shared-history hazard evidence at one fixed common horizon.

This is an offline diagnostic, not a controller or navigation qualification.
Caller supplies terminal-audited modality-eligible data and bound raw prefix
witnesses. Existing inventory joins are reused, not predecessor eligibility.
No file access, fitting, threshold selection, inference or privileged input.
"""
from collections import Counter
import re
import numpy as np
from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.pulse_timed_dataset_development import ROLES

HORIZON_NS = 2_000_000_000
GROUP_FIELDS = ('layout_id', 'context_kind', 'history_kind', 'support')
METRICS = ('contact_brier', 'minimum_score_contact_fraction', 'avoidable_contact_regret', 'hazard_concordance')


def mean_available(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def prefix_available(row):
    return (isinstance(row, dict) and row.get('status') == 'COMPLETE_PREFIX'
            and row.get('native_samples') == 1150 and row.get('frames') == 9
            and isinstance(row.get('sha256'), dict) and bool(row['sha256']))


def score_actions(logits, truth):
    """Tie-aware risk ranking; contact labels are observed outcomes, not safety."""
    logits, truth = np.asarray(logits, float), np.asarray(truth, float)
    if logits.shape != (6,) or truth.shape != (6,) or not np.isfinite(logits).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError('six finite hazard logits and six observed binary outcomes required')
    probability = np.exp(-np.logaddexp(0., -logits))
    minima = np.flatnonzero(logits == logits.min())
    selected_contact = float(truth[minima].mean())
    positive, negative = logits[truth == 1], logits[truth == 0]
    # Rank the logits directly; sigmoid saturation must not create false ties.
    wins = positive[:, None] > negative[None, :]
    ties = positive[:, None] == negative[None, :]
    concordance = float((wins + .5 * ties).mean()) if wins.size else None
    return dict(contact_brier=float(((probability - truth)**2).mean()),
        minimum_logit_action_set=minima.tolist(), minimum_score_contact_fraction=selected_contact,
        avoidable_contact_regret=selected_contact - float(truth.min()),
        hazard_concordance=concordance, discordant_outcome_pairs=int(wins.size),
        observed_contact_actions=int(truth.sum()), safe_alternative_observed=bool((truth == 0).any()),
        tie_interpretation='uniform average over exactly tied minimum scores; no action is executed')


def evaluate_matched_hazards(evaluation, prefixes, heads, *, role):
    """Keep every planned six-action group; require complete counterfactual data.

    Missing actions, unequal/unavailable prefixes and censored contact labels
    are retained separately. An unavailable head gets no score on a favorable
    subset. Groups/layouts with no positive-versus-negative contrast receive
    no concordance value, not a perfect ranking score. Layouts are the units.
    """
    if not isinstance(evaluation, IndependentPulseEvaluation) or role not in ROLES:
        raise ValueError('validated independent inventory/dataset view and development role required')
    if not isinstance(prefixes, dict) or not isinstance(heads, dict):
        raise ValueError('explicit prefix witnesses and named prediction heads required')
    if any(not isinstance(k, str) or re.fullmatch('[a-z][a-z0-9_]*', k) is None for k in heads):
        raise ValueError('unambiguous prediction head names required')
    planned = {}
    for episode in evaluation.inventory.episodes.values():
        if episode['role'] == role:
            group = tuple(episode[k] for k in GROUP_FIELDS)
            planned.setdefault(group, {})[episode['action_index']] = episode['episode_id']
    present = any(w['history_ready'] and evaluation.dataset.episode_roles[w['condition']]['role'] == role
                  for w in evaluation.dataset.windows)
    data = evaluation.arrays(role) if present else dict(indices=np.empty(0, np.int64), metadata=[],
        offsets_ns=np.empty((0, 8), np.int64), active=np.empty((0, 8), bool),
        targets=dict(contact=np.empty((0, 8)), contact_valid=np.empty((0, 8), bool)))
    lookup = {row['condition']: i for i, row in enumerate(data['metadata'])}
    groups, scored = [], []
    for group, actions in sorted(planned.items()):
        if set(actions) != set(range(6)): raise ValueError('exact six-action inventory group required')
        ids = [actions[a] for a in range(6)]
        missing = [a for a, c in enumerate(ids) if c not in lookup]
        missing_prefix = [a for a, c in enumerate(ids) if not prefix_available(prefixes.get(c))]
        unequal = ([] if missing_prefix else [a for a in range(1, 6)
            if prefixes[ids[a]]['sha256'] != prefixes[ids[0]]['sha256']])
        censored, positions, truth = [], [], []
        for a, case in enumerate(ids):
            if case not in lookup: continue
            i = lookup[case]
            slots = np.flatnonzero(data['active'][i] & (data['offsets_ns'][i] == HORIZON_NS))
            if len(slots) != 1: raise ValueError('exact common two-second action horizon required')
            h = int(slots[0]); positions.append((i, h))
            valid = data['targets']['contact_valid'][i, h]
            value = data['targets']['contact'][i, h]
            if not valid: censored.append(a)
            elif value not in (0., 1.): raise ValueError('observed contact truth must be binary')
            truth.append(float(value))
        issues = [name for name, values in (('MISSING_ACTIONS', missing), ('UNAVAILABLE_PREFIX', missing_prefix),
            ('UNEQUAL_PREFIX', unequal), ('CENSORED_CONTACT', censored)) if values]
        status = issues[0] if issues else ('OBSERVED_CONTRASTIVE' if 0 < sum(truth) < 6
            else 'OBSERVED_ALL_CONTACT' if sum(truth) == 6 else 'OBSERVED_NO_CONTACT')
        row = dict(zip(GROUP_FIELDS, group)) | dict(conditions=ids, status=status, issues=issues,
            missing_action_indices=missing, unavailable_prefix_actions=missing_prefix,
            unequal_prefix_actions=unequal, censored_contact_actions=censored,
            observed_contact_actions=int(sum(truth)) if not missing and not censored else None)
        groups.append(row)
        if not issues: scored.append((row, positions, np.asarray(truth)))
    layout_ids = sorted({g[0] for g in planned})
    metrics, unavailable = {}, {}
    for name, entry in heads.items():
        if not isinstance(entry, dict) or set(entry) != {'indices', 'prediction'}:
            raise ValueError('explicit exact row indices and outcome prediction array required')
        indices, prediction = np.asarray(entry['indices']), np.asarray(entry['prediction'], float)
        if indices.dtype.kind not in 'iu' or not np.array_equal(indices, data['indices']):
            raise ValueError('same exact ordered role rows required for every head')
        if prediction.shape != (len(indices), 8, 5): raise ValueError('eight-slot five-component prediction required')
        missing_groups = [row['conditions'] for row, positions, _ in scored
            if any(not np.isfinite(prediction[i, h, 4]) for i, h in positions)]
        if missing_groups:
            unavailable[name] = dict(reason='MISSING_HAZARD_PREDICTIONS_NO_SUBSET_SCORE', groups=missing_groups)
            continue
        values = [dict(zip(GROUP_FIELDS, [row[k] for k in GROUP_FIELDS])) | dict(
            conditions=row['conditions'], **score_actions([prediction[i, h, 4] for i, h in positions], truth))
            for row, positions, truth in scored]
        layouts = []
        for layout in layout_ids:
            selected = [v for v in values if v['layout_id'] == layout]
            layouts.append(dict(layout_id=layout, planned_groups=sum(g[0] == layout for g in planned),
                scored_groups=len(selected), contrastive_groups=sum(v['hazard_concordance'] is not None for v in selected),
                **{m: mean_available([v[m] for v in selected]) for m in METRICS}))
        metrics[name] = dict(groups=values, layouts=layouts,
            layout_macro={m: mean_available([v[m] for v in layouts]) for m in METRICS},
            contributing_layouts={m: sum(v[m] is not None for v in layouts) for m in METRICS})
    pairs = {}
    for left in sorted(metrics):
        for right in sorted(metrics):
            if left >= right: continue
            rows = [dict(layout_id=a['layout_id'], **{m: a[m] - b[m]
                if a[m] is not None and b[m] is not None else None for m in METRICS})
                for a, b in zip(metrics[left]['layouts'], metrics[right]['layouts'], strict=True)]
            pairs[left + ' minus ' + right] = dict(layout_differences=rows,
                macro_difference={m: mean_available([r[m] for r in rows]) for m in METRICS},
                contributing_layouts={m: sum(r[m] is not None for r in rows) for m in METRICS},
                higher_is_better=['hazard_concordance'], confidence_interval=None)
    return dict(horizon_ns=HORIZON_NS, role=role, resubstitution=role == 'train', planned_groups=len(groups),
        groups=groups, status_counts=dict(Counter(r['status'] for r in groups)), scored_groups=len(scored),
        all_planned_groups_scored=len(scored) == len(groups),
        planned_layouts=len(layout_ids), contrastive_groups=sum(r['status'] == 'OBSERVED_CONTRASTIVE' for r in groups),
        metrics=metrics, unavailable_heads=unavailable, paired_comparisons=pairs,
        all_requested_heads_comparable=bool(heads) and bool(scored) and not unavailable,
        source_and_eligibility_verified_by_interface=False, confidence_interval=None,
        scope='offline matched disallowed-contact diagnostic; not all unsafe events, execution, or navigation',
        training_performed=False, checkpoint_selection_performed=False, navigation_qualified=False, goal_achieved=False)
