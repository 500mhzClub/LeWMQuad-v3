"""Matched action-coverage endpoints; labels are used only by evaluation.

No layout is treated as independent of its own actions, windows, or seeds.
The unchanged temporal reducer supplies censoring and layout-first aggregation.
"""
import numpy as np

from lewm.temporal_prediction_metrics_development import reduce_predictions


def past_action(row):
    return row.get('prefix_action_index', 0 if row['offset_ns'] == 0 else row['action_index'])


def cell_key(row):
    return (past_action(row), row['action_index'], row['offset_ns'])


def coverage_shuffle(metadata):
    """Cross-layout donors keep past action, future action and offset fixed."""
    layouts = sorted({r['layout_id'] for r in metadata})
    lookup = {(r['layout_id'], *cell_key(r)): i for i, r in enumerate(metadata)}
    if len(lookup) != len(metadata):
        raise ValueError('duplicate layout/past/future/offset identity')
    donors = np.arange(len(metadata))
    eligible = np.zeros(len(metadata), dtype=bool)
    if len(layouts) < 2:
        return donors, eligible
    for i, row in enumerate(metadata):
        key = cell_key(row)
        if all((layout, *key) in lookup for layout in layouts):
            donor = layouts[(layouts.index(row['layout_id']) + 1) % len(layouts)]
            donors[i] = lookup[(donor, *key)]
            eligible[i] = True
    return donors, eligible


def coverage_report(prediction, batch, eligible=None):
    metadata = batch['metadata']
    n = len(metadata)
    active = np.asarray(batch['known_action_valid']).all(-1)
    old = np.array([m['coverage_source'] == 'old' for m in metadata])
    initial = np.array([m['offset_ns'] == 0 for m in metadata])
    moving = np.array([m['offset_ns'] == 1_000_000_000 and past_action(m) != 0 for m in metadata])
    switch = np.array([m['coverage_source'] == 'switch' for m in metadata])
    base = np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible)
    if base.shape != (n,) or base.dtype != bool:
        raise ValueError('explicit eligible population required')
    strata = {'all': np.ones(n, dtype=bool), 'old': old, 'old_initial': old & initial,
              'old_later': old & ~initial, 'moving_all_actions': moving,
              'moving_continuation': moving & old, 'moving_switch': moving & switch}
    result = {}
    for name, rows in strata.items():
        result[name] = {}
        for horizon, mask in (
            ('all_known', active),
            ('first_half_second', np.broadcast_to(np.arange(8) == 0, (n, 8))),
            ('three_seconds', np.broadcast_to(np.arange(8) == 5, (n, 8))),
        ):
            result[name][horizon] = reduce_predictions(
                prediction, batch['targets'], metadata, active, base & rows, mask)
    return result


def moving_decisions(prediction, batch, horizon):
    """Choose among five actually observed actions from each moving context.

    Fixed direction cue and cost match the existing local adapter. These are
    counterfactual offline choices, not executed model-controlled trajectories.
    """
    if horizon not in (0, 5):
        raise ValueError('fixed half-second or three-second horizon required')
    prediction = np.asarray(prediction)
    meta = batch['metadata']
    cv = np.asarray(batch['targets']['contact_valid'])
    mv = np.asarray(batch['targets']['motion_valid'])
    contact = np.asarray(batch['targets']['contact'])
    motion = np.asarray(batch['targets']['motion'])
    groups = {}
    for i, row in enumerate(meta):
        if row['offset_ns'] == 1_000_000_000 and past_action(row) != 0:
            groups.setdefault((row['layout_id'], past_action(row)), []).append(i)
    rows, unavailable = [], []
    for (layout, past), indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda i: meta[i]['action_index'])
        if [meta[i]['action_index'] for i in indices] != list(range(5)):
            raise ValueError('five observed moving-context actions required')
        if not cv[indices, horizon].all():
            unavailable.append({'layout_id': layout, 'past_action': past, 'reason': 'unknown contact horizon'})
            continue
        actual_contact = contact[indices, horizon].astype(bool)
        if not mv[np.array(indices)[~actual_contact], horizon].all():
            raise ValueError('noncontact choice missing actual motion')
        p = prediction[indices, horizon]
        if not np.isfinite(p).all():
            raise ValueError('nonfinite choice prediction')
        probability = 1 / (1 + np.exp(-np.clip(p[:, 4], -60, 60)))
        for cue_name, cue in (('forward', (.8, 0)), ('left', (0, .8)), ('right', (0, -.8))):
            score = 10 * probability + np.linalg.norm(p[:, :2] - cue, axis=1)
            chosen = int(np.argmin(score))  # fixed action-index tie break
            actual = np.full(5, 10.)
            actual[~actual_contact] = np.linalg.norm(motion[np.array(indices)[~actual_contact], horizon, :2] - cue, axis=1)
            rows.append({'layout_id': layout, 'past_action': past, 'cue': cue_name,
                         'chosen_action': chosen, 'contact': bool(actual_contact[chosen]),
                         'realized_cost': float(actual[chosen]), 'regret': float(actual[chosen] - actual.min()),
                         'always_stop_cost': float(actual[0]), 'always_stop_contact': bool(actual_contact[0])})
    layouts = []
    for layout in sorted({r['layout_id'] for r in meta}):
        local = [r for r in rows if r['layout_id'] == layout]
        layouts.append({'layout_id': layout, 'choices': len(local), **{
            k: float(np.mean([r[k] for r in local])) if local else None
            for k in ('contact', 'realized_cost', 'regret', 'always_stop_cost', 'always_stop_contact')},
            'stop_fraction': float(np.mean([r['chosen_action'] == 0 for r in local])) if local else None})
    return {'rows': rows, 'layouts': layouts, 'unavailable_contexts': unavailable,
            'scope': 'offline observed-action ranking, not executed navigation or measured unseen actions'}


def paired_layout_delta(first, second, key, bootstrap_seed=2026092499):
    """Seed-average first, bootstrap layouts second; never pool action rows."""
    if not first or len(first) != len(second):
        raise ValueError('nonempty paired seeds required')
    labels = [r['layout_id'] for r in first[0]]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError('unique layout identities required')
    values = []
    for a, b in zip(first, second, strict=True):
        if [r['layout_id'] for r in a] != labels or [r['layout_id'] for r in b] != labels:
            raise ValueError('layout pairing changed')
        if any(r[key] is None or not np.isfinite(r[key]) for r in [*a, *b]):
            return {'available': False, 'reason': 'missing endpoint; no complete-case layout deletion'}
        values.append([x[key] - y[key] for x, y in zip(a, b, strict=True)])
    array = np.asarray(values)
    layout = array.mean(0)
    rng = np.random.default_rng(bootstrap_seed)
    boot = layout[rng.integers(len(labels), size=(10000, len(labels)))].mean(1)
    return {'available': True, 'mean_delta': float(layout.mean()),
            'per_seed_mean_delta': array.mean(1).tolist(), 'layout_ids': labels,
            'per_layout_seed_mean_delta': layout.tolist(),
            'layout_bootstrap_95_percentile': np.quantile(boot, [.025, .975]).tolist(),
            'scope': 'negative favors first; descriptive reused development layouts, not confirmatory significance'}
