"""Training-label contact interactions, not visual utility or a policy.

No I/O, fitting, action execution, prediction scoring or threshold selection.
The caller supplies authenticated, exactly prefix-matched action groups and
the full prospective training-group roster. Construction history/support labels
are diagnostic strata only; they are not deployment inputs. Cross-group body
histories need not match, so even a reversal would not prove an RGB contribution.
"""
from collections import Counter, defaultdict
from itertools import combinations
import math


def summarize(groups, *, planned_group_ids):
    """Check for a common zero-contact action and strict pair-order reversals.

    A reversal requires a < b somewhere and a > b elsewhere within the same
    declared history/support stratum. Ties do not count as reversals. Censored
    groups remain in the denominator and suppress complete-population claims.
    Observed zeros describe recorded contact only, not guaranteed safety.
    """
    if (not isinstance(groups, list) or not isinstance(planned_group_ids, list)
            or not planned_group_ids or any(type(x) is not str or not x for x in planned_group_ids)
            or len(set(planned_group_ids)) != len(planned_group_ids)):
        raise ValueError('nonempty explicit unique planned training groups required')
    fields = {'group_id', 'role', 'layout_id', 'context_kind', 'history_kind',
              'support', 'matched_prefix', 'contact'}
    ids = []
    observed = []
    status = {}
    for g in groups:
        if not isinstance(g, dict) or set(g) != fields or g['role'] != 'train':
            raise ValueError('exact training-only contact group schema required')
        if (any(type(g[k]) is not str or not g[k] for k in
                ('group_id', 'layout_id', 'context_kind', 'history_kind', 'support'))
                or type(g['matched_prefix']) is not bool or not isinstance(g['contact'], list)
                or len(g['contact']) != 6):
            raise ValueError('explicit six-action group identity and prefix status required')
        for v in g['contact']:
            if v is not None and (type(v) not in (int, float) or not math.isfinite(v) or v not in (0, 1)):
                raise ValueError('observed binary contact or explicit censoring required')
        ids.append(g['group_id'])
        reason = ('UNMATCHED_PREFIX' if not g['matched_prefix'] else
                  'CENSORED_CONTACT' if None in g['contact'] else 'COMPLETE')
        status[g['group_id']] = reason
        if reason == 'COMPLETE':
            observed.append(g)
    if len(set(ids)) != len(ids) or set(ids) != set(planned_group_ids):
        raise ValueError('retain every planned group exactly once; no subset diagnostic')
    complete = len(observed) == len(groups)
    zero = set(range(6)) if observed else set()
    minimum = set(range(6)) if observed else set()
    patterns = Counter()
    strata = defaultdict(list)
    for g in observed:
        c = g['contact']
        zero.intersection_update(i for i, v in enumerate(c) if v == 0)
        minimum.intersection_update(i for i, v in enumerate(c) if v == min(c))
        patterns[tuple(int(v) for v in c)] += 1
        strata[g['history_kind'], g['support']].append(g)
    comparisons = []
    for (history, support), rows in sorted(strata.items()):
        pairs = []
        for a, b in combinations(range(6), 2):
            lower = sorted(g['group_id'] for g in rows if g['contact'][a] < g['contact'][b])
            higher = sorted(g['group_id'] for g in rows if g['contact'][a] > g['contact'][b])
            pairs.append(dict(action_pair=[a, b], lower_groups=lower, higher_groups=higher,
                              tied_groups=len(rows)-len(lower)-len(higher),
                              strict_reversal_observed=bool(lower and higher)))
        comparisons.append(dict(history_kind=history, support=support,
                                observed_groups=len(rows), pairs=pairs))
    return dict(role='train', planned_groups=len(planned_group_ids), observed_groups=len(observed),
        complete_population=complete, group_status=status, status_counts=dict(Counter(status.values())),
        contact_patterns=[dict(contact=list(p), groups=n) for p, n in sorted(patterns.items())],
        observed_subset_common_zero_contact_actions=sorted(zero) if observed else None,
        complete_population_common_zero_contact_actions=sorted(zero) if complete else None,
        observed_subset_common_minimum_contact_actions=sorted(minimum) if observed else None,
        complete_population_common_minimum_contact_actions=sorted(minimum) if complete else None,
        strict_pair_reversal_strata=sum(p['strict_reversal_observed'] for s in comparisons for p in s['pairs']),
        stratum_comparisons=comparisons,
        goal_progress_evaluated=False, rgb_contribution_established=False,
        cross_group_sensor_history_equivalence_established=False,
        training_performed=False, navigation_qualified=False, goal_achieved=False)
